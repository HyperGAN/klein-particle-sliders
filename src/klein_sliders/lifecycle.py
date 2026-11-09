"""Thin product factories; shared core owns updates, optimizer, EMA and packets."""
from dataclasses import replace
from pathlib import Path
import torch
from particle_sliders.lifecycle import Backend, Binding, Capabilities, UnsupportedOperation, native_cleanup
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg095 import ConditionalPG095, engine_options
from particle_sliders.field_critic import FieldCritic
from .contracts import PROTOCOL, WEIGHT_POLICY, SOURCE_PROFILE, INITIALIZATION_POLICY, digest, rate_overrides, validate_options, validate_weight_policy
from .initialization import Construction, initialize_owned, construction_for, validate_receipt
from .adapter import ParticleAdapter
from .native import NativeRuntime, KleinResponseAdapter, observation
from .cache import CachedResponses

CAPABILITIES = Capabilities("routed_particle_gan", render=False, render_outputs=(),
    base_precisions=("bfloat16", "float32"),
    qualification="New PG0.9.5/fresh-init source; matching native CPU, pretrained and GPU checks pending",
    unavailable_reasons=(("render", "Pretrained native sampler/decoder and reference editing need qualification"),))


# Training remains response/cache-only. New native exports advertise a real
# finished-image callback, with empirical native qualification still pending.
INFERENCE_CAPABILITIES = replace(CAPABILITIES, render=True, render_outputs=("pil",),
    unavailable_reasons=(),
    qualification="Native public-pipeline finished-image source integration; pretrained/GPU render qualification pending")


def preflight(options, request, staged):
    validate_weight_policy(options)
    if set(staged) - {"runtime", "construction"}:
        raise ValueError("only runtime and explicit construction intent may be staged")
    construction = staged.get("construction")
    if (not isinstance(construction, Construction) or construction.options_sha256 != digest(options)
            or construction.mode not in {"fresh", "resume"}):
        raise ValueError("use product open_training: fresh/resume intent is required before native allocation")
    if request["native"]:
        raise UnsupportedOperation("training accepts only the frozen velocity cache")
    runtime = staged.get("runtime")
    if runtime is not None:
        if (str(runtime.device) != options.get("device", "cpu")
                or runtime.dtype != getattr(torch, options.get("base_precision", "bfloat16"))):
            raise ValueError("borrowed runtime device/precision must match the session")
        if runtime.device.type != "cpu":
            raise UnsupportedOperation("borrowed accelerator runtimes require separate native qualification")


def attach(runtime, *, checkpointing=True):
    adapter = ParticleAdapter(runtime.transformer).to(device=runtime.device, dtype=torch.float32)
    try:
        if checkpointing:
            runtime.transformer.enable_gradient_checkpointing()
        return adapter, KleinResponseAdapter(runtime, adapter)
    except BaseException:
        adapter.close()
        raise


def prepare(options, *, runtime=None, construction=None):
    validate_options(options)
    if (not isinstance(construction, Construction) or construction.options_sha256 != digest(options)
            or construction.mode not in {"fresh", "resume"}):
        raise ValueError("use product open_training before optimizer construction")
    owns = runtime is None
    if owns:
        runtime = NativeRuntime(options["model_dir"], device=options.get("device", "cpu"),
                                precision=options.get("base_precision", "bfloat16"))
    adapter = None
    # Save the public checkpoint state of borrowed runtimes, too.
    restore_native = native_cleanup(runtime.transformer)
    closed = False
    def close():
        nonlocal closed
        if closed:
            return
        closed = True
        if adapter is not None:
            adapter.close()
        restore_native()
        if owns:
            runtime.close()
    try:
        rows = CachedResponses(options["cache_dir"], runtime, options["cache_index_sha256"])
        devices = [] if runtime.device.type == "cpu" else [runtime.device.index]
        with torch.random.fork_rng(devices=devices):
            from particle_sliders.field_critic import seed_initialization_rng
            seed_initialization_rng(options.get("seed", 7), devices)
            adapter, backend = attach(runtime)
            critic = FieldCritic(rows.response_shape[1], rows.features.shape[1],
                                 options.get("critic_width", 32)).to(runtime.device)
            receipt = initialize_owned(adapter, critic, construction, options)
            backend.fresh_initialization = receipt.as_dict()
        game = ConditionalPG095(backend, rows, critic, particles=adapter.particles,
            batch_size=options.get("batch", 2), total_steps=options.get("steps", 4),
            seed=options.get("seed", 7), sigma=options.get("sigma", 1.),
            response_scale=options.get("response_scale", 1.),
            recipe_overrides=rate_overrides(options), **engine_options(options))
        specification = NATIVE_BACKEND.specification(backend, model_identity=runtime.identity,
            protocol=PROTOCOL, configuration={"weight_policy": WEIGHT_POLICY,
                           "source_profile": SOURCE_PROFILE,
                           "initialization_policy": INITIALIZATION_POLICY,
                           "initialization": receipt.as_dict()})
        return Binding(game, close, inference=specification,
                       before_update=lambda request: bool(rows.identity()),
                       observation=lambda: observation(runtime, adapter))
    except BaseException:
        close()
        raise


def validate_inference(options, spec):
    if set(options) - {"model_dir", "device", "base_precision"}:
        raise ValueError("native inference accepts model_dir/device/base_precision only")
    if not isinstance(options.get("model_dir"), str) or not Path(options["model_dir"]).is_dir():
        raise ValueError("local staged model_dir required")
    if options.get("device", "cpu") not in ("cpu", "cuda:0", "cuda:1"):
        raise ValueError("explicit CPU or single GPU device required")
    if options.get("base_precision", "bfloat16") not in ("float32", "bfloat16"):
        raise ValueError("only FP32/BF16 native bases are supported")
    configuration = spec.configuration
    if (spec.identity["protocol"] != PROTOCOL
            or set(configuration) != {"weight_policy", "source_profile", "initialization_policy", "initialization"}
            or configuration["weight_policy"] != WEIGHT_POLICY
            or configuration["source_profile"] != SOURCE_PROFILE
            or configuration["initialization_policy"] != INITIALIZATION_POLICY):
        raise ValueError("requires the new PG0.9.5 RAW-only Klein export; historical artifacts retain original source")
    validate_receipt(configuration["initialization"], cohort=configuration["initialization"].get("cohort_id"))


def preflight_inference(options, request, spec):
    native = request["native"]
    if request["operation"] == "render":
        from .render import validate_render_request
        allowed = {"prompt", "height", "width", "steps", "images"}
        if set(native) - allowed:
            raise ValueError("unknown native finished-render arguments")
        validate_render_request(prompt=native.get("prompt"), height=native.get("height", 512),
            width=native.get("width", 512), steps=native.get("steps", 8), seed=0,
            strength=request["strength"], images=native.get("images"), output_type=request["output_type"])
        return
    required = {"latent", "embedding", "text_ids", "timestep"}
    if native and set(native) != required:
        raise ValueError("native response requires latent, embedding, text_ids and timestep together")
    if not native:
        return
    for name, rank in (("latent", 4), ("embedding", 3), ("text_ids", 3)):
        value = native[name]
        if not isinstance(value, torch.Tensor) or value.ndim != rank or value.shape[0] != 1:
            raise ValueError("invalid native " + name + " tensor geometry")


def prepare_inference(options, spec):
    validate_inference(options, spec)
    runtime = NativeRuntime(options["model_dir"], device=options.get("device", "cpu"),
                            precision=options.get("base_precision", "bfloat16"))
    adapter = None
    def close():
        if adapter is not None:
            adapter.close()
        runtime.close()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            adapter, backend = attach(runtime, checkpointing=False)
        backend.fresh_initialization = spec.configuration["initialization"]
        identity = native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL)
        def response(*, strength, seed, latent, embedding, text_ids, timestep):
            with backend.network.scaled(strength):
                return backend._velocity(dict(embeds=embedding, text_ids=text_ids), latent,
                                         torch.as_tensor(timestep)).detach().clone()
        def render(*, strength, seed, images=None, output_type="pil", **native):
            from .render import render_image
            return render_image(runtime, adapter, strength=strength, seed=seed,
                                images=images, output_type=output_type, **native)
        return InferenceBinding(backend.network, identity, close, response=response, render=render,
                                observation=lambda: observation(runtime, adapter))
    except BaseException:
        close()
        raise


NATIVE_BACKEND = InferenceBackend("flux2_klein", INFERENCE_CAPABILITIES, prepare_inference, validate_inference,
                                  preflight=preflight_inference)
BACKEND = Backend("flux2_klein", CAPABILITIES, prepare, validate_options, preflight=preflight)


def open_training(config, *, resume=None, **request):
    """Bind actual fresh/restore intent before factory/optimizers; no migration."""
    from copy import deepcopy
    from particle_sliders.lifecycle import read_packet
    # Source/policy rejection and full packet hash/config/receipt check precede
    # any native allocation. The exact core restore performs the final binding.
    validate_options(config.options)
    packet = deepcopy(resume) if isinstance(resume, dict) else read_packet(resume) if resume is not None else None
    construction = construction_for(config, packet)
    session = BACKEND.open(config, operation="resume" if packet is not None else "train",
                           construction=construction, **request)
    try:
        if packet is not None:
            session.restore(packet)
        return session
    except BaseException as primary:
        from particle_sliders.restoration import _restore_all
        _restore_all([session.close], primary=primary)
        raise
