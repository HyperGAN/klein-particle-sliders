"""Thin product factories; shared core owns updates, optimizer, EMA and packets."""
from contextlib import contextmanager
from pathlib import Path
import torch
from particle_sliders.lifecycle import Backend, Binding, Capabilities, UnsupportedOperation, native_cleanup
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.pg094 import ConditionalPG094
from particle_sliders.field_critic import FieldCritic
from .contracts import PROTOCOL, WEIGHT_POLICY, validate_options, validate_weight_policy
from .adapter import ParticleAdapter
from .native import NativeRuntime, KleinResponseAdapter, observation
from .cache import CachedResponses

CAPABILITIES = Capabilities("routed_particle_gan", render=False, render_outputs=(),
    base_precisions=("bfloat16", "float32"),
    qualification="Source integration and tiny CPU native conformance only; pretrained/GPU unqualified",
    unavailable_reasons=(("render", "Pretrained native sampler/decoder and reference editing need qualification"),))


def preflight(options, request, staged):
    validate_weight_policy(options)
    if set(staged) - {"runtime"}:
        raise ValueError("only a staged native runtime may be borrowed")
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


def prepare(options, *, runtime=None):
    validate_options(options)
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
        game = ConditionalPG094(backend, rows, critic, particles=adapter.particles,
            batch_size=options.get("batch", 2), total_steps=options.get("steps", 4),
            seed=options.get("seed", 7), sigma=options.get("sigma", 1.),
            response_scale=options.get("response_scale", 1.))
        specification = NATIVE_BACKEND.specification(backend, model_identity=runtime.identity,
            protocol=PROTOCOL, configuration={"weight_policy": WEIGHT_POLICY})
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
    if (spec.identity["protocol"] != PROTOCOL
            or spec.configuration != {"weight_policy": WEIGHT_POLICY}):
        raise ValueError("requires the fresh RAW-only Klein export; historical artifacts retain their original source")


def preflight_inference(options, request, spec):
    native = request["native"]
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
        identity = native_identity(backend, model_identity=runtime.identity, protocol=PROTOCOL)
        def response(*, strength, seed, latent, embedding, text_ids, timestep):
            with backend.network.scaled(strength):
                return backend._velocity(dict(embeds=embedding, text_ids=text_ids), latent,
                                         torch.as_tensor(timestep)).detach().clone()
        return InferenceBinding(backend.network, identity, close, response=response,
                                observation=lambda: observation(runtime, adapter))
    except BaseException:
        close()
        raise


NATIVE_BACKEND = InferenceBackend("flux2_klein", CAPABILITIES, prepare_inference, validate_inference,
                                  preflight=preflight_inference)
BACKEND = Backend("flux2_klein", CAPABILITIES, prepare, validate_options, preflight=preflight)
