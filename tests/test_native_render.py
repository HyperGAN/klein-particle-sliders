"""Real pinned native pipeline API with RANDOM TINY components, CPU only.

Authored for a separately admitted matching environment. No pretrained weights,
network calls, downloads, quality/effect result or genuine native qualification.
"""
from hashlib import sha256
import os
from types import SimpleNamespace
from zipfile import ZipFile
import numpy as np
import pytest
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from transformers import Qwen3Config, Qwen3ForCausalLM
from klein_sliders.activation_gate import CORE_MODULE, LatentGatePlan
from klein_sliders.adapter import ParticleAdapter
from klein_sliders.render import render_image, native_pipeline_state
from klein_sliders.scheduled_render import render as render_scheduled
from conftest import TinyRuntime


class TinyTokenizer:
    """Explicit synthetic deterministic token IDs; actual tiny Qwen3 runs them."""
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]
    def __call__(self, text, *, max_length, **kwargs):
        return dict(input_ids=torch.ones(1,max_length,dtype=torch.long),
                    attention_mask=torch.ones(1,max_length,dtype=torch.long))


def tiny_native_pipeline(runtime):
    """One actual tiny DiT, VAE, Qwen3 and native scheduler; never from_pretrained."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(319)
        vae=AutoencoderKLFlux2(latent_channels=2,block_out_channels=(4,4,4,4),
            layers_per_block=1,norm_num_groups=1,mid_block_add_attention=False,sample_size=256)
        text=Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=4,intermediate_size=8,
            num_hidden_layers=28,num_attention_heads=1,num_key_value_heads=1,
            head_dim=4,max_position_embeddings=256))
    vae.requires_grad_(False).eval();text.requires_grad_(False).eval()
    scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    pipe=Flux2KleinPipeline(scheduler=scheduler,vae=vae,text_encoder=text,
        tokenizer=TinyTokenizer(),transformer=runtime.transformer,is_distilled=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_real_native_pipeline_driver_matches_exact_OFF_and_finishes_ON():
    runtime=TinyRuntime();pipe=tiny_native_pipeline(runtime)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        adapter=ParticleAdapter(runtime.transformer)
    view=SimpleNamespace(device=runtime.device,transformer=runtime.transformer,load_renderer=lambda:pipe)
    before=torch.get_rng_state().clone();scheduler=pipe.scheduler
    try:
        with torch.no_grad(),native_pipeline_state(pipe),adapter.scaled(0.):
            expected=pipe(image=None,prompt="A watering can",height=256,width=256,
                num_inference_steps=2,guidance_scale=1.,num_images_per_prompt=1,
                generator=torch.Generator(device="cpu").manual_seed(7),output_type="pil",
                return_dict=True,max_sequence_length=128,text_encoder_out_layers=(9,18,27)).images[0]
        off=render_image(view,adapter,prompt="A watering can",height=256,width=256,
                         steps=2,seed=7,strength=0.)
        assert np.array_equal(np.asarray(expected),np.asarray(off))
        assert off.size==(256,256) and off.mode=="RGB"
        assert pipe.scheduler is scheduler and adapter.strength==0.
        # Explicit RAW adapter state fixture (not a trained/effectful slider).
        raw={name:value.detach().clone() for name,value in adapter.state_dict().items()}
        with torch.no_grad():
            raw[next(name for name in raw if name.endswith("up.weight"))].fill_(.01)
        adapter.load_state_dict(raw,strict=True)
        on=render_image(view,adapter,prompt="A watering can",height=256,width=256,
                        steps=2,seed=7,strength=1.)
        assert on.size==(256,256) and on.mode=="RGB"
        assert all(torch.equal(value,raw[name]) for name,value in adapter.state_dict().items())
        assert pipe.transformer is runtime.transformer
        assert pipe.scheduler is scheduler and adapter.strength==0.
        assert torch.equal(before,torch.get_rng_state())
        assert not torch.cuda.is_initialized()
    finally:
        adapter.close();runtime.close()


@pytest.fixture
def native_raw_session(tmp_path, bound, monkeypatch):
    from particle_sliders.inference import open_inference
    from klein_sliders import lifecycle
    from klein_sliders.native import NativeRuntime
    from test_lifecycle import klein_config, make_cache

    class FixtureRuntime(TinyRuntime, NativeRuntime):
        def __init__(self):
            super().__init__()
            self.pipe = tiny_native_pipeline(self)

        def load_renderer(self):
            return self.pipe

    cache, cache_sha = make_cache(tmp_path, bound)
    options = dict(model_dir=str(tmp_path), device="cpu", base_precision="float32")
    config = klein_config("flux2_klein", "tiny-scheduled-RAW", **options, cache_dir=str(cache),
                          cache_index_sha256=cache_sha, steps=1, batch=2, critic_width=4)
    with lifecycle.open_training(config, runtime=FixtureRuntime()) as training:
        training.train()
        raw = training.export_inference(selection="raw")
    runtime = FixtureRuntime()
    monkeypatch.setattr(lifecycle, "NativeRuntime", lambda *_args, **_kwargs: runtime)
    request = dict(prompt="A watering can", height=256, width=256, steps=2,
                   seed=7, strength=0.5)
    preview = {key: request[key] for key in ("prompt", "height", "width", "steps")}
    with open_inference(raw, backend=lifecycle.NATIVE_BACKEND, options=options,
                        native_request=preview) as session:
        yield session, runtime, request


def test_scheduled_render_uses_native_pipeline_and_restores_the_original_scope(native_raw_session):
    session, runtime, request = native_raw_session
    scheduler = runtime.pipe.scheduler
    before = session.render(**request)
    constant = render_scheduled(session, **request, step_gains=[1.0, 1.0])
    assert constant.tobytes() == before.tobytes()
    scheduled = render_scheduled(session, **request, step_gains=[1.5, 0.5])
    receipt = scheduled.info["strength_schedule"]
    assert receipt["effective_strengths"] == [0.75, 0.25]
    assert len(receipt["sigmas"]) == 3
    assert runtime.pipe.scheduler is scheduler
    assert not runtime.transformer._forward_pre_hooks and not runtime.transformer._forward_hooks


@pytest.fixture
def gate_options():
    wheel = os.environ.get("KLEIN_STEERING_CORE_WHEEL")
    if not wheel:
        pytest.skip("KLEIN_STEERING_CORE_WHEEL is needed for the exact shared gate")
    return dict(core_wheel=wheel, calibration_sha256=sha256(b"synthetic Klein fixture").hexdigest())


def test_native_cfp_gates_the_live_packed_latent_and_replays_exactly(native_raw_session, gate_options):
    session, runtime, request = native_raw_session
    observed = []

    def record(_module, _args, kwargs):
        observed.append(kwargs["hidden_states"].mean(dim=(0, 1), dtype=torch.float32))

    with runtime.transformer.register_forward_pre_hook(record, with_kwargs=True):
        session.render(**(request | {"strength": 0.0}))
    direction = torch.nn.functional.normalize(observed[0] - observed[1], dim=0)
    scores = [float(activation @ direction) for activation in observed]
    gate_plan = LatentGatePlan(**gate_options, method="cfp", parameter=sum(scores) / len(scores), direction=direction)
    observed.clear()
    with runtime.transformer.register_forward_pre_hook(record, with_kwargs=True):
        gated = render_scheduled(session, **request, gate_plan=gate_plan)
    trace = gated.info["strength_schedule"]
    assert [item["projection"] for item in trace["decisions"]] == [float(row @ direction) for row in observed]
    assert [item["fired"] for item in trace["decisions"]] == [False, True]
    assert trace["eligible"] == 2 and trace["fired"] == 1 and trace["gate_rate"] == 0.5
    assert trace["nominal_strength"] == request["strength"]
    assert trace["sigma_weighted_strength"] == request["strength"] * (trace["sigmas"][1] - trace["sigmas"][2])
    assert trace["activation_gate"]["calibration_sha256"] == gate_options["calibration_sha256"]
    assert gated.tobytes() == render_scheduled(session, **request, step_gains=[0, 1]).tobytes()


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_native_random_endpoints_match_ordinary_raw(native_raw_session, gate_options, rate):
    session, runtime, request = native_raw_session
    gate_plan = LatentGatePlan(**gate_options, method="random", parameter=rate, seed=1)
    expected = session.render(**(request | {"strength": request["strength"] * rate}))
    scheduler = runtime.pipe.scheduler
    gated = render_scheduled(session, **request, gate_plan=gate_plan)
    assert gated.tobytes() == expected.tobytes()
    assert gated.info["strength_schedule"]["gate_rate"] == rate
    assert runtime.pipe.scheduler is scheduler


def test_native_gate_requires_explicit_calibrated_strength_and_exclusive_schedule(native_raw_session, gate_options):
    session, runtime, request = native_raw_session
    gate_plan = LatentGatePlan(**gate_options, method="random", parameter=0.5, seed=1)
    with pytest.raises(ValueError, match="explicit nominal strength"):
        render_scheduled(session, **{key: value for key, value in request.items() if key != "strength"}, gate_plan=gate_plan)
    with pytest.raises(ValueError, match="Choose fixed Klein step gains"):
        render_scheduled(session, **request, step_gains=[1, 1], gate_plan=gate_plan)
    assert not runtime.transformer._forward_pre_hooks and not runtime.transformer._forward_hooks


def test_native_clabs_restarts_with_fixed_seed_and_restores_after_error(native_raw_session, gate_options, monkeypatch):
    session, runtime, request = native_raw_session
    gate_plan = LatentGatePlan(**gate_options, method="clabs", parameter=0.5, seed=1,
        direction=[1.0] + [0.0] * 7, clabs=dict(mean=0., std=1., epsilon=1e-6, setpoint=0.,
                    proportional_gain=0.2, integral_gain=0.1, beta=0.5))
    expected = render_scheduled(session, **request, gate_plan=gate_plan)
    scheduler = runtime.pipe.scheduler

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected native Euler failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(scheduler), "step", fail)
        with pytest.raises(RuntimeError, match="injected native Euler failure"):
            render_scheduled(session, **request, gate_plan=gate_plan)
    assert runtime.pipe.scheduler is scheduler
    assert not runtime.transformer._forward_pre_hooks and not runtime.transformer._forward_hooks
    again = render_scheduled(session, **request, gate_plan=gate_plan)
    assert again.tobytes() == expected.tobytes()
    assert again.info["strength_schedule"] == expected.info["strength_schedule"]
    trace = again.info["strength_schedule"]
    previous = 0.0
    for decision in trace["decisions"]:
        error = -decision["projection"] / (1.0 + 1e-6)
        previous = 0.5 * previous + 0.5 * error
        assert decision["probability"] == pytest.approx(min(1., max(0., 0.5 + 0.2 * error + 0.1 * previous)))


def test_tampered_steering_wheel_refused(tmp_path):
    path = tmp_path / "wrong.whl"
    with ZipFile(path, "w") as archive:
        archive.writestr(CORE_MODULE, "raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="exact shared steering module"):
        LatentGatePlan(core_wheel=path, method="random", parameter=0.0, seed=1, calibration_sha256="a" * 64)
