"""CPU coverage for the source-bound pretrained-shaped Klein velocity route."""

from hashlib import sha256
from itertools import pairwise
import json
import os
from pathlib import Path
from types import SimpleNamespace

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
import numpy as np
import pytest
import torch

from klein_sliders.contracts import digest
from klein_sliders.velocity_steering import (
    NATIVE50_SCHEMA, SCHEMA, VELOCITY_SHAPE, VelocitySteeringPlan, render_native_velocity,
)


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def tensor_sha(value):
    return sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def clock(steps):
    if steps == 8:
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(sigmas=[1, .95, .85, .8, .7, .55, .4, .2], device="cpu")
    else:
        assert steps == 50
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True)
        retrieve_timesteps(scheduler, steps, "cpu", sigmas=np.linspace(1., 1 / steps, steps),
                           mu=compute_empirical_mu(image_seq_len=VELOCITY_SHAPE[1], num_steps=steps))
        evidence = Path(__file__).resolve().parents[1] / "evidence/soft-diffuse-light-raw8-adaptive-steering.json"
        assert scheduler.sigmas.tolist() == json.loads(evidence.read_bytes())["sigmas"]
    scheduler.set_begin_index(0)
    return scheduler


@pytest.fixture(params=(8, 50), ids=("8-step", "50-step"))
def owned(tmp_path, request):
    supplied = os.environ.get("KLEIN_SPSR_CORE_WHEEL")
    if not supplied:
        pytest.skip("Set KLEIN_SPSR_CORE_WHEEL to the pinned shared wheel")
    raw = tmp_path / "RAW.pt"
    fit = tmp_path / "probe.pt"
    ref = tmp_path / "binding.json"
    adapter = torch.nn.Linear(4, 4).eval().requires_grad_(False)
    adapter.strength = 0
    torch.save(dict(network=adapter.state_dict(), weights="raw", ema_decay=None), raw)
    identity = {"fixture": "stock CPU Euler / packed Klein geometry"}
    steps = request.param
    grid = clock(steps).sigmas.tolist()
    train = "a" * 64
    module = "67ee03066e255fe105735820b0edba53757d943b93239b1279de707d23dbae59"
    state = {"probe.weight": torch.ones(1, 128)/128, "probe.bias": torch.tensor([0.125]),
             "direction.weight": torch.full((1, 128), 0.25)}
    torch.save(dict(probe=state, velocity_shape=list(VELOCITY_SHAPE), weights="raw", ema_decay=None,
                    native_raw_strength=0, plan={"sha256": train}, raw_checkpoint={"sha256": file_sha(raw)},
                    sigmas=grid, core_module_sha256=module), fit)
    schema = SCHEMA if steps == 8 else NATIVE50_SCHEMA
    ref.write_text(json.dumps(dict(schema=schema, weights="raw", ema_decay=None, fit_sha256=file_sha(fit),
        fit_plan_sha256=train, raw_sha256=file_sha(raw), model_identity_sha256=digest(identity),
        velocity_shape=list(VELOCITY_SHAPE), sigmas=grid)))
    plan = VelocitySteeringPlan(core_wheel=supplied, binding_path=ref, binding_sha256=file_sha(ref), fit_path=fit)
    return dict(plan=plan, adapter=adapter, raw_path=raw, model_identity=identity, binding=ref, fit=fit, grid=grid,
                steps=steps)


def native_request(seed=71):
    generator = torch.Generator().manual_seed(seed)
    start = torch.randn(VELOCITY_SHAPE, generator=generator, dtype=torch.bfloat16)
    cond = torch.ones((1,128,7680), dtype=torch.bfloat16)
    text = torch.zeros((1,128,4), dtype=torch.int64)
    request = dict(seed=seed, initial_sha256=tensor_sha(start), conditioning_sha256=tensor_sha(cond), text_ids_sha256=tensor_sha(text))
    return start, cond, text, request


def run(scope, *, alpha, mode="adaptive", trace=None, requested=None, cfg=False, wrong_text=False,
        mutate_grid=False, native_steps=None, begin_index=0):
    start, cond, text, native = native_request()
    if wrong_text:
        text += 1
    steps = scope["steps"] if native_steps is None else native_steps
    velocities = [torch.full_like(start, value / steps) for value in range(steps)]
    seen = []

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pos = 0

        def forward(self, *, hidden_states, encoder_hidden_states, txt_ids):
            value = velocities[self.pos]
            self.pos += 1
            return (value,)

    model = Transformer().eval()
    scheduler = clock(steps)
    if mutate_grid:
        scheduler.sigmas[1] = (scheduler.sigmas[0] + scheduler.sigmas[1]) / 2
    scheduler.set_begin_index(begin_index)
    pipe = SimpleNamespace(transformer=model, scheduler=scheduler, do_classifier_free_guidance=cfg)

    def render():
        latent = start
        for timestep in scheduler.timesteps:
            velocity, = model(hidden_states=latent, encoder_hidden_states=cond, txt_ids=text)
            seen.append(velocity)
            latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
        return latent

    args = {key: scope[key] for key in ("plan", "adapter", "raw_path", "model_identity")}
    try:
        result = render_native_velocity(render, transformer=model, pipeline=lambda: pipe, request=requested or native,
            alpha=alpha, mode=mode, from_adaptive=trace, check_live=lambda: None, **args)
    finally:
        assert not model._forward_hooks
    return result, seen, velocities


def test_adaptive_euler_identity_and_exact_bound_constant(owned):
    (off, off_record), seen, velocities = run(owned, alpha=0, mode="off")
    (zero, zero_record), _, _ = run(owned, alpha=0)
    assert torch.equal(off, zero)
    assert all(a is b for a, b in zip(seen, velocities, strict=True))
    assert off_record["exact_coefficient_dose"] == zero_record["exact_coefficient_dose"] == [0,1]

    (adapted, adaptive), _, velocities = run(owned, alpha=1)
    (fixed, constant), _, _ = run(owned, alpha=1, mode="same-atom-constant", trace=adaptive)
    assert not torch.equal(off, adapted) and not torch.equal(off, fixed)
    assert len({row["applied_coefficient"] for row in adaptive["steps"]}) > 1
    assert {row["applied_coefficient"] for row in constant["steps"]} == {adaptive["nearest_even_constant"]}
    assert constant["exact_coefficient_dose"] == adaptive["exact_nearest_constant_dose"]
    assert adaptive["sigmas"] == owned["grid"]
    assert len(adaptive["steps"]) == len(constant["steps"]) == owned["steps"]
    assert adaptive["integrated_realized_mean_BF16_edit_norm"] > 0
    assert all(row["interval"] == i for i, row in enumerate(constant["steps"]))
    assert all(a > b for a, b in pairwise(owned["grid"]))


def test_constant_requires_same_seed_alpha_and_exact_trace(owned):
    (_, measured), _, _ = run(owned, alpha=1)
    changed = dict(measured)
    changed["nearest_even_constant"] = 0
    with pytest.raises(ValueError, match="matching receipt differ"):
        run(owned, alpha=1, mode="same-atom-constant", trace=changed)
    with pytest.raises(ValueError, match="different fit, request or alpha"):
        run(owned, alpha=0, mode="same-atom-constant", trace=measured)
    with pytest.raises(ValueError, match="different fit, request or alpha"):
        run(owned, alpha=1, mode="same-atom-constant", trace=measured, requested=native_request(72)[-1])


def test_rejects_owner_model_clock_and_condition_change(owned):
    owned["adapter"].train()
    with pytest.raises(ValueError, match="RAW adapter must be frozen"):
        run(owned, alpha=0)
    owned["adapter"].eval()
    with pytest.raises(ValueError, match="fitted Klein native model"):
        run(dict(owned, model_identity={}), alpha=0)
    owned["adapter"].strength = 0.25
    with pytest.raises(ValueError, match="frozen RAW scale zero"):
        run(owned, alpha=0)
    owned["adapter"].strength = 0
    with torch.no_grad():
        owned["adapter"].bias.add_(1)
    with pytest.raises(ValueError, match="live Klein adapter differs"):
        run(owned, alpha=0)
    with torch.no_grad():
        owned["adapter"].load_state_dict(torch.load(owned["raw_path"], weights_only=True)["network"])
    with pytest.raises(RuntimeError, match="Euler grid differs"):
        run(owned, alpha=0, mutate_grid=True)
    with pytest.raises(RuntimeError, match="first Klein latent or prompt"):
        run(owned, alpha=0, wrong_text=True)
    with pytest.raises(RuntimeError, match="CFG1"):
        run(owned, alpha=0, cfg=True)
    with pytest.raises(RuntimeError, match="per Euler interval"):
        run(owned, alpha=0, begin_index=1)


def test_fit_and_live_clock_cannot_cross_profiles(owned):
    other_steps = 50 if owned["steps"] == 8 else 8
    with pytest.raises(RuntimeError, match="Euler grid differs"):
        run(owned, alpha=0, native_steps=other_steps)
    binding_path = owned["binding"]
    binding = json.loads(binding_path.read_bytes())
    other_schema = NATIVE50_SCHEMA if other_steps == 50 else SCHEMA
    changed = dict(binding, schema=other_schema)
    binding_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=f"measured {other_steps}-step"):
        VelocitySteeringPlan(core_wheel=os.environ["KLEIN_SPSR_CORE_WHEEL"], binding_path=binding_path,
            binding_sha256=file_sha(binding_path), fit_path=owned["fit"])
    changed["sigmas"] = clock(other_steps).sigmas.tolist()
    binding_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Probe fit differs"):
        VelocitySteeringPlan(core_wheel=os.environ["KLEIN_SPSR_CORE_WHEEL"], binding_path=binding_path,
            binding_sha256=file_sha(binding_path), fit_path=owned["fit"])


def test_refuses_fit_file_replacement(owned):
    with owned["fit"].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="probe fit SHA256 differs"):
        VelocitySteeringPlan(core_wheel=os.environ["KLEIN_SPSR_CORE_WHEEL"],
            binding_path=owned["binding"], binding_sha256=file_sha(owned["binding"]), fit_path=owned["fit"])
