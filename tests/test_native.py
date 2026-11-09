from dataclasses import replace
import torch
import pytest
from klein_sliders.contracts import EXPECTED, validate_architecture
from klein_sliders.adapter import ParticleAdapter
from klein_sliders.native import KleinResponseAdapter
from conftest import response_row


def test_exact_base_native_geometry_and_cleanup(runtime):
    row = response_row(runtime)
    z = row.latent
    ids = runtime.helper._prepare_latent_ids(z)
    with torch.no_grad():
        direct = runtime.transformer(hidden_states=runtime.helper._pack_latents(z),
            timestep=row.timestep.reshape(1) / 1000, guidance=None,
            encoder_hidden_states=row.neutral["embeds"], txt_ids=row.neutral["text_ids"],
            img_ids=ids, return_dict=False)[0]
        expected = runtime.helper._unpack_latents_with_ids(direct, ids, 4, 4)
    original = sum(len(m._forward_hooks) for m in runtime.transformer.modules())
    adapter = ParticleAdapter(runtime.transformer)
    backend = KleinResponseAdapter(runtime, adapter)
    assert len(adapter.names) == 6  # Four double-stream plus both fused single-stream projections.
    with torch.no_grad():
        for branch in adapter.branches:
            branch.up.weight.fill_(.03)
    assert torch.equal(backend.velocity(row.neutral, z, row.timestep, scale=0), expected)
    assert not torch.equal(backend.velocity(row.neutral, z, row.timestep, scale=1), expected)
    assert adapter.strength == 0
    adapter.close()
    adapter.close()
    assert sum(len(m._forward_hooks) for m in runtime.transformer.modules()) == original


def test_pack_roundtrip_uses_four_axis_ids(runtime):
    z = torch.arange(128 * 3 * 5).reshape(1, 128, 3, 5).float()
    ids = runtime.helper._prepare_latent_ids(z)
    assert ids.shape == (1, 15, 4)
    packed = runtime.helper._pack_latents(z)
    assert packed.shape == (1, 15, 128)
    assert torch.equal(runtime.helper._unpack_latents_with_ids(packed, ids, 3, 5), z)


def test_particle_gradients_and_frozen_base(bound):
    runtime, adapter, backend = bound
    with torch.no_grad():
        for branch in adapter.branches:
            branch.up.weight.normal_(std=.03)
    row = response_row(runtime)
    loss = backend.generated(backend.prepare(row)).square().mean()
    loss.backward()
    assert adapter.particles.grad is not None
    assert torch.isfinite(adapter.particles.grad).all() and adapter.particles.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in runtime.transformer.parameters())


def test_scale_restored_after_failure(bound):
    runtime, adapter, backend = bound
    row = response_row(runtime)
    with pytest.raises(RuntimeError):
        with backend.network.scaled(.7):
            assert adapter.strength == .7
            raise RuntimeError("fixture failure")
    assert adapter.strength == 0 and backend.network.scale == 1


@pytest.mark.parametrize("change", ["time", "channels", "text_ids", "guidance", "schedule"])
def test_invalid_native_input_refused(bound, change):
    runtime, adapter, backend = bound
    row = response_row(runtime)
    if change == "time":
        row = replace(row, timestep=torch.tensor(1001.))
    elif change == "channels":
        row = replace(row, latent=row.latent[:, :4])
    elif change == "text_ids":
        row.neutral["text_ids"] = row.neutral["text_ids"][..., :3]
    elif change == "guidance":
        row = replace(row, guidance=4.)
    else:
        from particle_sliders.pg09_backends import FlowSchedule
        row = replace(row, timestep=None, schedule=FlowSchedule.linear(2, time_scale=1000))
    with pytest.raises(ValueError):
        backend.prepare(row)


def test_exact_model_family_gate():
    validate_architecture(EXPECTED)
    for key in EXPECTED:
        altered = {**EXPECTED, key: None}
        with pytest.raises(ValueError):
            validate_architecture(altered)


def test_checkpoint_recompute_retains_forward_strength(bound):
    from particle_sliders.pg09_backends import bind_native_gradient_checkpointing
    runtime, adapter, backend = bound
    row = response_row(runtime)
    with torch.no_grad():
        for branch in adapter.branches:
            branch.up.weight.fill_(.02)
    first = backend.velocity(row.neutral, row.latent, row.timestep, scale=.5)
    first.square().mean().backward()
    expected = [p.grad.clone() for p in adapter.parameters()]
    adapter.zero_grad(set_to_none=True)
    runtime.transformer.enable_gradient_checkpointing()
    bind_native_gradient_checkpointing(runtime.transformer, backend.network)
    second = backend.velocity(row.neutral, row.latent, row.timestep, scale=.5)
    # Backward runs after the forward context has restored strength 0.
    assert adapter.strength == 0
    second.square().mean().backward()
    assert torch.equal(first, second)
    for parameter, value in zip(adapter.parameters(), expected):
        assert torch.equal(parameter.grad, value)
