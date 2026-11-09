import json
from pathlib import Path
import torch
import pytest
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from klein_sliders.cache import write_row, CachedResponses
from klein_sliders.contracts import PROTOCOL, WEIGHT_POLICY, INITIALIZATION_POLICY, file_hash
from klein_sliders.lifecycle import BACKEND, open_training
from conftest import TinyRuntime, response_row


def klein_config(*args, **options):
    return SessionConfig(*args, **(dict(pg_version="0.9.5", pg_source_profile="release-0.9.5",
                                      initialization_policy=INITIALIZATION_POLICY) | options))


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def make_cache(tmp_path, bound):
    runtime, adapter, backend = bound
    root = tmp_path / "cache"
    root.mkdir()
    rows = [write_row(root, i, backend.prepare(response_row(runtime, i))) for i in range(2)]
    index = dict(protocol=PROTOCOL, model=runtime.identity, precision=str(runtime.dtype),
                 response_shape=[1, 8, 4, 4], rows=rows, split="train")
    (root / "index.json").write_text(json.dumps(index))
    return root, file_hash(root / "index.json")


def test_native_core_training_resume_and_export(tmp_path, bound):
    root, sha = make_cache(tmp_path, bound)
    config = klein_config("flux2_klein", "tiny-cpu-only-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=2, batch=2, seed=7, critic_width=4, ema_decay=None,
        lr=.004, d_lr_mult=4.500000000000001, prior_lr_mult=7.499999999999999)
    before = torch.get_rng_state().clone()
    full_runtime, split_runtime, resume_runtime = TinyRuntime(), TinyRuntime(), TinyRuntime()
    sessions = []
    try:
        full = open_training(config, runtime=full_runtime); sessions.append(full)
        split = open_training(config, runtime=split_runtime); sessions.append(split)
        assert full.game.prior.z is full.game.network.particles
        assert full.game._signature["generator_arm"] == "routed_particle_gan"
        assert full.game._signature["pg_version"] == "0.9.5"
        overrides = {key: config.options[key] for key in ("lr", "d_lr_mult", "prior_lr_mult")}
        assert all(getattr(full.game.recipe, key) == value for key, value in overrides.items())
        assert [group["lr"] for group in full.game.opt_g.param_groups] == [
            overrides["lr"], overrides["lr"] * overrides["prior_lr_mult"]]
        assert {group["lr"] for group in full.game.opt_d.param_groups} == {
            overrides["lr"] * overrides["d_lr_mult"]}
        full.train()
        split.train(stop_after=1)
        resumed = open_training(config, resume=split.checkpoint(), runtime=resume_runtime); sessions.append(resumed)
        saved = resumed.checkpoint()
        assert torch.equal(resumed.evaluate(strength=0), resumed.game.rows[0].neutral)
        resumed.evaluate(strength=.5)
        equal(saved, resumed.checkpoint())
        resumed.train()
        equal(full.checkpoint(), resumed.checkpoint())
        packet = resumed.export_inference(selection="raw")
        assert resumed.ema.decay is None and resumed.ema.values is None
        assert resumed.checkpoint()["payload"]["ema"] == {"decay": None, "values": None}
        assert packet["selection"] == "raw"
        assert packet["specification"]["configuration"]["weight_policy"] == WEIGHT_POLICY
        assert packet["specification"]["configuration"]["initialization_policy"] == INITIALIZATION_POLICY
        with pytest.raises(ValueError):
            resumed.export_inference(selection="ema")
        with pytest.raises(ValueError):
            resumed.export(selection="ema")
        assert packet["schema"] == "particle-sliders-native-inference-v2"
        assert torch.equal(before, torch.get_rng_state())
    finally:
        for session in sessions:
            session.close()
    assert all(not m.is_gradient_checkpointing for m in (full_runtime.transformer,
               split_runtime.transformer, resume_runtime.transformer))
    assert all(not module._forward_hooks for module in full_runtime.transformer.modules())


def test_cache_corruption_refused_before_training(tmp_path, bound):
    root, sha = make_cache(tmp_path, bound)
    rows = CachedResponses(root, bound[0], sha)
    path = root / rows.records[0]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        rows[0]


def test_cache_path_escape_refused(tmp_path, bound):
    root, _ = make_cache(tmp_path, bound)
    path = root / "index.json"
    index = json.loads(path.read_text())
    index["rows"][0]["path"] = "../outside.pt"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="path/checksum"):
        CachedResponses(root, bound[0], file_hash(path))


def test_distinct_native_context_collisions_refused(tmp_path, bound):
    root, _ = make_cache(tmp_path, bound)
    path = root / "index.json"
    index = json.loads(path.read_text())
    index["rows"][1]["condition"] = index["rows"][0]["condition"]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="colliding"):
        CachedResponses(root, bound[0], file_hash(path))


def test_render_and_reference_editing_refused_before_runtime():
    config = klein_config("flux2_klein", "unqualified", model_dir="missing", cache_dir="missing")
    with pytest.raises(UnsupportedOperation):
        BACKEND.open(config, operation="render", references=True)


def test_native_inference_roundtrip_needs_no_cache_or_game(tmp_path, bound, monkeypatch):
    from particle_sliders.inference import open_inference
    import klein_sliders.lifecycle as product
    root, sha = make_cache(tmp_path, bound)
    config = klein_config("flux2_klein", "tiny-cpu-inference-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=1, batch=2, critic_width=4)
    row = response_row(bound[0])
    with open_training(config, runtime=TinyRuntime()) as training:
        training.train()
        expected = training.evaluate(strength=.5)
        artifact = training.export_inference()
    def no_game(*args, **kwargs):
        raise AssertionError("native-only inference must not reconstruct training/cache")
    monkeypatch.setattr(product, "NativeRuntime", lambda *a, **kw: TinyRuntime())
    monkeypatch.setattr(product, "CachedResponses", no_game)
    monkeypatch.setattr(product, "ConditionalPG095", no_game)
    import klein_sliders.initialization as initialization
    monkeypatch.setattr(initialization, "initialize_fresh_components", no_game)
    with open_inference(artifact, backend=product.NATIVE_BACKEND, operation="response",
                        options=dict(model_dir=str(tmp_path), device="cpu", base_precision="float32")) as inference:
        value = inference.response(strength=.5, latent=row.latent, embedding=row.neutral["embeds"],
                                   text_ids=row.neutral["text_ids"], timestep=500.)
        assert torch.equal(value, expected)


@pytest.mark.parametrize("decay", [0., .9, .995, False])
def test_weight_ema_refused_before_native_allocation(tmp_path, monkeypatch, decay):
    import klein_sliders.lifecycle as product
    def no_runtime(*args, **kwargs):
        raise AssertionError("policy refusal must precede native allocation")
    monkeypatch.setattr(product, "NativeRuntime", no_runtime)
    with pytest.raises(ValueError):
        config = klein_config("flux2_klein", "historical-EMA-not-RAW", ema_decay=decay,
                              model_dir=str(tmp_path), cache_dir="absent")
        open_training(config)
    with pytest.raises(ValueError, match="ema_decay=None"):
        product.prepare(dict(ema_decay=decay))


def test_ema_state_cannot_load_into_raw_session(tmp_path, bound):
    from copy import deepcopy
    from particle_sliders.lifecycle import CheckpointError
    from particle_sliders.pg09 import _hash
    root, sha = make_cache(tmp_path, bound)
    config = klein_config("flux2_klein", "tiny-fresh-RAW-packet-test", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=2, batch=2, critic_width=4, ema_decay=None)
    with open_training(config, runtime=TinyRuntime()) as session:
        session.train(stop_after=1)
        before = session.checkpoint()
        # Explicit malformed-state fixture, not a fabricated historical run receipt.
        incompatible = deepcopy(before)
        incompatible["payload"]["ema"] = dict(decay=.995,
                                               values=deepcopy(session.game.network.state_dict()))
        incompatible["sha256"] = _hash({key: value for key, value in incompatible.items() if key != "sha256"})
        with pytest.raises(CheckpointError):
            session.restore(incompatible)
        equal(before, session.checkpoint())


def test_historical_native_spec_refused_before_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import klein_sliders.lifecycle as product
    def no_runtime(*args, **kwargs):
        raise AssertionError("historical specification must be refused before native allocation")
    monkeypatch.setattr(product, "NativeRuntime", no_runtime)
    old_spec = SimpleNamespace(identity={"protocol": PROTOCOL}, configuration={})
    with pytest.raises(ValueError, match="RAW-only"):
        product.prepare_inference(dict(model_dir=str(tmp_path), device="cpu"), old_spec)


def test_fresh_receipt_zero_delta_and_resume_never_initializes(tmp_path, bound, monkeypatch):
    import klein_sliders.initialization as initialization
    root, sha = make_cache(tmp_path, bound)
    config = klein_config("flux2_klein", "new-fresh-receipt-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=2, batch=2, critic_width=4)
    base = TinyRuntime()
    base_values = {n: p.detach().clone() for n, p in base.transformer.named_parameters()}
    with open_training(config, runtime=base) as session:
        adapter = session.game.backend.adapter
        assert all(torch.count_nonzero(branch.up.weight) == 0 for branch in adapter.branches)
        assert all(torch.count_nonzero(branch.down.weight) > 0 for branch in adapter.branches)
        assert all(torch.equal(p, base_values[n]) and not p.requires_grad
                   for n, p in base.transformer.named_parameters())
        assert session.game.prior.z is adapter.particles
        receipt = session.game.signature["backend"]["fresh_initialization"]
        assert receipt["source_identity"]["initialization_policy"] == INITIALIZATION_POLICY
        assert receipt["components"]["adapter.bank"]["parameters"]["particles"]["host_declaration"] == {
            "kind": "R2Normal", "mean": 0.0, "std": 1.0}
        row = session.game.rows[0].row
        with adapter.scaled(1):
            fresh_on = session.game.backend._velocity(row.neutral, row.latent, row.timestep)
        with adapter.scaled(0):
            fresh_off = session.game.backend._velocity(row.neutral, row.latent, row.timestep)
        assert torch.equal(fresh_on, fresh_off)
        session.train(stop_after=1)
        packet = session.checkpoint()
    def refuse_initializer(*args, **kwargs):
        raise AssertionError("fresh initializer must not run while restoring or inferring")
    monkeypatch.setattr(initialization, "initialize_fresh_components", refuse_initializer)
    with open_training(config, resume=packet, runtime=TinyRuntime()) as resumed:
        equal(packet, resumed.checkpoint())
        resumed.train()
        assert resumed.game.completed_steps == 2


def test_missing_intent_and_old_source_refused_before_native_allocation(tmp_path, monkeypatch):
    import klein_sliders.lifecycle as product
    def refuse_native(*args, **kwargs):
        raise AssertionError("invalid intent/profile must refuse before native allocation")
    monkeypatch.setattr(product, "NativeRuntime", refuse_native)
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "index.json").write_text("{}")
    config = klein_config("flux2_klein", "fresh-v1", model_dir=str(tmp_path), cache_dir=str(cache),
                          cache_index_sha256=file_hash(cache / "index.json"), device="cpu")
    with pytest.raises(ValueError, match="intent"):
        BACKEND.open(config)
    with pytest.raises(ValueError, match="open_training"):
        product.prepare(config.options)
    historical = SessionConfig("flux2_klein", "old094", model_dir=str(tmp_path), cache_dir=str(cache))
    with pytest.raises(ValueError, match="new PG0.9.5"):
        open_training(historical)


def test_initialization_failure_unhooks_borrowed_native_without_optimizer(tmp_path, bound, monkeypatch):
    import klein_sliders.lifecycle as product
    import klein_sliders.initialization as initialization
    root, sha = make_cache(tmp_path, bound)
    config = klein_config("flux2_klein", "fresh-initializer-refusal-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=2, batch=2, critic_width=4)
    runtime = TinyRuntime()
    before = {n: p.detach().clone() for n, p in runtime.transformer.named_parameters()}
    def fail_initializer(*args, **kwargs):
        raise ValueError("declared staging failure")
    def no_optimizer_game(*args, **kwargs):
        raise AssertionError("game/optimizers must not be constructed after init failure")
    monkeypatch.setattr(initialization, "initialize_fresh_components", fail_initializer)
    monkeypatch.setattr(product, "ConditionalPG095", no_optimizer_game)
    with pytest.raises(ValueError, match="staging failure"):
        open_training(config, runtime=runtime)
    assert all(not module._forward_hooks for module in runtime.transformer.modules())
    assert not runtime.transformer.is_gradient_checkpointing
    assert all(torch.equal(p, before[n]) for n, p in runtime.transformer.named_parameters())
