import json
from pathlib import Path
import torch
import pytest
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from klein_sliders.cache import write_row, CachedResponses
from klein_sliders.contracts import PROTOCOL, file_hash
from klein_sliders.lifecycle import BACKEND
from conftest import TinyRuntime, response_row


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
    config = SessionConfig("flux2_klein", "tiny-cpu-only-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=2, batch=2, seed=7, critic_width=4, ema_decay=.9)
    before = torch.get_rng_state().clone()
    full_runtime, split_runtime, resume_runtime = TinyRuntime(), TinyRuntime(), TinyRuntime()
    sessions = []
    try:
        full = BACKEND.open(config, runtime=full_runtime); sessions.append(full)
        split = BACKEND.open(config, runtime=split_runtime); sessions.append(split)
        resumed = BACKEND.open(config, runtime=resume_runtime); sessions.append(resumed)
        assert full.game.prior.z is full.game.network.particles
        assert full.game._signature["generator_arm"] == "routed_particle_gan"
        assert full.game._signature["pg_version"] == "0.9.4"
        full.train()
        split.train(stop_after=1)
        resumed.restore(split.checkpoint())
        saved = resumed.checkpoint()
        assert torch.equal(resumed.evaluate(strength=0), resumed.game.rows[0].neutral)
        resumed.evaluate(strength=.5)
        equal(saved, resumed.checkpoint())
        resumed.train()
        equal(full.checkpoint(), resumed.checkpoint())
        packet = resumed.export_inference(selection="ema")
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
    config = SessionConfig("flux2_klein", "unqualified", model_dir="missing", cache_dir="missing")
    with pytest.raises(UnsupportedOperation):
        BACKEND.open(config, operation="render", references=True)


def test_native_inference_roundtrip_needs_no_cache_or_game(tmp_path, bound, monkeypatch):
    from particle_sliders.inference import open_inference
    import klein_sliders.lifecycle as product
    root, sha = make_cache(tmp_path, bound)
    config = SessionConfig("flux2_klein", "tiny-cpu-inference-v1", model_dir=str(tmp_path),
        cache_dir=str(root), cache_index_sha256=sha, device="cpu", base_precision="float32",
        steps=1, batch=2, critic_width=4)
    row = response_row(bound[0])
    with BACKEND.open(config, runtime=TinyRuntime()) as training:
        training.train()
        expected = training.evaluate(strength=.5)
        artifact = training.export_inference()
    def no_game(*args, **kwargs):
        raise AssertionError("native-only inference must not reconstruct training/cache")
    monkeypatch.setattr(product, "NativeRuntime", lambda *a, **kw: TinyRuntime())
    monkeypatch.setattr(product, "CachedResponses", no_game)
    monkeypatch.setattr(product, "ConditionalPG094", no_game)
    with open_inference(artifact, backend=product.NATIVE_BACKEND, operation="response",
                        options=dict(model_dir=str(tmp_path), device="cpu", base_precision="float32")) as inference:
        value = inference.response(strength=.5, latent=row.latent, embedding=row.neutral["embeds"],
                                   text_ids=row.neutral["text_ids"], timestep=500.)
        assert torch.equal(value, expected)
