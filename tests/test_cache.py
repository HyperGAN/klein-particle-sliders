import json
import torch
from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
from klein_sliders.cache import CachedResponses, prepare_cache


def test_tiny_cache_uses_actual_klein_pipeline_geometry_and_scheduler(tmp_path, bound):
    runtime, adapter, backend = bound
    runtime.pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=None, text_encoder=None, tokenizer=None, transformer=runtime.transformer, is_distilled=False)
    generator = torch.Generator().manual_seed(31)
    embeddings = torch.randn(1, 4, 12, generator=generator)
    def encode(text):
        value = embeddings + (.3 if text.startswith("positive") else 0)
        return dict(embeds=value, text_ids=runtime.helper._prepare_text_ids(value))
    runtime.encode = encode
    pairs = [dict(neutral="neutral a", positive="positive a"),
             dict(neutral="neutral b", positive="positive b")]
    sha = prepare_cache(runtime, adapter, pairs, tmp_path / "teachers", seeds=[7], size=64, steps=2)
    rows = CachedResponses(tmp_path / "teachers", runtime, sha)
    assert len(rows) == 4
    assert rows.response_shape == (1, 8, 4, 4)
    row = rows[0]
    assert torch.equal(row.neutral, backend.velocity(row.row.neutral, row.row.latent, row.row.timestep, scale=0))
    assert torch.equal(row.target, backend.velocity(row.row.positive, row.row.latent, row.row.timestep, scale=0))
    assert not torch.equal(row.target, row.neutral)
    assert all(not parameter.requires_grad for parameter in runtime.transformer.parameters())
