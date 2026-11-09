from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from klein_sliders.adapter import ParticleAdapter
from klein_sliders.native import KleinResponseAdapter
from particle_sliders.pg09_backends import ResponseRow


class TinyRuntime:
    """Actual native Klein transformer with random tiny weights; never pretrained evidence."""
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.helper = Flux2KleinPipeline
        self.identity = {"qualification": "random tiny CPU Klein fixture only"}
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(42)
            self.transformer = Flux2Transformer2DModel(in_channels=8, joint_attention_dim=12,
                num_layers=1, num_single_layers=1, num_attention_heads=2, attention_head_dim=8,
                axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False, mlp_ratio=2.)
        self.transformer.requires_grad_(False).eval()

    def close(self):
        self.transformer = None


@pytest.fixture(autouse=True)
def bounded_cpu():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)
    assert not torch.cuda.is_initialized()


@pytest.fixture
def runtime():
    return TinyRuntime()


@pytest.fixture
def bound(runtime):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        adapter = ParticleAdapter(runtime.transformer)
    backend = KleinResponseAdapter(runtime, adapter)
    yield runtime, adapter, backend
    adapter.close()


def response_row(runtime, index=0):
    generator = torch.Generator().manual_seed(19 + index)
    embeds = torch.randn(1, 4, 12, generator=generator)
    neutral = dict(embeds=embeds, text_ids=runtime.helper._prepare_text_ids(embeds))
    positive = dict(embeds=embeds + .3, text_ids=neutral["text_ids"].clone())
    return ResponseRow(neutral, positive, neutral, torch.randn(1, 8, 4, 4, generator=generator),
                       timestep=torch.tensor(500. + index * 20))
