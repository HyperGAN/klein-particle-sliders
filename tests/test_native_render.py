"""Real pinned native pipeline API with RANDOM TINY components, CPU only.

Authored for a separately admitted matching environment. No pretrained weights,
network calls, downloads, quality/effect result or genuine native qualification.
"""
from types import SimpleNamespace
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from transformers import Qwen3Config, Qwen3ForCausalLM
from klein_sliders.adapter import ParticleAdapter
from klein_sliders.render import render_image, native_pipeline_state
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
