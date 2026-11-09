"""Native Klein velocity geometry. No FLUX.1 packing, CFG embedding or pooled CLIP."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import torch
from particle_sliders.pg09_backends import (AdapterParameters, FrozenResponseAdapter,
                                          bind_native_gradient_checkpointing)
from .contracts import PROTOCOL, DIFFUSERS_REVISION, NATIVE_SOURCE_HASHES, file_hash, verify_model


class NativeRuntime:
    def __init__(self, directory, *, device="cpu", precision="bfloat16", encoder=False):
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        import inspect
        sources = {Path(inspect.getsourcefile(cls)).name: file_hash(inspect.getsourcefile(cls))
                   for cls in (Flux2KleinPipeline, Flux2Transformer2DModel)}
        if sources != NATIVE_SOURCE_HASHES:
            raise ValueError("installed native Klein source differs from the pinned Diffusers revision")
        self.identity = verify_model(directory)
        self.identity["native_sources"] = sources
        from importlib.metadata import version
        self.identity["runtime_versions"] = {name: version(name) for name in
                                             ("torch", "diffusers", "transformers", "accelerate")}
        self.device = torch.device(device)
        self.dtype = getattr(torch, precision)
        self.pipe = None
        self.helper = Flux2KleinPipeline
        if encoder:
            # Text encoder and VAE stay on CPU. Only the DiT occupies the selected GPU.
            self.pipe = Flux2KleinPipeline.from_pretrained(directory, torch_dtype=self.dtype,
                                                          local_files_only=True, is_distilled=False)
            for module in (self.pipe.transformer, self.pipe.text_encoder, self.pipe.vae):
                module.requires_grad_(False).eval()
            self.transformer = self.pipe.transformer
        else:
            self.transformer = Flux2Transformer2DModel.from_pretrained(
                directory, subfolder="transformer", torch_dtype=self.dtype, local_files_only=True)
            self.transformer.requires_grad_(False).eval()
        self.transformer.to(self.device)

    @torch.no_grad()
    def encode(self, text):
        if self.pipe is None:
            raise RuntimeError("encoding requires a separately loaded frozen text encoder")
        embeds, ids = self.pipe.encode_prompt(text, device="cpu", max_sequence_length=128)
        return dict(embeds=embeds.detach(), text_ids=ids.detach())

    def close(self):
        self.pipe = None
        self.transformer = None


class KleinResponseAdapter(FrozenResponseAdapter):
    family = "flux2_klein_base4b"
    time_coordinates = "native raw scheduler t in [0,1000]; transformer t=raw_t/1000"

    def __init__(self, runtime, adapter):
        if any(p.requires_grad for p in runtime.transformer.parameters()):
            raise ValueError("native base must remain frozen")
        self.runtime = runtime
        self.adapter = adapter
        self.transformer = runtime.transformer
        network = AdapterParameters(adapter.named_parameters(), adapter.scaled)
        super().__init__(network, model_identity=runtime.identity)
        bind_native_gradient_checkpointing(self.transformer, network)

    def _velocity(self, condition, latent, timestep):
        if (not isinstance(condition, dict) or set(condition) != {"embeds", "text_ids"}
                or latent.ndim != 4 or latent.shape[0] != 1
                or latent.shape[1] != self.transformer.config.in_channels
                or not latent.is_floating_point() or not torch.isfinite(latent).all()):
            raise ValueError("requires finite native normalized [1,128,H,W] field and Klein encoding")
        if timestep.numel() != 1 or not torch.isfinite(timestep).all() or not 0 <= float(timestep) <= 1000:
            raise ValueError("raw native timestep must be a scalar in [0,1000]")
        embeds, text_ids = condition["embeds"], condition["text_ids"]
        if (embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[1] < 1
                or embeds.shape[-1] != self.transformer.config.joint_attention_dim
                or text_ids.shape != (*embeds.shape[:2], 4)
                or not torch.isfinite(embeds).all() or not torch.isfinite(text_ids).all()):
            raise ValueError("invalid Qwen3 stacked embeddings or 4-axis text position IDs")
        z = latent.to(device=self.runtime.device, dtype=self.runtime.dtype)
        helper = self.runtime.helper
        packed = helper._pack_latents(z)
        image_ids = helper._prepare_latent_ids(z).to(z.device)
        out = self.transformer(
            hidden_states=packed, timestep=timestep.reshape(1).to(z) / 1000,
            guidance=None, encoder_hidden_states=embeds.to(z), txt_ids=text_ids.to(z.device),
            img_ids=image_ids, return_dict=False,
        )[0]
        if out.shape != packed.shape:
            raise ValueError("native Klein output token geometry differs")
        return helper._unpack_latents_with_ids(out, image_ids, z.shape[2], z.shape[3]).float()

    def prepare(self, row):
        if row.schedule is not None or row.guidance != 1 or row.scale != 1 or row.strength != 1:
            raise ValueError("initial Klein protocol is unipolar native velocity, CFG=1; no generic trajectory solver")
        return super().prepare(row)

    def metadata(self):
        return dict(super().metadata(), protocol=PROTOCOL, diffusers_revision=DIFFUSERS_REVISION,
                    particle_bank="actual product ParticleAdapter.particles", targets=list(self.adapter.names),
                    response_coordinates="BN-normalized patchified VAE field [1,128,H/16,W/16]",
                    image_conditioning=False, kv_cache=False, guidance_embedding=False,
                    fresh_initialization=getattr(self, "fresh_initialization", None))


@contextmanager
def observation(runtime, adapter):
    """Own native modes/buffers/strengths while core owns trainables and RNG."""
    from particle_sliders.lifecycle import native_cleanup
    restore = native_cleanup(runtime.transformer)
    old = adapter.strength
    buffers = [(b, b.detach().clone()) for b in runtime.transformer.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        adapter.strength = old
        restore()
