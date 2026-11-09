# Klein Particle Sliders

**Experimental source publication.** Historical tiny random native CPU qualification only; no pretrained model training, images, GPU-memory or quality result. This is FLUX.2-klein-base-4B (Apache model identity), not gated Flux3.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Native routed Particle Sliders for **[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)**, the undistilled Apache 2.0 model.

This is a source integration ready for bounded pretrained qualification. **No pretrained training, image quality or GPU-memory result is claimed.** The test suite uses a small, randomly initialized native Klein transformer on CPU. This public source is a historical/experimental EMA candidate: the existing
config uses `ema_decay=.995` and the CLI exports selection `ema`. It is not
RAW-only compliant. Publication does not alter that behavior or qualify it.

The first recipe learns **neutral → soft diffuse lighting** at strengths 0..1. It is unipolar: strength 0 preserves the base model and strength 1 applies the learned adapter. Negative strengths, reference-image edits, rendered image inference and ordinary LoRA/ComfyUI loading are outside this initial qualified interface. The four example prompt pairs are a starter dataset for qualification, not a production training set.

## Architecture

Dependency direction is this product → pinned `particle-sliders-core` → released ParticleGAN 0.9.4. The product owns native model loading, frozen teacher data, Klein latent/position geometry and adapter hooks. The shared core owns all GAN objectives, regularizers, optimizer laws, sampling, EMA, training/resume packets and native inference exports. There is no copied GAN trainer.

Adapters follow Anima's rank-8 routed-branch structure, calling the shared `winning_formulation().bridge()` and exposing the actual 128×4 particle parameter to the shared game's prior. Klein targets are double-stream attention Q/K/V/output and single-stream fused QKV/MLP-input/output projections. Text conditioning and the pretrained transformer stay frozen; adapter leaves and the spatial field critic are FP32. This is a new source/policy cohort, not an adoption of Anima checkpoints or its quality receipts. It uses the core's native `FieldCritic`, not Anima's historical global-mix error critic.

Klein uses Qwen3 intermediate layers 9/18/27, four-axis text/image position IDs, and normalized patchified VAE fields `[1,128,H/16,W/16]`. Native model time is raw scheduler time divided by 1000. Its 128 channels already contain the 2×2 VAE patch: the model boundary only rearranges the spatial lattice into tokens. Generic FLUX.1 packing and pooled CLIP conditioning are incompatible.

Frozen positive teacher velocities are evaluated at the **same latent and time** as the neutral response, along a neutral trajectory advanced by Klein's actual scheduler. The student receives neutral text. Raw native fields go to the shared adversarial game; no hidden-state sketch is presented as an image. The training protocol uses CFG=1 and no generic differentiable trajectory solver. Render and reference editing refuse before model allocation.

## Local setup and checks

Use an isolated Python 3.12 environment. Install a suitable PyTorch build for the selected host, then:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
.venv/bin/klein-slider plan --config configs/diffuse-light.json
```

Dependencies are exact source pins in `pyproject.toml` and `backend.lock.json`. The core is installed from its repository subdirectory, not an assumed PyPI release. CPU checks were executed with Python 3.12.15, Torch 2.13.0+cu126, Transformers 5.22.1 and Diffusers at commit `35087a6`. Other runtime combinations need their own checks.

## Prepare for a bounded GPU qualification

No GPU command has been run by this task. Coordinate admission with your existing training scheduler before cache generation or training. An explicit `cuda:1` in the example is a planned device, not a reservation or evidence it is idle. Preserve existing workloads, their leases and admission rules. This product does not override them or choose a GPU automatically. No live GPU inventory was available in the task sandbox.

1. Download and checksum the exact model snapshot. This is an explicit, potentially large download and does not allocate GPU memory:

   ```bash
   klein-slider stage-model --directory models/klein-base4b
   ```

2. After scheduler admission, first try a bounded low-resolution cache. The text encoder and VAE remain on CPU; the transformer goes on the explicitly chosen device. The normal starter is 512 px, 8 Euler steps, four prompt pairs and two seeds (64 rows). For first qualification use 256 px, 2 steps and one seed (8 rows):

   ```bash
   timeout 10m klein-slider cache \
     --model-dir models/klein-base4b \
     --pairs configs/diffuse-light-pairs.json \
     --output cache/diffuse-light \
     --device cuda:1 --size 256 --steps 2 --seeds 7
   ```

3. Put the command's `cache_index_sha256` into `configs/diffuse-light.json`; use the admitted device. The placeholder deliberately refuses training. Keep model/cache paths and the total step budget fixed when resuming. Then qualify only the first two updates:

   ```bash
   timeout 10m klein-slider train --config configs/diffuse-light.json \
     --output runs/diffuse-light --stop-after 2
   ```

4. Inspect native base parity, finite responses/gradients, prior ownership and peak GPU/host memory. Verify split/resume parity on the actual BF16 pretrained model before admitting longer training. Continuing the same run after admission uses:

   ```bash
   klein-slider train --config configs/diffuse-light.json \
     --output runs/diffuse-light --resume runs/diffuse-light/training.pt
   ```

The CLI emits `metrics.jsonl`, a shared-core `training.pt` packet and an EMA `native-inference.pt`. It checkpoints every 10 updates and at normal exit. A timeout may require resuming the last completed checkpoint. Files remain local; no upload or weight publication code is provided.

The core's source/policy pins, model manifest, cache index/row checksums and adapter target names participate in identities. Replacing source, configuration, data or model bytes is a new cohort, not an implicit checkpoint migration. Checkpoints include native training runtime details; exact CPU/GPU or precision migration is not claimed.

## Native response inference

`native-inference.pt` is a data-independent shared-core v2 export. Its loader constructs only the frozen native transformer and adapter; it does not reconstruct cache rows, a critic or an optimizer:

```python
from particle_sliders.inference import open_inference

with open_inference(
    "runs/diffuse-light/native-inference.pt",
    operation="response",
    options={"model_dir": "models/klein-base4b", "device": "cuda:1", "base_precision": "bfloat16"},
) as slider:
    velocity = slider.response(
        strength=0.5, latent=normalized_native_field,
        embedding=qwen_stacked_embeddings, text_ids=native_text_ids, timestep=500.0,
    )
```

This returns a native velocity field. A production image sampler/decoder and reference-image support require separate parity and quality qualification. Held-out prompts in `configs/heldout-prompts.json` are reserved for that work. Do not treat a response-field test as a rendered-image quality result.

## Sources and licensing

Source code is MIT, with the upstream Anima notice preserved in `LICENSE` and attribution in `NOTICE`. The pretrained model is separately Apache 2.0. Model weights are neither included nor published. This is an independent integration and does not imply Black Forest Labs endorsement.

- [Pinned shared core](https://github.com/HyperGAN/particle-sliders/tree/3e8052250d593a55d9f071cebae503e764342d05/packages/particle-sliders-core)
- [Anima routed-branch reference](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/particles.py)
- [Native Klein pipeline](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)
