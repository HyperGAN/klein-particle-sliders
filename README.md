# Klein Particle Sliders

**Current measured guidance — 2026-11-09:** saved RAW32 **G0.001/G0.0001 tie**,
each **+0.0625** at eight-step/CFG1 and **0.0** in the separate50-step/CFG4
comparison. Both are complete provisional model-instrument results with
`trait_qualified:false`; the50-step instrument saturated on all OFF/ONs.
No unique G, new default, formal trait validation or released weights is claimed.
See [rates, exact scopes and actual how-to](docs/finished-output-learning-rates.md)
and [portable choice/evidence metadata](docs/finished-output-rate.json).
G0.012 remains manually stopped; old results and false-negative records remain.

The retained pretrained cohort used explicit PG0.9.5 scoped initialization and
RAW/noEMA. Its matched native images and separate categorical results supersede
old “not yet run” statements as historical progress, without changing captured
factories or transferring a score/guard between sampling profiles.

## Source integration and historical qualification instructions

**Original publication snapshot:** tiny random native CPU qualification only; later retained execution is scoped in the guide above. This is FLUX.2-klein-base-4B (Apache model identity), not gated Flux3.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Native routed Particle Sliders for **[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)**, the undistilled Apache 2.0 model.

This source integration retains its historical CPU qualification; later pretrained results are documented separately above. The original qualification used a small, randomly initialized native Klein transformer on CPU. This fresh Source cohort requires `ema_decay=None`, refuses any non-None value before model/cache allocation, and exports RAW weights only. The original `.995`/EMA cohort is retained under `configs/historical/`; its old Source, packets and CPU receipt remain historical. They are not migrated or relabeled by this change. The combined PG0.9.5/scoped-init/RAW-render source passed 28 native CPU checks on 2026-11-09 in its matching isolated environment. The old18-test receipt remains historical. See [the current CPU receipt](NATIVE-CPU-20261109.md).

The first recipe learns **neutral → soft diffuse lighting** at strengths 0..1. It is unipolar: strength 0 preserves the base model and strength 1 applies the learned adapter. Negative strengths, reference-image edits and ordinary LoRA/ComfyUI loading are outside this interface. Native finished-image rendering is exposed below and CPU-qualified with tiny random components; pretrained quality remains unqualified. The four example prompt pairs are a starter dataset for qualification, not a production training set.

## Architecture

Dependency direction is this product → pinned `particle-sliders-core` → released ParticleGAN 0.9.5. The product owns native model loading, frozen teacher data, Klein latent/position geometry and adapter hooks. The shared core owns all GAN objectives, regularizers, optimizer laws, sampling, EMA, training/resume packets and native inference exports. There is no copied GAN trainer.

Adapters follow Anima's rank-8 routed-branch structure, calling the shared `winning_formulation().bridge()` and exposing the actual 128×4 particle parameter to the shared game's prior. Klein targets are double-stream attention Q/K/V/output and single-stream fused QKV/MLP-input/output projections. Text conditioning and the pretrained transformer stay frozen; adapter leaves and the spatial field critic are FP32. This is a new source/policy cohort, not an adoption of Anima checkpoints or its quality receipts. It uses the core's native `FieldCritic`, not Anima's historical global-mix error critic.

Klein uses Qwen3 intermediate layers 9/18/27, four-axis text/image position IDs, and normalized patchified VAE fields `[1,128,H/16,W/16]`. Native model time is raw scheduler time divided by 1000. Its 128 channels already contain the 2×2 VAE patch: the model boundary only rearranges the spatial lattice into tokens. Generic FLUX.1 packing and pooled CLIP conditioning are incompatible.

Frozen positive teacher velocities are evaluated at the **same latent and time** as the neutral response, along a neutral trajectory advanced by Klein's actual scheduler. The student receives neutral text. Raw native fields go to the shared adversarial game; no hidden-state sketch is presented as an image. The training protocol uses CFG=1 and no generic differentiable trajectory solver. The training backend is response-only; a separate cache-independent inference backend renders native images. Reference editing refuses before model allocation.

## Local setup and checks

Use an isolated Python 3.12 environment. Install a suitable PyTorch build for the selected host, then:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
.venv/bin/klein-slider plan --config configs/diffuse-light.json
```

Dependencies are exact source pins in `pyproject.toml` and `backend.lock.json`. The core is installed from its repository subdirectory at `4cc412d6ab6d01bb14b5f1e889ea6b79d3a2098d`, not an assumed PyPI release. **The new PG0.9.5/fresh-init/RAW-render cohort passed 28 native CPU checks.** The old18-test receipt used Python3.12.15/Torch2.13/Transformers5.22.1/Diffusers35087 with PG0.9.4 and EMA; it remains historical. The new source retains the exact Diffusers35087 guards; its actual isolated qualification used the dependencies recorded in [NATIVE-CPU-20261109.md](NATIVE-CPU-20261109.md).

## Historical first-qualification commands (do not repeat completed work)

This section records the original source-publication plan; actual retained work is now linked above. Coordinate admission with your existing training scheduler before cache generation or training. An explicit `cuda:1` in the example is a planned device, not a reservation or evidence it is idle. Preserve existing workloads, their leases and admission rules. This product does not override them or choose a GPU automatically. No live GPU inventory was available in the task sandbox.

1. Download and checksum the exact model snapshot. This is an explicit, potentially large download and does not allocate GPU memory:

   ```bash
   klein-slider stage-model --directory models/klein-base4b
   ```

2. After scheduler admission, first try a bounded low-resolution cache. The text encoder and VAE remain on CPU; the transformer goes on the explicitly chosen device. The normal starter is 512 px, 8 Euler steps, four prompt pairs and two seeds (64 rows). For first qualification use 256 px, 2 steps and one seed (8 rows):

   ```bash
   klein-slider cache \
     --model-dir models/klein-base4b \
     --pairs configs/diffuse-light-pairs.json \
     --output cache/diffuse-light \
     --device cuda:1 --size 256 --steps 2 --seeds 7
   ```

3. Put the command's `cache_index_sha256` into `configs/diffuse-light.json`; use the admitted device. The placeholder deliberately refuses training. Keep model/cache paths and the total step budget fixed when resuming. Then qualify only the first two updates:

   ```bash
   klein-slider train --config configs/diffuse-light.json \
     --output runs/diffuse-light --stop-after 2
   ```

4. Inspect native base parity, finite responses/gradients, prior ownership and peak GPU/host memory. The tiny native CPU tests already verify split/resume parity. Check actual pretrained outputs and resource use before admitting longer training; do not replay a completed prefix for another proof. Continuing the same run after admission uses:

   ```bash
   klein-slider train --config configs/diffuse-light.json \
     --output runs/diffuse-light --resume runs/diffuse-light/training.pt
   ```

The CLI emits `metrics.jsonl`, a shared-core `training.pt` packet and a RAW `native-inference.pt`. It checkpoints every 10 updates and at normal exit. Resume only its exact latest complete FULL training checkpoint; RAW exports are inference-only; there is no elapsed training deadline. Files remain local; no upload or weight publication code is provided.

The explicit [fresh initialization profile](FRESH-PG095-INITIALIZATION.md) uses the published shared helper before optimizer construction: named Kaiming-RMS QR for declared added adapter/field-critic leaves, exactly zero up weights retained, and R2Normal(std1) for the existing128×4 bank. This is a new experimental training recipe, not an improvement claim or reuse of an old LR optimum. `klein-slider train` and product `open_training(config, resume=...)` bind fresh/restore intent before the factory; exact resume skips initialization and retains the original receipt. Direct generic `BACKEND.open` without that explicit intent refuses. Native inference loads RAW and never calls the initializer. No universal initialization CLI flag is introduced. Native CPU checks and the tiny native rendered-image test pass. Later pretrained execution is recorded in the current guide; it does not change these CPU checks into a quality proof.

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

This returns a native velocity field. The separate finished-image renderer below uses the native pipeline and decoder. Pretrained quality and reference-image support remain unqualified. Held-out prompts in `configs/heldout-prompts.json` are reserved for that work. Do not treat a response-field test as a rendered-image quality result.

## Sources and licensing

Source code is MIT, with the upstream Anima notice preserved in `LICENSE` and attribution in `NOTICE`. The pretrained model is separately Apache 2.0. Model weights are neither included nor published. This is an independent integration and does not imply Black Forest Labs endorsement.

- [Pinned shared core](https://github.com/HyperGAN/particle-sliders/tree/4cc412d6ab6d01bb14b5f1e889ea6b79d3a2098d/packages/particle-sliders-core)
- [Anima routed-branch reference](https://github.com/HyperGAN/anima-particle-sliders/blob/c018f32759f49a05c2360a04b4a83924a6851913/lumen_studio/particles.py)
- [Native Klein pipeline](https://github.com/huggingface/diffusers/blob/35087a644038eb531f48581708d4763044cb4852/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py)

## Native finished-image rendering

The `render` command now has a cache-independent RAW-inference path that calls
this repository's exact pinned `Flux2KleinPipeline` and genuinely decodes a PIL
image. It uses the shared native export validator and strict network binding;
no game, critic, optimizer, teacher cache or initialization hook is reconstructed.
The training backend remains response/cache-only (`render=False`); the separate
native inference backend advertises its finished PIL callback for **new exports**.
Old source-bound exports remain tied to their original factory. They are not
normalized or relabeled to make them load with changed code.

```sh
klein-slider render --model-dir /absolute/staged/klein4b \
  --export /absolute/new-cohort/native-inference.pt --output /absolute/fresh/render \
  --prompt 'A watering can on a garden table' --seed 7 --size 256 \
  --steps 8 --strength 1 --device cuda:0 --base-precision bfloat16
```

This produces matched `OFF.png` (strength0) and `ON.png` (requested strength) from
one loaded RAW inference session. It records exact artifact bytes/internal
identity, original specification/cohort, prompt/native settings and both PNG
hashes. Both use independently constructed generators with the same seed. The
native scheduler is copied/restored per output, and the adapter strength and
common observation state are restored even if a call fails. Completed files and
an explicit failed receipt remain if the second output fails; there is no retry.

Only256/512 dimensions,2..50 native steps and unipolar strengths0..1 are exposed.
CFG remains1, text length128, Qwen3 layers9/18/27, no reference images/FLUX1
packing/guidance embedding. The original public pipeline owns four-axis IDs,
empirical-mu Euler schedule, BN inversion, unpatchification, VAE decode and image
postprocessing. A native hidden response is never presented as a finished image.
The renderer Source hash is part of verified model identity and checked again
before lazy loading. The pipeline reuses the **same already-bound transformer**;
only local text/VAE components are loaded on first render.

Rendering deliberately calls the public pipeline `.to(selected_device)` for
all three components. Budget roughly16GB BF16 model weights plus live activations
and adapter, rather than only the DiT. There is no CPU offload/device proxy or
second base transformer. Root must admit the actual full render memory profile.
This command does not acquire its own host/GPU resource authority.

**Original CPU qualification boundary:** later retained pretrained rendering has the separate scopes in the current guide. The 28 current CPU checks pass, including
`tests/test_native_render.py` with the actual pinned pipeline, random tiny
DiT/VAE/Qwen3 and a synthetic tokenizer. This establishes native software
conformance for the combined source, not pretrained image quality or memory
fit. Historical packets stay source-bound; there is no migration or automatic
training permission.
