# Klein Particle Sliders

**Memory (calculated):** native BF16 batch2 training at512²: **16 GiB VRAM /24 GiB RAM**; finished sampling: **24 GiB VRAM /32 GiB RAM**. See [MEMORY.md](MEMORY.md) for the complete budgets and assumptions.





**Current measured guidance — 2026-11-09:** saved RAW32 **G0.001/G0.0001 tie**,
each **+0.0625** at eight-step/CFG1 and **0.0** in the separate50-step/CFG4
comparison. Both are complete provisional model-instrument results with
`trait_qualified:false`; the50-step instrument saturated on all OFF/ONs.
No unique G, new default, formal trait validation or released weights is claimed.
See [rates, exact scopes and actual how-to](docs/finished-output-learning-rates.md)
and [portable choice/evidence metadata](https://github.com/HyperGAN/klein-particle-sliders/blob/98e13bea7ed14a3ee2477ea9eea74d5cec79c298/docs/finished-output-rate.json).
G0.012 remains manually stopped; old results and false-negative records remain.

The retained pretrained cohort used explicit the archived ParticleGAN release scoped initialization and
RAW/noEMA. Its matched native images and separate categorical results supersede
old “not yet run” statements as historical progress, without changing captured
factories or transferring a score/guard between sampling profiles.

## Source integration and historical qualification instructions

**Original publication snapshot:** tiny random native CPU qualification only; later retained execution is scoped in the guide above. This is FLUX.2-klein-base-4B (Apache model identity), not gated Flux3.

See [PUBLICATION.md](PUBLICATION.md) for sanitation, lineage and current CPU/CI limits.


Native routed Particle Sliders for **[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)**, the undistilled Apache 2.0 model.

This source integration retains its historical CPU qualification; later pretrained results are documented separately above. The original qualification used a small, randomly initialized native Klein transformer on CPU. This fresh Source cohort requires `ema_decay=None`, refuses any non-None value before model/cache allocation, and exports RAW weights only. The original `.995`/EMA cohort is retained under `configs/historical/`; its old Source, packets and CPU receipt remain historical. They are not migrated or relabeled by this change. The combined the archived ParticleGAN release/scoped-init/RAW-render source passed 28 native CPU checks on 2026-11-09 in its matching isolated environment. The old18-test receipt remains historical. See [the November 9 CPU receipt](NATIVE-CPU-20261109.md).

The first recipe learns **neutral → soft diffuse lighting** at strengths 0..1. It is unipolar: strength 0 preserves the base model and strength 1 applies the learned adapter. Negative strengths, reference-image edits and ordinary LoRA/ComfyUI loading are outside this interface. Native finished-image rendering is exposed below and CPU-qualified with tiny random components; pretrained quality remains unqualified. The four example prompt pairs are a starter dataset for qualification, not a production training set.

## Architecture

Dependency direction is this product → pinned `particle-sliders-core` → released ParticleGAN 0.11.1. The product owns native model loading, frozen teacher data, Klein latent/position geometry and adapter hooks. The shared core owns all GAN objectives, regularizers, optimizer laws, sampling, raw-weight training/resume packets and native inference exports. There is no copied GAN trainer.

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

Dependencies are exact source pins in `pyproject.toml` and `backend.lock.json`. The core is installed from its repository subdirectory at `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90`, not an assumed PyPI release. Current source uses an explicitly fresh ParticleGAN 0.11.1 cohort with RAW weights. The November 9 the archived ParticleGAN release cohort passed 28 native CPU checks. The old18-test receipt used Python3.12.15/Torch2.13/Transformers5.22.1/Diffusers35087 with the archived ParticleGAN release and EMA; it remains historical. The new source retains the exact Diffusers35087 guards; the earlier environment is recorded in [NATIVE-CPU-20261109.md](NATIVE-CPU-20261109.md).

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

The explicit [fresh initialization profile](FRESH-INITIALIZATION.md) uses the published shared helper before optimizer construction: named Kaiming-RMS QR for declared added adapter/field-critic leaves, exactly zero up weights retained, and R2Normal(std1) for the existing128×4 bank. This is a new experimental training recipe, not an improvement claim or reuse of an old LR optimum. `klein-slider train` and product `open_training(config, resume=...)` bind fresh/restore intent before the factory; exact resume skips initialization and retains the original receipt. Direct generic `BACKEND.open` without that explicit intent refuses. Native inference loads RAW and never calls the initializer. No universal initialization CLI flag is introduced. Native CPU checks and the tiny native rendered-image test pass. Later pretrained execution is recorded in the current guide; it does not change these CPU checks into a quality proof.

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

- [Pinned shared core](https://github.com/HyperGAN/particle-sliders/tree/ec1161f5d74481be3540cb0dfa63d0e8f8b54e90/packages/particle-sliders-core)
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

Pass an open RAW Python session to `klein_sliders.scheduled_render.render`
with `step_gains` and the original render arguments. Supply one gain per native
denoiser step, with every `strength * gain` in `[0, 1]`. The image exposes the
actual sigma grid and applied strengths in
`image.info["strength_schedule"]`. Constant gains matched the original
finished pixels in the tiny native CPU fixture.

On the archived Final Boss RAW64 profile, constant strength 1.0 led the
eight-case held-out comparison at +0.01083 mean finished-image gain.
At the same 2/3 nominal strength, late-heavy scored −0.01708, constant
scored −0.01917, and early-heavy scored −0.02500. All 40 images passed
independent integrity, and the eight winning outputs reproduced byte
for byte. A follow-up at equal mean strength 0.90 averaged −0.00750
constant, −0.00583 early-heavy and −0.00208 late-heavy on the same cases.
All 24 new images passed integrity. Constant strength 1.0 remains the
best tested profile. [Per-case evidence](evidence/final-boss-raw64-scheduled-strength.json)

A separate Soft Diffuse Light RAW8 panel used the native 50-step sampler
and two cases with registered shadow boundaries. At equal mean strength
0.5, constant averaged −0.00585, early-heavy −0.00527 and late-heavy
−0.00266. Saved constant strength 1 averaged −0.02450. All six new images
passed integrity. The measured gains against OFF were negative on average.
[Soft Diffuse Light evidence](evidence/soft-diffuse-light-raw8-scheduled-strength.json)

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

Earlier configuration and qualification receipts remain available in the [archived source](https://github.com/HyperGAN/klein-particle-sliders/tree/98e13bea7ed14a3ee2477ea9eea74d5cec79c298).

### Fitted S-PSR on native Euler velocity

The opt-in `klein_sliders.velocity_steering` renderer accepts an authenticated
eight- or 50-step, 512-pixel scale-zero RAW fit. The shared affine-ReLU probe reads
the FP32 channel mean of the packed velocity. The render applies its fitted
direction to the velocity consumed by the stock CFG 1 Euler scheduler. The
original stock owner loads and supplies the frozen RAW adapter, native
pipeline, and request. Existing fixed and gate entry points remain available.

Construct `VelocitySteeringPlan(core_wheel=..., binding_path=...,
binding_sha256=..., fit_path=...)` from a sealed replay binding. The JSON
binding uses `schema="klein-native8-velocity-replay-v1"` for eight steps or
`schema="klein-native50-velocity-replay-v1"` for 50 steps. Each binding records
the fitted probe and fit-plan hashes, exact RAW and native model identity
hashes, the packed velocity shape, and the full native sigma grid. The fit,
binding, and live scheduler must share the same grid. The model
identity hash uses `klein_sliders.contracts.digest`. Use a fitted
`weights="raw"`, `ema_decay=None` probe and pass a strength selected on the
separate calibration cases.

```python
from klein_sliders.velocity_steering import render_native_velocity

base = dict(
    plan=plan, transformer=runtime.transformer, pipeline=lambda: pipe,
    adapter=adapter, raw_path=raw, model_identity=runtime.identity,
    request=dict(seed=seed, initial_sha256=start_sha256,
                 conditioning_sha256=embedding_sha256,
                 text_ids_sha256=text_ids_sha256),
    check_live=check_live,
)
adaptive, trace = render_native_velocity(draw, **base, alpha=alpha)
constant, constant_trace = render_native_velocity(
    draw, **base, alpha=alpha, mode="same-atom-constant", from_adaptive=trace,
)
off, off_trace = render_native_velocity(draw, **base, alpha=0, mode="off")
```

Here `draw` runs one stock native request per call and reconstructs its
original RNG, conditioning, and scheduler state. The tensor hashes cover
the contiguous native bytes. The constant validates the same request, fit
and alpha against the adaptive trace, then uses the nearest-even FP32 value
of its exact sigma-weighted coefficient dose. Each receipt records the
applied coefficients, exact coefficient-space dose, and measured BF16 edit
norm integrated separately. Native calibration and held-out output evidence
remain separate from the fit.

### Calibrated activation gating

Use `gate_plan` on the same open RAW session. Klein averages the live packed
latent across spatial tokens in FP32 before each conditional forward. At
CFG 1 that forward occurs once per native Euler interval. CFP and CLABS
project this vector onto a calibrated unit direction in packed channel space.
The shared gate switches the saved routed adapter between OFF and the
explicit nominal strength. This applies the paper's allocation equations to
the native adapter residual. [Shared equations and calibration](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

```python
from klein_sliders.activation_gate import LatentGatePlan
from klein_sliders.scheduled_render import render

plan = LatentGatePlan(
    core_wheel=steering_wheel,
    method="cfp",
    parameter=calibrated_threshold,
    direction=fitted_packed_channel_direction,
    calibration_sha256=calibration_record_sha256,
)
image = render(session, gate_plan=plan, strength=calibrated_strength, **native_request)
receipt = image.info["strength_schedule"]
```

Build `steering_wheel` from core revision
`b1f780decc516808cfd3019d5e5b674457161f86` under
`packages/particle-sliders-core`. Klein verifies the exact steering module in
that wheel and loads it separately from the installed core that owns strict
RAW/FULL identities. For CLABS, supply the calibrated bias as `parameter`, an
explicit `seed`, and `clabs` with `mean`, `std`, `epsilon`, `setpoint`,
`proportional_gain`, `integral_gain`, and `beta`. The matched `random` control
takes the firing rate as `parameter` and an explicit `seed`. Keep its direction
unset. The receipt records each projection, firing probability, decision,
effective strength, actual sigma grid, realized gate rate, and sigma-weighted
strength. Replay a realized schedule with `step_gains` equal to those recorded
effective strengths divided by the original nominal strength.

The Final Boss RAW64 pilot calibrated both gates to 8 of 16 firings. On
five held-out cases, mean gains versus the equal-dose constant were
−0.03533 for CFP, −0.05400 for CLABS and +0.03467 for random gating. All
included images passed the independent integrity check.
[Per-case scores and applied strengths](evidence/final-boss-raw64-adaptive-steering.json)

A later exploratory comparison used the paper's 20% and 80% firing targets.
The 80% target rounded to 13 of 16 calibration intervals. Both CFP and CLABS
beat their own equal-dose constants on the reused exploratory cases.
[Exploratory firing-budget evidence](evidence/final-boss-raw64-exploratory-firing-budgets.json)

The first family-disjoint confirmation panel stopped when one OFF image failed
integrity. A separate, preregistered five-family panel kept the same RAW64
weights, sampler and 13-of-16 calibration. On that panel, CFP and CLABS fired
on all eight steps in every case, producing byte-identical images to their
equal-dose constants. Both methods averaged +0.02267 intended effect versus
OFF and exactly 0 versus the equal-dose constant. One random-gated image and
its constant failed integrity in the same case, so random has no aggregate
effect result. The confirmation supported keeping the existing constant
strength.
[Fresh confirmation traces and scores](evidence/final-boss-raw64-fresh-confirmation.json)

The [Soft Diffuse Light prompt pairs](configs/diffuse-light-pairs.json) supply
actual neutral and attribute prompts. Their archived 512-pixel BF16 eight-step
cache stores both teacher velocities at each shared neutral-trajectory latent.
Fit a direction from matched neutral and attribute latent trajectories at the
intended native resolution and step grid. Use separate cases and seeds for
active budget calibration and held-out evaluation. Compare against native OFF,
a constant dose control, and matched random allocation using the same RAW.
Rank by mean finished-image intended effect ON minus matched OFF and check
catastrophic output independently.

The Soft Diffuse Light RAW8 pilot fitted on portraits and calibrated on two
plant cases at nominal strength 0.5. Each gate fired on 50 of 100 calibration
intervals. The teapot and bedroom evaluations used 50 native Euler steps at
512 pixels and CFG 1. Both gates and their constant controls shared the
starting latent, and each constant matched the integrated sigma dose of its
gate.

Across the two held-out cases, CFP averaged −0.00560 versus OFF and +0.00162
versus its dose-matched constant. CLABS averaged −0.00867 versus OFF and
+0.00467 versus its dose-matched constant. All eight new renders passed
independent integrity checks. These results leave both settings experimental.
[Per-case scores, calibration, and applied intervals](evidence/soft-diffuse-light-raw8-adaptive-steering.json)

A separate prospective panel fixed four additional receiver and silhouette
registrations before timing. CFP averaged −0.00316 versus OFF and +0.00065
versus the same-dose constant. All 15 returned timing outputs passed
independent integrity. CLABS has three measured pairs because the third
scene’s matched constant remains unmeasured after the native memory stop.
[Four-case receiver evidence](evidence/soft-diffuse-light-raw8-receiver-timing-heldout.json)
