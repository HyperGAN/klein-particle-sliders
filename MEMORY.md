# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Native cached-response training | 1 | 16 | 24 | B2; 512²; text128; BF16; native block checkpointing |
| Finished-image sampling | 1 | 24 | 32 | B1; 256²/512²; text128; BF16; all components resident |
| Conditioning / teacher preparation | 1 | 12 | 32 | 512²; one teacher at a time; Qwen/VAE on CPU |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Native profile and calculation

Model: `black-forest-labs/FLUX.2-klein-base-4B@a3b4f4849157f664bdbc776fd7453c2783562f4d`. Source:
[contracts](src/klein_sliders/contracts.py), [loader](src/klein_sliders/native.py),
[adapter](src/klein_sliders/adapter.py), [lifecycle](src/klein_sliders/lifecycle.py)
and [render limits](src/klein_sliders/render.py).

* BF16 frozen residency: DiT serialized upper bound 7.219 GiB;
  Qwen3 7.492 GiB; VAE serialized upper bound 0.157 GiB.
  The renderer's `.pipe.to(device)` keeps all three together: **14.868 GiB**.
* Hidden3072, FF9216, 5 double +20 single blocks. Rank8 targets include20
  ordinary attention projections plus40 fused single-block hosts. Each single
  QKV+SwiGLU projection is3072→27648 and output12288→3072.
  `sum(8*(in+out))` = 8,355,840 LoRA scalars; 60 routed bridges ×6476,
  shared128×4 bank and 21,313-scalar width32 FieldCritic give **8,766,225 trainables**.
  Critic condition width106 is `4+3*(32+2)`: each of the THREE native contexts
  has32 projected coordinates plus count/RMS, followed by four time/sign/CFG
  metadata values. FP32 weights+gradients/update allowance = 0.653 GiB.
* Native checkpointing retains one hidden boundary per25 blocks:
  `2*(1024+128)*3072*2*26` = 0.343 GiB.
  One recomputed block's native envelope
  `2*(1024+128)*2*(12*3072+3*9216+4*3072)` = 0.330 GiB;
  round this plus routed128-way logits, penalty graphs and output fields to2 GiB.
  Add4 GiB workspace/allocator allowance and round total to16.
* Sampling: 14.868 frozen +4 forward/decode +4 workspace rounds to24.
  Encoder-only preparation keeps the Qwen/VAE on CPU; no model offload runs between
  native layers. Host budget32 derives from14.87 GiB complete pipeline +7.22
  largest component-loading copy + up to2 GiB cache +4 runtime. Cached-only training
  loads just the DiT: two7.22 GiB copies +2 cache +4 runtime round to24.

The table binds batch2, rank8, critic width32, text128, ≤128 cache rows and
memory-efficient attention. Public finished rendering accepts256/512 canvases;
training fields can differ, so larger requests require recalculation. No quantized
base is supported by this profile. GPU Source integration is not a blanket fit
qualification for every hardware/runtime. Configs are captured under [docs/memory](docs/memory/inputs.json).


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The PG 0.9.5 `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See released [PG 0.9.5 recipes](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/v0.9.5/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Native cached-response training | 14.215 → 16 | 20.438 → 24 |
| Finished-image sampling | 22.868 → 24 | 28.087 → 32 |
| Conditioning / teacher preparation | 11.219 → 12 | 28.087 → 32 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
