# New explicit PG0.9.5/fresh initialization cohort

This source profile pins shared core main merge
`4cc412d6ab6d01bb14b5f1e889ea6b79d3a2098d` and uses its public
`initialize_fresh_components` helper. It does not migrate original PG0.9.4/EMA
or intermediate RAW-only packets. Keep their original loaders and source pins.
The new config, source profile and complete initialization receipt participate in
training/export identities. Old18-test evidence is retained unchanged.

## Exact fresh policy

`klein-kaiming-RMS-QR-zero-up-bank-R2-std1-v1` applies only to owned new trainables:

| Component | Explicit method/law |
|---|---|
| Added down/router/MLP Linear leaves | Official deterministic orthogonal at the actual Kaiming(a=sqrt5)/default Linear RMS |
| Added up Linear leaves | Zero constructor tensors preserved exactly, with nonzero down leaves |
| FieldCritic Linear/Conv2d leaves | Official deterministic orthogonal at their existing default constructor RMS; no arbitrary readout/mix zeros |
| Existing actual128×4 particle Parameter | Official R2Normal(mean0,std1), preserving the bank's declared scale and ownership |
| Pretrained transformer, buffers and any restored state | Never sent to the fresh initializer |

Each nonempty trainable has a deterministic named seed derived from the explicit
policy, run seed and stable parameter name. Adding/reordering unrelated scopes
cannot shift existing draws. This is an opt-in experimental adapter recipe, not
a claim QR improves Transformer/LoRA training or that an old LR choice transfers.
It does not add a bank or change the shared GAN/optimizer formula. Weight EMA
remains disabled (`ema_decay=None`), exports/inference RAW only.

## Fresh versus exact restore

The product entrypoint routes intent **before** prepare/game/optimizer creation:

```python
from particle_sliders.lifecycle import SessionConfig
from klein_sliders.lifecycle import open_training

config = SessionConfig.from_dict(config_dictionary)
with open_training(config) as fresh:
    # Fresh scopes initialize once before the shared game's optimizers.
    fresh.train(stop_after=2)
    fresh.checkpoint("training.pt")

with open_training(config, resume="training.pt") as resumed:
    # Original initialization receipt is authenticated and reused; no initializer.
    resumed.train()
```

Use the same fixed total budget/config and the exact latest FULL training packet;
this example describes API placement, not admission of pretrained/GPU execution.
The product CLI uses this same route. The generic core currently opens a backend
before later restore and does not supply product construction intent, so calling
`BACKEND.open` without the product context fails before native allocation. No
universal CLI/config flag is advertised. Native response inference attaches shape
and loads the selected RAW state; it never calls the fresh initializer.

The receipt binds actual helper/PG initializer source, product policy/source,
cohort, named seeds, host laws, effective operations, trainable byte digests and
owned RNG-state digests. The full receipt is retained in game metadata and native
export configuration. Exact resume validates its original receipt before native
allocation and checks declared leaf names/shapes/seeds again before optimizers;
then the shared strict restore validates/loads the complete game/optimizer/bank/
RNG state. No packet is retagged or rewritten. Supplied fresh/restore context must
be truthful; neither the product nor helper claims to detect arbitrary hidden
loads or an optimizer a caller omits.

## Distribution and pending checks

Core0.14 intentionally excludes historical embedded `pg09_images`. This product
owns its conditioning provider, so `conditioning.py` extracts only the original
fixed policy/projection/context/features block byte-exact from pinned core
`3e8052250d593a55d9f071cebae503e764342d05`. Cache geometry, numeric operations,
collision rules and policy name are unchanged; no GAN code is copied. Source
provenance is recorded in `validation/conditioning-extraction.json`. This source
comparison does not substitute for native numerical checks.

Pending in an isolated matching dependency environment, with GPUs disabled:
actual tiny random native Klein full2 vs split1/exact-resume1, RAW-only export and
response inference, fresh zero-output/nonzero-down/frozen-base/actual-bank
ownership, original receipt reuse with initializer forbidden on restore/inference,
initialization-refusal cleanup before optimizers, and source/EMA/cohort refusals.
These are new-source CPU conformance checks, not pretrained/model quality proof,
new LR training, a renderer, an automatic catastrophic-output qualification or a
repetition of completed historical experiments.
