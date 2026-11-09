# Klein: two completed provisional ties

As of 2026-11-09, **G0.001 and G0.0001 tie** in two separate saved RAW32
comparisons of soft diffuse illumination. There is **no unique recommended G**,
new default, global optimum or automatic promotion. Both retain
`trait_qualified:false`: these are completed scoped model-instrument comparisons,
not formally validated trait measurements. No new numeric-anchor, calibration,
confirmation or repeated-rating gate is needed to report them honestly.

| Native512 sampling profile | G0.001 mean ON−OFF | G0.0001 mean ON−OFF | Scope |
|---|---|---|---|
| 8 steps / CFG1 | +0.0625 | +0.0625 | Four equally weighted cases |
| 50 steps / CFG4 | 0.0 | 0.0 | Four equally weighted cases; every OFF/ON saturated |

Each arm completed32 updates at horizon128/batch2/rank8/seed7, with the same
128×4 bank and fresh scoped PG0.9.5 policy. Cases are teapot7, woman19, plant7 and
bedroom19; OFFstrength0 and ONstrength1 use matched requests/seeds. The original
8-step normalized intensity pairs for **both** arms are teapot0.75→1.0,
woman1.0→1.0, plant0.75→0.75 and bedroom1.0→1.0: gains0.25/0/0/0 average0.0625.
For50-step/CFG4 all12 first intensities, including four shared OFFs and eight ONs,
are1.0 (rubric4/4); every gain is0. No old8-step OFF, score or guard was
transferred to this new profile. Saturation is a limitation, not evidence of an
LR improvement or a missing comparison.

Both original four-case photographic categorical checks and exact FULL32 guards
passed for the two lower-rate candidates. All eight **new**50-step ON categorical
observations also passed, but no new-profile FULL-bound guard is claimed.
G0.012 remains manually stopped after cyan/no-object collapse; its old detector
false negatives and saved checkpoints remain evidence, not an eligible winner.
A weak effect is separate from catastrophic failure.

[The portable summary](finished-output-rate.json) keeps both profiles, all
per-case values, exact floating-point actual rates, RAW/FULL metadata and original
result/decision hashes. Retained RAWs, caches and first responses are private,
source-bound campaign artifacts, not new released weights or bundled downloads.
It supports scalar arithmetic reproduction, not a standalone pretrained run.

## Exact common options and source limits

The two tied **product options fragments** are:

```json
{"lr": 0.001, "d_lr_mult": 18.0, "prior_lr_mult": 30.0, "ema_decay": null}
{"lr": 0.0001, "d_lr_mult": 180.0, "prior_lr_mult": 300.0, "ema_decay": null}
```

G=`lr`, D=`lr*d_lr_mult`, bank=`lr*prior_lr_mult`. The saved actual D for both
is0.018000000000000002. The bank products are respectively0.03 and
0.030000000000000002; preserve that recorded binary difference rather than
silently normalizing the recipes. The product passes the literal members to
`ConditionalPG095(recipe_overrides=...)`; the shared official GAN, full native
penalty and optimizer formula are unchanged. Omitted values keep declared
upstream defaults, not a claim of optimization.

For a genuinely new cohort, add the selected fragment to `options` in
[the existing config](../configs/diffuse-light.json), with actual model/cache
paths and cache checksum, fixed `steps=128`, batch2/seed7 and existing source/init
policy fields. `--stop-after 32` stops within that horizon. These actual commands
show interface placement; **do not repeat the completed arms**:

```sh
klein-slider train --config YOUR-CONFIG.json --output NEW-RUN --stop-after 32
klein-slider train --config SAME-CONFIG.json --output EXISTING-RUN --resume EXACT-LATEST-FULL.pt --stop-after 32
klein-slider render --model-dir STAGED-KLEIN --export SOURCE-MATCHED-RAW.pt --output NEW-PAIR --prompt 'A red teapot on a wooden table' --seed 7 --size 512 --steps 8 --strength 1 --device cuda:0 --base-precision bfloat16
```

Only a complete original FULL can resume training, with both optimizers, bank,
RNG, exact source/config and initialized receipt; RAW is inference-only. Use the
original source-matched loader for retained artifacts. The generic current
factory does not silently accept a changed captured research factory identity.
The public `render` CLI fixes **CFG1** and exposes no `--guidance-scale` option.
The saved50-step/CFG4 outputs used a separately bound execution profile; no public
CFG4 command is invented here. See the actual [CLI](../src/klein_sliders/cli.py)
and [renderer](../src/klein_sliders/render.py). Neither the recipe fragments nor
these commands allocate authority or establish ordinary hosted Generate readiness.

Fresh product `open_training(config)` explicitly applies named Kaiming-RMS QR
for added adapter/FieldCritic leaves and R2Normal(std1) to the existing bank,
before optimizers. It preserves zero-UP/nonzero down, actual Parameters and frozen
base. Exact `open_training(config, resume=FULL)` retains the original receipt
without reinitialization; inference never initializes. Adapter, critic and bank
use RAW current weights with `ema_decay=None`; no weight EMA/averaging is allowed.
Official scalar optimizer running statistics are unchanged. The old PG0.9.4/EMA
cohort remains historical, never migrated or proposed for fresh training.

## One metric, separate catastrophic eligibility

Rank eligible matched-budget candidates only by the equal-case arithmetic mean
of finished intended-effect intensity ON minus matched OFF. Keep negatives,
zero and exact ties; no forced winner. Teacher/student SSE, preservation, beauty,
confidence and weak trait do not select or break ties. Qualified catastrophic-only
judgments determine failure separately; unknown/missing judgments are unmeasured,
not passes or synthetic zeros. A pass does not qualify the trait instrument.

Reuse completed searches, teachers, RAW/FULL checkpoints, finished images and
frozen first observations. No repeated updates, old rerating, fresh confirmation
or extra validation loop is required. Another sampling profile needs its own
missing observations; both completed profiles above retain their own evidence.
