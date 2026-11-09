# Native CPU qualification — 2026-11-09

The combined source commit `8308d9ce87590b9c79521c4ad6e3193095f5b80c`
passed **28 tests in 3.27 seconds**. The actual worker exited0 and was reaped;
its owned process tree closed and reserved host resources were released.
This is new-cohort software conformance with randomly initialized tiny native
components. No pretrained model, GPU, teacher cache, real training control or
pretrained quality result was consumed or established.

The four new-cohort test modules cover fresh initialization before optimizer
construction, literal common learning-rate forwarding, actual native updates,
FULL exact resume, RAW/no-EMA export and cache-independent inference, and the
real pinned native pipeline producing finished PIL/PNG images. Saved states
are not reinitialized and the historical EMA cohort is not converted.

Actual environment: Python3.12, Torch2.9.1, ParticleGAN0.9.5;
`particle-sliders-core`0.14.1 from `4cc412d6ab6d01bb14b5f1e889ea6b79d3a2098d`;
Diffusers source `35087a644038eb531f48581708d4763044cb4852`;
Transformers5.22.1, Accelerate1.15.0 and Safetensors0.8.0.
The environment was isolated from historical captured runtimes. A setup attempt
first refused three unlisted resolver dependencies before any native test.
The continuation reused its actual prepared prefix and frozen resolution,
added the three exact wheel records, and ran only unfinished phases. Nothing
was retrained or rerated.

The retained actual qualification result is 11,325 bytes, SHA256
`2afe257beda453202bca57656de7925c3f7082f04e88776f9d5f6d079794b73c`.
Its Root terminal is 15,665 bytes, SHA256
`c2219ea723bb93db71668085742e17b4daaa753663e0713548bf7892ec3a9da5`.
The next step is a separately bounded pretrained cache/pilot and matched
native OFF/ON examples, followed by intended-effect LR measurements.
