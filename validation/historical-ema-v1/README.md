# Historical EMA source fixture

`test_lifecycle.py` is byte-identical to the original f2e9588e source test, retained as historical evidence rather than run against the new RAW policy. The original config is preserved at `configs/historical/diffuse-light-pg094-ema-v1.json`; `validation/cpu-qualification.json` is unchanged and belongs to that old Source. No packet, weight or qualification receipt was rewritten. Restore/inspect original packets with their exact original Source; this new factory does not reinterpret them as RAW or authorize EMA training.
