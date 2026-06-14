# Model handoff to sdv-py (manual, reviewed)

After Stage-2 training + the parity gate passes:

1. Validate each retrained model vs the shipped one via the library API — load both boosters
   and a feature matrix, then call `model_training.validate.prediction_parity(new, shipped, X)`
   (the `validate` CLI subcommand is not yet wired):
   ```python
   import xgboost as xgb
   from model_training.validate import prediction_parity
   new = xgb.Booster(); new.load_model("<new>.ubj")
   ref = xgb.Booster(); ref.load_model("<sdvpy>/cfb/models/<name>.ubj")
   report = prediction_parity(new, ref, X)
   assert report["within_tol"], report
   ```
2. Copy under review (open a sdv-py PR; never auto-overwrite):
   - `ep_model.ubj`, `wp_spread.ubj`, `qbr_model.ubj` -> `sportsdataverse/cfb/models/`
3. **WP-naive is new to sdv-py.** Also:
   - add `wp_naive.ubj` to `sportsdataverse/cfb/models/` and confirm the
     `[tool.setuptools.package-data]` glob (`cfb/models/*`) ships it;
   - in `cfb_pbp.py`: load a second booster from `wp_naive.ubj` and emit a `wp_*_naive` output
     alongside the spread WP (mirrors the spread path; uses `wp_final_names` minus `spread_time`);
   - bump the bundled-model note / CHANGELOG in sdv-py.
4. Re-run sdv-py's CFB tests; confirm EPA/WPA on a known game stay within tolerance.
