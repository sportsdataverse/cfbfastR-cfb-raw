# model_training fixtures

- `xgb_{ep_model,wp_spread_model,wp_naive_model}.ubj` — May-2021 R-trained reference models
  (gp-cfb-raw-keepers), converted from binary `.model` via xgboost 3.0 (binary format is
  unreadable in xgboost >=3.1). EP=8-feat/7-class; WP spread=10-feat; WP naive=9-feat.
  **Stage-1 parity references only** (divergent lineage; NOT the shipped models).
- `fd_model.ubj` — synthetic 3-round structural fixture (5-feat / 76-class / `multi:softprob`).
  Trained on random data to verify the feature contract; the full production model is trained
  on the CFB backfill by the `train-fd` CLI. cfb4th's internal `fd_model` uses an old xgboost
  binary format incompatible with xgboost ≥3.1, so a synthetic fixture is committed instead.
- `{epa,wpa}-model-test-items.json` — cfbscrapR-lineage reference plays from akeaswaran/cfb-pbp-analysis.
  `wpa-*` is in the shipped 13-feat WP contract (near-parity WP oracle); `epa-*` is 16-feat-lineage
  (ballpark EPA only). Sanity checks, not exact shipped-parity oracles.
