# CFB EP/WP Model-Training — R → Python Port Design Spec

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repo:** `cfbfastR-cfb-raw` (Python/uv) — new `python/model_training/` package
- **Source of truth (R originals):** `gp-cfb-raw-keepers/from-cfbfastR-raw/model_training/{00,01,02,03,04,06}*.R`
- **Inference parity target (sdv-py):** `sportsdataverse/cfb/model_vars.py` + `cfb_pbp.py` (`CFBPlayProcess`), bundled `sportsdataverse/cfb/models/{ep_model,wp_spread,qbr_model}.ubj`
- **Program:** **Track 1** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`). Tracks 2–6 (fourth-down, RB-eval, pregame-WP/Five-Factors, CPOE, NFL EP/WP) are specced separately.

## 1. Goal

Convert the cfbfastR R EP/WP model-training pipeline to Python, living in `cfbfastR-cfb-raw`,
to **retrain the EP, WP-spread, WP-naive, and QBR models** on the full **earliest-available→2025**
history that the CFB backfill produces. The retrained models drop back into sdv-py's
`cfb/models/*.ubj` so the library's EPA/WPA/QBR computation is reproducible and extendable from a
Python-native pipeline (no R toolchain). **WP-naive is newly promoted to a shipped artifact** (it was
trained as a by-product); shipping it requires a small sdv-py change to bundle + apply `wp_naive.ubj`.

Training inputs are **recreated faithfully from scratch, from raw ESPN game JSON** — not read
from the persisted `final.json`. The recreation reuses `CFBPlayProcess`'s own feature-building
functions ("most of the same functions"), so training features are built by the same code that
computes them at inference → train/inference parity by construction, while remaining a
self-contained pipeline that does not depend on whatever columns `final.json` happens to persist.

## 2. Background — what the investigation established

Hard evidence gathered while examining the R scripts and the shipped models (xgboost introspection
+ prediction comparison):

| Keepers `.model` | Date | Feats | Produced by | Relationship to shipped `.ubj` |
|---|---|---|---|---|
| `ep_model.model` | Dec 2020 | 8, `multi:softprob`, 7-class | *(pre-dates the scripts)* | **prediction-identical** to shipped `ep_model.ubj` (max abs diff `0.0`) |
| `wp_spread.model` | Dec 2020 | 13, `binary:logistic` | *(pre-dates the scripts)* | **prediction-identical** to shipped `wp_spread.ubj` (max abs diff `0.0`) |
| `xgb_ep_model.model` | May 2021 | 8, `multi:softprob`, 7-class | `02_epa_xgb_model.R` | same architecture, **different instance** (diff `0.336` vs shipped) |
| `xgb_wp_spread_model.model` | May 2021 | **10**, `binary:logistic` | `03_wpa_xgb_model.R` (spread) | **diverged** — shipped is 13-feat |
| `xgb_wp_naive_model.model` | May 2021 | 9, `binary:logistic` | `03_wpa_xgb_model.R` (naive) | never shipped |

**Conclusions that shape this design:**

1. **No keepers R script produced the shipped models.** The shipped `.ubj` are the Dec-2020
   canonical `.model` files converted to UBJ. The `00–06` scripts are a May-2021 retraining
   attempt whose WP output diverged (10-feat) and was never adopted.
2. **EP architecture matches** the shipped contract — shipped `ep_model.ubj` = **3675 trees
   (= 525 rounds × 7 classes), 8 feats** = keepers `02` (8-feat, `nrounds=525`, `multi:softprob`).
   `02` is the correct EP recipe *architecture* (the 16-feat cfbscrapR patsy EP and the 10-feat
   `model_training.R` are different lineages).
3. **The 13-feat shipped WP recipe IS `akeaswaran/cfb-pbp-analysis/cfbscrapR-wpa.ipynb`** (found by
   the prior-art survey; confirmed by tree count). Shipped `wp_spread.ubj` = **760 trees, 13 feats**,
   and that notebook trains `binary:logistic`, `nrounds=760`, on exactly `wp_final_names` (its
   `before_spread_columns` == `wp_final_names`, in order), saving `wp_spread.model` (= the Dec-2020
   keepers file = the shipped `.ubj`; `wp_spread_xgb_dump.json` is 760 trees). Its
   `spread_time = pos_team_spread*exp(-4*elapsed_share)` matches sdv-py `__add_spread_time` verbatim.
   **So WP is a faithful port, not a reconstruction.** Two recipe facts that differ from keepers
   `03` (the divergent 10-feat dead-end): the shipped WP uses **NO sample weights**
   (`label = (pos_team == winner)` only), and the hyperparameters are
   `eta=0.02, gamma=0.3445502, subsample=0.7204741, colsample_bytree=0.5714286, max_depth=5,
   min_child_weight=14, nrounds=760`.
4. **The R EP→`ExpScoreDiff` weight vector is scrambled.** `03` computes `ep` for the WP feature
   `ExpScoreDiff` with `weights <- c(0, 3, -3, -2, -7, 2, 7)` against class order
   `[TD, Opp_TD, FG, Opp_FG, Safety, Opp_Safety, No_Score]` (i.e. `TD→0, FG→-3, Safety→-7`),
   which is wrong. sdv-py inference uses the **correct**
   `ep_class_to_score_mapping = {0:7, 1:-7, 2:3, 3:-3, 4:2, 5:-2, 6:0}`.
5. **Old binary `.model` is unreadable in xgboost ≥3.1.** The May-2021 reference models load only
   in xgboost 3.0; they were converted once to UBJ for use as Stage-1 validation fixtures.
6. **The shipped `qbr_model.ubj` (6-feat, `reg:squarederror`) has no training script** in keepers,
   but its inputs and target are recoverable: `CFBPlayProcess.__process_qbr` already computes all 6
   `qbr_vars` features (`qbr_epa, sack_epa, pass_epa, rush_epa, pen_epa, spread` as per-QB weighted
   means), and `akeaswaran/cfb_qbr` supplies the missing piece — the training **target**, ESPN raw QBR
   (scraped to `composite.csv`) — plus the canonical `qbr_epa` definition (EPA floored at −5, −3.5 on
   fumbles, wp garbage-time weighting), which `CFBPlayProcess` already implements identically. QBR
   retraining is therefore **in scope** (see §3 #10, §7).
7. **Prior-art survey of `akeaswaran/cfb-pbp-analysis`:** **`cfbscrapR-wpa.ipynb` IS the shipped WP
   recipe** (see #3) — the big find. The others are NOT shipped recipes: `CFB-EPA_AKE.ipynb` is the
   cfbscrapR **16-feat patsy** EP (CoreML `CFBEPA.mlmodel`; `xgb_dump.json` = 2100 trees = 300×7;
   class order `No_Score=4`); `win-prob.ipynb` is a **pregame 5-factor team** WP (`pgwp_model.model`),
   not play-level; `model_training.R` is a modern **10-feat** EP variant (correct weights, vectorized
   labeling). Reusable: the vectorized next-score labeling (§6.1); `epa/wpa-model-test-items.json` as
   **sanity** fixtures (cfbscrapR-lineage — the WPA set was generated by `cfbscrapR-wpa.ipynb` itself
   and IS in the shipped WP feature contract, so it's a near-parity WP oracle; the EPA set is
   16-feat-lineage, ballpark only).

## 3. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Port location | `cfbfastR-cfb-raw/python/model_training/` (depends on `sportsdataverse` already in `pyproject`). |
| 2 | Training-data source | The **CFB backfill** (2002/2004→2025). Inputs recreated **from raw**, not from `final.json`. |
| 3 | Feature recreation | Build from raw by **reusing `CFBPlayProcess` feature functions** (not a separate re-implementation, not the persisted output). |
| 4 | Staging | **Stage 1 faithful replica** (validated vs the May-2021 `xgb_*` reference models) **then Stage 2 parity upgrade** (drop-in shipped contract). |
| 5 | Stage-1 EP weights | Replicate the **scrambled** `c(0,3,-3,-2,-7,2,7)` so predictions match the R reference. |
| 6 | Stage-2 EP weights | Use the **correct** `ep_class_to_score_mapping` (match sdv-py inference). |
| 7 | WP feature set | Stage 1 = R's 10-feat spread / 9-feat naive; Stage 2 = shipped **13-feat** `wp_final_names`. |
| 8 | Figures | **Recreate all 4 calibration plots with exceptional, bespoke styling** (plotnine) **plus** emit calibration **data tables**. Plotting code is retained, not dropped. |
| 9 | Figures engine | **plotnine** (ggplot2 port) — near-verbatim translation of the R `theme()` blocks. |
| 10 | QBR model | **In scope.** Faithful GAM port (`cfb_qbr/qbr.R → qbr.py`) landed now as a reference ancestor; **Stage 2 adds a 6-feat XGBoost drop-in** for the shipped `qbr_model.ubj` (features = `CFBPlayProcess` `qbr_vars`, target = ESPN raw QBR). |
| 11 | Model handoff to sdv-py | **Manual, reviewed copy** of `ep_model.ubj` / `wp_spread.ubj` / `wp_naive.ubj` / `qbr_model.ubj` into `sdv-py/sportsdataverse/cfb/models/` (no automated overwrite). **WP-naive is new to sdv-py** → also a small `CFBPlayProcess`/`model_vars` change to bundle + apply it (emit a `wp_*_naive` output alongside the spread WP). |
| 12 | Season coverage | Train from the **earliest available backfill season → 2025**. Backfill floor is 2004 (also ESPN's QBR floor); attempt 2002–2003 only if the backfill yields usable data for them. |

## 4. Data flow

```
CFB backfill: scrape_cfb_json.py → cfb/json/raw/{game_id}.json   (verbatim ESPN summary)
   │
   ▼  model_training reads RAW (not final.json) and rebuilds inputs from scratch
features.py
   ├─ CFBPlayProcess(gameId, path_to_json=raw_dir).cfb_pbp_disk().run_processing_pipeline()
   │     → in-memory play_df with start.*/end.* feature columns  (same functions as inference)
   └─ select model inputs:  Stage 1 → R subset (8 EP / 10 WP / 9 naive)
                            Stage 2 → shipped contract (8 EP / 13 WP)
next_score.py  (port of 06)
   └─ find_game_next_score_half(drives) → NSH (signed half-points) / DSH per play
ingest.py  (port of 01)
   └─ NSH → Next_Score → label (7-class);  weights Drive_Score_Dist_W / ScoreDiff_W / Total_W
   └─ play-type + down cleaning; write pbp_full.parquet  (idempotent, season-partitioned)
train_ep.py (02) / train_wp.py (03) / train_qbr.py (Stage 2)
   └─ XGBoost → models/{ep_model,wp_spread,wp_naive,qbr_model}.ubj   (+ optional LOSO CV)
   └─ QBR: features = qbr_vars from CFBPlayProcess; target = ESPN raw QBR (composite.csv)
validate.py + figures.py
   └─ prediction-parity vs reference .ubj; calibration tables (parquet/csv) + 4 styled PNGs
   ▼  (Stage 2 only, manual)
copy ep_model.ubj / wp_spread.ubj → sdv-py/sportsdataverse/cfb/models/
```

The one thing the processor never emits — and the only genuinely new logic — is the **label**
(next score in half). Inference does not need an outcome label, so the processor doesn't compute
it; `next_score.py` is the irreducible core of the port.

## 5. Module architecture — `python/model_training/`

| Module | Ports R | Responsibility |
|---|---|---|
| `next_score.py` | `06_data_ingest_utils.R` | `find_game_next_score_half(drive_df)` + `find_next_score(play_i, score_plays_i, dat_drive)`. NSH = signed next-score points from posteam perspective; DSH = drive id of that score; half-boundary → 0 (no score before half); defense-TD list flips the scoring sign. |
| `ingest.py` | `01_data_ingest.R` | Per season: NSH→`Next_Score`→7-class `label`; recency/score-diff weights; kickoff down→-1, drop `down<1`, drop OT games, drop ESPN partial games, drop special-teams play types. Writes `pbp_full.parquet`. |
| `features.py` | (01 select/rename) | Run `CFBPlayProcess` on raw → in-memory `play_df`; select/rename to `ep_final_names` (8) and the stage-appropriate WP set; `ExpScoreDiff = pos_score_diff + ep` and `ExpScoreDiff_Time_Ratio = ExpScoreDiff/(game_seconds_remaining+1)` with **stage-dependent** EP weights. |
| `train_ep.py` | `02_epa_xgb_model.R` | `booster=gbtree, objective=multi:softprob, eval_metric=mlogloss, num_class=7, eta=0.025, gamma=1, subsample=0.8, colsample_bytree=0.8, max_depth=5, min_child_weight=1`, `nrounds=525`, `weight=ScoreDiff_W`. Optional LOSO CV. Save `ep_model.ubj`. |
| `train_wp.py` | `cfbscrapR-wpa.ipynb` (shipped recipe) / `03` (Stage-1 ref only) | **Spread — the confirmed shipped recipe (760 trees / 13 feats):** `binary:logistic`, `eta=0.02, gamma=0.3445502, subsample=0.7204741, colsample_bytree=0.5714286, max_depth=5, min_child_weight=14`, `nrounds=760`; features = `wp_final_names` (13); **NO sample weights**; `label = (pos_team == winner)` (winner from home/away points). **Naive:** drops `spread_time` (12 feats), `eta=0.2, gamma=0, subsample=0.8, colsample_bytree=0.8, max_depth=4, min_child_weight=1`, `nrounds=65`. (Stage-1's faithful-replica target is keepers `03`'s divergent 10-feat model — `eta=0.05,…,nrounds=534`, `weight=ScoreDiff_W` — kept only to validate the labeling/training machinery; the shipped model uses the cfbscrapR-wpa recipe above.) Save `wp_spread.ubj` / `wp_naive.ubj`. |
| `train_qbr.py` | `cfb_qbr/qbr.R` (GAM) → 6-feat XGBoost | **Stage 2.** features = the 6 `qbr_vars` per QB-game (`CFBPlayProcess` weighted means: `qbr_epa, sack_epa, pass_epa, rush_epa, pen_epa, spread`); target = ESPN raw QBR (`qbr_scrape.py → composite.csv`); `objective=reg:squarederror`. The faithful 1-feat GAM port (`raw_qbr ~ s(qbr_epa)`) lands in `akeaswaran/cfb_qbr` (`qbr.py`) **now**, as the conceptual ancestor + canonical `qbr_epa` source. |
| `validate.py` | (CV/calibration math) | Prediction-parity harness vs reference `.ubj` (Stage 1) / shipped `.ubj` (Stage 2); LOSO-binned (0.05) calibration → `bin_pred_prob` vs `bin_actual_prob`; weighted calibration error; QBR fit vs ESPN raw QBR (corr + MAE). Emits tables as parquet/csv. |
| `figures.py` | ggplot blocks in 02/03 | plotnine recreation of the 4 calibration plots (see §8). |
| `cli.py` | `00_play_by_play_train.R` | Subcommands `ingest \| train-ep \| train-wp \| validate \| figures`; flags `--stage {1,2}`, `--seasons A:B`, `--loso`, `--raw-dir`, `--out-dir`. |

`model.matrix(~ . + 0)` in the R is a no-op to port: every feature is already numeric
(`down_1..4` are 0/1 ints), so the Python feature matrix is the selected columns as-is — no
dummy expansion.

## 6. Label & weight derivation (the portable IP)

### 6.1 Next score in half (`next_score.py`, port of `06`)
For each game, find the next scoring event **within the same half** for every play. NSH carries the
signed point value from the possessing team's perspective (`+7/+3/+2` for posteam TD/FG/Safety,
negated for the opponent); `No_Score` (0) when the half ends before any score. Defensive-TD play
types flip the scoring team's sign. DSH records the drive id of the next score (used for the
recency/score-distance weight).

**Implementation: prefer the vectorized form over the keepers `06` row loop.** akeaswaran's
`cfb-pbp-analysis/model_training.R` computes the same labels with a `tidyr::fill(.direction="up")`
of the scoring drive/team/type within `(game_id, half)` — which ports to a polars
`fill_null(strategy="backward").over(["game_id", "half"])` (next-score-looking-forward). This is
far simpler and faster than the per-play `find_next_score` loop and produces identical labels; the
loop form is kept only as a reference oracle in tests.

### 6.2 7-class label (`ingest.py`, port of `01`)
`NSH ∈ {7,3,2,-2,-3,-7} → {TD, FG, Safety, Opp_Safety, Opp_FG, Opp_TD}`, else `No_Score`;
encoded `TD=0, Opp_TD=1, FG=2, Opp_FG=3, Safety=4, Opp_Safety=5, No_Score=6` (matches the
shipped EP model's class order — verified by `num_class=7` introspection).

### 6.3 Weights
`Drive_Score_Dist_W` (drives-to-next-score recency), `ScoreDiff_W` (down-weight blowouts),
`Total_W = Drive_Score_Dist_W + ScoreDiff_W`, `Total_W_Scaled` (normalized). EP and WP both
train with `weight = ScoreDiff_W` (per the R scripts).

### 6.4 Cleaning (port of `01` filters)
kickoff down → -1; drop `down < 1`; drop OT/overtime games; drop ESPN partial games (4th-qtr
`clock_minutes == 0` heuristic); drop special-teams/non-scrimmage play types; drop the explicit
buggy `game_id` blocklist carried in `02`/`03`.

## 7. Stages

### Stage 1 — faithful replica
- Builds the **R feature subsets** (8 EP / 10 WP-spread / 9 WP-naive) and replicates the
  **scrambled EP weights** so the derived `ExpScoreDiff` matches R.
- **Validation target:** the May-2021 `xgb_*` reference models, pre-converted to UBJ (xgboost 3.0)
  and committed as test fixtures under `tests/fixtures/model_training/`.
- **Parity bar:** prediction-distribution within tolerance on a shared feature matrix
  (byte-identical R↔Py XGBoost is unattainable); **deterministic intermediates — NSH, labels,
  weights — asserted exactly** against a small recomputed reference. This proves the
  labeling/weighting/training port is correct before extending it.

### Stage 2 — parity upgrade (the shipped models)
- Builds the **shipped 13-feat WP** contract (`wp_final_names`) and the **correct EP weights**.
- **WP is a faithful port of `cfbscrapR-wpa.ipynb`** (the confirmed shipped recipe): exact 13 feats,
  the exact hyperparameters, `nrounds=760`, **no sample weights**, `label = (pos_team == winner)`.
  Inputs come from the backfill (`CFBPlayProcess` already emits `spread_time`,
  `ExpScoreDiff_Time_Ratio`, `adj_TimeSecsRem`, `pos_team_receives_2H_kickoff`, …); winner + spread
  from the ESPN summary / `homeTeamSpread`. Because the shipped `wp_spread.ubj` **is** this recipe's
  output, Stage-2 WP can be validated to near-equality against it directly.
- Trains EP + WP-spread + **WP-naive** + the 6-feat QBR XGBoost on **earliest-available→2025**.
  WP-naive is the same `cfbscrapR-wpa.ipynb` recipe minus `spread_time` (12 feats, `nrounds=65`,
  `eta=0.2, max_depth=4`); it ships as `wp_naive.ubj` (decision #11).
- **QBR (`train_qbr.py`):** features = the 6 `qbr_vars` (per-QB weighted means from
  `CFBPlayProcess`); target = ESPN raw QBR (`composite.csv` from `qbr_scrape.py`);
  `objective=reg:squarederror`. Drop-in for `qbr_model.ubj`.
- **Regression gate:** retrained `ep_model.ubj` / `wp_spread.ubj` / `qbr_model.ubj` predict within a
  documented tolerance of the *shipped* models on a held-out historical season (so existing
  EPA/WPA/QBR do not shift), then coverage extends to the new seasons.
- Output models are copied into sdv-py manually under review (decision #11).

## 8. Figures (`figures.py`) — bespoke styling + data tables

Recreate the 4 calibration plots with exceptional, brand-consistent styling, matching the R
originals:

- **`xgb_ep_cv_loso_calibration_results`** — 7 facets, one per scoring-event class.
- **`wp_spread_cv_loso_calibration_results`** — 4 facets, one per quarter.
- **`wp_spread_no_home_cv_loso_calibration_results`** — 4 facets (spread model w/o `is_home`).
- **`wp_naive_cv_loso_calibration_results`** — 4 facets.

Each: points sized by `n_plays`, loess smooth, `y=x` dashed reference, "More times / Fewer times
than expected" annotations, `facet_wrap`, and the calibration-error caption.

**Styling target (from the R `theme()` blocks):** garnet `#500f1b` accent, `grey95`/`grey99`
panel/plot fills, **Gill Sans MT** with a cross-platform fallback chain (the font is
Windows-only; degrade gracefully), bottom-aligned size legend, and the **cfbfastR hex logo**
overlaid bottom-right (the R `add_logo()` step). `logo.png` is absent from keepers — source the
cfbfastR hex from the `cfbfastR` package assets.

**Engine: plotnine** — the R uses ggplot2; plotnine ports `geom_point` / `geom_smooth(method="loess")`
/ `facet_wrap` / `theme(...)` almost line-for-line, so the bespoke look transfers faithfully.

**Plus data tables:** every figure's underlying calibration frame
(`qtr`/`class`, `bin_pred_prob`, `n_plays`, `n_wins`, `bin_actual_prob`) is written as
parquet + csv next to the PNG, so the numbers are inspectable independent of the chart.

## 9. Dependencies

Add to `cfbfastR-cfb-raw/pyproject.toml`:
- runtime/training: `xgboost>=2.0` (explicit; currently transitive via sdv-py), `numpy` (transitive).
  The QBR XGBoost (Stage 2) needs no new dep.
- figures (own group): `plotnine`, `statsmodels` (loess backend), `pillow` (logo overlay).

The standalone GAM reference (`akeaswaran/cfb_qbr/qbr.py`) needs `pygam` — but that lives in the
`cfb_qbr` repo, **not** a `cfbfastR-cfb-raw` dependency; the in-port QBR model is XGBoost.

`polars` / `pyarrow` / `pandas` are already present. No GPU. XGBoost training is CPU-bound and
benefits from the repo's existing ProcessPool convention for per-season fan-out.

## 10. Build sequence

1. **Scaffold** `python/model_training/` package + `cli.py`; add deps; commit Stage-1 reference
   UBJ fixtures (converted from the May-2021 `.model` via xgboost 3.0).
2. **`next_score.py`** + unit tests asserting NSH/DSH on a hand-built drive fixture (exact match).
3. **`ingest.py`** — labels + weights from raw via `features.py`; assert intermediates exactly on
   a small season subset.
4. **`train_ep.py`** (Stage 1) → validate predictions vs `xgb_ep_model` reference within tolerance.
5. **`train_wp.py`** (Stage 1, spread + naive) → validate vs `xgb_wp_spread`/`xgb_wp_naive` refs.
6. **`figures.py`** + `validate.py` — calibration tables + 4 styled plots; eyeball vs keepers PNGs.
   (The faithful `cfb_qbr` GAM port — `qbr.py` — is already landed as the QBR reference.)
7. **Stage 2** — flip feature sets to the shipped contract + correct EP weights; **add `train_qbr.py`**
   (6-feat XGBoost; ESPN raw QBR target via `qbr_scrape.py → composite.csv`); train all three on full
   history; regression-gate each vs the shipped `.ubj`; copy into sdv-py under review.

## 11. Risks & open items

- **Stage-1 exactness ceiling.** R↔Py XGBoost will not bit-match; the contract is prediction
  tolerance + exact deterministic intermediates. If even tolerance fails, the divergence is in
  feature construction (modern processor vs 2021 R derivation) — diagnosable via the intermediate
  asserts.
- **2002–2003 coverage (now in scope, decision #12).** The R trained on 2002→2020; the CFB backfill
  floor is 2004. Stage 2 trains earliest-available→2025; 2002–2003 are attempted only if the backfill
  yields usable data — confirm the realized floor before training.
- **Gill Sans MT availability.** Windows-only; the fallback chain must keep the plots legible on
  Linux CI. Final brand-perfect renders may be a local (Windows) step.
- **cfbfastR hex logo asset.** Must be sourced and vendored into the repo (with attribution).
- **QBR target alignment.** The 6-feat `qbr_model.ubj` retrain depends on ESPN raw QBR
  (`composite.csv`) joining cleanly to per-QB-game `qbr_vars`; ESPN QBR coverage starts ~2004 and is
  QB-week grain. The shipped-model recipe is *reconstructed* (features known from `CFBPlayProcess`,
  target = ESPN QBR), not a literal script — validated by the Stage-2 parity gate. The faithful
  `cfb_qbr` GAM (`qbr.py`) is a 1-feat ancestor, not a drop-in for `qbr_model.ubj`.
- **Model handoff.** Overwriting sdv-py's bundled `.ubj` is consequential; kept manual + reviewed,
  never automated by this pipeline.
