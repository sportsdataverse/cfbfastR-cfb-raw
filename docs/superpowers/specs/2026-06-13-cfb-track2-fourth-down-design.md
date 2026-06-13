# CFB Fourth-Down Yards-Gained Model — Design Spec

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repo:** `cfbfastR-cfb-raw` (Python/uv) — new `python/model_training/fourth_down/` sub-package
- **Source of truth (notebook):** `../cfb-pbp-analysis/fourth-downs.ipynb` (akeaswaran / Jason Lee lineage)
- **Decision layer (R):** `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfb4th` (`R/decision_functions.R` — `get_go_wp()`)
- **Program:** **Track 2** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`). Tracks 1/3–6 are specced separately.

## 1. Goal

Port the cfb4th yards-gained XGBoost model to Python, living in `cfbfastR-cfb-raw/python/model_training/fourth_down/`, to **retrain `fd_model.ubj`** (the 5-feature, 76-class `multi:softprob` model that projects yards gained on any 3rd/4th-down play) on the full **earliest-available→2025** history that the CFB backfill produces. The retrained model is a drop-in for the yards-gained core consumed by cfb4th's `get_go_wp()` decision layer (go/punt/FG expected-value).

**Scope of this track:** the **yards-gained model itself** — the probabilistic distribution over `clip(yards_gained, -10, 65) + 10 → class 0..75`. The full decision-EV layer (`get_go_wp`, `get_fg_wp`, `get_punt_wp`) is **out of scope** for the core model port but is documented as a downstream integration point in §10.

The retrained `.ubj` is kept in `cfbfastR-cfb-raw/python/model_training/fourth_down/` for human review; it is not automatically deployed to any consumer package.

## 2. Background — what the investigation established

### 2.1 Recipe confirmation (xgboost introspection)

The shipped `fd_model.model` / `fd_model.ubj` (bundled inside the cfb4th R package as internal sysdata) introspects to:

| Artifact | Trees | Formula | Confirms |
|---|---|---|---|
| `fd_model.model` (cfb4th sysdata) | **11 932** | 11 932 ÷ 76 classes = **157 rounds** | `nrounds=157` from notebook |

The tree count uniquely confirms the recipe: `nrounds=157`, `num_class=76`.

### 2.2 Notebook lineage

`fourth-downs.ipynb` (akeaswaran, in `cfb-pbp-analysis`) is a faithful Python port of Jason Lee's original cfb4th R training script (`data-raw/_go_for_it_cfb_mod.R`). Both notebooks:
- Train on 2014–2020 cfbscrapR PBP joined to CFBD betting lines
- Use the same 5-feature set, same `multi:softprob` / 76-class objective, same hyperparameters
- The R script (`_go_for_it_cfb_mod.R`) tuned via a 20-point Latin hypercube grid and selected the best params; the notebook hard-codes those final params verbatim

The Python notebook is the authoritative training recipe for this port. The R script is confirmatory.

### 2.3 Feature and label contracts

All facts below are read directly from `fourth-downs.ipynb` cells 5–7 and confirmed against `_go_for_it_cfb_mod.R` lines 46–91:

**Features (5, exact):**

| Column | Source | Notes |
|---|---|---|
| `down` | `start.down` on the play | 3 or 4 (filter gate) |
| `distance` | `start.distance` | yards to first down; > 0 (filter gate) |
| `yards_to_goal` | `start.yardsToEndzone` | > 0; combined filter: `distance <= yards_to_goal` (see §2.4 filter #5) |
| `posteam_total` | derived | `home_total = (homeTeamSpread + overUnder)/2`; `away_total = (overUnder - homeTeamSpread)/2`; `posteam_total = home_total if is_home else away_total` |
| `posteam_spread` | derived | `posteam_spread = homeTeamSpread if is_home else -homeTeamSpread` (positive = underdog; negative = favorite) |

**Label:** `clip(yardsGained, -10, 65) + 10` → integer class `0..75` (class 0 = 10-yard loss; class 10 = no gain; class 75 = 65-yard gain).

**XGBoost params (exact, from notebook cell 7):**

```python
nrounds = 157
params = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": 76,
    "eta": 0.07,
    "gamma": 4.325037e-09,
    "subsample": 0.5385424,
    "colsample_bytree": 0.6666667,
    "max_depth": 4,
    "min_child_weight": 7,
}
```

### 2.4 Data filter (training rows selected)

From notebook cell 6 + R script lines 28–75 (aligned logic; see filter #5 for one directional difference between sources — R script is canonical):

1. `down in {3, 4}`
2. `(rush == 1 OR pass == 1) OR first_down_penalty == 1`
3. `distance > 0`
4. `yards_to_goal > 0`
5. `distance <= yards_to_goal` — keep plays where yards needed for a first down does not exceed yards to the end zone (avoids geometrically impossible distances). The R script (`filter(distance <= yards_to_goal)`) is the primary recipe and this direction is semantically correct. The notebook used the opposite filter direction, which is a notebook-specific artifact; the R script's formulation is adopted here.
6. `posteam_total` not null (spread + overUnder must be present)
7. `posteam_spread` not null

### 2.5 Spread convention

The notebook derives posteam_total/spread from game-level `spread` + `overUnder` joined from CFBD lines CSV. In the ESPN backfill's `final.json`:
- `homeTeamSpread` (doc level) = the spread from the home team's perspective. Convention: **negative = home team is favored** (CFBD convention). Example: `homeTeamSpread = 48.5` means the home team is a +48.5 underdog — i.e., the away team is the heavy favorite.
- `overUnder` (doc level) = game total.
- Per-play `start.pos_team_spread` = already computed by `CFBPlayProcess` from the game-level `homeTeamSpread` with the posteam perspective applied (negative if posteam is favored, positive if underdog).

**The per-play `start.pos_team_spread` column is available on every play in `final.json` and already carries the correct posteam-perspective spread.** This is the direct equivalent of `posteam_spread` — no re-derivation needed. Similarly, `posteam_total` must be derived from the doc-level `homeTeamSpread` + `overUnder` plus the per-play `start.is_home` flag (since `CFBPlayProcess` does not pre-compute `posteam_total` on individual plays).

**Derivation in Python (from `final.json` per play):**

```python
home_total = (homeTeamSpread + overUnder) / 2
away_total = (overUnder - homeTeamSpread) / 2
posteam_total = pl.when(pl.col("start.is_home") == True).then(home_total).otherwise(away_total)
posteam_spread = pl.col("start.pos_team_spread")   # already correct; skip re-derivation
```

The doc-level `homeTeamSpread` and `overUnder` are broadcast to every play from the game document (both are already available as per-play columns in `final.json` — confirmed by introspection).

### 2.6 Relationship to cfb4th decision layer

`cfb4th/R/decision_functions.R::get_go_wp()` is the decision layer that wraps the yards model. Its flow:
1. Calls `stats::predict(fd_model, data)` on the 5-feature play-situation matrix.
2. Expands the 76-class probability vector into a long-form `(play × gain)` frame.
3. For each possible gain value (`−10..65`), caps gains at `yards_to_goal` (TD) or floors losses to keep the ball on the 1-yard line, then updates the game situation (posteam flip on turnover-on-downs, score adjustment on TD, clock run-off, timeout swap).
4. For each resulting situation, calls `add_ep()` + `add_wp()` (the EP/WP models from Track 1).
5. Weights each outcome's WP by the predicted probability to get `go_wp` (the expected WP of going for it).

This track ports **only** the yards model (step 1). Steps 2–5 — which depend on the EP/WP models (Track 1) already being retrained — are the decision-layer integration point noted in §10.

## 3. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Port location | `cfbfastR-cfb-raw/python/model_training/fourth_down/` — a new sub-package under the existing `model_training/` package. Keeps fourth-down logic isolated without fragmenting the repo. |
| 2 | Training-data source | The **CFB backfill's `final.json`** per-game plays — the same source as Track 1. Filtered to `down in {3,4}`, plays with rush or pass or first-down-penalty, and present `overUnder`/`homeTeamSpread`. |
| 3 | Feature derivation | `posteam_spread = start.pos_team_spread` (already on the play); `posteam_total` derived from doc-level `homeTeamSpread` + `overUnder` broadcast to plays via `start.is_home`. The 5-feature matrix is exactly `[down, distance, yards_to_goal, posteam_total, posteam_spread]`. |
| 4 | Label | `label = int(clip(yardsGained, −10, 65) + 10)` → integer class `0..75`. `yardsGained` is the `yardsGained` column on the play record. |
| 5 | No label for non-plays | `first_down_penalty` plays where no yardage is recorded: the R script and notebook include them in the filter but the notebook drops rows where `yards_gained` is null (the label step fails silently on null). **Decision: require non-null `yardsGained`; drop null rows after the filter. First-down-penalty plays without recorded `yardsGained` are excluded.** |
| 6 | Params | Exactly the notebook's confirmed recipe: `nrounds=157`, `num_class=76`, `eta=0.07`, `gamma=4.325037e-09`, `subsample=0.5385424`, `colsample_bytree=0.6666667`, `max_depth=4`, `min_child_weight=7`. Stored in `fourth_down/constants.py`. |
| 7 | Season coverage | Train from **earliest available backfill season → 2025** (same as Track 1, decision #12). Original training was 2014–2020; the wider window is the goal of the port. |
| 8 | Validation | **Structure assert** (5 feats, 76 classes, 157 × 76 = 11 932 trees) against the shipped `fd_model.ubj` converted from cfb4th sysdata as a reference fixture. **Prediction-distribution** comparison (max-abs-diff tolerance) vs the reference on a shared feature matrix drawn from the test fixture. |
| 9 | Decision-EV layer | **Out of scope for the core model.** Noted in §10 as the downstream integration point. A stub `fourth_down_decision.py` with the `get_go_wp_py()` signature (no implementation) is included so the integration contract is documented. |
| 10 | Model handoff | **Manual, reviewed** — the retrained `fd_model.ubj` is kept in `python/model_training/fourth_down/` and copied to any consumer (cfb4th Python port, game-on-paper) under explicit review. Never auto-deployed. |
| 11 | No sample weights | The yards-gained model trains with **no sample weights** (neither the notebook nor the R script passes a `weight=` argument to `xgb.train`). This differs from EP/WP (Track 1) which uses `ScoreDiff_W`. |
| 12 | `overUnder` availability | Confirmed present on `final.json` plays as a per-play broadcast column (`p.get('overUnder')` confirmed by introspection). If absent on a play (pre-odds-resolution games), that play is excluded from the training set (decision #5). |

## 4. Data flow

```
CFB backfill: scrape_cfb_json.py → cfb/json/raw/{game_id}.json   (verbatim ESPN summary)
   │
   ▼  reprocess_cfb_json.py → CFBPlayProcess (odds_override resolved) → cfb/json/final/{game_id}.json
fourth_down/features.py
   ├─ read cfb/json/final/{game_id}.json → plays
   ├─ filter: down in {3,4}, rush|pass|first_down_penalty, distance > 0, yards_to_goal > 0,
   │          distance <= yards_to_goal, overUnder not null, yardsGained not null
   ├─ derive: posteam_total from (homeTeamSpread + overUnder) / 2 using start.is_home
   │          posteam_spread = start.pos_team_spread (already on play)
   └─ label:  int(clip(yardsGained, -10, 65) + 10) → class 0..75
fourth_down/train.py
   └─ XGBoost DMatrix(X[5 feats], label=y, no weights)
      → xgb.train(FD_PARAMS, dtrain, nrounds=157)
      → save fd_model.ubj
fourth_down/validate.py
   └─ structure assert: num_features==5, num_class==76, n_trees==11932
      prediction-distribution vs reference fd_model.ubj (tolerance check)
      calibration: for each gain bin, predicted P(gain ≥ distance) vs empirical first-down rate
fourth_down/figures.py
   └─ plotnine: feature-importance bar chart + gain-distribution calibration plot
                (bespoke cfbfastR styling, data tables alongside PNGs)
```

The feature matrix is shallow: 5 numeric columns read from `final.json` plays that already carry the spread/overUnder broadcast. The label is derived from `yardsGained` (a raw ESPN field), which the processor does not transform.

## 5. Module architecture — `python/model_training/fourth_down/`

| Module | Responsibility |
|---|---|
| `__init__.py` | Package marker; re-exports `train_fourth_down`, `fd_features`, `FD_PARAMS`, `FD_FEATURES`. |
| `constants.py` | `FD_FEATURES` (list of 5 in exact model order), `FD_PARAMS` (XGBoost param dict), `FD_NROUNDS = 157`, `FD_NUM_CLASS = 76`, `FD_CLIP_LOW = -10`, `FD_CLIP_HIGH = 65`, `FD_LABEL_OFFSET = 10`. |
| `features.py` | `fd_features(plays_df) -> (X, y)` — filter plays frame, derive `posteam_total` + `posteam_spread`, build label, return `(pd.DataFrame[5 cols], np.ndarray[int])`. No weights (decision #11). |
| `train.py` | `train_fourth_down(df, nrounds=FD_NROUNDS) -> xgb.Booster` — wraps `fd_features` + `xgb.train`. |
| `validate.py` | `assert_structure(booster)` — verifies `num_features==5, num_class==76, n_trees==11932`; `prediction_parity(model_a, model_b, X)` re-used from Track 1 `validate.py`; `calibration_fd(booster, X, y_true_yards)` — predicted first-down probability vs empirical rate, binned by field zone. |
| `figures.py` | plotnine feature-importance bar + gain-distribution calibration (bespoke garnet `#500f1b` styling, sidecar data tables). |
| `cli.py` | `train-fd` subcommand: `--final-dir`, `--out`, `--seasons`. |
| `fourth_down_decision.py` | **Stub only** — `get_go_wp_py(pbp_df, fd_model, ep_model, wp_model) -> pd.DataFrame` signature with a `NotImplementedError` body. Documents the integration contract with the Track-1 EP/WP models without implementing the decision layer. |

The Track-1 `model_training/` package already exists and provides `validate.prediction_parity`, `figures.write_calibration`, and the `constants` discipline conventions. The fourth-down sub-package reuses those helpers rather than duplicating them.

## 6. Feature and label derivation (the portable IP)

### 6.1 `posteam_total` derivation

The notebook derives this from CFBD game-level lines. In the backfill:

```python
# doc-level fields broadcast to every play by CFBPlayProcess
home_total = (pl.col("homeTeamSpread") + pl.col("overUnder")) / 2
away_total = (pl.col("overUnder") - pl.col("homeTeamSpread")) / 2
posteam_total = pl.when(pl.col("start.is_home") == True).then(home_total).otherwise(away_total)
```

`start.is_home` is a boolean (True if the possessing team is the home team). Both `homeTeamSpread` and `overUnder` are per-play columns in `final.json` (broadcast from the game document by `CFBPlayProcess`).

### 6.2 `posteam_spread`

`start.pos_team_spread` is already set by `CFBPlayProcess` with the correct posteam perspective. Positive = possessing team is the underdog; negative = possessing team is the favorite. This matches the notebook's `posteam_spread = -spread if offense == away else spread` computation (where the notebook's `spread` is the home-team-perspective spread in CFBD convention, negative for home favorite). No re-derivation needed in the feature builder.

### 6.3 Label

```python
label = (pl.col("yardsGained").clip(FD_CLIP_LOW, FD_CLIP_HIGH) + FD_LABEL_OFFSET).cast(pl.Int32)
```

Class `0` = 10-yard loss; class `10` = no gain; class `75` = 65-yard gain. The 76 classes cover the integer range `[−10, 65]` inclusive (76 values).

### 6.4 Filter

Applied in order (each drops rows before the next):

1. `start.down in {3, 4}` — only third and fourth downs
2. `(rush == True OR pass == True) OR firstD_by_penalty == True` — only plays where a ball-carrying or penalty-awarded outcome is possible (excludes kick plays, timeouts, etc.)
3. `start.distance > 0` — avoid degenerate distances
4. `start.yardsToEndzone > 0` — avoid goal-line edge cases
5. `start.distance <= start.yardsToEndzone` — distance-to-first must not exceed distance-to-endzone (R script `filter(distance <= yards_to_goal)`)
6. `overUnder not null AND homeTeamSpread not null` — spread inputs required for `posteam_total`
7. `yardsGained not null` — label required

`rush` and `pass` in `final.json` are boolean columns set by `CFBPlayProcess`. `firstD_by_penalty` maps to the `firstD_by_penalty` column (or `start.first_down_penalty` depending on schema — the features module must handle both names defensively).

## 7. Validation

### 7.1 Structure assert

Against the cfb4th reference `fd_model.ubj` (converted once from cfb4th sysdata):

```
booster.num_features() == 5
booster.num_class == 76          # from save_config()
n_trees == 11932                 # from num_boosted_rounds() * num_class = 157 * 76
feature_names == FD_FEATURES     # exact column order
```

A test fixture `tests/fixtures/model_training/fd_model.ubj` is produced once by loading cfb4th's internal `fd_model` via `xgboost::xgb.save(fd_model, "fd_model.ubj")` from R. This is the parity oracle.

### 7.2 Prediction-distribution parity

Same harness as Track 1: `prediction_parity(reference, retrained, X_shared)` on a shared 5-feature matrix drawn from the fixture. Target: max-abs-diff < documented tolerance (the reference and retrained models will differ due to different training windows, so a numerical close-match is not expected — the structure assert and distribution shape are the relevant checks). The tolerance is **documented as N/A for this track**: we are retraining on more data, not replicating the exact model.

**Sensible check instead:** the retrained model's predicted gain distributions (marginal P(gain ≥ distance) by `(down, distance, yards_to_goal)` bucket) should be monotonically reasonable (e.g., P(first down on 4th-and-1) > P(first down on 4th-and-10)).

### 7.3 Calibration

For each play in a held-out season: `P(first_down) = sum(P(gain ≥ distance for that play))`. Group by `(yards_to_goal, distance)` decile buckets and compare predicted first-down probability against empirical first-down rate. Emit as a calibration table (parquet + csv) and a plotnine calibration scatter.

## 8. Figures (`figures.py`) — bespoke styling + data tables

Two figures:

- **Feature importance bar chart** — plotnine `geom_col`, sorted by gain, garnet `#500f1b`, with the 5 feature names on the y-axis. Emits `fd_feature_importance.png` + `.csv`.
- **Gain-distribution calibration** — predicted `P(gain ≥ distance)` vs empirical first-down rate, faceted by down (3rd vs 4th). Points sized by play count. Loess smooth. `y=x` dashed reference. Calibration-error caption. Emits `fd_calibration.png` + `.csv`.

Styling: garnet `#500f1b` accent, `grey95`/`grey99` panel/plot fills, Gill Sans MT with cross-platform fallback chain (identical to Track 1 `figures.py` conventions).

## 9. Dependencies

No new dependencies beyond what Track 1 already adds:
- `xgboost>=2.0` (already required by Track 1)
- `plotnine`, `statsmodels`, `pillow` (already in the `figures` dependency group)
- `polars` / `pyarrow` / `pandas` / `numpy` (already present)

The `fourth_down/` sub-package inherits the Track-1 `model_training` dep group. No additional `pyproject.toml` changes needed beyond what Track 1 introduced.

## 10. Decision-EV integration point (out of scope, documented)

`cfb4th`'s `get_go_wp()` (R) consumes the yards-gained model as follows:

1. Passes the 5-feature situation matrix to `predict(fd_model, data)` → 76-class probability vector per play.
2. Expands into a long `(play × gain)` frame: for each class `k`, `gain = k − 10` (reversing the label offset).
3. Caps gain at `yards_to_goal` (TD), floors loss to keep ball at the 1.
4. Updates game situation per outcome (possession flip on turnover-on-downs, score +6 on TD, spread flip, clock −6 seconds).
5. Calls EP/WP models (Track 1 output) on each resulting situation.
6. Weights WP by predicted probability → `go_wp = Σ P(gain=k) × WP(situation after gain=k)`.

A Python port of this decision layer — `fourth_down_decision.py::get_go_wp_py(pbp_df, fd_model, ep_model, wp_model)` — is the natural follow-on to Track 2 once Track 1's EP/WP models are retrained. The stub included in this track documents the function signature and the gain-expansion logic without implementing the full decision tree (which requires the EP/WP models as inputs).

The cfb4th R package also includes `get_fg_wp()` (wraps a GAM `fg_model`) and `get_punt_wp()` (wraps a `punt_df` distribution). These are separate models, not in scope.

## 11. Build sequence

1. **Scaffold** `python/model_training/fourth_down/` sub-package (`__init__.py`, `constants.py`); commit reference fixture `tests/fixtures/model_training/fd_model.ubj` (converted from cfb4th sysdata).
2. **`features.py`** + unit tests: filter logic, `posteam_total` derivation, `posteam_spread` passthrough, label `clip+10`, shape/dtype assertions.
3. **`train.py`** (trainer) + structure-assert test: 5-feat / 76-class / `multi:softprob` on a synthetic frame; verify `n_trees == nrounds * num_class` on a minimal run.
4. **`validate.py`** (structure assert + calibration harness) + tests against the reference fixture.
5. **`figures.py`** (feature-importance + calibration plots) + smoke test (PNG + CSV emitted).
6. **`cli.py`** (`train-fd` subcommand) + test.
7. **`fourth_down_decision.py`** — stub only; `get_go_wp_py` signature + `NotImplementedError` + docstring documenting the integration contract.
8. **(Data-dependent, skip if no backfill)** Full `train-fd` run on earliest-available→2025; validate calibration; commit `fd_model.ubj` to `python/model_training/fourth_down/artifacts/` (gitignored by default; committed under review).

## 12. Risks and open items

- **`yardsGained` null rate in backfill.** `yardsGained` is null on some plays in `final.json` (confirmed by introspection: the sampled 3rd-down pass play had `yardsGained: None`). The feature builder drops these rows (decision #5). The null rate on 3rd/4th-down rush/pass plays should be measured in Phase 2 to confirm the training set is not severely degraded. If null rates are high for a particular season range, that range may need to be excluded.
- **`overUnder` / `homeTeamSpread` availability in pre-2014 backfill.** The original model trained on 2014–2020. The backfill may not have reliable spread data for seasons before ~2012 (ESPN's odds coverage is thinner pre-2014). A pre-training audit of `overUnder` completeness by season is needed before widening the window; if null rates are high the pre-2014 seasons should be excluded (or use a fallback of `overUnder=55.0` as a population mean — but this must be a deliberate decision, not a silent default).
- **`firstD_by_penalty` column name.** In `final.json` plays, the penalty-first-down flag may be named `firstD_by_penalty` or `start.firstD_by_penalty` depending on the processor version. The feature builder must handle both defensively (try `firstD_by_penalty`; fall back to `start.firstD_by_penalty`; default to `False` if absent).
- **`start.is_home` dtype.** Introspection shows `start.is_home` on the play record is an integer (`1`/`0`) or boolean depending on the play. The polars expression `pl.col("start.is_home") == True` handles both cleanly (polars int `1` == `True` is `True`). Confirm in Phase 2 tests.
- **cfb4th reference fixture.** Extracting `fd_model.ubj` from cfb4th's internal sysdata requires loading the package in R and running `xgboost::xgb.save(cfb4th:::fd_model, "fd_model.ubj")`. This is a one-time operation; the resulting fixture must be committed to `tests/fixtures/model_training/`. If cfb4th is not installed in the current environment, an alternative is to train a known-params model on a synthetic frame and assert structure only (not prediction parity vs the original).
- **Decision-EV layer timing.** `get_go_wp_py()` in `fourth_down_decision.py` depends on Track 1's retrained EP/WP models (via `add_ep()` / `add_wp()` equivalents). Do not implement the decision layer until Track 1's Stage-2 models are validated and their Python inference paths are confirmed working.
- **Parity bar for the retrained model.** Unlike Track 1 (which can compare against Dec-2020 reference models trained on the same years), the retrained fd model will use a wider data window than the original (earliest-available→2025 vs 2014–2020). Exact prediction parity is not achievable or expected. The validation contract is **structure** (5-feat / 76-class / 157-round) + **calibration reasonableness** (predicted first-down probability tracks empirical first-down rate), not numerical closeness to the reference.
