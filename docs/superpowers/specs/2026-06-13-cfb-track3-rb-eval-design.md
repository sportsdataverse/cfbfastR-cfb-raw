# CFB RB-Eval (DAKOTA xREPA) — Design Spec

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repo:** `cfbfastR-cfb-raw` (Python/uv) — new `python/rb_eval/` package
- **Source of truth (R original):** `../cfb-pbp-analysis/rb_eval_model.R`
- **Program:** **Track 3** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`).
  Tracks 1 (EP/WP/QBR), 2 (fourth-down), 4 (pregame WP + Five Factors), 5 (CPOE), 6 (NFL EP/WP) are specced separately.

## 1. Goal

Port the DAKOTA-lineage R RB-evaluation model to Python in `cfbfastR-cfb-raw`, producing
**expected rushing EPA (xREPA)** per rusher-season via a **weighted bivariate GAM** on prior-season
`epa_per_play` and `success`. The output is a **season-grain metric** — a CSV/parquet table of
`(rusher_player_name, season, exp_rb_epa, unadjusted_epa, ...)` plus a LOSO calibration figure and
table. Unlike Tracks 1–2, this model is NOT bundled into sdv-py as a `.ubj`; it is a downstream
analytical artifact.

Training data come from the backfill's per-game **`final.json` plays (= `CFBPlayProcess` output)**,
already carrying `yds_rushed`, `rush`, `rusher_player_name`, `start.down`, `start.distance`,
`EPA`, `pos_team`, and the pre-computed `highlight_yards`, `second_level_yards`, `open_field_yards`
columns.

## 2. Background — what the R source established

The R script `rb_eval_model.R` (akeaswaran/cfb-pbp-analysis) does the following:

1. **Filtering:** rushing plays (`rush==1`), valid `posteam` + `epa` + `rusher_player_name`,
   name != "TEAM", seasons 2006–2019 from cfbfastR-data RDS files.
2. **Play-level feature derivation:** `fo_success` (Football Outsiders success rate by down),
   `is_rush_opp` (`yds_rushed >= 4`), Football Outsiders yardage decomposition
   (`adj_yardage`, `line_yards`, `second_level_yards`, `open_field_yards`, `highlight_yards`).
3. **Per-(rusher, season) summarization:** `n_plays`, `n_opps`, `unadjusted_epa`, `epa`
   (clamp < −4.5), `success`, `highlight_yards/n_opps`.
4. **Minimum volume filter:** `n_plays > 100`.
5. **Lag by 1 season** (sorted within rusher by season): `lepa`, `lsuccess`, `lhlite_yds`,
   `lunad_epa`, `lplays`; **weight = (n_plays² + lplays²)^0.5**.
6. **GAM training:** `mgcv::gam(target ~ s(epa_per_play) + s(success), weights=weight)` with
   `target = unadjusted_epa` (current season), `epa_per_play = lepa`, `success = lsuccess`.
7. **LOSO CV** by season; calibration of binned `exp_rb_epa` vs actual `unadjusted_epa`.

Key finding: `highlight_yards` is **computed but NOT used in the GAM formula** — it appears in
`model_data` but the `gam()` call is `target ~ s(epa_per_play) + s(success)` only. It is
included in the output as an analytical column for descriptive use.

## 3. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Port location | `cfbfastR-cfb-raw/python/rb_eval/` (sibling to `model_training/`). |
| 2 | Training-data source | The backfill's `final.json` plays — **not** cfbfastR-data RDS files. The `final.json` already carries all needed columns from `CFBPlayProcess`. |
| 3 | Season floor | The R script trained on 2006–2019. The backfill floor is 2004 (see §11 open items). Train from **earliest available season → 2025**; the LOSO loop naturally skips seasons with no prior-season lag. |
| 4 | Highlight-yards decomposition | **Reuse the pre-computed columns from `final.json`** (`highlight_yards`, `second_level_yards`, `open_field_yards`, `adj_rush_yardage`) rather than recomputing. The sdv-py formula diverges from the R source (see §5.1); the pre-computed columns ARE the sdv-py-canonical values and are what any downstream consumer sees. |
| 5 | GAM engine | `pygam.LinearGAM(s(0) + s(1))` fit with `sample_weight=weight`. Already in `pyproject.toml` under `[dependency-groups] gam`. |
| 6 | Model persistence | Pickle (`joblib.dump`) the fitted GAM alongside the calibration table and figure. The pickle is a **local analytical artifact** — NOT bundled into sdv-py. |
| 7 | Output | `rb_eval/output/xrepa_loso.parquet` (per-rusher-season LOSO predictions), `xrepa_final.pkl` (full-data GAM), `calibration.parquet` / `.csv` / `_calibration.png`. |
| 8 | Figures | plotnine + bespoke cfbfastR styling (garnet `#500f1b`, Gill Sans MT fallback). The R calibration plot is a simple scatter + y=x dashed reference + calibration-error annotation; replicate that shape with cfbfastR brand polish using the existing `figures.write_calibration` helper from `model_training/figures.py` (reuse, do not duplicate). |
| 9 | fo_success | Compute fresh from `yds_rushed` + `start.down` + `start.distance` (exactly as the R `case_when`). Do **not** rely on `EPA_success`, which differs for `down>=3` with `yds_rushed >= start_ydstogo` but still leaving gains short of a first down. |
| 10 | Module home | `python/rb_eval/` — season-grain metric, distinct data shape (aggregation-first, no per-play training labels), deserves its own subpackage. Re-exports the `figures.write_calibration` helper from `model_training`. |

## 4. Data flow

```
CFB backfill: scrape_cfb_json.py → cfb/json/raw/{game_id}.json (verbatim ESPN summary)
   │
   ▼  rb_eval reads the backfill's final.json plays (CFBPlayProcess output)
python/rb_eval/features.py
   ├─ read cfb/json/final/{game_id}.json → rush plays (rush==1, rusher_player_name != "TEAM")
   ├─ compute fo_success (R formula: down1 → yds>=0.5*dist, down2 → 0.7*dist, down≥3 → dist)
   ├─ is_rush_opp = (yds_rushed >= 4)
   └─ reuse pre-computed highlight_yards, adj_rush_yardage (sdv-py-canonical values)
   │
   ▼
python/rb_eval/aggregate.py
   ├─ group by (rusher_player_name, season)
   ├─ summarize: n_plays, n_opps, unadjusted_epa, epa (clamped ≥-4.5), success, highlight_yards
   ├─ filter n_plays > 100
   └─ lag by 1 season (per rusher): lepa, lsuccess, lhlite_yds, lunad_epa, lplays
       weight = (n_plays² + lplays²)^0.5
   │
   ▼
python/rb_eval/train.py
   ├─ fit pygam.LinearGAM(s(0) + s(1)) on (lepa, lsuccess) → target=unadjusted_epa w/ sample_weight
   ├─ LOSO CV by season → xrepa_loso.parquet
   └─ joblib.dump full-data GAM → xrepa_final.pkl
   │
   ▼
python/rb_eval/validate.py  +  figures (via model_training.figures.write_calibration)
   ├─ bin exp_rb_epa by 0.05, compute weighted calibration error + weighted R²
   └─ calibration.parquet / .csv / _calibration.png
```

## 5. Module architecture — `python/rb_eval/`

| Module | Ports R section | Responsibility |
|---|---|---|
| `python/rb_eval/__init__.py` | — | Package marker + version. |
| `python/rb_eval/features.py` | `pbp_db` block (lines 18–52) | `load_rush_plays(final_dir, seasons)` — read `final.json`, filter rush plays, compute `fo_success` + `is_rush_opp`. Reuse pre-computed `highlight_yards`. Returns `pl.DataFrame`. |
| `python/rb_eval/aggregate.py` | `lrbs` block (lines 54–77) + `model_data` rename (79–86) | `build_rusher_seasons(rush_df)` — group by (rusher, season); `unadjusted_epa`, `epa` (clamped); `success`; `highlight_yards/n_opps`; filter n>100; lag 1 season; weight. `build_model_data(rusher_seasons)` — rename to `target/epa_per_play/success/highlight_yards/weight/season`. |
| `python/rb_eval/train.py` | GAM block (lines 104–115) + LOSO loop (lines 98–115) | `train_xrepa(model_data) -> LinearGAM` — `LinearGAM(s(0) + s(1)).fit(X, y, weights=w)`. `loso_cv(model_data, seasons) -> pl.DataFrame` — leave-one-season-out predictions. `save_model(gam, path)` / `load_model(path)` — joblib. |
| `python/rb_eval/validate.py` | calibration block (lines 119–176) | `calibration_table(cv_results, bin_size) -> pl.DataFrame`; `weighted_cal_error(table) -> float`; `weighted_r2(table) -> float`. |
| `python/rb_eval/cli.py` | — | Subcommands `features \| aggregate \| train \| validate \| figures`. Flags `--final-dir`, `--out-dir`, `--seasons A:B`. |
| `tests/rb_eval/` | — | One `test_*.py` per module (see §6). |
| `tests/fixtures/rb_eval/` | — | Synthetic play frame; small hand-rolled rusher-season reference. |

## 5.1 Formula divergence: R source vs sdv-py (final.json)

The R script computes `adj_yardage = ifelse(yds_rushed > 10, 10, yds_rushed)` — capped at **10**.
sdv-py's `CFBPlayProcess` computes `adj_rush_yardage` capped at **8**, with a piecewise line_yards
formula that differs from the R `case_when`. Specifically:

| Quantity | R formula | sdv-py formula in final.json |
|---|---|---|
| `adj_yardage` | `min(yds, 10)` | `min(yds, 8)` (cap at 8) |
| `line_yards` (yds≥4, yds≤8) | `0.5 * adj` | `3 + 0.5 * (adj - 3)` |
| `line_yards` (yds>8) | `0.5 * adj` | `5.5` (flat) |
| `second_level_yards` (yds≥5) | `0.5 * (adj - 5)` | `0.5 * (adj - 4)` (threshold at 4) |
| `open_field_yards` (yds>10) | `yds - adj` | `yds - adj` (same, but adj differs) |

**Consequence:** the `highlight_yards` values pre-baked into `final.json` are computed with the
sdv-py formula, not the R formula. Decision #4 uses the pre-computed values and treats the sdv-py
formula as canonical. The LOSO calibration is evaluated against itself (sdv-py-formula features →
sdv-py-formula target), so internal consistency holds. This is a **documented divergence** from the
R-original `rb_eval_model.R` — not a bug, but a policy decision to match cfbfastR-py's inference
environment.

The `fo_success` formula (used for `success`) is computed **fresh** (not pre-baked as a column)
because it differs from `EPA_success` for the `down>=3` case and is not stored directly in
`final.json`. See §5.2.

## 5.2 fo_success derivation (must compute fresh)

`EPA_success` in `final.json` diverges from the Football Outsiders `fo_success` definition
for plays where `down >= 3` and `yds_rushed >= start_ydstogo` but the drive did not convert
(a rare edge case related to penalty-extended drives). The R formula is authoritative:

```python
# fo_success = True if the run meets the FO threshold for the given down
# down1: yds >= 0.5 * dist; down2: yds >= 0.7 * dist; down>=3: yds >= dist
fo_success = (
    pl.when(pl.col("start.down") == 1)
    .then(pl.col("yds_rushed") >= 0.5 * pl.col("start.distance"))
    .when(pl.col("start.down") == 2)
    .then(pl.col("yds_rushed") >= 0.7 * pl.col("start.distance"))
    .otherwise(pl.col("yds_rushed") >= pl.col("start.distance"))
)
```

## 5.3 Lag derivation (within-rusher sort by season)

The R `lag()` call inside `group_by(rusher_player_name, season) %>% mutate(...)` operates on the
**sorted within-rusher** sequence — because the data is already sorted by season before the group_by
and the lag is along the season axis. The polars equivalent:

```python
# Sort by (rusher_player_name, season) before lagging
df.sort(["rusher_player_name", "season"]).with_columns(
    lepa=pl.col("epa").shift(1).over("rusher_player_name"),
    lsuccess=pl.col("success").shift(1).over("rusher_player_name"),
    lhlite_yds=pl.col("highlight_yards").shift(1).over("rusher_player_name"),
    lunad_epa=pl.col("unadjusted_epa").shift(1).over("rusher_player_name"),
    lplays=pl.col("n_plays").shift(1).over("rusher_player_name"),
)
```

The lag produces `null` for each rusher's first season; those rows are dropped before GAM fitting
(they have no `lepa`/`lsuccess` predecessor). `weight = (n_plays**2 + lplays**2)**0.5` is
undefined for the first-season row and also dropped.

## 6. Test strategy

All tests run offline. No live API calls. Three fixture types:

1. **Synthetic play frames** (`tests/fixtures/rb_eval/synth_plays.json`) — hand-crafted rows for
   testing `fo_success` computation, `is_rush_opp`, and highlight-yards reuse.
2. **Synthetic rusher-season frame** — generated inline in test helpers, covering the lag/weight
   logic and the n>100 filter boundary.
3. **GAM smoke test** — fits `LinearGAM(s(0)+s(1))` on a 50-row synthetic `model_data`; asserts
   output shape (one prediction per row) and that fitted values are finite floats.

| Test file | Covers |
|---|---|
| `tests/rb_eval/test_features.py` | `load_rush_plays` filter + `fo_success` formula (all 3 downs, edge: yds=0) |
| `tests/rb_eval/test_aggregate.py` | `build_rusher_seasons` epa clamp, `n_plays>100` filter, lag 1-season, weight formula |
| `tests/rb_eval/test_train.py` | `train_xrepa` shape/type; `loso_cv` produces one row per test-season rusher; `save`/`load` roundtrip |
| `tests/rb_eval/test_validate.py` | `calibration_table` bin arithmetic; `weighted_cal_error` formula |
| `tests/rb_eval/test_cli.py` | subcommand presence; `--help` exits 0 |

## 7. GAM details

**Library:** `pygam` (`LinearGAM`), already in `[dependency-groups] gam` in `pyproject.toml`.

**Formula:** `LinearGAM(s(0) + s(1))` where column 0 = `epa_per_play` (prior-season clamp-epa),
column 1 = `success` (prior-season FO success rate). This is a direct port of
`mgcv::gam(target ~ s(epa_per_play) + s(success), ...)`.

**Fitting call:**
```python
from pygam import LinearGAM, s
gam = LinearGAM(s(0) + s(1))
gam.fit(X, y, weights=w)  # X shape (n, 2); y = target; w = weight
```

`pygam` exposes `sample_weight` via the `weights=` parameter of `.fit()`. Default spline order
and number of splines are left at pygam's defaults (matches mgcv's default thin-plate splines
in spirit, though not in basis type). No grid-search or cross-validation on the GAM
hyperparameters — the R source uses defaults, so we do too.

**Predict:**
```python
exp_rb_epa = gam.predict(X_test)  # shape (n,)
```

**Persistence:** `joblib.dump(gam, path)` / `joblib.load(path)`. Pickle is appropriate for a
local analytical artifact (not a sdv-py bundled model). Document the `pygam` version in a
`model_card.json` sidecar (version string + training date + season range + n_rushers).

## 8. LOSO cross-validation

Leave-one-season-out: for each held-out season `s`, train on `model_data[season != s]`, predict on
`model_data[season == s]`. Accumulate across all seasons into a single `cv_results` frame.

```python
for season in sorted(model_data["season"].unique()):
    train = model_data.filter(pl.col("season") != season)
    test  = model_data.filter(pl.col("season") == season)
    # drop rows with null lag features (first-season rushers)
    train = train.drop_nulls(["epa_per_play", "success", "weight"])
    if train.is_empty() or test.is_empty():
        continue
    gam = train_xrepa(train)
    preds = gam.predict(test[["epa_per_play", "success"]].to_numpy())
    cv_results.append(test.with_columns(pl.Series("exp_rb_epa", preds)))
```

The LOSO loop is the **only calibration signal** — there is no separate holdout set, matching the
R source exactly.

## 9. Figures — calibration plot

The R calibration chart plots `bin_pred_epa` vs `bin_actual_epa` with points sized by
`total_instances`, a y=x dashed reference line, and annotation text
("Higher/Lower than predicted") positioned in the upper-left and lower-right quadrants.
No faceting (xREPA is a scalar — no natural facet variable like WP's quarter). The figure is
produced by calling the existing `model_training.figures.write_calibration` helper **adapted**
for a non-faceted layout (pass `by="all"` or a constant column for the facet variable, then
suppress the facet strip in the theme).

Styling target: garnet `#500f1b`, `grey95`/`grey99` panels, Gill Sans MT with fallback,
`coord_equal()` (x and y ranges are the same EPA scale), calibration-error + weighted R²
caption. Points sized by `total_instances` (number of rushers in the bin, not plays).

**Data table:** emit `calibration.parquet` + `calibration.csv` alongside the PNG — same
convention as Track 1 figures.

## 10. CLI

```
uv run python -m rb_eval.cli features   --final-dir cfb/json/final  --out cfb/rb_eval/rush_plays.parquet
uv run python -m rb_eval.cli aggregate  --plays cfb/rb_eval/rush_plays.parquet  --out cfb/rb_eval/rusher_seasons.parquet
uv run python -m rb_eval.cli train      --seasons cfb/rb_eval/rusher_seasons.parquet  --out cfb/rb_eval/
uv run python -m rb_eval.cli validate   --loso cfb/rb_eval/xrepa_loso.parquet  --out cfb/rb_eval/
uv run python -m rb_eval.cli figures    --table cfb/rb_eval/calibration.parquet  --out cfb/rb_eval/
```

Each subcommand is idempotent. Default `--out cfb/rb_eval/`. A `--seasons A:B` flag filters to
that season range for development iteration.

## 11. Risks and open items

- **Season floor.** The R script trained on 2006–2019. The backfill floor is 2004 (ESPN's data
  starts around 2004). Confirm the actual first season with `> 100 rushing plays per rusher` once
  the backfill is populated for 2004–2005. The lag derivation means the effective prediction start
  is `floor + 1`. Set `--seasons 2006:2025` as the default until the 2004–2005 data quality is
  confirmed.
- **`yds_rushed` vs `statYardage` column name.** Investigation of `final.json` confirms
  `yds_rushed` is present (it is the `CFBPlayProcess`-derived column, not the raw ESPN
  `statYardage`). If a game predates the `CFBPlayProcess` enrichment step (e.g., a partial backfill
  from raw JSON without reprocess), `yds_rushed` may be null; the feature loader must filter those
  rows. `statYardage` is available as a fallback but may include penalty yardage — do not use it
  as a substitute.
- **`pos_team` is a team ID (integer), not a name string.** The column `pos_team` in `final.json`
  carries the ESPN team ID (e.g., `194` for Ohio State), not the name. `start.pos_team.name` is
  the string. The filter `!is.na(posteam)` from the R source maps to `pos_team.is_not_null()` in
  polars. No downstream use of the team name is required by the model; this is a filter-only column.
- **highlight_yards = 0 for n_opps = 0.** The per-rusher-season aggregation computes
  `highlight_yards = sum(highlight_yards) / n_opps`. If `n_opps == 0` (no run gained >= 4 yards),
  this is a division by zero. Guard with `pl.when(pl.col("n_opps") > 0).then(...).otherwise(0.0)`.
  (The R script silently propagates NaN here; the n>100 filter mostly eliminates zero-opp rushers.)
- **pygam default spline basis differs from mgcv.** mgcv uses thin-plate regression splines by
  default; pygam uses B-splines. The LOSO calibration is the validation target — if calibration
  error is substantially worse than the R original, consider tuning `n_splines` or `lam` on the
  training splits. The R source uses `mgcv` defaults; we accept a reasonable calibration match
  (within 10% relative error) as the parity bar.
- **Module home.** Decision #10 places this in `python/rb_eval/`. The alternative was
  `python/model_training/` (Track 1 home), but the season-grain aggregation-first design warrants
  its own subpackage. This is consistent with the program's "each track, its own spec → plan →
  build cycle" principle.
- **No sdv-py handoff.** The xREPA artifact is NOT bundled into sdv-py. It is a research-grade
  metric table produced by this pipeline, consumed by downstream dashboards (e.g.,
  `game-on-paper-app`). Document this explicitly in `rb_eval/README.md`.
