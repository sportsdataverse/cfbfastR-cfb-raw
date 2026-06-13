# CFB Track 5 — CPOE (Completion Percentage Over Expected) Design Spec

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repo:** `cfbfastR-cfb-raw` (Python/uv) — `python/cpoe/` package (proposed)
- **Source of truth (R original):** `../cfb-pbp-analysis/cpoe_model.R`
- **Program:** **Track 5** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`).
- **Prerequisite gate:** feasibility analysis (Phase 0 of the implementation plan) — this spec documents the analysis already performed as part of spec authorship and records its conclusion.

---

## 1. Goal

Deliver a **Completion Percentage Over Expected (CPOE)** metric for CFB play-by-play that is:

1. Trained on a CFB-native data source (not StatsBomb AMF).
2. Produces a per-play probability of completion that is well-calibrated.
3. Produces a per-QB (and per-play) CPOE = `actual_completion − expected_completion` that is a meaningful quarterback quality signal, not a noisy artefact of situation.

The R `cpoe_model.R` script shows *what a high-quality CPOE model looks like* and is the
conceptual ancestor — it is not portable as written, because it was trained on the
**StatsBomb American Football open data**, a source that does not cover the CFB population
and whose richest features (air yards, target field coordinates, target separation, QB pressure)
are entirely absent from the ESPN `final.json` that is the CFB backfill's canonical output.
**Track 5 is therefore a re-basing effort, not a port.**

---

## 2. Background — the R original: `cpoe_model.R`

### 2.1 Data source

`cpoe_model.R` fetches data from:

```
https://raw.githubusercontent.com/statsbomb/amf-open-data/main/data/events/tb12_events_dataset_{start}_{end}.csv
https://raw.githubusercontent.com/statsbomb/amf-open-data/main/data/plays/tb12_plays_dataset_{start}_{end}.csv
```

Seasons: 2017–18 through 2022–23 (six season-pairs). The "tb12" dataset is StatsBomb's
American football (NFL-style and college) open data offering. The event grain is one row per
**throw event**, joined to the play metadata. The data carries rich pre-snap and mid-air
throw attributes that do not exist in ESPN's CFB play-by-play.

### 2.2 Label

`label = event_success` — a binary flag on the throw event indicating whether the pass was
completed (1) or incomplete/intercepted (0).

### 2.3 Feature set (exact, from `pass_data` dplyr::select + mutate block, lines 31–63)

After joining events to plays and filtering to pass events with non-null `play_pass_made` and
non-null `event_success`, the following columns form the model input:

| Column | Source | Description |
|---|---|---|
| `event_pass_air_yards` | StatsBomb events | Straight-line distance (yards) the ball traveled through the air from QB release to target catch point. The single most predictive feature — proxies throw depth, difficulty, and receiver route depth. |
| `play_target_separation` | StatsBomb plays | Yards of separation between the intended receiver and the nearest covering defender at the moment of the throw. Direct measure of throw difficulty (tight window vs open receiver). |
| `play_qb_pressure` | StatsBomb plays | Boolean — was the QB under pressure on the play? Null-coerced to `FALSE`. |
| `endline_receiver_dist` | Derived: `110 − event_pass_target_x` | Yards from target field coordinate to the end line. Captures throw location along the field length (long/back-of-end-zone routes are harder). |
| `sideline_receiver_dist` | Derived: `min(event_pass_target_y, 53.33 − event_pass_target_y)` | Yards from target field coordinate to the nearer sideline. Sideline routes and out-of-bounds attempts are harder to complete. |

Two derived features are built from raw StatsBomb coordinates, then `event_pass_target_x` and
`event_pass_target_y` are dropped. The final feature matrix has **5 predictors**.

### 2.4 Hyperparameters

```
booster          = "gbtree"
objective        = "binary:logistic"
eval_metric      = "logloss"
eta              = 0.025
gamma            = 5
subsample        = 0.8
colsample_bytree = 0.8
max_depth        = 4
min_child_weight = 6
base_score       = mean(label)   # set to population completion rate
nrounds          = 560
```

### 2.5 Validation scheme

Leave-one-season-out (LOSO) cross-validation over 6 season-pairs (2017–18 → 2022–23).
Calibration is stratified by air-yards distance bucket:

| Bucket | Rule |
|---|---|
| Short | `event_pass_air_yards < 5` |
| Intermediate | `5 ≤ event_pass_air_yards < 15` |
| Deep | `event_pass_air_yards ≥ 15` |

Calibration metric: weighted mean of per-distance bin calibration error.

The script also trains a final model on all seasons (`pass_train`) and provides two sample
inputs/outputs via a `gt` table — confirming the final model is usable for inference.

---

## 3. CRITICAL FINDING — Feature-Availability Analysis

> **This section is the pivot of the spec. All design decisions flow from it.**

### 3.1 What is in ESPN `final.json` on pass plays

Inspecting `cfb/json/final/401628455.json` (a 2024 regular-season game, 169 plays of which 65
are pass plays) confirms the following columns are present on every pass play:

| Column | Present | Notes |
|---|---|---|
| `completion` | YES | 0/1 boolean — the label |
| `pass_attempt` | YES | filter for pass plays |
| `start.down` | YES | 1–4 |
| `start.distance` | YES | yards to first down |
| `start.yardsToEndzone` | YES | yards to end zone (field position proxy) |
| `pos_score_diff_start` | YES | score differential at snap |
| `start.TimeSecsRem` | YES | clock context |
| `start.spread_time` | YES | spread × time decay (WP feature) |
| `statYardage` | YES | total yards on play (generally accurate) |
| `passer_player_name` | YES | join key for CPOE aggregation |
| `receiver_player_name` | YES (sparse) | present ~40% of plays |
| `wp_before` | YES | win probability at snap |
| `EPA_pass` | YES | per-play EPA |
| `sack_vec` | YES | sack flag |
| `int` | YES | interception flag |
| `start.pos_team_score` / `start.def_pos_team_score` | YES | raw scores |
| `period` | YES | quarter |

### 3.2 What is NOT in ESPN `final.json`

| StatsBomb Feature | In ESPN `final.json`? | Notes |
|---|---|---|
| `event_pass_air_yards` | **NO** | Not collected/stored by ESPN. Partially recoverable from CFBD for some seasons (~2020+) but not available in the existing backfill pipeline. |
| `play_target_separation` | **NO** | Requires player-tracking data (NextGen Stats equivalent). Not available for college football at all. |
| `event_pass_target_x` / `_y` | **NO** | Requires player-tracking field coordinates. Not available for college football at all. |
| `endline_receiver_dist` | **NO** | Derived from `event_pass_target_x`; same unavailability. |
| `sideline_receiver_dist` | **NO** | Derived from `event_pass_target_y`; same unavailability. |
| `play_qb_pressure` | **NO** | Not in ESPN PBP text. Some CFBD advanced stats provide pressure rates at the team-game level, not per-play. |

**All five predictors of the StatsBomb-trained model are absent from the CFB ESPN backfill.**

Additionally, `yds_receiving` (which could partially proxy air yards) is null on approximately
91% of completions in the inspected game — it is not reliably populated. `statYardage` is
present but represents total yards on the play (including yards after catch), not throw distance.

### 3.3 Feasibility verdict

**The StatsBomb-trained CPOE model cannot be ported to CFB ESPN data as written.** Zero of its
five features exist in the `final.json` backfill. The re-basing path is mandatory, and the
richness of any CFB-native model will be substantially lower than the StatsBomb baseline.

The question then becomes: **is a reduced CFB-native CPOE feasible, and if so, what form?**

---

## 4. Candidate Approaches

Three candidate approaches are evaluated below in order of recommended priority.

### Approach A — Reduced game-state model (RECOMMENDED)

**Feasibility: YES, with clear caveats.**

Train a `binary:logistic` completion-probability model using only game-state features that are
present on every ESPN pass play. The feature set is orthogonal to throw-level features but
captures situation — an "opportunity-adjusted" completion probability.

**Proposed feature set (all present in `final.json`):**

| Feature | Rationale |
|---|---|
| `start.down` | Down matters — 3rd-and-long pressures the QB to attempt deep/difficult throws. |
| `start.distance` | Distance to first down is the primary throw-depth forcing function available to us. |
| `start.yardsToEndzone` | Field position: red-zone passes are shorter on average; backed-up QBs may throw more conservatively. |
| `pos_score_diff_start` | Score deficit forces more downfield attempts; lead situations favor safe passes. |
| `start.TimeSecsRem` | Late-game clock pressure changes throw selection. |
| `start.is_home` | Noise/crowd effects; minor but free. |
| `period` | Garbage time vs. competitive quarters signal different throw profiles. |
| `passing_down` (binary) | cfbfastR-derived: 2nd & 8+, 3rd/4th & 5+. Strong proxy for likely deep throw. |

**Label:** `completion` (0/1), filtered to `pass_attempt == True`.

**Training source:** CFB backfill `final.json`, earliest-available → 2025.

**What this measures:** CPOE relative to down/distance/field position/game state — i.e., "was
the QB completing passes at a higher rate than QBs in similar situations?" This is analogous to
nflfastR's `cpoe` variable, which similarly excludes air-yards and uses completion-opportunity
features.

**Key limitation:** This model conflates throw difficulty with game state. A QB who attempts
only check-downs on 3rd-and-9 will appear to outperform expectations, not because he is better,
but because his coach called safer routes. Approach A is a **situational CPOE**, not a
**throw-difficulty CPOE**. This is a material difference that must be documented prominently.

**Output:** `cp_model.ubj`, per-play `expected_completion`, per-pass-play `cpoe` =
`completion − expected_completion`, and a per-QB-game (or per-QB-season) CPOE aggregation.

---

### Approach B — CFBD air-yards enrichment (CONDITIONAL)

**Feasibility: CONDITIONAL on CFBD data availability and quality.**

The College Football Data (CFBD) API provides `advancedBoxScore` data that includes team-level
passing statistics, and for seasons approximately 2020+, the CFBD PBP endpoint occasionally
includes `air_yards` on individual plays. If CFBD air-yards coverage is at least 60% complete
for post-2020 seasons, a hybrid approach is viable: use Approach A features plus CFBD
`air_yards` where present, and train a two-tier model or impute from game state where missing.

**Investigation required (Phase 0, Task 0.2 in the plan):** Fetch a sample of CFBD PBP data
for seasons 2020–2024, compute the per-play fill rate of `air_yards`, and report it. If fill
rate ≥ 60%, this approach unlocks a stronger model for the covered period. If < 60%, it adds
complexity without meaningfully improving over Approach A.

**Not pursued further in this spec until Phase 0 Task 0.2 is complete.**

---

### Approach C — Requires player-tracking data (INFEASIBLE on current sources)

**Feasibility: NO on any currently available CFB data source.**

`play_target_separation` (the second most important feature in the StatsBomb model) requires
sub-second player tracking at the moment of throw. No public CFB data source provides this.
NFL Next Gen Stats (NGS) has it for the NFL; there is no equivalent college football dataset
as of 2026. This feature is absent from ESPN, CFBD, and every other publicly available CFB
source inspected.

Similarly, `play_qb_pressure` per-play is not available — CFBD provides team-level pressure
rates (not per play). `event_pass_target_x/y` require tracking coordinates.

**Conclusion:** a model that matches the StatsBomb CPOE's feature richness is infeasible for
CFB on any available public data source today.

---

## 5. Recommended Path: Approach A + Optional B

The recommended implementation path is:

1. **Implement Approach A** (reduced game-state model, 8 features, `final.json` only) as the
   primary deliverable. This produces a usable CFB CPOE that parallels the nflfastR convention.
2. **Investigate Approach B** in Phase 0. If CFBD air-yards fill rate is ≥ 60% for post-2020,
   add air yards as a 9th feature for the seasons where it is available and document the
   coverage gap.
3. **Never claim equivalence to the StatsBomb model** in documentation — the feature sets are
   categorically different. The CFB CPOE measures "situation-adjusted completion percentage",
   not "throw-difficulty-adjusted completion percentage".

---

## 6. Architecture

### 6.1 Grain

One row per pass attempt (`pass_attempt == True`), per game, per player. CPOE is computed at
the play grain and may be aggregated to QB-game or QB-season.

### 6.2 Data source

CFB backfill `cfb/json/final/{game_id}.json` plays — the `CFBPlayProcess` output. Same
source as Track 1 (EP/WP models). Filter: `pass_attempt == True`, drop `sack_vec == True`
(sacks are not pass attempts in the completion sense), drop `penalty_no_play == True`.

### 6.3 Module location

`python/cpoe/` (new package, sibling to `python/model_training/`):

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker + `__version__`. |
| `features.py` | `cp_matrix(df) → (X, y, keys)` — select/rename Approach-A feature columns from final.json plays. |
| `train_cp.py` | `train_cp(df) → Booster` — XGBoost `binary:logistic`, Approach-A params. |
| `cpoe.py` | `compute_cpoe(df, model) → pl.DataFrame` — per-play `expected_completion` + `cpoe`; `aggregate_cpoe(df, by) → pl.DataFrame` — QB-game or QB-season rollup. |
| `validate.py` | Calibration tables (binned by distance bucket approximation: `yards_to_gain` ≈ Short/Intermediate/Deep proxy), parity check. |
| `figures.py` | plotnine calibration plots, bespoke cfbfastR styling (same theme as Track 1). |
| `cli.py` | `ingest \| train \| predict \| validate \| figures` subcommands. |

### 6.4 Hyperparameters (Approach A — to be tuned)

Starting point (mirrors `cpoe_model.R` except `base_score` is removed — XGBoost 2.x
computes it from label mean automatically):

```
booster          = "gbtree"
objective        = "binary:logistic"
eval_metric      = "logloss"
eta              = 0.025
gamma            = 5
subsample        = 0.8
colsample_bytree = 0.8
max_depth        = 4
min_child_weight = 6
nrounds          = 400   # to be tuned by LOSO CV
```

The `nrounds` and `min_child_weight` will be tuned by LOSO CV (leave-one-season-out) in Phase
2, as the reduced feature set may need different tree depth/count to achieve calibration.

### 6.5 LOSO CV scheme

Leave-one-season-out over all available seasons (earliest-available → 2025). Calibration
stratified by `distance_bucket`:

| Bucket | Rule (ESPN proxy for air yards) |
|---|---|
| Short | `start.distance ≤ 3` |
| Intermediate | `4 ≤ start.distance ≤ 8` |
| Long | `start.distance ≥ 9` |

Note: `start.distance` (yards to first down) is the closest available proxy for throw depth.
It is a coarse proxy — on 3rd-and-10 the throw could still be a short screen. This is the key
limitation of Approach A; document it on every calibration figure.

### 6.6 Output artifacts

- `cp_model.ubj` — the trained completion-probability model.
- Calibration plots (`cp_cv_loso_calibration_{distance}.png` + `.csv` / `.parquet`) per
  distance bucket.
- Per-play inference emits columns: `expected_completion`, `cpoe` (when joined to actual play
  data).
- **This model is NOT bundled into sdv-py** — it is a `cfbfastR-cfb-raw` artifact only, used
  offline in the backfill/metrics pipeline. CPOE is a derived metric, not a shipped model like
  EP/WP. If it is eventually surfaced in sdv-py it should emit per-play and per-QB-season
  columns from precomputed tables, not from bundled model inference.

---

## 7. Feature Crosswalk (Approach A)

The following table maps shipped `final.json` column names to the CPOE feature names used in
`python/cpoe/features.py`:

| Feature name | Source column in `final.json` | Notes |
|---|---|---|
| `down` | `start.down` | 1-hot or ordinal (test both) |
| `distance` | `start.distance` | Yards to first down |
| `yards_to_goal` | `start.yardsToEndzone` | Field position |
| `pos_score_diff` | `pos_score_diff_start` | Score diff at snap |
| `secs_remaining` | `start.TimeSecsRem` | Clock |
| `is_home` | `start.is_home` | Binary |
| `period` | `period` | 1–4 |
| `passing_down` | `passing_down` | Binary: 2nd & 8+, 3rd/4th & 5+ |

Optional (if Approach B is pursued):

| Feature name | Source | Availability |
|---|---|---|
| `air_yards` | CFBD PBP API | Post-2020, ~40–70% fill rate (to be confirmed Phase 0) |

---

## 8. Figures

One calibration plot per distance bucket (Short / Intermediate / Long), same bespoke styling as
Track 1 (`figures.py` in `model_training/`):

- Points sized by `n_plays`, loess smooth, `y=x` dashed reference.
- "More times / Fewer times than expected" annotations.
- `facet_wrap(~distance_bucket)`.
- Calibration error caption per bucket + weighted overall.
- Caption note: "Distance buckets approximate throw depth via yards-to-first-down (ESPN pbp);
  not equivalent to air-yards stratification."

---

## 9. Risks and Limitations

| Risk | Severity | Mitigation |
|---|---|---|
| Reduced feature set produces a noisy CPOE signal | HIGH | Document clearly that this is game-state CPOE, not throw-difficulty CPOE. Validate calibration by distance bucket. |
| CFBD air-yards sparse/unreliable (Approach B) | HIGH | Phase 0 Task 0.2 gates Approach B. Only pursue if ≥ 60% fill rate. |
| LOSO calibration overfits on small seasons (pre-2010) | MEDIUM | Drop seasons with < 5,000 pass attempts from CV folds; include them in the final training run. |
| Receiver player name sparsity (~40% populated) | LOW | Approach A does not use receiver_player_name; skip it. |
| `statYardage` ≠ air yards (confirmed) | INFORMATIONAL | statYardage = total play yards, not throw distance. Do not use it as an air-yards proxy. |
| Model not equivalent to the StatsBomb original | INFORMATIONAL | Document prominently. The R `cpoe_model.R` is not a port target; it is a conceptual reference. |

---

## 10. Decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Base approach | Approach A (8-feature game-state model). |
| 2 | Training source | CFB backfill `final.json`, pass plays only (`pass_attempt == True`, `sack_vec == False`). |
| 3 | StatsBomb data | Not used. The R script's data source is unavailable for CFB and its features have no ESPN equivalent. |
| 4 | CFBD air yards | Investigate in Phase 0 Task 0.2; adopt only if ≥ 60% fill rate. |
| 5 | Grain | Play-level; aggregate to QB-game or QB-season for the CPOE metric. |
| 6 | Model handoff to sdv-py | Not bundled into sdv-py. This is a backfill/metrics artifact only. |
| 7 | Conventional commits | Yes. No AI co-author trailers (stated once, applies to all Track 5 commits). |

---

## 11. Open Items

- **Phase 0 Task 0.2:** CFBD air-yards fill-rate analysis — gates Approach B.
- **Hyperparameter tuning:** `nrounds` and `min_child_weight` TBD by LOSO CV (Phase 2).
- **Down encoding:** test ordinal integer vs. 1-hot (down_1..down_4 as in EP/WP models).
- **Sack handling:** confirm whether `sack_vec == True` plays should be treated as incomplete
  passes or dropped from the training set (current decision: drop).
- **Per-QB-season CPOE aggregation:** define the minimum pass-attempt threshold for a season
  CPOE to be considered reliable (proposed: ≥ 100 attempts).
