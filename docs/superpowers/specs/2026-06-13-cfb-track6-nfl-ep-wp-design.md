# NFL EP/WP/QBR Model-Training — Design Spec (Track 6)

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Program:** **Track 6** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`)
- **Implementation home:** **NOT `cfbfastR-cfb-raw`** — see §3 Decision #1 (cross-repo key decision)
- **Docs home:** `cfbfastR-cfb-raw/docs/superpowers/` (umbrella program lives here)

---

## 1. Goal

Survey and scope the retraining pipeline for sdv-py's three bundled NFL models
(`nfl/models/ep_model.ubj`, `wp_spread.ubj`, `qbr_model.ubj`) using a Python-native
pipeline sourced from nflverse play-by-play. The retrained models drop back into
sdv-py's `sportsdataverse/nfl/models/` so `NFLPlayProcess`'s EPA, WPA, and QBR
computation is reproducible, extendable, and no longer dependent on an opaque
one-time training event. This spec documents what the survey established, what
remains unknown, and the architectural decisions that must be settled before
implementation begins.

---

## 2. Survey findings — bundled NFL model introspection

### 2.1 Model inventory

Three models ship under `sportsdataverse/nfl/models/` and are loaded at module
import time in `nfl_pbp.py`:

| Model file | Size | `num_features` | `objective` | `num_class` | `num_trees` |
|---|---|---|---|---|---|
| `ep_model.ubj` | 9.0 MB | **8** | `multi:softprob` | **7** | **525** |
| `wp_spread.ubj` | 1.7 MB | **13** | `binary:logistic` | — | **760** |
| `qbr_model.ubj` | 72 KB | **6** | `reg:squarederror` | — | **45** |

All three have `feature_names = None` (no embedded feature name list), so the
inference contract is defined entirely by the column selection / rename logic in
`nfl_pbp.py` → `model_vars.py`.

### 2.2 Critical finding: NFL models are architecturally identical to the CFB models

The NFL `ep_model.ubj`, `wp_spread.ubj`, and `qbr_model.ubj` carry **the exact
same architecture** as the CFB counterparts already established by Track 1's
introspection:

- EP: 8 features, `multi:softprob`, 7 classes, 525 trees — identical to CFB
  `ep_model.ubj`.
- WP spread: 13 features, `binary:logistic`, 760 trees — identical to CFB
  `wp_spread.ubj`.
- QBR: 6 features, `reg:squarederror`, 45 trees — identical to CFB
  `qbr_model.ubj`.

This alignment is not coincidental. The sdv-py `nfl/model_vars.py` defines the
same feature-name lists (`ep_final_names`, `wp_final_names`, `qbr_vars`) as
`cfb/model_vars.py`, and the two `ep_class_to_score_mapping` dicts are
identical (`{0:7, 1:-7, 2:3, 3:-3, 4:2, 5:-2, 6:0}`). The inference code paths
in `NFLPlayProcess` are structurally mirrored from `CFBPlayProcess`.

**Implication:** the NFL models almost certainly share a common training lineage
with the CFB models — either they were trained together or one informed the
other. This needs to be confirmed by sourcing the nflfastR training scripts (§5).

### 2.3 NFL inference feature contracts (from `nfl_pbp.py` + `model_vars.py`)

The `nfl/model_vars.py` module defines the canonical feature lists. These mirror
the CFB contracts with NFL-specific source column names:

**EP features (8):** `ep_final_names`

```python
["TimeSecsRem", "yards_to_goal", "distance",
 "down_1", "down_2", "down_3", "down_4", "pos_score_diff_start"]
```

Source columns in the plays frame:

- Start (normal): `ep_start_columns` — `start.TimeSecsRem`, `start.yardsToEndzone`,
  `start.distance`, `down_1..4`, `pos_score_diff_start`
- Start (touchback): `ep_start_touchback_columns` — same but
  `start.yardsToEndzone.touchback`
- End: `ep_end_columns` — analogous `end.*` columns

**WP features (13):** `wp_final_names`

```python
["pos_team_receives_2H_kickoff", "spread_time", "TimeSecsRem",
 "adj_TimeSecsRem", "ExpScoreDiff_Time_Ratio", "pos_score_diff_start",
 "down", "distance", "yards_to_goal", "is_home",
 "pos_team_timeouts_rem_before", "def_pos_team_timeouts_rem_before", "period"]
```

Source columns: `wp_start_columns` / `wp_start_touchback_columns` /
`wp_end_columns` — all in the `start.*` / `end.*` namespace. `spread_time` is
computed by `__add_spread_time` via `pos_team_spread * exp(-4 * elapsed_share)`,
identical to the CFB formula. `ExpScoreDiff_Time_Ratio = start.ExpScoreDiff /
(start.adj_TimeSecsRem + 1)`.

**QBR features (6):** `qbr_vars`

```python
["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"]
```

Computed by `__process_qbr` as weighted per-QB-game means (same logic as CFB's
`__process_qbr`; `qbr_epa = min(EPA, -5)` / `-3.5 on fumbles`; `sack_epa` on
non-fumble sacks, `pass_epa`/`rush_epa`/`pen_epa` gated by play type).

**WP also uses two additional touchback columns not in wp_final_names:**

- `wp_start_touchback_columns`: same 13 features but `start.yardsToEndzone.touchback`
  in place of `start.yardsToEndzone` and `start.ExpScoreDiff_Time_Ratio_touchback`
  in place of `start.ExpScoreDiff_Time_Ratio`. Both are renamed to `wp_final_names`
  before prediction — so the model sees the same 13 feature names regardless.

### 2.4 Data source: nflverse-pbp

sdv-py already bundles `load_nfl_pbp(seasons)` (nflreadpy parity; see
`nfl_loaders.py`). This reads from:

```
https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet
```

Coverage: 1999–present. The nflverse PBP is the pre-enriched nflfastR output
(EPA/WPA/QBR already applied). It is therefore **not** a raw input for training
in the producer==consumer sense — it is the output of `NFLPlayProcess`'s
equivalent (nflfastR), not a raw ESPN feed. This is a meaningful difference from
Track 1's `final.json` approach and is discussed in §6.

---

## 3. Decisions (locked or pending)

| # | Decision | Choice | Status |
|---|---|---|---|
| **1** | **Implementation repo** | **NOT `cfbfastR-cfb-raw`** — that repo is CFB-only. The NFL model-training pipeline must live in a separate NFL-focused repo: a future `nfl-raw` scraper/pipeline repo, or an `nflverse-py-data` repo, or a standalone `nfl-model-training` module. The exact home is **pending** (see §4). | Pending — blocking decision |
| **2** | Training-data source (option A) | Use the enriched **nflverse-pbp** parquet (already in sdv-py loaders). Contains all the engineered features needed (EP/WP columns, spread, down/distance, score diff). The models' training labels (next score in half) must still be derived from the raw play sequence — the enriched PBP does not include intermediate NSH. | Option A identified; feasibility TBD |
| **3** | Training-data source (option B) | Run `NFLPlayProcess` on raw ESPN NFL PBP (scraped game-by-game) to produce a play-level frame — the NFL analogue of the CFB `final.json`. No such raw scraper exists in sdv-py today (NFL data is loaded from nflverse releases, not scraped from ESPN). | Option B requires new scraper; higher effort |
| **4** | Recipe sourcing | The nflfastR EP/WP training scripts must be located. The known prior art is the nflfastR R package (`https://github.com/nflverse/nflfastR`) and its companion data repo. Training scripts are likely under `inst/` or a `data-raw/` directory. **This must be confirmed before implementation** (see §5). | Pending — Phase 0 task |
| **5** | Cross-repo handoff | Retrained `.ubj` models are **manually copied** into `sdv-py/sportsdataverse/nfl/models/` under review — same protocol as CFB Track 1 (never auto-overwritten). | Locked (follows Track 1 precedent) |
| **6** | NFL WP-naive | The CFB track ships a `wp_naive.ubj` (WP without spread). No equivalent exists in the NFL bundle. Whether to train and ship an NFL WP-naive is **deferred** to the recipe-sourcing phase. | Deferred |
| **7** | QBR target for NFL | The QBR training target for NFL is ESPN's raw NFL QBR (the same endpoint family as CFB: `sports.core.api.espn.com/.../football/leagues/nfl/.../qbr`). ESPN NFL QBR coverage and scraping logistics are not yet confirmed. | Pending |

---

## 4. Cross-repo home — key architectural decision

`cfbfastR-cfb-raw` is scoped to college football. The NFL training pipeline
**must not land here**. Three candidate homes exist:

### Option A — New `nfl-raw` repo (recommended)
Create a dedicated `nfl-raw` repository mirroring the `cfbfastR-cfb-raw`
architecture: a raw scraper + reprocessor + model-training pipeline for NFL.
- Pro: clean separation; mirrors the CFB/NFL split already present in nflverse.
- Pro: can house a future raw ESPN NFL scraper (analogous to `cfbfastR-cfb-raw`'s
  `scrape_cfb_json.py`).
- Con: new repo to create and maintain.

### Option B — `nflverse-py-data` or community repo
Contribute the training scripts to an existing nflverse-Python community repo
(if one exists or is planned).
- Pro: aligns with the nflverse ecosystem.
- Con: dependency on external maintainers; unknown.

### Option C — `sdv-py` internal `nfl/model_training/` sub-package
Add a `model_training/` sub-package inside `sportsdataverse/nfl/` — analogous
to `cfbfastR-cfb-raw/python/model_training/` but inside the library itself.
- Pro: no new repo.
- Con: couples training tooling to the inference library; inflates the sdv-py dep
  surface (xgboost, plotnine, statsmodels as optional extras); against the
  intent of keeping sdv-py lean.

**Recommendation: Option A (new `nfl-raw` repo).** This spec documents the
design; the plan's Phase 0 includes making this decision concrete and creating
the target repo scaffold. Until that decision is made, all implementation tasks
are on hold.

---

## 5. Recipe sourcing — what is known vs. unknown

### Known

- The bundled NFL models share the exact architecture as the CFB models (Track 1
  finding: same feature counts, same objectives, same tree counts).
- The nflfastR R package (`https://github.com/nflverse/nflfastR`) is the
  authoritative source for the EP/WP training pipeline — it is the NFL analogue
  of cfbscrapR.
- nflfastR's EP/WP model training scripts are expected to be in the package's
  `data-raw/` directory (R convention for scripts that generate bundled data) or
  a companion data repo (`nflverse/nflverse-models` or similar).
- The CFB models were found to be **prediction-identical** to the corresponding
  keepers Dec-2020 `.model` files. If the NFL models share this provenance, the
  nflfastR training repo at or before Dec 2020 should contain the originating
  scripts.
- The nflfastR paper (López, Thompson, Rowlingson, 2020 — arxiv 2009.12394
  "nflfastR: Open NFL Play-by-Play Data with Estimated Probabilities")
  describes the EP/WP methodology. The `multi:softprob` 7-class EP and
  `binary:logistic` WP architecture is documented in that paper.

### Unknown (Phase 0 tasks)

1. **Exact training scripts and their location** in the nflfastR repo / companion
   repos. Must be found and read — do not assume from architecture alone.
2. **nflfastR hyperparameters.** Are the nrounds/eta/gamma/subsample/etc. for the
   NFL models the same as the CFB models (which share architecture with the same
   tree counts)? This would indicate either a shared training run or a deliberate
   alignment. The tree counts matching exactly (525, 760, 45) is a strong hint but
   not confirmation.
3. **Training data source used by nflfastR.** nflfastR trains on nflscrapR/nflfastR-
   processed PBP. Whether this was the enriched PBP parquet or a raw ESPN feed
   determines which of options A/B in §3 Decision #2 is feasible.
4. **Season coverage.** What season range was the bundled model trained on?
5. **NFL WP-naive recipe.** Did nflfastR train a naive (no-spread) WP model?
6. **ESPN NFL QBR scraping.** Is the `sports.core.api.espn.com` QBR endpoint
   accessible for NFL in the same format as CFB? What is the season coverage?

---

## 6. Data source discussion — NFL vs. CFB producer==consumer contract

Track 1's central design insight was that `final.json` (the `CFBPlayProcess`
output with odds resolved) provides training features that are **identical** to
what the model sees at inference — producer==consumer parity by construction.

The NFL situation is more complex:

**Option A: nflverse-pbp enriched parquet (pre-enriched by nflfastR).** This is
the output of the nflfastR inference pipeline, not its input. Using it as
training data would mean training on nflfastR's EP/WPA values — valid for
features derived from them (like `ExpScoreDiff` = `pos_score_diff_start`'s
expected value analog), but the EP feature columns themselves are derived from
raw game state (`TimeSecsRem`, `yardsToEndzone`, `distance`, `down`, `score
diff`), all of which are present in the nflverse PBP. The WP `spread_time` is
derivable from the spread and `adj_TimeSecsRem` (both present in nflverse PBP).
The **labels** (next score in half) are **not** in the nflverse PBP and must be
rederived.

**Option B: run `NFLPlayProcess` on raw ESPN NFL PBP.** This is the exact
producer==consumer path — the same code that computes the features at inference
produces the training frame. But it requires a raw ESPN NFL scraper (no such
thing exists in this codebase today). This is the higher-fidelity approach but
requires significantly more infrastructure.

**Practical recommendation for Phase 0:** assess whether the nflverse PBP
contains all the raw game-state columns needed to reconstruct the 8 EP features
and 13 WP features. If yes, Option A is the training-data path (no new scraper
needed). If key columns are absent or already-processed values that mask the
raw state, a targeted raw scraper is needed.

---

## 7. Inference contract (from `nfl_pbp.py`) — full detail

`NFLPlayProcess` applies the three models in the same structural pattern as
`CFBPlayProcess`. The key inference methods:

**EP application:** `play_df[ep_start_columns]`, renamed to `ep_final_names`,
passed to `ep_model.predict(DMatrix(...))`. `__calculate_ep_exp_val` applies
`ep_class_to_score_mapping = {0:7, 1:-7, 2:3, 3:-3, 4:2, 5:-2, 6:0}`. EPA =
EP_end − EP_start. Touchback variant uses `yardsToEndzone.touchback`.

**WP application:** `play_df[wp_start_columns]`, renamed to `wp_final_names`,
passed to `wp_model.predict(DMatrix(...))`. WPA = WP_end − WP_start.
`spread_time = pos_team_spread * exp(-4 * elapsed_share)` where
`elapsed_share = (3600 − adj_TimeSecsRem) / 3600`. `ExpScoreDiff_Time_Ratio =
ExpScoreDiff / (adj_TimeSecsRem + 1)`. Touchback variant substitutes
`yardsToEndzone.touchback` and `ExpScoreDiff_Time_Ratio_touchback`.

**QBR application:** per-QB-game weighted means of `qbr_vars` computed by
`__process_qbr`, then `qbr_model.predict(DMatrix(pass_qbr[qbr_vars]))`.

**No WP-naive model** is currently bundled or applied in `NFLPlayProcess`.

---

## 8. Dependencies + tooling

The NFL model-training pipeline will need:

- `xgboost>=2.0` (same as Track 1)
- `polars>=1.0` / `pandas` / `pyarrow` (training frame I/O)
- `plotnine` + `statsmodels` + `pillow` (calibration figures; same as Track 1)
- `requests` / network access (nflverse-pbp download or ESPN scraping)
- `uv` (packaging, consistent with sdv-py + cfbfastR-cfb-raw)

No `pygam` dependency — QBR is XGBoost, not GAM, for the NFL model.

---

## 9. Parallel to CFB Track 1 — what transfers

The NFL training pipeline is structurally a near-copy of the CFB Track 1
pipeline. Many of the same modules are needed:

| CFB Track 1 module | NFL Track 6 analogue | Delta |
|---|---|---|
| `next_score.py` | `nfl_next_score.py` | NFL play types differ; scoring values same |
| `ingest.py` | `nfl_ingest.py` | NFL data source (nflverse-pbp vs final.json) |
| `features.py` | `nfl_features.py` | Same column crosswalk logic; NFL source cols |
| `train_ep.py` | `nfl_train_ep.py` | Same params (if recipe confirms); same contract |
| `train_wp.py` | `nfl_train_wp.py` | Same params (if recipe confirms); same contract |
| `train_qbr.py` | `nfl_train_qbr.py` | ESPN NFL QBR target (same endpoint family) |
| `validate.py` | `nfl_validate.py` | Same parity harness; different shipped refs |
| `figures.py` | `nfl_figures.py` | Same plotnine styling; nflverse/nflfastR hex |
| `cli.py` | `nfl_cli.py` | Same subcommand structure |
| `constants.py` | `nfl_constants.py` | NFL feature lists from `nfl/model_vars.py` |

The NFL `constants.py` can be generated almost verbatim from `nfl/model_vars.py`
(the same `ep_final_names`, `wp_final_names`, `qbr_vars` symbols exist there).

---

## 10. Risks and open items

| Risk | Severity | Mitigation |
|---|---|---|
| **Cross-repo home undecided.** Implementation is blocked until the target repo is agreed on. | High | Phase 0 task 0.0: decide and create the repo. |
| **nflfastR training scripts not found.** If the exact recipe (hyperparams, feature set, label derivation) cannot be sourced, the pipeline is a reconstruction rather than a port. | High | Phase 0 task 0.2: read `nflfastR/data-raw/` exhaustively; escalate to the nflverse community if absent. |
| **Hyperparameter alignment.** The NFL and CFB models share architecture and tree counts exactly. If this is not coincidental, the NFL models may use the same hyperparameters as the CFB models (discovered in Track 1). Confirming this would significantly accelerate the NFL port. | Medium | Phase 0 task 0.2 (read the nflfastR scripts). |
| **nflverse-pbp column availability.** The pre-enriched PBP may not include all raw game-state columns needed to reconstruct features without re-running NFLPlayProcess. | Medium | Phase 0 task 0.3: column audit of nflverse-pbp parquet. |
| **NFL WP-naive** (no-spread WP) does not exist in the current NFL bundle. Whether to add one depends on the recipe survey. | Low | Deferred to Phase 0. |
| **ESPN NFL QBR endpoint.** Same family as CFB (`sports.core.api.espn.com`); availability and scrape logistics unconfirmed. | Low–Medium | Phase 0 task 0.4. |
| **Model handoff.** Overwriting sdv-py's bundled NFL `.ubj` is consequential (affects all `NFLPlayProcess` EPA/WPA/QBR outputs). Always manual + reviewed. | High | Locked protocol (§3 Decision #5). |
| **Season coverage.** nflverse PBP goes back to 1999. Does training on 1999–2025 improve model quality? Or does pre-2002 data hurt signal? | Low | Empirical; evaluate during training. |

---

## 11. What this spec does NOT cover

- Fourth-down model for NFL (out of scope for this track).
- Any NFL model beyond EP / WP / QBR (not currently bundled in sdv-py).
- Changes to `NFLPlayProcess` inference logic beyond what is needed to bundle
  a retrained model. **(SUPERSEDED by §12 — see below: the bundled NFL models are CFB copies, so
  retraining real NFL models REQUIRES changing the inference feature contract.)**
- Automated deployment or CI-triggered model updates (decision #5: always manual).

---

## 12. UPDATE — canonical recipe found (`fastrmodels/data-raw/models.R`)

The Phase-0 "find the nflfastR recipe" survey is **resolved**. Ben Baldwin's
`nflverse/fastrmodels/data-raw/models.R` (local: `…/nflverse-dev/fastrmodels/data-raw/models.R`)
estimates the canonical nflfastR EP/WP/CP/FG models. Two findings reshape this track:

### 12.1 The bundled sdv-py NFL models are **CFB-model copies**, not real NFL models

Introspecting `sportsdataverse/nfl/models/*.ubj`:

| sdv-py NFL bundle | feats / trees | = nflfastR canonical? | = CFB shipped? |
|---|---|---|---|
| `ep_model.ubj` | **8 / 3675** (525×7) | ❌ (nflfastR EP = **18**-feat) | ✅ identical to CFB EP |
| `wp_spread.ubj` | **13 / 760** | ❌ (nflfastR WP = **12**-feat / 534) | ✅ identical to CFB WP |
| `qbr_model.ubj` | **6 / 45** | ❌ (no nflfastR QBR model) | ✅ identical to CFB QBR |

So `NFLPlayProcess` currently computes NFL EPA/WPA/QBR with **CFB-trained weights on the CFB
feature contract** — a latent fidelity gap: sdv-py's NFL metrics will diverge from nflverse's
official values (which come from the real 18-feat EP / 12-feat WP models). **Track 6 is therefore
a correctness fix, not a drop-in retrain:** it must (a) train *real* NFL models from the nflfastR
recipe AND (b) update `NFLPlayProcess`'s feature construction from the CFB 8/13-feat contract to
the NFL 18/12-feat contract. The handoff (decision #5, manual) now includes inference-code changes,
not just `.ubj` swaps — scope this carefully.

### 12.2 The canonical nflfastR recipe (exact, from `models.R`)

**EP** — `multi:softprob`, `num_class=7`, `eta=0.025, gamma=1, subsample=0.8, colsample_bytree=0.8,
max_depth=5, min_child_weight=1`, `nrounds=525`, `set.seed(2013)`, `weight=Total_W_Scaled`.
**18 features:** `half_seconds_remaining, yardline_100, home, retractable, dome, outdoors, ydstogo,
era0, era1, era2, era3, era4, down1, down2, down3, down4, posteam_timeouts_remaining,
defteam_timeouts_remaining`. Label class order = CFB's (`TD=0 … No_Score=6`). Data:
`nflfastR-data/models/cal_data.rds`.

**WP-spread** — `binary:logistic`, `eta=0.05, gamma=.79012017, subsample=0.9224245,
colsample_bytree=5/12, max_depth=5, min_child_weight=7`, `nrounds=534`, **`monotone_constraints =
"(0,0,0,0,0,1,1,-1,-1,-1,1,-1)"`**. **12 features:** `receive_2h_ko, spread_time, home,
half_seconds_remaining, game_seconds_remaining, Diff_Time_Ratio, score_differential, down, ydstogo,
yardline_100, posteam_timeouts_remaining, defteam_timeouts_remaining`. `label = (posteam == Winner)`,
filter `qtr <= 4`. Data: `guga31bb/metrics/wp_tuning/cal_data.rds`.

**WP-naive** — same as spread minus `spread_time` (11 features), `eta=0.2, gamma=0, subsample=0.8,
colsample_bytree=0.8, max_depth=4, min_child_weight=1`, `nrounds=65`.

**Bonus models in the same script** (adjacent, optional additions to this track):
- **CP** (completion probability) — `binary:logistic, eta=0.025, gamma=5, subsample=0.8,
  colsample_bytree=0.8, max_depth=4, min_child_weight=6, base_score=mean(complete_pass)`,
  `nrounds=560`; features from `prepare_cp_data()`. **This is Track 5's true lineage** (CFB
  `cpoe_model.R` uses these exact params; StatsBomb was an experiment overlay).
- **FG** (field goal) — `mgcv::bam(sp ~ s(yardline_100, by = interaction(era, model_roof)) +
  model_roof + era, family = "binomial")` (a GAM, not XGBoost).

### 12.3 Cross-checks (confirm the CFB forensics)

- **CFB keepers `03` = a copy of this nflfastR WP-spread recipe** (identical `eta=0.05/.79012017/
  5÷12/534/min_child_weight=7`). That's why keepers `03` was the divergent CFB dead-end — it was the
  NFL recipe, never the shipped CFB WP (which is cfbscrapR-wpa's 760-tree model).
- The feature *derivations* (`make_model_mutations`, `prepare_wp_data`, `prepare_cp_data`:
  era/roof dummies, `spread_time`, `Diff_Time_Ratio`) live in **nflfastR's** `R/helper_add_ep_wp.R`
  / `helper_add_cp_cpoe.R` (NOT in fastrmodels) — the one remaining Phase-0 read.

### 12.4 Revised Phase-0 (most of it is now done)

1. ~~Find the recipe~~ → **done** (above).
2. Read nflfastR `R/helper_add_ep_wp.R` + `helper_add_cp_cpoe.R` for the exact feature
   derivations (era buckets, roof one-hot, `spread_time`, `Diff_Time_Ratio`, `prepare_cp_data`).
3. Decide the cross-repo home (`nfl-raw`) + data sourcing (`nflfastR-data` / `guga31bb/metrics`
   cal_data, or regenerate from nflverse pbp).
4. Scope the `NFLPlayProcess` feature-contract change (8/13-feat CFB-shape → 18/12-feat NFL-shape) —
   this is the largest piece and was previously unrecognized.
