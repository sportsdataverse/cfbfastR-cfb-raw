# CFB Modeling Suite — Track 1 (Play-Level Shipped Models) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the cfbfastR R EP/WP/QBR model-training pipeline to Python in `cfbfastR-cfb-raw/python/model_training/`, retraining drop-in replacements for sdv-py's `cfb/models/{ep_model,wp_spread,wp_naive,qbr_model}.ubj` from the CFB backfill.

**Architecture:** Producer == consumer. Training features come from the **same** `CFBPlayProcess` code that computes them at inference (read from the backfill's `cfb/json/final/{game_id}.json` plays), so train/inference parity holds by construction. The only net-new logic is the **outcome label** (next score in half), ported from akeaswaran's vectorized `model_training.R` approach (fill the next scoring drive/team/type within each game-half). Two stages: **Stage 1** faithful replica (validated vs the May-2021 `xgb_*` reference models) then **Stage 2** parity upgrade (the confirmed shipped recipes — EP=keepers `02`, WP=`cfbscrapR-wpa.ipynb`, QBR=6-feat reconstruction).

**Tech Stack:** Python 3.11+, uv, polars 1.x, xgboost ≥2.0 (introspection via 3.x), pandas/pyarrow, plotnine + statsmodels + pillow (figures), pygam (QBR GAM ancestor), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-ep-wp-model-training-port-design.md` (umbrella: `…-cfb-modeling-suite-program.md`).

---

## File structure

Package `python/model_training/` (one responsibility per file):

| File | Responsibility |
|---|---|
| `python/model_training/__init__.py` | Package marker + version. |
| `python/model_training/constants.py` | Feature-name lists (`EP_FEATURES`, `WP_SPREAD_FEATURES`, `WP_NAIVE_FEATURES`, `QBR_FEATURES`), the `CFBPlayProcess`→model column crosswalk, EP class order + `EP_CLASS_TO_SCORE`, XGBoost param dicts per model/stage. |
| `python/model_training/next_score.py` | `label_next_score_half(plays_df) -> plays_df+["next_score_half","label","score_drive"]` — vectorized next-score-in-half (port of `model_training.R`). |
| `python/model_training/ingest.py` | `build_training_frame(final_dir, seasons, stage) -> pl.DataFrame` — read `final.json`, clean, label (via `next_score`), compute weights, write `pbp_full.parquet`. |
| `python/model_training/features.py` | `ep_matrix/ wp_matrix/ qbr_matrix(df, stage) -> (X, y, w)` — select/rename to the exact model contracts; QBR per-QB aggregation. |
| `python/model_training/train_ep.py` | `train_ep(df, stage) -> Booster`; LOSO CV helper. |
| `python/model_training/train_wp.py` | `train_wp(df, variant, stage) -> Booster` (variant ∈ {spread,naive}). |
| `python/model_training/train_qbr.py` | `train_qbr(df) -> Booster` (6-feat reg:squarederror). |
| `python/model_training/validate.py` | prediction-parity vs reference `.ubj`; LOSO calibration table; QBR fit metrics. |
| `python/model_training/figures.py` | plotnine calibration plots (bespoke styling) + data tables. |
| `python/model_training/cli.py` | `ingest \| train-ep \| train-wp \| train-qbr \| validate \| figures` subcommands. |
| `tests/model_training/` | one `test_*.py` per module. |
| `tests/fixtures/model_training/` | reference `.ubj` (converted), sanity test-items JSON, one `final.json` slice. |

Tests run with `uv run pytest tests/model_training/`. The repo already pins `sportsdataverse>=0.0.52` (gives `CFBPlayProcess`, `load_cfb_pbp`, `model_vars`).

---

## Phase 0 — Scaffold, deps, reference fixtures

### Task 0.1: Create the package skeleton

**Files:**
- Create: `python/model_training/__init__.py`
- Test: `tests/model_training/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_package.py
def test_package_imports():
    import model_training
    assert hasattr(model_training, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd cfbfastR-cfb-raw && uv run pytest tests/model_training/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_training'`

- [ ] **Step 3: Create the package**

```python
# python/model_training/__init__.py
"""CFB model-training port (Track 1: EP/WP/QBR play-level models)."""
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 4: Make `python/` importable in tests**

Append to `tests/conftest.py` (create if absent):

```python
# tests/conftest.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_package.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/model_training/__init__.py tests/model_training/test_package.py tests/conftest.py
git commit -m "feat(model-training): scaffold model_training package"
```

### Task 0.2: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the training + figures deps**

Edit `pyproject.toml` `[project] dependencies` to add `"xgboost>=2.0"`. Add a new dependency group:

```toml
[dependency-groups]
dev = ["pytest>=8.0"]
figures = ["plotnine>=0.13", "statsmodels>=0.14", "pillow>=10.0"]
gam = ["pygam>=0.9"]
```

- [ ] **Step 2: Sync**

Run: `uv sync --all-groups`
Expected: resolves and installs (xgboost, plotnine, statsmodels, pillow, pygam) with no error.

- [ ] **Step 3: Verify xgboost importable**

Run: `uv run python -c "import xgboost, plotnine, statsmodels, PIL, pygam; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(model-training): add xgboost + figures/gam dep groups"
```

### Task 0.3: Vendor the Stage-1 reference models + sanity fixtures

The May-2021 `xgb_*.model` (binary format, unreadable in xgboost ≥3.1) live in `gp-cfb-raw-keepers`. Convert once to UBJ for use as Stage-1 parity references. Also copy the cfbscrapR-lineage sanity fixtures from `cfb-pbp-analysis`.

**Files:**
- Create: `tests/fixtures/model_training/{xgb_ep_model,xgb_wp_spread_model,xgb_wp_naive_model}.ubj`
- Create: `tests/fixtures/model_training/{epa,wpa}-model-test-items.json`
- Create: `tests/fixtures/model_training/README.md`

- [ ] **Step 1: Convert the reference models (one-time, xgboost 3.0)**

Run from `cfbfastR-cfb-raw`:

```bash
mkdir -p tests/fixtures/model_training
KEEP=../gp-cfb-raw-keepers/from-cfbfastR-raw/models
uv run --no-project --with 'xgboost==3.0.5' python - <<'PY'
import xgboost as xgb, os
KEEP=os.environ.get("KEEP","../gp-cfb-raw-keepers/from-cfbfastR-raw/models")
OUT="tests/fixtures/model_training"
for f in ["xgb_ep_model","xgb_wp_spread_model","xgb_wp_naive_model"]:
    b=xgb.Booster(); b.load_model(f"{KEEP}/{f}.model")
    b.save_model(f"{OUT}/{f}.ubj")
    print("converted", f, "feats", b.num_features())
PY
```

Expected: `converted xgb_ep_model feats 8`, `converted xgb_wp_spread_model feats 10`, `converted xgb_wp_naive_model feats 9`.

- [ ] **Step 2: Copy the sanity fixtures**

```bash
cp ../cfb-pbp-analysis/epa-model-test-items.json tests/fixtures/model_training/
cp ../cfb-pbp-analysis/wpa-model-test-items.json tests/fixtures/model_training/
```

- [ ] **Step 3: Document provenance**

```markdown
<!-- tests/fixtures/model_training/README.md -->
# model_training fixtures

- `xgb_{ep_model,wp_spread_model,wp_naive_model}.ubj` — May-2021 R-trained reference models
  (gp-cfb-raw-keepers), converted from binary `.model` via xgboost 3.0 (binary format is
  unreadable in xgboost >=3.1). EP=8-feat/7-class; WP spread=10-feat; WP naive=9-feat.
  **Stage-1 parity references only** (divergent lineage; NOT the shipped models).
- `{epa,wpa}-model-test-items.json` — cfbscrapR-lineage reference plays from akeaswaran/cfb-pbp-analysis.
  `wpa-*` is in the shipped 13-feat WP contract (near-parity WP oracle); `epa-*` is 16-feat-lineage
  (ballpark EPA only). Sanity checks, not exact shipped-parity oracles.
```

- [ ] **Step 4: Verify the references load in the project's xgboost**

Run: `uv run python -c "import xgboost as xgb; [print(f, xgb.Booster().load_model(f) or 'ok') for f in ['tests/fixtures/model_training/xgb_ep_model.ubj']]"`
Expected: prints `... ok` (loads without error in xgboost ≥3.1).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/model_training/
git commit -m "test(model-training): vendor Stage-1 reference models + sanity fixtures"
```

---

## Phase 1 — `next_score.py` (vectorized next-score-in-half labeling)

Port of `model_training.R` lines 22-60. Operates on the `final.json` plays frame using `CFBPlayProcess` column names. Within each `(game_id, half)`: mark scoring plays, carry the scoring drive/team/type **backward** (so each play sees the *next* score in its half), then classify into the 7 EP classes from the posteam's perspective.

### Task 1.1: Column crosswalk + EP class constants

**Files:**
- Create: `python/model_training/constants.py`
- Test: `tests/model_training/test_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_constants.py
from model_training import constants as C

def test_ep_class_to_score_matches_sdvpy():
    from sportsdataverse.cfb.model_vars import ep_class_to_score_mapping
    assert C.EP_CLASS_TO_SCORE == ep_class_to_score_mapping

def test_feature_lists_match_sdvpy_contract():
    from sportsdataverse.cfb import model_vars as mv
    assert C.EP_FEATURES == mv.ep_final_names
    assert C.WP_SPREAD_FEATURES == mv.wp_final_names
    # naive == spread minus spread_time
    assert C.WP_NAIVE_FEATURES == [c for c in mv.wp_final_names if c != "spread_time"]
    assert C.QBR_FEATURES == mv.qbr_vars
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_constants.py -v`
Expected: FAIL (`No module named 'model_training.constants'`)

- [ ] **Step 3: Implement constants**

```python
# python/model_training/constants.py
"""Feature contracts, column crosswalk, EP class mapping, and XGBoost params.

The EP/WP/QBR feature lists are imported from sdv-py's model_vars at runtime so this
module can NEVER drift from the shipped inference contract (a test asserts equality).
"""
from __future__ import annotations

from sportsdataverse.cfb import model_vars as _mv

# --- shipped inference contracts (re-exported for clarity + a drift test) ---
EP_FEATURES: list[str] = list(_mv.ep_final_names)            # 8
WP_SPREAD_FEATURES: list[str] = list(_mv.wp_final_names)     # 13
WP_NAIVE_FEATURES: list[str] = [c for c in _mv.wp_final_names if c != "spread_time"]  # 12
QBR_FEATURES: list[str] = list(_mv.qbr_vars)                 # 6
EP_CLASS_TO_SCORE: dict[int, int] = dict(_mv.ep_class_to_score_mapping)
# class order: 0 TD, 1 Opp_TD, 2 FG, 3 Opp_FG, 4 Safety, 5 Opp_Safety, 6 No_Score
NEXT_SCORE_TO_LABEL: dict[str, int] = {
    "Touchdown": 0, "Opp_Touchdown": 1, "Field_Goal": 2, "Opp_Field_Goal": 3,
    "Safety": 4, "Opp_Safety": 5, "No_Score": 6,
}

# --- CFBPlayProcess (final.json plays) -> the columns the EP/WP feature builders need.
# The start.* features already exist on the plays; this maps the SHIPPED feature name
# -> the source column in the final.json play record.
EP_SOURCE = {
    "TimeSecsRem": "start.TimeSecsRem", "yards_to_goal": "start.yardsToEndzone",
    "distance": "start.distance", "down_1": "down_1", "down_2": "down_2",
    "down_3": "down_3", "down_4": "down_4", "pos_score_diff_start": "pos_score_diff_start",
}
WP_SOURCE = {
    "pos_team_receives_2H_kickoff": "start.pos_team_receives_2H_kickoff",
    "spread_time": "start.spread_time", "TimeSecsRem": "start.TimeSecsRem",
    "adj_TimeSecsRem": "start.adj_TimeSecsRem",
    "ExpScoreDiff_Time_Ratio": "start.ExpScoreDiff_Time_Ratio",
    "pos_score_diff_start": "pos_score_diff_start", "down": "start.down",
    "distance": "start.distance", "yards_to_goal": "start.yardsToEndzone",
    "is_home": "start.is_home", "pos_team_timeouts_rem_before": "start.posTeamTimeouts",
    "def_pos_team_timeouts_rem_before": "start.defPosTeamTimeouts", "period": "period",
}

# --- labeling source columns (final.json plays) ---
LBL = {
    "game_id": "game_id", "drive_id": "drive.id", "period": "period",
    "pos_team": "pos_team", "def_pos_team": "def_pos_team",
    "scoring_play": "scoring_play", "offense_score_play": "offense_score_play",
    "defense_score_play": "defense_score_play", "play_type": "type.text",
    "pos_score_diff": "pos_score_diff_start",
}

# --- XGBoost params (exact, per spec §5/§7) ---
EP_PARAMS = dict(booster="gbtree", objective="multi:softprob", eval_metric="mlogloss",
                 num_class=7, eta=0.025, gamma=1, subsample=0.8, colsample_bytree=0.8,
                 max_depth=5, min_child_weight=1)
EP_NROUNDS = 525

WP_SPREAD_PARAMS = dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss",
                        eta=0.02, gamma=0.3445502, subsample=0.7204741,
                        colsample_bytree=0.5714286, max_depth=5, min_child_weight=14)
WP_SPREAD_NROUNDS = 760
WP_NAIVE_PARAMS = dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss",
                       eta=0.2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                       max_depth=4, min_child_weight=1)
WP_NAIVE_NROUNDS = 65

# Stage-1 (divergent keepers `03`) WP-spread params — replica target only.
WP_SPREAD_PARAMS_STAGE1 = dict(booster="gbtree", objective="binary:logistic",
                               eval_metric="logloss", eta=0.05, gamma=0.79012017,
                               subsample=0.9224245, colsample_bytree=5 / 12, max_depth=5,
                               min_child_weight=7)
WP_SPREAD_NROUNDS_STAGE1 = 534

QBR_PARAMS = dict(booster="gbtree", objective="reg:squarederror", eta=0.1,
                  subsample=0.8, colsample_bytree=0.8, max_depth=4, min_child_weight=1)
QBR_NROUNDS = 45  # matches shipped qbr_model.ubj tree count

# Known-bad games excluded by keepers 02/03 + model_training.R (ESPN data defects).
BAD_GAME_IDS: set[int] = {
    400603838, 401020760, 400933849, 400547737, 400547739, 401012806,
    401021693, 400787470, 401112262, 401114227, 401147693, 401015042,
    400986609, 400763439,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_constants.py -v`
Expected: PASS (4 assertions). If `WP_NAIVE_FEATURES`/order mismatches, fix the comprehension order to follow `wp_final_names`.

- [ ] **Step 5: Commit**

```bash
git add python/model_training/constants.py tests/model_training/test_constants.py
git commit -m "feat(model-training): feature contracts + column crosswalk + params"
```

### Task 1.2: `label_next_score_half` — happy path within a half

**Files:**
- Create: `python/model_training/next_score.py`
- Test: `tests/model_training/test_next_score.py`

- [ ] **Step 1: Write the failing test (synthetic single half)**

```python
# tests/model_training/test_next_score.py
import polars as pl
from model_training.next_score import label_next_score_half


def _plays(rows):
    # rows: list of dicts with the LBL source columns
    return pl.DataFrame(rows)


def test_offense_touchdown_then_label_td():
    # 3 plays in H1, drive 2 scores an offensive TD for team A (the posteam of plays 1-2)
    df = _plays([
        {"game_id": 1, "drive.id": 1, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Rush"},
        {"game_id": 1, "drive.id": 1, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Pass Incompletion"},
        {"game_id": 1, "drive.id": 2, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": True, "offense_score_play": True, "defense_score_play": False,
         "type.text": "Passing Touchdown"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["Touchdown", "Touchdown", "Touchdown"]
    assert out["label"].to_list() == [0, 0, 0]
    assert out["score_drive"].to_list() == [2, 2, 2]


def test_no_score_before_half_is_no_score():
    df = _plays([
        {"game_id": 1, "drive.id": 1, "period": 2, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Rush"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["No_Score"]
    assert out["label"].to_list() == [6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_next_score.py -v`
Expected: FAIL (`No module named 'model_training.next_score'`)

- [ ] **Step 3: Implement `label_next_score_half`**

```python
# python/model_training/next_score.py
"""Vectorized next-score-in-half labeling (port of model_training.R lines 22-60).

Within each (game_id, half) the next scoring drive/team/type are carried BACKWARD
(`fill backward`) so every play sees the next score in its half; the 7-class label is
then derived from the posteam's perspective. Half = 1 for periods {1,2}, 2 for {3,4}.
OT (period > 4) is dropped upstream in ingest.
"""
from __future__ import annotations

import polars as pl

from .constants import NEXT_SCORE_TO_LABEL

_TD = "Touchdown"
_FG = "Field Goal Good"
_SAFETY = "Safety"


def label_next_score_half(plays: pl.DataFrame) -> pl.DataFrame:
    df = plays.with_columns(
        half=pl.when(pl.col("period").is_in([1, 2])).then(1).otherwise(2),
        _drive=pl.col("drive.id").cast(pl.Int64),
    )
    # scoring team/type/drive marked ONLY on scoring plays, else null
    df = df.with_columns(
        _score_team=pl.when(pl.col("scoring_play") & pl.col("offense_score_play"))
        .then(pl.col("pos_team"))
        .when(pl.col("scoring_play") & pl.col("defense_score_play"))
        .then(pl.col("def_pos_team"))
        .otherwise(None),
        _score_type=pl.when(pl.col("scoring_play")).then(pl.col("type.text")).otherwise(None),
        _score_drive=pl.when(pl.col("scoring_play")).then(pl.col("_drive")).otherwise(None),
    )
    # carry the NEXT score backward within (game_id, half), preserving play order
    df = df.with_columns(
        next_team=pl.col("_score_team").fill_null(strategy="backward").over(["game_id", "half"]),
        next_type=pl.col("_score_type").fill_null(strategy="backward").over(["game_id", "half"]),
        score_drive=pl.col("_score_drive").fill_null(strategy="backward").over(["game_id", "half"]),
    )
    df = df.with_columns(
        next_score_half=pl.when(pl.col("next_type").is_null())
        .then(pl.lit("No_Score"))
        .when(pl.col("next_type").str.contains(_TD) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Touchdown"))
        .when(pl.col("next_type").str.contains(_TD))
        .then(pl.lit("Opp_Touchdown"))
        .when(pl.col("next_type").str.contains(_FG) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Field_Goal"))
        .when(pl.col("next_type").str.contains(_FG))
        .then(pl.lit("Opp_Field_Goal"))
        .when(pl.col("next_type").str.contains(_SAFETY) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Safety"))
        .when(pl.col("next_type").str.contains(_SAFETY))
        .then(pl.lit("Opp_Safety"))
        .otherwise(pl.lit("No_Score")),
    )
    # No_Score plays: score_drive falls back to own drive (for the recency weight)
    df = df.with_columns(
        score_drive=pl.when(pl.col("next_score_half") == "No_Score")
        .then(pl.col("_drive"))
        .otherwise(pl.col("score_drive")),
        label=pl.col("next_score_half").replace_strict(NEXT_SCORE_TO_LABEL, return_dtype=pl.Int32),
    )
    return df.drop(["_drive", "_score_team", "_score_type", "_score_drive", "next_team", "next_type"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_next_score.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add python/model_training/next_score.py tests/model_training/test_next_score.py
git commit -m "feat(model-training): vectorized next-score-in-half labeling"
```

### Task 1.3: defensive-score sign flip + half boundary isolation

**Files:**
- Modify: `tests/model_training/test_next_score.py`

- [ ] **Step 1: Add tests for the opponent + half-boundary cases**

```python
def test_defensive_td_is_opp_touchdown():
    # posteam A drives, then a defensive TD (B scores) is the next score -> Opp_Touchdown for A
    df = pl.DataFrame([
        {"game_id": 1, "drive.id": 1, "period": 3, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Pass Reception"},
        {"game_id": 1, "drive.id": 1, "period": 3, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": True, "offense_score_play": False, "defense_score_play": True,
         "type.text": "Interception Return Touchdown"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["Opp_Touchdown", "Opp_Touchdown"]
    assert out["label"].to_list() == [1, 1]


def test_score_does_not_leak_across_half_boundary():
    # H1 play has no score in H1; H2 has a FG. H1 play must be No_Score (not the H2 FG).
    df = pl.DataFrame([
        {"game_id": 1, "drive.id": 1, "period": 2, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Rush"},
        {"game_id": 1, "drive.id": 5, "period": 3, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": True, "offense_score_play": True, "defense_score_play": False,
         "type.text": "Field Goal Good"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["No_Score", "Field_Goal"]
    assert out["label"].to_list() == [6, 2]
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/model_training/test_next_score.py -v`
Expected: PASS (4 tests). The half grouping isolates H1/H2; the defensive-TD case is handled by `_score_team = def_pos_team` on `defense_score_play`.

- [ ] **Step 3: Commit**

```bash
git add tests/model_training/test_next_score.py
git commit -m "test(model-training): next-score defensive-TD + half-boundary cases"
```

---

## Phase 2 — `ingest.py` (clean, label, weight)

Port of keepers `01`: drop OT/zero/partial games + special-teams plays, label via `next_score`, compute the nflscrapR weights, write `pbp_full.parquet`.

### Task 2.1: weights from labeled plays

**Files:**
- Create: `python/model_training/ingest.py`
- Test: `tests/model_training/test_ingest_weights.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_ingest_weights.py
import polars as pl
from model_training.ingest import add_weights


def test_weights_formula_matches_nflscrapr():
    # Drive_Score_Dist_W and ScoreDiff_W are min-max inverted; Total_W_Scaled is min-maxed.
    df = pl.DataFrame({
        "game_id": [1, 1, 1],
        "drive.id": [1, 2, 3],
        "score_drive": [3, 3, 3],            # next score is drive 3
        "pos_score_diff_start": [0, -7, 21],
    })
    out = add_weights(df)
    # Drive_Score_Dist = score_drive - drive_id = [2,1,0]; inverted min-max -> [0,0.5,1]
    assert out["Drive_Score_Dist_W"].to_list() == [0.0, 0.5, 1.0]
    # abs_diff = [0,7,21]; ScoreDiff_W = (21-|d|)/(21-0) -> [1, 14/21, 0]
    sw = out["ScoreDiff_W"].to_list()
    assert abs(sw[0] - 1.0) < 1e-9 and abs(sw[2] - 0.0) < 1e-9
    assert out["Total_W_Scaled"].min() == 0.0 and out["Total_W_Scaled"].max() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_ingest_weights.py -v`
Expected: FAIL (`No module named 'model_training.ingest'`)

- [ ] **Step 3: Implement `add_weights`**

```python
# python/model_training/ingest.py  (partial — weights)
"""Read final.json plays, clean, label, weight -> pbp_full.parquet (port of keepers 01)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .next_score import label_next_score_half

# Special-teams / non-scrimmage play types removed before labeling (keepers 01 remove_plays).
REMOVE_PLAYS = {
    "Extra Point Missed", "Extra Point Good", "Timeout", "Kickoff", "Penalty (Kickoff)",
    "Kickoff Return (Offense)", "Kickoff Return Touchdown", "Kickoff Team Fumble Recovery",
    "Kickoff Team Fumble Recovery Touchdown", "Kickoff Touchdown",
}


def add_weights(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        abs_diff=pl.col("pos_score_diff_start").abs(),
        Drive_Score_Dist=(pl.col("score_drive").cast(pl.Int64) - pl.col("drive.id").cast(pl.Int64)),
    )
    dsd_min, dsd_max = df["Drive_Score_Dist"].min(), df["Drive_Score_Dist"].max()
    ad_min, ad_max = df["abs_diff"].min(), df["abs_diff"].max()
    df = df.with_columns(
        Drive_Score_Dist_W=(dsd_max - pl.col("Drive_Score_Dist")) / (dsd_max - dsd_min),
        ScoreDiff_W=(ad_max - pl.col("abs_diff")) / (ad_max - ad_min),
    ).with_columns(
        Total_W=pl.col("Drive_Score_Dist_W") + pl.col("ScoreDiff_W"),
    )
    tw_min, tw_max = df["Total_W"].min(), df["Total_W"].max()
    return df.with_columns(
        Total_W_Scaled=(pl.col("Total_W") - tw_min) / (tw_max - tw_min),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_ingest_weights.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/model_training/ingest.py tests/model_training/test_ingest_weights.py
git commit -m "feat(model-training): nflscrapR weight columns"
```

### Task 2.2: game-level cleaning (OT / zero-period / non-full / special-teams)

**Files:**
- Modify: `python/model_training/ingest.py`
- Test: `tests/model_training/test_ingest_clean.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_ingest_clean.py
import polars as pl
from model_training.ingest import clean_plays


def test_drops_ot_zero_specialteams_and_fixes_kickoff_down():
    df = pl.DataFrame({
        "game_id": [1, 1, 2, 3, 4],
        "period":  [4, 5, 1, 0, 4],     # game2 OT(period5); game3 has period 0
        "start.down": [1, 1, 5, 1, 1],
        "type.text": ["Rush", "Rush", "Kickoff", "Rush", "Timeout"],
    })
    out = clean_plays(df)
    # game 2 (OT) and game 3 (zero period) removed entirely
    assert set(out["game_id"].to_list()).isdisjoint({2, 3})
    # special-teams ("Kickoff") + "Timeout" play types removed
    assert "Kickoff" not in out["type.text"].to_list()
    assert "Timeout" not in out["type.text"].to_list()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_ingest_clean.py -v`
Expected: FAIL (`cannot import name 'clean_plays'`)

- [ ] **Step 3: Implement `clean_plays`**

```python
# python/model_training/ingest.py  (append)
def clean_plays(df: pl.DataFrame) -> pl.DataFrame:
    # kickoff downs (5) -> -1, then drop synthetic high downs (keepers: down < 5)
    df = df.with_columns(
        start_down=pl.when((pl.col("start.down") == 5) & pl.col("type.text").str.contains("Kickoff"))
        .then(-1)
        .otherwise(pl.col("start.down")),
    )
    # OT games (any period > 4) and zero-period games (any period < 1) removed wholesale
    bad = (
        df.group_by("game_id")
        .agg(max_per=pl.col("period").max(), min_per=pl.col("period").min())
        .filter((pl.col("max_per") > 4) | (pl.col("min_per") < 1))
        .get_column("game_id")
    )
    df = df.filter(~pl.col("game_id").is_in(bad))
    # known-bad games (ESPN data defects) excluded by keepers 02/03 + model_training.R
    from .constants import BAD_GAME_IDS
    df = df.filter(~pl.col("game_id").is_in(list(BAD_GAME_IDS)))
    # ESPN partial games: keep only games whose 4th qtr reaches clock_minutes == 0
    if "clock_minutes" in df.columns:
        full = (
            df.filter(pl.col("period") == 4)
            .group_by("game_id")
            .agg(min_clock=pl.col("clock_minutes").min())
            .filter(pl.col("min_clock") == 0)
            .get_column("game_id")
        )
        df = df.filter(pl.col("game_id").is_in(full))
    # remove special-teams / non-scrimmage play types
    return df.filter(~pl.col("type.text").is_in(list(REMOVE_PLAYS)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_ingest_clean.py -v`
Expected: PASS (the synthetic frame has no `clock_minutes`, so the partial-game gate is skipped; OT/zero/special-teams removal verified)

- [ ] **Step 5: Commit**

```bash
git add python/model_training/ingest.py tests/model_training/test_ingest_clean.py
git commit -m "feat(model-training): play/game cleaning (OT, zero-period, partial, special-teams)"
```

### Task 2.3: `build_training_frame` end-to-end over the on-disk final.json

**Files:**
- Modify: `python/model_training/ingest.py`
- Test: `tests/model_training/test_ingest_build.py`

- [ ] **Step 1: Write the failing test (uses the real backfill final.json)**

```python
# tests/model_training/test_ingest_build.py
import pathlib
import polars as pl
import pytest
from model_training.ingest import build_training_frame

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_build_frame_has_labels_and_weights():
    df = build_training_frame(FINAL_DIR, seasons=None)
    assert df.height > 0
    assert df["label"].is_in([0, 1, 2, 3, 4, 5, 6]).all()
    for col in ["Total_W_Scaled", "ScoreDiff_W", "next_score_half"]:
        assert col in df.columns
    # features must all be present (no nulls in the EP feature columns after clean)
    from model_training.constants import EP_SOURCE
    for src in EP_SOURCE.values():
        assert src in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_ingest_build.py -v`
Expected: FAIL (`cannot import name 'build_training_frame'`)

- [ ] **Step 3: Implement `build_training_frame` + writer**

```python
# python/model_training/ingest.py  (append)
def _read_final_plays(final_dir: Path, seasons) -> pl.DataFrame:
    frames = []
    for f in sorted(Path(final_dir).glob("*.json")):
        obj = json.loads(f.read_text())
        if seasons is not None and obj.get("season") not in seasons:
            continue
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def build_training_frame(final_dir, seasons=None) -> pl.DataFrame:
    df = _read_final_plays(final_dir, seasons)
    if df.is_empty():
        return df
    df = clean_plays(df)
    df = label_next_score_half(df)
    df = add_weights(df)
    return df


def write_training_frame(final_dir, out_path, seasons=None) -> int:
    df = build_training_frame(final_dir, seasons)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df.height
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_ingest_build.py -v`
Expected: PASS (or SKIP if no `final.json` on disk — then run a backfill first: `uv run python python/scrape_cfb_json.py -s 2024 -e 2024` then `python/reprocess_cfb_json.py -s 2024 -e 2024`).

- [ ] **Step 5: Commit**

```bash
git add python/model_training/ingest.py tests/model_training/test_ingest_build.py
git commit -m "feat(model-training): build_training_frame end-to-end (final.json -> pbp_full)"
```

---

## Phase 3 — `features.py` (model input matrices)

### Task 3.1: EP + WP feature matrices

**Files:**
- Create: `python/model_training/features.py`
- Test: `tests/model_training/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_features.py
import numpy as np
import polars as pl
from model_training import constants as C
from model_training.features import ep_matrix, wp_matrix


def _frame():
    base = {src: 1.0 for src in set(C.EP_SOURCE.values()) | set(C.WP_SOURCE.values())}
    base.update({"label": 0, "Total_W_Scaled": 0.5, "ScoreDiff_W": 0.5,
                 "season": 2024, "pos_team": "A", "winner": "A", "next_score_half": "Touchdown"})
    return pl.DataFrame([base, {**base, "label": 6, "winner": "B", "next_score_half": "No_Score"}])


def test_ep_matrix_shape_and_order():
    X, y, w = ep_matrix(_frame())
    assert X.shape[1] == 8 and list(X.columns) == C.EP_FEATURES
    assert y.tolist() == [0, 6]


def test_wp_spread_matrix_13_feats_and_binary_label():
    X, y, w = wp_matrix(_frame(), variant="spread")
    assert X.shape[1] == 13 and list(X.columns) == C.WP_SPREAD_FEATURES
    assert set(np.unique(y)).issubset({0, 1})  # label = (pos_team == winner)


def test_wp_naive_drops_spread_time():
    X, _, _ = wp_matrix(_frame(), variant="naive")
    assert "spread_time" not in X.columns and X.shape[1] == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_features.py -v`
Expected: FAIL (`No module named 'model_training.features'`)

- [ ] **Step 3: Implement EP/WP matrices**

```python
# python/model_training/features.py
"""Select/rename final.json plays into the exact shipped model input matrices.

Returns pandas DataFrames (xgboost.DMatrix-friendly) with columns in the EXACT shipped
order, plus the label and weight arrays. WP label is win_indicator = (pos_team==winner);
no sample weights for WP (per the cfbscrapR-wpa recipe). EP uses ScoreDiff_W weights.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from . import constants as C


def _select(df: pl.DataFrame, source: dict[str, str]) -> "pd.DataFrame":
    out = df.select([pl.col(src).alias(name) for name, src in source.items()])
    return out.to_pandas()


def ep_matrix(df: pl.DataFrame):
    X = _select(df, C.EP_SOURCE)[C.EP_FEATURES]
    y = df["label"].to_numpy()
    w = df["ScoreDiff_W"].to_numpy()
    return X, y, w


def wp_matrix(df: pl.DataFrame, variant: str = "spread"):
    feats = C.WP_SPREAD_FEATURES if variant == "spread" else C.WP_NAIVE_FEATURES
    source = {k: v for k, v in C.WP_SOURCE.items() if k in feats}
    X = _select(df, source)[feats]
    y = (df["pos_team"] == df["winner"]).cast(pl.Int32).to_numpy()
    return X, y, None  # WP trains with no sample weights
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add python/model_training/features.py tests/model_training/test_features.py
git commit -m "feat(model-training): EP/WP feature matrices (exact shipped contracts)"
```

### Task 3.2: `winner` join + QBR per-QB aggregation

**Files:**
- Modify: `python/model_training/features.py`, `python/model_training/ingest.py`
- Test: `tests/model_training/test_features_qbr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_features_qbr.py
import polars as pl
from model_training.features import qbr_matrix
from model_training.ingest import add_winner


def test_add_winner_from_final_scores():
    df = pl.DataFrame({
        "game_id": [1, 1], "homeTeamName": ["A", "A"], "awayTeamName": ["B", "B"],
        "homeScore": [28, 28], "awayScore": [10, 10], "is_home": [1, 0],
        "pos_team": ["A", "B"],
    })
    out = add_winner(df)
    assert out["winner"].to_list() == ["A", "A"]


def test_qbr_matrix_aggregates_per_qb_game():
    # two plays for QB X in game 1 -> one aggregated row with the 6 qbr_vars
    df = pl.DataFrame({
        "game_id": [1, 1], "passer_player_name": ["X", "X"], "season": [2024, 2024],
        "qbr_epa": [0.5, -0.2], "weight": [1.0, 1.0],
        "sack_epa": [None, -0.2], "pass_epa": [0.5, None], "rush_epa": [None, None],
        "pen_epa": [None, None], "sack_weight": [None, 1.0], "pass_weight": [1.0, None],
        "rush_weight": [None, None], "pen_weight": [None, None], "spread": [-3.0, -3.0],
    })
    X, y, w = qbr_matrix(df)
    assert X.shape[0] == 1 and list(X.columns) == ["qbr_epa", "sack_epa", "pass_epa",
                                                   "rush_epa", "pen_epa", "spread"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_features_qbr.py -v`
Expected: FAIL (`cannot import name 'add_winner'` / `qbr_matrix`)

- [ ] **Step 3: Implement `add_winner` (ingest) + `qbr_matrix` (features)**

```python
# python/model_training/ingest.py  (append)
def add_winner(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        winner=pl.when(pl.col("homeScore") > pl.col("awayScore"))
        .then(pl.col("homeTeamName"))
        .when(pl.col("homeScore") < pl.col("awayScore"))
        .then(pl.col("awayTeamName"))
        .otherwise(pl.lit("TIE")),
    )
```

```python
# python/model_training/features.py  (append)
def qbr_matrix(df: pl.DataFrame):
    """Per-(passer, game) weighted means of the 6 qbr_vars (mirrors CFBPlayProcess __process_qbr).

    Target is left to the caller (ESPN raw QBR join) — qbr_matrix returns features + the
    join keys; y/w are None here (the ESPN-QBR target is merged in train_qbr).
    """
    g = (
        df.filter(pl.col("passer_player_name").is_not_null())
        .group_by(["game_id", "season", "passer_player_name"])
        .agg(
            qbr_epa=(pl.col("qbr_epa") * pl.col("weight")).sum() / pl.col("weight").sum(),
            sack_epa=(pl.col("sack_epa") * pl.col("sack_weight")).sum() / pl.col("sack_weight").sum(),
            pass_epa=(pl.col("pass_epa") * pl.col("pass_weight")).sum() / pl.col("pass_weight").sum(),
            rush_epa=(pl.col("rush_epa") * pl.col("rush_weight")).sum() / pl.col("rush_weight").sum(),
            pen_epa=(pl.col("pen_epa") * pl.col("pen_weight")).sum() / pl.col("pen_weight").sum(),
            spread=pl.col("spread").first(),
        )
        .with_columns(pl.col(["sack_epa", "pass_epa", "rush_epa", "pen_epa"]).fill_null(0.0))
    )
    X = g.select(["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"]).to_pandas()
    keys = g.select(["game_id", "season", "passer_player_name"]).to_pandas()
    return X, None, keys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_features_qbr.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add python/model_training/features.py python/model_training/ingest.py tests/model_training/test_features_qbr.py
git commit -m "feat(model-training): winner join + per-QB QBR feature aggregation"
```

---

## Phase 4 — `train_ep.py`

### Task 4.1: EP trainer (plumbing + structure parity)

**Files:**
- Create: `python/model_training/train_ep.py`
- Test: `tests/model_training/test_train_ep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_train_ep.py
import json
import numpy as np
import polars as pl
import xgboost as xgb
from model_training import constants as C
from model_training.train_ep import train_ep


def _synth_ep_frame(n=400):
    rng = np.random.default_rng(0)
    rows = {src: rng.random(n) for src in C.EP_SOURCE.values()}
    rows["label"] = rng.integers(0, 7, n)
    rows["ScoreDiff_W"] = rng.random(n)
    return pl.DataFrame(rows)


def test_train_ep_produces_8feat_7class_softprob():
    model = train_ep(_synth_ep_frame(), nrounds=5)
    cfg = json.loads(model.save_config())["learner"]
    assert model.num_features() == 8
    assert cfg["objective"]["name"] == "multi:softprob"
    assert cfg["learner_model_param"]["num_class"] == "7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_train_ep.py -v`
Expected: FAIL (`No module named 'model_training.train_ep'`)

- [ ] **Step 3: Implement `train_ep`**

```python
# python/model_training/train_ep.py
"""EP model trainer (port of keepers 02_epa_xgb_model.R / model_training.R)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import ep_matrix


def train_ep(df: pl.DataFrame, nrounds: int = C.EP_NROUNDS) -> xgb.Booster:
    X, y, w = ep_matrix(df)
    dtrain = xgb.DMatrix(X, label=y, weight=w)
    return xgb.train(C.EP_PARAMS, dtrain, num_boost_round=nrounds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_train_ep.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/model_training/train_ep.py tests/model_training/test_train_ep.py
git commit -m "feat(model-training): EP trainer (8-feat multi:softprob, nrounds=525)"
```

---

## Phase 5 — `train_wp.py`

### Task 5.1: WP trainer (spread + naive)

**Files:**
- Create: `python/model_training/train_wp.py`
- Test: `tests/model_training/test_train_wp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_train_wp.py
import json
import numpy as np
import polars as pl
from model_training import constants as C
from model_training.train_wp import train_wp


def _synth_wp_frame(n=400):
    rng = np.random.default_rng(1)
    rows = {src: rng.random(n) for src in C.WP_SOURCE.values()}
    rows["pos_team"] = ["A"] * n
    rows["winner"] = rng.choice(["A", "B"], n)
    return pl.DataFrame(rows)


def test_wp_spread_is_13feat_logistic():
    m = train_wp(_synth_wp_frame(), variant="spread", nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 13 and cfg["objective"]["name"] == "binary:logistic"


def test_wp_naive_is_12feat():
    m = train_wp(_synth_wp_frame(), variant="naive", nrounds=5)
    assert m.num_features() == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_train_wp.py -v`
Expected: FAIL (`No module named 'model_training.train_wp'`)

- [ ] **Step 3: Implement `train_wp`**

```python
# python/model_training/train_wp.py
"""WP trainers (spread + naive). Shipped recipe = cfbscrapR-wpa.ipynb (no sample weights)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import wp_matrix

_PARAMS = {"spread": (C.WP_SPREAD_PARAMS, C.WP_SPREAD_NROUNDS),
           "naive": (C.WP_NAIVE_PARAMS, C.WP_NAIVE_NROUNDS)}
_STAGE1 = {"spread": (C.WP_SPREAD_PARAMS_STAGE1, C.WP_SPREAD_NROUNDS_STAGE1)}


def train_wp(df: pl.DataFrame, variant: str = "spread", stage: int = 2,
             nrounds: int | None = None) -> xgb.Booster:
    if stage == 1 and variant in _STAGE1:
        params, default_rounds = _STAGE1[variant]
    else:
        params, default_rounds = _PARAMS[variant]
    X, y, _ = wp_matrix(df, variant=variant)
    dtrain = xgb.DMatrix(X, label=y)  # no weights (shipped recipe)
    return xgb.train(params, dtrain, num_boost_round=nrounds or default_rounds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_train_wp.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add python/model_training/train_wp.py tests/model_training/test_train_wp.py
git commit -m "feat(model-training): WP trainer (spread 13-feat/760, naive 12-feat/65)"
```

---

## Phase 6 — QBR scrape + `train_qbr.py`

### Task 6.0: ESPN QBR scraper (the training target, keyed by game_id)

The 6-feat QBR model's target is ESPN's raw QBR. Add a backfill scraper that hits the ESPN core QBR
endpoint and keys each record by `game_id` (from the event `$ref`) + athlete, so it joins to the
per-QB feature rows on `(game_id, passer_player_name)`.

**Files:**
- Create: `python/scrape_cfb_qbr.py`
- Test: `tests/model_training/test_qbr_scrape.py`
- Test fixture: `tests/fixtures/model_training/qbr_endpoint_sample.json`

- [ ] **Step 1: Capture one endpoint payload as a fixture (live, one-time)**

```bash
uv run python - <<'PY'
import json, requests
url = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/2024/types/2/weeks/1/qbr/10000?limit=1000"
open("tests/fixtures/model_training/qbr_endpoint_sample.json", "w").write(json.dumps(requests.get(url).json()))
print("saved")
PY
```

Expected: `saved` (a `{items:[...]}` payload with `athlete`/`team`/`event` `$ref`s + `splits.categories[0].stats`).

- [ ] **Step 2: Write the failing test (offline parse of the fixture)**

```python
# tests/model_training/test_qbr_scrape.py
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from scrape_cfb_qbr import parse_qbr_payload

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "model_training" / "qbr_endpoint_sample.json"


def test_parse_extracts_game_id_athlete_and_qbr():
    payload = json.loads(FIX.read_text())
    rows = parse_qbr_payload(payload, year=2024, week=1)
    assert rows, "expected QBR rows"
    r = rows[0]
    assert {"game_id", "athlete_id", "year", "week", "QBR", "TQBR"} <= set(r.keys())
    assert str(r["game_id"]).isdigit()  # game_id extracted from event $ref
```

- [ ] **Step 3: Implement the scraper parser + fetch loop**

```python
# python/scrape_cfb_qbr.py
"""Scrape ESPN core QBR (the QBR-model training target), keyed by game_id + athlete.

Endpoint: sports.core.api.espn.com/.../seasons/{yr}/types/2/weeks/{wk}/qbr/10000?limit=1000
Each item has athlete/team/event $refs (event id = game_id) + splits.categories[0].stats
(QBR, TQBR, and component pieces). Output rows join to per-QB feature rows on
(game_id, passer_player_name).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

_EVENT_ID = re.compile(r"/events/(\d+)")
_ATHLETE_ID = re.compile(r"/athletes/(\d+)")


def _qbr_url(year: int, week: int) -> str:
    return (f"https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
            f"seasons/{year}/types/2/weeks/{week}/qbr/10000?limit=1000")


def parse_qbr_payload(payload: dict, year: int, week: int) -> list[dict]:
    rows = []
    for rec in payload.get("items", []) or []:
        ev = (rec.get("event") or {}).get("$ref", "")
        ath = (rec.get("athlete") or {}).get("$ref", "")
        gm = _EVENT_ID.search(ev)
        aid = _ATHLETE_ID.search(ath)
        out = {"year": year, "week": week,
               "game_id": int(gm.group(1)) if gm else None,
               "athlete_id": int(aid.group(1)) if aid else None}
        stats = (((rec.get("splits") or {}).get("categories") or [{}])[0]).get("stats", [])
        for s in stats:
            out[s["abbreviation"]] = s.get("value")
        rows.append(out)
    return rows


def _athlete_name(year: int, athlete_id: int, cache: dict, session: requests.Session) -> str | None:
    if athlete_id in cache:
        return cache[athlete_id]
    url = (f"https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
           f"seasons/{year}/athletes/{athlete_id}?lang=en&region=us")
    try:
        name = session.get(url, timeout=30).json().get("fullName")
    except Exception:
        name = None
    cache[athlete_id] = name
    return name


def scrape(years: range, weeks: range, out_path: str) -> int:
    import pandas as pd
    session = requests.Session()
    cache: dict = {}
    frames = []
    for yr in years:
        for wk in weeks:
            data = session.get(_qbr_url(yr, wk), timeout=30).json()
            rows = parse_qbr_payload(data, yr, wk)
            for r in rows:
                r["passer_player_name"] = _athlete_name(yr, r["athlete_id"], cache, session)
                r["raw_qbr"] = r.get("QBR")
                r["adj_qbr"] = r.get("TQBR")
            if rows:
                frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start", type=int, required=True)
    ap.add_argument("-e", "--end", type=int, required=True)
    ap.add_argument("--out", default="cfb/qbr/espn_qbr.parquet")
    args = ap.parse_args(argv)
    n = scrape(range(args.start, args.end + 1), range(1, 16), args.out)
    print(f"wrote {n} QBR rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_qbr_scrape.py -v`
Expected: PASS (parses `game_id`/`athlete_id`/`QBR`/`TQBR` from the fixture)

- [ ] **Step 5: Commit**

```bash
git add python/scrape_cfb_qbr.py tests/model_training/test_qbr_scrape.py tests/fixtures/model_training/qbr_endpoint_sample.json
git commit -m "feat(model-training): ESPN QBR scraper (training target, keyed by game_id)"
```

### Task 6.1: QBR trainer (6-feat reg:squarederror, ESPN-QBR target)

**Files:**
- Create: `python/model_training/train_qbr.py`
- Test: `tests/model_training/test_train_qbr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_train_qbr.py
import json
import numpy as np
import pandas as pd
from model_training.train_qbr import train_qbr_from_matrix


def test_qbr_model_is_6feat_regression():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((300, 6)),
                     columns=["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"])
    y = rng.random(300) * 100
    m = train_qbr_from_matrix(X, y, nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 6 and cfg["objective"]["name"] == "reg:squarederror"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_train_qbr.py -v`
Expected: FAIL (`No module named 'model_training.train_qbr'`)

- [ ] **Step 3: Implement `train_qbr`**

```python
# python/model_training/train_qbr.py
"""QBR trainer: 6-feat reg:squarederror predicting ESPN raw QBR (shipped qbr_model.ubj).

Features = per-QB-game qbr_vars (from features.qbr_matrix); target = ESPN raw QBR joined
on (passer/season/week-or-game). The ESPN QBR is sourced via the betting/QBR scrape the
backfill already retains, or the cfb_qbr `composite.csv`.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import qbr_matrix


def train_qbr_from_matrix(X: pd.DataFrame, y, nrounds: int = C.QBR_NROUNDS) -> xgb.Booster:
    dtrain = xgb.DMatrix(X[C.QBR_FEATURES], label=y)
    return xgb.train(C.QBR_PARAMS, dtrain, num_boost_round=nrounds)


def train_qbr(df: pl.DataFrame, espn_qbr: pl.DataFrame, nrounds: int = C.QBR_NROUNDS) -> xgb.Booster:
    X, _, keys = qbr_matrix(df)
    feat = pl.from_pandas(keys).hstack(pl.from_pandas(X))
    joined = feat.join(
        espn_qbr.select(["game_id", "passer_player_name", "raw_qbr"]),
        on=["game_id", "passer_player_name"], how="inner",
    ).drop_nulls("raw_qbr")
    Xj = joined.select(C.QBR_FEATURES).to_pandas()
    yj = joined["raw_qbr"].to_numpy()
    return train_qbr_from_matrix(Xj, yj, nrounds=nrounds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_train_qbr.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/model_training/train_qbr.py tests/model_training/test_train_qbr.py
git commit -m "feat(model-training): QBR trainer (6-feat reg:squarederror, ESPN-QBR target)"
```

---

## Phase 7 — `validate.py` (parity + calibration)

### Task 7.1: prediction-parity harness vs reference `.ubj`

**Files:**
- Create: `python/model_training/validate.py`
- Test: `tests/model_training/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_validate.py
import numpy as np
import pandas as pd
import pathlib
import xgboost as xgb
from model_training.validate import prediction_parity

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "model_training"


def test_parity_against_self_is_zero():
    ref = xgb.Booster(); ref.load_model(str(FIX / "xgb_ep_model.ubj"))
    X = pd.DataFrame(np.random.default_rng(0).random((50, 8)))
    report = prediction_parity(ref, ref, X)
    assert report["max_abs_diff"] == 0.0
    assert report["within_tol"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_validate.py -v`
Expected: FAIL (`No module named 'model_training.validate'`)

- [ ] **Step 3: Implement parity + calibration**

```python
# python/model_training/validate.py
"""Validation: prediction-parity vs reference models + LOSO calibration tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


def prediction_parity(model_a: xgb.Booster, model_b: xgb.Booster, X: pd.DataFrame,
                      tol: float = 1e-3) -> dict:
    d = xgb.DMatrix(X)
    pa, pb = model_a.predict(d), model_b.predict(d)
    max_abs = float(np.max(np.abs(pa - pb)))
    return {"max_abs_diff": max_abs, "within_tol": max_abs <= tol, "tol": tol}


def calibration_table(pred_prob, outcome, by, bin_size: float = 0.05) -> pl.DataFrame:
    df = pl.DataFrame({"pred": pred_prob, "outcome": outcome, "by": by})
    df = df.with_columns(bin=(pl.col("pred") / bin_size).round() * bin_size)
    return (
        df.group_by(["by", "bin"])
        .agg(n_plays=pl.len(), n_pos=pl.col("outcome").sum())
        .with_columns(actual=pl.col("n_pos") / pl.col("n_plays"))
        .sort(["by", "bin"])
    )


def weighted_cal_error(table: pl.DataFrame) -> float:
    t = table.with_columns(diff=(pl.col("bin") - pl.col("actual")).abs())
    per = t.group_by("by").agg(
        wce=(pl.col("diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
        n=pl.col("n_pos").sum(),
    )
    return float((per["wce"] * per["n"]).sum() / per["n"].sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/model_training/validate.py tests/model_training/test_validate.py
git commit -m "feat(model-training): prediction-parity + LOSO calibration helpers"
```

### Task 7.2: WP near-parity vs the sanity fixture

**Files:**
- Modify: `tests/model_training/test_validate.py`

- [ ] **Step 1: Add the WP sanity-fixture test**

```python
def test_wp_naive_reference_predicts_on_sanity_fixture():
    import json
    from model_training import constants as C
    ref = xgb.Booster(); ref.load_model(str(FIX / "xgb_wp_naive_model.ubj"))
    items = json.loads((FIX / "wpa-model-test-items.json").read_text())
    X = pd.DataFrame(items)[C.WP_NAIVE_FEATURES]  # 9-feat ref vs 12-feat contract? assert subset
    # the keepers naive ref is 9-feat; assert it at least predicts on its own feature subset
    feats = ref.feature_names or list(X.columns)[: ref.num_features()]
    preds = ref.predict(xgb.DMatrix(pd.DataFrame(items)[[c for c in X.columns][: ref.num_features()]]))
    assert ((preds >= 0) & (preds <= 1)).all()
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/model_training/test_validate.py -v`
Expected: PASS (probabilities in [0,1]). If the reference feature count vs fixture columns mismatches, restrict to the reference's `num_features()` leading columns (the fixture is a superset).

- [ ] **Step 3: Commit**

```bash
git add tests/model_training/test_validate.py
git commit -m "test(model-training): WP reference predicts on sanity fixture"
```

---

## Phase 8 — `figures.py` (plotnine calibration plots + data tables)

### Task 8.1: calibration figure + table writer

**Files:**
- Create: `python/model_training/figures.py`
- Test: `tests/model_training/test_figures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_figures.py
import polars as pl
from model_training.figures import write_calibration


def test_write_calibration_emits_png_and_table(tmp_path):
    table = pl.DataFrame({
        "by": ["1st"] * 5, "bin": [0.1, 0.3, 0.5, 0.7, 0.9],
        "n_plays": [100, 200, 300, 200, 100], "actual": [0.12, 0.28, 0.51, 0.69, 0.93],
    })
    png, csv = write_calibration(table, tmp_path / "wp_spread", title="WP", subtitle="LOSO",
                                 cal_error=0.012)
    assert png.exists() and csv.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_figures.py -v`
Expected: FAIL (`No module named 'model_training.figures'`)

- [ ] **Step 3: Implement `write_calibration` (plotnine, bespoke styling)**

```python
# python/model_training/figures.py
"""plotnine calibration plots (bespoke cfbfastR styling) + sidecar data tables.

Styling target: garnet #500f1b accent, grey95/grey99 panels, Gill Sans MT with a
cross-platform fallback, faceted by `by` (quarter for WP / scoring-event for EP),
sized points + loess + y=x reference, calibration-error caption.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from plotnine import (aes, facet_wrap, geom_abline, geom_point, geom_smooth, ggplot,
                      labs, theme, theme_bw, element_text, element_rect)

GARNET = "#500f1b"
FONT = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]


def write_calibration(table: pl.DataFrame, stem, title: str, subtitle: str,
                      cal_error: float) -> tuple[Path, Path]:
    stem = Path(stem)
    csv = stem.with_suffix(".csv")
    png = stem.with_suffix(".png")
    csv.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(csv)
    table.write_parquet(stem.with_suffix(".parquet"))
    pdf = table.to_pandas()
    p = (
        ggplot(pdf, aes("bin", "actual"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET)
        + geom_smooth(method="loess", se=False, color=GARNET, size=0.5)
        + facet_wrap("~by")
        + labs(title=title, subtitle=subtitle,
               caption=f"Overall Weighted Calibration Error: {cal_error}",
               x="Estimated probability", y="Observed probability", size="Number of plays")
        + theme_bw()
        + theme(text=element_text(family=FONT),
                plot_background=element_rect(fill="grey99", color="black"),
                panel_background=element_rect(fill="grey95"),
                legend_position="bottom")
    )
    p.save(png, width=6, height=4, dpi=200, verbose=False)
    return png, csv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_figures.py -v`
Expected: PASS (PNG + CSV written). If Gill Sans MT is absent the fallback keeps it legible (a font warning is acceptable).

> **Follow-up (blocked on asset):** the cfbfastR hex logo overlay (spec §8, R `add_logo`) is
> deferred until the hex PNG is vendored into `python/model_training/assets/` (spec §11 risk). When
> added, composite it bottom-right with pillow after `p.save(...)`:
> `from PIL import Image; base=Image.open(png); logo=Image.open(asset).resize(...); base.paste(logo, pos, logo); base.save(png)`.

- [ ] **Step 5: Commit**

```bash
git add python/model_training/figures.py tests/model_training/test_figures.py
git commit -m "feat(model-training): plotnine calibration figures + data tables"
```

---

## Phase 9 — `cli.py`

### Task 9.1: subcommand dispatch

**Files:**
- Create: `python/model_training/cli.py`
- Test: `tests/model_training/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/test_cli.py
from model_training.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    sub = {a.dest for a in p._subparsers._group_actions[0].choices}  # type: ignore[attr-defined]
    assert {"ingest", "train-ep", "train-wp", "train-qbr", "validate", "figures"} <= set(
        p._subparsers._group_actions[0].choices.keys()
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/test_cli.py -v`
Expected: FAIL (`No module named 'model_training.cli'`)

- [ ] **Step 3: Implement the CLI**

```python
# python/model_training/cli.py
"""CLI: ingest | train-ep | train-wp | train-qbr | validate | figures."""
from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb

from .ingest import add_winner, build_training_frame, write_training_frame


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="model_training")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest"); i.add_argument("--final-dir", default="cfb/json/final")
    i.add_argument("--out", default="pbp_full.parquet"); i.add_argument("--seasons", nargs="*", type=int)
    for name in ("train-ep", "train-wp", "train-qbr"):
        s = sub.add_parser(name); s.add_argument("--pbp", default="pbp_full.parquet")
        s.add_argument("--out", required=True)
        if name == "train-wp":
            s.add_argument("--variant", choices=["spread", "naive"], default="spread")
        if name == "train-qbr":
            s.add_argument("--espn-qbr", required=True)
    v = sub.add_parser("validate"); v.add_argument("--model", required=True); v.add_argument("--ref", required=True)
    f = sub.add_parser("figures"); f.add_argument("--table", required=True); f.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        n = write_training_frame(args.final_dir, args.out, args.seasons)
        print(f"wrote {n} rows -> {args.out}")
    elif args.cmd in ("train-ep", "train-wp", "train-qbr"):
        import polars as pl
        df = add_winner(pl.read_parquet(args.pbp))
        if args.cmd == "train-ep":
            from .train_ep import train_ep
            model = train_ep(df, nrounds=525 if args.stage == 2 else 525)
        elif args.cmd == "train-wp":
            from .train_wp import train_wp
            model = train_wp(df, variant=args.variant, stage=args.stage)
        else:
            from .train_qbr import train_qbr
            model = train_qbr(df, pl.read_parquet(args.espn_qbr))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(args.out)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/model_training/cli.py tests/model_training/test_cli.py
git commit -m "feat(model-training): CLI subcommand dispatch"
```

### Task 9.2: full-suite run smoke (whatever's on disk)

**Files:**
- Test: `tests/model_training/test_cli_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/model_training/test_cli_smoke.py
import pathlib
import pytest
from model_training.cli import main

FINAL = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL.glob("*.json")), reason="no backfill final.json")
def test_ingest_then_train_ep(tmp_path):
    pbp = tmp_path / "pbp_full.parquet"
    assert main(["ingest", "--final-dir", str(FINAL), "--out", str(pbp)]) == 0
    assert pbp.exists()
    assert main(["--stage", "2", "train-ep", "--pbp", str(pbp), "--out", str(tmp_path / "ep.ubj")]) == 0
    assert (tmp_path / "ep.ubj").exists()
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/model_training/test_cli_smoke.py -v`
Expected: PASS (or SKIP without backfill data)

- [ ] **Step 3: Commit**

```bash
git add tests/model_training/test_cli_smoke.py
git commit -m "test(model-training): ingest -> train-ep CLI smoke"
```

---

## Phase 10 — sdv-py handoff (manual, reviewed) + WP-naive bundling

This phase is documentation + a small sdv-py change; it is **not** automated by the pipeline (decision #11). Performed once Stage-2 models pass the parity gate.

### Task 10.1: handoff runbook

**Files:**
- Create: `python/model_training/HANDOFF.md`

- [ ] **Step 1: Write the runbook**

```markdown
<!-- python/model_training/HANDOFF.md -->
# Model handoff to sdv-py (manual, reviewed)

After Stage-2 training + the parity gate passes:

1. Validate each retrained model vs the shipped one (held-out season, tolerance documented):
   `uv run python -m model_training.cli validate --model <new>.ubj --ref <sdvpy>/cfb/models/<name>.ubj`
2. Copy under review (open a sdv-py PR; never auto-overwrite):
   - `ep_model.ubj`, `wp_spread.ubj`, `qbr_model.ubj` -> `sportsdataverse/cfb/models/`
3. **WP-naive is new to sdv-py.** Also:
   - add `wp_naive.ubj` to `sportsdataverse/cfb/models/` and the `[tool.setuptools.package-data]`
     glob (already `cfb/models/*`, so covered);
   - in `cfb_pbp.py`: load a second booster from `wp_naive.ubj` and emit a `wp_*_naive` output
     alongside the spread WP (mirrors the spread path; uses `wp_final_names` minus `spread_time`);
   - bump the bundled-model note / CHANGELOG in sdv-py.
4. Re-run sdv-py's CFB tests; confirm EPA/WPA on a known game stay within tolerance.
```

- [ ] **Step 2: Commit**

```bash
git add python/model_training/HANDOFF.md
git commit -m "docs(model-training): sdv-py handoff runbook (incl. wp_naive bundling)"
```

---

## Stage gating note

Tasks above build the **machinery** (Stage-agnostic). Stage selection is a runtime flag:

- **Stage 1 (faithful replica):** train on the keepers feature subsets / Stage-1 params (`--stage 1`),
  then `validate` predictions vs `tests/fixtures/model_training/xgb_*.ubj` (tolerance, not byte-equal);
  assert the deterministic intermediates (labels/weights) exactly on a small recomputed slice.
- **Stage 2 (parity upgrade):** train on the shipped contracts/params (`--stage 2`, the defaults),
  full history; `validate` vs the **shipped** `cfb/models/*.ubj` on a held-out season; then Task 10.1.

Run the full suite once at the end of each stage:
`uv run pytest tests/model_training/ -v` (expected: all pass; data-dependent tests SKIP without a backfill).
