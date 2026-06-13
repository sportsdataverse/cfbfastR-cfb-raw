# CFB Modeling Suite — Track 2 (Fourth-Down Yards-Gained Model) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Commit authorship note:** All commits in this plan carry the author Saiem Gilani. **No AI co-author trailers (`Co-Authored-By: Claude …` or similar) should appear on any commit.** SportsDataverse policy prohibits AI attribution on commits — the human author is the sole attributable contributor.

**Goal:** Port the cfb4th yards-gained XGBoost model (5 features, 76 classes, 157 rounds, `multi:softprob`) to Python in `cfbfastR-cfb-raw/python/model_training/fourth_down/`, retraining `fd_model.ubj` on the full CFB backfill history (earliest-available → 2025).

**Architecture:** The 5-feature matrix (`down`, `distance`, `yards_to_goal`, `posteam_total`, `posteam_spread`) is derived directly from `final.json` plays already produced by the backfill's `CFBPlayProcess`. `posteam_spread` = `start.pos_team_spread` (already on the play). `posteam_total` = `(homeTeamSpread + overUnder)/2` if home else `(overUnder - homeTeamSpread)/2`, where both doc-level fields are broadcast to every play. The label = `int(clip(yardsGained, −10, 65) + 10)`. No sample weights (the original model trains without them). Validation = structure assert (5 feats, 76 classes, 157 × 76 = 11 932 trees) + calibration reasonableness (predicted first-down rate vs empirical).

**Tech Stack:** Python 3.11+, uv, polars 1.x, xgboost ≥ 2.0, pandas/pyarrow, plotnine + statsmodels (figures), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-track2-fourth-down-design.md` (umbrella: `…-cfb-modeling-suite-program.md`).

**Prerequisite:** Track 1 phases 0–2 already complete (the `python/model_training/` package exists; `constants.py`, `features.py`, `validate.py`, and `figures.py` are implemented). This plan adds a `fourth_down/` sub-package inside that package.

---

## File structure

New sub-package `python/model_training/fourth_down/`:

| File | Responsibility |
|---|---|
| `python/model_training/fourth_down/__init__.py` | Package marker; re-exports `train_fourth_down`, `fd_features`, `FD_PARAMS`, `FD_FEATURES`. |
| `python/model_training/fourth_down/constants.py` | `FD_FEATURES`, `FD_PARAMS`, `FD_NROUNDS=157`, `FD_NUM_CLASS=76`, clip bounds, label offset. |
| `python/model_training/fourth_down/features.py` | `fd_features(plays_df) -> (pd.DataFrame, np.ndarray)` — filter + derive `posteam_total` + label. |
| `python/model_training/fourth_down/train.py` | `train_fourth_down(df, nrounds) -> xgb.Booster`. |
| `python/model_training/fourth_down/validate.py` | `assert_structure(booster)` + `calibration_fd(booster, X, y_yards)`. |
| `python/model_training/fourth_down/figures.py` | Feature-importance bar + gain-distribution calibration (plotnine, bespoke styling, data tables). |
| `python/model_training/fourth_down/cli.py` | `train-fd` subcommand. |
| `python/model_training/fourth_down/fourth_down_decision.py` | Stub: `get_go_wp_py()` signature + `NotImplementedError`. |
| `tests/model_training/fourth_down/test_constants.py` | Constants shape + value assertions. |
| `tests/model_training/fourth_down/test_features.py` | Filter logic, derivation, label. |
| `tests/model_training/fourth_down/test_train.py` | Structure assert on synthetic frame. |
| `tests/model_training/fourth_down/test_validate.py` | Structure + parity vs fixture. |
| `tests/model_training/fourth_down/test_figures.py` | PNG + CSV smoke. |
| `tests/model_training/fourth_down/test_cli.py` | Subcommand presence. |
| `tests/fixtures/model_training/fd_model.ubj` | cfb4th reference model (one-time convert from R). |
| `tests/fixtures/model_training/fd_fixture_plays.json` | Small synthetic play slice for offline tests. |

Tests run with `uv run pytest tests/model_training/fourth_down/`. The repo already pins `sportsdataverse>=0.0.52` and the Track-1 `model_training` package is already importable from `python/`.

---

## Phase 0 — Scaffold sub-package, reference fixture, constants

### Task 0.1: Create the sub-package skeleton

**Files:**
- Create: `python/model_training/fourth_down/__init__.py`
- Create: `tests/model_training/fourth_down/__init__.py`
- Test: `tests/model_training/fourth_down/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_package.py
def test_fourth_down_package_imports():
    from model_training import fourth_down
    assert hasattr(fourth_down, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/fourth_down/test_package.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down'`)

- [ ] **Step 3: Create the package**

```python
# python/model_training/fourth_down/__init__.py
"""CFB model-training Track 2: fourth-down yards-gained model (5-feat, 76-class multi:softprob)."""
from __future__ import annotations

__version__ = "0.1.0"
```

Also create `tests/model_training/fourth_down/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/fourth_down/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/__init__.py tests/model_training/fourth_down/__init__.py tests/model_training/fourth_down/test_package.py
git commit -m "feat(fourth-down): scaffold fourth_down sub-package"
```

### Task 0.2: Vendor the cfb4th reference model fixture

The cfb4th internal `fd_model` must be extracted to UBJ once from R.

**Files:**
- Create: `tests/fixtures/model_training/fd_model.ubj`
- Modify: `tests/fixtures/model_training/README.md`

- [ ] **Step 1: Extract the reference model from cfb4th (one-time, requires R + cfb4th installed)**

From an R session with cfb4th installed:

```r
library(cfb4th)
xgboost::xgb.save(cfb4th:::fd_model, "tests/fixtures/model_training/fd_model.ubj")
cat("saved; num_features:", cfb4th:::fd_model$nfeatures, "\n")
```

Expected: `num_features: 5` (confirms the 5-feature contract).

Alternative if cfb4th is not installed in the current environment: train a minimal reference on a synthetic 5-feat frame with `nrounds=3` and commit that as a structural-only fixture (then update when cfb4th is available).

- [ ] **Step 2: Verify the fixture loads in the project's xgboost**

Run:

```bash
uv run python - <<'PY'
import xgboost as xgb
b = xgb.Booster()
b.load_model("tests/fixtures/model_training/fd_model.ubj")
import json
cfg = json.loads(b.save_config())["learner"]
n_trees = b.num_boosted_rounds()
print("num_features:", b.num_features())
print("num_class:", cfg["learner_model_param"]["num_class"])
print("n_trees:", n_trees, "= nrounds × 76 =", n_trees // 76, "× 76")
PY
```

Expected: `num_features: 5`, `num_class: 76`, `n_trees: 11932`.

- [ ] **Step 3: Append fixture note to README**

Append to `tests/fixtures/model_training/README.md`:

```markdown
- `fd_model.ubj` — cfb4th internal `fd_model` (Jason Lee / akeaswaran lineage), extracted
  from cfb4th R package sysdata via `xgboost::xgb.save(cfb4th:::fd_model, ...)`.
  5-feat / 76-class / `multi:softprob` / 157 rounds (11932 trees).
  **Track-2 parity reference only.** Retrained on wider window; not the shipped artifact.
```

- [ ] **Step 4: Commit**

```
git add tests/fixtures/model_training/fd_model.ubj tests/fixtures/model_training/README.md
git commit -m "test(fourth-down): vendor cfb4th fd_model reference fixture (5-feat/76-class/157 rounds)"
```

### Task 0.3: `constants.py` — params, feature list, clip bounds

**Files:**
- Create: `python/model_training/fourth_down/constants.py`
- Test: `tests/model_training/fourth_down/test_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_constants.py
from model_training.fourth_down import constants as C


def test_feature_count():
    assert len(C.FD_FEATURES) == 5


def test_feature_order():
    # exact column order the model was trained on (notebook cell 7)
    assert C.FD_FEATURES == ["down", "distance", "yards_to_goal", "posteam_total", "posteam_spread"]


def test_params_objective():
    assert C.FD_PARAMS["objective"] == "multi:softprob"
    assert C.FD_PARAMS["num_class"] == 76
    assert C.FD_PARAMS["eta"] == 0.07
    assert abs(C.FD_PARAMS["gamma"] - 4.325037e-09) < 1e-15


def test_nrounds_and_label_math():
    assert C.FD_NROUNDS == 157
    assert C.FD_NUM_CLASS == 76
    assert C.FD_NROUNDS * C.FD_NUM_CLASS == 11932
    assert C.FD_CLIP_LOW == -10
    assert C.FD_CLIP_HIGH == 65
    assert C.FD_LABEL_OFFSET == 10
    # class 0 = 10-yard loss; class 75 = 65-yard gain
    assert C.FD_CLIP_LOW + C.FD_LABEL_OFFSET == 0
    assert C.FD_CLIP_HIGH + C.FD_LABEL_OFFSET == 75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/fourth_down/test_constants.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.constants'`)

- [ ] **Step 3: Implement constants**

```python
# python/model_training/fourth_down/constants.py
"""Feature contract, XGBoost params, and label bounds for the fourth-down yards model.

Recipe source: fourth-downs.ipynb (akeaswaran / Jason Lee lineage), confirmed by
cfb4th:::fd_model tree count: 157 rounds × 76 classes = 11932 trees.
"""
from __future__ import annotations

# --- feature contract (exact column order, model was trained on these 5 features) ---
FD_FEATURES: list[str] = [
    "down",
    "distance",
    "yards_to_goal",
    "posteam_total",
    "posteam_spread",
]

# --- label bounds (clip + offset: label = clip(yardsGained, LOW, HIGH) + OFFSET) ---
FD_CLIP_LOW: int = -10    # 10-yard loss = class 0
FD_CLIP_HIGH: int = 65    # 65-yard gain = class 75
FD_LABEL_OFFSET: int = 10
FD_NUM_CLASS: int = 76    # classes 0..75 covering integer gains -10..65

# --- XGBoost params (exact, from notebook cell 7 + _go_for_it_cfb_mod.R lines 188-200) ---
FD_PARAMS: dict = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": FD_NUM_CLASS,
    "eta": 0.07,
    "gamma": 4.325037e-09,
    "subsample": 0.5385424,
    "colsample_bytree": 0.6666667,
    "max_depth": 4,
    "min_child_weight": 7,
}
FD_NROUNDS: int = 157

# --- source column names in final.json plays ---
# (the feature builder reads these from the plays frame)
FD_SOURCE: dict[str, str] = {
    "down": "start.down",
    "distance": "start.distance",
    "yards_to_goal": "start.yardsToEndzone",
    # posteam_total is DERIVED (not a direct source column)
    # posteam_spread is read from start.pos_team_spread
    "posteam_spread": "start.pos_team_spread",
}
# Spread + total source columns (doc-level, broadcast to every play by CFBPlayProcess)
FD_SPREAD_COL: str = "homeTeamSpread"     # home-team-perspective spread (negative = home favored)
FD_OVERUNDER_COL: str = "overUnder"       # game total
FD_IS_HOME_COL: str = "start.is_home"     # 1/True if possessing team is home
FD_YARDS_GAINED_COL: str = "yardsGained"  # label source
FD_RUSH_COL: str = "rush"                 # boolean/int — play filter
FD_PASS_COL: str = "pass"                 # boolean/int — play filter
FD_FIRST_DOWN_PENALTY_COLS: tuple[str, ...] = ("firstD_by_penalty", "start.firstD_by_penalty")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/fourth_down/test_constants.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/constants.py tests/model_training/fourth_down/test_constants.py
git commit -m "feat(fourth-down): constants (5-feat contract + params nrounds=157/76-class)"
```

---

## Phase 1 — `features.py` (filter + derivation + label)

### Task 1.1: `posteam_total` derivation + `posteam_spread` passthrough

**Files:**
- Create: `python/model_training/fourth_down/features.py`
- Create: `tests/fixtures/model_training/fd_fixture_plays.json`
- Test: `tests/model_training/fourth_down/test_features.py`

- [ ] **Step 1: Create a small synthetic fixture for offline tests**

```python
# run once to produce tests/fixtures/model_training/fd_fixture_plays.json
import json

plays = [
    # play 1: home offense, 3rd down, drive play with yards gained
    {
        "start.down": 3, "start.distance": 5, "start.yardsToEndzone": 40,
        "start.pos_team_spread": -7.0,   # home is favored (spread is negative)
        "homeTeamSpread": -7.0, "overUnder": 55.0,
        "start.is_home": 1,
        "yardsGained": 7,
        "rush": True, "pass": False, "firstD_by_penalty": False,
    },
    # play 2: away offense, 4th down
    {
        "start.down": 4, "start.distance": 2, "start.yardsToEndzone": 15,
        "start.pos_team_spread": 7.0,    # away is underdog (spread is positive from posteam view)
        "homeTeamSpread": -7.0, "overUnder": 55.0,
        "start.is_home": 0,
        "yardsGained": -3,
        "rush": False, "pass": True, "firstD_by_penalty": False,
    },
    # play 3: 1st down — should be filtered out
    {
        "start.down": 1, "start.distance": 10, "start.yardsToEndzone": 70,
        "start.pos_team_spread": 0.0,
        "homeTeamSpread": 0.0, "overUnder": 50.0,
        "start.is_home": 1,
        "yardsGained": 5,
        "rush": True, "pass": False, "firstD_by_penalty": False,
    },
    # play 4: null overUnder — should be filtered out
    {
        "start.down": 4, "start.distance": 1, "start.yardsToEndzone": 5,
        "start.pos_team_spread": 3.0,
        "homeTeamSpread": None, "overUnder": None,
        "start.is_home": 0,
        "yardsGained": 2,
        "rush": True, "pass": False, "firstD_by_penalty": False,
    },
    # play 5: null yardsGained — should be filtered out
    {
        "start.down": 3, "start.distance": 8, "start.yardsToEndzone": 30,
        "start.pos_team_spread": -3.0,
        "homeTeamSpread": -3.0, "overUnder": 48.0,
        "start.is_home": 1,
        "yardsGained": None,
        "rush": True, "pass": False, "firstD_by_penalty": False,
    },
]

with open("tests/fixtures/model_training/fd_fixture_plays.json", "w") as f:
    json.dump(plays, f, indent=2)
print("wrote fixture")
```

Run this once: `uv run python - <<'PY' ... PY` (or write and delete the script).

- [ ] **Step 2: Write the failing tests**

```python
# tests/model_training/fourth_down/test_features.py
import json
import math
import pathlib

import numpy as np
import polars as pl
import pytest

from model_training.fourth_down.features import fd_features

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training" / "fd_fixture_plays.json"


def _load_plays() -> pl.DataFrame:
    plays = json.loads(FIX.read_text())
    return pl.DataFrame(plays, infer_schema_length=None)


def test_filter_keeps_only_3rd_and_4th():
    X, y = fd_features(_load_plays())
    # play 3 (down=1) and plays 4/5 (null overUnder / null yardsGained) must be dropped
    assert len(X) == 2


def test_feature_columns_and_order():
    from model_training.fourth_down.constants import FD_FEATURES
    X, y = fd_features(_load_plays())
    assert list(X.columns) == FD_FEATURES


def test_posteam_total_home_offense():
    # play 1: is_home=1, homeTeamSpread=-7.0, overUnder=55.0
    # home_total = (-7 + 55) / 2 = 24.0
    X, y = fd_features(_load_plays())
    home_row = X[X["start.down"] == 3] if "start.down" in X.columns else X.iloc[0:1]
    # use row index 0 (home offense row)
    assert abs(X.iloc[0]["posteam_total"] - 24.0) < 1e-9


def test_posteam_total_away_offense():
    # play 2: is_home=0, homeTeamSpread=-7.0, overUnder=55.0
    # away_total = (55 - (-7)) / 2 = 31.0
    X, y = fd_features(_load_plays())
    assert abs(X.iloc[1]["posteam_total"] - 31.0) < 1e-9


def test_posteam_spread_passthrough():
    # play 1: start.pos_team_spread = -7.0 (already correct)
    # play 2: start.pos_team_spread = 7.0 (already correct)
    X, y = fd_features(_load_plays())
    assert X.iloc[0]["posteam_spread"] == -7.0
    assert X.iloc[1]["posteam_spread"] == 7.0


def test_label_clip_and_offset():
    # play 1: yardsGained=7 -> clip(-10,65) -> 7 -> +10 -> 17
    # play 2: yardsGained=-3 -> clip(-10,65) -> -3 -> +10 -> 7
    X, y = fd_features(_load_plays())
    assert y[0] == 17
    assert y[1] == 7


def test_label_dtype_is_integer():
    _, y = fd_features(_load_plays())
    assert y.dtype in (np.int32, np.int64, int)


def test_no_weights_returned():
    # fd_features returns only (X, y) — no weights (decision #11)
    result = fd_features(_load_plays())
    assert len(result) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/model_training/fourth_down/test_features.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.features'`)

- [ ] **Step 4: Implement `fd_features`**

```python
# python/model_training/fourth_down/features.py
"""Build the 5-feature matrix and 76-class label for the fourth-down yards model.

Input: a polars DataFrame of final.json plays (or a concat of multiple games).
Output: (X: pd.DataFrame[5 cols], y: np.ndarray[int]) — no sample weights (decision #11).

Feature derivation:
  posteam_total  = (homeTeamSpread + overUnder)/2  if start.is_home else (overUnder-homeTeamSpread)/2
  posteam_spread = start.pos_team_spread  (already correct posteam perspective, set by CFBPlayProcess)
  down, distance, yards_to_goal = start.* columns directly

Label:
  label = int(clip(yardsGained, -10, 65) + 10)  — class 0..75
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .constants import (
    FD_CLIP_HIGH,
    FD_CLIP_LOW,
    FD_FEATURES,
    FD_FIRST_DOWN_PENALTY_COLS,
    FD_IS_HOME_COL,
    FD_LABEL_OFFSET,
    FD_OVERUNDER_COL,
    FD_SPREAD_COL,
    FD_YARDS_GAINED_COL,
)


def _first_down_penalty_col(df: pl.DataFrame) -> str:
    """Return whichever first-down-penalty column name is present in the frame."""
    for name in FD_FIRST_DOWN_PENALTY_COLS:
        if name in df.columns:
            return name
    return FD_FIRST_DOWN_PENALTY_COLS[0]  # will yield nulls; filter handles it gracefully


def fd_features(plays: pl.DataFrame) -> tuple["pd.DataFrame", np.ndarray]:  # type: ignore[type-arg]
    """Filter plays and build the (X, y) pair for the fourth-down yards-gained model.

    Args:
        plays: polars DataFrame of final.json play records (all downs, all play types).

    Returns:
        X: pandas DataFrame with exactly the 5 columns in FD_FEATURES order.
        y: integer ndarray of class labels (0..75).
    """
    fdp_col = _first_down_penalty_col(plays)

    # --- step 1: down filter (keep 3rd and 4th only) ---
    df = plays.filter(pl.col("start.down").is_in([3, 4]))

    # --- step 2: play-type filter ---
    rush = pl.col("rush").cast(pl.Boolean) if plays.schema.get("rush") not in (pl.Boolean,) else pl.col("rush")
    pass_ = pl.col("pass").cast(pl.Boolean) if plays.schema.get("pass") not in (pl.Boolean,) else pl.col("pass")
    if fdp_col in df.columns:
        fdp = pl.col(fdp_col).fill_null(False).cast(pl.Boolean)
    else:
        fdp = pl.lit(False)
    df = df.filter(rush | pass_ | fdp)

    # --- step 3: distance / yards_to_goal guards ---
    df = df.filter(
        (pl.col("start.distance") > 0)
        & (pl.col("start.yardsToEndzone") > 0)
        & (pl.col("start.distance") <= pl.col("start.yardsToEndzone"))  # R filter: distance <= yards_to_goal
    )

    # --- step 4: spread / overUnder must be present ---
    df = df.filter(
        pl.col(FD_SPREAD_COL).is_not_null()
        & pl.col(FD_OVERUNDER_COL).is_not_null()
    )

    # --- step 5: yardsGained must be present (label source) ---
    df = df.filter(pl.col(FD_YARDS_GAINED_COL).is_not_null())

    if df.is_empty():
        import pandas as pd
        X_empty = pd.DataFrame(columns=FD_FEATURES)
        return X_empty, np.array([], dtype=np.int32)

    # --- derive posteam_total ---
    home_total = (pl.col(FD_SPREAD_COL) + pl.col(FD_OVERUNDER_COL)) / 2.0
    away_total = (pl.col(FD_OVERUNDER_COL) - pl.col(FD_SPREAD_COL)) / 2.0
    df = df.with_columns(
        posteam_total=pl.when(pl.col(FD_IS_HOME_COL).cast(pl.Boolean) == True)
        .then(home_total)
        .otherwise(away_total),
        posteam_spread=pl.col("start.pos_team_spread"),
    )

    # --- build label ---
    df = df.with_columns(
        _label=(
            pl.col(FD_YARDS_GAINED_COL)
            .cast(pl.Float64)
            .clip(FD_CLIP_LOW, FD_CLIP_HIGH)
            + FD_LABEL_OFFSET
        ).cast(pl.Int32)
    )

    # --- select the 5 feature columns (in exact model order) ---
    col_map = {
        "down": "start.down",
        "distance": "start.distance",
        "yards_to_goal": "start.yardsToEndzone",
        "posteam_total": "posteam_total",    # derived above
        "posteam_spread": "posteam_spread",  # derived above
    }
    X = df.select(
        [pl.col(col_map[f]).alias(f) for f in FD_FEATURES]
    ).to_pandas()

    y = df["_label"].to_numpy()
    return X, y
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_features.py -v`
Expected: PASS (7 tests). If `posteam_total` derivation fails on the `cast(pl.Boolean)` for integer `is_home`, the `pl.when(... == True)` handles both 0/1 integers and booleans cleanly.

- [ ] **Step 6: Commit**

```
git add python/model_training/fourth_down/features.py tests/model_training/fourth_down/test_features.py tests/fixtures/model_training/fd_fixture_plays.json
git commit -m "feat(fourth-down): feature builder (5-feat matrix + clip+10 label, no weights)"
```

### Task 1.2: Edge-case filter coverage

**Files:**
- Modify: `tests/model_training/fourth_down/test_features.py`

- [ ] **Step 1: Add edge-case tests**

```python
# append to tests/model_training/fourth_down/test_features.py

def test_clip_low_yields_class_0():
    """yardsGained < -10 is clipped to -10 -> class 0."""
    df = pl.DataFrame([{
        "start.down": 4, "start.distance": 3, "start.yardsToEndzone": 10,
        "start.pos_team_spread": 2.0,
        "homeTeamSpread": 2.0, "overUnder": 50.0, "start.is_home": 1,
        "yardsGained": -20.0,  # should clip to -10 -> class 0
        "rush": True, "pass": False, "firstD_by_penalty": False,
    }])
    _, y = fd_features(df)
    assert y[0] == 0


def test_clip_high_yields_class_75():
    """yardsGained > 65 is clipped to 65 -> class 75."""
    df = pl.DataFrame([{
        "start.down": 3, "start.distance": 10, "start.yardsToEndzone": 80,
        "start.pos_team_spread": -14.0,
        "homeTeamSpread": -14.0, "overUnder": 60.0, "start.is_home": 1,
        "yardsGained": 80.0,  # should clip to 65 -> class 75
        "rush": False, "pass": True, "firstD_by_penalty": False,
    }])
    _, y = fd_features(df)
    assert y[0] == 75


def test_distance_greater_than_yards_to_goal_excluded():
    """distance > yards_to_goal should be excluded (filter step 3)."""
    df = pl.DataFrame([{
        "start.down": 4, "start.distance": 20, "start.yardsToEndzone": 5,
        "start.pos_team_spread": 0.0,
        "homeTeamSpread": 0.0, "overUnder": 50.0, "start.is_home": 1,
        "yardsGained": 3.0,
        "rush": True, "pass": False, "firstD_by_penalty": False,
    }])
    X, y = fd_features(df)
    assert len(X) == 0


def test_empty_input_returns_empty_frame():
    X, y = fd_features(pl.DataFrame())
    assert len(X) == 0 and len(y) == 0


def test_first_down_penalty_included_without_rush_pass():
    """A play with only first_down_penalty=True (no rush/pass) should be included."""
    df = pl.DataFrame([{
        "start.down": 4, "start.distance": 2, "start.yardsToEndzone": 10,
        "start.pos_team_spread": 3.0,
        "homeTeamSpread": -3.0, "overUnder": 50.0, "start.is_home": 0,
        "yardsGained": 2.0,
        "rush": False, "pass": False, "firstD_by_penalty": True,
    }])
    X, y = fd_features(df)
    assert len(X) == 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_features.py -v`
Expected: PASS (12 tests total)

- [ ] **Step 3: Commit**

```
git add tests/model_training/fourth_down/test_features.py
git commit -m "test(fourth-down): edge-case filter coverage (clip bounds, distance guard, first-down penalty)"
```

---

## Phase 2 — `train.py` (trainer)

### Task 2.1: Trainer — structure assert on synthetic frame

**Files:**
- Create: `python/model_training/fourth_down/train.py`
- Test: `tests/model_training/fourth_down/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_train.py
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from model_training.fourth_down import constants as C
from model_training.fourth_down.train import train_fourth_down


def _synth_fd_frame(n: int = 500) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "down": rng.integers(3, 5, n),
        "distance": rng.integers(1, 20, n).astype(float),
        "yards_to_goal": rng.integers(5, 99, n).astype(float),
        "posteam_total": rng.uniform(20, 70, n),
        "posteam_spread": rng.uniform(-40, 40, n),
    })
    y = rng.integers(0, 76, n)
    return X, y


def test_train_returns_booster():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=3)
    assert isinstance(m, xgb.Booster)


def test_train_structure_5feat_76class_softprob():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=3)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 5
    assert cfg["objective"]["name"] == "multi:softprob"
    assert cfg["learner_model_param"]["num_class"] == "76"


def test_train_nrounds_produces_expected_tree_count():
    """With nrounds=5 and num_class=76, tree count = 5 * 76 = 380."""
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=5)
    assert m.num_boosted_rounds() == 5


def test_train_feature_names_match_fd_features():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=2)
    assert m.feature_names == C.FD_FEATURES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model_training/fourth_down/test_train.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.train'`)

- [ ] **Step 3: Implement `train_fourth_down`**

```python
# python/model_training/fourth_down/train.py
"""Fourth-down yards-gained model trainer.

Trains the 5-feature, 76-class multi:softprob XGBoost model that projects yards gained
on any 3rd/4th-down play. No sample weights (the original model trains without them,
unlike the EP/WP models in Track 1). Feature input is the X pandas DataFrame returned
by fd_features(); the caller is responsible for the df -> (X, y) split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import FD_FEATURES, FD_NROUNDS, FD_PARAMS


def train_fourth_down(
    X: pd.DataFrame,
    y: np.ndarray,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Train the fourth-down yards-gained model.

    Args:
        X: Feature matrix with exactly FD_FEATURES columns in the correct order.
        y: Integer label array (class 0..75).
        nrounds: Number of boosting rounds (default 157, the confirmed recipe value).

    Returns:
        Trained xgboost.Booster saved as multi:softprob with 5 features and 76 classes.
    """
    dtrain = xgb.DMatrix(X[FD_FEATURES], label=y)
    return xgb.train(FD_PARAMS, dtrain, num_boost_round=nrounds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_train.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/train.py tests/model_training/fourth_down/test_train.py
git commit -m "feat(fourth-down): trainer (5-feat multi:softprob, nrounds=157, num_class=76)"
```

### Task 2.2: End-to-end trainer from plays DataFrame

**Files:**
- Modify: `python/model_training/fourth_down/train.py`
- Modify: `tests/model_training/fourth_down/test_train.py`

- [ ] **Step 1: Add the end-to-end overload test**

```python
# append to tests/model_training/fourth_down/test_train.py
import pathlib
import polars as pl
from model_training.fourth_down.train import train_from_plays

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training" / "fd_fixture_plays.json"


def test_train_from_plays_with_fixture():
    import json
    plays = pl.DataFrame(json.loads(FIX.read_text()), infer_schema_length=None)
    m = train_from_plays(plays, nrounds=2)
    assert isinstance(m, xgb.Booster)
    assert m.num_features() == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/fourth_down/test_train.py::test_train_from_plays_with_fixture -v`
Expected: FAIL (`cannot import name 'train_from_plays'`)

- [ ] **Step 3: Add `train_from_plays` convenience wrapper**

```python
# python/model_training/fourth_down/train.py  (append)
from .features import fd_features
import polars as pl


def train_from_plays(
    plays: pl.DataFrame,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Filter plays, build features, and train in one step.

    Args:
        plays: polars DataFrame of final.json play records.
        nrounds: Number of boosting rounds.

    Returns:
        Trained Booster (or raises ValueError if no training rows survive the filter).
    """
    X, y = fd_features(plays)
    if len(X) == 0:
        raise ValueError("No training rows survived the fourth-down feature filter. "
                         "Check that plays include 3rd/4th-down rush/pass rows with "
                         "overUnder, homeTeamSpread, and yardsGained present.")
    return train_fourth_down(X, y, nrounds=nrounds)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_train.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/train.py tests/model_training/fourth_down/test_train.py
git commit -m "feat(fourth-down): train_from_plays convenience wrapper (filter + train in one step)"
```

---

## Phase 3 — `validate.py` (structure assert + calibration harness)

### Task 3.1: Structure assert against the reference fixture

**Files:**
- Create: `python/model_training/fourth_down/validate.py`
- Test: `tests/model_training/fourth_down/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_validate.py
import pathlib

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from model_training.fourth_down import constants as C
from model_training.fourth_down.validate import assert_structure, calibration_fd

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training"
REF_MODEL = FIX / "fd_model.ubj"


@pytest.mark.skipif(not REF_MODEL.exists(), reason="fd_model.ubj fixture not on disk (run Task 0.2)")
def test_assert_structure_passes_on_reference():
    ref = xgb.Booster()
    ref.load_model(str(REF_MODEL))
    assert_structure(ref)  # should not raise


def test_assert_structure_fails_on_wrong_feat_count():
    """A 4-feature model should fail the structure assert."""
    import numpy as np
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 4)), columns=["a", "b", "c", "d"])
    y = rng.integers(0, 76, 50)
    import xgboost as xgb
    from model_training.fourth_down.constants import FD_PARAMS
    m = xgb.train(FD_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=2)
    with pytest.raises(AssertionError, match="num_features"):
        assert_structure(m)


def test_calibration_fd_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "down": rng.integers(3, 5, 200),
        "distance": rng.integers(1, 20, 200).astype(float),
        "yards_to_goal": rng.integers(5, 99, 200).astype(float),
        "posteam_total": rng.uniform(20, 70, 200),
        "posteam_spread": rng.uniform(-40, 40, 200),
    })
    y_yards = rng.integers(-10, 66, 200)
    # Use a minimal synthetic model (not the reference — avoid needing the fixture)
    from model_training.fourth_down.train import train_fourth_down
    y_label = (y_yards + 10).astype(int)
    m = train_fourth_down(X, y_label, nrounds=3)
    table = calibration_fd(m, X, y_yards)
    # table must have: pred_fd_prob, empirical_fd_rate columns
    assert "pred_fd_prob" in table.columns and "empirical_fd_rate" in table.columns
    assert table["pred_fd_prob"].between(0, 1).all()
    assert table["empirical_fd_rate"].between(0, 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model_training/fourth_down/test_validate.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.validate'`)

- [ ] **Step 3: Implement `assert_structure` + `calibration_fd`**

```python
# python/model_training/fourth_down/validate.py
"""Validation helpers for the fourth-down yards-gained model.

assert_structure: verifies the 5-feat / 76-class / multi:softprob contract.
calibration_fd:   builds a predicted first-down probability vs empirical rate table.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from .constants import FD_FEATURES, FD_NUM_CLASS, FD_NROUNDS


def assert_structure(booster: xgb.Booster) -> None:
    """Assert that a Booster matches the fourth-down model's structural contract.

    Raises:
        AssertionError: if num_features, num_class, or objective does not match.
    """
    cfg = json.loads(booster.save_config())["learner"]
    num_class = int(cfg["learner_model_param"]["num_class"])
    objective = cfg["objective"]["name"]
    n_feats = booster.num_features()

    assert n_feats == 5, (
        f"num_features={n_feats}, expected 5. "
        "Confirm the model was trained on [down, distance, yards_to_goal, "
        "posteam_total, posteam_spread]."
    )
    assert num_class == FD_NUM_CLASS, (
        f"num_class={num_class}, expected {FD_NUM_CLASS}. "
        "Model must cover integer gains -10..65 (76 classes)."
    )
    assert objective == "multi:softprob", (
        f"objective={objective!r}, expected 'multi:softprob'."
    )


def assert_structure_full(booster: xgb.Booster) -> None:
    """Full structure assert including tree count (requires the reference nrounds=157)."""
    assert_structure(booster)
    n_trees = booster.num_boosted_rounds()
    assert n_trees == FD_NROUNDS, (
        f"num_boosted_rounds={n_trees}, expected {FD_NROUNDS}. "
        "If this is a retrained model on a wider window, use assert_structure() instead."
    )


def calibration_fd(
    booster: xgb.Booster,
    X: pd.DataFrame,
    y_yards: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted first-down probability vs empirical first-down rate.

    For each play with distance d, first-down occurs when actual yards_gained >= d.
    Predicted P(first down) = sum over classes k where (k - 10) >= d of P(class=k).

    Args:
        booster: Trained fourth-down model.
        X: Feature matrix (must include 'distance' column).
        y_yards: Actual yards gained for each play (-10..65 range, not the class).
        n_bins: Number of quantile bins for grouping predicted probability.

    Returns:
        pandas DataFrame with columns:
            bin_center, pred_fd_prob, empirical_fd_rate, n_plays.
    """
    dmat = xgb.DMatrix(X[FD_FEATURES])
    raw = booster.predict(dmat)  # shape (n_plays * 76,) for old xgb or (n_plays, 76) for new
    if raw.ndim == 1:
        probs = raw.reshape(-1, FD_NUM_CLASS)  # (n_plays, 76)
    else:
        probs = raw

    distance = X["distance"].to_numpy()
    n_plays = len(X)

    # For each play i, compute P(yards_gained >= distance[i])
    # class k corresponds to gain = k - 10; first down when gain >= distance
    gains = np.arange(FD_NUM_CLASS) - 10  # [-10, -9, ..., 65]
    pred_fd = np.array([
        probs[i, gains >= distance[i]].sum()
        for i in range(n_plays)
    ])
    empirical_fd = (y_yards >= distance).astype(float)

    # bin by predicted probability quantile
    bins = np.quantile(pred_fd, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    bin_idx = np.searchsorted(bins, pred_fd, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    rows = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": float((bins[b] + bins[b + 1]) / 2),
            "pred_fd_prob": float(pred_fd[mask].mean()),
            "empirical_fd_rate": float(empirical_fd[mask].mean()),
            "n_plays": int(mask.sum()),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_validate.py -v`
Expected: PASS (3 tests; the structure-against-reference test SKIPs if the fixture is absent)

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/validate.py tests/model_training/fourth_down/test_validate.py
git commit -m "feat(fourth-down): structure assert + first-down calibration harness"
```

---

## Phase 4 — `figures.py` (feature importance + calibration plot)

### Task 4.1: Feature-importance bar + calibration scatter

**Files:**
- Create: `python/model_training/fourth_down/figures.py`
- Test: `tests/model_training/fourth_down/test_figures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_figures.py
import numpy as np
import pandas as pd
import pytest
from model_training.fourth_down.figures import write_fd_figures


def _synth_cal_table():
    return pd.DataFrame({
        "bin_center": [0.1, 0.3, 0.5, 0.7, 0.9],
        "pred_fd_prob": [0.10, 0.29, 0.51, 0.71, 0.88],
        "empirical_fd_rate": [0.12, 0.28, 0.50, 0.73, 0.87],
        "n_plays": [150, 300, 400, 300, 150],
    })


def _synth_importance():
    return pd.DataFrame({
        "Feature": ["posteam_total", "distance", "yards_to_goal", "down", "posteam_spread"],
        "Gain": [0.38, 0.27, 0.20, 0.10, 0.05],
    })


def test_write_fd_figures_emits_expected_files(tmp_path):
    cal_png, imp_png = write_fd_figures(
        cal_table=_synth_cal_table(),
        importance=_synth_importance(),
        out_dir=tmp_path,
        cal_error=0.021,
    )
    assert cal_png.exists()
    assert imp_png.exists()
    # sidecar data tables
    assert (tmp_path / "fd_calibration.csv").exists()
    assert (tmp_path / "fd_feature_importance.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/fourth_down/test_figures.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.figures'`)

- [ ] **Step 3: Implement figures**

```python
# python/model_training/fourth_down/figures.py
"""plotnine calibration + feature-importance figures for the fourth-down yards model.

Bespoke cfbfastR styling: garnet #500f1b accent, grey95/grey99 panels,
Gill Sans MT with cross-platform fallback chain. Emits PNGs + sidecar data tables.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    geom_col,
    geom_point,
    geom_smooth,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_bw,
)

GARNET = "#500f1b"
FONT = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]


def _theme_fd():
    return (
        theme_bw()
        + theme(
            text=element_text(family=FONT),
            plot_background=element_rect(fill="grey99", color="black"),
            panel_background=element_rect(fill="grey95"),
            legend_position="bottom",
        )
    )


def _save(p, path: Path, width=6, height=4, dpi=200) -> Path:
    p.save(str(path), width=width, height=height, dpi=dpi, verbose=False)
    return path


def write_fd_figures(
    cal_table: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir,
    cal_error: float,
) -> tuple[Path, Path]:
    """Write calibration scatter and feature-importance bar chart.

    Args:
        cal_table: DataFrame with columns bin_center, pred_fd_prob, empirical_fd_rate, n_plays.
        importance: DataFrame with columns Feature, Gain (from xgb.importance).
        out_dir: Directory to write PNGs + CSVs into.
        cal_error: Overall weighted calibration error (float, shown in caption).

    Returns:
        (calibration_png_path, importance_png_path)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- sidecar data tables ---
    cal_table.to_csv(out / "fd_calibration.csv", index=False)
    importance.to_csv(out / "fd_feature_importance.csv", index=False)

    # --- calibration scatter ---
    cal_p = (
        ggplot(cal_table, aes("pred_fd_prob", "empirical_fd_rate"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET)
        + geom_smooth(method="loess", se=False, color=GARNET, size=0.5)
        + scale_x_continuous(limits=[0, 1])
        + labs(
            title="Fourth-Down Yards Model — First-Down Calibration",
            subtitle="Predicted P(first down) vs Empirical First-Down Rate",
            caption=f"Weighted Calibration Error: {cal_error:.4f}",
            x="Predicted P(first down)",
            y="Empirical first-down rate",
            size="Number of plays",
        )
        + _theme_fd()
    )
    cal_png = _save(cal_p, out / "fd_calibration.png")

    # --- feature importance bar ---
    imp_sorted = importance.sort_values("Gain", ascending=True).copy()
    imp_p = (
        ggplot(imp_sorted, aes(x="Feature", y="Gain"))
        + geom_col(fill=GARNET)
        + coord_flip()
        + labs(
            title="Fourth-Down Yards Model — Feature Importance",
            subtitle="XGBoost Gain (higher = more important)",
            x="Feature",
            y="Gain",
        )
        + _theme_fd()
    )
    imp_png = _save(imp_p, out / "fd_feature_importance.png")

    return cal_png, imp_png
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/fourth_down/test_figures.py -v`
Expected: PASS (PNG + CSV written for both figures). Font warning for Gill Sans MT on non-Windows is acceptable.

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/figures.py tests/model_training/fourth_down/test_figures.py
git commit -m "feat(fourth-down): plotnine calibration + feature-importance figures (bespoke styling)"
```

---

## Phase 5 — `cli.py` (`train-fd` subcommand)

### Task 5.1: Subcommand dispatch

**Files:**
- Create: `python/model_training/fourth_down/cli.py`
- Test: `tests/model_training/fourth_down/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_cli.py
from model_training.fourth_down.cli import build_parser


def test_train_fd_subcommand_present():
    p = build_parser()
    choices = p._subparsers._group_actions[0].choices  # type: ignore[attr-defined]
    assert "train-fd" in choices


def test_train_fd_accepts_final_dir_and_out():
    p = build_parser()
    args = p.parse_args(["train-fd", "--final-dir", "/tmp/final", "--out", "/tmp/fd.ubj"])
    assert args.final_dir == "/tmp/final"
    assert args.out == "/tmp/fd.ubj"
    assert args.seasons is None


def test_train_fd_accepts_seasons():
    p = build_parser()
    args = p.parse_args(["train-fd", "--final-dir", "cfb/json/final",
                         "--out", "fd.ubj", "--seasons", "2018", "2019", "2020"])
    assert args.seasons == [2018, 2019, 2020]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model_training/fourth_down/test_cli.py -v`
Expected: FAIL (`No module named 'model_training.fourth_down.cli'`)

- [ ] **Step 3: Implement the CLI**

```python
# python/model_training/fourth_down/cli.py
"""CLI for the fourth-down yards model training pipeline.

Usage:
    uv run python -m model_training.fourth_down.cli train-fd \\
        --final-dir cfb/json/final \\
        --out python/model_training/fourth_down/artifacts/fd_model.ubj \\
        --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fourth_down_train")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fd = sub.add_parser("train-fd", help="Train the fourth-down yards-gained model.")
    fd.add_argument("--final-dir", default="cfb/json/final",
                    help="Directory containing final.json play files.")
    fd.add_argument("--out", required=True,
                    help="Output path for the trained fd_model.ubj.")
    fd.add_argument("--seasons", nargs="*", type=int, default=None,
                    help="Seasons to include (default: all available).")
    fd.add_argument("--nrounds", type=int, default=None,
                    help="Override nrounds (default: 157).")
    fd.add_argument("--validate", action="store_true",
                    help="Run structure assert + calibration after training.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "train-fd":
        import polars as pl
        from .constants import FD_NROUNDS
        from .train import train_from_plays

        final_dir = Path(args.final_dir)
        if not final_dir.exists():
            print(f"ERROR: --final-dir {final_dir} does not exist.")
            return 1

        # read all final.json plays
        frames = []
        for fpath in sorted(final_dir.glob("*.json")):
            obj = json.loads(fpath.read_text())
            season = obj.get("season")
            if args.seasons is not None and season not in args.seasons:
                continue
            plays = obj.get("plays") or []
            if plays:
                frames.append(pl.DataFrame(plays, infer_schema_length=None))
        if not frames:
            print("ERROR: No plays found. Check --final-dir and --seasons.")
            return 1

        all_plays = pl.concat(frames, how="diagonal_relaxed")
        print(f"Loaded {all_plays.height} plays from {len(frames)} games.")

        nrounds = args.nrounds or FD_NROUNDS
        model = train_from_plays(all_plays, nrounds=nrounds)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out))
        print(f"Saved fd_model -> {out} ({model.num_boosted_rounds()} rounds, "
              f"{model.num_features()} features)")

        if args.validate:
            from .validate import assert_structure
            assert_structure(model)
            print("Structure assert passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/model_training/fourth_down/test_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/cli.py tests/model_training/fourth_down/test_cli.py
git commit -m "feat(fourth-down): CLI train-fd subcommand (--final-dir, --out, --seasons, --validate)"
```

---

## Phase 6 — Decision-layer stub + `__init__.py` re-exports

### Task 6.1: `fourth_down_decision.py` stub

**Files:**
- Create: `python/model_training/fourth_down/fourth_down_decision.py`
- Test: `tests/model_training/fourth_down/test_decision_stub.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/model_training/fourth_down/test_decision_stub.py
import pytest
from model_training.fourth_down.fourth_down_decision import get_go_wp_py


def test_get_go_wp_py_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Track 1"):
        get_go_wp_py(None, None, None, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/model_training/fourth_down/test_decision_stub.py -v`
Expected: FAIL (`cannot import name 'get_go_wp_py'`)

- [ ] **Step 3: Implement the stub**

```python
# python/model_training/fourth_down/fourth_down_decision.py
"""Fourth-down decision layer: expected-value comparison of go/punt/FG.

STUB — not yet implemented. Depends on Track 1's retrained EP/WP models.

Integration contract with cfb4th/R/decision_functions.R::get_go_wp():
  1. Pass the 5-feature situation matrix (down, distance, yards_to_goal,
     posteam_total, posteam_spread) to fd_model.predict() → 76-class probability
     vector per play.
  2. Expand into a long (play × gain) frame: gain = class_index - 10.
  3. Cap gain at yards_to_goal (TD); floor loss so the ball stays at the 1-yard line.
  4. Update game situation per outcome:
       - possession flip on turnover-on-downs (gain < distance)
       - +6 points and possession flip on TD (gain == yards_to_goal)
       - spread sign flip on possession change
       - clock run-off: TimeSecsRem -= 6; adj_TimeSecsRem -= 6 (min 0)
  5. Call add_ep(situation) + add_wp(situation) from Track 1's Python models.
  6. Weight each outcome's WP by P(gain=k) → go_wp = Σ P(k) × WP(outcome_k).

This function signature mirrors cfb4th::get_go_wp(); implement after Track 1's
EP/WP Python inference paths are confirmed working (Stage 2 retrain complete).
"""
from __future__ import annotations


def get_go_wp_py(pbp_df, fd_model, ep_model, wp_model):
    """Compute the expected win probability of going for it on 4th down.

    Args:
        pbp_df: Play-by-play DataFrame (polars or pandas) with the fourth-down situations.
                Must contain the 5 fourth-down features plus the EP/WP inference columns
                (TimeSecsRem, adj_TimeSecsRem, pos_score_diff_start, etc.).
        fd_model: Trained fourth-down yards-gained XGBoost Booster (Track 2 output).
        ep_model: Trained EP XGBoost Booster (Track 1 Stage-2 output, multi:softprob 7-class).
        wp_model: Trained WP-spread XGBoost Booster (Track 1 Stage-2 output, binary:logistic).

    Returns:
        pbp_df augmented with columns: go_wp, first_down_prob, wp_succeed, wp_fail.

    Raises:
        NotImplementedError: Always — Track 1 Stage-2 EP/WP models must be complete before
            implementing the decision layer. See cfb4th/R/decision_functions.R::get_go_wp()
            for the reference implementation.
    """
    raise NotImplementedError(
        "get_go_wp_py() is not yet implemented. "
        "Implement after Track 1 Stage-2 EP/WP Python inference paths are confirmed "
        "working. See cfb4th/R/decision_functions.R::get_go_wp() for the reference."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/model_training/fourth_down/test_decision_stub.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```
git add python/model_training/fourth_down/fourth_down_decision.py tests/model_training/fourth_down/test_decision_stub.py
git commit -m "feat(fourth-down): decision-layer stub (get_go_wp_py signature + NotImplementedError, awaits Track 1 EP/WP)"
```

### Task 6.2: `__init__.py` re-exports + full-suite smoke

**Files:**
- Modify: `python/model_training/fourth_down/__init__.py`
- Test: `tests/model_training/fourth_down/test_package.py`

- [ ] **Step 1: Update the package init**

```python
# python/model_training/fourth_down/__init__.py
"""CFB model-training Track 2: fourth-down yards-gained model (5-feat, 76-class multi:softprob).

Usage:
    from model_training.fourth_down import train_from_plays, fd_features, FD_PARAMS, FD_FEATURES
    model = train_from_plays(plays_df)
    model.save_model("fd_model.ubj")
"""
from __future__ import annotations

__version__ = "0.1.0"

from .constants import FD_FEATURES, FD_NROUNDS, FD_NUM_CLASS, FD_PARAMS
from .features import fd_features
from .train import train_fourth_down, train_from_plays

__all__ = [
    "FD_FEATURES",
    "FD_NROUNDS",
    "FD_NUM_CLASS",
    "FD_PARAMS",
    "fd_features",
    "train_fourth_down",
    "train_from_plays",
]
```

- [ ] **Step 2: Add re-export test + full-suite smoke**

```python
# append to tests/model_training/fourth_down/test_package.py
from model_training import fourth_down as fd


def test_re_exports():
    assert hasattr(fd, "FD_FEATURES")
    assert hasattr(fd, "FD_PARAMS")
    assert hasattr(fd, "fd_features")
    assert hasattr(fd, "train_fourth_down")
    assert hasattr(fd, "train_from_plays")
```

- [ ] **Step 3: Run the full fourth-down test suite**

Run: `uv run pytest tests/model_training/fourth_down/ -v`
Expected: ALL PASS (or SKIP for the fixture-dependent structure-assert test if `fd_model.ubj` is not yet on disk)

- [ ] **Step 4: Commit**

```
git add python/model_training/fourth_down/__init__.py tests/model_training/fourth_down/test_package.py
git commit -m "feat(fourth-down): __init__ re-exports + package API surface"
```

---

## Phase 7 — Data-dependent validation (backfill on disk)

This phase requires the CFB backfill's `final.json` files to be on disk. The tests use `pytest.mark.skipif` to skip gracefully when they are absent.

### Task 7.1: Null-rate audit + full-data smoke train

**Files:**
- Test: `tests/model_training/fourth_down/test_data_audit.py`

- [ ] **Step 1: Write the data-dependent audit test**

```python
# tests/model_training/fourth_down/test_data_audit.py
import pathlib
import pytest

FINAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_fourth_down_null_rate_acceptable():
    """Verify yardsGained null rate on 3rd/4th-down rush/pass plays is < 20%.

    A high null rate indicates a data quality issue that would reduce the training set
    to a biased subsample. If this test fails, audit which seasons/game_ids have null
    yardsGained and consider excluding them (as the R script's BAD_GAME_IDS list does).
    """
    import json
    import polars as pl

    frames = []
    for f in sorted(FINAL_DIR.glob("*.json"))[:200]:  # audit on first 200 games
        obj = json.loads(f.read_text())
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        pytest.skip("no plays in first 200 final.json files")

    df = pl.concat(frames, how="diagonal_relaxed")
    # pre-filter: 3rd/4th down rush or pass
    subset = df.filter(
        pl.col("start.down").is_in([3, 4])
        & (pl.col("rush").cast(pl.Boolean) | pl.col("pass").cast(pl.Boolean))
    )
    if subset.height == 0:
        pytest.skip("no 3rd/4th down plays in sample")

    null_rate = subset["yardsGained"].null_count() / subset.height
    print(f"yardsGained null rate on 3rd/4th-down rush/pass: {null_rate:.1%} "
          f"({subset['yardsGained'].null_count()} / {subset.height})")
    assert null_rate < 0.20, (
        f"yardsGained null rate {null_rate:.1%} exceeds 20% threshold. "
        "Audit the backfill for plays missing yardsGained."
    )


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_smoke_train_on_backfill(tmp_path):
    import json
    import polars as pl
    from model_training.fourth_down import train_from_plays
    from model_training.fourth_down.validate import assert_structure

    frames = []
    for f in sorted(FINAL_DIR.glob("*.json"))[:50]:  # 50 games for speed
        obj = json.loads(f.read_text())
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        pytest.skip("no plays found")

    df = pl.concat(frames, how="diagonal_relaxed")
    model = train_from_plays(df, nrounds=5)  # fast run, not the final model
    assert_structure(model)
    out = tmp_path / "fd_model_smoke.ubj"
    model.save_model(str(out))
    assert out.exists()
    print(f"Smoke train complete: {model.num_boosted_rounds()} rounds, "
          f"{model.num_features()} features, file: {out.stat().st_size} bytes")
```

- [ ] **Step 2: Run tests (skip if no data)**

Run: `uv run pytest tests/model_training/fourth_down/test_data_audit.py -v`
Expected: PASS (or SKIP if no backfill data). If the null-rate assert fails, investigate which seasons/game_ids have null `yardsGained` and add them to a `BAD_GAME_IDS_FD` constant in `constants.py`.

- [ ] **Step 3: Commit**

```
git add tests/model_training/fourth_down/test_data_audit.py
git commit -m "test(fourth-down): null-rate audit + smoke train (skipped without backfill)"
```

---

## Stage gating note

All tasks above build the machinery (stage-agnostic). The fourth-down model has only one stage (no faithful-replica vs parity-upgrade distinction as in Track 1, because there is no divergent lineage):

- **Stage 1 (offline, any machine):** `uv run pytest tests/model_training/fourth_down/ -v` — all non-data-dependent tests pass; data-dependent tests skip.
- **Stage 2 (with backfill, full training):** run `uv run python -m model_training.fourth_down.cli train-fd --final-dir cfb/json/final --out python/model_training/fourth_down/artifacts/fd_model.ubj --validate` on a machine with the full backfill. Inspect calibration outputs. Commit `fd_model.ubj` under review.

Run the full sub-package suite at the end of each phase:
`uv run pytest tests/model_training/fourth_down/ -v`
(expected: all pass; Task 7.x tests SKIP without a backfill)
