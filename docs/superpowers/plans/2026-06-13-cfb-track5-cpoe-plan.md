# CFB Track 5 — CPOE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CFB-native Completion Percentage Over Expected (CPOE) pipeline in
`cfbfastR-cfb-raw/python/cpoe/` that trains on ESPN `final.json` pass plays, produces a
per-play completion probability model, and emits per-QB-game and per-QB-season CPOE.

**Architecture:** Eight game-state features from `CFBPlayProcess` output (`final.json` plays),
`binary:logistic` XGBoost, LOSO calibration. The StatsBomb-trained R original (`cpoe_model.R`)
is a conceptual reference only — all five of its throw-level features are absent from the ESPN
backfill. See the design spec (§3) for the full feature-availability analysis.

**Tech Stack:** Python 3.11+, uv, polars 1.x, xgboost ≥ 2.0, plotnine + statsmodels (figures),
pandas/pyarrow, pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-track5-cpoe-design.md`
**Program umbrella:** `docs/superpowers/specs/2026-06-13-cfb-modeling-suite-program.md`

**Commit convention:** Conventional Commits (`feat(cpoe): ...`, `test(cpoe): ...`, etc.).
No AI co-author trailers on any commit in this track.

---

## PHASE 0 — Feasibility Gate

> **This phase is gated.** Phases 1–N are conditional on the Phase 0 verdict.
> The Phase 0 analysis was partially performed during spec authorship (ESPN `final.json`
> inspection confirmed: all five StatsBomb features absent, `yds_receiving` 91% null on
> completions). Task 0.2 (CFBD air-yards) is the one remaining open item.

### Task 0.1: Document the completed ESPN final.json feature-availability analysis

**Files:**
- Create: `python/cpoe/FEASIBILITY.md`

This task captures the already-completed investigation as a committed artifact, so the
Phase 0 verdict is reproducible by any contributor.

- [ ] **Step 1: Create the feasibility document**

```markdown
<!-- python/cpoe/FEASIBILITY.md -->
# Track 5 CPOE — Phase 0 Feasibility Analysis

## Source model features (`cpoe_model.R`, StatsBomb AMF)

The R original trains on StatsBomb `tb12_events_dataset_{start}_{end}.csv` joined to
`tb12_plays_dataset_{start}_{end}.csv`. The five model features are:

| Feature | Present in ESPN final.json? |
|---|---|
| event_pass_air_yards | NO |
| play_target_separation | NO |
| event_pass_target_x | NO |
| event_pass_target_y | NO |
| play_qb_pressure | NO |
| endline_receiver_dist (derived: 110 - target_x) | NO |
| sideline_receiver_dist (derived: min(y, 53.33-y)) | NO |

## ESPN final.json inspection (game: 401628455, 2024 season)

Inspection via `python/cpoe/inspect_final_json.py` (see Task 0.1 script).

Pass plays found: 65. Completions: 36. All five StatsBomb features: None on every play.
`yds_receiving`: null on 33/36 completions (~91%). `statYardage` = total play yards, not
throw distance; not a usable air-yards proxy.

## Verdict

The StatsBomb-trained CPOE cannot be ported to CFB ESPN data. Approach A (8-feature
game-state model) is the primary path. CFBD air-yards (Approach B) requires Task 0.2 investigation.

## Available features for Approach A (Reduced Game-State Model)

| Feature | Column | Always populated? |
|---|---|---|
| Down | start.down | YES |
| Distance to 1st | start.distance | YES |
| Yards to goal | start.yardsToEndzone | YES |
| Score diff | pos_score_diff_start | YES |
| Seconds remaining | start.TimeSecsRem | YES |
| Is home | start.is_home | YES |
| Period | period | YES |
| Passing down flag | passing_down | YES |
```

- [ ] **Step 2: Commit**

```bash
git add python/cpoe/FEASIBILITY.md
git commit -m "docs(cpoe): Phase 0 feasibility analysis — StatsBomb features absent from ESPN final.json"
```

---

### Task 0.2: CFBD air-yards fill-rate investigation (gates Approach B)

**Files:**
- Create: `python/cpoe/inspect_cfbd_air_yards.py` (one-time investigation script)
- Output: print/write a fill-rate report to stdout or `python/cpoe/CFBD_AIR_YARDS.md`

**Expected output:** For a sample of CFBD PBP data (seasons 2020–2024, ~5 games per season),
report: total pass plays, pass plays with `air_yards` non-null, fill rate %.

- [ ] **Step 1: Write the investigation script**

```python
# python/cpoe/inspect_cfbd_air_yards.py
"""One-time investigation: what is the CFBD air_yards fill rate on CFB pass plays?

Uses cfbd-python (pip install cfbd) or a direct requests call to the CFBD API.
Run: uv run python python/cpoe/inspect_cfbd_air_yards.py

CFBD PBP endpoint: https://api.collegefootballdata.com/plays?seasonType=regular&year=Y&week=W&team=T
Air yards field name: 'yards_to_sticks' (not 'air_yards') — confirm from actual response.
"""
import json
import requests

CFBD_BASE = "https://api.collegefootballdata.com"
SAMPLE_GAMES = [
    (2020, 1, "Alabama"),
    (2021, 1, "Ohio State"),
    (2022, 1, "Georgia"),
    (2023, 1, "Michigan"),
    (2024, 1, "Texas"),
]

def get_plays(year, week, team, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{CFBD_BASE}/plays"
    params = {"seasonType": "regular", "year": year, "week": week, "team": team}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    total_pass = 0
    total_with_air = 0
    for year, week, team in SAMPLE_GAMES:
        try:
            plays = get_plays(year, week, team)
            pass_plays = [p for p in plays if "pass" in str(p.get("play_type","")).lower()]
            # check what field names are available
            if pass_plays:
                print(f"\n{year} wk{week} {team}: {len(pass_plays)} pass plays")
                print("  Available fields:", list(pass_plays[0].keys()))
                # look for any air_yards / depth column
                for candidate in ["air_yards", "yards_to_sticks", "pass_length"]:
                    has = sum(1 for p in pass_plays if p.get(candidate) is not None)
                    print(f"  {candidate} non-null: {has}/{len(pass_plays)} ({has/len(pass_plays)*100:.0f}%)")
            total_pass += len(pass_plays)
        except Exception as e:
            print(f"  ERROR {year} {team}: {e}")
    print(f"\nSummary: {total_pass} pass plays sampled")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the investigation**

```bash
uv run python python/cpoe/inspect_cfbd_air_yards.py
```

- [ ] **Step 3: Record the verdict**

If fill rate ≥ 60%: proceed to implement Approach B enhancement in Phase 3 (Task 3.2).
If fill rate < 60%: document "CFBD air_yards insufficient; Approach A only" and skip Task 3.2.

Write the verdict as a comment in `python/cpoe/FEASIBILITY.md` under a new `## CFBD Air Yards`
section:

```markdown
## CFBD Air Yards (Task 0.2 result)

CFBD air_yards fill rate: XX% (YY/ZZ pass plays across 5 sampled game-seasons).
Verdict: [Approach B FEASIBLE / INFEASIBLE]. [See Task 3.2 / Approach A only.]
```

- [ ] **Step 4: Commit**

```bash
git add python/cpoe/FEASIBILITY.md python/cpoe/inspect_cfbd_air_yards.py
git commit -m "docs(cpoe): CFBD air-yards fill-rate verdict (Phase 0 Task 0.2)"
```

> **DECISION GATE:** If Approach B is INFEASIBLE, skip Phase 3 Task 3.2 entirely.
> All subsequent phases proceed regardless of the Task 0.2 verdict (Approach A is always built).

---

## PHASE 1 — Package Scaffold + Constants

### Task 1.1: Create the `cpoe` package skeleton

**Files:**
- Create: `python/cpoe/__init__.py`
- Create: `tests/cpoe/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_package.py
def test_package_imports():
    import cpoe
    assert hasattr(cpoe, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_package.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'cpoe'`)

- [ ] **Step 3: Create the package**

```python
# python/cpoe/__init__.py
"""CFB CPOE (Completion Percentage Over Expected) — Track 5 of the CFB Modeling Suite.

Game-state completion probability model trained on ESPN final.json pass plays.
NOT a port of the StatsBomb-trained cpoe_model.R — all five StatsBomb throw-level
features are absent from the ESPN CFB backfill. See FEASIBILITY.md.
"""
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 4: Make `python/` importable in `tests/cpoe/`**

Verify `tests/conftest.py` already has `sys.path.insert(0, ".../python")` (added by Track 1
Task 0.1). If not, add it.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_package.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/cpoe/__init__.py tests/cpoe/test_package.py
git commit -m "feat(cpoe): scaffold cpoe package"
```

### Task 1.2: Feature constants

**Files:**
- Create: `python/cpoe/constants.py`
- Test: `tests/cpoe/test_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_constants.py
from cpoe import constants as C


def test_approach_a_features_all_in_final_json():
    """All Approach A features must be known final.json column names."""
    KNOWN_FINAL_JSON_COLS = {
        "start.down", "start.distance", "start.yardsToEndzone",
        "pos_score_diff_start", "start.TimeSecsRem", "start.is_home",
        "period", "passing_down",
    }
    for src in C.CP_FEATURE_SOURCE.values():
        assert src in KNOWN_FINAL_JSON_COLS, f"{src} not a known final.json column"


def test_feature_list_length_approach_a():
    assert len(C.CP_FEATURES_A) == 8


def test_xgb_params_are_binary_logistic():
    assert C.CP_PARAMS["objective"] == "binary:logistic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_constants.py -v`
Expected: FAIL (`No module named 'cpoe.constants'`)

- [ ] **Step 3: Implement constants**

```python
# python/cpoe/constants.py
"""Feature contracts and XGBoost params for the CFB CPOE model (Approach A).

Approach A: 8 game-state features from ESPN final.json pass plays.
Approach B extension (air_yards from CFBD) is gated by Phase 0 Task 0.2 verdict.

StatsBomb original features (cpoe_model.R) are documented here for reference;
all five are absent from ESPN final.json — see FEASIBILITY.md.
"""
from __future__ import annotations

# --- StatsBomb original feature set (REFERENCE ONLY; absent from ESPN CFB pbp) ---
STATSBOMB_FEATURES: list[str] = [
    "event_pass_air_yards",      # throw distance through air (Euclidean)
    "play_target_separation",    # yards from nearest defender at catch point
    "play_qb_pressure",          # QB under pressure (bool, null->False)
    "endline_receiver_dist",     # 110 - event_pass_target_x
    "sideline_receiver_dist",    # min(y, 53.33-y) from event_pass_target_y
]
STATSBOMB_NROUNDS = 560
STATSBOMB_SEASONS = list(range(2017, 2023))  # 2017-18 through 2022-23

# --- Approach A: game-state features (ESPN final.json, always present) ---
# Maps feature name used in model -> source column name in final.json plays.
CP_FEATURE_SOURCE: dict[str, str] = {
    "down":           "start.down",
    "distance":       "start.distance",
    "yards_to_goal":  "start.yardsToEndzone",
    "pos_score_diff": "pos_score_diff_start",
    "secs_remaining": "start.TimeSecsRem",
    "is_home":        "start.is_home",
    "period":         "period",
    "passing_down":   "passing_down",
}
CP_FEATURES_A: list[str] = list(CP_FEATURE_SOURCE.keys())  # length == 8

# --- Approach B extension (conditional on Phase 0 Task 0.2) ---
# If CFBD air_yards fill rate >= 60%, add this feature for post-2020 seasons.
CP_FEATURE_SOURCE_B: dict[str, str] = {
    **CP_FEATURE_SOURCE,
    "air_yards": "air_yards",   # from CFBD PBP join; null where unavailable
}
CP_FEATURES_B: list[str] = list(CP_FEATURE_SOURCE_B.keys())  # length == 9

# --- XGBoost params (Approach A starting point; nrounds tuned by LOSO CV) ---
CP_PARAMS: dict = {
    "booster":          "gbtree",
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "eta":              0.025,
    "gamma":            5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "max_depth":        4,
    "min_child_weight": 6,
}
CP_NROUNDS: int = 400  # starting point; tuned in Phase 2 LOSO CV

# --- Distance buckets (yards-to-first-down proxy for throw depth) ---
# Note: this is a coarse proxy for air yards; documented on every calibration figure.
DISTANCE_BUCKETS: dict[str, tuple] = {
    "Short":        (0, 3),    # start.distance <= 3
    "Intermediate": (4, 8),    # 4 <= start.distance <= 8
    "Long":         (9, 9999), # start.distance >= 9
}

# --- Minimum pass attempts for a QB-season CPOE to be considered reliable ---
MIN_ATTEMPTS_SEASON: int = 100
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_constants.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/constants.py tests/cpoe/test_constants.py
git commit -m "feat(cpoe): feature contracts + XGBoost params (Approach A, 8 game-state features)"
```

---

## PHASE 2 — `features.py` + `train_cp.py`

### Task 2.1: Pass-play feature matrix

**Files:**
- Create: `python/cpoe/features.py`
- Test: `tests/cpoe/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_features.py
import polars as pl
import numpy as np
from cpoe import constants as C
from cpoe.features import cp_matrix, filter_pass_plays


def _pass_frame(n=50):
    rng = np.random.default_rng(0)
    rows = {src: rng.random(n).tolist() for src in C.CP_FEATURE_SOURCE.values()}
    rows["pass_attempt"] = [True] * n
    rows["sack_vec"] = [False] * n
    rows["penalty_no_play"] = [False] * n
    rows["completion"] = rng.integers(0, 2, n).tolist()
    rows["game_id"] = [1] * n
    rows["season"] = [2024] * n
    rows["passer_player_name"] = ["QB A"] * n
    return pl.DataFrame(rows)


def test_cp_matrix_has_8_features():
    X, y, keys = cp_matrix(_pass_frame())
    assert X.shape[1] == 8
    assert list(X.columns) == C.CP_FEATURES_A


def test_cp_matrix_label_is_binary():
    X, y, keys = cp_matrix(_pass_frame())
    assert set(y).issubset({0, 1})


def test_filter_pass_plays_drops_sacks_and_no_plays():
    df = _pass_frame(10)
    df = df.with_columns(
        sack_vec=pl.Series([True] + [False] * 9),
        penalty_no_play=pl.Series([False] * 9 + [True]),
    )
    out = filter_pass_plays(df)
    assert out.height == 8  # 10 - 1 sack - 1 no-play
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_features.py -v`
Expected: FAIL (`No module named 'cpoe.features'`)

- [ ] **Step 3: Implement `features.py`**

```python
# python/cpoe/features.py
"""Select/rename ESPN final.json pass plays into the CP model input matrix.

Returns pandas DataFrames (xgboost DMatrix-ready) + label + join keys.
Pass-play filter: pass_attempt==True, sack_vec==False, penalty_no_play==False.
"""
from __future__ import annotations

import polars as pl

from . import constants as C


def filter_pass_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only genuine pass attempts (not sacks, not penalty-no-play)."""
    return df.filter(
        (pl.col("pass_attempt") == True)  # noqa: E712
        & (pl.col("sack_vec") == False)   # noqa: E712
        & (pl.col("penalty_no_play") == False)  # noqa: E712
    )


def cp_matrix(df: pl.DataFrame, approach: str = "A"):
    """Build the CP model input matrix from final.json pass plays.

    Args:
        df: polars DataFrame of pass plays (already filtered).
        approach: "A" (8-feature game-state) or "B" (9-feature with air_yards).

    Returns:
        (X: pd.DataFrame, y: np.ndarray, keys: pd.DataFrame)
        X has columns in CP_FEATURES_A (or CP_FEATURES_B) order.
        y is the binary completion label (0/1).
        keys is (game_id, season, passer_player_name) for CPOE aggregation joins.
    """
    source = C.CP_FEATURE_SOURCE if approach == "A" else C.CP_FEATURE_SOURCE_B
    feats = C.CP_FEATURES_A if approach == "A" else C.CP_FEATURES_B

    X = (
        df.select([pl.col(src).cast(pl.Float32).alias(name) for name, src in source.items()])
        .select(feats)
        .to_pandas()
    )
    y = df["completion"].cast(pl.Int32).to_numpy()
    keys = df.select(["game_id", "season", "passer_player_name"]).to_pandas()
    return X, y, keys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/features.py tests/cpoe/test_features.py
git commit -m "feat(cpoe): pass-play feature matrix + filter (Approach A, 8-feat game-state)"
```

### Task 2.2: CP model trainer

**Files:**
- Create: `python/cpoe/train_cp.py`
- Test: `tests/cpoe/test_train_cp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_train_cp.py
import json
import numpy as np
import pandas as pd
from cpoe.train_cp import train_cp_from_matrix


def test_cp_model_is_binary_logistic_8feat():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.random((300, 8)),
                     columns=["down","distance","yards_to_goal","pos_score_diff",
                              "secs_remaining","is_home","period","passing_down"])
    y = rng.integers(0, 2, 300)
    m = train_cp_from_matrix(X, y, nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 8
    assert cfg["objective"]["name"] == "binary:logistic"


def test_cp_predictions_in_unit_interval():
    import xgboost as xgb
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.random((50, 8)),
                     columns=["down","distance","yards_to_goal","pos_score_diff",
                              "secs_remaining","is_home","period","passing_down"])
    y = rng.integers(0, 2, 50)
    m = train_cp_from_matrix(X, y, nrounds=5)
    preds = m.predict(xgb.DMatrix(X))
    assert ((preds >= 0) & (preds <= 1)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_train_cp.py -v`
Expected: FAIL (`No module named 'cpoe.train_cp'`)

- [ ] **Step 3: Implement `train_cp.py`**

```python
# python/cpoe/train_cp.py
"""CP model trainer — 8-feature binary:logistic XGBoost (Approach A).

Optionally extends to 9-feature (Approach B with CFBD air_yards) if Phase 0 Task 0.2
confirmed fill rate >= 60%.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import cp_matrix, filter_pass_plays


def train_cp_from_matrix(X: pd.DataFrame, y, nrounds: int = C.CP_NROUNDS) -> xgb.Booster:
    """Train CP model from a pre-built feature matrix.

    Args:
        X: Feature matrix (columns must match CP_FEATURES_A in order).
        y: Binary label array (1=completion, 0=incomplete/int).
        nrounds: Number of boosting rounds (default CP_NROUNDS; tune with LOSO CV).

    Returns:
        Trained XGBoost Booster.
    """
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(C.CP_PARAMS, dtrain, num_boost_round=nrounds)


def train_cp(df: pl.DataFrame, approach: str = "A",
             nrounds: int = C.CP_NROUNDS) -> xgb.Booster:
    """Train CP model from a final.json plays DataFrame.

    Args:
        df: polars DataFrame from build_cp_training_frame (unfiltered or pre-filtered).
        approach: "A" or "B" (see constants).
        nrounds: Boosting rounds.

    Returns:
        Trained XGBoost Booster.
    """
    df = filter_pass_plays(df)
    X, y, _ = cp_matrix(df, approach=approach)
    return train_cp_from_matrix(X, y, nrounds=nrounds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_train_cp.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/train_cp.py tests/cpoe/test_train_cp.py
git commit -m "feat(cpoe): CP model trainer (binary:logistic, 8-feat, nrounds tunable)"
```

### Task 2.3: Ingest — build CP training frame from final.json

**Files:**
- Create: `python/cpoe/ingest.py`
- Test: `tests/cpoe/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_ingest.py
import pathlib
import polars as pl
import pytest
from cpoe.ingest import build_cp_training_frame
from cpoe.features import filter_pass_plays
from cpoe import constants as C

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_build_cp_training_frame():
    df = build_cp_training_frame(FINAL_DIR)
    assert df.height > 0
    # All pass-play rows must have the feature columns
    pass_df = filter_pass_plays(df)
    for feat_col in C.CP_FEATURE_SOURCE.values():
        assert feat_col in pass_df.columns, f"missing {feat_col}"
    # Completion must be 0 or 1
    assert pass_df["completion"].is_in([0, 1]).all()


def test_build_cp_frame_is_pass_attempts():
    """build_cp_training_frame must retain ALL play types (filtering is done in features.py)."""
    import json, pathlib
    f = list(FINAL_DIR.glob("*.json"))
    if not f:
        pytest.skip("no backfill final.json")
    with open(f[0]) as fh:
        obj = json.load(fh)
    plays = obj.get("plays", [])
    n_pass = sum(1 for p in plays if p.get("pass_attempt"))
    # the frame must contain pass_attempt column
    df = build_cp_training_frame(FINAL_DIR)
    assert "pass_attempt" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_ingest.py -v`
Expected: FAIL (`No module named 'cpoe.ingest'`)

- [ ] **Step 3: Implement `ingest.py`**

```python
# python/cpoe/ingest.py
"""Read final.json plays for CP model training.

The CP training frame is simpler than the EP/WP frame in model_training/ingest.py:
no labeling (outcome = completion flag already on the play), no NSH logic.
We just read all plays, keep pass-relevant columns, and let features.py filter.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def build_cp_training_frame(final_dir, seasons=None) -> pl.DataFrame:
    """Read final.json pass plays for CP model training.

    Args:
        final_dir: Path to cfb/json/final/ directory.
        seasons: Optional list of seasons to include; None = all.

    Returns:
        polars DataFrame with all plays (not yet filtered to pass_attempt only;
        filtering happens in features.filter_pass_plays).
    """
    frames = []
    for f in sorted(Path(final_dir).glob("*.json")):
        with open(f) as fh:
            obj = json.load(fh)
        if seasons is not None and obj.get("season") not in seasons:
            continue
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_ingest.py -v`
Expected: PASS (or SKIP without backfill data)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/ingest.py tests/cpoe/test_ingest.py
git commit -m "feat(cpoe): ingest final.json plays for CP training frame"
```

---

## PHASE 3 — LOSO CV + Hyperparameter Tuning

### Task 3.1: LOSO CV over seasons (Approach A)

**Files:**
- Create: `python/cpoe/loso.py`
- Test: `tests/cpoe/test_loso.py`

The LOSO CV loop also tunes `nrounds` via early stopping on the held-out season's logloss.

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_loso.py
import numpy as np
import pandas as pd
import polars as pl
from cpoe.loso import loso_cv


def _synth_frame(n=300):
    rng = np.random.default_rng(5)
    return pl.DataFrame({
        "down":           rng.integers(1, 5, n).tolist(),
        "distance":       rng.integers(1, 20, n).tolist(),
        "yards_to_goal":  rng.integers(1, 100, n).tolist(),
        "pos_score_diff": rng.integers(-28, 28, n).tolist(),
        "secs_remaining": rng.integers(0, 3600, n).tolist(),
        "is_home":        rng.integers(0, 2, n).tolist(),
        "period":         rng.integers(1, 5, n).tolist(),
        "passing_down":   rng.integers(0, 2, n).tolist(),
        "completion":     rng.integers(0, 2, n).tolist(),
        "pass_attempt":   [True] * n,
        "sack_vec":       [False] * n,
        "penalty_no_play": [False] * n,
        "game_id":        rng.integers(100, 200, n).tolist(),
        "passer_player_name": ["QB A"] * n,
        "season":         rng.choice([2021, 2022, 2023], n).tolist(),
    })


def test_loso_cv_returns_predictions_for_each_season():
    cv_out = loso_cv(_synth_frame(), nrounds=5)
    # cv_out must have predicted CP and actual completion
    assert "cp" in cv_out.columns
    assert "completion" in cv_out.columns
    assert "season" in cv_out.columns
    # preds in [0,1]
    assert float(cv_out["cp"].min()) >= 0.0
    assert float(cv_out["cp"].max()) <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_loso.py -v`
Expected: FAIL (`No module named 'cpoe.loso'`)

- [ ] **Step 3: Implement `loso.py`**

```python
# python/cpoe/loso.py
"""Leave-one-season-out CV for the CP model.

For each season held out:
  - Train on all other seasons.
  - Predict on the held-out season.
  - Collect preds + labels into a long-form DataFrame.

Also used to tune nrounds: the CV run with the best aggregate logloss is stored.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import cp_matrix, filter_pass_plays
from .train_cp import train_cp_from_matrix


def loso_cv(df: pl.DataFrame, approach: str = "A",
            nrounds: int = C.CP_NROUNDS) -> pl.DataFrame:
    """Run LOSO CV; return long-form DataFrame with cp (predicted) + completion + season.

    Args:
        df: Full plays DataFrame (unfiltered; filtering done here).
        approach: "A" or "B".
        nrounds: Boosting rounds per fold.

    Returns:
        polars DataFrame with columns: season, cp, completion, + all CP_FEATURE_SOURCE values.
    """
    df = filter_pass_plays(df)
    seasons = sorted(df["season"].unique().to_list())
    folds = []
    for s in seasons:
        train = df.filter(pl.col("season") != s)
        test  = df.filter(pl.col("season") == s)
        if train.is_empty() or test.is_empty():
            continue
        X_tr, y_tr, _ = cp_matrix(train, approach=approach)
        X_te, y_te, keys = cp_matrix(test, approach=approach)
        model = train_cp_from_matrix(X_tr, y_tr, nrounds=nrounds)
        preds = model.predict(xgb.DMatrix(X_te))
        fold = test.with_columns(pl.Series("cp", preds.tolist()))
        folds.append(fold)
    if not folds:
        return pl.DataFrame()
    return pl.concat(folds, how="diagonal_relaxed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_loso.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/loso.py tests/cpoe/test_loso.py
git commit -m "feat(cpoe): LOSO CV loop (leave-one-season-out, returns per-season predictions)"
```

### Task 3.2: (CONDITIONAL) Approach B — CFBD air-yards feature

> **Gate:** Implement ONLY if Phase 0 Task 0.2 confirmed CFBD air_yards fill rate ≥ 60%.
> If not, mark this task SKIPPED and note the verdict.

**Files:**
- Modify: `python/cpoe/features.py` (add CFBD air_yards join path)
- Modify: `python/cpoe/ingest.py` (add optional CFBD PBP join)
- Test: `tests/cpoe/test_features_b.py`

- [ ] **Step 1: Confirm Phase 0 Task 0.2 verdict before starting**

If `FEASIBILITY.md` says Approach B is INFEASIBLE, mark this task SKIPPED.

- [ ] **Step 2: Implement CFBD air_yards join in `ingest.py`**

Add a `load_cfbd_air_yards(seasons) -> pl.DataFrame` function that fetches CFBD PBP data with
air_yards for the specified seasons. Store it at `cfb/cfbd/cfbd_pbp_air_yards.parquet` in the
backfill.

- [ ] **Step 3: Extend `features.py` to join air_yards**

```python
def cp_matrix_b(df: pl.DataFrame, cfbd_air: pl.DataFrame):
    """Build Approach B CP matrix (9-feat) with CFBD air_yards joined."""
    # join on (game_id, passer_player_name, play_text_fuzzy)
    # fill nulls with median(air_yards) by distance bucket
    ...
```

- [ ] **Step 4: Run tests and commit if passing**

```bash
git commit -m "feat(cpoe): Approach B — CFBD air_yards as 9th feature (conditional on Phase 0)"
```

---

## PHASE 4 — `cpoe.py` (CPOE computation + aggregation)

### Task 4.1: Per-play CPOE

**Files:**
- Create: `python/cpoe/cpoe.py`
- Test: `tests/cpoe/test_cpoe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_cpoe.py
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from cpoe.cpoe import compute_cpoe, aggregate_cpoe
from cpoe import constants as C


def _mock_model(n_feats=8, preds=0.55):
    """Return a mock Booster-like callable for testing."""
    class _Mock:
        def predict(self, dm):
            return np.full(dm.num_row(), preds)
    return _Mock()


def _pass_df(n=20):
    rng = np.random.default_rng(6)
    rows = {src: rng.random(n).tolist() for src in C.CP_FEATURE_SOURCE.values()}
    rows["pass_attempt"] = [True] * n
    rows["sack_vec"] = [False] * n
    rows["penalty_no_play"] = [False] * n
    rows["completion"] = rng.integers(0, 2, n).tolist()
    rows["game_id"] = [1] * n
    rows["season"] = [2024] * n
    rows["passer_player_name"] = ["QB A"] * (n // 2) + ["QB B"] * (n // 2)
    return pl.DataFrame(rows)


def test_compute_cpoe_adds_columns():
    df = compute_cpoe(_pass_df(), model=_mock_model())
    assert "expected_completion" in df.columns
    assert "cpoe" in df.columns


def test_cpoe_formula():
    # CPOE = actual - expected; mock model always predicts 0.55
    df = _pass_df()
    out = compute_cpoe(df, model=_mock_model(preds=0.55))
    expected_cpoes = [c - 0.55 for c in out["completion"].to_list()]
    actual_cpoes = out["cpoe"].to_list()
    assert all(abs(a - e) < 1e-5 for a, e in zip(actual_cpoes, expected_cpoes))


def test_aggregate_cpoe_per_qb():
    df = _pass_df()
    out = compute_cpoe(df, model=_mock_model())
    agg = aggregate_cpoe(out, by=["season", "passer_player_name"])
    assert "cpoe" in agg.columns
    assert "attempts" in agg.columns
    assert agg.height == 2  # QB A and QB B
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_cpoe.py -v`
Expected: FAIL (`No module named 'cpoe.cpoe'`)

- [ ] **Step 3: Implement `cpoe.py`**

```python
# python/cpoe/cpoe.py
"""Compute per-play expected completion and CPOE from a trained CP model.

cpoe = completion - expected_completion (positive = better than expected).
Aggregate to QB-game or QB-season via aggregate_cpoe().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import xgboost as xgb

from .features import cp_matrix, filter_pass_plays
from . import constants as C

if TYPE_CHECKING:
    from xgboost import Booster


def compute_cpoe(df: pl.DataFrame, model: "Booster",
                 approach: str = "A") -> pl.DataFrame:
    """Add expected_completion and cpoe columns to a pass-plays DataFrame.

    Args:
        df: Pass plays from final.json (filtered or unfiltered).
        model: Trained CP XGBoost Booster.
        approach: "A" or "B".

    Returns:
        df with added columns: expected_completion (float), cpoe (float).
        Rows that fail the pass-play filter are returned with nulls in both columns.
    """
    pass_mask = (
        (pl.col("pass_attempt") == True)  # noqa: E712
        & (pl.col("sack_vec") == False)   # noqa: E712
        & (pl.col("penalty_no_play") == False)  # noqa: E712
    )
    pass_df = df.filter(pass_mask)
    X, _, _ = cp_matrix(pass_df, approach=approach)
    preds = model.predict(xgb.DMatrix(X))
    pass_df = pass_df.with_columns(
        expected_completion=pl.Series(preds.tolist()),
        cpoe=(pl.col("completion").cast(pl.Float32) - pl.Series(preds.tolist())),
    )
    # left-join back to keep all rows (non-pass rows get nulls)
    return df.join(
        pass_df.select(["game_id", "id", "expected_completion", "cpoe"]),
        on=["game_id", "id"],
        how="left",
    ) if "id" in df.columns else pass_df


def aggregate_cpoe(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Aggregate per-play CPOE to QB-game or QB-season.

    Args:
        df: Output of compute_cpoe (has expected_completion, cpoe columns).
        by: Group-by columns, e.g. ["season", "passer_player_name"].

    Returns:
        Aggregated DataFrame with: cpoe (mean), attempts, completions,
        actual_completion_pct, expected_completion_pct.
    """
    return (
        df.filter(pl.col("cpoe").is_not_null())
        .group_by(by)
        .agg(
            attempts=pl.len(),
            completions=pl.col("completion").cast(pl.Int32).sum(),
            cpoe=pl.col("cpoe").mean(),
            actual_completion_pct=pl.col("completion").cast(pl.Float32).mean(),
            expected_completion_pct=pl.col("expected_completion").mean(),
        )
        .sort(by)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_cpoe.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/cpoe.py tests/cpoe/test_cpoe.py
git commit -m "feat(cpoe): per-play expected_completion + CPOE + QB aggregation"
```

---

## PHASE 5 — `validate.py` + `figures.py`

### Task 5.1: Calibration validation

**Files:**
- Create: `python/cpoe/validate.py`
- Test: `tests/cpoe/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_validate.py
import polars as pl
from cpoe.validate import calibration_table, distance_bucket


def _cv_frame(n=200):
    import numpy as np
    rng = np.random.default_rng(7)
    return pl.DataFrame({
        "cp": rng.uniform(0, 1, n).tolist(),
        "completion": rng.integers(0, 2, n).tolist(),
        "distance": rng.integers(1, 20, n).tolist(),
        "season": rng.choice([2021, 2022], n).tolist(),
    })


def test_calibration_table_has_expected_columns():
    df = _cv_frame()
    df = df.with_columns(distance_bucket=distance_bucket(pl.col("distance")))
    tbl = calibration_table(df)
    for col in ["distance_bucket", "bin_pred_prob", "n_plays", "n_complete", "bin_actual_prob"]:
        assert col in tbl.columns


def test_distance_bucket_assigns_correctly():
    df = pl.DataFrame({"distance": [1, 3, 4, 8, 9, 15]})
    buckets = df.with_columns(bucket=distance_bucket(pl.col("distance")))["bucket"].to_list()
    assert buckets == ["Short", "Short", "Intermediate", "Intermediate", "Long", "Long"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_validate.py -v`
Expected: FAIL (`No module named 'cpoe.validate'`)

- [ ] **Step 3: Implement `validate.py`**

```python
# python/cpoe/validate.py
"""Calibration tables for the CP model LOSO CV output.

Stratified by distance bucket (yards-to-first-down proxy for air yards depth).
Note: distance_bucket is a coarse proxy — document on all figures.
"""
from __future__ import annotations

import polars as pl

from . import constants as C


def distance_bucket(col: pl.Expr) -> pl.Expr:
    """Classify start.distance into Short / Intermediate / Long buckets."""
    short_hi = C.DISTANCE_BUCKETS["Short"][1]
    mid_hi = C.DISTANCE_BUCKETS["Intermediate"][1]
    return (
        pl.when(col <= short_hi).then(pl.lit("Short"))
        .when(col <= mid_hi).then(pl.lit("Intermediate"))
        .otherwise(pl.lit("Long"))
    )


def calibration_table(cv_df: pl.DataFrame, bin_size: float = 0.05) -> pl.DataFrame:
    """Compute binned calibration stats from LOSO CV output.

    Args:
        cv_df: LOSO CV output with cp (predicted), completion (actual), distance_bucket.
        bin_size: Bin width for predicted probability.

    Returns:
        polars DataFrame with per-(distance_bucket, bin) calibration stats.
    """
    return (
        cv_df.with_columns(
            bin_pred_prob=(pl.col("cp") / bin_size).round() * bin_size,
        )
        .group_by(["distance_bucket", "bin_pred_prob"])
        .agg(
            n_plays=pl.len(),
            n_complete=pl.col("completion").cast(pl.Int32).sum(),
        )
        .with_columns(
            bin_actual_prob=pl.col("n_complete") / pl.col("n_plays"),
        )
        .filter(pl.col("n_plays") > 10)
        .sort(["distance_bucket", "bin_pred_prob"])
    )


def weighted_cal_error(tbl: pl.DataFrame) -> dict[str, float]:
    """Compute per-bucket and overall weighted calibration error."""
    tbl = tbl.with_columns(
        cal_diff=(pl.col("bin_pred_prob") - pl.col("bin_actual_prob")).abs()
    )
    per = (
        tbl.group_by("distance_bucket")
        .agg(
            wce=(pl.col("cal_diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
            n=pl.col("n_complete").sum(),
        )
    )
    overall = float(
        (per["wce"] * per["n"]).sum() / per["n"].sum()
    )
    return {"per_bucket": per.to_dicts(), "overall": overall}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_validate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/validate.py tests/cpoe/test_validate.py
git commit -m "feat(cpoe): calibration table + distance bucket + weighted cal error"
```

### Task 5.2: Calibration figures

**Files:**
- Create: `python/cpoe/figures.py`
- Test: `tests/cpoe/test_figures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_figures.py
import polars as pl
from cpoe.figures import write_cp_calibration


def test_write_cp_calibration_emits_png_and_csv(tmp_path):
    tbl = pl.DataFrame({
        "distance_bucket": ["Short"] * 5 + ["Long"] * 5,
        "bin_pred_prob":   [0.3, 0.4, 0.5, 0.6, 0.7] * 2,
        "n_plays":         [50, 100, 200, 100, 50] * 2,
        "bin_actual_prob": [0.28, 0.42, 0.51, 0.62, 0.72] * 2,
    })
    png, csv = write_cp_calibration(tbl, tmp_path / "cp_cal", cal_error=0.015)
    assert png.exists() and csv.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_figures.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `figures.py`**

```python
# python/cpoe/figures.py
"""plotnine calibration plots for the CFB CP model (bespoke cfbfastR styling).

One facet per distance bucket (Short / Intermediate / Long).
Caption always notes that distance_bucket approximates throw depth via yards-to-first-down,
NOT via actual air yards.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from plotnine import (aes, facet_wrap, geom_abline, geom_point, geom_smooth,
                      geom_text, ggplot, labs, theme, theme_bw,
                      element_text, element_rect, coord_equal, scale_x_continuous,
                      scale_y_continuous)

GARNET = "#500f1b"
FONT = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]

_ANN = {"x": [0.25, 0.75], "y": [0.75, 0.25],
        "lab": ["More times\nthan expected", "Fewer times\nthan expected"]}


def write_cp_calibration(tbl: pl.DataFrame, stem, cal_error: float,
                         title: str = "CFB Completion Probability — LOSO Calibration",
                         subtitle: str = "Approach A (game-state model; distance = yards-to-first-down proxy)"):
    """Write a calibration PNG + CSV + parquet for the CP model.

    Args:
        tbl: Calibration table (distance_bucket, bin_pred_prob, n_plays, bin_actual_prob).
        stem: Path stem (no extension).
        cal_error: Overall weighted calibration error (for caption).
        title: Plot title.
        subtitle: Plot subtitle.

    Returns:
        (png_path, csv_path)
    """
    import pandas as pd
    stem = Path(stem)
    csv = stem.with_suffix(".csv")
    png = stem.with_suffix(".png")
    stem.parent.mkdir(parents=True, exist_ok=True)
    tbl.write_csv(csv)
    tbl.write_parquet(stem.with_suffix(".parquet"))

    pdf = tbl.to_pandas()
    ann = pd.DataFrame(_ANN)

    p = (
        ggplot(pdf, aes("bin_pred_prob", "bin_actual_prob"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET)
        + geom_smooth(method="loess", se=False, color=GARNET, size=0.5)
        + geom_text(data=ann, mapping=aes(x="x", y="y", label="lab"), size=8, inherit_aes=False)
        + facet_wrap("~distance_bucket", ncol=3)
        + coord_equal()
        + scale_x_continuous(limits=(0, 1))
        + scale_y_continuous(limits=(0, 1))
        + labs(
            title=title,
            subtitle=subtitle,
            caption=(
                f"Overall Weighted Calibration Error: {round(cal_error, 4)}\n"
                "Note: distance bucket approximates throw depth via yards-to-first-down, "
                "not actual air yards (unavailable in ESPN CFB pbp)."
            ),
            x="Estimated completion percentage",
            y="Observed completion percentage",
            size="Number of plays",
        )
        + theme_bw()
        + theme(
            text=element_text(family=FONT),
            plot_background=element_rect(fill="grey99", color="black"),
            panel_background=element_rect(fill="grey95"),
            legend_position="bottom",
        )
    )
    p.save(png, width=9, height=4, dpi=200, verbose=False)
    return png, csv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_figures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/figures.py tests/cpoe/test_figures.py
git commit -m "feat(cpoe): calibration figures + bespoke cfbfastR styling (distance-bucket facets)"
```

---

## PHASE 6 — `cli.py` + smoke tests

### Task 6.1: CLI subcommands

**Files:**
- Create: `python/cpoe/cli.py`
- Test: `tests/cpoe/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/cpoe/test_cli.py
from cpoe.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    choices = set(p._subparsers._group_actions[0].choices.keys())
    assert {"ingest", "train", "loso", "predict", "validate", "figures"} <= choices
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cpoe/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `cli.py`**

```python
# python/cpoe/cli.py
"""CLI: ingest | train | loso | predict | validate | figures."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cpoe")
    ap.add_argument("--approach", choices=["A", "B"], default="A")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("--final-dir", default="cfb/json/final")
    i.add_argument("--out", default="cfb/cpoe/cp_plays.parquet")
    i.add_argument("--seasons", nargs="*", type=int)

    t = sub.add_parser("train")
    t.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet")
    t.add_argument("--out", default="cfb/cpoe/cp_model.ubj")
    t.add_argument("--nrounds", type=int, default=400)

    lo = sub.add_parser("loso")
    lo.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet")
    lo.add_argument("--out", default="cfb/cpoe/loso_cv.parquet")
    lo.add_argument("--nrounds", type=int, default=400)

    pr = sub.add_parser("predict")
    pr.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet")
    pr.add_argument("--model", default="cfb/cpoe/cp_model.ubj")
    pr.add_argument("--out", default="cfb/cpoe/cpoe_plays.parquet")

    v = sub.add_parser("validate")
    v.add_argument("--loso", default="cfb/cpoe/loso_cv.parquet")
    v.add_argument("--out", default="cfb/cpoe/calibration.parquet")

    f = sub.add_parser("figures")
    f.add_argument("--calibration", default="cfb/cpoe/calibration.parquet")
    f.add_argument("--out-dir", default="cfb/cpoe/figures/")

    return ap


def main(argv=None) -> int:
    import polars as pl
    import xgboost as xgb
    from .ingest import build_cp_training_frame
    from .features import filter_pass_plays
    from .train_cp import train_cp
    from .loso import loso_cv
    from .cpoe import compute_cpoe, aggregate_cpoe
    from .validate import calibration_table, distance_bucket, weighted_cal_error
    from .figures import write_cp_calibration

    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        df = build_cp_training_frame(args.final_dir, args.seasons)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.out)
        print(f"wrote {df.height} plays -> {args.out}")

    elif args.cmd == "train":
        df = pl.read_parquet(args.plays)
        model = train_cp(df, approach=args.approach, nrounds=args.nrounds)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(args.out)
        print(f"saved model -> {args.out}")

    elif args.cmd == "loso":
        df = pl.read_parquet(args.plays)
        cv = loso_cv(df, approach=args.approach, nrounds=args.nrounds)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv.write_parquet(args.out)
        print(f"LOSO CV done -> {args.out}")

    elif args.cmd == "predict":
        df = pl.read_parquet(args.plays)
        model = xgb.Booster()
        model.load_model(args.model)
        out = compute_cpoe(df, model, approach=args.approach)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(args.out)
        print(f"wrote CPOE -> {args.out}")

    elif args.cmd == "validate":
        cv = pl.read_parquet(args.loso)
        cv = cv.with_columns(distance_bucket=distance_bucket(pl.col("distance")))
        tbl = calibration_table(cv)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tbl.write_parquet(args.out)
        err = weighted_cal_error(tbl)
        print(f"Overall calibration error: {err['overall']:.4f}")

    elif args.cmd == "figures":
        tbl = pl.read_parquet(args.calibration)
        err = tbl.with_columns(
            cal_diff=(pl.col("bin_pred_prob") - pl.col("bin_actual_prob")).abs()
        )
        overall = float(
            (err["cal_diff"] * err["n_plays"]).sum() / err["n_plays"].sum()
        )
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        write_cp_calibration(tbl, Path(args.out_dir) / "cp_calibration", cal_error=overall)
        print(f"figures written -> {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cpoe/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/cpoe/cli.py tests/cpoe/test_cli.py
git commit -m "feat(cpoe): CLI subcommands (ingest | train | loso | predict | validate | figures)"
```

### Task 6.2: End-to-end smoke test (conditional on backfill data)

**Files:**
- Test: `tests/cpoe/test_cli_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/cpoe/test_cli_smoke.py
import pathlib
import pytest
from cpoe.cli import main

FINAL = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL.glob("*.json")), reason="no backfill final.json")
def test_ingest_then_train(tmp_path):
    plays_out = tmp_path / "cp_plays.parquet"
    model_out = tmp_path / "cp_model.ubj"
    assert main(["ingest", "--final-dir", str(FINAL), "--out", str(plays_out)]) == 0
    assert plays_out.exists()
    assert main(["train", "--plays", str(plays_out), "--out", str(model_out), "--nrounds", "5"]) == 0
    assert model_out.exists()


@pytest.mark.skipif(not any(FINAL.glob("*.json")), reason="no backfill final.json")
def test_loso_produces_parquet(tmp_path):
    plays_out = tmp_path / "cp_plays.parquet"
    loso_out = tmp_path / "loso_cv.parquet"
    main(["ingest", "--final-dir", str(FINAL), "--out", str(plays_out)])
    assert main(["loso", "--plays", str(plays_out), "--out", str(loso_out), "--nrounds", "5"]) == 0
    assert loso_out.exists()
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/cpoe/test_cli_smoke.py -v`
Expected: PASS (or SKIP without backfill data — run `uv run python python/scrape_cfb_json.py -s 2024 -e 2024` + `reprocess_cfb_json.py` first)

- [ ] **Step 3: Commit**

```bash
git add tests/cpoe/test_cli_smoke.py
git commit -m "test(cpoe): end-to-end smoke (ingest -> train -> loso)"
```

---

## PHASE 7 — Full training run + LOSO calibration artifact

> This phase runs on the full backfill history and produces the final `cp_model.ubj` + LOSO
> calibration plots. It is a manual run step, not a CI step.

### Task 7.1: Full-history training run

- [ ] **Step 1: Run ingest on full history**

```bash
uv run python -m cpoe ingest \
  --final-dir cfb/json/final \
  --out cfb/cpoe/cp_plays.parquet
```

Expected: several hundred thousand pass plays loaded.

- [ ] **Step 2: Run LOSO CV to tune nrounds**

```bash
uv run python -m cpoe loso \
  --plays cfb/cpoe/cp_plays.parquet \
  --out cfb/cpoe/loso_cv.parquet \
  --nrounds 400
```

Inspect `loso_cv.parquet`: if overall calibration error is acceptable (< 0.03), proceed.
Otherwise adjust `nrounds` and repeat. Document the chosen `nrounds` in `FEASIBILITY.md`.

- [ ] **Step 3: Produce calibration artifacts**

```bash
uv run python -m cpoe validate --loso cfb/cpoe/loso_cv.parquet --out cfb/cpoe/calibration.parquet
uv run python -m cpoe figures --calibration cfb/cpoe/calibration.parquet --out-dir cfb/cpoe/figures/
```

- [ ] **Step 4: Train final model on full history**

```bash
uv run python -m cpoe train \
  --plays cfb/cpoe/cp_plays.parquet \
  --out cfb/cpoe/cp_model.ubj \
  --nrounds <tuned_value>
```

- [ ] **Step 5: Commit the artifacts**

```bash
git add cfb/cpoe/cp_model.ubj cfb/cpoe/calibration.parquet cfb/cpoe/figures/
git commit -m "feat(cpoe): full-history CP model (Approach A, <N> rounds) + LOSO calibration artifacts"
```

---

## Stage gating summary

| Phase | Condition | Status when gated |
|---|---|---|
| Phase 0 Task 0.1 | Always | (feasibility documented) |
| Phase 0 Task 0.2 | Always (investigation) | Determines Approach B eligibility |
| Phase 3 Task 3.2 | Approach B only (fill rate ≥ 60%) | SKIP if infeasible |
| Phases 1–2, 4–7 | Always (Approach A) | Not gated |

Run the full test suite after each phase:

```bash
uv run pytest tests/cpoe/ -v
```

All unit/integration tests should pass. Data-dependent tests SKIP without backfill data.
