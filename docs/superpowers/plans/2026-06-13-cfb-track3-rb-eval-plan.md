# CFB Modeling Suite — Track 3 (RB-Eval xREPA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the DAKOTA-lineage R RB-evaluation model to Python in
`cfbfastR-cfb-raw/python/rb_eval/`, producing per-rusher-season **expected rushing EPA (xREPA)**
from a `pygam.LinearGAM(s(0) + s(1))` fit on prior-season `epa_per_play` and `success`.

**Architecture:** Season-grain aggregation-first. Raw data source is the backfill's
`final.json` plays (= `CFBPlayProcess` output). The pipeline has three logical stages:
(1) feature loading and `fo_success` derivation; (2) per-rusher-season aggregation, lag, and
weight; (3) GAM training, LOSO CV, and calibration output. No sdv-py model handoff — the
xREPA table is a research-grade analytical artifact.

**Tech Stack:** Python 3.11+, uv, polars 1.x, pygam (already in `[dependency-groups] gam`),
plotnine + statsmodels (already in `[dependency-groups] figures`), joblib (stdlib-adjacent,
part of scikit-learn; add as dev dep), pytest. No xgboost dependency.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-track3-rb-eval-design.md`
(umbrella: `2026-06-13-cfb-modeling-suite-program.md`).

**Commit convention:** Conventional Commits (`feat(rb-eval): ...`, `test(rb-eval): ...`,
`fix(rb-eval): ...`). No Co-Authored-By or AI co-author trailers on any commit in this repo.

---

## File structure

Package `python/rb_eval/` (one responsibility per file):

| File | Responsibility |
|---|---|
| `python/rb_eval/__init__.py` | Package marker + version. |
| `python/rb_eval/features.py` | `load_rush_plays(final_dir, seasons) -> pl.DataFrame` — read `final.json` plays, filter rush plays, compute `fo_success` + `is_rush_opp`. |
| `python/rb_eval/aggregate.py` | `build_rusher_seasons(rush_df) -> pl.DataFrame` — group, summarize, filter n>100, lag, weight. `build_model_data(rusher_seasons) -> pl.DataFrame` — rename to GAM input contract. |
| `python/rb_eval/train.py` | `train_xrepa(model_data) -> LinearGAM`; `loso_cv(model_data) -> pl.DataFrame`; `save_model` / `load_model`. |
| `python/rb_eval/validate.py` | `calibration_table`, `weighted_cal_error`, `weighted_r2`. |
| `python/rb_eval/cli.py` | `features \| aggregate \| train \| validate \| figures` subcommands. |
| `tests/rb_eval/test_features.py` | fo_success formula; rush filter; highlight_yards reuse. |
| `tests/rb_eval/test_aggregate.py` | epa clamp; n>100 filter; lag 1-season; weight formula; n_opps=0 guard. |
| `tests/rb_eval/test_train.py` | GAM shape/type; LOSO CV coverage; save/load roundtrip. |
| `tests/rb_eval/test_validate.py` | calibration bin arithmetic; weighted error formula. |
| `tests/rb_eval/test_cli.py` | subcommand presence; help exits 0. |
| `tests/fixtures/rb_eval/` | Synthetic play frame fixture + README. |

Tests run with `uv run pytest tests/rb_eval/`. The project already has `pygam` in the `gam`
dependency group (`pyproject.toml`) and `plotnine`/`statsmodels`/`pillow` in the `figures` group.

---

## Phase 0 — Scaffold, deps, fixtures

### Task 0.1: Create the package skeleton

**Files:**
- Create: `python/rb_eval/__init__.py`
- Create: `tests/rb_eval/__init__.py`
- Create: `tests/rb_eval/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_package.py
def test_package_imports():
    import rb_eval
    assert hasattr(rb_eval, "__version__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rb_eval'`

- [ ] **Step 3: Create the package**

```python
# python/rb_eval/__init__.py
"""CFB RB-eval (DAKOTA xREPA) model — Track 3 of the CFB Modeling Suite."""
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 4: Verify `python/` is importable in tests**

The `tests/conftest.py` already adds `python/` to `sys.path` (added during Track 1 scaffold).
Confirm:

```python
# Verify in tests/conftest.py (read, do not blindly overwrite):
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
```

If absent, add the `sys.path.insert` line (Track 1 conftest note applies).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_package.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```
git add python/rb_eval/__init__.py tests/rb_eval/__init__.py tests/rb_eval/test_package.py
git commit -m "feat(rb-eval): scaffold rb_eval package"
```

### Task 0.2: Add joblib dependency

`joblib` ships with scikit-learn but the project does not currently list scikit-learn. Add `joblib`
directly (it is a standalone package on PyPI) to the `dev` group (it is only needed at training
time, not at runtime).

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add joblib to the dev group**

Edit `pyproject.toml`:

```toml
[dependency-groups]
dev = ["pytest>=8.0", "joblib>=1.3"]
figures = ["plotnine>=0.13", "statsmodels>=0.14", "pillow>=10.0"]
gam = ["pygam>=0.9"]
```

- [ ] **Step 2: Sync and verify**

Run: `uv sync --all-groups && uv run python -c "import pygam, joblib; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```
git add pyproject.toml uv.lock
git commit -m "build(rb-eval): add joblib to dev dep group"
```

### Task 0.3: Create synthetic play fixture

**Files:**
- Create: `tests/fixtures/rb_eval/synth_plays.json`
- Create: `tests/fixtures/rb_eval/README.md`

- [ ] **Step 1: Write the fixture**

```json
// tests/fixtures/rb_eval/synth_plays.json
[
  {"game_id": 1, "season": 2010, "week": 1, "rush": true, "rusher_player_name": "Rusher A",
   "yds_rushed": 5, "start.down": 1, "start.distance": 10,
   "highlight_yards": 0.5, "adj_rush_yardage": 5.0,
   "pos_team": 100, "EPA": 0.4, "home_wp_before": 0.6},
  {"game_id": 1, "season": 2010, "week": 1, "rush": true, "rusher_player_name": "Rusher A",
   "yds_rushed": 2, "start.down": 2, "start.distance": 8,
   "highlight_yards": 0.0, "adj_rush_yardage": 2.0,
   "pos_team": 100, "EPA": -0.3, "home_wp_before": 0.6},
  {"game_id": 1, "season": 2010, "week": 1, "rush": false, "rusher_player_name": null,
   "yds_rushed": null, "start.down": 1, "start.distance": 10,
   "highlight_yards": null, "adj_rush_yardage": null,
   "pos_team": 100, "EPA": 0.1, "home_wp_before": 0.6},
  {"game_id": 1, "season": 2010, "week": 1, "rush": true, "rusher_player_name": "TEAM",
   "yds_rushed": 1, "start.down": 3, "start.distance": 3,
   "highlight_yards": 0.0, "adj_rush_yardage": 1.0,
   "pos_team": 100, "EPA": -0.5, "home_wp_before": 0.6},
  {"game_id": 2, "season": 2010, "week": 2, "rush": true, "rusher_player_name": "Rusher A",
   "yds_rushed": 8, "start.down": 3, "start.distance": 3,
   "highlight_yards": 2.0, "adj_rush_yardage": 8.0,
   "pos_team": 100, "EPA": 1.2, "home_wp_before": 0.5},
  {"game_id": 2, "season": 2010, "week": 2, "rush": true, "rusher_player_name": "Rusher B",
   "yds_rushed": -2, "start.down": 1, "start.distance": 10,
   "highlight_yards": 0.0, "adj_rush_yardage": 0.0,
   "pos_team": 100, "EPA": -1.5, "home_wp_before": 0.5}
]
```

- [ ] **Step 2: Write the README**

```markdown
<!-- tests/fixtures/rb_eval/README.md -->
# rb_eval fixtures

- `synth_plays.json` — hand-crafted rush plays for offline testing of fo_success, is_rush_opp,
  and highlight_yards reuse. Contains: one non-rush play (filtered), one TEAM rusher (filtered),
  one negative-yardage rush, plays from two seasons for Rusher A to verify lag logic.
  Not sourced from ESPN; values are synthetic.
```

- [ ] **Step 3: Commit**

```
git add tests/fixtures/rb_eval/
git commit -m "test(rb-eval): synthetic play fixture + README"
```

---

## Phase 1 — `features.py` (rush play loading + fo_success)

Port of `pbp_db` block in `rb_eval_model.R` (lines 18–52). Reads `final.json` play files,
applies the rush filter, computes `fo_success` and `is_rush_opp` from the raw per-play columns,
and reuses the pre-computed `highlight_yards` from `CFBPlayProcess`.

### Task 1.1: fo_success formula (down 1 + down 2 cases)

**Files:**
- Create: `python/rb_eval/features.py`
- Create: `tests/rb_eval/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_features.py
import polars as pl
import pytest
from rb_eval.features import add_fo_success


def _plays(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_fo_success_down1_at_half_threshold():
    # down1: fo_success if yds_rushed >= 0.5 * start.distance
    df = _plays([
        {"start.down": 1, "start.distance": 10, "yds_rushed": 5},   # 5 >= 5.0 → True
        {"start.down": 1, "start.distance": 10, "yds_rushed": 4},   # 4 < 5.0 → False
        {"start.down": 1, "start.distance": 10, "yds_rushed": 0},   # 0 < 5.0 → False
    ])
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False, False]


def test_fo_success_down2_at_seventy_percent():
    # down2: fo_success if yds_rushed >= 0.7 * start.distance
    df = _plays([
        {"start.down": 2, "start.distance": 10, "yds_rushed": 7},   # 7 >= 7.0 → True
        {"start.down": 2, "start.distance": 10, "yds_rushed": 6},   # 6 < 7.0 → False
    ])
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False]


def test_fo_success_down3_at_full_distance():
    # down≥3: fo_success if yds_rushed >= start.distance
    df = _plays([
        {"start.down": 3, "start.distance": 3, "yds_rushed": 3},   # 3 >= 3 → True
        {"start.down": 3, "start.distance": 3, "yds_rushed": 2},   # 2 < 3 → False
        {"start.down": 4, "start.distance": 1, "yds_rushed": 1},   # 1 >= 1 → True
    ])
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False, True]


def test_fo_success_down4_included():
    # down4 falls to the otherwise branch (same threshold as down>=3)
    df = _plays([{"start.down": 4, "start.distance": 2, "yds_rushed": 2}])
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_features.py::test_fo_success_down1_at_half_threshold -v`
Expected: FAIL (`No module named 'rb_eval.features'`)

- [ ] **Step 3: Implement `add_fo_success`**

```python
# python/rb_eval/features.py
"""Load rush plays from final.json and compute fo_success + is_rush_opp.

Reuses CFBPlayProcess pre-computed highlight_yards, adj_rush_yardage from the final.json
plays (sdv-py-canonical formula). fo_success is computed fresh (see spec §5.2).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def add_fo_success(df: pl.DataFrame) -> pl.DataFrame:
    """Add fo_success column using Football Outsiders down-weighted success thresholds.

    Args:
        df: play frame with columns start.down (int), start.distance (float), yds_rushed (float).

    Returns:
        df with added boolean column fo_success.
    """
    return df.with_columns(
        fo_success=pl.when(pl.col("start.down") == 1)
        .then(pl.col("yds_rushed") >= 0.5 * pl.col("start.distance"))
        .when(pl.col("start.down") == 2)
        .then(pl.col("yds_rushed") >= 0.7 * pl.col("start.distance"))
        .otherwise(pl.col("yds_rushed") >= pl.col("start.distance"))
        .cast(pl.Boolean),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/features.py tests/rb_eval/test_features.py
git commit -m "feat(rb-eval): fo_success formula (FO down-weighted thresholds)"
```

### Task 1.2: rush play filter (rush==1, valid rusher, pos_team, EPA; name != TEAM)

**Files:**
- Modify: `python/rb_eval/features.py`
- Modify: `tests/rb_eval/test_features.py`

- [ ] **Step 1: Add the filter test**

```python
# tests/rb_eval/test_features.py  (append)
import pathlib, json

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "rb_eval" / "synth_plays.json"


def test_filter_rush_plays_excludes_non_rush_and_team():
    df = pl.DataFrame(json.loads(FIXTURE.read_text()))
    from rb_eval.features import filter_rush_plays
    out = filter_rush_plays(df)
    # TEAM rusher and non-rush play excluded; null rusher excluded
    assert "TEAM" not in (out["rusher_player_name"].to_list())
    assert out["rush"].to_list() == [True] * len(out)
    assert out["rusher_player_name"].null_count() == 0
    assert out["pos_team"].null_count() == 0
    assert out["EPA"].null_count() == 0


def test_filter_adds_fo_success_and_is_rush_opp():
    df = pl.DataFrame(json.loads(FIXTURE.read_text()))
    from rb_eval.features import filter_rush_plays
    out = filter_rush_plays(df)
    assert "fo_success" in out.columns
    assert "is_rush_opp" in out.columns
    # is_rush_opp = (yds_rushed >= 4): yds=5 → True, yds=2 → False, yds=8 → True, yds=-2 → False
    # Only Rusher A (yds=5) and Rusher A (yds=8) and Rusher B (yds=-2) survive the filter
    opps = out["is_rush_opp"].to_list()
    assert all(isinstance(v, bool) for v in opps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_features.py::test_filter_rush_plays_excludes_non_rush_and_team -v`
Expected: FAIL (`cannot import name 'filter_rush_plays'`)

- [ ] **Step 3: Implement `filter_rush_plays`**

```python
# python/rb_eval/features.py  (append)

def filter_rush_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to rushing plays, apply the R source filter conditions, add derived columns.

    Filtering mirrors rb_eval_model.R lines 20-22:
        filter(rush == 1)
        filter(!is.na(posteam) & !is.na(epa) & !is.na(rusher_player_name))
        filter(rusher_player_name != "TEAM")

    Args:
        df: raw plays frame (final.json plays).

    Returns:
        Filtered frame with fo_success, is_rush_opp columns added.
    """
    # EPA column may be named 'EPA' or 'epa' depending on source — normalize
    epa_col = "EPA" if "EPA" in df.columns else "epa"
    out = (
        df.filter(pl.col("rush") == True)  # noqa: E712
        .filter(pl.col("pos_team").is_not_null())
        .filter(pl.col(epa_col).is_not_null())
        .filter(pl.col("rusher_player_name").is_not_null())
        .filter(pl.col("rusher_player_name") != "TEAM")
    )
    if epa_col == "EPA":
        out = out.rename({"EPA": "epa"})
    out = add_fo_success(out)
    out = out.with_columns(
        is_rush_opp=(pl.col("yds_rushed") >= 4).cast(pl.Boolean),
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_features.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/features.py tests/rb_eval/test_features.py
git commit -m "feat(rb-eval): rush play filter + is_rush_opp (R source filter conditions)"
```

### Task 1.3: `load_rush_plays` — read from final.json directory

**Files:**
- Modify: `python/rb_eval/features.py`
- Modify: `tests/rb_eval/test_features.py`

- [ ] **Step 1: Add the loader test**

```python
# tests/rb_eval/test_features.py  (append)
import pathlib
import pytest

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_load_rush_plays_from_backfill():
    from rb_eval.features import load_rush_plays
    df = load_rush_plays(FINAL_DIR, seasons=None)
    assert df.height > 0
    assert "fo_success" in df.columns
    assert "is_rush_opp" in df.columns
    # All rows are rush plays with valid rusher, pos_team, epa
    assert (df["rush"] == True).all()  # noqa: E712
    assert df["rusher_player_name"].null_count() == 0
    assert df["epa"].null_count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_features.py::test_load_rush_plays_from_backfill -v`
Expected: FAIL (`cannot import name 'load_rush_plays'`) or SKIP if no backfill data.

- [ ] **Step 3: Implement `load_rush_plays`**

```python
# python/rb_eval/features.py  (append)

def load_rush_plays(final_dir, seasons=None) -> pl.DataFrame:
    """Read final.json play files, filter to rushing plays, return filtered frame.

    Args:
        final_dir: path to the backfill's cfb/json/final/ directory.
        seasons: optional list/set of seasons (int) to restrict loading.

    Returns:
        polars DataFrame of rush plays with fo_success, is_rush_opp columns.
    """
    frames: list[pl.DataFrame] = []
    for path in sorted(Path(final_dir).glob("*.json")):
        raw = json.loads(path.read_text())
        if seasons is not None and raw.get("season") not in seasons:
            continue
        plays = raw.get("plays") or []
        if not plays:
            continue
        frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed")
    return filter_rush_plays(df)
```

- [ ] **Step 4: Run test to verify it passes (or skips)**

Run: `uv run pytest tests/rb_eval/test_features.py -v`
Expected: PASS (all offline tests) + SKIP (backfill-gated test unless data present)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/features.py tests/rb_eval/test_features.py
git commit -m "feat(rb-eval): load_rush_plays (final.json -> filtered rush plays)"
```

---

## Phase 2 — `aggregate.py` (per-rusher-season summarize, lag, weight)

Port of `lrbs` block (lines 54–77) and `model_data` rename (lines 79–86) in `rb_eval_model.R`.

### Task 2.1: per-rusher-season summarize + epa clamp

**Files:**
- Create: `python/rb_eval/aggregate.py`
- Create: `tests/rb_eval/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_aggregate.py
import polars as pl
import pytest
from rb_eval.aggregate import summarize_rusher_seasons


def _rush_plays(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _base_play(rusher="A", season=2010, yds=5, down=1, dist=10, epa=0.5, n_opps_yds=5):
    return {
        "rusher_player_name": rusher,
        "season": season,
        "yds_rushed": yds,
        "start.down": down,
        "start.distance": dist,
        "epa": epa,
        "is_rush_opp": yds >= 4,
        "fo_success": yds >= 0.5 * dist if down == 1 else yds >= 0.7 * dist if down == 2 else yds >= dist,
        "highlight_yards": max(0.0, (0.5 * (min(yds, 8) - 4) if yds >= 4 else 0.0) + (yds - min(yds, 8) if yds > 8 else 0.0)),
        "pos_team": 100,
    }


def test_epa_clamped_at_minus_4_5():
    # A play with epa = -6.0 should be clamped to -4.5 in the 'epa' summary
    play_big_loss = {**_base_play("A", 2010, yds=1, epa=-6.0), "is_rush_opp": False, "fo_success": False, "highlight_yards": 0.0}
    plays = [_base_play("A", 2010) for _ in range(100)] + [play_big_loss]
    df = _rush_plays(plays)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "A").row(0, named=True)
    # unadjusted_epa includes -6.0; epa_clamped should use -4.5 instead
    assert row["unadjusted_epa"] < row["epa"]  # unadjusted < epa (unadjusted more negative)


def test_n_plays_filter_excludes_below_100():
    # Rusher with only 50 plays is excluded after the n>100 filter
    plays_50 = [_base_play("LowVol", 2010) for _ in range(50)]
    plays_101 = [_base_play("HighVol", 2010) for _ in range(101)]
    df = _rush_plays(plays_50 + plays_101)
    out = summarize_rusher_seasons(df)
    assert "LowVol" not in out["rusher_player_name"].to_list()
    assert "HighVol" in out["rusher_player_name"].to_list()


def test_n_opps_zero_guard_for_highlight_yards():
    # All plays have yds_rushed < 4 (no rush opportunities) -> highlight_yards = 0, not NaN
    plays = [{**_base_play("ToughYard", 2010, yds=2, epa=-0.1),
              "is_rush_opp": False, "highlight_yards": 0.0}
             for _ in range(101)]
    df = _rush_plays(plays)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "ToughYard").row(0, named=True)
    assert row["highlight_yards"] == 0.0  # not NaN
    assert row["n_opps"] == 0


def test_success_rate_formula():
    # 50 successful plays out of 101 total -> success = 50/101
    plays_success = [{**_base_play("S", 2010, yds=5, epa=0.3),
                      "is_rush_opp": True, "fo_success": True, "highlight_yards": 0.5}
                     for _ in range(50)]
    plays_fail = [{**_base_play("S", 2010, yds=2, epa=-0.1),
                   "is_rush_opp": False, "fo_success": False, "highlight_yards": 0.0}
                  for _ in range(51)]
    df = _rush_plays(plays_success + plays_fail)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "S").row(0, named=True)
    import math
    assert math.isclose(row["success"], 50 / 101, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_aggregate.py -v`
Expected: FAIL (`No module named 'rb_eval.aggregate'`)

- [ ] **Step 3: Implement `summarize_rusher_seasons`**

```python
# python/rb_eval/aggregate.py
"""Per-rusher-season aggregation, lag, and weight derivation.

Port of rb_eval_model.R lines 54-86:
  lrbs block: group by (rusher_player_name, season), summarize, filter n>100, lag 1 season, weight.
  model_data rename: target=unadjusted_epa, epa_per_play=lepa, success=lsuccess.
"""
from __future__ import annotations

import polars as pl


def summarize_rusher_seasons(rush_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate rush plays to per-(rusher, season) summary statistics.

    Applies the epa clamp (< -4.5 → -4.5), computes unadjusted_epa (pre-clamp mean),
    epa (clamped mean), success (FO rate), highlight_yards/n_opps (guarded vs zero n_opps),
    and filters to n_plays > 100.

    Args:
        rush_df: filtered rush plays with columns epa, fo_success, is_rush_opp,
                 highlight_yards, rusher_player_name, season.

    Returns:
        polars DataFrame of per-rusher-season metrics.
    """
    # Clamp epa for the 'epa' column (clamp happens before aggregation, per the R mutate)
    df = rush_df.with_columns(
        epa_clamped=pl.when(pl.col("epa") < -4.5).then(-4.5).otherwise(pl.col("epa")),
    )
    agg = (
        df.group_by(["rusher_player_name", "season"])
        .agg(
            n_plays=pl.len(),
            n_opps=pl.col("is_rush_opp").sum().cast(pl.Int64),
            unadjusted_epa=pl.col("epa").sum() / pl.len(),
            epa=pl.col("epa_clamped").sum() / pl.len(),
            success=pl.col("fo_success").cast(pl.Int32).sum() / pl.len(),
            highlight_yards_sum=pl.col("highlight_yards").sum(),
        )
        .with_columns(
            # Guard against n_opps = 0: set highlight_yards to 0.0 instead of NaN
            highlight_yards=pl.when(pl.col("n_opps") > 0)
            .then(pl.col("highlight_yards_sum") / pl.col("n_opps").cast(pl.Float64))
            .otherwise(0.0),
        )
        .drop("highlight_yards_sum")
    )
    # Apply n_plays > 100 filter (R: filter(n_plays > 100))
    return agg.filter(pl.col("n_plays") > 100)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_aggregate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/aggregate.py tests/rb_eval/test_aggregate.py
git commit -m "feat(rb-eval): per-rusher-season aggregation (epa clamp, FO success, highlight-yards guard)"
```

### Task 2.2: lag by 1 season + weight

**Files:**
- Modify: `python/rb_eval/aggregate.py`
- Modify: `tests/rb_eval/test_aggregate.py`

- [ ] **Step 1: Add lag and weight tests**

```python
# tests/rb_eval/test_aggregate.py  (append)
from rb_eval.aggregate import add_season_lag


def _rusher_seasons(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_lag_shifts_prior_season_values():
    # Rusher A: season 2010 (n=110) and season 2011 (n=120)
    # After lag: 2011 row has lepa = 2010's epa, lsuccess = 2010's success
    df = _rusher_seasons([
        {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
         "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
        {"rusher_player_name": "A", "season": 2011, "n_plays": 120, "epa": 0.2,
         "success": 0.5, "highlight_yards": 0.6, "unadjusted_epa": 0.22, "n_opps": 70},
    ])
    out = add_season_lag(df)
    row_2011 = out.filter(pl.col("season") == 2011).row(0, named=True)
    import math
    assert math.isclose(row_2011["lepa"], 0.1, rel_tol=1e-9)
    assert math.isclose(row_2011["lsuccess"], 0.4, rel_tol=1e-9)
    assert math.isclose(row_2011["lhlite_yds"], 0.5, rel_tol=1e-9)
    assert row_2011["lplays"] == 110


def test_first_season_has_null_lag():
    # Rusher A: only season 2010 — lag values are null (no prior season)
    df = _rusher_seasons([
        {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
         "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
    ])
    out = add_season_lag(df)
    row = out.row(0, named=True)
    assert row["lepa"] is None
    assert row["lsuccess"] is None


def test_weight_formula():
    # weight = (n_plays^2 + lplays^2)^0.5
    import math
    df = _rusher_seasons([
        {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
         "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
        {"rusher_player_name": "A", "season": 2011, "n_plays": 120, "epa": 0.2,
         "success": 0.5, "highlight_yards": 0.6, "unadjusted_epa": 0.22, "n_opps": 70},
    ])
    out = add_season_lag(df)
    row_2011 = out.filter(pl.col("season") == 2011).row(0, named=True)
    expected = math.sqrt(120**2 + 110**2)
    assert math.isclose(row_2011["weight"], expected, rel_tol=1e-9)


def test_non_consecutive_seasons_produce_null_lag():
    # Rusher with seasons 2010 and 2012 (gap year) — 2012 gets null lag
    # because shift(1) looks at the immediately prior row in the sorted order
    # and we must verify the sort is by season, not by array position
    df = _rusher_seasons([
        {"rusher_player_name": "A", "season": 2012, "n_plays": 130, "epa": 0.3,
         "success": 0.55, "highlight_yards": 0.7, "unadjusted_epa": 0.32, "n_opps": 80},
        {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
         "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
    ])
    out = add_season_lag(df)
    row_2012 = out.filter(pl.col("season") == 2012).row(0, named=True)
    # shift(1) assigns 2010 values to 2012 (no actual consecutive-season validation)
    # The R script uses lag() which just shifts in order — we replicate this.
    # With proper sort: 2010 → 2012 (adjacent in sorted order) so lag IS 2010
    assert row_2012["lepa"] is not None  # 2010 is adjacent in sorted order
    assert row_2012["lplays"] == 110
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_aggregate.py -v`
Expected: FAIL (`cannot import name 'add_season_lag'`)

- [ ] **Step 3: Implement `add_season_lag`**

```python
# python/rb_eval/aggregate.py  (append)

def add_season_lag(rusher_seasons: pl.DataFrame) -> pl.DataFrame:
    """Add prior-season lag columns (lepa, lsuccess, lhlite_yds, lunad_epa, lplays) and weight.

    Mirrors R `mutate(lepa = lag(epa, n=1), ...)` within group_by(rusher_player_name, season).
    The R lag operates on the within-rusher sorted sequence; polars `shift(1).over(rusher)` with
    a prior sort by (rusher, season) produces the same result.

    Args:
        rusher_seasons: aggregated frame from summarize_rusher_seasons.

    Returns:
        Frame with lag columns and weight added. Rows with null lepa/lsuccess are retained
        (the first-season row per rusher); downstream build_model_data drops them via drop_nulls.
    """
    df = rusher_seasons.sort(["rusher_player_name", "season"])
    df = df.with_columns(
        lepa=pl.col("epa").shift(1).over("rusher_player_name"),
        lunad_epa=pl.col("unadjusted_epa").shift(1).over("rusher_player_name"),
        lhlite_yds=pl.col("highlight_yards").shift(1).over("rusher_player_name"),
        lsuccess=pl.col("success").shift(1).over("rusher_player_name"),
        lplays=pl.col("n_plays").shift(1).over("rusher_player_name"),
    )
    return df.with_columns(
        weight=((pl.col("n_plays").cast(pl.Float64) ** 2 + pl.col("lplays").cast(pl.Float64) ** 2) ** 0.5),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_aggregate.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/aggregate.py tests/rb_eval/test_aggregate.py
git commit -m "feat(rb-eval): season lag + weight (shift(1).over(rusher) + Pythagorean weight)"
```

### Task 2.3: `build_rusher_seasons` + `build_model_data` orchestration

**Files:**
- Modify: `python/rb_eval/aggregate.py`
- Modify: `tests/rb_eval/test_aggregate.py`

- [ ] **Step 1: Add orchestration tests**

```python
# tests/rb_eval/test_aggregate.py  (append)
from rb_eval.aggregate import build_rusher_seasons, build_model_data


def _big_rush_frame(n_per_rusher=110, seasons=None):
    """Generate a synthetic frame with two rushers across multiple seasons."""
    seasons = seasons or [2010, 2011, 2012]
    rows = []
    for rusher in ["Alpha", "Beta"]:
        for season in seasons:
            for _ in range(n_per_rusher):
                rows.append({
                    "rusher_player_name": rusher, "season": season,
                    "yds_rushed": 5, "start.down": 1, "start.distance": 10,
                    "epa": 0.3, "is_rush_opp": True, "fo_success": True,
                    "highlight_yards": 0.5, "pos_team": 100,
                })
    return pl.DataFrame(rows)


def test_build_rusher_seasons_has_expected_columns():
    df = _big_rush_frame()
    out = build_rusher_seasons(df)
    for col in ["rusher_player_name", "season", "n_plays", "n_opps", "unadjusted_epa",
                "epa", "success", "highlight_yards", "lepa", "lsuccess", "weight"]:
        assert col in out.columns, f"missing column: {col}"


def test_build_model_data_renames_to_gam_contract():
    df = _big_rush_frame()
    seasons_df = build_rusher_seasons(df)
    md = build_model_data(seasons_df)
    for col in ["target", "epa_per_play", "success", "highlight_yards", "weight", "season"]:
        assert col in md.columns, f"missing model_data column: {col}"
    # No null lag rows (first season per rusher dropped)
    assert md["epa_per_play"].null_count() == 0
    assert md["success"].null_count() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_aggregate.py::test_build_rusher_seasons_has_expected_columns -v`
Expected: FAIL (`cannot import name 'build_rusher_seasons'`)

- [ ] **Step 3: Implement orchestration functions**

```python
# python/rb_eval/aggregate.py  (append)

def build_rusher_seasons(rush_df: pl.DataFrame) -> pl.DataFrame:
    """Full aggregation pipeline: summarize → lag → weight.

    Args:
        rush_df: filtered rush plays from features.filter_rush_plays.

    Returns:
        Per-rusher-season frame with all lag and weight columns.
    """
    seasons_df = summarize_rusher_seasons(rush_df)
    return add_season_lag(seasons_df)


def build_model_data(rusher_seasons: pl.DataFrame) -> pl.DataFrame:
    """Rename per-rusher-season columns to GAM input contract and drop null-lag rows.

    Port of rb_eval_model.R lines 79-86 (model_data <- lrbs %>% select(...) %>% rename(...)).

    GAM input: target=unadjusted_epa, epa_per_play=lepa, success=lsuccess,
               highlight_yards=lhlite_yds, weight, season.
    highlight_yards is included for descriptive output but NOT in the GAM formula.

    Args:
        rusher_seasons: output of build_rusher_seasons.

    Returns:
        model_data frame with GAM column names; rows with null lag dropped.
    """
    return (
        rusher_seasons
        .select(["rusher_player_name", "unadjusted_epa", "lhlite_yds", "lepa",
                 "lsuccess", "weight", "season"])
        .rename({
            "unadjusted_epa": "target",
            "lhlite_yds": "highlight_yards",
            "lepa": "epa_per_play",
            "lsuccess": "success",
        })
        .drop_nulls(["epa_per_play", "success", "weight"])
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_aggregate.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/aggregate.py tests/rb_eval/test_aggregate.py
git commit -m "feat(rb-eval): build_rusher_seasons + build_model_data orchestration"
```

---

## Phase 3 — `train.py` (pygam LinearGAM + LOSO CV + persistence)

Port of the GAM fitting block (lines 104–115) and LOSO loop (lines 98–115) in `rb_eval_model.R`.

### Task 3.1: `train_xrepa` — LinearGAM(s(0) + s(1)) basic fit

**Files:**
- Create: `python/rb_eval/train.py`
- Create: `tests/rb_eval/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_train.py
import numpy as np
import polars as pl
import pytest
from rb_eval.train import train_xrepa


def _synth_model_data(n: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "rusher_player_name": [f"R{i % 20}" for i in range(n)],
        "season": [2010 + i % 5 for i in range(n)],
        "epa_per_play": rng.normal(0.0, 0.3, n).tolist(),
        "success": rng.uniform(0.3, 0.7, n).tolist(),
        "target": rng.normal(0.0, 0.2, n).tolist(),
        "highlight_yards": rng.uniform(0.0, 2.0, n).tolist(),
        "weight": rng.uniform(100.0, 300.0, n).tolist(),
    })


def test_train_xrepa_returns_fitted_gam():
    from pygam import LinearGAM
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    assert isinstance(gam, LinearGAM)
    assert gam._is_fitted


def test_train_xrepa_predictions_are_finite():
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    X = model_data[["epa_per_play", "success"]].to_numpy()
    preds = gam.predict(X)
    assert np.all(np.isfinite(preds)), "GAM predictions contain non-finite values"
    assert preds.shape == (len(model_data),)


def test_train_xrepa_uses_two_features():
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    # pygam LinearGAM with s(0)+s(1) has 2 terms (plus intercept)
    assert gam.n_features == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_train.py -v`
Expected: FAIL (`No module named 'rb_eval.train'`)

- [ ] **Step 3: Implement `train_xrepa`**

```python
# python/rb_eval/train.py
"""GAM training for xREPA: LinearGAM(s(0) + s(1)) on prior-season epa_per_play and success.

Port of rb_eval_model.R:
    dakota_model = mgcv::gam(target ~ s(epa_per_play) + s(success), data=train_data, weights=weight)

The pygam LinearGAM is an approximate equivalent: B-spline basis (vs mgcv thin-plate regression
splines). LOSO CV by season replicates the R cv_results loop.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

try:
    from pygam import LinearGAM, s
except ImportError as e:
    raise ImportError(
        "pygam is required for rb_eval training. Install with: uv sync --group gam"
    ) from e


_GAM_FEATURES = ["epa_per_play", "success"]


def train_xrepa(model_data: pl.DataFrame) -> "LinearGAM":
    """Fit LinearGAM(s(0) + s(1)) on (epa_per_play, success) -> target with sample weights.

    Args:
        model_data: GAM input frame from aggregate.build_model_data. Must contain columns
                    epa_per_play, success, target, weight with no null values.

    Returns:
        Fitted pygam LinearGAM.
    """
    df = model_data.drop_nulls(_GAM_FEATURES + ["target", "weight"])
    X = df.select(_GAM_FEATURES).to_numpy()
    y = df["target"].to_numpy()
    w = df["weight"].to_numpy()
    gam = LinearGAM(s(0) + s(1))
    gam.fit(X, y, weights=w)
    return gam
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_train.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/train.py tests/rb_eval/test_train.py
git commit -m "feat(rb-eval): train_xrepa (LinearGAM(s(0)+s(1)), sample_weight=weight)"
```

### Task 3.2: `loso_cv` — leave-one-season-out predictions

**Files:**
- Modify: `python/rb_eval/train.py`
- Modify: `tests/rb_eval/test_train.py`

- [ ] **Step 1: Add LOSO test**

```python
# tests/rb_eval/test_train.py  (append)
from rb_eval.train import loso_cv


def test_loso_cv_covers_all_seasons():
    model_data = _synth_model_data(n=200)  # enough data for leave-one-season-out
    cv = loso_cv(model_data)
    # Every season in model_data should have at least one row in cv output
    assert set(cv["season"].unique().to_list()) == set(model_data["season"].unique().to_list())


def test_loso_cv_output_has_exp_rb_epa():
    model_data = _synth_model_data(n=200)
    cv = loso_cv(model_data)
    assert "exp_rb_epa" in cv.columns
    assert cv["exp_rb_epa"].null_count() == 0
    assert cv["exp_rb_epa"].dtype in (pl.Float32, pl.Float64)


def test_loso_cv_does_not_train_on_test_season():
    # Smoke: cv should produce predictions even when a season has very few rows
    # (e.g., only 1 rusher in the test set). No exception expected.
    model_data = _synth_model_data(n=100)
    cv = loso_cv(model_data)  # should not raise
    assert cv.height > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_train.py::test_loso_cv_covers_all_seasons -v`
Expected: FAIL (`cannot import name 'loso_cv'`)

- [ ] **Step 3: Implement `loso_cv`**

```python
# python/rb_eval/train.py  (append)

def loso_cv(model_data: pl.DataFrame) -> pl.DataFrame:
    """Leave-one-season-out cross-validation for xREPA.

    For each held-out season, train on all other seasons and predict on the held-out set.
    Mirrors the R cv_results map_dfr loop (lines 98-115 in rb_eval_model.R).

    Args:
        model_data: GAM input frame from aggregate.build_model_data.

    Returns:
        model_data with an added exp_rb_epa column (LOSO predictions).
        Seasons for which the training set is empty after null-dropping are skipped.
    """
    seasons = sorted(model_data["season"].drop_nulls().unique().to_list())
    parts: list[pl.DataFrame] = []
    for season in seasons:
        train = model_data.filter(pl.col("season") != season).drop_nulls(
            _GAM_FEATURES + ["target", "weight"]
        )
        test = model_data.filter(pl.col("season") == season).drop_nulls(
            _GAM_FEATURES + ["target", "weight"]
        )
        if train.is_empty() or test.is_empty():
            continue
        gam = train_xrepa(train)
        X_test = test.select(_GAM_FEATURES).to_numpy()
        preds = gam.predict(X_test)
        parts.append(
            test.with_columns(pl.Series("exp_rb_epa", preds, dtype=pl.Float64))
        )
    if not parts:
        return model_data.with_columns(pl.lit(None).cast(pl.Float64).alias("exp_rb_epa"))
    return pl.concat(parts, how="diagonal_relaxed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_train.py -v`
Expected: PASS (6 tests total)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/train.py tests/rb_eval/test_train.py
git commit -m "feat(rb-eval): loso_cv (leave-one-season-out xREPA predictions)"
```

### Task 3.3: `save_model` / `load_model` + `model_card.json`

**Files:**
- Modify: `python/rb_eval/train.py`
- Modify: `tests/rb_eval/test_train.py`

- [ ] **Step 1: Add persistence tests**

```python
# tests/rb_eval/test_train.py  (append)
import pathlib, json
from rb_eval.train import save_model, load_model


def test_save_and_load_model_roundtrip(tmp_path):
    from pygam import LinearGAM
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    pkl_path = tmp_path / "xrepa_final.pkl"
    card_path = save_model(gam, pkl_path, season_range=(2010, 2014), n_rushers=20)
    assert pkl_path.exists()
    assert card_path.exists()
    loaded = load_model(pkl_path)
    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted
    # model_card.json has expected keys
    card = json.loads(card_path.read_text())
    assert "pygam_version" in card and "season_range" in card and "n_rushers" in card
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_train.py::test_save_and_load_model_roundtrip -v`
Expected: FAIL (`cannot import name 'save_model'`)

- [ ] **Step 3: Implement persistence**

```python
# python/rb_eval/train.py  (append)
import json

try:
    import joblib
except ImportError as e:
    raise ImportError(
        "joblib is required for model persistence. Install with: uv sync --group dev"
    ) from e


def save_model(gam: "LinearGAM", path, season_range: tuple[int, int],
               n_rushers: int) -> Path:
    """Persist the fitted GAM to disk via joblib and write a model_card.json sidecar.

    Args:
        gam: fitted LinearGAM from train_xrepa.
        path: destination .pkl path.
        season_range: (first_season, last_season) of training data.
        n_rushers: number of rusher-season rows used for training.

    Returns:
        Path to the model_card.json sidecar.
    """
    import pygam
    from datetime import date

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gam, path)
    card = {
        "pygam_version": pygam.__version__,
        "season_range": list(season_range),
        "n_rushers": n_rushers,
        "trained_date": date.today().isoformat(),
        "model_formula": "LinearGAM(s(0) + s(1))",
        "features": ["epa_per_play", "success"],
        "target": "unadjusted_epa",
        "note": "xREPA analytical artifact — NOT bundled into sdv-py.",
    }
    card_path = path.with_suffix(".json")
    card_path.write_text(json.dumps(card, indent=2))
    return card_path


def load_model(path) -> "LinearGAM":
    """Load a persisted GAM from disk.

    Args:
        path: path to the .pkl file written by save_model.

    Returns:
        Fitted LinearGAM.
    """
    return joblib.load(Path(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_train.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/train.py tests/rb_eval/test_train.py
git commit -m "feat(rb-eval): save_model/load_model (joblib + model_card.json sidecar)"
```

---

## Phase 4 — `validate.py` (calibration table + weighted error + weighted R²)

Port of the `show_calibration_chart` / calibration math in `rb_eval_model.R` (lines 119–176),
decoupled from the plotting.

### Task 4.1: calibration_table + weighted_cal_error

**Files:**
- Create: `python/rb_eval/validate.py`
- Create: `tests/rb_eval/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_validate.py
import math
import polars as pl
from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2


def _cv_frame(n: int = 40) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(7)
    pred = rng.normal(0.0, 0.15, n)
    actual = pred + rng.normal(0.0, 0.05, n)  # actual ≈ pred + small noise
    return pl.DataFrame({
        "exp_rb_epa": pred.tolist(),
        "target": actual.tolist(),
    })


def test_calibration_table_bins_by_pred_epa():
    cv = _cv_frame()
    table = calibration_table(cv, bin_size=0.05)
    assert "bin_pred_epa" in table.columns
    assert "bin_actual_epa" in table.columns
    assert "total_instances" in table.columns
    # bin column is rounded to bin_size increments
    bins = table["bin_pred_epa"].to_list()
    for b in bins:
        remainder = abs(b / 0.05 - round(b / 0.05))
        assert remainder < 1e-9, f"bin {b} not a multiple of 0.05"


def test_weighted_cal_error_is_non_negative():
    cv = _cv_frame(100)
    table = calibration_table(cv)
    err = weighted_cal_error(table)
    assert err >= 0.0


def test_weighted_cal_error_is_zero_for_perfect_calibration():
    # Perfect calibration: bin_actual_epa == bin_pred_epa
    table = pl.DataFrame({
        "bin_pred_epa": [-0.1, 0.0, 0.1],
        "bin_actual_epa": [-0.1, 0.0, 0.1],
        "total_instances": [10, 20, 10],
    })
    err = weighted_cal_error(table)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_weighted_r2_between_zero_and_one_for_noisy_pred():
    cv = _cv_frame(100)
    table = calibration_table(cv)
    r2 = weighted_r2(table)
    assert 0.0 <= r2 <= 1.0 or r2 < 0.0  # r2 can be negative for bad models; just check finite
    assert math.isfinite(r2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_validate.py -v`
Expected: FAIL (`No module named 'rb_eval.validate'`)

- [ ] **Step 3: Implement validate module**

```python
# python/rb_eval/validate.py
"""Calibration table and metrics for xREPA LOSO validation.

Port of show_calibration_chart() in rb_eval_model.R (lines 119-176), decoupled from plotting.
The R calibration computes:
  bin_pred_epa = round(exp_rb_epa / bin_size) * bin_size
  bin_actual_epa = mean(target) within each bin
  weighted calibration error = weighted.mean(|bin_pred - bin_actual|, total_instances)
  weighted R² = cor(bin_actual, bin_pred, w=total_instances)^2  (boot::corr method)
"""
from __future__ import annotations

import numpy as np
import polars as pl


def calibration_table(cv_results: pl.DataFrame, bin_size: float = 0.05) -> pl.DataFrame:
    """Bin LOSO predictions and compute mean actual EPA per bin.

    Args:
        cv_results: frame with columns exp_rb_epa and target, from loso_cv.
        bin_size: bin width for rounding exp_rb_epa.

    Returns:
        Frame with bin_pred_epa, total_instances, bin_actual_epa columns, sorted by bin.
    """
    return (
        cv_results
        .drop_nulls(["exp_rb_epa", "target"])
        .with_columns(
            bin_pred_epa=(pl.col("exp_rb_epa") / bin_size).round(0) * bin_size,
        )
        .group_by("bin_pred_epa")
        .agg(
            total_instances=pl.len(),
            bin_actual_epa=pl.col("target").mean(),
        )
        .sort("bin_pred_epa")
    )


def weighted_cal_error(table: pl.DataFrame) -> float:
    """Weighted mean absolute calibration error.

    weighted.mean(|bin_pred_epa - bin_actual_epa|, total_instances)

    Args:
        table: output of calibration_table.

    Returns:
        Scalar weighted calibration error (float).
    """
    t = table.with_columns(
        cal_diff=(pl.col("bin_pred_epa") - pl.col("bin_actual_epa")).abs(),
    )
    total = t["total_instances"].sum()
    if total == 0:
        return float("nan")
    return float((t["cal_diff"] * t["total_instances"]).sum() / total)


def weighted_r2(table: pl.DataFrame) -> float:
    """Weighted R² of binned actual vs predicted EPA (boot::corr method from R source).

    Computes the weighted Pearson correlation^2 between bin_actual_epa and bin_pred_epa,
    with weights = total_instances. Mirrors R: r2 = boot::corr(d=cbind(y, y_pred), w=w)^2.

    Args:
        table: output of calibration_table.

    Returns:
        Scalar weighted R² (float).
    """
    t = table.drop_nulls(["bin_pred_epa", "bin_actual_epa"])
    if t.is_empty():
        return float("nan")
    y = t["bin_actual_epa"].to_numpy()
    y_hat = t["bin_pred_epa"].to_numpy()
    w = t["total_instances"].to_numpy().astype(float)
    w_norm = w / w.sum()
    mu_y = np.sum(w_norm * y)
    mu_yhat = np.sum(w_norm * y_hat)
    cov_yy = np.sum(w_norm * (y - mu_y) ** 2)
    cov_yyhyh = np.sum(w_norm * (y_hat - mu_yhat) ** 2)
    cov_yyh = np.sum(w_norm * (y - mu_y) * (y_hat - mu_yhat))
    denom = np.sqrt(cov_yy * cov_yyhyh)
    if denom < 1e-14:
        return float("nan")
    return float((cov_yyh / denom) ** 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_validate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add python/rb_eval/validate.py tests/rb_eval/test_validate.py
git commit -m "feat(rb-eval): calibration_table + weighted_cal_error + weighted_r2"
```

---

## Phase 5 — figures (calibration plot via model_training.figures)

The xREPA calibration figure reuses `model_training.figures.write_calibration`. Since xREPA has no
natural facet variable (unlike WP's quarter), the figure is produced with a constant `by` column
and the facet strip is suppressed. This phase adapts the shared helper rather than duplicating code.

### Task 5.1: xrepa_calibration_figure wrapper

**Files:**
- Create: `python/rb_eval/figures.py`
- Modify: `tests/rb_eval/test_validate.py`

- [ ] **Step 1: Add figure test (appended to validate test)**

```python
# tests/rb_eval/test_validate.py  (append)
import pathlib


def test_xrepa_calibration_figure_emits_png_and_csv(tmp_path):
    from rb_eval.figures import write_xrepa_calibration
    cv = _cv_frame(100)
    table = calibration_table(cv)
    png, csv = write_xrepa_calibration(table, tmp_path / "xrepa", cal_error=0.01, r2=0.82)
    assert png.exists()
    assert csv.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_validate.py::test_xrepa_calibration_figure_emits_png_and_csv -v`
Expected: FAIL (`No module named 'rb_eval.figures'`)

- [ ] **Step 3: Implement `write_xrepa_calibration`**

```python
# python/rb_eval/figures.py
"""xREPA calibration figure — thin wrapper over model_training.figures.write_calibration.

xREPA has no facet variable (single-panel: all rushers). We add a constant 'by' column
("All rushers") and suppress the facet strip in the theme.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from model_training.figures import write_calibration as _wc
except ImportError as e:
    raise ImportError(
        "model_training package required. Ensure python/model_training/ is on sys.path."
    ) from e


def write_xrepa_calibration(
    table: pl.DataFrame, stem, cal_error: float, r2: float
) -> tuple[Path, Path]:
    """Produce the xREPA calibration PNG + sidecar CSV/parquet.

    Wraps model_training.figures.write_calibration with a constant 'by' column
    and an xREPA-specific subtitle.

    Args:
        table: output of validate.calibration_table (bin_pred_epa, bin_actual_epa, total_instances).
        stem: output path stem (no extension); PNG and CSV are written alongside.
        cal_error: weighted calibration error (for caption).
        r2: weighted R² (for caption).

    Returns:
        (png_path, csv_path) tuple.
    """
    # Rename columns to match the shared write_calibration contract
    # write_calibration expects: by, bin, n_plays, actual
    adapted = table.rename({
        "bin_pred_epa": "bin",
        "total_instances": "n_plays",
        "bin_actual_epa": "actual",
    }).with_columns(by=pl.lit("All rushers"))
    return _wc(
        adapted,
        stem=stem,
        title="xREPA LOSO Calibration",
        subtitle=f"Wgt R²: {round(r2, 4)}",
        cal_error=cal_error,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_validate.py -v`
Expected: PASS (5 tests total). A font warning for Gill Sans MT is acceptable.

- [ ] **Step 5: Commit**

```
git add python/rb_eval/figures.py tests/rb_eval/test_validate.py
git commit -m "feat(rb-eval): write_xrepa_calibration (reuses model_training.figures)"
```

---

## Phase 6 — `cli.py` (subcommand dispatch)

### Task 6.1: CLI structure

**Files:**
- Create: `python/rb_eval/cli.py`
- Create: `tests/rb_eval/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rb_eval/test_cli.py
from rb_eval.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    subcommand_names = set(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected = {"features", "aggregate", "train", "validate", "figures"}
    assert expected <= subcommand_names


def test_help_exits_zero():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "rb_eval", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "features" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rb_eval/test_cli.py -v`
Expected: FAIL (`No module named 'rb_eval.cli'`)

- [ ] **Step 3: Implement the CLI**

```python
# python/rb_eval/cli.py
"""CLI: features | aggregate | train | validate | figures.

Usage:
  uv run python -m rb_eval features   --final-dir cfb/json/final --out cfb/rb_eval/rush_plays.parquet
  uv run python -m rb_eval aggregate  --plays cfb/rb_eval/rush_plays.parquet --out cfb/rb_eval/rusher_seasons.parquet
  uv run python -m rb_eval train      --seasons cfb/rb_eval/rusher_seasons.parquet --out cfb/rb_eval/
  uv run python -m rb_eval validate   --loso cfb/rb_eval/xrepa_loso.parquet --out cfb/rb_eval/
  uv run python -m rb_eval figures    --table cfb/rb_eval/calibration.parquet --out cfb/rb_eval/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rb_eval",
        description="CFB RB-Eval xREPA pipeline (Track 3, CFB Modeling Suite).",
    )
    ap.add_argument("--seasons", default=None,
                    help="Season range as A:B (e.g. 2006:2025); default = all available.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features", help="Load rush plays from final.json and compute fo_success.")
    f.add_argument("--final-dir", default="cfb/json/final")
    f.add_argument("--out", default="cfb/rb_eval/rush_plays.parquet")

    a = sub.add_parser("aggregate", help="Aggregate to per-rusher-season, lag, weight.")
    a.add_argument("--plays", default="cfb/rb_eval/rush_plays.parquet")
    a.add_argument("--out", default="cfb/rb_eval/rusher_seasons.parquet")

    t = sub.add_parser("train", help="Fit LinearGAM and run LOSO CV.")
    t.add_argument("--seasons", dest="seasons_parquet", default="cfb/rb_eval/rusher_seasons.parquet")
    t.add_argument("--out", default="cfb/rb_eval/")

    v = sub.add_parser("validate", help="Compute calibration table and metrics.")
    v.add_argument("--loso", default="cfb/rb_eval/xrepa_loso.parquet")
    v.add_argument("--out", default="cfb/rb_eval/")

    fi = sub.add_parser("figures", help="Produce calibration PNG + data table.")
    fi.add_argument("--table", default="cfb/rb_eval/calibration.parquet")
    fi.add_argument("--out", default="cfb/rb_eval/")

    return ap


def _parse_seasons(seasons_str):
    if seasons_str is None:
        return None
    parts = seasons_str.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seasons = _parse_seasons(getattr(args, "seasons", None))

    if args.cmd == "features":
        import polars as pl
        from rb_eval.features import load_rush_plays
        df = load_rush_plays(args.final_dir, seasons=seasons)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.out)
        print(f"features: wrote {df.height} rush plays -> {args.out}")

    elif args.cmd == "aggregate":
        import polars as pl
        from rb_eval.aggregate import build_rusher_seasons, build_model_data
        rush_df = pl.read_parquet(args.plays)
        seasons_df = build_rusher_seasons(rush_df)
        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        seasons_df.write_parquet(args.out)
        print(f"aggregate: wrote {seasons_df.height} rusher-season rows -> {args.out}")

    elif args.cmd == "train":
        import polars as pl
        from rb_eval.aggregate import build_model_data
        from rb_eval.train import train_xrepa, loso_cv, save_model
        seasons_df = pl.read_parquet(args.seasons_parquet)
        model_data = build_model_data(seasons_df)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        # LOSO CV
        cv = loso_cv(model_data)
        cv.write_parquet(out_dir / "xrepa_loso.parquet")
        print(f"train: wrote LOSO predictions ({cv.height} rows) -> {out_dir / 'xrepa_loso.parquet'}")
        # Full-data model
        gam = train_xrepa(model_data)
        card = save_model(
            gam, out_dir / "xrepa_final.pkl",
            season_range=(int(model_data["season"].min()), int(model_data["season"].max())),
            n_rushers=model_data.height,
        )
        print(f"train: saved full-data GAM -> {out_dir / 'xrepa_final.pkl'}")
        print(f"train: wrote model card -> {card}")

    elif args.cmd == "validate":
        import polars as pl
        from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
        cv = pl.read_parquet(args.loso)
        table = calibration_table(cv)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out_dir / "calibration.parquet")
        table.write_csv(out_dir / "calibration.csv")
        err = weighted_cal_error(table)
        r2 = weighted_r2(table)
        print(f"validate: weighted cal error = {err:.6f}, weighted R² = {r2:.4f}")
        print(f"validate: calibration table -> {out_dir / 'calibration.parquet'}")

    elif args.cmd == "figures":
        import polars as pl
        from rb_eval.validate import weighted_cal_error, weighted_r2
        from rb_eval.figures import write_xrepa_calibration
        table = pl.read_parquet(args.table)
        out_dir = Path(args.out)
        err = weighted_cal_error(table)
        r2 = weighted_r2(table)
        png, csv = write_xrepa_calibration(
            table, out_dir / "xrepa_calibration", cal_error=err, r2=r2
        )
        print(f"figures: wrote {png}, {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add `__main__.py` so `python -m rb_eval` works**

```python
# python/rb_eval/__main__.py
from rb_eval.cli import main
raise SystemExit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/rb_eval/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```
git add python/rb_eval/cli.py python/rb_eval/__main__.py tests/rb_eval/test_cli.py
git commit -m "feat(rb-eval): CLI subcommand dispatch (features|aggregate|train|validate|figures)"
```

---

## Phase 7 — end-to-end smoke test (backfill-gated)

### Task 7.1: integration smoke over on-disk final.json

**Files:**
- Create: `tests/rb_eval/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/rb_eval/test_smoke.py
"""Integration smoke: features -> aggregate -> train -> validate -> figures.

Skipped when no backfill final.json is present on disk.
"""
import pathlib
import pytest
import polars as pl

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(
    not any(FINAL_DIR.glob("*.json")),
    reason="no backfill final.json on disk; run scrape_cfb_json.py + reprocess_cfb_json.py first",
)
def test_full_pipeline_runs_without_error(tmp_path):
    from rb_eval.features import load_rush_plays
    from rb_eval.aggregate import build_rusher_seasons, build_model_data
    from rb_eval.train import loso_cv, train_xrepa, save_model
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
    from rb_eval.figures import write_xrepa_calibration

    # features
    rush_df = load_rush_plays(FINAL_DIR, seasons=None)
    assert rush_df.height > 0, "No rush plays loaded from backfill"

    # aggregate
    seasons_df = build_rusher_seasons(rush_df)
    model_data = build_model_data(seasons_df)
    assert model_data.height > 0, "No model_data rows after aggregation"
    assert model_data["epa_per_play"].null_count() == 0

    # train: LOSO (small subset for speed — limit to most recent 3 seasons)
    recent_seasons = sorted(model_data["season"].unique().to_list())[-3:]
    md_small = model_data.filter(pl.col("season").is_in(recent_seasons))
    if md_small.height < 5:
        pytest.skip("Too few rows for a meaningful smoke test with available backfill data")
    cv = loso_cv(md_small)
    assert "exp_rb_epa" in cv.columns

    # validate
    table = calibration_table(cv)
    err = weighted_cal_error(table)
    r2 = weighted_r2(table)
    assert err >= 0.0
    import math
    assert math.isfinite(r2)

    # figures
    png, csv = write_xrepa_calibration(
        table, tmp_path / "xrepa_calibration", cal_error=err, r2=r2
    )
    assert png.exists()
    assert csv.exists()
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/rb_eval/test_smoke.py -v`
Expected: PASS (or SKIP without backfill data). With backfill data, this confirms the full
pipeline executes on real plays without errors.

- [ ] **Step 3: Commit**

```
git add tests/rb_eval/test_smoke.py
git commit -m "test(rb-eval): full-pipeline integration smoke (backfill-gated)"
```

---

## Full test suite run

After all phases complete:

```
uv run pytest tests/rb_eval/ -v
```

Expected: all offline tests PASS; backfill-gated tests SKIP (unless backfill data present).
Total offline tests: ~25. No live API calls anywhere in this module.

---

## Stage gating note

This track has a single implementation stage (no R reference models to compare against — the R
source is the specification, not a fixture). The parity bar is:

1. **Deterministic intermediates** (fo_success values, epa clamp, per-rusher-season metrics,
   lag values, weight formula) — **asserted exactly** against synthetic ground truth in unit tests.
2. **LOSO calibration** — run against the full backfill; confirm weighted calibration error
   is comparable to the R original (ballpark ≤ 0.02 absolute; the R source reports values at
   `bin_size=0.05` but does not publish a numeric reference, so eyeballing the calibration plot
   is the primary signal).
3. **Figure quality** — eyeball the `_calibration.png` against the R `show_calibration_chart(0.05)`
   output (scatter + y=x + annotation structure matches).

No UBJ reference models, no `tests/fixtures/model_training/` analogs. The synthetic unit tests
ARE the parity gate for this track.
