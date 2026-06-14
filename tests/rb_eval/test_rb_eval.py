"""T3 RB-Eval xREPA — consolidated unit tests.

Tests are fully offline (synthetic polars DataFrames; no live API calls, no disk fixtures needed).
Run with: uv run pytest tests/rb_eval/test_rb_eval.py -v
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plays(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _synth_model_data(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "rusher_player_name": [f"R{i % 10}" for i in range(n)],
        "season": [2010 + i % 4 for i in range(n)],
        "epa_per_play": rng.normal(0.0, 0.3, n).tolist(),
        "success": rng.uniform(0.3, 0.7, n).tolist(),
        "target": rng.normal(0.0, 0.2, n).tolist(),
        "highlight_yards": rng.uniform(0.0, 2.0, n).tolist(),
        "weight": rng.uniform(100.0, 300.0, n).tolist(),
    })


# ---------------------------------------------------------------------------
# 1. fo_success tests
# ---------------------------------------------------------------------------

def test_fo_success_down1():
    """Down 1: success if yards_gained >= 0.5 * distance (50% rule)."""
    from rb_eval.features import fo_success

    df = _plays([
        {"down": 1, "distance": 9, "yards_gained": 5},   # 5 >= 4.5 → True
        {"down": 1, "distance": 9, "yards_gained": 4},   # 4 < 4.5 → False
    ])
    result = fo_success(df)
    assert result["fo_success"].to_list() == [True, False]


def test_fo_success_down1_fail():
    """Down 1: yards_gained=4, distance=9 → 4 < 4.5 → False."""
    from rb_eval.features import fo_success

    df = _plays([{"down": 1, "distance": 9, "yards_gained": 4}])
    result = fo_success(df)
    assert result["fo_success"].to_list() == [False]


def test_fo_success_down2():
    """Down 2: success if yards_gained >= 0.7 * distance (70% rule)."""
    from rb_eval.features import fo_success

    df = _plays([
        {"down": 2, "distance": 9, "yards_gained": 7},   # 7 >= 6.3 → True
        {"down": 2, "distance": 9, "yards_gained": 6},   # 6 < 6.3 → False
    ])
    result = fo_success(df)
    assert result["fo_success"].to_list() == [True, False]


def test_fo_success_down3_always_false():
    """Down 3: cfb4th parity — excluded from fo_success calculation (False)."""
    from rb_eval.features import fo_success

    # Per cfb4th parity, down 3 is not included in the 50%/70% rules.
    # The spec says: .when(down>=4, yds>=dist).otherwise(False)
    # → down 3 falls to otherwise(False) regardless of yards.
    df = _plays([
        {"down": 3, "distance": 3, "yards_gained": 10},   # plenty of yards, still False
        {"down": 3, "distance": 3, "yards_gained": 3},    # exactly distance, still False
        {"down": 3, "distance": 3, "yards_gained": 0},    # 0, False
    ])
    result = fo_success(df)
    assert result["fo_success"].to_list() == [False, False, False]


# ---------------------------------------------------------------------------
# 2. epa_clamp tests
# ---------------------------------------------------------------------------

def test_epa_clamp_below():
    """epa=-5.0 → epa_clamped=-4.5 (clamped at floor)."""
    from rb_eval.features import clamp_epa

    df = _plays([{"epa": -5.0}])
    result = clamp_epa(df)
    assert math.isclose(result["epa_clamped"][0], -4.5)


def test_epa_clamp_at_floor():
    """epa=-4.5 → epa_clamped=-4.5 (exactly at floor, no change)."""
    from rb_eval.features import clamp_epa

    df = _plays([{"epa": -4.5}])
    result = clamp_epa(df)
    assert math.isclose(result["epa_clamped"][0], -4.5)


def test_epa_clamp_above_floor():
    """epa=-4.0 → epa_clamped=-4.0 (above floor, no change)."""
    from rb_eval.features import clamp_epa

    df = _plays([{"epa": -4.0}])
    result = clamp_epa(df)
    assert math.isclose(result["epa_clamped"][0], -4.0)


def test_epa_clamp_positive():
    """epa=1.0 → epa_clamped=1.0 (positive, no change)."""
    from rb_eval.features import clamp_epa

    df = _plays([{"epa": 1.0}])
    result = clamp_epa(df)
    assert math.isclose(result["epa_clamped"][0], 1.0)


# ---------------------------------------------------------------------------
# 3. n_plays filter test
# ---------------------------------------------------------------------------

def test_n_plays_filter():
    """aggregate 200 plays for rusher A, 50 for B → only A in output."""
    from rb_eval.features import fo_success, clamp_epa

    def _make_plays(rusher: str, n: int, season: int = 2020) -> list[dict]:
        return [
            {
                "rusher_player_name": rusher,
                "season": season,
                "down": 1,
                "distance": 10,
                "yards_gained": 5,
                "epa": 0.3,
            }
            for _ in range(n)
        ]

    rows = _make_plays("A", 200) + _make_plays("B", 50)
    df = pl.DataFrame(rows)
    df = fo_success(df)
    df = clamp_epa(df)

    from rb_eval.features import aggregate_per_rusher

    result = aggregate_per_rusher(df)
    names = result["rusher_player_name"].to_list()
    assert "A" in names
    assert "B" not in names


# ---------------------------------------------------------------------------
# 4. Pythagorean weight test
# ---------------------------------------------------------------------------

def test_pythagorean_weight():
    """weight = sqrt(n^2 + lag_n^2); n=200, lag_n=150 → ~250.0."""
    from rb_eval.features import aggregate_per_rusher

    # Build two seasons for the same rusher so lag is populated.
    def _make_plays(season: int, n: int) -> list[dict]:
        return [
            {
                "rusher_player_name": "Alpha",
                "season": season,
                "down": 1,
                "distance": 10,
                "yards_gained": 5,
                "epa": 0.3,
                "fo_success": True,
                "epa_clamped": 0.3,
            }
            for _ in range(n)
        ]

    # Season 2019: 150 plays; Season 2020: 200 plays
    df = pl.DataFrame(_make_plays(2019, 150) + _make_plays(2020, 200))

    from rb_eval.model import add_lag_features, add_weight

    agg = aggregate_per_rusher(df)
    lagged = add_lag_features(agg)
    weighted = add_weight(lagged)

    row_2020 = weighted.filter(pl.col("season") == 2020)
    assert row_2020.height == 1
    w = row_2020["weight"][0]
    expected = math.sqrt(200**2 + 150**2)
    assert math.isclose(w, expected, rel_tol=1e-6), f"expected {expected}, got {w}"


# ---------------------------------------------------------------------------
# 5. Lag features test
# ---------------------------------------------------------------------------

def test_lag_features():
    """season 2020 epa_per_play used as lag for season 2021 row."""
    from rb_eval.model import add_lag_features

    df = pl.DataFrame([
        {"rusher_player_name": "Alpha", "season": 2020, "n_plays": 150,
         "epa_per_play": 0.25, "success": 0.45},
        {"rusher_player_name": "Alpha", "season": 2021, "n_plays": 180,
         "epa_per_play": 0.30, "success": 0.50},
    ])

    result = add_lag_features(df)
    row_2021 = result.filter(pl.col("season") == 2021).row(0, named=True)

    assert math.isclose(row_2021["lepa"], 0.25, rel_tol=1e-9), \
        f"Expected lepa=0.25, got {row_2021['lepa']}"
    assert math.isclose(row_2021["lsuccess"], 0.45, rel_tol=1e-9), \
        f"Expected lsuccess=0.45, got {row_2021['lsuccess']}"
    assert row_2021["lplays"] == 150, \
        f"Expected lplays=150, got {row_2021['lplays']}"


# ---------------------------------------------------------------------------
# 6. GAM fit test
# ---------------------------------------------------------------------------

def test_gam_fit():
    """Create 20-row df; fit GAM; predict returns 20 values."""
    pytest.importorskip("pygam", reason="pygam not installed — run: uv sync --group gam")
    from rb_eval.model import fit_rb_eval_model

    rng = np.random.default_rng(99)
    n = 20
    X = np.column_stack([
        rng.normal(0.0, 0.3, n),    # epa_per_play
        rng.uniform(0.3, 0.7, n),   # success
    ])
    y = rng.normal(0.0, 0.2, n)    # unadjusted_epa

    model = fit_rb_eval_model(X, y)
    preds = model.predict(X)

    assert len(preds) == n, f"Expected {n} predictions, got {len(preds)}"
    assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"
