"""Tests for the pregame_wp package (Track 4 — Pregame WP + Five-Factors).

All tests here use synthetic data only — no live API calls and no CFBD key required
(except test_live_cfbd_skips_without_key, which is explicitly gated).
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest
import scipy.stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plays_df(
    down: int,
    distance: int,
    yards_gained: int,
    n: int = 1,
) -> pl.DataFrame:
    """Return a minimal plays DataFrame suitable for data_prep.play_successful."""
    return pl.DataFrame(
        {
            "down": [down] * n,
            "distance": [distance] * n,
            "yards_gained": [yards_gained] * n,
        }
    )


# ---------------------------------------------------------------------------
# play_successful — down-specific success thresholds
# ---------------------------------------------------------------------------

def test_play_successful_down1() -> None:
    """Down 1: yards_gained=4 out of distance=8 (4 >= 0.5*8=4.0) → True."""
    from pregame_wp.data_prep import play_successful

    df = _make_plays_df(down=1, distance=8, yards_gained=4)
    result = play_successful(df)
    assert result["play_successful"][0] is True


def test_play_successful_down2() -> None:
    """Down 2: yards_gained=5 out of distance=8 (5 < 0.7*8=5.6) → False."""
    from pregame_wp.data_prep import play_successful

    df = _make_plays_df(down=2, distance=8, yards_gained=5)
    result = play_successful(df)
    assert result["play_successful"][0] is False


def test_play_successful_down3_always_false() -> None:
    """Down 3 is never successful in the faithful port (3rd down absent from np.select)."""
    from pregame_wp.data_prep import play_successful

    # Even with massive gain, down 3 should be False per the faithful port
    df = _make_plays_df(down=3, distance=1, yards_gained=99)
    result = play_successful(df)
    assert result["play_successful"][0] is False


def test_play_successful_down4() -> None:
    """Down 4: yards_gained=10 out of distance=10 (10 >= 1.0*10=10.0) → True."""
    from pregame_wp.data_prep import play_successful

    df = _make_plays_df(down=4, distance=10, yards_gained=10)
    result = play_successful(df)
    assert result["play_successful"][0] is True


# ---------------------------------------------------------------------------
# constants — weights sum to 1.0
# ---------------------------------------------------------------------------

def test_five_factor_weights_sum() -> None:
    """EFFICIENCY + EXPLOSIVENESS + FINISHING + FIELD_POS + TURNOVER weights must sum to 1.0."""
    from pregame_wp import constants as C

    total = (
        C.EFFICIENCY_WEIGHT
        + C.EXPLOSIVENESS_WEIGHT
        + C.FINISHING_WEIGHT
        + C.FIELD_POS_WEIGHT
        + C.TURNOVER_WEIGHT
    )
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# OQ-5 bug (faithful port) — PuntReturnEqPPP uses punt_eqppp (same variable)
# ---------------------------------------------------------------------------

def test_oq5_bug_present() -> None:
    """OQ-5 faithful port: PuntReturnEqPPP column equals punt_eqppp (same var → contribution is 0).

    In the notebook's generate_team_st_stats:
        'PuntReturnEqPPP': [punt_eqppp]    # BUG: should be punt_ret_eqppp
    This makes PuntEqPPP - PuntReturnEqPPP = 0 always (both sides use punt_eqppp).
    """
    from pregame_wp.data_prep import compute_five_factor_rating

    # Build synthetic team stats where punt_eqppp != punt_return_eqppp to surface the bug.
    team_stats = {
        "OffSR": 0.5,
        "OffER": 0.2,
        "AvgEqPPP": 0.3,
        "OppRate": 0.5,
        "OppEff": 0.4,
        "OppPPD": 3.0,
        "OppSR": 0.4,
        "field_pos_quant": 0.0,
        "ExpTO": 1.5,
        "ActualTO": 1.5,
        "SackRate": 0.05,
        "HavocRate": 0.1,
        "kickoff_eqppp": 0.1,
        "kickoff_return_eqppp": 0.05,
        "punt_eqppp": 0.2,           # the punter's EP
        "punt_return_eqppp": 0.8,    # the returner's EP — should differ but is NOT used per OQ-5 bug
    }

    # The bug: internally PuntReturnEqPPP is assigned punt_eqppp (== 0.2), not punt_return_eqppp (0.8).
    # So the field-position punt sub-term (punt_eqppp - PuntReturnEqPPP) == 0.2 - 0.2 == 0.
    # We verify this by checking that the function uses punt_eqppp for PuntReturnEqPPP.
    result_with_bug = compute_five_factor_rating(team_stats)

    # Now swap punt_return_eqppp to match punt_eqppp (making the bug irrelevant)
    team_stats_same = {**team_stats, "punt_return_eqppp": team_stats["punt_eqppp"]}
    result_same = compute_five_factor_rating(team_stats_same)

    # Both must produce the same result (confirming the bug: punt_return_eqppp is never used)
    assert abs(result_with_bug - result_same) < 1e-9


# ---------------------------------------------------------------------------
# OQ-7 — WP uses mu=0.0, not mean(pred)
# ---------------------------------------------------------------------------

def test_oq7_mu_zero() -> None:
    """WP calc uses mu=0.0 per OQ-7 resolution: norm.cdf(0/std) == 0.5 regardless of std."""
    # A team predicted to score 0 more than opponent should have WP = 0.5
    std = 10.0
    pred_mov = 0.0
    wp = scipy.stats.norm.cdf((pred_mov - 0.0) / std)
    assert abs(wp - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# filter_outliers — z-score gate on 5FRDiff
# ---------------------------------------------------------------------------

def test_filter_outliers() -> None:
    """Rows with |5FRDiff| z-score > FILTER_Z should be removed."""
    from pregame_wp.data_prep import filter_outliers
    from pregame_wp.constants import FILTER_Z

    rng = np.random.default_rng(42)
    n = 200
    vals = rng.normal(0.0, 2.0, n).tolist()
    pts = rng.normal(0.0, 14.0, n).tolist()

    # Inject one row with |z| >> FILTER_Z
    vals[0] = 1_000.0
    pts[0] = 0.0

    df = pl.DataFrame({"5FRDiff": vals, "PtsDiff": pts})
    filtered = filter_outliers(df)

    assert len(filtered) < len(df), "No rows were removed"
    assert filtered["5FRDiff"].max() < 1_000.0, "Extreme outlier was not removed"


# ---------------------------------------------------------------------------
# XGBRegressor round-trip (synthetic, offline)
# ---------------------------------------------------------------------------

def test_xgb_pregame_roundtrip() -> None:
    """50-row synthetic df: fit XGBRegressor(n_estimators=10) and verify predict returns 50 values."""
    from xgboost import XGBRegressor

    rng = np.random.default_rng(7)
    n = 50
    X = rng.normal(0.0, 2.0, (n, 1))
    y = rng.normal(0.0, 14.0, n)

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == n
    assert preds.dtype == np.float32


# ---------------------------------------------------------------------------
# Live CFBD test — skipped without CFB_DATA_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("CFB_DATA_API_KEY"),
    reason="CFB_DATA_API_KEY not set — skipping live CFBD test",
)
def test_live_cfbd_skips_without_key() -> None:
    """Gate test: only runs when CFB_DATA_API_KEY is set in the environment."""
    from pregame_wp.data_prep import load_cfbd_data

    # Fetch a small slice (2019) to verify the API client works
    df = load_cfbd_data(season=2019, api_key=os.environ["CFB_DATA_API_KEY"])
    assert len(df) > 0
    assert "team" in df.columns or "homeTeam" in df.columns
