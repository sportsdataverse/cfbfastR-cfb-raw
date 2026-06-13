"""Phase 4 — cpoe module tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL


@pytest.fixture()
def trained_booster():
    """Train a tiny booster on synthetic data."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        col: rng.integers(0, 10, n) for col in FEATURE_COLS
    })
    df["completion"] = rng.integers(0, 2, n)
    from cpoe.train_cp import train_cp_model
    return train_cp_model(df[FEATURE_COLS], df[TARGET_COL])


@pytest.fixture()
def pass_plays_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 20
    df = pd.DataFrame({col: rng.integers(0, 10, n) for col in FEATURE_COLS})
    df["completion"] = rng.integers(0, 2, n)
    return df


def test_cpoe_imports():
    from cpoe.cpoe import compute_cpoe  # noqa: F401


def test_compute_cpoe_returns_dataframe(pass_plays_df, trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    assert isinstance(result, pd.DataFrame)


def test_compute_cpoe_has_cp_pred_col(pass_plays_df, trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    assert "cp_pred" in result.columns


def test_compute_cpoe_has_cpoe_col(pass_plays_df, trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    assert "cpoe" in result.columns


def test_cpoe_length_matches_input(pass_plays_df, trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    assert len(result) == len(pass_plays_df)


def test_cp_pred_in_unit_interval(pass_plays_df, trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    assert (result["cp_pred"] >= 0.0).all()
    assert (result["cp_pred"] <= 1.0).all()


def test_cpoe_arithmetic(pass_plays_df, trained_booster):
    """cpoe == completion - cp_pred for every row."""
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pass_plays_df, trained_booster)
    expected = result["completion"] - result["cp_pred"]
    np.testing.assert_array_almost_equal(result["cpoe"].values, expected.values)


def test_cpoe_mean_near_zero_on_large_sample(trained_booster):
    """On a balanced random sample, mean CPOE should be close to zero."""
    from cpoe.cpoe import compute_cpoe
    rng = np.random.default_rng(99)
    n = 500
    df = pd.DataFrame({col: rng.integers(0, 10, n) for col in FEATURE_COLS})
    # balanced completions
    df["completion"] = (rng.random(n) > 0.5).astype(int)
    result = compute_cpoe(df, trained_booster)
    assert abs(result["cpoe"].mean()) < 0.5  # sanity check only


def test_compute_cpoe_empty_input(trained_booster):
    from cpoe.cpoe import compute_cpoe
    result = compute_cpoe(pd.DataFrame(), trained_booster)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
