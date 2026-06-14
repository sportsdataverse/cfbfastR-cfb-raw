"""Phase 5 Task 5.1 — validate module tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def pred_df() -> pd.DataFrame:
    """20-row DF with synthetic CP predictions and actual completions."""
    rng = np.random.default_rng(42)
    n = 20
    cp_pred = rng.random(n)
    completion = (rng.random(n) > 0.5).astype(int)
    return pd.DataFrame({"cp_pred": cp_pred, "completion": completion})


@pytest.fixture()
def trained_booster_and_df():
    from cpoe.constants import FEATURE_COLS, TARGET_COL
    from cpoe.train_cp import train_cp_model
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({col: rng.integers(0, 10, n) for col in FEATURE_COLS})
    df[TARGET_COL] = rng.integers(0, 2, n)
    booster = train_cp_model(df[FEATURE_COLS], df[TARGET_COL])
    return booster, df


def test_validate_imports():
    from cpoe.validate import calibration_metrics  # noqa: F401


def test_calibration_metrics_returns_dict(pred_df):
    from cpoe.validate import calibration_metrics
    result = calibration_metrics(pred_df["completion"], pred_df["cp_pred"])
    assert isinstance(result, dict)


def test_calibration_metrics_has_required_keys(pred_df):
    from cpoe.validate import calibration_metrics
    result = calibration_metrics(pred_df["completion"], pred_df["cp_pred"])
    for key in ("log_loss", "brier_score", "n"):
        assert key in result, f"Missing key: {key}"


def test_calibration_log_loss_non_negative(pred_df):
    from cpoe.validate import calibration_metrics
    result = calibration_metrics(pred_df["completion"], pred_df["cp_pred"])
    assert result["log_loss"] >= 0.0


def test_calibration_brier_score_range(pred_df):
    from cpoe.validate import calibration_metrics
    result = calibration_metrics(pred_df["completion"], pred_df["cp_pred"])
    assert 0.0 <= result["brier_score"] <= 1.0


def test_calibration_n_matches_input(pred_df):
    from cpoe.validate import calibration_metrics
    result = calibration_metrics(pred_df["completion"], pred_df["cp_pred"])
    assert result["n"] == len(pred_df)


def test_perfect_classifier_low_loss():
    from cpoe.validate import calibration_metrics
    y = np.array([1, 1, 0, 0])
    p = np.array([0.99, 0.99, 0.01, 0.01])
    result = calibration_metrics(y, p)
    assert result["log_loss"] < 0.1
    assert result["brier_score"] < 0.01


def test_calibration_bins_returns_dataframe(pred_df):
    from cpoe.validate import calibration_bins
    df = calibration_bins(pred_df["completion"], pred_df["cp_pred"])
    assert isinstance(df, pd.DataFrame)
    assert "bin_mid" in df.columns
    assert "actual_rate" in df.columns
    assert "pred_rate" in df.columns
    assert "n" in df.columns


def test_shapley_importance_imports():
    from cpoe.validate import feature_importance  # noqa: F401


def test_feature_importance_returns_dict(trained_booster_and_df):
    from cpoe.validate import feature_importance
    booster, _ = trained_booster_and_df
    result = feature_importance(booster)
    assert isinstance(result, dict)
    assert len(result) > 0
