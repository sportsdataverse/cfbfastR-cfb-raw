"""Phase 2 Task 2.2 — train_cp module tests."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL


@pytest.fixture()
def tiny_df() -> pd.DataFrame:
    """40-row synthetic dataset — enough for XGBoost to train without error."""
    rng = np.random.default_rng(42)
    n = 40
    data = {
        "down": rng.integers(1, 5, n),
        "distance": rng.integers(1, 20, n),
        "yards_to_goal": rng.integers(1, 100, n),
        "score_diff": rng.integers(-28, 29, n),
        "seconds_remaining": rng.integers(0, 3600, n),
        "is_home": rng.integers(0, 2, n),
        "period": rng.integers(1, 5, n),
        "passing_down": rng.integers(0, 2, n),
        "completion": rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


def test_train_cp_imports():
    from cpoe.train_cp import train_cp_model  # noqa: F401


def test_train_returns_booster(tiny_df):
    import xgboost as xgb
    from cpoe.train_cp import train_cp_model
    X = tiny_df[FEATURE_COLS]
    y = tiny_df[TARGET_COL]
    booster = train_cp_model(X, y)
    assert isinstance(booster, xgb.Booster)


def test_predict_returns_probabilities(tiny_df):
    import xgboost as xgb
    from cpoe.train_cp import train_cp_model
    X = tiny_df[FEATURE_COLS]
    y = tiny_df[TARGET_COL]
    booster = train_cp_model(X, y)
    dmat = xgb.DMatrix(X)
    preds = booster.predict(dmat)
    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_save_and_load_round_trip(tiny_df, tmp_path: pathlib.Path):
    import xgboost as xgb
    from cpoe.train_cp import load_cp_model, save_cp_model, train_cp_model
    X = tiny_df[FEATURE_COLS]
    y = tiny_df[TARGET_COL]
    booster = train_cp_model(X, y)
    model_path = tmp_path / "test_cp.ubj"
    save_cp_model(booster, model_path)
    assert model_path.exists()
    loaded = load_cp_model(model_path)
    assert isinstance(loaded, xgb.Booster)
    # predictions must be identical after round-trip
    dmat = xgb.DMatrix(X)
    np.testing.assert_array_equal(booster.predict(dmat), loaded.predict(dmat))


def test_load_missing_raises(tmp_path: pathlib.Path):
    from cpoe.train_cp import load_cp_model
    with pytest.raises(FileNotFoundError):
        load_cp_model(tmp_path / "nonexistent.ubj")


def test_train_cp_model_accepts_numpy(tiny_df):
    """train_cp_model must accept bare numpy arrays in addition to DataFrames."""
    from cpoe.train_cp import train_cp_model
    import xgboost as xgb
    X = tiny_df[FEATURE_COLS].to_numpy()
    y = tiny_df[TARGET_COL].to_numpy()
    booster = train_cp_model(X, y)
    assert isinstance(booster, xgb.Booster)
