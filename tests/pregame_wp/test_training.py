import numpy as np
import pandas as pd
import pytest
from pregame_wp.training import filter_outliers, train_pgwp_model


def _make_training_df(n=500, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "5FRDiff": rng.normal(0, 2, n),
        "PtsDiff": rng.normal(0, 14, n),
    })


def test_filter_outliers_removes_extreme_rows():
    df = _make_training_df(200, seed=0)
    df.loc[0, "5FRDiff"] = 100.0
    filtered = filter_outliers(df)
    assert len(filtered) < len(df)
    assert filtered["5FRDiff"].max() < 100.0


def test_filter_outliers_keeps_normal_rows():
    df = _make_training_df(200, seed=2)
    filtered = filter_outliers(df)
    # most rows should survive (no extreme outliers in seed=2 normal sample)
    assert len(filtered) > 180


def test_train_returns_xgb_model():
    import xgboost as xgb
    df = _make_training_df(500)
    model, mu, std = train_pgwp_model(df)
    assert isinstance(model, xgb.XGBRegressor)


def test_train_n_estimators_10():
    df = _make_training_df(500)
    model, mu, std = train_pgwp_model(df)
    assert model.n_estimators == 10


def test_train_mu_is_zero():
    # OQ-7 resolution: mu = 0.0 (symmetric point-diff)
    df = _make_training_df(500)
    model, mu, std = train_pgwp_model(df)
    assert mu == 0.0


def test_train_std_positive():
    df = _make_training_df(500)
    model, mu, std = train_pgwp_model(df)
    assert std > 0


def test_train_std_from_full_training_preds():
    # std is derived from full training set predictions, not a test split
    df = _make_training_df(500)
    model, mu, std = train_pgwp_model(df)
    # Verify std matches std of model predictions on the same df
    preds = model.predict(df[["5FRDiff"]])
    assert abs(std - float(np.std(preds))) < 1e-6
