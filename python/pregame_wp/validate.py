"""Validation metrics for the Pregame WP model (Track 4).

Computes MAE, RMSE on predicted point differential, and Brier score
on predicted win probability vs actual outcome.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from xgboost import XGBRegressor

from .wp import pregame_wp_from_pred


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between predicted and actual point differentials.

    Args:
        y_true: Actual point differentials.
        y_pred: Predicted point differentials.

    Returns:
        MAE as a float.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between predicted and actual point differentials.

    Args:
        y_true: Actual point differentials.
        y_pred: Predicted point differentials.

    Returns:
        RMSE as a float.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def brier_score(
    y_outcome: np.ndarray,
    y_wp: np.ndarray,
) -> float:
    """Brier score: mean((predicted WP - actual outcome)^2).

    Lower is better. Perfect calibration: 0.0.

    Args:
        y_outcome: Actual binary outcomes (1 = team won, 0 = team lost).
        y_wp: Predicted win probabilities in [0, 1].

    Returns:
        Brier score as a float.
    """
    return float(np.mean((y_wp - y_outcome) ** 2))


def validate_model(
    model: XGBRegressor,
    df: pl.DataFrame,
    std: float,
    mu: float = 0.0,
) -> dict[str, float]:
    """Compute validation metrics on a held-out (or full) DataFrame.

    Args:
        model: Fitted XGBRegressor.
        df: DataFrame with columns '5FRDiff', 'PtsDiff', and optionally 'outcome'
            (1 if the team with positive 5FRDiff won, 0 otherwise).
        std: Standard deviation for WP CDF normalization.
        mu: Center of normal distribution (default 0.0 per OQ-7).

    Returns:
        dict with keys 'mae', 'rmse', and optionally 'brier_score'.
    """
    X = df[["5FRDiff"]].to_numpy()
    y_true = df["PtsDiff"].to_numpy()

    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    result: dict[str, float] = {"mae": mae, "rmse": rmse}

    if "outcome" in df.columns:
        y_outcome = df["outcome"].to_numpy().astype(float)
        y_wp = np.array([pregame_wp_from_pred(float(p), std, mu) for p in y_pred])
        result["brier_score"] = brier_score(y_outcome, y_wp)

    return result
