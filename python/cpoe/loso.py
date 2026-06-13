"""Leave-One-Season-Out cross-validation for the CFB CP model.

For each held-out season:
  1. Train on all other seasons.
  2. Predict CP probabilities on the held-out season.
  3. Record log-loss, Brier score, and play count.

Input DataFrame must have a ``season`` column plus FEATURE_COLS + TARGET_COL.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from .constants import FEATURE_COLS, TARGET_COL, XGB_NROUNDS, XGB_PARAMS
from .train_cp import train_cp_model


def run_loso_cv(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    return_preds: bool = False,
    nrounds: int = XGB_NROUNDS,
    params: dict | None = None,
) -> dict[str, Any]:
    """Run LOSO cross-validation.

    Args:
        df: DataFrame with ``season_col``, FEATURE_COLS, and TARGET_COL.
        season_col: Column that identifies the season (int year).
        return_preds: If True, each fold record includes a ``cp_pred``
            array of length ``n_plays``.
        nrounds: Boosting rounds per fold.
        params: XGBoost params dict (defaults to constants.XGB_PARAMS).

    Returns:
        Dict with keys:
            ``folds``   — list of per-fold result dicts.
            ``summary`` — dict with ``mean_log_loss``, ``mean_brier_score``.

    Raises:
        ValueError: If fewer than 2 distinct seasons are present.
    """
    seasons = sorted(df[season_col].unique())
    if len(seasons) < 2:
        raise ValueError(
            f"run_loso_cv requires at least 2 distinct seasons; got {seasons}."
        )

    folds: list[dict[str, Any]] = []

    for held_out in seasons:
        train_df = df[df[season_col] != held_out]
        test_df = df[df[season_col] == held_out]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL].to_numpy()

        booster = train_cp_model(X_train, y_train, nrounds=nrounds, params=params)
        preds = booster.predict(xgb.DMatrix(X_test))

        # clip for numerical safety
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)

        fold: dict[str, Any] = {
            "season": int(held_out),
            "n_plays": len(y_test),
            "log_loss": float(log_loss(y_test, preds_clipped)),
            "brier_score": float(brier_score_loss(y_test, preds_clipped)),
        }
        if return_preds:
            fold["cp_pred"] = preds.tolist()

        folds.append(fold)

    mean_log_loss = float(np.mean([f["log_loss"] for f in folds]))
    mean_brier = float(np.mean([f["brier_score"] for f in folds]))

    return {
        "folds": folds,
        "summary": {
            "mean_log_loss": mean_log_loss,
            "mean_brier_score": mean_brier,
            "n_seasons": len(seasons),
        },
    }
