"""Compute CPOE = Completion - Predicted Completion Probability.

CPOE > 0  → QB completed passes at a higher rate than expected.
CPOE < 0  → QB completed passes at a lower rate than expected.

This is purely arithmetic once we have the CP model predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import FEATURE_COLS, TARGET_COL


def compute_cpoe(
    df: pd.DataFrame,
    booster: xgb.Booster,
) -> pd.DataFrame:
    """Add ``cp_pred`` and ``cpoe`` columns to a pass-play DataFrame.

    Args:
        df: DataFrame with FEATURE_COLS + TARGET_COL (from
            ``extract_pass_features``).
        booster: Trained XGBoost Booster (from ``train_cp_model``).

    Returns:
        Copy of ``df`` with two additional columns:
            ``cp_pred`` — predicted completion probability [0, 1].
            ``cpoe``    — actual completion minus ``cp_pred``.
        Empty DataFrame (zero rows) if ``df`` is empty.
    """
    if df.empty:
        return pd.DataFrame()

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    dmat = xgb.DMatrix(df[feature_cols])
    preds = booster.predict(dmat)

    out = df.copy()
    out["cp_pred"] = preds.astype(float)

    if TARGET_COL in out.columns:
        out["cpoe"] = out[TARGET_COL].astype(float) - out["cp_pred"]
    else:
        out["cpoe"] = np.nan

    return out.reset_index(drop=True)
