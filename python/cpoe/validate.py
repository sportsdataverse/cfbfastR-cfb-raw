"""Calibration metrics, binned calibration curve, and feature importance.

Used for post-training diagnostics of the CFB CP model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss


def calibration_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute scalar calibration metrics.

    Args:
        y_true: Binary completion labels (0/1).
        y_pred: Predicted CP probabilities [0, 1].

    Returns:
        Dict with keys ``log_loss``, ``brier_score``, ``n``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def calibration_bins(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute binned calibration curve data.

    Splits predicted probabilities into ``n_bins`` equal-width bins
    and computes the mean actual completion rate vs. mean predicted rate.

    Args:
        y_true: Binary completion labels.
        y_pred: Predicted CP probabilities.
        n_bins: Number of bins (default: 10).

    Returns:
        DataFrame with columns: ``bin_mid``, ``actual_rate``, ``pred_rate``, ``n``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if lo == edges[-2]:  # include right edge on last bin
            mask = (y_pred >= lo) & (y_pred <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin_mid": float((lo + hi) / 2),
                "actual_rate": float(y_true[mask].mean()),
                "pred_rate": float(y_pred[mask].mean()),
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def feature_importance(
    booster: xgb.Booster,
    *,
    importance_type: str = "gain",
) -> dict[str, float]:
    """Extract XGBoost built-in feature importance scores.

    Args:
        booster: Trained XGBoost Booster.
        importance_type: One of ``"gain"``, ``"weight"``, ``"cover"``.

    Returns:
        Dict mapping feature name → importance score (sorted descending by score).
    """
    raw = booster.get_score(importance_type=importance_type)
    return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))
