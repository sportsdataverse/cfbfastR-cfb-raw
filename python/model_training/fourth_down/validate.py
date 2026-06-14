"""Validation helpers for the fourth-down yards-gained model.

assert_structure: verifies the 5-feat / 76-class / multi:softprob contract.
calibration_fd:   builds predicted first-down probability vs empirical rate table.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import FD_FEATURES, FD_NUM_CLASS, FD_NROUNDS


def assert_structure(booster: xgb.Booster) -> None:
    """Assert that a Booster matches the fourth-down model's structural contract.

    Checks num_features == 5, num_class == 76, objective == multi:softprob.

    Args:
        booster: An xgboost.Booster to validate.

    Raises:
        AssertionError: if num_features, num_class, or objective does not match.
    """
    cfg = json.loads(booster.save_config())["learner"]
    num_class = int(cfg["learner_model_param"]["num_class"])
    objective = cfg["objective"]["name"]
    n_feats = booster.num_features()

    assert n_feats == 5, (
        f"num_features={n_feats}, expected 5. "
        "Confirm the model was trained on [down, distance, yards_to_goal, "
        "posteam_total, posteam_spread]."
    )
    assert num_class == FD_NUM_CLASS, (
        f"num_class={num_class}, expected {FD_NUM_CLASS}. "
        "Model must cover integer gains -10..65 (76 classes)."
    )
    assert objective == "multi:softprob", (
        f"objective={objective!r}, expected 'multi:softprob'."
    )


def validate_fd_model(
    model: xgb.Booster,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute top-1 accuracy and log-loss for the fourth-down model.

    Args:
        model: Trained fourth-down Booster.
        test_df: pandas DataFrame with FD_FEATURES columns and a 'fd_label' column.

    Returns:
        dict with keys 'top1_accuracy' and 'log_loss'.
    """
    from sklearn.metrics import accuracy_score, log_loss

    X = test_df[FD_FEATURES]
    y_true = test_df["fd_label"].to_numpy()

    dmat = xgb.DMatrix(X)
    raw = model.predict(dmat)
    if raw.ndim == 1:
        probs = raw.reshape(-1, FD_NUM_CLASS)
    else:
        probs = raw

    y_pred = probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, probs, labels=list(range(FD_NUM_CLASS))))

    return {"top1_accuracy": acc, "log_loss": ll}


def calibration_fd(
    booster: xgb.Booster,
    X: pd.DataFrame,
    y_yards: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Predicted first-down probability vs empirical first-down rate.

    For each play with distance d, a first down occurs when actual yards_gained >= d.
    Predicted P(first down) = sum over classes k where (k - 10) >= d of P(class=k).

    Args:
        booster: Trained fourth-down model.
        X: Feature matrix (must include 'distance' column, using FD_FEATURES order).
        y_yards: Actual yards gained for each play (raw yards, not class label).
        n_bins: Number of quantile bins for grouping predicted probability.

    Returns:
        pandas DataFrame with columns:
            bin_center, pred_fd_prob, empirical_fd_rate, n_plays.
    """
    dmat = xgb.DMatrix(X[FD_FEATURES])
    raw = booster.predict(dmat)
    if raw.ndim == 1:
        probs = raw.reshape(-1, FD_NUM_CLASS)
    else:
        probs = raw

    distance = X["distance"].to_numpy()
    n_plays = len(X)

    # class k corresponds to gain = k - 10; first down when gain >= distance
    gains = np.arange(FD_NUM_CLASS) - 10  # [-10, -9, ..., 65]
    pred_fd = np.array([
        probs[i, gains >= distance[i]].sum()
        for i in range(n_plays)
    ])
    empirical_fd = (y_yards >= distance).astype(float)

    # bin by predicted probability quantile
    bins = np.quantile(pred_fd, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.array([0.0, 1.0])
    bin_idx = np.searchsorted(bins, pred_fd, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    rows = []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": float((bins[b] + bins[b + 1]) / 2),
            "pred_fd_prob": float(pred_fd[mask].mean()),
            "empirical_fd_rate": float(empirical_fd[mask].mean()),
            "n_plays": int(mask.sum()),
        })
    return pd.DataFrame(rows)
