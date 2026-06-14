"""XGBoost CP model training, save, and load helpers (Track 5, Approach A).

The model mirrors the nflfastR `cpoe_model.R` hyper-parameters exactly
(binary:logistic, eta=0.025, max_depth=4, etc.).  See constants.XGB_PARAMS.
"""
from __future__ import annotations

import pathlib
from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb

from .constants import FEATURE_COLS, TARGET_COL, XGB_NROUNDS, XGB_PARAMS

ArrayLike = Union[pd.DataFrame, np.ndarray]


def train_cp_model(
    X: ArrayLike,
    y: ArrayLike,
    *,
    nrounds: int = XGB_NROUNDS,
    params: dict | None = None,
    verbose_eval: bool = False,
) -> xgb.Booster:
    """Train the CP model and return the fitted XGBoost Booster.

    Args:
        X: Feature matrix (n_plays × len(FEATURE_COLS)).  DataFrame or ndarray.
        y: Binary completion labels (0/1).  Series or 1-D ndarray.
        nrounds: Number of boosting rounds (default: XGB_NROUNDS = 560).
        params: XGBoost parameter dict.  Defaults to constants.XGB_PARAMS.
        verbose_eval: Print eval log every round if True.

    Returns:
        Fitted ``xgb.Booster``.
    """
    _params = dict(XGB_PARAMS if params is None else params)
    dmat = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_COLS) if isinstance(X, np.ndarray) else None)
    booster = xgb.train(
        _params,
        dmat,
        num_boost_round=nrounds,
        verbose_eval=verbose_eval,
    )
    return booster


def save_cp_model(booster: xgb.Booster, path: pathlib.Path | str) -> None:
    """Save a trained Booster to an UBJ file.

    Args:
        booster: Fitted XGBoost Booster.
        path: Destination path (conventionally ``cfb_cp_model.ubj``).
    """
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p))


def load_cp_model(path: pathlib.Path | str) -> xgb.Booster:
    """Load a saved CP model from disk.

    Args:
        path: Path to a UBJ (or JSON/bin) XGBoost model file.

    Returns:
        Loaded ``xgb.Booster``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CP model not found: {p}")
    booster = xgb.Booster()
    booster.load_model(str(p))
    return booster
