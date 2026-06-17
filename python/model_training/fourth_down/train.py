"""Fourth-down yards-gained model trainer.

Trains the 5-feature, 76-class multi:softprob XGBoost model that projects yards gained
on any 3rd/4th-down play. No sample weights (the original model trains without them,
unlike the EP/WP models in Track 1). Feature input is the X pandas DataFrame returned
by fd_features(); the caller is responsible for the df -> (X, y) split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from .constants import FD_FEATURES, FD_NROUNDS, FD_PARAMS


def train_fourth_down(
    X: pd.DataFrame,
    y: np.ndarray,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Train the fourth-down yards-gained model.

    Args:
        X: Feature matrix with exactly FD_FEATURES columns in the correct order.
        y: Integer label array (class 0..75).
        nrounds: Number of boosting rounds (default 157, the confirmed recipe value).

    Returns:
        Trained xgboost.Booster with multi:softprob objective, 5 features, 76 classes.
    """
    dtrain = xgb.DMatrix(X[FD_FEATURES], label=y)
    return xgb.train(FD_PARAMS, dtrain, num_boost_round=nrounds)


def train_from_plays(
    plays: pl.DataFrame,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Filter plays, build features, and train in one step.

    Args:
        plays: polars DataFrame of final.json play records.
        nrounds: Number of boosting rounds.

    Returns:
        Trained Booster.

    Raises:
        ValueError: if no training rows survive the fourth-down feature filter.
    """
    from .features import fd_features

    X, y = fd_features(plays)
    if len(X) == 0:
        raise ValueError(
            "No training rows survived the fourth-down feature filter. "
            "Check that plays include 3rd/4th-down rush/pass rows with "
            "overUnder, homeTeamSpread, and yardsGained present."
        )
    return train_fourth_down(X, y, nrounds=nrounds)
