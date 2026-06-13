"""EP model trainer (port of keepers 02_epa_xgb_model.R / model_training.R)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import ep_matrix


def train_ep(df: pl.DataFrame, nrounds: int = C.EP_NROUNDS) -> xgb.Booster:
    X, y, w = ep_matrix(df)
    dtrain = xgb.DMatrix(X, label=y, weight=w)
    return xgb.train(C.EP_PARAMS, dtrain, num_boost_round=nrounds)
