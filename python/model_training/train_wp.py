"""WP trainers (spread + naive). Shipped recipe = cfbscrapR-wpa.ipynb (no sample weights)."""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import wp_matrix

_PARAMS = {"spread": (C.WP_SPREAD_PARAMS, C.WP_SPREAD_NROUNDS),
           "naive": (C.WP_NAIVE_PARAMS, C.WP_NAIVE_NROUNDS)}
_STAGE1 = {"spread": (C.WP_SPREAD_PARAMS_STAGE1, C.WP_SPREAD_NROUNDS_STAGE1)}


def train_wp(df: pl.DataFrame, variant: str = "spread", stage: int = 2,
             nrounds: int | None = None) -> xgb.Booster:
    if stage == 1 and variant in _STAGE1:
        params, default_rounds = _STAGE1[variant]
    else:
        params, default_rounds = _PARAMS[variant]
    X, y, _ = wp_matrix(df, variant=variant)
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(params, dtrain, num_boost_round=nrounds or default_rounds)
