"""QBR trainer: 6-feat reg:squarederror predicting ESPN raw QBR (shipped qbr_model.ubj).

Features = per-QB-game qbr_vars (from features.qbr_matrix); target = ESPN raw QBR joined
on (game_id, passer_player_name). The ESPN QBR is produced by python/scrape_cfb_qbr.py.
"""
from __future__ import annotations

import pandas as pd
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import qbr_matrix


def train_qbr_from_matrix(X: pd.DataFrame, y, nrounds: int = C.QBR_NROUNDS) -> xgb.Booster:
    dtrain = xgb.DMatrix(X[C.QBR_FEATURES], label=y)
    return xgb.train(C.QBR_PARAMS, dtrain, num_boost_round=nrounds)


def train_qbr(df: pl.DataFrame, espn_qbr: pl.DataFrame, nrounds: int = C.QBR_NROUNDS) -> xgb.Booster:
    X, _, keys = qbr_matrix(df)
    feat = pl.from_pandas(keys).hstack(pl.from_pandas(X))
    joined = feat.join(
        espn_qbr.select(["game_id", "passer_player_name", "raw_qbr"]),
        on=["game_id", "passer_player_name"], how="inner",
    ).drop_nulls("raw_qbr")
    Xj = joined.select(C.QBR_FEATURES).to_pandas()
    yj = joined["raw_qbr"].to_numpy()
    return train_qbr_from_matrix(Xj, yj, nrounds=nrounds)
