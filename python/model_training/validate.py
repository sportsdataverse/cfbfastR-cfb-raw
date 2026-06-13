"""Validation: prediction-parity vs reference models + LOSO calibration tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb


def prediction_parity(model_a: xgb.Booster, model_b: xgb.Booster, X: pd.DataFrame,
                      tol: float = 1e-3) -> dict:
    d = xgb.DMatrix(X)
    pa, pb = model_a.predict(d), model_b.predict(d)
    max_abs = float(np.max(np.abs(pa - pb)))
    return {"max_abs_diff": max_abs, "within_tol": max_abs <= tol, "tol": tol}


def calibration_table(pred_prob, outcome, by, bin_size: float = 0.05) -> pl.DataFrame:
    df = pl.DataFrame({"pred": pred_prob, "outcome": outcome, "by": by})
    df = df.with_columns(bin=(pl.col("pred") / bin_size).round() * bin_size)
    return (
        df.group_by(["by", "bin"])
        .agg(n_plays=pl.len(), n_pos=pl.col("outcome").sum())
        .with_columns(actual=pl.col("n_pos") / pl.col("n_plays"))
        .sort(["by", "bin"])
    )


def weighted_cal_error(table: pl.DataFrame) -> float:
    t = table.with_columns(diff=(pl.col("bin") - pl.col("actual")).abs())
    per = t.group_by("by").agg(
        wce=(pl.col("diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
        n=pl.col("n_pos").sum(),
    )
    return float((per["wce"] * per["n"]).sum() / per["n"].sum())
