"""Calibration table and metrics for xREPA LOSO validation.

Port of show_calibration_chart() in rb_eval_model.R (lines 119-176), decoupled from plotting.
R calibration:
  bin_pred_epa   = round(exp_rb_epa / bin_size) * bin_size
  bin_actual_epa = mean(target) within each bin
  weighted cal error = weighted.mean(|bin_pred - bin_actual|, total_instances)
  weighted R²    = cor(bin_actual, bin_pred, w=total_instances)^2  (boot::corr method)
"""
from __future__ import annotations

import numpy as np
import polars as pl


def calibration_table(cv_results: pl.DataFrame, bin_size: float = 0.05) -> pl.DataFrame:
    """Bin LOSO predictions and compute mean actual EPA per bin."""
    return (
        cv_results
        .drop_nulls(["exp_rb_epa", "target"])
        .with_columns(
            bin_pred_epa=(pl.col("exp_rb_epa") / bin_size).round(0) * bin_size,
        )
        .group_by("bin_pred_epa")
        .agg(
            total_instances=pl.len(),
            bin_actual_epa=pl.col("target").mean(),
        )
        .sort("bin_pred_epa")
    )


def weighted_cal_error(table: pl.DataFrame) -> float:
    """Weighted mean absolute calibration error: weighted.mean(|bin_pred - bin_actual|, n)."""
    t = table.with_columns(
        cal_diff=(pl.col("bin_pred_epa") - pl.col("bin_actual_epa")).abs(),
    )
    total = t["total_instances"].sum()
    if total == 0:
        return float("nan")
    return float((t["cal_diff"] * t["total_instances"]).sum() / total)


def weighted_r2(table: pl.DataFrame) -> float:
    """Weighted R² of binned actual vs predicted EPA (boot::corr method from R source)."""
    t = table.drop_nulls(["bin_pred_epa", "bin_actual_epa"])
    if t.is_empty():
        return float("nan")
    y = t["bin_actual_epa"].to_numpy()
    y_hat = t["bin_pred_epa"].to_numpy()
    w = t["total_instances"].to_numpy().astype(float)
    w_norm = w / w.sum()
    mu_y = np.sum(w_norm * y)
    mu_yhat = np.sum(w_norm * y_hat)
    cov_yy = np.sum(w_norm * (y - mu_y) ** 2)
    cov_yyhyh = np.sum(w_norm * (y_hat - mu_yhat) ** 2)
    cov_yyh = np.sum(w_norm * (y - mu_y) * (y_hat - mu_yhat))
    denom = np.sqrt(cov_yy * cov_yyhyh)
    if denom < 1e-14:
        return float("nan")
    return float((cov_yyh / denom) ** 2)
