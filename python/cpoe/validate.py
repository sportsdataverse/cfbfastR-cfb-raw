"""Calibration tables and Brier score for the CFB CP model LOSO CV output.

Stratified by distance bucket (yards-to-first-down proxy for air yards depth).
Note: distance_bucket is a COARSE proxy — this is documented on all figures.
"""
from __future__ import annotations

import polars as pl

from . import constants as C
from .features import assign_distance_bucket


# ---------------------------------------------------------------------------
# Distance bucket expression helper
# ---------------------------------------------------------------------------

def distance_bucket(col: pl.Expr) -> pl.Expr:
    """Classify a distance column expression into Short / Intermediate / Long.

    Thresholds (inclusive):
      Short:        col <= 3
      Intermediate: 4 <= col <= 8
      Long:         col >= 9

    Args:
        col: A polars Expr referencing a distance column.

    Returns:
        A polars Expr that evaluates to one of "Short", "Intermediate", "Long".
    """
    short_hi = C.DISTANCE_BUCKETS["Short"][1]
    mid_hi = C.DISTANCE_BUCKETS["Intermediate"][1]
    return (
        pl.when(col <= short_hi)
        .then(pl.lit("Short"))
        .when(col <= mid_hi)
        .then(pl.lit("Intermediate"))
        .otherwise(pl.lit("Long"))
    )


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------

def calibration_table(
    cv_df: pl.DataFrame,
    bin_size: float = 0.05,
    min_plays: int = 10,
) -> pl.DataFrame:
    """Compute binned calibration stats from LOSO CV output.

    Bins predicted completion probability into `bin_size`-wide bins, then
    computes actual completion rate per (distance_bucket, bin).

    Args:
        cv_df: LOSO CV output with predicted_cp (or cp), completion,
            and distance_bucket columns.
        bin_size: Bin width for predicted probability (default 0.05 = 5%).
        min_plays: Drop bins with fewer than this many plays (default 10).

    Returns:
        polars DataFrame with columns:
            distance_bucket, bin_pred_prob, n_plays, n_complete, bin_actual_prob.
    """
    # Support either predicted_cp (loso_calibrate output) or cp (loso_cv output)
    pred_col = "predicted_cp" if "predicted_cp" in cv_df.columns else "cp"

    # Ensure distance_bucket is present
    if "distance_bucket" not in cv_df.columns:
        if "distance" in cv_df.columns:
            cv_df = cv_df.with_columns(
                distance_bucket(pl.col("distance")).alias("distance_bucket")
            )
        else:
            cv_df = cv_df.with_columns(pl.lit("Unknown").alias("distance_bucket"))

    return (
        cv_df.with_columns(
            bin_pred_prob=(
                (pl.col(pred_col) / bin_size).round() * bin_size
            ).round(4),
        )
        .group_by(["distance_bucket", "bin_pred_prob"])
        .agg(
            n_plays=pl.len(),
            n_complete=pl.col("completion").cast(pl.Int32).sum(),
        )
        .with_columns(
            bin_actual_prob=(pl.col("n_complete").cast(pl.Float64) / pl.col("n_plays")),
        )
        .filter(pl.col("n_plays") >= min_plays)
        .sort(["distance_bucket", "bin_pred_prob"])
    )


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(cv_df: pl.DataFrame) -> dict[str, float]:
    """Compute overall and per-bucket Brier score from LOSO CV output.

    Brier score = mean((predicted - actual)^2). Lower is better.

    Args:
        cv_df: LOSO CV output with predicted_cp (or cp), completion,
            and distance_bucket columns.

    Returns:
        dict with keys "overall" (float) and "per_bucket" (list of dicts).
    """
    pred_col = "predicted_cp" if "predicted_cp" in cv_df.columns else "cp"

    if "distance_bucket" not in cv_df.columns:
        if "distance" in cv_df.columns:
            cv_df = cv_df.with_columns(
                distance_bucket(pl.col("distance")).alias("distance_bucket")
            )
        else:
            cv_df = cv_df.with_columns(pl.lit("Unknown").alias("distance_bucket"))

    df = cv_df.with_columns(
        brier_sq=(
            (pl.col(pred_col).cast(pl.Float64) - pl.col("completion").cast(pl.Float64)) ** 2
        )
    )

    overall = float(df["brier_sq"].mean())

    per_bucket = (
        df.group_by("distance_bucket")
        .agg(
            brier=pl.col("brier_sq").mean(),
            n=pl.len(),
        )
        .sort("distance_bucket")
        .to_dicts()
    )

    return {"overall": overall, "per_bucket": per_bucket}


# ---------------------------------------------------------------------------
# Weighted calibration error
# ---------------------------------------------------------------------------

def weighted_cal_error(tbl: pl.DataFrame) -> dict:
    """Compute per-bucket and overall weighted calibration error from calibration_table output.

    Args:
        tbl: Output of calibration_table (has distance_bucket, bin_pred_prob,
            n_plays, bin_actual_prob).

    Returns:
        dict with "per_bucket" (list of dicts) and "overall" (float).
    """
    tbl = tbl.with_columns(
        cal_diff=(pl.col("bin_pred_prob") - pl.col("bin_actual_prob")).abs()
    )
    per = (
        tbl.group_by("distance_bucket")
        .agg(
            wce=(pl.col("cal_diff") * pl.col("n_plays")).sum() / pl.col("n_plays").sum(),
            n=pl.col("n_plays").sum(),
        )
        .sort("distance_bucket")
    )
    total_n = int(per["n"].sum())
    overall = float(
        (per["wce"] * per["n"]).sum() / total_n
    ) if total_n > 0 else float("nan")

    return {"per_bucket": per.to_dicts(), "overall": overall}
