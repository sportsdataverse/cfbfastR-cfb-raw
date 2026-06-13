"""xREPA calibration figure — thin wrapper over model_training.figures.write_calibration."""
from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from model_training.figures import write_calibration as _wc
except ImportError as e:
    raise ImportError(
        "model_training package required. Ensure python/model_training/ is on sys.path."
    ) from e


def write_xrepa_calibration(
    table: pl.DataFrame,
    stem: str | Path,
    cal_error: float,
    r2: float,
) -> tuple[Path, Path]:
    """Produce the xREPA calibration PNG + sidecar CSV.

    Renames calibration_table columns to the write_calibration contract and adds
    a constant 'by' column ("All rushers") so the shared helper renders a single panel.
    """
    adapted = (
        table
        .rename({
            "bin_pred_epa": "bin",
            "total_instances": "n_plays",
            "bin_actual_epa": "actual",
        })
        .with_columns(by=pl.lit("All rushers"))
    )
    return _wc(
        adapted,
        stem=stem,
        title="xREPA LOSO Calibration",
        subtitle=f"Wgt R²: {round(r2, 4)}",
        cal_error=cal_error,
    )
