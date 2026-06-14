"""xREPA calibration figure — thin wrapper over model_training.figures.write_calibration.

xREPA has no natural facet variable (single-panel: all rushers combined). We add a constant
'by' column ("All rushers") so the shared write_calibration function renders a single facet.

Column mapping:
  calibration_table output  →  write_calibration input
  bin_pred_epa              →  bin
  total_instances           →  n_plays
  bin_actual_epa            →  actual
  (constant "All rushers")  →  by
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def write_xrepa_calibration(
    table: pl.DataFrame,
    stem: str | Path,
    cal_error: float,
    r2: float,
) -> tuple[Path, Path]:
    """Produce the xREPA calibration PNG + sidecar CSV.

    Wraps model_training.figures.write_calibration with a constant 'by' column
    and an xREPA-specific title/subtitle.

    Args:
        table: output of validate.calibration_table — must have columns
               bin_pred_epa, total_instances, bin_actual_epa.
        stem: output path stem (no extension). Siblings <stem>.png, <stem>.csv,
              and <stem>.parquet are written next to each other.
        cal_error: weighted calibration error scalar (for figure caption).
        r2: weighted R² scalar (for figure subtitle).

    Returns:
        (png_path, csv_path) two-tuple of Path objects.
    """
    try:
        from model_training.figures import write_calibration as _wc
    except ImportError as exc:
        raise ImportError(
            "model_training package required for figures. "
            "Ensure python/model_training/ is on sys.path."
        ) from exc

    # Rename to the write_calibration column contract
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
        subtitle=f"Weighted R²: {round(r2, 4)}",
        cal_error=cal_error,
    )
