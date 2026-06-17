"""plotnine calibration + feature-importance figures for the fourth-down yards model.

Bespoke cfbfastR styling: garnet #500f1b accent, grey95/grey99 panels,
Gill Sans MT with cross-platform fallback chain. Emits PNGs + sidecar data tables.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import skmisc  # noqa: F401 — loess smoothing

    _SMOOTHER = "loess"
except ImportError:
    _SMOOTHER = "lm"

from plotnine import (
    aes,
    coord_flip,
    element_rect,
    element_text,
    geom_abline,
    geom_col,
    geom_point,
    geom_smooth,
    ggplot,
    labs,
    scale_x_continuous,
    theme,
    theme_bw,
)

GARNET = "#500f1b"
GREY95 = "#f2f2f2"  # equivalent to R's grey(0.95)
GREY99 = "#fcfcfc"  # equivalent to R's grey(0.99)
FONT = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]


def _theme_fd():
    return theme_bw() + theme(
        text=element_text(family=FONT),
        plot_background=element_rect(fill=GREY99, color="black"),
        panel_background=element_rect(fill=GREY95),
        legend_position="bottom",
    )


def _save(p, path: Path, width: int = 6, height: int = 4, dpi: int = 200) -> Path:
    p.save(str(path), width=width, height=height, dpi=dpi, verbose=False)
    return path


def write_fd_figures(
    cal_table: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir,
    cal_error: float,
) -> tuple[Path, Path]:
    """Write calibration scatter and feature-importance bar chart.

    Args:
        cal_table: DataFrame with columns bin_center, pred_fd_prob, empirical_fd_rate, n_plays.
        importance: DataFrame with columns Feature, Gain (from xgb.importance).
        out_dir: Directory to write PNGs + CSVs into.
        cal_error: Overall weighted calibration error (float, shown in caption).

    Returns:
        (calibration_png_path, importance_png_path)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal_table.to_csv(out / "fd_calibration.csv", index=False)
    importance.to_csv(out / "fd_feature_importance.csv", index=False)

    cal_p = (
        ggplot(cal_table, aes("pred_fd_prob", "empirical_fd_rate"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET)
        + geom_smooth(method=_SMOOTHER, se=False, color=GARNET, size=0.5)
        + scale_x_continuous(limits=[0, 1])
        + labs(
            title="Fourth-Down Yards Model — First-Down Calibration",
            subtitle="Predicted P(first down) vs Empirical First-Down Rate",
            caption=f"Weighted Calibration Error: {cal_error:.4f}",
            x="Predicted P(first down)",
            y="Empirical first-down rate",
            size="Number of plays",
        )
        + _theme_fd()
    )
    cal_png = _save(cal_p, out / "fd_calibration.png")

    imp_sorted = importance.sort_values("Gain", ascending=True).copy()
    imp_p = (
        ggplot(imp_sorted, aes(x="Feature", y="Gain"))
        + geom_col(fill=GARNET)
        + coord_flip()
        + labs(
            title="Fourth-Down Yards Model — Feature Importance",
            subtitle="XGBoost Gain (higher = more important)",
            x="Feature",
            y="Gain",
        )
        + _theme_fd()
    )
    imp_png = _save(imp_p, out / "fd_feature_importance.png")

    return cal_png, imp_png
