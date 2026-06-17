"""Diagnostic figures for the CFB CPOE model (Track 5).

Requires the ``figures`` dependency group:
    uv sync --group figures

CFB/cfbfastR palette:
    Garnet  #500f1b   (primary)
    Gold    #c99700   (secondary)
    Steel   #4a4a4a   (neutral)
"""
from __future__ import annotations

import pandas as pd
import plotnine as p9

_GARNET = "#500f1b"
_GOLD = "#c99700"
_STEEL = "#4a4a4a"

_THEME = p9.theme_minimal() + p9.theme(
    text=p9.element_text(family="Gill Sans MT", color=_STEEL),
    axis_text=p9.element_text(size=9),
    axis_title=p9.element_text(size=10),
    plot_title=p9.element_text(size=12, face="bold"),
)


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    *,
    title: str = "CFB CP Model — Calibration",
) -> p9.ggplot:
    """Plot actual completion rate vs. predicted CP probability.

    Args:
        calibration_df: Output of ``validate.calibration_bins()``.
            Must contain columns ``bin_mid``, ``actual_rate``, ``pred_rate``.
        title: Plot title.

    Returns:
        plotnine ``ggplot`` object.
    """
    fig = (
        p9.ggplot(calibration_df, p9.aes(x="bin_mid"))
        + p9.geom_abline(slope=1, intercept=0, color=_STEEL, linetype="dashed", size=0.6)
        + p9.geom_line(p9.aes(y="actual_rate"), color=_GARNET, size=1.0)
        + p9.geom_point(p9.aes(y="actual_rate", size="n"), color=_GARNET, alpha=0.8)
        + p9.scale_size_continuous(range=(2, 6), guide=False)
        + p9.scale_x_continuous(limits=(0, 1), labels=lambda x: [f"{v:.0%}" for v in x])
        + p9.scale_y_continuous(limits=(0, 1), labels=lambda y: [f"{v:.0%}" for v in y])
        + p9.labs(
            title=title,
            x="Predicted Completion Probability",
            y="Actual Completion Rate",
        )
        + _THEME
    )
    return fig


def plot_cpoe_distribution(
    cpoe_df: pd.DataFrame,
    *,
    cpoe_col: str = "cpoe",
    title: str = "CPOE Distribution",
) -> p9.ggplot:
    """Plot the distribution of CPOE values.

    Args:
        cpoe_df: DataFrame with a ``cpoe`` column (or custom ``cpoe_col``).
        cpoe_col: Name of the CPOE column.
        title: Plot title.

    Returns:
        plotnine ``ggplot`` object.
    """
    fig = (
        p9.ggplot(cpoe_df, p9.aes(x=cpoe_col))
        + p9.geom_histogram(fill=_GARNET, color="white", bins=30, alpha=0.9)
        + p9.geom_vline(xintercept=0, color=_STEEL, linetype="dashed", size=0.8)
        + p9.labs(
            title=title,
            x="CPOE (Completion − Expected Completion)",
            y="Count",
        )
        + _THEME
    )
    return fig
