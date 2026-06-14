"""Diagnostic figures for the Pregame WP model (Track 4).

Generates scatter plots of predicted vs actual point differential.
Uses matplotlib; plotnine (ggplot2-style) is available in the 'figures' dep group.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


def scatter_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Pregame WP — Predicted vs Actual Point Differential",
    output_path: Optional[str | Path] = None,
) -> None:
    """Scatter plot of predicted vs actual point differential.

    Args:
        y_true: Actual point differentials.
        y_pred: Predicted point differentials from the model.
        title: Plot title.
        output_path: If provided, save the figure to this path (PNG/SVG).
                     If None, call plt.show() instead.

    Raises:
        ImportError: if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for figures. Install with: uv add matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, y_true, alpha=0.35, s=15, color="#2196F3", edgecolors="none")

    # Reference line: perfect prediction
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#E53935", linewidth=1.2, linestyle="--", label="y=x")

    ax.set_xlabel("Predicted Point Differential")
    ax.set_ylabel("Actual Point Differential")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def scatter_from_df(
    model,  # XGBRegressor — avoid import cycle
    df: pl.DataFrame,
    *,
    output_path: Optional[str | Path] = None,
) -> None:
    """Convenience wrapper: predict from df and call scatter_predicted_vs_actual.

    Args:
        model: Fitted XGBRegressor with a single feature '5FRDiff'.
        df: DataFrame with columns '5FRDiff' and 'PtsDiff'.
        output_path: Optional file path to save the figure.
    """
    X = df[["5FRDiff"]].to_numpy()
    y_true = df["PtsDiff"].to_numpy()
    y_pred = model.predict(X)
    scatter_predicted_vs_actual(y_true, y_pred, output_path=output_path)
