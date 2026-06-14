"""Phase 5 Task 5.2 — figures module tests.

Tests are skipped if plotnine is not installed (optional dependency).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

plotnine = pytest.importorskip("plotnine")


@pytest.fixture()
def calibration_df() -> pd.DataFrame:
    """Minimal calibration-bins DataFrame for plot tests."""
    return pd.DataFrame({
        "bin_mid": [0.1, 0.3, 0.5, 0.7, 0.9],
        "actual_rate": [0.08, 0.28, 0.52, 0.71, 0.88],
        "pred_rate": [0.1, 0.3, 0.5, 0.7, 0.9],
        "n": [30, 50, 80, 50, 30],
    })


@pytest.fixture()
def cpoe_df() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 100
    return pd.DataFrame({
        "cpoe": rng.normal(0, 0.15, n),
        "passer": rng.choice(["QB_A", "QB_B", "QB_C"], n),
    })


def test_figures_imports():
    from cpoe.figures import plot_calibration_curve  # noqa: F401


def test_plot_calibration_returns_ggplot(calibration_df):
    import plotnine as p9
    from cpoe.figures import plot_calibration_curve
    fig = plot_calibration_curve(calibration_df)
    assert isinstance(fig, p9.ggplot)


def test_plot_cpoe_dist_imports():
    from cpoe.figures import plot_cpoe_distribution  # noqa: F401


def test_plot_cpoe_dist_returns_ggplot(cpoe_df):
    import plotnine as p9
    from cpoe.figures import plot_cpoe_distribution
    fig = plot_cpoe_distribution(cpoe_df)
    assert isinstance(fig, p9.ggplot)
