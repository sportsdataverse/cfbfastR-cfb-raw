"""plotnine calibration plots (bespoke cfbfastR styling) + sidecar data tables.

Styling target: garnet #500f1b accent, grey95/grey99 panels, Gill Sans MT with a
cross-platform fallback, faceted by `by` (quarter for WP / scoring-event for EP),
sized points + linear smoother + y=x reference, calibration-error caption.

Note: loess smoothing requires the optional ``scikit-misc`` package.  When it is
not installed the smoother falls back to ``method="lm"`` (ordinary least squares),
which is correct for a binary-outcome calibration check over the [0, 1] range.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    geom_point,
    geom_smooth,
    ggplot,
    labs,
    theme,
    theme_bw,
)

GARNET = "#500f1b"
GREY95 = "#f2f2f2"  # equivalent to R's grey(0.95)
GREY99 = "#fcfcfc"  # equivalent to R's grey(0.99)
FONT = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]

try:
    import skmisc  # noqa: F401

    _SMOOTHER = "loess"
except ModuleNotFoundError:
    _SMOOTHER = "lm"


def write_calibration(
    table: pl.DataFrame,
    stem: Path | str,
    title: str,
    subtitle: str,
    cal_error: float,
) -> tuple[Path, Path]:
    """Render a calibration plot and write sidecar data tables.

    Produces a PNG figure (plotnine, bespoke cfbfastR styling) together with a
    CSV and a Parquet copy of the underlying calibration frame.  The smoother is
    loess when ``scikit-misc`` is available, otherwise ordinary least squares.

    Args:
        table: Calibration frame with columns ``by``, ``bin``, ``n_plays``,
            ``actual``.  ``by`` is used as the facet variable (e.g. quarter label
            for WP, or scoring-event label for EP).
        stem: Output path stem (no extension).  Siblings ``<stem>.png``,
            ``<stem>.csv``, and ``<stem>.parquet`` are written next to each other.
        title: Plot title string.
        subtitle: Plot subtitle string (e.g. ``"LOSO"``).
        cal_error: Overall weighted calibration error scalar, shown in the figure
            caption.

    Returns:
        A two-tuple ``(png_path, csv_path)`` of :class:`pathlib.Path` objects for
        the written PNG and CSV files respectively.

    Raises:
        ValueError: If ``table`` is missing any of the required columns.

    Example:
        Quick start::

            import polars as pl
            from model_training.figures import write_calibration

            table = pl.DataFrame({
                "by": ["Q1"] * 5,
                "bin": [0.1, 0.3, 0.5, 0.7, 0.9],
                "n_plays": [100, 200, 300, 200, 100],
                "actual": [0.12, 0.28, 0.51, 0.69, 0.93],
            })
            png, csv = write_calibration(table, "/tmp/wp_cal", title="WP",
                                         subtitle="LOSO", cal_error=0.012)
            print(png)  # /tmp/wp_cal.png
    """
    stem = Path(stem)
    csv = stem.with_suffix(".csv")
    png = stem.with_suffix(".png")
    stem.parent.mkdir(parents=True, exist_ok=True)

    table.write_csv(csv)
    table.write_parquet(stem.with_suffix(".parquet"))

    pdf = table.to_pandas()

    p = (
        ggplot(pdf, aes("bin", "actual"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET)
        + geom_smooth(method=_SMOOTHER, se=False, color=GARNET, size=0.5)
        + facet_wrap("~by")
        + labs(
            title=title,
            subtitle=subtitle,
            caption=f"Overall Weighted Calibration Error: {cal_error}",
            x="Estimated probability",
            y="Observed probability",
            size="Number of plays",
        )
        + theme_bw()
        + theme(
            text=element_text(family=FONT),
            plot_background=element_rect(fill=GREY99, color="black"),
            panel_background=element_rect(fill=GREY95),
            legend_position="bottom",
        )
    )
    p.save(png, width=6, height=4, dpi=200, verbose=False)
    return png, csv
