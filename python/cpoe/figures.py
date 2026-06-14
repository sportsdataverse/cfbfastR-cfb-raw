"""Calibration plots for the CFB CP model (Approach A, distance-bucket facets).

Uses plotnine with bespoke cfbfastR styling (garnet accent, grey panels).
One facet per distance bucket (Short / Intermediate / Long).

Caption always notes that distance_bucket approximates throw depth via
yards-to-first-down, NOT via actual air yards (unavailable in ESPN CFB pbp).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------
GARNET = "#500f1b"
FONT_FAMILY = ["Gill Sans MT", "DejaVu Sans", "sans-serif"]

_ANN_DATA = {
    "x": [0.20, 0.80],
    "y": [0.80, 0.20],
    "lab": ["More times\nthan expected", "Fewer times\nthan expected"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_cp_calibration(
    tbl: pl.DataFrame,
    stem: str | Path,
    cal_error: float,
    title: str = "CFB Completion Probability — LOSO Calibration",
    subtitle: str = (
        "Approach A (game-state model; "
        "distance_bucket = yards-to-first-down proxy for throw depth)"
    ),
) -> tuple[Path, Path]:
    """Write a calibration PNG, CSV, and Parquet for the CP model.

    Creates a 3-facet calibration plot (Short / Intermediate / Long distance buckets)
    with a diagonal reference line, bubble points scaled by n_plays, and a
    LOESS smoother overlay.

    Args:
        tbl: Calibration table from validate.calibration_table
            (columns: distance_bucket, bin_pred_prob, n_plays, bin_actual_prob).
        stem: Path stem (no extension). PNG, CSV, and Parquet files are written
            using this stem with the appropriate extension.
        cal_error: Overall weighted calibration error for the figure caption.
        title: Plot title.
        subtitle: Plot subtitle.

    Returns:
        (png_path, csv_path) as Path objects.
    """
    import pandas as pd
    from plotnine import (
        aes,
        coord_equal,
        element_rect,
        element_text,
        facet_wrap,
        geom_abline,
        geom_point,
        geom_smooth,
        geom_text,
        ggplot,
        labs,
        scale_x_continuous,
        scale_y_continuous,
        theme,
        theme_bw,
    )

    stem = Path(stem)
    csv_path = stem.with_suffix(".csv")
    png_path = stem.with_suffix(".png")
    parquet_path = stem.with_suffix(".parquet")

    stem.parent.mkdir(parents=True, exist_ok=True)

    # Write tabular outputs
    tbl.write_csv(csv_path)
    tbl.write_parquet(parquet_path)

    # Convert to pandas for plotnine
    pdf = tbl.to_pandas()
    ann = pd.DataFrame(_ANN_DATA)

    caption = (
        f"Overall Weighted Calibration Error: {round(cal_error, 4)}\n"
        "Note: distance_bucket approximates throw depth via yards-to-first-down, "
        "not actual air yards (unavailable in ESPN CFB play-by-play data)."
    )

    p = (
        ggplot(pdf, aes("bin_pred_prob", "bin_actual_prob"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black")
        + geom_point(aes(size="n_plays"), color=GARNET, alpha=0.8)
        + geom_smooth(method="loess", se=False, color=GARNET, size=0.6)
        + geom_text(
            data=ann,
            mapping=aes(x="x", y="y", label="lab"),
            size=8,
            color="grey40",
            inherit_aes=False,
        )
        + facet_wrap("~distance_bucket", ncol=3)
        + coord_equal()
        + scale_x_continuous(limits=(0, 1))
        + scale_y_continuous(limits=(0, 1))
        + labs(
            title=title,
            subtitle=subtitle,
            caption=caption,
            x="Estimated completion percentage",
            y="Observed completion percentage",
            size="Number of plays",
        )
        + theme_bw()
        + theme(
            text=element_text(family=FONT_FAMILY),
            plot_background=element_rect(fill="grey99", color="black"),
            panel_background=element_rect(fill="grey95"),
            legend_position="bottom",
        )
    )

    p.save(str(png_path), width=9, height=4, dpi=200, verbose=False)

    return png_path, csv_path
