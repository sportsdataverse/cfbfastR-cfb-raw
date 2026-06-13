"""Per-rusher-season aggregation, lag, and weight derivation.

Port of rb_eval_model.R lines 54-86:
  lrbs block: group by (rusher_player_name, season), summarize, filter n>100, lag 1 season, weight.
  model_data rename: target=unadjusted_epa, epa_per_play=lepa, success=lsuccess.
"""
from __future__ import annotations

import polars as pl


def summarize_rusher_seasons(rush_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate rush plays to per-(rusher, season) summary statistics.

    Applies the epa clamp (< -4.5 → -4.5), computes unadjusted_epa (pre-clamp mean),
    epa (clamped mean), success (FO rate), highlight_yards/n_opps (guarded vs zero n_opps),
    and filters to n_plays > 100.
    """
    df = rush_df.with_columns(
        epa_clamped=pl.when(pl.col("epa") < -4.5).then(-4.5).otherwise(pl.col("epa")),
    )
    agg = (
        df.group_by(["rusher_player_name", "season"])
        .agg(
            n_plays=pl.len(),
            n_opps=pl.col("is_rush_opp").sum().cast(pl.Int64),
            unadjusted_epa=pl.col("epa").sum() / pl.len(),
            epa=pl.col("epa_clamped").sum() / pl.len(),
            success=pl.col("fo_success").cast(pl.Int32).sum() / pl.len(),
            highlight_yards_sum=pl.col("highlight_yards").sum(),
        )
        .with_columns(
            highlight_yards=pl.when(pl.col("n_opps") > 0)
            .then(pl.col("highlight_yards_sum") / pl.col("n_opps").cast(pl.Float64))
            .otherwise(0.0),
        )
        .drop("highlight_yards_sum")
    )
    return agg.filter(pl.col("n_plays") > 100)


def add_season_lag(rusher_seasons: pl.DataFrame) -> pl.DataFrame:
    """Add prior-season lag columns (lepa, lsuccess, lhlite_yds, lunad_epa, lplays) and weight.

    Mirrors R `mutate(lepa = lag(epa, n=1), ...)` within group_by(rusher_player_name).
    Sort by (rusher, season) ensures the shift is season-ordered.
    """
    df = rusher_seasons.sort(["rusher_player_name", "season"])
    df = df.with_columns(
        lepa=pl.col("epa").shift(1).over("rusher_player_name"),
        lunad_epa=pl.col("unadjusted_epa").shift(1).over("rusher_player_name"),
        lhlite_yds=pl.col("highlight_yards").shift(1).over("rusher_player_name"),
        lsuccess=pl.col("success").shift(1).over("rusher_player_name"),
        lplays=pl.col("n_plays").shift(1).over("rusher_player_name"),
    )
    return df.with_columns(
        weight=(
            (pl.col("n_plays").cast(pl.Float64) ** 2
             + pl.col("lplays").cast(pl.Float64) ** 2) ** 0.5
        ),
    )


def build_rusher_seasons(rush_df: pl.DataFrame) -> pl.DataFrame:
    """Full aggregation pipeline: summarize → lag → weight."""
    return add_season_lag(summarize_rusher_seasons(rush_df))


def build_model_data(rusher_seasons: pl.DataFrame) -> pl.DataFrame:
    """Rename per-rusher-season columns to GAM input contract and drop null-lag rows.

    GAM input: target=unadjusted_epa, epa_per_play=lepa, success=lsuccess,
               highlight_yards=lhlite_yds, weight, season.
    """
    return (
        rusher_seasons
        .select([
            "rusher_player_name", "unadjusted_epa", "lhlite_yds",
            "lepa", "lsuccess", "weight", "season",
        ])
        .rename({
            "unadjusted_epa": "target",
            "lhlite_yds": "highlight_yards",
            "lepa": "epa_per_play",
            "lsuccess": "success",
        })
        .drop_nulls(["epa_per_play", "success", "weight"])
    )
