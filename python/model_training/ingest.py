"""Read final.json plays, clean, label, weight -> pbp_full.parquet (port of keepers 01)."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .next_score import label_next_score_half

REMOVE_PLAYS = {
    "Extra Point Missed", "Extra Point Good", "Timeout", "Kickoff", "Penalty (Kickoff)",
    "Kickoff Return (Offense)", "Kickoff Return Touchdown", "Kickoff Team Fumble Recovery",
    "Kickoff Team Fumble Recovery Touchdown", "Kickoff Touchdown",
}


def _norm_inv(expr: pl.Expr, lo, hi) -> pl.Expr:
    """Inverted min-max: (hi - x) / (hi - lo).  Returns lit(1.0) when span is zero or bounds are null."""
    if lo is None or hi is None or hi == lo:
        return pl.lit(1.0)
    return (hi - expr) / (hi - lo)


def _minmax(expr: pl.Expr, lo, hi) -> pl.Expr:
    """Standard min-max: (x - lo) / (hi - lo).  Returns lit(1.0) when span is zero or bounds are null."""
    if lo is None or hi is None or hi == lo:
        return pl.lit(1.0)
    return (expr - lo) / (hi - lo)


def add_weights(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        abs_diff=pl.col("pos_score_diff_start").abs(),
        Drive_Score_Dist=(pl.col("score_drive").cast(pl.Int64) - pl.col("drive.id").cast(pl.Int64)),
    )
    dsd_min, dsd_max = df["Drive_Score_Dist"].min(), df["Drive_Score_Dist"].max()
    ad_min, ad_max = df["abs_diff"].min(), df["abs_diff"].max()
    df = df.with_columns(
        Drive_Score_Dist_W=_norm_inv(pl.col("Drive_Score_Dist"), dsd_min, dsd_max),
        ScoreDiff_W=_norm_inv(pl.col("abs_diff"), ad_min, ad_max),
    ).with_columns(
        Total_W=pl.col("Drive_Score_Dist_W") + pl.col("ScoreDiff_W"),
    )
    tw_min, tw_max = df["Total_W"].min(), df["Total_W"].max()
    return df.with_columns(
        Total_W_Scaled=_minmax(pl.col("Total_W"), tw_min, tw_max),
    )


def clean_plays(df: pl.DataFrame) -> pl.DataFrame:
    bad = (
        df.group_by("game_id")
        .agg(max_per=pl.col("period").max(), min_per=pl.col("period").min())
        .filter((pl.col("max_per") > 4) | (pl.col("min_per") < 1))
        .get_column("game_id")
    )
    df = df.filter(~pl.col("game_id").is_in(bad.to_list()))
    from .constants import BAD_GAME_IDS
    df = df.filter(~pl.col("game_id").is_in(list(BAD_GAME_IDS)))
    # ESPN partial games: keep only games whose 4th qtr reaches a 0-minute clock. The
    # column is `clock_minutes` in keepers' pbp_train but `clock.minutes` (ESPN dot-notation)
    # on CFBPlayProcess final.json -- support whichever is present.
    clock_col = next((c for c in ("clock_minutes", "clock.minutes") if c in df.columns), None)
    if clock_col is not None:
        full = (
            df.filter(pl.col("period") == 4)
            .group_by("game_id")
            .agg(min_clock=pl.col(clock_col).min())
            .filter(pl.col("min_clock") == 0)
            .get_column("game_id")
        )
        df = df.filter(pl.col("game_id").is_in(full.to_list()))
    df = df.filter(pl.col("start.down").is_between(1, 4))
    return df.filter(~pl.col("type.text").is_in(list(REMOVE_PLAYS)))


def _read_final_plays(final_dir: Path, seasons) -> pl.DataFrame:
    frames = []
    for f in sorted(Path(final_dir).glob("*.json")):
        obj = json.loads(f.read_text())
        if seasons is not None and obj.get("season") not in seasons:
            continue
        plays = obj.get("plays") or []
        if plays:
            frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _coerce_scoring_bools(df: pl.DataFrame) -> pl.DataFrame:
    # final.json may carry scoring flags as int/null; label_next_score_half needs clean bools
    for c in ("scoring_play", "offense_score_play", "defense_score_play"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Boolean).fill_null(False).alias(c))
    return df


def build_training_frame(final_dir, seasons=None) -> pl.DataFrame:
    df = _read_final_plays(final_dir, seasons)
    if df.is_empty():
        return df
    df = clean_plays(df)
    df = _coerce_scoring_bools(df)
    df = label_next_score_half(df)
    df = add_weights(df)
    return df


def write_training_frame(final_dir, out_path, seasons=None) -> int:
    df = build_training_frame(final_dir, seasons)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return df.height


def add_winner(df: pl.DataFrame) -> pl.DataFrame:
    # homeScore/awayScore on play records are running scores; the winner is decided by the
    # game's FINAL (max) score per game_id.
    return (
        df.with_columns(
            _home_final=pl.col("homeScore").max().over("game_id"),
            _away_final=pl.col("awayScore").max().over("game_id"),
        )
        .with_columns(
            winner=pl.when(pl.col("_home_final") > pl.col("_away_final"))
            .then(pl.col("homeTeamName"))
            .when(pl.col("_home_final") < pl.col("_away_final"))
            .then(pl.col("awayTeamName"))
            .otherwise(pl.lit("TIE")),
        )
        .drop(["_home_final", "_away_final"])
    )
