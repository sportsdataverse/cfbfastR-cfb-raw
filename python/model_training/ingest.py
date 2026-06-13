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


def add_weights(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        abs_diff=pl.col("pos_score_diff_start").abs(),
        Drive_Score_Dist=(pl.col("score_drive").cast(pl.Int64) - pl.col("drive.id").cast(pl.Int64)),
    )
    dsd_min, dsd_max = df["Drive_Score_Dist"].min(), df["Drive_Score_Dist"].max()
    ad_min, ad_max = df["abs_diff"].min(), df["abs_diff"].max()
    df = df.with_columns(
        Drive_Score_Dist_W=(dsd_max - pl.col("Drive_Score_Dist")) / (dsd_max - dsd_min),
        ScoreDiff_W=(ad_max - pl.col("abs_diff")) / (ad_max - ad_min),
    ).with_columns(
        Total_W=pl.col("Drive_Score_Dist_W") + pl.col("ScoreDiff_W"),
    )
    tw_min, tw_max = df["Total_W"].min(), df["Total_W"].max()
    return df.with_columns(
        Total_W_Scaled=(pl.col("Total_W") - tw_min) / (tw_max - tw_min),
    )


def clean_plays(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        start_down=pl.when((pl.col("start.down") == 5) & pl.col("type.text").str.contains("Kickoff"))
        .then(-1)
        .otherwise(pl.col("start.down")),
    )
    bad = (
        df.group_by("game_id")
        .agg(max_per=pl.col("period").max(), min_per=pl.col("period").min())
        .filter((pl.col("max_per") > 4) | (pl.col("min_per") < 1))
        .get_column("game_id")
    )
    df = df.filter(~pl.col("game_id").is_in(bad.to_list()))
    from .constants import BAD_GAME_IDS
    df = df.filter(~pl.col("game_id").is_in(list(BAD_GAME_IDS)))
    if "clock_minutes" in df.columns:
        full = (
            df.filter(pl.col("period") == 4)
            .group_by("game_id")
            .agg(min_clock=pl.col("clock_minutes").min())
            .filter(pl.col("min_clock") == 0)
            .get_column("game_id")
        )
        df = df.filter(pl.col("game_id").is_in(full.to_list()))
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
