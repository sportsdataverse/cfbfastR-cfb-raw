"""Vectorized next-score-in-half labeling (port of model_training.R lines 22-60).

Within each (game_id, half) the next scoring drive/team/type are carried BACKWARD
(`fill backward`) so every play sees the next score in its half; the 7-class label is
then derived from the posteam's perspective. Half = 1 for periods {1,2}, 2 for {3,4}.
OT (period > 4) is dropped upstream in ingest.
"""
from __future__ import annotations

import polars as pl

from .constants import NEXT_SCORE_TO_LABEL

_TD = "Touchdown"
_FG = "Field Goal Good"
_SAFETY = "Safety"


def label_next_score_half(plays: pl.DataFrame) -> pl.DataFrame:
    df = plays.with_columns(
        half=pl.when(pl.col("period").is_in([1, 2])).then(1).otherwise(2),
        _drive=pl.col("drive.id").cast(pl.Int64),
    )
    df = df.with_columns(
        _score_team=pl.when(pl.col("scoring_play") & pl.col("offense_score_play"))
        .then(pl.col("pos_team"))
        .when(pl.col("scoring_play") & pl.col("defense_score_play"))
        .then(pl.col("def_pos_team"))
        .otherwise(None),
        _score_type=pl.when(pl.col("scoring_play")).then(pl.col("type.text")).otherwise(None),
        _score_drive=pl.when(pl.col("scoring_play")).then(pl.col("_drive")).otherwise(None),
    )
    df = df.with_columns(
        next_team=pl.col("_score_team").fill_null(strategy="backward").over(["game_id", "half"]),
        next_type=pl.col("_score_type").fill_null(strategy="backward").over(["game_id", "half"]),
        score_drive=pl.col("_score_drive").fill_null(strategy="backward").over(["game_id", "half"]),
    )
    df = df.with_columns(
        next_score_half=pl.when(pl.col("next_type").is_null())
        .then(pl.lit("No_Score"))
        .when(pl.col("next_type").str.contains(_TD) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Touchdown"))
        .when(pl.col("next_type").str.contains(_TD))
        .then(pl.lit("Opp_Touchdown"))
        .when(pl.col("next_type").str.contains(_FG, literal=True) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Field_Goal"))
        .when(pl.col("next_type").str.contains(_FG, literal=True))
        .then(pl.lit("Opp_Field_Goal"))
        .when(pl.col("next_type").str.contains(_SAFETY) & (pl.col("pos_team") == pl.col("next_team")))
        .then(pl.lit("Safety"))
        .when(pl.col("next_type").str.contains(_SAFETY))
        .then(pl.lit("Opp_Safety"))
        .otherwise(pl.lit("No_Score")),
    )
    df = df.with_columns(
        score_drive=pl.when(pl.col("next_score_half") == "No_Score")
        .then(pl.col("_drive"))
        .otherwise(pl.col("score_drive")),
        label=pl.col("next_score_half").replace_strict(NEXT_SCORE_TO_LABEL, return_dtype=pl.Int32),
    )
    return df.drop(["_drive", "_score_team", "_score_type", "_score_drive", "next_team", "next_type"])
