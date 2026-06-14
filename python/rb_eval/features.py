"""Rush play loading, fo_success derivation, EPA clamping, and per-rusher aggregation.

Column conventions (cfb4th / CFBPlayProcess final.json → rb_eval normalization):
  - down        : integer down number (1–4)
  - distance    : yards to first down
  - yards_gained: yards gained on the play (mapped from yds_rushed in final.json)
  - epa         : EPA per play (mapped from EPA/epa in final.json)
  - epa_clamped : epa with lower_bound clamp at RB_EVAL_EPA_CLAMP (-4.5)
  - fo_success  : Football Outsiders down-weighted success (see spec §5.2)
  - is_rush_opp : True if yards_gained >= 4 (eligible for highlight_yards)

The fo_success formula mirrors cfb4th's `play_successful` for rushing:
  down 1: yards_gained >= 0.50 * distance
  down 2: yards_gained >= 0.70 * distance
  down 3: ALWAYS False  (cfb4th parity — 3rd-down conversion is a binary, not a success metric)
  down 4: yards_gained >= distance

When loading from final.json (load_rush_plays), input column names from CFBPlayProcess are
remapped at the filter stage to the canonical rb_eval names above.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl

from .constants import RB_EVAL_EPA_CLAMP, RB_EVAL_MIN_PLAYS, RB_EVAL_RUSH_OPP_THRESHOLD


# ---------------------------------------------------------------------------
# Core feature functions (operate on rb_eval canonical column names)
# ---------------------------------------------------------------------------

def fo_success(df: pl.DataFrame) -> pl.DataFrame:
    """Add fo_success column using Football Outsiders down-weighted success thresholds.

    Mirrors cfb4th parity:
      - down 1: success if yards_gained >= 0.5 * distance
      - down 2: success if yards_gained >= 0.7 * distance
      - down 3: ALWAYS False (excluded from the cfb4th success formula)
      - down 4: success if yards_gained >= distance

    Args:
        df: play frame with columns down (int), distance (float), yards_gained (float).

    Returns:
        df with added boolean column fo_success.
    """
    return df.with_columns(
        fo_success=pl.when(pl.col("down") == 1)
        .then(pl.col("yards_gained") >= 0.5 * pl.col("distance"))
        .when(pl.col("down") == 2)
        .then(pl.col("yards_gained") >= 0.7 * pl.col("distance"))
        .when(pl.col("down") >= 4)
        .then(pl.col("yards_gained") >= pl.col("distance"))
        .otherwise(False)
        .cast(pl.Boolean),
    )


def clamp_epa(df: pl.DataFrame) -> pl.DataFrame:
    """Add epa_clamped column with lower bound at RB_EVAL_EPA_CLAMP (-4.5).

    Args:
        df: play frame with column epa (float).

    Returns:
        df with added column epa_clamped.
    """
    return df.with_columns(
        epa_clamped=pl.col("epa").clip(lower_bound=RB_EVAL_EPA_CLAMP),
    )


def aggregate_per_rusher(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate rush plays to per-(rusher_player_name, season) summary statistics.

    Expects df to already have fo_success and epa_clamped columns (call fo_success()
    and clamp_epa() first). Filters to n_plays > RB_EVAL_MIN_PLAYS (100).

    Args:
        df: play frame with columns rusher_player_name, season, epa, epa_clamped,
            fo_success, yards_gained.

    Returns:
        Aggregated polars DataFrame with columns:
            rusher_player_name, season, n_plays, epa_per_play, success, unadjusted_epa.
    """
    agg = (
        df.group_by(["rusher_player_name", "season"])
        .agg(
            n_plays=pl.len(),
            epa_per_play=pl.col("epa_clamped").mean(),
            success=pl.col("fo_success").cast(pl.Float64).mean(),
            unadjusted_epa=pl.col("epa").mean(),
        )
        .filter(pl.col("n_plays") > RB_EVAL_MIN_PLAYS)
    )
    return agg


# ---------------------------------------------------------------------------
# final.json loading helpers
# ---------------------------------------------------------------------------

def _normalize_rush_play(df: pl.DataFrame) -> pl.DataFrame:
    """Rename final.json / CFBPlayProcess column names to rb_eval canonical names.

    final.json uses: yds_rushed, start.down, start.distance, EPA, rush, pos_team,
    rusher_player_name, highlight_yards (pre-computed by CFBPlayProcess).
    rb_eval uses: yards_gained, down, distance, epa, rush, pos_team, rusher_player_name.

    Only renames columns that exist (so unit tests with already-canonical names pass through).
    """
    rename_map: dict[str, str] = {}
    if "yds_rushed" in df.columns and "yards_gained" not in df.columns:
        rename_map["yds_rushed"] = "yards_gained"
    if "start.down" in df.columns and "down" not in df.columns:
        rename_map["start.down"] = "down"
    if "start.distance" in df.columns and "distance" not in df.columns:
        rename_map["start.distance"] = "distance"
    if "EPA" in df.columns and "epa" not in df.columns:
        rename_map["EPA"] = "epa"
    if rename_map:
        df = df.rename(rename_map)
    return df


def filter_rush_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Filter to rushing plays; apply validity conditions; add fo_success + is_rush_opp.

    Filtering mirrors rb_eval_model.R lines 20-22:
        filter(rush == 1)
        filter(!is.na(posteam) & !is.na(epa) & !is.na(rusher_player_name))
        filter(rusher_player_name != "TEAM")

    Args:
        df: raw plays frame (final.json plays, pre-normalization acceptable).

    Returns:
        Filtered frame with fo_success, epa_clamped, is_rush_opp columns added,
        using canonical rb_eval column names.
    """
    df = _normalize_rush_play(df)
    out = (
        df.filter(pl.col("rush") == True)  # noqa: E712
        .filter(pl.col("pos_team").is_not_null())
        .filter(pl.col("epa").is_not_null())
        .filter(pl.col("rusher_player_name").is_not_null())
        .filter(pl.col("rusher_player_name") != "TEAM")
    )
    out = fo_success(out)
    out = clamp_epa(out)
    out = out.with_columns(
        is_rush_opp=(pl.col("yards_gained") >= RB_EVAL_RUSH_OPP_THRESHOLD).cast(pl.Boolean),
    )
    return out


def load_rush_plays(
    final_dir: str | Path,
    seasons: Iterable[int] | None = None,
) -> pl.DataFrame:
    """Read final.json play files, filter to rushing plays, return filtered frame.

    Args:
        final_dir: path to the backfill's cfb/json/final/ directory. Each .json file
            is a game-level dict with a "plays" list (CFBPlayProcess output).
        seasons: optional iterable of seasons (int) to restrict loading.
            Pass None to load all available files.

    Returns:
        polars DataFrame of rush plays with fo_success, epa_clamped, is_rush_opp columns.
        Returns an empty DataFrame when no matching plays are found.
    """
    seasons_set: set[int] | None = None if seasons is None else set(seasons)
    frames: list[pl.DataFrame] = []
    for path in sorted(Path(final_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if seasons_set is not None and raw.get("season") not in seasons_set:
            continue
        plays = raw.get("plays") or []
        if not plays:
            continue
        frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames, how="diagonal_relaxed")
    return filter_rush_plays(df)
