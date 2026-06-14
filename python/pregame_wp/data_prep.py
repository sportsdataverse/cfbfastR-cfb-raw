"""Data preparation helpers for the Pregame WP + Five-Factors pipeline (Track 4).

Faithful port of akeaswaran's win-prob.ipynb cells 20, 22, and 24.

Key design decisions:
  OQ-2: 3rd-down plays default to False in play_successful (absent from conditions).
  OQ-5: PuntReturnEqPPP = punt_eqppp (punter EP) — faithful bug port. See constants.py.
  OQ-7: WP uses mu=0.0 (symmetric) and full-training-set std.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .constants import (
    EXPLOSIVE_THRESHOLD,
    FILTER_Z,
    FILTER_Z2,
    SR_DOWN1,
    SR_DOWN2,
    SR_DOWN4,
)


# ---------------------------------------------------------------------------
# play_successful (polars, faithful port of np.select conditions)
# ---------------------------------------------------------------------------

def play_successful(df: pl.DataFrame) -> pl.DataFrame:
    """Add a boolean 'play_successful' column using down-specific success thresholds.

    Down 1: yards_gained >= 0.5 * distance
    Down 2: yards_gained >= 0.7 * distance
    Down 3: always False (intentionally absent from conditions — faithful OQ-2 port)
    Down 4: yards_gained >= 1.0 * distance

    Args:
        df: DataFrame with columns 'down', 'distance', 'yards_gained'.

    Returns:
        df with an added boolean column 'play_successful'.
    """
    return df.with_columns(
        pl.when(pl.col("down") == 1)
        .then(pl.col("yards_gained") >= SR_DOWN1 * pl.col("distance"))
        .when(pl.col("down") == 2)
        .then(pl.col("yards_gained") >= SR_DOWN2 * pl.col("distance"))
        .when(pl.col("down") == 4)
        .then(pl.col("yards_gained") >= SR_DOWN4 * pl.col("distance"))
        .otherwise(False)  # down 3 and any other value → False
        .alias("play_successful")
    )


# ---------------------------------------------------------------------------
# generate_team_play_stats (pandas-style dict of team-level summary stats)
# ---------------------------------------------------------------------------

def generate_team_play_stats(plays_df: pl.DataFrame) -> dict[str, Any]:
    """Compute per-team offensive play statistics for the Five-Factors pipeline.

    Computes OffSR (offensive success rate), OffER (explosive play rate), and
    AvgEqPPP (average expected points per play) from a play-level DataFrame.

    Note (OQ-5 — FAITHFUL BUG PORT): The punt_return_eqppp field is computed but
    PuntReturnEqPPP is deliberately set equal to punt_eqppp (the punter's EP), so
    the PuntEqPPP - PuntReturnEqPPP sub-term is always zero. This matches the
    notebook's generate_team_st_stats behavior.

    Args:
        plays_df: DataFrame with columns including 'down', 'distance', 'yards_gained',
                  and optionally 'ep_start', 'ep_end' for EqPPP lookups.

    Returns:
        dict with keys: OffSR, OffER, AvgEqPPP, punt_eqppp, punt_return_eqppp
    """
    # Add play_successful column if not already present
    if "play_successful" not in plays_df.columns:
        plays_df = play_successful(plays_df)

    # Explosive: yards_gained > EXPLOSIVE_THRESHOLD
    if "play_explosive" not in plays_df.columns:
        plays_df = plays_df.with_columns(
            (pl.col("yards_gained") > EXPLOSIVE_THRESHOLD).alias("play_explosive")
        )

    n_plays = len(plays_df)
    if n_plays == 0:
        return {
            "OffSR": 0.0,
            "OffER": 0.0,
            "AvgEqPPP": 0.0,
            "punt_eqppp": 0.0,
            "punt_return_eqppp": 0.0,
        }

    off_sr = plays_df["play_successful"].mean()
    off_er = plays_df["play_explosive"].mean()

    # EqPPP: use ep_end - ep_start if available, else yards_gained proxy
    if "EqPPP" in plays_df.columns:
        avg_eqppp = plays_df["EqPPP"].mean()
    elif "ep_end" in plays_df.columns and "ep_start" in plays_df.columns:
        avg_eqppp = (plays_df["ep_end"] - plays_df["ep_start"]).mean()
    else:
        avg_eqppp = 0.0

    # Punt EqPPP (placeholder — populated by generate_team_st_stats in full pipeline)
    punt_eqppp = 0.0

    # OQ-5 BUG (FAITHFUL PORT): PuntReturnEqPPP = punt_eqppp (same variable)
    # This makes the field-position punt sub-term always zero.
    punt_return_eqppp = punt_eqppp  # BUG: should be punt_ret_eqppp

    return {
        "OffSR": float(off_sr) if off_sr is not None else 0.0,
        "OffER": float(off_er) if off_er is not None else 0.0,
        "AvgEqPPP": float(avg_eqppp) if avg_eqppp is not None else 0.0,
        "punt_eqppp": punt_eqppp,
        "punt_return_eqppp": punt_return_eqppp,  # always == punt_eqppp per OQ-5 bug
    }


# ---------------------------------------------------------------------------
# compute_five_factor_rating — composite 5FR from team stats dict
# ---------------------------------------------------------------------------

def _translate(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linear re-scale value from [in_min, in_max] to [out_min, out_max]."""
    if in_max == in_min:
        return (out_min + out_max) / 2.0
    clamped = max(in_min, min(in_max, value))
    return out_min + (clamped - in_min) / (in_max - in_min) * (out_max - out_min)


def compute_five_factor_rating(team_stats: dict[str, Any]) -> float:
    """Compute the composite Five-Factors Rating (5FR) from a team-stats dict.

    Weights: Efficiency 35%, Explosiveness 30%, Finishing Drives 15%,
             Field Position 10%, Turnovers 10%.

    OQ-5 BUG (FAITHFUL PORT):
    PuntReturnEqPPP is set to punt_eqppp (not punt_return_eqppp), so the punt
    sub-term in the field-position factor is always zero.

    Args:
        team_stats: dict with keys OffSR, OffER, AvgEqPPP, OppRate, OppEff,
                    OppPPD, OppSR, field_pos_quant, ExpTO, ActualTO,
                    SackRate, HavocRate, kickoff_eqppp, kickoff_return_eqppp,
                    punt_eqppp, punt_return_eqppp.

    Returns:
        Composite 5FR as a float.
    """
    from .constants import (
        EFF_DOMAIN,
        EFFICIENCY_WEIGHT,
        EXPLOSIVENESS_WEIGHT,
        FIELD_POS_WEIGHT,
        FIN_DRV_PPD_DOMAIN,
        FIN_DRV_RATE_DOMAIN,
        FIN_DRV_SR_DOMAIN,
        FINISHING_WEIGHT,
        FLD_POS_QUANT_DOMAIN,
        FP_KICK_WEIGHT,
        FP_PUNT_WEIGHT,
        FP_SR_WEIGHT,
        FP_TO_WEIGHT,
        TRNOVR_HAVOC_DOMAIN,
        TRNOVR_LUCK_DOMAIN,
        TRNOVR_SACK_DOMAIN,
        TURNOVER_WEIGHT,
    )

    # --- Efficiency index (OffSR vs opponent OffSR) ---
    off_sr = float(team_stats.get("OffSR", 0.0))
    def_sr = float(team_stats.get("OppSR", 0.5))  # opponent offensive SR against us
    eff_diff = off_sr - def_sr
    eff_index = _translate(eff_diff, *EFF_DOMAIN)

    # --- Explosiveness index (AvgEqPPP vs opponent AvgEqPPP) ---
    avg_eqppp = float(team_stats.get("AvgEqPPP", 0.0))
    def_eqppp = float(team_stats.get("def_AvgEqPPP", 0.0))
    expl_diff = avg_eqppp - def_eqppp
    # Explosiveness uses a data-driven domain; default to EFF_DOMAIN scale
    expl_index = _translate(expl_diff, -2.0, 2.0, 0.0, 10.0)

    # --- Finishing drives index (OppRate, OppEff, OppPPD vs opponent) ---
    opp_rate_diff = float(team_stats.get("OppRate", 0.5)) - float(team_stats.get("def_OppRate", 0.5))
    opp_ppd_diff = float(team_stats.get("OppPPD", 3.0)) - float(team_stats.get("def_OppPPD", 3.0))
    opp_sr_diff = float(team_stats.get("OppSR", 0.4)) - float(team_stats.get("def_OppSR", 0.4))

    fin_ppd = _translate(opp_ppd_diff, *FIN_DRV_PPD_DOMAIN)
    fin_rate = _translate(opp_rate_diff, *FIN_DRV_RATE_DOMAIN)
    fin_sr = _translate(opp_sr_diff, *FIN_DRV_SR_DOMAIN)
    finish_index = fin_ppd + fin_rate + fin_sr

    # --- Field position index ---
    # OQ-5 BUG (FAITHFUL PORT): PuntReturnEqPPP = punt_eqppp (not punt_return_eqppp)
    kickoff_eqppp = float(team_stats.get("kickoff_eqppp", 0.0))
    kickoff_return_eqppp = float(team_stats.get("kickoff_return_eqppp", 0.0))
    punt_eqppp = float(team_stats.get("punt_eqppp", 0.0))
    PuntReturnEqPPP = punt_eqppp  # BUG: uses punt_eqppp instead of punt_return_eqppp

    to_fp = float(team_stats.get("ActualTO", 0.0)) - float(team_stats.get("def_ActualTO", 0.0))

    field_pos_quant = (
        FP_SR_WEIGHT * (off_sr - def_sr)
        + FP_TO_WEIGHT * to_fp
        + FP_KICK_WEIGHT * (kickoff_eqppp - kickoff_return_eqppp)
        + FP_PUNT_WEIGHT * (punt_eqppp - PuntReturnEqPPP)  # always 0 per OQ-5 bug
    )
    fp_index = _translate(field_pos_quant, *FLD_POS_QUANT_DOMAIN)

    # --- Turnover index (ExpTO - ActualTO luck + sack rate + havoc rate) ---
    exp_to = float(team_stats.get("ExpTO", 1.0))
    actual_to = float(team_stats.get("ActualTO", 1.0))
    sack_rate = float(team_stats.get("SackRate", 0.0))
    def_sack_rate = float(team_stats.get("def_SackRate", 0.0))
    havoc_rate = float(team_stats.get("HavocRate", 0.0))
    def_havoc_rate = float(team_stats.get("def_HavocRate", 0.0))

    to_luck = _translate(exp_to - actual_to, *TRNOVR_LUCK_DOMAIN)
    to_sack = _translate(sack_rate - def_sack_rate, *TRNOVR_SACK_DOMAIN)
    to_havoc = _translate(havoc_rate - def_havoc_rate, *TRNOVR_HAVOC_DOMAIN)
    turnover_index = to_luck + to_sack + to_havoc

    # --- Composite 5FR ---
    return (
        EFFICIENCY_WEIGHT * eff_index
        + EXPLOSIVENESS_WEIGHT * expl_index
        + FINISHING_WEIGHT * finish_index
        + FIELD_POS_WEIGHT * fp_index
        + TURNOVER_WEIGHT * turnover_index
    )


# ---------------------------------------------------------------------------
# filter_outliers (polars, z-score based)
# ---------------------------------------------------------------------------

def filter_outliers(
    df: pl.DataFrame,
    z: float = FILTER_Z,
    z2: float = FILTER_Z2,
) -> pl.DataFrame:
    """Remove rows where |5FRDiff| z-score > z OR |PtsDiff| z-score > z2.

    Args:
        df: DataFrame with columns '5FRDiff' and 'PtsDiff'.
        z: Z-score threshold for 5FRDiff (default FILTER_Z = 3.2).
        z2: Z-score threshold for PtsDiff (default FILTER_Z2 = 3.0).

    Returns:
        Filtered DataFrame (outlier rows removed).
    """
    five_fr = pl.col("5FRDiff")
    pts = pl.col("PtsDiff")

    five_fr_mean = df["5FRDiff"].mean()
    five_fr_std = df["5FRDiff"].std()
    pts_mean = df["PtsDiff"].mean()
    pts_std = df["PtsDiff"].std()

    if five_fr_std is None or five_fr_std == 0.0:
        return df
    if pts_std is None or pts_std == 0.0:
        return df

    return df.filter(
        ((five_fr - five_fr_mean).abs() / five_fr_std <= z)
        & ((pts - pts_mean).abs() / pts_std <= z2)
    )


# ---------------------------------------------------------------------------
# load_cfbd_data (CFBD API client — gate import so module loads without cfbd)
# ---------------------------------------------------------------------------

def load_cfbd_data(season: int, api_key: str) -> pl.DataFrame:
    """Fetch team game stats for a season via the CFBD Python client.

    The 'cfbd' package is imported lazily so this module loads cleanly even
    when cfbd is not installed. Tests that call this function are gated with
    @pytest.mark.skipif(not os.getenv("CFB_DATA_API_KEY")).

    Args:
        season: CFB season year (e.g., 2019).
        api_key: CFBD API key (from CFB_DATA_API_KEY env var).

    Returns:
        polars DataFrame with team game stats columns.

    Raises:
        ImportError: if the 'cfbd' package is not installed.
    """
    try:
        import cfbd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'cfbd' package is required for load_cfbd_data. "
            "Install with: uv add cfbd  (or pip install cfbd)"
        ) from exc

    configuration = cfbd.Configuration()
    configuration.api_key["Authorization"] = api_key
    configuration.api_key_prefix["Authorization"] = "Bearer"

    api_client = cfbd.ApiClient(configuration)
    games_api = cfbd.GamesApi(api_client)

    game_stats = games_api.get_team_game_stats(year=season)

    rows: list[dict[str, object]] = []
    for stat in game_stats:
        row: dict[str, object] = {
            "game_id": stat.id,
            "season": season,
            "team": stat.school,
        }
        if stat.teams:
            for team_stat in stat.teams:
                for stat_obj in team_stat.stats or []:
                    col_name = f"{team_stat.school}_{stat_obj.category}"
                    row[col_name] = stat_obj.stat
        rows.append(row)

    if not rows:
        return pl.DataFrame({"game_id": [], "season": [], "team": []})

    import pandas as pd

    pdf = pd.DataFrame(rows)
    return pl.from_pandas(pdf)
