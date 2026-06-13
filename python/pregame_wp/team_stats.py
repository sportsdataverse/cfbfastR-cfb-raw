"""Per-team per-game statistics for the Five-Factors pipeline.

Faithful port of win-prob.ipynb cells 22 and 24.

OQ-5 note: generate_team_st_stats assigns punt_eqppp (not punt_ret_eqppp) to
PuntReturnEqPPP, matching the notebook bug. PuntEqPPP - PuntReturnEqPPP = 0 always.
"""
from __future__ import annotations

import pandas as pd

from .constants import EXP_TO_FUM_WEIGHT, EXP_TO_INT_WEIGHT, SCORING_OPP_THRESHOLD
from .ep_curve import ep_at


# ---------------------------------------------------------------------------
# Play-level stats
# ---------------------------------------------------------------------------

def generate_team_play_stats(
    df: pd.DataFrame,
    team: str,
    off_types: list[str],
    st_types: list[str],
) -> pd.DataFrame:
    """OffSR, OffER, AvgEqPPP, IsoPPP for one team in one game."""
    off = df[(df["offense"] == team) & (df["play_type"].isin(off_types))].copy()
    if off.empty:
        return pd.DataFrame([{"Team": team, "OffSR": 0.0, "OffER": 0.0,
                              "AvgEqPPP": 0.0, "IsoPPP": 0.0, "Plays": 0}])
    n = len(off)
    off_sr = off["play_successful"].mean()
    off_er = off["play_explosive"].mean()
    avg_eqppp = off["EqPPP"].mean()
    successful = off[off["play_successful"] == True]  # noqa: E712
    iso_ppp = successful["EqPPP"].mean() if not successful.empty else 0.0
    return pd.DataFrame([{
        "Team": team,
        "OffSR": off_sr,
        "OffER": off_er,
        "AvgEqPPP": avg_eqppp,
        "IsoPPP": iso_ppp,
        "Plays": n,
    }])


# ---------------------------------------------------------------------------
# Drive-level stats
# ---------------------------------------------------------------------------

def generate_team_drive_stats(
    df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """OppRate, OppEff, OppPPD for one team in one game."""
    drives = df[df["offense"] == team].copy()
    if drives.empty:
        return pd.DataFrame([{"Team": team, "OppRate": 0.0, "OppEff": 0.0,
                              "OppPPD": 0.0, "OppSR": 0.0}])
    n_drives = len(drives)
    scoring_opps = drives[
        drives["drive_start_yardline"] + drives["drive_yards"] >= SCORING_OPP_THRESHOLD
    ]
    n_opps = len(scoring_opps)
    opp_rate = n_opps / n_drives if n_drives > 0 else 0.0
    if n_opps == 0:
        return pd.DataFrame([{"Team": team, "OppRate": opp_rate, "OppEff": 0.0,
                              "OppPPD": 0.0, "OppSR": 0.0}])
    opp_eff = scoring_opps["drive_scoring"].mean()
    opp_ppd = scoring_opps["drive_pts"].mean()
    opp_sr = scoring_opps["drive_scoring"].sum() / n_drives
    return pd.DataFrame([{
        "Team": team,
        "OppRate": opp_rate,
        "OppEff": opp_eff,
        "OppPPD": opp_ppd,
        "OppSR": opp_sr,
    }])


# ---------------------------------------------------------------------------
# Turnover stats
# ---------------------------------------------------------------------------

def generate_team_turnover_stats(
    df: pd.DataFrame,
    offense: str,
    defense: str,
) -> pd.DataFrame:
    """ExpTO, ActualTO, HavocRate, SackRate for one team in one game (offense perspective)."""
    off = df[df["offense"] == offense].copy()
    if off.empty:
        return pd.DataFrame([{"Team": offense, "ExpTO": 0.0, "ActualTO": 0,
                              "HavocRate": 0.0, "SackRate": 0.0}])

    n_plays = len(off)

    # Pass deflections: incomplete passes with "broken up" in play_text
    pds = off[
        (off["play_type"] == "Pass Incompletion")
        & (off["play_text"].str.contains("broken up", case=False, na=False))
    ]
    n_pd = len(pds)

    # Interceptions
    ints = off[off["play_type"] == "Interception"]
    n_int = len(ints)

    # Fumbles recovered by opponent
    fums = off[off["play_type"] == "Fumble Recovery (Opponent)"]
    n_fum = len(fums)

    exp_to = EXP_TO_INT_WEIGHT * (n_pd + n_int) + EXP_TO_FUM_WEIGHT * n_fum
    actual_to = n_int + n_fum

    # Havoc: interceptions + fumbles recovered by defense + sacks
    sacks = off[off["play_type"] == "Sack"]
    n_sack = len(sacks)
    havoc = n_int + n_fum + n_sack
    havoc_rate = havoc / n_plays if n_plays > 0 else 0.0
    sack_rate = n_sack / n_plays if n_plays > 0 else 0.0

    return pd.DataFrame([{
        "Team": offense,
        "ExpTO": exp_to,
        "ActualTO": actual_to,
        "HavocRate": havoc_rate,
        "SackRate": sack_rate,
    }])


# ---------------------------------------------------------------------------
# Special teams stats
# ---------------------------------------------------------------------------

def generate_team_st_stats(
    df: pd.DataFrame,
    team: str,
    ep_data: list[float],
    punt_sr: dict[int, float],
) -> pd.DataFrame:
    """Kickoff/punt ST stats including EqPPP values.

    OQ-5 faithful port: PuntReturnEqPPP = PuntEqPPP (punt_eqppp), NOT punt_ret_eqppp.
    This means PuntEqPPP - PuntReturnEqPPP = 0 always, matching the notebook.
    """
    kicks = df[(df["offense"] == team) & (df["play_type"] == "Kickoff")].copy()
    punts = df[(df["offense"] == team) & (df["play_type"] == "Punt")].copy()

    # --- kickoff ---
    if kicks.empty:
        kick_sr, kick_eqppp = 0.0, 0.0
        kick_ret_eqppp = 0.0
    else:
        kick_sr = kicks["kick_yards"].mean() / 100.0 if "kick_yards" in kicks.columns else 0.0
        kick_eqppp = kicks.apply(
            lambda x: ep_at(ep_data, int(x.get("yard_line", 35) + x.get("kick_yards", 0)))
            - ep_at(ep_data, int(x.get("yard_line", 35))),
            axis=1,
        ).mean()
        kick_ret_eqppp = kicks.apply(
            lambda x: ep_at(ep_data, int(x.get("return_yards", 0))),
            axis=1,
        ).mean()

    # --- punt ---
    if punts.empty:
        punt_sr_val, punt_eqppp, punt_ret_eqppp = 0.0, 0.0, 0.0
    else:
        punt_sr_val = punts.apply(
            lambda x: float(x.get("kick_yards", 0) > (
                punt_sr.get(int(x.get("yard_line", 50)), 40.0)
            )),
            axis=1,
        ).mean()
        punt_eqppp = punts.apply(
            lambda x: ep_at(ep_data, int(x.get("yard_line", 30) + x.get("kick_yards", 0)))
            - ep_at(ep_data, int(x.get("yard_line", 30))),
            axis=1,
        ).mean()
        punt_ret_eqppp = punts.apply(
            lambda x: ep_at(ep_data, int(x.get("return_yards", 0))),
            axis=1,
        ).mean()

    return pd.DataFrame([{
        "Team": team,
        "KickoffSR": kick_sr,
        "KickoffEqPPP": kick_eqppp,
        "KickoffReturnEqPPP": kick_ret_eqppp,
        "PuntSR": punt_sr_val,
        "PuntEqPPP": punt_eqppp,
        # OQ-5: faithful port uses punt_eqppp (not punt_ret_eqppp) for PuntReturnEqPPP
        "PuntReturnEqPPP": punt_eqppp,
        "PuntReturnIsoPPP": punt_ret_eqppp,
    }])
