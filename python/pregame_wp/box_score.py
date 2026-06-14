"""Full-game box score: per-team Five-Factors stats + 5FR + 5FRDiff.

Port of win-prob.ipynb cell 24 calculate_box_score.
"""
from __future__ import annotations

import pandas as pd

from .five_factors import calculate_five_factors_rating
from .play_features import add_play_features
from .team_stats import (
    generate_team_drive_stats,
    generate_team_play_stats,
    generate_team_st_stats,
    generate_team_turnover_stats,
)

_OFF_TYPES = [
    "Rush", "Pass Reception", "Pass Incompletion", "Rushing Touchdown",
    "Passing Touchdown", "Fumble Recovery (Opponent)", "Sack",
]
_ST_TYPES = ["Kickoff", "Punt", "Field Goal Good", "Field Goal Missed", "Kickoff Return TD"]
_BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]


def calculate_box_score_from_frames(
    plays: pd.DataFrame,
    drives: pd.DataFrame,
    ep_data: list[float],
    punt_sr: dict[int, float],
    eq_ppp_global_min: float = -2.0,
    eq_ppp_global_max: float = 2.0,
) -> pd.DataFrame:
    """Compute per-team 5FR box score from pre-loaded play/drive frames.

    Args:
        plays: Play-by-play with columns offense, defense, play_type, down,
               distance, yards_gained, yard_line, play_text.
        drives: Drive log with offense, defense, drive_start_yardline,
                drive_yards, drive_scoring, drive_pts.
        ep_data: EP curve list (len 101).
        punt_sr: {yardline: ExpPuntNet} dict.
        eq_ppp_global_min: Global EqPPP min from training PBP (for expl index domain).
        eq_ppp_global_max: Global EqPPP max from training PBP.

    Returns:
        DataFrame with one row per team: OffSR, OffER, AvgEqPPP, IsoPPP,
        OppRate, OppEff, OppPPD, OppSR, ExpTO, ActualTO, HavocRate, SackRate,
        KickoffEqPPP, PuntEqPPP, PuntReturnEqPPP, 5FR, 5FRDiff.
    """
    teams = sorted(plays["offense"].unique())
    if len(teams) != 2:
        raise ValueError(f"Expected exactly 2 teams, got {teams}")

    # Enrich plays with EqPPP / play_successful / play_explosive
    plays = add_play_features(plays, ep_data, _ST_TYPES, _BAD_TYPES)

    rows = []
    for team in teams:
        opponent = [t for t in teams if t != team][0]
        play_stats = generate_team_play_stats(plays, team, _OFF_TYPES, _ST_TYPES)
        drive_stats = generate_team_drive_stats(drives, team)
        to_stats = generate_team_turnover_stats(plays, team, opponent)
        st_stats = generate_team_st_stats(plays, team, ep_data, punt_sr)

        row = {
            "Team": team,
            **{c: play_stats[c].iloc[0] for c in ["OffSR", "OffER", "AvgEqPPP", "IsoPPP", "Plays"]},
            **{c: drive_stats[c].iloc[0] for c in ["OppRate", "OppEff", "OppPPD", "OppSR"]},
            **{c: to_stats[c].iloc[0] for c in ["ExpTO", "ActualTO", "HavocRate", "SackRate"]},
            **{c: st_stats[c].iloc[0] for c in [
                "KickoffSR", "KickoffEqPPP", "KickoffReturnEqPPP",
                "PuntSR", "PuntEqPPP", "PuntReturnEqPPP",
            ]},
        }
        rows.append(row)

    box = pd.DataFrame(rows)

    # Compute per-factor diffs (team - opponent)
    for stat in ["OffSR", "AvgEqPPP", "OppPPD", "OppRate", "OppSR", "ActualTO",
                 "SackRate", "HavocRate"]:
        box[f"{stat}Diff"] = box[stat] - box[stat].iloc[::-1].values

    # Attach global EqPPP bounds for explosiveness domain
    box["_eq_ppp_min"] = eq_ppp_global_min
    box["_eq_ppp_max"] = eq_ppp_global_max

    # 5FR composite
    box["5FR"] = box.apply(calculate_five_factors_rating, axis=1)

    # 5FRDiff is antisymmetric (A's diff = -B's diff)
    box["5FRDiff"] = box["5FR"] - box["5FR"].iloc[::-1].values

    return box.reset_index(drop=True)
