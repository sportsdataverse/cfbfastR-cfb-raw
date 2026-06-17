"""CFB pregame win-probability + Five-Factors team ratings (Track 4, CFB Modeling Suite).

Usage:
    from pregame_wp.ep_curve import load_ep_curve, load_punt_sr, ep_at, eqppp
    from pregame_wp.play_features import add_play_features
    from pregame_wp.team_stats import generate_team_play_stats, generate_team_drive_stats
    from pregame_wp.five_factors import translate, calculate_five_factors_rating
    from pregame_wp.box_score import calculate_box_score_from_frames
    from pregame_wp.training import filter_outliers, train_pgwp_model, save_pgwp_model
    from pregame_wp.predict import five_fr_to_wp, generate_win_prob
    from pregame_wp.talent import calculate_roster_talent, calculate_returning_production
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import data_ingest
from .box_score import calculate_box_score_from_frames
from .ep_curve import ep_at, eqppp, load_ep_curve, load_punt_sr
from .five_factors import calculate_five_factors_rating, translate
from .play_features import add_play_features
from .predict import five_fr_to_wp, generate_win_prob
from .talent import calculate_returning_production, calculate_roster_talent
from .team_stats import (
    generate_team_drive_stats,
    generate_team_play_stats,
    generate_team_st_stats,
    generate_team_turnover_stats,
)
from .training import filter_outliers, save_pgwp_model, train_pgwp_model

__all__ = [
    "__version__",
    "load_ep_curve", "load_punt_sr", "ep_at", "eqppp",
    "add_play_features",
    "generate_team_play_stats", "generate_team_drive_stats",
    "generate_team_turnover_stats", "generate_team_st_stats",
    "translate", "calculate_five_factors_rating",
    "calculate_box_score_from_frames",
    "filter_outliers", "train_pgwp_model", "save_pgwp_model",
    "five_fr_to_wp", "generate_win_prob",
    "calculate_roster_talent", "calculate_returning_production",
]
