"""Feature contracts, column crosswalk, EP class mapping, and XGBoost params.

The EP/WP/QBR feature lists are imported from sdv-py's model_vars at runtime so this
module can NEVER drift from the shipped inference contract (a test asserts equality).
"""
from __future__ import annotations

from sportsdataverse.cfb import model_vars as _mv

# --- shipped inference contracts (re-exported for clarity + a drift test) ---
EP_FEATURES: list[str] = list(_mv.ep_final_names)            # 8
WP_SPREAD_FEATURES: list[str] = list(_mv.wp_final_names)     # 13
WP_NAIVE_FEATURES: list[str] = [c for c in _mv.wp_final_names if c != "spread_time"]  # 12
QBR_FEATURES: list[str] = list(_mv.qbr_vars)                 # 6
EP_CLASS_TO_SCORE: dict[int, int] = dict(_mv.ep_class_to_score_mapping)
# class order: 0 TD, 1 Opp_TD, 2 FG, 3 Opp_FG, 4 Safety, 5 Opp_Safety, 6 No_Score
NEXT_SCORE_TO_LABEL: dict[str, int] = {
    "Touchdown": 0, "Opp_Touchdown": 1, "Field_Goal": 2, "Opp_Field_Goal": 3,
    "Safety": 4, "Opp_Safety": 5, "No_Score": 6,
}

# --- CFBPlayProcess (final.json plays) -> the columns the EP/WP feature builders need.
EP_SOURCE = {
    "TimeSecsRem": "start.TimeSecsRem", "yards_to_goal": "start.yardsToEndzone",
    "distance": "start.distance", "down_1": "down_1", "down_2": "down_2",
    "down_3": "down_3", "down_4": "down_4", "pos_score_diff_start": "pos_score_diff_start",
}
WP_SOURCE = {
    "pos_team_receives_2H_kickoff": "start.pos_team_receives_2H_kickoff",
    "spread_time": "start.spread_time", "TimeSecsRem": "start.TimeSecsRem",
    "adj_TimeSecsRem": "start.adj_TimeSecsRem",
    "ExpScoreDiff_Time_Ratio": "start.ExpScoreDiff_Time_Ratio",
    "pos_score_diff_start": "pos_score_diff_start", "down": "start.down",
    "distance": "start.distance", "yards_to_goal": "start.yardsToEndzone",
    "is_home": "start.is_home", "pos_team_timeouts_rem_before": "start.posTeamTimeouts",
    "def_pos_team_timeouts_rem_before": "start.defPosTeamTimeouts", "period": "period",
}

# --- labeling source columns (final.json plays) ---
LBL = {
    "game_id": "game_id", "drive_id": "drive.id", "period": "period",
    "pos_team": "pos_team", "def_pos_team": "def_pos_team",
    "scoring_play": "scoring_play", "offense_score_play": "offense_score_play",
    "defense_score_play": "defense_score_play", "play_type": "type.text",
    "pos_score_diff": "pos_score_diff_start",
}

# --- XGBoost params (exact, confirmed recipes — do not alter the numbers) ---
EP_PARAMS = dict(booster="gbtree", objective="multi:softprob", eval_metric="mlogloss",
                 num_class=7, eta=0.025, gamma=1, subsample=0.8, colsample_bytree=0.8,
                 max_depth=5, min_child_weight=1)
EP_NROUNDS = 525

WP_SPREAD_PARAMS = dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss",
                        eta=0.02, gamma=0.3445502, subsample=0.7204741,
                        colsample_bytree=0.5714286, max_depth=5, min_child_weight=14)
WP_SPREAD_NROUNDS = 760
WP_NAIVE_PARAMS = dict(booster="gbtree", objective="binary:logistic", eval_metric="logloss",
                       eta=0.2, gamma=0, subsample=0.8, colsample_bytree=0.8,
                       max_depth=4, min_child_weight=1)
WP_NAIVE_NROUNDS = 65

# Stage-1 (divergent keepers `03`) WP-spread params — replica target only.
WP_SPREAD_PARAMS_STAGE1 = dict(booster="gbtree", objective="binary:logistic",
                               eval_metric="logloss", eta=0.05, gamma=0.79012017,
                               subsample=0.9224245, colsample_bytree=5 / 12, max_depth=5,
                               min_child_weight=7)
WP_SPREAD_NROUNDS_STAGE1 = 534

QBR_PARAMS = dict(booster="gbtree", objective="reg:squarederror", eta=0.1,
                  subsample=0.8, colsample_bytree=0.8, max_depth=4, min_child_weight=1)
QBR_NROUNDS = 45  # matches shipped qbr_model.ubj tree count

# Known-bad games excluded by keepers 02/03 + model_training.R (ESPN data defects).
BAD_GAME_IDS: set[int] = {
    400603838, 401020760, 400933849, 400547737, 400547739, 401012806,
    401021693, 400787470, 401112262, 401114227, 401147693, 401015042,
    400986609, 400763439,
}
