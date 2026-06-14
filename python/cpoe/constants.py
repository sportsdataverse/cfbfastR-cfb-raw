"""Feature contracts, source-column crosswalk, and XGBoost params for the CFB CPOE model.

Approach A: 8 game-state features from ESPN final.json pass plays (always populated).
Approach B: 9-feature extension with CFBD air_yards (gated by Phase 0 Task 0.2 verdict).

The StatsBomb-trained R original (`cpoe_model.R`) uses five throw-level features that
are entirely absent from the ESPN CFB play-by-play backfill. See FEASIBILITY.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# StatsBomb original feature set — REFERENCE ONLY; absent from ESPN CFB pbp
# ---------------------------------------------------------------------------
STATSBOMB_FEATURES: list[str] = [
    "event_pass_air_yards",       # throw distance through air (Euclidean)
    "play_target_separation",     # yards from nearest defender at catch point
    "play_qb_pressure",           # QB under pressure (bool, null->False)
    "endline_receiver_dist",      # 110 - event_pass_target_x
    "sideline_receiver_dist",     # min(y, 53.33-y) from event_pass_target_y
]
STATSBOMB_NROUNDS: int = 560
STATSBOMB_SEASONS: list[int] = list(range(2017, 2023))  # 2017-18 through 2022-23

# ---------------------------------------------------------------------------
# Approach A: game-state features — ESPN final.json (CFBPlayProcess output)
# ---------------------------------------------------------------------------
# Maps canonical feature name (used in model) → source column in final.json plays.
CPOE_SOURCE_COLS: dict[str, str] = {
    "down":           "start.down",
    "distance":       "start.distance",
    "yards_to_goal":  "start.yardsToEndzone",
    "pos_score_diff": "pos_score_diff_start",
    "secs_remaining": "start.TimeSecsRem",
    "is_home":        "start.is_home",
    "period":         "period",
    # passing_down is derived (not a raw ESPN column); set last so derivation happens first
    "passing_down":   "passing_down",
}

# The ordered feature list used in DMatrix / model column contract.
CPOE_FEATURES: list[str] = list(CPOE_SOURCE_COLS.keys())  # length == 8

# ---------------------------------------------------------------------------
# Approach B extension (conditional on Phase 0 Task 0.2)
# If CFBD air_yards fill rate >= 60%, add this feature for post-2020 seasons.
# ---------------------------------------------------------------------------
CPOE_SOURCE_COLS_B: dict[str, str] = {
    **CPOE_SOURCE_COLS,
    "air_yards": "air_yards",   # from CFBD PBP join; null where unavailable
}
CPOE_FEATURES_B: list[str] = list(CPOE_SOURCE_COLS_B.keys())  # length == 9

# ---------------------------------------------------------------------------
# XGBoost params — Approach A (binary:logistic)
# nrounds tuned by LOSO CV in Phase 3.
# ---------------------------------------------------------------------------
CPOE_PARAMS: dict = {
    "booster":          "gbtree",
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "eta":              0.025,
    "gamma":            5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "max_depth":        4,
    "min_child_weight": 5,
}
CPOE_NROUNDS: int = 400  # starting point; tune in Phase 3 LOSO CV

# ---------------------------------------------------------------------------
# Distance buckets — yards-to-first-down proxy for throw depth
# Note: this is a COARSE proxy for air yards; document on every calibration figure.
# ---------------------------------------------------------------------------
DISTANCE_BUCKETS: dict[str, tuple[int, int]] = {
    "Short":        (0, 3),    # distance <= 3
    "Intermediate": (4, 8),    # 4 <= distance <= 8
    "Long":         (9, 9999), # distance >= 9
}

# ---------------------------------------------------------------------------
# Minimum pass attempts for a QB-season CPOE to be considered reliable
# ---------------------------------------------------------------------------
MIN_ATTEMPTS_SEASON: int = 100
