"""Shared constants for the CFB CPOE pipeline (Track 5, Approach A).

Hyper-parameters mirror the nflfastR `cpoe_model.R` recipe:
    binary:logistic, eta=0.025, gamma=5, subsample=0.8,
    colsample_bytree=0.8, max_depth=4, min_child_weight=6, nrounds=560.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Feature / target columns
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "down",
    "distance",
    "yards_to_goal",
    "score_diff",
    "seconds_remaining",
    "is_home",
    "period",
    "passing_down",
]

TARGET_COL: str = "completion"

# ---------------------------------------------------------------------------
# Throw-depth proxy buckets (yards-to-first-down based; open upper on "long")
# (lo, hi_inclusive)  —  hi=None means unbounded.
# ---------------------------------------------------------------------------

THROW_DEPTH_BUCKETS: dict[str, tuple[int, int | None]] = {
    "short": (0, 3),
    "intermediate": (4, 8),
    "long": (9, None),
}

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters (exact nflfastR parity)
# ---------------------------------------------------------------------------

XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eta": 0.025,
    "gamma": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 4,
    "min_child_weight": 6,
    "eval_metric": "logloss",
}

XGB_NROUNDS: int = 560

# ---------------------------------------------------------------------------
# Pass-play type filter (ESPN playType values)
# ---------------------------------------------------------------------------

PASS_PLAY_TYPES: frozenset[str] = frozenset(
    {
        "Pass Reception",
        "Pass Incompletion",
        "Pass Interception Return",
        "Passing Touchdown",
        "Sack",
        "Interception Return Touchdown",
        "Pass",
    }
)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

MIN_SEASON: int = 2014
MODEL_FILENAME: str = "cfb_cp_model.ubj"
