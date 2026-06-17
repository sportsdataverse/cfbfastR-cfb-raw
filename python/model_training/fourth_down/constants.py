"""Feature contract, XGBoost params, and label bounds for the fourth-down yards model.

Recipe source: fourth-downs.ipynb (akeaswaran / Jason Lee lineage), confirmed by
cfb4th:::fd_model tree count: 157 rounds × 76 classes = 11932 trees.
"""
from __future__ import annotations

# --- feature contract (exact column order, model was trained on these 5 features) ---
FD_FEATURES: list[str] = [
    "down",
    "distance",
    "yards_to_goal",
    "posteam_total",
    "posteam_spread",
]

# --- label bounds (clip + offset: label = clip(yardsGained, LOW, HIGH) + OFFSET) ---
FD_CLIP_LOW: int = -10  # 10-yard loss = class 0
FD_CLIP_HIGH: int = 65  # 65-yard gain = class 75
FD_LABEL_OFFSET: int = 10
FD_NUM_CLASS: int = 76  # classes 0..75 covering integer gains -10..65

# --- XGBoost params (exact, from notebook cell 7 + _go_for_it_cfb_mod.R lines 188-200) ---
FD_PARAMS: dict = {
    "booster": "gbtree",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": FD_NUM_CLASS,
    "eta": 0.07,
    "gamma": 4.325037e-09,
    "subsample": 0.5385424,
    "colsample_bytree": 0.6666667,
    "max_depth": 4,
    "min_child_weight": 7,
}
FD_NROUNDS: int = 157

# --- source column names in final.json plays ---
FD_SOURCE: dict[str, str] = {
    "down": "start.down",
    "distance": "start.distance",
    "yards_to_goal": "start.yardsToEndzone",
    # posteam_total is DERIVED (not a direct source column)
    # posteam_spread is read from start.pos_team_spread
    "posteam_spread": "start.pos_team_spread",
}

# Spread + total source columns (doc-level, broadcast to every play by CFBPlayProcess)
FD_SPREAD_COL: str = "homeTeamSpread"  # home-team-perspective spread (negative = home favored)
FD_OVERUNDER_COL: str = "overUnder"  # game total
FD_IS_HOME_COL: str = "start.is_home"  # 1/True if possessing team is home
FD_YARDS_GAINED_COL: str = "yardsGained"  # label source
FD_RUSH_COL: str = "rush"  # boolean/int — play filter
FD_PASS_COL: str = "pass"  # boolean/int — play filter
FD_FIRST_DOWN_PENALTY_COLS: tuple[str, ...] = ("firstD_by_penalty", "start.firstD_by_penalty")
