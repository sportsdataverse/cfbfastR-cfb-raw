"""Constants for the RB-Eval xREPA pipeline.

These mirror the spec §5 design document and the DAKOTA R source.
"""
from __future__ import annotations

# Features fed to the LinearGAM (prior-season values, post-lag and rename)
RB_EVAL_FEATURES: list[str] = ["epa_per_play", "success"]

# Target column (current-season unadjusted/unclamped EPA per play)
RB_EVAL_TARGET: str = "unadjusted_epa"

# Minimum rushing plays for a rusher-season to be included in the model
RB_EVAL_MIN_PLAYS: int = 100

# EPA per-play clamp floor (plays below this are clamped before averaging)
RB_EVAL_EPA_CLAMP: float = -4.5

# Minimum yards for a rush to count as a "rush opportunity" (for highlight_yards)
RB_EVAL_RUSH_OPP_THRESHOLD: int = 4

# Column name written by loso_cv for out-of-fold predictions
RB_EVAL_PRED_COL: str = "exp_rb_epa"
