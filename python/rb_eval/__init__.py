"""CFB RB-eval (DAKOTA xREPA) model — Track 3 of the CFB Modeling Suite.

Usage:
    from rb_eval.features import load_rush_plays
    from rb_eval.aggregate import build_rusher_seasons, build_model_data
    from rb_eval.train import train_xrepa, loso_cv, save_model, load_model
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
    from rb_eval.figures import write_xrepa_calibration
"""
from __future__ import annotations

__version__ = "0.1.0"

from .aggregate import add_season_lag, build_model_data, build_rusher_seasons, summarize_rusher_seasons
from .features import add_fo_success, filter_rush_plays, load_rush_plays
from .train import loso_cv, load_model, save_model, train_xrepa
from .validate import calibration_table, weighted_cal_error, weighted_r2

__all__ = [
    "__version__",
    "add_fo_success", "filter_rush_plays", "load_rush_plays",
    "add_season_lag", "summarize_rusher_seasons", "build_rusher_seasons", "build_model_data",
    "train_xrepa", "loso_cv", "save_model", "load_model",
    "calibration_table", "weighted_cal_error", "weighted_r2",
]
