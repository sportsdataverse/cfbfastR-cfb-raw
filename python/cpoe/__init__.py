"""CFB CPOE (Completion Percentage Over Expected) — Track 5 of the CFB Modeling Suite.

Approach A (8-feature game-state model) only.  Approach B (CFBD air_yards) is
INFEASIBLE — see FEASIBILITY.md for the investigation record.

Public surface
--------------
constants   : feature lists, throw-depth buckets, training hyper-parameters
features    : extract_pass_features(plays_df) → polars DataFrame
train_cp    : train_cp_model(X, y) → booster; save/load helpers
ingest      : load_season_pass_plays(season, ...) → polars DataFrame
loso        : run_loso_cv(seasons, raw_dir) → dict with per-fold metrics
cpoe        : compute_cpoe(plays_df, booster) → polars DataFrame
validate    : calibration metrics + shapley helpers
figures     : calibration + CPOE distribution plots (plotnine / cfbfastR palette)
cli         : __main__ entry-point for the full training pipeline
"""
from __future__ import annotations

__version__ = "0.1.0"
