"""CFB RB-eval (DAKOTA xREPA) model — Track 3 of the CFB Modeling Suite.

Port of rb_eval_model.R (DAKOTA lineage). Produces per-rusher-season expected rushing
EPA (xREPA) from a pygam.LinearGAM(s(0) + s(1)) fit on prior-season epa_per_play and success.

Pipeline stages:
  1. features  — load final.json rush plays, compute fo_success + is_rush_opp
  2. aggregate — per-rusher-season group-by, epa clamp, lag, Pythagorean weight
  3. model     — LinearGAM fit, LOSO CV, save/load
  4. validate  — calibration table, weighted cal-error, weighted R²
  5. figures   — calibration PNG via model_training.figures
  6. cli       — features | aggregate | train | validate | figures subcommands
"""
from __future__ import annotations

__version__ = "0.1.0"
