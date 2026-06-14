"""CFB CPOE (Completion Percentage Over Expected) — Track 5 of the CFB Modeling Suite.

Game-state completion probability model trained on ESPN final.json pass plays.
NOT a port of the StatsBomb-trained cpoe_model.R — all five StatsBomb throw-level
features are absent from the ESPN CFB backfill. See FEASIBILITY.md.

Architecture: Approach A — 8 game-state features from CFBPlayProcess output,
binary:logistic XGBoost, LOSO calibration.
"""
from __future__ import annotations

__version__ = "0.1.0"
