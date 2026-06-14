"""CFB pregame win-probability + Five-Factors team ratings (Track 4).

Architecture:
  Five-Factors pipeline → per-game box score → 5FRDiff → 10-tree XGBRegressor
  → normal-CDF WP. Data source is CFBD (not ESPN backfill).

Known bugs (faithful port):
  OQ-5: PuntReturnEqPPP uses punt_eqppp (punter EP) instead of punt_ret_eqppp
        (returner EP), making the punt sub-term in field_pos always zero.
  OQ-7: WP conversion uses mu=0.0 (symmetric by construction) and std derived
        from full training set predictions.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "constants",
    "data_prep",
    "model",
    "wp",
    "validate",
    "figures",
    "cli",
]
