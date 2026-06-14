"""All numeric constants for the Pregame WP + Five-Factors system (Track 4).

Bug note (OQ-5 — FAITHFUL PORT):
  In the notebook's generate_team_st_stats, PuntReturnEqPPP is assigned
  punt_eqppp (the punter's EP) rather than punt_ret_eqppp (the returner's EP),
  making the PuntEqPPP - PuntReturnEqPPP field-position sub-term always zero.
  The variables punt_ret_eqppp / punt_ret_isoppp are computed but never used.
  Phase 0 decision: preserve this behavior (Option A) for parity with the
  trained pgwp_model.model. See docs/superpowers/plans/…-track4-pregame-wp… §0.1.

Mu/std note (OQ-7 resolution):
  WP conversion uses mu=0.0 (point-differential is symmetric; each game has one
  positive and one negative entry) and std derived from full-training-set
  predictions (not a test-split stat). The notebook used test-split statistics
  which is non-reproducible without a fixed seed.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Five-Factor composite weights (must sum to 1.0)
# ---------------------------------------------------------------------------
EFFICIENCY_WEIGHT: float = 0.35
EXPLOSIVENESS_WEIGHT: float = 0.30
FINISHING_WEIGHT: float = 0.15
FIELD_POS_WEIGHT: float = 0.10
TURNOVER_WEIGHT: float = 0.10

# ---------------------------------------------------------------------------
# Down-specific success-rate thresholds
# ---------------------------------------------------------------------------
SR_DOWN1: float = 0.5   # yards_gained >= 0.5 * distance
SR_DOWN2: float = 0.7   # yards_gained >= 0.7 * distance
SR_DOWN4: float = 1.0   # yards_gained >= 1.0 * distance  (must convert)
# Down 3 is intentionally absent from the np.select conditions → default False (OQ-2).

EXPLOSIVE_THRESHOLD: int = 15   # yards for a play to be classified as explosive

# ---------------------------------------------------------------------------
# Scoring opportunity threshold (start_yardline + yards_gained >= threshold)
# ---------------------------------------------------------------------------
SCORING_OPP_THRESHOLD: int = 60

# ---------------------------------------------------------------------------
# translate() linear-scale domains (inMin, inMax, outMin, outMax)
# ---------------------------------------------------------------------------
EFF_DOMAIN: tuple[float, float, float, float] = (-1.0, 1.0, 0.0, 10.0)
FIN_DRV_PPD_DOMAIN: tuple[float, float, float, float] = (-7.0, 7.0, 0.0, 3.5)
FIN_DRV_RATE_DOMAIN: tuple[float, float, float, float] = (-1.0, 1.0, 0.0, 4.0)
FIN_DRV_SR_DOMAIN: tuple[float, float, float, float] = (-1.0, 1.0, 0.0, 2.5)
FLD_POS_QUANT_DOMAIN: tuple[float, float, float, float] = (-10.0, 10.0, 0.0, 10.0)
TRNOVR_LUCK_DOMAIN: tuple[float, float, float, float] = (-5.0, 5.0, 0.0, 3.0)
TRNOVR_SACK_DOMAIN: tuple[float, float, float, float] = (-1.0, 1.0, 0.0, 3.0)
TRNOVR_HAVOC_DOMAIN: tuple[float, float, float, float] = (-1.0, 1.0, 0.0, 4.0)

# ---------------------------------------------------------------------------
# Field-position sub-factor weights (within the quant formula)
# ---------------------------------------------------------------------------
FP_SR_WEIGHT: float = 0.37
FP_TO_WEIGHT: float = 0.21
FP_KICK_WEIGHT: float = 0.22
FP_PUNT_WEIGHT: float = 0.20

# ---------------------------------------------------------------------------
# Expected turnover formula coefficients (spec §4.7)
# ---------------------------------------------------------------------------
EXP_TO_INT_WEIGHT: float = 0.22   # weight for (PD + INT) interceptions component
EXP_TO_FUM_WEIGHT: float = 0.49   # weight for fumbles

# ---------------------------------------------------------------------------
# Kickoff / punt thresholds
# ---------------------------------------------------------------------------
KICKOFF_NET_SUCCESS: int = 40       # yards for a kickoff net to be "successful"
KICKOFF_RETURN_SUCCESS: int = 24    # yards for a kickoff return to be "successful"
TOUCHBACK_RETURN_YARDS: int = 25    # assumed return yards on a touchback
PUNT_TOUCHBACK_RETURN_YARDS: int = 20  # assumed return yards on a punt touchback

# ---------------------------------------------------------------------------
# Training + outlier filter
# ---------------------------------------------------------------------------
PREGAME_WP_PARAMS: dict[str, object] = {"n_estimators": 10}
FILTER_Z: float = 3.2    # z-score threshold for |5FRDiff| outlier removal
FILTER_Z2: float = 3.0   # z-score threshold for |PtsDiff| outlier removal
TRAIN_SPLIT: float = 0.80
XGB_N_ESTIMATORS: int = 10
XGB_SEED: int = 123

# ---------------------------------------------------------------------------
# WP normalization (OQ-7 resolution)
# ---------------------------------------------------------------------------
WP_MU: float = 0.0  # symmetric by construction; std computed from full training preds

# ---------------------------------------------------------------------------
# Home-field advantage (points)
# ---------------------------------------------------------------------------
HFA_NORMAL: float = 2.5
HFA_COVID: float = 1.0   # 2020 COVID season (reduced/absent crowds)

# ---------------------------------------------------------------------------
# Talent / returning production
# ---------------------------------------------------------------------------
TALENT_FCS_PERCENTILE: float = 0.02    # 2nd-percentile FBS rating floor for FCS teams
RETURNING_PROD_FLOOR_PERCENTILE: float = 0.02
PRESEASON_WEEKS: int = 4               # weeks <= 4 trigger returning-production adjustment
