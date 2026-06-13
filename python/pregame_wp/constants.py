"""All numeric constants for the Five-Factors system.

Bug note (OQ-5): PuntReturnEqPPP is assigned punt_eqppp (the punting team's EP),
not punt_ret_eqppp (the return team's EP), in the notebook's generate_team_st_stats.
This makes PuntEqPPP - PuntReturnEqPPP = 0 always. This faithful port preserves that
behavior; the field-position factor's punt term effectively contributes nothing.

Mu/std note (OQ-7): WP conversion uses mu=0.0 (point-differential is symmetric by
construction — each game contributes one +ve and one -ve entry) and std derived from
full-training-set predictions rather than a test-split, which was non-reproducible in
the notebook without a fixed random seed.
"""
from __future__ import annotations

# --- 5FR factor weights (must sum to 1.0) ---
EFF_WEIGHT = 0.35
EXPL_WEIGHT = 0.30
FIN_DRV_WEIGHT = 0.15
FLD_POS_WEIGHT = 0.10
TRNOVR_WEIGHT = 0.10

# --- translate() domain tuples: (inMin, inMax, outMin, outMax) ---
EFF_DOMAIN = (-1.0, 1.0, 0.0, 10.0)
FIN_DRV_PPD_DOMAIN = (-7.0, 7.0, 0.0, 3.5)
FIN_DRV_RATE_DOMAIN = (-1.0, 1.0, 0.0, 4.0)
FIN_DRV_SR_DOMAIN = (-1.0, 1.0, 0.0, 2.5)
FLD_POS_QUANT_DOMAIN = (-10.0, 10.0, 0.0, 10.0)
TRNOVR_LUCK_DOMAIN = (-5.0, 5.0, 0.0, 3.0)
TRNOVR_SACK_DOMAIN = (-1.0, 1.0, 0.0, 3.0)
TRNOVR_HAVOC_DOMAIN = (-1.0, 1.0, 0.0, 4.0)

# --- field position sub-factor weights ---
FP_SR_WEIGHT = 0.37
FP_TO_WEIGHT = 0.21
FP_KICK_WEIGHT = 0.22
FP_PUNT_WEIGHT = 0.20

# --- success rate thresholds (down-specific) ---
SR_DOWN1 = 0.5
SR_DOWN2 = 0.7
SR_DOWN4 = 1.0
EXPLOSIVE_THRESHOLD = 15  # yards; play_explosive = yards_gained >= threshold

# --- scoring opportunity threshold ---
SCORING_OPP_THRESHOLD = 60  # start_yardline + yards >= 60 marks a scoring opp

# --- expected turnover formula weights ---
EXP_TO_INT_WEIGHT = 0.22
EXP_TO_FUM_WEIGHT = 0.49

# --- kickoff / punt thresholds ---
KICKOFF_NET_SUCCESS = 40    # net yards for a successful kickoff
KICKOFF_RETURN_SUCCESS = 24  # return yards for a successful kick return
TOUCHBACK_RETURN_YARDS = 25  # conventional return yards assumed on touchback
PUNT_TOUCHBACK_RETURN_YARDS = 20

# --- training / outlier ---
OUTLIER_Z_5FR = 3.2
OUTLIER_Z_PTS = 3.0
TRAIN_SPLIT = 0.80
XGB_N_ESTIMATORS = 10
XGB_SEED = 123

# --- WP normalization (OQ-7: mu=0.0; std computed from full training preds at train time) ---
WP_MU = 0.0

# --- recruiting talent ---
TALENT_FCS_PERCENTILE = 0.02
RETURNING_PROD_FLOOR_PERCENTILE = 0.02
PRESEASON_WEEKS = 4  # weeks <= PRESEASON_WEEKS trigger returning-production adjustment

# --- home-field advantage adjustments (points) ---
HFA_NORMAL = 2.5
HFA_COVID = 1.0  # 2020 season (reduced/no crowds)
