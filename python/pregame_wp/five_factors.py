"""Five-factor index functions + composite 5FR rating.

Faithful port of win-prob.ipynb cell 24 translate / create_*_index /
calculate_five_factors_rating.

OQ-3 resolution: AvgEqPPPDiff (mean EqPPP across ALL offensive plays) drives the
explosiveness index, matching what the model was trained on.  IsoPPP is computed
and stored but NOT used in any index — consistent with the notebook.
"""
from __future__ import annotations

import pandas as pd

from .constants import (
    EFF_DOMAIN,
    EFF_WEIGHT,
    EXPL_WEIGHT,
    FIN_DRV_PPD_DOMAIN,
    FIN_DRV_RATE_DOMAIN,
    FIN_DRV_SR_DOMAIN,
    FIN_DRV_WEIGHT,
    FLD_POS_QUANT_DOMAIN,
    FLD_POS_WEIGHT,
    FP_KICK_WEIGHT,
    FP_PUNT_WEIGHT,
    FP_SR_WEIGHT,
    FP_TO_WEIGHT,
    TRNOVR_HAVOC_DOMAIN,
    TRNOVR_LUCK_DOMAIN,
    TRNOVR_SACK_DOMAIN,
    TRNOVR_WEIGHT,
)


def translate(
    value: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
) -> float:
    """Linear interpolation from [in_min, in_max] → [out_min, out_max], clamped."""
    value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


def create_eff_index(row) -> float:
    """Efficiency index [0–10] from OffSRDiff."""
    return translate(row.OffSRDiff, *EFF_DOMAIN)


def create_expl_index(row) -> float:
    """Explosiveness index [0–10] from AvgEqPPPDiff (OQ-3: not IsoPPP)."""
    eq_min = getattr(row, "_eq_ppp_min", None)
    eq_max = getattr(row, "_eq_ppp_max", None)
    if eq_min is None:
        # fall back to row dict access (e.g. pandas Series)
        eq_min = row["_eq_ppp_min"] if hasattr(row, "__getitem__") else -2.0
        eq_max = row["_eq_ppp_max"] if hasattr(row, "__getitem__") else 2.0
    return translate(row.AvgEqPPPDiff, float(eq_min), float(eq_max), 0.0, 10.0)


def create_finish_drive_index(row) -> float:
    """Finishing drives index [0–10] from PPD + OppRate + OppSR."""
    ppd = translate(row.OppPPDDiff, *FIN_DRV_PPD_DOMAIN)
    rate = translate(row.OppRateDiff, *FIN_DRV_RATE_DOMAIN)
    sr = translate(row.OppSRDiff, *FIN_DRV_SR_DOMAIN)
    return ppd + rate + sr


def create_fp_index(row) -> float:
    """Field position index [0–10] from kick/punt EqPPP + SR + TO."""
    # Kick EP diff
    kick_ep_diff = row.KickoffEqPPP - row.KickoffReturnEqPPP
    # Punt EP diff (OQ-5: PuntReturnEqPPP = PuntEqPPP → this term is always 0)
    punt_ep_diff = row.PuntEqPPP - row.PuntReturnEqPPP
    # OffSR contribution (normalized to [0, 1] range)
    sr_contrib = row.OffSRDiff
    # TO contribution (uses ActualTO sign — positive means fewer turnovers for us)
    to_contrib = -row.ActualTODiff / max(1, float(row.Plays))

    quant = (
        FP_SR_WEIGHT * sr_contrib
        + FP_TO_WEIGHT * to_contrib
        + FP_KICK_WEIGHT * kick_ep_diff
        + FP_PUNT_WEIGHT * punt_ep_diff
    )
    return translate(quant, *FLD_POS_QUANT_DOMAIN)


def create_turnover_index(row) -> float:
    """Turnover index [0–10] from luck (ExpTO-ActualTO), sack rate, havoc rate."""
    luck_diff = row.ExpTO - row.ActualTO
    luck = translate(luck_diff, *TRNOVR_LUCK_DOMAIN)
    sack = translate(row.SackRateDiff, *TRNOVR_SACK_DOMAIN)
    havoc = translate(row.HavocRateDiff, *TRNOVR_HAVOC_DOMAIN)
    return luck + sack + havoc


def calculate_five_factors_rating(row) -> float:
    """Composite 5FR = weighted sum of five factor indices."""
    eff = create_eff_index(row)
    expl = create_expl_index(row)
    fin_drv = create_finish_drive_index(row)
    fp = create_fp_index(row)
    trnovr = create_turnover_index(row)
    return (
        EFF_WEIGHT * eff
        + EXPL_WEIGHT * expl
        + FIN_DRV_WEIGHT * fin_drv
        + FLD_POS_WEIGHT * fp
        + TRNOVR_WEIGHT * trnovr
    )
