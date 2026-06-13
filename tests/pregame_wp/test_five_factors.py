import pandas as pd
from pregame_wp.five_factors import (
    translate,
    create_eff_index,
    create_expl_index,
    create_finish_drive_index,
    create_fp_index,
    create_turnover_index,
    calculate_five_factors_rating,
)


def test_translate_midpoint():
    # midpoint of [-1,1] should map to midpoint of [0,10] = 5
    assert translate(0.0, -1.0, 1.0, 0.0, 10.0) == 5.0


def test_translate_min_max():
    assert translate(-1.0, -1.0, 1.0, 0.0, 10.0) == 0.0
    assert translate(1.0, -1.0, 1.0, 0.0, 10.0) == 10.0


def test_translate_clamps_below():
    # value below inMin should clamp to outMin
    assert translate(-2.0, -1.0, 1.0, 0.0, 10.0) == 0.0


def test_translate_clamps_above():
    # value above inMax should clamp to outMax
    assert translate(2.0, -1.0, 1.0, 0.0, 10.0) == 10.0


def test_eff_index_from_zero_diff():
    row = type("R", (), {"OffSRDiff": 0.0})()
    assert create_eff_index(row) == 5.0


def test_eff_index_positive_diff():
    row = type("R", (), {"OffSRDiff": 0.5})()
    idx = create_eff_index(row)
    assert idx > 5.0


def test_expl_index_from_zero_diff():
    row = type("R", (), {"AvgEqPPPDiff": 0.0, "_eq_ppp_min": -2.0, "_eq_ppp_max": 2.0})()
    idx = create_expl_index(row)
    assert abs(idx - 5.0) < 1e-9


def test_finish_drive_index_all_zero():
    row = type("R", (), {"OppPPDDiff": 0.0, "OppRateDiff": 0.0, "OppSRDiff": 0.0})()
    # midpoints of each sub-domain
    from pregame_wp import constants as C
    mid_ppd = (C.FIN_DRV_PPD_DOMAIN[2] + C.FIN_DRV_PPD_DOMAIN[3]) / 2
    mid_rate = (C.FIN_DRV_RATE_DOMAIN[2] + C.FIN_DRV_RATE_DOMAIN[3]) / 2
    mid_sr = (C.FIN_DRV_SR_DOMAIN[2] + C.FIN_DRV_SR_DOMAIN[3]) / 2
    expected = mid_ppd + mid_rate + mid_sr
    assert abs(create_finish_drive_index(row) - expected) < 1e-9


def test_fp_index_from_zero_quant():
    row = type("R", (), {
        "KickoffEqPPP": 0.0, "KickoffReturnEqPPP": 0.0,
        "PuntEqPPP": 0.0, "PuntReturnEqPPP": 0.0,
        "OffSRDiff": 0.0, "ActualTODiff": 0.0, "Plays": 40,
    })()
    idx = create_fp_index(row)
    assert abs(idx - 5.0) < 1e-9  # quant=0 -> translate(0,-10,10,0,10) = 5


def test_turnover_index_balanced():
    # ExpTO == ActualTO → luckDiff = 0; sack/havoc = 0 → full score at midpoint
    row = type("R", (), {
        "ExpTO": 1.0, "ActualTO": 1.0,
        "SackRateDiff": 0.0, "HavocRateDiff": 0.0,
    })()
    idx = create_turnover_index(row)
    # luckDiff=0 → translate(0,-5,5,0,3)=1.5; sack=0→mid=1.5; havoc=0→mid=2.0
    from pregame_wp import constants as C
    mid_luck = (C.TRNOVR_LUCK_DOMAIN[2] + C.TRNOVR_LUCK_DOMAIN[3]) / 2
    mid_sack = (C.TRNOVR_SACK_DOMAIN[2] + C.TRNOVR_SACK_DOMAIN[3]) / 2
    mid_havoc = (C.TRNOVR_HAVOC_DOMAIN[2] + C.TRNOVR_HAVOC_DOMAIN[3]) / 2
    assert abs(idx - (mid_luck + mid_sack + mid_havoc)) < 1e-9


def test_five_factors_rating_symmetry():
    # With all diffs = 0, rating should be near 5.0 (symmetric)
    row = pd.Series({
        "OffSRDiff": 0.0, "AvgEqPPPDiff": 0.0,
        "OppPPDDiff": 0.0, "OppRateDiff": 0.0, "OppSRDiff": 0.0,
        "ActualTODiff": 0.0, "Plays": 40,
        "KickoffEqPPP": 0.0, "KickoffReturnEqPPP": 0.0,
        "PuntEqPPP": 0.0, "PuntReturnEqPPP": 0.0,
        "ExpTO": 1.0, "ActualTO": 1.0,
        "SackRateDiff": 0.0, "HavocRateDiff": 0.0,
        "_eq_ppp_min": -2.0, "_eq_ppp_max": 2.0,
    })
    ffr = calculate_five_factors_rating(row)
    # All sub-indices land at their midpoints → weighted sum = 5.0
    assert abs(ffr - 5.0) < 0.1
