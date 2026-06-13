from pregame_wp import constants as C


def test_factor_weights_sum_to_one():
    total = C.EFF_WEIGHT + C.EXPL_WEIGHT + C.FIN_DRV_WEIGHT + C.FLD_POS_WEIGHT + C.TRNOVR_WEIGHT
    assert abs(total - 1.0) < 1e-9


def test_outlier_thresholds_defined():
    assert C.OUTLIER_Z_5FR == 3.2
    assert C.OUTLIER_Z_PTS == 3.0


def test_success_rate_thresholds():
    assert C.SR_DOWN1 == 0.5
    assert C.SR_DOWN2 == 0.7
    assert C.SR_DOWN4 == 1.0


def test_scoring_opp_threshold():
    assert C.SCORING_OPP_THRESHOLD == 60


def test_xgb_params():
    assert C.XGB_N_ESTIMATORS == 10
    assert C.XGB_SEED == 123


def test_wp_mu_is_zero():
    # OQ-7 resolution: symmetric point-differential -> mu = 0.0
    assert C.WP_MU == 0.0
