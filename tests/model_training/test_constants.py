from model_training import constants as C


def test_ep_class_to_score_matches_sdvpy():
    from sportsdataverse.cfb.model_vars import ep_class_to_score_mapping
    assert C.EP_CLASS_TO_SCORE == ep_class_to_score_mapping


def test_feature_lists_match_sdvpy_contract():
    from sportsdataverse.cfb import model_vars as mv
    assert C.EP_FEATURES == mv.ep_final_names
    assert C.WP_SPREAD_FEATURES == mv.wp_final_names
    assert C.WP_NAIVE_FEATURES == [c for c in mv.wp_final_names if c != "spread_time"]
    assert C.QBR_FEATURES == mv.qbr_vars
