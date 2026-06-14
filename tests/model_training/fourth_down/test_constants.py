from model_training.fourth_down import constants as C


def test_feature_count():
    assert len(C.FD_FEATURES) == 5


def test_feature_order():
    assert C.FD_FEATURES == ["down", "distance", "yards_to_goal", "posteam_total", "posteam_spread"]


def test_params_objective():
    assert C.FD_PARAMS["objective"] == "multi:softprob"
    assert C.FD_PARAMS["num_class"] == 76
    assert C.FD_PARAMS["eta"] == 0.07
    assert abs(C.FD_PARAMS["gamma"] - 4.325037e-09) < 1e-15


def test_nrounds_and_label_math():
    assert C.FD_NROUNDS == 157
    assert C.FD_NUM_CLASS == 76
    assert C.FD_NROUNDS * C.FD_NUM_CLASS == 11932
    assert C.FD_CLIP_LOW == -10
    assert C.FD_CLIP_HIGH == 65
    assert C.FD_LABEL_OFFSET == 10
    # class 0 = 10-yard loss; class 75 = 65-yard gain
    assert C.FD_CLIP_LOW + C.FD_LABEL_OFFSET == 0
    assert C.FD_CLIP_HIGH + C.FD_LABEL_OFFSET == 75
