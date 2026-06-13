from pregame_wp.ep_curve import load_ep_curve, load_punt_sr, ep_at, eqppp


def test_ep_curve_has_101_entries():
    ep = load_ep_curve()
    assert len(ep) == 101  # yardlines 0-100


def test_ep_at_midfield_is_positive():
    ep = load_ep_curve()
    assert ep_at(ep, 50) > 0  # possession at own 50 is positive EP


def test_eqppp_10_yard_gain_from_20():
    ep = load_ep_curve()
    val = eqppp(ep, yard_line=20, yards_gained=10)
    # EP should increase from yard_line 20 to 30
    assert val == ep_at(ep, 30) - ep_at(ep, 20)


def test_eqppp_clamps_at_100():
    ep = load_ep_curve()
    # 90-yard gain from yl=80 should clamp to ep[100] - ep[80]
    val = eqppp(ep, yard_line=80, yards_gained=90)
    assert val == ep_at(ep, 100) - ep_at(ep, 80)


def test_punt_sr_has_100_entries():
    punt_sr = load_punt_sr()
    assert len(punt_sr) == 100
