import numpy as np
from pregame_wp.predict import five_fr_to_wp


def test_wp_is_50_when_mov_equals_mu():
    wp = five_fr_to_wp(0.0, mu=0.0, std=10.0)
    assert abs(wp - 0.5) < 1e-9


def test_wp_greater_than_50_for_positive_mov():
    wp = five_fr_to_wp(7.0, mu=0.0, std=10.0)
    assert wp > 0.5


def test_wp_less_than_50_for_negative_mov():
    wp = five_fr_to_wp(-7.0, mu=0.0, std=10.0)
    assert wp < 0.5


def test_wp_in_unit_interval():
    for mov in [-50.0, -10.0, 0.0, 10.0, 50.0]:
        wp = five_fr_to_wp(mov, mu=0.0, std=10.0)
        assert 0.0 < wp < 1.0


def test_wp_is_symmetric():
    wp_pos = five_fr_to_wp(14.0, mu=0.0, std=10.0)
    wp_neg = five_fr_to_wp(-14.0, mu=0.0, std=10.0)
    assert abs(wp_pos + wp_neg - 1.0) < 1e-9
