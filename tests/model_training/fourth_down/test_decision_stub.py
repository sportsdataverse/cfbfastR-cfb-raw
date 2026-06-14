import pytest

from model_training.fourth_down.fourth_down_decision import get_go_wp_py


def test_get_go_wp_py_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Track 1"):
        get_go_wp_py(None, None, None, None)
