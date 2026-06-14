import numpy as np
import pandas as pd

from model_training.fourth_down.figures import write_fd_figures


def _synth_cal_table():
    return pd.DataFrame(
        {
            "bin_center": [0.1, 0.3, 0.5, 0.7, 0.9],
            "pred_fd_prob": [0.10, 0.29, 0.51, 0.71, 0.88],
            "empirical_fd_rate": [0.12, 0.28, 0.50, 0.73, 0.87],
            "n_plays": [150, 300, 400, 300, 150],
        }
    )


def _synth_importance():
    return pd.DataFrame(
        {
            "Feature": ["posteam_total", "distance", "yards_to_goal", "down", "posteam_spread"],
            "Gain": [0.38, 0.27, 0.20, 0.10, 0.05],
        }
    )


def test_write_fd_figures_emits_expected_files(tmp_path):
    cal_png, imp_png = write_fd_figures(
        cal_table=_synth_cal_table(),
        importance=_synth_importance(),
        out_dir=tmp_path,
        cal_error=0.021,
    )
    assert cal_png.exists()
    assert imp_png.exists()
    assert (tmp_path / "fd_calibration.csv").exists()
    assert (tmp_path / "fd_feature_importance.csv").exists()
