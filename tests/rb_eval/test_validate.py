import math

import numpy as np
import polars as pl
import pytest

from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2


def _cv_frame(n: int = 40) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    pred = rng.normal(0.0, 0.15, n)
    actual = pred + rng.normal(0.0, 0.05, n)
    return pl.DataFrame({
        "exp_rb_epa": pred.tolist(),
        "target": actual.tolist(),
    })


def test_calibration_table_bins_by_pred_epa():
    cv = _cv_frame()
    table = calibration_table(cv, bin_size=0.05)
    assert "bin_pred_epa" in table.columns
    assert "bin_actual_epa" in table.columns
    assert "total_instances" in table.columns
    for b in table["bin_pred_epa"].to_list():
        remainder = abs(b / 0.05 - round(b / 0.05))
        assert remainder < 1e-9, f"bin {b} not a multiple of 0.05"


def test_weighted_cal_error_is_non_negative():
    cv = _cv_frame(100)
    table = calibration_table(cv)
    err = weighted_cal_error(table)
    assert err >= 0.0


def test_weighted_cal_error_is_zero_for_perfect_calibration():
    table = pl.DataFrame({
        "bin_pred_epa": [-0.1, 0.0, 0.1],
        "bin_actual_epa": [-0.1, 0.0, 0.1],
        "total_instances": [10, 20, 10],
    })
    err = weighted_cal_error(table)
    assert math.isclose(err, 0.0, abs_tol=1e-12)


def test_weighted_r2_is_finite():
    cv = _cv_frame(100)
    table = calibration_table(cv)
    r2 = weighted_r2(table)
    assert math.isfinite(r2)


def test_xrepa_calibration_figure_emits_png_and_csv(tmp_path):
    from rb_eval.figures import write_xrepa_calibration
    cv = _cv_frame(100)
    table = calibration_table(cv)
    png, csv = write_xrepa_calibration(table, tmp_path / "xrepa", cal_error=0.01, r2=0.82)
    assert png.exists()
    assert csv.exists()
