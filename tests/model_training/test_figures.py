import polars as pl

from model_training.figures import write_calibration


def test_write_calibration_emits_png_and_table(tmp_path):
    table = pl.DataFrame({
        "by": ["1st"] * 5,
        "bin": [0.1, 0.3, 0.5, 0.7, 0.9],
        "n_plays": [100, 200, 300, 200, 100],
        "actual": [0.12, 0.28, 0.51, 0.69, 0.93],
    })
    png, csv = write_calibration(
        table,
        tmp_path / "wp_spread",
        title="WP",
        subtitle="LOSO",
        cal_error=0.012,
    )
    assert png.exists() and csv.exists()
