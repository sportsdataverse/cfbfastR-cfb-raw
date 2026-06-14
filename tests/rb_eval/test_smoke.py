"""Integration smoke: features -> aggregate -> train -> validate -> figures.

Skipped when no backfill final.json is present on disk.
"""
import math
import pathlib

import polars as pl  # noqa: E402
import pytest  # noqa: E402

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(
    not any(FINAL_DIR.glob("*.json")),
    reason="no backfill final.json on disk; run scrape_cfb_json.py + reprocess_cfb_json.py first",
)
def test_full_pipeline_runs_without_error(tmp_path):
    from rb_eval.aggregate import build_model_data, build_rusher_seasons
    from rb_eval.features import load_rush_plays
    from rb_eval.figures import write_xrepa_calibration
    from rb_eval.train import loso_cv
    from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2

    rush_df = load_rush_plays(FINAL_DIR, seasons=None)
    assert rush_df.height > 0, "No rush plays loaded from backfill"

    seasons_df = build_rusher_seasons(rush_df)
    model_data = build_model_data(seasons_df)
    if model_data.height == 0:
        pytest.skip("No multi-season rushers in backfill data yet — need ≥2 seasons per rusher")
    assert model_data["epa_per_play"].null_count() == 0

    recent_seasons = sorted(model_data["season"].unique().to_list())[-3:]
    md_small = model_data.filter(pl.col("season").is_in(recent_seasons))
    if md_small.height < 5:
        pytest.skip("Too few rows for a meaningful smoke test with available backfill data")
    cv = loso_cv(md_small)
    assert "exp_rb_epa" in cv.columns

    table = calibration_table(cv)
    err = weighted_cal_error(table)
    r2 = weighted_r2(table)
    assert err >= 0.0
    assert math.isfinite(r2)

    png, csv = write_xrepa_calibration(
        table, tmp_path / "xrepa_calibration", cal_error=err, r2=r2
    )
    assert png.exists()
    assert csv.exists()
