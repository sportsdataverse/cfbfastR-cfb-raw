import json
import pathlib

import numpy as np
import polars as pl
import pytest

from rb_eval.train import loso_cv, load_model, save_model, train_xrepa


def _synth_model_data(n: int = 80) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "rusher_player_name": [f"R{i % 20}" for i in range(n)],
        "season": [2010 + i % 5 for i in range(n)],
        "epa_per_play": rng.normal(0.0, 0.3, n).tolist(),
        "success": rng.uniform(0.3, 0.7, n).tolist(),
        "target": rng.normal(0.0, 0.2, n).tolist(),
        "highlight_yards": rng.uniform(0.0, 2.0, n).tolist(),
        "weight": rng.uniform(100.0, 300.0, n).tolist(),
    })


def test_train_xrepa_returns_fitted_gam():
    from pygam import LinearGAM
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    assert isinstance(gam, LinearGAM)
    assert gam._is_fitted


def test_train_xrepa_predictions_are_finite():
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    X = model_data[["epa_per_play", "success"]].to_numpy()
    preds = gam.predict(X)
    assert np.all(np.isfinite(preds)), "GAM predictions contain non-finite values"
    assert preds.shape == (len(model_data),)


def test_train_xrepa_uses_two_features():
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    # LinearGAM(s(0)+s(1)) has exactly 2 spline terms (plus an intercept term)
    spline_terms = [t for t in gam.terms if t.__class__.__name__ == "SplineTerm"]
    assert len(spline_terms) == 2


def test_loso_cv_covers_all_seasons():
    model_data = _synth_model_data(n=200)
    cv = loso_cv(model_data)
    assert set(cv["season"].unique().to_list()) == set(model_data["season"].unique().to_list())


def test_loso_cv_output_has_exp_rb_epa():
    model_data = _synth_model_data(n=200)
    cv = loso_cv(model_data)
    assert "exp_rb_epa" in cv.columns
    assert cv["exp_rb_epa"].null_count() == 0
    assert cv["exp_rb_epa"].dtype in (pl.Float32, pl.Float64)


def test_loso_cv_does_not_raise_on_small_data():
    model_data = _synth_model_data(n=100)
    cv = loso_cv(model_data)
    assert cv.height > 0


def test_save_and_load_model_roundtrip(tmp_path):
    from pygam import LinearGAM
    model_data = _synth_model_data()
    gam = train_xrepa(model_data)
    pkl_path = tmp_path / "xrepa_final.pkl"
    card_path = save_model(gam, pkl_path, season_range=(2010, 2014), n_rushers=20)
    assert pkl_path.exists()
    assert card_path.exists()
    loaded = load_model(pkl_path)
    assert isinstance(loaded, LinearGAM)
    assert loaded._is_fitted
    card = json.loads(card_path.read_text())
    assert "pygam_version" in card
    assert "season_range" in card
    assert "n_rushers" in card
