"""Phase 6 smoke test: full CLI pipeline on synthetic parquet data."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL


def _make_plays(n: int, game_id: str, rng: np.random.Generator) -> pd.DataFrame:
    pass_types = ["Pass Reception", "Pass Incompletion", "Passing Touchdown"]
    df = pd.DataFrame({
        "game_id": game_id,
        "playType": rng.choice(pass_types, n),
        "start.down": rng.integers(1, 5, n),
        "start.distance": rng.integers(1, 20, n),
        "start.yardsToEndzone": rng.integers(1, 99, n),
        "pos_score_diff_start": rng.integers(-21, 22, n),
        "start.TimeSecsRem": rng.integers(0, 3600, n),
        "start.is_home": rng.integers(0, 2, n),
        "period": rng.integers(1, 5, n),
        "passing_down": rng.integers(0, 2, n),
        "completion": rng.integers(0, 2, n),
    })
    return df


@pytest.fixture()
def synthetic_raw_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    rng = np.random.default_rng(42)
    for season in (2021, 2022, 2023):
        for game_id in (f"{season}0001", f"{season}0002"):
            gd = tmp_path / str(season) / game_id
            gd.mkdir(parents=True)
            plays = _make_plays(30, game_id, rng)
            plays.to_parquet(gd / "plays.parquet")
    return tmp_path


def test_cli_smoke_no_loso(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--raw-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out"),
        "--seasons", "2021", "2022", "2023",
    ])
    assert rc == 0
    assert (tmp_path / "out" / "cfb_cp_model.ubj").exists()
    assert not (tmp_path / "out" / "loso_cv.json").exists()


def test_cli_smoke_with_loso(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--raw-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out_loso"),
        "--seasons", "2021", "2022", "2023",
        "--loso",
    ])
    assert rc == 0
    cv_path = tmp_path / "out_loso" / "loso_cv.json"
    assert cv_path.exists()
    cv = json.loads(cv_path.read_text())
    assert "folds" in cv
    assert len(cv["folds"]) == 3
    assert (tmp_path / "out_loso" / "cfb_cp_model.ubj").exists()


def test_cli_no_seasons_returns_nonzero(synthetic_raw_dir, tmp_path):
    from cpoe.cli import main
    rc = main([
        "--raw-dir", str(synthetic_raw_dir),
        "--out-dir", str(tmp_path / "out_fail"),
    ])
    assert rc != 0


def test_cli_missing_raw_dir_returns_nonzero(tmp_path):
    from cpoe.cli import main
    rc = main([
        "--raw-dir", str(tmp_path / "nonexistent"),
        "--out-dir", str(tmp_path / "out_fail"),
        "--seasons", "2021",
    ])
    assert rc != 0
