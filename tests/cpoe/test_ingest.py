"""Phase 2 Task 2.3 — ingest module tests."""
from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL

# ---------------------------------------------------------------------------
# Minimal fixture: two pass plays in a cfbfastR-processed plays DataFrame
# (parquet on disk, per-game subdirectory layout used by cfbfastR-cfb-raw).
# ---------------------------------------------------------------------------

_PLAYS_FIXTURE = [
    {
        "game_id": "401628455",
        "playType": "Pass Reception",
        "start.down": 1,
        "start.distance": 10,
        "start.yardsToEndzone": 65,
        "pos_score_diff_start": 7,
        "start.TimeSecsRem": 1800,
        "start.is_home": True,
        "period": 2,
        "passing_down": 0,
        "completion": 1,
    },
    {
        "game_id": "401628455",
        "playType": "Pass Incompletion",
        "start.down": 3,
        "start.distance": 8,
        "start.yardsToEndzone": 30,
        "pos_score_diff_start": -3,
        "start.TimeSecsRem": 400,
        "start.is_home": False,
        "period": 4,
        "passing_down": 1,
        "completion": 0,
    },
    {
        "game_id": "401628455",
        "playType": "Rush",
        "start.down": 1,
        "start.distance": 10,
        "start.yardsToEndzone": 50,
        "pos_score_diff_start": 0,
        "start.TimeSecsRem": 3600,
        "start.is_home": True,
        "period": 1,
        "passing_down": 0,
        "completion": 0,
    },
]


@pytest.fixture()
def season_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A mock cfbfastR-cfb-raw season/game directory with a plays parquet."""
    plays_df = pd.DataFrame(_PLAYS_FIXTURE)
    season_path = tmp_path / "2024" / "regular"
    game_dir = season_path / "401628455"
    game_dir.mkdir(parents=True)
    plays_df.to_parquet(game_dir / "plays.parquet")
    return season_path


def test_ingest_imports():
    from cpoe.ingest import load_season_pass_plays  # noqa: F401


def test_load_season_returns_dataframe(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert isinstance(df, pd.DataFrame)


def test_load_season_only_pass_plays(season_dir):
    """Rush play must be excluded; only 2 pass plays."""
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert len(df) == 2


def test_load_season_has_feature_cols(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing: {col}"


def test_load_season_has_target_col(season_dir):
    from cpoe.ingest import load_season_pass_plays
    df = load_season_pass_plays(season_dir)
    assert TARGET_COL in df.columns


def test_load_season_empty_dir_returns_empty(tmp_path):
    from cpoe.ingest import load_season_pass_plays
    (tmp_path / "empty_season").mkdir()
    df = load_season_pass_plays(tmp_path / "empty_season")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_season_multiple_games(tmp_path: pathlib.Path):
    """Two game dirs → combined DataFrame."""
    from cpoe.ingest import load_season_pass_plays
    plays_df = pd.DataFrame(_PLAYS_FIXTURE)
    for game_id in ("401628455", "401628456"):
        gd = tmp_path / game_id
        gd.mkdir()
        plays_df["game_id"] = game_id
        plays_df.to_parquet(gd / "plays.parquet")
    df = load_season_pass_plays(tmp_path)
    assert len(df) == 4  # 2 pass plays × 2 games
