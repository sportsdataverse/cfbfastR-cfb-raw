"""Tests for pregame_wp.data_ingest (Phase 6).

Live tests are gated by CFB_DATA_API_KEY env var; they hit the real CFBD API
and are skipped in offline CI.

Notes on live test game:
    2019 CFP Semifinal Peach Bowl (game_id=401135278):
        LSU 63 – Oklahoma 28, played 2019-12-28.
        CFBD season=2019, seasonType=postseason, week=1.
"""
from __future__ import annotations

import json
import os
import pathlib

import pandas as pd
import pytest

HAS_KEY = bool(os.environ.get("CFB_DATA_API_KEY"))
skip_no_key = pytest.mark.skipif(not HAS_KEY, reason="CFB_DATA_API_KEY not set")

# Known game: 2019 CFP National Championship
_LIVE_GAME_ID = "401135278"
_LIVE_YEAR = 2019
_LIVE_WEEK = 1
_LIVE_SEASON_TYPE = "postseason"


# ---------------------------------------------------------------------------
# Import smoke test (offline — no key required)
# ---------------------------------------------------------------------------

def test_module_imports_without_error():
    from pregame_wp import data_ingest  # noqa: F401


def test_public_api_present():
    from pregame_wp import data_ingest
    for name in ("fetch_games", "fetch_plays", "fetch_drives", "load_game_frames"):
        assert hasattr(data_ingest, name), f"Missing: {name}"


# ---------------------------------------------------------------------------
# Unit tests for response → DataFrame transformations (offline, fixture-based)
# ---------------------------------------------------------------------------

# Fixtures use snake_case (as normalize_* produces after CFBD camelCase mapping)
_PLAYS_FIXTURE = [
    {
        "offense": "Alabama",
        "defense": "Georgia",
        "play_type": "Rush",
        "down": 1,
        "distance": 10,
        "yards_gained": 4,
        "yard_line": 35,
        "play_text": "Bryce Young run for 4 yards",
    },
    {
        "offense": "Georgia",
        "defense": "Alabama",
        "play_type": "Pass Reception",
        "down": 2,
        "distance": 6,
        "yards_gained": 10,
        "yard_line": 50,
        "play_text": "Stetson Bennett pass to receiver for 10 yards",
    },
]

_PLAYS_CAMEL_FIXTURE = [
    {
        "offense": "Alabama",
        "defense": "Georgia",
        "playType": "Rush",
        "down": 1,
        "distance": 10,
        "yardsGained": 4,
        "yardsToGoal": 65,  # 100 - 35
        "playText": "Bryce Young run for 4 yards",
    },
]

_DRIVES_FIXTURE = [
    {
        "offense": "Alabama",
        "defense": "Georgia",
        "drive_start_yardline": 25,
        "drive_yards": 45,
        "drive_scoring": True,
        "drive_pts": 7,
    },
    {
        "offense": "Georgia",
        "defense": "Alabama",
        "drive_start_yardline": 30,
        "drive_yards": 30,
        "drive_scoring": False,
        "drive_pts": 0,
    },
]


def test_normalize_plays_returns_dataframe():
    from pregame_wp.data_ingest import normalize_plays
    df = normalize_plays(_PLAYS_FIXTURE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    required = {"offense", "defense", "play_type", "down", "distance", "yards_gained", "yard_line"}
    assert required <= set(df.columns)


def test_normalize_plays_dtypes():
    from pregame_wp.data_ingest import normalize_plays
    df = normalize_plays(_PLAYS_FIXTURE)
    for col in ("down", "distance", "yards_gained", "yard_line"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"


def test_normalize_plays_camelcase_keys():
    """normalize_plays() must handle CFBD camelCase field names."""
    from pregame_wp.data_ingest import normalize_plays
    df = normalize_plays(_PLAYS_CAMEL_FIXTURE)
    assert "play_type" in df.columns
    assert "yards_gained" in df.columns
    assert "yard_line" in df.columns
    assert df["play_type"].iloc[0] == "Rush"
    assert df["yards_gained"].iloc[0] == 4
    assert df["yard_line"].iloc[0] == 65  # yardsToGoal stored as-is in yard_line


def test_normalize_drives_returns_dataframe():
    from pregame_wp.data_ingest import normalize_drives
    df = normalize_drives(_DRIVES_FIXTURE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    required = {"offense", "defense", "drive_start_yardline", "drive_yards", "drive_scoring", "drive_pts"}
    assert required <= set(df.columns)


def test_normalize_plays_empty_input():
    from pregame_wp.data_ingest import normalize_plays
    df = normalize_plays([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_normalize_drives_empty_input():
    from pregame_wp.data_ingest import normalize_drives
    df = normalize_drives([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_load_game_frames_from_disk(tmp_path: pathlib.Path):
    """load_game_frames() reads from pre-cached JSON files on disk."""
    from pregame_wp.data_ingest import load_game_frames

    game_id = "401415063"
    game_dir = tmp_path / str(game_id)
    game_dir.mkdir()
    (game_dir / "plays.json").write_text(json.dumps(_PLAYS_FIXTURE))
    (game_dir / "drives.json").write_text(json.dumps(_DRIVES_FIXTURE))

    plays, drives = load_game_frames(game_id, raw_dir=tmp_path)
    assert isinstance(plays, pd.DataFrame)
    assert isinstance(drives, pd.DataFrame)
    assert len(plays) == 2
    assert len(drives) == 2


def test_load_game_frames_missing_raises(tmp_path: pathlib.Path):
    from pregame_wp.data_ingest import load_game_frames
    with pytest.raises(FileNotFoundError):
        load_game_frames("999999999", raw_dir=tmp_path)


# ---------------------------------------------------------------------------
# Live integration tests (require CFB_DATA_API_KEY)
# ---------------------------------------------------------------------------

@skip_no_key
def test_fetch_games_returns_list():
    from pregame_wp.data_ingest import fetch_games
    games = fetch_games(season=2019, season_type="regular")
    assert isinstance(games, list)
    assert len(games) > 0
    # CFBD returns camelCase keys
    first = games[0]
    assert "id" in first
    # homeTeam or home_team depending on API version
    assert "homeTeam" in first or "home_team" in first


@skip_no_key
def test_fetch_plays_for_known_game():
    """Fetch all postseason week-1 plays and spot-check structure."""
    from pregame_wp.data_ingest import fetch_plays
    plays = fetch_plays(year=_LIVE_YEAR, week=_LIVE_WEEK, season_type=_LIVE_SEASON_TYPE)
    assert isinstance(plays, list)
    assert len(plays) > 50


@skip_no_key
def test_fetch_drives_for_known_game():
    from pregame_wp.data_ingest import fetch_drives
    drives = fetch_drives(
        year=_LIVE_YEAR,
        season_type=_LIVE_SEASON_TYPE,
        week=_LIVE_WEEK,
        game_id=_LIVE_GAME_ID,
    )
    assert isinstance(drives, list)
    assert len(drives) > 0  # may be all week-1 or filtered, either is valid


@skip_no_key
def test_fetch_plays_normalized_columns():
    from pregame_wp.data_ingest import (
        fetch_plays,
        filter_plays_to_game,
        normalize_plays,
    )
    raw = fetch_plays(year=_LIVE_YEAR, week=_LIVE_WEEK, season_type=_LIVE_SEASON_TYPE)
    game_plays = filter_plays_to_game(raw, "LSU", "Clemson")
    df = normalize_plays(game_plays)
    required = {"offense", "defense", "play_type", "down", "distance", "yards_gained", "yard_line"}
    assert required <= set(df.columns)


@skip_no_key
def test_fetch_and_cache_game(tmp_path: pathlib.Path):
    """fetch_and_cache() writes plays.json + drives.json, then load_game_frames reads them back."""
    from pregame_wp.data_ingest import fetch_and_cache, load_game_frames
    fetch_and_cache(
        game_id=_LIVE_GAME_ID,
        year=_LIVE_YEAR,
        week=_LIVE_WEEK,
        raw_dir=tmp_path,
        season_type=_LIVE_SEASON_TYPE,
    )

    game_dir = tmp_path / _LIVE_GAME_ID
    assert (game_dir / "plays.json").exists()
    assert (game_dir / "drives.json").exists()

    plays, drives = load_game_frames(_LIVE_GAME_ID, raw_dir=tmp_path)
    assert len(plays) > 50
    assert len(drives) > 5


@skip_no_key
def test_e2e_box_score_from_live_game(tmp_path: pathlib.Path):
    """Full pipeline: fetch → normalize → box_score for one known game."""
    from pregame_wp.box_score import calculate_box_score_from_frames
    from pregame_wp.data_ingest import fetch_and_cache, load_game_frames
    from pregame_wp.ep_curve import load_ep_curve, load_punt_sr

    fetch_and_cache(
        game_id=_LIVE_GAME_ID,
        year=_LIVE_YEAR,
        week=_LIVE_WEEK,
        raw_dir=tmp_path,
        season_type=_LIVE_SEASON_TYPE,
    )
    plays, drives = load_game_frames(_LIVE_GAME_ID, raw_dir=tmp_path)

    ep_data = load_ep_curve()
    punt_sr = load_punt_sr()
    box = calculate_box_score_from_frames(plays, drives, ep_data, punt_sr)

    assert len(box) == 2
    assert "5FR" in box.columns
    assert "5FRDiff" in box.columns
    # 5FRDiff is antisymmetric
    assert abs(box["5FRDiff"].iloc[0] + box["5FRDiff"].iloc[1]) < 1e-9
