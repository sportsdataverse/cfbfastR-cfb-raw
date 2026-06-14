"""CFBD data ingest for pregame_wp pipeline (Phase 6).

Fetches games, play-by-play, and drives from the College Football Data API
(https://api.collegefootballdata.com) and normalizes them into the DataFrame
shape expected by calculate_box_score_from_frames().

Environment:
    CFB_DATA_API_KEY  — CFBD API bearer token (required for live fetch).
                        Can be set in a .env file at the project root;
                        uv run loads it automatically since uv 0.4.

API notes:
    * The CFBD API returns camelCase field names (e.g. ``yardsGained``,
      ``playType``).  normalize_plays() / normalize_drives() convert them
      to the snake_case shape the pipeline expects.
    * GET /plays does NOT support gameId filtering — ``year`` + ``week`` are
      required; the returned plays are then filtered client-side to the two
      teams of the target game.
    * GET /drives supports ``gameId`` but the client-side filter is applied
      as a safety net regardless.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

_BASE_URL = "https://api.collegefootballdata.com"
_TIMEOUT = 30

# CFBD camelCase → pipeline snake_case for plays
_PLAY_KEY_MAP: dict[str, str] = {
    "playType": "play_type",
    "playText": "play_text",
    "yardsGained": "yards_gained",
    "yardsToGoal": "yard_line",  # CFBD yardsToGoal = yards from opponent end zone
}

# CFBD camelCase → pipeline snake_case for drives
_DRIVE_KEY_MAP: dict[str, str] = {
    "startYardline": "drive_start_yardline",
    "startYardLine": "drive_start_yardline",
    "yards": "drive_yards",
    "scoring": "drive_scoring",
    "isScore": "drive_scoring",
    "startPeriod": "drive_start_period",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.environ.get("CFB_DATA_API_KEY") or os.environ.get("CFBD_DATA_API_KEY")
    if not key:
        raise EnvironmentError(
            "CFB_DATA_API_KEY not set. "
            "Add it to your .env file or export it before running."
        )
    return key


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
    }
    url = f"{_BASE_URL}{path}"
    resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _team_key(game: dict[str, Any], side: str) -> str:
    """Extract home or away team name from a CFBD game record (handles camelCase)."""
    camel = f"{side}Team"
    snake = f"{side}_team"
    return str(game.get(camel) or game.get(snake) or "")


# ---------------------------------------------------------------------------
# Fetch raw data from CFBD API
# ---------------------------------------------------------------------------

def fetch_games(
    season: int,
    season_type: str = "regular",
    team: str | None = None,
    week: int | None = None,
) -> list[dict[str, Any]]:
    """Return game metadata list for a season.

    The CFBD API returns camelCase keys (``homeTeam``, ``awayTeam``, etc.).

    Args:
        season: Year (e.g. 2019).
        season_type: ``"regular"`` | ``"postseason"`` | ``"both"``.
        team: Optional team name filter.
        week: Optional week number filter.

    Returns:
        List of game dicts as returned by the API.
    """
    params: dict[str, Any] = {"year": season, "seasonType": season_type}
    if team:
        params["team"] = team
    if week is not None:
        params["week"] = week
    return _get("/games", params)


def fetch_plays(
    year: int,
    week: int,
    season_type: str = "regular",
    offense: str | None = None,
) -> list[dict[str, Any]]:
    """Return raw play list for a week from CFBD.

    The ``/plays`` endpoint requires both ``year`` and ``week``; there is no
    ``gameId`` filter — filter to a specific game client-side via
    ``filter_plays_to_game()``.

    Args:
        year: Season year (e.g. 2019).
        week: Week number within the season.
        season_type: ``"regular"`` | ``"postseason"``.
        offense: Optional team name to pre-filter by offense (reduces result set).

    Returns:
        List of play dicts as returned by ``/plays``.
    """
    params: dict[str, Any] = {
        "year": int(year),
        "week": int(week),
        "seasonType": season_type,
    }
    if offense:
        params["offense"] = offense
    return _get("/plays", params)


def fetch_drives(
    year: int,
    season_type: str = "regular",
    week: int | None = None,
    game_id: str | int | None = None,
) -> list[dict[str, Any]]:
    """Return raw drive list from CFBD.

    Args:
        year: Season year.
        season_type: ``"regular"`` | ``"postseason"``.
        week: Optional week filter.
        game_id: Optional CFBD game identifier filter.

    Returns:
        List of drive dicts as returned by ``/drives``.
    """
    params: dict[str, Any] = {
        "year": int(year),
        "seasonType": season_type,
    }
    if week is not None:
        params["week"] = int(week)
    if game_id is not None:
        params["gameId"] = int(game_id)
    return _get("/drives", params)


def filter_plays_to_game(
    plays: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> list[dict[str, Any]]:
    """Filter a week's play list to just the plays from one specific matchup.

    Both ``offense`` and ``defense`` must be one of the two teams.

    Args:
        plays: Full play list for a week (from ``fetch_plays()``).
        home_team: One of the two teams in the matchup.
        away_team: The other team.

    Returns:
        Subset of plays involving only those two teams.
    """
    teams = {home_team, away_team}
    return [
        p for p in plays
        if p.get("offense") in teams and p.get("defense") in teams
    ]


def filter_drives_to_game(
    drives: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> list[dict[str, Any]]:
    """Filter a drive list to just the drives from one specific matchup."""
    teams = {home_team, away_team}
    return [
        d for d in drives
        if d.get("offense") in teams and d.get("defense") in teams
    ]


# ---------------------------------------------------------------------------
# Normalize raw lists → DataFrames
# ---------------------------------------------------------------------------

def normalize_plays(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize CFBD play list to the shape expected by box_score pipeline.

    Handles both camelCase keys (from live CFBD API) and snake_case keys
    (from test fixtures / older ingest versions).

    Args:
        raw: List of play dicts.

    Returns:
        DataFrame with columns: offense, defense, play_type, down,
        distance, yards_gained, yard_line, play_text.
    """
    _required_cols = [
        "offense", "defense", "play_type", "down",
        "distance", "yards_gained", "yard_line", "play_text",
    ]
    if not raw:
        return pd.DataFrame(columns=_required_cols)

    df = pd.json_normalize(raw)

    rename = {k: v for k, v in _PLAY_KEY_MAP.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)

    for col in ("down", "distance", "yards_gained", "yard_line"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "play_text" not in df.columns:
        df["play_text"] = ""
    if "play_type" not in df.columns:
        df["play_type"] = ""

    available = [c for c in _required_cols if c in df.columns]
    return df[available].reset_index(drop=True)


def normalize_drives(raw: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize CFBD drive list to the shape expected by box_score pipeline.

    Args:
        raw: List of drive dicts.

    Returns:
        DataFrame with columns: offense, defense, drive_start_yardline,
        drive_yards, drive_scoring, drive_pts.
    """
    _required_cols = [
        "offense", "defense", "drive_start_yardline",
        "drive_yards", "drive_scoring", "drive_pts",
    ]
    if not raw:
        return pd.DataFrame(columns=_required_cols)

    df = pd.json_normalize(raw)

    rename = {k: v for k, v in _DRIVE_KEY_MAP.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(columns=rename)

    if "drive_pts" not in df.columns:
        for src in ("points", "drivePoints", "drive_points"):
            if src in df.columns:
                df = df.rename(columns={src: "drive_pts"})
                break
    if "drive_pts" not in df.columns:
        df["drive_pts"] = 0

    for col in ("drive_start_yardline", "drive_yards", "drive_pts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "drive_scoring" in df.columns:
        df["drive_scoring"] = df["drive_scoring"].astype(bool)

    available = [c for c in _required_cols if c in df.columns]
    return df[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def fetch_and_cache(
    game_id: str | int,
    year: int,
    week: int,
    raw_dir: Path | str,
    season_type: str = "regular",
) -> None:
    """Fetch plays + drives for one game and write them to disk.

    Workflow:
    1. Look up the game record to get home/away team names.
    2. Fetch all plays for the week filtered by home team offense.
    3. Filter plays client-side to just the two teams.
    4. Fetch drives and filter similarly.
    5. Write ``{raw_dir}/{game_id}/plays.json`` and ``drives.json``.

    Args:
        game_id: CFBD game identifier.
        year: Season year.
        week: Week number within the season.
        raw_dir: Root directory for per-game JSON cache.
        season_type: ``"regular"`` | ``"postseason"``.
    """
    raw_dir = Path(raw_dir)
    game_dir = raw_dir / str(game_id)
    game_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: get game metadata to identify the two teams
    games = fetch_games(season=year, season_type=season_type, week=week)
    target = next((g for g in games if str(g.get("id")) == str(game_id)), None)
    if target is None:
        raise ValueError(f"Game {game_id} not found in {year} {season_type} week {week}")
    home_team = _team_key(target, "home")
    away_team = _team_key(target, "away")

    # Step 2–3: fetch plays from both teams (CFBD filters by offense only)
    home_raw = fetch_plays(year=year, week=week, season_type=season_type, offense=home_team)
    away_raw = fetch_plays(year=year, week=week, season_type=season_type, offense=away_team)
    game_plays = (
        filter_plays_to_game(home_raw, home_team, away_team)
        + filter_plays_to_game(away_raw, home_team, away_team)
    )
    (game_dir / "plays.json").write_text(json.dumps(game_plays), encoding="utf-8")

    # Step 4: fetch drives and filter
    all_drives = fetch_drives(year=year, season_type=season_type, week=week, game_id=game_id)
    game_drives = filter_drives_to_game(all_drives, home_team, away_team)
    (game_dir / "drives.json").write_text(json.dumps(game_drives), encoding="utf-8")


def load_game_frames(
    game_id: str | int,
    raw_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-cached plays + drives JSON from disk.

    Args:
        game_id: CFBD game identifier.
        raw_dir: Root directory containing per-game subdirectories.

    Returns:
        Tuple of ``(plays_df, drives_df)`` ready for
        ``calculate_box_score_from_frames()``.

    Raises:
        FileNotFoundError: If plays.json or drives.json are missing.
    """
    game_dir = Path(raw_dir) / str(game_id)
    plays_path = game_dir / "plays.json"
    drives_path = game_dir / "drives.json"

    if not plays_path.exists():
        raise FileNotFoundError(f"plays.json not found: {plays_path}")
    if not drives_path.exists():
        raise FileNotFoundError(f"drives.json not found: {drives_path}")

    plays = json.loads(plays_path.read_text(encoding="utf-8"))
    drives = json.loads(drives_path.read_text(encoding="utf-8"))

    return normalize_plays(plays), normalize_drives(drives)
