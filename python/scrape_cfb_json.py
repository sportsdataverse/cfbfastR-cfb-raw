"""Core CFB scraper: per-game raw + enriched final + standalone aux/extras."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import sportsdataverse as sdv
from sportsdataverse.cfb import CFBPlayProcess

from _cfb_raw_utils import (PROCESSING_VERSION, _safe, filter_undone,
                            games_for_seasons, get_logger, load_schedule_master,
                            most_recent_cfb_season, run_pool, stamp, write_json_atomic)
from cfb_betting import capture_betting
from cfb_team_box_extra import team_box_extra_from_summary


# --- thin sdv-py adapters (monkeypatch points in tests) ---
def _participants(gid):
    return sdv.cfb.espn_cfb_play_participants(game_id=gid, return_as_pandas=True).to_dict("records")


def _rosters(gid):
    return sdv.cfb.espn_cfb_game_rosters(game_id=gid, return_as_pandas=True).to_dict("records")


def _officials(gid):
    return sdv.cfb.espn_cfb_event_officials(event_id=gid)


def _power_index(gid):
    return sdv.cfb.espn_cfb_event_powerindex(event_id=gid)


def _odds_full(gid):
    return sdv.cfb.espn_cfb_event_odds(event_id=gid)


def _propbets(gid):
    return sdv.cfb.espn_cfb_event_propbets(event_id=gid)


def _home_away_ids(raw: dict):
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors") or []
    home = away = None
    for c in comps:
        tid = c.get("team", {}).get("id")
        if c.get("homeAway") == "home":
            home = tid
        elif c.get("homeAway") == "away":
            away = tid
    return home, away


def download_game(game_id: int, season: int, rescrape: bool, logger=None):
    logger = logger or get_logger("cfb_json", season)
    try:
        # 1. bank RAW first
        raw = CFBPlayProcess(gameId=game_id, raw=True).espn_cfb_pbp()
        write_json_atomic(raw, f"cfb/json/raw/{game_id}.json")

        # 2. enrich
        proc = CFBPlayProcess(gameId=game_id)
        proc.espn_cfb_pbp()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)

        # 3. aux (endpoint-backed) — each _safe-wrapped so one failure doesn't kill the game
        participants = _safe(_participants, game_id, logger=logger, default=[])
        rosters = _safe(_rosters, game_id, logger=logger, default=[])
        betting = capture_betting(
            raw, proc,
            odds_full=_safe(_odds_full, game_id, logger=logger, default=[]),
            propbets=_safe(_propbets, game_id, logger=logger, default=[]),
        )
        officials = _safe(_officials, game_id, logger=logger, default=[])
        power_index = _safe(_power_index, game_id, logger=logger, default={})

        # 4. team_box_extra: prefer summary (de-dup gate); {} if summary lacks it
        team_extra = team_box_extra_from_summary(raw, [home_id, away_id]) or {}

        injuries = raw.get("injuries") or []
        game_notes = raw.get("gameNotes") or []
        week = result.get("week")

        # 5. standalone datasets (each is an offline-reprocess source)
        standalone = {
            "rosters": rosters, "play_participants": participants, "betting": betting,
            "officials": officials, "power_index": power_index, "team_box_extra": team_extra,
        }
        for name, obj in standalone.items():
            write_json_atomic(stamp(obj, game_id=game_id, season=season, week=week),
                              f"cfb/{name}/json/{season}/{game_id}.json")

        # 6. embed + write FINAL last
        result.update(
            id=game_id, season=season, week=week,
            season_type=result.get("season_type") or raw.get("header", {}).get("season", {}).get("type"),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            play_participants=participants, game_rosters=rosters, betting=betting,
            officials=officials, power_index=power_index, team_box_extra=team_extra,
            injuries=injuries, game_notes=game_notes,
            homeTeamId=home_id, awayTeamId=away_id,
        )
        write_json_atomic(result, f"cfb/json/final/{game_id}.json")
        return "ok"
    except Exception:
        logger.exception("download_game failed: %s", game_id)
        return "error"


def _worker(args):
    """Module-level (picklable) wrapper for ProcessPoolExecutor. args = (game_id, season, rescrape)."""
    game_id, season, rescrape = args
    return download_game(game_id, season, rescrape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    master = load_schedule_master()
    for season in range(args.start_year, end + 1):
        logger = get_logger("cfb_json", season)
        games = filter_undone(games_for_seasons(master, season, season), rescrape=rescrape)
        logger.info("season %s: %d games to scrape (rescrape=%s)", season, len(games), rescrape)
        run_pool(_worker, [(g, season, rescrape) for g in games],
                 kind="process", desc=f"cfb {season}")


if __name__ == "__main__":
    main()
