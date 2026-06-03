"""Rebuild final/{id}.json from on-disk raw + standalone aux, fully offline."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
from pathlib import Path

from sportsdataverse.cfb import CFBPlayProcess

from _cfb_raw_utils import PROCESSING_VERSION, get_logger, run_pool, write_json_atomic
from cfb_betting import odds_override_from_betting

RAW_DIR = Path("cfb/json/raw")
FINAL_DIR = Path("cfb/json/final")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _aux(ds: str, season: int, game_id: int):
    return _read(Path(f"cfb/{ds}/json/{season}/{game_id}.json"), {})


def _aux_list(ds: str, season: int, game_id: int):
    obj = _aux(ds, season, game_id)
    return obj.get("data", obj) if isinstance(obj, dict) else obj


def _final_is_current(game_id: int) -> bool:
    f = FINAL_DIR / f"{game_id}.json"
    if not f.exists():
        return False
    return _read(f, {}).get("processing_version") == PROCESSING_VERSION


def _home_away_ids(raw: dict):
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors") or []
    home = away = None
    for c in comps:
        if c.get("homeAway") == "home":
            home = c.get("team", {}).get("id")
        elif c.get("homeAway") == "away":
            away = c.get("team", {}).get("id")
    return home, away


def reprocess_game(game_id: int, season: int, refresh_aux: bool, force: bool, logger=None):
    logger = logger or get_logger("cfb_reprocess", season)
    try:
        if not force and _final_is_current(game_id):
            return "skipped"
        raw = _read(RAW_DIR / f"{game_id}.json", None)
        if raw is None:
            logger.warning("no raw on disk for %s", game_id)
            return "missing_raw"

        betting = _aux("betting", season, game_id)
        override = odds_override_from_betting(betting) if betting else None

        proc = CFBPlayProcess(gameId=game_id, path_to_json=str(RAW_DIR),
                              odds_override=override)
        proc.cfb_pbp_disk()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)
        result.update(
            id=game_id, season=season, week=result.get("week"),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            betting=betting,
            game_rosters=_aux_list("rosters", season, game_id),
            play_participants=_aux_list("play_participants", season, game_id),
            power_index=_aux("power_index", season, game_id),
            team_box_extra=_aux("team_box_extra", season, game_id),
            injuries=raw.get("injuries") or [],
            game_notes=raw.get("gameNotes") or [],
            homeTeamId=home_id, awayTeamId=away_id,
        )
        write_json_atomic(result, str(FINAL_DIR / f"{game_id}.json"))
        return "rebuilt"
    except Exception:
        logger.exception("reprocess failed: %s", game_id)
        return "error"


def _worker(args):
    """Module-level (picklable) wrapper for ProcessPoolExecutor.
    args = (game_id, season, refresh_aux, force)."""
    game_id, season, refresh_aux, force = args
    return reprocess_game(game_id, season, refresh_aux, force)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=None)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refresh-aux", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    master = pd.read_parquet("cfb/cfb_schedule_master.parquet")
    if not args.all:
        start = args.start_year
        end = args.end_year or start
        master = master[(master["season"] >= start) & (master["season"] <= end)]
    pairs = list(master[["game_id", "season"]].itertuples(index=False, name=None))
    pairs = [(int(g), int(s), args.refresh_aux, args.force)
             for g, s in pairs if (RAW_DIR / f"{g}.json").exists()]
    run_pool(_worker, pairs, kind="process", desc="reprocess")


if __name__ == "__main__":
    main()
