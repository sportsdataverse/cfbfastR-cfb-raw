"""Rebuild final/{id}.json from on-disk raw + standalone aux, fully offline."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import functools
import json
from pathlib import Path

from sportsdataverse.cfb import CFBPlayProcess

from _cfb_raw_utils import PROCESSING_VERSION, get_logger, run_pool, season_type_from_raw, write_json_atomic
from cfb_betting import odds_override_from_betting

RAW_DIR = Path("cfb/json/raw")
FINAL_DIR = Path("cfb/json/final")
CONSENSUS_PATH = Path("cfb/odds_consensus.parquet")


@functools.lru_cache(maxsize=1)
def _consensus_map() -> dict:
    """game_id -> odds_override from the cfb_line_odds multi-book consensus.

    Preferred over the per-game ESPN betting aux: it's a median spread/total across
    sportsbooks keyed by the real ESPN game_id (2006-2025). Built by
    cfbfastR-cfb-data/python/betting/build_odds_consensus.py. Loaded once per worker
    process; an absent aux yields an empty map (callers fall back to ESPN betting)."""
    if not CONSENSUS_PATH.exists():
        return {}
    import polars as pl

    df = pl.read_parquet(CONSENSUS_PATH)
    return {
        int(r["game_id"]): {
            "gameSpread": float(r["gameSpread"]),
            "overUnder": float(r["overUnder"]),
            "homeFavorite": bool(r["homeFavorite"]),
            "gameSpreadAvailable": bool(r["gameSpreadAvailable"]),
        }
        for r in df.iter_rows(named=True)
    }


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _aux(ds: str, season: int, game_id: int):
    return _read(Path(f"cfb/{ds}/json/{game_id}.json"), {})


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


def reprocess_game(game_id: int, season: int, force: bool, logger=None):
    logger = logger or get_logger("cfb_reprocess", season)
    try:
        if not force and _final_is_current(game_id):
            return "skipped"
        raw = _read(RAW_DIR / f"{game_id}.json", None)
        if raw is None:
            logger.warning("no raw on disk for %s", game_id)
            return "missing_raw"

        betting = _aux("betting", season, game_id)
        # prefer the cfb_line_odds multi-book consensus; fall back to the ESPN betting aux
        consensus = _consensus_map().get(int(game_id))
        if consensus is not None:
            override, odds_source = consensus, "cfb_line_odds"
        else:
            override = odds_override_from_betting(betting)
            odds_source = "espn_betting" if override is not None else None
        betting_embed = {k: v for k, v in betting.items()
                         if k not in ("game_id", "season", "week")}

        proc = CFBPlayProcess(gameId=game_id, path_to_json=str(RAW_DIR),
                              odds_override=override,
                              game_roster=_aux_list("game_rosters", season, game_id),
                              participants=_aux_list("play_participants", season, game_id))
        proc.join_participants = False  # offline: use the supplied participants/roster, never the network
        proc.cfb_pbp_disk()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)
        result.update(
            id=game_id, season=season, week=result.get("week"),
            season_type=season_type_from_raw(raw),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            betting=betting_embed,
            game_rosters=_aux_list("game_rosters", season, game_id),
            play_participants=_aux_list("play_participants", season, game_id),
            power_index=_aux("power_index", season, game_id),
            team_box_extra=_aux("team_box_extra", season, game_id),
            injuries=raw.get("injuries") or [],
            game_notes=raw.get("gameNotes") or [],
            homeTeamId=home_id, awayTeamId=away_id,
            odds_source=odds_source or result.get("odds_source"),
        )
        write_json_atomic(result, str(FINAL_DIR / f"{game_id}.json"))
        return "rebuilt"
    except Exception:
        logger.exception("reprocess failed: %s", game_id)
        return "error"


def _worker(args):
    """Module-level (picklable) wrapper for ProcessPoolExecutor.
    args = (game_id, season, force)."""
    game_id, season, force = args
    return reprocess_game(game_id, season, force)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=None)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    master = pd.read_parquet("cfb/cfb_schedule_master.parquet")
    if not args.all:
        start = args.start_year
        end = args.end_year or start
        master = master[(master["season"] >= start) & (master["season"] <= end)]
    pairs = list(master[["game_id", "season"]].itertuples(index=False, name=None))
    pairs = [(int(g), int(s), args.force)
             for g, s in pairs if (RAW_DIR / f"{g}.json").exists()]
    # CFB_REPROCESS_WORKERS bounds pool concurrency (the 0.0.69 pipeline bundles
    # several models per worker; the default cpu-2 can OOM-kill a worker on a long
    # all-years sweep). 0/unset -> run_pool's default (cpu-2).
    workers = int(os.getenv("CFB_REPROCESS_WORKERS", "0")) or None
    run_pool(_worker, pairs, kind="process", desc="reprocess", workers=workers)


if __name__ == "__main__":
    main()
