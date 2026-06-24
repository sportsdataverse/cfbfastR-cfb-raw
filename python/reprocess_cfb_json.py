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
        betting_embed = {k: v for k, v in betting.items() if k not in ("game_id", "season", "week")}

        proc = CFBPlayProcess(
            gameId=game_id,
            path_to_json=str(RAW_DIR),
            odds_override=override,
            game_roster=_aux_list("game_rosters", season, game_id),
            participants=_aux_list("play_participants", season, game_id),
        )
        proc.join_participants = False  # offline: use the supplied participants/roster, never the network
        proc.cfb_pbp_disk()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)
        result.update(
            id=game_id,
            season=season,
            week=result.get("week"),
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
            homeTeamId=home_id,
            awayTeamId=away_id,
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


# Worker-internal thread caps. Each worker runs a polars + multi-model XGBoost
# pipeline; left unbounded, every worker spawns ~cpu threads, so N workers x C cores
# oversubscribes a many-core box into thrash (the cause of a past stalled sweep). Pin
# each worker to one internal thread so total threads ~= worker count. setdefault lets
# an explicit env override win.
_THREAD_CAP_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "XGBOOST_NTHREAD",
)


def _auto_workers(requested: int | None, n_items: int, logger) -> int:
    """RAM-aware worker count. An explicit `requested` (CLI/env) wins; otherwise size to
    free memory so the model-bundling 0.0.69 pipeline can't OOM-kill a worker (the
    cpu-2 default does, on a long all-years sweep)."""
    cpu_ceil = max(1, (os.cpu_count() or 2) - 2)
    if requested:
        return max(1, min(requested, n_items))
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / 1e9
        per_gb = float(os.getenv("CFB_REPROCESS_WORKER_GB", "2.5"))
        ram_cap = max(1, int(free_gb * 0.7 / per_gb))
        n = max(1, min(cpu_ceil, ram_cap, n_items))
        logger.info(
            "auto workers=%d (cpu_ceil=%d, ram_cap=%d @ %.1fGB/worker, free=%.1fGB)",
            n,
            cpu_ceil,
            ram_cap,
            per_gb,
            free_gb,
        )
        return n
    except Exception:  # noqa: BLE001 - psutil optional; degrade to cpu-2
        n = max(1, min(cpu_ceil, n_items))
        logger.info("psutil unavailable; workers=%d (cpu-2)", n)
        return n


def _run_with_recovery(pairs, workers: int, force: bool, logger) -> None:
    """Run the pool, auto-recovering from an OOM-killed worker: halve the worker count
    and relaunch. Completed games are already on disk, so each relaunch resumes from the
    checkpoint (`_final_is_current` skips them). If a worker is still killed at 1, a
    single game exceeds the budget and we re-raise rather than spin forever."""
    attempt = 0
    while True:
        remaining = [p for p in pairs if force or not _final_is_current(p[0])]
        if not remaining:
            logger.info("nothing left to reprocess (all current)")
            return
        logger.info("sweep attempt %d: %d game(s) at %d worker(s)", attempt + 1, len(remaining), workers)
        try:
            run_pool(_worker, remaining, kind="process", desc=f"reprocess w={workers}", workers=workers)
            return
        except BrokenProcessPool:
            if workers <= 1:
                logger.error("worker killed at workers=1 -- a single game exceeds the memory budget; aborting")
                raise
            workers = max(1, workers // 2)
            attempt += 1
            logger.warning("worker died (likely OOM) -> halving to %d worker(s) and resuming", workers)


def _verify(pairs, logger) -> tuple[int, int, int]:
    """Confirm every targeted game now has a final at the current PROCESSING_VERSION.
    Returns (current, stale_or_missing, current_but_empty). Empty-but-current games are
    reported (could be legitimately play-less, e.g. cancelled), not treated as failures."""
    current = stale = empty = 0
    incomplete: list[int] = []
    for game_id, _season, _force in pairs:
        d = _read(FINAL_DIR / f"{game_id}.json", {})
        if d.get("processing_version") == PROCESSING_VERSION:
            current += 1
            if (d.get("count") or 0) == 0:
                empty += 1
        else:
            stale += 1
            incomplete.append(game_id)
    logger.info(
        "verify: %d/%d current (%d empty-but-current), %d stale/missing",
        current,
        len(pairs),
        empty,
        stale,
    )
    if incomplete:
        logger.warning("not-current after sweep (first 20): %s", incomplete[:20])
    return current, stale, empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=None)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="explicit worker count (overrides CFB_REPROCESS_WORKERS and RAM auto-sizing)",
    )
    args = ap.parse_args()

    logger = get_logger("cfb_reprocess", "sweep")

    # Pin worker-internal threads before the pool spawns (children inherit env).
    for var in _THREAD_CAP_VARS:
        os.environ.setdefault(var, "1")

    import pandas as pd

    master = pd.read_parquet("cfb/cfb_schedule_master.parquet")
    if not args.all:
        start = args.start_year
        end = args.end_year or start
        master = master[(master["season"] >= start) & (master["season"] <= end)]
    pairs = list(master[["game_id", "season"]].itertuples(index=False, name=None))
    pairs = [(int(g), int(s), args.force) for g, s in pairs if (RAW_DIR / f"{g}.json").exists()]
    if not pairs:
        logger.info("no games with raw on disk in range; nothing to do")
        return

    # CFB_REPROCESS_WORKERS / --workers bound pool concurrency; unset -> RAM-aware
    # sizing (the 0.0.69 pipeline bundles several models per worker and the cpu-2
    # default can OOM-kill a worker on a long all-years sweep).
    requested = args.workers or (int(os.getenv("CFB_REPROCESS_WORKERS", "0")) or None)
    workers = _auto_workers(requested, len(pairs), logger)
    _run_with_recovery(pairs, workers, args.force, logger)
    _verify(pairs, logger)


if __name__ == "__main__":
    main()
