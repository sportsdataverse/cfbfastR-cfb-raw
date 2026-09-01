"""Reprocess every final/ game whose processing_version is stale, from raw on disk.

`espn_cfb_60_reprocess.py` enumerates its work from `cfb_schedule_master.parquet`,
so a game that has raw + final but is absent from the schedule master is never
targeted -- even with `--force`, and even though the verify pass only checks the
games it targeted, so it reports "0 stale/missing" while those games keep an old
stamp indefinitely.

That is not hypothetical: after a full `-s 2004 -e 2025 --force` sweep, 712 games
across 2007-2023 still carried stamps as old as `0.0.59+1`, and 244 of a
254-game sample still showed the `rushing_power_rate == 1.0` defect the sweep was
run to clear.

This driver takes the complement: every final/ whose stamp differs from the
current PROCESSING_VERSION and whose raw exists, regardless of schedule-master
membership. It reuses `reprocess_game` so the output is identical to a normal
sweep -- only the targeting differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfb_raw_scrape._cfb_raw_utils import PROCESSING_VERSION, get_logger  # noqa: E402
from espn_cfb_60_reprocess import FINAL_DIR, RAW_DIR, _run_with_recovery  # noqa: E402


def _stale_pairs(force: bool) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    for f in sorted(FINAL_DIR.glob("*.json")):
        gid = f.stem
        if not (RAW_DIR / f"{gid}.json").exists():
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("processing_version") == PROCESSING_VERSION and not force:
            continue
        season = d.get("season")
        if isinstance(season, dict):
            season = season.get("year")
        if not isinstance(season, int):
            # No usable season on a stale final: fall back to the ESPN id's date
            # prefix (YYMMDDxxx before 2014, 4-prefixed after), which is all
            # reprocess_game needs it for (aux lookup paths).
            season = 2000 + int(gid[:2]) if not gid.startswith("4") else None
        if not isinstance(season, int):
            continue
        out.append((int(gid), int(season), True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--workers", type=int, default=None)
    ap.add_argument(
        "--force", action="store_true", help="reprocess even if the stamp is current"
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logger = get_logger("cfb_reprocess", "stale-by-stamp")
    pairs = _stale_pairs(args.force)
    logger.info(
        f"PROCESSING_VERSION={PROCESSING_VERSION}; {len(pairs)} game(s) stale by stamp"
    )
    if not pairs:
        return 0
    seasons: dict[int, int] = {}
    for _g, s, _f in pairs:
        seasons[s] = seasons.get(s, 0) + 1
    logger.info(f"by season: {dict(sorted(seasons.items()))}")
    if args.dry_run:
        return 0

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    # Reuse the sweep's own runner rather than driving a pool here: it unpacks the
    # (game_id, season, force) tuples and carries the OOM auto-recovery (halve the
    # workers and resume from the on-disk checkpoint) that a long sweep needs.
    _run_with_recovery(pairs, workers, True, logger)

    remaining = _stale_pairs(False)
    logger.info(f"after: {len(remaining)} still stale")
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
