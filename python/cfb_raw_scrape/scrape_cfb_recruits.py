"""Scrape 247 recruit classes into the raw store.

Unlike every other scraper here this one is CLASS-YEAR keyed, not game keyed, so
it does not use the schedule master or the game-oriented helpers in
``_cfb_raw_utils`` -- it pages a class year until the feed runs out.

Why this exists as a producer job at all: a recruiting class is IMMUTABLE once
its signing period closes, but the consumer (``cfb_roster_talent``) accumulates
a 4-season window, so computing talent live re-fetched the same frozen classes
once per target season -- roughly 96 pages / 20 minutes per call, and ~3.3
hours to publish 2016-2025. Scraped once into the raw store, a class year is
never fetched again.

Raw payloads are persisted UNPARSED (``return_parsed=False``). The parser has
already been wrong once about this feed (see the unexpanded-institution note in
sdv-py's ``cfb_roster_talent``), and a raw store means the next parser fix is
re-applied offline instead of costing another full re-scrape.

Layout::

    cfb/recruits/json/{year}/page_0001.json   raw RDB payload, one per page
    cfb/recruits/json/{year}/_manifest.json   written LAST; its presence means done

Pacing/retry knobs are shared with sdv-py so both sides tune together:
``SDV_PY_247_RETRIES`` (3), ``SDV_PY_247_DELAY`` (0.5s), ``SDV_PY_247_BACKOFF`` (2.0s).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import sportsdataverse as sdv
from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    is_recruiting_class_final,
    most_recent_cfb_season,
    write_json_atomic,
)

DATASET = "recruits"
SPORT_KEY = 1  # football

#: 500 exceeds what the RDB serves inside the client's 3s timeout (measured:
#: 50/100/250 return in full, 500 raises curl(28)). 250 is the largest
#: measured-good size and halves the request count versus 100.
PAGE_SIZE = 250

#: A class year stops changing once its signing period closes. Only the current
#: cycle (and, until February, the one just signed) is worth re-scraping.
FIRST_CLASS_YEAR = 2000


def _env_num(name: str, default: float, cast) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = cast(raw)
    except (TypeError, ValueError):
        return default
    return val if val >= 0 else default


def _fetch_page(year: int, page: int, logger) -> dict | None:
    """One raw page, with bounded retry. 247 resets connections under paging."""
    attempts = max(1, int(_env_num("SDV_PY_247_RETRIES", 3, int)))
    backoff = _env_num("SDV_PY_247_BACKOFF", 2.0, float)
    for attempt in range(1, attempts + 1):
        try:
            return sdv.cfb.sports247_recruits(
                sport_key=SPORT_KEY,
                year=year,
                page_size=PAGE_SIZE,
                page=page,
                return_parsed=False,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors vary by backend
            if attempt == attempts:
                logger.error(
                    "%s %s page %s failed after %s attempts: %s",
                    DATASET,
                    year,
                    page,
                    attempts,
                    exc,
                )
                return None
            time.sleep(backoff * attempt)
    return None


def _rows(payload) -> int:
    """Row count of a raw RDB page.

    The envelope is ``{"pagination": {...}, "players": [...]}``; prefer the
    named key and fall back to the first list-valued one so a renamed envelope
    degrades instead of silently counting zero.
    """
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("players"), list):
            return len(payload["players"])
        for v in payload.values():
            if isinstance(v, list):
                return len(v)
    return 0


def _expected(payload) -> int | None:
    """Total recruits the feed says exist for this class year, if it says.

    The RDB ships ``pagination: {currentPage, itemsPerPage, count, pageCount}``
    -- e.g. 2023 reports count=6213. That is an INDEPENDENT statement of how
    much data there is, so completeness becomes a check rather than an
    inference from "the last page was short". A short page can also mean the
    feed truncated, and that is precisely the failure this producer exists to
    stop shipping silently.
    """
    if not isinstance(payload, dict):
        return None
    pag = payload.get("pagination")
    if isinstance(pag, dict) and isinstance(pag.get("count"), int):
        return pag["count"]
    return None


def year_dir(year: int) -> Path:
    return Path(f"cfb/{DATASET}/json/{year}")


def is_complete(year: int, today=None) -> bool:
    """Manifest present AND the class has closed.

    The manifest alone is not enough. It is written after the last page of
    whatever the feed held AT THAT MOMENT, and an open class keeps signing --
    so a manifest for a live class certifies a snapshot, not a total. The 2027
    class was banked on 2026-08-06 at 4,779 rows while signed classes run
    5,678-5,952, and under the manifest-only rule it would have stayed frozen
    through its own signing day.

    Deliberately NOT checking rows against a stated count here: the 26 already
    banked years carry ``expected: null`` (the field postdates them), so
    requiring agreement would invalidate every one of them and trigger a full
    re-scrape of classes that are correct and closed. Stage 52 records
    ``expected`` from the start and does check it.
    """
    if not is_recruiting_class_final(year, today):
        return False
    return (year_dir(year) / "_manifest.json").is_file()


def scrape_year(year: int, *, logger) -> dict:
    """Page a class year to exhaustion; write the manifest last.

    The manifest is the completion marker and is deliberately written AFTER the
    final page, so an interrupted run leaves the year looking unfinished rather
    than silently truncated -- the failure mode that let an empty talent table
    ship for weeks.
    """
    delay = _env_num("SDV_PY_247_DELAY", 0.5, float)
    page, total, pages_written, failed = 1, 0, 0, []
    expected: int | None = None
    while True:
        payload = _fetch_page(year, page, logger)
        if payload is None:
            failed.append(page)
            break
        if expected is None:
            expected = _expected(payload)
            if expected is not None:
                logger.info("%s %s: feed reports %s recruits", DATASET, year, expected)
        n = _rows(payload)
        if n == 0:
            break
        write_json_atomic(
            {
                "year": year,
                "page": page,
                "page_size": PAGE_SIZE,
                "sport_key": SPORT_KEY,
                "payload": payload,
            },
            year_dir(year) / f"page_{page:04d}.json",
        )
        total += n
        pages_written += 1
        logger.info(
            "%s %s: page %s -> %s rows (cum %s/%s)",
            DATASET,
            year,
            page,
            n,
            total,
            expected or "?",
        )
        if n < PAGE_SIZE or (expected is not None and total >= expected):
            break
        page += 1
        time.sleep(delay)

    # Completeness is CHECKED against the feed's own count, never inferred from
    # a short final page -- a truncated feed also ends with a short page, and
    # calling that "done" is how an incomplete class ships looking healthy.
    short = expected is not None and total != expected
    if short:
        logger.error(
            "%s %s: got %s rows, feed reported %s", DATASET, year, total, expected
        )
    manifest = {
        "year": year,
        "pages": pages_written,
        "rows": total,
        "expected_rows": expected,
        "page_size": PAGE_SIZE,
        "sport_key": SPORT_KEY,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "failed_pages": failed,
        "complete": not failed and total > 0 and not short,
    }
    if manifest["complete"]:
        write_json_atomic(manifest, year_dir(year) / "_manifest.json")
    else:
        # No manifest => the next run retries this year. A class is never empty,
        # so zero rows is a fetch failure, not a small class.
        logger.error(
            "%s %s INCOMPLETE: rows=%s failed_pages=%s (no manifest written)",
            DATASET,
            year,
            total,
            failed,
        )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-s", "--start_year", type=int, default=most_recent_cfb_season() + 1
    )
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument(
        "-r",
        "--rescrape",
        action="store_true",
        help="re-scrape years that already have a manifest (default: skip them -- classes are immutable)",
    )
    args = ap.parse_args()
    end = args.end_year or args.start_year
    if args.start_year < FIRST_CLASS_YEAR:
        raise SystemExit(
            f"start_year {args.start_year} predates the {FIRST_CLASS_YEAR} floor"
        )

    summary = []
    for year in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", year)
        if is_complete(year) and not args.rescrape:
            logger.info(
                "%s %s: already complete, skipping (--rescrape to force)", DATASET, year
            )
            continue
        summary.append(scrape_year(year, logger=logger))

    done = [m for m in summary if m["complete"]]
    print(
        json.dumps(
            {
                "years": len(summary),
                "complete": len(done),
                "rows": sum(m["rows"] for m in summary),
            }
        )
    )
    if summary and not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
