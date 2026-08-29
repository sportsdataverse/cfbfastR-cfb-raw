"""Stage 07 implementation -- ESPN CFB per-season conference standings.

Season-keyed, not game-keyed: one ESPN call per season returns the whole
conference tree, so this is the cheapest stage in the pipeline. That is why it
takes no schedule master and runs no pool -- adding either would be machinery
for a single request.

Payload shape, read from a real 2024 response rather than assumed::

    {id, name, children: [                    # 11 conferences in 2024
        {id, name, abbreviation, isConference,
         standings: {season, seasonType, entries: [
             {team: {...}, stats: [ ... 78 stats ... ]}   # 14 teams in the AAC
         ]}}
    ]}

**Completeness is a check, not an assumption.** A season with zero conferences,
or conferences with zero entries, is a failed fetch that ESPN returned 200 for --
the same 200-with-empty-payload throttle the ESPN Site v2 notes warn about. We
refuse to bank such a payload, so a later run retries it instead of skipping a
hole forever.
"""

from __future__ import annotations

import argparse

import sportsdataverse as sdv

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    most_recent_cfb_season,
    write_json_atomic,
)

DATASET = "standings"

#: A season is only banked when the payload clears these. Sized from the 2024
#: response (11 conferences) with generous headroom -- the point is to catch an
#: empty or truncated payload, not to assert a conference count.
MIN_CONFERENCES = 1
MIN_ENTRIES = 1


def out_path(season: int) -> str:
    return f"cfb/{DATASET}/json/{season}.json"


def _fetch(season: int) -> dict:
    return sdv.cfb.espn_cfb_standings(season=season, return_parsed=False)


def summarize(payload: dict) -> dict:
    """Count what actually came back, for the completeness check and the log.

    Returns:
        ``{"conferences": int, "entries": int}`` -- entries summed across every
        conference, which is the number that would silently be zero on a
        throttled 200.
    """
    children = payload.get("children") or []
    entries = sum(
        len((c.get("standings") or {}).get("entries") or []) for c in children
    )
    return {"conferences": len(children), "entries": entries}


def is_complete(season: int) -> bool:
    """True when the season is banked AND its banked payload is non-empty.

    Presence is not validity: an unparseable or empty file must not count as
    done, or a bad fetch blocks its own retry forever.
    """
    from pathlib import Path
    import json

    p = Path(out_path(season))
    if not p.is_file():
        return False
    try:
        counts = summarize(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False
    return counts["conferences"] >= MIN_CONFERENCES and counts["entries"] >= MIN_ENTRIES


def write_one(season: int, logger) -> dict:
    """Fetch and bank one season. Raises rather than banking an empty payload."""
    payload = _fetch(season)
    counts = summarize(payload)
    if counts["conferences"] < MIN_CONFERENCES or counts["entries"] < MIN_ENTRIES:
        raise RuntimeError(
            f"{DATASET} {season}: refusing to bank an empty payload "
            f"({counts['conferences']} conferences, {counts['entries']} entries) -- "
            "ESPN returns 200 with an absent array under throttle"
        )
    payload["season"] = season
    write_json_atomic(payload, out_path(season))
    logger.info(
        "%s %s: %d conferences, %d entries",
        DATASET,
        season,
        counts["conferences"],
        counts["entries"],
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument(
        "-r",
        "--rescrape",
        type=str,
        default="false",
        help="re-fetch seasons already banked (default: skip them)",
    )
    args = ap.parse_args(argv)
    end = args.end_year or args.start_year
    # Tolerant str2bool: an unknown value is False, never an exception -- a cron
    # typo must not trigger a full re-scrape.
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")

    rc = 0
    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        if is_complete(season) and not rescrape:
            logger.info(
                "%s %s: already banked, skipping (-r true to force)", DATASET, season
            )
            continue
        try:
            write_one(season, logger)
        except Exception as exc:  # noqa: BLE001 -- one bad season must not stop the range
            logger.error("%s %s: %s", DATASET, season, exc)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
