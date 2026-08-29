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

#: ESPN group ids, probed live 2026-08-29 for season 2024:
#:
#:     80  FBS                  11 conferences, 120 teams   <- the default
#:     81  FCS                  14 conferences, 117 teams
#:     35  "Division II/III"     2 conferences,   0 teams   <- named, never populated
#:     36  "All Star"            0 / 0
#:     90  "NCAA Division I"     2 / 0   (the parent of 80/81; standings live in the children)
#:
#: **D2 and D3 are NOT available.** Group 35 exists and is named, but ESPN has
#: never published standings entries under it -- so this stage covers FBS and
#: FCS, which is everything that exists, not everything we would like.
#:
#: A leaf conference id (21 = MVFC, 22 = Ivy) also works and returns its
#: standings at the ROOT rather than under children -- see summarize().
DIVISIONS = {"fbs": 80, "fcs": 81}

#: A season is only banked when the payload clears these. Sized from the 2024
#: response (11 conferences) with generous headroom -- the point is to catch an
#: empty or truncated payload, not to assert a conference count.
MIN_CONFERENCES = 1
MIN_ENTRIES = 1

#: There is NO week or seasontype parameter -- probed live 2026-08-29: the
#: wrapper forwards unknown kwargs to download(), which raises TypeError on
#: `week=` and `seasontype=`, and standings_type accepts only its default
#: ("expanded" and "regularseason" both return 0 conferences / 0 entries).
#: The default response reports seasonType=3, name="overall" -- i.e. the
#: CURRENT cumulative standings, which for a finished season are the finals.
#:
#: That makes a mid-season fetch a SNAPSHOT, not a final answer, which is why
#: is_complete() below refuses to call the in-progress season done. Banking a
#: September snapshot of a season that ends in January and never refreshing it
#: is the same failure the 2027 recruiting class hit: completeness is not
#: finality.


def out_path(season: int, division: str) -> str:
    return f"cfb/{DATASET}/json/{division}/{season}.json"


def _fetch(season: int, group: int) -> dict:
    # standings_type is deliberately NOT passed: probed live, "expanded"
    # returns 0 conferences and 0 entries where the default returns 14/117.
    return sdv.cfb.espn_cfb_standings(season=season, group=group, return_parsed=False)


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
    # A LEAF conference group (21 = MVFC, 22 = Ivy) carries its standings at the
    # ROOT with children == []. Counting only children[] reported 0 entries for a
    # payload that actually had 11, and the completeness guard would then have
    # refused to bank a perfectly good response.
    root = len((payload.get("standings") or {}).get("entries") or [])
    return {
        "conferences": len(children) or (1 if root else 0),
        "entries": entries + root,
    }


def is_season_final(season: int, today=None) -> bool:
    """True once `season` can no longer change.

    A CFB season labelled 2024 runs into January 2025, so it is only final from
    February of the following year. Conservative on purpose: re-fetching a
    finished season costs one request, while freezing a live one costs the
    entire rest of its schedule.
    """
    from datetime import date

    today = today or date.today()
    return (today.year, today.month) >= (season + 1, 2)


def is_complete(season: int, division: str, today=None) -> bool:
    """True when the season is banked, non-empty, AND already finished.

    Three ways this returns False, each a real failure mode:

    - the file is absent
    - the file exists but is empty or unparseable (presence is not validity, so
      a bad fetch must not block its own retry)
    - the season is still being played (completeness is not finality -- ESPN
      serves current cumulative standings, so an in-progress season's file is a
      snapshot that must keep refreshing)
    """
    if not is_season_final(season, today):
        return False
    from pathlib import Path
    import json

    p = Path(out_path(season, division))
    if not p.is_file():
        return False
    try:
        counts = summarize(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False
    return counts["conferences"] >= MIN_CONFERENCES and counts["entries"] >= MIN_ENTRIES


def write_one(season: int, division: str, logger) -> dict:
    """Fetch and bank one (season, division). Raises rather than banking empty."""
    payload = _fetch(season, DIVISIONS[division])
    counts = summarize(payload)
    if counts["conferences"] < MIN_CONFERENCES or counts["entries"] < MIN_ENTRIES:
        raise RuntimeError(
            f"{DATASET} {season} {division}: refusing to bank an empty payload "
            f"({counts['conferences']} conferences, {counts['entries']} entries) -- "
            "ESPN returns 200 with an absent array under throttle"
        )
    payload["season"] = season
    payload["division"] = division
    write_json_atomic(payload, out_path(season, division))
    logger.info(
        "%s %s %s: %d conferences, %d entries",
        DATASET,
        season,
        division,
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
    ap.add_argument(
        "-d",
        "--divisions",
        default=",".join(DIVISIONS),
        help=f"comma-separated subset of {sorted(DIVISIONS)} (default: all)",
    )
    args = ap.parse_args(argv)
    end = args.end_year or args.start_year
    # Tolerant str2bool: an unknown value is False, never an exception -- a cron
    # typo must not trigger a full re-scrape.
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")

    divisions = [d.strip() for d in args.divisions.split(",") if d.strip()]
    unknown = [d for d in divisions if d not in DIVISIONS]
    if unknown:
        raise SystemExit(f"unknown division(s) {unknown}; known: {sorted(DIVISIONS)}")

    rc = 0
    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        for division in divisions:
            if is_complete(season, division) and not rescrape:
                logger.info(
                    "%s %s %s: already banked, skipping (-r true to force)",
                    DATASET,
                    season,
                    division,
                )
                continue
            try:
                write_one(season, division, logger)
            except Exception as exc:  # noqa: BLE001 -- one bad division must not stop the rest
                logger.error("%s %s %s: %s", DATASET, season, division, exc)
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
