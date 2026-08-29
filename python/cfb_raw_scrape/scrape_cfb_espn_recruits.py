"""Stage 52 implementation -- ESPN's OWN CFB recruiting classes.

Stage 50 scrapes 247Sports. This is a different provider with a different
universe, different grades and different rankings -- not a duplicate, and not a
substitute. Both are worth having; neither is the other's oracle.

**Which of ESPN's two recruiting routes, and why it is not a close call.**
ESPN exposes the same class two ways, and they do not return the same thing::

    /recruiting/{year}/athletes    <- THIS ONE
    /seasons/{season}/recruits

Measured live 2026-08-29 for 2024:

===============================  =======  ==========================
route                            count    items arrive
===============================  =======  ==========================
``/recruiting/2024/athletes``    3,815    **50/50 pre-hydrated**
``/seasons/2024/recruits``       4,095    0/50 -- bare ``$ref`` only
===============================  =======  ==========================

Both hydrate to the identical shape, so the second route buys nothing and costs
one extra request per recruit -- ~3,800 per class year, against a Core v2 host
that 403s under load. The pre-hydrated route ships everything inline:
``athlete`` (name, height, weight, position, highSchool, hometown),
``recruitingClass``, ``status`` (Signed / Committed / Undecided), ``grade``,
``attributes`` (rank, positionRank, stateRank, regionRank, and the SPARQ-style
40-yard dash / 3-cone / 20-yard shuttle grades), and ``schools[]`` -- every
school that recruited the player, each with a visit date and a per-school
commitment status.

That ``schools[]`` array is the reason this dataset is worth capturing at all:
it is recruiting-PROCESS data (who visited where, and when), not just a final
signing. Nothing else in this repo carries it.

The 280-recruit gap between the two counts is NOT reconciled here. Do not
assume one is a superset of the other; if a downstream consumer needs that
answer, measure it rather than inferring it from these numbers.

**Coverage: 2006-2028**, read from ``/recruiting`` itself (23 years, all pages
present) rather than assumed. Future class years are real -- ESPN publishes 2027
and 2028 already.

**Pagination is mandatory and the counts are large.** count=5,193 for 2026 and
3,815 for 2024, against a default page size of 100. Core v2 states ``count``
independently of what it returns, so completeness is a CHECK here, not an
inference from "the last page looked short" -- a short page is equally the
signature of a truncated feed.

**This stage cannot use the generated wrapper, and the reason is a defect.**
``espn_cfb_recruiting_players`` exposes ``limit`` but NO ``page`` -- unlike
``espn_cfb_season_players`` and ``espn_cfb_leaders``, which both have it. Passing
``page=`` falls through to ``**kwargs`` and dies as
``TypeError: download() got an unexpected keyword argument 'page'`` (verified
2026-08-29). So the wrapper can return at most one page of a feed that needs
four, with no error and no signal that anything is missing -- a caller gets
1,000 of 3,815 and a well-formed payload.

That is the FOURTH instance in this repo of generated wrappers being unable to
reach ESPN's query params (``espn_cfb_positions``, ``espn_cfb_team_roster``, the
absent Core v2 team-statistics wrapper, and now this). ``docs/ESPN_ROSTERS.md``
section 16 records the pattern. The fix belongs in sdv-py's codegen; until it
lands, this stage goes through ``dl_utils.download`` directly, which paginates
correctly: 1000 + 1000 + 1000 + 815 = 3,815 = ``count``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    is_recruiting_class_final,
    write_json_atomic,
)
from sportsdataverse.dl_utils import download

DATASET = "espn_recruits"

#: Measured good at 1000 (count=5,193 -> 6 pages). The endpoint's own default is
#: 100, which would need 52 pages for the same year.
PAGE_SIZE = 1000

#: ESPN's /recruiting index lists 2006-2028 (23 years, probed 2026-08-29).
FIRST_CLASS_YEAR = 2006

#: Hard stop so a bad count can never spin forever.
MAX_PAGES = 200

#: The route the generated wrapper cannot paginate -- see the module docstring.
RECRUITS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/recruiting/{year}/athletes"
)


def _fetch_page(year: int, page: int) -> dict:
    """One page, through the HTTP chokepoint with an explicit limit AND page."""
    resp = download(
        url=RECRUITS_URL.format(year=year), params={"limit": PAGE_SIZE, "page": page}
    )
    return resp.json() if resp is not None else {}


def year_dir(year: int) -> Path:
    return Path(f"cfb/{DATASET}/json/{year}")


def page_path(year: int, page: int) -> Path:
    return year_dir(year) / f"page_{page:04d}.json"


def manifest_path(year: int) -> Path:
    return year_dir(year) / "_manifest.json"


def is_complete(year: int, today: date | None = None) -> bool:
    """Manifest present, its own totals agree, AND the class has closed."""
    if not is_recruiting_class_final(year, today):
        return False
    p = manifest_path(year)
    if not p.is_file():
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(m.get("rows")) and m.get("rows") == m.get("expected")


def _rows(payload: dict) -> int:
    return len(payload.get("items") or [])


def _expected(payload: dict) -> int | None:
    """What Core v2 says the class holds, independent of what it returned."""
    c = payload.get("count")
    return c if isinstance(c, int) else None


def scrape_year(year: int, *, logger) -> dict:
    """Page a class year to exhaustion, then write the manifest LAST.

    The manifest is the completion marker and is written only after the final
    page and only when the row total matches ESPN's own count, so an interrupted
    or truncated run leaves the year looking unfinished rather than silently
    short.
    """
    rows = 0
    expected: int | None = None
    pages = 0
    for page in range(1, MAX_PAGES + 1):
        payload = _fetch_page(year, page)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{DATASET} {year} page {page}: non-dict payload")
        n = _rows(payload)
        if expected is None:
            expected = _expected(payload)
        if n == 0:
            break
        write_json_atomic(payload, str(page_path(year, page)))
        rows += n
        pages = page
        if expected is not None and rows >= expected:
            break
    else:
        raise RuntimeError(f"{DATASET} {year}: hit MAX_PAGES={MAX_PAGES}")

    if expected is None:
        raise RuntimeError(
            f"{DATASET} {year}: feed stated no count -- cannot verify completeness"
        )
    if rows != expected:
        raise RuntimeError(
            f"{DATASET} {year}: banked {rows} rows but the feed states {expected} "
            "-- refusing to write the manifest for a short class"
        )

    manifest = {
        "dataset": DATASET,
        "class_year": year,
        "rows": rows,
        "expected": expected,
        "pages": pages,
        "page_size": PAGE_SIZE,
        "source": "espn core v2 /recruiting/{year}/athletes",
    }
    write_json_atomic(manifest, str(manifest_path(year)))
    logger.info("%s %s: %d rows across %d pages", DATASET, year, rows, pages)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=date.today().year)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args(argv)
    start = max(args.start_year, FIRST_CLASS_YEAR)
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")

    rc = 0
    for year in range(start, end + 1):
        logger = get_logger(f"cfb_{DATASET}", year)
        if is_complete(year) and not rescrape:
            logger.info("%s %s: already banked and closed, skipping", DATASET, year)
            continue
        try:
            scrape_year(year, logger=logger)
        except Exception as exc:  # noqa: BLE001 -- one year must not stop the range
            logger.error("%s %s: %s", DATASET, year, exc)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
