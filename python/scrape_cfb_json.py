"""Core CFB scraper: per-game raw + enriched final + standalone aux/extras."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

import sportsdataverse as sdv
from sportsdataverse.cfb import CFBPlayProcess

from cfb_raw_scrape._cfb_raw_utils import (
    PROCESSING_VERSION,
    _safe,
    filter_hollow,
    filter_ids_file,
    filter_undone,
    games_for_seasons,
    get_logger,
    load_schedule_master,
    most_recent_cfb_season,
    run_pool,
    season_type_from_raw,
    status_state,
    stamp,
    write_json_guarded,
)
import cfb_raw_scrape.proxy_pool as proxy_pool
from cfb_raw_scrape.cfb_betting import capture_betting
from cfb_raw_scrape.cfb_team_box_extra import team_box_extra_from_summary

# FPI/powerindex + full event_odds only return data for recent CFB seasons
# (probe: 2014 empty, 2024 populated). Tunable.
EXTRAS_MIN_SEASON = 2015

# sdv-py download() defaults to num_retries=15 (16 attempts) and retries even a
# definitive ESPN 404 (NoESPNDataError) — catastrophic for the rosters/
# participants path, which fans out ~250 Core v2 $ref calls per game: every
# no-roster game (FCS / data gaps) burns 16 attempts x ~51s of backoff on a 404
# that will never succeed, and under real throttle the 16x retries amplify load
# and deepen the rate-limit. Cap retries low for the extras: fail fast on a 404,
# keep a small cushion for genuine transients. Override via CFB_EXTRAS_RETRIES.
EXTRAS_NUM_RETRIES = int(os.getenv("CFB_EXTRAS_RETRIES", "2"))


# --- thin sdv-py adapters (monkeypatch points in tests) ---
def _participants(gid):
    return sdv.cfb.espn_cfb_play_participants(
        game_id=gid, return_as_pandas=True, num_retries=EXTRAS_NUM_RETRIES
    ).to_dict("records")


def _rosters(gid):
    return sdv.cfb.espn_cfb_game_rosters(
        game_id=gid, return_as_pandas=True, num_retries=EXTRAS_NUM_RETRIES
    ).to_dict("records")


def _power_index(gid):
    # sportsdataverse 0.0.51+ renamed event_powerindex -> game_powerindex and
    # defaults return_parsed=True; we want the raw Core v2 {items} dict to bank.
    return sdv.cfb.espn_cfb_game_powerindex(
        event_id=gid, return_parsed=False, num_retries=EXTRAS_NUM_RETRIES
    )


def _odds_full(gid):
    # 0.0.51+ rename: event_odds -> game_odds; raw dict via return_parsed=False.
    return sdv.cfb.espn_cfb_game_odds(
        event_id=gid, return_parsed=False, num_retries=EXTRAS_NUM_RETRIES
    )


def _home_away_ids(raw: dict):
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get(
        "competitors"
    ) or []
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
    # Rotate to the next proxy for this game. Per-GAME (not per-request) is the
    # right granularity: one game is ~224 sequential Core v2 calls, so rotating
    # per request would throw away connection reuse for no extra IP spread.
    # Returns None when proxying is off or the bandwidth reserve was reached.
    proxy_hostport = proxy_pool.apply_to_env(season)
    try:
        # 1. bank RAW first -- but never let a degraded fetch clobber good data.
        # If ESPN answered with a 5xx/empty body, the allowlist dict collapses to
        # a ~250-byte stub; writing it would destroy a banked 50-400KB summary,
        # and every downstream output derives from it. Refuse the write AND skip
        # the rest of the game so raw/ and final/ can't drift apart.
        raw = CFBPlayProcess(gameId=game_id, raw=True).espn_cfb_pbp()
        if not write_json_guarded(raw, f"cfb/json/raw/{game_id}.json", logger=logger):
            logger.warning(
                "degraded summary for %s (proxy=%s) -- keeping banked copy, skipping game",
                game_id,
                proxy_hostport or "direct",
            )
            return "degraded"

        # 2. enrich
        proc = CFBPlayProcess(gameId=game_id)
        proc.espn_cfb_pbp()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)

        # 3. aux (endpoint-backed) — each _safe-wrapped so one failure doesn't kill the game
        participants = _safe(_participants, game_id, logger=logger, default=[])
        rosters = _safe(_rosters, game_id, logger=logger, default=[])

        recent = season >= EXTRAS_MIN_SEASON
        odds_full = (
            _safe(_odds_full, game_id, logger=logger, default=[]) if recent else []
        )
        power_index = (
            _safe(_power_index, game_id, logger=logger, default={}) if recent else {}
        )
        betting = capture_betting(raw, proc, odds_full=odds_full, propbets=[])

        # 4. team_box_extra: prefer summary (de-dup gate); {} if summary lacks it
        team_extra = team_box_extra_from_summary(raw, [home_id, away_id]) or {}

        injuries = raw.get("injuries") or []
        game_notes = raw.get("gameNotes") or []
        week = result.get("week")

        # 5. standalone datasets (each is an offline-reprocess source)
        standalone = {
            "game_rosters": rosters,
            "play_participants": participants,
            "betting": betting,
            "power_index": power_index,
            "team_box_extra": team_extra,
        }
        # Same no-shrink guard per dataset: _safe() defaults each of these to
        # []/{} on a transient failure, and writing that empty default over a
        # good banked roster/participants file is the identical clobber.
        for name, obj in standalone.items():
            write_json_guarded(
                stamp(obj, game_id=game_id, season=season, week=week),
                f"cfb/{name}/json/{game_id}.json",
                logger=logger,
            )

        # 6. embed + write FINAL last
        result.update(
            id=game_id,
            season=season,
            week=week,
            season_type=season_type_from_raw(raw),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            play_participants=participants,
            game_rosters=rosters,
            betting=betting,
            power_index=power_index,
            team_box_extra=team_extra,
            injuries=injuries,
            game_notes=game_notes,
            homeTeamId=home_id,
            awayTeamId=away_id,
        )
        # A summary fetched before kickoff carries no plays. Banking it as final
        # makes filter_undone treat the game as already scraped, so it is skipped
        # for the REST OF THE SEASON while the job keeps reporting green -- the
        # 2026-08-02 reprocess banked 946 such shells, one per unplayed 2026 game.
        #
        # The aux datasets written above ARE meaningful before kickoff (rosters,
        # betting lines, power index), so they are kept; only the final is withheld.
        if status_state(raw) == "pre":
            logger.info(
                "pre-game summary for %s -- aux banked, final withheld until kickoff",
                game_id,
            )
            return "pregame"

        write_json_guarded(result, f"cfb/json/final/{game_id}.json", logger=logger)
        return "ok"
    except Exception:
        logger.exception("download_game failed: %s", game_id)
        return "error"


def _worker(args):
    """Module-level (picklable) wrapper for ProcessPoolExecutor. args = (game_id, season, rescrape)."""
    game_id, season, rescrape = args
    return download_game(game_id, season, rescrape)


def _scrape_workers():
    """Game-pool worker count. CFB_SCRAPE_WORKERS env overrides the default
    (cpu_count-2); lower it (e.g. 3) to stay under ESPN's Core v2 rate limit
    on large backfills."""
    val = os.getenv("CFB_SCRAPE_WORKERS")
    return int(val) if val else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    ap.add_argument(
        "--hollow",
        type=str,
        default="false",
        help="rescrape only games flagged as hollow_extras/no_final in logs/scrape_failures.csv",
    )
    ap.add_argument(
        "--ids-file",
        default=None,
        help=(
            "rescrape exactly the game ids in this file (whitespace-separated). "
            "Used by scripts/retry_degraded_games.sh, which harvests degraded ids "
            "out of the logs. Forces a re-scrape of those ids regardless of -r."
        ),
    )
    args = ap.parse_args()
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    hollow = str(args.hollow).lower() in ("1", "true", "yes")
    master = load_schedule_master()
    for season in range(args.start_year, end + 1):
        logger = get_logger("cfb_json", season)
        all_games = games_for_seasons(master, season, season)
        if args.ids_file:
            # An explicit id list is a targeted recovery pass: the caller already
            # decided these games need re-fetching, so it forces the re-scrape.
            # Intersected with the season's schedule so a stray id cannot send the
            # scraper after a game that is not this season's.
            games = filter_ids_file(all_games, args.ids_file)
            logger.info(
                "season %s: %d games from %s are in this season's schedule",
                season,
                len(games),
                args.ids_file,
            )
        elif hollow:
            games = filter_hollow(all_games)
            logger.info(
                "season %s: %d hollow/missing games to rescrape", season, len(games)
            )
        else:
            games = filter_undone(all_games, rescrape=rescrape)
            logger.info(
                "season %s: %d games to scrape (rescrape=%s)",
                season,
                len(games),
                rescrape,
            )
        run_pool(
            _worker,
            # thread the parsed -r flag through; hollow mode is a forced
            # re-scrape of flagged games regardless of -r
            [(g, season, rescrape or hollow or bool(args.ids_file)) for g in games],
            kind="process",
            desc=f"cfb {season}",
            workers=_scrape_workers(),
        )


if __name__ == "__main__":
    main()
