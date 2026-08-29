# CLAUDE.md — cfbfastR-cfb-raw

Python/uv scraper for ESPN college-football game JSON. Sibling of `cfbfastR-cfb-data` (R).

## Commands
- `uv sync` — install (editable sdv-py from ../../sdv-py for dev; requires sportsdataverse>=0.0.69).
- `uv run pytest` — offline test suite. Live tests: `CFB_LIVE_TESTS=1 uv run pytest -m live`.
- `source scripts/_venv.sh` then `"$PY" python/espn_cfb_04_pbp_scrape.py -s YYYY -e YYYY -r false` — scrape.
- `"$PY" python/reprocess_cfb_json.py -s YYYY -e YYYY --force` — offline rebuild.
  **Never `uv run` for a scrape or reprocess** — it re-syncs the env mid-run and has
  already swapped the interpreter under a live job (2026-07-28). `scripts/_venv.sh`
  resolves it once: `CFB_PY` override → this repo's `.venv` → loud failure.

## Conventions
- SDK boundary: all ESPN access via `sportsdataverse.cfb` (`CFBPlayProcess`, `espn_cfb_*`).
  Bug fixes go upstream to sdv-py, not here.
- Per-game task order: **raw first**, **final last** (final's existence = completion marker).
- Every aux/extra is persisted standalone AND embedded in final (offline-reprocess source).
- `write_json_atomic` for every write. `_safe()`-wrap every extra endpoint.
- ProcessPool callables must be module-level (lambdas aren't picklable) — see `_worker`.
- Commit message format is load-bearing: `"CFB Raw Update (Start: YYYY End: YYYY)"` /
  `"CFB Reprocess Update (Start: YYYY End: YYYY)"` — the `-data` trigger greps the years.
- Bump `SCHEMA_REV` when the final shape / enrichment inputs change.
- `python/scrape_cfb_qbr.py` is executed cross-repo by `cfbfastR-cfb-data`'s
  `cfb_model_pipeline.yml` (checks this repo out as `_raw`) — not an orphan;
  coordinate any rename/move/CLI change with that workflow. It is deliberately
  **retained under its original name**: the numbered stage
  `espn_cfb_08_qbr_scrape.py` is a shim over it, so the external caller keeps
  working. Retiring the old name needs that workflow updated FIRST.
- Never add AI co-author trailers to commits.

## Pipeline stages

`python/espn_cfb_NN_<name>_scrape.py` are thin shims — the directory listing IS
the pipeline. The implementation stays in the `scrape_cfb_*` modules they import,
which also keeps the cross-repo callers above working.

| NN | stage | implementation | in the daily loop? |
|---|---|---|---|
| 01 | teams | `cfb_raw_scrape/scrape_cfb_teams.py` | yes |
| 02 | schedules | `cfb_raw_scrape/scrape_cfb_schedules.py` | yes |
| 03 | team_rosters | *(reserved — not built)* | — |
| 04 | pbp / json | `cfb_raw_scrape/scrape_cfb_pbp.py` | yes |
| 05 | player_stats | *(reserved — not built)* | — |
| 06 | team_stats | *(reserved — not built)* | — |
| 07 | standings | *(reserved — not built)* | — |
| 08 | qbr | `cfb_raw_scrape/scrape_cfb_qbr.py` | yes |
| 09 | power_index | `cfb_raw_scrape/scrape_cfb_power_index.py` | yes |
| 50 | recruits | `cfb_raw_scrape/scrape_cfb_recruits.py` | monthly, preflight-gated |
| 51 | player_core | *(reserved — not built)* | monthly |

**The numbers are this repo's COLD-START EXECUTION ORDER** (renumbered
2026-08-29), not the cross-repo ESPN family slots. Reading the `python/`
listing top to bottom gives you a working pipeline.

**This DIVERGES from nba / mbb / wnba / wbb deliberately.** Those repos number by
cross-repo dataset identity, where `04` means game_rosters everywhere. CFB
numbers by its own dependency chain instead, because the CFB pipeline has joins
the others do not — teams feeding the extended schedule interface, and
`scrape_cfb_pbp` fetching rosters + participants inline. **Do not "fix" a CFB
number to match a sibling repo.** A reserved number stays EMPTY until built.

**Rosters and participants are not stages.** `scrape_cfb_pbp` (04) calls
`_rosters` and `_participants` per game and embeds both in `json/final`
(verified: 223 roster entries and 172 participants in a real file). Separate
stages meant a second ~250 Core v2 `$ref` fan-out per game against an endpoint
that 403s under load, for data 04 already has. They were deleted 2026-08-29.

**A deleted stage's number is RECLAIMED; only a RESERVED number holds its
place.** pbp moved 06 -> 04, qbr 10 -> 08, power_index 11 -> 09, and the
reserved player_stats / team_stats / standings shifted with them. The sequence
stays dense over live stages so the listing keeps reading as the pipeline; the
reserved slots are the only gaps, and each is a stage that will exist.

The pre-existing `cfb/game_rosters` and `cfb/play_participants` trees (20,696
files each) are left committed but are no longer written by anything. Nothing in
this repo or in `cfbfastR-cfb-data`'s ingest reads them — checked before the
deletion — so they are a frozen historical artifact, not a live dataset.

## Layout: entry points at the top, implementations in the package

`python/` holds only what a driver invokes — the numbered stage shims plus
`reprocess_cfb_json.py`, `filter_stale.py`, `preflight_build.py`,
`verify_season_fill.py`, `reprocess_stale_by_stamp.py`. Every scraper
implementation lives in `python/cfb_raw_scrape/` and is imported from there
(`from cfb_raw_scrape.scrape_cfb_teams import main`). The directory listing at
the top level IS the pipeline; the package is where the work is.

The ordered sequence in `scripts/daily_cfb_scraper.sh` remains the executable
truth for what actually runs each night.

## Scope: scraping + reprocess only

This repo is **scraping + offline-reprocess only**. The native Python model suite
(`model_training` + `rb_eval` / `pregame_wp` / `cpoe` + the reports/publish packages)
was **decommissioned out of `-raw` into `cfbfastR-cfb-data/python/`** (merged 2026-06-17);
`HANDOFF.md` moved with it. **Don't reintroduce ML deps here** — enrichment bug fixes go
to sdv-py, modeling work goes to `-data`.

`python/cfb_raw_scrape/` holds the scraper implementations, each named for the
stage it serves (`scrape_cfb_teams`, `_schedules`, `_game_rosters`,
`_play_participants`, `_pbp`, `_qbr`, `_power_index`, `_recruits`). `python/`
itself holds only entry points. Reprocess worker count is bounded by the
`CFB_REPROCESS_WORKERS` env var.

## Spec
`docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`
