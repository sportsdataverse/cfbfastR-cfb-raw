# CLAUDE.md — cfbfastR-cfb-raw

Python/uv scraper for ESPN college-football game JSON. Sibling of `cfbfastR-cfb-data` (R).

## Commands
- `uv sync` — install (editable sdv-py from ../../sdv-py for dev; requires sportsdataverse>=0.0.69).
- `uv run pytest` — offline test suite. Live tests: `CFB_LIVE_TESTS=1 uv run pytest -m live`.
- `source scripts/_venv.sh` then `"$PY" python/espn_cfb_06_pbp_scrape.py -s YYYY -e YYYY -r false` — scrape.
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
  `espn_cfb_10_qbr_scrape.py` is a shim over it, so the external caller keeps
  working. Retiring the old name needs that workflow updated FIRST.
- Never add AI co-author trailers to commits.

## Pipeline stages

`python/espn_cfb_NN_<name>_scrape.py` are thin shims — the directory listing IS
the pipeline. The implementation stays in the `scrape_cfb_*` modules they import,
which also keeps the cross-repo callers above working.

| NN | stage | implementation |
|---|---|---|
| 01 | schedules | `scrape_cfb_schedules.py` |
| 02 | pbp | `scrape_cfb_json.py` |
| 04 | game_rosters | `scrape_cfb_game_rosters.py` |
| 10 | recruits | `scrape_cfb_recruits.py` |
| 11 | play_participants | `scrape_cfb_participants.py` |
| 12 | power_index | `scrape_cfb_power_index.py` |
| 13 | qbr | `scrape_cfb_qbr.py` |
| 14 | teams | `scrape_cfb_teams.py` |

**03, 05–09 are HOLES and stay empty.** 01–09 are the shared ESPN family slots
(03 standings, 05 draft, 06 player_stats, 07 team_stats, 08 team_rosters,
09 player_core) which CFB does not scrape. A number means the same dataset in
every ESPN `-raw` repo — that is worth more than a dense sequence, so never
compact them. CFB-only datasets start at 10.

The number is intended **build** order, not run order: the ordered sequence in
`scripts/daily_cfb_scraper.sh` is the executable truth.

## Scope: scraping + reprocess only

This repo is **scraping + offline-reprocess only**. The native Python model suite
(`model_training` + `rb_eval` / `pregame_wp` / `cpoe` + the reports/publish packages)
was **decommissioned out of `-raw` into `cfbfastR-cfb-data/python/`** (merged 2026-06-17);
`HANDOFF.md` moved with it. **Don't reintroduce ML deps here** — enrichment bug fixes go
to sdv-py, modeling work goes to `-data`.

`python/` holds the scrapers (`scrape_cfb_json`, `_game_rosters`, `_participants`,
`_power_index`, `_qbr`, `_schedules`) + `reprocess_cfb_json.py`. Reprocess worker count is
bounded by the `CFB_REPROCESS_WORKERS` env var.

## Spec
`docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`
