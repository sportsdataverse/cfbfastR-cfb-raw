# CLAUDE.md — cfbfastR-cfb-raw

Python/uv scraper for ESPN college-football game JSON. Sibling of `cfbfastR-cfb-data` (R).

## Commands
- `uv sync` — install (editable sdv-py from ../../sdv-py for dev; requires sportsdataverse>=0.0.69).
- `uv run pytest` — offline test suite. Live tests: `CFB_LIVE_TESTS=1 uv run pytest -m live`.
- `uv run python python/scrape_cfb_json.py -s YYYY -e YYYY -r false` — scrape.
- `uv run python python/reprocess_cfb_json.py -s YYYY -e YYYY --force` — offline rebuild.

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
  coordinate any rename/move/CLI change with that workflow.
- Never add AI co-author trailers to commits.

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
