# CLAUDE.md — cfbfastR-cfb-raw

Python/uv scraper for ESPN college-football game JSON. Sibling of `cfbfastR-cfb-data` (R).

## Commands
- `uv sync` — install (editable sdv-py from ../../sdv-py for dev; requires sportsdataverse>=0.0.52).
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
- Never add AI co-author trailers to commits.

## Spec
`docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`
