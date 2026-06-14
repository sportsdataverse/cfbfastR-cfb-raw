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

## Model training

Five native Python model packages live under `python/`:

| Package | Entry point | Dep group |
|---|---|---|
| `model_training` (T1) | `python -m model_training` | — |
| `model_training/fourth_down` (T2) | `python -m model_training.fourth_down` | — |
| `rb_eval` (T3) | `python -m rb_eval` | `gam` (pygam) |
| `pregame_wp` (T4) | `python -m pregame_wp` | — |
| `cpoe` (T5) | `python -m cpoe` | — |

Figures for all tracks require `uv sync --group figures` (plotnine).
Tests for T3 (pygam) require `uv sync --group gam`; they skip cleanly otherwise.

See `python/model_training/HANDOFF.md` for the sdv-py integration checklist.

## Spec
`docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`
