# cfbfastR CFB JSON Consolidation — Design Spec

- **Date:** 2026-06-03
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repos:** `cfbfastR-cfb-raw` (new, Python/uv), `cfbfastR-cfb-data` (new, R)

## 1. Goal

Re-think CFB JSON scraping/consolidation as a modern, two-repo pipeline that mirrors the
sibling ecosystems (`hoopR-nba-raw/-data`, `wehoop-wnba-raw/-data`,
`fastRhockey-nhl-raw/-data`). Each college-football game is stored as two
single-game JSON files:

- **`raw`** — verbatim ESPN summary payload, exactly as the API returns it.
- **`final`** — fully enriched output of the `cfb_pbp` pipeline (EPA/WPA/QBR plays,
  advanced box score) plus play participants, game rosters, and normalized betting —
  self-contained per game.

The system must support (a) full historical backfill (2004→present; 2004 matches the
existing `cfbfastR-data` coverage floor), (b) incremental
in-season daily runs, and critically (c) **rebuilding `final` from `raw` already on disk**
when processing logic changes — no re-scrape, no full reshape.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Relationship to legacy `cfbfastR-raw`/`-data` | **Clean-slate replacement.** New `cfbfastR-cfb-raw` + `cfbfastR-cfb-data` follow the `{pkg}-{league}-raw/-data` naming. Legacy repos untouched, retired later (out of scope). |
| 2 | Scraper language | **Python**, packaged as a **uv** project. |
| 3 | `-data` rectangularization language | **R** — match sibling `-data` repos, reuse cfbfastR `pbp_output_schema.R` + `piggyback` releases. |
| 4 | Final JSON enrichment depth | **Full** `run_processing_pipeline()` (EPA/WPA/QBR + advBoxScore) + `play_participants` + `game_rosters` + normalized `betting`. |
| 5 | Datasets produced by `-data` | `play_by_play`, ESPN `team_box`/`player_box`, advanced box (`adv_box`), `play_participants`, `drives`, `rosters`, `betting`, `schedules`. |
| 6 | Concurrency / scope | **ProcessPool** (CPU-bound XGBoost bypasses GIL) + full backfill via `-s/-e` year args + incremental daily. |
| 7 | Rosters, participants & betting placement | **Both** embedded in `final/{id}.json` **and** written as standalone per-game JSON datasets. `play_participants` & `game_rosters` each hit their own ESPN endpoint (fetched once per game); `betting` is normalized from the summary (no extra call). |
| 8 | Reprocess-from-disk | New first-class capability gated by a `processing_version` stamp (see §7). |

## 3. SDK boundary

All ESPN fetching + enrichment lives in **sdv-py** (`sportsdataverse`), not in the
scraper. The scraper stays thin; bug fixes go upstream. Key entry points:

- `CFBPlayProcess(gameId, raw=True).espn_cfb_pbp()` → verbatim ESPN summary dict.
- `CFBPlayProcess(gameId).espn_cfb_pbp(); .run_processing_pipeline()` → enriched plays
  + `advBoxScore` + drive data (15-step pipeline; loads `ep_model.ubj`,
  `wp_spread.ubj`, `qbr_model.ubj` at import).
- `CFBPlayProcess(gameId, path_to_json=...).cfb_pbp_disk()` → load raw **from disk**
  (offline reprocess).
- `espn_cfb_play_participants(game_id)` → per-play participant mapping. **Hits its own
  ESPN endpoint** (not derivable from the summary payload).
- `espn_cfb_game_rosters(game_id)` → game roster. **Hits its own ESPN endpoint** (not
  derivable from the summary payload).

> **Odds are an enrichment input, not a passthrough (see §12.2).** Inside
> `run_processing_pipeline()`, `__helper_cfb_pickcenter()` (cfb_pbp.py) sets
> `gameSpread`/`overUnder`/`homeFavorite`, which feed **every play's EPA/WPA**
> (`wp_spread.ubj`). For 2024+ games ESPN empties the summary `pickcenter`, so the helper
> cascades to a **live** `sports.core.api.espn.com/.../odds` call
> (`__helper__espn_cfb_odds_information__`), falling back to hardcoded defaults
> `(2.5, 55.5, True, False)` on failure. Offline reprocess must NOT hit that endpoint and
> must NOT inherit defaults — so the resolved odds are persisted at scrape time (§6.3) and
> injected on reprocess (§7.1). Required upstream sdv-py rework is tracked in §12.2.
- `espn_cfb_schedule(...)` → weekly/season schedule.

> **Note:** cfbfastR R package is currently on branch `refactor/pbp-epa-wpa-modular`;
> the sdv-py CFB enrichment engine is the live contract this repo pins.

## 4. Repository architecture

```
cfbfastR-cfb-raw  (Python/uv)
  scrape ESPN -> cfb/json/{raw,final}/{game_id}.json
              + schedule master + standalone rosters/betting JSON
  on push: peter-evans/repository-dispatch (event-type daily_cfb_data) ─┐
                                                                        ▼
cfbfastR-cfb-data (R)
  read final JSON (raw.githubusercontent.com/.../cfb/json/final/{id}.json)
  -> rectangularize -> parquet/csv/rds  -> piggyback release to sportsdataverse-data
```

- **raw repo** owns scraping + enrichment; the `final` JSON is the contract.
- **data repo** owns rectangularization + releases; reuses `pbp_output_schema.R`.

## 5. Directory layout

### 5.1 `cfbfastR-cfb-raw/`

```
python/
  _cfb_raw_utils.py        # logging, ProcessPool runner, manifest reader, on-disk filter,
                           #   atomic JSON write, season helpers, PROCESSING_VERSION
  scrape_cfb_schedules.py  # season -> cfb/schedules/{parquet,rds,csv} + master
  scrape_cfb_json.py       # game_id -> raw + final + standalone rosters/participants/betting
                           #   (core: hits summary + participants + rosters endpoints ONCE each)
  reprocess_cfb_json.py    # raw-on-disk -> final/{id}.json              (offline rebuild)
  # optional single-dataset refreshers (NOT in the daily loop; hit one endpoint each):
  scrape_cfb_rosters.py      # game_id -> cfb/rosters/json/{season}/{id}.json
  scrape_cfb_participants.py # game_id -> cfb/play_participants/json/{season}/{id}.json
  scrape_cfb_betting.py      # game_id -> cfb/betting/json/{season}/{id}.json (from summary)
scripts/
  daily_cfb_scraper.sh     # season-loop orchestrator (schedules -> json[+all aux])
  backfill_cfb.sh          # wrapper: full 2004->present range
  reprocess_cfb.sh         # season-loop, reprocess only, commits "CFB Reprocess Update (...)"
cfb/
  json/raw/{game_id}.json
  json/final/{game_id}.json
  schedules/{parquet,rds,csv}/cfb_schedule_{year}.*
  rosters/json/{season}/{game_id}.json
  play_participants/json/{season}/{game_id}.json
  betting/json/{season}/{game_id}.json
  cfb_schedule_master.{parquet,rds}
logs/{scraper}_logfile_{year}.log
.github/workflows/
  scrape_cfb_raw.yml             # cron (Aug->Jan) + workflow_dispatch
  cfbfastR_cfb_data_trigger.yml  # on push -> repository_dispatch to -data
pyproject.toml                   # uv project
uv.lock
.python-version
CLAUDE.md  README.md  .gitignore  .Rbuildignore-equivalent (.gitattributes)
```

### 5.2 `cfbfastR-cfb-data/`

```
R/
  espn_cfb_01_pbp_creation.R           # final JSON -> play_by_play_{year}.parquet/csv.gz
  espn_cfb_02_team_box_creation.R      # ESPN team box
  espn_cfb_03_player_box_creation.R    # ESPN player box
  espn_cfb_04_adv_box_creation.R       # advBoxScore (adv team + adv player) -- SEE §9 open item
  espn_cfb_05_play_participants_creation.R
  espn_cfb_06_drives_creation.R
  espn_cfb_07_rosters_creation.R
  espn_cfb_08_betting_creation.R       # odds/lines/pickcenter/predictor/ATS
  espn_cfb_09_schedules_creation.R
  upload_releases.R                    # piggyback -> sportsdataverse-data
  run_summary.R                        # $GITHUB_STEP_SUMMARY
scripts/daily_cfb_R_processor.sh
cfb/
  pbp/{parquet,csv}/play_by_play_{year}.{parquet,csv.gz}
  team_box/  player_box/  adv_box/  play_participants/  drives/  rosters/  betting/  schedules/
    -> {parquet,rds}/{dataset}_{year}.*
  cfb_{dataset}_in_data_repo.{parquet,rds}     # master indices
logs/{processor}_logfile_{year}.log
.github/workflows/daily_cfb.yml          # repository_dispatch + cron + workflow_dispatch
CLAUDE.md  README.md  DESCRIPTION-or-renv (R deps)
```

Output `cfb/` is committed to each repo; GitHub doubles as store + CDN. Flat per-season
files (no nested `game_id/` dirs) keep `arrow::open_dataset` + per-season binds fast.

## 6. Data contract: raw vs. final JSON

### 6.1 `cfb/json/raw/{game_id}.json`

Verbatim ESPN summary payload, no transformation. Written **first**, before enrichment,
for crash resilience and offline reprocess. Top-level keys as ESPN returns them:
`boxscore, drives, header, gameInfo, leaders, scoringPlays, winprobability, odds,
pickcenter, predictor, againstTheSpread, broadcasts, videos, standings, format`.
(`drives` is sometimes absent/empty upstream.)

### 6.2 `cfb/json/final/{game_id}.json`

```jsonc
{
  "id": 401628455,
  "season": 2024, "week": 1, "season_type": 2,
  "processing_version": "0.0.51+1",          // §7 version gate
  "count": 185,
  "plays": [ { /* 150+ enriched cols: EPA, wpa, *_player_name, down/dist, yardage, drive.* */ } ],
  "play_participants": [ { "play_id": ..., "athlete_id": ..., "role": ... } ],
  "advBoxScore": { "teams": {}, "players": {} },   // SEE §9 open item
  "boxScore":    { /* raw ESPN team/player box passthrough */ },
  "game_rosters": [ { "game_id":..., "season":..., "team_id":..., "athlete_id":..., "name":..., "position":..., "jersey":... } ],
  "betting": {
      // resolved odds actually used by enrichment (the EPA/WPA inputs) — persisted so
      // reprocess injects these instead of re-fetching/falling back to defaults:
      "game_spread": -7.5, "over_under": 52.5,
      "home_favorite": true, "home_team_spread": -7.5, "game_spread_available": true,
      "odds_source": "summary_pickcenter" | "core_odds_api" | "default",
      // raw payloads (whichever applied) carried for forensics + re-normalization:
      "pickcenter": [], "odds": [], "predictor": {}, "against_the_spread": [],
      "odds_core_items": []   // verbatim sports.core.api odds items when that path was used
  },
  "drives": { /* raw ESPN drives passthrough */ },
  "scoringPlays": [], "winprobability": [], "leaders": [],
  "header": {}, "gameInfo": {}, "standings": {},
  "homeTeamId": 333, "awayTeamId": 99, "timeouts": {},
  "broadcasts": [], "videos": []
}
```

- Top-level `id/season/week/season_type` are **echoed** (self-describing returns).
- `betting` is **normalized in Python** (stable shape, null-safe defaults) rather than
  passed through raw — insulates the R binder from ESPN's `odds`/`pickcenter`/ATS drift.
  It also carries the **resolved odds** (`game_spread`/`over_under`/`home_favorite`/
  `game_spread_available`) + `odds_source` so reprocess reuses the exact EPA/WPA inputs
  (§7.1) and never re-fetches the live core-odds endpoint.
- `drives` stays a **raw passthrough**; the enriched `plays` already carry `drive.*`
  columns, so the drives dataset in `-data` is a light re-normalization.

### 6.3 Standalone aux JSON

Three standalone per-game datasets, each the same data embedded in `final`, each echoing
`game_id`/`season`/`week` for self-describing joins:

- `cfb/rosters/json/{season}/{id}.json` — the `game_rosters` array (from the rosters endpoint).
- `cfb/play_participants/json/{season}/{id}.json` — the `play_participants` array (from the
  participants endpoint).
- `cfb/betting/json/{season}/{id}.json` — the normalized `betting` object, including the
  **resolved odds** (`game_spread`/`over_under`/`home_favorite`/`game_spread_available` +
  `odds_source`) and the verbatim `odds_core_items` when the live core-odds endpoint was
  used. This is the **offline source of the EPA/WPA spread inputs** for 2024+ games whose
  summary `pickcenter` is empty.

These standalone files are the **authoritative offline source** for the endpoint-backed
data: because participants, rosters, and (for 2024+) the resolved odds can't be
reconstructed from the on-disk summary, persisting them here lets the offline reprocess
(§7) read them from disk rather than depending on a prior `final` existing or hitting the
network.

### 6.4 How `-data` resolves the per-game JSON (enumeration + retrieval)

`raw/` and `final/` are **flat** (`cfb/json/{raw,final}/{game_id}.json`) — sibling
convention. The `-data` repo links to them over **HTTP**, never via a checkout of the raw
repo. The flow per `-data` run:

1. **Season range** — extracted from the `repository_dispatch` commit message
   (`grep -oP 'Start:\s*\K[0-9]{4}'` / `End:`), or from `workflow_dispatch` inputs, or
   defaulted to `most_recent_cfb_season()`.
2. **Enumerate `game_id`s** — the **schedule master is the index**. ESPN `game_id`s are not
   season-decodable, so `-data` downloads
   `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/cfb_schedule_master.parquet`
   once, `arrow::read_parquet()`s it, and filters to `season ∈ [start, end]` to get the
   `game_id` list (and `season`/`week`/team meta for stamping).
3. **Fetch each `final` JSON** — construct
   `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/json/final/{game_id}.json`
   and read with `jsonlite::fromJSON()`. Aux-only datasets (rosters, participants, betting)
   read from their season-partitioned URLs (`cfb/{dataset}/json/{season}/{game_id}.json`)
   when built independently of the flagship PBP; otherwise they are taken from the embedded
   blocks in the already-fetched `final` JSON to avoid duplicate requests.
4. **Bind + write** — each `espn_cfb_0N_*.R` script binds the per-game frames into a
   per-season table, conforms PBP to `pbp_output_schema.R`, writes parquet/csv/rds, and
   updates `cfb_{dataset}_in_data_repo.parquet`.

**Robustness against the raw CDN cache.** `raw.githubusercontent.com` serves with a ~5-min
cache, so a just-pushed `final` can briefly 404. Mitigations: (a) the `-data` cron fires
offset **+2h** after raw (§10.3), well past the cache window; (b) per-file fetch uses a
small retry-with-backoff on 404/transient errors; (c) a fetch that still fails is logged
and skipped (the game is picked up on the next run), never aborting the season bind.

> **Implementation note:** for a large historical *recreate* (≈17k files), HTTP-per-file is
> slow but rare and resumable. If it becomes a bottleneck, an opt-in local-checkout fast
> path can be added later without changing the `final` contract — out of scope here.

## 7. Reprocess-from-disk (rebuild `final` from `raw`)

Two rebuild axes, two costs:

1. **Reprocess** = `raw JSON -> final JSON` (Python, ~5-10s/game XGBoost) — **expensive**;
   this section. Done from disk when `CFBPlayProcess` logic changes.
2. **Recreate** = `final JSON -> parquet` (R, in `-data`) — cheap reshape, already covered
   by re-running the `espn_cfb_0N_*.R` scripts.

### 7.1 `python/reprocess_cfb_json.py`

- Args: `-s/-e` season range (or `--all`), `--force`, `--refresh-aux`.
- Iterates `cfb/json/raw/*.json` on disk (NOT the schedule master, NOT ESPN).
- Per game:
  - `CFBPlayProcess(gameId, path_to_json="cfb/json/raw").cfb_pbp_disk()` → load raw from disk.
  - **Inject persisted odds** from `cfb/betting/json/{season}/{id}.json`
    (`game_spread`/`over_under`/`home_favorite`/`game_spread_available`) so
    `__helper_cfb_pickcenter` uses them instead of cascading to the live core-odds endpoint
    or hardcoded defaults — this keeps the EPA/WPA spread inputs identical to scrape time.
    (Requires the sdv-py rework in §12.2.)
  - `run_processing_pipeline()` → rebuild `plays` / `advBoxScore` / `drives` with the
    injected spread.
  - `betting` re-emitted from the disk betting JSON (deterministic, offline).
  - `game_rosters` from `cfb/rosters/json/{season}/{id}.json` on disk (offline); only
    re-fetch via `espn_cfb_game_rosters()` if `--refresh-aux`.
  - `play_participants` from `cfb/play_participants/json/{season}/{id}.json` on disk
    (offline); only re-fetch via `espn_cfb_play_participants()` if `--refresh-aux`.
  - Stamp `processing_version`; atomic write `final/{id}.json`.

Because participants and rosters are endpoint-backed (not in the summary), their standalone
disk JSON is the offline source — no dependence on a prior `final`. If a standalone aux file
is missing for a game, reprocess logs it and (a) re-fetches if `--refresh-aux`, else (b)
emits the field empty and flags the game for a targeted refresher run.

| `final` field | reprocess source | network? |
|---|---|---|
| `plays`, `advBoxScore`, `drives` | re-run pipeline on disk raw + injected odds | no |
| `betting` (incl. resolved odds) | disk `betting/json/{season}/{id}.json` | no (yes if `--refresh-aux`) |
| `game_rosters` | disk `rosters/json/{season}/{id}.json` | no (yes if `--refresh-aux`) |
| `play_participants` | disk `play_participants/json/{season}/{id}.json` | no (yes if `--refresh-aux`) |
| `header`/`gameInfo`/`scoringPlays`/… | passthrough disk raw | no |

### 7.2 Version gate

`_cfb_raw_utils.py` defines `PROCESSING_VERSION = f"{sportsdataverse.__version__}+{SCHEMA_REV}"`
and stamps it into every `final` JSON. Reprocess **skips** games whose
`final.processing_version` already matches the current value — so after an enrichment
change you bump `SCHEMA_REV` (or sdv-py ships a new version) and only stale games rebuild.
`--force` rebuilds everything; reprocess is idempotent and resumable (re-run after a crash
skips already-current games for free).

### 7.3 Trigger reuse

`scripts/reprocess_cfb.sh` uses the same season-loop + temp-log + git-commit idiom, runs
only `reprocess_cfb_json.py`, and commits `"CFB Reprocess Update (Start: $i End: $i)"`.
The `Start:/End:` tokens are preserved, so the existing push trigger fires and `-data`
recreates parquet for exactly those seasons — no special wiring.

## 8. Python scraper internals + orchestration

### 8.1 `_cfb_raw_utils.py`

- `get_logger(name, year)` — `FileHandler(logs/{name}_logfile_{year}.log)` + `StreamHandler`
  (both the GH Actions log and the tracked logfile capture output).
- `run_pool(fn, items, *, kind="process"|"thread", workers, desc)` — tqdm-wrapped pool.
  ProcessPool for enrichment (`scrape_cfb_json`, `reprocess_cfb_json`); ThreadPool for
  IO-only (`scrape_cfb_schedules`, rosters, betting).
- `load_schedule_master()`, `games_for_seasons(master, start, end)`.
- `filter_undone(games, dir, rescrape=False)` — drop games whose `final/{id}.json` exists.
- `write_json_atomic(obj, path)` — write `{id}.json.tmp` then `os.replace`.
- `most_recent_cfb_season()` — Aug rollover heuristic / sdv-py helper.
- `PROCESSING_VERSION`, `SCHEMA_REV`.

### 8.2 `scrape_cfb_json.py` per-game task (raw-first ordering)

```python
def download_game(game_id, season, rescrape, logger):
    raw = CFBPlayProcess(gameId=game_id, raw=True).espn_cfb_pbp()
    write_json_atomic(raw, f"cfb/json/raw/{game_id}.json")        # bank raw first
    proc = CFBPlayProcess(gameId=game_id)
    proc.espn_cfb_pbp(); result = proc.run_processing_pipeline()  # ~5-10s, XGBoost
    # endpoint-backed aux — fetched ONCE each, written both standalone AND embedded:
    participants = espn_cfb_play_participants(game_id=game_id).to_dicts()
    rosters      = espn_cfb_game_rosters(game_id=game_id).to_dicts()
    # betting carries the RESOLVED odds the pipeline actually used (proc.gameSpread, etc. —
    # incl. the core-odds path for 2024+) + the raw payloads + odds_source, so reprocess
    # can inject them offline (§7.1, §12.2):
    betting      = _capture_betting(raw, proc)                    # resolved odds + payloads
    write_json_atomic(_stamp(participants, game_id, season),
                      f"cfb/play_participants/json/{season}/{game_id}.json")
    write_json_atomic(_stamp(rosters, game_id, season),
                      f"cfb/rosters/json/{season}/{game_id}.json")
    write_json_atomic(_stamp(betting, game_id, season),
                      f"cfb/betting/json/{season}/{game_id}.json")
    result["play_participants"] = participants
    result["game_rosters"]      = rosters
    result["betting"]           = betting
    result.update(id=game_id, season=season, week=..., season_type=...,
                  processing_version=PROCESSING_VERSION)
    write_json_atomic(result, f"cfb/json/final/{game_id}.json")   # final written LAST
```

Each task is wrapped so a bad game logs `logger.exception(...)` and the pool continues.
Ordering is deliberate: **raw banked first** (survives an enrichment crash), **final
written last** (its existence is the incremental-completion marker for `filter_undone`).
The daily path hits each ESPN endpoint exactly once per game; the standalone
`scrape_cfb_{rosters,participants,betting}.py` scripts exist only for targeted
single-dataset refresh outside the daily loop.

### 8.3 `scripts/daily_cfb_scraper.sh`

Season loop with the sibling temp-log-then-commit git idiom:

```bash
for i in $(seq "$START" "$END"); do
  TMPLOG=$(mktemp /tmp/cfb_raw_${i}.XXXXXX.log)
  { git pull >/dev/null; git config --local user.email "action@github.com"; git config --local user.name "Github Action"
    uv run python python/scrape_cfb_schedules.py -s $i -e $i -r $RESCRAPE   # schedules FIRST
    uv run python python/scrape_cfb_json.py      -s $i -e $i -r $RESCRAPE   # raw+final+all aux
    git pull >/dev/null; git add cfb/* ; git commit -m "CFB Raw Update (Start: $i End: $i)" || echo "No changes"
    git pull >/dev/null; git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  git pull --rebase >/dev/null || true; git add logs/cfb_raw_logfile_${i}.log
  git commit -m "CFB Raw log update (Start: $i End: $i)" >/dev/null || true; git push >/dev/null; rm -f "$TMPLOG"
done
```

- **Schedules run first** — every other scraper reads `cfb_schedule_master.parquet`.
- **Commit message is load-bearing** — `"CFB Raw Update (Start: YYYY End: YYYY)"` is the
  only channel carrying the season range across the repo boundary.

## 9. uv project configuration

- `pyproject.toml` + `uv.lock` + `.python-version` (replaces `requirements.txt`).
- Deps: `sportsdataverse>=0.0.51`, `pandas`, `polars`, `pyarrow`, `tqdm`.
- `[tool.uv.sources]` path source for local dev (`sportsdataverse = { path = "../../sdv-py", editable = true }`
  — relative to repo root: `cfbfastR-dev/cfbfastR-cfb-raw` → `sdv-dev/sdv-py`),
  while `uv.lock` pins the published release for reproducible CI.
- All script invocations use `uv run python ...` so the locked env is used everywhere.

## 10. GitHub Actions

### 10.1 `cfbfastR-cfb-raw/.github/workflows/scrape_cfb_raw.yml`

- Triggers: `schedule` (CFB calendar windows — Aug preseason → Jan CFP, UTC) +
  `workflow_dispatch` (`start_year`/`end_year`/`rescrape` inputs).
- Steps: `actions/checkout` → `astral-sh/setup-uv` → `uv sync --frozen` → empty-input
  fallback to `most_recent_cfb_season()` → `bash scripts/daily_cfb_scraper.sh`.
- Token: `secrets.GITHUB_TOKEN` (same-repo commits).

### 10.2 `cfbfastR-cfb-raw/.github/workflows/cfbfastR_cfb_data_trigger.yml`

- On push → `peter-evans/repository-dispatch@v4`, `token: secrets.SDV_GH_TOKEN`
  (cross-repo PAT), `repository: sportsdataverse/cfbfastR-cfb-data`,
  `event-type: daily_cfb_data`, client payload includes the commit message.

### 10.3 `cfbfastR-cfb-data/.github/workflows/daily_cfb.yml`

- Triggers: `repository_dispatch: [daily_cfb_data]` + `schedule` (offset +2h after raw) +
  `workflow_dispatch`.
- Extract years from commit message: `grep -oP 'Start:\s*\K[0-9]{4}'` / `'End:\s*\K[0-9]{4}'`.
- Steps: `r-lib/actions/setup-r` + deps (`sportsdataverse/cfbfastR`, `ropensci/piggyback`)
  → `bash scripts/daily_cfb_R_processor.sh` (runs `espn_cfb_0N_*.R` sequentially,
  non-fatal per script, exit code propagated) → `upload_releases.R` → `run_summary.R`.

## 11. Logging

- Per-season files: `logs/{scraper}_logfile_{year}.log` (raw) /
  `logs/{processor}_logfile_{year}.log` (data).
- Written to `/tmp` during the script, copied to tracked `logs/` after git ops, committed
  separately (avoids conflicts inside the pull/commit/push block).
- Python `get_logger` dual-sinks to file + stdout; GH Actions captures stdout.
- Use `::error ::` / `::warning ::` workflow annotations for failures.

## 12. Open work items / risks

1. **advBoxScore expansion (flagged).** The `create_box_score()` output (team + player
   advanced stats) likely needs flattening/expansion refinement in
   `espn_cfb_04_adv_box_creation.R`, and possibly upstream tweaks in sdv-py's
   `create_box_score()`. Treat as its own sub-task when reached; do not assume the current
   shape is release-ready.
2. **Odds-resolution rework in sdv-py (prerequisite for offline reprocess).** The spread/OU
   are EPA/WPA inputs, not passthroughs (see §3 callout). `__helper_cfb_pickcenter()` uses
   the summary `pickcenter` when present, but for 2024+ games it cascades to a **live**
   `sports.core.api.espn.com/.../odds` call (`__helper__espn_cfb_odds_information__`),
   defaulting to `(2.5, 55.5, True, False)` on failure. Required work:
   (a) **prefer the raw-summary keys** (`pickcenter`/`odds`/`predictor`/`againstTheSpread`,
   all in raw) whenever they carry the spread — investigate whether the summary `odds` key
   holds it for 2024+ (if so, no extra endpoint is ever needed);
   (b) make the live core-odds call happen **only at scrape time**, exposing the resolved
   odds + raw items on the processor so they're persisted to the betting JSON (§6.3);
   (c) add an **injected-odds path** to `CFBPlayProcess` (e.g. accept resolved odds /
   read from `path_to_json`) so reprocess (§7.1) supplies the persisted spread and the
   live endpoint + defaults are never reached offline.
   Until (c) lands, reprocess of 2024+ games is not provably offline — treat as a gating
   sub-task of Plan 1.
3. **`espn_cfb_play_participants` / `espn_cfb_game_rosters` are endpoint-backed
   (confirmed).** Both hit their own ESPN endpoints, not the summary. Therefore both are
   persisted as standalone per-game JSON (§6.3) so the offline reprocess reads them from
   disk (§7.1) — no dependence on a prior `final`. `--refresh-aux` re-hits the endpoints
   when an aux *parser* changes. Implementation note: confirm the exact endpoint URLs and
   whether participant `athlete_id`s require a further `$ref` resolve (which would add a
   call); if so, the standalone participant JSON already captures the resolved result, so
   reprocess stays offline regardless.
4. **`drives` upstream absence.** ESPN sometimes omits `drives`; the drives dataset and the
   `drive.*` play columns must degrade gracefully (empty, not error).
5. **Backfill volume.** 2004→present × thousands of games × ~5-10s enrichment is a long
   run; ProcessPool + `filter_undone` + `processing_version` make it resumable. Initial
   backfill likely run in season chunks rather than one job.
6. **cfbfastR `pbp_output_schema.R` reuse.** `-data` PBP creation should conform to the
   ~150-column canonical schema; verify alignment with the modular-refactor branch.

## 13. Out of scope

- Retiring/migrating legacy `cfbfastR-raw` / `cfbfastR-data` (later).
- Changes to the sdv-py enrichment engine beyond what's needed for advBoxScore (§12.1) and
  the odds-resolution rework (§12.2) — both of which ARE in scope as gating sub-tasks.
- KenPom/other non-ESPN sources.
