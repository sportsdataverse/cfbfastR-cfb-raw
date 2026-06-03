# cfbfastR-cfb-data Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cfbfastR-cfb-data` — an **R** repo that reads the per-game enriched `final` JSON from `cfbfastR-cfb-raw` (over HTTP) and rectangularizes each block into release parquet/csv/rds, committed in-repo **and** published to `sportsdataverse-data` releases, on the CFB calendar via `repository_dispatch` from `-raw`.

**Architecture:** Pure **reshape, not re-enrichment** — the `-raw` pipeline already ran `CFBPlayProcess` in Python, so `final.plays` already holds the ~150 enriched PBP columns and `final.advBoxScore` already holds 8 box sections. Each R creation script: enumerate games for a season range (from the `-raw` schedule master), fetch each `final/{game_id}.json` via `jsonlite`, pluck one block, bind across games, conform columns, write `cfb/{dataset}/{parquet,rds,csv}/`, and `piggyback::pb_upload` to **both** `sportsdataverse/cfbfastR-cfb-data` and `sportsdataverse/sportsdataverse-data`. PBP conforms to cfbfastR's real `.pbp_apply_output_schema()`. A bounded sdv-py expansion adds the missing advBoxScore pieces (Phase H).

**Tech Stack:** R (≥4.1), testthat; arrow, jsonlite, dplyr, purrr, data.table, piggyback, cli, glue, optparse, curl; cfbfastR (for `.pbp_apply_output_schema` + `most_recent_cfb_season`); bash; GitHub Actions (`r-lib/actions`). Phase H also touches Python/sdv-py.

**Spec:** `docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md` (§5.2, §6.4, §10.3, §12).

**Decisions (locked this session):** Release target = **both** (commit in-repo parquet + piggyback to `sportsdataverse-data`). v1 scope = **per-game datasets only** (defer rankings/recruiting/QBR season-joins). advBoxScore = **expand in Plan 2** (Phase H).

**Repo:** create `sportsdataverse/cfbfastR-cfb-data` (public). Local path target: `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfbfastR-cfb-data`.
**sdv-py:** `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\sdv-py` (Phase H; branch off `main`).
**raw repo (input):** `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb/...`

---

## The `final` JSON contract this repo consumes (per game)

From `cfbfastR-cfb-raw` (spec §6.2, post-prune §12.8). Top-level keys:
- `id, season, week, season_type, processing_version, count`
- `plays` — list of ~150-col enriched play dicts (EPA/WPA/QBR, `*_player_name`, down/dist, `drive.*`).
- `play_participants` — list of `{play_id, athlete_id, role, ...}`.
- `advBoxScore` — `{pass, rush, receiver, team, situational, defensive, turnover, drives}` (see Phase H).
- `boxScore` — raw ESPN team/player box passthrough.
- `game_rosters` — list of roster rows.
- `betting` — `{game_spread, over_under, home_favorite, home_team_spread, game_spread_available, odds_source, pickcenter, odds, predictor, against_the_spread, odds_core_items, odds_full, propbets}`.
- `power_index` — FPI dict (recent seasons; `{}` for old).
- `team_box_extra` — `{ "{team_id}": { record, linescores, statistics, leaders } }`.
- `drives` — raw ESPN drives passthrough.
- `injuries`, `game_notes`, `scoringPlays`, `winprobability`, `leaders`, `header`, `gameInfo`, `standings`, `homeTeamId`, `awayTeamId`, `timeouts`, `broadcasts`, `videos`.
- **Dropped (not present):** `officials`, betting `propbets` is always `[]`.

---

## File Structure

```
cfbfastR-cfb-data/
  DESCRIPTION                       # data-repo metadata + Imports + Remotes
  R/
    _data_utils.R                   # shared: enumerate games, fetch final JSON, bind, write, pb_upload-both, manifest
    espn_cfb_01_pbp_creation.R      # final.plays -> .pbp_apply_output_schema -> play_by_play_{y}
    espn_cfb_02_team_box_creation.R # final.boxScore (ESPN team box)
    espn_cfb_03_player_box_creation.R # final.boxScore (ESPN player box)
    espn_cfb_04_adv_box_creation.R  # final.advBoxScore -> adv_{team,passing,rushing,receiving,defensive,turnover,drives,situational}
    espn_cfb_05_play_participants_creation.R
    espn_cfb_06_drives_creation.R
    espn_cfb_07_rosters_creation.R
    espn_cfb_08_betting_creation.R
    espn_cfb_09_schedules_creation.R
    espn_cfb_10_linescores_creation.R   # from team_box_extra (recent)
    espn_cfb_11_power_index_creation.R   # FPI (recent)
    espn_cfb_13_injuries_creation.R
    releases_init.R                 # create release tags on both repos (idempotent)
    run_summary.R                   # $GITHUB_STEP_SUMMARY
  scripts/daily_cfb_R_processor.sh
  cfb/                              # committed parquet/rds/csv per dataset per season + cfb_{ds}_in_data_repo.csv
  logs/
  tests/testthat/                  # testthat unit tests on the pure reshape fns (offline, fixture-driven)
  tests/testthat/fixtures/final_<gid>.json
  .github/workflows/daily_cfb.yml
  CLAUDE.md  README.md  .gitignore
```

## Conventions

- Each creation script is callable as `Rscript R/espn_cfb_0N_*.R -s <start> -e <end>` (optparse) and exposes pure, testable reshape functions in `_data_utils.R` or its own file so testthat can exercise them on a fixture without network.
- Tests: `testthat`, offline by default (fixture JSON); live tests gated by `Sys.getenv("CFB_LIVE_TESTS") == "1"` via `testthat::skip_if_not`.
- Release tags: **`espn_cfb_*`** namespace (matches the hoopR/wehoop sibling convention). PBP = **`espn_cfb_pbp`**; others = `espn_cfb_{dataset}` (e.g. `espn_cfb_team_box`, `espn_cfb_player_box`, `espn_cfb_adv_team_box`, …). Per-season file stem `{dataset}_{year}` (PBP stem `play_by_play_{year}`).
- **Cutover note (load_cfb_pbp):** `cfbfastR::load_cfb_pbp()` currently reads the *legacy* `cfbfastR_cfb_pbp` release (2014–2022, old pipeline). The new pipeline publishes to `espn_cfb_pbp` **for now**, leaving the legacy tag untouched while the new data is validated. A later cutover (out of v1 scope) either repoints `load_cfb_pbp()` at `espn_cfb_pbp` or promotes the `espn_cfb_pbp` assets into `cfbfastR_cfb_pbp`.
- "Both" publish: commit `cfb/**` in-repo AND `pb_upload` to `sportsdataverse/cfbfastR-cfb-data` + `sportsdataverse/sportsdataverse-data`.
- Column-drift resilience: bind with `data.table::rbindlist(fill = TRUE)`; select with `dplyr::any_of()`.
- Commit-message contract (consumed by nothing downstream of `-data`, but keep tidy): `"CFB Data Updated (Start: $i End: $i)"`.
- Conventional Commits; NEVER AI co-author trailers.

---

# Phase F — scaffold + shared utils

### Task F1: repo scaffold + DESCRIPTION + gitignore

**Files:** Create `DESCRIPTION`, `.gitignore`, `tests/testthat.R`, `R/.gitkeep`.

- [ ] **Step 1: `git init` the repo locally**

Run:
```bash
mkdir -p /c/Users/saiem/Documents/GitHub-Data/sdv-dev/cfbfastR-dev/cfbfastR-cfb-data
cd /c/Users/saiem/Documents/GitHub-Data/sdv-dev/cfbfastR-dev/cfbfastR-cfb-data
git init -q && git branch -M main
git config user.name "saiemgilani" && git config user.email "saiem.gilani@gmail.com"
```
Expected: empty repo on `main`.

- [ ] **Step 2: Create `DESCRIPTION`**

```
Package: cfbfastR.cfb.data
Title: College Football Data (cfbfastR data repository)
Version: 0.0.1
Authors@R: person("Saiem", "Gilani", email = "saiem.gilani@gmail.com", role = c("aut", "cre"))
License: CC BY 4.0
Encoding: UTF-8
Depends:
    R (>= 4.1.0)
Imports:
    arrow (>= 14.0.0),
    cli,
    curl,
    data.table (>= 1.14.0),
    dplyr (>= 1.1.0),
    glue,
    jsonlite (>= 1.8.0),
    optparse,
    piggyback (>= 0.1.5),
    purrr (>= 1.0.0),
    readr,
    rlang
Suggests:
    testthat (>= 3.0.0)
Remotes:
    sportsdataverse/cfbfastR
Config/testthat/edition: 3
```

- [ ] **Step 3: Create `.gitignore`**

```
.Rproj.user
.Rhistory
.RData
.Ruserdata
*.Rproj
.DS_Store
.vscode/
.claude/
/tmp/
```

- [ ] **Step 4: Create `tests/testthat.R`**

```r
library(testthat)
testthat::test_dir("tests/testthat")
```

- [ ] **Step 5: Commit**

```bash
git add DESCRIPTION .gitignore tests/testthat.R
git commit -m "chore: scaffold cfbfastR-cfb-data (DESCRIPTION, gitignore, testthat)"
```

---

### Task F2: capture a fixture `final` JSON for offline tests

**Files:** `tests/testthat/fixtures/final_<gid>.json`

- [ ] **Step 1: Generate one enriched final JSON from the -raw scraper**

The `-raw` repo (sibling, `../cfbfastR-cfb-raw`) has the working scraper. Produce one game's `final` JSON and copy it as a fixture:
```bash
cd ../cfbfastR-cfb-raw
uv run python -c "import sys; sys.path.insert(0,'python'); import scrape_cfb_json as sj; sj.download_game(401628455, season=2024, rescrape=True)"
mkdir -p ../cfbfastR-cfb-data/tests/testthat/fixtures
cp cfb/json/final/401628455.json ../cfbfastR-cfb-data/tests/testthat/fixtures/final_401628455.json
cd ../cfbfastR-cfb-data
```
Expected: `tests/testthat/fixtures/final_401628455.json` exists (a real enriched game). If 401628455 fails live, pick any completed FBS game id and update the fixture name + the `GID` constant used in tests.

- [ ] **Step 2: Sanity-check the fixture has the expected blocks**

Run:
```bash
Rscript -e 'j <- jsonlite::fromJSON("tests/testthat/fixtures/final_401628455.json", simplifyVector = FALSE); cat(sort(names(j)), sep="\n")'
```
Expected: prints keys incl. `plays`, `advBoxScore`, `boxScore`, `betting`, `play_participants`, `drives`, `game_rosters`, `team_box_extra`, `injuries`, `power_index`, `id`, `season`, `week`.

- [ ] **Step 3: Commit the fixture**

```bash
git add tests/testthat/fixtures/final_401628455.json
git commit -m "test: add enriched final JSON fixture for offline reshape tests"
```

---

### Task F3: `_data_utils.R` — fetch, enumerate, write, dual-upload

**Files:** Create `R/_data_utils.R`; Test `tests/testthat/test-data-utils.R`

- [ ] **Step 1: Write failing tests**

`tests/testthat/test-data-utils.R`:
```r
source(file.path("..", "..", "R", "_data_utils.R"))

FIX <- testthat::test_path("fixtures", "final_401628455.json")

test_that("read_final_json parses a game and returns a named list", {
  g <- read_final_json(FIX)
  expect_true(is.list(g))
  expect_true(all(c("plays", "advBoxScore", "betting", "id", "season") %in% names(g)))
})

test_that("write_dataset writes parquet + rds + csv and a manifest row", {
  tmp <- withr::local_tempdir()
  withr::local_dir(tmp)
  df <- data.frame(game_id = c(1L, 2L), x = c(10, 20))
  write_dataset(df, dataset = "demo", season = 2024, stem = "demo")
  expect_true(file.exists("cfb/demo/parquet/demo_2024.parquet"))
  expect_true(file.exists("cfb/demo/rds/demo_2024.rds"))
  expect_true(file.exists("cfb/demo/csv/demo_2024.csv.gz"))
  m <- readr::read_csv("cfb/demo/cfb_demo_in_data_repo.csv", show_col_types = FALSE)
  expect_equal(m$season, 2024)
  expect_equal(m$row_count, 2)
})

test_that("season_game_ids filters the master to a season", {
  tmp <- withr::local_tempdir()
  master <- file.path(tmp, "m.parquet")
  arrow::write_parquet(
    data.frame(game_id = c(1L, 2L, 3L), season = c(2023L, 2024L, 2024L)), master)
  ids <- season_game_ids_from_master(master, 2024)
  expect_setequal(ids, c(2L, 3L))
})
```
(If `withr` isn't installed, add it to Suggests + install; or use `tempfile()`/`on.exit` instead.)

- [ ] **Step 2: Run, confirm FAIL**

Run: `Rscript -e 'testthat::test_local()'` (or `R CMD INSTALL` deps first). Expected: FAIL — functions undefined.

- [ ] **Step 3: Implement `R/_data_utils.R`**

```r
# Shared helpers for cfbfastR-cfb-data creation scripts.
# Pure-ish reshape + IO helpers; network isolated to fetch_* so reshape fns stay testable.

RAW_BASE <- "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-cfb-raw/main/cfb"
PUBLISH_REPOS <- c("sportsdataverse/cfbfastR-cfb-data", "sportsdataverse/sportsdataverse-data")

read_final_json <- function(path_or_url) {
  jsonlite::fromJSON(path_or_url, simplifyVector = FALSE)
}

# Download the -raw schedule master parquet to a temp file and return game_ids for a season.
season_game_ids_from_master <- function(master_path_or_url, season) {
  df <- arrow::read_parquet(master_path_or_url)
  ids <- df$game_id[df$season == season]
  unique(as.integer(ids[!is.na(ids)]))
}

fetch_master_local <- function() {
  dest <- tempfile(fileext = ".parquet")
  curl::curl_download(paste0(RAW_BASE, "/cfb_schedule_master.parquet"), dest, quiet = TRUE)
  dest
}

final_url <- function(game_id) sprintf("%s/json/final/%s.json", RAW_BASE, game_id)

# Fetch + parse one final JSON with retry/backoff; returns NULL on persistent failure (logged).
fetch_final <- function(game_id, tries = 3L) {
  for (i in seq_len(tries)) {
    out <- tryCatch(read_final_json(final_url(game_id)), error = function(e) e)
    if (!inherits(out, "error")) return(out)
    Sys.sleep(min(2^(i - 1), 5))
  }
  cli::cli_alert_warning("fetch_final failed for {game_id}")
  NULL
}

# Bind a list of per-game data.frames (drift-safe).
bind_games <- function(frames) {
  frames <- Filter(function(x) !is.null(x) && nrow(x) > 0, frames)
  if (length(frames) == 0) return(data.frame())
  data.table::rbindlist(frames, fill = TRUE, use.names = TRUE) |> as.data.frame()
}

# Write parquet + rds + gzipped csv under cfb/{dataset}/ and append a manifest row.
write_dataset <- function(df, dataset, season, stem) {
  if (is.null(df) || nrow(df) == 0) {
    cli::cli_alert_info("{dataset} {season}: 0 rows, skipping write")
    return(invisible(NULL))
  }
  base <- file.path("cfb", dataset)
  for (sub in c("parquet", "rds", "csv")) dir.create(file.path(base, sub), recursive = TRUE, showWarnings = FALSE)
  arrow::write_parquet(df, file.path(base, "parquet", sprintf("%s_%d.parquet", stem, season)))
  saveRDS(df, file.path(base, "rds", sprintf("%s_%d.rds", stem, season)))
  readr::write_csv(df, file.path(base, "csv", sprintf("%s_%d.csv.gz", stem, season)))
  .append_manifest(dataset, season, nrow(df))
  invisible(df)
}

.append_manifest <- function(dataset, season, row_count) {
  f <- file.path("cfb", dataset, sprintf("cfb_%s_in_data_repo.csv", dataset))
  row <- data.frame(season = as.integer(season), row_count = as.integer(row_count),
                    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE))
  if (file.exists(f)) {
    old <- readr::read_csv(f, show_col_types = FALSE)
    row <- dplyr::bind_rows(old[old$season != season, , drop = FALSE], row)
  }
  row <- row[order(row$season), , drop = FALSE]
  readr::write_csv(row, f)
}

# Upload one file to BOTH publish repos under a release tag (idempotent overwrite).
pb_upload_both <- function(file, tag, repos = PUBLISH_REPOS, token = Sys.getenv("GITHUB_PAT")) {
  for (repo in repos) {
    tryCatch(
      piggyback::pb_upload(file = file, repo = repo, tag = tag, overwrite = TRUE, .token = token),
      error = function(e) cli::cli_alert_danger("pb_upload {repo}@{tag} {basename(file)}: {conditionMessage(e)}")
    )
  }
}

# Publish all three formats for a dataset+season to both repos.
publish_dataset <- function(dataset, season, stem, tag) {
  base <- file.path("cfb", dataset)
  for (ext in c("parquet", "rds", "csv")) {
    sub <- if (ext == "csv") "csv" else ext
    fn  <- if (ext == "csv") sprintf("%s_%d.csv.gz", stem, season) else sprintf("%s_%d.%s", stem, season, ext)
    f <- file.path(base, sub, fn)
    if (file.exists(f)) pb_upload_both(f, tag)
  }
}

# Generic per-season driver: enumerate -> fetch -> reshape(each) -> bind -> write -> publish.
build_season <- function(season, dataset, stem, tag, reshape_fn,
                         master = fetch_master_local(), live = TRUE) {
  ids <- season_game_ids_from_master(master, season)
  cli::cli_alert_info("{dataset} {season}: {length(ids)} games")
  frames <- lapply(ids, function(gid) {
    g <- if (live) fetch_final(gid) else NULL
    if (is.null(g)) return(NULL)
    tryCatch(reshape_fn(g), error = function(e) { cli::cli_alert_warning("{dataset} {gid}: {conditionMessage(e)}"); NULL })
  })
  df <- bind_games(frames)
  write_dataset(df, dataset, season, stem)
  if (live && nrow(df) > 0) publish_dataset(dataset, season, stem, tag)
  invisible(df)
}
```

- [ ] **Step 4: Install deps + run tests**

Run:
```bash
Rscript -e 'install.packages(c("arrow","jsonlite","dplyr","data.table","purrr","readr","cli","glue","curl","piggyback","optparse","withr","testthat","rlang"), repos="https://cloud.r-project.org")' 2>&1 | tail -3
Rscript -e 'testthat::test_local()'
```
Expected: the 3 `_data_utils` tests pass. (Network helpers `fetch_*` are not exercised here.)

- [ ] **Step 5: Commit**

```bash
git add R/_data_utils.R tests/testthat/test-data-utils.R
git commit -m "feat(utils): final-JSON fetch, season enumeration, dual-repo publish, dataset writer"
```

---

# Phase G — per-game dataset creation scripts

Each task adds one creation script + a reshape function tested against the fixture. The reshape function takes a parsed `final` list and returns a data.frame (one or more rows). Scripts share the `build_season()` driver from F3.

### Task G1: PBP (`espn_cfb_01_pbp_creation.R`) — conform to cfbfastR schema

**Files:** Create `R/espn_cfb_01_pbp_creation.R`; Test `tests/testthat/test-pbp.R`

- [ ] **Step 1: Failing test**

`tests/testthat/test-pbp.R`:
```r
source(file.path("..", "..", "R", "_data_utils.R"))
source(file.path("..", "..", "R", "espn_cfb_01_pbp_creation.R"))
GID <- 401628455
g <- read_final_json(testthat::test_path("fixtures", sprintf("final_%d.json", GID)))

test_that("reshape_pbp returns one row per play with key identifiers", {
  df <- reshape_pbp(g)
  expect_s3_class(df, "data.frame")
  expect_gt(nrow(df), 50)             # a real game has many plays
  expect_equal(nrow(df), length(g$plays))
  expect_true(all(c("game_id", "season") %in% names(df)))
  expect_true(all(df$game_id == g$id))
})
```

- [ ] **Step 2: Run, confirm FAIL** — `Rscript -e 'testthat::test_local()'` → `reshape_pbp` undefined.

- [ ] **Step 3: Implement `R/espn_cfb_01_pbp_creation.R`**

```r
suppressPackageStartupMessages({
  library(dplyr); library(purrr); library(data.table); library(arrow)
  library(jsonlite); library(glue); library(optparse); library(cli)
})
if (!exists("read_final_json")) source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_data_utils.R"))

# plays is a list-of-dicts; flatten each to a 1-row frame, bind, stamp identity.
reshape_pbp <- function(g) {
  plays <- g$plays
  if (is.null(plays) || length(plays) == 0) return(data.frame())
  df <- data.table::rbindlist(
    lapply(plays, function(p) as.data.frame(lapply(p, function(v) if (length(v) == 1) v else I(list(v))),
                                            stringsAsFactors = FALSE)),
    fill = TRUE, use.names = TRUE) |> as.data.frame()
  df$game_id <- as.integer(g$id)
  df$season  <- as.integer(g$season)
  if (!is.null(g$week)) df$week <- as.integer(g$week)
  df
}

# Apply cfbfastR's canonical column ordering/tiering if available (drift-safe otherwise).
conform_pbp <- function(df, output = "default") {
  if (nrow(df) == 0) return(df)
  fn <- tryCatch(getFromNamespace(".pbp_apply_output_schema", "cfbfastR"), error = function(e) NULL)
  if (!is.null(fn)) return(fn(df, output = output))
  df  # cfbfastR not installed (tests) — return as-is; CI has cfbfastR
}

build_pbp_season <- function(season, master = fetch_master_local(), live = TRUE) {
  build_season(season, dataset = "pbp", stem = "play_by_play", tag = "espn_cfb_pbp",
               reshape_fn = function(g) conform_pbp(reshape_pbp(g)), master = master, live = live)
}

if (sys.nframe() == 0 || identical(environment(), globalenv())) {
  if (!interactive() && length(commandArgs(trailingOnly = TRUE)) > 0) {
    opt <- optparse::parse_args(optparse::OptionParser(option_list = list(
      optparse::make_option(c("-s", "--start_year"), type = "integer"),
      optparse::make_option(c("-e", "--end_year"), type = "integer"))))
    master <- fetch_master_local()
    for (y in opt$start_year:opt$end_year) build_pbp_season(y, master = master, live = TRUE)
  }
}
```
> **Verify during execution:** confirm `.pbp_apply_output_schema` is exported-or-internal in the installed cfbfastR (the explore found it in `R/pbp_output_schema.R`). If its name/signature differs, adjust `conform_pbp`. The reshape test does NOT require cfbfastR (conform is a no-op without it), so it passes offline.

- [ ] **Step 4: Run, confirm PASS** — `Rscript -e 'testthat::test_local()'` → pbp test passes.

- [ ] **Step 5: Commit**

```bash
git add R/espn_cfb_01_pbp_creation.R tests/testthat/test-pbp.R
git commit -m "feat(pbp): reshape final.plays -> play_by_play, conform to cfbfastR schema, tag espn_cfb_pbp"
```

### Task G2: ESPN team_box + player_box (`02`, `03`)

**Files:** `R/espn_cfb_02_team_box_creation.R`, `R/espn_cfb_03_player_box_creation.R`; Test `tests/testthat/test-boxscore.R`

- [ ] **Step 1: Failing test** — `tests/testthat/test-boxscore.R`:
```r
source(file.path("..", "..", "R", "_data_utils.R"))
source(file.path("..", "..", "R", "espn_cfb_02_team_box_creation.R"))
source(file.path("..", "..", "R", "espn_cfb_03_player_box_creation.R"))
GID <- 401628455
g <- read_final_json(testthat::test_path("fixtures", sprintf("final_%d.json", GID)))

test_that("reshape_team_box returns two team rows stamped with game_id/season", {
  df <- reshape_team_box(g)
  expect_s3_class(df, "data.frame")
  if (nrow(df) > 0) {
    expect_lte(nrow(df), 2)
    expect_true(all(df$game_id == g$id))
  }
})
test_that("reshape_player_box returns rows stamped with game_id", {
  df <- reshape_player_box(g)
  expect_s3_class(df, "data.frame")
  if (nrow(df) > 0) expect_true(all(df$game_id == g$id))
})
```

- [ ] **Step 2: Run, confirm FAIL.**

- [ ] **Step 3: Implement.** `R/espn_cfb_02_team_box_creation.R`:
```r
suppressPackageStartupMessages({ library(dplyr); library(data.table); library(arrow); library(jsonlite); library(optparse); library(cli) })
if (!exists("read_final_json")) source(file.path(dirname(sys.frame(1)$ofile %||% "."), "_data_utils.R"))

# ESPN team box from final$boxScore$teams[]: each has $team + $statistics (list of {name,displayValue}).
reshape_team_box <- function(g) {
  teams <- g$boxScore$teams
  if (is.null(teams) || length(teams) == 0) return(data.frame())
  rows <- lapply(teams, function(t) {
    stats <- t$statistics %||% list()
    kv <- setNames(
      lapply(stats, function(s) s$displayValue %||% s$value %||% NA),
      vapply(stats, function(s) s$name %||% s$label %||% "stat", character(1)))
    df <- as.data.frame(kv, stringsAsFactors = FALSE, check.names = TRUE)
    df$team_id <- as.integer(t$team$id %||% NA)
    df$home_away <- t$homeAway %||% NA_character_
    df
  })
  out <- data.table::rbindlist(rows, fill = TRUE) |> as.data.frame()
  out$game_id <- as.integer(g$id); out$season <- as.integer(g$season)
  out
}
build_team_box_season <- function(season, master = fetch_master_local(), live = TRUE)
  build_season(season, "team_box", "team_box", "espn_cfb_team_box", reshape_team_box, master, live)
# (argparse main block identical to G1; calls build_team_box_season)
```
`R/espn_cfb_03_player_box_creation.R` — same shape, `reshape_player_box(g)` reads `g$boxScore$players[]` (per-team `statistics` groups → athletes); stem/tag `player_box`/`espn_cfb_player_box`.
> **Verify during execution:** inspect the fixture's `boxScore$teams[[1]]$statistics` and `boxScore$players` shapes (`Rscript -e 'str(jsonlite::fromJSON(...)$boxScore, max.level=3)'`) and adjust the field paths; ESPN box nests players as `players[[i]]$statistics[[j]]$athletes[[k]]`.

- [ ] **Step 4: Run, confirm PASS.**
- [ ] **Step 5: Commit** — `feat(box): ESPN team_box + player_box reshape from final.boxScore`.

### Task G3: play_participants, drives, rosters, injuries (`05`, `06`, `07`, `13`)

**Files:** the four scripts + `tests/testthat/test-simple-blocks.R`

These are direct list-to-frame reshapes of self-describing blocks. Pattern per dataset:
```r
reshape_<ds> <- function(g) {
  x <- g[["<key>"]]
  if (is.null(x) || length(x) == 0) return(data.frame())
  df <- data.table::rbindlist(lapply(x, as.data.frame), fill = TRUE) |> as.data.frame()
  df$game_id <- as.integer(g$id); df$season <- as.integer(g$season); df
}
```
- [ ] **Step 1: Failing test** asserting each returns a data.frame stamped with `game_id` (rows ≥ 0):
```r
source(file.path("..","..","R","_data_utils.R"))
for (f in c("05_play_participants","06_drives","07_rosters","13_injuries"))
  source(file.path("..","..","R", sprintf("espn_cfb_%s_creation.R", f)))
g <- read_final_json(testthat::test_path("fixtures","final_401628455.json"))
test_that("block reshapes are stamped data.frames", {
  for (fn in list(reshape_play_participants, reshape_drives, reshape_rosters, reshape_injuries)) {
    df <- fn(g); expect_s3_class(df, "data.frame")
    if (nrow(df) > 0) expect_true(all(df$game_id == g$id))
  }
})
```
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** the four scripts using the pattern above: `05` key `play_participants` (tag `espn_cfb_play_participants`, stem `play_participants`); `06` key `drives` (tag `espn_cfb_drives`, stem `drives`) — **note** `drives` is a nested ESPN object keyed by team, so `reshape_drives` must `purrr::map_dfr` over `g$drives` unrolling `$plays` into drive rows (verify shape on the fixture; if `drives` is `{}` for the game, return empty); `07` key `game_rosters` (tag `espn_cfb_rosters`, stem `rosters`); `13` key `injuries` (tag `espn_cfb_injuries`, stem `injuries`). Each gets a `build_<ds>_season()` + argparse main calling `build_season(...)`.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat(blocks): play_participants/drives/rosters/injuries reshapes`.

### Task G4: betting, schedules, linescores, power_index (`08`, `09`, `10`, `11`)

**Files:** four scripts + `tests/testthat/test-meta-blocks.R`

- [ ] **Step 1: Failing test:**
```r
source(file.path("..","..","R","_data_utils.R"))
for (f in c("08_betting","09_schedules","10_linescores","11_power_index"))
  source(file.path("..","..","R", sprintf("espn_cfb_%s_creation.R", f)))
g <- read_final_json(testthat::test_path("fixtures","final_401628455.json"))
test_that("betting is one self-describing row", {
  df <- reshape_betting(g)
  expect_equal(nrow(df), 1L)
  expect_true(all(c("game_id","game_spread","over_under","odds_source") %in% names(df)))
})
test_that("schedules row carries game meta", {
  df <- reshape_schedule_row(g)
  expect_equal(nrow(df), 1L)
  expect_true(all(c("game_id","season","home_id","away_id") %in% names(df)))
})
test_that("linescores + power_index degrade to empty when absent", {
  expect_s3_class(reshape_linescores(g), "data.frame")
  expect_s3_class(reshape_power_index(g), "data.frame")
})
```
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement:**
  - `08_betting` — `reshape_betting(g)`: take scalar fields of `g$betting` (`game_spread, over_under, home_favorite, home_team_spread, game_spread_available, odds_source`) into a 1-row df + `game_id/season/week`; drop the list payloads (`odds`, `odds_full`, `pickcenter`, `predictor`, `against_the_spread`, `propbets`) for the rectangular release (keep them only in JSON). Tag `espn_cfb_betting`.
  - `09_schedules` — `reshape_schedule_row(g)`: pull `home_id/away_id/home_team/away_team/season/week/season_type/game_date/venue` from `g$header`/`g$gameInfo` into a 1-row game-meta frame; bound across games = the season schedule. Tag `espn_cfb_schedules`, stem `cfb_schedule`. (This also produces the per-season schedule the repo serves.)
  - `10_linescores` — `reshape_linescores(g)`: from `g$team_box_extra[[team_id]]$linescores`, long-form (team_id, period, value); empty if `team_box_extra` absent (older seasons). Tag `espn_cfb_linescores`.
  - `11_power_index` — `reshape_power_index(g)`: flatten `g$power_index` (FPI) to a per-team or per-game row; empty `{}` → empty df (recent-only). Tag `espn_cfb_power_index`.
  Each gets `build_<ds>_season()` + argparse main.
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat(meta): betting/schedules/linescores/power_index reshapes`.

---

# Phase H — advBoxScore (R rectangularization + sdv-py expansion)

### Task H1 (sdv-py): expand `create_box_score()` with missing player/specialist sections

**Files:** Modify `sportsdataverse/cfb/cfb_pbp.py` (`create_box_score`); Test `tests/cfb/test_create_box_score.py`. (Branch off `main`: `feat/cfb-advbox-expansion`.)

**Context:** `create_box_score` returns 8 sections (`pass, rush, receiver, team, situational, defensive, turnover, drives`). The §12.1 gaps: no **defensive player-level** stats, no **specialist (kicker/punter/returner)** player stats. Add two player-level sections derivable from existing play columns (`sack_player_name`, `interception_player_name`, `pass_breakup_player_name`, `fumble_player_name`; `fg_kicker_player_name`, `punter_player_name`, `kickoff_returner_player_name`, `punt_returner_player_name`, and their yardage cols).

- [ ] **Step 1: Capture a fixture + write failing test** (`tests/cfb/test_create_box_score.py`): build a `CFBPlayProcess` from a saved raw fixture, run `run_processing_pipeline()`, assert `result["advBoxScore"]` now has keys `"defensive_players"` and `"specialists"`, each a non-empty list of dicts with `pos_team`/`def_pos_team` + player name + the expected stat fields. (Use a fixture game known to have a sack + a field goal.)
- [ ] **Step 2: Run, confirm FAIL** (`python -m pytest tests/cfb/test_create_box_score.py -v`).
- [ ] **Step 3: Implement** two new grouped aggregations in `create_box_score` (mirroring the existing polars `group_by` + `agg` pattern used for `pass`/`rush`): `defensive_players` grouped by `["def_pos_team", "<defender>_player_name"]` summing TFL/sack/INT/PBU/forced-fumble counts + EPA; `specialists` grouped by kicker/punter/returner name with FG made/att, punt count + yards, return yards. Add both to the returned dict.
  > **Verify during execution:** confirm the exact enriched play-column names against a real processed game (`[c for c in play_df.columns if 'player_name' in c or 'yds_' in c]`); the earlier exploration confirmed `sack_player_name`, `interception_player_name`, `fg_kicker_player_name`, `punter_player_name`, `yds_*` exist. Keep it bounded to what's derivable; do NOT invent new tracking data.
- [ ] **Step 4: Run, confirm PASS.** Also run `python -m pytest tests/cfb -q` to ensure no regression in existing box-score behavior.
- [ ] **Step 5: Commit + PR** — `feat(cfb): add defensive_players + specialists sections to create_box_score`; bump CHANGELOG (new `## 0.0.53 (unreleased)` section), open PR. (This ships in a later sdv-py release; `-data` Phase H2 tolerates absence via `any_of`.)

### Task H2 (R): rectangularize all advBoxScore sections

**Files:** `R/espn_cfb_04_adv_box_creation.R`; Test `tests/testthat/test-advbox.R`

- [ ] **Step 1: Failing test:**
```r
source(file.path("..","..","R","_data_utils.R"))
source(file.path("..","..","R","espn_cfb_04_adv_box_creation.R"))
g <- read_final_json(testthat::test_path("fixtures","final_401628455.json"))
test_that("adv box sections rectangularize to stamped frames", {
  out <- reshape_adv_box(g)        # named list of data.frames
  expect_true(is.list(out))
  for (nm in c("adv_team","adv_passing","adv_rushing","adv_receiving",
               "adv_defensive","adv_turnover","adv_drives","adv_situational")) {
    expect_true(nm %in% names(out))
    expect_s3_class(out[[nm]], "data.frame")
    if (nrow(out[[nm]]) > 0) expect_true(all(out[[nm]]$game_id == g$id))
  }
})
```
- [ ] **Step 2: FAIL.**
- [ ] **Step 3: Implement** `reshape_adv_box(g)` returning a named list, one frame per `g$advBoxScore` section: `team`→`adv_team`, `pass`→`adv_passing`, `rush`→`adv_rushing`, `receiver`→`adv_receiving`, `defensive`→`adv_defensive`, `turnover`→`adv_turnover`, `drives`→`adv_drives`, `situational`→`adv_situational`. Each: `data.table::rbindlist(lapply(section, as.data.frame), fill=TRUE)` then stamp `game_id`/`season`. Tolerate missing/empty sections (`any_of`); include `defensive_players`/`specialists` when present (Phase H1) via `any_of`. The `build_adv_box_season()` writes **each** section as its own dataset/tag (`espn_cfb_adv_team`, `espn_cfb_adv_passing`, …) — extend `build_season` to accept a multi-frame reshape (loop the named list, write+publish each with `stem = nm`, `tag = sprintf("espn_cfb_%s", nm)`).
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** — `feat(advbox): rectangularize all advBoxScore sections into per-section release datasets`.

---

# Phase I — orchestration + CI + docs + remote

### Task I1: `releases_init.R` + `run_summary.R`

- [ ] Implement `R/releases_init.R` — idempotently `piggyback::pb_release_create()` each tag (`espn_cfb_pbp`, `_team_box`, `_player_box`, `_adv_*`, `_play_participants`, `_drives`, `_rosters`, `_betting`, `_schedules`, `_linescores`, `_power_index`, `_injuries`) on BOTH publish repos (wrap each in tryCatch; "already exists" is fine). Run once manually before the first data run.
- [ ] Implement `R/run_summary.R` — read each `cfb/*/cfb_*_in_data_repo.csv`, print a per-dataset season×row_count table to stdout and append a markdown table to `$GITHUB_STEP_SUMMARY` (if set). optparse `-s/-e`.
- [ ] Commit — `feat(release): release-tag init + run summary`.

### Task I2: `scripts/daily_cfb_R_processor.sh`

- [ ] Implement the season-loop + temp-log-then-commit git idiom (mirror Plan 1's `daily_cfb_scraper.sh`), running the creation scripts in order (`01`→`13`, skipping `12`) per season with `Rscript R/espn_cfb_0N_*.R -s $i -e $i`, each non-fatal (`|| { rc=$?; echo "::warning ::..."; SEASON_RC=$rc; }`), exit-code propagated; commit `"CFB Data Updated (Start: $i End: $i)"`; then `Rscript R/run_summary.R`. Add `END_YEAR=${END_YEAR:-$START_YEAR}` default (Plan 1 I3 lesson). `bash -n` check. Commit.

### Task I3: `.github/workflows/daily_cfb.yml`

- [ ] Implement the workflow: triggers `repository_dispatch: [daily_cfb_data]` + `schedule` (CFB calendar, offset after `-raw`) + `workflow_dispatch` (`start_year`/`end_year`); `r-lib/actions/setup-r` + `setup-r-dependencies` (extra-packages: `sportsdataverse/cfbfastR`, `ropensci/piggyback`, local `.`); extract year range from `github.event.client_payload.commit_message` via `grep -oP 'Start:\s*\K[0-9]{4}'` / `'End:\s*\K[0-9]{4}'` with `most_recent_cfb_season()` fallback; run `bash scripts/daily_cfb_R_processor.sh`. `GITHUB_PAT: ${{ secrets.SDV_GH_TOKEN }}` (needs cross-repo write to publish to `sportsdataverse-data`). YAML-lint. Commit.

### Task I4: README + CLAUDE.md

- [ ] README: what it produces (per-dataset tables + release tags incl. `espn_cfb_pbp` consumed by `cfbfastR::load_cfb_pbp()`), the reshape-not-re-enrich architecture, the dual-publish model, how `repository_dispatch` from `-raw` drives it. CLAUDE.md: conventions (reshape fns pure + testable on fixture, `any_of` drift-safety, tags, dual `pb_upload`, no AI co-authors, commit-message format). Commit.

### Task I5: create remote + push + wire the trigger

- [ ] **Step 1:** `gh repo create sportsdataverse/cfbfastR-cfb-data --public --source=. --remote=origin --push --description "..."`.
- [ ] **Step 2:** Run `Rscript R/releases_init.R` once (creates the release tags on both repos) — requires `GITHUB_PAT`/`SDV_GH_TOKEN` with write to `sportsdataverse-data`.
- [ ] **Step 3:** Verify the `-raw` → `-data` link end-to-end: push to `-raw` (or `workflow_dispatch`) and confirm `cfbfastR_cfb_data_trigger.yml` now dispatches successfully (the target repo exists) and `daily_cfb.yml` starts. Confirm `SDV_GH_TOKEN` org-secret is inherited by `cfbfastR-cfb-data` (same as `-raw`).
- [ ] **Step 4:** Smoke test: `workflow_dispatch` `daily_cfb.yml` for `-s 2024 -e 2024`; confirm parquet committed + a release asset uploaded to the `espn_cfb_pbp` tag on **both** repos (`piggyback::pb_list(repo, tag = "espn_cfb_pbp")`). Note: `cfbfastR::load_cfb_pbp()` reads the legacy `cfbfastR_cfb_pbp` tag, so it will NOT pick up `espn_cfb_pbp` yet — that's the deferred cutover (§73 cutover note). Verify the new asset loads directly: `arrow::read_parquet(piggyback URL for espn_cfb_pbp/play_by_play_2024.parquet)`.

---

## Self-review notes (author)

- **Spec coverage:** §5.2 datasets → Phases G+H (officials/propbets dropped per §12.8; rankings/recruiting/QBR deferred per this session's v1-scope decision). §6.4 HTTP linkage + master enumeration → F3 `fetch_final`/`season_game_ids_from_master`. §10.3 workflow/trigger → I3. Release "both" → F3 `pb_upload_both` + committed `cfb/`. §12.1 advBox expansion → H1+H2.
- **Reshape-not-re-enrich** is the load-bearing simplification: PBP conforms to cfbfastR's real `.pbp_apply_output_schema` (verified in `R/pbp_output_schema.R`); other datasets are block reshapes. No EPA/WPA recompute in R.
- **Release tags use the `espn_cfb_*` namespace** ("publish to `espn_cfb_pbp` for now" — this session). `cfbfastR::load_cfb_pbp()` reads the *legacy* `cfbfastR_cfb_pbp` tag (verified in `R/load_cfb_pbp.R`), so the new pipeline's `espn_cfb_pbp` output is intentionally separate until the deferred cutover (§73 cutover note) — legacy data stays untouched during rollout. Box/other loaders don't exist yet, so their `espn_cfb_*` tags are new.
- **Verification points flagged inline** (not placeholders): ESPN `boxScore` player nesting (G2), `drives` nesting (G3), enriched play-column names for the advBox expansion (H1), and `.pbp_apply_output_schema` name/signature (G1) — each must be confirmed against the fixture/installed package during execution, with a concrete fallback.
- **Offline tests** rely on one committed fixture (`final_401628455.json`); reshape fns are pure (no network), so the suite runs without GH/ESPN. cfbfastR is optional for the reshape tests (`conform_pbp` no-ops without it); CI installs it.
- **Phase H1 ships in a later sdv-py release** (0.0.53); `-data` tolerates its absence (`any_of`), so Plan 2 is not blocked on it.
```
