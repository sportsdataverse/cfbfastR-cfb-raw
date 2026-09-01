# ESPN CFB rosters — collection strategy, column union, gotchas

Recon for the `cfb_rosters` dataset (compiled in `cfbfastR-cfb-data` by
`cfbfastR-cfb-data`'s `python/cfb_data_build/rosters_espn.py`, published to the `espn_cfb_rosters` tag).

## 1. Collection strategy — and the request-count math

Four candidate paths to a season-wide ESPN roster were compared:

| # | Path | Requests for 2004–2025 | Verdict |
|---|---|---:|---|
| A | Site v2 `teams/{id}/roster?season=Y` | ~262 × 22 = **5,764** | **DEAD for history.** `?season=` is *accepted and echoed back* but every position bucket comes back empty. Verified on team 333 for 2003/2004/2010/2014/2019/2023/2025: `athletes[]` has all 6 buckets, `items` length 0 in every one. Only the *current* season returns athletes. This is why the released 2023–2025 assets exist and 2004–2022 do not. |
| B | Core v2 `seasons/{Y}/teams/{id}/athletes` | 1 list + ~110 athlete `$ref` per team → ~29k/season → **~640,000** | Rejected. The list payload is `$ref`-only (verified: 2014 team 333 → `count:150`, items are bare `$ref`s), so every athlete costs its own request. At the low concurrency Core v2 tolerates (it 403s under load) this is a multi-day job. |
| C | Core v2 `events/{e}/competitions/{c}/competitors/{t}/roster` (per game) | ~20,700 games × 2 teams | Already scraped. See below. |
| D | **Re-use the existing capture (chosen)** | **0 new scrape requests** | `cfbfastR-cfb-raw` has already run path C: `cfb/game_rosters/json/{game_id}.json`, **20,695 files / 6.2 GB**, seasons 2004–2026, each holding the FULL ESPN athlete record for every rostered player in that game. |

**Chosen: D.** The season roster is the union of a season's per-game rosters,
collapsed to one row per `(season, team_id, athlete_id)` with the *last*
appearance supplying the attribute values. New requests needed: **zero for the
rosters themselves**, plus the two small references below.

Two references complete the dataset (both owned by the `cfb_teams` pipeline,
consumed here — not re-captured):

| Reference | Path in `cfbfastR-cfb-raw` | Requests |
|---|---|---:|
| League position list (74) | `cfb/reference/positions.json` | 1 index + 74 `$ref` = **75**, once ever |
| Season team lists / division | `cfb/teams/json/{season}.json` (`divisions`: `80`=fbs, `81`=fcs) | 2 per season |

**Coverage of the existing capture** (checked against `cfb_schedule_master.parquet`,
19,586 game ids): 16 master ids have no roster capture (14 in 2020, 2 in 2021 —
COVID-era cancellations); 1,125 captured games are not in the master (mostly 2026
and non-master events) and are dropped, matching every other dataset in the repo,
which enumerates from the master. Effective coverage **99.92 %**. Gaps are filled by
re-running stage 06 (`cfb_raw_scrape/scrape_cfb_pbp.py`), which fetches rosters
per game and embeds them in `json/final` — **no new scraper was written**,
because the raw store this dataset needs was already complete.

> The standalone `scrape_cfb_game_rosters.py` this section originally named was
> deleted 2026-08-29: it duplicated the fetch that stage 06 already performs.
> The `cfb/game_rosters/` tree it wrote is still committed but is no longer
> updated; read the rosters out of `json/final` instead.

### Compile cost

Reading the raw over HTTP (`raw.githubusercontent.com`, the `RAW_BASE` contract):
measured **10.4 MB/s at 8 workers, 53 ms/game** → the whole 2004–2025 backfill is
~20,000 requests / 6.2 GB / **~18 minutes**. Compiling from `cfb/json/final/{id}.json`
(what every other dataset here reads) would be the same request count over **55 GB**
— a ~9× larger read for identical output. That is the whole reason this compiler
reads the standalone `game_rosters` block instead of going through `fetch_final`.

## 2. The full column union across seasons

2.96 M athlete-game rows scanned across 2004–2025 → **77 distinct keys**. Nine are
per-game circumstance and are dropped when collapsing to a season roster
(`game_id`, `week`, `starter`, `did_not_play`, `order`, `home_away`, `winner`,
`valid`, `statistics_href`); the remaining 68 are the ESPN union, pinned as
`ESPN_COLS` in the compiler. **No key is present in every season** — hence the
pinned constant: a season that never received a key gets it null-filled rather
than shifting the column set.

| First season present | Keys |
|---|---|
| 2004 (68 of 68 minus below) | the athlete block (`athlete_id`, `athlete_uid`, `athlete_guid`, `athlete_type`, `first_name`, `middle_name`, `last_name`, `full_name`, `athlete_display_name`, `short_name`, `slug`, `jersey`, `weight`, `display_weight`, `height`, `display_height`, `age`, `date_of_birth`, `linked`, `active`, `alternate_ids_sdr`, `birth_place_city/_state/_country`, `birth_country_alternate_id`, `birth_country_abbreviation`, `flag_href/_alt/_rel`, `headshot_href/_alt`, `experience_years/_display_value/_abbreviation`, `status_id/_name/_type/_abbreviation`, `athlete_href`, `position_href`) + the full `team_*` block + `draft_display_text/_round/_year/_selection` |
| **2005** | `hand_type`, `hand_abbreviation`, `hand_display_value`, `citizenship`, `nickname` |
| **2008** | `jersey_right`, `display_name` |
| sparse / partial | `draft_*` appears in only 12 of 22 seasons; `draft_team_href` in 11 (2004–2021 only); `nickname` in 16 (2005–2020) |

Output adds: `season`, `division`, `position_id`, `position`,
`position_abbreviation`, `position_name`, `position_leaf`, `position_parent_id`,
`games_rostered` → **78 columns, identical in every season.**

## 3. Position reference — all 74

Captured at `cfb/reference/positions.json`. `leaf = n` means it is a *grouping*
node (`Offense`/`Defense`/`Special Teams`/`Athlete`/`Unknown`/`Guard`/`Tackle`/
`Back`/`Setter`), not an assignable position. Duplicated abbreviations are real
(ESPN has four `LS`, three `PK`/`P`, two `C`, two `Unknown`) — **join on `id`,
never on `abbreviation`**.

| id | displayName | abbr | leaf | parent |
|---:|---|---|:--:|---:|
| 0 | Unknown | `-` | n |  |
| 1 | Wide Receiver | `WR` | y | 70 |
| 2 | Left Tackle | `LT` | y | 46 |
| 3 | Left Guard | `LG` | y | 47 |
| 4 | Center | `C` | y | 70 |
| 5 | Right Guard | `RG` | y | 46 |
| 6 | Right Tackle | `RT` | y | 47 |
| 7 | Tight End | `TE` | y | 70 |
| 8 | Quarterback | `QB` | y | 70 |
| 9 | Running Back | `RB` | y | 70 |
| 10 | Fullback | `FB` | y | 9 |
| 11 | Left Defensive End | `LDE` | y | 31 |
| 12 | Nose Tackle | `NT` | y | 32 |
| 13 | Right Defensive End | `RDE` | y | 31 |
| 14 | Left Outside Linebacker | `LOLB` | y | 30 |
| 15 | Left Inside Linebacker | `LILB` | y | 30 |
| 16 | Right Inside Linebacker | `RILB` | y | 30 |
| 17 | Right Outside Linebacker | `ROLB` | y | 30 |
| 18 | Left Cornerback | `LCB` | y | 29 |
| 19 | Right Cornerback | `RCB` | y | 29 |
| 20 | Strong Safety | `SS` | y | 36 |
| 21 | Free Safety | `FS` | y | 36 |
| 22 | Place Kicker | `PK` | y | 72 |
| 23 | Punter | `P` | y | 72 |
| 24 | Left Defensive Tackle | `LDT` | y | 32 |
| 25 | Right Defensive Tackle | `RDT` | y | 32 |
| 26 | Weakside Linebacker | `WLB` | y | 30 |
| 27 | Middle Linebacker | `MLB` | y | 30 |
| 28 | Strongside Linebacker | `SLB` | y | 30 |
| 29 | Cornerback | `CB` | y | 35 |
| 30 | Linebacker | `LB` | y | 71 |
| 31 | Defensive End | `DE` | y | 71 |
| 32 | Defensive Tackle | `DT` | y | 71 |
| 33 | Under Tackle | `UT` | y | 46 |
| 34 | Nickel Back | `NB` | y | 35 |
| 35 | Defensive Back | `DB` | y | 71 |
| 36 | Safety | `S` | y | 35 |
| 37 | Defensive Lineman | `DL` | y | 71 |
| 39 | Long Snapper | `LS` | y | 72 |
| 45 | Offensive Lineman | `OL` | y | 70 |
| 46 | Offensive Tackle | `OT` | y | 70 |
| 47 | Offensive Guard | `OG` | y | 70 |
| 50 | Athlete | `ATH` | n |  |
| 70 | Offense | `OFF` | n |  |
| 71 | Defense | `DEF` | n |  |
| 72 | Special Teams | `ST` | n |  |
| 73 | Guard | `G` | n |  |
| 74 | Tackle | `T` | n |  |
| 75 | Nose Guard | `NG` | y | 37 |
| 76 | Punt Returner | `PR` | y | 72 |
| 77 | Kick Returner | `KR` | y | 72 |
| 78 | Long Snapper | `LS` | y | 72 |
| 79 | Holder | `H` | y | 72 |
| 80 | Place Kicker | `PK` | y | 72 |
| 90 | Inside Linebacker | `ILB` | y | 30 |
| 91 | Center | `C` | y | 45 |
| 94 | Punter | `P` | y | 72 |
| 96 | Long Snapper | `LS` | y | 72 |
| 99 | Unknown | `-` | n |  |
| 100 | Flanker | `FL` | y | 70 |
| 101 | Halfback | `HB` | y | 70 |
| 102 | Tailback | `TB` | y | 70 |
| 103 | Left Halfback | `LHB` | y | 101 |
| 104 | Right Halfback | `RHB` | y | 101 |
| 105 | Left Linebacker | `LLB` | y | 30 |
| 106 | Right Linebacker | `RLB` | y | 30 |
| 107 | Outside Linebacker | `OLB` | y | 30 |
| 108 | Left Safety | `LSF` | y | 36 |
| 109 | Right Safety | `RSF` | y | 36 |
| 110 | Middle Guard | `MG` | y | 47 |
| 111 | Split End | `SE` | y | 71 |
| 218 | Setter | `SETTER` | n |  |
| 219 | Back | `B` | n |  |
| 264 | EDGE | `EDGE` | y | 71 |

Only 27 of the 74 ids are actually used by CFB roster rows. The most common are
`0` (Unknown, 17.4 % of all athlete-game rows), `45` OL, `1` WR, `30` LB, `35` DB,
`37` DL, `9` RB, `7` TE, `8` QB, `36` S, `29` CB, `31` DE.

## 4. Gotchas

1. **`athletes[]` is a list of POSITION BUCKETS, not athletes.** The Site v2 roster
   payload's `athletes` is `[{position: "offense"|"defense"|"specialTeam"|
   "injuredReserveOrOut"|"suspended"|"practiceSquad", items: [...]}]`. Flatten
   `athletes[].items[]`. (Not used by this pipeline — see gotcha 2 — but it is the
   trap anyone reaching for `espn_cfb_team_roster` hits first.)
2. **`?season=` on the Site v2 roster is a silent no-op.** It is echoed back in the
   response `season` field, so the payload *looks* season-scoped, while every
   `items` array is empty. There is no failure signal. Historical rosters are
   simply not available from that endpoint.
3. **`position_href` is the only position information ESPN ships on a roster row.**
   There is no `position` object, id, or abbreviation. The released
   `rosters_2023/2024/2025` assets shipped the href verbatim, which is why they had
   no usable position column. Resolve `.../positions/{id}` against the reference.
4. **Grouping positions are not positions.** Nine ids have `leaf: false`. A roster
   row can legitimately carry one (`50` Athlete, `70` Offense). Filter on
   `position_leaf` if you need assignable positions only.
5. **Position ids `0` and `99` are both "Unknown".** Pre-2014 rosters lean heavily
   on `0` — 17.4 % of all athlete-game rows league-wide. Do not read a non-null
   `position` as "ESPN knew the position".
6. **Pre-2014 rosters are ~4× thinner.** ~50 k athlete-game rows/season for
   2004–2013 vs ~205 k for 2014+ (ESPN's per-game participant coverage begins in
   earnest in 2014). Season roster counts follow. This is a real ESPN coverage
   cliff, not a scrape gap.
7. **2020 is short** (754 captured games vs ~930 typical) and 14 of the 16
   uncaptured master ids are 2020 — COVID cancellations.
8. **2026 has 946 captured game files and 0 roster rows** (scheduled, unplayed).
   The build range stops at 2025 for that reason; 2026 fills in as games are played.
9. **`athlete_id` and `team_id` are `int` in the raw JSON — pin them to `Int64`.**
   Never route an id through a float (a float-origin id stringifies as `"123.0"`).
   The compiler casts at the boundary and asserts dtype agreement on both joins
   (`position_id`, `team_id`) before joining.
10. **`jersey` is space-padded** (`"88 "`). Always `str.strip_chars()`. It is also
    a *string*, not a number (`"00"` is a real jersey), and is nullable.
11. **`height` / `weight` / `age` / `date_of_birth` are nullable and get more so the
    further back you go**; `experience_*` is nullable throughout.
12. **A transferring athlete legitimately produces two rows in one season** — one
    per `(team_id, athlete_id)`. The grain is `(season, team_id, athlete_id)`, NOT
    `(season, athlete_id)`. De-duplicating on `athlete_id` alone silently drops
    real transfer/mid-season-move rows.
13. **The same athlete appears in many games** — the collapse keeps the LAST
    appearance by `(week, game_id)`, so attribute values are the most recent ESPN
    had. `games_rostered` records how many game rosters the athlete appeared on;
    a value of 1 usually means a walk-on or a single-game call-up.
14. **`division` is not in the team payload.** `cfb/teams/json/{season}.json`
    `divisions` is the only authoritative source (group `80` = fbs, `81` = fcs).
    A team not in either group gets a null `division`.
15. **A team's own `groups` `$ref` sits under season type 3** while the 80/81
    children live under type 2 — only the group id is comparable across the two,
    never the URL.
16. **The generated `espn_cfb_positions` wrapper hardcodes `params={}`**, so `limit`
    cannot be passed and ESPN pages at 25 with no `page` kwarg. The reference
    capture goes through `dl_utils.download(url, params={"limit": 200})` instead.

    **This bites the roster endpoint too, and it costs data.** `espn_cfb_team_roster`
    has the same limitation — `params=` raises `TypeError` from
    `_codegen_runtime._get()` — and the roster route pages at **100**. Measured on
    Alabama (333): the wrapper returns 100 players, `limit=500` returns **120**.
    Stage 03 therefore fetches through `dl_utils.download` with an explicit
    `limit` and refuses to bank a roster whose player count equals the limit,
    because a full page is indistinguishable from a complete squad.
17. **The old `espn_cfb_rosters` release was two datasets under one tag.**
    `roster_2004…2022` (18 columns) are byte-equivalent MIRRORS of the CFBD-sourced
    `cfbfastR-data/rosters/parquet/cfb_rosters_{season}.parquet` — the exact file
    `sportsdataverse.cfb.load_cfb_rosters` already fetches — while
    `rosters_2023…2025` (66–71 columns) are genuinely ESPN-derived and FBS-only
    (~148 teams). Two naming stems, two schemas, two sources, one tag.
