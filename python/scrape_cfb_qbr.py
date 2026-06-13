"""Scrape ESPN core QBR (the QBR-model training target), keyed by game_id + athlete.

Endpoint: sports.core.api.espn.com/.../seasons/{yr}/types/2/weeks/{wk}/qbr/10000?limit=1000
Each item has athlete/team/event $refs (event id = game_id) + splits.categories[0].stats
(QBR, TQBR, and component pieces). Output rows join to per-QB feature rows on
(game_id, passer_player_name).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

_EVENT_ID = re.compile(r"/events/(\d+)")
_ATHLETE_ID = re.compile(r"/athletes/(\d+)")


def _qbr_url(year: int, week: int) -> str:
    return (f"https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
            f"seasons/{year}/types/2/weeks/{week}/qbr/10000?limit=1000")


def parse_qbr_payload(payload: dict, year: int, week: int) -> list[dict]:
    rows = []
    for rec in payload.get("items", []) or []:
        ev = (rec.get("event") or {}).get("$ref", "")
        ath = (rec.get("athlete") or {}).get("$ref", "")
        gm = _EVENT_ID.search(ev)
        aid = _ATHLETE_ID.search(ath)
        out = {"year": year, "week": week,
               "game_id": int(gm.group(1)) if gm else None,
               "athlete_id": int(aid.group(1)) if aid else None}
        stats = (((rec.get("splits") or {}).get("categories") or [{}])[0]).get("stats", [])
        for s in stats:
            out[s["abbreviation"]] = s.get("value")
        rows.append(out)
    return rows


def _athlete_name(year: int, athlete_id: int, cache: dict, session: requests.Session) -> str | None:
    if athlete_id in cache:
        return cache[athlete_id]
    url = (f"https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
           f"seasons/{year}/athletes/{athlete_id}?lang=en&region=us")
    try:
        name = session.get(url, timeout=30).json().get("fullName")
    except Exception:
        name = None
    cache[athlete_id] = name
    return name


def scrape(years, weeks, out_path: str) -> int:
    import pandas as pd
    session = requests.Session()
    cache: dict = {}
    frames = []
    for yr in years:
        for wk in weeks:
            resp = session.get(_qbr_url(yr, wk), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rows = parse_qbr_payload(data, yr, wk)
            for r in rows:
                r["passer_player_name"] = _athlete_name(yr, r["athlete_id"], cache, session)
                r["raw_qbr"] = r.get("QBR")
                r["adj_qbr"] = r.get("TQBR")
            if rows:
                frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return len(df)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start", type=int, required=True)
    ap.add_argument("-e", "--end", type=int, required=True)
    ap.add_argument("--out", default="cfb/qbr/espn_qbr.parquet")
    args = ap.parse_args(argv)
    n = scrape(range(args.start, args.end + 1), range(1, 16), args.out)
    print(f"wrote {n} QBR rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
