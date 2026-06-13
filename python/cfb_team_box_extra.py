"""Derive per-team box 'extra' fields from the summary, per the de-dup gate.

Returns None when the summary lacks the data, signalling the caller to fall back
to the event_competitor_* endpoints.
"""
from __future__ import annotations


def _competitors(raw: dict) -> list:
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors")
    return comps or []


def _box_teams(raw: dict) -> list:
    return raw.get("boxscore", {}).get("teams") or []


def _leaders(raw: dict) -> list:
    return raw.get("leaders") or []


def team_box_extra_from_summary(raw: dict, team_ids):
    comps = _competitors(raw)
    if not comps:
        return None
    by_team = {}
    box_by_id = {str(t.get("team", {}).get("id")): t for t in _box_teams(raw)}
    lead_by_id = {str(lead.get("team", {}).get("id")): lead for lead in _leaders(raw)}
    for c in comps:
        tid = str(c.get("team", {}).get("id"))
        by_team[tid] = {
            "record": c.get("record") or [],
            "linescores": c.get("linescores") or [],
            "statistics": (box_by_id.get(tid, {}).get("statistics") or []),
            "leaders": (lead_by_id.get(tid, {}).get("leaders") or []),
        }
    # require at least record/linescores for both requested teams to consider it complete
    for tid in (str(t) for t in team_ids):
        if tid not in by_team or not by_team[tid]["linescores"]:
            return None
    return by_team
