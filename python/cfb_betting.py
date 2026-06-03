"""Normalize ESPN betting payloads into a stable, null-safe shape."""
from __future__ import annotations


def capture_betting(raw: dict, proc, *, odds_full=None, propbets=None) -> dict:
    spread = proc.gameSpread
    home_fav = bool(proc.homeFavorite)
    home_team_spread = -abs(spread) if home_fav else abs(spread)
    return {
        # resolved odds (EPA/WPA inputs) — persisted so reprocess injects them
        "game_spread": spread,
        "over_under": proc.overUnder,
        "home_favorite": home_fav,
        "home_team_spread": home_team_spread,
        "game_spread_available": bool(proc.gameSpreadAvailable),
        "odds_source": getattr(proc, "odds_source", None),
        # raw payloads for forensics + re-normalization
        "pickcenter": raw.get("pickcenter") or [],
        "odds": raw.get("odds") or [],
        "predictor": raw.get("predictor") or {},
        "against_the_spread": raw.get("againstTheSpread") or [],
        "odds_core_items": raw.get("odds_core_items") or [],
        "odds_full": odds_full or [],
        "propbets": propbets or [],
    }


def odds_override_from_betting(betting: dict) -> dict:
    """Reconstruct the CFBPlayProcess odds_override from a persisted betting dict."""
    return {
        "gameSpread": betting["game_spread"],
        "overUnder": betting["over_under"],
        "homeFavorite": betting["home_favorite"],
        "gameSpreadAvailable": betting["game_spread_available"],
    }
