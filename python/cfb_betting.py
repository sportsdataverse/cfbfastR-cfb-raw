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


def _odds_scalar(v):
    """Coerce a persisted odds value to a float scalar (or None).

    Historical betting aux (≈2012-2022) persisted ``game_spread`` / ``over_under``
    as stringified single-element arrays (e.g. ``'[3.]'``, ``'[-7.5]'``,
    ``'[61.5]'``) or as JSON lists; coerce those to a plain float so the
    ``CFBPlayProcess(odds_override=...)`` ``float()`` coercion in ``__init__``
    can't ``ValueError``. Returns None when missing or unparseable.
    """
    if v is None or isinstance(v, bool):  # bool is an int subclass — never a spread
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return _odds_scalar(v[0]) if v else None
    if isinstance(v, str):
        s = v.strip().strip("[]").split(",")[0].strip()
        try:
            return float(s) if s else None
        except ValueError:
            return None
    return None


def odds_override_from_betting(betting: dict):
    """Reconstruct CFBPlayProcess odds_override from a persisted betting dict.
    Returns None if the betting dict is missing the resolved spread (caller then lets
    CFBPlayProcess resolve normally)."""
    if not betting or betting.get("game_spread") is None:
        return None
    spread = _odds_scalar(betting.get("game_spread"))
    if spread is None:  # present but unparseable -> let CFBPlayProcess resolve normally
        return None
    over_under = _odds_scalar(betting.get("over_under"))
    if over_under is None:
        over_under = 55.5  # CFBPlayProcess default; the spread is the key EPA/WPA input
    return {
        "gameSpread": spread,
        "overUnder": over_under,
        "homeFavorite": bool(betting.get("home_favorite")),
        "gameSpreadAvailable": bool(betting.get("game_spread_available", False)),
    }
