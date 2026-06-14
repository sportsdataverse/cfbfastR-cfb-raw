"""EP curve + punt success-rate lookup tables (from assets/ep.csv + punt_sr.csv)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_ASSETS = Path(__file__).parent / "assets"


def load_ep_curve() -> list[float]:
    """Return EP values indexed by integer yardline (ep[yardline], len=101)."""
    df = pd.read_csv(_ASSETS / "ep.csv", encoding="utf-8")
    return df["ep"].tolist()


def load_punt_sr() -> dict[int, float]:
    """Return {yardline: ExpPuntNet} mapping (yardlines 1-100)."""
    df = pd.read_csv(_ASSETS / "punt_sr.csv", encoding="utf-8")
    return dict(zip(df["Yardline"].astype(int), df["ExpPuntNet"]))


def load_fg_sr() -> dict[int, tuple[float, float]]:
    """Return {distance: (accuracy, exp_fg_value)} mapping."""
    df = pd.read_csv(_ASSETS / "fg_sr.csv", encoding="utf-8")
    return {
        int(row["Distance"]): (float(row["Accuracy"]), float(row["ExpFGValue"]))
        for _, row in df.iterrows()
    }


def ep_at(ep: list[float], yardline: int) -> float:
    """Look up EP at a yardline, clamped to [0, 100]."""
    return ep[max(0, min(100, int(yardline)))]


def eqppp(ep: list[float], yard_line: int, yards_gained: int) -> float:
    """EqPPP = EP(yl + yards) - EP(yl), clamped to keep destination in [0, 100]."""
    return ep_at(ep, yard_line + yards_gained) - ep_at(ep, yard_line)


def determine_kick_ep(
    ep: list[float],
    kick_yardline: int,
    distance: int,
    return_yards: int,
) -> float:
    """Net EP value for a kick play: EP(land+return) - EP(kick)."""
    landing = kick_yardline + distance
    net_yardline = landing - return_yards
    return ep_at(ep, net_yardline) - ep_at(ep, kick_yardline)
