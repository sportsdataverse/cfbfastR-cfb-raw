import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "cfb_betting.py"
spec = importlib.util.spec_from_file_location("cfb_betting", P)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class _Proc:
    gameSpread = -7.5
    overUnder = 52.5
    homeFavorite = True
    gameSpreadAvailable = True
    odds_source = "summary_pickcenter"


def test_capture_betting_stable_shape():
    raw = {"pickcenter": [{"spread": -7.5}], "odds": [], "predictor": {},
           "againstTheSpread": []}
    out = b.capture_betting(raw, _Proc(), odds_full=[{"provider": "x"}], propbets=[])
    assert out["game_spread"] == -7.5
    assert out["over_under"] == 52.5
    assert out["home_favorite"] is True
    assert out["home_team_spread"] == -7.5  # favorite -> -abs(spread)
    assert out["game_spread_available"] is True
    assert out["odds_source"] == "summary_pickcenter"
    for k in ("pickcenter", "odds", "predictor", "against_the_spread",
              "odds_core_items", "odds_full", "propbets"):
        assert k in out


def test_capture_betting_handles_missing_keys():
    out = b.capture_betting({}, _Proc(), odds_full=None, propbets=None)
    assert out["pickcenter"] == [] and out["odds"] == [] and out["propbets"] == []
    assert out["predictor"] == {}


def test_home_team_spread_sign_when_away_favorite():
    class _AwayFav(_Proc):
        homeFavorite = False
    out = b.capture_betting({}, _AwayFav(), odds_full=None, propbets=None)
    assert out["home_team_spread"] == 7.5  # away favorite -> +abs(spread)


def test_odds_override_from_betting_roundtrip():
    betting = {"game_spread": -10.5, "over_under": 60.0, "home_favorite": True,
               "game_spread_available": True}
    o = b.odds_override_from_betting(betting)
    assert o == {"gameSpread": -10.5, "overUnder": 60.0,
                 "homeFavorite": True, "gameSpreadAvailable": True}
