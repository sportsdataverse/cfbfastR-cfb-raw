import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "cfb_team_box_extra.py"
spec = importlib.util.spec_from_file_location("cfb_team_box_extra", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _raw_with_header():
    return {
        "header": {"competitions": [{"competitors": [
            {"team": {"id": "333"}, "record": [{"summary": "1-0"}],
             "linescores": [{"value": 7}, {"value": 14}]},
            {"team": {"id": "99"}, "record": [{"summary": "0-1"}],
             "linescores": [{"value": 3}, {"value": 0}]},
        ]}]},
        "boxscore": {"teams": [
            {"team": {"id": "333"}, "statistics": [{"name": "totalYards", "displayValue": "400"}]},
            {"team": {"id": "99"}, "statistics": [{"name": "totalYards", "displayValue": "250"}]},
        ]},
        "leaders": [{"team": {"id": "333"}, "leaders": []},
                    {"team": {"id": "99"}, "leaders": []}],
    }


def test_team_box_extra_from_summary_present():
    out = m.team_box_extra_from_summary(_raw_with_header(), ["333", "99"])
    assert out is not None
    assert out["333"]["record"] == [{"summary": "1-0"}]
    assert out["333"]["linescores"] == [{"value": 7}, {"value": 14}]
    assert out["333"]["statistics"][0]["displayValue"] == "400"


def test_team_box_extra_returns_none_when_summary_lacks_it():
    out = m.team_box_extra_from_summary({"header": {"competitions": [{}]}}, ["333", "99"])
    assert out is None
