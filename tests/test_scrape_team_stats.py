"""Stage 06 team_stats -- against a REAL trimmed Core v2 capture.

`tests/fixtures/cfb_team_stats_333_2024_trimmed.json` is an actual
``.../seasons/2024/types/2/teams/333/statistics`` response (11 categories /
285 stats live) cut to 3 categories x 2 stats.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from cfb_raw_scrape import scrape_cfb_team_stats as ts

FIXTURE = Path(__file__).parent / "fixtures" / "cfb_team_stats_333_2024_trimmed.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _Log:
    def info(self, *a):
        pass

    def error(self, *a):
        pass


def _teams(tmp_path, season, ids):
    p = tmp_path / ts.teams_path(season)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"divisions": {"80": ids}}), encoding="utf-8")


def test_summarize_counts_categories_and_stats(payload):
    assert ts.summarize(payload) == {"categories": 3, "stats": 6}


def test_summarize_on_empty_and_scaffolded_payloads():
    assert ts.summarize({}) == {"categories": 0, "stats": 0}
    # ESPN can ship category scaffolding with every stats list empty -- a
    # non-zero category count carrying zero actual data.
    scaffold = {"splits": {"categories": [{"name": "passing", "stats": []}] * 4}}
    assert ts.summarize(scaffold) == {"categories": 4, "stats": 0}


def test_url_puts_season_and_type_in_the_PATH():
    """A query-param season can be accepted and ignored (that is exactly what
    /athletes/{id}/stats does). A path segment cannot."""
    u = ts.STATS_URL.format(season=2024, seasontype=2, team_id=333)
    assert u.endswith("/seasons/2024/types/2/teams/333/statistics")
    assert "?" not in u


def test_season_types_capture_both_regular_and_cumulative():
    """Measured: type 3 totals EXCEED type 2 in every season, so type 3 is
    cumulative incl. postseason -- not a postseason-only split."""
    assert ts.SEASON_TYPES == (2, 3)


def test_min_season_is_the_probed_floor():
    assert ts.MIN_SEASON == 2004  # 2003 returns 404 for both types


def test_team_ids_reads_stage_01(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _teams(tmp_path, 2024, ["333", "99", "333"])
    assert ts.team_ids(2024) == ["99", "333"]


def test_team_ids_RAISES_when_stage_01_has_not_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="run stage 01"):
        ts.team_ids(2024)


def test_write_one_banks_a_real_payload(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts, "_fetch", lambda s, t, tid: payload)
    assert ts.write_one(2024, 2, 333, _Log()) == {"categories": 3, "stats": 6}
    written = json.loads(
        (tmp_path / ts.out_path(2024, 2, 333)).read_text(encoding="utf-8")
    )
    assert written["season_requested"] == 2024
    assert written["season_type_requested"] == 2


def test_write_one_REFUSES_a_scaffolded_payload(tmp_path, monkeypatch):
    """Categories present but every stats[] empty is the subtler throttle --
    a category count alone would call this a success."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ts,
        "_fetch",
        lambda s, t, tid: {
            "splits": {"categories": [{"name": "passing", "stats": []}]}
        },
    )
    with pytest.raises(RuntimeError, match="refusing to bank an empty payload"):
        ts.write_one(2024, 2, 333, _Log())
    assert not (tmp_path / ts.out_path(2024, 2, 333)).exists()


def test_write_one_REFUSES_an_empty_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts, "_fetch", lambda s, t, tid: {})
    with pytest.raises(RuntimeError):
        ts.write_one(2024, 2, 333, _Log())


def test_is_complete_is_false_for_a_LIVE_season(tmp_path, monkeypatch, payload):
    """Season stats accumulate weekly, so a mid-season file is a snapshot."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ts.out_path(2026, 2, 333)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert ts.is_complete(2026, 2, 333, today=date(2026, 10, 1)) is False
    assert ts.is_complete(2026, 2, 333, today=date(2027, 2, 1)) is True


def test_is_complete_rejects_presence_without_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ts.out_path(2024, 2, 333)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"splits": {"categories": []}}), encoding="utf-8")
    assert ts.is_complete(2024, 2, 333) is False
    p.write_text("{not json", encoding="utf-8")
    assert ts.is_complete(2024, 2, 333) is False


def test_main_covers_every_type_for_every_team(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    _teams(tmp_path, 2024, ["333", "99"])
    calls = []
    monkeypatch.setattr(
        ts, "_fetch", lambda s, t, tid: calls.append((s, t, tid)) or payload
    )
    assert ts.main(["-s", "2024", "-e", "2024"]) == 0
    assert sorted(calls) == [
        (2024, 2, "333"),
        (2024, 2, "99"),
        (2024, 3, "333"),
        (2024, 3, "99"),
    ]


def test_main_clamps_to_the_season_floor(tmp_path, monkeypatch, payload):
    """2003 is a 404 for both types -- never request it."""
    monkeypatch.chdir(tmp_path)
    _teams(tmp_path, 2004, ["333"])
    calls = []
    monkeypatch.setattr(ts, "_fetch", lambda s, t, tid: calls.append(s) or payload)
    ts.main(["-s", "2001", "-e", "2004"])
    assert calls and min(calls) == ts.MIN_SEASON


def test_main_one_bad_team_does_not_stop_the_sweep(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    _teams(tmp_path, 2024, ["333", "99"])

    def _f(s, t, tid):
        if tid == "99":
            raise RuntimeError("boom")
        return payload

    monkeypatch.setattr(ts, "_fetch", _f)
    assert ts.main(["-s", "2024", "-e", "2024"]) == 1
    assert (tmp_path / ts.out_path(2024, 2, 333)).is_file()
