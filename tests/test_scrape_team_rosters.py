"""Stage 03 team_rosters -- against a REAL trimmed Alabama capture.

`tests/fixtures/cfb_team_roster_333_trimmed.json` is an actual response with each
position group cut to 2 players. The 6-group shape (offense / defense /
specialTeam / injuredReserveOrOut / suspended / practiceSquad) is ESPN's, not
mine -- a hand-written payload would only confirm my guess about it.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from cfb_raw_scrape import scrape_cfb_team_rosters as tr

FIXTURE = Path(__file__).parent / "fixtures" / "cfb_team_roster_333_trimmed.json"


@pytest.fixture
def roster():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _Log:
    def info(self, *a):
        pass

    def error(self, *a):
        pass


def test_count_players_sums_every_position_group(roster):
    # 3 populated groups x 2 players after trimming; the 3 empty groups must not
    # break the sum, and must not be silently dropped from the shape either.
    assert tr.count_players(roster) == 6
    assert len(roster["athletes"]) == 6


def test_count_players_on_an_empty_payload():
    assert tr.count_players({}) == 0
    assert tr.count_players({"athletes": []}) == 0


def test_team_ids_reads_stage_01_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / tr.teams_path(2026)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"divisions": {"80": ["2", "5"], "81": ["57"], "35": ["999"]}}),
        encoding="utf-8",
    )
    # 35 (D2/D3) is deliberately NOT enumerated -- those teams have no roster
    # endpoint, the same asymmetry as standings.
    assert tr.team_ids(2026) == ["2", "5", "57"]
    assert tr.team_ids(2026, groups=("35",)) == ["999"]


def test_team_ids_RAISES_when_stage_01_has_not_run(tmp_path, monkeypatch):
    """A missing precondition must be loud. An empty list would report
    '0 rosters' as a successful run."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="run stage 01"):
        tr.team_ids(2026)


def test_write_one_banks_the_current_season(tmp_path, monkeypatch, roster):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tr, "_fetch", lambda team_id: roster)
    n = tr.write_one(roster["season"]["year"], 333, _Log())
    assert n == 6
    written = json.loads(
        (tmp_path / tr.out_path(roster["season"]["year"], 333)).read_text(
            encoding="utf-8"
        )
    )
    assert written["season_requested"] == roster["season"]["year"]


def test_write_one_REFUSES_a_season_the_endpoint_cannot_serve(
    tmp_path, monkeypatch, roster
):
    """The endpoint has no season parameter. Banking its current payload under
    2024 would look like a successful backfill and be a fabrication."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tr, "_fetch", lambda team_id: roster)
    with pytest.raises(RuntimeError, match="endpoint served season"):
        tr.write_one(2024, 333, _Log())
    assert not (tmp_path / tr.out_path(2024, 333)).exists()


def test_write_one_refuses_a_roster_with_no_players(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        tr, "_fetch", lambda team_id: {"season": {"year": 2026}, "athletes": []}
    )
    with pytest.raises(RuntimeError, match="0 players"):
        tr.write_one(2026, 333, _Log())


def test_is_season_final_matches_the_standings_rule():
    assert tr.is_season_final(2026, date(2026, 10, 1)) is False
    assert tr.is_season_final(2026, date(2027, 1, 20)) is False
    assert tr.is_season_final(2026, date(2027, 2, 1)) is True


def test_is_complete_never_true_for_a_live_season(tmp_path, monkeypatch, roster):
    """Rosters churn all year -- transfers, injuries, suspensions. A banked file
    is a snapshot, so the live season must keep refreshing."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / tr.out_path(2026, 333)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(roster), encoding="utf-8")
    assert tr.is_complete(2026, 333, today=date(2026, 10, 1)) is False
    assert tr.is_complete(2026, 333, today=date(2027, 2, 1)) is True


def test_is_complete_rejects_an_empty_or_corrupt_bank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / tr.out_path(2024, 333)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"athletes": []}), encoding="utf-8")
    assert tr.is_complete(2024, 333) is False
    p.write_text("{not json", encoding="utf-8")
    assert tr.is_complete(2024, 333) is False


def test_main_reports_red_when_stage_01_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tr.main(["-s", "2026", "-e", "2026"]) == 1


def test_main_one_bad_team_does_not_stop_the_sweep(tmp_path, monkeypatch, roster):
    monkeypatch.chdir(tmp_path)
    year = roster["season"]["year"]
    p = tmp_path / tr.teams_path(year)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"divisions": {"80": ["333", "999"]}}), encoding="utf-8")

    def _fetch(team_id):
        if str(team_id) == "999":
            raise RuntimeError("boom")
        return roster

    monkeypatch.setattr(tr, "_fetch", _fetch)
    rc = tr.main(["-s", str(year), "-e", str(year)])
    assert rc == 1  # the driver sees red
    assert (tmp_path / tr.out_path(year, 333)).exists()  # the good team still landed
