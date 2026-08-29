"""Stage 51 player_core -- against a REAL trimmed Core v2 athlete record."""

import json
from pathlib import Path

import pytest
from cfb_raw_scrape import scrape_cfb_player_core as pc

FIXTURE = Path(__file__).parent / "fixtures" / "cfb_player_core_4426339.json"


@pytest.fixture
def core():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _Log:
    def info(self, *a):
        pass

    def error(self, *a):
        pass


def test_the_fixture_is_a_real_identifiable_record(core):
    assert core["id"] in (4426339, "4426339")
    assert core["fullName"] == "Spencer Rattler"
    assert pc.is_valid(core) is True


def test_team_and_college_are_ref_only(core):
    """Confirms the endpoint's shape rather than trusting the docstring: both
    are {"$ref"} and nothing else, which is WHY hydrating them is banned --
    it would triple the request count for ids already in the ref URL."""
    for key in ("team", "college"):
        if core.get(key) is not None:
            assert list(core[key]) == ["$ref"]


def test_is_valid_rejects_an_envelope_without_a_record():
    assert pc.is_valid({}) is False
    assert pc.is_valid({"id": 1}) is False  # no name
    assert pc.is_valid({"fullName": "X"}) is False  # no id
    assert pc.is_valid({"id": 1, "displayName": "X"}) is True


def test_write_one_banks_a_real_record(tmp_path, monkeypatch, core):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pc, "espn_cfb_player_core", lambda **kw: core, raising=True)
    assert pc.write_one(4426339, _Log()) == "Spencer Rattler"
    assert (tmp_path / pc.out_path(4426339)).is_file()


def test_write_one_REFUSES_an_unidentifiable_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pc, "espn_cfb_player_core", lambda **kw: {"$ref": "x"}, raising=True
    )
    with pytest.raises(RuntimeError, match="no id/name"):
        pc.write_one(1, _Log())
    assert not (tmp_path / pc.out_path(1)).exists()


def test_is_complete_refetches_an_ACTIVE_athlete(tmp_path, monkeypatch, core):
    """An active athlete's record is a moving target -- team, jersey, weight.
    Catches the transfer who has not played yet, whom no live season flags."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / pc.out_path(1)
    p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(json.dumps({**core, "active": False}), encoding="utf-8")
    assert pc.is_complete(1) is True

    p.write_text(json.dumps({**core, "active": True}), encoding="utf-8")
    assert pc.is_complete(1) is False


def test_is_complete_honours_the_shared_stale_set(tmp_path, monkeypatch, core):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / pc.out_path(7)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**core, "active": False}), encoding="utf-8")
    assert pc.is_complete(7) is True
    assert pc.is_complete(7, stale={"7"}) is False


def test_is_complete_rejects_empty_or_corrupt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / pc.out_path(1)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"$ref": "x"}), encoding="utf-8")
    assert pc.is_complete(1) is False
    p.write_text("{not json", encoding="utf-8")
    assert pc.is_complete(1) is False
    assert pc.is_complete(999999) is False


def test_reuses_stage_05_athlete_enumeration():
    """One source of truth for 'who played' -- not a second copy that can drift."""
    from cfb_raw_scrape import scrape_cfb_player_stats as ps

    assert pc.athletes_by_season is ps.athletes_by_season
    assert pc.stale_athletes is ps.stale_athletes


def test_main_reports_red_when_stage_04_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert pc.main(["-s", "2024", "-e", "2024"]) == 1
