"""Stage 05 player_stats -- against a REAL trimmed athlete capture.

`tests/fixtures/cfb_player_stats_4426339_trimmed.json` is an actual
``common/v3 .../athletes/{id}/stats`` response with each category's
``statistics`` list cut to one entry. The 5-category shape is ESPN's.
"""

import json
from datetime import date
from pathlib import Path

import pytest
from cfb_raw_scrape import scrape_cfb_player_stats as ps

FIXTURE = Path(__file__).parent / "fixtures" / "cfb_player_stats_4426339_trimmed.json"


@pytest.fixture
def career():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class _Log:
    def info(self, *a):
        pass

    def error(self, *a):
        pass


def _roster(tmp_path, game_id, season, athlete_ids):
    p = tmp_path / "cfb" / "game_rosters" / "json" / f"{game_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "game_id": game_id,
                "season": season,
                "data": [{"athlete_id": a} for a in athlete_ids],
            }
        ),
        encoding="utf-8",
    )


def test_count_categories(career):
    assert ps.count_categories(career) == 5
    assert ps.count_categories({}) == 0
    assert ps.count_categories({"categories": []}) == 0


def test_athletes_by_season_reads_stage_04(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _roster(tmp_path, 1, 2024, [10, 11])
    _roster(tmp_path, 2, 2024, [11, 12])
    _roster(tmp_path, 3, 2023, [99])
    per = ps.athletes_by_season(2024, 2024)
    assert per == {2024: {"10", "11", "12"}}


def test_athletes_by_season_skips_unplayed_games(tmp_path, monkeypatch):
    """1,655 games in the tree carry data: [] -- future fixtures. Not an error."""
    monkeypatch.chdir(tmp_path)
    _roster(tmp_path, 1, 2026, [])
    _roster(tmp_path, 2, 2026, [7])
    assert ps.athletes_by_season(2026, 2026) == {2026: {"7"}}


def test_athletes_by_season_RAISES_when_stage_04_has_not_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="run stage 04"):
        ps.athletes_by_season(2024, 2024)


def test_stale_athletes_flags_anyone_in_a_LIVE_season():
    """A career payload GROWS. An athlete still playing must re-fetch, or their
    file freezes at the season we first saw them -- the 2027-recruiting bug."""
    per = {2024: {"a", "b"}, 2026: {"b", "c"}}
    # 2024 is final by Aug 2026; 2026 is not.
    stale = ps.stale_athletes(per, today=date(2026, 8, 29))
    assert stale == {"b", "c"}
    # "a" only ever played a finished season, so their career cannot change.
    assert "a" not in stale


def test_stale_athletes_empty_once_every_season_is_final():
    assert ps.stale_athletes({2024: {"a"}}, today=date(2026, 8, 29)) == set()


def test_is_complete_is_false_for_a_stale_athlete(tmp_path, monkeypatch, career):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ps.out_path(4426339)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(career), encoding="utf-8")
    assert ps.is_complete(4426339) is True
    assert ps.is_complete(4426339, stale={"4426339"}) is False


def test_is_complete_rejects_empty_or_corrupt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ps.out_path(1)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"categories": []}), encoding="utf-8")
    assert ps.is_complete(1) is False
    p.write_text("{not json", encoding="utf-8")
    assert ps.is_complete(1) is False
    assert ps.is_complete(999999) is False


def test_write_one_banks_a_real_payload(tmp_path, monkeypatch, career):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ps, "espn_cfb_player_stats_v3", lambda **kw: career, raising=True
    )
    assert ps.write_one(4426339, _Log()) == 5
    assert (tmp_path / ps.out_path(4426339)).is_file()


def test_write_one_REFUSES_an_empty_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ps, "espn_cfb_player_stats_v3", lambda **kw: {"athlete": {}}, raising=True
    )
    with pytest.raises(RuntimeError, match="0 categories"):
        ps.write_one(1, _Log())
    assert not (tmp_path / ps.out_path(1)).exists()


def test_write_one_does_NOT_pass_season(tmp_path, monkeypatch, career):
    """ESPN ignores the season param -- measured, identical payload for 2023,
    2024 and None. Passing it would imply a scoping that does not exist."""
    monkeypatch.chdir(tmp_path)
    seen = {}

    def _f(**kw):
        seen.update(kw)
        return career

    monkeypatch.setattr(ps, "espn_cfb_player_stats_v3", _f, raising=True)
    ps.write_one(4426339, _Log())
    assert "season" not in seen
    assert seen["athlete_id"] == 4426339


def test_main_reports_red_when_stage_04_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ps.main(["-s", "2024", "-e", "2024"]) == 1


def test_main_one_bad_athlete_does_not_stop_the_sweep(tmp_path, monkeypatch, career):
    monkeypatch.chdir(tmp_path)
    _roster(tmp_path, 1, 2024, [10, 11])

    def _f(*, athlete_id, **kw):
        if str(athlete_id) == "11":
            raise RuntimeError("boom")
        return career

    monkeypatch.setattr(ps, "espn_cfb_player_stats_v3", _f, raising=True)
    assert ps.main(["-s", "2024", "-e", "2024"]) == 1
    assert (tmp_path / ps.out_path(10)).is_file()
