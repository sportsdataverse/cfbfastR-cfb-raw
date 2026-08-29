"""Stage 07 standings -- exercised against a REAL trimmed ESPN capture.

The fixture is `tests/fixtures/cfb_standings_2024_trimmed.json`: an actual 2024
response with `children` cut to 2 conferences and each `standings.entries` cut
to 3 teams. Trimmed for size, never hand-written -- a synthetic payload would
only prove the code agrees with my guess about ESPN's shape, which is the
failure mode this repo has hit before.
"""

import json
from pathlib import Path

import pytest

from cfb_raw_scrape import scrape_cfb_standings as ss

FIXTURE = Path(__file__).parent / "fixtures" / "cfb_standings_2024_trimmed.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_summarize_counts_conferences_and_entries(payload):
    counts = ss.summarize(payload)
    assert counts == {"conferences": 2, "entries": 6}


def test_summarize_survives_a_conference_with_no_standings(payload):
    # ESPN ships conference groups with no standings block at all early in a
    # season; that must count as zero entries, not raise.
    payload["children"].append({"id": "999", "name": "Empty Group"})
    assert ss.summarize(payload)["entries"] == 6


def test_write_one_banks_a_real_payload(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ss, "_fetch", lambda season: payload)

    class _Log:
        def info(self, *a):
            pass

    counts = ss.write_one(2024, _Log())
    assert counts == {"conferences": 2, "entries": 6}
    written = json.loads((tmp_path / ss.out_path(2024)).read_text(encoding="utf-8"))
    assert written["season"] == 2024
    assert len(written["children"]) == 2


def test_write_one_REFUSES_an_empty_payload(tmp_path, monkeypatch):
    """The 200-with-absent-array throttle must not be banked as a real season."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ss, "_fetch", lambda season: {"id": "1", "children": []})

    class _Log:
        def info(self, *a):
            pass

    with pytest.raises(RuntimeError, match="refusing to bank an empty payload"):
        ss.write_one(2024, _Log())
    assert not (tmp_path / ss.out_path(2024)).exists()


def test_write_one_refuses_conferences_with_zero_entries(tmp_path, monkeypatch):
    """Conferences present but every entries[] empty is the subtler throttle."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ss, "_fetch", lambda season: {"children": [{"standings": {"entries": []}}]}
    )

    class _Log:
        def info(self, *a):
            pass

    with pytest.raises(RuntimeError):
        ss.write_one(2024, _Log())


def test_is_complete_rejects_presence_without_content(tmp_path, monkeypatch):
    """Presence is not validity -- an empty banked file must not block its retry."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ss.out_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"children": []}), encoding="utf-8")
    assert ss.is_complete(2024) is False


def test_is_complete_rejects_a_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ss.out_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert ss.is_complete(2024) is False


def test_is_complete_accepts_the_real_payload(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ss.out_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert ss.is_complete(2024) is True


def test_main_skips_a_banked_season_unless_rescraped(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ss.out_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    monkeypatch.setattr(ss, "_fetch", lambda season: calls.append(season) or payload)

    assert ss.main(["-s", "2024", "-e", "2024"]) == 0
    assert calls == []  # skipped

    assert ss.main(["-s", "2024", "-e", "2024", "-r", "true"]) == 0
    assert calls == [2024]  # forced


def test_main_bad_rescrape_value_is_false_not_an_error(tmp_path, monkeypatch, payload):
    """A cron typo must not trigger a full re-scrape, and must not raise."""
    monkeypatch.chdir(tmp_path)
    p = tmp_path / ss.out_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    monkeypatch.setattr(ss, "_fetch", lambda season: calls.append(season) or payload)
    assert ss.main(["-s", "2024", "-e", "2024", "-r", "ture"]) == 0
    assert calls == []


def test_main_one_bad_season_does_not_stop_the_range(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)

    def _fetch(season):
        if season == 2023:
            raise RuntimeError("boom")
        return payload

    monkeypatch.setattr(ss, "_fetch", _fetch)
    rc = ss.main(["-s", "2023", "-e", "2024"])
    assert rc == 1  # the driver sees red
    assert (tmp_path / ss.out_path(2024)).exists()  # but 2024 still landed
