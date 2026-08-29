"""Stage 52 espn_recruits -- pagination, count verification, class finality."""

import json
from datetime import date

import pytest
from cfb_raw_scrape import scrape_cfb_espn_recruits as er


class _Log:
    def info(self, *a):
        pass

    def error(self, *a):
        pass


def _page(n, count=3815):
    return {"count": count, "pageIndex": 1, "items": [{"athlete": {}}] * n}


def test_fetch_page_sends_BOTH_limit_and_page(monkeypatch):
    """The generated wrapper has limit but no page, so page= dies as a TypeError
    inside download(). That is why this stage bypasses it -- assert the params
    actually go out, not merely that the call returns."""
    seen = {}

    class _R:
        def json(self):
            return {}

    monkeypatch.setattr(
        er,
        "download",
        lambda url, params=None, **kw: (seen.update(url=url, params=params), _R())[1],
    )
    er._fetch_page(2024, 3)
    assert seen["params"] == {"limit": er.PAGE_SIZE, "page": 3}
    assert "/recruiting/2024/athletes" in seen["url"]


def test_scrape_year_pages_to_exhaustion(tmp_path, monkeypatch):
    """Real 2024 shape: 1000 + 1000 + 1000 + 815 = 3815 = count."""
    monkeypatch.chdir(tmp_path)
    sizes = {1: 1000, 2: 1000, 3: 1000, 4: 815}
    monkeypatch.setattr(er, "_fetch_page", lambda y, p: _page(sizes.get(p, 0)))
    m = er.scrape_year(2024, logger=_Log())
    assert m["rows"] == 3815 == m["expected"]
    assert m["pages"] == 4
    assert (tmp_path / er.manifest_path(2024)).is_file()
    assert (tmp_path / er.page_path(2024, 4)).is_file()


def test_scrape_year_REFUSES_a_short_class(tmp_path, monkeypatch):
    """A feed that returns fewer rows than it says exist is truncated. No
    manifest may be written, or the year reads as complete forever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        er, "_fetch_page", lambda y, p: _page(1000 if p == 1 else 0, count=3815)
    )
    with pytest.raises(RuntimeError, match="banked 1000 rows but the feed states 3815"):
        er.scrape_year(2024, logger=_Log())
    assert not (tmp_path / er.manifest_path(2024)).is_file()


def test_scrape_year_REFUSES_a_feed_that_states_no_count(tmp_path, monkeypatch):
    """Without an independent count, completeness would be an inference from
    'the last page looked short' -- which is also what truncation looks like."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        er, "_fetch_page", lambda y, p: {"items": [{}] * 10 if p == 1 else []}
    )
    with pytest.raises(RuntimeError, match="stated no count"):
        er.scrape_year(2024, logger=_Log())


def test_is_class_final_keeps_an_OPEN_class_refreshable():
    """The guard stage 50 lacks: the 2027 class was banked 'complete' at 4,779
    rows in Aug 2026, months before it signs."""
    assert er.is_recruiting_class_final(2027, date(2026, 8, 29)) is False
    assert (
        er.is_recruiting_class_final(2027, date(2027, 2, 1)) is False
    )  # signing day-ish
    assert er.is_recruiting_class_final(2027, date(2027, 4, 1)) is True
    assert er.is_recruiting_class_final(2024, date(2026, 8, 29)) is True


def test_is_complete_requires_finality_AND_agreeing_totals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / er.manifest_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)

    p.write_text(json.dumps({"rows": 3815, "expected": 3815}), encoding="utf-8")
    assert er.is_complete(2024, today=date(2026, 8, 29)) is True

    # an open class is never complete, however good its manifest looks
    assert er.is_complete(2027, today=date(2026, 8, 29)) is False

    # a manifest that disagrees with itself is not a completion marker
    p.write_text(json.dumps({"rows": 100, "expected": 3815}), encoding="utf-8")
    assert er.is_complete(2024, today=date(2026, 8, 29)) is False


def test_is_complete_rejects_missing_or_corrupt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert er.is_complete(2024, today=date(2026, 8, 29)) is False
    p = tmp_path / er.manifest_path(2024)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert er.is_complete(2024, today=date(2026, 8, 29)) is False


def test_main_one_bad_year_does_not_stop_the_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def _f(y, p):
        if y == 2023:
            raise RuntimeError("boom")
        return _page(10 if p == 1 else 0, count=10)

    monkeypatch.setattr(er, "_fetch_page", _f)
    assert er.main(["-s", "2023", "-e", "2024"]) == 1
    assert (tmp_path / er.manifest_path(2024)).is_file()
