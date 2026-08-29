"""Tests for the 247 recruit-class scraper.

The interesting logic is completeness accounting, not fetching: this producer
exists because an empty recruit feed shipped undetected for weeks, so the tests
concentrate on "can a partial scrape ever look finished".
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from cfb_raw_scrape import scrape_cfb_recruits as sr  # noqa: E402


def _page(n_players: int, *, count: int | None = None) -> dict:
    p = {"players": [{"key": i} for i in range(n_players)]}
    if count is not None:
        p["pagination"] = {
            "currentPage": 1,
            "itemsPerPage": sr.PAGE_SIZE,
            "count": count,
            "pageCount": 1,
        }
    return p


def test_rows_reads_the_named_key() -> None:
    assert sr._rows(_page(7)) == 7


def test_rows_falls_back_to_first_list_if_envelope_renamed() -> None:
    """A renamed envelope must degrade to a count, not silently return zero.

    Zero would look exactly like "this class has no recruits", which is the
    failure this whole producer exists to stop shipping.
    """
    assert sr._rows({"pagination": {"count": 3}, "recruits": [1, 2, 3]}) == 3


def test_rows_handles_missing_and_empty() -> None:
    assert sr._rows(None) == 0
    assert sr._rows({}) == 0
    assert sr._rows(_page(0)) == 0


def test_expected_reads_the_feeds_own_count() -> None:
    assert sr._expected(_page(5, count=6183)) == 6183
    assert sr._expected(_page(5)) is None  # no pagination block -> unknown, not zero


def _run(tmp_path, monkeypatch, pages: list, year: int = 2016):
    """Drive scrape_year against a scripted page sequence in a temp cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SDV_PY_247_DELAY", "0")
    monkeypatch.setenv("SDV_PY_247_BACKOFF", "0")
    seq = list(pages)

    def _fetch(y, page, logger):
        return seq[page - 1] if page - 1 < len(seq) else None

    monkeypatch.setattr(sr, "_fetch_page", _fetch)
    import logging

    return sr.scrape_year(year, logger=logging.getLogger("t"))


def test_complete_scrape_writes_a_manifest(tmp_path, monkeypatch) -> None:
    full, last = sr.PAGE_SIZE, 13
    man = _run(tmp_path, monkeypatch, [_page(full, count=full + last), _page(last)])
    assert man["complete"] is True
    assert man["rows"] == full + last == man["expected_rows"]
    assert sr.is_complete(2016)


def test_truncated_scrape_is_not_marked_complete(tmp_path, monkeypatch) -> None:
    """A short page is NOT proof of completion when the feed says otherwise.

    This is the whole point of reading `pagination.count`: a truncated feed
    also ends with a short page, so inferring "done" from page length would
    publish a partial class that looks healthy.
    """
    man = _run(tmp_path, monkeypatch, [_page(sr.PAGE_SIZE, count=99_999), _page(10)])
    assert man["complete"] is False
    assert man["rows"] == sr.PAGE_SIZE + 10
    assert man["expected_rows"] == 99_999
    assert not sr.is_complete(2016), "no manifest may exist for an incomplete year"


def test_failed_page_leaves_year_incomplete(tmp_path, monkeypatch) -> None:
    """An exhausted-retry page must not be silently dropped from the class."""
    man = _run(tmp_path, monkeypatch, [_page(sr.PAGE_SIZE, count=10_000), None])
    assert man["complete"] is False
    assert man["failed_pages"] == [2]
    assert not sr.is_complete(2016)


def test_empty_feed_is_never_complete(tmp_path, monkeypatch) -> None:
    """Zero recruits is a fetch failure -- a recruiting class is never empty."""
    man = _run(tmp_path, monkeypatch, [_page(0, count=0)])
    assert man["complete"] is False
    assert man["rows"] == 0
    assert not sr.is_complete(2016)


def test_pages_are_written_with_the_raw_payload(tmp_path, monkeypatch) -> None:
    """The store keeps the UNPARSED payload so a parser fix replays offline."""
    _run(tmp_path, monkeypatch, [_page(3, count=3)])
    stored = json.loads(
        (sr.year_dir(2016) / "page_0001.json").read_text(encoding="utf-8")
    )
    assert stored["year"] == 2016 and stored["page"] == 1
    assert stored["payload"]["players"] == [{"key": 0}, {"key": 1}, {"key": 2}]
    assert "pagination" in stored["payload"]


def test_page_size_stays_within_the_measured_serving_limit() -> None:
    """500 raises curl(28) against the 3s client budget; 250 is measured-good."""
    assert sr.PAGE_SIZE <= 250, sr.PAGE_SIZE


@pytest.mark.parametrize("bad", ["", "abc", None])
def test_env_overrides_ignore_unusable_values(monkeypatch, bad) -> None:
    if bad is None:
        monkeypatch.delenv("SDV_PY_247_RETRIES", raising=False)
    else:
        monkeypatch.setenv("SDV_PY_247_RETRIES", bad)
    assert sr._env_num("SDV_PY_247_RETRIES", 3, int) == 3
