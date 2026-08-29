import json

# --- added after the 2027 class was found frozen at 4,779 rows -------------


def test_is_complete_requires_the_class_to_have_CLOSED():
    """A manifest certifies "we paged what existed then", not "the class is
    done". 2027 was banked 2026-08-06 at 4,779 rows against signed classes of
    5,678-5,952, and the manifest-only rule would have frozen it there."""
    from datetime import date

    import cfb_raw_scrape.scrape_cfb_recruits as sr

    # signed classes stay complete -- their banks must NOT be re-scraped
    assert sr.is_recruiting_class_final(2026, date(2026, 8, 29)) is True
    assert sr.is_recruiting_class_final(2024, date(2026, 8, 29)) is True
    # the open class must refresh
    assert sr.is_recruiting_class_final(2027, date(2026, 8, 29)) is False
    assert sr.is_recruiting_class_final(2028, date(2026, 8, 29)) is False
    # and it closes only after signing day passes
    assert sr.is_recruiting_class_final(2027, date(2027, 4, 1)) is True


def test_is_complete_does_not_invalidate_the_26_banked_years(tmp_path, monkeypatch):
    """The banked manifests carry expected: null (the field postdates them).
    Requiring rows==expected here would re-scrape 26 correct, closed classes."""
    from datetime import date

    import cfb_raw_scrape.scrape_cfb_recruits as sr

    monkeypatch.chdir(tmp_path)
    p = tmp_path / sr.year_dir(2024) / "_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rows": 5770, "expected": None}), encoding="utf-8")
    assert sr.is_complete(2024, today=date(2026, 8, 29)) is True


def test_an_open_class_is_incomplete_even_with_a_manifest(tmp_path, monkeypatch):
    from datetime import date

    import cfb_raw_scrape.scrape_cfb_recruits as sr

    monkeypatch.chdir(tmp_path)
    p = tmp_path / sr.year_dir(2027) / "_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rows": 4779, "expected": None}), encoding="utf-8")
    assert sr.is_complete(2027, today=date(2026, 8, 29)) is False
