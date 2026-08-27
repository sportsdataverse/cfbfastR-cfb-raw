import json
from pathlib import Path
import importlib.util
import pandas as pd

UTILS = Path(__file__).parents[1] / "python" / "_cfb_raw_utils.py"
spec = importlib.util.spec_from_file_location("_cfb_raw_utils", UTILS)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_write_json_atomic_creates_dirs_and_no_tmp(tmp_path):
    target = tmp_path / "a" / "b" / "401.json"
    u.write_json_atomic({"id": 401, "x": [1, 2]}, target)
    assert target.exists()
    assert json.loads(target.read_text())["id"] == 401
    assert not list(target.parent.glob("*.tmp")), "temp file left behind"


def test_write_json_atomic_emits_valid_json_for_nan_inf(tmp_path):
    # Python's json writes bare NaN/Infinity (invalid JSON; R/JS/Go reject it).
    # write_json_atomic must coerce nan/inf -> null so cross-language consumers can parse.
    target = tmp_path / "g.json"
    obj = {"a": float("nan"), "b": float("inf"), "c": float("-inf"),
           "plays": [{"name": "x", "epa": float("nan"), "yds": 3.5}], "ok": 1}
    u.write_json_atomic(obj, target)
    raw = target.read_text()
    assert "NaN" not in raw and "Infinity" not in raw, "invalid JSON literal emitted"
    # strict parse (no NaN tolerance) must succeed
    loaded = json.loads(raw, parse_constant=_reject_constant)
    assert loaded["a"] is None and loaded["b"] is None and loaded["c"] is None
    assert loaded["plays"][0]["epa"] is None
    assert loaded["plays"][0]["yds"] == 3.5
    assert loaded["ok"] == 1


def _reject_constant(c):  # json.loads(parse_constant=) fires only for NaN/Infinity tokens
    raise AssertionError(f"non-standard JSON constant present: {c}")


def test_processing_version_format():
    v = u.PROCESSING_VERSION
    assert "+" in v and v.split("+")[1].isdigit()


def test_stamp_adds_identity():
    out = u.stamp({"k": 1}, game_id=401, season=2024, week=1)
    assert out["game_id"] == 401 and out["season"] == 2024 and out["week"] == 1
    assert out["k"] == 1


def test_stamp_list_wraps_with_meta():
    out = u.stamp([{"a": 1}], game_id=401, season=2024, week=1)
    assert out["game_id"] == 401
    assert out["data"] == [{"a": 1}]


def test_filter_undone_drops_existing(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402, 403], dir=str(final_dir), rescrape=False)
    assert out == [402, 403]


def test_filter_undone_rescrape_keeps_all(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402], dir=str(final_dir), rescrape=True)
    assert out == [401, 402]


def test_games_for_seasons_filters_range(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2, 3], "season": [2003, 2004, 2005]}).to_parquet(master)
    games = u.games_for_seasons(u.load_schedule_master(str(master)), 2004, 2005)
    assert sorted(games) == [2, 3]


def test_season_type_from_raw_variants():
    assert u.season_type_from_raw({"header": {"season": {"type": 2}}}) == 2
    assert u.season_type_from_raw({"header": {"season": {"type": {"id": "3"}}}}) == 3
    assert u.season_type_from_raw({"header": {"competitions": [{"type": {"id": 2}}]}}) == 2
    assert u.season_type_from_raw({"header": {}}) is None


def _big(n):
    """A payload whose serialized size is comfortably above the guard threshold."""
    return {"plays": [{"id": i, "text": "x" * 40} for i in range(n)]}


def test_write_json_guarded_refuses_to_clobber_with_a_degraded_payload(tmp_path):
    """ESPN 5xx/empty bodies collapse to a ~250-byte stub. Writing that over a
    banked 50-400KB summary is what destroyed 11 games in the 2004 pilot."""
    target = tmp_path / "401.json"
    u.write_json_atomic(_big(400), target)
    before = target.stat().st_size

    wrote = u.write_json_guarded({"header": {}, "boxscore": {}}, target)

    assert wrote is False, "degraded write should be refused"
    assert target.stat().st_size == before, "banked copy must be untouched"
    assert len(json.loads(target.read_text())["plays"]) == 400


def test_write_json_guarded_allows_growth_and_modest_change(tmp_path):
    """Real content changes must pass through -- the guard only blocks collapse."""
    target = tmp_path / "401.json"
    u.write_json_atomic(_big(100), target)

    assert u.write_json_guarded(_big(400), target) is True, "growth must be allowed"
    assert len(json.loads(target.read_text())["plays"]) == 400

    # shrinking to 80% of current is a plausible real edit, not a collapse
    assert u.write_json_guarded(_big(330), target) is True
    assert len(json.loads(target.read_text())["plays"]) == 330


def test_write_json_guarded_writes_when_no_existing_file(tmp_path):
    """First write of a game has nothing to protect."""
    target = tmp_path / "sub" / "401.json"
    assert u.write_json_guarded({"id": 401}, target) is True
    assert json.loads(target.read_text())["id"] == 401


def test_write_json_guarded_replaces_an_existing_empty_stub(tmp_path):
    """The recovery direction: a good payload must overwrite a prior stub."""
    target = tmp_path / "401.json"
    u.write_json_atomic({}, target)
    assert u.write_json_guarded(_big(400), target) is True
    assert len(json.loads(target.read_text())["plays"]) == 400


def _final(final_dir, game_id, count, state):
    """Bank a final with the two fields the shell test reads."""
    doc = {"count": count}
    if state is not None:
        doc["header"] = {"competitions": [{"status": {"type": {"state": state}}}]}
    (final_dir / f"{game_id}.json").write_text(json.dumps(doc))


def test_status_state_reads_pre_in_post():
    for state in ("pre", "in", "post"):
        doc = {"header": {"competitions": [{"status": {"type": {"state": state}}}]}}
        assert u.status_state(doc) == state
    assert u.status_state({}) is None


def test_filter_undone_rescrapes_a_pregame_shell(tmp_path):
    """The 2026 blocker: a final banked before kickoff must NOT count as scraped.

    946 of these (one per unplayed 2026 game) were banked by the 2026-08-02
    reprocess. With the old existence-only check every 2026 game was skipped for
    the whole season while the job reported green.
    """
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _final(final_dir, 401, count=0, state="pre")
    assert u.filter_undone([401], dir=str(final_dir), rescrape=False) == [401]


def test_filter_undone_settles_a_finished_game_with_no_plays(tmp_path):
    """A finished game with no ESPN pbp source legitimately banks count == 0.

    Real example: 242410193. A naive "zero plays -> undone" rule would re-scrape
    every one of these on every daily run, forever.
    """
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _final(final_dir, 402, count=0, state="post")
    assert u.filter_undone([402], dir=str(final_dir), rescrape=False) == []


def test_filter_undone_keeps_a_real_scrape_done(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _final(final_dir, 403, count=150, state="post")
    assert u.filter_undone([403], dir=str(final_dir), rescrape=False) == []


def test_filter_undone_treats_an_unreadable_final_as_undone(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "404.json").write_text("{not json")
    assert u.filter_undone([404], dir=str(final_dir), rescrape=False) == [404]


def test_status_state_survives_a_null_header():
    """`.get(k, {})` returns None when the key EXISTS and is null, so a null
    header used to raise AttributeError out of filter_undone and abort the pass."""
    assert u.status_state({"header": None}) is None
    assert u.status_state({"header": {"competitions": None}}) is None
    assert u.status_state({"header": {"competitions": [None]}}) is None


def test_filter_undone_survives_a_malformed_final(tmp_path):
    """Neither case may raise -- an escaping error aborts the whole filter pass,
    which is the failure mode this module exists to prevent.

    They resolve differently on purpose:

    * a non-numeric `count` cannot be evaluated at all -> schedule the game.
    * a null header is merely missing the pre-game SIGNAL. Absence of evidence is
      not evidence of a shell, so it is left alone -- the same answer `{}` has
      always got (see test_filter_undone_drops_existing).
    """
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "405.json").write_text(json.dumps({"count": "not-a-number"}))
    (final_dir / "406.json").write_text(json.dumps({"count": 0, "header": None}))
    assert u.filter_undone([405, 406], dir=str(final_dir), rescrape=False) == [405]
