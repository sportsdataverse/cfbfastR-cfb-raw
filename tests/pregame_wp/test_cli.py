import os
import pathlib
import subprocess
import sys

from pregame_wp.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    choices = set(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected = {"build-boxes", "train", "predict-matchup"}
    assert expected <= choices


def test_help_exits_zero():
    python_dir = str(pathlib.Path(__file__).resolve().parents[2] / "python")
    env = {**os.environ, "PYTHONPATH": python_dir}
    result = subprocess.run(
        [sys.executable, "-m", "pregame_wp", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "build-boxes" in result.stdout
