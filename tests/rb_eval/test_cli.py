import subprocess
import sys

from rb_eval.cli import build_parser


def test_subcommands_present():
    p = build_parser()
    subcommand_names = set(p._subparsers._group_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected = {"features", "aggregate", "train", "validate", "figures"}
    assert expected <= subcommand_names


def test_help_exits_zero():
    import os
    import pathlib
    python_dir = str(pathlib.Path(__file__).resolve().parents[2] / "python")
    env = {**os.environ, "PYTHONPATH": python_dir}
    result = subprocess.run(
        [sys.executable, "-m", "rb_eval", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "features" in result.stdout
