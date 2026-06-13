"""Phase 6 — cli module tests (offline, no filesystem writes required)."""
from __future__ import annotations

import argparse

import pytest


def test_cli_module_importable():
    import cpoe.cli  # noqa: F401


def test_build_parser_returns_argumentparser():
    from cpoe.cli import build_parser
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_has_raw_dir_arg():
    from cpoe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--raw-dir", "/some/path", "--out-dir", "/out"])
    assert args.raw_dir == "/some/path"


def test_parser_has_out_dir_arg():
    from cpoe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--raw-dir", "/some/path", "--out-dir", "/out"])
    assert args.out_dir == "/out"


def test_parser_has_seasons_arg():
    from cpoe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "--raw-dir", "/r", "--out-dir", "/o",
        "--seasons", "2021", "2022", "2023",
    ])
    assert args.seasons == [2021, 2022, 2023]


def test_parser_loso_flag_defaults_false():
    from cpoe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--raw-dir", "/r", "--out-dir", "/o"])
    assert args.loso is False


def test_parser_loso_flag_sets_true():
    from cpoe.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["--raw-dir", "/r", "--out-dir", "/o", "--loso"])
    assert args.loso is True


def test_main_importable():
    from cpoe.cli import main  # noqa: F401
