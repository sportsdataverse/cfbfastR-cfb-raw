"""CLI for the fourth-down yards model training pipeline.

Usage:
    uv run python -m model_training.fourth_down.cli train-fd \\
        --final-dir cfb/json/final \\
        --out python/model_training/fourth_down/artifacts/fd_model.ubj \\
        --seasons 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fourth_down_train")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fd = sub.add_parser("train-fd", help="Train the fourth-down yards-gained model.")
    fd.add_argument("--final-dir", default="cfb/json/final", help="Directory containing final.json play files.")
    fd.add_argument("--out", required=True, help="Output path for the trained fd_model.ubj.")
    fd.add_argument("--seasons", nargs="*", type=int, default=None, help="Seasons to include (default: all).")
    fd.add_argument("--nrounds", type=int, default=None, help="Override nrounds (default: 157).")
    fd.add_argument("--validate", action="store_true", help="Run structure assert after training.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "train-fd":
        import polars as pl

        from .constants import FD_NROUNDS
        from .train import train_from_plays

        final_dir = Path(args.final_dir)
        if not final_dir.exists():
            print(f"ERROR: --final-dir {final_dir} does not exist.")
            return 1

        frames = []
        for fpath in sorted(final_dir.glob("*.json")):
            obj = json.loads(fpath.read_text())
            season = obj.get("season")
            if args.seasons is not None and season not in args.seasons:
                continue
            plays = obj.get("plays") or []
            if plays:
                frames.append(pl.DataFrame(plays, infer_schema_length=None))

        if not frames:
            print("ERROR: No plays found. Check --final-dir and --seasons.")
            return 1

        all_plays = pl.concat(frames, how="diagonal_relaxed")
        print(f"Loaded {all_plays.height} plays from {len(frames)} games.")

        nrounds = args.nrounds or FD_NROUNDS
        model = train_from_plays(all_plays, nrounds=nrounds)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out))
        print(f"Saved fd_model -> {out} ({model.num_boosted_rounds()} rounds, {model.num_features()} features)")

        if args.validate:
            from .validate import assert_structure

            assert_structure(model)
            print("Structure assert passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
