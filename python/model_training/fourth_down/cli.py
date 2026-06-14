"""CLI for the fourth-down yards model training pipeline.

Usage:
    uv run python -m model_training.fourth_down.cli train \\
        --input-parquet pbp.parquet \\
        --output-model python/model_training/fourth_down/artifacts/fd_model.ubj

    uv run python -m model_training.fourth_down.cli validate \\
        --model fd_model.ubj \\
        --input-parquet pbp.parquet

    uv run python -m model_training.fourth_down.cli figures \\
        --results fd_calibration.csv \\
        --output-dir figures/

Direct final-dir mode (reads raw final.json plays):
    uv run python -m model_training.fourth_down.cli train \\
        --final-dir cfb/json/final \\
        --output-model fd_model.ubj \\
        --seasons 2018 2019 2020
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fourth_down_train",
        description="Fourth-down yards-gained model training pipeline.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- train subcommand ---
    tr = sub.add_parser("train", help="Train the fourth-down yards-gained model.")
    _grp = tr.add_mutually_exclusive_group()
    _grp.add_argument(
        "--input-parquet",
        default=None,
        help="Path to a pre-built plays parquet (from model_training ingest).",
    )
    _grp.add_argument(
        "--final-dir",
        default=None,
        help="Directory containing final.json play files (alternative to --input-parquet).",
    )
    tr.add_argument(
        "--output-model",
        required=True,
        help="Output path for the trained fd_model.ubj.",
    )
    tr.add_argument(
        "--seasons",
        nargs="*",
        type=int,
        default=None,
        help="Seasons to include when reading from --final-dir (default: all available).",
    )
    tr.add_argument(
        "--nrounds",
        type=int,
        default=None,
        help="Override boosting rounds (default: 157).",
    )
    tr.add_argument(
        "--validate",
        action="store_true",
        help="Run structure assert after training.",
    )

    # --- validate subcommand ---
    vl = sub.add_parser("validate", help="Validate a trained model.")
    vl.add_argument("--model", required=True, help="Path to fd_model.ubj.")
    vl.add_argument(
        "--input-parquet",
        required=True,
        help="Path to plays parquet for evaluation.",
    )

    # --- figures subcommand ---
    fg = sub.add_parser("figures", help="Generate calibration and feature-importance figures.")
    fg.add_argument("--results", required=True, help="Path to fd_calibration.csv.")
    fg.add_argument("--output-dir", required=True, help="Directory to write PNGs + CSVs.")

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "train":
        import polars as pl
        from .constants import FD_NROUNDS
        from .train import train_fd_model

        nrounds = args.nrounds or FD_NROUNDS

        if getattr(args, "input_parquet", None):
            plays = pl.read_parquet(args.input_parquet)
        elif getattr(args, "final_dir", None):
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
                play_list = obj.get("plays") or []
                if play_list:
                    frames.append(pl.DataFrame(play_list, infer_schema_length=None))
            if not frames:
                print("ERROR: No plays found. Check --final-dir and --seasons.")
                return 1
            plays = pl.concat(frames, how="diagonal_relaxed")
            print(f"Loaded {plays.height} plays from {len(frames)} games.")
        else:
            print("ERROR: Provide either --input-parquet or --final-dir.")
            return 1

        out = Path(args.output_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        model = train_fd_model(plays, output_path=str(out), nrounds=nrounds)
        print(
            f"Saved fd_model -> {out} "
            f"({model.num_boosted_rounds()} rounds, {model.num_features()} features)"
        )

        if args.validate:
            from .validate import assert_structure
            assert_structure(model)
            print("Structure assert passed.")

    elif args.cmd == "validate":
        import polars as pl
        import xgboost as xgb
        from .validate import assert_structure, calibration_fd
        from .features import derive_fd_features
        from .constants import FD_FEATURES
        from .train import _filter_plays

        model = xgb.Booster()
        model.load_model(args.model)
        assert_structure(model)
        print("Structure assert passed.")

        plays = pl.read_parquet(args.input_parquet)
        filtered = _filter_plays(plays)
        enriched = derive_fd_features(filtered)
        X = enriched.select(FD_FEATURES).to_pandas()
        y_yards = filtered["yardsGained"].to_numpy()

        cal = calibration_fd(model, X, y_yards)
        print(cal.to_string())

    elif args.cmd == "figures":
        import pandas as pd
        from .figures import write_fd_figures

        cal_table = pd.read_csv(args.results)
        # minimal importance stub if no importance CSV available alongside
        importance_path = Path(args.results).parent / "fd_feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path)
        else:
            importance = pd.DataFrame({"Feature": ["(unavailable)"], "Gain": [1.0]})

        cal_error = float(
            abs(cal_table["pred_fd_prob"] - cal_table["empirical_fd_rate"]).mean()
        ) if "pred_fd_prob" in cal_table.columns else 0.0

        cal_png, imp_png = write_fd_figures(
            cal_table=cal_table,
            importance=importance,
            out_dir=args.output_dir,
            cal_error=cal_error,
        )
        print(f"Wrote: {cal_png}, {imp_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
