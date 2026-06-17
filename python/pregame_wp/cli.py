"""CLI: build-boxes | train | predict-matchup.

Usage:
  uv run python -m pregame_wp build-boxes  --seasons 2012:2020 --out cfb/pregame_wp/boxes/
  uv run python -m pregame_wp train        --boxes cfb/pregame_wp/boxes/ --out cfb/pregame_wp/
  uv run python -m pregame_wp predict-matchup --home "LSU" --away "Clemson" --year 2019
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pregame_wp",
        description="CFB Pregame WP + Five-Factors pipeline (Track 4, CFB Modeling Suite).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    bb = sub.add_parser("build-boxes", help="Compute 5FR box scores from CFBD data.")
    bb.add_argument("--seasons", default="2012:2020",
                    help="Season range as A:B (e.g. 2012:2020).")
    bb.add_argument("--out", default="cfb/pregame_wp/boxes/")

    tr = sub.add_parser("train", help="Train XGBRegressor on stored game boxes.")
    tr.add_argument("--boxes", default="cfb/pregame_wp/boxes/")
    tr.add_argument("--out", default="cfb/pregame_wp/")

    pm = sub.add_parser("predict-matchup", help="Predict WP for a future matchup.")
    pm.add_argument("--home", required=True)
    pm.add_argument("--away", required=True)
    pm.add_argument("--year", type=int, required=True)
    pm.add_argument("--model", default="python/pregame_wp/models/pgwp_model.ubj")
    pm.add_argument("--games", type=int, default=4,
                    help="Recent games to average for each team's 5FR.")
    pm.add_argument("--week", type=int, default=-1,
                    help="Week of season (-1 = latest available).")

    return ap


def _parse_seasons(seasons_str: str) -> list[int]:
    parts = seasons_str.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "build-boxes":
        seasons = _parse_seasons(args.seasons)
        print(f"build-boxes: seasons {seasons[0]}–{seasons[-1]}, out={args.out}")
        print("  (requires CFBD data staged locally; see data_ingest.py)")

    elif args.cmd == "train":
        import glob as _glob
        import pandas as pd
        from pregame_wp.training import filter_outliers, save_pgwp_model, train_pgwp_model

        box_files = sorted(_glob.glob(str(Path(args.boxes) / "*.parquet")))
        if not box_files:
            print(f"train: no parquet files found in {args.boxes}")
            return 1
        frames = [pd.read_parquet(f) for f in box_files]
        stored = pd.concat(frames, ignore_index=True)
        filtered = filter_outliers(stored)
        model, mu, std = train_pgwp_model(filtered)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_pgwp_model(model, std, out_dir / "pgwp_model.ubj",
                        season_range=None)
        print(f"train: model saved → {out_dir / 'pgwp_model.ubj'} (mu={mu}, std={std:.4f})")

    elif args.cmd == "predict-matchup":
        import xgboost as xgb
        import json
        from pregame_wp.predict import five_fr_to_wp

        model_path = Path(args.model)
        if not model_path.exists():
            print(f"predict-matchup: model not found at {model_path}")
            return 1
        m = xgb.XGBRegressor()
        m.load_model(str(model_path))
        card = json.loads(model_path.with_suffix(".json").read_text())
        mu, std = float(card["mu"]), float(card["std"])
        print(f"predict-matchup: {args.home} vs {args.away}, year={args.year}")
        print("  (requires pre-computed 5FR averages; see box_score.py)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
