"""CLI for the Pregame WP + Five-Factors pipeline (Track 4).

Subcommands:
  ingest    -- Fetch CFBD data (requires CFB_DATA_API_KEY env var).
  train     -- Build box scores and train the XGBRegressor (offline).
  validate  -- Compute MAE/RMSE/Brier on held-out data (offline).
  predict   -- Predict WP for a single matchup (offline if model is on disk).
  figures   -- Generate diagnostic scatter plot (offline).

Usage:
    python -m pregame_wp ingest --season 2019
    python -m pregame_wp train --input boxes.parquet --model pregame_wp/models/pgwp.ubj
    python -m pregame_wp validate --input boxes.parquet --model pregame_wp/models/pgwp.ubj --std 7.5
    python -m pregame_wp predict --model pgwp.ubj --five-factor-diff 2.1 --std 7.5
    python -m pregame_wp figures --input boxes.parquet --model pgwp.ubj --out scatter.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pregame_wp",
        description="CFB Pregame Win Probability + Five-Factors modeling pipeline.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Fetch CFBD team game stats (requires CFB_DATA_API_KEY).")
    p_ingest.add_argument("--season", required=True, type=int, help="CFB season year (e.g. 2019).")
    p_ingest.add_argument("--out", default="data/cfbd", help="Output directory for CFBD data files.")

    # train
    p_train = sub.add_parser("train", help="Train XGBRegressor on 5FRDiff -> PtsDiff (offline).")
    p_train.add_argument("--input", dest="input_path", required=True,
                         help="Path to parquet/CSV with columns '5FRDiff' and 'PtsDiff'.")
    p_train.add_argument("--model", dest="model_path",
                         default="python/pregame_wp/models/pgwp_model.ubj",
                         help="Output path for the trained .ubj model file.")
    p_train.add_argument("--std-out", dest="std_path", default=None,
                         help="Optional: write std to a text file for later use.")

    # validate
    p_val = sub.add_parser("validate", help="Compute MAE/RMSE/Brier on held-out data (offline).")
    p_val.add_argument("--input", dest="input_path", required=True,
                       help="Path to box-score parquet/CSV.")
    p_val.add_argument("--model", dest="model_path", required=True,
                       help="Path to trained .ubj model file.")
    p_val.add_argument("--std", dest="std_val", required=True, type=float,
                       help="Training std (from train step).")

    # predict
    p_pred = sub.add_parser("predict", help="Predict pregame WP for a single matchup (offline).")
    p_pred.add_argument("--model", dest="model_path", required=True,
                        help="Path to trained .ubj model file.")
    p_pred.add_argument("--five-factor-diff", dest="ffr_diff", required=True, type=float,
                        help="5FRDiff = home_5FR - away_5FR.")
    p_pred.add_argument("--std", dest="std_val", required=True, type=float,
                        help="Training std for WP CDF normalization.")

    # figures
    p_fig = sub.add_parser("figures", help="Generate scatter plot of predicted vs actual (offline).")
    p_fig.add_argument("--input", dest="input_path", required=True,
                       help="Path to box-score parquet/CSV.")
    p_fig.add_argument("--model", dest="model_path", required=True,
                       help="Path to trained .ubj model file.")
    p_fig.add_argument("--out", dest="out_path", default="scatter.png",
                       help="Output figure path (PNG/SVG).")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "ingest":
        api_key = os.environ.get("CFB_DATA_API_KEY")
        if not api_key:
            print(
                "ERROR: CFB_DATA_API_KEY environment variable is not set. "
                "Set it and re-run: export CFB_DATA_API_KEY=<your_key>",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Fetching CFBD team game stats for season {args.season}...")
        from pregame_wp.data_prep import load_cfbd_data

        df = load_cfbd_data(season=args.season, api_key=api_key)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"team_game_stats_{args.season}.parquet"
        df.write_parquet(str(out_file))
        print(f"Saved {len(df)} rows to {out_file}")

    elif args.cmd == "train":
        import polars as pl
        from pregame_wp.model import save_model, train_pregame_model_with_stats

        p = Path(args.input_path)
        df = pl.read_parquet(str(p)) if p.suffix == ".parquet" else pl.read_csv(str(p))
        print(f"Loaded {len(df)} rows from {args.input_path}")

        model, mu, std = train_pregame_model_with_stats(df)
        print(f"Trained model: n_estimators={model.n_estimators}, mu={mu:.4f}, std={std:.4f}")

        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        save_model(model, args.model_path)
        print(f"Saved model to {args.model_path}")

        if args.std_path:
            Path(args.std_path).write_text(f"{std}\n")
            print(f"Saved std={std:.6f} to {args.std_path}")

    elif args.cmd == "validate":
        import polars as pl
        from pregame_wp.model import load_model
        from pregame_wp.validate import validate_model

        p = Path(args.input_path)
        df = pl.read_parquet(str(p)) if p.suffix == ".parquet" else pl.read_csv(str(p))
        model = load_model(args.model_path)
        metrics = validate_model(model, df, std=args.std_val)

        print("Validation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    elif args.cmd == "predict":
        from pregame_wp.model import load_model
        from pregame_wp.wp import pregame_wp

        model = load_model(args.model_path)
        wp = pregame_wp(model, args.ffr_diff, args.std_val)
        away_wp = 1.0 - wp
        print(f"5FRDiff: {args.ffr_diff:+.4f}")
        print(f"Home WP: {wp:.4f} ({wp*100:.1f}%)")
        print(f"Away WP: {away_wp:.4f} ({away_wp*100:.1f}%)")

    elif args.cmd == "figures":
        import polars as pl
        from pregame_wp.figures import scatter_from_df
        from pregame_wp.model import load_model

        p = Path(args.input_path)
        df = pl.read_parquet(str(p)) if p.suffix == ".parquet" else pl.read_csv(str(p))
        model = load_model(args.model_path)
        scatter_from_df(model, df, output_path=args.out_path)
        print(f"Saved figure to {args.out_path}")


if __name__ == "__main__":
    main()
