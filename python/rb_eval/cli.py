"""CLI: features | aggregate | train | validate | figures.

Usage:
  uv run python -m rb_eval features   --final-dir cfb/json/final --out cfb/rb_eval/rush_plays.parquet
  uv run python -m rb_eval aggregate  --plays cfb/rb_eval/rush_plays.parquet --out cfb/rb_eval/rusher_seasons.parquet
  uv run python -m rb_eval train      --seasons cfb/rb_eval/rusher_seasons.parquet --out cfb/rb_eval/
  uv run python -m rb_eval validate   --loso cfb/rb_eval/xrepa_loso.parquet --out cfb/rb_eval/
  uv run python -m rb_eval figures    --table cfb/rb_eval/calibration.parquet --out cfb/rb_eval/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rb_eval",
        description="CFB RB-Eval xREPA pipeline (Track 3, CFB Modeling Suite).",
    )
    ap.add_argument(
        "--seasons",
        default=None,
        help="Season range as A:B (e.g. 2006:2025); default = all available.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features", help="Load rush plays from final.json and compute fo_success.")
    f.add_argument("--final-dir", default="cfb/json/final")
    f.add_argument("--out", default="cfb/rb_eval/rush_plays.parquet")

    a = sub.add_parser("aggregate", help="Aggregate to per-rusher-season, lag, weight.")
    a.add_argument("--plays", default="cfb/rb_eval/rush_plays.parquet")
    a.add_argument("--out", default="cfb/rb_eval/rusher_seasons.parquet")

    t = sub.add_parser("train", help="Fit LinearGAM and run LOSO CV.")
    t.add_argument("--seasons", dest="seasons_parquet",
                   default="cfb/rb_eval/rusher_seasons.parquet")
    t.add_argument("--out", default="cfb/rb_eval/")

    v = sub.add_parser("validate", help="Compute calibration table and metrics.")
    v.add_argument("--loso", default="cfb/rb_eval/xrepa_loso.parquet")
    v.add_argument("--out", default="cfb/rb_eval/")

    fi = sub.add_parser("figures", help="Produce calibration PNG + data table.")
    fi.add_argument("--table", default="cfb/rb_eval/calibration.parquet")
    fi.add_argument("--out", default="cfb/rb_eval/")

    return ap


def _parse_seasons(seasons_str: str | None) -> list[int] | None:
    if seasons_str is None:
        return None
    parts = seasons_str.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seasons = _parse_seasons(getattr(args, "seasons", None))

    if args.cmd == "features":
        import polars as pl
        from rb_eval.features import load_rush_plays
        df = load_rush_plays(args.final_dir, seasons=seasons)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(args.out)
        print(f"features: wrote {df.height} rush plays -> {args.out}")

    elif args.cmd == "aggregate":
        import polars as pl
        from rb_eval.aggregate import build_rusher_seasons
        rush_df = pl.read_parquet(args.plays)
        seasons_df = build_rusher_seasons(rush_df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        seasons_df.write_parquet(args.out)
        print(f"aggregate: wrote {seasons_df.height} rusher-season rows -> {args.out}")

    elif args.cmd == "train":
        import polars as pl
        from rb_eval.aggregate import build_model_data
        from rb_eval.train import loso_cv, save_model, train_xrepa
        seasons_df = pl.read_parquet(args.seasons_parquet)
        model_data = build_model_data(seasons_df)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv = loso_cv(model_data)
        cv.write_parquet(out_dir / "xrepa_loso.parquet")
        print(f"train: LOSO predictions ({cv.height} rows) -> {out_dir / 'xrepa_loso.parquet'}")
        gam = train_xrepa(model_data)
        card = save_model(
            gam,
            out_dir / "xrepa_final.pkl",
            season_range=(int(model_data["season"].min()), int(model_data["season"].max())),
            n_rushers=model_data.height,
        )
        print(f"train: saved full-data GAM -> {out_dir / 'xrepa_final.pkl'}")
        print(f"train: model card -> {card}")

    elif args.cmd == "validate":
        import polars as pl
        from rb_eval.validate import calibration_table, weighted_cal_error, weighted_r2
        cv = pl.read_parquet(args.loso)
        table = calibration_table(cv)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out_dir / "calibration.parquet")
        table.write_csv(out_dir / "calibration.csv")
        err = weighted_cal_error(table)
        r2 = weighted_r2(table)
        print(f"validate: weighted cal error = {err:.6f}, weighted R² = {r2:.4f}")
        print(f"validate: calibration table -> {out_dir / 'calibration.parquet'}")

    elif args.cmd == "figures":
        import polars as pl
        from rb_eval.figures import write_xrepa_calibration
        from rb_eval.validate import weighted_cal_error, weighted_r2
        table = pl.read_parquet(args.table)
        out_dir = Path(args.out)
        err = weighted_cal_error(table)
        r2 = weighted_r2(table)
        png, csv = write_xrepa_calibration(
            table, out_dir / "xrepa_calibration", cal_error=err, r2=r2
        )
        print(f"figures: wrote {png}, {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
