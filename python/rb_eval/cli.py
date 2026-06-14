"""CLI: features | aggregate | train | validate | figures subcommands.

Usage:
  python -m rb_eval features   --final-dir cfb/json/final --out cfb/rb_eval/rush_plays.parquet
  python -m rb_eval aggregate  --plays cfb/rb_eval/rush_plays.parquet \
                               --out cfb/rb_eval/rusher_seasons.parquet
  python -m rb_eval train      --seasons cfb/rb_eval/rusher_seasons.parquet \
                               --out cfb/rb_eval/
  python -m rb_eval validate   --loso cfb/rb_eval/xrepa_loso.parquet --out cfb/rb_eval/
  python -m rb_eval figures    --table cfb/rb_eval/calibration.parquet --out cfb/rb_eval/

Season range: --season-range A:B  (e.g. 2006:2025); default = all available.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build and return the rb_eval argument parser with all subcommands."""
    ap = argparse.ArgumentParser(
        prog="rb_eval",
        description="CFB RB-Eval xREPA pipeline (Track 3, CFB Modeling Suite).",
    )
    ap.add_argument(
        "--season-range",
        default=None,
        metavar="A:B",
        help="Season range as A:B (inclusive, e.g. 2006:2025); default = all.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- features ---
    f = sub.add_parser(
        "features",
        help="Load rush plays from final.json and compute fo_success + epa_clamped.",
    )
    f.add_argument("--final-dir", default="cfb/json/final",
                   help="Directory containing game-level final.json files.")
    f.add_argument("--out", default="cfb/rb_eval/rush_plays.parquet",
                   help="Output parquet path for filtered rush plays.")

    # --- aggregate ---
    a = sub.add_parser(
        "aggregate",
        help="Aggregate to per-rusher-season rows, add lag and Pythagorean weight.",
    )
    a.add_argument("--plays", default="cfb/rb_eval/rush_plays.parquet",
                   help="Input parquet path from the features step.")
    a.add_argument("--out", default="cfb/rb_eval/rusher_seasons.parquet",
                   help="Output parquet path for per-rusher-season frame.")

    # --- train ---
    t = sub.add_parser(
        "train",
        help="Fit LinearGAM(s(0)+s(1)) and run LOSO CV.",
    )
    t.add_argument("--seasons", dest="seasons_parquet",
                   default="cfb/rb_eval/rusher_seasons.parquet",
                   help="Input parquet path from the aggregate step.")
    t.add_argument("--out", default="cfb/rb_eval/",
                   help="Output directory for xrepa_loso.parquet and xrepa_final.pkl.")

    # --- validate ---
    v = sub.add_parser(
        "validate",
        help="Compute calibration table, weighted cal-error, and weighted R².",
    )
    v.add_argument("--loso", default="cfb/rb_eval/xrepa_loso.parquet",
                   help="Input parquet path from the train step (LOSO CV output).")
    v.add_argument("--out", default="cfb/rb_eval/",
                   help="Output directory for calibration.parquet + calibration.csv.")

    # --- figures ---
    fi = sub.add_parser(
        "figures",
        help="Produce xREPA calibration PNG + data table.",
    )
    fi.add_argument("--table", default="cfb/rb_eval/calibration.parquet",
                    help="Input parquet path from the validate step.")
    fi.add_argument("--out", default="cfb/rb_eval/",
                    help="Output directory for xrepa_calibration.png + .csv.")

    return ap


def _parse_season_range(season_range: str | None) -> list[int] | None:
    """Parse --season-range A:B into a list of integers [A, A+1, ..., B]."""
    if season_range is None:
        return None
    parts = season_range.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(parts[0])]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the rb_eval CLI.

    Args:
        argv: argument list (defaults to sys.argv[1:] when None).

    Returns:
        Exit code (0 = success).
    """
    args = build_parser().parse_args(argv)
    seasons = _parse_season_range(getattr(args, "season_range", None))

    if args.cmd == "features":
        import polars as pl  # noqa: F401 — polars available for dtype work if needed
        from rb_eval.features import load_rush_plays

        df = load_rush_plays(args.final_dir, seasons=seasons)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        print(f"features: wrote {df.height} rush plays -> {out_path}")

    elif args.cmd == "aggregate":
        import polars as pl
        from rb_eval.features import aggregate_per_rusher, fo_success, clamp_epa
        from rb_eval.model import add_lag_features, add_weight

        rush_df = pl.read_parquet(args.plays)
        # Re-apply fo_success/clamp if columns absent (idempotent if already present)
        if "fo_success" not in rush_df.columns:
            rush_df = fo_success(rush_df)
        if "epa_clamped" not in rush_df.columns:
            rush_df = clamp_epa(rush_df)
        agg = aggregate_per_rusher(rush_df)
        lagged = add_lag_features(agg)
        weighted = add_weight(lagged)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        weighted.write_parquet(out_path)
        print(f"aggregate: wrote {weighted.height} rusher-season rows -> {out_path}")

    elif args.cmd == "train":
        import polars as pl
        from rb_eval.model import build_model_data, train_xrepa, loso_cv, save_model

        seasons_df = pl.read_parquet(args.seasons_parquet)
        model_data = build_model_data(seasons_df)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # LOSO CV
        cv = loso_cv(model_data)
        loso_path = out_dir / "xrepa_loso.parquet"
        cv.write_parquet(loso_path)
        print(f"train: wrote LOSO predictions ({cv.height} rows) -> {loso_path}")

        # Full-data model
        gam = train_xrepa(model_data)
        pkl_path = out_dir / "xrepa_final.pkl"
        card_path = save_model(
            gam,
            pkl_path,
            season_range=(
                int(model_data["season"].min()),
                int(model_data["season"].max()),
            ),
            n_rushers=model_data.height,
        )
        print(f"train: saved full-data GAM -> {pkl_path}")
        print(f"train: wrote model card -> {card_path}")

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
        from rb_eval.validate import weighted_cal_error, weighted_r2
        from rb_eval.figures import write_xrepa_calibration

        table = pl.read_parquet(args.table)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        err = weighted_cal_error(table)
        r2 = weighted_r2(table)
        png, csv = write_xrepa_calibration(
            table, out_dir / "xrepa_calibration", cal_error=err, r2=r2
        )
        print(f"figures: wrote {png}, {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
