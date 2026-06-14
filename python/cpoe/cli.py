"""CLI: ingest | train | loso | predict | validate | figures.

Entry point: python -m cpoe.cli <subcommand> [options]
             or via the cpoe console_scripts entry point.

All subcommands follow the same data-flow:
  ingest  → cp_plays.parquet
  train   → cp_model.ubj          (from cp_plays.parquet)
  loso    → loso_cv.parquet       (from cp_plays.parquet; LOSO calibration)
  predict → cpoe_plays.parquet    (from cp_plays.parquet + cp_model.ubj)
  validate→ calibration.parquet   (from loso_cv.parquet)
  figures → figures/cp_*.png|csv  (from calibration.parquet)
"""
from __future__ import annotations

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cpoe",
        description="CFB CPOE pipeline — ingest | train | loso | predict | validate | figures",
    )
    ap.add_argument(
        "--approach",
        choices=["A", "B"],
        default="A",
        help="Feature approach: A (8 game-state features) or B (9, with CFBD air_yards)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- ingest ---
    i = sub.add_parser("ingest", help="Read final.json plays into a parquet training frame")
    i.add_argument("--final-dir", default="cfb/json/final",
                   help="Path to cfb/json/final/ directory")
    i.add_argument("--out", default="cfb/cpoe/cp_plays.parquet",
                   help="Output parquet path")
    i.add_argument("--seasons", nargs="*", type=int,
                   help="Season years to include (default: all)")

    # --- train ---
    t = sub.add_parser("train", help="Train the CP model from cp_plays.parquet")
    t.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet",
                   help="Input plays parquet")
    t.add_argument("--out", default="cfb/cpoe/cp_model.ubj",
                   help="Output model path (.ubj)")
    t.add_argument("--nrounds", type=int, default=400,
                   help="XGBoost boosting rounds")

    # --- loso ---
    lo = sub.add_parser("loso", help="Leave-one-season-out calibration CV")
    lo.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet")
    lo.add_argument("--out", default="cfb/cpoe/loso_cv.parquet")
    lo.add_argument("--nrounds", type=int, default=400)
    lo.add_argument("--seasons", nargs="*", type=int,
                    help="Seasons to include in LOSO loop (default: all)")

    # --- predict ---
    pr = sub.add_parser("predict", help="Add expected_completion + cpoe to plays")
    pr.add_argument("--plays", default="cfb/cpoe/cp_plays.parquet")
    pr.add_argument("--model", default="cfb/cpoe/cp_model.ubj")
    pr.add_argument("--out", default="cfb/cpoe/cpoe_plays.parquet")

    # --- validate ---
    v = sub.add_parser("validate", help="Compute calibration table from LOSO CV output")
    v.add_argument("--loso", default="cfb/cpoe/loso_cv.parquet")
    v.add_argument("--out", default="cfb/cpoe/calibration.parquet")

    # --- figures ---
    f = sub.add_parser("figures", help="Write calibration PNG + CSV from calibration table")
    f.add_argument("--calibration", default="cfb/cpoe/calibration.parquet")
    f.add_argument("--out-dir", default="cfb/cpoe/figures/")

    return ap


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import polars as pl
    import xgboost as xgb

    args = build_parser().parse_args(argv)
    approach = args.approach

    # ---- ingest ----
    if args.cmd == "ingest":
        from .ingest import build_cp_training_frame

        df = build_cp_training_frame(args.final_dir, seasons=args.seasons)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        print(f"wrote {df.height} plays -> {out}")

    # ---- train ----
    elif args.cmd == "train":
        from .train import train_cp_model

        df = pl.read_parquet(args.plays)
        model = train_cp_model(df, output_path=args.out, approach=approach,
                               nrounds=args.nrounds)
        print(f"saved model -> {args.out}")

    # ---- loso ----
    elif args.cmd == "loso":
        from .calibrate import loso_calibrate

        df = pl.read_parquet(args.plays)
        cv = loso_calibrate(df, seasons=args.seasons, approach=approach,
                            nrounds=args.nrounds)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv.write_parquet(out)
        print(f"LOSO CV done ({cv.height} rows) -> {out}")

    # ---- predict ----
    elif args.cmd == "predict":
        from .train import compute_cpoe

        df = pl.read_parquet(args.plays)
        model = xgb.Booster()
        model.load_model(args.model)
        out_df = compute_cpoe(df, model, approach=approach)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(out)
        print(f"wrote CPOE plays -> {out}")

    # ---- validate ----
    elif args.cmd == "validate":
        from .validate import calibration_table, distance_bucket, weighted_cal_error

        cv = pl.read_parquet(args.loso)
        # Ensure distance_bucket is present
        if "distance_bucket" not in cv.columns and "distance" in cv.columns:
            cv = cv.with_columns(distance_bucket(pl.col("distance")).alias("distance_bucket"))
        tbl = calibration_table(cv)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tbl.write_parquet(out)
        err = weighted_cal_error(tbl)
        print(f"Overall weighted calibration error: {err['overall']:.4f}")
        print(f"calibration table written -> {out}")

    # ---- figures ----
    elif args.cmd == "figures":
        from .figures import write_cp_calibration
        from .validate import weighted_cal_error

        tbl = pl.read_parquet(args.calibration)
        err_result = weighted_cal_error(tbl)
        overall_err = err_result["overall"]
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        png, csv = write_cp_calibration(
            tbl,
            stem=out_dir / "cp_calibration",
            cal_error=overall_err,
        )
        print(f"figures written -> {png}, {csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
