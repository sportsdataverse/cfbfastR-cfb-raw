"""CLI: ingest | train-ep | train-wp | train-qbr | validate | figures."""
from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import add_winner, build_training_frame, write_training_frame  # noqa: F401


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="model_training")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--final-dir", default="cfb/json/final")
    i.add_argument("--out", default="pbp_full.parquet")
    i.add_argument("--seasons", nargs="*", type=int)
    for name in ("train-ep", "train-wp", "train-qbr"):
        s = sub.add_parser(name)
        s.add_argument("--pbp", default="pbp_full.parquet")
        s.add_argument("--out", required=True)
        if name == "train-wp":
            s.add_argument("--variant", choices=["spread", "naive"], default="spread")
        if name == "train-qbr":
            s.add_argument("--espn-qbr", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--model", required=True)
    v.add_argument("--ref", required=True)
    f = sub.add_parser("figures")
    f.add_argument("--table", required=True)
    f.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ingest":
        n = write_training_frame(args.final_dir, args.out, args.seasons)
        print(f"wrote {n} rows -> {args.out}")
    elif args.cmd in ("train-ep", "train-wp", "train-qbr"):
        import polars as pl

        df = add_winner(pl.read_parquet(args.pbp))
        if args.cmd == "train-ep":
            from .train_ep import train_ep

            model = train_ep(df)
        elif args.cmd == "train-wp":
            from .train_wp import train_wp

            model = train_wp(df, variant=args.variant, stage=args.stage)
        else:
            from .train_qbr import train_qbr

            model = train_qbr(df, pl.read_parquet(args.espn_qbr))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(args.out)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
