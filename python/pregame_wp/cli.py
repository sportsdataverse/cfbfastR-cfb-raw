"""CLI for the Pregame WP + Five-Factors pipeline (Track 4).

Subcommands:
  ingest    — Fetch CFBD data (requires CFB_DATA_API_KEY env var).
  train     — Build box scores and train the XGBRegressor (offline).
  validate  — Compute MAE/RMSE/Brier on held-out data (offline).
  predict   — Predict WP for a single matchup (offline if model is on disk).
  figures   — Generate diagnostic scatter plot (offline).

Usage:
    python -m pregame_wp.cli ingest --season 2019
    python -m pregame_wp.cli train --input boxes.parquet --model pregame_wp/models/pgwp.ubj
    python -m pregame_wp.cli validate --input boxes.parquet --model pregame_wp/models/pgwp.ubj
    python -m pregame_wp.cli predict --model pgwp.ubj --five-factor-diff 2.1 --std 7.5
    python -m pregame_wp.cli figures --input boxes.parquet --model pgwp.ubj --out scatter.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """CFB Pregame Win Probability + Five-Factors modeling pipeline."""


# ---------------------------------------------------------------------------
# ingest subcommand (requires CFB_DATA_API_KEY)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--season", required=True, type=int, help="CFB season year (e.g. 2019).")
@click.option(
    "--out",
    default="data/cfbd",
    show_default=True,
    help="Output directory for CFBD data files.",
)
def ingest(season: int, out: str) -> None:
    """Fetch CFBD team game stats for SEASON and save to OUT/.

    Requires CFB_DATA_API_KEY environment variable.
    """
    api_key = os.environ.get("CFB_DATA_API_KEY")
    if not api_key:
        click.echo(
            "ERROR: CFB_DATA_API_KEY environment variable is not set. "
            "Set it and re-run: export CFB_DATA_API_KEY=<your_key>",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Fetching CFBD team game stats for season {season}...")

    from pregame_wp.data_prep import load_cfbd_data

    df = load_cfbd_data(season=season, api_key=api_key)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"team_game_stats_{season}.parquet"
    df.write_parquet(str(out_file))

    click.echo(f"Saved {len(df)} rows to {out_file}")


# ---------------------------------------------------------------------------
# train subcommand (offline)
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    help="Path to parquet/CSV with columns '5FRDiff' and 'PtsDiff'.",
)
@click.option(
    "--model",
    "model_path",
    default="python/pregame_wp/models/pgwp_model.ubj",
    show_default=True,
    help="Output path for the trained .ubj model file.",
)
@click.option(
    "--std-out",
    "std_path",
    default=None,
    help="Optional: write std to a text file for later use.",
)
def train(input_path: str, model_path: str, std_path: str | None) -> None:
    """Train the XGBRegressor on 5FRDiff → PtsDiff after outlier removal.

    Works entirely offline — no CFBD API key required.
    """
    import polars as pl

    from pregame_wp.model import save_model, train_pregame_model_with_stats

    p = Path(input_path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(str(p))
    else:
        df = pl.read_csv(str(p))

    click.echo(f"Loaded {len(df)} rows from {input_path}")
    model, mu, std = train_pregame_model_with_stats(df)
    click.echo(f"Trained model: n_estimators={model.n_estimators}, mu={mu:.4f}, std={std:.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)
    click.echo(f"Saved model to {model_path}")

    if std_path:
        Path(std_path).write_text(f"{std}\n")
        click.echo(f"Saved std={std:.6f} to {std_path}")


# ---------------------------------------------------------------------------
# validate subcommand (offline)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_path", required=True, help="Path to box-score parquet/CSV.")
@click.option("--model", "model_path", required=True, help="Path to trained .ubj model file.")
@click.option("--std", "std_val", required=True, type=float, help="Training std (from train step).")
def validate(input_path: str, model_path: str, std_val: float) -> None:
    """Compute MAE, RMSE, and (if 'outcome' column present) Brier score.

    Works entirely offline.
    """
    import polars as pl

    from pregame_wp.model import load_model
    from pregame_wp.validate import validate_model

    p = Path(input_path)
    df = pl.read_parquet(str(p)) if p.suffix == ".parquet" else pl.read_csv(str(p))

    model = load_model(model_path)
    metrics = validate_model(model, df, std=std_val)

    click.echo("Validation metrics:")
    for k, v in metrics.items():
        click.echo(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# predict subcommand (offline)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "model_path", required=True, help="Path to trained .ubj model file.")
@click.option("--five-factor-diff", "ffr_diff", required=True, type=float,
              help="5FRDiff = home_5FR - away_5FR.")
@click.option("--std", "std_val", required=True, type=float,
              help="Training std for WP CDF normalization.")
def predict(model_path: str, ffr_diff: float, std_val: float) -> None:
    """Predict pregame win probability for a matchup from 5FRDiff and training std.

    Works entirely offline.
    """
    from pregame_wp.model import load_model
    from pregame_wp.wp import pregame_wp

    model = load_model(model_path)
    wp = pregame_wp(model, ffr_diff, std_val)
    away_wp = 1.0 - wp

    click.echo(f"5FRDiff: {ffr_diff:+.4f}")
    click.echo(f"Home WP: {wp:.4f} ({wp*100:.1f}%)")
    click.echo(f"Away WP: {away_wp:.4f} ({away_wp*100:.1f}%)")


# ---------------------------------------------------------------------------
# figures subcommand (offline)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "input_path", required=True, help="Path to box-score parquet/CSV.")
@click.option("--model", "model_path", required=True, help="Path to trained .ubj model file.")
@click.option("--out", "out_path", default="scatter.png", show_default=True,
              help="Output figure path (PNG/SVG).")
def figures(input_path: str, model_path: str, out_path: str) -> None:
    """Generate scatter plot of predicted vs actual point differential.

    Works entirely offline. Requires matplotlib (pip install matplotlib).
    """
    import polars as pl

    from pregame_wp.figures import scatter_from_df
    from pregame_wp.model import load_model

    p = Path(input_path)
    df = pl.read_parquet(str(p)) if p.suffix == ".parquet" else pl.read_csv(str(p))

    model = load_model(model_path)
    scatter_from_df(model, df, output_path=out_path)
    click.echo(f"Saved figure to {out_path}")


if __name__ == "__main__":
    cli()
