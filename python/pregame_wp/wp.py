"""Win-probability calculation using the normal CDF (Track 4 — Pregame WP).

OQ-7 resolution: mu = 0.0 (point-differential is symmetric; each game has one
positive and one negative entry). std is derived from full training-set predictions
(not a test-split stat as in the original notebook).

WP = norm.cdf((model.predict([[5FRDiff]])[0] - mu) / std)
   = norm.cdf(pred / std)                        # with mu = 0.0
"""
from __future__ import annotations

from scipy.stats import norm
from xgboost import XGBRegressor

from .constants import WP_MU


def pregame_wp(
    model: XGBRegressor,
    five_factor_diff: float,
    std: float,
    mu: float = WP_MU,
) -> float:
    """Compute pregame win probability for the team with the given 5FRDiff.

    Args:
        model: Fitted XGBRegressor (single feature: 5FRDiff).
        five_factor_diff: home_5FR - away_5FR (positive favours home team).
        std: Standard deviation of training-set predictions (for CDF normalization).
        mu: Center of the normal distribution (default 0.0 per OQ-7).

    Returns:
        Win probability in [0, 1] for the team with the positive 5FRDiff.

    Example:
        Quick start::

            from pregame_wp.wp import pregame_wp
            wp = pregame_wp(model, five_factor_diff=2.1, std=7.5)
            print(f"Home WP: {wp:.3f}")
    """
    pred = float(model.predict([[five_factor_diff]])[0])
    return float(norm.cdf((pred - mu) / std))


def pregame_wp_from_pred(pred_mov: float, std: float, mu: float = WP_MU) -> float:
    """Compute WP directly from a predicted margin of victory.

    Useful when the model prediction is already available and you only need
    the CDF transformation.

    Args:
        pred_mov: Predicted margin of victory (positive = home team wins).
        std: Standard deviation from training-set predictions.
        mu: Center (default 0.0).

    Returns:
        Win probability in [0, 1].
    """
    return float(norm.cdf((pred_mov - mu) / std))
