"""Win probability inference: XGBRegressor prediction → normal-CDF WP.

Port of win-prob.ipynb cells 66–80.
"""
from __future__ import annotations

from scipy.stats import norm


def five_fr_to_wp(predicted_mov: float, mu: float, std: float) -> float:
    """Convert a predicted margin-of-victory to a win probability.

    Args:
        predicted_mov: Model output (5FRDiff → predicted PtsDiff).
        mu: Mean of the training prediction distribution (0.0 per OQ-7).
        std: Std dev of training predictions (full-set, per OQ-7).

    Returns:
        Win probability in (0, 1).
    """
    if std <= 0:
        return 0.5
    return float(norm.cdf((predicted_mov - mu) / std))


def generate_win_prob(
    fr_diff: float,
    model,
    mu: float,
    std: float,
    hfa: float = 0.0,
) -> float:
    """Generate WP from a 5FRDiff value + trained model + normalization params.

    Args:
        fr_diff: Pre-computed 5FRDiff (home - away).
        model: Fitted XGBRegressor.
        mu: Normalization mu (0.0 per OQ-7).
        std: Normalization std.
        hfa: Home-field advantage adjustment in points (added to predicted MOV).

    Returns:
        Win probability for the home team in (0, 1).
    """
    import numpy as np
    X = np.array([[fr_diff]])
    predicted_mov = float(model.predict(X)[0]) + hfa
    return five_fr_to_wp(predicted_mov, mu=mu, std=std)
