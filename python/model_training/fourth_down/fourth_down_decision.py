"""Fourth-down decision layer: expected-value comparison of go/punt/FG.

STUB — not yet implemented. Depends on Track 1's retrained EP/WP models.

Integration contract with cfb4th/R/decision_functions.R::get_go_wp():
  1. Pass the 5-feature situation matrix to fd_model.predict() → 76-class probability
     vector per play.
  2. Expand into a long (play × gain) frame: gain = class_index - 10.
  3. Cap gain at yards_to_goal (TD); floor loss so ball stays at the 1-yard line.
  4. Update game situation per outcome (possession flip, +6 on TD, spread flip, clock).
  5. Call add_ep(situation) + add_wp(situation) from Track 1's Python models.
  6. Weight each outcome's WP by P(gain=k) → go_wp = Σ P(k) × WP(outcome_k).

Implement after Track 1 Stage-2 EP/WP Python inference paths are confirmed working.
See cfb4th/R/decision_functions.R::get_go_wp() for the reference implementation.
"""
from __future__ import annotations


def get_go_wp_py(pbp_df, fd_model, ep_model, wp_model):
    """Compute the expected win probability of going for it on 4th down.

    Args:
        pbp_df: Play-by-play DataFrame with the fourth-down situations.
        fd_model: Trained fourth-down yards-gained XGBoost Booster (Track 2 output).
        ep_model: Trained EP XGBoost Booster (Track 1 Stage-2 output, multi:softprob 7-class).
        wp_model: Trained WP-spread XGBoost Booster (Track 1 Stage-2 output, binary:logistic).

    Returns:
        pbp_df augmented with columns: go_wp, first_down_prob, wp_succeed, wp_fail.

    Raises:
        NotImplementedError: Always — Track 1 Stage-2 EP/WP models must be complete first.
    """
    raise NotImplementedError(
        "get_go_wp_py() is not yet implemented. "
        "Implement after Track 1 Stage-2 EP/WP Python inference paths are confirmed "
        "working. See cfb4th/R/decision_functions.R::get_go_wp() for the reference."
    )
