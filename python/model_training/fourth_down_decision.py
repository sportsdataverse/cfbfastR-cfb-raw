"""Fourth-down decision layer: expected-value comparison of go/punt/FG.

STUB — not yet implemented. Depends on Track 1's retrained EP/WP models.

Integration contract with cfb4th/R/decision_functions.R::get_go_wp():
  1. Pass the 5-feature situation matrix (down, distance, yards_to_goal,
     posteam_total, posteam_spread) to fd_model.predict() -> 76-class probability
     vector per play.
  2. Expand into a long (play x gain) frame: gain = class_index - 10.
  3. Cap gain at yards_to_goal (TD); floor loss so the ball stays at the 1-yard line.
  4. Update game situation per outcome:
       - possession flip on turnover-on-downs (gain < distance)
       - +6 points and possession flip on TD (gain == yards_to_goal)
       - spread sign flip on possession change
       - clock run-off: TimeSecsRem -= 6; adj_TimeSecsRem -= 6 (min 0)
  5. Call add_ep(situation) + add_wp(situation) from Track 1's Python models.
  6. Weight each outcome's WP by P(gain=k) -> go_wp = Sum P(k) x WP(outcome_k).

This function signature mirrors cfb4th::get_go_wp(); implement after Track 1's
EP/WP Python inference paths are confirmed working (Stage 2 retrain complete).
"""
from __future__ import annotations


def get_go_wp_py(pbp_df, fd_model, ep_model, wp_model):
    """Compute the expected win probability of going for it on 4th down.

    Args:
        pbp_df: Play-by-play DataFrame (polars or pandas) with the fourth-down situations.
                Must contain the 5 fourth-down features plus the EP/WP inference columns
                (TimeSecsRem, adj_TimeSecsRem, pos_score_diff_start, etc.).
        fd_model: Trained fourth-down yards-gained XGBoost Booster (Track 2 output).
        ep_model: Trained EP XGBoost Booster (Track 1 Stage-2 output, multi:softprob 7-class).
        wp_model: Trained WP-spread XGBoost Booster (Track 1 Stage-2 output, binary:logistic).

    Returns:
        pbp_df augmented with columns: go_wp, first_down_prob, wp_succeed, wp_fail.

    Raises:
        NotImplementedError: Always -- Track 1 Stage-2 EP/WP models must be complete before
            implementing the decision layer. See cfb4th R package for the reference
            implementation.
    """
    raise NotImplementedError(
        "fourth_down_decision not yet implemented; "
        "see cfb4th R package for the reference implementation"
    )
