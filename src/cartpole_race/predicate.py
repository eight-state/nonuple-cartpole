"""The sole 1 kHz sampled-state success predicate for the N9 replay."""

from __future__ import annotations

import numpy as np

from cartpole_race.lqr import wrap_state_error


def in_success_set(
    model,
    state: np.ndarray,
    theta_tol: float = np.deg2rad(5.0),
    theta_rate_tol: float = 0.5,
    cart_position_tol: float = 2.0,
    cart_rate_tol: float = 0.5,
) -> bool:
    """Return whether one state lies in the released upright hold set."""
    state = np.asarray(state, dtype=float).reshape(-1)
    error = wrap_state_error(state, model.x_equilibrium("up"), model.n)
    return bool(
        np.all(np.abs(error[1 : 1 + model.n]) <= theta_tol)
        and np.all(np.abs(state[model.nq + 1 :]) <= theta_rate_tol)
        and abs(state[0]) <= cart_position_tol
        and abs(state[model.nq]) <= cart_rate_tol
    )


def longest_continuous_hold_s(model, states: np.ndarray, control_dt_s: float) -> float:
    """Return the longest interval across consecutive 1 kHz logged states in the hold set."""
    run = best = 0
    for state in states:
        run = run + 1 if in_success_set(model, state) else 0
        best = max(best, run)
    return max(0, best - 1) * control_dt_s
