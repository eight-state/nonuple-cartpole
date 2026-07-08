"""LQR tests: closed-loop ``A - B K`` is Hurwitz at upright for n=1,2,3."""

from __future__ import annotations

import numpy as np
import pytest

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import (
    StaticLQRPolicy,
    static_lqr,
    wrap_state_error,
    wrap_to_pi,
)

from tests.conftest import spec_for


@pytest.mark.parametrize("n", [1, 2, 3])
def test_closed_loop_hurwitz(n: int) -> None:
    """All eigenvalues of A - B K have strictly negative real part."""
    model = NLinkCartPole(spec_for(n))
    K, P = static_lqr(model)
    x_up = model.x_equilibrium("up")
    A, B = model.linearize(x_up, 0.0)
    eig = np.linalg.eigvals(A - B @ K)
    assert np.max(eig.real) < -1e-6, f"not Hurwitz: max Re = {np.max(eig.real)}"
    # P must be symmetric positive definite.
    assert np.allclose(P, P.T, atol=1e-8)
    assert np.min(np.linalg.eigvalsh(P)) > 0.0


@pytest.mark.parametrize("n", [1, 2, 3])
def test_static_policy_stabilizes_small_perturbation(n: int) -> None:
    """A small upright perturbation converges into the success set in 5 s."""
    from cartpole_race.funnels import in_success_set

    model = NLinkCartPole(spec_for(n))
    K, P = static_lqr(model)
    pol = StaticLQRPolicy(model, K)
    pol.P = P
    x0 = model.x_equilibrium("up").copy()
    x0[1] += np.deg2rad(3.0)  # tip the first link 3 degrees
    _, x_log, _ = model.rollout_zoh(
        x0, pol, 5.0, model.spec.control_dt_s, model.spec.rk4_max_step_s
    )
    assert in_success_set(model, x_log[-1])


def test_wrap_to_pi() -> None:
    """Angle wrapping maps to (-pi, pi]."""
    assert np.isclose(wrap_to_pi(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap_to_pi(-np.pi - 0.1), np.pi - 0.1)
    assert np.isclose(wrap_to_pi(0.0), 0.0)


def test_wrap_state_error_only_wraps_angles() -> None:
    """Cart/velocity errors are plain diffs; angle errors are wrapped."""
    model = NLinkCartPole(spec_for(2))
    x = model.x_equilibrium("up").copy()
    x_ref = model.x_equilibrium("up").copy()
    x[0] = 3.0  # cart far away -> not wrapped
    x[1] = 2.0 * np.pi - 0.01  # nearly full turn -> wraps near -0.01
    e = wrap_state_error(x, x_ref, 2)
    assert np.isclose(e[0], 3.0)
    assert abs(e[1]) < 0.1
