"""Hard-gate test: RK4 ZOH rollout determinism (bitwise identical)."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from tests.conftest import spec_for


def _sine_policy(amplitude: float, omega: float):
    """A deterministic, state-independent-but-time-varying test policy."""

    def policy(x: np.ndarray, t: float) -> float:
        # Mix a time term and a state term so the policy exercises both paths.
        return amplitude * np.sin(omega * t) + 0.5 * float(x[0])

    return policy


def test_rk4_rollout_bitwise_identical(model: NLinkCartPole) -> None:
    """Same seed/spec/policy => bitwise-identical logged states."""
    spec = model.spec
    x0 = model.x_equilibrium("down").copy()
    x0[1 : 1 + model.n] += 0.01  # small offset off hanging
    policy = _sine_policy(amplitude=20.0, omega=3.0)

    t1, x1, u1 = model.rollout_zoh(
        x0,
        policy,
        t_final=0.5,
        control_dt=spec.control_dt_s,
        rk4_max_step=spec.rk4_max_step_s,
        seed=42,
    )
    t2, x2, u2 = model.rollout_zoh(
        x0,
        policy,
        t_final=0.5,
        control_dt=spec.control_dt_s,
        rk4_max_step=spec.rk4_max_step_s,
        seed=42,
    )

    assert np.array_equal(t1, t2)
    assert np.array_equal(x1, x2)  # bitwise identical
    assert np.array_equal(u1, u2)


def test_rk4_rollout_identical_across_instances() -> None:
    """A fresh model instance reproduces the same rollout bit-for-bit."""
    spec = spec_for(3)
    m_a = NLinkCartPole(spec)
    m_b = NLinkCartPole(spec)
    x0 = m_a.x_equilibrium("down").copy()
    x0[1 : 1 + m_a.n] += 0.02
    policy = _sine_policy(amplitude=15.0, omega=5.0)

    _, xa, _ = m_a.rollout_zoh(
        x0, policy, 0.3, spec.control_dt_s, spec.rk4_max_step_s, seed=1
    )
    _, xb, _ = m_b.rollout_zoh(
        x0, policy, 0.3, spec.control_dt_s, spec.rk4_max_step_s, seed=1
    )
    assert np.array_equal(xa, xb)
