"""Hard-gate tests: shape, equilibrium, and energy sanity."""

from __future__ import annotations

import numpy as np
import pytest

from cartpole_race.dynamics import NLinkCartPole
from tests.conftest import spec_for


# --- Gate 1: shape -----------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_state_shape_and_finite(n: int) -> None:
    """nx = 2*(n+1) and f(x, u) is finite for every n in 1..6."""
    model = NLinkCartPole(spec_for(n))
    assert model.nx == 2 * (n + 1)

    rng = np.random.default_rng(0)
    x = rng.standard_normal(model.nx)
    u = 3.7
    xdot = model.f_num(x, u)
    assert xdot.shape == (model.nx,)
    assert np.all(np.isfinite(xdot))


# --- Gate 2: equilibria (u = 0 => accelerations ~ 0) -------------------------
@pytest.mark.parametrize("kind", ["up", "down"])
def test_equilibrium_zero_acceleration(model: NLinkCartPole, kind: str) -> None:
    """Upright and hanging equilibria have zero acceleration at u = 0."""
    x_eq = model.x_equilibrium(kind)
    xdot = model.f_num(x_eq, 0.0)
    # Velocities are already zero; accelerations are the second half.
    acc = xdot[model.nq :]
    assert np.max(np.abs(acc)) < 1e-9, f"{kind} acc max {np.max(np.abs(acc))}"
    # Velocity block should remain exactly zero.
    assert np.max(np.abs(xdot[: model.nq])) < 1e-9


# --- Gate 5: energy sanity (u = 0, zero damping, tiny step) ------------------
def test_energy_drift_small(model: NLinkCartPole) -> None:
    """Free response (u=0, no damping) conserves energy to a tiny drift over 2 s.

    RK4 is not symplectic, so we assert a generous but fixed bound rather than
    exact conservation. A small-amplitude perturbation off hanging is used.
    """
    rng = np.random.default_rng(7)
    x0 = model.x_equilibrium("down").copy()
    # Small kick on angles and angular rates only (keep it bounded).
    x0[1 : 1 + model.n] += rng.uniform(-0.05, 0.05, size=model.n)
    x0[model.nq + 1 :] += rng.uniform(-0.05, 0.05, size=model.n)

    e0 = model.energy(x0)

    # Tiny fixed RK4 step, u = 0, integrate 2 s.
    dt = 1e-4
    n_steps = int(round(2.0 / dt))
    x = x0.copy()
    for _ in range(n_steps):
        x = model.rk4_step(x, 0.0, dt)
    e1 = model.energy(x)

    drift = abs(e1 - e0)
    # Fixed absolute threshold (Joules). Energies here are O(1 J) or less.
    assert drift < 1e-4, f"energy drift {drift:.3e} J over 2 s (n={model.n})"
