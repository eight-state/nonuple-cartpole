"""Adversarial falsifier: CEM and CMA-ES boundary search (proposal harness).

Per the proposal 'Proof harness -> CEM/CMA-ES adversarial boundary search' and
'falsification.py' in the M1 build list. The adversary searches inside a
claimed funnel level (``eta * rho``) for an initial perturbation whose rollout
FAILS the success predicate, minimizing a success margin. Reused by M5.

The decision variable is a perturbation ``dx`` constrained to the ellipsoid
``dx' M dx <= eta * rho`` (``M`` is ``P_static`` for the static funnel or
``S(t0)`` for a TVLQR funnel). We parameterize the interior of the ellipsoid
and minimize a scalar margin that is negative when the rollout succeeds and
rises toward/above zero as it approaches/crosses failure; finding a point with
margin >= 0 (failure) inside the set falsifies the funnel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.funnels import in_success_set
from cartpole_race.lqr import wrap_state_error
from cartpole_race.rollout import SETTLE_TIME_S

MarginFn = Callable[[np.ndarray], float]


@dataclass
class FalsifyResult:
    """Outcome of an adversarial search."""

    failure_found: bool
    best_margin: float
    best_dx: np.ndarray
    n_evals: int
    method: str


def success_margin_static(
    model: NLinkCartPole,
    policy,
    P: np.ndarray,
    rho: float,
    eta: float,
    hold_time_s: float = 5.0,
):
    """Build a margin function for the static funnel ``eta * rho``.

    The margin is constructed so that **margin >= 0 means failure** (a found
    counterexample) and more-negative means a more comfortable success. We
    combine: hold-time deficit, track violation, force-saturation excess, final
    angle error. Points are mapped from an unconstrained vector ``z`` into the
    P-ellipsoid of level ``eta*rho``.

    Returns:
        ``(margin_fn, decode)`` where ``decode(z) -> dx`` recovers the
        perturbation actually evaluated (for reporting).
    """
    spec = model.spec
    n = model.n
    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s
    track = spec.track_half_length_m
    fbound = spec.force_bound_n
    x_up = model.x_equilibrium("up")
    level = eta * rho
    # Settling budget: a boundary point may take a brief transient to enter the
    # hold set; the predicate is "enters AND remains" for hold_time_s. Roll a
    # little longer and require the FINAL hold_time_s in-set, matching
    # rollout.static_hold_rollout. Without this the hold-deficit term flags
    # clean catches that settle in a few ms as spurious "failures".
    settle_time_s = SETTLE_TIME_S
    total_t = hold_time_s + settle_time_s

    # Whitening: dx = L z, with L from Cholesky of inv(P), so dx'P dx = z'z.
    Pinv = np.linalg.inv(P)
    L = np.linalg.cholesky(Pinv)

    def decode(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float).reshape(-1)
        r = np.linalg.norm(z)
        # Squash radius into [0, sqrt(level)] so we always stay inside the set.
        target_r = np.sqrt(level) * (np.tanh(r) if r > 0 else 0.0)
        if r > 1e-12:
            z_scaled = z * (target_r / r)
        else:
            z_scaled = z
        return L @ z_scaled

    def margin_fn(z: np.ndarray) -> float:
        dx = decode(z)
        x0 = x_up + dx
        _, x_log, u_log = model.rollout_zoh(
            x0, policy, total_t, control_dt, rk4_max
        )
        # Track violation (positive if breached).
        track_excess = float(np.max(np.abs(x_log[:, 0])) - track)
        # Force saturation excess (positive if over bound).
        force_excess = float(np.max(np.abs(u_log)) - fbound) if len(u_log) else -fbound
        # Hold deficit: how far the tail is from being in-set.
        in_set = np.array([in_success_set(model, xx) for xx in x_log])
        tail_len = 0
        for j in range(len(in_set) - 1, -1, -1):
            if in_set[j]:
                tail_len += 1
            else:
                break
        tail_time = tail_len * control_dt
        hold_deficit = hold_time_s - tail_time  # >0 means did not hold full 5s
        # Final angle error (normalized).
        e = wrap_state_error(x_log[-1], x_up, n)
        ang_err = float(np.max(np.abs(e[1 : 1 + n])) - np.deg2rad(5.0))

        # Failure if ANY hard violation, or hold not achieved. Build a margin
        # that is >=0 on failure. Use the max of the violation signals; for the
        # hold-deficit (soft) scale so that a missed hold registers as >0.
        margin = max(
            track_excess,
            force_excess,
            ang_err,
            (hold_deficit - 1e-6),  # any deficit -> >0 (failure)
        )
        return margin

    return margin_fn, decode


def cem_minimize(
    margin_fn: MarginFn,
    dim: int,
    n_evals: int,
    seed: int = 0,
    pop: int = 32,
    elite_frac: float = 0.25,
    init_sigma: float = 1.0,
) -> FalsifyResult:
    """Cross-Entropy Method maximizing the margin (to find failure >= 0).

    We maximize the margin (failure-seeking) by minimizing ``-margin``.

    Args:
        margin_fn: ``z -> margin`` (>=0 means failure).
        dim: Dimension of ``z``.
        n_evals: Evaluation budget.
        seed: RNG seed.
        pop: Population per generation.
        elite_frac: Fraction kept as elites.
        init_sigma: Initial std.

    Returns:
        :class:`FalsifyResult`.
    """
    rng = np.random.default_rng(seed)
    mean = np.zeros(dim)
    sigma = np.full(dim, init_sigma)
    n_elite = max(2, int(pop * elite_frac))

    best_margin = -np.inf
    best_z = mean.copy()
    evals = 0
    while evals < n_evals:
        Z = rng.normal(mean, sigma, size=(pop, dim))
        margins = np.array([margin_fn(z) for z in Z])
        evals += pop
        order = np.argsort(margins)[::-1]  # descending margin (failure-seeking)
        if margins[order[0]] > best_margin:
            best_margin = float(margins[order[0]])
            best_z = Z[order[0]].copy()
        if best_margin >= 0.0:
            break
        elites = Z[order[:n_elite]]
        mean = elites.mean(axis=0)
        sigma = elites.std(axis=0) + 1e-6

    return FalsifyResult(
        failure_found=best_margin >= 0.0,
        best_margin=best_margin,
        best_dx=best_z,
        n_evals=evals,
        method="CEM",
    )


def cmaes_minimize(
    margin_fn: MarginFn,
    dim: int,
    n_evals: int,
    seed: int = 0,
    init_sigma: float = 0.5,
) -> FalsifyResult:
    """CMA-ES failure search via the ``cma`` package (maximize margin).

    ``cma`` minimizes, so we minimize ``-margin``.

    Args:
        margin_fn: ``z -> margin`` (>=0 means failure).
        dim: Dimension of ``z``.
        n_evals: Evaluation budget.
        seed: RNG seed.
        init_sigma: Initial step size.

    Returns:
        :class:`FalsifyResult`.
    """
    import cma

    best_margin = -np.inf
    best_z = np.zeros(dim)
    evals = 0

    es = cma.CMAEvolutionStrategy(
        np.zeros(dim),
        init_sigma,
        {"seed": int(seed) + 1, "verbose": -9, "maxfevals": n_evals},
    )
    while not es.stop() and evals < n_evals:
        solutions = es.ask()
        costs = []
        for z in solutions:
            m = margin_fn(np.asarray(z))
            costs.append(-m)  # minimize -margin
            evals += 1
            if m > best_margin:
                best_margin = float(m)
                best_z = np.asarray(z).copy()
        es.tell(solutions, costs)
        if best_margin >= 0.0:
            break

    return FalsifyResult(
        failure_found=best_margin >= 0.0,
        best_margin=best_margin,
        best_dx=best_z,
        n_evals=evals,
        method="CMA-ES",
    )
