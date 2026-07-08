"""CasADi ``mapaccum`` closed-loop rollouts — compiler-free ~14x speedup.

The Python per-control-tick loop in :func:`cartpole_race.rollout.static_hold_rollout`
is the gate bottleneck (~6-10 s per 2-3 link rollout). This module folds the
*entire* rollout into a single CasADi VM call by building one symbolic control
tick (saturated static-LQR + 4 RK4 substeps) and ``mapaccum``-ing it over the
horizon. It uses the SAME ``model.f`` dynamics ca.Function — single source of
truth (Principle 1). Verified to reproduce the Python-loop trajectory to
floating-point agreement (see scripts/bench_mapaccum.py and the consistency
test).
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.funnels import in_success_set
from cartpole_race.rollout import SETTLE_TIME_S


class StaticHoldEvaluator:
    """Pre-built mapaccum evaluator for the saturated static-LQR hold test.

    Build once per ``(model, K)`` (amortizes the symbolic graph), then call
    :meth:`evaluate` per initial state. Reproduces the success predicate of
    :func:`cartpole_race.rollout.static_hold_rollout`.
    """

    def __init__(
        self,
        model: NLinkCartPole,
        K: np.ndarray,
        hold_time_s: float = 5.0,
        settle_time_s: float = SETTLE_TIME_S,
    ) -> None:
        self.model = model
        self.K = np.asarray(K, dtype=float)
        self.hold_time_s = hold_time_s
        spec = model.spec
        self.control_dt = spec.control_dt_s
        self.track = spec.track_half_length_m
        self.fbound = spec.force_bound_n
        self.n = model.n
        self.nx = model.nx
        self.xref = np.asarray(model.x_equilibrium("up")).reshape(-1)
        self.total_t = hold_time_s + settle_time_s
        self.n_ticks = int(round(self.total_t / self.control_dt))
        self.hold_ticks = int(round(hold_time_s / self.control_dt))
        self._roll = self._build(spec)

    def _control(self, x):
        """Symbolic/numeric saturated static-LQR control for state ``x``."""
        e = x - ca.DM(self.xref)
        parts = [e[i] for i in range(self.nx)]
        for i in range(1, 1 + self.n):  # wrap angle errors
            parts[i] = ca.atan2(ca.sin(e[i]), ca.cos(e[i]))
        e = ca.vertcat(*parts)
        u = -ca.mtimes(ca.DM(self.K), e)
        return ca.fmax(ca.fmin(u, self.fbound), -self.fbound)

    def _build(self, spec):
        rk4 = spec.rk4_max_step_s
        n_sub = int(round(self.control_dt / rk4))
        x = ca.MX.sym("x", self.nx)
        u = self._control(x)
        xx = x
        for _ in range(n_sub):
            k1 = self.model.f(xx, u)
            k2 = self.model.f(xx + 0.5 * rk4 * k1, u)
            k3 = self.model.f(xx + 0.5 * rk4 * k2, u)
            k4 = self.model.f(xx + rk4 * k3, u)
            xx = xx + (rk4 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        step = ca.Function("step", [x], [xx])
        return step.mapaccum(self.n_ticks)

    def evaluate(self, x0: np.ndarray) -> tuple[bool, dict]:
        """Roll the closed loop and apply the locked hold predicate."""
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        traj = np.asarray(self._roll(x0))  # (nx, n_ticks), states after each tick
        xfull = np.column_stack([x0, traj])  # include x0
        # Control applied at each pre-step state (for the force bound check).
        e = xfull[:, :-1] - self.xref[:, None]
        e[1 : 1 + self.n] = np.arctan2(
            np.sin(e[1 : 1 + self.n]), np.cos(e[1 : 1 + self.n])
        )
        u = -(self.K @ e)
        u = np.clip(u, -self.fbound, self.fbound).ravel()
        max_force = float(np.max(np.abs(u))) if u.size else 0.0
        max_cart = float(np.max(np.abs(xfull[0, :])))
        track_ok = bool(max_cart <= self.track)
        force_ok = bool(max_force <= self.fbound + 1e-6)
        in_set = np.array(
            [in_success_set(self.model, xfull[:, j]) for j in range(xfull.shape[1])]
        )
        tail = 0
        for j in range(len(in_set) - 1, -1, -1):
            if in_set[j]:
                tail += 1
            else:
                break
        tail_time = tail * self.control_dt
        success = bool(track_ok and force_ok and tail_time >= self.hold_time_s - 1e-9)
        info = {
            "max_force": max_force,
            "min_track_margin": float(self.track - max_cart),
            "final_state": xfull[:, -1].tolist(),
            "tail_hold_s": tail_time,
            "track_ok": track_ok,
            "force_ok": force_ok,
        }
        return success, info
