"""Optimized trajopt solver — the fast-stack gate solver (FastColloc).

Same NLP as collocation.solve_trajopt (RK4 multiple-shooting defects, force
bound, track limit, terminal ball, w_u*sumsqr(U) + terminal Qf), with:

  1. Per-step flat SX function F_step (the 4 nested f-calls expanded ONCE,
     ~20k instructions) embedded per node — cuts MX call nodes 4x and lets
     CasADi differentiate through one flat graph per node.
     Measured: jac_g 1.9x faster, hess_l 1.6x faster than the gate's graph.
  2. Parametric x0 (nlpsol parameter p) — one compiled solver object serves
     every IC / every replan of the same shape. Construction amortized.
  3. Optional dual warm start (lam_g0/lam_x0 + warm_start_init_point=yes):
     a single continuous IPOPT run (NOT chunking — the chunked-restart
     divergence mode does not apply).
  4. Optional hessian_approximation=limited-memory (kills the dominant
     Hessian eval entirely; iteration count may inflate — benchmark).

Iterate paths differ from the banked gate in the last bits (different AD
accumulation order) => any use on a real gate is a NEW GATE VERSION with a
full re-bank, per the determinism rules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import casadi as ca
import numpy as np


@dataclass
class FastResult:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    success: bool
    solver_status: str
    max_defect: float
    objective: float
    iter_count: int
    wall_s: float
    lam_g: np.ndarray = field(repr=False, default=None)
    lam_x: np.ndarray = field(repr=False, default=None)


class FastColloc:
    """Build-once, solve-many trajopt NLP (parametric in x0)."""

    def __init__(
        self,
        model,
        *,
        horizon_s: float,
        n_nodes: int,
        force_bound: float,
        terminal_tol_rad: float | None,
        w_u: float = 1e-4,
        w_v: float = 0.0,
        qf_diag: np.ndarray | None = None,
        max_iter: int = 1500,
        linear_solver: str = "mumps",
        mu_strategy: str = "adaptive",
        acceptable_tol: float = 1e-4,
        acceptable_iter: int = 8,
        hessian: str = "exact",          # "exact" | "lbfgs"
        warm_duals: bool = False,
        print_level: int = 0,
        plugin: str = "ipopt",            # "ipopt" | "fatrop"
        map_threads: int = 0,             # >1: thread-parallel defect map
        ipopt_extra: dict | None = None,  # extra raw ipopt options (e.g.
                                          # acceptable_constr_viol_tol,
                                          # max_cpu_time)
        zoh_sub: int = 0,                 # >1: ZOH-consistent node map with
                                          # zoh_sub RK4 sub-steps per node
        term_demand: tuple | None = None, # (K_row, bound_N): adds constraint
                                          # |K_row.(x_N - x_up)| <= bound_N so
                                          # the hold's initial force demand is
                                          # bounded WITHOUT reshaping the whole
                                          # trajectory (knife-edge safe)
    ) -> None:
        spec = model.spec
        nx = model.nx
        n = model.n
        N = n_nodes
        h = horizon_s / N
        self.model = model
        self.nx, self.n, self.N, self.h = nx, n, N, h
        self.horizon_s = horizon_s
        x_t = np.asarray(model.x_equilibrium("up")).reshape(-1)
        if qf_diag is None:
            qf_diag = np.concatenate(
                [[10.0], 200.0 * np.ones(n), [10.0], 50.0 * np.ones(n)])

        # ---- per-step flat SX RK4 (built once; cheap) ----
        # zoh_sub>1: compose zoh_sub RK4 sub-steps of h/zoh_sub per node so the
        # NLP's node map matches the simulator's ZOH tick chain EXACTLY. This
        # kills the one-step-vs-substeps integrator mismatch that dominates the
        # densified-reference boundary jumps (~1e-5, >>NLP defect) and breaks
        # closed-loop tracking at n>=11 (measured 2026-07-07).
        self.zoh_sub = int(zoh_sub or 0)
        xs = ca.SX.sym("x", nx)
        us = ca.SX.sym("u")

        def _rk4(xx, hh):
            k1 = model.f(xx, us)
            k2 = model.f(xx + 0.5 * hh * k1, us)
            k3 = model.f(xx + 0.5 * hh * k2, us)
            k4 = model.f(xx + hh * k3, us)
            return xx + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        nsub = max(1, self.zoh_sub)
        xx = xs
        for _ in range(nsub):
            xx = _rk4(xx, h / nsub)
        self.F_step = ca.Function("F", [xs, us], [xx])

        # ---- NLP (MX chain over F_step), x0 as parameter ----
        # fatrop's auto structure detection needs stage-interleaved variables
        # [x0,u0,x1,u1,...,xN]; ipopt keeps the simple [vec(X); U] layout.
        self.stagewise = (plugin == "fatrop")
        p = ca.MX.sym("p", nx)
        if self.stagewise:
            z = ca.MX.sym("z", N * (nx + 1) + nx)
            Xs = [z[k * (nx + 1): k * (nx + 1) + nx] for k in range(N)]
            Xs.append(z[N * (nx + 1):])
            Us = [z[k * (nx + 1) + nx] for k in range(N)]
        else:
            X = ca.MX.sym("X", nx, N + 1)
            U = ca.MX.sym("U", 1, N)
            z = ca.veccat(X, U)
            Xs = [X[:, k] for k in range(N + 1)]
            Us = [U[0, k] for k in range(N)]
        if map_threads and map_threads > 1 and not self.stagewise:
            # one (possibly threaded) map node; derivatives inherit the map
            Fmap = self.F_step.map(N, "thread", map_threads)
            Xnext = Fmap(X[:, :N], U)
            g = ca.vertcat(ca.vec(Xs[0] - p), ca.vec(X[:, 1:] - Xnext))
        else:
            g_parts = [Xs[0] - p]
            for k in range(N):
                g_parts.append(Xs[k + 1] - self.F_step(Xs[k], Us[k]))
            g = ca.vertcat(*[ca.vec(gp) for gp in g_parts])
        err = Xs[N] - ca.DM(x_t)
        self.lbg_vec = np.zeros(int(g.shape[0]))
        self.ubg_vec = np.zeros(int(g.shape[0]))
        if term_demand is not None:
            assert not self.stagewise, "term_demand: ipopt layout only"
            Krow, dbound = term_demand
            g = ca.vertcat(g, ca.mtimes(ca.DM(np.asarray(Krow).reshape(1, nx)),
                                        err))
            self.lbg_vec = np.concatenate([self.lbg_vec, [-float(dbound)]])
            self.ubg_vec = np.concatenate([self.ubg_vec, [float(dbound)]])
        obj = w_u * ca.sumsqr(ca.vertcat(*Us)) + ca.mtimes(
            [err.T, ca.DM(np.diag(qf_diag)), err])
        # Running velocity penalty (w_v): must match the nominal's objective so
        # the per-IC replans reproduce the GENTLE (trackable) swing-up rather
        # than drifting back to the force-optimal violent one. Velocities are
        # state rows [n+1:] (cart_vel + link rates). Default 0.0 = unchanged.
        if w_v > 0.0:
            obj = obj + w_v * ca.sumsqr(
                ca.vertcat(*[Xs[k][n + 1:] for k in range(N + 1)]))
        self.nvar = int(z.shape[0])
        self.ng = int(g.shape[0])
        self._nX = nx * (N + 1)

        # bounds
        lbx = -np.inf * np.ones(self.nvar)
        ubx = np.inf * np.ones(self.nvar)
        tl = spec.track_half_length_m

        def _xi(k):   # flat index of state k's first entry
            return k * (nx + 1) if self.stagewise else k * nx

        def _ui(k):   # flat index of control k
            return k * (nx + 1) + nx if self.stagewise else self._nX + k

        for k in range(N):
            lbx[_ui(k)] = -force_bound
            ubx[_ui(k)] = force_bound
        for k in range(N + 1):
            lbx[_xi(k)] = -tl
            ubx[_xi(k)] = tl
        if terminal_tol_rad is not None:
            base = _xi(N)
            for i in range(1, 1 + n):
                lbx[base + i] = x_t[i] - terminal_tol_rad
                ubx[base + i] = x_t[i] + terminal_tol_rad
            for i in range(1 + n, nx):
                lbx[base + i] = -terminal_tol_rad
                ubx[base + i] = terminal_tol_rad
        self.lbx, self.ubx = lbx, ubx

        nlp = {"x": z, "p": p, "f": obj, "g": g}
        if plugin == "fatrop":
            opts = {"print_time": False,
                    "structure_detection": "auto",
                    "equality": [True] * self.ng,
                    "debug": False,
                    "fatrop": {"max_iter": max_iter, "tol": 1e-8,
                               "print_level": print_level}}
        else:
            ipopt = {"max_iter": max_iter, "print_level": print_level,
                     "linear_solver": linear_solver, "tol": 1e-8,
                     "acceptable_tol": acceptable_tol,
                     "acceptable_iter": acceptable_iter,
                     "mu_strategy": mu_strategy}
            if hessian == "lbfgs":
                ipopt["hessian_approximation"] = "limited-memory"
            if warm_duals:
                ipopt.update({
                    "warm_start_init_point": "yes",
                    "warm_start_bound_push": 1e-9,
                    "warm_start_slack_bound_push": 1e-9,
                    "warm_start_mult_bound_push": 1e-9,
                    "warm_start_bound_frac": 1e-9,
                    "warm_start_slack_bound_frac": 1e-9,
                })
            if ipopt_extra:
                ipopt.update(ipopt_extra)
            opts = {"print_time": False, "ipopt": ipopt}
        t0 = time.perf_counter()
        self.solver = ca.nlpsol("fast", plugin, nlp, opts)
        self.construct_s = time.perf_counter() - t0
        self._defect_map = self.F_step.map(N)

    # ------------------------------------------------------------------
    def solve(
        self,
        x0: np.ndarray,
        x_init: np.ndarray | None = None,
        u_init: np.ndarray | None = None,
        lam_g0: np.ndarray | None = None,
        lam_x0: np.ndarray | None = None,
    ) -> FastResult:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        x_t = np.asarray(self.model.x_equilibrium("up")).reshape(-1)
        if x_init is None:
            x_init = np.linspace(x0, x_t, self.N + 1)
        if u_init is None:
            u_init = np.zeros(self.N)
        xi = np.asarray(x_init, dtype=float)[: self.N + 1]
        ui = np.asarray(u_init, dtype=float)[: self.N].reshape(-1)
        if self.stagewise:
            z0 = np.concatenate(
                [np.concatenate([xi[k], [ui[k]]]) for k in range(self.N)]
                + [xi[self.N]])
        else:
            # veccat(X, U): node-major flat X (C-order of (N+1, nx)) then U.
            z0 = np.concatenate([xi.reshape(-1), ui])
        kw = dict(x0=z0, p=x0, lbx=self.lbx, ubx=self.ubx,
                  lbg=self.lbg_vec, ubg=self.ubg_vec)
        if lam_g0 is not None:
            kw["lam_g0"] = lam_g0
        if lam_x0 is not None:
            kw["lam_x0"] = lam_x0
        t0 = time.perf_counter()
        sol = self.solver(**kw)
        wall = time.perf_counter() - t0
        st = self.solver.stats()
        ok = bool(st.get("success", False))
        status = st.get("return_status", "?")
        zv = np.asarray(sol["x"]).reshape(-1)
        if self.stagewise:
            stage = zv[: self.N * (self.nx + 1)].reshape(self.N, self.nx + 1)
            Xv = np.vstack([stage[:, : self.nx],
                            zv[self.N * (self.nx + 1):].reshape(1, self.nx)])
            Uv = stage[:, self.nx].copy()
        else:
            Xv = zv[: self._nX].reshape(self.N + 1, self.nx)
            Uv = zv[self._nX:].reshape(self.N)
        # numeric defect via one mapped call (same F_step graph)
        Xn = np.asarray(self._defect_map(Xv[:-1].T, Uv.reshape(1, -1)))
        max_def = float(np.max(np.abs(Xn - Xv[1:].T)))
        return FastResult(
            t=np.linspace(0.0, self.horizon_s, self.N + 1),
            x=Xv, u=Uv, success=ok, solver_status=status,
            max_defect=max_def, objective=float(sol["f"]),
            iter_count=int(st.get("iter_count", -1)), wall_s=wall,
            lam_g=np.asarray(sol["lam_g"]).reshape(-1),
            lam_x=np.asarray(sol["lam_x"]).reshape(-1),
        )
