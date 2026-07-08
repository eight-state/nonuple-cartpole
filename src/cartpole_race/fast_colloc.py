"""Persistent-NLP collocation solver (attempt3 speedup).

Root cause of the slow 1ms polish: scripts/n7_polish_conv.py drove IPOPT one
iteration at a time (crash mitigation) by calling
``collocation.solve_trajopt(..., max_iter=1)`` in a Python loop. EACH call
constructs a fresh ``ca.Opti()`` and CasADi re-builds + re-compiles the whole
NLP (128016 eq-constraints, ~2.18M Jacobian nonzeros, ~1.08M Hessian
nonzeros). That construction/codegen is what eats the ~300-340 s/chunk -- the
single IPOPT iteration itself is cheap.

This module builds the NLP graph and the ``nlpsol`` solver object ONCE, then
steps IPOPT ``chunk_iter`` iterations per call by re-entering the SAME compiled
solver with the previous primal/dual iterate as the warm start
(``warm_start_init_point=yes``). A crash-resume reloads the saved iterate into
the same solver -- no reconstruction.

Distinct from collocation.py / solve_trajopt; nothing here touches the running
supervised polish or its checkpoints.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import casadi as ca
import numpy as np

from cartpole_race.dynamics import NLinkCartPole


def _rk4_step_sym(model, x, u, h):
    k1 = model.f(x, u)
    k2 = model.f(x + 0.5 * h * k1, u)
    k3 = model.f(x + 0.5 * h * k2, u)
    k4 = model.f(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _zoh_step_sym(model, x, u, control_dt, n_sub):
    dt_sub = control_dt / n_sub
    xx = x
    for _ in range(n_sub):
        xx = _rk4_step_sym(model, xx, u, dt_sub)
    return xx


@dataclass
class _State:
    """Mutable warm-start iterate carried across chunked solves."""
    x: np.ndarray          # primal (nvar,)
    lam_g: np.ndarray      # constraint multipliers (ng,)
    lam_x: np.ndarray      # bound multipliers (nvar,)


class PersistentColloc:
    """Build the trajopt NLP + solver ONCE; step IPOPT in chunks via warm start.

    The decision vector packs ``X`` (nx*(N+1)) then ``U`` (N), column-major in
    the same order ``ca.veccat`` produces, so we can unpack to the (N+1,nx) /
    (N,) arrays the rest of the pipeline expects.
    """

    def __init__(
        self,
        model: NLinkCartPole,
        x0: np.ndarray,
        *,
        horizon_s: float,
        n_nodes: int,
        force_bound: float,
        qf_diag: np.ndarray,
        terminal_tol_rad: float | None = None,
        w_u: float = 1e-4,
        zoh_consistent: bool = False,
        linear_solver: str | None = None,
        mu_strategy: str | None = None,
        print_level: int = 5,
        outfile: str | None = None,
        chunk_iter: int = 1,
    ) -> None:
        spec = model.spec
        nx = model.nx
        n = model.n
        N = n_nodes
        self.model = model
        self.nx = nx
        self.n = n
        self.N = N
        self.zoh_consistent = zoh_consistent

        control_dt = spec.control_dt_s
        n_sub = max(1, int(np.ceil(control_dt / spec.rk4_max_step_s)))
        h = control_dt if zoh_consistent else horizon_s / N
        self.control_dt = control_dt
        self.n_sub = n_sub
        self.h = h
        x_t = np.asarray(model.x_equilibrium("up")).reshape(-1)
        self.x_t = x_t

        # Decision variables.
        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", 1, N)
        z = ca.veccat(X, U)  # column-major: X[:,0],X[:,1],...,U[0,:]
        self.nvar = z.shape[0]

        def _step(xk, uk):
            if zoh_consistent:
                return _zoh_step_sym(model, xk, uk, control_dt, n_sub)
            return _rk4_step_sym(model, xk, uk, h)

        # Equality constraints g(z) = 0: initial condition + RK4/ZOH defects.
        g_parts = [X[:, 0] - ca.DM(np.asarray(x0).reshape(-1))]
        for k in range(N):
            g_parts.append(X[:, k + 1] - _step(X[:, k], U[0, k]))
        g = ca.vertcat(*[ca.vec(gp) for gp in g_parts])
        self.ng = g.shape[0]

        # Objective.
        err_T = X[:, N] - ca.DM(x_t)
        Qf = ca.DM(np.diag(qf_diag))
        obj = w_u * ca.sumsqr(U) + ca.mtimes([err_T.T, Qf, err_T])

        nlp = {"x": z, "f": obj, "g": g}

        linear_solver = linear_solver or os.environ.get(
            "CARTPOLE_LINEAR_SOLVER", "mumps")
        mu_strategy = mu_strategy or os.environ.get(
            "CARTPOLE_MU_STRATEGY", "monotone")
        ipopt_opts = {
            "max_iter": chunk_iter,
            "print_level": print_level,
            "linear_solver": linear_solver,
            "tol": 1e-8,
            "acceptable_tol": 1e-6,
            "acceptable_iter": 15,
            "mu_strategy": mu_strategy,
            # The whole point: re-enter the SAME solver with the prior iterate
            # as the warm start instead of rebuilding.
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-9,
            "warm_start_slack_bound_push": 1e-9,
            "warm_start_mult_bound_push": 1e-9,
            "warm_start_bound_frac": 1e-9,
            "warm_start_slack_bound_frac": 1e-9,
            # Keep mu fixed across chunk boundaries so a chunked solve tracks a
            # single continuous IPOPT run (otherwise each chunk would reset the
            # barrier and stall).
        }
        if outfile:
            ipopt_opts["output_file"] = outfile
            ipopt_opts["file_print_level"] = max(print_level, 5)
            ipopt_opts["print_frequency_iter"] = 1
        self._ipopt_opts = ipopt_opts

        self.solver = ca.nlpsol(
            "fastcolloc", "ipopt", nlp,
            {"print_time": False, "ipopt": ipopt_opts},
        )

        # Bounds. Equalities: lbg = ubg = 0.
        self.lbg = np.zeros(self.ng)
        self.ubg = np.zeros(self.ng)
        lbx = -np.inf * np.ones(self.nvar)
        ubx = np.inf * np.ones(self.nvar)
        # Unpack helper indices.
        self._nX = nx * (N + 1)
        # Force bounds on U (last N entries).
        lbx[self._nX:] = -force_bound
        ubx[self._nX:] = force_bound
        # Track limit on cart position X[0, :].
        tl = spec.track_half_length_m
        for k in range(N + 1):
            lbx[k * nx + 0] = -tl
            ubx[k * nx + 0] = tl
        # Hard terminal ball (final knot X[:, N]) when requested.
        if terminal_tol_rad is not None:
            base = N * nx
            for i in range(1, 1 + n):  # angles -> target +/- tol
                lbx[base + i] = x_t[i] - terminal_tol_rad
                ubx[base + i] = x_t[i] + terminal_tol_rad
            for i in range(1 + n, nx):  # velocities -> 0 +/- tol
                lbx[base + i] = -terminal_tol_rad
                ubx[base + i] = terminal_tol_rad
        self.lbx = lbx
        self.ubx = ubx

        self.state: _State | None = None

    # -- (un)pack -------------------------------------------------------
    def pack(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """(N+1,nx),(N,) -> flat decision vector (column-major X then U)."""
        return np.concatenate([X.T.reshape(-1), np.asarray(U).reshape(-1)])

    def unpack(self, z: np.ndarray):
        X = np.asarray(z[: self._nX]).reshape(self.nx, self.N + 1).T
        U = np.asarray(z[self._nX:]).reshape(self.N)
        return X, U

    # -- warm-start management -----------------------------------------
    def set_initial(self, X: np.ndarray, U: np.ndarray) -> None:
        z = self.pack(X, U)
        self.state = _State(
            x=z, lam_g=np.zeros(self.ng), lam_x=np.zeros(self.nvar))

    def load_state(self, d: dict) -> None:
        self.state = _State(x=d["z"], lam_g=d["lam_g"], lam_x=d["lam_x"])

    def dump_state(self) -> dict:
        assert self.state is not None
        return {"z": self.state.x, "lam_g": self.state.lam_g,
                "lam_x": self.state.lam_x}

    # -- one chunk ------------------------------------------------------
    def step(self):
        """Run ``chunk_iter`` IPOPT iterations from the carried iterate.

        Returns (X, U, stats). Updates ``self.state`` in place with the new
        primal/dual iterate so the next call resumes from here in the SAME
        compiled solver -- no reconstruction.
        """
        assert self.state is not None, "call set_initial/load_state first"
        sol = self.solver(
            x0=self.state.x,
            lam_g0=self.state.lam_g,
            lam_x0=self.state.lam_x,
            lbg=self.lbg, ubg=self.ubg, lbx=self.lbx, ubx=self.ubx,
        )
        znew = np.asarray(sol["x"]).reshape(-1)
        self.state = _State(
            x=znew,
            lam_g=np.asarray(sol["lam_g"]).reshape(-1),
            lam_x=np.asarray(sol["lam_x"]).reshape(-1),
        )
        X, U = self.unpack(znew)
        return X, U, self.solver.stats()

    # -- numeric defect (shared dynamics) ------------------------------
    def defect(self, X: np.ndarray, U: np.ndarray) -> float:
        md = 0.0
        for k in range(self.N):
            if self.zoh_consistent:
                xn = _zoh_step_sym(self.model, X[k], U[k],
                                   self.control_dt, self.n_sub)
            else:
                xn = _rk4_step_sym(self.model, X[k], U[k], self.h)
            xn = np.asarray(xn).reshape(-1)
            md = max(md, float(np.max(np.abs(xn - X[k + 1]))))
        return md
