"""Direct trajectory optimization over the shared dynamics (CasADi Opti).

One OCP solver serves BOTH:
  - the MPC/trajopt CATCH (from a near-upright arrival, drive to upright while
    keeping every link inside an angle tube -- the constraint a fixed LQR gain
    cannot honor, which is why the LQR catch whips links past +-5deg);
  - the swing-up (from hanging, same machinery, looser link bounds).

Multiple-shooting with RK4 defects, using the SAME ``model.f`` ca.Function as
the simulator/linearizer (Principle 1: single-source dynamics). Returns the
optimal state/control trajectory; a TVLQR tracker (tvlqr.py) closes the loop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import casadi as ca
import numpy as np

from cartpole_race.dynamics import NLinkCartPole


@dataclass
class TrajOptResult:
    """Optimal trajectory from :func:`solve_trajopt`."""

    t: np.ndarray            # (N+1,) time grid
    x: np.ndarray            # (N+1, nx) states
    u: np.ndarray            # (N,) controls
    success: bool
    solver_status: str
    max_defect: float
    objective: float


def _rk4_step_sym(model: NLinkCartPole, x, u, h: float):
    """One symbolic RK4 step using the shared dynamics ``model.f``."""
    k1 = model.f(x, u)
    k2 = model.f(x + 0.5 * h * k1, u)
    k3 = model.f(x + 0.5 * h * k2, u)
    k4 = model.f(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _zoh_step_sym(model: NLinkCartPole, x, u, control_dt: float, n_sub: int):
    """Integrate one ZOH control interval with ``n_sub`` equal RK4 substeps.

    This mirrors :meth:`NLinkCartPole.rollout_zoh` EXACTLY: control held
    constant over ``control_dt``, integrated with ``n_sub`` RK4 substeps of
    size ``control_dt / n_sub`` (the simulator's fixed substep schedule). Using
    this in the collocation defect makes the transcription integrate
    identically to the real simulator (gate G2).
    """
    dt_sub = control_dt / n_sub
    xx = x
    for _ in range(n_sub):
        xx = _rk4_step_sym(model, xx, u, dt_sub)
    return xx


def solve_trajopt(
    model: NLinkCartPole,
    x0: np.ndarray,
    *,
    horizon_s: float,
    n_nodes: int,
    angle_tube_rad: float | None = None,
    terminal_tol_rad: float | None = None,
    terminal_vel_tol: float | None = None,
    force_bound: float | None = None,
    track_limit: float | None = None,
    qf_diag: np.ndarray | None = None,
    w_u: float = 1e-3,
    w_v: float = 0.0,
    x_target: np.ndarray | None = None,
    x_init_guess: np.ndarray | None = None,
    u_init_guess: np.ndarray | None = None,
    max_iter: int = 2000,
    print_level: int = 0,
    zoh_consistent: bool = False,
) -> TrajOptResult:
    """Solve a fixed-horizon OCP driving ``x0`` to ``x_target`` (default upright).

    Args:
        model: Shared dynamics object.
        x0: Initial state (fixed).
        horizon_s: Trajectory duration.
        n_nodes: Number of control intervals (N); N+1 knot points.
        angle_tube_rad: If set, ``|theta_i(t)| <= angle_tube`` at every node (the
            key catch constraint). None = no link-angle constraint.
        force_bound: ``|u| <= force_bound`` (default: spec force bound).
        track_limit: ``|x_cart| <= track_limit`` (default: spec track).
        qf_diag: Terminal-cost diagonal weights (default: heavy on angles/vels).
        w_u: Control-effort weight.
        x_target: Terminal target (default: upright equilibrium).
        x_init_guess / u_init_guess: Warm starts.
        max_iter / print_level: IPOPT settings.

    Returns:
        :class:`TrajOptResult`.
    """
    spec = model.spec
    nx = model.nx
    n = model.n
    N = n_nodes
    h = horizon_s / N

    # G2 integration consistency: when ``zoh_consistent`` is set, each control
    # interval spans exactly one simulator control tick (``control_dt``) and the
    # defect is integrated with the SAME fixed RK4 substep schedule that
    # :meth:`NLinkCartPole.rollout_zoh` uses (``n_sub`` substeps of
    # ``control_dt / n_sub``). Then a dense ZOH replay of the planned controls
    # reproduces the planned states to ~1e-6 (no integration mismatch).
    control_dt = spec.control_dt_s
    n_sub = max(1, int(np.ceil(control_dt / spec.rk4_max_step_s)))
    if zoh_consistent:
        h = control_dt
    fb = force_bound if force_bound is not None else spec.force_bound_n
    tl = track_limit if track_limit is not None else spec.track_half_length_m
    x_t = (np.asarray(x_target) if x_target is not None
           else np.asarray(model.x_equilibrium("up"))).reshape(-1)
    if qf_diag is None:
        qf_diag = np.concatenate(
            [[10.0], 200.0 * np.ones(n), [10.0], 50.0 * np.ones(n)]
        )

    opti = ca.Opti()
    X = opti.variable(nx, N + 1)
    U = opti.variable(1, N)

    # Initial condition.
    opti.subject_to(X[:, 0] == ca.DM(np.asarray(x0).reshape(-1)))

    # Dynamics defects (multiple shooting, shared f). When zoh_consistent, the
    # step integrates one control tick with the simulator's RK4 substeps.
    def _step(xk, uk):
        if zoh_consistent:
            return _zoh_step_sym(model, xk, uk, control_dt, n_sub)
        return _rk4_step_sym(model, xk, uk, h)

    for k in range(N):
        x_next = _step(X[:, k], U[0, k])
        opti.subject_to(X[:, k + 1] == x_next)

    # Bounds.
    opti.subject_to(opti.bounded(-fb, ca.vec(U), fb))
    opti.subject_to(opti.bounded(-tl, X[0, :], tl))
    if angle_tube_rad is not None:
        for i in range(1, 1 + n):
            opti.subject_to(opti.bounded(-angle_tube_rad, X[i, :], angle_tube_rad))
    # Hard terminal ball: drive angles AND velocities into a tiny neighbourhood
    # of the target (the real catch requirement -- a soft Qf just parks links at
    # the tube edge). Cart position free within track.
    if terminal_tol_rad is not None:
        # Separate velocity tolerance: the saturated hold basin is GENEROUS in
        # angle but tiny in velocity, so the terminal ANGLE can be tightened
        # cheaply while the velocity box stays at a sane width. Forcing the
        # terminal velocity to the (tiny) angle tol is what made the n=9 solve
        # intractable. Default = terminal_tol_rad (backward-compatible).
        vtol = terminal_tol_rad if terminal_vel_tol is None else terminal_vel_tol
        for i in range(1, 1 + n):  # angles -> ~0
            opti.subject_to(opti.bounded(x_t[i] - terminal_tol_rad,
                                         X[i, N], x_t[i] + terminal_tol_rad))
        for i in range(1 + n, nx):  # all velocities -> within vtol
            opti.subject_to(opti.bounded(-vtol, X[i, N], vtol))

    # Objective: control effort + terminal cost-to-go on the target.
    err_T = X[:, N] - ca.DM(x_t)
    Qf = ca.DM(np.diag(qf_diag))
    # Running velocity penalty (w_v): discourages a whip-fast swing-up. The
    # terminal ball is a HARD constraint, so this only smooths the PATH to
    # upright, it cannot trade away reaching it. Needed at n>=9 where the
    # force-optimal (w_v=0) swing-up is too fast for the TVLQR linearization
    # to track. Default 0.0 leaves all other callers unchanged.
    obj = w_u * ca.sumsqr(U) + ca.mtimes([err_T.T, Qf, err_T])
    if w_v > 0.0:
        obj = obj + w_v * ca.sumsqr(X[1 + n:, :])
    opti.minimize(obj)

    # Warm starts.
    if x_init_guess is not None:
        opti.set_initial(X, x_init_guess.T if x_init_guess.shape[0] == N + 1
                         else x_init_guess)
    else:
        # Linear interpolation x0 -> target as a default guess.
        guess = np.linspace(np.asarray(x0).reshape(-1), x_t, N + 1).T
        opti.set_initial(X, guess)
    if u_init_guess is not None:
        opti.set_initial(U, u_init_guess.reshape(1, N))

    # Linear solver: HSL (ma57/ma27/ma97) is NOT installed in this build
    # (libhsl.dll missing -> DYNAMIC_LIBRARY_FAILURE, IPOPT bails immediately,
    # returning the unconverged warm start as "failed"). Working solvers here are
    # MUMPS (default, robust) and SPRAL. Allow override via env for experiments.
    linear_solver = os.environ.get("CARTPOLE_LINEAR_SOLVER", "mumps")
    _atol = float(os.environ.get("CARTPOLE_ACCEPTABLE_TOL", "1e-6"))
    _aiter = int(os.environ.get("CARTPOLE_ACCEPTABLE_ITER", "15"))
    ipopt_opts = {"max_iter": max_iter, "print_level": print_level,
                  "linear_solver": linear_solver,
                  "tol": 1e-8, "acceptable_tol": _atol, "acceptable_iter": _aiter,
                  "mu_strategy": os.environ.get("CARTPOLE_MU_STRATEGY", "adaptive")}
    if os.environ.get("CARTPOLE_ACCEPTABLE_VIOL"):
        ipopt_opts["acceptable_constr_viol_tol"] = float(
            os.environ["CARTPOLE_ACCEPTABLE_VIOL"])
    if os.environ.get("CARTPOLE_BOUND_PUSH"):
        bp = float(os.environ["CARTPOLE_BOUND_PUSH"])
        ipopt_opts["bound_push"] = bp
        ipopt_opts["bound_frac"] = bp
    if os.environ.get("CARTPOLE_MAX_CPU_S"):
        ipopt_opts["max_cpu_time"] = float(os.environ["CARTPOLE_MAX_CPU_S"])
    _of = os.environ.get("CARTPOLE_IPOPT_OUTFILE")
    if _of:
        # Native IPOPT file output flushes per-iteration -> live progress under
        # nohup despite C-runtime stdout buffering.
        ipopt_opts["output_file"] = _of
        ipopt_opts["file_print_level"] = max(print_level, 5)
        ipopt_opts["print_frequency_iter"] = 1
    opti.solver("ipopt", {"print_time": False}, ipopt_opts)
    try:
        sol = opti.solve()
        status = "solved"
        ok = True
        Xv = np.array(sol.value(X)).reshape(nx, N + 1).T
        Uv = np.array(sol.value(U)).reshape(N)
        objv = float(sol.value(obj))
    except RuntimeError:
        # Recover the last iterate even on non-convergence.
        status = "failed"
        ok = False
        Xv = np.array(opti.debug.value(X)).reshape(nx, N + 1).T
        Uv = np.array(opti.debug.value(U)).reshape(N)
        objv = float("nan")

    # Defect check via the shared (numeric) dynamics.
    max_def = 0.0
    for k in range(N):
        xn = np.asarray(_step(Xv[k], Uv[k])).reshape(-1)
        max_def = max(max_def, float(np.max(np.abs(xn - Xv[k + 1]))))

    t_final = N * control_dt if zoh_consistent else horizon_s
    return TrajOptResult(
        t=np.linspace(0.0, t_final, N + 1), x=Xv, u=Uv,
        success=ok, solver_status=status, max_defect=max_def, objective=objv,
    )
