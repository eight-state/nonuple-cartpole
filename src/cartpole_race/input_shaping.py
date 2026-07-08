"""Modal-nulling final-approach command shaper (the surviving frontier tool H1).

The zig-zag differential mode is the wall: unstable (~26 rad/s) AND
near-uncontrollable by the cart, so the static/TVLQR stabiliser has a
microscopic zig-zag basin (~0.02-0.1 deg). This shaper does NOT stabilise the
mode (it is unstable -- it regrows). It buys the *cleanest possible handoff*:
over a short final-approach window it designs the cart-force sequence that drives
the projection of the state onto the unstable differential ("zig-zag") eigenmode
to ~0 at the handoff instant, so the downstream stabiliser inherits a near-silent
zig-zag and can grab it before it regrows. Useful as the terminal cap of a
swing-up, feeding the catch.

Mode definition (authoritative): the zig-zag mode is mode-0 of the
angle-acceleration block ``G = d(qdd)/dq`` restricted to the angle subspace
(NOT the naive alternating +-1 pattern, which is ill-conditioned and saturates).
The shape for the locked 6-link spec is ~``[-0.29, 0.62, -0.59, 0.38, -0.18,
0.08]``. We target BOTH the modal coordinate ``q_z = c_pos @ (x - xup)`` and the
modal velocity ``qd_z = c_vel @ (x - xup)`` at the terminal node.

Method: linearise at upright; build the finite-horizon ZOH controllability map
(Phi, Gamma) matching ``rollout_zoh``; solve the min-effort force sequence that
zeros the two zig-zag modal coordinates (position + velocity) at the terminal
node.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from cartpole_race.dynamics import NLinkCartPole


@dataclass
class ZigzagMode:
    """The G-block zig-zag mode targeted for nulling.

    Attributes:
        growth_rate2: eigenvalue ``lam`` of G for the mode (rad^2/s^2). The
            growth rate is ``sqrt(lam)`` when positive (unstable).
        shape: unit-norm 6-vector angle shape of the zig-zag mode.
        c_pos: row-vector (length nx) s.t. ``q_z = c_pos @ (x - xup)``.
        c_vel: row-vector (length nx) s.t. ``qd_z = c_vel @ (x - xup)``.
    """

    growth_rate2: float
    shape: np.ndarray
    c_pos: np.ndarray
    c_vel: np.ndarray


def identify_zigzag_mode(model: NLinkCartPole) -> ZigzagMode:
    """Return the G-block zig-zag mode (mode 0) and its (coord, velocity) maps.

    The zig-zag mode is the highest-growth eigenvector of the angle-acceleration
    block ``G = A[nq+1:, 1:nq]`` (angles -> angular accelerations, angle
    subspace). This matches ``frontier_lib.zigzag_proj``.
    """
    n = model.n
    nq = n + 1
    xup = np.asarray(model.x_equilibrium("up"))
    A, _ = model.linearize(xup, 0.0)
    A = np.asarray(A)
    # Angle-acceleration block G = d(qdd_angles)/d(angles).
    A21 = A[nq:, :nq]
    G = A21[1:, 1:]
    w2, V = np.linalg.eig(G)
    idx = np.argsort(-w2.real)
    lam = w2[idx].real
    Vr = V[:, idx].real
    v = Vr[:, 0] / np.linalg.norm(Vr[:, 0])
    c_pos = np.zeros(model.nx)
    c_vel = np.zeros(model.nx)
    c_pos[1:1 + n] = v
    c_vel[nq + 1:nq + 1 + n] = v
    return ZigzagMode(float(lam[0]), v, c_pos, c_vel)


def _zoh_maps(A: np.ndarray, B: np.ndarray, dt: float, N: int):
    """Discrete ZOH lifted maps ``Phi = Ad^N`` and ``Gamma`` (nx, N).

    Column ``k`` of Gamma is ``Ad^(N-1-k) Bd`` -- the effect of the force held
    during tick ``k`` on the terminal state at node ``N``. This matches the
    per-tick ZOH integration used by ``rollout_zoh`` (continuous A, B
    discretised by the matrix exponential).
    """
    nx = A.shape[0]
    M = np.zeros((nx + 1, nx + 1))
    M[:nx, :nx] = A * dt
    M[:nx, nx:] = B * dt
    E = sla.expm(M)
    Ad = E[:nx, :nx]
    Bd = E[:nx, nx:]
    Gamma = np.zeros((nx, N))
    for k in range(N):
        Gamma[:, k] = (np.linalg.matrix_power(Ad, N - 1 - k) @ Bd).ravel()
    Phi = np.linalg.matrix_power(Ad, N)
    return Ad, Bd, Phi, Gamma


def design_nulling_command(
    model: NLinkCartPole,
    x0: np.ndarray,
    *,
    horizon_s: float = 0.06,
    mode: ZigzagMode | None = None,
    force_bound: float | None = None,
    null_velocity: bool = True,
    ridge: float = 0.0,
) -> np.ndarray:
    """Min-effort ZOH force sequence that zeros the zig-zag modal coord at T.

    Solves ``min ||u||`` subject to ``c_pos @ (Phi dx0 + Gamma u) = 0`` and
    (if ``null_velocity``) ``c_vel @ (Phi dx0 + Gamma u) = 0``, where
    ``dx0 = x0 - xup``. The min-norm solution is returned, clipped to the force
    bound. Length-N sequence at the sim control rate.

    Args:
        model: shared dynamics object.
        x0: initial (arrival) state.
        horizon_s: shaping window length (seconds).
        mode: zig-zag mode (computed if None).
        force_bound: clip bound; defaults to spec.force_bound_n.
        null_velocity: also null the modal velocity at T (recommended).
        ridge: optional Tikhonov term on the min-norm solve for conditioning.

    Returns:
        Per-tick force sequence (length N).
    """
    spec = model.spec
    dt = spec.control_dt_s
    N = max(1, int(round(horizon_s / dt)))
    fb = force_bound if force_bound is not None else spec.force_bound_n
    if mode is None:
        mode = identify_zigzag_mode(model)
    xup = np.asarray(model.x_equilibrium("up"))
    A, B = model.linearize(xup, 0.0)
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1, 1)
    _, _, Phi, Gamma = _zoh_maps(A, B, dt, N)
    dx0 = np.asarray(x0).reshape(-1) - xup

    rows = [mode.c_pos @ Gamma]
    b = [-(mode.c_pos @ Phi @ dx0)]
    if null_velocity:
        rows.append(mode.c_vel @ Gamma)
        b.append(-(mode.c_vel @ Phi @ dx0))
    Mmat = np.vstack(rows)
    bvec = np.array(b)
    # Min-norm: u = M^T (M M^T + ridge I)^-1 b.
    reg = max(ridge, 1e-12)
    u = Mmat.T @ np.linalg.solve(Mmat @ Mmat.T + reg * np.eye(Mmat.shape[0]), bvec)
    return np.clip(u, -fb, fb)


class ZigzagCapPolicy:
    """Feedforward terminal anti-zig-zag cap as a ``(state, t) -> force`` policy.

    Plays a precomputed min-effort nulling sequence over the final-approach
    window (one-shot, open-loop feedforward). After the window expires the cap
    outputs zero (or hands off to a downstream stabiliser supplied separately).
    """

    def __init__(
        self,
        model: NLinkCartPole,
        x0: np.ndarray,
        *,
        horizon_s: float = 0.06,
        force_bound: float | None = None,
        null_velocity: bool = True,
    ) -> None:
        self.model = model
        self.dt = model.spec.control_dt_s
        self.u_seq = design_nulling_command(
            model, x0, horizon_s=horizon_s, force_bound=force_bound,
            null_velocity=null_velocity,
        )
        self.N = len(self.u_seq)

    def __call__(self, x: np.ndarray, t: float) -> float:
        del x
        k = int(round(t / self.dt))
        if 0 <= k < self.N:
            return float(self.u_seq[k])
        return 0.0
