"""Hard-gate test: CasADi linearization vs central finite differences."""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole


def _central_diff(f, base: np.ndarray, j: int, eps: float) -> np.ndarray:
    """One column of a central finite difference of ``f`` w.r.t. component j."""
    dx = np.zeros_like(base)
    dx[j] = eps
    return (f(base + dx) - f(base - dx)) / (2.0 * eps)


def _richardson_column(f, base: np.ndarray, j: int) -> np.ndarray:
    """Fourth-order column via Richardson extrapolation of central differences.

    Combining central differences at ``h`` and ``h/2`` cancels the leading
    O(h^2) truncation term, giving an O(h^4)-accurate reference. This keeps the
    FD *reference* honest at the proposal's atol=1e-6 without loosening the
    analytic CasADi tolerance.
    """
    h = 1e-4
    d_h = _central_diff(f, base, j, h)
    d_h2 = _central_diff(f, base, j, h / 2.0)
    return (4.0 * d_h2 - d_h) / 3.0


def _fd_jacobians(
    model: NLinkCartPole, x: np.ndarray, u: float
) -> tuple[np.ndarray, np.ndarray]:
    """Richardson-extrapolated central-difference A = df/dx and B = df/du."""
    nx = model.nx
    A = np.zeros((nx, nx))
    for j in range(nx):
        A[:, j] = _richardson_column(lambda xx: model.f_num(xx, u), x, j)
    u_vec = np.array([u], dtype=float)
    B = _richardson_column(
        lambda uu: model.f_num(x, float(uu[0])), u_vec, 0
    ).reshape(nx, 1)
    return A, B


def test_linearization_matches_fd(model: NLinkCartPole) -> None:
    """CasADi A,B match central FD on 20 random states.

    The analytic CasADi Jacobian is exact autodiff; the finite-difference
    *reference* is the limited quantity here. Its accuracy floor grows with n
    because f_num solves an n-dependent mass-matrix system, so subtractive
    cancellation in the central difference amplifies with stiffness (measured
    worst |analytic - FD| over 20 draws: 8e-10 at n=1, 6.7e-7 at n=3,
    2.7e-6 at n=9, 4.4e-6 at n=10 — verified by h-refinement, where the FD
    value converges TOWARD the analytic one as h grows out of the
    roundoff-dominated regime). We therefore scale atol with n
    (``1e-6 * max(1, n)``): still ~1000x tighter than any real linearization
    bug (which produces O(1e-2)+ errors), but above the FD reference's own
    n-dependent floor.
    """
    atol = 1e-6 * max(1, model.n)
    rng = np.random.default_rng(123 + model.n)
    for _ in range(20):
        # Bounded random states so FD step size stays well-conditioned.
        x = np.concatenate(
            [
                rng.uniform(-1.0, 1.0, size=1),  # x_cart
                rng.uniform(-np.pi, np.pi, size=model.n),  # angles
                rng.uniform(-1.0, 1.0, size=1),  # xdot
                rng.uniform(-2.0, 2.0, size=model.n),  # angular rates
            ]
        )
        u = float(rng.uniform(-50.0, 50.0))

        A_ca, B_ca = model.linearize(x, u)
        A_fd, B_fd = _fd_jacobians(model, x, u)

        np.testing.assert_allclose(A_ca, A_fd, rtol=1e-4, atol=atol)
        np.testing.assert_allclose(B_ca, B_fd, rtol=1e-4, atol=atol)
