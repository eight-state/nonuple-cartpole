"""Discrete-time TVLQR along a dense (1 ms) nominal.

The continuous-Riccati TVLQR (tvlqr.py) interpolates gains on an n_eval grid
(20 ms at n_eval=400 over 8 s) and designs in continuous time; along the n=7
swing-up that loop is UNSTABLE (monodromy rho=47.85). Here: exact ZOH
discretization (block matrix exponential) of the linearization at EVERY
control tick, backward discrete Riccati, per-tick gains, no interpolation.
"""
import numpy as np
import scipy.linalg as sla

from cartpole_race.lqr import make_Q, make_R, static_lqr, wrap_state_error


def zoh_AB(model, x, u, dt):
    A, B = model.linearize(x, float(u))
    nx = A.shape[0]
    M = np.zeros((nx + 1, nx + 1))
    M[:nx, :nx] = A * dt
    M[:nx, nx] = np.asarray(B).reshape(-1) * dt
    E = sla.expm(M)
    return E[:nx, :nx], E[:nx, nx]


class DiscreteTVLQR:
    """u_k = u_nom[k] - K_k (x - x_nom[k]); gains from discrete Riccati."""

    def __init__(self, model, X, U, dt, Qf=None, Q=None, R=None,
                 progress=False):
        n = model.n
        nx = model.nx
        N = len(U)
        assert len(X) == N + 1
        Q = make_Q(n) if Q is None else Q
        R = make_R() if R is None else R
        Rv = float(np.asarray(R).reshape(-1)[0])
        if Qf is None:
            _, Qf = static_lqr(model)
        self.model = model
        self.n = n
        self.X = X
        self.U = U
        self.dt = dt
        self.N = N
        Ad = np.empty((N, nx, nx))
        Bd = np.empty((N, nx))
        for k in range(N):
            Ad[k], Bd[k] = zoh_AB(model, X[k], U[k], dt)
            if progress and k % 1000 == 0:
                print(f"  zoh_AB {k}/{N}", flush=True)
        Kk = np.empty((N, nx))
        S = Qf.copy()
        for k in range(N - 1, -1, -1):
            a, b = Ad[k], Bd[k]
            sb = S @ b
            den = Rv + b @ sb
            kk = (a.T @ sb) / den          # K row: u = -kk^T e
            Kk[k] = kk
            Acl = a - np.outer(b, kk)
            S = Q + Rv * np.outer(kk, kk) + Acl.T @ S @ Acl
            S = 0.5 * (S + S.T)
        self.K = Kk
        self.Ad, self.Bd = Ad, Bd
        self.S0 = S

    def policy(self, x, t):
        k = min(max(int(round(t / self.dt)), 0), self.N - 1)
        e = wrap_state_error(x, self.X[k], self.n)
        return float(self.U[k] - self.K[k] @ e)

    def monodromy(self):
        nx = self.Ad.shape[1]
        M = np.eye(nx)
        for k in range(self.N):
            Acl = self.Ad[k] - np.outer(self.Bd[k], self.K[k])
            M = Acl @ M
        return float(np.max(np.abs(np.linalg.eigvals(M))))
