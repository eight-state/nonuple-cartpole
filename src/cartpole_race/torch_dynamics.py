"""Batched PyTorch clone of the EXACT n-link cart-pole EOM in dynamics.py.

This exists ONLY to make RL training fast (vectorized over thousands of envs).
It is verified bit-for-bit (to ~1e-12) against the authoritative CasADi
``NLinkCartPole.f_num`` at construction time via :func:`verify_against_casadi`.
The GATE never uses this file -- the gate runs the real CasADi
``rollout_zoh``. This is the training simulator only.

Same coordinate convention: q=[x_cart, theta_1..n], theta=0 => UP, pi => DOWN.
Same 150 N cart-force saturation, same 1 ms ZOH, same RK4 substeps.
"""

from __future__ import annotations

import numpy as np
import torch

from cartpole_race.env_spec import CartPoleSpec


class TorchCartPole:
    """Vectorized EOM + RK4 ZOH integrator, exact clone of dynamics.py."""

    def __init__(self, spec: CartPoleSpec, device: str = "cpu",
                 dtype: torch.dtype = torch.float64) -> None:
        self.spec = spec
        self.n = spec.n_links
        self.nx = spec.nx
        self.device = device
        self.dtype = dtype
        n = self.n
        mc = spec.cart_mass_kg
        ms = np.array(spec.link_masses_kg, dtype=float)
        ll = np.array(spec.link_lengths_m, dtype=float)
        self.g = spec.gravity_m_s2
        self.mtot = float(mc + ms.sum())
        # reach matrix a[i,j]: link j's lever on link i's COM
        a = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j < i:
                    a[i, j] = ll[j]
                elif j == i:
                    a[i, j] = 0.5 * ll[i]
        Iv = ms * ll ** 2 / 12.0
        W = np.einsum("i,ik,il->kl", ms, a, a)
        hc = np.einsum("i,ik->k", ms, a)
        self.W = torch.tensor(W, device=device, dtype=dtype)
        self.hc = torch.tensor(hc, device=device, dtype=dtype)
        self.Iv = torch.tensor(Iv, device=device, dtype=dtype)
        self.fbound = spec.force_bound_n
        # precompute static index/structure tensors (avoid per-call rebuild)
        nn = n + 1
        eye = torch.eye(n, device=device, dtype=dtype)
        self._delta = (eye[:, None, :] - eye[None, :, :])  # (n,n,n) [k,l,r]
        self._diag = torch.diag_embed(self.Iv)             # (n,n)
        self._idx = torch.arange(n, device=device)
        self._nn = nn

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """xdot = f(x,u). x:(B,nx), u:(B,). Exact analytic, no autograd."""
        B = x.shape[0]
        n = self.n
        nn = n + 1
        th = x[:, 1:1 + n]
        qd = x[:, n + 1:]
        c = torch.cos(th)
        s = torch.sin(th)
        dth = th[:, :, None] - th[:, None, :]
        cdth = torch.cos(dth)
        sdth = torch.sin(dth)
        hc_c = self.hc[None, :] * c  # (B,n)
        # Mass matrix (build via cat, avoids slow advanced-index scatter)
        col0 = torch.cat([torch.full((B, 1), self.mtot, device=self.device,
                                     dtype=self.dtype), hc_c], dim=1)  # (B,nn)
        Mang = self.W[None] * cdth + self._diag[None]  # (B,n,n)
        top = torch.cat([col0[:, None, :1], hc_c[:, None, :]], dim=2)  # (B,1,nn)
        bottom = torch.cat([hc_c[:, :, None], Mang], dim=2)            # (B,n,nn)
        M = torch.cat([top, bottom], dim=1)                           # (B,nn,nn)
        # dMdq[b,i,j,r] = d M[i,j] / d theta_r  (q idx 1+r -> theta_r; cart->0)
        dMdq = torch.zeros(B, nn, nn, nn, device=self.device, dtype=self.dtype)
        # cart-angle coupling: d M[0,1+k]/d th_k = -hc[k] sin th_k (diagonal in k,r)
        cart_coup = (-self.hc[None, :] * s)  # (B,n)
        dMdq[:, 0, 1:, 1:] = torch.diag_embed(cart_coup)
        dMdq[:, 1:, 0, 1:] = torch.diag_embed(cart_coup)
        # angle-angle: -W[k,l] sin(th_k-th_l) (delta_rk - delta_rl)
        WS = self.W[None] * sdth                       # (B,n,n)
        block = -WS[:, :, :, None] * self._delta[None]  # (B,n,n,n)
        dMdq[:, 1:, 1:, 1:] = block
        # Christoffel: C_k = sum_ij (dM[k,i]/dq_j - 0.5 dM[i,j]/dq_k) qd_i qd_j
        Cqd = (torch.einsum("bkij,bi,bj->bk", dMdq, qd, qd)
               - 0.5 * torch.einsum("bijk,bi,bj->bk", dMdq, qd, qd))
        G = torch.zeros(B, nn, device=self.device, dtype=self.dtype)
        G[:, 1:] = -self.g * s * self.hc[None, :]
        Q = torch.zeros(B, nn, device=self.device, dtype=self.dtype)
        Q[:, 0] = u
        rhs = Q - Cqd - G
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        return torch.cat([qd, qdd], dim=1)

    def rk4_step(self, x, u, dt):
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * dt * k1, u)
        k3 = self.f(x + 0.5 * dt * k2, u)
        k4 = self.f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_zoh(self, x, u_raw, control_dt=None, rk4_max=None):
        """One control tick: clip force to +/-fbound, RK4 substeps. Exact ZOH."""
        control_dt = control_dt or self.spec.control_dt_s
        rk4_max = rk4_max or self.spec.rk4_max_step_s
        u = torch.clamp(u_raw, -self.fbound, self.fbound)
        n_sub = max(1, int(np.ceil(control_dt / rk4_max)))
        dt_sub = control_dt / n_sub
        for _ in range(n_sub):
            x = self.rk4_step(x, u, dt_sub)
        return x


def verify_against_casadi(n: int = 7, n_pts: int = 300, seed: int = 0):
    """Return max abs error of TorchCartPole.f vs CasADi f_num over random pts."""
    from cartpole_race.dynamics import NLinkCartPole
    spec = CartPoleSpec().with_n_links(n)
    cas = NLinkCartPole(spec)
    tc = TorchCartPole(spec)
    rng = np.random.default_rng(seed)
    maxerr = 0.0
    for _ in range(n_pts):
        xv = rng.normal(0, 1.5, spec.nx)
        uv = rng.normal(0, 80)
        ref = cas.f_num(xv, uv)
        got = tc.f(torch.tensor(xv[None]), torch.tensor([uv]))[0].numpy()
        maxerr = max(maxerr, float(np.max(np.abs(ref - got))))
    return maxerr
