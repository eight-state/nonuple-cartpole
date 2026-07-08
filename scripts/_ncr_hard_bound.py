"""Controller-INDEPENDENT null-controllable-region analysis at upright.

Per unstable mode i (real lambda_i > 0, biorthogonal left vec w_i):
    z_i = w_i^T x,   dz_i/dt = lambda_i z_i + (w_i^T B) u,  |u| <= umax
HARD necessary condition for recoverability under ANY admissible input:
    |z_i| <= umax * |w_i^T B| / lambda_i   =: c_i
Joint NCR support function (exact, convex):
    h(eta) = umax * integral_0^inf |B^T expm(-A_u^T t) eta| dt
Membership x in NCR  iff  eta^T x <= h(eta) for all eta.

We compute:
 1. c_i per mode for n=6,7 (the hard widths).
 2. For pure-angle perturbations (the _radv2_nlbasin scenario), the max
    perturbation scale (deg) at which the per-mode necessary condition still
    holds, per random direction -> controller-independent CEILING on the
    angle-basin. Compare vs the measured saturated-LQR basin.
 3. Same for pure angular-rate perturbations (rad/s).
 4. Joint-NCR membership check by support-function falsification over sampled
    eta directions (tighter than per-mode).
"""
import os, sys
from pathlib import Path
import numpy as np

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from scipy.linalg import expm


def unstable_modal(A, B):
    """Return (lams, W, cvec) for real unstable modes with biorthogonal w_i."""
    lam, V = np.linalg.eig(A)
    Wfull = np.linalg.inv(V)  # rows are biorthogonal left eigenvectors
    idx = [i for i in range(len(lam)) if lam[i].real > 1e-9]
    # all unstable modes should be real for upright n-pendulum
    lams = np.array([lam[i].real for i in idx])
    W = np.array([Wfull[i].real for i in idx])
    order = np.argsort(lams)
    return lams[order], W[order]


def run(n):
    spec = CartPoleSpec().with_n_links(n)
    m = NLinkCartPole(spec)
    umax = spec.force_bound_n
    xup = m.x_equilibrium("up")
    A, B = m.linearize(xup, 0.0)
    B = B.reshape(-1)
    lams, W = unstable_modal(A, B)
    print(f"\n===== n={n}  umax={umax} N =====")
    print(f"{'lam':>10} {'|w.B|':>12} {'c_i = umax|w.B|/lam':>22}")
    cs = []
    for lam_i, w in zip(lams, W):
        wb = abs(w @ B)
        c = umax * wb / lam_i
        cs.append(c)
        print(f"{lam_i:10.4f} {wb:12.4e} {c:22.4e}")
    cs = np.array(cs)

    nx = m.nx
    rng = np.random.default_rng(1)  # same seed as _radv2_nlbasin
    # --- scenario A: pure angle perturbations (nlbasin scenario) ---
    print("\n[A] pure-angle perturbation ceiling (controller-independent):")
    print("  per random direction: max scale s (deg) s.t. all |z_i| <= c_i")
    ceil_deg = []
    for _ in range(12):
        d = rng.normal(0, 1.0, n)  # direction in angle space
        dx = np.zeros(nx); dx[1:1 + n] = np.deg2rad(d)
        z = np.abs(W @ dx)  # |z_i| per unit-deg-sigma draw
        s_max = np.min(np.where(z > 0, cs / z, np.inf))
        ceil_deg.append(s_max)
    ceil_deg = np.array(ceil_deg)
    print(f"  draws (sigma=1deg shape): ceiling deg: min {ceil_deg.min():.4f} "
          f"med {np.median(ceil_deg):.4f} max {ceil_deg.max():.4f}")

    # --- scenario B: pure angular-rate perturbations ---
    print("[B] pure ang-rate perturbation ceiling (rad/s):")
    ceil_w = []
    for _ in range(12):
        d = rng.normal(0, 1.0, n)
        dx = np.zeros(nx); dx[2 + n:] = d
        z = np.abs(W @ dx)
        ceil_w.append(np.min(np.where(z > 0, cs / z, np.inf)))
    ceil_w = np.array(ceil_w)
    print(f"  ceiling rad/s: min {ceil_w.min():.4e} med {np.median(ceil_w):.4e} "
          f"max {ceil_w.max():.4e}")

    # --- scenario C: full gate-style perturbation sigma=0.02 everywhere ---
    print("[C] gate-style sigma=0.02 (pos/ang/vel) draws: fraction passing the")
    print("    per-mode NECESSARY condition (any controller):")
    npass = 0; N = 200
    margins = []
    for _ in range(N):
        dx = np.zeros(nx)
        dx[0] = rng.normal(0, 0.02)
        dx[1:1 + n] = rng.normal(0, 0.02, n)
        dx[1 + n] = rng.normal(0, 0.02)
        dx[2 + n:] = rng.normal(0, 0.02, n)
        z = np.abs(W @ dx)
        ratio = np.max(z / cs)
        margins.append(ratio)
        if ratio <= 1.0:
            npass += 1
    margins = np.array(margins)
    print(f"  pass {npass}/{N}; worst-mode ratio |z|/c: med {np.median(margins):.2f} "
          f"min {margins.min():.2f} max {margins.max():.2f}")
    print(f"  (ratio>1 means PROVABLY unrecoverable under any bounded input)")

    # --- joint NCR support function along the fast left eigvec (tightness) ---
    Au = np.diag(lams)  # modal coords: decoupled
    Bu = W @ B
    # support along e_i in modal coords == c_i exactly (decoupled), so the
    # per-mode bound IS the box; joint set is smaller. Estimate shrinkage via
    # random eta in modal space:
    def h(eta, T=4.0, K=4000):
        ts = np.linspace(0, T, K)
        vals = np.abs((Bu * np.exp(-np.outer(ts, lams))) @ eta)
        return umax * np.trapezoid(vals, ts)
    rng2 = np.random.default_rng(7)
    shrink = []
    for _ in range(64):
        eta = rng2.normal(0, 1, len(lams))
        eta /= np.linalg.norm(eta)
        box = np.sum(np.abs(eta) * cs)  # support of the per-mode box
        shrink.append(h(eta) / box)
    shrink = np.array(shrink)
    print(f"[D] joint-NCR vs per-mode box support ratio: med {np.median(shrink):.3f} "
          f"min {shrink.min():.3f}  (1.0 = box tight)")
    return lams, W, cs


for n in (6, 7):
    run(n)
