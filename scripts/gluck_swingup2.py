"""Gluck/Kugi-style swing-up v2 -- PHYSICAL n=4 nominal.

Method (per C:/Users/verom/.building/gaps/gluck-kugi-method.md):
exact I/O inversion (cart accel = input via affine-in-force inversion of the
cart-accel row of f_num) + stable inversion of the unstable angle internal
dynamics.

Refinements over scripts/gluck_swingup.py (which left n>=3 degenerate: status=2
singular Jacobian, ~2850-7650 m/s^2 absurd accel):

  1. ACCEL SATURATION: cart accel is reparametrized through a smooth tanh so
     |s..*(t)| <= a_max by construction:
        s..*(t) = a_max * tanh( (sum_i p_i sin(i*pi*t/T)) / a_max ).
     The sine basis vanishes at t=0,T => s..*(0)=s..*(T)=0 (C0 feedforward).

  2. MULTIPLE-SHOOTING STABLE INVERSION (the key fix the original lacked):
     a single global scipy.solve_bvp over 2n GLOBAL Fourier params couples every
     node to every param through the UNSTABLE internal dynamics -> ill-conditioned
     Newton Jacobian (status=2) for n>=3. Instead split [0,T] into M segments,
     carry the segment-boundary angle states as unknowns, RK4-integrate each
     short segment forward (short horizon caps the exponential growth of the
     unstable modes), and solve continuity + BCs + accel params with
     scipy.optimize.least_squares (trf). Far better conditioned. The whole
     M-segment integration is VECTORIZED with a CasADi .map(M) so a residual
     eval is nsub*4 CasADi calls, not M*nsub*4.

  3. CONTINUATION in link count: warm-start n from the converged n-1 solution
     (resample boundary states, copy last link for the new link, pad params).
     Also continuation in a_max (loosen then tighten) if a stage stalls.

  4. Equivalent cart FORCE recovered pose-by-pose (cart accel affine in force);
     the recovered FORCE is replayed through model.rollout_zoh at 1 ms to test
     dynamic consistency on the real model.

Usage: gluck_swingup2.py [nmax] [--save]
"""
import os, sys, time, json, warnings
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")
import casadi as ca
from scipy.optimize import least_squares
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.dynamics import NLinkCartPole

LOG = None
def log(*a):
    print(*a, flush=True)
    if LOG is not None:
        print(*a, file=LOG, flush=True)


def make_model(n):
    return NLinkCartPole(CartPoleSpec().with_n_links(n))


def build_fns(m):
    """Symbolic angle-accel(phi,phid,a) and force(phi,phid,a) via affine-in-force
    inversion of the cart-acceleration row of the dynamics."""
    n = m.n; ci = n + 1
    phi = ca.SX.sym("phi", n); phid = ca.SX.sym("phid", n); a = ca.SX.sym("a")
    x = ca.vertcat(0.0, phi, 0.0, phid)
    xd0 = m.f(x=x, u=0.0)["xdot"]; xd1 = m.f(x=x, u=1.0)["xdot"]
    c0 = xd0[ci]; c1 = xd1[ci] - c0; u = (a - c0) / c1
    xdu = m.f(x=x, u=u)["xdot"]
    fang = ca.Function("fang", [phi, phid, a], [xdu[ci + 1: ci + 1 + n]])
    fforce = ca.Function("fforce", [phi, phid, a], [u])
    return fang, fforce


def accel_vec(t, p, T, a_max, NP):
    """Saturated cart accel at array of times t (t may be array)."""
    t = np.atleast_1d(t)
    s = np.zeros_like(t, dtype=float)
    for i in range(NP):
        s = s + p[i] * np.sin((i + 1) * np.pi * t / T)
    return a_max * np.tanh(s / a_max)


class MultiShoot:
    """Vectorized multiple-shooting solver for the angle internal dynamics."""

    def __init__(self, m, fang, T, a_max, NP, M, nsub):
        self.m = m; self.fang = fang; self.n = m.n
        self.T = T; self.a_max = a_max; self.NP = NP; self.M = M; self.nsub = nsub
        self.ts = np.linspace(0, T, M + 1)
        self.fmap = fang.map(M)
        self.nz = 2 * m.n

    def integrate_all(self, Zstarts, p):
        """RK4-integrate ALL M segments forward by one segment each, in lockstep.
        Zstarts: (M, 2n) start state of each segment. Returns (M, 2n) end states."""
        n = self.n; M = self.M; nsub = self.nsub
        Z = Zstarts.T.copy()                 # (2n, M)
        t0 = self.ts[:-1]                     # (M,)
        h = (self.ts[1:] - self.ts[:-1])     # (M,)
        dt = h / nsub                         # (M,)

        def deriv(Z, tcur):
            phi = Z[:n, :]; phid = Z[n:, :]
            a = accel_vec(tcur, p, self.T, self.a_max, self.NP)  # (M,)
            aa = np.asarray(self.fmap(phi, phid, a.reshape(1, -1)))  # (n, M)
            return np.vstack([phid, aa])

        tcur = t0.copy()
        for _ in range(nsub):
            k1 = deriv(Z, tcur)
            k2 = deriv(Z + 0.5 * dt * k1, tcur + 0.5 * dt)
            k3 = deriv(Z + 0.5 * dt * k2, tcur + 0.5 * dt)
            k4 = deriv(Z + dt * k3, tcur + dt)
            Z = Z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            tcur = tcur + dt
        return Z.T                            # (M, 2n)

    def residual(self, w):
        n = self.n; M = self.M; nz = self.nz
        Z = w[:nz * (M + 1)].reshape(M + 1, nz)
        p = w[nz * (M + 1):]
        zend = self.integrate_all(Z[:M], p)   # (M, 2n)
        cont = (Z[1:] - zend).reshape(-1)
        bc = np.concatenate([Z[0, :n] - np.pi, Z[0, n:], Z[M, :n], Z[M, n:]])
        return np.concatenate([bc, cont])

    def jac_sparsity(self):
        """Block sparsity of the residual Jacobian for cheap grouped finite-diff.
        Rows: [4n BC] + [nz per continuity segment]. Cols: [nz*(M+1) states]+[NP]."""
        from scipy.sparse import lil_matrix
        n = self.n; nz = self.nz; M = self.M; NP = self.NP
        nrows = 4 * n + nz * M
        ncols = nz * (M + 1) + NP
        S = lil_matrix((nrows, ncols))
        # BC rows: depend on Z[0] (first nz cols) and Z[M] (last state block)
        S[0:2 * n, 0:nz] = 1                       # start BC -> Z[0]
        S[2 * n:4 * n, nz * M:nz * (M + 1)] = 1    # end BC -> Z[M]
        # continuity rows for segment j depend on Z[j], Z[j+1], and all p
        r0 = 4 * n
        for j in range(M):
            rows = slice(r0 + j * nz, r0 + (j + 1) * nz)
            S[rows, nz * j:nz * (j + 1)] = 1       # Z[j]
            S[rows, nz * (j + 1):nz * (j + 2)] = 1  # Z[j+1]
            S[rows, nz * (M + 1):] = 1             # params p
        return S.tocsr()

    def solve(self, Z0, p0, maxfev=140, ftol=1e-6, xtol=1e-8):
        w0 = np.concatenate([Z0.reshape(-1), p0])
        res = least_squares(self.residual, w0, method="trf",
                            ftol=ftol, xtol=xtol, gtol=1e-10,
                            jac_sparsity=self.jac_sparsity(),
                            max_nfev=maxfev, verbose=0)
        Z = res.x[:self.nz * (self.M + 1)].reshape(self.M + 1, self.nz)
        p = res.x[self.nz * (self.M + 1):]
        return res, Z, p


def smooth_Z(n, ts, T):
    s = ts / T
    sh = 1.0 - (3 * s**2 - 2 * s**3)
    dsh = -(6 * s - 6 * s**2) / T
    Z = np.zeros((len(ts), 2 * n))
    for j in range(n):
        Z[:, j] = np.pi * sh
        Z[:, n + j] = np.pi * dsh
    return Z


def warm_Z(prev_Z, prev_ts, prev_n, new_n, new_ts, prev_p, new_NP):
    """Resample previous boundary states onto the new mesh; copy last link for the
    new link; pad Fourier coeffs."""
    M1 = len(new_ts)
    Z = np.zeros((M1, 2 * new_n))
    for j in range(new_n):
        src = min(j, prev_n - 1)
        ang = np.interp(new_ts, prev_ts, prev_Z[:, src])
        rate = np.interp(new_ts, prev_ts, prev_Z[:, prev_n + src])
        Z[:, j] = ang
        Z[:, new_n + j] = rate
    p = np.zeros(new_NP)
    p[:len(prev_p)] = prev_p
    return Z, p


def reconstruct(m, fang, Z, p, ts, T, a_max, NP, dt):
    """Reconstruct the multiple-shooting trajectory on the 1ms grid by integrating
    each segment from ITS OWN shooting node Z[j] (NOT globally from Z[0]) -- a
    single global forward integration would blow up the unstable internal
    dynamics. Each control-grid point is integrated from the start of its segment.
    """
    n = m.n
    N = int(round(T / dt))
    tg = np.linspace(0, T, N + 1)
    M = len(ts) - 1
    phi = np.zeros((n, N + 1)); phid = np.zeros((n, N + 1)); a = np.zeros(N + 1)

    def deriv(zz, tt):
        aa = np.asarray(fang(zz[:n], zz[n:], accel_vec(tt, p, T, a_max, NP)[0])).reshape(-1)
        return np.concatenate([zz[n:], aa])

    def rk4(z, t0, t1, sub):
        ddt = (t1 - t0) / sub; t = t0
        for _ in range(sub):
            k1 = deriv(z, t); k2 = deriv(z + 0.5 * ddt * k1, t + 0.5 * ddt)
            k3 = deriv(z + 0.5 * ddt * k2, t + 0.5 * ddt); k4 = deriv(z + ddt * k3, t + ddt)
            z = z + (ddt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4); t += ddt
        return z

    # Incremental O(N) forward integration; reset to the shooting node Z[seg]
    # at each segment boundary so the unstable dynamics never integrate across
    # more than one segment (this is the whole point of multiple shooting).
    seg = 0
    z = Z[0].copy()
    for k in range(N + 1):
        t = tg[k]
        if seg < M - 1 and t >= ts[seg + 1] - 1e-12:
            seg += 1
            z = Z[seg].copy()          # snap to shooting node at boundary
        phi[:, k] = z[:n]; phid[:, k] = z[n:]
        a[k] = accel_vec(t, p, T, a_max, NP)[0]
        if k < N:
            z = rk4(z, t, tg[k + 1], 2)
    return tg, phi, phid, a


def evaluate(m, fang, fforce, Z, p, ts, T, a_max, NP, label, save_path=None):
    n = m.n; dt = m.spec.control_dt_s
    tg, phi, phid, a = reconstruct(m, fang, Z, p, ts, T, a_max, NP, dt)
    N = len(tg) - 1

    termdeg = float(np.rad2deg(np.max(np.abs(((phi[:, -1] + np.pi) % (2 * np.pi)) - np.pi))))
    termrate = float(np.max(np.abs(phid[:, -1])))
    # shooting-node terminal (the optimizer's own terminal BC)
    node_termdeg = float(np.rad2deg(np.max(np.abs(((Z[-1, :n] + np.pi) % (2 * np.pi)) - np.pi))))

    forces = np.asarray(fforce.map(N + 1)(phi, phid, a.reshape(1, -1))).reshape(-1)
    peak_a = float(np.nanmax(np.abs(a)))
    peak_F = float(np.nanmax(np.abs(forces)))

    sdot = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tg))])
    spos = np.concatenate([[0.0], np.cumsum(0.5 * (sdot[1:] + sdot[:-1]) * np.diff(tg))])
    X = np.zeros((N + 1, m.nx))
    X[:, 0] = spos; X[:, 1:1 + n] = phi.T; X[:, n + 1] = sdot; X[:, n + 2:2 + 2 * n] = phid.T

    u_ff = forces[:-1].copy()
    def policy(x, t):
        k = int(round(t / dt)); k = max(0, min(k, len(u_ff) - 1)); return u_ff[k]
    x0 = m.x_equilibrium("down")
    t_log, x_log, u_log = m.rollout_zoh(x0, policy, T, dt, m.spec.rk4_max_step_s)
    phi_replay = x_log[-1, 1:1 + n]
    replay_termdeg = float(np.rad2deg(np.max(np.abs(((phi_replay + np.pi) % (2 * np.pi)) - np.pi))))
    dev = float(np.rad2deg(np.max(np.abs(x_log[:, 1:1 + n] - X[:, 1:1 + n]))))
    fclip = m.spec.force_bound_n
    n_clipped = int(np.sum(np.abs(u_ff) > fclip))

    # one-step RK4 dynamic defect of the saved nominal (boundary-jump indicator)
    max_def = 0.0
    for k in range(0, N, 5):  # subsample for speed
        xn = m.rk4_step(X[k], float(u_ff[k]), dt)
        max_def = max(max_def, float(np.rad2deg(np.max(np.abs(xn[1:1 + n] - X[k + 1, 1:1 + n])))))

    # CLOSED-LOOP consistency: TVLQR tracking the nominal (the honest test --
    # pure open-loop feedforward on an unstable plant always diverges; even the
    # repo's committed n4 nominal drifts ~83deg open-loop).
    cl_termdeg = float("nan"); cl_maxdev = float("nan"); cl_peakF = float("nan")
    try:
        from cartpole_race.tvlqr import TVLQR, make_Q, make_R
        import scipy.linalg as sla
        A_up, B_up = m.linearize(m.x_equilibrium("up"), 0.0)
        Qd = make_Q(n); Rd = make_R()
        P = sla.solve_continuous_are(A_up, B_up, Qd, Rd)  # upright static Riccati
        tv = TVLQR(m, tg, X, forces, 25.0 * P, n_eval=300)  # track nominal (X, forces)
        def clpol(x, t):
            return float(tv.policy(x, t))
        _, xcl, ucl = m.rollout_zoh(x0, clpol, T, dt, m.spec.rk4_max_step_s)
        cl_termdeg = float(np.rad2deg(np.max(np.abs(((xcl[-1, 1:1 + n] + np.pi) % (2 * np.pi)) - np.pi))))
        cl_maxdev = float(np.rad2deg(np.max(np.abs(xcl[:, 1:1 + n] - X[:, 1:1 + n]))))
        cl_peakF = float(np.max(np.abs(ucl)))
    except Exception as e:
        log(f"[{label}] TVLQR closed-loop replay failed: {e}")

    rec = dict(n=n, T=T, a_max=a_max, M=len(ts) - 1,
               term_recon_deg=round(termdeg, 4), term_node_deg=round(node_termdeg, 4),
               termrate=round(termrate, 4),
               peak_accel=round(peak_a, 2), peak_force=round(peak_F, 2),
               replay_ol_termdeg=round(replay_termdeg, 3), replay_ol_maxdev_deg=round(dev, 2),
               replay_cl_termdeg=round(cl_termdeg, 3), replay_cl_maxdev_deg=round(cl_maxdev, 2),
               replay_cl_peakF=round(cl_peakF, 2),
               max_1step_defect_deg=round(max_def, 4),
               n_force_clipped=n_clipped, force_bound=fclip)
    log(f"[{label}] term(recon)={termdeg:.3f}deg term(node)={node_termdeg:.3f}deg rate={termrate:.3f} "
        f"| peak|a|={peak_a:.1f} m/s^2 peak|F|={peak_F:.1f} N "
        f"| OL_replay_term={replay_termdeg:.1f}deg | CL_replay_term={cl_termdeg:.2f}deg "
        f"CL_maxdev={cl_maxdev:.1f}deg CL_peakF={cl_peakF:.0f}N max_defect={max_def:.3f}deg "
        f"clip={n_clipped}/{len(u_ff)}")

    if save_path is not None:
        np.savez(save_path, states=X, forces=u_ff, t=tg, accel=a, T=T, n=n,
                 control_dt=dt, replay_states=x_log, replay_forces=u_log,
                 termdeg=termdeg, replay_termdeg=replay_termdeg,
                 peak_accel=peak_a, peak_force=peak_F, p=p, a_max=a_max)
        log(f"[{label}] saved nominal -> {save_path}")
    return rec


def run(nmax, save):
    results = []
    # (T, a_max, M, nsub) per n. a_max generous; M segments scale with difficulty.
    sched = {2: (3.0, 45.0, 16, 12), 3: (4.0, 55.0, 24, 12), 4: (5.0, 70.0, 40, 12)}
    prev = None  # (Z, ts, n, p)
    for n in range(2, nmax + 1):
        T, a_max, M, nsub = sched[n]
        NP = 2 * n
        m = make_model(n); fang, fforce = build_fns(m)
        ts = np.linspace(0, T, M + 1)
        if prev is None:
            Z0 = smooth_Z(n, ts, T); p0 = np.zeros(NP); p0[0] = 5.0
        else:
            pZ, pts, pn, pp = prev
            Z0, p0 = warm_Z(pZ, pts, pn, n, ts, pp, NP)
        ms = MultiShoot(m, fang, T, a_max, NP, M, nsub)
        t0 = time.time()
        # tighten the final (deliverable) n=4 stage so segment-boundary defects
        # shrink enough for TVLQR to track the nominal.
        res, Z, p = ms.solve(Z0, p0)  # maxfev=140, ftol=1e-6 (works for n=2,3)
        el = time.time() - t0
        save_path = "runs/r2/nom_n4_gluck.npz" if (save and n == nmax and n == 4) else None
        rec = evaluate(m, fang, fforce, Z, p, ts, T, a_max, NP,
                       f"n={n} MS T={T} a_max={a_max} M={M}", save_path=save_path)
        rec["nfev"] = int(res.nfev); rec["cost"] = float(res.cost); rec["solve_s"] = round(el, 1)
        results.append(rec)
        good = rec["term_recon_deg"] < 1.5 and rec["peak_accel"] < 200 and np.isfinite(rec["peak_force"])
        if good:
            prev = (Z, ts, n, p)
        else:
            log(f"[n={n}] stage weak (term={rec['term_recon_deg']:.2f}, "
                f"peak|a|={rec['peak_accel']:.1f}); a_max continuation")
            ok = False
            for am in (a_max * 1.5, a_max * 0.8, a_max * 2.0):
                ms2 = MultiShoot(m, fang, T, am, NP, M, nsub)
                res2, Z2, p2 = ms2.solve(Z0, p0)
                rec2 = evaluate(m, fang, fforce, Z2, p2, ts, T, am, NP,
                                f"n={n} MS a_max={am:.0f}",
                                save_path=save_path)
                rec2["nfev"] = int(res2.nfev); results.append(rec2)
                if rec2["term_recon_deg"] < 1.5 and rec2["peak_accel"] < 200:
                    prev = (Z2, ts, n, p2); ok = True; break
            if not ok and prev is None:
                prev = (Z, ts, n, p)
    return results


if __name__ == "__main__":
    nmax = 4; save = False
    for arg in sys.argv[1:]:
        if arg == "--save":
            save = True
        else:
            nmax = int(arg)
    LOG = open("runs/r2/gluck2_run.log", "w")
    results = run(nmax, save)
    log("\n=== SUMMARY ===")
    for r in results:
        log(json.dumps(r))
    LOG.close()
