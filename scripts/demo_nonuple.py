"""Clean, reproducible demo: closed-loop swing-up + balance of the n=9
(NONUPLE) inverted pendulum on a cart, in the REAL saturated simulator —
rendered to results/demo_nonuple.gif.

    uv run python scripts/demo_nonuple.py

One deterministic UNPERTURBED rollout: exact-ZOH discrete TVLQR along the
shipped dense nominal over [0, 10 s], handoff to the static LQR, then hold.
Unlike the n<=8 demos, the hold is watched over a 10 s window: the n=9 catch
has a ~2.4 s settling transient (the upright has 9 unstable modes, fastest
growth ~35/s), during which the saturating static-LQR catch excurses to ~14 deg
before recovering. Success is the SAME predicate the release gate uses: a
continuous >= 5 s in-success-set run (predicate v1). The GIF is asserted PASS
before saving (never a failed run). No re-solving, no ensemble, no pre-roll
(the pre-roll is only for perturbed ICs; see scripts/gate_n9_preroll.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.funnels import in_success_set
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from _dtvlqr import DiscreteTVLQR
import nominal as NOM

N_LINKS = 9
HOLD_S = 5.0
HOLD_WIN = 10.0  # watch the hold over a 10 s window (n=9 transient is ~2.4 s)
GIF_NAME = "demo_nonuple.gif"


def main() -> int:
    n = N_LINKS
    spec = CartPoleSpec().with_n_links(n)
    m = NLinkCartPole(spec)
    dt = spec.control_dt_s
    fb = spec.force_bound_n

    d = np.load(NOM.NOMINAL.path)
    X = d["x"]; U = d["u"]; T = float(d["horizon"])
    print(f"[demo n={n}] nominal {NOM.NOMINAL.file}: {T} s, "
          f"peak ff {np.abs(U).max():.1f} N")

    tv = DiscreteTVLQR(m, X, U, dt)
    K, P = static_lqr(m)
    sp_ = StaticLQRPolicy(m, K)
    sp_.P = P

    x0 = m.x_equilibrium("down")

    def policy(x, t):
        if t < T:
            return float(np.clip(tv.policy(x, t), -fb, fb))
        return sp_(x, t)

    total = T + HOLD_WIN + 1.0
    t_log, x_log, u_log = m.rollout_zoh(x0, policy, total, dt,
                                        spec.rk4_max_step_s)
    # assert predicate v1: longest continuous in-success-set run >= 5 s,
    # measured as elapsed time (best-1)*dt to match the gate scripts and
    # rollout.static_hold_rollout.
    in_set = np.array([in_success_set(m, xx) for xx in x_log])
    run = 0
    best = 0
    for v in in_set:
        run = run + 1 if v else 0
        best = max(best, run)
    ok = max(0, best - 1) * dt >= HOLD_S - 1e-9 and \
        np.max(np.abs(x_log[:, 0])) <= spec.track_half_length_m
    print(f"[demo n={n}] longest hold {max(0, best - 1) * dt:.1f} s, peak force "
          f"{np.abs(u_log).max():.1f} N -> {'PASS' if ok else 'FAIL'}")
    assert ok, "demo rollout failed the predicate; refusing to render"

    _save_gif(t_log, x_log, u_log, m, T)
    return 0


def _save_gif(t_log, x_log, u_log, model, horizon):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = model.n
    L = model.spec.link_lengths_m[0]
    fps = 25
    step = int(round(1.0 / (fps * 0.001)))  # sim ticks per frame
    frames = range(0, len(x_log), step)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-5.4, 5.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="#999", lw=1)
    title = ax.set_title("")
    cart, = ax.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = ax.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    ftxt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

    def pts(state):
        xs = [state[0]]
        ys = [0.0]
        for i in range(n):
            xs.append(xs[-1] + L * np.sin(state[1 + i]))
            ys.append(ys[-1] + L * np.cos(state[1 + i]))
        return xs, ys

    def update(fi):
        s = x_log[fi]
        xs, ys = pts(s)
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        t = t_log[fi]
        phase = "swing-up" if t < horizon else "balance"
        title.set_text(f"n={n} cart-pole — {phase}  t={t:5.2f} s")
        ui = min(fi, len(u_log) - 1)
        ftxt.set_text(f"force {u_log[ui]:+6.1f} N (|u|<=150)")
        return cart, chain, title, ftxt

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = REPO / "results" / GIF_NAME
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[demo] saved {out}")


if __name__ == "__main__":
    raise SystemExit(main())
