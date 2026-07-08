"""ONE-COMMAND reproduction of the headline n=9 (nonuple) cart-pole result.

    uv run python reproduce_n9.py            # fast: unperturbed pass + rho
    uv run python reproduce_n9.py --gate     # full 24-IC preroll gate, 3 seeds

Fast mode (~1 min):
  1. Loads the shipped n=9 dense nominal (configs/nominal.py).
  2. Verifies the nominal's grid facts (10000 ticks, 10 s) and reports its
     4 ms parent's RK4-4ms transcription defect (expected ~8.2e-8).
  3. Builds exact-ZOH DISCRETE-time TVLQR along the dense nominal and reports
     the closed-loop monodromy spectral radius rho (expected ~0.0736 < 1).
  4. Runs the UNPERTURBED closed loop in the real saturated simulator
     (rollout_zoh, hard 150 N clip, 1 ms ZOH, RK4 substeps) from exact
     hanging, hands off near upright, then the static-LQR hold, watched over a
     10 s window (the n=9 catch has a ~2.4 s settling transient). Success is
     predicate v1: a continuous >= 5 s in-success-set run.
     Expected: PASS, swing peak ~41 N, handoff ~0.0115 deg.

Gate mode (--gate): the PRE-ROLL perturbed-IC gate at sigma=0.02 (identical
  perturbation model, seeds, simulator, and predicate as the n=5..8 releases):
  per-IC LQR-about-down pre-roll (settles the sigma=0.02 hanging perturbation
  back to the nominal start) -> discrete TVLQR track the swing-up nominal ->
  static-LQR hold. NO per-IC NLP (~1000x cheaper than the n=8 composite replan).
  Banked result: 24/24 on EACH of seeds 12345, 777, 2024 (72/72), ~4 min/seed
  on 4 workers. Banked JSONs are in results/gate_n9_preroll_seed*.json.

Determinism: each rollout is fixed-step RK4 + ZOH and deterministic.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

import numpy as np  # noqa: E402

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from cartpole_race.funnels import in_success_set  # noqa: E402
from cartpole_race.lqr import StaticLQRPolicy, static_lqr  # noqa: E402
from configs.nominal import NOMINAL, NOMINAL_4MS  # noqa: E402
from _dtvlqr import DiscreteTVLQR  # noqa: E402

N = 9
HOLD_S = 5.0
HOLD_WIN = 10.0


def _parent_defect(m) -> float:
    d = np.load(NOMINAL_4MS.path)
    X = np.asarray(d["x"], float); U = np.asarray(d["u"], float).reshape(-1)
    h = float(d["horizon"]) / len(U)
    worst = 0.0
    for k in range(len(U)):
        x = X[k]; u = float(U[k])
        k1 = m.f(x, u); k2 = m.f(x + 0.5 * h * np.asarray(k1).reshape(-1), u)
        k3 = m.f(x + 0.5 * h * np.asarray(k2).reshape(-1), u)
        k4 = m.f(x + h * np.asarray(k3).reshape(-1), u)
        xn = x + (h / 6.0) * (np.asarray(k1).reshape(-1)
                              + 2 * np.asarray(k2).reshape(-1)
                              + 2 * np.asarray(k3).reshape(-1)
                              + np.asarray(k4).reshape(-1))
        worst = max(worst, float(np.max(np.abs(xn - X[k + 1]))))
    return worst


def main() -> int:
    spec = CartPoleSpec().with_n_links(N)
    m = NLinkCartPole(spec)
    dt = spec.control_dt_s
    d = np.load(NOMINAL.path)
    X = d["x"]; U = d["u"]; T = float(d["horizon"])
    assert len(U) == NOMINAL.n_nodes and abs(T - NOMINAL.horizon_s) < 1e-9
    print(f"[1] nominal: {NOMINAL.file} ({NOMINAL.label}), "
          f"{len(U)} ticks, {T} s, peak ff {np.abs(U).max():.1f} N")

    defect = _parent_defect(m)
    print(f"    4 ms parent RK4-4ms transcription defect = {defect:.3e} "
          f"(expected ~8.2e-8)")

    print("[2] building exact-ZOH discrete TVLQR along the nominal ...")
    tv = DiscreteTVLQR(m, X, U, dt)
    rho = tv.monodromy()
    print(f"    closed-loop monodromy rho = {rho:.4g}  (expected ~0.0736 < 1)")

    print("[3] UNPERTURBED closed loop in the real saturated sim ...")
    x0 = m.x_equilibrium("down")
    K, P = static_lqr(m)
    sp_ = StaticLQRPolicy(m, K); sp_.P = P
    fb = spec.force_bound_n

    def pol(x, t):
        if t < T:
            return float(np.clip(tv.policy(x, t), -fb, fb))
        return sp_(x, t)

    total = T + HOLD_WIN + 1.0
    t1, x1, u1 = m.rollout_zoh(x0, pol, total, dt, spec.rk4_max_step_s)
    xup = m.x_equilibrium("up")
    # handoff deviation at t = T (end of the swing-up track)
    kT = int(round(T / dt))
    xh = x1[min(kT, len(x1) - 1)]
    hdev = np.rad2deg(np.max(np.abs(((xh[1:1 + N] - xup[1:1 + N] + np.pi)
                                     % (2 * np.pi)) - np.pi)))
    in_set = np.array([in_success_set(m, xx) for xx in x1])
    run = best = 0
    for v in in_set:
        run = run + 1 if v else 0
        best = max(best, run)
    hold_s = max(0, best - 1) * dt
    ok = hold_s >= HOLD_S - 1e-9 and \
        np.max(np.abs(x1[:, 0])) <= spec.track_half_length_m
    print(f"    swing-up handoff dev {hdev:.4f} deg, peak force "
          f"{np.abs(u1).max():.1f} N")
    print(f"    hold (predicate v1, 10 s window): longest continuous "
          f"{hold_s:.1f} s -> {'PASS' if ok else 'FAIL'}")
    if not (ok and rho < 1.0):
        print("REPRODUCTION FAILED")
        return 1
    print("\n*** n=9 SWING-UP + BALANCE: UNPERTURBED CLOSED-LOOP PASS ***")

    if "--gate" in sys.argv:
        import subprocess
        workers = str(max(1, (os.cpu_count() or 2) - 1))
        rc = 0
        for seed in ("12345", "777", "2024"):
            print(f"\n[4] preroll gate (24 ICs, sigma=0.02, seed {seed}) — "
                  f"no per-IC NLP, ~4 min/seed on 4 workers ...")
            r = subprocess.run([sys.executable,
                                str(REPO / "scripts" / "gate_n9_preroll.py"),
                                "24", seed, "9.0", workers], cwd=str(REPO))
            rc = rc or r.returncode
        return rc
    print("\n(run with --gate to regenerate the full 24-IC preroll gate on all "
          "three seeds; banked result is 24/24 each — see README and "
          "results/gate_n9_preroll_seed*.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
