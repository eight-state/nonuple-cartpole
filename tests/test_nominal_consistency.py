"""Committed checks on the shipped n=9 dense nominal's rigor facts.

The n=9 dense nominal (``results/nom_n9_dense1ms_wv6en4.npz``) is the 1 ms
DENSIFICATION of a 4 ms w_v=6e-4 collocation solve: within each 4 ms segment it
IS the simulator's own ZOH integration of the held node force (defect ~0), and
at 4 ms node boundaries it carries the parent solve's transcription seam.
The committed claims tested here (all MEASURED on this artifact):

  1. intra-segment ticks reproduce exactly under the simulator's ZOH step
     (defect < 1e-10; measured 0.0),
  2. node-boundary seams < 5e-5 state units (measured max 1.129e-5; the
     velocity-penalized nominal is smooth, so ~375x cleaner than n=8's 4.2e-3),
  3. the 4 ms parent solve's own RK4-4ms transcription defect < 5e-7
     (measured 8.248e-08, matching the solve log),
  4. peak feedforward force < 50 N (measured 41.35 N) — 3.6x margin to the
     150 N bound.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec

REPO = Path(__file__).resolve().parent.parent
DENSE = REPO / "results" / "nom_n9_dense1ms_wv6en4.npz"
PARENT = REPO / "results" / "nom_n9_4ms_wv6en4.npz"
N = 9
STRIDE = 4  # 1 ms ticks per 4 ms parent node


def _zoh_step(model, x, u, control_dt, rk4_max_step):
    n_sub = max(1, int(np.ceil(control_dt / rk4_max_step)))
    dt_sub = control_dt / n_sub
    xx = np.asarray(x, dtype=float).reshape(-1).copy()
    for _ in range(n_sub):
        xx = model.rk4_step(xx, u, dt_sub)
    return xx


def _model():
    spec = CartPoleSpec().with_n_links(N)
    return NLinkCartPole(spec), spec


def test_dense_nominal_intra_segment_exact() -> None:
    """Ticks NOT at 4 ms boundaries reproduce under the sim's ZOH step."""
    assert DENSE.exists(), f"missing nominal: {DENSE}"
    d = np.load(DENSE)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, spec = _model()
    assert (X.shape[1] // 2) - 1 == N
    max_defect = 0.0
    for k in range(len(U)):
        if k % STRIDE == 0:
            continue  # first tick of a segment steps across the parent seam
        xn = _zoh_step(model, X[k], float(U[k]), spec.control_dt_s,
                       spec.rk4_max_step_s)
        max_defect = max(max_defect, float(np.max(np.abs(xn - X[k + 1]))))
    assert max_defect < 1e-10, (
        f"intra-segment densification not exact: {max_defect:.3e}")


def test_dense_nominal_seams_bounded() -> None:
    """4 ms node-boundary seams stay below the committed 5e-5 bound."""
    d = np.load(DENSE)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, spec = _model()
    worst = 0.0
    for k in range(len(U)):
        if k % STRIDE != 0:
            continue  # seam-crossing ticks: first tick of each segment
        xn = _zoh_step(model, X[k], float(U[k]), spec.control_dt_s,
                       spec.rk4_max_step_s)
        worst = max(worst, float(np.max(np.abs(xn - X[k + 1]))))
    # n=9 measured max seam: 1.129e-5 (vs 4.233e-3 at n=8). The w_v=6e-4
    # nominal is smooth, so the RK4-4ms vs exact-ZOH mismatch is tiny; the
    # closed loop absorbs it (peak demand ~41 N, monodromy rho=0.0736).
    assert worst < 5e-5, f"node seam too large: {worst:.3e}"


def test_parent_solve_transcription_defect() -> None:
    """The 4 ms parent satisfies its own RK4-4ms transcription to < 5e-7.

    Sampled spot-check (every 7th node) for speed; the full-grid defect
    (8.248e-08, matching the solve log) is reported by reproduce_n9.py.
    """
    d = np.load(PARENT)
    X = np.asarray(d["x"], float)
    U = np.asarray(d["u"], float).reshape(-1)
    model, _ = _model()
    h = float(d["horizon"]) / len(U)
    worst = 0.0
    for k in range(0, len(U), 7):  # sampled (every 7th node) for speed
        x = X[k]
        u = float(U[k])
        k1 = model.f(x, u); k2 = model.f(x + 0.5 * h * np.asarray(k1).reshape(-1), u)
        k3 = model.f(x + 0.5 * h * np.asarray(k2).reshape(-1), u)
        k4 = model.f(x + h * np.asarray(k3).reshape(-1), u)
        xn = x + (h / 6.0) * (np.asarray(k1).reshape(-1)
                              + 2 * np.asarray(k2).reshape(-1)
                              + 2 * np.asarray(k3).reshape(-1)
                              + np.asarray(k4).reshape(-1))
        worst = max(worst, float(np.max(np.abs(xn - X[k + 1]))))
    assert worst < 5e-7, f"parent transcription defect: {worst:.3e}"


def test_peak_feedforward_force_margin() -> None:
    d = np.load(DENSE)
    U = np.asarray(d["u"], float)
    assert float(np.abs(U).max()) < 50.0
