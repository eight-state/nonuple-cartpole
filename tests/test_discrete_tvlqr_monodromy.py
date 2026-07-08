"""Committed gate: the exact-ZOH discrete TVLQR closed loop CONTRACTS along
the shipped n=9 nominal (monodromy spectral radius < 1).

This is the controller fact behind the swing-up: the repo's continuous-Riccati
TVLQR with interpolated gains is closed-loop unstable along this nominal, while
the discrete-time design (per-tick exact ZOH discretization + backward discrete
Riccati, scripts/_dtvlqr.py) gives rho ~= 0.0736. See docs/METHOD.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from cartpole_race.dynamics import NLinkCartPole  # noqa: E402
from cartpole_race.env_spec import CartPoleSpec  # noqa: E402
from _dtvlqr import DiscreteTVLQR  # noqa: E402


def test_discrete_tvlqr_monodromy_contracts() -> None:
    d = np.load(REPO / "results" / "nom_n9_dense1ms_wv6en4.npz")
    spec = CartPoleSpec().with_n_links(9)
    m = NLinkCartPole(spec)
    tv = DiscreteTVLQR(m, np.asarray(d["x"], float),
                       np.asarray(d["u"], float), spec.control_dt_s)
    rho = tv.monodromy()
    assert rho < 1.0, f"closed loop does not contract: rho={rho:.4g}"
    assert rho < 0.5, f"rho unexpectedly large vs banked 0.0736: {rho:.4g}"
