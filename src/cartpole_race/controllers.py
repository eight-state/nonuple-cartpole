"""Compose static + TVLQR catch policies behind a common ``policy(x, t)``.

Per the proposal 'Controllers and handoff'. In M1 only the TVLQR catch and the
static LQR are active (no swing-up). This module exposes:

- :class:`StaticController` — thin wrapper exposing ``policy(x, t)``.
- :class:`CatchThenHoldController` — TVLQR over ``[0, tf]`` then static LQR,
  with the proposal's TVLQR->static dwell/margin switch applied (the full
  dwell machine lives in :mod:`rollout`; this class gives a simple composed
  policy for funnel estimation and quick rollouts).
"""

from __future__ import annotations

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.lqr import StaticLQRPolicy, static_lqr
from cartpole_race.tvlqr import TVLQR, build_upright_tvlqr


class Controller:
    """Common controller interface: callable ``(state, t) -> force``."""

    def __call__(self, x: np.ndarray, t: float) -> float:  # pragma: no cover
        raise NotImplementedError

    # Alias so callers can pass either ``ctrl`` or ``ctrl.policy``.
    def policy(self, x: np.ndarray, t: float) -> float:
        return self(x, t)


class StaticController(Controller):
    """Static-LQR-only controller."""

    def __init__(self, model: NLinkCartPole) -> None:
        K, P = static_lqr(model)
        self.model = model
        self.lqr = StaticLQRPolicy(model, K)
        self.lqr.P = P
        self.K = K
        self.P = P

    def __call__(self, x: np.ndarray, t: float) -> float:
        return self.lqr(x, t)


class CatchThenHoldController(Controller):
    """TVLQR catch for ``t < tf`` then static LQR hold (saturated by sim).

    The composition is time-based for funnel estimation: during the catch
    horizon the TVLQR feedback is used; after ``tf`` the static policy takes
    over. The simulator boundary applies force saturation.
    """

    def __init__(
        self,
        model: NLinkCartPole,
        horizon: float,
        qf_scale: float = 25.0,
    ) -> None:
        self.model = model
        self.tvlqr: TVLQR = build_upright_tvlqr(model, horizon, qf_scale=qf_scale)
        self.static = StaticController(model)
        self.tf = float(horizon)

    def __call__(self, x: np.ndarray, t: float) -> float:
        if t < self.tf:
            return self.tvlqr.policy(x, t)
        return self.static(x, t)
