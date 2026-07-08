"""Thin rollout helpers + the handoff mode machine (M1: TVLQR + STATIC only).

Wraps :meth:`NLinkCartPole.rollout_zoh` (the single rollout) and implements the
proposal's dwell/margin handoff machine. Modes:
``SWINGUP -> TVLQR_CATCH -> STATIC_LQR -> {SUCCESS, FAIL}`` — one-way. In M1
only ``TVLQR_CATCH`` and ``STATIC_LQR`` are active (no swing-up).

TVLQR catch -> static LQR switch (proposal), iff all hold:
    1. ``V_static(x) = e'P e <= 0.60 * rho_static``;
    2. ``|u_static| <= 0.70 * Fmax``;
    3. ``|x_cart| <= 0.50 * track_half_length``;
    4. dwell 500 ms continuously.
Then static LQR must hold 5 s to count as success. If static exits
``0.9 * rho_static``, fall back to the active TVLQR if its window still
applies; otherwise declare failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.funnels import in_success_set
from cartpole_race.lqr import StaticLQRPolicy, static_lqr, wrap_state_error
from cartpole_race.tvlqr import TVLQR

MODE_TVLQR_CATCH = "TVLQR_CATCH"
MODE_STATIC_LQR = "STATIC_LQR"
MODE_SUCCESS = "SUCCESS"
MODE_FAIL = "FAIL"

# Settling budget granted before the locked 5 s hold window must hold. The
# proposal predicate is "enters AND remains" in the hold set, so a boundary
# point is allowed a brief transient to settle before the continuous-hold clock
# starts. The hold requirement itself (>= hold_time_s continuously in-set) is
# unchanged; this only stops zero-settling-budget false negatives.
SETTLE_TIME_S = 1.0

# Proposal-locked TVLQR->static switch margins.
SWITCH_V_FRACTION = 0.60
SWITCH_U_FRACTION = 0.70
SWITCH_X_FRACTION = 0.50
SWITCH_DWELL_S = 0.50
STATIC_EXIT_FRACTION = 0.90


@dataclass
class HandoffResult:
    """Outcome of a handoff rollout."""

    success: bool
    mode_sequence: list[str]
    t_log: np.ndarray
    x_log: np.ndarray
    u_log: np.ndarray
    switch_time: float | None
    hold_time_achieved: float
    max_force: float
    min_track_margin: float
    max_force_demanded: float = 0.0
    saturated: bool = False
    failure_reason: str = ""
    mode_log: np.ndarray = field(default_factory=lambda: np.empty(0))


class HandoffPolicy:
    """Stateful TVLQR->static handoff policy (one ``rollout_zoh`` call).

    This is the proposal's dwell/margin mode machine packaged as a single
    ``(state, t) -> force`` callable so the entire handoff rollout goes through
    exactly ONE :meth:`NLinkCartPole.rollout_zoh` invocation (the single shared
    rollout). The machine updates its mode at the start of each control tick
    based on the state handed in by the rollout.

    Modes (M1): ``TVLQR_CATCH -> STATIC_LQR``. The one-way switch fires after
    the proposal's 500 ms dwell once the V/force/track margins all hold; static
    exit above ``0.9*rho_static`` falls back to TVLQR while its window applies.
    Success (5 s hold) and failure are evaluated post-rollout from the log.
    """

    def __init__(
        self,
        model: NLinkCartPole,
        tvlqr: TVLQR,
        P_static: np.ndarray,
        rho_static: float,
        catch_horizon: float,
        static_lqr_policy: StaticLQRPolicy,
    ) -> None:
        self.model = model
        self.n = model.n
        self.tvlqr = tvlqr
        self.P = P_static
        self.rho = rho_static
        self.catch_horizon = catch_horizon
        self.static = static_lqr_policy
        self.x_up = model.x_equilibrium("up")
        self.fbound = model.spec.force_bound_n
        self.track = model.spec.track_half_length_m
        self.dwell_ticks = int(round(SWITCH_DWELL_S / model.spec.control_dt_s))

        self.mode = MODE_TVLQR_CATCH
        self.dwell = 0
        self.switch_time: float | None = None
        self.mode_seq = [MODE_TVLQR_CATCH]

    def __call__(self, x: np.ndarray, t: float) -> float:
        e = wrap_state_error(x, self.x_up, self.n)
        v_static = float(e @ self.P @ e)

        if self.mode == MODE_TVLQR_CATCH:
            u = self.tvlqr.policy(x, min(t, self.catch_horizon))
            u_static = float(-(self.static.K @ e).item())
            cond = (
                v_static <= SWITCH_V_FRACTION * self.rho
                and abs(u_static) <= SWITCH_U_FRACTION * self.fbound
                and abs(x[0]) <= SWITCH_X_FRACTION * self.track
            )
            self.dwell = self.dwell + 1 if cond else 0
            if self.dwell >= self.dwell_ticks:
                self.mode = MODE_STATIC_LQR
                self.switch_time = t
                self.mode_seq.append(MODE_STATIC_LQR)
            return u

        # STATIC_LQR
        if v_static > STATIC_EXIT_FRACTION * self.rho and t < self.catch_horizon:
            self.mode = MODE_TVLQR_CATCH
            self.mode_seq.append(MODE_TVLQR_CATCH)
            self.dwell = 0
            return self.tvlqr.policy(x, min(t, self.catch_horizon))
        return self.static(x, t)


def simulate_handoff(
    model: NLinkCartPole,
    x0: np.ndarray,
    tvlqr: TVLQR,
    P_static: np.ndarray,
    rho_static: float,
    catch_horizon: float,
    hold_time_s: float = 5.0,
    static_lqr_policy: StaticLQRPolicy | None = None,
) -> HandoffResult:
    """Run TVLQR catch then static-LQR hold via ONE shared rollout.

    Uses :class:`HandoffPolicy` (the dwell/margin mode machine) inside a single
    :meth:`NLinkCartPole.rollout_zoh` call, then evaluates the locked success
    predicate (continuous 5 s hold) from the returned log.

    Args:
        model: Shared dynamics object.
        x0: Initial state.
        tvlqr: Built catch TVLQR.
        P_static: Static Riccati matrix.
        rho_static: Static funnel level.
        catch_horizon: TVLQR horizon ``tf``.
        hold_time_s: Required continuous static hold for success.
        static_lqr_policy: Optional prebuilt static policy.

    Returns:
        :class:`HandoffResult`.
    """
    spec = model.spec
    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s
    track = spec.track_half_length_m
    fbound = spec.force_bound_n

    if static_lqr_policy is None:
        K, _ = static_lqr(model)
        static_lqr_policy = StaticLQRPolicy(model, K)
    static_lqr_policy.P = P_static

    total_t = catch_horizon + hold_time_s + 1.0
    pol = HandoffPolicy(
        model, tvlqr, P_static, rho_static, catch_horizon, static_lqr_policy
    )
    # Capture the RAW demanded force (pre-saturation) per control tick. The
    # shared rollout (dynamics.rollout_zoh) calls the policy exactly once per
    # tick with the raw demand and clips it internally to +/- fbound, so this
    # wrapper records the identical raw sequence the release logs as u_raw_log
    # WITHOUT changing the rollout's 3-tuple contract.
    u_raw_log: list[float] = []

    def _recording_pol(xx: np.ndarray, tt: float) -> float:
        u_raw = float(pol(xx, tt))
        u_raw_log.append(u_raw)
        return u_raw

    t_log, x_log, u_log = model.rollout_zoh(
        x0, _recording_pol, total_t, control_dt, rk4_max
    )
    u_raw_arr = np.asarray(u_raw_log, dtype=float)

    max_force = float(np.max(np.abs(u_log))) if len(u_log) else 0.0
    # Raw demanded force (pre-clip) and whether the bound was hit on any tick.
    max_force_demanded = (
        float(np.max(np.abs(u_raw_arr))) if len(u_raw_arr) else 0.0
    )
    saturated = bool(
        len(u_raw_arr) and np.any(np.abs(u_raw_arr) > fbound + 1e-6)
    )
    min_margin = float(track - np.max(np.abs(x_log[:, 0])))
    track_ok = bool(np.all(np.abs(x_log[:, 0]) <= track))
    force_ok = bool(np.all(np.abs(u_log) <= fbound + 1e-6))

    # Continuous hold: longest in-success-set suffix.
    in_set = np.array([in_success_set(model, xx) for xx in x_log])
    tail_len = 0
    for j in range(len(in_set) - 1, -1, -1):
        if in_set[j]:
            tail_len += 1
        else:
            break
    # Elapsed time spanned by tail_len in-set samples is (tail_len - 1) ticks
    # (the gap count), not tail_len; guard against the empty tail.
    hold_achieved = max(0, tail_len - 1) * control_dt

    reason = ""
    if not track_ok:
        reason = "track_violation"
    elif not force_ok:
        reason = "force_violation"
    elif hold_achieved < hold_time_s - 1e-9:
        reason = "no_5s_hold"

    success = bool(track_ok and force_ok and hold_achieved >= hold_time_s - 1e-9)
    mode_seq = pol.mode_seq + ([MODE_SUCCESS] if success else [MODE_FAIL])

    return HandoffResult(
        success=success,
        mode_sequence=mode_seq,
        t_log=t_log,
        x_log=x_log,
        u_log=u_log,
        switch_time=pol.switch_time,
        hold_time_achieved=hold_achieved,
        max_force=max_force,
        min_track_margin=min_margin,
        max_force_demanded=max_force_demanded,
        saturated=saturated,
        failure_reason=reason,
    )


def static_hold_rollout(
    model: NLinkCartPole,
    x0: np.ndarray,
    policy,
    hold_time_s: float = 5.0,
    settle_time_s: float = SETTLE_TIME_S,
) -> tuple[bool, dict]:
    """Roll the static policy and test the locked hold predicate.

    The proposal's success predicate is "**enters and remains** in the upright
    hold set continuously for >= ``hold_time_s``". A point on (or near) the
    funnel boundary legitimately needs a brief settling transient before it is
    inside the 5 deg / 0.5 rad·s hold set, so the rollout must allow time to
    settle and then require the FINAL ``hold_time_s`` window to be continuously
    in-set. Rolling for exactly ``hold_time_s`` and demanding the whole window
    in-set (zero settling budget) spuriously rejects clean catches that settle
    in a few ms — the artifact that previously failed gate clause C4.

    Success iff: force/track respected over the WHOLE rollout AND the trajectory
    is continuously in the locked success set for the final ``hold_time_s``.

    Args:
        model: Shared dynamics object.
        x0: Initial state.
        policy: ``(state, t) -> force`` policy.
        hold_time_s: Required continuous in-set hold (the locked 5 s).
        settle_time_s: Additional settling budget before the hold window.

    Returns:
        ``(success, info)`` with ``info`` carrying max_force, min_track_margin,
        final_state, the achieved continuous hold, and per-bound flags.
    """
    spec = model.spec
    control_dt = spec.control_dt_s
    rk4_max = spec.rk4_max_step_s
    track = spec.track_half_length_m
    fbound = spec.force_bound_n

    total_t = hold_time_s + settle_time_s
    t_log, x_log, u_log = model.rollout_zoh(
        x0, policy, total_t, control_dt, rk4_max
    )
    max_force = float(np.max(np.abs(u_log))) if len(u_log) else 0.0
    min_margin = float(track - np.max(np.abs(x_log[:, 0])))
    track_ok = bool(np.all(np.abs(x_log[:, 0]) <= track))
    force_ok = bool(np.all(np.abs(u_log) <= fbound + 1e-6))

    # Continuous in-set tail (longest all-True suffix of the in-set mask).
    in_set = np.array([in_success_set(model, xx) for xx in x_log])
    tail_len = 0
    for j in range(len(in_set) - 1, -1, -1):
        if in_set[j]:
            tail_len += 1
        else:
            break
    # Elapsed time spanned by tail_len in-set samples is (tail_len - 1) ticks
    # (the gap count), not tail_len; guard against the empty tail.
    tail_time = max(0, tail_len - 1) * control_dt

    success = bool(track_ok and force_ok and tail_time >= hold_time_s - 1e-9)
    info = {
        "max_force": max_force,
        "min_track_margin": min_margin,
        "final_state": x_log[-1].tolist(),
        "tail_hold_s": tail_time,
        "track_ok": track_ok,
        "force_ok": force_ok,
    }
    return success, info
