"""Focused checks for the frozen nominal and exact-ZOH controller."""

import numpy as np

from cartpole_race.release import audit_nominals, build_release_stack


def test_nominal_consistency_and_discrete_tvlqr_contraction() -> None:
    stack = build_release_stack()
    nominal = audit_nominals(stack)
    rho = stack.tracker.monodromy()
    assert nominal["dense_ticks"] == 10_000
    assert nominal["parent_rk4_4ms_defect"] < 5e-7
    assert nominal["dense_4ms_seam"] < 5e-5
    assert np.isfinite(rho)
    assert 0.0 <= rho < 0.2
