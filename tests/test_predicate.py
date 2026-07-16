"""The public five-second claim measures elapsed intervals, not samples."""

import numpy as np

from cartpole_race.predicate import longest_continuous_hold_s
from cartpole_race.release import build_model


def test_five_seconds_requires_5001_one_millisecond_states() -> None:
    model = build_model()
    upright = model.x_equilibrium("up")
    short = np.repeat(upright[None, :], 5_000, axis=0)
    exact = np.repeat(upright[None, :], 5_001, axis=0)
    assert longest_continuous_hold_s(model, short, 0.001) == 4.999
    assert longest_continuous_hold_s(model, exact, 0.001) == 5.0
