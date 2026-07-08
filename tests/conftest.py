"""Shared test fixtures for the dynamics hard-gate suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import CartPoleSpec, load_spec

# Parametrized dynamics gates at n = 1, 2, 3, 9.
N_LINKS_SWEEP = [1, 2, 3, 9]

# Resolve the config relative to the repo root so tests run from any CWD.
_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "env-base.yaml"
_BASE_SPEC = load_spec(_CONFIG)


def spec_for(n: int) -> CartPoleSpec:
    """Race spec resized to ``n`` links."""
    return _BASE_SPEC.with_n_links(n)


@pytest.fixture(params=N_LINKS_SWEEP, ids=lambda n: f"n{n}")
def n_links(request) -> int:
    """Parametrized link count across the hard-gate sweep."""
    return request.param


@pytest.fixture
def model(n_links: int) -> NLinkCartPole:
    """An :class:`NLinkCartPole` for the swept link count."""
    return NLinkCartPole(spec_for(n_links))
