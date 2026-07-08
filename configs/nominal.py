"""Single source of truth for the n=9 nominal trajectory and its grid.

The shipped n=9 dense nominal is the 1 ms DENSIFICATION of a 4 ms collocation
solve (`nom_n9_4ms_wv6en4.npz`, RK4-4ms transcription defect 8.248e-08, peak
feedforward 41.35 N, terminal 0.0115 deg, from the Glueck MS continuation seed
`nom_n9_gluck.npz`): each node's constant force integrated through the
simulator's exact ZOH stepping (4x RK4 0.25 ms substeps per 1 ms tick; max
node-boundary seam 1.129e-05, ~50x smaller than n=8's because the
velocity-penalized nominal is smooth). Closed-loop validation runs the REAL
saturated plant at 1 ms with exact-ZOH discrete TVLQR (monodromy rho = 0.0736).

The velocity penalty (`w_v = 6e-4`, a running Sum vel^2 cost in the collocation
objective) is the one n=9-specific ingredient: the force-optimal (w_v=0) nominal
converges to the same 0.0115 deg terminal but swings up too violently (peak link
rate 46.9 rad/s vs 12.5 rad/s here) to track closed-loop in the saturated sim.
See docs/METHOD.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent
REPO = CONFIGS_DIR.parent
RESULTS = REPO / "results"


@dataclass(frozen=True)
class NominalSpec:
    file: str
    grid_dt_s: float
    n_nodes: int
    horizon_s: float
    is_native_1ms: bool
    label: str

    @property
    def path(self) -> Path:
        return RESULTS / self.file


NOMINAL = NominalSpec(
    file="nom_n9_dense1ms_wv6en4.npz",
    grid_dt_s=0.001,
    n_nodes=10000,
    horizon_s=10.0,
    is_native_1ms=True,
    label="densified 4 ms w_v=6e-4 collocation nominal (1 ms grid)",
)

NOMINAL_4MS = NominalSpec(
    file="nom_n9_4ms_wv6en4.npz",
    grid_dt_s=0.004,
    n_nodes=2500,
    horizon_s=10.0,
    is_native_1ms=False,
    label="4 ms w_v=6e-4 collocation parent solve",
)
