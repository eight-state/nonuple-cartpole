"""Capture-basin mapper: spawn-safe multiprocessing over near-upright states.

Per the proposal M1 build list and 'Proof harness':
- multiprocessing **spawn-safe** across 20 workers, seeds via
  ``numpy.random.SeedSequence(master).spawn(n_jobs)``;
- each worker rebuilds the shared dynamics artifact ONCE at start;
- sample near-upright states, roll static LQR or TVLQR, record success;
- write ``runs/<ts>/*.parquet``; 2D + modal plots show anisotropy.

All integration goes through the single :meth:`NLinkCartPole.rollout_zoh`. The
worker functions are module-level (picklable under the spawn start method on
Windows).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import load_spec
from cartpole_race.funnels import (
    sample_shell,
    unstable_left_basis,
    unstable_modal_coord,
)
from cartpole_race.lqr import StaticLQRPolicy, static_lqr, wrap_state_error
from cartpole_race.rollout import simulate_handoff, static_hold_rollout
from cartpole_race.tvlqr import build_upright_tvlqr

N_WORKERS = 20


# ----------------------------------------------------------------------
# Sampling specifications
# ----------------------------------------------------------------------
@dataclass
class SampleBox:
    """Uniform near-upright sampling box (errors added to upright)."""

    theta: float  # +- rad per angle
    thetad: float  # +- rad/s per angle
    x: float  # +- m
    xdot: float  # +- m/s

    def draw(self, n_links: int, rng: np.random.Generator) -> np.ndarray:
        """Draw one error vector of length ``2*(n+1)``."""
        nx = 2 * (n_links + 1)
        e = np.zeros(nx)
        e[0] = rng.uniform(-self.x, self.x)
        e[1 : 1 + n_links] = rng.uniform(-self.theta, self.theta, n_links)
        e[1 + n_links] = rng.uniform(-self.xdot, self.xdot)
        e[2 + n_links :] = rng.uniform(-self.thetad, self.thetad, n_links)
        return e


@dataclass
class MapJob:
    """A capture-basin mapping job (passed to workers)."""

    config_path: str
    controller: str  # "static" or "tvlqr"
    n_samples: int
    mode: str  # "box" or "shell"
    box: SampleBox | None = None
    shell_rho: float = 0.0  # for mode == "shell" (level on P or S0)
    shell_eta: float = 1.0
    catch_horizon: float = 1.0
    hold_time_s: float = 5.0
    rho_static: float = 0.0  # for the handoff machine
    extra: dict = field(default_factory=dict)


# Module-level cache so each spawned worker builds the model once.
_WORKER_STATE: dict = {}


def _limit_native_threads() -> None:
    """Pin native math libraries to one thread in this (worker) process.

    With 20 spawn workers on a 20-core machine, each BLAS/OpenMP runtime would
    otherwise allocate a per-core thread arena, multiplying the committed
    memory per process and spiking the Windows commit charge past the lazily
    grown page file during the simultaneous-spawn burst (observed crash:
    ``OSError WinError 1455 — The paging file is too small``). One thread per
    worker also avoids CPU oversubscription. Safe to call repeatedly.
    """
    for v in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(v, "1")


def _worker_init(config_path: str, controller: str, catch_horizon: float) -> None:
    """Initialize per-worker shared dynamics + controllers (built once)."""
    _limit_native_threads()
    spec = load_spec(config_path)
    model = NLinkCartPole(spec)
    K, P = static_lqr(model)
    static_pol = StaticLQRPolicy(model, K)
    static_pol.P = P
    W_u = unstable_left_basis(model)
    from cartpole_race.fast_rollout import StaticHoldEvaluator

    state = {
        "model": model,
        "P": P,
        "static_pol": static_pol,
        "W_u": W_u,
        # mapaccum closed-loop evaluator (~8-15x faster; verified equivalent).
        "static_eval": StaticHoldEvaluator(model, K, hold_time_s=5.0),
    }
    if controller == "tvlqr":
        # Longer horizon (1.5 s) keeps S(t0) near the gentle infinite-horizon
        # solution so the single-input catch does not instantly saturate.
        tvlqr = build_upright_tvlqr(model, catch_horizon, qf_scale=5.0)
        state["tvlqr"] = tvlqr
    _WORKER_STATE.clear()
    _WORKER_STATE.update(state)


def _eval_one(args: tuple) -> dict:
    """Evaluate a single sampled IC in a worker. Returns a flat result dict."""
    (job_dict, seed_int) = args
    job = _job_from_dict(job_dict)
    model: NLinkCartPole = _WORKER_STATE["model"]
    P = _WORKER_STATE["P"]
    static_pol = _WORKER_STATE["static_pol"]
    W_u = _WORKER_STATE["W_u"]
    n = model.n
    x_up = model.x_equilibrium("up")
    rng = np.random.default_rng(seed_int)

    # Draw the initial error.
    if job.mode == "box":
        e = job.box.draw(n, rng)
    else:  # shell
        if job.controller == "tvlqr":
            S0 = _WORKER_STATE["tvlqr"].S_at(0.0)
            e = sample_shell(S0, job.shell_eta * job.shell_rho, 1, rng)[0]
        else:
            e = sample_shell(P, job.shell_eta * job.shell_rho, 1, rng)[0]
    x0 = x_up + e

    z = unstable_modal_coord(model, x0, W_u)

    if job.controller == "static":
        # mapaccum evaluator (verified equivalent to static_hold_rollout).
        success, info = _WORKER_STATE["static_eval"].evaluate(x0)
        mode_seq = "STATIC_LQR"
        reason = "" if success else "static_no_hold"
        max_force = info["max_force"]
        min_margin = info["min_track_margin"]
        final_state = info["final_state"]
    else:  # tvlqr handoff
        res = simulate_handoff(
            model,
            x0,
            _WORKER_STATE["tvlqr"],
            P,
            job.rho_static,
            job.catch_horizon,
            hold_time_s=job.hold_time_s,
            static_lqr_policy=static_pol,
        )
        success = res.success
        mode_seq = ">".join(res.mode_sequence)
        reason = res.failure_reason
        max_force = res.max_force
        min_margin = res.min_track_margin
        final_state = res.x_log[-1].tolist()

    rec = {
        "seed": int(seed_int),
        "n_links": n,
        "controller": job.controller,
        "success": bool(success),
        "x_cart": float(x0[0]),
        "xdot": float(x0[1 + n]),
        "mode_sequence": mode_seq,
        "max_force": float(max_force),
        "min_track_margin": float(min_margin),
        "failure_reason": reason,
        "v_static0": float(
            wrap_state_error(x0, x_up, n) @ P @ wrap_state_error(x0, x_up, n)
        ),
    }
    # Per-angle physical coords.
    for i in range(n):
        rec[f"theta{i}"] = float(e[1 + i])
        rec[f"thetad{i}"] = float(e[2 + n + i])
    # Modal coords.
    for i in range(len(z)):
        rec[f"z{i}"] = float(z[i])
    # Final state stored compactly.
    rec["final_x_cart"] = float(final_state[0])
    return rec


def _job_to_dict(job: MapJob) -> dict:
    d = dict(job.__dict__)
    if job.box is not None:
        d["box"] = job.box.__dict__
    return d


def _job_from_dict(d: dict) -> MapJob:
    d = dict(d)
    if d.get("box") is not None:
        d["box"] = SampleBox(**d["box"])
    return MapJob(**d)


def run_mapper(
    job: MapJob,
    n_workers: int = N_WORKERS,
    master_seed: int = 12345,
    out_root: str | Path = "runs",
    tag: str = "",
) -> tuple[pd.DataFrame, Path]:
    """Run the capture-basin mapper across ``n_workers`` spawn-safe processes.

    Args:
        job: Mapping job specification.
        n_workers: Worker process count (20 per the gate).
        master_seed: Master seed for ``SeedSequence`` spawning.
        out_root: Root directory for ``runs/<ts>/``.
        tag: Optional filename tag.

    Returns:
        ``(df, parquet_path)`` results and where they were written.
    """
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(job.n_samples)
    seed_ints = [int(s.generate_state(1)[0]) for s in child_seeds]

    job_dict = _job_to_dict(job)
    args = [(job_dict, s) for s in seed_ints]

    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(job.config_path, job.controller, job.catch_horizon),
        maxtasksperchild=64,
    ) as pool:
        records = pool.map(_eval_one, args, chunksize=8)
    wall = time.time() - t0

    df = pd.DataFrame.from_records(records)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{job.controller}_{job.mode}"
    if tag:
        name += f"_{tag}"
    parquet_path = out_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)

    # Stash wall-clock + summary alongside.
    summary = {
        "wall_s": wall,
        "n_samples": job.n_samples,
        "n_workers": n_workers,
        "success_fraction": float(df["success"].mean()),
        "controller": job.controller,
        "mode": job.mode,
    }
    (out_dir / f"{name}_summary.json").write_text(
        pd.Series(summary).to_json(indent=2), encoding="utf-8"
    )
    return df, parquet_path


__all__ = [
    "MapJob",
    "SampleBox",
    "run_mapper",
    "N_WORKERS",
]
