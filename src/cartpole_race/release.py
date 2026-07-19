"""The supported N9 replay and immutable-evidence audit."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cartpole_race.discrete_tvlqr import DiscreteTVLQR
from cartpole_race.dynamics import NLinkCartPole
from cartpole_race.env_spec import load_spec
from cartpole_race.evidence import AUTHORITY_SHA256
from cartpole_race.lqr import static_lqr, wrap_state_error
from cartpole_race.predicate import longest_continuous_hold_s

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
WORKING = REPO / ".working"
CONFIG = REPO / "configs" / "env-base.yaml"
DENSE = RESULTS / "nom_n9_dense1ms_wv6en4.npz"
PARENT = RESULTS / "nom_n9_4ms_wv6en4.npz"
N_LINKS = 9
HOLD_REQUIRED_S = 5.0
HOLD_WINDOW_S = 10.0

GATE_FILES = (
    RESULTS / "gate_n9_preroll_seed12345.json",
    RESULTS / "gate_n9_preroll_seed777.json",
    RESULTS / "gate_n9_preroll_seed2024.json",
)


@dataclass(frozen=True)
class ReleaseStack:
    """Frozen reference plus feedback gains rebuilt from it locally."""

    model: NLinkCartPole
    nominal_states: np.ndarray
    nominal_controls: np.ndarray
    horizon_s: float
    tracker: DiscreteTVLQR
    static_gain: np.ndarray


@dataclass(frozen=True)
class LiveRun:
    """One fresh hanging-start simulation and its derived metrics."""

    times: np.ndarray
    states: np.ndarray
    applied_controls: np.ndarray
    raw_controls: np.ndarray
    metrics: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_authority_bytes() -> list[dict[str, str]]:
    """Rehash every frozen source and evidence input from raw file bytes."""
    records = []
    for relative_path, expected in AUTHORITY_SHA256.items():
        path = REPO / relative_path
        _require(path.is_file(), f"missing authority: {relative_path}")
        actual = file_sha256(path)
        _require(actual == expected, f"authority bytes changed: {relative_path}")
        records.append({"file": relative_path, "sha256": actual})
    return records


def build_model() -> NLinkCartPole:
    spec = load_spec(CONFIG)
    _require(spec.n_links == N_LINKS, "plant link count differs from N9")
    _require(math.isclose(spec.control_dt_s, 0.001), "control period differs from 1 ms")
    _require(math.isclose(spec.rk4_max_step_s, 0.00025), "RK4 step differs from 0.25 ms")
    return NLinkCartPole(spec)


def _load_npz(path: Path, expected_ticks: int) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as archive:
        states = np.asarray(archive["x"], dtype=float)
        controls = np.asarray(archive["u"], dtype=float).reshape(-1)
        horizon_s = float(np.asarray(archive["horizon"]).item())
        n_links = int(np.asarray(archive["n"]).item())
        force_bound_n = float(np.asarray(archive["force"]).item())
    _require(states.shape == (expected_ticks + 1, 20), f"state shape: {path.name}")
    _require(controls.shape == (expected_ticks,), f"control shape: {path.name}")
    _require(n_links == N_LINKS, f"link count: {path.name}")
    _require(force_bound_n == 150.0, f"force metadata: {path.name}")
    _require(np.all(np.isfinite(states)) and np.all(np.isfinite(controls)), path.name)
    return states, controls, horizon_s


def build_release_stack() -> ReleaseStack:
    """Load the fixed dense reference and rebuild both feedback controllers."""
    audit_authority_bytes()
    model = build_model()
    states, controls, horizon_s = _load_npz(DENSE, 10_000)
    _require(horizon_s == 10.0, "dense nominal horizon differs from 10 s")
    tracker = DiscreteTVLQR(model, states, controls, model.spec.control_dt_s)
    static_gain, _ = static_lqr(model)
    return ReleaseStack(model, states, controls, horizon_s, tracker, static_gain)


def run_live(stack: ReleaseStack | None = None) -> LiveRun:
    """Integrate one fresh hanging-start closed loop in the saturated plant."""
    stack = build_release_stack() if stack is None else stack
    model = stack.model
    upright = model.x_equilibrium("up")
    raw_controls: list[float] = []

    def policy(state: np.ndarray, time_s: float) -> float:
        if time_s < stack.horizon_s:
            raw = stack.tracker.policy(state, time_s)
        else:
            raw = -(stack.static_gain @ wrap_state_error(state, upright, model.n)).item()
        raw_controls.append(raw)
        return raw

    times, states, applied = model.rollout_zoh(
        model.x_equilibrium("down"),
        policy,
        stack.horizon_s + HOLD_WINDOW_S + 1.0,
        model.spec.control_dt_s,
        model.spec.rk4_max_step_s,
    )
    raw = np.asarray(raw_controls, dtype=float)
    handoff_tick = int(round(stack.horizon_s / model.spec.control_dt_s))
    handoff_error = wrap_state_error(states[handoff_tick], upright, model.n)
    handoff_deg = float(np.rad2deg(np.max(np.abs(handoff_error[1 : 1 + model.n]))))
    hold_s = longest_continuous_hold_s(model, states, model.spec.control_dt_s)
    track_peak_m = float(np.max(np.abs(states[:, 0])))
    clip_ticks = int(np.count_nonzero(np.abs(raw) > model.spec.force_bound_n + 1e-9))
    passed = bool(
        np.all(np.isfinite(states))
        and hold_s >= HOLD_REQUIRED_S - 1e-9
        and track_peak_m <= model.spec.track_half_length_m
        and np.max(np.abs(applied)) <= model.spec.force_bound_n + 1e-9
    )
    metrics = {
        "loaded_artifacts": {
            "plant_yaml": CONFIG.relative_to(REPO).as_posix(),
            "dense_nominal": DENSE.relative_to(REPO).as_posix(),
            "dense_nominal_sha256": AUTHORITY_SHA256[DENSE.relative_to(REPO).as_posix()],
            "saved_nominal_state_trace_loaded": True,
            "saved_nominal_role": "controller reference only; never rendered",
            "saved_controller": "none",
            "saved_rollout_states_rendered": False,
        },
        "recomputed": {
            "discrete_tvlqr": "exact-ZOH linearizations and backward Riccati gains",
            "static_lqr": "upright continuous Riccati gain",
            "states": "fresh rollout_zoh states from exact hanging",
            "controls": "fresh raw demands and simulator-applied clipped forces",
        },
        "closed_loop": {
            "monodromy_rho": stack.tracker.monodromy(),
            "handoff_max_angle_error_deg": handoff_deg,
            "longest_sampled_hold_s": hold_s,
            "track_peak_abs_m": track_peak_m,
            "applied_peak_force_n": float(np.max(np.abs(applied))),
            "raw_peak_force_n": float(np.max(np.abs(raw))),
            "clip_ticks": clip_ticks,
            "passed": passed,
        },
    }
    if not passed:
        raise RuntimeError(f"fresh N9 closed loop failed: {metrics['closed_loop']}")
    return LiveRun(times, states, applied, raw, metrics)


def _dense_defects(stack: ReleaseStack) -> tuple[float, float]:
    intra_segment = seam = 0.0
    for tick, (state, control, next_state) in enumerate(
        zip(
            stack.nominal_states[:-1],
            stack.nominal_controls,
            stack.nominal_states[1:],
            strict=True,
        )
    ):
        stepped = state.copy()
        for _ in range(4):
            stepped = stack.model.rk4_step(
                stepped, float(control), stack.model.spec.rk4_max_step_s
            )
        defect = float(np.max(np.abs(stepped - next_state)))
        if tick % 4 == 0:
            seam = max(seam, defect)
        else:
            intra_segment = max(intra_segment, defect)
    return intra_segment, seam


def audit_nominals(stack: ReleaseStack | None = None) -> dict[str, Any]:
    """Recompute consistency bounds for both frozen nominal artifacts."""
    stack = build_release_stack() if stack is None else stack
    intra_segment, seam = _dense_defects(stack)
    parent_states, parent_controls, parent_horizon_s = _load_npz(PARENT, 2_500)
    step_s = parent_horizon_s / len(parent_controls)
    parent_defect = 0.0
    for state, control, next_state in zip(
        parent_states[:-1], parent_controls, parent_states[1:], strict=True
    ):
        stepped = stack.model.rk4_step(state, float(control), step_s)
        parent_defect = max(parent_defect, float(np.max(np.abs(stepped - next_state))))
    _require(intra_segment < 1e-10, "dense intra-segment defect exceeds bound")
    _require(seam < 5e-5, "dense 4 ms seam exceeds bound")
    _require(parent_defect < 5e-7, "parent nominal defect exceeds bound")
    return {
        "dense_file": DENSE.relative_to(REPO).as_posix(),
        "dense_sha256": AUTHORITY_SHA256[DENSE.relative_to(REPO).as_posix()],
        "parent_file": PARENT.relative_to(REPO).as_posix(),
        "parent_sha256": AUTHORITY_SHA256[PARENT.relative_to(REPO).as_posix()],
        "dense_ticks": len(stack.nominal_controls),
        "horizon_s": stack.horizon_s,
        "peak_feedforward_n": float(np.max(np.abs(stack.nominal_controls))),
        "dense_intra_segment_defect": intra_segment,
        "dense_4ms_seam": seam,
        "parent_rk4_4ms_defect": parent_defect,
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float]:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
    ) / denominator
    return [round(center - half_width, 4), round(min(1.0, center + half_width), 4)]


def audit_banked_gates() -> dict[str, Any]:
    """Rehash and rederive historical rows without rerunning perturbations."""
    records = []
    for path in GATE_FILES:
        relative = path.relative_to(REPO).as_posix()
        _require(file_sha256(path) == AUTHORITY_SHA256[relative], f"gate hash: {path.name}")
        record = json.loads(path.read_text(encoding="utf-8"))
        rows = record.get("results", [])
        _require(len(rows) == record.get("n_ic") == 24, f"gate rows: {path.name}")
        _require([row.get("tag") for row in rows] == list(range(24)), f"tags: {path.name}")
        for row in rows:
            _require(row.get("success") is True, f"failed historical row: {path.name}")
            _require(row.get("track_ok") is True, f"track row: {path.name}")
            _require(row.get("fail") is None, f"failure label: {path.name}")
            _require(float(row.get("hold_s", 0.0)) >= HOLD_REQUIRED_S, path.name)
            _require(float(row.get("peakF", math.inf)) <= 150.0, path.name)
        successes = sum(row["success"] for row in rows)
        _require(record.get("n_success") == successes == 24, f"success count: {path.name}")
        interval = wilson_interval(successes, len(rows))
        _require(record.get("wilson95") == interval, f"Wilson interval: {path.name}")
        records.append(
            {
                "file": relative,
                "sha256": AUTHORITY_SHA256[relative],
                "seed": record.get("seed"),
                "successes": successes,
                "trials": len(rows),
                "wilson95": interval,
            }
        )
    _require({record["seed"] for record in records} == {12345, 777, 2024}, "seed set")
    return {
        "scope": "historical banked rows audited; perturbations not rerun",
        "files": records,
        "total_successes": sum(record["successes"] for record in records),
        "total_trials": sum(record["trials"] for record in records),
    }


def _working_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(WORKING.resolve())
    except ValueError as error:
        raise ValueError("generated output must stay below .working") from error
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = _working_path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_gif(run: LiveRun, path: Path, horizon_s: float) -> Path:
    """Render only the supplied fresh states to an ignored GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    path = _working_path(path)
    model = build_model()
    fps = 25
    frame_step = int(round(1.0 / (fps * model.spec.control_dt_s)))
    frames = range(0, len(run.states), frame_step)
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=80)
    axis.set_xlim(-6.2, 6.2)
    axis.set_ylim(-5.4, 5.6)
    axis.set_aspect("equal")
    axis.axhline(0, color="#999", lw=1)
    title = axis.set_title("")
    cart, = axis.plot([], [], "s", ms=14, color="#1f4e9c")
    chain, = axis.plot([], [], "-o", lw=2, ms=4, color="#c1452b")
    force_text = axis.text(0.02, 0.95, "", transform=axis.transAxes, fontsize=9)

    def points(state: np.ndarray) -> tuple[list[float], list[float]]:
        xs = [float(state[0])]
        ys = [0.0]
        for index in range(model.n):
            xs.append(xs[-1] + model.spec.link_lengths_m[index] * np.sin(state[1 + index]))
            ys.append(ys[-1] + model.spec.link_lengths_m[index] * np.cos(state[1 + index]))
        return xs, ys

    def update(frame_index: int):
        xs, ys = points(run.states[frame_index])
        cart.set_data([xs[0]], [0.0])
        chain.set_data(xs, ys)
        elapsed_s = run.times[frame_index]
        phase = "swing-up" if elapsed_s < horizon_s else "balance"
        title.set_text(f"N9 cart-pole: {phase}, t={elapsed_s:5.2f} s")
        control_index = min(frame_index, len(run.applied_controls) - 1)
        force_text.set_text(
            f"applied force {run.applied_controls[control_index]:+6.1f} N (|u| <= 150)"
        )
        return cart, chain, title, force_text

    animation = FuncAnimation(figure, update, frames=frames, blit=False)
    animation.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(figure)
    return path


def _report_live(run: LiveRun) -> None:
    closed_loop = run.metrics["closed_loop"]
    print(
        f"[live] handoff {closed_loop['handoff_max_angle_error_deg']:.7f} deg; "
        f"hold {closed_loop['longest_sampled_hold_s']:.3f} s; "
        f"applied/raw peak {closed_loop['applied_peak_force_n']:.7f}/"
        f"{closed_loop['raw_peak_force_n']:.7f} N; clips {closed_loop['clip_ticks']}; "
        f"rho {closed_loop['monodromy_rho']:.7g} -> PASS",
        flush=True,
    )


def demo_main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    stack = build_release_stack()
    run = run_live(stack)
    metrics_path = _write_json(WORKING / "n9" / "live-metrics.json", run.metrics)
    gif_path = render_gif(run, WORKING / "n9" / "demo.gif", stack.horizon_s)
    _report_live(run)
    print(f"[render] {gif_path.relative_to(REPO).as_posix()}", flush=True)
    print(f"[metrics] {metrics_path.relative_to(REPO).as_posix()}", flush=True)
    return 0


def verify_main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    stack = build_release_stack()
    sources = audit_authority_bytes()
    nominals = audit_nominals(stack)
    gates = audit_banked_gates()
    run = run_live(stack)
    report = {
        "authority_bytes": sources,
        "nominal_artifacts": nominals,
        "banked_gate_evidence": gates,
        "fresh_live": run.metrics,
    }
    report_path = _write_json(WORKING / "n9-verify" / "verification.json", report)
    print(f"[authority] {len(sources)} frozen inputs match", flush=True)
    print(
        f"[gate] {gates['total_successes']}/{gates['total_trials']} historical rows; "
        "perturbations not rerun",
        flush=True,
    )
    print(
        f"[nominal] parent/dense defects {nominals['parent_rk4_4ms_defect']:.3e}/"
        f"{nominals['dense_4ms_seam']:.3e}",
        flush=True,
    )
    _report_live(run)
    print(f"[report] {report_path.relative_to(REPO).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(demo_main())
