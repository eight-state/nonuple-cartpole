# METHOD: what is new at n=9

Delta-doc. The full method (the impossibility verdict and its refutation, the
NCR analysis, the discrete-TVLQR fix, densification, and the gate
evidence-construction history) lives in the n=7 and n=8 releases' METHODs. This
document records the n=9 specifics: the **velocity penalty** that made the
swing-up trackable, and the **pre-roll gate** that replaced the n=8 per-IC
replan.

## 1. Seed: Glück MS continuation (one more rung)

`gluck_n9_from_n8.py` is the exact analog of the n7→n8 continuation: sample the
saved n=8 MS trajectory at the n=9 segment boundaries, copy link 8's angle/rate
onto the new link 9, extend the horizon, run the same MultiShoot solver. Result
`nom_n9_gluck.npz`: a genuine open-loop n=9 swing-up plan (whose open-loop
replay diverges, as every raw MS seed's does — that is what the polish and
feedback are for).

## 2. Polish: ONE-SHOT 4 ms collocation with a velocity penalty (`_n9_oneshot_wv.py`)

2500-node, 4 ms RK4 collocation NLP (IPOPT/MUMPS, single continuous solve),
warm-started from the seed, with a **running velocity penalty** `w_v * Σ vel²`
added to the objective (`w_v = 6e-4`). Converged: RK4-4ms transcription defect
**8.248e-08**, peak feedforward **41.35 N** of 150, terminal **0.0115°**.

**Why `w_v` — the one n=9-specific lesson.** The force-optimal (`w_v=0`) nominal
solves cleanly (defect 6.0e-12, terminal 0.0115° open-loop) but its closed-loop
track **diverges at t≈2.96 s** to a 128.98° handoff, pinning 150 N for 70% of
ticks. Root cause: the force-optimal swing-up is too violent — peak link rate
**46.9 rad/s** vs n=8's trackable 30.6 — past the 1 ms-ZOH TVLQR
linearization's validity in the saturated nonlinear sim. Giving it *unlimited*
force diverges worse (NaN at t=2.96 s), which is the decisive proof it is NOT a
force-authority wall. Penalizing Σvel² cuts the peak rate to **12.5 rad/s** and
restores trackability, at negligible force cost (41.4 N peak). Softening or
hardening the TVLQR gains (R-sweep ×1…×10000) does not fix the `w_v=0` nominal —
it is the trajectory, not the gains.

## 3. Densify + control

Densification onto the exact 1 ms sim grid: max node-boundary seam **1.129e-5**
(~50× *smaller* than n=8's 4.233e-3, because the `w_v` nominal is smooth).
Exact-ZOH discrete-time TVLQR along the dense nominal: **monodromy rho =
0.0736** (contractive; the continuous-Riccati interpolated-gain TVLQR is
closed-loop unstable along this nominal — the discrete design is the fix, as at
n=7/8).

## 4. Unperturbed closed-loop result (and the refuted "catch is the blocker")

Real saturated 1 ms sim from exact hanging: handoff **0.0115°**, swing peak
41.4 N, then static-LQR hold → **PASS**. `reproduce_n9.py` and
`scripts/verify_n9_solve.py` both reproduce it.

**F4, refuted — the catch is not the blocker.** An earlier verdict said the
static-LQR hold from the 0.0115° handoff "holds 3.6 s then drifts." That was a
**hold-window artifact**: the n=9 catch has a ~2.4 s settling transient (the
upright has 9 unstable modes, fastest growth ~35/s, τ≈28 ms), so a 6 s rollout
window shows only ~3.6 s of the 5 s hold. Extend the window and the SAME
static-LQR hold from the SAME handoff holds: 6 s→3.6 s, 8 s→5.6 s, 12 s→9.6 s.
It PASSES the 5 s predicate. The saturating transient excurses to ~14° (inside
the NCR) and recovers. This is why `reproduce_n9.py`, `demo_nonuple.py`, and the
gate all watch the hold over a 10 s window.

## 5. Pre-roll gate + why it works (the delta vs n=8)

The n=8 release absorbed σ=0.02 with a **per-IC swing-up replan** (composite;
~hours/IC, ~weeks of wall/seed). At n=9 that is intractable on this box:
exact-Hessian IPOPT and fatrop both blew past 585 s for ONE solve; the n=9
symbolic dynamics graph makes every NLP derivative eval brutally slow.

The pre-roll sidesteps it. **The σ=0.02 perturbation lives at the *hanging*
start, which is a *stable* equilibrium.** So instead of re-optimizing per IC,
run a fixed **LQR-about-down** from the perturbed IC to actively settle it back
to the nominal start `x_nom[0]` (adaptive: until link angle+rate residual <
1.5e-3, i.e. inside the swing-up track's robustness radius; ~3.5–8.5 s), THEN
run the verified fixed-nominal TVLQR track + static-LQR hold. One LQR gain,
computed once; ~1000× cheaper than a per-IC collocation solve.

**Why the pre-roll must be tight.** The fixed-nominal track is genuinely
marginal at n=9 — it diverges during the up-swing-through-horizontal (~t=2.3 s)
for even σ≈0.005 *raw* perturbations. Settling angle+rate below ~1.5e-3 (and
regulating the CART hard in the down-LQR, else it drifts ~0.3 m and returns
slowly, breaking the track) puts every IC inside the track's radius → 24/24.

Result: **24/24 on each of seeds 12345, 777, 2024** (72/72), ~4 min/seed on 4
workers, Wilson-95 lower bound 0.862. Config (down-LQR gains, tol, cap) was
tuned on seed 12345 then **validated unchanged on 777 & 2024**. Banked with
provenance in `results/gate_n9_preroll_seed{12345,777,2024}.json`.

## 6. Controller-independent recoverability (the keystone guard)

The static-LQR *catch basin* is thin (~0.001° in arbitrary directions), but that
is a controller property, not a wall — reading it as a wall would repeat the
n=7 keystone error. `scripts/_ncr_hard_bound.py` gives the n=9 upright's
controller-independent null-controllable region as **7.1°–22.3°** (pure angle),
and all 200/200 σ=0.02 gate draws pass the per-mode recoverability necessary
condition (worst ratio 0.35 < 1). The swing-up always delivers the handoff in
the same clean ~0.0115° direction, which is inside the basin, so the catch never
needs handoff-perturbation robustness.

## 7. Cross-n statement, updated

Five rungs (n=5..9) now support it: the binding constraints are transcription
fidelity, gain discretization, basin realization, and solver mode (controller
numerics, all), not single-input actuator authority. Peak feedforward force has
stayed a fraction of the 150 N actuator across the ladder (41.4 N at n=9) while
the spectrum stiffens and the unstable-mode count grows.
