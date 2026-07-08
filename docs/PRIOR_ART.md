# Prior art: multi-link inverted pendulums (n=9)

This is the accounting behind the claim that this repo is, to our knowledge,
the **first public n=9 cart-pole swing-up-and-balance artifact by any method**
(unperturbed closed-loop leg + a banked perturbed-IC pre-roll gate, 24/24 on
three seeds), and certainly the first open-source, code-reproducible one. The
full prior-art tables for n=5 and n=6 (Glück 2013 hardware triple, Lam &
Davison 2006 base-torque stabilization, the Ozana quintuple simulation video,
Kotelovych 2024 Isaac Sim n=5 balance, yacine's 2026 public RL n=6 post) live
in the sibling releases:

- n=5: https://github.com/eight-state/quintuple-cartpole
- n=6: https://github.com/eight-state/sextuple-cartpole (docs/PRIOR_ART.md)

The table applies with one more row; what changes at n=9:

| Work | System | Links | Task | Why distinct from this repo |
|---|---|---|---|---|
| Oh et al. 2025 (RL) | Cart | n=4 | Swing-up + balance | The standing **published** frontier by any method. RL; five links short. |
| Lam & Davison 2006 | Bottom-pivot torque chain (**not a cart**) | up to n=7 | **Balance only** | Different plant (base torque), different task (local stabilization, never swing-up). |
| yacine (@yacineMTB), 2026 | Cart (MuJoCo, pufferlib PPO) | n=6 | Swing-up + balance (RL) | Public-first at n=6 (conceded in the n=6 repo). No released code artifact at higher n seen. |
| Our n=5..7 releases | Cart | 5, 6, 7 | Swing-up + balance | Predecessors; n=7 was judged "~99% impossible" by our own adversarial program before that repo refuted it. |
| Our n=8 release (octuple-cartpole) | Cart | n=8 | Swing-up + balance | The immediate predecessor: full perturbed **composite** gate (24/24 × 2 seeds) via per-IC replanning (~hours/IC). |
| **This repo (nonuple-cartpole)** | Cart (single 150 N force) | **n=9** | **Swing-up + balance** | To our knowledge no public n=9 cart-pole swing-up claim exists by ANY method. Reproducible from a clean clone in one command. Banked perturbed-IC **pre-roll** gate: 24/24 × 3 seeds, no per-IC NLP (~150–230× faster than the n=8 composite). |

## Honest scope

Same boundary as the siblings: simulation only (1 kHz saturated ODE sim, not
hardware), full-state feedback, exact model, deterministic; robustness is
empirical (script-verified counts under a documented perturbation distribution
and committed predicate v1), not a theorem. The pre-roll gate is a *lighter*
controller than n=8's composite (one fixed LQR-about-down + one fixed nominal,
no per-IC replanning), and its honest rough edge is a saturating static-LQR
catch (150 N, ~14° transient that recovers) — disclosed in the README. The
"first" claim is "first public artifact we could find," dated 2026-07-04
(first n=9 σ=0.02 gate pass); it is falsifiable by counter-example and we will
concede priority exactly as the n=6 repo did if a prior public n=9 claim
surfaces.
