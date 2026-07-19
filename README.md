# N9 cart-pole replay

`n9-demo` produces one fresh hanging-start, force-saturated simulation and locally recomputes both feedback gains. `.working/n9/live-metrics.json` identifies the fixed dense nominal states and controls as its loaded tracking reference; `.working/n9/demo.gif` renders only the freshly integrated states.

```sh
uv sync --locked
uv run n9-demo
uv run n9-verify
```

> Supported distribution is a complete source checkout at the reviewed revision. Run `uv sync --locked` and the documented `uv run` commands from that checkout. The commands require repository-root configuration and evidence inputs. Wheels, sdists, package-index releases, and installs outside that checkout are unsupported and must not be published.

The YAML authority defines nine 0.5 m, 0.1 kg links on a 1 kg cart. The simulator uses 1 kHz zero-order hold, four 0.25 ms RK4 substeps, and a 150 N force-magnitude limit. Acceptance is evaluated at 1 kHz control-boundary samples: a five-second hold is 5,001 consecutive passing states, spanning `(5,001 - 1) × 0.001 = 5.0` seconds. State and track checks are sampled acceptance envelopes, not continuous-time or physical-rail claims. Applied force remains bounded over each zero-order-held interval by simulator clipping.

`n9-verify` checks canonical bytes for the frozen numerical sources, two nominal artifacts, and three banked gate records. It counts and structurally checks 24/24 stored historical success flags per seed and 72/72 total, then runs the same fresh stack; it does not re-evaluate historical outcomes or rerun trials. N9 excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness evidence.
