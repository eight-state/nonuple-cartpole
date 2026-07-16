# N9 cart-pole replay

`n9-demo` produces one fresh hanging-start, force-saturated simulation and locally recomputes both feedback gains. `.working/n9/live-metrics.json` identifies the fixed dense nominal states and controls as its loaded tracking reference; `.working/n9/demo.gif` renders only the freshly integrated states.

```sh
uv sync --locked
uv run n9-demo
uv run n9-verify
```

The YAML authority defines nine 0.5 m, 0.1 kg links on a 1 kg cart. The simulator uses 1 kHz zero-order hold, four 0.25 ms RK4 substeps, and a 150 N force-magnitude limit. `n9-verify` accepts the run only when every wrapped link angle stays within 5 degrees, each link rate within 0.5 rad/s, the cart within 2 m, and its speed within 0.5 m/s for 5 continuous seconds.

`n9-verify` checks canonical bytes for the frozen numerical sources, two nominal artifacts, and three banked gate records. It rederives 24/24 per seed and 72/72 total from their rows, then runs the same fresh stack. Gate records contain historical simulation evidence. N9 excludes nominal synthesis, perturbation reruns, hardware evidence, and robustness evidence.
