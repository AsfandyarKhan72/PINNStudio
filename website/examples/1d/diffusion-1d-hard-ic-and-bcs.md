---
search:
  boost: 0.5
---

# Diffusion 1D – hard IC and BCs

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - du_xx + exp(-t)*(sin(pi*x) - pi**2*sin(pi*x))
```

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `sin(pi*x)` | not used as a loss term |

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 32, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 40 domain, 0 boundary, 0 initial, 10,000 test
- **Loss weights:** `1` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** output transform `t*(1 - x**2)*u + sin(pi*x)` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Diffusion 1D – hard IC and BCs** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)*exp(-t)`

## Source and citation

- DeepXDE script: [diffusion_1d_exactBC.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/diffusion_1d_exactBC.py)
- DeepXDE documentation: [diffusion.1d.exactBC.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.exactBC.html)
- Source paper (from the benchmark records): [Wu, Zhu, Tan, Kartha & Lu (2023), CMAME 403](https://arxiv.org/abs/2207.10289) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Configuration and benchmark records

- [`benchmarks/tuning/Diffusion 1D – hard IC and BCs (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Diffusion%201D%20%E2%80%93%20hard%20IC%20and%20BCs%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-diffusion-equation-with-hard-ic-and-bc) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/diffusion-1d-hard-ic-and-bcs.md) (`benchmarks/problems/diffusion-1d-hard-ic-and-bcs.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
