---
search:
  boost: 0.5
---

# Poisson 1D – hard BCs

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the benchmark records):

\[-\frac{d^2u}{dx^2} = \sum_{k=1}^{4} k\sin(kx) + 8\sin(8x), \qquad x \in [0, \pi]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx - (sin(x) + 2*sin(2*x) + 3*sin(3*x) + 4*sin(4*x)) - 8*sin(8*x)
```

## Domain

- Interval: \(x \in [0, 3.14159]\)

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 50, inputs (x), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 64 domain, 0 boundary, 400 test
- **Loss weights:** `1` (equations, then boundary-condition rows)
- **Methods:** output transform `x*(pi - x)*u + x` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.0001 | inverse time (1000, 0.3) |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Poisson 1D – hard BCs** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = x + sin(x) + sin(2*x)/2 + sin(3*x)/3 + sin(4*x)/4 + sin(8*x)/8`

## Source and citation

- DeepXDE script: [Poisson_Dirichlet_1d_exactBC.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_Dirichlet_1d_exactBC.py)
- Source paper (from the benchmark records): [Yu, Lu, Meng & Karniadakis (2022), CMAME 393 (gPINNs)](https://arxiv.org/abs/2111.02801) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 1D – hard BCs (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20hard%20BCs%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-poisson-1d-hard-boundary-conditions) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-1d-hard-bcs.md) (`benchmarks/problems/poisson-1d-hard-bcs.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
