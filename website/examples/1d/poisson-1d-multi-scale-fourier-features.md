---
search:
  boost: 0.5
---

# Poisson 1D – multi-scale Fourier features

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

\[\frac{d^2u}{dx^2} = -(\pi A)^2\sin(\pi A x) - 0.1(\pi B)^2\sin(\pi B x), \qquad x \in [0, 1],\ A = 2,\ B = 50\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_xx + (pi*A)**2*sin(pi*A*x) + 0.1*(pi*B)**2*sin(pi*B*x)
```

Constants: `A = 2`, `B = 50`

## Domain

- Interval: \(x \in [0, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** multi-scale Fourier-feature network (MsFFN), hidden layers 3 × 100, inputs (x), 1 output; activation tanh, initializer Glorot uniform; Fourier-feature scales 1, 10
- **Points:** 1,280 domain, 2 boundary, 10,000 test
- **Loss weights:** `1,1` (equations, then boundary-condition rows)
- **Methods:** [multi-scale Fourier features](../../reference/networks.md); collocation-point resampling every 1 iterations

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | inverse time (2000, 0.9) |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Poisson 1D – multi-scale Fourier features** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*A*x) + 0.1*sin(pi*B*x)`

## Source and citation

- DeepXDE script: [Poisson_multiscale_1d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_multiscale_1d.py)
- Source paper (from the benchmark records): [Wang, Wang & Perdikaris (2021), CMAME 384](https://arxiv.org/abs/2012.10047) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- Multi-scale Fourier feature network (scales 1 and 10): the solution has a slow wave sin(2 pi x) and a fast one 0.1 sin(50 pi x). DeepXDE's script passes the exact solution as the boundary value; it is 0 at x = 0 and 1. Collocation points are redrawn every iteration.

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 1D – multi-scale Fourier features (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20multi-scale%20Fourier%20features%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-poisson-1d-multi-scale-fourier-features) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-1d-multi-scale-fourier-features.md) (`benchmarks/problems/poisson-1d-multi-scale-fourier-features.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
