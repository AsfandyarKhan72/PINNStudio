---
search:
  boost: 0.5
---

# Integro-differential ODE – Volterra kernel

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | ODE (t) |
| **Regime** | Time-dependent |
| **Geometry** | — (t only) |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[\frac{du}{dt} + u(t) = \int_0^t e^{s - t}\ u(s)\ ds, \qquad t \in [0, 5]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t + u - integral(exp(s - t)*u)
```

## Domain

- Time interval: \(t \in [0, 5]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `1` | loss term |

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 10 domain, 2 boundary, 100 test
- **Loss weights:** `1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── ODE ──** heading, **Integro-differential ODE – Volterra kernel** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = exp(-t)*cosh(t)`

## Source and citation

- DeepXDE script: [Volterra_IDE.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)
- Source paper (from the README): [Lu, Meng, Mao & Karniadakis (2021), SIAM Rev. 63 (DeepXDE)](https://arxiv.org/abs/1907.04502) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- integral(exp(s - t)*u) is the integral of the kernel e^(s - t) times u(s) from 0 to t, by 20-point Gauss-Legendre quadrature of the network (DeepXDE's dde.data.IDE, quad_deg = 20). Trained with L-BFGS only, as in DeepXDE's script.

## Configuration and benchmark records

- [`benchmarks/tuning/Integro-differential ODE – Volterra kernel (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Integro-differential%20ODE%20%E2%80%93%20Volterra%20kernel%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/integro-differential-ode-volterra-kernel.md) (`benchmarks/problems/integro-differential-ode-volterra-kernel.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
