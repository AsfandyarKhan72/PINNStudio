---
search:
  boost: 0.5
---

# Fractional diffusion 1D

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

\[\frac{\partial u}{\partial t} = D_{0+}^{\alpha}u + D_{1-}^{\alpha}u - e^{-t}\left(x^3(1-x)^3 + f_\alpha(x)\right), \qquad x \in [0, 1],\ t \in [0, 1],\ \alpha = 1.8\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - frac(u) + exp(-t)*(x**3*(1 - x)**3 + gamma(4)/gamma(4 - alpha)*(x**(3 - alpha) + (1 - x)**(3 - alpha)) - 3*gamma(5)/gamma(5 - alpha)*(x**(4 - alpha) + (1 - x)**(4 - alpha)) + 3*gamma(6)/gamma(6 - alpha)*(x**(5 - alpha) + (1 - x)**(5 - alpha)) - gamma(7)/gamma(7 - alpha)*(x**(6 - alpha) + (1 - x)**(6 - alpha)))
```

Constants: `alpha = 1.8`

## Domain

- Interval: \(x \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `x**3*(1 - x)**3` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 400 domain, 2 boundary, 2 initial, 100 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** output transform `x*(1 - x)*t*u + x**3*(1 - x)**3` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Fractional diffusion 1D** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = exp(-t)*x**3*(1 - x)**3`

## Source and citation

- DeepXDE script: [fractional_diffusion_1d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_diffusion_1d.py)

## Notes

- Time-dependent fractional equation (DeepXDE's TimeFPDE, static mesh of 52 points per time); the boundary and initial conditions are imposed exactly by the output transform.

## Configuration and benchmark records

- [`benchmarks/tuning/Fractional diffusion 1D (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Fractional%20diffusion%201D%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-fractional-diffusion-equation) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/fractional-diffusion-1d.md) (`benchmarks/problems/fractional-diffusion-1d.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
