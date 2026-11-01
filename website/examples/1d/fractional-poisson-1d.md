---
search:
  boost: 0.5
---

# Fractional Poisson 1D

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[D_{0+}^{\alpha}u + D_{1-}^{\alpha}u = f(x), \qquad x \in [0, 1],\ \alpha = 1.5\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
frac(u) - (gamma(4)/gamma(4 - alpha)*(x**(3 - alpha) + (1 - x)**(3 - alpha)) - 3*gamma(5)/gamma(5 - alpha)*(x**(4 - alpha) + (1 - x)**(4 - alpha)) + 3*gamma(6)/gamma(6 - alpha)*(x**(5 - alpha) + (1 - x)**(5 - alpha)) - gamma(7)/gamma(7 - alpha)*(x**(6 - alpha) + (1 - x)**(6 - alpha)))
```

Constants: `alpha = 1.5`

## Domain

- Interval: \(x \in [0, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 20, inputs (x), 1 output; activation tanh, initializer Glorot normal
- **Points:** 101 domain, 2 boundary, 100 test
- **Loss weights:** `1,1` (equations, then boundary-condition rows)
- **Methods:** output transform `x*(1 - x)*u` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: a larger network (hidden layers twice as wide, one more layer).* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 5 × 40, inputs (x), 1 output; activation tanh, initializer Glorot normal
    - **Points:** 101 domain, 2 boundary, 100 test
    - **Loss weights:** `1,1` (equations, then boundary-condition rows)
    - **Methods:** output transform `x*(1 - x)*u` ([reference](../../reference/networks.md))

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 10,000 | 0.001 | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Fractional Poisson 1D** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = x**3*(1 - x)**3`

## Source and citation

- DeepXDE script: [fractional_Poisson_1d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_1d.py)
- Source paper (from the README): [Pang, Lu & Karniadakis (2019), SIAM J. Sci. Comput. 41 (fPINNs)](https://arxiv.org/abs/1811.08967) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- frac(u) is the sum of the left and right Riemann-Liouville derivatives of order alpha, discretized by DeepXDE's FPDE (Grunwald-Letnikov, static mesh of 101 points). DeepXDE's own script supports only the TensorFlow and Paddle backends; PINNStudio applies the same FPDE matrix with PyTorch.

## Configuration and benchmark records

- [`benchmarks/tuning/Fractional Poisson 1D (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Fractional%20Poisson%201D%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/fractional-poisson-1d.md) (`benchmarks/problems/fractional-poisson-1d.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
