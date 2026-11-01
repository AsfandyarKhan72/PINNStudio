---
search:
  boost: 0.5
---

# Poisson 1D – Dirichlet/Periodic

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

Mathematical form (from the benchmark records):

\[-\frac{d^2u}{dx^2} = \pi^2\sin(\pi x), \qquad x \in [-1, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx - pi**2*sin(pi*x)
```

## Domain

- Interval: \(x \in [-1, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `left` | `u` | `0` | — |
| Periodic BC | `right` | `u` | — | axis x, derivative order 0 |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 50, inputs (x), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 16 domain, 2 boundary, 100 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 100, inputs (x), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 48 domain, 6 boundary, 100 test
    - **Loss weights:** `1,1,1` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 30,000 | 0.001 | none |
    | 2 | L-BFGS | 15,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Poisson 1D – Dirichlet/Periodic** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)`

## Source and citation

- DeepXDE script: [Poisson_periodic_1d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Poisson_periodic_1d.py)
- DeepXDE documentation: [poisson.1d.dirichletperiodic.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/poisson.1d.dirichletperiodic.html)

## Notes

- DeepXDE's script passes sin(pi x) as the left boundary value; it is 0 at x = -1.

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 1D – Dirichlet - Periodic (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20Dirichlet%20-%20Periodic%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-poisson-1d-dirichlet-periodic) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-1d-dirichlet-periodic.md) (`benchmarks/problems/poisson-1d-dirichlet-periodic.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
