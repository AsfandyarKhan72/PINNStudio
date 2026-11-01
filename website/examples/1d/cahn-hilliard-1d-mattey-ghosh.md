# Cahn-Hilliard 1D – Mattey & Ghosh

| | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `h`, `mu` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{\partial h}{\partial t} = \frac{\partial^2 \mu}{\partial x^2}, \qquad \mu = \kappa (h^3 - h) - \alpha\kappa \frac{\partial^2 h}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]\]

Where \(\alpha = 0.02\) and \(\kappa = 1\). As in the paper (its phase-space form, eq. 19) and the Wight & Zhao example, \(\mu\) is a second output, so no residual needs a fourth derivative. The same paper's 1D Allen-Cahn problem is the [Allen-Cahn 1D – Wight & Zhao](allen-cahn-1d-wight-zhao.md) example, and its 2D Allen-Cahn IBVP-1 is [Allen-Cahn 2D – Mattey & Ghosh](../2d/allen-cahn-2d-mattey-ghosh.md).

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dh_t - dmu_xx
mu - 1.0*(h**3 - h) + 0.02*dh_xx
```

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `h` | `cos(pi*x) - exp(-4*(pi*x)**2)` | loss term |
| `mu` | `0` | not used as a loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Periodic BC | `left` | `h` | — | axis x, derivative order 0 |
| Periodic BC | `left` | `mu` | — | axis x, derivative order 0 |
| Periodic BC | `left` | `h` | — | axis x, derivative order 1 |
| Periodic BC | `left` | `mu` | — | axis x, derivative order 1 |

## Settings

**Settings: Default** — the paper's method (Mattey & Ghosh 2022, bc-PINN)

- **Network:** FNN, hidden layers 6 × 128, inputs (x, t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 10,000 domain, 20 boundary, 128 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,64.0` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** time windows (10 windows, transfer learning); backward-compatible PINN (bc-PINN) across the windows; initial-condition guided learning

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |
| 2 | L-BFGS | 50,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: Mattey & Ghosh's bc-PINN.* The change is applied to the template's own settings (not the paper's method that Default loads).

    - **Network:** FNN, hidden layers 6 × 128, inputs (x, t), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 10,000 domain, 20 boundary, 128 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,64.0` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** time windows (10 windows, transfer learning); backward-compatible PINN (bc-PINN) across the windows; initial-condition guided learning

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 10,000 | 0.001 | none |
    | 2 | L-BFGS | 50,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Cahn-Hilliard 1D – Mattey & Ghosh** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/1D/cahn_hilliard_mattey/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/1D/cahn_hilliard_mattey)

## Source and citation

- Paper: [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474)
- README: Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474), section 5.2 (eq.

## Notes

- Forward only: the Inverse mode of this template is withdrawn for now because the parameters were not found (pinnstudio/core/example_catalog.py, FORWARD_ONLY_TEMPLATES).

## Configuration and benchmark records

- [`benchmarks/tuning/Cahn-Hilliard 1D – Mattey & Ghosh (PINNStudio template) (best setup) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Cahn-Hilliard%201D%20%E2%80%93%20Mattey%20%26%20Ghosh%20%28PINNStudio%20template%29%20%28best%20setup%29%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/cahn-hilliard-1d-mattey-ghosh.md) (`benchmarks/problems/cahn-hilliard-1d-mattey-ghosh.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
