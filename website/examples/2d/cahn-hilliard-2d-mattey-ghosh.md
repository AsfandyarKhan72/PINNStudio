---
search:
  boost: 0.5
---

# Cahn-Hilliard 2D – Mattey & Ghosh

| Example | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 2D (x, y, t) |
| **Regime** | Time-dependent |
| **Geometry** | Rectangle |
| **Outputs** | `h`, `mu` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial h}{\partial t'} = 0.005\left(\frac{\partial^2 \mu}{\partial x^2} + \frac{\partial^2 \mu}{\partial y^2}\right), \qquad \mu = h^3 - h - \alpha\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right), \qquad (x, y) \in [0, 1]^2,\ t' \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dh_t - 0.005*(dmu_xx + dmu_yy)
mu - 1.0*(h**3 - h) + 0.0002*(dh_xx + dh_yy)
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
dh_t - 0.005*(dmu_xx + dmu_yy)
mu - kappa*(h**3 - h) + alpha*(dh_xx + dh_yy)
```

## Domain

- Rectangle: \(x \in [0, 1]\), \(y \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `h` | `0.4*cos(4*pi*x)*cos(4*pi*(x + y))` | loss term |
| `mu` | `0` | not used as a loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Periodic BC | `left` | `h` | — | axis x, derivative order 0 |
| Periodic BC | `bottom` | `h` | — | axis y, derivative order 0 |
| Periodic BC | `left` | `mu` | — | axis x, derivative order 0 |
| Periodic BC | `bottom` | `mu` | — | axis y, derivative order 0 |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `alpha` | 1 | 0.0002 |
| `kappa` | 0.1 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/2D/cahn_hilliard_mattey/t_1_h.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/cahn_hilliard_mattey/t_1_h.txt) → output `h`, weight 100

From the benchmark records: same template, switched to **Inverse**, recovering \(\alpha\) and \(\kappa\) (initial guesses 1 and 0.1, true values 0.0002 and 1) from \(h\) measured at \(t' = 1\).

## Settings

### Forward mode

**Settings: Default** — the paper's method (Mattey & Ghosh 2022, bc-PINN)

- **Network:** FNN, hidden layers 5 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 50,000 domain, 1,600 boundary, 4,096 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,64.0` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [time windows (4 windows, transfer learning)](../../reference/time-windows.md); [backward-compatible PINN (bc-PINN) across the windows](../../reference/time-windows.md); [initial-condition guided learning](../../reference/pretraining.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 40,000 | 0.001 | none |
| 2 | L-BFGS | 75,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× the iterations of every training phase.* The change is applied to the template's own settings (not the paper's method that Default loads).

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [SA-PINN point weights](../../reference/point-weights.md); [exact initial condition](../../reference/exact-constraints.md); [exact periodicity (periodic input features)](../../reference/networks.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | none |
    | 2 | L-BFGS | 60,000 | — | none |

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)
- **Methods:** [SA-PINN point weights](../../reference/point-weights.md); [exact initial condition](../../reference/exact-constraints.md); [exact periodicity (periodic input features)](../../reference/networks.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: measured data at every reference time after t = 0.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)
    - **Methods:** [SA-PINN point weights](../../reference/point-weights.md); [exact initial condition](../../reference/exact-constraints.md); [exact periodicity (periodic input features)](../../reference/networks.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |
    | 2 | L-BFGS | 20,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Cahn-Hilliard 2D – Mattey & Ghosh** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Cahn-Hilliard 2D – Mattey & Ghosh** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/2D/cahn_hilliard_mattey/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/2D/cahn_hilliard_mattey)

## Source and citation

- Paper: [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474)
- From the benchmark records: Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474), section 6.2, IBVP-2: a pattern of stripes that sharpens into the two phases,.

## Configuration and benchmark records

- [`benchmarks/tuning/Cahn-Hilliard 2D – Mattey & Ghosh (PINNStudio template) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Cahn-Hilliard%202D%20%E2%80%93%20Mattey%20%26%20Ghosh%20%28PINNStudio%20template%29%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Cahn-Hilliard 2D – Mattey & Ghosh (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Cahn-Hilliard%202D%20%E2%80%93%20Mattey%20%26%20Ghosh%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#2d-cahn-hilliard-mattey--ghosh) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/cahn-hilliard-2d-mattey-ghosh.md) (`benchmarks/problems/cahn-hilliard-2d-mattey-ghosh.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
