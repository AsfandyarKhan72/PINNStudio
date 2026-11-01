---
search:
  boost: 0.5
---

# Allen-Cahn 2D – Mattey & Ghosh

| Example | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 2D (x, y, t) |
| **Regime** | Time-dependent |
| **Geometry** | Rectangle |
| **Outputs** | `h` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{\partial h}{\partial t} = c_1^2\left(\frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2}\right) - c_2(h^3 - h), \qquad (x, y) \in [0, 1]^2,\ t \in (0, 1]\]

Where \(c_1^2 = 0.0001\) and \(c_2 = 1\).

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dh_t - 0.0001*(dh_xx + dh_yy) + 1.0*(h**3 - h)
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
dh_t - c1_sq*(dh_xx + dh_yy) + c2*(h**3 - h)
```

## Domain

- Rectangle: \(x \in [0, 1]\), \(y \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `h` | `sin(4*pi*x)*cos(4*pi*y)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Periodic BC | `left` | `h` | — | axis x, derivative order 0 |
| Periodic BC | `bottom` | `h` | — | axis y, derivative order 0 |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `c1_sq` | 1 | 0.0001 |
| `c2` | 0.1 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/2D/allen_cahn_mattey/t_0.25.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/allen_cahn_mattey/t_0.25.txt) → output `h`, weight 100
- [`reference_data/2D/allen_cahn_mattey/t_0.5.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/allen_cahn_mattey/t_0.5.txt) → output `h`, weight 100
- [`reference_data/2D/allen_cahn_mattey/t_0.75.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/allen_cahn_mattey/t_0.75.txt) → output `h`, weight 100
- [`reference_data/2D/allen_cahn_mattey/t_1.0.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/allen_cahn_mattey/t_1.0.txt) → output `h`, weight 100

From the README: same template, switched to **Inverse**, jointly recovering both \(c_1^2\) (interfacial-thickness-squared) and \(c_2\) (reaction) as unknown parameters; \(c_2\)'s initial guess starts at \(0.1\) rather than \(1.0\), since \(1.0\) is already its true value. The template measures \(h\) at every reference time after \(t = 0\) (\(t\) = 0.25, 0.5, 0.75 and 1, four data files, each with weight 100) and trains with Adam 20,000 → L-BFGS 20,000 (Forward keeps L-BFGS 10,000).

## Settings

### Forward mode

**Settings: Default** — the paper's method (Mattey & Ghosh 2022, bc-PINN)

- **Network:** FNN, hidden layers 6 × 128, inputs (x, y, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 20,000 domain, 640 boundary, 4,096 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,64.0` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [time windows (5 windows, transfer learning)](../../reference/time-windows.md); [backward-compatible PINN (bc-PINN) across the windows](../../reference/time-windows.md); [initial-condition guided learning](../../reference/pretraining.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 12,000 | 0.001 | none |
| 2 | L-BFGS | 10,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: exact IC, exact periodicity, the whole time interval, SA-PINN weights.* The change is applied to the template's own settings (not the paper's method that Default loads).

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 10,000 domain, 400 boundary, 512 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [SA-PINN point weights](../../reference/point-weights.md); [exact initial condition](../../reference/exact-constraints.md); [exact periodicity (periodic input features)](../../reference/networks.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |
    | 2 | L-BFGS | 10,000 | — | none |

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 10,000 domain, 400 boundary, 512 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,100.0,100.0,100.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: exact IC, exact periodicity, the whole time interval, SA-PINN weights.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 10,000 domain, 400 boundary, 512 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [SA-PINN point weights](../../reference/point-weights.md); [exact initial condition](../../reference/exact-constraints.md); [exact periodicity (periodic input features)](../../reference/networks.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |
    | 2 | L-BFGS | 20,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Allen-Cahn 2D – Mattey & Ghosh** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Allen-Cahn 2D – Mattey & Ghosh** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/2D/allen_cahn_mattey/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/2D/allen_cahn_mattey)

## Source and citation

- Paper: [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474)
- README: Benchmark problem after [Mattey & Ghosh (2022)](https://doi.org/10.1016/j.cma.2021.114474).

## Configuration and benchmark records

- [`benchmarks/tuning/Allen-Cahn 2D – Mattey & Ghosh (PINNStudio template) (best setup) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Allen-Cahn%202D%20%E2%80%93%20Mattey%20%26%20Ghosh%20%28PINNStudio%20template%29%20%28best%20setup%29%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Allen-Cahn 2D – Mattey & Ghosh (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Allen-Cahn%202D%20%E2%80%93%20Mattey%20%26%20Ghosh%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/allen-cahn-2d-mattey-ghosh.md) (`benchmarks/problems/allen-cahn-2d-mattey-ghosh.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
