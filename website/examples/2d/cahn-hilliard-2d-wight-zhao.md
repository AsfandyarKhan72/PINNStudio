---
search:
  boost: 0.5
---

# Cahn-Hilliard 2D – Wight & Zhao

| Example | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 2D (x, y, t) |
| **Regime** | Time-dependent |
| **Geometry** | Rectangle |
| **Outputs** | `phi`, `mu` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial \phi}{\partial t} = \lambda\left(\frac{\partial^2 \mu}{\partial x^2} + \frac{\partial^2 \mu}{\partial y^2}\right), \qquad \mu = \phi^3 - \phi - \varepsilon^2\left(\frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}\right), \qquad (x, y) \in [-1, 1]^2,\ t \in [0, 1]\]

Where \(\lambda = 1\) and \(\varepsilon = 0.05\). As in the 1D example, the chemical potential \(\mu\) is a second output (the paper's network is also \((\phi, \mu)\)), so no residual needs a fourth derivative.

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dphi_t - (dmu_xx + dmu_yy)
mu - (phi**3 - phi) + 0.0025*(dphi_xx + dphi_yy)
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
dphi_t - (dmu_xx + dmu_yy)
mu - (phi**3 - phi) + eps_sq*(dphi_xx + dphi_yy)
```

## Domain

- Rectangle: \(x \in [-1, 1]\), \(y \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `phi` | `tanh((0.4 - (sqrt((x - 0.28)**2 + y**2 + 1e-12) + sqrt((x + 0.28)**2 + y**2 + 1e-12) - sqrt((sqrt((x - 0.28)**2 + y**2 + 1e-12) - sqrt((x + 0.28)**2 + y**2 + 1e-12))**2 + 0.0004))/2)/0.1)` | loss term |
| `mu` | `0` | not used as a loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Periodic BC | `left` | `phi` | — | axis x, derivative order 0 |
| Periodic BC | `bottom` | `phi` | — | axis y, derivative order 0 |
| Periodic BC | `left` | `mu` | — | axis x, derivative order 0 |
| Periodic BC | `bottom` | `mu` | — | axis y, derivative order 0 |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `eps_sq` | 1 | 0.0025 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/2D/cahn_hilliard_wight/t_1_phi.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/2D/cahn_hilliard_wight/t_1_phi.txt) → output `phi`, weight 100

From the benchmark records: same template, switched to **Inverse**, recovering \(\varepsilon^2\) (initial guess 1, true value 0.0025) from \(\phi\) measured at \(t = 1\).

## Settings

### Forward mode

**Settings: Default** — the paper's method (Wight & Zhao 2021)

- **Network:** FNN, hidden layers 6 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,1000.0` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [time windows (5 windows)](../../reference/time-windows.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |
| 2 | L-BFGS | 10,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: the template's own settings (PINNStudio's best for it).*

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,100.0` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [exact periodicity (periodic input features)](../../reference/networks.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |
    | 2 | L-BFGS | 20,000 | — | none |

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, y, t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 20,000 domain, 800 boundary, 1,024 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0,1.0,1.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)
- **Methods:** [exact periodicity (periodic input features)](../../reference/networks.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Cahn-Hilliard 2D – Wight & Zhao** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Cahn-Hilliard 2D – Wight & Zhao** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/2D/cahn_hilliard_wight/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/2D/cahn_hilliard_wight)

## Source and citation

- Paper: [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086)
- From the benchmark records: Benchmark problem after [Wight & Zhao (2021)](https://doi.org/10.4208/cicp.OA-2020-0086), section 3.3 (eqs.

## Configuration and benchmark records

- [`benchmarks/tuning/Cahn-Hilliard 2D – Wight & Zhao (PINNStudio template) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Cahn-Hilliard%202D%20%E2%80%93%20Wight%20%26%20Zhao%20%28PINNStudio%20template%29%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Cahn-Hilliard 2D – Wight & Zhao (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Cahn-Hilliard%202D%20%E2%80%93%20Wight%20%26%20Zhao%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#2d-cahn-hilliard-wight--zhao) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/cahn-hilliard-2d-wight-zhao.md) (`benchmarks/problems/cahn-hilliard-2d-wight-zhao.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
