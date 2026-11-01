---
search:
  boost: 0.5
---

# Navier-Stokes 3D – Beltrami flow

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 3D (x, y, z, t) |
| **Regime** | Time-dependent |
| **Geometry** | Cuboid |
| **Outputs** | `u`, `v`, `w`, `p` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u + \nabla p = \nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad (x, y, z) \in [-1, 1]^3,\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t + (u*du_x + v*du_y + w*du_z) + dp_x - (du_xx + du_yy + du_zz)
dv_t + (u*dv_x + v*dv_y + w*dv_z) + dp_y - (dv_xx + dv_yy + dv_zz)
dw_t + (u*dw_x + v*dw_y + w*dw_z) + dp_z - (dw_xx + dw_yy + dw_zz)
du_x + dv_y + dw_z
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
du_t + (u*du_x + v*du_y + w*du_z) + dp_x - nu*(du_xx + du_yy + du_zz)
dv_t + (u*dv_x + v*dv_y + w*dv_z) + dp_y - nu*(dv_xx + dv_yy + dv_zz)
dw_t + (u*dw_x + v*dw_y + w*dw_z) + dp_z - nu*(dw_xx + dw_yy + dw_zz)
du_x + dv_y + dw_z
```

## Domain

- Cuboid: \(x \in [-1, 1]\), \(y \in [-1, 1]\), \(z \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `-(exp(x)*sin(y + z) + exp(z)*cos(x + y))` | loss term |
| `v` | `-(exp(y)*sin(z + x) + exp(x)*cos(y + z))` | loss term |
| `w` | `-(exp(z)*sin(x + y) + exp(y)*cos(z + x))` | loss term |
| `p` | `0` | not used as a loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `-(exp(x)*sin(y + z) + exp(z)*cos(x + y))*exp(-t)` | — |
| Dirichlet BC | `True (the whole boundary)` | `v` | `-(exp(y)*sin(z + x) + exp(x)*cos(y + z))*exp(-t)` | — |
| Dirichlet BC | `True (the whole boundary)` | `w` | `-(exp(z)*sin(x + y) + exp(y)*cos(z + x))*exp(-t)` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `nu` | 0.5 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/beltrami_inverse/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/beltrami_inverse/u_obs.txt) → output `u`, weight 100
- [`reference_data/deepxde/beltrami_inverse/v_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/beltrami_inverse/v_obs.txt) → output `v`, weight 100
- [`reference_data/deepxde/beltrami_inverse/w_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/beltrami_inverse/w_obs.txt) → output `w`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y, z, t), 4 outputs; activation tanh, initializer Glorot normal
- **Points:** 50,000 domain, 5,000 boundary, 5,000 initial, 10,000 test
- **Loss weights:** `1,1,1,1,100,100,100,100,100,100` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y, z, t), 4 outputs; activation tanh, initializer Glorot normal
- **Points:** 50,000 domain, 5,000 boundary, 5,000 initial, 10,000 test
- **Loss weights:** `1,1,1,1,100,100,100,100,100,100` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× the L-BFGS iterations.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 50, inputs (x, y, z, t), 4 outputs; activation tanh, initializer Glorot normal
    - **Points:** 50,000 domain, 5,000 boundary, 5,000 initial, 10,000 test
    - **Loss weights:** `1,1,1,1,100,100,100,100,100,100` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 30,000 | 0.001 | none |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 3D ──** heading, **Navier-Stokes 3D – Beltrami flow** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 3D ──** heading, **Navier-Stokes 3D – Beltrami flow** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = -(exp(x)*sin(y + z) + exp(z)*cos(x + y))*exp(-t)`, `v = -(exp(y)*sin(z + x) + exp(x)*cos(y + z))*exp(-t)`, `w = -(exp(z)*sin(x + y) + exp(y)*cos(z + x))*exp(-t)`, `p = -0.5*(exp(2*x) + exp(2*y) + exp(2*z) + 2*sin(x + y)*cos(z + x)*exp(y + z) + 2*sin(y + z)*cos(x + y)*exp(z + x) + 2*sin(z + x)*cos(y + z)*exp(x + y))*exp(-2*t)`

## Source and citation

- DeepXDE script: [Beltrami_flow.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py)
- Source paper (from the README): [Jin, Cai, Li & Karniadakis (2021), J. Comput. Phys. 426, 109951 (NSFnets), Table 4](https://doi.org/10.1016/j.jcp.2020.109951) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- Only the velocity has boundary and initial conditions, so the pressure p is determined only up to a constant: its predicted values are typically off by a fixed offset, which makes p's relative error large (and growing with t, as p decays like exp(-2t)) even when u, v, w are accurate.
- Inverse mode (PINNStudio's addition): the viscosity nu (1 in the forward equations; initial guess 0.5) is learned from u, v and w measured at 500 points each at t = 0.5 and t = 1 (reference_data/deepxde/beltrami_inverse).

## Configuration and benchmark records

- [`benchmarks/tuning/Navier-Stokes 3D – Beltrami flow (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Navier-Stokes%203D%20%E2%80%93%20Beltrami%20flow%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Navier-Stokes 3D – Beltrami flow, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Navier-Stokes%203D%20%E2%80%93%20Beltrami%20flow%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/navier-stokes-3d-beltrami-flow.md) (`benchmarks/problems/navier-stokes-3d-beltrami-flow.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
