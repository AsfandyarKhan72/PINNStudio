---
search:
  boost: 0.5
---

# Poisson 3D – sphere

| Example | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 3D (x, y, z) |
| **Regime** | Steady |
| **Geometry** | Sphere |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[-\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} - \frac{\partial^2 u}{\partial z^2} = 1, \qquad x^2 + y^2 + z^2 \le 1\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx - du_yy - du_zz - 1
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
-du_xx - du_yy - du_zz - source_term
```

## Domain

- Sphere: centre (0, 0, 0), radius 1

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `source_term` | 0.1 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/3D/poisson_sphere/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/3D/poisson_sphere/u_obs.txt) → output `u`, weight 100

From the README: same template, switched to **Inverse**, recovering the source term (fixed at \(1\) in Forward mode) as an unknown parameter from 300 measured points

## Settings

### Forward mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y, z), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 2,000 domain, 300 boundary, 2,000 test
- **Loss weights:** `1.0,1.0` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y, z), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 2,000 domain, 300 boundary, 2,000 test
- **Loss weights:** `1.0,1.0,100.0` (equations, then boundary-condition rows, then measured data)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 3D ──** heading, **Poisson 3D – sphere** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 3D ──** heading, **Poisson 3D – sphere** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/3D/poisson_sphere/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/poisson_sphere)

## Source and citation

- PINNStudio's own problem (no source paper).

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 3D – sphere (PINNStudio template) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%203D%20%E2%80%93%20sphere%20%28PINNStudio%20template%29%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Poisson 3D – sphere (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%203D%20%E2%80%93%20sphere%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-3d-sphere.md) (`benchmarks/problems/poisson-3d-sphere.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
