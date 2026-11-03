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

Mathematical form (from the benchmark records):

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

From the benchmark records: same template, switched to **Inverse**, recovering the source term (fixed at \(1\) in Forward mode) as an unknown parameter from 300 measured points

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

## Definition and records

- **Definition:** template "3D Poisson (Sphere)" in `MainWindow._on_template_selected`, [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py)
    - Inverse mode: `MainWindow.INVERSE_AUTO_VARS["3D Poisson (Sphere)"]` (unknowns) and `INVERSE_AUTO_OBS["3D Poisson (Sphere)"]` (measured data), same file
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** [`reference_data/3D/poisson_sphere/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/poisson_sphere), written by [`tools/make_poisson_reference_data.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/make_poisson_reference_data.py)
- **Measured data** (Inverse): [`reference_data/3D/poisson_sphere/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/poisson_sphere), written by [`tools/make_poisson_reference_data.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/make_poisson_reference_data.py)
- **Benchmark records:** [`benchmarks/problems/poisson-3d-sphere.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-3d-sphere.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#3d-poisson-sphere), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md)
- **Tuning-study configurations** (records of the tuning study; open one with **Problem › Open Saved Problem...**; not presets of the Examples list): [`Poisson 3D – sphere (PINNStudio template) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%203D%20%E2%80%93%20sphere%20%28PINNStudio%20template%29%20%28tuned%29.pinn.json), [`Poisson 3D – sphere (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%203D%20%E2%80%93%20sphere%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json)
- **Cite:** PINNStudio's own problem (no source paper)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
