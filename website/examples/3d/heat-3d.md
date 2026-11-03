---
search:
  boost: 0.5
---

# Heat 3D

| Example | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 3D (x, y, z, t) |
| **Regime** | Time-dependent |
| **Geometry** | Cuboid |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

<figure markdown="span">
  ![Animated 3D voxel plot of the heat-equation PINN solution in the unit cube](../../assets/images/examples/heat-3d.gif){ loading=lazy }
  <figcaption>Animation from the repository's figures (`assets/Images/Top_GIFs/`): the PINN solution of this example over time.</figcaption>
</figure>

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial u}{\partial t} = 0.4\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right), \qquad (x, y, z) \in [0, 1]^3,\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - 0.4*(du_xx + du_yy + du_zz)
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
du_t - alpha*(du_xx + du_yy + du_zz)
```

## Domain

- Cuboid: \(x \in [0, 1]\), \(y \in [0, 1]\), \(z \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `0.0` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Neumann BC | `left` | `u` | `0` | — |
| Dirichlet BC | `right` | `u` | `1` | — |
| Neumann BC | `bottom` | `u` | `0` | — |
| Neumann BC | `top` | `u` | `0` | — |
| Neumann BC | `front` | `u` | `0` | — |
| Neumann BC | `back` | `u` | `0` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `alpha` | 1 | 0.4 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/3D/heat/t_1.0_LessData.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/3D/heat/t_1.0_LessData.txt) → output `u`, weight 100

From the benchmark records: same template, switched to **Inverse**, recovering the diffusivity coefficient (fixed at \(0.4\) in Forward mode) as an unknown parameter.

## Settings

### Forward mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, y, z, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 15,000 domain, 1,000 boundary, 1,000 initial, 10,000 test
- **Loss weights:** `1.0,1.0,100.0,1.0,1.0,1.0,1.0,100.0` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, y, z, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 15,000 domain, 1,000 boundary, 1,000 initial, 10,000 test
- **Loss weights:** `1.0,1.0,100.0,1.0,1.0,1.0,1.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 10× the weight of the measured data.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 128, inputs (x, y, z, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 15,000 domain, 1,000 boundary, 1,000 initial, 10,000 test
    - **Loss weights:** `1.0,1.0,100.0,1.0,1.0,1.0,1.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |
    | 2 | L-BFGS | 20,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 3D ──** heading, **Heat 3D** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 3D ──** heading, **Heat 3D** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/3D/heat/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/heat)

## Definition and records

- **Definition:** template "3D Heat" in `MainWindow._on_template_selected`, [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py)
    - Inverse mode: `MainWindow.INVERSE_AUTO_VARS["3D Heat"]` (unknowns) and `INVERSE_AUTO_OBS["3D Heat"]` (measured data), same file
    - Settings: Tuned: entry "3D Heat" in [`pinnstudio/resources/tuned_settings.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/resources/tuned_settings.json)
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** [`reference_data/3D/heat/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/heat), no generator script in the repository (checked against an independent solution by [`benchmarks/reference_regeneration/check_other_references.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration/check_other_references.py))
- **Measured data** (Inverse): [`reference_data/3D/heat/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/3D/heat), no generator script in the repository (checked against an independent solution by [`benchmarks/reference_regeneration/check_other_references.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/reference_regeneration/check_other_references.py))
- **Benchmark records:** [`benchmarks/problems/heat-3d.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/heat-3d.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#3d-heat), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md)
- **Tuning-study configurations** (records of the tuning study; open one with **Problem › Open Saved Problem...**; not presets of the Examples list): [`Heat 3D (PINNStudio template) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Heat%203D%20%28PINNStudio%20template%29%20%28tuned%29.pinn.json), [`Heat 3D (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Heat%203D%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json)
- **Cite:** PINNStudio's own problem (no source paper)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
