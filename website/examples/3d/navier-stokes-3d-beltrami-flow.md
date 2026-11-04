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

Mathematical form (from the benchmark records):

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

- [`reference_data/deepxde/beltrami_inverse/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/reference_data/deepxde/beltrami_inverse/u_obs.txt) → output `u`, weight 100
- [`reference_data/deepxde/beltrami_inverse/v_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/reference_data/deepxde/beltrami_inverse/v_obs.txt) → output `v`, weight 100
- [`reference_data/deepxde/beltrami_inverse/w_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/reference_data/deepxde/beltrami_inverse/w_obs.txt) → output `w`, weight 100

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

## Results

The configuration selected in PINNStudio's benchmark study, run on one RTX 4090. Forward: relative L₂ error of each output at the last reference time. Inverse: each parameter's relative error and, per seed, the largest of them (worst parameter error).

### Forward problem

**Settings:** Default — the example's own settings (not in the tuning sweep); the example has no separate Tuned preset for this mode. **Values:** median over seeds 2026, 1, 2.

| Output | Relative L₂ |
|---|---|
| u | 1.4 × 10⁻³ |
| v | 9.8 × 10⁻⁴ |
| w | 1.1 × 10⁻³ |

<figure markdown="span">
  ![Forward result: PINN, reference and error, u](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_surface_comparison_u.png){ loading=lazy }
  <figcaption>Forward, Default settings: PINN, reference and error, u.</figcaption>
</figure>

<figure markdown="span">
  ![Forward result: 3D view, u](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_view3d_comparison_u.png){ loading=lazy }
  <figcaption>Forward, Default settings: 3D view, u.</figcaption>
</figure>

??? note "Other outputs (6 figures)"

    <figure markdown="span">
      ![Forward result: PINN, reference and error, p](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_surface_comparison_p.png){ loading=lazy }
      <figcaption>Forward, Default settings: PINN, reference and error, p.</figcaption>
    </figure>

    <figure markdown="span">
      ![Forward result: PINN, reference and error, v](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_surface_comparison_v.png){ loading=lazy }
      <figcaption>Forward, Default settings: PINN, reference and error, v.</figcaption>
    </figure>

    <figure markdown="span">
      ![Forward result: PINN, reference and error, w](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_surface_comparison_w.png){ loading=lazy }
      <figcaption>Forward, Default settings: PINN, reference and error, w.</figcaption>
    </figure>

    <figure markdown="span">
      ![Forward result: 3D view, p](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_view3d_comparison_p.png){ loading=lazy }
      <figcaption>Forward, Default settings: 3D view, p.</figcaption>
    </figure>

    <figure markdown="span">
      ![Forward result: 3D view, v](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_view3d_comparison_v.png){ loading=lazy }
      <figcaption>Forward, Default settings: 3D view, v.</figcaption>
    </figure>

    <figure markdown="span">
      ![Forward result: 3D view, w](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow/base_view3d_comparison_w.png){ loading=lazy }
      <figcaption>Forward, Default settings: 3D view, w.</figcaption>
    </figure>

### Inverse problem

**Settings:** Tuned — 3× L-BFGS iterations. **Values:** median over seeds 2026, 1, 2.

| Parameter | True value | Recovered | Relative error |
|---|---|---|---|
| ν | 1 | 0.998265 | 0.17 % |

Worst parameter error: **0.17 %** (the median over the seeds of each seed's largest error).

<figure markdown="span">
  ![Inverse result: PINN, reference and error, u](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_surface_comparison_u.png){ loading=lazy }
  <figcaption>Inverse, Tuned settings: PINN, reference and error, u.</figcaption>
</figure>

<figure markdown="span">
  ![Inverse result: 3D view, u](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_view3d_comparison_u.png){ loading=lazy }
  <figcaption>Inverse, Tuned settings: 3D view, u.</figcaption>
</figure>

<figure markdown="span">
  ![Inverse result: parameter convergence](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_param_plot.png){ loading=lazy }
  <figcaption>Inverse, Tuned settings: parameter convergence.</figcaption>
</figure>

??? note "Other outputs (6 figures)"

    <figure markdown="span">
      ![Inverse result: PINN, reference and error, p](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_surface_comparison_p.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: PINN, reference and error, p.</figcaption>
    </figure>

    <figure markdown="span">
      ![Inverse result: PINN, reference and error, v](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_surface_comparison_v.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: PINN, reference and error, v.</figcaption>
    </figure>

    <figure markdown="span">
      ![Inverse result: PINN, reference and error, w](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_surface_comparison_w.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: PINN, reference and error, w.</figcaption>
    </figure>

    <figure markdown="span">
      ![Inverse result: 3D view, p](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_view3d_comparison_p.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: 3D view, p.</figcaption>
    </figure>

    <figure markdown="span">
      ![Inverse result: 3D view, v](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_view3d_comparison_v.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: 3D view, v.</figcaption>
    </figure>

    <figure markdown="span">
      ![Inverse result: 3D view, w](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/figures/tuning/Beltrami_flow_inv/lbfgs3_view3d_comparison_w.png){ loading=lazy }
      <figcaption>Inverse, Tuned settings: 3D view, w.</figcaption>
    </figure>

**Full benchmark record:** every setting tried, the per-seed values and all figures are in [the benchmark records on GitHub](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/examples.md#dx-beltrami-flow).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 3D ──** heading, **Navier-Stokes 3D – Beltrami flow** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 3D ──** heading, **Navier-Stokes 3D – Beltrami flow** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = -(exp(x)*sin(y + z) + exp(z)*cos(x + y))*exp(-t)`, `v = -(exp(y)*sin(z + x) + exp(x)*cos(y + z))*exp(-t)`, `w = -(exp(z)*sin(x + y) + exp(y)*cos(z + x))*exp(-t)`, `p = -0.5*(exp(2*x) + exp(2*y) + exp(2*z) + 2*sin(x + y)*cos(z + x)*exp(y + z) + 2*sin(y + z)*cos(x + y)*exp(z + x) + 2*sin(z + x)*cos(y + z)*exp(x + y))*exp(-2*t)`

## Notes

- Only the velocity has boundary and initial conditions, so the pressure p is determined only up to a constant: its predicted values are typically off by a fixed offset, which makes p's relative error large (and growing with t, as p decays like exp(-2t)) even when u, v, w are accurate.
- Inverse mode (PINNStudio's addition): the viscosity nu (1 in the forward equations; initial guess 0.5) is learned from u, v and w measured at 500 points each at t = 0.5 and t = 1 (reference_data/deepxde/beltrami_inverse).

## Definition and records

- **Definition:** `EXAMPLES` entry "Beltrami flow" in [`pinnstudio/core/deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/deepxde_examples.py)
    - Inverse mode (PINNStudio's addition): `ADDED_INVERSE["Beltrami flow"]`, same file
    - Settings: Tuned: entry "Beltrami flow" in [`pinnstudio/resources/tuned_settings.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/resources/tuned_settings.json)
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** the exact-solution formula under *Reference solution* (no data files)
- **Measured data** (Inverse): [`reference_data/deepxde/beltrami_inverse/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/v1.6.0/reference_data/deepxde/beltrami_inverse), written by [`tools/make_deepxde_reference_data.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tools/make_deepxde_reference_data.py)
- **Benchmark records:** [`benchmarks/problems/navier-stokes-3d-beltrami-flow.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/problems/navier-stokes-3d-beltrami-flow.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/examples.md#dx-beltrami-flow), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/README.md)
- **Tuning-study configurations** (records of the tuning study; open one with **Problem › Open Saved Problem...**; not presets of the Examples list): [`Navier-Stokes 3D – Beltrami flow (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/tuning/Navier-Stokes%203D%20%E2%80%93%20Beltrami%20flow%20%28tuned%29.pinn.json), [`Navier-Stokes 3D – Beltrami flow, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/tuning/Navier-Stokes%203D%20%E2%80%93%20Beltrami%20flow%2C%20inverse%20mode%20%28tuned%29.pinn.json)
- **DeepXDE example:** [Beltrami_flow.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Beltrami_flow.py)
- **Cite:** [Lu et al. 2021](../../about/references.md#lu2021deepxde) (DeepXDE); the problem follows [Jin et al. 2021](../../about/references.md#jin2021)

How the benchmark values were measured and checked: [Reproducibility](../../reproducibility/benchmarks.md).

[← Back to the example gallery](../index.md)
