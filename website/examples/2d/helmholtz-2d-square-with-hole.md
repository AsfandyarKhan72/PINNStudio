---
search:
  boost: 0.5
---

# Helmholtz 2D – square with hole

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Rectangle − Disk |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[-\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 2\pi\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx - du_yy - k0**2*u - k0**2*sin(k0*x)*sin(k0*y)
```

Constants: `n = 1`, `k0 = 2*pi*n`, `R = 1/4`

## Domain

- Custom geometry (**Geometry** › Custom): Rectangle (x_min = -0.5, x_max = 0.5, y_min = -0.5, y_max = 0.5), minus Disk (cx = 0, cy = 0, r = 0.25)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Neumann BC | `np.isclose(x**2 + y**2, R**2)` | `u` | `-(k0*cos(k0*x)*sin(k0*y)*x + k0*sin(k0*x)*cos(k0*y)*y)/R` | — |
| Dirichlet BC | `left or right or bottom or top` | `u` | `sin(k0*x)*sin(k0*y)` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 350, inputs (x, y), 1 output; activation sin, initializer Glorot uniform
- **Points:** 225 domain, 240 boundary, 900 test
- **Loss weights:** `1,10,100` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 5,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: an L-BFGS phase after Adam.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 3 × 350, inputs (x, y), 1 output; activation sin, initializer Glorot uniform
    - **Points:** 225 domain, 240 boundary, 900 test
    - **Loss weights:** `1,10,100` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 5,000 | 0.001 | none |
    | 2 | L-BFGS | 15,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Helmholtz 2D – square with hole** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(k0*x)*sin(k0*y)`

## Source and citation

- DeepXDE script: [Helmholtz_Neumann_2d_hole.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Neumann_2d_hole.py)
- DeepXDE documentation: [helmholtz.2d.neumann.hole.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/helmholtz.2d.neumann.hole.html)

## Configuration and benchmark records

- [`benchmarks/tuning/Helmholtz 2D – square with hole (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Helmholtz%202D%20%E2%80%93%20square%20with%20hole%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/helmholtz-2d-square-with-hole.md) (`benchmarks/problems/helmholtz-2d-square-with-hole.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
