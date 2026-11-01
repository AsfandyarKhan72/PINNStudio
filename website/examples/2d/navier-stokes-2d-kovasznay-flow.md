# Navier-Stokes 2D – Kovasznay flow

| | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Rectangle |
| **Outputs** | `u`, `v`, `p` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[(\mathbf u\cdot\nabla)\mathbf u + \nabla p = \frac{1}{Re}\nabla^2\mathbf u, \qquad \nabla\cdot\mathbf u = 0, \qquad Re = 20\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
u*du_x + v*du_y + dp_x - 1/Re*(du_xx + du_yy)
u*dv_x + v*dv_y + dp_y - 1/Re*(dv_xx + dv_yy)
du_x + dv_y
```

Constants: `Re = 20`, `nu = 1/Re`, `l = 1/(2*nu) - sqrt(1/(4*nu**2) + 4*pi**2)`

## Domain

- Rectangle: \(x \in [-0.5, 1]\), \(y \in [-0.5, 1.5]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `1 - exp(l*x)*cos(2*pi*y)` | — |
| Dirichlet BC | `True (the whole boundary)` | `v` | `l/(2*pi)*exp(l*x)*sin(2*pi*y)` | — |
| Dirichlet BC | `right` | `p` | `0.5*(1 - exp(2*l*x))` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `Re` | 10 | 20 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/kovasznay_inverse/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/kovasznay_inverse/u_obs.txt) → output `u`, weight 100
- [`reference_data/deepxde/kovasznay_inverse/v_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/kovasznay_inverse/v_obs.txt) → output `v`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y), 3 outputs; activation tanh, initializer Glorot normal
- **Points:** 2,601 domain, 400 boundary, 100,000 test
- **Loss weights:** `1,1,1,1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y), 3 outputs; activation tanh, initializer Glorot normal
- **Points:** 2,601 domain, 400 boundary, 100,000 test
- **Loss weights:** `1,1,1,1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 5 × 100, inputs (x, y), 3 outputs; activation tanh, initializer Glorot normal
    - **Points:** 7,803 domain, 1,200 boundary, 100,000 test
    - **Loss weights:** `1,1,1,1,1,1` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 90,000 | 0.001 | none |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Navier-Stokes 2D – Kovasznay flow** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Navier-Stokes 2D – Kovasznay flow** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = 1 - exp(l*x)*cos(2*pi*y)`, `v = l/(2*pi)*exp(l*x)*sin(2*pi*y)`, `p = 0.5*(1 - exp(2*l*x))`

## Source and citation

- DeepXDE script: [Kovasznay_flow.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Kovasznay_flow.py)
- DeepXDE documentation: [Kovasznay.flow.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html)
- Source paper (from the README): [Jin, Cai, Li & Karniadakis (2021), J. Comput. Phys. 426, 109951 (NSFnets)](https://doi.org/10.1016/j.jcp.2020.109951) — studies a related problem.

## Notes

- Constants Re, nu and l are defined as in DeepXDE's script.
- Inverse mode (PINNStudio's addition): the Reynolds number Re (initial guess 10, true 20) is learned from u and v measured at 500 points (reference_data/deepxde/kovasznay_inverse); the boundary values keep the true Re.

## Configuration and benchmark records

- [`benchmarks/tuning/Navier-Stokes 2D – Kovasznay flow (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Navier-Stokes%202D%20%E2%80%93%20Kovasznay%20flow%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [`benchmarks/tuning/Navier-Stokes 2D – Kovasznay flow, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Navier-Stokes%202D%20%E2%80%93%20Kovasznay%20flow%2C%20inverse%20mode%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/navier-stokes-2d-kovasznay-flow.md) (`benchmarks/problems/navier-stokes-2d-kovasznay-flow.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
