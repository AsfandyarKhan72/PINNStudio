---
search:
  boost: 0.5
---

# Linear elasticity 2D – plate

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Rectangle |
| **Outputs** | `ux`, `uy`, `sxx`, `syy`, `sxy` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} = f_x, \qquad \frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} = f_y\]

\[\sigma_{xx} = (\lambda + 2\mu)\varepsilon_{xx} + \lambda\varepsilon_{yy}, \quad \sigma_{yy} = (\lambda + 2\mu)\varepsilon_{yy} + \lambda\varepsilon_{xx}, \quad \sigma_{xy} = 2\mu\varepsilon_{xy}\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dsxx_x + dsxy_y - (-lmbd*(4*pi**2*cos(2*pi*x)*sin(pi*y) - Q*y**3*pi*cos(pi*x)) - mu*(pi**2*cos(2*pi*x)*sin(pi*y) - Q*y**3*pi*cos(pi*x)) - 8*mu*pi**2*cos(2*pi*x)*sin(pi*y))
dsxy_x + dsyy_y - (lmbd*(3*Q*y**2*sin(pi*x) - 2*pi**2*cos(pi*y)*sin(2*pi*x)) - mu*(2*pi**2*cos(pi*y)*sin(2*pi*x) + Q*y**4*pi**2*sin(pi*x)/4) + 6*Q*mu*y**2*sin(pi*x))
(2*mu + lmbd)*dux_x + lmbd*duy_y - sxx
(2*mu + lmbd)*duy_y + lmbd*dux_x - syy
mu*(dux_y + duy_x) - sxy
```

Constants: `lmbd = 1`, `mu = 0.5`, `Q = 4`

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
dsxx_x + dsxy_y - (-(1)*(4*pi**2*cos(2*pi*x)*sin(pi*y) - Q*y**3*pi*cos(pi*x)) - (0.5)*(pi**2*cos(2*pi*x)*sin(pi*y) - Q*y**3*pi*cos(pi*x)) - 8*(0.5)*pi**2*cos(2*pi*x)*sin(pi*y))
dsxy_x + dsyy_y - ((1)*(3*Q*y**2*sin(pi*x) - 2*pi**2*cos(pi*y)*sin(2*pi*x)) - (0.5)*(2*pi**2*cos(pi*y)*sin(2*pi*x) + Q*y**4*pi**2*sin(pi*x)/4) + 6*Q*(0.5)*y**2*sin(pi*x))
(2*mu + lmbd)*dux_x + lmbd*duy_y - sxx
(2*mu + lmbd)*duy_y + lmbd*dux_x - syy
mu*(dux_y + duy_x) - sxy
```

Constants: `Q = 4`

## Domain

- Rectangle: \(x \in [0, 1]\), \(y \in [0, 1]\)

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `lmbd` | 2 | 1 |
| `mu` | 1 | 0.5 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/elasticity_inverse/ux_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/elasticity_inverse/ux_obs.txt) → output `ux`, weight 100
- [`reference_data/deepxde/elasticity_inverse/uy_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/elasticity_inverse/uy_obs.txt) → output `uy`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** parallel FNN (PFNN), hidden layers 4 × 40, inputs (x, y), 5 outputs; activation tanh, initializer Glorot uniform
- **Points:** 500 domain, 500 boundary, 100 test
- **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)
- **Methods:** output transform `ux*y*(1 - y)|uy*x*(1 - x)*y|sxx*x*(1 - x)|syy*(1 - y) + (lmbd + 2*mu)*Q*sin(pi*x)|sxy` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** parallel FNN (PFNN), hidden layers 5 × 80, inputs (x, y), 5 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 1,500 domain, 1,500 boundary, 100 test
    - **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)
    - **Methods:** output transform `ux*y*(1 - y)|uy*x*(1 - x)*y|sxx*x*(1 - x)|syy*(1 - y) + (lmbd + 2*mu)*Q*sin(pi*x)|sxy` ([reference](../../reference/networks.md))

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | none |
    | 2 | L-BFGS | 60,000 | — | none |

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** parallel FNN (PFNN), hidden layers 4 × 40, inputs (x, y), 5 outputs; activation tanh, initializer Glorot uniform
- **Points:** 500 domain, 500 boundary, 100 test
- **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)
- **Methods:** output transform `ux*y*(1 - y)|uy*x*(1 - x)*y|sxx*x*(1 - x)|syy*(1 - y) + (1 + 2*0.5)*Q*sin(pi*x)|sxy` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 20,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Linear elasticity 2D – plate** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Linear elasticity 2D – plate** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Forward mode: exact solution, **Error Analysis** › reference from formula: `ux = cos(2*pi*x)*sin(pi*y)`, `uy = sin(pi*x)*Q*y**4/4`, `sxx = (2*mu + lmbd)*(-2*pi*sin(2*pi*x)*sin(pi*y)) + lmbd*sin(pi*x)*Q*y**3`, `syy = (2*mu + lmbd)*sin(pi*x)*Q*y**3 + lmbd*(-2*pi*sin(2*pi*x)*sin(pi*y))`, `sxy = mu*(pi*cos(2*pi*x)*cos(pi*y) + pi*cos(pi*x)*Q*y**4/4)`
- Inverse mode: exact solution, **Error Analysis** › reference from formula: `ux = cos(2*pi*x)*sin(pi*y)`, `uy = sin(pi*x)*Q*y**4/4`, `sxx = (2*(0.5) + (1))*(-2*pi*sin(2*pi*x)*sin(pi*y)) + (1)*sin(pi*x)*Q*y**3`, `syy = (2*(0.5) + (1))*sin(pi*x)*Q*y**3 + (1)*(-2*pi*sin(2*pi*x)*sin(pi*y))`, `sxy = (0.5)*(pi*cos(2*pi*x)*cos(pi*y) + pi*cos(pi*x)*Q*y**4/4)`

## Notes

- Boundary conditions are imposed exactly through the output transform (DeepXDE's default 'hard' setting), so the Boundary Conditions panel is empty. Training: 20,000 Adam + 20,000 L-BFGS iterations instead of DeepXDE's 5,000 Adam, which are not enough for this problem (L2 errors of 0.2-1.6).
- Inverse mode (PINNStudio's addition): the Lamé parameters lmbd (initial guess 2, true 1) and mu (initial guess 1, true 0.5) are learned from the displacements ux and uy measured on a 19 x 19 grid (reference_data/deepxde/elasticity_inverse). The body force keeps the true values (it is the known load), so lmbd and mu are unknown only in the stress-strain law.

## Definition and records

- **Definition:** `EXAMPLES` entry "Linear elastostatic plate" in [`pinnstudio/core/deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/deepxde_examples.py)
    - Inverse mode (PINNStudio's addition): `ADDED_INVERSE["Linear elastostatic plate"]`, same file
    - Settings: Tuned: entry "Linear elastostatic plate" in [`pinnstudio/resources/tuned_settings.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/resources/tuned_settings.json)
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** the exact-solution formula under *Reference solution* (no data files)
- **Measured data** (Inverse): [`reference_data/deepxde/elasticity_inverse/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/deepxde/elasticity_inverse), written by [`tools/make_deepxde_reference_data.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/make_deepxde_reference_data.py)
- **Benchmark records:** [`benchmarks/problems/linear-elasticity-2d-plate.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/linear-elasticity-2d-plate.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-linear-elastostatic-plate), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md)
- **Tuning-study configurations** (records of the tuning study; open one with **Problem › Open Saved Problem...**; not presets of the Examples list): [`Linear elasticity 2D – plate (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Linear%20elasticity%202D%20%E2%80%93%20plate%20%28tuned%29.pinn.json), [`Linear elasticity 2D – plate, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Linear%20elasticity%202D%20%E2%80%93%20plate%2C%20inverse%20mode%20%28tuned%29.pinn.json)
- **Methods** (Default and Tuned settings): [Networks and Fourier features](../../reference/networks.md)
- **DeepXDE example:** [elasticity_plate.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/elasticity_plate.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/elasticity.plate.html))
- **Cite:** [Lu et al. 2021](../../about/references.md#lu2021deepxde) (DeepXDE); the problem follows [Haghighat et al. 2021](../../about/references.md#haghighat2021)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
