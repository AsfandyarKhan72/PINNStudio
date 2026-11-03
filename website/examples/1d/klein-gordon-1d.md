---
search:
  boost: 0.5
---

# Klein-Gordon 1D

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} + u^2 = -x\cos t + x^2\cos^2 t, \qquad x \in [-1, 1],\ t \in [0, 10]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_tt + alpha*du_xx + beta*u + gamma*u**k + x*cos(t) - x**2*cos(t)**2
```

Constants: `alpha = -1`, `beta = 0`, `gamma = 1`, `k = 2`

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 10]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `x` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `x*cos(t)` | — |
| Operator BC (advanced) | initial time | — | `du_t` | applied at the initial time |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `gamma` | 0.5 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/klein_gordon_inverse/t_1.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_1.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_2.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_2.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_3.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_3.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_4.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_4.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_5.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_5.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_6.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_6.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_7.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_7.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_8.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_8.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_9.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_9.txt) → output `u`, weight 100
- [`reference_data/deepxde/klein_gordon_inverse/t_10.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/klein_gordon_inverse/t_10.txt) → output `u`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 2 × 40, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 30,000 domain, 1,500 boundary, 1,500 initial, 6,000 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | inverse time (3000, 0.9) |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: a larger network (hidden layers twice as wide, one more layer).* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 3 × 80, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 30,000 domain, 1,500 boundary, 1,500 initial, 6,000 test
    - **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | inverse time (3000, 0.9) |
    | 2 | L-BFGS | 15,000 | — | none |

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 2 × 40, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 30,000 domain, 1,500 boundary, 1,500 initial, 6,000 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | inverse time (3000, 0.9) |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 3 × 80, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 90,000 domain, 4,500 boundary, 4,500 initial, 6,000 test
    - **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | inverse time (3000, 0.9) |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Klein-Gordon 1D** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Klein-Gordon 1D** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = x*cos(t)`

## Notes

- The second initial condition du/dt = 0 is applied at t = 0. (DeepXDE's script passes it to an OperatorBC, which selects spatial-boundary points rather than t = 0.)
- Inverse mode (PINNStudio's addition): the coefficient gamma of the nonlinear term (initial guess 0.5, true 1) is learned from u measured at t = 1, 2, ..., 10 (reference_data/deepxde/klein_gordon_inverse). alpha and beta stay known: the solution u = x cos t has u_xx = 0, so the data cannot tell alpha, and beta = 0.

## Definition and records

- **Definition:** `EXAMPLES` entry "Klein-Gordon equation" in [`pinnstudio/core/deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/deepxde_examples.py)
    - Inverse mode (PINNStudio's addition): `ADDED_INVERSE["Klein-Gordon equation"]`, same file
    - Settings: Tuned: entry "Klein-Gordon equation" in [`pinnstudio/resources/tuned_settings.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/resources/tuned_settings.json)
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** the exact-solution formula under *Reference solution* (no data files)
- **Measured data** (Inverse): [`reference_data/deepxde/klein_gordon_inverse/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/deepxde/klein_gordon_inverse), written by [`tools/make_deepxde_reference_data.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/make_deepxde_reference_data.py)
- **Benchmark records:** [`benchmarks/problems/klein-gordon-1d.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/klein-gordon-1d.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-klein-gordon-equation), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md)
- **Tuning-study configurations** (records of the tuning study; open one with **Problem › Open Saved Problem...**; not presets of the Examples list): [`Klein-Gordon 1D (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Klein-Gordon%201D%20%28tuned%29.pinn.json), [`Klein-Gordon 1D, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Klein-Gordon%201D%2C%20inverse%20mode%20%28tuned%29.pinn.json)
- **DeepXDE example:** [Klein_Gordon.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Klein_Gordon.py) ([documentation](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/klein.gordon.html))
- **Cite:** [Lu et al. 2021](../../about/references.md#lu2021deepxde) (DeepXDE)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
