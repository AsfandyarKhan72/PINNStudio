---
search:
  boost: 0.5
---

# Linear ODE system

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | ODE (t) |
| **Regime** | Time-dependent |
| **Geometry** | — (t only) |
| **Outputs** | `y1`, `y2` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{dy_1}{dt} = y_2, \qquad \frac{dy_2}{dt} = -y_1, \qquad t \in [0, 10]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
dy1_t - y2
dy2_t + y1
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
dy1_t - y2
dy2_t + w*y1
```

## Domain

- Time interval: \(t \in [0, 10]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `y1` | `0` | loss term |
| `y2` | `1` | loss term |

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `w` | 2 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/ode_system_inverse/y1_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/ode_system_inverse/y1_obs.txt) → output `y1`, weight 100
- [`reference_data/deepxde/ode_system_inverse/y2_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/ode_system_inverse/y2_obs.txt) → output `y2`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 50, inputs (t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 35 domain, 2 boundary, 100 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 3 × 50, inputs (t), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 35 domain, 2 boundary, 100 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 100, inputs (t), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 105 domain, 6 boundary, 100 test
    - **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | none |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── ODE ──** heading, **Linear ODE system** › **Default**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── ODE ──** heading, **Linear ODE system** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `y1 = sin(t)`, `y2 = cos(t)`

## Source and citation

- DeepXDE script: [ode_system.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/ode_system.py)
- DeepXDE documentation: [ode.system.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.system.html)

## Notes

- Inverse mode (PINNStudio's addition): the coefficient w in y2' = -w y1 (1 in the forward equations; initial guess 2) is learned from y1 and y2 at 40 times (reference_data/deepxde/ode_system_inverse). An L-BFGS phase follows Adam.

## Configuration and benchmark records

- [`benchmarks/tuning/Linear ODE system (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Linear%20ODE%20system%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Linear ODE system, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Linear%20ODE%20system%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/linear-ode-system.md) (`benchmarks/problems/linear-ode-system.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
