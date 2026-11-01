---
search:
  boost: 0.5
---

# Euler beam 1D

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{d^4u}{dx^4} = -1, \qquad x \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_xxxx + 1
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
EI*du_xxxx + 1
```

## Domain

- Interval: \(x \in [0, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `left` | `u` | `0` | — |
| Neumann BC | `left` | `u` | `0` | — |
| Operator BC (advanced) | `right` | — | `du_xx` | — |
| Operator BC (advanced) | `right` | — | `du_xxx` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `EI` | 2 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/euler_beam_inverse/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/euler_beam_inverse/u_obs.txt) → output `u`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 10 domain, 2 boundary, 100 test
- **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 40, inputs (x), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 30 domain, 6 boundary, 100 test
    - **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 30,000 | 0.001 | none |
    | 2 | L-BFGS | 15,000 | — | none |

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 3 × 20, inputs (x), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 10 domain, 2 boundary, 100 test
- **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: a larger network (hidden layers twice as wide, one more layer).* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 40, inputs (x), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 10 domain, 2 boundary, 100 test
    - **Loss weights:** `1,1,1,1,1` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 10,000 | 0.001 | none |
    | 2 | L-BFGS | 15,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Euler beam 1D** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Euler beam 1D** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = -x**4/24 + x**3/6 - x**2/4`

## Source and citation

- DeepXDE script: [Euler_beam.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Euler_beam.py)
- DeepXDE documentation: [eulerbeam.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/eulerbeam.html)

## Notes

- Inverse mode (PINNStudio's addition): the bending stiffness EI (1 in the forward equation; initial guess 2) is learned from the deflection measured at 20 points (reference_data/deepxde/euler_beam_inverse). An L-BFGS phase follows Adam.

## Configuration and benchmark records

- [`benchmarks/tuning/Euler beam 1D (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Euler%20beam%201D%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Euler beam 1D, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Euler%20beam%201D%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/euler-beam-1d.md) (`benchmarks/problems/euler-beam-1d.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
