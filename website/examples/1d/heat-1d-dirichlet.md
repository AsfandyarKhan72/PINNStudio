---
search:
  boost: 0.5
---

# Heat 1D – Dirichlet

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

\[\frac{\partial u}{\partial t} = 0.4\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - a*du_xx
```

Constants: `a = 0.4`, `L = 1`, `n = 1`

Inverse-mode form (from the benchmark records):

\[\frac{\partial u}{\partial t} = a\ \frac{\partial^2 u}{\partial x^2}\]

## Domain

- Interval: \(x \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `sin(n*pi*x/L)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `a` | 1 | 0.4 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/1D/heat/t_1.0.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/1D/heat/t_1.0.txt) → output `u`, weight 100

From the benchmark records: pick the example from the **Inverse** list: the diffusivity \(a\) is unknown (initial guess 1, true value 0.4) and is learned together with \(u\) from \(u\) measured at \(t = 1\) (the exact solution, [`reference_data/1D/heat/t_1.0.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/1D/heat/t_1.0.txt)).

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 2,540 domain, 80 boundary, 160 initial, 2,540 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 40, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
    - **Points:** 7,620 domain, 240 boundary, 480 initial, 2,540 test
    - **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | none |
    | 2 | L-BFGS | 45,000 | — | none |

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** FNN, hidden layers 3 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 2,540 domain, 80 boundary, 160 initial, 2,540 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 40, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
    - **Points:** 7,620 domain, 240 boundary, 480 initial, 2,540 test
    - **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.001 | none |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Heat 1D – Dirichlet** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Heat 1D – Dirichlet** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Forward mode: exact solution, **Error Analysis** › reference from formula: `u = exp(-n**2*pi**2*a*t/L**2)*sin(n*pi*x/L)`
- Inverse mode: exact solution, **Error Analysis** › reference from formula: `u = exp(-n**2*pi**2*0.4*t/L**2)*sin(n*pi*x/L)`

## Source and citation

- DeepXDE script: [heat.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/heat.py)
- DeepXDE documentation: [heat.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html)

## Notes

- Inverse mode (PINNStudio's addition): the diffusivity a is unknown (initial guess 1, true 0.4) and learned from u measured at t = 1 (the exact solution, reference_data/1D/heat/t_1.0.txt).

## Configuration and benchmark records

- [`benchmarks/tuning/Heat 1D – Dirichlet (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Heat%201D%20%E2%80%93%20Dirichlet%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Heat 1D – Dirichlet, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Heat%201D%20%E2%80%93%20Dirichlet%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-heat-equation) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/heat-1d-dirichlet.md) (`benchmarks/problems/heat-1d-dirichlet.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
