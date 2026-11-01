---
search:
  boost: 0.5
---

# Diffusion 1D – unknown coefficient

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial u}{\partial t} = C\frac{\partial^2 u}{\partial x^2} - e^{-t}\left(\sin(\pi x) - \pi^2\sin(\pi x)\right), \qquad x \in [-1, 1],\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - C*du_xx + exp(-t)*(sin(pi*x) - pi**2*sin(pi*x))
```

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `sin(pi*x)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `C` | 2 | 1 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/diffusion_inverse/u_obs_t1.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/diffusion_inverse/u_obs_t1.txt) → output `u`, weight 1

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 32, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 40 domain, 20 boundary, 10 initial, 10,000 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 50,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× the iterations of every training phase.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 3 × 32, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 40 domain, 20 boundary, 10 initial, 10,000 test
    - **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 150,000 | 0.001 | none |

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Diffusion 1D – unknown coefficient** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)*exp(-t)`

## Source and citation

- DeepXDE script: [diffusion_1d_inverse.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_1d_inverse.py)
- DeepXDE documentation: [diffusion.1d.inverse.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/diffusion.1d.inverse.html)

## Notes

- Unknown diffusion coefficient C (true value 1), measured u at t = 1. DeepXDE's script passes the exact solution sin(pi x) exp(-t) as the boundary value; it is 0 at x = -1 and 1.

## Configuration and benchmark records

- [`benchmarks/tuning/Diffusion 1D – unknown coefficient (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Diffusion%201D%20%E2%80%93%20unknown%20coefficient%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-inverse-diffusion-equation) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/diffusion-1d-unknown-coefficient.md) (`benchmarks/problems/diffusion-1d-unknown-coefficient.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
