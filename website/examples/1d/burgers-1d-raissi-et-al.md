# Burgers 1D – Raissi et al.

| | |
|---|---|
| **Source** | PINNStudio template |
| **Problem** | Forward and Inverse |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \frac{0.01}{\pi}\frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t + 1.0*u*du_x - (0.01/pi)*du_xx
```

As typed in PINNStudio's **Equations** page (Inverse mode) (each expression is a residual, set to zero):

```text
du_t + lambda_1*u*du_x - (lambda_2/pi)*du_xx
```

Inverse-mode form (from the README):

\[\frac{\partial u}{\partial t} + \lambda_1 u\frac{\partial u}{\partial x} - \lambda_2\frac{\partial^2 u}{\partial x^2} = 0\]

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `-sin(pi*x)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `left` | `u` | `0.0` | — |
| Dirichlet BC | `right` | `u` | `0.0` | — |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `lambda_1` | 0.1 | 1 |
| `lambda_2` | 1 | 0.01 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/1D/burgers/t_1.0.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/1D/burgers/t_1.0.txt) → output `u`, weight 100

From the README: same template, switched to **Inverse**, jointly recovering both coefficients of the data-driven-discovery form of this equation in [Raissi, Perdikaris & Karniadakis (2019)](https://doi.org/10.1016/j.jcp.2018.10.045), Appendix B, eq. B.1.

## Settings

### Forward mode

**Settings: Default** — the paper's method (Raissi et al. 2019)

- **Network:** FNN, hidden layers 8 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 10,000 domain, 50 boundary, 50 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,1.0` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | L-BFGS | 50,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: Raissi et al.'s settings.* The change is applied to the template's own settings (not the paper's method that Default loads).

    - **Network:** FNN, hidden layers 8 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 10,000 domain, 50 boundary, 50 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,1.0` (equations, then boundary-condition rows, then initial conditions)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | L-BFGS | 50,000 | — | none |

### Inverse mode

**Settings: Default** — the template's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 8,000 domain, 200 boundary, 200 initial, 1,000 test
- **Loss weights:** `1.0,1.0,1.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 15,000 | 0.001 | none |
| 2 | L-BFGS | 10,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 10× the data weight, 3× the L-BFGS iterations and a larger network.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 4 × 40, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 8,000 domain, 200 boundary, 200 initial, 1,000 test
    - **Loss weights:** `1.0,1.0,1.0,100.0,100.0` (equations, then boundary-condition rows, then initial conditions, then measured data)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 15,000 | 0.001 | none |
    | 2 | L-BFGS | 30,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Burgers 1D – Raissi et al.** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Burgers 1D – Raissi et al.** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/1D/burgers/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/1D/burgers)

## Source and citation

- Paper: [Raissi et al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045)

## Configuration and benchmark records

- [`benchmarks/tuning/Burgers 1D – Raissi et al. (PINNStudio template) (best setup) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Burgers%201D%20%E2%80%93%20Raissi%20et%20al.%20%28PINNStudio%20template%29%20%28best%20setup%29%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [`benchmarks/tuning/Burgers 1D – Raissi et al. (PINNStudio template), inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Burgers%201D%20%E2%80%93%20Raissi%20et%20al.%20%28PINNStudio%20template%29%2C%20inverse%20mode%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/burgers-1d-raissi-et-al.md) (`benchmarks/problems/burgers-1d-raissi-et-al.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
