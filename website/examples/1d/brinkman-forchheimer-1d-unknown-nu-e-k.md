# Brinkman-Forchheimer 1D – unknown ν_e, K

| | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[-\frac{\nu_e}{\epsilon}\frac{d^2u}{dx^2} + \frac{\nu u}{K} = g, \qquad x \in [0, 1],\ \epsilon = 0.4,\ \nu = 10^{-3},\ g = 1\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-v_e/e*du_xx + v*u/K - g
```

Constants: `g = 1`, `v = 1e-3`, `e = 0.4`

## Domain

- Interval: \(x \in [0, 1]\)

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `v_e` | 0.1 | 0.001 |
| `K` | 0.1 | 0.001 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/brinkman_forchheimer/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/brinkman_forchheimer/u_obs.txt) → output `u`, weight 1

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x), 1 output; activation tanh, initializer Glorot uniform
- **Points:** 100 domain, 0 boundary, 500 test
- **Loss weights:** `1` (equations, then boundary-condition rows)
- **Methods:** output transform `x*(1 - x)*u`

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 30,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× the iterations of every training phase.* The change is applied to the example's own settings.

    - **Network:** FNN, hidden layers 3 × 20, inputs (x), 1 output; activation tanh, initializer Glorot uniform
    - **Points:** 100 domain, 0 boundary, 500 test
    - **Loss weights:** `1` (equations, then boundary-condition rows)
    - **Methods:** output transform `x*(1 - x)*u`

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 90,000 | 0.001 | none |

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Brinkman-Forchheimer 1D – unknown ν_e, K** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = 1 - cosh(20*(x - 0.5))/cosh(10)`

## Source and citation

- DeepXDE script: [brinkman_forchheimer.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/brinkman_forchheimer.py)
- Source paper (from the README): [Yu, Lu, Meng & Karniadakis (2022), CMAME 393 (gPINNs)](https://arxiv.org/abs/2111.02801) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- Unknown effective viscosity v_e and permeability K (both 1e-3), from 5 measurements; u = 0 at both walls through the output transform.

## Configuration and benchmark records

- [`benchmarks/tuning/Brinkman-Forchheimer 1D – unknown ν_e, K (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Brinkman-Forchheimer%201D%20%E2%80%93%20unknown%20%CE%BD_e%2C%20K%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/brinkman-forchheimer-1d-unknown-e-k.md) (`benchmarks/problems/brinkman-forchheimer-1d-unknown-e-k.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
