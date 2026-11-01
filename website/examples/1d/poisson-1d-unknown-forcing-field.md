---
search:
  boost: 0.5
---

# Poisson 1D – unknown forcing field

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u`, `q` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[-\frac{d^2u}{dx^2} + q(x) = 0, \qquad x \in [-1, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx + q
```

## Domain

- Interval: \(x \in [-1, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |
| Point Set BC | — | `u` | (values from the file) | points from `reference_data/deepxde/elliptic_inverse_field/u_obs.txt` |

## Unknowns and measured data (Inverse)

This example learns an unknown **field** as an extra network output (it has no scalar trainable parameters), from measurements given as a Point Set boundary condition; it runs as a Forward problem, as DeepXDE's script does.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** parallel FNN (PFNN), hidden layers 3 × 20, inputs (x), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 200 domain, 2 boundary, 1,000 test
- **Loss weights:** `1,100,1000` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.0001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** parallel FNN (PFNN), hidden layers 4 × 40, inputs (x), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 600 domain, 6 boundary, 1,000 test
    - **Loss weights:** `1,100,1000` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 60,000 | 0.0001 | none |
    | 2 | L-BFGS | 15,000 | — | none |

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Poisson 1D – unknown forcing field** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)`, `q = -pi**2*sin(pi*x)`

## Source and citation

- DeepXDE script: [elliptic_inverse_field.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field.py)
- DeepXDE documentation: [elliptic.inverse.field.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse/elliptic.inverse.field.html)

## Notes

- The unknown is a field, q(x) (true: -pi^2 sin(pi x)), learned as a second network output from 100 measurements of u; there are no unknown constants, so it runs in Forward mode with the measurements as a Point Set BC. The boundary value sin(pi x) in DeepXDE's script is 0 at x = -1 and 1.

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 1D – unknown forcing field (batches) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20unknown%20forcing%20field%20%28batches%29%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Poisson 1D – unknown forcing field (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20unknown%20forcing%20field%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-inverse-poisson-with-unknown-forcing-field) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-1d-unknown-forcing-field.md) (`benchmarks/problems/poisson-1d-unknown-forcing-field.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
