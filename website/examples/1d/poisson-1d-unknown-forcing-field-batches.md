# Poisson 1D – unknown forcing field (batches)

| | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u`, `q` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

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
| Point Set BC | — | `u` | (values from the file) | points from `reference_data/deepxde/elliptic_inverse_field/u_obs_10000.txt`; batch size 100 |

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

No **Tuned** choice in the GUI: the **Tuned** item is greyed out for this example. `pinnstudio/resources/tuned_settings.json` does hold a tuned setting for it (under its inverse entry), but the example loads as a Forward problem, so the Examples list does not offer it. The tuning-study configuration file is linked under *Configuration and benchmark records*.

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Poisson 1D – unknown forcing field (batches)** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)`, `q = -pi**2*sin(pi*x)`

## Source and citation

- DeepXDE script: [elliptic_inverse_field_batch.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/elliptic_inverse_field_batch.py)

## Notes

- 10,000 measurements of u, used 100 at a time (resampled every iteration). The boundary value sin(pi x) in DeepXDE's script is 0 at x = -1 and 1.

## Configuration and benchmark records

- [`benchmarks/tuning/Poisson 1D – unknown forcing field (batches) (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Poisson%201D%20%E2%80%93%20unknown%20forcing%20field%20%28batches%29%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/poisson-1d-unknown-forcing-field-batches.md) (`benchmarks/problems/poisson-1d-unknown-forcing-field-batches.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
