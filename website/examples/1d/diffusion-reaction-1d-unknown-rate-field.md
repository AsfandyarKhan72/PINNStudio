---
search:
  boost: 0.5
---

# Diffusion-reaction 1D – unknown rate field

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 1D (x) |
| **Regime** | Steady |
| **Geometry** | Interval |
| **Outputs** | `u`, `k` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the benchmark records):

\[\lambda\frac{d^2u}{dx^2} - k(x)\ u = \sin(2\pi x), \qquad \lambda = 0.01,\ x \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
l*du_xx - u*k - sin(2*pi*x)
```

Constants: `l = 0.01`

## Domain

- Interval: \(x \in [0, 1]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |
| Point Set BC | — | `u` | (values from the file) | points from `reference_data/deepxde/diffusion_reaction_rate/u_obs.txt` |

## Unknowns and measured data (Inverse)

This example learns an unknown **field** as an extra network output (it has no scalar trainable parameters), from measurements given as a Point Set boundary condition; it runs as a Forward problem, as DeepXDE's script does.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** parallel FNN (PFNN), hidden layers 2 × 20, inputs (x), 2 outputs; activation tanh, initializer Glorot uniform
- **Points:** 50 domain, 8 boundary, 1,000 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: a larger network (hidden layers twice as wide, one more layer).* The change is applied to the example's own settings.

    - **Network:** parallel FNN (PFNN), hidden layers 3 × 40, inputs (x), 2 outputs; activation tanh, initializer Glorot uniform
    - **Points:** 50 domain, 8 boundary, 1,000 test
    - **Loss weights:** `1,1,1` (equations, then boundary-condition rows)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 20,000 | 0.001 | none |

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Diffusion-reaction 1D – unknown rate field** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = `, `k = 0.1 + exp(-0.5*(x - 0.5)**2/0.15**2)`

## Source and citation

- DeepXDE script: [diffusion_reaction_rate.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/diffusion_reaction_rate.py)
- Source paper (from the benchmark records): [Yu, Lu, Meng & Karniadakis (2022), CMAME 393 (gPINNs)](https://arxiv.org/abs/2111.02801) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- The unknown reaction rate k(x) (true: 0.1 + exp(-0.5 (x - 0.5)^2 / 0.15^2)) is a second network output, learned from 100 measurements of u (computed with scipy's solve_bvp, as DeepXDE does).

## Configuration and benchmark records

- [`benchmarks/tuning/Diffusion-reaction 1D – unknown rate field (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Diffusion-reaction%201D%20%E2%80%93%20unknown%20rate%20field%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-inverse-diffusion-reaction-with-space-dependent-rate) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/diffusion-reaction-1d-unknown-rate-field.md) (`benchmarks/problems/diffusion-reaction-1d-unknown-rate-field.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
