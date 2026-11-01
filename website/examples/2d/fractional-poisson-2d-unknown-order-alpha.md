---
search:
  boost: 0.5
---

# Fractional Poisson 2D – unknown order α

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Inverse (DeepXDE inverse example) |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Disk |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[-(-\Delta)^{\alpha/2}u = 2^{\alpha_0}\Gamma(2 + \tfrac{\alpha_0}{2})\Gamma(1 + \tfrac{\alpha_0}{2})\left(1 - (1 + \tfrac{\alpha_0}{2})|x|^2\right), \qquad |x| < 1\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-frac(u)*exp(lgamma((1 - alpha)/2) + lgamma((2 + alpha)/2))/(2*pi**1.5) - 2**a0*gamma(2 + a0/2)*gamma(1 + a0/2)*(1 - (1 + a0/2)*(x**2 + y**2))
```

Constants: `a0 = 1.8`

## Domain

- Disk: centre (0, 0), radius 1

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `alpha` | 1.5 | 1.8 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/fractional_inverse_2d/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/fractional_inverse_2d/u_obs.txt) → output `u`, weight 100

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 20, inputs (x, y), 1 output; activation tanh, initializer Glorot normal
- **Points:** 64 domain, 0 boundary, 100 test
- **Loss weights:** `1` (equations, then boundary-condition rows)
- **Methods:** output transform `(1 - x**2 - y**2)*u` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Inverse: toolbar **Examples:** › **Inverse** › under the **── 2D ──** heading, **Fractional Poisson 2D – unknown order α** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = abs(1 - x**2 - y**2)**(1 + a0/2)`

## Source and citation

- DeepXDE script: [fractional_Poisson_2d_inverse.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/fractional_Poisson_2d_inverse.py)

## Notes

- Unknown order alpha (true 1.8, start 1.5) of the fractional Laplacian on the unit disk, from 30 random measurements (seeded, written by tools/make_deepxde_reference_data.py). lgamma keeps the constant C(alpha, 2) differentiable in alpha, as DeepXDE's script does.

## Configuration and benchmark records

- [`benchmarks/tuning/Fractional Poisson 2D – unknown order α (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Fractional%20Poisson%202D%20%E2%80%93%20unknown%20order%20%CE%B1%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/fractional-poisson-2d-unknown-order.md) (`benchmarks/problems/fractional-poisson-2d-unknown-order.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
