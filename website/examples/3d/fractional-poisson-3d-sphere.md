# Fractional Poisson 3D – sphere

| | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 3D (x, y, z) |
| **Regime** | Steady |
| **Geometry** | Sphere |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[-(-\Delta)^{\alpha/2}u = \frac{2^{\alpha}\Gamma(2 + \frac{\alpha}{2})\Gamma(\frac{3 + \alpha}{2})}{\Gamma(\frac{3}{2})}\left(1 - (1 + \tfrac{\alpha}{3})|x|^2\right), \qquad |x| < 1,\ \alpha = 1.8\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
frac(u)*gamma((1 - alpha)/2)*gamma((3 + alpha)/2)/(2*pi**2) - 2**alpha*gamma(2 + alpha/2)*gamma((3 + alpha)/2)/gamma(3/2)*(1 - (1 + alpha/3)*(x**2 + y**2 + z**2))
```

Constants: `alpha = 1.8`

## Domain

- Sphere: centre (0, 0, 0), radius 1

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 20, inputs (x, y, z), 1 output; activation tanh, initializer Glorot normal
- **Points:** 256 domain, 0 boundary, 100 test
- **Loss weights:** `1` (equations, then boundary-condition rows)
- **Methods:** output transform `(1 - x**2 - y**2 - z**2)*u`

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 3D ──** heading, **Fractional Poisson 3D – sphere** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = abs(1 - x**2 - y**2 - z**2)**(1 + alpha/2)`

## Source and citation

- DeepXDE script: [fractional_Poisson_3d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/fractional_Poisson_3d.py)

## Notes

- 3D version: 8 x 8 directions, 100 Grunwald-Letnikov points per unit length; u = 0 on the sphere through the output transform.

## Configuration and benchmark records

- [`benchmarks/tuning/Fractional Poisson 3D – sphere (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Fractional%20Poisson%203D%20%E2%80%93%20sphere%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/fractional-poisson-3d-sphere.md) (`benchmarks/problems/fractional-poisson-3d-sphere.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
