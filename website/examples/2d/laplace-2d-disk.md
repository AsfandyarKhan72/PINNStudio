---
search:
  boost: 0.5
---

# Laplace 2D – disk

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Rectangle in (r, θ) |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[r\frac{\partial u}{\partial r} + r^2\frac{\partial^2 u}{\partial r^2} + \frac{\partial^2 u}{\partial \theta^2} = 0, \qquad r \in [0, 1],\ \theta \in [0, 2\pi]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
x*du_x + x**2*du_xx + du_yy
```

## Domain

- Rectangle in polar coordinates (x is r, y is θ): \(x \in [0, 1]\), \(y \in [0, 6.28319]\)

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `right` | `u` | `cos(y)` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x, y), 1 output; activation tanh, initializer Glorot normal
- **Points:** 2,540 domain, 80 boundary, 1,000 test
- **Loss weights:** `1,1` (equations, then boundary-condition rows)
- **Methods:** input features `x*sin(y); x*cos(y)` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 15,000 | 0.001 | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Laplace 2D – disk** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = x*cos(y)`

## Source and citation

- DeepXDE script: [Laplace_disk.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Laplace_disk.py)
- DeepXDE documentation: [laplace.disk.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/laplace.disk.html)

## Notes

- Solved in polar coordinates as DeepXDE does: x is the radius r in [0, 1] and y the angle theta in [0, 2 pi], so plots show the (r, theta) rectangle.

## Configuration and benchmark records

- [`benchmarks/tuning/Laplace 2D – disk (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Laplace%202D%20%E2%80%93%20disk%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/laplace-2d-disk.md) (`benchmarks/problems/laplace-2d-disk.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
