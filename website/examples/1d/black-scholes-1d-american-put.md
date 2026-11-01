---
search:
  boost: 0.5
---

# Black-Scholes 1D – American put

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

## Governing equations

Mathematical form (from the README):

\[\min\left(-\mathcal{L}V,\ V - \max(K - S, 0)\right) = 0, \qquad \mathcal{L}V = -\frac{\partial V}{\partial \tau} + \frac{\sigma^2}{2}S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
relu(-du_t + sigma**2/2*x**2*du_xx + r*x*du_x - r*u)
relu(relu(K - x) - u)
sqrt((u - relu(K - x))**2 + (du_t - sigma**2/2*x**2*du_xx - r*x*du_x + r*u)**2 + 1e-8) - ((u - relu(K - x)) + (du_t - sigma**2/2*x**2*du_xx - r*x*du_x + r*u))
```

Constants: `r = 0.05`, `sigma = 0.2`, `K = 1`

## Domain

- Interval: \(x \in [0, 4]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `np.maximum(K - x, 0)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `left` | `u` | `1` | — |
| Dirichlet BC | `right` | `u` | `0` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 128, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 6,000 domain, 1,500 boundary, 1,500 initial, 1,000 test
- **Loss weights:** `1,10,5,100,100,100` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** collocation-point resampling every 1000 iterations

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 12,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Black-Scholes 1D – American put** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- None bundled; you can add your own files in the **Error Analysis** dialog.

## Source and citation

- DeepXDE script: [american_put.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/american_put.py)

## Notes

- Strike K, rate r and volatility sigma are constants. DeepXDE resamples points only during its L-BFGS phase; here resampling (every 1000 iterations) is on for the whole run. No closed-form reference (DeepXDE compares against a binomial-tree price).

## Configuration and benchmark records

- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/black-scholes-1d-american-put.md) (`benchmarks/problems/black-scholes-1d-american-put.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
