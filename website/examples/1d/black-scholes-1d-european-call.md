# Black-Scholes 1D – European call

| | |
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

\[\frac{\partial V}{\partial \tau} = \frac{\sigma^2}{2}S^2\frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV, \qquad S \in [0, 150],\ \tau \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t - (sigma**2/2*x**2*du_xx + r*x*du_x - r*u)
```

Constants: `K = 50`, `r = 0.05`, `sigma = 0.2`

## Domain

- Interval: \(x \in [0, 150]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `np.maximum(x - K, 0)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `left` | `u` | `0` | — |
| Dirichlet BC | `right` | `u` | `x_max - K*exp(-r*t)` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 4 × 28, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 2,047 domain, 63 boundary, 127 initial, 2,047 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 20,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Black-Scholes 1D – European call** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = np.maximum(x, 1e-10)*norm_cdf((log(np.maximum(x, 1e-10)/K) + (r + sigma**2/2)*np.maximum(t, 1e-8))/(sigma*sqrt(np.maximum(t, 1e-8)))) - K*exp(-r*np.maximum(t, 1e-8))*norm_cdf((log(np.maximum(x, 1e-10)/K) + (r + sigma**2/2)*np.maximum(t, 1e-8))/(sigma*sqrt(np.maximum(t, 1e-8))) - sigma*sqrt(np.maximum(t, 1e-8)))`

## Source and citation

- DeepXDE script: [black_scholes_call.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/black_scholes_call.py)
- DeepXDE documentation: [black_scholes.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/black_scholes.html)

## Notes

- x is the stock price S in [0, 150] and t the time to maturity tau; strike K, rate r and volatility sigma are constants.

## Configuration and benchmark records

- [`benchmarks/tuning/Black-Scholes 1D – European call (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Black-Scholes%201D%20%E2%80%93%20European%20call%20%28tuned%29.pinn.json) — problem file of a tuning-study configuration (open it with **Problem › Open Saved Problem...**)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/black-scholes-1d-european-call.md) (`benchmarks/problems/black-scholes-1d-european-call.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
