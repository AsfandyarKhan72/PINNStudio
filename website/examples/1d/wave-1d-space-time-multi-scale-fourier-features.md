---
search:
  boost: 0.5
---

# Wave 1D – space-time multi-scale Fourier features

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward and Inverse (the Inverse mode is PINNStudio's addition) |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default, Tuned |

## Governing equations

Mathematical form (from the README):

\[\frac{\partial^2 u}{\partial t^2} = C^2\frac{\partial^2 u}{\partial x^2}, \qquad x \in [0, 1],\ t \in [0, 1],\ C = 10\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_tt - C**2*du_xx
```

Constants: `A = 2`, `C = 10`

## Domain

- Interval: \(x \in [0, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `sin(pi*x) + sin(A*pi*x)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |
| Operator BC (advanced) | initial time | — | `du_t` | applied at the initial time |

## Unknowns and measured data (Inverse)

| Trainable parameter | Initial value | True value |
|---|---|---|
| `C` | 5 | 10 |

Measured data (**Inverse** page, measured-data files):

- [`reference_data/deepxde/wave_inverse/u_obs.txt`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/reference_data/deepxde/wave_inverse/u_obs.txt) → output `u`, weight 100

## Settings

### Forward mode

**Settings: Default** — DeepXDE's own settings

- **Network:** space-time multi-scale Fourier-feature network (STMsFFN), hidden layers 3 × 100, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform; scales 1 (space), 1, 10 (time)
- **Points:** 360 domain, 360 boundary, 360 initial, 10,000 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [multi-scale Fourier features](../../reference/networks.md); collocation-point resampling every 1 iterations; input features `(x - 0.5)*2*sqrt(3); (t - 0.5)*2*sqrt(3)` ([reference](../../reference/networks.md)); [loss weights set from the initial losses](../../reference/loss-weights.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | inverse time (2000, 0.9) |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** space-time multi-scale Fourier-feature network (STMsFFN), hidden layers 4 × 200, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform; scales 1 (space), 1, 10 (time)
    - **Points:** 1,080 domain, 1,080 boundary, 1,080 initial, 10,000 test
    - **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [multi-scale Fourier features](../../reference/networks.md); collocation-point resampling every 1 iterations; input features `(x - 0.5)*2*sqrt(3); (t - 0.5)*2*sqrt(3)` ([reference](../../reference/networks.md)); [loss weights set from the initial losses](../../reference/loss-weights.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 30,000 | 0.001 | inverse time (2000, 0.9) |
    | 2 | L-BFGS | 15,000 | — | none |

### Inverse mode

**Settings: Default** — DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)

- **Network:** space-time multi-scale Fourier-feature network (STMsFFN), hidden layers 3 × 100, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform; scales 1 (space), 1, 10 (time)
- **Points:** 5,000 domain, 360 boundary, 360 initial, 10,000 test
- **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [multi-scale Fourier features](../../reference/networks.md); collocation-point resampling every 1 iterations; input features `(x - 0.5)*2*sqrt(3); (t - 0.5)*2*sqrt(3)` ([reference](../../reference/networks.md)); [loss weights set from the initial losses](../../reference/loss-weights.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | inverse time (2000, 0.9) |
| 2 | L-BFGS | 15,000 | — | none |

??? note "Settings: Tuned"

    The Training Log reports: *Tuned settings: 3× iterations, a larger network, 3× points and an L-BFGS phase together.* The change is applied to the example's own settings.

    - **Network:** space-time multi-scale Fourier-feature network (STMsFFN), hidden layers 4 × 200, inputs (x, t), 1 output; activation tanh, initializer Glorot uniform; scales 1 (space), 1, 10 (time)
    - **Points:** 15,000 domain, 1,080 boundary, 1,080 initial, 10,000 test
    - **Loss weights:** `1,1,1,1` (equations, then boundary-condition rows, then initial conditions)
    - **Methods:** [multi-scale Fourier features](../../reference/networks.md); collocation-point resampling every 1 iterations; input features `(x - 0.5)*2*sqrt(3); (t - 0.5)*2*sqrt(3)` ([reference](../../reference/networks.md)); [loss weights set from the initial losses](../../reference/loss-weights.md)

    | Phase | Optimizer | Iterations | Learning rate | Decay |
    |---|---|---|---|---|
    | 1 | Adam | 30,000 | 0.001 | inverse time (2000, 0.9) |
    | 2 | L-BFGS | 45,000 | — | none |

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Wave 1D – space-time multi-scale Fourier features** › **Default** or **Tuned**, then **Solve**.
- Inverse: toolbar **Examples:** › **Inverse** › under the **── 1D ──** heading, **Wave 1D – space-time multi-scale Fourier features** › **Default** or **Tuned**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Forward mode: exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)*cos(C*pi*t) + sin(A*pi*x)*cos(A*C*pi*t)`
- Inverse mode: exact solution, **Error Analysis** › reference from formula: `u = sin(pi*x)*cos(10*pi*t) + sin(A*pi*x)*cos(A*10*pi*t)`

## Source and citation

- DeepXDE script: [wave_1d.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py)
- Source paper (from the README): [Wang, Wang & Perdikaris (2021), CMAME 384, Sec. 4.3](https://arxiv.org/abs/2012.10047) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- Space-time multi-scale Fourier feature network (space scale 1, time scales 1 and 10) for a wave that oscillates fast in time. Loss weights are set from the initial losses (5 / L_i(0)), as in DeepXDE's script; the inputs are rescaled to zero mean and unit variance by an input-transform formula; collocation points are redrawn every iteration -- pseudo-random here: DeepXDE's script keeps the default Hammersley distribution, which is deterministic, so its resampler redraws the same points and the network overfits them (test loss stays near 1). DeepXDE's script passes the exact solution as the boundary value; it is 0 at x = 0 and 1.
- Inverse mode (PINNStudio's addition): the wave speed C (initial guess 5, true 10) is learned from u measured at 2,000 random points in space and time (reference_data/deepxde/wave_inverse). At regular times the fast mode cos(20 pi t) is sampled in step with its period, so a slower wave fits the same data (at t = 0.25, 0.5, 0.75 and 1, speeds 10 and 6 give the same u). 5,000 residual points instead of the forward example's 360. An L-BFGS phase follows Adam.

## Configuration and benchmark records

- [`benchmarks/tuning/Wave 1D – space-time multi-scale Fourier features (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Wave%201D%20%E2%80%93%20space-time%20multi-scale%20Fourier%20features%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [`benchmarks/tuning/Wave 1D – space-time multi-scale Fourier features, inverse mode (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Wave%201D%20%E2%80%93%20space-time%20multi-scale%20Fourier%20features%2C%20inverse%20mode%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/wave-1d-space-time-multi-scale-fourier-features.md) (`benchmarks/problems/wave-1d-space-time-multi-scale-fourier-features.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
