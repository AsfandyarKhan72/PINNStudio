---
search:
  boost: 0.5
---

# Burgers 1D – RAR

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 1D (x, t) |
| **Regime** | Time-dependent |
| **Geometry** | Interval |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

<figure markdown="span">
  ![Collocation points of a Burgers 1D RAR run after round 2, the added points along the shock at x = 0](../../assets/images/examples/burgers-1d-rar-points.png){ loading=lazy }
  <figcaption>Figure from the repository's figures (`assets/Images/1D-Burgers/`): a Burgers 1D run with RAR, round 2 — original points, points added in earlier rounds and points added in this round. No record states which example's settings the run used.</figcaption>
</figure>

<figure markdown="span">
  ![PINN solution, reference solution and error over (t, x) of a Burgers 1D RAR run after round 2](../../assets/images/examples/burgers-1d-rar-solution.png){ loading=lazy }
  <figcaption>Figure from the repository's figures (`assets/Images/1D-Burgers/`): the same Burgers 1D RAR run, PINN, reference and error after round 2.</figcaption>
</figure>

## Governing equations

Mathematical form (from the benchmark records):

\[\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \frac{0.01}{\pi}\frac{\partial^2 u}{\partial x^2}, \qquad x \in [-1, 1],\ t \in [0, 1]\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
du_t + u*du_x - 0.01/pi*du_xx
```

## Domain

- Interval: \(x \in [-1, 1]\)
- Time interval: \(t \in [0, 1]\)

## Initial conditions

| Output | Initial value | How it is imposed (Settings: Default) |
|---|---|---|
| `u` | `-sin(pi*x)` | loss term |

## Boundary conditions

| Type | Where | Output | Value | Details |
|---|---|---|---|---|
| Dirichlet BC | `True (the whole boundary)` | `u` | `0` | — |

## Settings

**Settings: Default** — DeepXDE's own settings

- **Network:** FNN, hidden layers 3 × 20, inputs (x, t), 1 output; activation tanh, initializer Glorot normal
- **Points:** 2,500 domain, 100 boundary, 160 initial, 1,000 test
- **Loss weights:** `1,1,1` (equations, then boundary-condition rows, then initial conditions)
- **Methods:** [residual-based adaptive refinement (RAR, 3 rounds of 100 points)](../../reference/rar.md)

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |
| 2 | L-BFGS | 15,000 | — | none |

No separate **Tuned** preset: the **Tuned** item is greyed out for this example (its Default settings are PINNStudio's best).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 1D ──** heading, **Burgers 1D – RAR** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Reference data files: [`reference_data/1D/burgers/`](https://github.com/AsfandyarKhan72/PINNStudio/tree/main/reference_data/1D/burgers)

## Source and citation

- DeepXDE script: [Burgers_RAR.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Burgers_RAR.py)
- DeepXDE documentation: [burgers.rar.html](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.rar.html)
- Source paper (from the benchmark records): [Lu, Meng, Mao & Karniadakis (2021), SIAM Rev. 63 (DeepXDE)](https://arxiv.org/abs/1907.04502) — DeepXDE's example follows the equation and conditions studied there, with its own numerical settings.

## Notes

- Time runs from 0 to 1 (DeepXDE's script stops at 0.99, where its Burgers.npz data ends); Error Analysis compares with the exact (Cole-Hopf) solution at t = 0, 0.25, 0.5, 0.75 and 1. RAR: 3 rounds, each adding the 100 points with the largest residual (out of 100,000 candidates) and retraining, with no residual tolerance. DeepXDE's script adds 1 point per round until the mean residual is below 0.005, but after the first training the mean residual is already below that (about 3e-3), so its loop would add a single point and stop -- too little to show refinement.

## Configuration and benchmark records

- [`benchmarks/tuning/Burgers 1D – RAR (tuned).pinn.json`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/tuning/Burgers%201D%20%E2%80%93%20RAR%20%28tuned%29.pinn.json) — a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)
- [Entry in the examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/examples.md#dx-burgers-equation-with-rar) (`benchmarks/examples.md`)
- [Benchmark record](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/problems/burgers-1d-rar.md) (`benchmarks/problems/burgers-1d-rar.md`)
- [Entry in the benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/README.md) (`benchmarks/README.md`)

Verified error values are reported in the [Reproducibility](../../reproducibility/benchmarks.md) section, not on this page.

[← Back to the example gallery](../index.md)
