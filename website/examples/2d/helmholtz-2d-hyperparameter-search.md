---
search:
  boost: 0.5
---

# Helmholtz 2D – hyperparameter search

| Example | |
|---|---|
| **Source** | DeepXDE example |
| **Problem** | Forward |
| **Dimension** | 2D (x, y) |
| **Regime** | Steady |
| **Geometry** | Rectangle |
| **Outputs** | `u` |
| **Settings** | Default (no separate Tuned preset) |

<figure markdown="span">
  ![Heat maps of the PINN solution, the reference solution and the error of the 2D Helmholtz problem](../../assets/images/examples/helmholtz-2d-hpo-surface.png){ loading=lazy }
  <figcaption>The search's starting configuration (run 0: learning rate 10⁻³, 4 × 50, sin; relative L₂ 3.3 × 10⁻¹), before the 50-run Bayesian search — PINN, reference and error. The selected run (lowest final training loss) is under *Results*.</figcaption>
</figure>

## Governing equations

Mathematical form (from the benchmark records):

\[-\nabla^2 u - k_0^2 u = k_0^2\sin(k_0 x)\sin(k_0 y), \qquad k_0 = 4\pi,\ (x, y) \in [0, 1]^2\]

As typed in PINNStudio's **Equations** page (each expression is a residual, set to zero):

```text
-du_xx - du_yy - k0**2*u - k0**2*sin(k0*x)*sin(k0*y)
```

Constants: `n = 2`, `k0 = 2*pi*n`

## Domain

- Rectangle: \(x \in [0, 1]\), \(y \in [0, 1]\)

## Boundary conditions

No boundary-condition rows: the boundary values are imposed exactly through the network's output transform (see **Methods** below), or the problem has no boundary terms.

## Settings

**Settings: Default** — a 50-run Bayesian search (Parameter Sweep), the configuration selected by the lowest final training loss; each run starts from DeepXDE's network and training below

- **Network:** FNN, hidden layers 4 × 50, inputs (x, y), 1 output; activation sin, initializer Glorot uniform
- **Points:** 400 domain, 80 boundary, 3,600 test
- **Loss weights:** `1` (equations, then boundary-condition rows)
- **Methods:** output transform `x*(1 - x)*y*(1 - y)*u` ([reference](../../reference/networks.md))

| Phase | Optimizer | Iterations | Learning rate | Decay |
|---|---|---|---|---|
| 1 | Adam | 10,000 | 0.001 | none |

No **Tuned** preset: the example is its own search, and its configuration is chosen by the lowest final training loss, not by the error against the reference.

## Results

The configuration selected in PINNStudio's benchmark study, run on one RTX 4090. Forward: relative L₂ error of each output at the last reference time. Inverse: each parameter's relative error and, per seed, the largest of them (worst parameter error).

### Forward problem

**Settings:** Default — the example's Bayesian search, 50 runs; selected: the run with the lowest final training loss (run 49); the example has no separate Tuned preset for this mode. **Values:** one run, seed 2026, selected by its final training loss (not the reference error); the search's 50 runs are in [its record](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/records/helmholtz_hpo_2026-10-25/README.md).

| Output | Relative L₂ |
|---|---|
| u | 1.3 × 10⁻³ |

<figure markdown="span">
  ![Forward result: PINN, reference and error](https://raw.githubusercontent.com/AsfandyarKhan72/PINNStudio/v1.6.0/benchmarks/records/helmholtz_hpo_2026-10-25/best_run_49_surface_comparison.png){ loading=lazy }
  <figcaption>Forward, Default settings: PINN, reference and error.</figcaption>
</figure>

**Full benchmark record:** every setting tried, the per-seed values and all figures are in [the benchmark records on GitHub](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/examples.md#dx-helmholtz-2d-hyperparameter-optimization).

## Loading it in PINNStudio

- Forward: toolbar **Examples:** › **Forward** › under the **── 2D ──** heading, **Helmholtz 2D – hyperparameter search** › **Default**, then **Solve**.

Loading an example fills in every setting; anything you change before **Solve** is listed in the Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.

## Reference solution

- Exact solution, **Error Analysis** › reference from formula: `u = sin(k0*x)*sin(k0*y)`

## Notes

- Bayesian search over learning rate (1e-4 to 5e-2, log scale), hidden layers (1 to 10), neurons per layer (5 to 500) and activation (sin, sigmoid, tanh): the baseline (1e-3, 4 x 50, sin) and 9 random runs (10 before the model, the baseline counted), then runs chosen by a Gaussian-process model, 50 in all (Parameter Sweep, mode Bayesian search). Each run minimizes the final training loss (DeepXDE's script uses the lowest test loss). Click Run Sweep.

## Definition and records

- **Definition:** `EXAMPLES` entry "Helmholtz 2D – hyperparameter optimization" in [`pinnstudio/core/deepxde_examples.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/deepxde_examples.py)
- **Equations:** [Governing equations](#governing-equations) on this page
- **Reference data:** the exact-solution formula under *Reference solution* (no data files)
- **Benchmark records:** [`benchmarks/problems/helmholtz-2d-hyperparameter-search.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/problems/helmholtz-2d-hyperparameter-search.md), its entry in the [examples overview](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/examples.md#dx-helmholtz-2d-hyperparameter-optimization), the [benchmark collection](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/benchmarks/README.md)
- **Methods** (Default and Tuned settings): [Networks and Fourier features](../../reference/networks.md)
- **DeepXDE example:** [Helmholtz_Dirichlet_2d_HPO.py](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Helmholtz_Dirichlet_2d_HPO.py)
- **Cite:** [Lu et al. 2021](../../about/references.md#lu2021deepxde) (DeepXDE); the problem follows [Escapil-Inchauspé & Ruz 2023](../../about/references.md#escapil2023)

How the benchmark values were measured and checked: [Reproducibility](../../reproducibility/benchmarks.md).

[← Back to the example gallery](../index.md)
