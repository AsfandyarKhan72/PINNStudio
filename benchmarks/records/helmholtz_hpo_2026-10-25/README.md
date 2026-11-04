# Helmholtz 2D – hyperparameter search: the benchmark record

The record behind the value 1.3 × 10⁻³ reported for *Helmholtz 2D – hyperparameter search* (benchmark pages,
Supplementary Table S2). The example is adapted from DeepXDE's `Helmholtz_Dirichlet_2d_HPO.py` (Escapil-Inchauspé &
Ruz, *Neurocomputing* 561 (2023) 126826). In PINNStudio it is a Parameter Sweep in Bayesian mode: PINNStudio's own
Gaussian-process optimizer with Expected Improvement (`pinnstudio/core/bayes_opt.py`), the same sweep that is
available for every forward and inverse problem.

## How this result differs from the other benchmarks

**The configuration is selected by training loss, not by the reference solution.** The search evaluates 50
training configurations and selects the one with the lowest **final training loss** (`sweep_bayes_objective="loss"`
in the example's definition, `pinnstudio/core/deepxde_examples.py`). Only after that is the selected run's relative
L₂ error evaluated against the reference solution (the exact solution sin(k₀x) sin(k₀y), written by Error Analysis
at the evaluation points). For every other benchmark, the configuration is selected from the reference errors
(`benchmarks/tables/README.md`, *Selection*).

## The run

- **Benchmark run:** the seed-2026 run of the benchmark study, 2026-10-25 19:19 to 2026-10-26 02:04 (wall clock on the
  shared GPU machine, NVIDIA GeForce RTX 4090), workstation job `a0e_base`.
- **Seeds:** the training seed is 2026 (`random_seed`, the PINNStudio default, also set by the benchmark runner for
  `base_s2026`). The search's own seed is 0 (`sweep_bayes_seed`, default).
- **Search:** 50 evaluations. The baseline (learning rate 10⁻³, 4 hidden layers of 50, sin) comes first, then 9
  random points (10 before the Gaussian-process model, the baseline counted), then 40 points chosen by Expected
  Improvement. Each evaluation is 10,000 Adam iterations. (DeepXDE's script selects by the lowest test loss;
  PINNStudio's example selects by the final training loss.)
- **Search space:** learning rate 10⁻⁴ to 5 × 10⁻² (log), 1 to 10 hidden layers, 5 to 500 neurons per layer, activation
  sin / sigmoid / tanh.
- **Source commit:** not written into this job's folder. The same search was run in six earlier workstation jobs
  between 2026-10-12 and 2026-10-20, on the code of each date, and all seven `bayes_search.csv` files are byte for
  byte identical (MD5 `13d3d0c0d7a0…`), down to the same selected run, loss and error.

## Result

| | Run | Configuration | Final training loss | Relative L₂ |
|---|---|---|---|---|
| Baseline (the example's starting point) | 0 | lr 10⁻³, 4 × 50, sin | 5.6561 | 3.31 × 10⁻¹ |
| **Selected (lowest final training loss)** | **49** | **lr 1.499 × 10⁻³, 3 × 87, sin** | **0.020182** | **1.281 × 10⁻³** |
| For comparison: lowest L₂ of the 50 (not used for selection) | 25 | lr 1.868 × 10⁻³, 3 × 172, sin | 0.038243 | 1.272 × 10⁻³ |

The reported value, 1.3 × 10⁻³, is the selected run's 1.281 × 10⁻³ at two significant digits. The lowest error among
the 50 runs, which the selection rule does not look at, is almost the same (1.272 × 10⁻³).

**Why the per-seed table has no row for it.** `benchmarks/tables/forward_per_seed.csv` is built from the benchmark
runner's records. For this example the runner's record (`records/corrected_2026-10-31/a0e.jsonl`, variant `base`,
seed 2026) ended with `rc = 1` after 1.9 s and no error: the runner lost track of the sweep, which then ran to
completion in its own process. This folder is the record of that completed sweep.

## Files

| File | Contents |
|---|---|
| `bayes_search.csv` | the sweep's own table: every evaluation, its configuration, final loss and the best so far |
| `runs_summary.csv` | the same 50 runs with each run's relative L₂ (from its `error_metrics.txt`) and folder name |
| `bayes_convergence.png` | the sweep's convergence plot |
| `runs/run_NN_*/` | for every run: `model_config.json` (the network), `training_log.txt` (the loss), `error_metrics.txt` (the relative L₂ at the reference points) |
| `best_run_49_surface_comparison.png`, `baseline_run_00_surface_comparison.png` | PINN, reference and error of the selected run and of the baseline |
| `runner_log.txt` | the benchmark runner's own log of the job (it ends where the runner lost track of the sweep) |

The trained models (`.pt`, 165 MB for the 50 runs) stay on the GPU machine, in
`.gpu_runner/done/a0e_base/out/Helmholtz_2D_hyperparameter_optimization/base_s2026/` (ask the authors).

## Reproducing it

In PINNStudio: Examples → Forward → *Helmholtz 2D – hyperparameter search* (Settings: Default; the example has no
separate Tuned preset), then **Run Sweep**. The Parameter Sweep runs the 50 evaluations and writes `bayes_search.csv`,
`bayes_convergence.png` and one folder per run; the best run is printed at the end of the Training Log. With the
default seeds (training 2026, search 0) on the GPU machine, every run of this search so far has given the table above.
