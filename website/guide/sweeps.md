# Parameter sweeps

A **parameter sweep** runs the same problem several times, changing one or more settings — a learning
rate, a network size, a PDE coefficient, a loss weight, a training method — and saves every run, so the
results can be compared. It can also search for good settings with Bayesian optimization. Everything is
set on the **Parameter sweep** page; the sweep runs with the main **Solve** button.

![The Parameter sweep page with Enable Parameter Sweep ticked: Mode One at a time, Also run the current settings (baseline) unticked, Parameter 1 set to Phase 1: Learning rate with the list of values 0.001, 0.0005, 0.0001, and Add Parameter; the toolbar button reads Run Sweep](../assets/screenshots/guide_sweep_row.png){ .screenshot loading=lazy }

Background: [Sweeps and Bayesian optimization](../reference/sweeps.md),
[Bayesian optimization](../theory/bayesian.md).

## Set up a sweep

1. Set the problem up as for a single run; this is the *base* setup every run starts from.
2. Tick **Enable Parameter Sweep**. The **Solve** button becomes **Run Sweep**.
3. Choose the **Mode:** (below).
4. For each setting to vary, use a **Parameter N** block (**Add Parameter** adds one, ✕ removes one):
    - **Parameter:** — the setting, listed by category (for example *Network: Hidden layers*,
      *Phase 1: Learning rate*, *Constants: nu*).
    - **Values:** — **List** (comma-separated values), or **Linear** / **Log** (**Steps** values spread
      evenly, or evenly on a log scale, from **Min** to **Max**). A setting with fixed choices (an
      activation, an optimizer, on/off) takes a list of its choices, which the box shows.
5. Optionally tick **Also run the current settings (baseline)**.
6. Click **Run Sweep**.

The **Parameter** list is built from the current setup each time you open it, so it only offers
settings that exist: phase settings for the phases you have, RAR settings only with RAR on, inverse
settings only for an inverse problem. A row whose setting no longer exists is marked
*(not available now)*.

## Modes

| **Mode:** | Runs |
|---|---|
| **One at a time** | One run per value of each parameter in turn, the other parameters at their base values |
| **All combinations** | Every value of every parameter, crossed: N parameters with k values each make kᴺ runs |
| **Paired values** | Parameter 1's first value with parameter 2's first value, and so on; every parameter needs the same number of values |
| **Bayesian search** | Each run is chosen from the results so far by a Gaussian-process model, within each parameter's **Min** and **Max** (a list gives its choices); **Steps** is not used |

**Bayesian search** has three more settings: **Runs:** (the total number of runs, baseline included),
**Random first:** (runs with random settings before the model takes over) and **Minimize:** —
**Final loss**, **Final loss (unweighted)** (each loss term divided by its weight; comparable when loss
weights are searched) or **L2 error** (needs [Error Analysis](error-analysis.md) to be set up).

**Also run the current settings (baseline)** adds a first run with the base settings. Until you click it,
it follows the mode: on for **One at a time** with two or more parameters and for **Bayesian search**,
off otherwise.

## What can be swept

| Category | Settings |
|---|---|
| Training | Random seed; L-BFGS phase at the end (its iterations; 0 removes it) |
| Phase N | Learning rate (Adam/AdamW phases), Iterations, Optimizer |
| Network | Hidden layers, Neurons per layer, Activation, Kernel initializer, Weight decay, Architecture (FNN, PFNN, MsFFN, STMsFFN, as the network allows) |
| Collocation Points | Domain, Boundary, Initial and Test points, Point distribution |
| Loss Weights | Each loss term's weight (when all phases share the weights) |
| Constants | Each named constant of the Equations page |
| Training Methods | Per-point weights (None, RBA, SA-PINN), Causal training, Exact periodicity, Exact initial condition, Time-adaptive training |
| Inverse | Each parameter's initial guess, each data file's loss weight, the measured data's grid points per axis or random points, Measured data: times |
| RAR | The RAR settings, when RAR is on |
| Time Adaptive | Each step group's steps, IC grid resolution, when time-adaptive training is on |
| Input Transform, Output Transform | Each scale, when the transform is on |

Each is offered only where the window itself allows it. Constants defined from a swept constant follow it, and
every expression that uses it gets the swept value.

## Run it and read the results

**Run Sweep** first checks the base settings like **Solve**. The runs are then trained one after another. Every
run's settings are checked before it trains, as Solve does: a combination that cannot run is skipped with the
reason in the Training Log and marked *invalid* in the summary.
The Training Log says where the sweep is saved, then for each run *Run i/n: …*, its output (prefixed
*[Sweep i/n]*), and its final loss and relative L₂ error. **Stop** ends the sweep. At the end the Overview
shows the last finished run's figures.

The sweep is saved in the **Save to** folder as `<example>_sweep__<date>_<time>/`, with one subfolder per
run (`run_01__…`, named after the values it set; `…__baseline` for the baseline) and:

- `sweep_summary.csv` — one row per run: its values, status, final loss and error;
- `sweep_manifest.json` — the sweep's settings and every run's result;
- `sweep_error_vs_parameter.png` — the error and final loss against the swept values.

Each run folder is a normal run folder, so the runs also appear in the **History** tab of the Results
area, where **Compare** plots several of them together.

## Example: network size for Burgers 1D

A sweep like Table 1 of [Raissi et al. (2019)](../about/references.md#raissi2019): load **Burgers 1D – Raissi et al.**, set **Mode:** to **All
combinations**, and add two parameters, *Network: Hidden layers* with the values 2, 4, 6, 8 and *Network:
Neurons per layer* with 10, 20, 40. That makes 12 runs, each a full training run with the example's other
settings. `sweep_error_vs_parameter.png` then plots the error against the number of layers, one line per width,
and selecting the runs of one width in **History** and clicking **Compare** shows their loss curves together.

In the recorded run of this sweep two hidden layers were too few for the shock at \( x = 0 \), the error dropped
by more than an order of magnitude from two to four layers, and the widest networks were the most accurate at
every depth. The record, with the table of errors and the figures, is
[`benchmarks/network_size_sweep.md`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/benchmarks/network_size_sweep.md).
It was made with the example's settings of that time (Adam and L-BFGS, 8,000 points); the example's
**Settings: Default** is now the paper's method (L-BFGS only, 10,000 points), so a sweep today gives other
numbers. Tutorial [13](../tutorials/13-compare.md) runs a shorter sweep step by step.

## Settings on this page

| GUI label | Meaning | Default | Saved as |
|---|---|---|---|
| **Enable Parameter Sweep** | Solve runs the sweep | off | `sweep_enabled` |
| **Mode:** | One at a time, All combinations, Paired values, Bayesian search | One at a time | `sweep_mode` (`oat`, `grid`, `zip`, `bayes`) |
| **Also run the current settings (baseline)** | Baseline run first | automatic | `sweep_include_baseline` (`auto`, `on`, `off`) |
| **Runs:** | Bayesian: total runs (2–1000) | 20 | `sweep_bayes_calls` |
| **Random first:** | Bayesian: random runs first (1–100) | 5 | `sweep_bayes_initial` |
| **Minimize:** | Final loss, Final loss (unweighted), L2 error | Final loss | `sweep_bayes_objective` (`loss`, `loss_unweighted`, `l2`) |
| **Parameter:**, **Values:**, **Min**, **Max**, **Steps** | One swept setting | one empty row; Steps 5 (2–100) | `sweep_parameters` (JSON list of `id`, `mode`, `values` or `min`, `max`, `n`) |
