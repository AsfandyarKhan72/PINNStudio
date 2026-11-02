---
search:
  boost: 2
---

# Sweeps and Bayesian optimization

## What it does

A parameter sweep runs the same problem several times with one or more settings changed, each run in its
own folder, and summarizes the final loss, the error and (inverse problems) the parameter errors of every
run. Each run is an ordinary Solve: the sweep changes a copy of the configuration and generates the normal
script for it. Four modes are available: one at a time, all combinations (a grid), paired values, and
**Bayesian search**, in which each run is chosen by a Gaussian-process model of the results so far with
expected improvement, as in DeepXDE's Helmholtz hyperparameter-optimization example (see
[Theory: Bayesian optimization](../theory/bayesian.md)).

## When to use it

Convergence and sensitivity studies (points, network size, iterations, a PDE coefficient), repeating a
setup with other seeds, comparing training methods, and hyperparameter search.

## How to enable it

**Parameter sweep** page → **Enable Parameter Sweep**. Set **Mode:**, then one block per parameter
(**── Parameter *n* ──**: **Parameter:**, **Values:** **List** with comma-separated values, or **Linear** /
**Log** with **Min**, **Max** and **Steps**); **➕ Add Parameter** adds a block. **Solve** then runs the
sweep (the button reads **Run Sweep** while it is enabled). **About sweeps** opens a short description.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Enable Parameter Sweep** | Solve runs the sweep | off | — | `sweep_enabled` |
| **Mode:** | **One at a time**, **All combinations**, **Paired values**, **Bayesian search** | One at a time | — | `sweep_mode` (`oat`, `grid`, `zip`, `bayes`) |
| **Also run the current settings (baseline)** | A first run with the settings unchanged | on for One at a time with 2+ parameters and for Bayesian search, off otherwise (until clicked) | — | `sweep_include_baseline` (`auto`, `on`, `off`) |
| **Parameter:** | What to vary (the list below) | — | — | `sweep_parameters` (`id`) |
| **Values:** | **List**, **Linear** or **Log** | List | — | `sweep_parameters` (`mode`) |
| list | Comma-separated values, e.g. `0.001, 0.005, 0.01` | — | — | `values` |
| **Min** / **Max** / **Steps** | Range and number of values (Log needs Min, Max > 0) | — / 1 / 5 | −10¹²–10¹²; Steps 2–100 | `min`, `max`, `n` |
| **Runs:** (Bayesian) | Total runs, the baseline included | 20 | 2–1000 | `sweep_bayes_calls` |
| **Random first:** (Bayesian) | Starting runs before the model takes over (an in-space baseline counts as one; the rest are random) | 5 | 1–100 | `sweep_bayes_initial` |
| **Minimize:** (Bayesian) | **Final loss**, **Final loss (unweighted)** (each term divided by its weight), **L2 error** | Final loss | — | `sweep_bayes_objective` (`loss`, `loss_unweighted`, `l2`) |
| (no control) | Seed of the Bayesian search's random numbers | 0 | integer | `sweep_bayes_seed` |

The runs are saved under the **Save to** folder (else `~/PINNStudio_Results/parameter_sweep_results`) in
`<label>_sweep__<date>_<time>/`, one subfolder per run.

### Sweepable parameters

The **Parameter:** list (`sweep_registry.available_params`) shows, by category, what the current problem
allows:

| Category | Parameters | Offered when |
|---|---|---|
| Network | Weight decay, Hidden layers, Neurons per layer, Activation, Kernel initializer, Architecture (FNN, PFNN, MsFFN, STMsFFN) | always (architectures as the problem allows) |
| Collocation Points | Domain points, Boundary points, Initial points, Test points, Point distribution | always (Initial points: time-dependent PDEs) |
| Phase *n* | Learning rate, Iterations, Optimizer | per training phase (Learning rate: Adam/AdamW phases) |
| Loss Weights | one per term: PDE *n*, BC *n*, IC *n* | **Use same weights for all phases** on |
| Training | Random seed; L-BFGS phase at the end: iterations (0 = none) | always; the L-BFGS entry not with RAR |
| Training Methods | Per-point weights (None, RBA, SA-PINN), Causal training, Exact periodicity, Exact initial condition, Time-adaptive training (on/off) | causal: time-dependent PDEs; the others where the window allows them |
| Inverse | *name*: initial guess; Obs *n*: Data loss weight; Measured data: grid points per axis; Measured data: random points; Measured data: times | inverse problems |
| RAR | Training rounds, Residual sampling points, Points added per cycle, Adam iterations, L-BFGS iterations | Method RAR |
| Time Adaptive | Group *n* steps; IC grid resolution | Method Time Adaptive Training |
| Input Transform / Output Transform | the scale of each input / output | the transform enabled |
| Constants | every named constant of the Equations page | the problem has named constants |

A swept constant changes every expression that uses it (equations, conditions, transforms, the
exact-solution reference), and constants defined from it follow.

!!! info "New in v1.6.0 (in development)"
    - The **Constants** category: every named constant is a sweep parameter (grid and Bayesian sweeps).
    - The Training, Training Methods and **Architecture** entries, and **Measured data: times**.
    - Every run's settings are checked before it trains, as Solve does; a combination that cannot run is
      skipped with the reason in the Training Log and marked "invalid" in the summary.

## Formulation

**One at a time**: one run per value of each parameter in turn, the others at their current values.
**All combinations**: the Cartesian product (\( k^N \) runs for \( N \) parameters with \( k \) values).
**Paired values**: the \( n \)-th values of all parameters together (all need the same number of values).

**Bayesian search** (`pinnstudio/core/bayes_opt.py`, NumPy only): each parameter is a dimension mapped to
\( [0, 1] \) — a Linear or Log range (log-scaled) for a number, or its list of values as categorical
choices. After **Random first:** starting runs (an in-space baseline counts as one of them; the rest are random), the surrogate is a Gaussian
process with a Matérn 5/2 kernel on these coordinates, fitted to \( \log_{10} \) of the objective (when all
values are positive), normalized, with the length scale chosen from \( \{0.1, 0.2, 0.35, 0.6, 1\} \) by
marginal likelihood. The next point maximizes the **expected improvement**

\[
\mathrm{EI}(\mathbf{z}) = (y^\ast - \mu(\mathbf{z}) - \xi)\,\Phi(Z) + \sigma(\mathbf{z})\,\varphi(Z),
\qquad Z = \frac{y^\ast - \mu(\mathbf{z}) - \xi}{\sigma(\mathbf{z})},\qquad \xi = 0.01,
\]

(\( y^\ast \) the best normalized objective so far) over 3000 random candidates and 1000 perturbations of
the best point, skipping points already run. A failed run counts as ten times the worst value seen. The
objective is the final training loss (weighted, or unweighted), or the Error Analysis relative L2 error
(the mean over the reference times).

## Output files

In the sweep folder: `sweep_summary.csv` (every run, its values, final loss, error and parameter errors),
`sweep_error_vs_parameter.png` (error and final loss against the swept values), `sweep_manifest.json`, and
for Bayesian search `bayes_search.csv` (every run, its objective and the best so far) and
`bayes_convergence.png` (best objective against the run).

## Defaults and limitations

- **Bayesian search uses expected improvement with ξ = 0.01**; the acquisition function and ξ cannot be
  changed. **Steps** is not used in Bayesian search.
- The baseline teaches the Bayesian model only if it lies inside the search space.
- Runs are sequential; each is a full training run.

## Supported problem types

All; the parameter list adapts to the problem.

## Compatibility with other features

Refused by `validate()`:

> Parameter Sweep is enabled but no parameters have been added -- add at least one from the Parameter
> Sweep tab, or turn Parameter Sweep off to run a normal single Solve.

> Parameter Sweep entry #*n* ('…') is not a sweepable parameter for this problem's current setup … --
> remove and re-add it.

> Specified Combinations mode needs every swept parameter to have the same number of values; got … --
> adjust the value lists/ranges so they match.

> Bayesian search needs at least 2 runs (the baseline and one more). / … at least 1 random starting run.

> Bayesian search on the L2 error needs Error Analysis reference data (files or an exact-solution
> formula) -- or minimize the final loss instead.

> Parameter Sweep: 'Measured data: grid points per axis' and 'Measured data: random points' each set how
> the measured data are reduced, so they can't be swept together -- sweep one of them.

Plus the per-entry checks (at least one value, Min < Max, at least 2 steps, positive Min and Max for Log).

## Common mistakes and troubleshooting

- **A parameter is missing from the list**: it is offered only where the window allows it (e.g. loss
  weights need **Use same weights for all phases**; RAR parameters need Method RAR).
- **Comparing loss values across swept loss weights**: use **Final loss (unweighted)**.

## Example

**Helmholtz 2D – hyperparameter search** in **Examples** (a Bayesian search over the learning rate, the
network size and the activation, as in DeepXDE's HPO example). See
[Tutorial 13: Comparing configurations](../tutorials/13-compare.md),
[Tutorial 14: Bayesian hyperparameter search](../tutorials/14-bayesian.md) and the
[Parameter sweeps guide page](../guide/sweeps.md).

## Related

- **Tutorials:** [13. Comparing training configurations](../tutorials/13-compare.md), [14. Bayesian hyperparameter search](../tutorials/14-bayesian.md)
- **Examples:** [Helmholtz 2D – hyperparameter search](../examples/2d/helmholtz-2d-hyperparameter-search.md)
- **Theory:** [Bayesian optimization](../theory/bayesian.md)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/sweeps.md"
