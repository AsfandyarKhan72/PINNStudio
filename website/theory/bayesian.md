# Bayesian optimization

The settings of a PINN (learning rate, network size, number of points, loss weights, ...) affect the result
strongly, and every evaluation of a setting is a full training run. The **Parameter sweep** page can run a
problem over many settings, either over a fixed plan or by Bayesian search, which chooses each new setting
from the results so far. See [Sweeps and Bayesian optimization](../reference/sweeps.md) and the
[guide page](../guide/sweeps.md).

## Parameter sweeps

A sweep varies one or more parameters \(p_1, \dots, p_P\) (**Values:** `List`, or `Linear` / `Log` with **Min**,
**Max** and **Steps**) and runs every resulting configuration as its own training
run, one after the other. **Mode:**

| Mode | Runs |
|---|---|
| **One at a time** | each parameter's values in turn, the other parameters at their current values: \(\sum_q n_q\) runs |
| **All combinations** | the Cartesian product of all value lists: \(\prod_q n_q\) runs |
| **Paired values** | the \(m\)-th value of every parameter together: \(n\) runs (all lists of equal length \(n\)) |
| **Bayesian search** | a number of runs chosen one at a time by the method below |

**Also run the current settings (baseline)** adds one run with the unchanged settings first (on by default for
One at a time with two or more parameters and for Bayesian search). Every run's settings are validated before it
starts; an impossible combination is skipped and marked invalid. A sweep is a set of independent experiments: it
reports the final loss and, when Error Analysis is set up, the error of every run, but does not judge which
differences are significant (repeat runs with other seeds for that).

## Bayesian search as implemented

The method follows DeepXDE's hyperparameter-optimization example (Helmholtz 2D), which uses scikit-optimize's
`gp_minimize` with expected improvement. PINNStudio implements it with NumPy only, in
`pinnstudio/core/bayes_opt.py`.

### Search space

Each swept parameter is one dimension, mapped to the unit interval:

- a numeric parameter with **Min**/**Max**: \(u = (v - v_{\min})/(v_{\max} - v_{\min})\), or the same in
  \(\log v\) for a logarithmic range (**Steps** is not used); integers are rounded when mapped back;
- a parameter given as a list of values (or a choice such as the activation): a categorical dimension with
  \(n\) choices, choice \(c\) at \(u = (c + \tfrac12)/n\).

### Surrogate model

After \(n_0\) = **Random first:** starting runs (an in-space baseline counts as one of them; the others are at
uniformly random points), the objective
values \(y_1, \dots, y_n\) observed at the points \(\mathbf{u}_1, \dots, \mathbf{u}_n\) are modelled by a Gaussian
process:

- the values are transformed to \(\log_{10} y\) when all are positive (losses and errors span decades), then
  standardised to zero mean and unit variance, \(\tilde y\);
- the kernel is the Matérn 5/2 kernel with unit variance,

  \[
  k(\mathbf{u}, \mathbf{u}') = \Big(1 + \sqrt{5}\,\frac{d}{\ell} + \frac{5}{3}\,\frac{d^2}{\ell^2}\Big)
  \exp\!\Big(-\sqrt{5}\,\frac{d}{\ell}\Big), \qquad d = \lVert \mathbf{u} - \mathbf{u}' \rVert_2 ,
  \]

  one length scale \(\ell\) for all dimensions, plus \(10^{-6}\) on the diagonal;
- the length scale is chosen from \(\{0.1, 0.2, 0.35, 0.6, 1.0\}\) by the largest log marginal likelihood
  \(-\tfrac12 \tilde y^{\mathsf T} K^{-1} \tilde y - \tfrac12 \log\det K\) (constant omitted).

The posterior mean and standard deviation at a candidate \(\mathbf{u}\) are
\(\mu(\mathbf{u}) = \mathbf{k}^{\mathsf T} K^{-1}\tilde y\) and
\(\sigma(\mathbf{u}) = \sqrt{\max(1 - \mathbf{k}^{\mathsf T}K^{-1}\mathbf{k},\ 10^{-12})}\).

### Acquisition

The next point maximises the expected improvement for minimisation (Jones, Schonlau & Welch, 1998),

\[
\mathrm{EI}(\mathbf{u}) = I\,\Phi(z) + \sigma(\mathbf{u})\,\varphi(z), \qquad
I = \tilde y_{\min} - \mu(\mathbf{u}) - \xi, \quad z = \frac{I}{\sigma(\mathbf{u})}, \quad \xi = 0.01,
\]

with \(\Phi\), \(\varphi\) the standard normal distribution and density. EI is maximised over 4000 candidates:
3000 uniformly random points and 1000 perturbations of the best point so far (Gaussian, standard deviation 0.05
in unit coordinates, clipped to \([0, 1]\)). The best candidate whose rounded settings have not been run yet is
chosen. The random numbers come from **sweep_bayes_seed** (stored in the configuration, default 0).

### Objective

**Minimize:** chooses the value \(y\) of a finished run:

| Choice | \(y\) |
|---|---|
| **Final loss** | the last recorded total training loss \(\sum_k w_k \mathcal{L}_k\) |
| **Final loss (unweighted)** | \(\sum_k \mathcal{L}_k\), comparable between runs with different loss weights |
| **L2 error** | the mean relative \(L_2\) error over the reference times ([Error measures](metrics.md)); needs Error Analysis reference data |

A failed run is recorded as ten times the worst value seen so far (or \(10^5\)), so the search moves away from it.
The baseline run teaches the model only if its settings lie inside the search space. The search ends after
**Runs:** runs in total (the baseline included); the best run, all runs (`bayes_search.csv`) and the best value
against the run number (`bayes_convergence.png`) are written to the sweep's folder.

### Differences from `gp_minimize`

PINNStudio's optimizer is a small re-implementation, not scikit-optimize: the kernel amplitude is fixed, the
single length scale is chosen from five values instead of being optimised per dimension, there is no noise
model beyond the small diagonal term, and the acquisition is maximised over random candidates rather than by
a gradient-based optimiser. Categorical choices are placed on the unit interval in their list order, so the
model assumes that neighbouring choices behave alike.

!!! note
    Training is stochastic. Two runs with the same settings and a different seed can differ more than two
    settings do. Minimising the final loss is not the same as minimising the error; when a reference solution
    exists, the **L2 error** objective measures what matters, but it then tunes the settings on that reference.

## Implementation and source code

The Gaussian-process search above is `BayesOptimizer` in `pinnstudio/core/bayes_opt.py`, driven run by run from
the sweep runner:

--8<-- "_snippets/source/sweeps.md"
