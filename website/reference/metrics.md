# Error metrics

## What it does

**Error Analysis** compares the trained network with a reference solution — reference files, or an exact
solution formula — at the reference points, and reports the relative \( L_2 \) error, the mean squared
error, the maximum and the mean absolute error for each reference time, plus the relative \( L_2 \) error
over all reference times together. For inverse problems with a known **True value**, the error of each
identified parameter is reported in sweep summaries. See [Theory: error measures](../theory/metrics.md).

## When to use it

Whenever a reference solution is available: the training loss alone does not show how accurate the
solution is.

## How to enable it

**Results** page → **Error Analysis** opens the dialog: **Plot Types** (**Line comparison (PINN vs
reference at each time)**, **Surface comparison (PINN | Reference | Error)**), **Error Metrics to Compute**
(**L2 Relative**, **MSE**, **Max Error**), **Reference:** (**Reference files** with a list of files, or
**Exact solution formula** with one formula per output, **Compare at times:** and **Grid:**). A loaded
example sets its own reference. Error Analysis then runs at the end of every Solve.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **L2 Relative** | Compute and show the relative L2 error | on | — | `ea_metric_l2` |
| **MSE** | Compute and show the MSE | off | — | `ea_metric_mse` |
| **Max Error** | Compute and show the maximum absolute error | off | — | `ea_metric_max` |
| **Line comparison …** / **Surface comparison …** | Comparison figures | on / on | — | `ea_do_line`, `ea_do_surface` |
| **Reference:** | Reference files or Exact solution formula | Reference files | — | `ea_ref_mode` (`files`, `formula`) |
| reference files | List of (time, file[, output]) | none | — | `ea_files` |
| formula rows (*name* =) | Exact solution per output in `x, y, z, t` (numpy names, `erf`, `norm_cdf`) | blank | — | `ea_ref_formula` (`|`-separated) |
| **Compare at times:** | Comparison times | blank = 5 evenly spaced over \( [t_{\min}, t_{\max}] \) | comma-separated | `ea_ref_times` |
| **Grid:** | Points per axis of the formula's reference grid | 0 = 201 (1D), 101 (2D), 31 (3D) | integer | `ea_ref_grid` |

## Formulation

At each reference time \( t \), with the reference values \( u_{\mathrm{ref}}(\mathbf{x}_n) \) at the
reference points \( \mathbf{x}_n \), \( n = 1, \dots, N \), and the network's prediction
\( u_\theta(\mathbf{x}_n, t) \) *at the same points* (no interpolation of the reference):

\[
e_{L_2} = \frac{\lVert u_\theta - u_{\mathrm{ref}}\rVert_2}{\lVert u_{\mathrm{ref}}\rVert_2 + 10^{-10}},\quad
\mathrm{MSE} = \frac1N\sum_n \big(u_\theta - u_{\mathrm{ref}}\big)^2,\quad
e_{\max} = \max_n |u_\theta - u_{\mathrm{ref}}|,\quad
e_{\mathrm{mean}} = \frac1N\sum_n |u_\theta - u_{\mathrm{ref}}| .
\]

Over all \( T \) reference times together (every reference point at every time):

\[
e_{L_2}^{\mathrm{all}} = \left(\frac{\sum_{t}\lVert u_\theta(\cdot, t) - u_{\mathrm{ref}}(\cdot, t)\rVert_2^2}{\sum_t \lVert u_{\mathrm{ref}}(\cdot, t)\rVert_2^2}\right)^{1/2},
\]

reported when there is more than one reference time. For an inverse unknown with true value
\( \lambda^\ast \), the parameter error is

\[
e_\lambda = \frac{|\lambda - \lambda^\ast|}{|\lambda^\ast|}\times 100\,\% .
\]

The compared field is the output the file names (`t_<time>_<output>.txt`), or with a file that names no
output, the field the Results page plots.

## Output files

In the run folder, `error_analysis/`:

- `error_metrics.txt`: CSV with the columns `t`, then `L2_relative`, `MSE`, `Max_error` for the ticked
  metrics, and always `Mean_abs_error`;
- `error_spacetime.txt`: `times,L2_relative_all_times`;
- the line and surface comparison figures (and, for a 3D problem on a box, `view3d_comparison.png`).

The Training Log prints the same numbers (the relative L2 error is written L₂ there). A parameter sweep
writes the error of every run, and for inverse problems each parameter's error in percent, to
`sweep_summary.csv`.

!!! info "New in v1.6.0 (in development)"
    The relative L2 error over all reference times together (`error_spacetime.txt`).

## Defaults and limitations

- The metrics are evaluated at the reference points only; there is no integration over the domain.
- A reference point with a missing value (NaN, e.g. outside a shape) is dropped.
- The \( 10^{-10} \) in the denominator only matters for a reference that is (almost) zero everywhere,
  where the relative error is not meaningful.
- The exact-solution formula is written to reference files by the scripts, so Error Analysis is the same
  for both reference kinds.

## Supported problem types

All dimensions, forward and inverse; with time windows each reference time is compared with the window
that covers it.

## Compatibility with other features

Bayesian search on the L2 error needs reference data: "Bayesian search on the L2 error needs Error
Analysis reference data (files or an exact-solution formula) -- or minimize the final loss instead."

## Common mistakes and troubleshooting

- **A file is not matched**: the time is read from the file name; check it against the time interval.
- **Large relative error with a small absolute error**: the reference is close to zero at that time;
  look at the maximum or mean absolute error.

## Example

Most DeepXDE examples use an exact-solution formula, e.g. **Poisson 1D – Dirichlet** and
**Heat 1D – Dirichlet**; the phase-field templates use reference files. See
[Tutorial 17: Error analysis](../tutorials/17-error-analysis.md) and the
[Error Analysis guide page](../guide/error-analysis.md).
