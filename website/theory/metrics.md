# Error measures

The training loss shows how well the network satisfies the equations and conditions at the training
points; it is not the error of the solution. PINNStudio measures the error only against a reference solution,
in **Error Analysis** ([guide](../guide/error-analysis.md), [reference](../reference/metrics.md)). This page
gives the formulas exactly as the generated scripts evaluate them.

## Reference points

A reference solution is given either as files or as a formula:

- **Files.** One file per reference time (for a steady problem, one file), with one row per point:
  the coordinates, the time (time-dependent 2D and 3D files) and the value. Rows with a missing value
  (NaN, for example a point outside the shape) are dropped.
- **Formula.** The script evaluates the formula on a grid over the geometry's bounding box, keeps the grid
  points inside the geometry, and writes them as ordinary reference files (201 points per axis in 1D,
  101 in 2D, 31 in 3D, unless a grid size is given; five evenly spaced times over \([t_{\min}, t_{\max}]\)
  unless times are given).

For each reference time \(t_s\) the network is evaluated **at the reference points themselves**,
\(\hat u_n = u_\theta(\mathbf{x}_n, t_s)\), \(n = 1, \dots, N_s\); nothing is interpolated for the metrics. With
time windows, each reference time is evaluated with the network of the window that contains it. The
compared quantity is the selected output or a derived field (for example \(\sqrt{u^2 + v^2}\)) when one is
set; problems with several outputs are compared output group by output group, each with its own metrics file.

## Metrics at one reference time

With \(\hat u, u \in \mathbb{R}^{N_s}\) the PINN and reference values at time \(t_s\):

| Metric (Error Analysis checkbox) | Formula |
|---|---|
| Relative \(L_2\) error (**L2 Relative**, on by default) | \( \varepsilon_{L_2}(t_s) = \dfrac{\lVert \hat u - u \rVert_2}{\lVert u \rVert_2 + 10^{-10}} = \dfrac{\big(\sum_n (\hat u_n - u_n)^2\big)^{1/2}}{\big(\sum_n u_n^2\big)^{1/2} + 10^{-10}} \) |
| Mean squared error (**MSE**) | \( \mathrm{MSE}(t_s) = \dfrac{1}{N_s}\sum_n (\hat u_n - u_n)^2 \) |
| Maximum error (**Max Error**) | \( e_{\max}(t_s) = \max_n \lvert \hat u_n - u_n \rvert \) |
| Mean absolute error (always written) | \( \bar e(t_s) = \dfrac{1}{N_s}\sum_n \lvert \hat u_n - u_n \rvert \) |

The ticked metrics and the mean absolute error are printed in the Training Log for every reference time and
written to `error_analysis/error_metrics.txt` (columns `t`, `L2_relative`, `MSE`, `Max_error`,
`Mean_abs_error`). The \(10^{-10}\) in the denominator only prevents a division by zero for a reference that is
zero everywhere; the per-round diagnostics of RAR use \(10^{-12}\).

The relative \(L_2\) error is computed over the reference points, so it is a discrete norm weighted by the
point distribution of the reference file, not a quadrature of
\(\lVert \hat u - u \rVert_{L^2(\Omega)} / \lVert u \rVert_{L^2(\Omega)}\). For an evenly spaced grid the two are
close.

## All reference times together

When there is more than one reference time, the script also reports the relative \(L_2\) error over every
reference point at every time,

\[
\varepsilon_{L_2}^{\text{all}} =
\left( \frac{\sum_s \sum_n (\hat u_{s,n} - u_{s,n})^2}{\sum_s \sum_n u_{s,n}^2} \right)^{1/2},
\]

in the Training Log and in `error_analysis/error_spacetime.txt`. This is the single number most papers report
for a time-dependent problem. It is not the mean of the per-time errors: times where \(u\) is large weigh more.

!!! info "New in v1.6.0 (in development)"
    The error over all reference times together (`error_spacetime.txt`) is added in the development version.

## What other parts of PINNStudio report

- **Results › Error analysis table.** The per-time relative \(L_2\) error and mean absolute error, with a summary
  line giving the mean and the worst relative \(L_2\) error over the reference times.
- **Parameter sweeps and Bayesian search.** The "L2 error" of a run is the mean of the `L2_relative` column
  over every row of the run's metrics files, that is over the reference times (and output groups); a steady
  problem has one row.
- **Parameter relative error** (inverse problems, sweep summary). For an unknown with a known, non-zero
  **True value**,

  \[
  e_\lambda = \frac{\lvert \hat\lambda - \lambda_{\text{true}} \rvert}{\lvert \lambda_{\text{true}} \rvert} \times 100\,\%,
  \]

  where \(\hat\lambda\) is the last value in the run's parameter history. Without a true value (or with
  \(\lambda_{\text{true}} = 0\)) no error is given.

!!! warning
    In the time-window Error Analysis, a reference time for which no window's network can be found is filled
    with zeros (a warning is printed), which gives a relative error of about 1 for that time. Check the
    Training Log when one time stands out.

## Reading the numbers

An error measure is only as good as the reference. A reference computed on a coarse grid, or with a
different formulation of the problem, limits what the comparison can show. Report which reference was used
and how it was obtained; see [Reference data](../reproducibility/reference-data.md).
