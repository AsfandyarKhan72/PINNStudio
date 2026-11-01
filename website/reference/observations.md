---
search:
  boost: 2
---

# Observational data

## What it does

In an inverse problem the **Measured data** box lists the data files the solution is fitted to. Each file
is one loss term (`dde.icbc.PointSetBC` on one output, or `dde.icbc.PointSetOperatorBC` for an expression
of the outputs) with its own weight. By default every measured point is also an *anchor*: a collocation
point where the equation is checked. The data can be thinned to a coarser grid or a random subset without
changing the files.

## When to use it

For every inverse problem (at least one file is required). Several files are useful when different
outputs, or different times, are measured.

## How to enable it

**Inverse problem** page → **Measured data**: one block per file with **File:**, **Measures:** and
**Loss weight:**; **➕ Add data file** adds a block. Under the blocks: **Check the equation at the measured
points too (anchors)** and **Use:**. A summary line under **Use:** says what the files hold and how many
points are kept (e.g. "101 × 101 grid, 10,201 points → 51 × 51 = 2,601 points").

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **File:** | A measured-data file (`.txt`, `.dat`, `.csv`) | — | path | `inverse_obs_files_json` (`path`) |
| **Measures:** | Which output the values are, or **Custom...** for an expression of the outputs | Output 1 | Output 1 (name), …, Custom... | `inverse_obs_files_json` (`output_idx`, `custom_expr`) |
| custom expression | Expression of the outputs by name, e.g. `sqrt(u**2+v**2)` | blank | expression | `custom_expr` |
| **Loss weight:** | Weight of this file's data term | 100 | any number | `inverse_obs_files_json` (`weight`) |
| **Check the equation at the measured points too (anchors)** | Measured points are also collocation points | on | — | `inverse_obs_anchors` |
| **Use:** | **All points**, **Grid: points per axis** or **Random points** | All points | — | `inverse_obs_reduce` (`all`, `grid`, `random`) |
| grid number | Points kept along each space axis | 51 | 2–100,000 | `inverse_obs_grid_n` |
| random number | Points kept per file | 1000 | 1–10,000,000 | `inverse_obs_random_n` |

### File format

Plain text, one point per line: the input coordinates as the problem has them, then the measured value —
`x, t, u` (1D), `x, y, t, u` (2D), `x, y, z, t, u` (3D), without `t` for a stationary problem, and `t, u`
for an ODE. Numbers are separated by commas, semicolons or spaces; blank lines, lines starting with `#` or
`%`, and lines that are not numbers (a header) are skipped. Rows with a missing or non-finite value
(`nan`, `inf`) or with a different number of columns from the most common one are skipped too, and the
Training Log says how many. The window, both scripts and Error Analysis read the files the same way.

## Formulation

File \( m \) with points \( (\mathbf{x}_n, d_n) \) adds

\[
w_m\,\ell\big(g_m(\mathbf{x}_n) - d_n\big)_{n=1}^{N_m},\qquad
g_m = u_{\theta,k}\ \text{(Measures: Output } k\text{)}\ \text{or}\ F(u_{\theta,1}, \dots)\ \text{(Custom...)},
\]

after the equation, boundary and initial terms in the loss-weight list. A custom expression is evaluated on
the network's tensors with the math functions (`sqrt` includes a shift of \( 10^{-12} \) so its gradient
stays finite at zero). With anchors on, the points of all files are added to the training points
(`anchors=`), together with any **Extra training points** from the **Collocation points** page.

**Reduction** (**Use:**), applied to every file:

- **Grid: points per axis** keeps \( N \) evenly spaced values along each space axis (the edges kept), the
  same for every time in the file, e.g. a 101 × 101 grid with 51 gives 51 × 51 per time. A file that is not
  a full grid at every time keeps \( N^{d} \) random points instead (\( d \) = number of coordinate columns,
  time counted).
- **Random points** keeps \( N \) points per file, chosen with the run's random seed (seed 0 when the seed
  is off), so the choice repeats.

## Defaults and limitations

- The shipped inverse examples weight their data 100.
- Every kept point is used at every training step. Tens of thousands of points make training slow, above
  all on the CPU; the Training Log prints a note at 20,000 points or more.
- With very few measured points (the window's help mentions 5 points, Brinkman-Forchheimer), keep the
  anchors on.
- A custom expression uses the outputs, not their derivatives.

## Supported problem types

Inverse problems of every dimension, stationary and time-dependent.

## Compatibility with other features

> Inverse mode needs at least one observed-data file to fit against, but no observation row has a file
> selected.

In a sweep, **Measured data: grid points per axis** and **Measured data: random points** cannot be swept
together: "Parameter Sweep: 'Measured data: grid points per axis' and 'Measured data: random points' each
set how the measured data are reduced, so they can't be swept together -- sweep one of them." The category
**Inverse** also offers **Measured data: times** (the data at every reference time instead of the file
picked) and each file's weight (see [Sweeps](sweeps.md)).

## Common mistakes and troubleshooting

- **Wrong column count**: a 2D time-dependent file needs four columns (`x, y, t, value`); a stationary one
  three. A file with fewer columns than coordinates + 1 stops the script; single rows with an extra or a
  missing column are skipped with a note.
- **The fit is good but the unknown is wrong**: too few points, or points that do not excite the term the
  unknown multiplies; add data at other times or places.

## Example

**Navier-Stokes 2D – Kovasznay flow** (two data files), **Diffusion-reaction 1D** (Inverse, four files),
**Klein-Gordon 1D** (Inverse, ten files) and **Allen-Cahn 3D – Wight & Zhao** (Inverse, **Use:** grid) in
**Examples**, category **Inverse**. See [Tutorial 7: Importing observational data](../tutorials/07-data.md)
and the [Inverse problems guide page](../guide/inverse.md).

## See also

- [Theory › Forward and inverse problems](../theory/forward-inverse.md)
- [Tutorial 7: measured data](../tutorials/07-data.md)
