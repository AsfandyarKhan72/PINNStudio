# Initial conditions and imported data

## What it does

For a time-dependent problem, the **Initial conditions** page gives each output its value at
\( t = t_{\min} \): a formula in the space coordinates, or (for the first output) values read from a
file. Each initial condition is one loss term (`dde.icbc.IC`, or `dde.icbc.PointSetBC` for a file) with
its own weight, unless it is built into the network exactly (see
[Exact constraints](exact-constraints.md)).

## When to use it

For every time-dependent PDE. Use a file when the initial state is measured or computed elsewhere
(another solver, an experiment) and has no closed form. An ODE takes its initial values from the same
page (values at \( t_0 \)); a stationary problem has no initial condition and the page is hidden.

## How to enable it

**Initial conditions** page, one block per output (**── Output *n* (name) ──**):

1. **IC for *name*** (ticked by default) makes the formula a loss term; untick it for an output without
   an initial condition (e.g. a chemical potential).
2. Type the formula in the box under it, e.g. `sin(pi*x)`, `exp(-(x**2+y**2))`.
3. **Exact (built into the network)** builds it into the output instead (see
   [Exact constraints](exact-constraints.md)).
4. For the first output only: **Load IC from file (x,t,c format)** (2D: `x,y,t,c`, 3D: `x,y,z,t,c`) and
   **Browse** for the file.

**Initial-condition syntax** opens examples for the current dimension.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **IC for *name*** | This output's initial condition is a loss term | on | — | `ic_active` (comma-separated `True`/`False`) |
| formula box | \( u_0(\mathbf{x}) \) in `x`, `y`, `z` | 1D `sin(pi*x)`, 2D `sin(4*pi*x)*cos(4*pi*y)` | expression | `ic_expressions` (`|`-separated) |
| **Exact (built into the network)** | Exact initial condition | off | — | `ic_hard` |
| **Load IC from file (… format)** | Output 0's initial condition from a file | off | — | `forward_ic_from_file`, `forward_ic_file` |
| **Also check the equation at the file's points** | The file's points are also anchors of the equation | off | — | `ic_file_anchors` |
| **Initial points:** (Collocation points page) | Points at \( t = t_{\min} \) for formula ICs | 200 | 0–2,000,000 | `num_initial` |
| IC weight (Loss weights page) | Weight of each IC term | 100 for a new problem | any number | `loss_weights_multi` |

The formula may use the math functions (`sin`, `cos`, `exp`, `sqrt`, `tanh`, `pi`, …) and the named
constants; `np.` and `x[:,0]` are not needed. A `t` in the formula means \( t_{\min} \).

## Formulation

For output \( i \) with an initial condition as a loss term:

\[
\mathcal{L}_{\mathrm{IC},i} = \ell\big(u_{\theta,i}(\mathbf{x}, t_{\min}) - u_{0,i}(\mathbf{x})\big)
\]

at the initial points (formula) or at the file's points (file). An ODE's initial values become Dirichlet
rows at the start \( t = t_0 \) of the 1D problem PINNStudio solves internally; a derivative initial
value \( u'(t_0) = a \) is an Operator BC `du_t - a` applied at the initial time.

### Initial condition from a file

The file is plain text without a required header: numbers separated by commas, semicolons or spaces;
lines that do not parse as numbers (a header, a comment starting with `#` or `%`) are skipped, and so are
rows with a missing or non-finite value (`nan`, `inf`) or with a different number of columns from the
most common one (the log prints how many rows were skipped). Columns: one per space axis, then \( t \), then
the value — `x, t, c` in 1D, `x, y, t, c` in 2D, `x, y, z, t, c` in 3D. Only the rows with
\( t = t_{\min} \) (to within \( 10^{-8} \)) are used; the file must have at least one. The points are
used as given (no interpolation), as a `PointSetBC` on output 0.

## Defaults and limitations

- **The file sets output 0 only.** Other outputs keep their formulas (or none). The option is shown under
  the first output only.
- With the file on, the formula box, **IC for …** and **Exact** of the first output are disabled (the file
  always wins).
- The file's points enter only the initial-condition loss. **Also check the equation at the file's
  points** (shown under the file field while **Load IC from file** is ticked; `ic_file_anchors`, off by
  default) also makes them collocation points (anchors) of the equation. It is not used with time
  windows, where each window has its own initial points.
- With [time windows](time-windows.md) the initial condition applies to the first window; later windows
  start from the previous window's solution.

!!! info "New in v1.6.0 (in development)"
    An initial condition written with `t` (for example an exact solution pasted into the box) used to stop
    both scripts with `NameError: t`; `t` now means \( t_{\min} \) there.

## Supported problem types

Time-dependent 1D, 2D and 3D problems (formula and file), ODEs (formula), forward and inverse.

## Compatibility with other features

- A file cannot be combined with an exact initial condition: "Exact initial condition: needs the initial
  condition as a formula, not from a file."
- [IC pre-training](pretraining.md) needs at least one initial condition as a loss term (or a file):
  "IC pre-training fits the network to the initial conditions given as loss terms, but no output has one
  (an exact initial condition needs no pre-training) -- untick IC pre-training."

## Common mistakes and troubleshooting

- **"IC file … has no rows at t = …"**: the file's time column has no value equal to \( t_{\min} \) of the
  **Time interval** (check the column order: time is the column after the space columns).
- **The IC term dominates or is ignored**: change its weight on the [Loss weights](loss-weights.md) page
  (a new problem starts with 100).

## Example

Every time-dependent example has formula initial conditions, e.g. **Burgers 1D – Raissi et al.**
(`-sin(pi*x)`) and **Cahn-Hilliard 2D – Wight & Zhao** (an IC for \( \phi \), none for \( \mu \)). No shipped
example reads its initial condition from a file. See the
[Initial conditions guide page](../guide/initial-conditions.md) and
[Tutorial 7: Importing observational data](../tutorials/07-data.md).
