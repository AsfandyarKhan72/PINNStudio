# Initial conditions

For a time-dependent problem, the **Initial conditions** page gives each output's value at the start of
the time interval, \( t = t_{\min} \). It has one block per output (set by **Outputs** on the
[Equations](equations.md) page). The page is not shown for a stationary problem.

![The Initial conditions page of the 1D Burgers example: the block Output 1 (u) with IC for u ticked, Exact (built into the network) unticked, the expression -sin(pi*x), and Load IC from file (x,t,c format) unticked](../assets/screenshots/burgers_ic.png){ .screenshot loading=lazy }

Background: [Initial conditions and imported data](../reference/initial-conditions.md),
[Exact constraints](../reference/exact-constraints.md).

## Give an initial condition as a formula

Type the initial value as an expression of the space coordinates (`x`; `x, y`; `x, y, z`), using the
same functions as the equations, for example `-sin(pi*x)`, `exp(-(x**2+y**2))` or `0`. Named constants
can be used. In an ODE problem the initial value is usually a number. Click **Initial-condition syntax**
for examples for the current dimension.

!!! info "New in v1.6.0 (in development)"
    An initial condition written with `t` (for example an exact solution copied into the box) is
    evaluated with `t` equal to `t_min`. Before, both scripts stopped with `NameError: t`.

Each ticked **IC for …** is one loss term, **IC N** on the [Loss weights](loss-weights.md) page, with a
default weight of 100.

![The Initial conditions page of a new 1D problem with two outputs: a block for u with sin(pi*x) and a block for v with cos(pi*x), each with its own IC for and Exact check boxes; only the first block offers Load IC from file](../assets/screenshots/guide_ic_two_outputs.png){ .screenshot loading=lazy }

## Leave an output without an initial condition

Untick **IC for …** for an output that has no initial condition of its own, for example the chemical
potential of a Cahn–Hilliard system, which follows from the other output. That output then has no IC
loss term and no IC weight.

## Build the initial condition into the network

Tick **Exact (built into the network)** to make the initial condition hold exactly instead of fitting
it. The output becomes

\[ u(x, t) = \mathrm{IC}(x) + \tau\, N(x, t), \qquad \tau = \frac{t - t_{\min}}{t_{\max} - t_{\min}}, \]

where \( N \) is the network, so \( u = \mathrm{IC} \) at \( t_{\min} \) whatever the network learns. The
IC loss term is then switched off (its **IC for …** box is unticked and greyed out) and its weight row
disappears. The Neural network page notes that the output transform is in use.

An exact initial condition needs the condition as a formula (not a file) and cannot be combined with
**Time Adaptive Training** or with your own output-transform formula; **Solve** says so if you try.

## Load the initial condition from a file

For output 1 only, tick **Load IC from file (x,t,c format)** and browse for a text file. Its columns are
the coordinates, the time and the value: `x, t, c` in 1D, `x, y, t, c` in 2D, `x, y, z, t, c` in 3D,
separated by commas, semicolons or spaces. Only the rows at the start time \( t_{\min} \) are used; the
run stops with an error if the file has none. Header and comment lines are skipped, and so are rows with
a missing value (NaN) or a different number of columns than most rows, with a note in the Training Log.
While the file is used, the formula box is hidden and **Exact** cannot be ticked.

A second check box then appears, **Also check the equation at the file's points** (off by default). Ticked,
the file's points are also used as collocation points (anchors), so the equation's residual is evaluated
there as well as the initial-condition loss. It is not used with time windows, where each window has its
own initial points.

## Initial conditions on derivatives

A condition on a time derivative, such as \( u_t(x, 0) = 0 \) for a wave equation, or \( u'(0) = 2 \)
in an ODE, is not an initial value. Add it on the [Boundary conditions](boundary-conditions.md) page as
an **Operator BC (advanced)** row with **Applies on: the initial time (t = t₀)** and the function
`du_t` (or `du_t - 2`).

## How this page connects to the others

- **Time-adaptive training** uses the initial condition for its first time window; the later windows
  start from the previous window's solution ([Adaptive training](adaptive.md)).
- **Initial-condition pre-training** on the Training page first fits the network to the initial
  conditions given as loss terms ([Training phases](training.md#initial-condition-pre-training)).
- An inverse problem uses the same initial conditions.

## Settings on this page

| GUI label | Meaning | Default (new 1D problem) | Saved as |
|---|---|---|---|
| **IC for** *name* | The output has an initial condition (a loss term) | on | `ic_active` (comma-separated, one per output) |
| **Exact (built into the network)** | Impose the IC exactly through the output | off | `ic_hard` |
| (expression box) | Initial value as a formula | output 1: `sin(pi*x)`, others: `cos(pi*x)` (2D: `sin(4*pi*x)*cos(4*pi*y)`, `cos(4*pi*x)*sin(4*pi*y)`) | `ic_expressions` (separated by a vertical bar) |
| **Load IC from file (… format)** | Output 1's IC from a file | off | `forward_ic_from_file` |
| (file box) | The IC file | empty | `forward_ic_file` |
| **Also check the equation at the file's points** | The IC file's points are also collocation points (anchors) | off | `ic_file_anchors` |
