---
search:
  boost: 2
---

# Boundary conditions

## What it does

The **Boundary conditions** page is a list of rows; each row becomes one DeepXDE condition object
(`dde.icbc.DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`, `PointSetBC`, `PointSetOperatorBC`,
`OperatorBC`, `Interface2DBC`) and one loss term with its own weight. A row has a **Type:**, the
**Output:** it applies to, **Where:** on the boundary it holds and a **Value:** (or the type's own
fields).

## When to use it

For every problem with boundary conditions. A condition can also be built into the network exactly instead
(an output transform, or [exact periodicity](exact-constraints.md)); a row is the soft (loss-term) form.

## How to enable it

**Boundary conditions** page → **➕ Add Boundary Condition**, then set the row's fields. A loaded example
fills the list with its own rows. **"Where" examples** opens a short reference; **✕** removes a row.
Edges picked on the **Sketch** tab with **Edges → BCs** can create or extend rows (see
[Geometry](geometry.md#sketch)).

## Settings

| Label | Meaning | Default | Range / choices | Config field (`custom_bc_json` row) |
|---|---|---|---|---|
| **Type:** | The condition type | Dirichlet BC | Dirichlet BC, Neumann BC, Robin BC, Periodic BC, Point Set BC, Point Set Operator BC (advanced, from file), Operator BC (advanced), Interface 2D BC (advanced, Rectangle/Polygon only) | `type`: `dirichlet`, `neumann`, `robin`, `periodic`, `pointset`, `pointset_operator`, `operator`, `interface2d` |
| **Output:** | Output the row constrains (1 = first output) | 1 | the problem's outputs | `component` (0-based) |
| **Where:** | Part of the boundary: a named choice or **Custom expression…** | Whole boundary | see below | `location` |
| **Value:** / **Function (func):** | Target value or operator expression | `0` | expression | `value` |
| **Periodic in:**, **Derivative order:** | Periodic axis and whether values (0) or first derivatives (1) are matched | x, 0 | the problem's axes; 0–1 | `axis`, `deriv_order` |
| **Points:** | Point Set rows: **From file** (coordinate columns, then the value column(s); commas, semicolons or spaces; header, missing-value and wrong-width rows skipped with a note), **Typed list** or **Generated** (a count, **on the boundary (Where)** or **in the domain**) | From file | count 1–1,000,000 (default 10) | `points_mode`, `points_file`, `points_list`, `points_count`, `points_where` |
| **Target values:** | **From the file's value column(s)** or **Formula** (`|` between components) | From the file | — | `values_mode`, `values_formula` |
| **Components (comma-separated):** | Point Set BC: several outputs against one file (one value column each) | blank (uses **Output:**) | output numbers | `components` |
| **Batch size (0 = off):**, **Shuffle** | Point Set rows: a random batch of the points, resampled during training | 0, on | 0–1,000,000 | `batch_size`, `shuffle` |
| **Applies on:** | Operator BC: **the boundary** or **the initial time (t = t₀)** | the boundary | — | `apply_at` (`boundary`, `initial`) |
| **Where (side 2):**, **Direction:** | Interface 2D BC: the second edge and `normal` or `tangent` | blank, normal | — | `location2`, `direction` |

### Where: location expressions and side names

The **Where:** picker offers **Whole boundary** (`True`), **Left side (x = x_min)** (`left`), **Right
side (x = x_max)** (`right`), **Left and right sides** (`left or right`), and in 2D/3D the **Bottom**/**Top**
(`bottom`/`top`, \( y = y_{\min}/y_{\max} \)) and in 3D the **Front**/**Back** (`front`/`back`,
\( z = z_{\min}/z_{\max} \)) sides and their pairs. In an ODE the choices are **Both ends (t₀ and t₁)**,
**Start (t = t₀)** (`start`) and **End (t = t₁)** (`end`).

**Custom expression…** opens a box for any true/false Python expression in `x`, `y`, `z` (and `t`), the
domain limits `x_min`, `x_max`, `y_min`, … , the side names, `np` and the math functions, e.g.
`x <= 0.5 and bottom` or `np.isclose(x**2 + y**2, 0.25)`. It is evaluated only at points already on the
boundary, so it only has to say *which part*. A side name is true where the coordinate equals that limit
(`np.isclose`); for a non-box shape the limits are its bounding box.

Value expressions use `x`, `y`, `z`, `t`, the domain limits, the named constants and the math functions;
a Robin value may also use `u` (this row's output) and every output by name.

## Formulation

With \( \mathbf{n} \) the outward unit normal and \( \Gamma \) the part of the boundary picked by
**Where:**, each type adds the loss term \( \ell(e) \) of the error \( e \) below at its points:

| Type | Condition | Error trained |
|---|---|---|
| Dirichlet | \( u = g \) on \( \Gamma \) | \( u - g \) |
| Neumann | \( \partial u/\partial n = g \) on \( \Gamma \) | \( \nabla u\cdot\mathbf{n} - g \) |
| Robin | \( \partial u/\partial n = f(\mathbf{x}, u) \) on \( \Gamma \) | \( \nabla u\cdot\mathbf{n} - f(\mathbf{x}, u) \) |
| Periodic, order 0 | \( u(\mathbf{x}) = u(\mathbf{x}') \) | \( u(\mathbf{x}) - u(\mathbf{x}') \) |
| Periodic, order 1 | \( \partial_a u(\mathbf{x}) = \partial_a u(\mathbf{x}') \) | \( \partial_a u(\mathbf{x}) - \partial_a u(\mathbf{x}') \) |
| Point Set | \( u(\mathbf{x}_n) = d_n \) at listed points | \( u(\mathbf{x}_n) - d_n \) |
| Point Set Operator | \( \mathcal{G}[u](\mathbf{x}_n) = d_n \) at listed points (anywhere in the domain) | \( \mathcal{G}[u](\mathbf{x}_n) - d_n \) |
| Operator | \( \mathcal{G}[u] = 0 \) on \( \Gamma \) (or at \( t = t_0 \)) | \( \mathcal{G}[u] \) |
| Interface 2D | DeepXDE's `Interface2DBC` on two edges | \( \langle \mathbf{u}(\mathbf{x}_1), \mathbf{d}_1\rangle + \langle \mathbf{u}(\mathbf{x}_2), \mathbf{d}_2\rangle - g(\mathbf{x}_1) \) |

- **Neumann and Robin values are the outward normal derivative** \( \partial u/\partial n \): on a right
  side it is \( +\partial u/\partial x \), on a left side \( -\partial u/\partial x \) (likewise in
  \( y \) and \( z \)). A Neumann value of `4` on the left side of \([0,1]\) means \( u_x(0) = -4 \).
- **Periodic**: \( \mathbf{x}' \) is the periodic image of \( \mathbf{x} \) on the paired side along the
  axis \( a \) (**Periodic in:** x pairs left ↔ right, y bottom ↔ top, z front ↔ back). Each row matches
  either the values (**Derivative order:** 0) or the first derivatives along the axis (1); for both, add
  two rows. A periodic row needs no **Where:** (the window hides it when it is a side of the pair or the
  whole boundary).
- **Operator** and **Point Set Operator** functions use the same derivative shorthand as the equations
  (`u`, `du_x`, `du_xx`, …) and DeepXDE's raw `inputs`, `outputs` and `X`. An Operator row must equal
  zero, so the target is written into it (`du_x - 4`). Applied at the initial time it gives a second
  initial condition such as \( u_t(x, 0) = 0 \) (`du_t`).
- **Interface 2D** follows DeepXDE: for paired points on the two edges, the two-component output is
  projected on each edge's outward normal (**Direction:** `normal`) or its tangent (`tangent`, the normal
  rotated 90° clockwise), the projections are added and compared with **Value:** evaluated on the first
  edge.

## Defaults and limitations

- A Point Set file has the coordinate columns (x[, y[, z]][, t]) and then the value column(s); `.csv`
  files are comma-separated, others whitespace-separated. A typed list separates points with `;`.
- A Point Set row with **Generated** points **on the boundary (Where)** draws random boundary points and
  keeps those where **Where:** holds; when a small region gives fewer than the count, both scripts draw more
  candidates (up to five more, each four times larger), print a note if the count is still not reached, and
  stop if no point matches.
- A Point Set row with a batch size is resampled by DeepXDE's `PDEPointResampler(bc_points=True)`, added
  automatically when the **Point Resampling** callback is off.
- Interface 2D needs uniform boundary points: choosing it switches **Point distribution:** to `uniform`.
- An Operator or Point Set Operator function that uses `X` disables the test set (`num_test=None`), as
  DeepXDE requires.
- No shipped example uses a Point Set BC or an Interface 2D BC row.

## Supported problem types

All dimensions and both problem types. Periodic: Interval, Rectangle, Cuboid. Interface 2D: 2D Rectangle
or Polygon with two outputs. Operator at the initial time: time-dependent problems without time windows.

## Compatibility with other features

Refused before training (code generator):

> Periodic BC requires an Interval (1D), Rectangle (2D), or Cuboid (3D) geometry (current geometry: …).

> Interface 2D BC requires a Rectangle or Polygon geometry (current geometry: …). / Interface 2D BC
> requires Output Size = 2 …

> BC *n*: an Operator BC set to apply at the initial time needs a time-dependent problem (this one is
> steady-state). / … isn't supported with Time-Adaptive training (only the first time window contains
> t = t_min). Use Adaptation = None or RAR for this problem.

Refused by `validate()` with per-point weights:

> Per-point weights (…) belong to fixed training points, but a Point Set condition with a batch size
> resamples the points every 100 steps (as in DeepXDE) and the weights would start over each time -- set
> its batch size to 0, or Per-point weights to None.

## Common mistakes and troubleshooting

- **Sign of a Neumann value** on a left or bottom side: the value is the outward normal derivative (see
  above).
- **A condition that selects no point** has a NaN loss; both scripts name the empty condition before
  training. Typical cause: a side name on a curved boundary, or a `Where` expression that is never true
  on the boundary (use `np.isclose` instead of `==`).
- **Periodic row on a disk or a custom shape** is refused; use an Operator row that compares the two sides
  by hand, or a box domain.

## Example

**Poisson 1D – Dirichlet/Neumann**, **Poisson 1D – Dirichlet/Robin**, **Poisson 1D – Dirichlet/Periodic**
and **Poisson 1D – Dirichlet/point-set operator** (one condition type each), **Helmholtz 2D – sound-hard
scattering** (Neumann and Robin on a combined shape), **Second-order ODE** and **Klein-Gordon 1D**
(Operator at the initial time) in **Examples**. See the
[Boundary conditions guide page](../guide/boundary-conditions.md).

## Related

- **Tutorials:** [3. 2D heat equation](../tutorials/03-heat-2d.md), [8. Custom geometry](../tutorials/08-geometry.md)
- **Examples:** [Poisson 1D – Dirichlet/Neumann](../examples/1d/poisson-1d-dirichlet-neumann.md), [Poisson 1D – Dirichlet/Robin](../examples/1d/poisson-1d-dirichlet-robin.md), [Poisson 1D – Dirichlet/Periodic](../examples/1d/poisson-1d-dirichlet-periodic.md), [Poisson 1D – Dirichlet/point-set operator](../examples/1d/poisson-1d-dirichlet-point-set-operator.md), [Helmholtz 2D – sound-hard scattering](../examples/2d/helmholtz-2d-sound-hard-scattering.md), [Klein-Gordon 1D](../examples/1d/klein-gordon-1d.md)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/boundary-conditions.md"
