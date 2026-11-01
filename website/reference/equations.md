---
search:
  boost: 2
---

# ODEs and PDEs

## What it does

The **Equations** page defines the outputs of the network, the governing equations as residuals that
training drives to zero, and named constants. PINNStudio turns each equation into a DeepXDE residual
function built from automatic-differentiation derivatives (`dde.grad.jacobian`, `dde.grad.hessian`), as
in Raissi et al. ([2019](../about/references.md#raissi2019)) and
Lu et al. ([2021](../about/references.md#lu2021deepxde)). Two special terms are available:
`integral(...)` for integro-differential ODEs and `frac(...)` for fractional equations.

## When to use it

For every problem: the equations are the physics the network is trained on.

## How to enable it

1. **Problem** page → **Problem Dimension**: **ODE  (t)**, **1D  (x)**, **2D  (x, y)** or
   **3D  (x, y, z)**; **Problem Type**: **Time Dependent** or **Stationary** (an ODE is always over
   \( t \); **Stationary** is disabled for it).
2. **Equations** page → **PDE Definition**: set **Outputs:** and **Equations:**, name each output
   (**Output 1 name:** …), and type each residual in **PDE 1 (residual = 0):** ….
3. **Derivative syntax** opens the list of derivative names for the current dimension;
   **+ Add constant** adds a named constant.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Outputs:** | Number of network outputs | 1 | 1–10 | `num_outputs` |
| **Equations:** | Number of residuals; reset to the number of outputs whenever **Outputs:** changes | 1 | 1–10 | `num_equations` (0 = one per output) |
| **Output *n* name:** | Name used in the expressions | `u`, `v`, `w`, `p`, then `u5`, … | a Python name, see below | `output_names` (comma-separated) |
| **PDE *n* (residual = 0):** | The residual expression | `du_t - 0.4 * du_xx` (1D; a new ODE system: `du_t + 0.4 * u`) | expression | `pde_expressions` (`|`-separated) |
| **Constants:** | Named values, e.g. `Re = 20`, `nu = 1/Re` | none | number or expression of math functions and earlier constants | `constants` (JSON list of `{name, value}`) |
| **∫ points:** (ODE only) | Gauss–Legendre points for `integral(...)` | 20 | 2–200 | `integral_points` |
| **Fractional  α:** | Order of `frac(...)` (shown while an equation uses it) | 1.5 | number in (0, 2), a constant, or an inverse unknown's name | `fractional_alpha` |
| **Resolution:** | Discretization of `frac(...)` | 100 | 1D: one integer; 2D: two (directions, points per unit length, e.g. `8, 100`); 3D: three (`8, 8, 100`); each at least 2 | `fractional_resolution` |
| **Mesh:** | `dynamic` or `static` fractional mesh | dynamic | dynamic, static (static on a 1D Interval only) | `fractional_meshtype` |

## Expression syntax

An equation is written as its residual, the expression that must equal zero:
`du_t + u*du_x - 0.01/pi*du_xx` for \( u_t + u u_x - (0.01/\pi) u_{xx} = 0 \).

Names an expression can use:

- the outputs by name (`u`, `v`, …) and their derivatives (below);
- the coordinates `x`, `y`, `z`, `t` (as the dimension has them), e.g. a source term
  `- exp(-t)*sin(pi*x)`;
- the named constants, and in an inverse problem the unknowns' names;
- the functions `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `exp`, `log`,
  `log10`, `sqrt`, `abs`, `ceil`, `floor`, `relu`, `sigmoid`, `lgamma`, and the constant `pi`. They are
  applied to the network's tensors (`sin(...)` becomes `torch.sin(...)`); a call written as `torch.…`
  or `np.…` is left as written.

### Derivative names

A derivative of output `u` is `du_` followed by the variables it is taken with respect to, e.g. `du_x`,
`du_xx`, `du_xt`. For an output named `phi` it is `dphi_x`, and so on. The names the generated residual
function computes are:

| Dimension | Time dependent | Stationary |
|---|---|---|
| ODE (variable \( t \)) | `du_t`, `du_tt`, `du_ttt`, `du_tttt` | — |
| 1D | `du_x`, `du_t`, `du_xx`, `du_tt`, `du_xt`, `du_xxx`, `du_xxxx`, `du_xxtt`, `du_tttt` | `du_x`, `du_xx`, `du_xxx`, `du_xxxx` |
| 2D | `du_x`, `du_y`, `du_t`, `du_xx`, `du_yy`, `du_xy`, `du_tt`, `du_xt`, `du_yt`, `du_xxxx`, `du_yyyy`, `du_xxyy`, `du_xxtt`, `du_yytt` | `du_x`, `du_y`, `du_xx`, `du_yy`, `du_xy`, `du_xxxx`, `du_yyyy`, `du_xxyy` |
| 3D | `du_x`, `du_y`, `du_z`, `du_t`, `du_xx`, `du_yy`, `du_zz`, `du_tt`, `du_xy`, `du_xz`, `du_yz`, `du_xt`, `du_yt`, `du_zt`, `du_xxxx`, `du_yyyy`, `du_zzzz`, `du_xxyy`, `du_xxzz`, `du_yyzz`, `du_xxtt`, `du_yytt`, `du_zztt` | the same without any name containing `t` |

First derivatives are computed with `dde.grad.jacobian`, second derivatives with `dde.grad.hessian`,
`du_xxx` as the Jacobian of `du_xx`, and the fourth-order names as the Hessian of the matching
second derivative (e.g. `du_xxyy` = \( \partial_{yy} u_{xx} \)). Mixed derivatives across different
outputs are not named; differentiate each output separately. An ODE is solved internally as a
stationary 1D problem in \( x = t \): `du_t` becomes `du_x`, `du_tt` becomes `du_xx`, and so on.

!!! note
    The **Derivative syntax** list on the page does not show `du_xxx` (1D) or `du_yytt` (2D), although
    the generated scripts compute them (`du_xxx` is used, for example, for a KdV term).

### Output names

**Output *n* name:** must be a Python name, unique, and not a name that already means something in the
scripts: a coordinate, the array `y` the outputs are read from, `np`, `dde`, `torch`, a math function
(`sin`, `pi`, …), a Python keyword (`lambda`), a side name, or a derivative-like name (`du_x`). In an ODE
an output cannot be called `x` or `t`; with `integral(...)` terms it cannot be called `s`.

!!! info "New in v1.6.0 (in development)"
    Output names are checked before Solve. An output named `y`, `np`, `sin`, `lambda`, a name with a
    space, or one name for two outputs used to crash a script or (with `y`) silently train a different
    problem.
    Derivative names outside the table above (e.g. `du_xxy`) are refused before Solve; they used to stop
    both scripts with `NameError` at the first training step.

### Integral terms `integral(...)`

`integral(f)` is the integral of `f` from the start of the range \( t_0 \) to the current point \( t \):

\[
\texttt{integral}(f)(t) = \int_{t_0}^{t} f\big(s, t, u(s)\big)\,\mathrm{d}s,
\]

where `f` is written in the integration variable `s`, the current point `t` and the outputs evaluated at
`s`, e.g. `du_t + u - integral(exp(s - t)*u)` (a Volterra integro-differential equation). The script
evaluates it by Gauss–Legendre quadrature of the network with **∫ points:** nodes, as DeepXDE's
`dde.data.IDE` does. Available for ODE problems and stationary 1D problems (where the variable is `x`).

### Fractional operator `frac(...)`

`frac(u)` is DeepXDE's discretized fractional operator of order α applied to output `u`
(`dde.data.FPDE`, Grünwald–Letnikov discretization). In 1D it is the sum of the left and right
Riemann–Liouville derivatives; in 2D and 3D it is the integral over directions of the directional
derivative, which times \( C(\alpha, D) = \Gamma\big(\tfrac{1-\alpha}{2}\big)\Gamma\big(\tfrac{D+\alpha}{2}\big) / \big(2\pi^{(D+1)/2}\big) \)
is minus the fractional Laplacian. α can be a number, a constant, or an inverse unknown's name (then α is
learned). The test loss of a fractional problem uses the training points (DeepXDE's fractional data), not
**Test points**.

## Formulation

For outputs \( u_1, \dots, u_m \) and equations \( \mathcal{N}_1, \dots, \mathcal{N}_K \), the residual
function returns \( [\mathcal{N}_1[u_\theta](\mathbf{x}), \dots, \mathcal{N}_K[u_\theta](\mathbf{x})] \)
at the collocation points; each equation is one loss term with its own weight (see
[Loss weights](loss-weights.md)). The number of equations need not equal the number of outputs.

## Defaults and limitations

- Named constants are substituted by their values in every expression of the problem (equations,
  conditions, transforms, the exact-solution reference) before the scripts are written.
- `integral(...)` and `frac(...)` cover the forms above only; other integral operators are not available.

## Supported problem types

All dimensions; `integral(...)` on ODE and stationary 1D problems; `frac(...)` on 1D, 2D and 3D problems.

## Compatibility with other features

Refused by `validate()`:

> Equations: du_xxy is not available (see Derivative syntax on the Equations page) -- write the equation
> with the derivatives PINNStudio provides, e.g. a third derivative as a new output.

This is printed for any derivative name outside the table above (here `du_xxy`).

> integral(...) terms are available for ODE problems (and steady 1D problems): the integral runs over the
> one variable, from its start to the current point.

> frac(...) problems can't use RAR, Time-Adaptive training or IC pre-training (DeepXDE's fractional data
> keeps its own collocation points).

> A static fractional mesh works only on a 1D Interval -- use dynamic.

> Fractional order alpha must be between 0 and 2; got ….

> ODE problems use t as the variable (and x internally), so an output can't be named 'x' (or 't') --
> rename it (e.g. y1, y2).

Causal training and time windows are not available for an ODE (see [Causal training](causal.md)).

## Common mistakes and troubleshooting

- **A name is not recognised**: the equation preview under the box marks names it does not know (a typo
  in an output name, a constant not defined yet, an unknown of an inverse problem in Forward mode).
- **A derivative of a stationary problem with `t`**: a stationary problem has no time column; remove the
  time derivative or switch to **Time Dependent**.

## Example

**Euler beam 1D** (`du_xxxx`), **Integro-differential ODE** (`integral(...)`), **Fractional Poisson 1D**
and **Fractional Poisson 2D – disk** (`frac(...)`), and **Linear ODE system** in **Examples**. See
[Tutorial 1: First forward ODE](../tutorials/01-ode.md) and the
[Equations guide page](../guide/equations.md).

## Related

- **Tutorials:** [1. First forward ODE](../tutorials/01-ode.md), [18. Your own PDE (Fisher-KPP)](../tutorials/18-own-problem.md)
- **Examples:** [Euler beam 1D](../examples/1d/euler-beam-1d.md), [Integro-differential ODE](../examples/ode/integro-differential-ode.md), [Fractional Poisson 1D](../examples/1d/fractional-poisson-1d.md), [Fractional Poisson 2D – disk](../examples/2d/fractional-poisson-2d-disk.md), [Linear ODE system](../examples/ode/linear-ode-system.md)
- **Theory:** [Losses and automatic differentiation](../theory/losses.md)
- **Method references:** [Raissi et al. 2019](../about/references.md#raissi2019); [Lu et al. 2021](../about/references.md#lu2021deepxde) (DeepXDE)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/equations.md"
