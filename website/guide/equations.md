# Equations, outputs and constants

The **Equations** page defines what the network predicts (its *outputs*) and the equations those outputs
must satisfy. Each equation is written as a *residual*: an expression that is zero when the equation
holds. For the viscous Burgers equation \( u_t + u u_x = \nu u_{xx} \) with \( \nu = 0.01/\pi \) you type

```text
du_t + u*du_x - (0.01/pi)*du_xx
```

and the page typesets it underneath, so you can check it at a glance.

![The Equations page of the 1D Burgers example: Outputs 1, Equations 1, the Add constant button, Output 1 name u, and the PDE 1 box with its typeset equation below it](../assets/screenshots/burgers_equations.png){ .screenshot loading=lazy }

Background: [ODEs and PDEs](../reference/equations.md), [Coupled systems](../reference/coupled.md).

## Set the number of outputs first

**Outputs** is the number of fields the network predicts (1 to 10), for example 1 for a heat equation,
2 for the real and imaginary parts of a Schrödinger equation, 3 for a 2D flow (u, v, p). The output
count drives several other pages:

- one **Output N name** box and one equation box per output here;
- one block per output on the [Initial conditions](initial-conditions.md) page;
- one **IC** weight per output on the [Loss weights](loss-weights.md) page;
- one row per output in the network's output transform ([Neural network](network.md));
- the output lists on the Results, Restore & visualize and Inverse problem pages, and the **Output**
  number of each boundary condition.

!!! warning "Changing Outputs resets the equation boxes"
    When you change **Outputs**, the name and equation boxes and the initial-condition expressions are
    rebuilt with their default contents (names `u`, `v`, `w`, `p`, …). Set the number of outputs before
    typing the equations. (Changing **Equations** alone keeps what you typed.)

![The Equations page of a new 1D problem with Outputs set to 2: two output names, u and v, each followed by its own equation box](../assets/screenshots/guide_equations_two_outputs.png){ .screenshot loading=lazy }

**Equations** is the number of residuals. It follows **Outputs** whenever that changes, and can then be
set separately (1 to 10): three equations for one output, or one equation for two outputs when the
second output is an unknown field of an inverse problem. Each equation is one **PDE** loss term with its
own weight.

## Name the outputs

Each output's name is how you refer to it in equations, conditions, transforms, plots and Error
Analysis. Use a short Python-style name: letters, digits and `_`, not starting with a digit.

!!! info "New in v1.6.0 (in development)"
    Output names are checked. A name that is not a valid identifier, a Python word (`lambda`), a
    coordinate, the array the outputs are read from (`y`), a module name (`np`, `dde`), a math function
    (`sin`, `pi`), a side name (`left`, …) or a name used for two outputs is refused, with the message on
    the Equations page. Before, such names could crash a script or silently train a different problem.
    In an ODE, outputs cannot be called `x` or `t`, and with `integral(...)` terms not `s`.

## Write the equations

Derivatives are written `d<output>_<variables>`:

| You type | Means |
|---|---|
| `du_x`, `du_y`, `du_z`, `du_t` | \( \partial u/\partial x \), … |
| `du_xx`, `du_yy`, `du_tt` | second derivatives |
| `du_xy`, `du_xt`, `du_yt`, … | mixed derivatives |
| `du_xxxx`, `du_xxyy`, `du_xxtt`, … | fourth derivatives |

The coordinates (`x`, `y`, `z`, `t` as the dimension has them), the output names, the named constants,
the unknown parameters of an inverse problem and the functions `sin`, `cos`, `exp`, `log`, `sqrt`,
`tanh` and `pi` can all be used. A source term is simply part of the expression, for example
`du_t - 0.4*du_xx - exp(-t)*sin(pi*x)`. Click **Derivative syntax** for the full list for the current
dimension.

Below each box the equation is typeset. A name the page does not know (a typo, or a constant you have
not defined) is listed in orange under it, as in *Unknown names: …*.

### ODE problems

In an **ODE  (t)** problem the only variable is `t`: write `du_t`, `du_tt`. One equation per output
gives a first-order system, for example `dy1_t - y2` and `dy2_t + y1`. For integro-differential
equations, `integral(f)` is the integral of `f` from the start of the time interval to the current `t`,
with `s` as the integration variable; **∫ points** sets the number of Gauss–Legendre points (2 to 200,
default 20). For example `du_t + u - integral(exp(s - t)*u)`.

![The Equations page of a new ODE system with the Derivative syntax help open, explaining du_t, du_tt and integral(f); the default equation still contains du_xx and is flagged as an unknown name](../assets/screenshots/guide_ode_equations.png){ .screenshot loading=lazy }

A new ODE system starts with the 1D default equation; replace it with your own.

### Fractional equations

`frac(u)` applies a fractional operator of order α to output `u` (DeepXDE's fractional PDE). While an
equation uses `frac(...)`, a row **Fractional  α:**, **Resolution:**, **Mesh:** appears. α can be a
number, a constant or the name of an unknown parameter (to learn α). See the tooltip and the
[ODEs and PDEs reference](../reference/equations.md).

## Named constants

To keep equations readable, define constants with **Add constant**: a name and a value, for example
`Re = 20` and `nu = 1/Re`. A value can be a number or an expression in math functions and the constants
defined above it. The computed value is shown next to it (or a warning sign if it cannot be evaluated).
Constants can be used in the equations, boundary and initial conditions, transforms and the
exact-solution formula of Error Analysis.

!!! info "New in v1.6.0 (in development)"
    Every named constant is a parameter of the [parameter sweep](sweeps.md) (category **Constants**).
    Constants defined from a swept constant follow it.

## Inverse problems

In an inverse problem the names of the unknown parameters (Inverse problem page) can be used in the
equations like constants, for example `du_t - D*du_xx`. See
[Inverse problems and measured data](inverse.md).

## Settings on this page

| GUI label | Meaning | Default | Saved as |
|---|---|---|---|
| **Outputs:** | Number of network outputs (1–10) | 1 | `num_outputs` |
| **Equations:** | Number of residuals (1–10); follows Outputs when that changes | 1 | `num_equations` (0 when equal to `num_outputs`) |
| **∫ points:** | Quadrature points for `integral(...)` (ODE only, 2–200) | 20 | `integral_points` |
| **Constants:** / **Add constant** | Named values `name = value` | none | `constants` (JSON list) |
| **Output N name:** | Name of output N | `u`, `v`, `w`, `p`, then `u5`, … | `output_names` (comma-separated) |
| **PDE N (residual = 0):** | Residual expression N | 1D: `du_t - 0.4 * du_xx` | `pde_expressions` (separated by a vertical bar) |
| **Fractional  α:** | Order of `frac(...)` | 1.5 | `fractional_alpha` |
| **Resolution:** | Fractional discretisation | 100 | `fractional_resolution` |
| **Mesh:** | `dynamic` or `static` (static: 1D interval only) | dynamic | `fractional_meshtype` |
