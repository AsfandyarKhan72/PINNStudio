---
search:
  boost: 2
---

# Coupled systems

## What it does

A coupled system has several outputs (fields) and several equations that link them, e.g. a velocity and
a pressure, a phase field \( \phi \) and a chemical potential \( \mu \), or the real and imaginary parts
of a complex field. PINNStudio trains one network with one output per field (or, with PFNN, one
sub-network per field) and one residual per equation. The number of equations does not have to equal
the number of outputs.

## When to use it

- Systems of ODEs (\( y_1' = y_2,\ y_2' = -y_1 \)).
- Systems of PDEs: Navier–Stokes (\( u, v, p \)), linear elasticity, Cahn–Hilliard split into two
  second-order equations, Schrödinger as two real equations.
- An inverse problem with an unknown *field*: an extra output that appears in the equation but has no
  condition of its own (one equation for two outputs).
- More equations than outputs, e.g. three residual terms on one output.

## How to enable it

**Equations** page → **PDE Definition**: set **Outputs:** (1–10), name each output, set
**Equations:** (1–10; it is reset to the number of outputs whenever **Outputs:** changes) and type one residual per
**PDE *n* (residual = 0):** box. Each output gets its own initial condition on the **Initial conditions**
page, and every boundary-condition row has an **Output:** number.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Outputs:** | Number of fields | 1 | 1–10 | `num_outputs` |
| **Equations:** | Number of residuals | reset to **Outputs:** when that changes | 1–10 | `num_equations` (0 = one per output) |
| **Output *n* name:** | Field names | `u`, `v`, `w`, `p`, `u5`, … | see [ODEs and PDEs](equations.md#output-names) | `output_names` |
| **Network type:** | One shared network or one per output | FNN | FNN, PFNN, MsFFN, STMsFFN | `network_type` |
| **Output:** (BC row) | Which output a boundary condition constrains (1 = first) | 1 | the problem's outputs | `custom_bc_json` (`component`, 0-based) |
| **Points from:** (RAR) | Which equation's residual ranks RAR's new points | All outputs (combined) | All outputs (combined), Output 1, … | `rar_output_selector` (−1 = all) |

## Formulation

The network maps the inputs to \( (u_1, \dots, u_m) \); equation \( k \) may use every output and its
derivatives (`du_x`, `dv_y`, …). Each equation is one PDE loss term:

\[
\mathcal{L}_{\mathrm{PDE}} = \sum_{k=1}^{K} w_k^{\mathrm{PDE}}\,\ell\big(\mathcal{N}_k[u_{1},\dots,u_{m}]\big),
\]

in the order the equations are entered. Boundary and initial conditions are per output. The loss-weight
order is: one weight per equation, one per boundary-condition row, one per output with an initial
condition as a loss term, then one per measured-data file (see [Loss weights](loss-weights.md)).

## Defaults and limitations

- With FNN all outputs share every hidden layer; with PFNN each output has its own hidden layers (see
  [Networks](networks.md)). Outputs of very different size often train better with PFNN or with an
  output scale.
- Derivative names are per output; a derivative of an expression of several outputs is written with the
  chain rule by hand.
- An output with no initial condition (Cahn–Hilliard's \( \mu \)) leaves its **IC for …** box unticked;
  its loss term is then omitted.

## Supported problem types

All dimensions, forward and inverse.

## Compatibility with other features

- **Interface 2D BC** is offered only for a 2D problem with exactly two outputs.
- **Exact periodicity** makes every output periodic along a periodic axis; each output then needs its own
  periodic boundary-condition row on that axis, otherwise Solve stops with
  "Exact periodicity makes every output periodic, but … has no periodic boundary condition in … -- add a
  periodic BC row for it, or untick 'Exact periodicity'."
- **Per-point weights** keep one weight per point and per equation.

## Common mistakes and troubleshooting

- **Two outputs with the same name**, or an output named like a coordinate or a function, is refused
  (see [ODEs and PDEs](equations.md#output-names)).
- **Wrong output in a condition**: the **Output:** number in a boundary-condition row is 1-based in the
  window (1 = the first output).
- **One field dominates the loss**: set the equations' weights separately on the **Loss weights** page,
  or use [Scale by initial losses](loss-weights.md).

## Example

**Linear ODE system** and **Lotka-Volterra ODE** (ODE systems), **Navier-Stokes 2D – Kovasznay flow**
(\( u, v, p \)), **Cahn-Hilliard 2D – Wight & Zhao** (\( \phi, \mu \)), **Schrödinger 1D**
(\( u, v \)), **Linear elasticity 2D – plate** (PFNN), **Black-Scholes 1D – American put** (three
equations for one output) and **Poisson 1D – unknown forcing field** (Inverse: an unknown field) in
**Examples**. See [Tutorial 4: Coupled PDE system](../tutorials/04-coupled.md).

## See also

- [Tutorial 4: coupled system](../tutorials/04-coupled.md)
