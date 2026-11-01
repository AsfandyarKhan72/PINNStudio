# Exact constraints

## What it does

An exact (hard) constraint builds a condition into the network, so it holds whatever the network learns
and needs no loss term (Lagaris et al., [1998](../about/references.md#lagaris1998)) (see [Theory: exact constraints](../theory/exact-constraints.md)). PINNStudio offers
two, both written by the generators as an output or input transform
(`net.apply_output_transform`, `net.apply_feature_transform`):

- **Exact initial condition**: \( u = u_0(\mathbf{x}) + \tau\,N_\theta \) with
  \( \tau = (t - t_{\min})/(t_{\max} - t_{\min}) \), so \( u(\mathbf{x}, t_{\min}) = u_0(\mathbf{x}) \)
  exactly.
- **Exact periodicity**: every axis with a periodic boundary condition enters the network as
  \( \cos(2\pi x/L), \sin(2\pi x/L) \) instead of \( x \), so the solution and all its derivatives are
  periodic with period \( L \) by construction.

Any other hard constraint (e.g. \( u = 0 \) on \( x = 0 \) and \( x = 1 \) as `x*(1-x)*u`) is written by
hand as an output-transform formula on the **Neural network** page (see [Networks](networks.md)).

## When to use it

When a soft initial condition or periodic condition is the part of the loss the network fits worst, in
particular for sharp fronts and phase fields. Making periodicity exact, and trying the initial condition
both exact and as a loss term, is the first step the **Automatic weighting** reference on the
**Loss weights** page suggests for hard problems.

## How to enable it

- **Initial conditions** page → for each output, **Exact (built into the network)**. Ticking it unticks
  and disables **IC for …** (its loss term is removed).
- **Boundary conditions** page → **Exact periodicity (periodic inputs)**. Needs at least one **Periodic
  BC** row; the periodic rows can stay (their loss is then about zero).

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Exact (built into the network)** | Exact initial condition for this output | off | per output | `ic_hard` (comma-separated `True`/`False`) |
| **Exact periodicity (periodic inputs)** | Periodic input features on every periodic axis | off | — | `periodic_inputs` |

## Formulation

**Exact initial condition.** For output \( i \) with the box ticked:

\[
u_i(\mathbf{x}, t) = u_{0,i}(\mathbf{x}) + \tau(t)\,\big(a_i N_{\theta,i}(\mathbf{x}, t) + b_i\big),
\qquad \tau(t) = \frac{t - t_{\min}}{t_{\max} - t_{\min}},
\]

where \( a_i, b_i \) are the output transform's **Scale/shift** values when that transform is on (else 1
and 0). Outputs without the box keep their own scale and shift. The generator writes this as an output
formula, e.g. `(sin(pi*x)) + t*u` for \( t \in [0, 1] \).

**Exact periodicity.** For each spatial axis \( a \) with a Periodic BC row, of length
\( L_a = a_{\max} - a_{\min} \):

\[
a \;\mapsto\; \big(\cos(2\pi a / L_a),\ \sin(2\pi a / L_a)\big),
\]

the other spatial coordinates are passed unchanged and \( t \) is appended last, e.g. on
\( x \in [-1, 1] \): `cos(pi*x); sin(pi*x); t`. The first layer is sized to the number of features. One
harmonic is used (period \( L_a \)); the features are not shifted by \( a_{\min} \), which does not change
the period.

## Defaults and limitations

- Exact periodicity uses **one harmonic** per periodic axis; higher harmonics come from the network's
  nonlinearity.
- Exact periodicity applies to **every output** along a periodic axis (the features are shared).
- The exact initial condition needs the condition as a formula in `x`, `y`, `z` (not a file, not
  `np.`/`x[...]` code).
- The factor \( \tau \) is linear in time; there is no choice of other blending functions.

## Supported problem types

Exact initial condition: time-dependent 1D, 2D and 3D problems. Exact periodicity: 1D, 2D and 3D problems
on an Interval, Rectangle or Cuboid (Periodic BC rows need these shapes), stationary or time-dependent.

## Compatibility with other features

Refused by `validate()`:

> Exact initial condition: needs a time-dependent PDE problem.

> Exact initial condition can't be combined with time-adaptive training yet (each time step starts from
> the previous step's solution, not the formula) -- untick 'Exact' on the Initial Conditions page, or
> switch time-adaptive training off.

> Exact initial condition: needs the initial condition as a formula, not from a file.

> Exact initial condition (output *n*): write the initial condition as a formula in x, y, z (e.g.
> sin(pi*x)), not ….

> Exact initial condition can't be combined with your own output transform formula -- write the condition
> into the formula yourself (IC + t*u), or turn the formula off.

> Exact periodicity: needs a spatial axis with a periodic boundary condition. (ODE)

> Exact periodicity: no boundary condition is periodic -- add a periodic BC row (or untick 'Exact
> periodicity' on the Boundary Conditions page).

> Exact periodicity makes every output periodic, but … has no periodic boundary condition in … -- add a
> periodic BC row for it, or untick 'Exact periodicity'.

> Exact periodicity can't be combined with the Network page's input transform -- turn the input transform
> off, or write the cos/sin features into its formula yourself.

The **Neural network** page shows a note while either constraint is on. [IC pre-training](pretraining.md)
with every initial condition exact is refused (nothing to pre-train). Both constraints combine with
per-point weights, causal training and MsFFN; with STMsFFN the generated feature list ends with `t`, as
STMsFFN requires.

!!! info "New in v1.6.0 (in development)"
    Exact periodicity with an output that has no periodic boundary condition on that axis is refused (the
    output was silently made periodic before).

## Common mistakes and troubleshooting

- **The solution is periodic although only one output should be**: exact periodicity cannot be limited to
  one output; use soft Periodic BC rows instead.
- **A wrong period**: the period is the domain length along the axis; the domain must span exactly one
  period.

## Example

**Allen-Cahn 1D – Wight & Zhao** with **Settings: Tuned** (exact initial condition and exact periodicity,
with causal training and an MsFFN network), **Cahn-Hilliard 1D – Wight & Zhao** with **Settings: Tuned**
(exact initial condition for the phase field only, exact periodicity, SA-PINN), and
**Diffusion 1D – hard IC and BCs** (a hand-written output-transform formula) in **Examples**. See
[Tutorial 9: Exact constraints](../tutorials/09-exact.md).
