# Exact constraints

A condition given as a loss term is satisfied only approximately, and its weight competes with the residual.
A condition can instead be built into the network's form, so that it holds exactly for every \(\theta\) and
needs no loss term ("hard constraints", an idea going back to
[Lagaris, Likas & Fotiadis, 1998](../about/references.md#lagaris1998); PINNStudio builds them on DeepXDE's output
and feature transforms, [Lu et al., 2021](../about/references.md#lu2021deepxde)). PINNStudio offers two such constraints and,
for anything else, the formula modes of the network's input and output transforms. See
[Exact constraints](../reference/exact-constraints.md).

## Exact initial condition

On the **Initial conditions** page, **Exact (built into the network)** for an output \(i\) with the initial
condition \(u_{0,i}(\mathbf{x})\) replaces that output of the network \(N_\theta\) by

\[
u_{\theta,i}(\mathbf{x}, t) = u_{0,i}(\mathbf{x}) + \tau(t)\, N_{\theta,i}(\mathbf{x}, t),
\qquad
\tau(t) = \frac{t - t_{\min}}{t_{\max} - t_{\min}} .
\]

Since \(\tau(t_{\min}) = 0\), \(u_{\theta,i}(\mathbf{x}, t_{\min}) = u_{0,i}(\mathbf{x})\) exactly; the output's
initial-condition loss term is removed. If the output also has an output scale \(a\) and shift \(b\) (**Enable
output transform** in scale mode), they apply to the network part: \(u_{0,i} + \tau\,(a N_{\theta,i} + b)\).
Outputs without an exact initial condition are unchanged.

The constraint is written into the generated script as an output-transform formula (in
`codegen._exact_constraints`), for example `(sin(pi*x)) + t*u` when \(t_{\min} = 0\), \(t_{\max} = 1\). Both
generators use the same formula.

Conditions for its use (checked before training):

- a time-dependent PDE (not a steady problem, not an ODE);
- the initial condition written as a formula in `x`, `y`, `z` (for example `sin(pi*x)`), not read from a file and
  not in the `np.`/`x[:, 0]` array notation;
- not combined with time-adaptive training (later windows start from the previous window's solution, not from
  the formula), and not with your own output-transform formula (write the condition into the formula
  instead).

\(\tau\) is linear in \(t\). Other choices used in the literature (for example \(1 - e^{-(t - t_{\min})}\) or a
learned distance function) are not offered as options but can be written as an output-transform formula.

## Exact periodicity

**Exact periodicity (periodic inputs)** on the **Boundary conditions** page replaces every coordinate \(x_a\)
that has a periodic boundary condition by two input features,

\[
x_a \ \longmapsto\ \left(\cos\frac{2\pi x_a}{L_a},\ \sin\frac{2\pi x_a}{L_a}\right),
\qquad L_a = x_{a,\max} - x_{a,\min},
\]

and passes the other coordinates (and \(t\)) unchanged. Any function of these features is \(L_a\)-periodic in
\(x_a\), so the network and all its derivatives are periodic by construction and the periodic boundary
conditions hold exactly. The feature list is written as an input-feature formula, for example
`cos(pi*x); sin(pi*x); t` for \(x \in [-1, 1]\); the first layer of the network is sized to the number of
features. This way of enforcing periodicity exactly is described by
[Dong & Ni (2021)](../about/references.md#dong2021); periodic input embeddings of this kind are also used, for
example, by
[Wang, Sankaran & Perdikaris (2024)](../about/references.md#wang2024causal); PINNStudio uses only the first
harmonic of the period (one cosine and one sine per periodic axis).

Conditions for its use:

- at least one periodic boundary-condition row, on a spatial axis;
- every output must have a periodic condition on that axis (the features make every output periodic);
- not combined with the **Neural network** page's input transform (write the features into its formula
  instead).

The periodic boundary-condition rows themselves remain in the loss; they are satisfied exactly and therefore
contribute (up to rounding) zero.

## User-defined constraints

The **Neural network** page's transforms in formula mode cover other exact constraints:

- an **output transform** formula, one expression per output in the coordinates and the raw outputs, for
  example `x*(1-x)*y*(1-y)*u` for \(u = 0\) on the boundary of the unit square;
- an **input feature** formula, for example `x*sin(y); x*cos(y)`.

Several DeepXDE examples shipped with PINNStudio use these (for example the Lotka-Volterra example's output
transform `r*tanh(t) + 100/ub|p*tanh(t) + 15/ub`, which imposes the initial values). A formula you write is not
checked for whether it really satisfies the condition; verify the trained network against it.
