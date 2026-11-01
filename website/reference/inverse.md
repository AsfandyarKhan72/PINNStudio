---
search:
  boost: 2
---

# Inverse problems

## What it does

An inverse problem learns unknown quantities of the equations together with the solution, from measured
data. PINNStudio supports two kinds of unknown:

- **Unknown constants** (a diffusion coefficient, a reaction rate, a wave speed): each is a scalar
  `dde.Variable` trained along with the network, as in Raissi et al.
  ([2019](../about/references.md#raissi2019)) and DeepXDE's inverse examples
  ([Lu et al., 2021](../about/references.md#lu2021deepxde)).
- **Unknown fields** (a source term \( f(x) \), a space-dependent rate): an extra network output that
  appears in the equations and has no boundary or initial condition of its own; see
  [Coupled systems](coupled.md).

## When to use it

When some coefficient or field of the model is not known but the solution (or an expression of it) has
been measured at some points.

## How to enable it

1. **Problem** page → **Problem Type** → **Inverse**. The **Inverse problem** page appears in the model
   tree, and **Parameter Convergence** becomes the selected plot type on the **Results** page.
2. **Inverse problem** page → **Unknown parameters**: one row per unknown (**Name**, **Initial guess**,
   **True value**); **➕ Add parameter** adds a row.
3. Write each name into the equations on the **Equations** page, e.g. `du_t - D*du_xx`. A Quick Example
   loaded in Inverse mode puts its first unknown into the equation itself; any further one you add must be
   written in by hand (the page says so).
4. **Measured data**: one block per data file (see [Observational data](observations.md)).

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Name** | The unknown's name as used in the equations | `trainable_variable_1` | a Python name (letters, digits, `_`, not starting with a digit, not a keyword) | `inverse_variables_json` (`name`) |
| **Initial guess** | Value when training starts | 1.0 | any number | `inverse_variables_json` (`init`) |
| **True value** | Known answer, drawn as a dashed line on the convergence plot and used for the parameter error; never used in training | blank | optional number | `inverse_variables_json` (`true`) |
| **Save parameters:** (Results page) | How often the unknowns are recorded | Every 100 iters | No, Every 100 iters, Every 1000 iters, or any number of iterations typed in | `inv_param_save` |

The measured-data settings are listed on [Observational data](observations.md).

## Formulation

The unknowns \( \boldsymbol\lambda \) enter the residual, and every measured-data file adds a data term
(a `dde.icbc.PointSetBC`, or a `PointSetOperatorBC` for a custom expression):

\[
\mathcal{L}(\theta, \boldsymbol\lambda) = \sum_k w_k^{\mathrm{PDE}}\,\ell\big(\mathcal{N}_k[u_\theta;\boldsymbol\lambda]\big)
 + \sum_j w_j^{\mathrm{BC}}\,\ell\big(\mathcal{B}_j[u_\theta]\big)
 + \sum_i w_i^{\mathrm{IC}}\,\ell\big(\cdot\big)
 + \sum_m w_m^{\mathrm{data}}\,\ell\big(g_m(u_\theta)(\mathbf{x}_n) - d_n\big)_{n=1}^{N_m},
\]

where \( g_m \) is the measured output (or the custom expression of the outputs) of file \( m \), and
\( (\mathbf{x}_n, d_n) \) its points and values.

Every training phase compiles the model with `external_trainable_variables` set to the unknowns, so
Adam, L-BFGS and the [RAR](rar.md) rounds all update them. The values are written to
`<name>_convergence.txt` (iteration, value) in the run folder at the **Save parameters** period, and the
final value is printed as `Final inferred <name>: …`.

## Defaults and limitations

- **No bounds or positivity.** An unknown is an unconstrained real number. If it must be positive, train
  its logarithm instead: name the unknown `logD` and write `exp(logD)` in the equation, e.g.
  `du_t - exp(logD)*du_xx`; the identified value is then \( e^{\text{logD}} \).
- A poor initial guess, too little data or a data weight that is too small can leave an unknown
  unidentified; the convergence plot shows whether it has settled.
- Time-adaptive training is not offered in Inverse mode: the **Time Adaptive Training** method is
  removed from the **Adaptive training** page while **Inverse** is selected (it does not train the
  unknowns window by window).

## Supported problem types

ODE, 1D, 2D and 3D, stationary and time-dependent. The fractional order α of `frac(...)` can itself be an
unknown (see [ODEs and PDEs](equations.md#fractional-operator-frac)).

## Compatibility with other features

Refused by `validate()`:

> Inverse mode needs at least one observed-data file to fit against, but no observation row has a file
> selected.

> Trainable variable "…" can't be used as a name in the equations: use letters, digits and _ (not
> starting with a digit), and not a Python word such as lambda.

Time windows are not available (see above). RAR, per-point weights, causal training, exact constraints,
IC pre-training and parameter sweeps (including the unknowns' initial guesses and the data weights, see
[Sweeps](sweeps.md)) can be used.

## Common mistakes and troubleshooting

- **The unknown never changes**: its name is not in the equation (check the equation preview on the
  **Equations** page, which marks names it does not know), or a second unknown was added but not written
  into the equation.
- **Training is slow**: every measured point is used at every step and, with **Check the equation at the
  measured points too (anchors)** on, is also a collocation point. Reduce the data with **Use:** or untick
  the anchors (see [Observational data](observations.md)). The Training Log prints a note when the
  measured data has 20,000 points or more.

## Example

**Diffusion 1D – unknown coefficient**, **Navier-Stokes 2D – Kovasznay flow** (Inverse) and
**Poisson 1D – unknown forcing field** (an unknown field) in **Examples**, category **Inverse**. See
[Tutorial 6: Inverse parameter estimation](../tutorials/06-inverse.md).

## Related

- **Tutorial:** [6. Inverse parameter estimation](../tutorials/06-inverse.md)
- **Examples:** [Diffusion 1D – unknown coefficient](../examples/1d/diffusion-1d-unknown-coefficient.md), [Navier-Stokes 2D – Kovasznay flow](../examples/2d/navier-stokes-2d-kovasznay-flow.md), [Poisson 1D – unknown forcing field](../examples/1d/poisson-1d-unknown-forcing-field.md); more in the gallery under [Inverse problems](../examples/index.md#inverse)
- **Theory:** [Forward and inverse problems](../theory/forward-inverse.md)
- **Method references:** [Raissi et al. 2019](../about/references.md#raissi2019); [Lu et al. 2021](../about/references.md#lu2021deepxde) (DeepXDE)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/inverse.md"
