---
search:
  boost: 2
---

# Forward problems

## What it does

A forward problem solves for the unknown field(s) of a differential equation whose coefficients, domain,
boundary conditions and (for a time-dependent problem) initial condition are all given. PINNStudio
trains a neural network \( u_\theta(\mathbf{x}, t) \) so that the equation residual is small at the
collocation points and the conditions hold at the boundary and initial points, as in Raissi et al.
([2019](../about/references.md#raissi2019)), using DeepXDE ([Lu et al., 2021](../about/references.md#lu2021deepxde)).

## When to use it

Whenever every quantity in the equations is known and you want the solution. If a coefficient or a source
field is unknown and you have measurements, use an [inverse problem](inverse.md).

## How to enable it

**Problem** page → **Problem Type** → **Forward** (the default). Then set the
[equations](equations.md), the [geometry](geometry.md), the [collocation points](../guide/points.md),
the [initial conditions](initial-conditions.md) and the [boundary conditions](boundary-conditions.md),
and press **Solve**.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Forward** / **Inverse** | Problem type | Forward | — | `problem_type` (`"Forward"`, `"Inverse"`) |
| **Time Dependent** / **Stationary** | With or without a time axis | Time Dependent | — | `steady_state` |
| **Domain points:** | Collocation points inside the domain (space–time for a time-dependent problem) | 2000 | 1–10,000,000 | `num_domain` |
| **Boundary points:** | Points on the boundary | 200 | 0–2,000,000 | `num_boundary` |
| **Initial points:** | Points at \( t = t_{\min} \) | 200 | 0–2,000,000 | `num_initial` |
| **Test points:** | Points for the test loss (not trained on) | 1000 | 10–10,000,000 | `num_test` |
| **Point distribution:** | How the points are sampled | Hammersley | Hammersley, uniform, Halton, LHS, Sobol, pseudo | `point_distribution` |
| **Extra training points:** | Additional fixed points (DeepXDE anchors) | None | None, From a file, Random in a box | `anchors_mode`, `anchors_file`, `anchors_count` (1000), `anchors_box` |

## Formulation

With residuals \( r_k = \mathcal{N}_k[u_\theta] \) of the \( K \) equations, boundary conditions
\( \mathcal{B}_j \) and initial conditions \( \mathcal{I}_i \), the training loss is

\[
\mathcal{L}(\theta) = \sum_{k=1}^{K} w_k^{\mathrm{PDE}} \,\ell\big(r_k\big)
 + \sum_{j} w_j^{\mathrm{BC}} \,\ell\big(\mathcal{B}_j[u_\theta]\big)
 + \sum_{i} w_i^{\mathrm{IC}} \,\ell\big(u_{\theta,i}(\cdot, t_{\min}) - u_{0,i}\big),
\]

where \( \ell \) is the phase's loss function (MSE by default, see [optimizers](optimizers.md)) evaluated
at the term's own points, and the weights are those of the [Loss weights](loss-weights.md) page. The
generated script builds `dde.data.TimePDE` on a `GeometryXTime` for a time-dependent problem and
`dde.data.PDE` on the spatial geometry for a stationary one (no initial condition, no initial points).

## Defaults and limitations

- A new problem (**Problem › New**) starts with one output `u`, the equation `du_t - 0.4 * du_xx` (a new ODE
  system: `du_t + 0.4 * u` with `u(0) = 1`), the
  default network (3 hidden layers of 64, tanh) and two training phases (Adam 10,000 iterations at
  learning rate 0.001, then L-BFGS 10,000 iterations).
- Training the network does not guarantee an accurate solution. Check the result against a reference
  with [Error Analysis](metrics.md) whenever one is available.

## Supported problem types

ODE, 1D, 2D and 3D; stationary and time-dependent (an ODE is always solved over \( t \)); every
[geometry](geometry.md) the dimension offers.

## Compatibility with other features

Every training feature can be used with a forward problem, subject to the restrictions on its own page.
Extra training points are refused with time windows:

> Extra training points (anchors) aren't supported with Time-Adaptive training (each step covers only
> part of the time range). Use Adaptation = None or RAR.

## Common mistakes and troubleshooting

- **A condition that selects no point** (for example `left` on a disk, which has no flat side) has a NaN
  loss. Both scripts print which condition is empty and why before training.
- **Test points** only affect the test loss shown in the log and the loss plot; they are not used for
  training.

## Example

**Heat 2D** and **Burgers 1D – Raissi et al.** in **Examples** (Forward). See
[Tutorial 2: 1D Burgers equation](../tutorials/02-burgers.md) and
[Tutorial 3: 2D heat equation](../tutorials/03-heat-2d.md).

## Related

- **Tutorials:** [2. 1D Burgers equation](../tutorials/02-burgers.md), [3. 2D heat equation](../tutorials/03-heat-2d.md)
- **Examples:** [Heat 2D](../examples/2d/heat-2d.md), [Burgers 1D – Raissi et al.](../examples/1d/burgers-1d-raissi-et-al.md); more in the gallery under [Forward problems](../examples/index.md#forward)
- **Theory:** [Forward and inverse problems](../theory/forward-inverse.md)
- **Method references:** [Raissi et al. 2019](../about/references.md#raissi2019); [Lu et al. 2021](../about/references.md#lu2021deepxde) (DeepXDE)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/problem.md"
