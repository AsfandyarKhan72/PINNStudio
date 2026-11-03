# Losses and automatic differentiation

## Derivatives by automatic differentiation

The derivatives in an equation are derivatives of the network with respect to its inputs. The generated
script computes them with DeepXDE's `dde.grad.jacobian` and `dde.grad.hessian`, which call PyTorch's reverse-mode
automatic differentiation (`torch.autograd`) on the network's computational graph. The result is exact for
the network (up to floating-point rounding), not a finite-difference approximation, and it is itself
differentiable with respect to \(\theta\), so the residual can be minimised by gradient descent.

The names used in the equation boxes map to these calls:

| Typed name | Derivative | Computed as |
|---|---|---|
| `du_x`, `du_y`, `du_z`, `du_t` | \(\partial u/\partial x\), ... | `dde.grad.jacobian(y, x, i, j)` (always computed) |
| `du_xx`, `du_yy`, `du_xy`, `du_xt`, ... | \(\partial^2 u/\partial x^2\), \(\partial^2 u/\partial x\,\partial y\), ... | `dde.grad.hessian(y, x, component=i, i=a, j=b)`, only when the name appears in an equation |
| `du_xxxx`, `du_xxyy`, `du_xxtt`, ... | fourth derivatives | the Hessian of the corresponding second derivative |

Here `u` stands for any output name; for an output `phi` the names are `dphi_x`, `dphi_xx`, and so on. The
full syntax, including integral and fractional terms, is on the [Equations guide page](../guide/equations.md).

## Residuals

For each equation \(j\) the residual at a point is the typed expression evaluated with the network,

\[
r_j(\mathbf{x}, t; \theta) = \mathcal{N}_j[u_\theta](\mathbf{x}, t).
\]

For example, the 1D Burgers equation typed as `du_t + u*du_x - 0.01/pi*du_xx` has the residual
\( r = \partial_t u_\theta + u_\theta \partial_x u_\theta - \tfrac{0.01}{\pi}\partial_{xx} u_\theta \). An unknown
coefficient of an inverse problem enters the residual by its name ([Forward and inverse problems](forward-inverse.md)).

## Loss terms

DeepXDE builds one loss term per equation and one per condition. With a point-wise error \(e\) and the loss
function \(\ell\) chosen for the training phase,

\[
\begin{aligned}
\mathcal{L}_{r,j}(\theta) &= \ell\big(\{ r_j(\mathbf{x}, t; \theta) : (\mathbf{x}, t) \in \mathcal{T}_r \}\big)
&& \text{(equation } j\text{)},\\
\mathcal{L}_{b}(\theta) &= \ell\big(\{ \mathcal{B}_b[u_\theta](\mathbf{x}, t) - g_b(\mathbf{x}, t) : (\mathbf{x}, t) \in \mathcal{T}_b \cap \Gamma_b \}\big)
&& \text{(boundary condition } b\text{)},\\
\mathcal{L}_{0,i}(\theta) &= \ell\big(\{ u_{\theta,i}(\mathbf{x}, t_{\min}) - u_{0,i}(\mathbf{x}) : \mathbf{x} \in \mathcal{T}_0 \}\big)
&& \text{(initial condition of output } i\text{)}.
\end{aligned}
\]

For a Dirichlet condition \(\mathcal{B}_b[u] = u_i\); for a Neumann condition the normal derivative
\(\partial u_i/\partial n\); for a Robin condition \(\partial u_i/\partial n - h(\mathbf{x}, u)\); a periodic
condition compares the value (or a derivative) at the two opposite sides; an operator condition is an
arbitrary expression of the outputs and their derivatives; a point-set condition compares the outputs
with given values at given points. Measured data in an inverse problem adds one point-set term per data file
([Forward and inverse problems](forward-inverse.md)).

In DeepXDE the residual set \(\mathcal{T}_r\) is the whole set of sampled training points: the domain points,
the boundary and initial points, and the anchors (extra points, measured-data points when they are anchors,
RAR points). Each condition term uses the subset of the sampled boundary or initial points where its
location holds, or its own point list.

## Loss functions

Each Training Phase has its own **Loss function:** (the IC pre-training block has one too). For an error
vector \(e \in \mathbb{R}^N\) on the term's points:

| Choice | Value | Notes |
|---|---|---|
| `MSE` (default) | \( \ell(e) = \dfrac{1}{N}\sum_{n=1}^{N} e_n^2 \) | DeepXDE's `mean_squared_error` |
| `MAE` | \( \ell(e) = \dfrac{1}{N}\sum_{n=1}^{N} \lvert e_n \rvert \) | DeepXDE's `mean_absolute_error` |

The list offers only these two. The residuals and conditions are compared with a target of zero, so
DeepXDE's mean relative \(L_2\) error and mean absolute percentage error would divide by zero (and the
latter exists only for TensorFlow), and `softplus` is not a DeepXDE loss. A problem saved by an older
version with one of those losses is refused before training with a message asking for MSE or MAE.

## The weighted total loss

The quantity the optimizer minimises is

\[
\mathcal{L}(\theta) = \sum_{j=1}^{J} w_{r,j}\,\mathcal{L}_{r,j}
+ \sum_{b} w_{b}\,\mathcal{L}_{b}
+ \sum_{i \in \mathcal{I}_0} w_{0,i}\,\mathcal{L}_{0,i}
+ \sum_{f} w_{d,f}\,\mathcal{L}_{d,f},
\]

where \(\mathcal{I}_0\) is the set of outputs whose initial condition is a loss term (an
[exact](exact-constraints.md) initial condition, or an output without one, has no term) and the last sum
runs over measured-data files of an inverse problem. The weights are given on the **Loss weights** page, in
the order of this formula: one weight per equation (**PDE 1 (u):**, ...), one per boundary-condition row
(**BC 1 (...)**), one per initial condition, then one per data file. Each Training Phase can have its own
weights unless **Use same weights for all phases** is ticked. The weights can also be set automatically
([Adaptive loss weighting](loss-weighting.md)). See also [Loss weights](../reference/loss-weights.md).

Weight decay (**Weight decay (L2):** in **Settings › Optimizer Settings...**) is passed to the optimizer as its
`weight_decay` parameter, which adds \(\alpha\,\theta\) to the gradient of the network weights (an \(L_2\)
penalty \(\tfrac{\alpha}{2}\lVert\theta\rVert^2\) for Adam; decoupled decay for AdamW). It does not act on the unknowns
of an inverse problem, cannot be combined with L-BFGS or NNCG (DeepXDE refuses it), and is refused
when **Network type:** is PFNN, MsFFN or STMsFFN (DeepXDE's networks of those types have no regularization argument).

## What the Training Log reports

DeepXDE records the weighted terms \(w_k \mathcal{L}_k\) on the training points and on the test points every **Record every:** iterations (**Settings › Plot Settings...**, **Loss** tab). At the end of a run the script prints

\[
\text{Final loss} = \sum_k w_k \mathcal{L}_k,
\qquad
\text{Final loss (unweighted)} = \sum_k \frac{w_k \mathcal{L}_k}{w_k} = \sum_k \mathcal{L}_k ,
\]

from the last recorded training losses (terms with weight zero are left out of the unweighted sum). The
unweighted value makes runs with different weights comparable, for example in a loss-weight sweep. Neither
number measures the error of the solution; see [Error measures](metrics.md).
