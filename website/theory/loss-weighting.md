# Adaptive loss weighting

The terms of the total loss can differ by orders of magnitude, and within the residual term some points or
some times can matter more than others. PINNStudio offers weighting at two levels, all on the **Loss weights**
page (see [Loss weights](../reference/loss-weights.md), [RBA and SA-PINN](../reference/point-weights.md) and
[Causal training](../reference/causal.md)):

- **between terms:** fixed weights \(w_k\), or weights set from the initial losses;
- **inside the residual term** (**Automatic weighting**): per-point weights (RBA, SA-PINN) or causal weights
  in time.

The weights between terms still apply when a residual term is weighted inside.

## Fixed weights

\[
\mathcal{L}(\theta) = \sum_k w_k\,\mathcal{L}_k(\theta),
\]

with one weight per term in the order of [Losses and automatic differentiation](losses.md#the-weighted-total-loss).
The weights can differ between Training Phases.

## Scaling by the initial losses

With **Scale by initial losses: wᵢ = c / Lᵢ(0), c =** ticked, the script evaluates every loss term once before
training (no optimizer step) and sets

\[
w_k = \frac{c}{\max\!\big(\mathcal{L}_k(\theta_0),\ 10^{-30}\big)},
\]

so that every term starts at the value \(c\). \(\mathcal{L}_k(\theta_0)\) is the unweighted term of the initial
network, computed with the loss function of the first Training Phase and, when IC pre-training is on, with the
pre-trained network. These weights replace the manual weights in every phase. This is the scaling of DeepXDE's
wave-equation example, `loss_weights = 5 / initial_losses` (\(c = 5\)). It is not available together with
time-adaptive training.

## Per-point weights

Per-point weights give every training point \(n\) and every equation \(j\) its own weight \(\lambda_{n,j}\) and
multiply the residual by it before the loss function is applied:

\[
\mathcal{L}_{r,j}^{\lambda}(\theta) = \ell\big(\{\lambda_{n,j}\, r_j(\mathbf{z}_n;\theta)\}_n\big)
\ \overset{\text{MSE}}{=}\ \frac{1}{N}\sum_{n=1}^{N} \lambda_{n,j}^2\, r_j(\mathbf{z}_n;\theta)^2 .
\]

Only the training points are weighted; the test loss, RAR candidates and resampling candidates use the plain
residual. All weights start at \(\lambda = 1\). They belong to a fixed point set: when the training points
change (a RAR round, the next time window) the weights start again from 1, and a configuration that redraws
the points during training (the Point Resampling callback, or a point-set condition with a batch size) is
refused. Boundary-condition, initial-condition and data terms are not weighted point by point.

### Residual-based attention (RBA)

[Anagnostopoulos et al. (2024)](../about/references.md#anagnostopoulos2024). At every training step (in every
phase except L-BFGS), before the residual is weighted, the weights are updated from the current residual
without a gradient:

\[
\lambda_{n,j} \leftarrow \gamma\,\lambda_{n,j} + \eta\,\frac{\lvert r_j(\mathbf{z}_n) \rvert}{\max_m \lvert r_j(\mathbf{z}_m) \rvert},
\]

with **Decay γ:** (default 0.999) and **Step η:** (default 0.01). The maximum is taken per equation over all
training points, including the sampled boundary and initial points (DeepXDE evaluates the residual there too, then
leaves those rows out of the PDE loss); the same rows count in the mean used by **Keep the mean weight at 1**. A point whose normalised residual stays at \(a \in [0, 1]\) approaches
\(\lambda = \eta a / (1-\gamma)\), so with the defaults the weights lie between about 0 and 10.

### Self-adaptive weights (SA-PINN)

[McClenny & Braga-Neto (2023)](../about/references.md#mcclenny2023). The weights are trainable and are updated
by gradient **ascent** on the same loss the network minimises,

\[
\min_\theta\ \max_{\lambda \ge 0}\ \mathcal{L}(\theta, \lambda),
\]

with a separate Adam optimizer (PyTorch `Adam(maximize=True)`, learning rate **Weights' learning rate:**,
default 0.005) taking one step after every network step, followed by \(\lambda \leftarrow \max(\lambda, 0)\).
Because the residual is multiplied by \(\lambda\) and squared by the MSE, the effective weight of a point is
\(\lambda^2\), which corresponds to the polynomial mask \(m(\lambda) = \lambda^2\) of the paper. With
**Keep the mean weight at 1** ticked, the weights are divided by their mean (per equation) after every step;
this normalisation is a PINNStudio option, not part of the published method.

!!! note "Difference from the published method"
    McClenny & Braga-Neto also give the boundary and initial points self-adaptive weights. PINNStudio weights
    only the residual points, so an initial condition given as a loss term can be outweighed by large residual
    weights; combine SA-PINN with an [exact initial condition](exact-constraints.md) where possible.

### During L-BFGS

L-BFGS needs a fixed objective. **During L-BFGS:** decides what it minimises: **Plain loss (weights off)**
(the default) drops the point weights, so L-BFGS refines the ordinary loss; **Keep the weights** holds them at
the values the previous phase ended with.

## Causal training

[Wang, Sankaran & Perdikaris (2024)](../about/references.md#wang2024causal) observed that a PINN can fit late
times before early ones, which violates causality for evolution equations, and proposed weighting the residual
by time so that later times count only once earlier times are fitted.

### As implemented

With **Causal training** ticked and **Time chunks:** \(= M\), let \(t_0\) and \(t_1\) be the smallest and largest
time among the current training points (the whole interval, or one time window). Each point \(n\) is assigned
to a chunk

\[
i(n) = \min\!\Big(\Big\lfloor M\,\frac{t_n - t_0}{t_1 - t_0} \Big\rfloor,\ M - 1\Big),
\]

the chunk losses are the mean squared residual summed over the equations,

\[
L_i = \frac{1}{N_i} \sum_{n : i(n) = i}\ \sum_{j} r_j(\mathbf{z}_n)^2,
\]

and the chunk weights are

\[
w_i = \exp\!\Big(-\varepsilon \sum_{k < i} L_k\Big), \qquad w_0 = 1,
\]

computed without a gradient at every training step. Every residual is multiplied by \(\sqrt{w_{i(n)}}\), so
with the MSE the residual term of equation \(j\) becomes

\[
\mathcal{L}_{r,j}^{\text{causal}} = \frac{1}{N} \sum_{n} w_{i(n)}\, r_j(\mathbf{z}_n)^2
= \sum_{i} \frac{N_i}{N}\, w_i\, L_{i,j},
\]

which equals the paper's \(\frac{1}{M}\sum_i w_i L_{i,j}\) when the chunks hold equally many points.

**Raise ε step by step (0.01 → 100)** (on by default) follows the paper's Algorithm 1: \(\varepsilon\) starts
at 0.01 and moves to 0.1, 1, 10 and 100, each time once every chunk weight is above 0.99. Otherwise
**Fixed ε:** is used. The schedule starts again for every new time window. The Training Log reports every
1000 steps the smallest weight and the time up to which all weights are above 0.99. During L-BFGS the weights
are held at their last values.

### Differences from the paper

- The paper uses a fixed temporal grid; PINNStudio bins the (randomly sampled) training points into chunks, so
  the chunk sizes \(N_i\) vary slightly and enter the loss as shown above.
- DeepXDE evaluates the residual at all training points, including the sampled boundary and initial points,
  so these points also enter the chunk losses \(L_i\).
- The chunk losses sum the squared residuals of all equations, while each equation keeps its own loss term.
- The scaling by \(\sqrt{w}\) gives the paper's weighting for the MSE. With the MAE loss the effective weight is
  \(\sqrt{w_i}\) instead of \(w_i\).

Causal training needs a time-dependent PDE (it is refused for a steady problem or an ODE) and at least two
chunks. Combining it with per-point weights is allowed, but both reweight the same residuals; the tooltip in
the window advises against it.
