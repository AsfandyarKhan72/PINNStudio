# RBA and SA-PINN

## What it does

Per-point weights give every training point's PDE residual its own weight \( \lambda_n \), which grows
where the residual stays large — usually at sharp fronts and interfaces. Two methods are available:

- **RBA**, residual-based attention (Anagnostopoulos, Toscano, Stergiopulos & Karniadakis,
  [2024](../about/references.md#anagnostopoulos2024)): after every Adam step,
  \( \lambda \leftarrow \gamma\lambda + \eta\,|r|/\max|r| \). Nothing extra is trained.
- **SA-PINN**, self-adaptive weights (McClenny & Braga-Neto,
  [2023](../about/references.md#mcclenny2023)): the weights are trained by gradient *ascent* on the same
  loss the network minimizes.

Only the PDE residual at the training (collocation) points is weighted; boundary, initial and data terms,
and the test loss, stay as they are.

## When to use it

Where a plain PINN struggles: phase fields (Allen–Cahn, Cahn–Hilliard), sharp fronts. On a problem a plain
PINN already solves well they can make the result worse; compare with a run without them.

## How to enable it

**Training** → **Loss weights** page → **Automatic weighting** → **Per-point weights:**
**Residual-based attention (RBA)** or **Self-adaptive (SA-PINN)**. **During L-BFGS:** then decides what
happens in L-BFGS phases. **About automatic weighting** opens a short guide.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Per-point weights:** | None, Residual-based attention (RBA), Self-adaptive (SA-PINN) | None | — | `point_weights` (`None`, `RBA`, `SA-PINN`) |
| **Decay γ:** (RBA) | Memory of the weights; they level off at about \( \eta/(1-\gamma) \) | 0.999 | \( 0 < \gamma < 1 \) | `rba_gamma` |
| **Step η:** (RBA) | How much \( \|r\|/\max\|r\| \) adds each step | 0.01 | \( > 0 \) | `rba_eta` |
| **Weights' learning rate:** (SA-PINN) | Learning rate of the weights' gradient ascent (Adam) | 0.005 | 10⁻⁵–10 | `sa_lr` |
| **Keep the mean weight at 1** (SA-PINN) | Rescale the weights to mean 1 after every step | off | — | `sa_normalize` |
| **During L-BFGS:** | **Plain loss (weights off)** or **Keep the weights** | Plain loss (weights off) | — | `point_weights_lbfgs` (`plain`, `keep`) |

## Formulation

For equation \( k \) and training point \( n \), the residual becomes \( \lambda_{n,k}\,r_k(\mathbf{x}_n) \),
so with the MSE loss the PDE term is

\[
\mathcal{L}_{\mathrm{PDE},k} = \frac{1}{N}\sum_{n=1}^{N}\big(\lambda_{n,k}\,r_k(\mathbf{x}_n)\big)^2 .
\]

The weights start at 1, one per training point and equation.

**RBA**, once per Adam step (no gradient through \( \lambda \)):

\[
\lambda_{n,k} \leftarrow \gamma\,\lambda_{n,k} + \eta\,\frac{|r_k(\mathbf{x}_n)|}{\max_m |r_k(\mathbf{x}_m)|}.
\]

**SA-PINN**: \( \lambda \) is a trainable tensor updated by Adam with `maximize=True` and the learning rate
**Weights' learning rate:**, after each network step; the weights are clipped at 0. With
**Keep the mean weight at 1**, each equation's weights are divided by their mean after every step (not in
the paper); unticked, they only grow, by about the learning rate per step, as in McClenny & Braga-Neto.

**During L-BFGS** (L-BFGS needs a fixed objective): *Plain loss* drops the weights (L-BFGS refines the
ordinary loss); *Keep the weights* holds them at the values Adam ended with, without updating them.

## Defaults and limitations

- The weights belong to fixed training points: when the points change (a RAR round, the next time window)
  they start again at 1.
- *Plain loss* is the default because keeping the weights made already-accurate problems (1D Burgers,
  Kovasznay flow) worse in PINNStudio's benchmarks; *Keep the weights* helped where the plain PINN
  struggles, and SA-PINN needs it to have a lasting effect (see the tool tips and
  `benchmarks/training_methods.md`).
- SA-PINN weights only the PDE points, so an initial condition given as a loss term can be outweighed;
  use it with an exact initial condition (the tool tip reports \( u = 0 \) learned on 2D Allen-Cahn of
  Wight & Zhao otherwise).

## Supported problem types

All dimensions, stationary and time-dependent, forward and inverse; with RAR and time windows (without
resampling rounds).

## Compatibility with other features

Refused by `validate()`:

> Per-point weights (…) belong to fixed training points, but the Point Resampler callback replaces the
> points every … steps and the weights would start over each time -- turn the Point Resampler off, or set
> Per-point weights to None.

> Per-point weights (…) belong to fixed training points, but a Point Set condition with a batch size
> resamples the points every 100 steps (as in DeepXDE) and the weights would start over each time -- set
> its batch size to 0, or Per-point weights to None.

> Time-adaptive training: resampling adds collocation points, which per-point weights (…) can't follow --
> set Per-point weights to None, or resampling rounds to 0.

> Per-point weights (RBA): the decay gamma must be between 0 and 1. / … the step eta must be positive.
> / Per-point weights (SA-PINN): the learning rate must be positive.

The page warns (without refusing) when causal training is also on: both reweight the same residuals and
together did worse than either alone in PINNStudio's benchmarks.

!!! info "New in v1.6.0 (in development)"
    Per-point weights with a point resampler (the Point Resampling callback, or a batched Point Set
    condition) are refused; the weights used to start over at every resampling, which silently turned the
    method off.

## Common mistakes and troubleshooting

- **The loss grows without bound with SA-PINN**: tick **Keep the mean weight at 1**, or lower the weights'
  learning rate.
- **No effect in a run that is mostly L-BFGS**: with *Plain loss* the weights are off in L-BFGS; use
  *Keep the weights*.
- The Training Log prints the mean and maximum weight every 1000 iterations.

## Example

**Allen-Cahn 2D – Wight & Zhao** with **Settings: Tuned** (RBA, kept during L-BFGS, with time windows and
exact periodicity), and **Cahn-Hilliard 1D – Wight & Zhao**, **Allen-Cahn 2D – Mattey & Ghosh** and
**Cahn-Hilliard 2D – Mattey & Ghosh** with **Settings: Tuned** (SA-PINN, kept during L-BFGS, with an exact
initial condition) in **Examples**. See [Tutorial 11: RBA and SA-PINN](../tutorials/11-point-weights.md).
