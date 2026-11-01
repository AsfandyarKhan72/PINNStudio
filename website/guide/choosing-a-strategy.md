# Choosing a training strategy

PINNStudio offers several ways to change how a PINN is trained: more than one optimizer phase, adaptive
collocation points, per-point and causal weights, exact constraints, time windows, pre-training on the
initial condition, and the loss weights. This page is meant to help you decide which of them to try for a
given problem. It does not repeat the settings; each section links to the reference page (settings,
formulation, refused combinations) and the theory page (background) of the method.

!!! warning "No method is always better"
    Every method on this page was proposed for a particular kind of difficulty, and the papers that propose
    them report improvements on their own test problems. None of them is better on every problem, and each
    of them can make a result worse on a problem that a plain PINN already solves well. Whether a method
    helps depends on the equation, the domain, the network, the number of points and the training budget.
    Treat the suggestions below as things to try, and decide with a comparison against reference data, not
    from the loss alone (see [Compare before you choose](#compare-before-you-choose)).

## Start from a plain run

Before adding a method, train the problem once as plainly as possible: an FNN, the default training phases
(Adam, then L-BFGS), loss terms for the initial and boundary conditions, and no adaptive method. Then look
at three things:

1. **The error against a reference**, in the **Error analysis** tab of the Results area
   ([Error Analysis](error-analysis.md)). Without a reference solution you cannot tell whether a method
   improved the result or only lowered the loss.
2. **The individual loss terms** (**Live loss** tab, **Each loss term**, or the **Loss** tab of
   **Plot settings**). A total loss that is small while one term stays large means that one condition is
   not satisfied.
3. **Where and when the error is large**: in a small region (a front, a shock, a boundary layer), at late
   times, at the initial time, or everywhere.

The answer to the third question is what the decision table below is organised by. The list of known
failure modes of PINNs in [Scope and limitations](../theory/limitations.md) is a useful check at this
point.

## Decision table

| If you see… | consider… | watch out for… |
|---|---|---|
| The loss stops falling under Adam, or the error is still large at the end of Adam | An L-BFGS phase after Adam ([below](#adam-followed-by-l-bfgs)) | Weight decay must be 0; float32 can stop L-BFGS early |
| The error is concentrated in small regions (fronts, shocks, boundary layers) | RAR ([below](#residual-based-adaptive-refinement-rar)) | Extra training rounds; not with `frac(...)`; constant learning rate in the rounds |
| Fronts or interfaces that stay poorly resolved, with a residual that stays large at some points | Per-point weights, RBA or SA-PINN ([below](#per-point-weights-rba-and-sa-pinn)) | Not with point resampling; can worsen an already accurate problem; SA-PINN can outweigh a soft initial condition |
| The initial or periodic condition is the term fitted worst | Exact constraints ([below](#exact-constraints)) | Exact IC needs a formula and is refused with time windows and with your own output-transform formula |
| A time-dependent solution that is right early and wrong later, or that settles on a trivial state | Causal training ([below](#causal-training)) or time windows ([below](#time-windows)) | Causal training needs a time-dependent PDE, and together with per-point weights it did worse than either alone in PINNStudio's own study |
| A long time interval or a stiff evolution (phase fields) that one network follows poorly | Time windows, possibly with transfer learning and bc-PINN ([below](#time-windows)) | Training time grows with the number of windows; errors can accumulate from window to window; forward problems only |
| Training goes to \( u \equiv 0 \) or another trivial solution because the initial condition is fitted too slowly | IC pre-training ([below](#ic-pre-training)), a larger IC weight, or an exact initial condition | Pre-training needs at least one IC given as a loss term |
| Loss terms that differ by orders of magnitude | Hand-set loss weights or **Scale by initial losses** ([below](#loss-weights)) | Automatic scaling is computed once and is refused with time windows |

## Adam followed by L-BFGS

**What it targets.** Adam makes steady progress from random weights but can stall at a loss it no longer
reduces; the quasi-Newton method L-BFGS, started from where Adam ended, often reduces the loss further. Adam
followed by L-BFGS is PINNStudio's default (**Training phases**: phase 1 Adam, phase 2 L-BFGS). Rathore et
al. ([2024](../about/references.md#rathore2024)) study the PINN loss landscape and the combination of
first- and second-order optimizers; the same paper proposes the NNCG optimizer that PINNStudio offers for a
phase. The examples show that the choice depends on the problem and its source: **Burgers 1D – Raissi et
al.** with **Settings: Default** trains with L-BFGS only, as Raissi et al.
([2019](../about/references.md#raissi2019)) did, while the Wight & Zhao and Mattey & Ghosh examples
with **Settings: Default** use Adam and then L-BFGS, following those papers.

**What it costs.** An L-BFGS iteration evaluates the loss and its gradient more than once (line search), and
L-BFGS stores **maxcor:** correction pairs. L-BFGS can also stop well before its iteration count when its
tolerances are met, so its time is hard to predict. More phases are more settings to choose: each has its
own iterations, learning rate and decay, loss function and, optionally, loss weights.

**Limitations in PINNStudio.**

- A run that uses L-BFGS (or NNCG) anywhere, including RAR's L-BFGS step, is refused when weight decay is
  greater than 0 (`_validate_optimizer_settings` in the main window).
- Per-point weights are switched off during L-BFGS by default (**During L-BFGS:** **Plain loss (weights
  off)**), and causal weights are held at their last values; L-BFGS needs a fixed objective.
- In float32, L-BFGS may stop early; float64 (**Precision:**) is slower but can go further.
- RAR rounds, resampling rounds and IC-guided learning use a constant learning rate taken from the phases,
  without the phase's decay.

**Read more.** [Optimizers and training phases](../reference/optimizers.md) ·
[Training phases](training.md) · [Physics-informed neural networks](../theory/pinns.md).

## Residual-based adaptive refinement (RAR)

**What it targets.** Uniformly sampled collocation points can resolve a sharp, localised feature poorly.
RAR adds, after the training phases, the candidate points with the largest PDE residual and trains again,
for a number of rounds. It follows DeepXDE's residual-based adaptive refinement (Lu et al.,
[2021](../about/references.md#lu2021deepxde)); Wu et al. ([2023](../about/references.md#wu2023rar))
compare fixed and residual-based adaptive sampling. **Burgers 1D – RAR** reproduces DeepXDE's Burgers-RAR
example, and **Allen-Cahn 1D – Wight & Zhao** with **Settings: Default** uses RAR for Wight & Zhao's
resampling trial.

**What it costs.** Every round evaluates the residual at all **Residual sampling points:** and then trains
for its **Adam iterations:** and **L-BFGS iterations:** in addition to the training phases, so the total
training time grows roughly with the number of rounds. The training set grows by **Points added per
round:** each round. Settings: rounds, candidates, points per round, iterations per round, an optional stop
tolerance, and which equation ranks the points.

**Limitations in PINNStudio.**

- `frac(...)` problems are refused with RAR.
- RAR and time windows are alternatives in **Method:**; they cannot be combined.
- The rounds train with the last first-order phase's learning rate, without its decay.
- The candidates are drawn uniformly; a probability-weighted variant (RAR-D) is not offered.
- With per-point weights the weights start again at 1 after each round, because they belong to fixed points.

**Read more.** [RAR](../reference/rar.md) ·
[Adaptive sampling](../theory/adaptive-sampling.md#residual-based-adaptive-refinement-rar) ·
[Tutorial 10: RAR](../tutorials/10-rar.md).

## Per-point weights: RBA and SA-PINN

**What it targets.** Points where the residual stays large, typically at fronts and interfaces of phase
fields. Each training point's PDE residual gets its own weight, which grows where the residual stays large:
by an update rule in residual-based attention, **RBA** (Anagnostopoulos et al.,
[2024](../about/references.md#anagnostopoulos2024)), or by gradient ascent in self-adaptive PINNs,
**SA-PINN** (McClenny & Braga-Neto, [2023](../about/references.md#mcclenny2023)). The tuned settings of
**Allen-Cahn 2D – Wight & Zhao** use RBA, and those of **Cahn-Hilliard 1D – Wight & Zhao** and
**Allen-Cahn 2D – Mattey & Ghosh** use SA-PINN, both with the weights kept during L-BFGS.

**What it costs.** One weight per training point and equation. RBA adds an inexpensive update after every
Adam step; SA-PINN trains the weights with their own Adam optimizer, which adds a little more. Settings: the
method, **Decay γ:** and **Step η:** (RBA) or **Weights' learning rate:** (SA-PINN), and **During L-BFGS:**.

**Limitations in PINNStudio.**

- Refused together with the **Point Resampling** callback, and with a Point Set condition that has a batch
  size, because the weights belong to fixed points.
- Refused together with resampling rounds in time windows.
- Only the PDE residual is weighted. With SA-PINN, an initial condition given as a loss term can be
  outweighed; the reference page suggests an exact initial condition with SA-PINN.
- With the default **Plain loss (weights off)**, a run that is mostly L-BFGS sees little of the weights;
  **Keep the weights** holds them at their Adam values. The reference page notes that keeping them made
  already-accurate problems worse in PINNStudio's benchmarks.
- The **Loss weights** page warns when causal training is also on: in PINNStudio's own study the two
  together did worse than either alone.

**Read more.** [RBA and SA-PINN](../reference/point-weights.md) ·
[Adaptive loss weighting](../theory/loss-weighting.md#per-point-weights) ·
[Tutorial 11: RBA and SA-PINN](../tutorials/11-point-weights.md).

## Exact constraints

**What it targets.** A soft condition is only as well satisfied as its loss term is minimised. An exact
(hard) constraint builds the condition into the network output, so it holds whatever the network learns
(Lagaris et al., [1998](../about/references.md#lagaris1998)). PINNStudio offers two: an **exact initial
condition** (**Exact (built into the network)** on the **Initial conditions** page) and **exact
periodicity** (**Exact periodicity (periodic inputs)** on the **Boundary conditions** page), which feeds
\( \cos(2\pi x/L), \sin(2\pi x/L) \) to the network instead of \( x \) (Dong & Ni,
[2021](../about/references.md#dong2021)). Other hard constraints can be written by hand as an
output-transform formula on the **Neural network** page; **Burgers 2D – Mathias et al.** with
**Settings: Default** builds its initial and boundary conditions into the output this way, following
Mathias et al. ([2022](../about/references.md#mathias2022)).

**What it costs.** Almost nothing in time; a loss term disappears. The cost is in flexibility: the network
can no longer trade the condition against the equation, and an exact initial condition with a poorly chosen
blending can make the rest harder to fit. Try the initial condition both exact and as a loss term. An exact
initial condition should also be a smooth formula: a kink (from a `max` or an `abs`) has derivatives the network
cannot represent.

**Limitations in PINNStudio** (all refused by the check that runs before each Solve):

- Exact initial condition: needs a time-dependent PDE, cannot be combined with time-adaptive
  training, needs the condition as a formula in `x`, `y`, `z`, not from a file or as
  `np.`/`x[...]` code, and cannot be combined with your own output-transform formula
 ; write the condition into the formula yourself instead.
- Exact periodicity: needs a periodic boundary condition on a spatial axis, makes
  **every** output periodic, so every output needs a periodic condition on that axis, and cannot
  be combined with the Network page's input transform.
- IC pre-training is refused when every initial condition is exact (nothing to pre-train).

**Read more.** [Exact constraints](../reference/exact-constraints.md) ·
[Theory: exact constraints](../theory/exact-constraints.md) ·
[Tutorial 9: Exact constraints](../tutorials/09-exact.md).

## Causal training

**What it targets.** A time-dependent PINN trained on the whole interval at once can fit late times before
early times are right, and converge to a solution that violates causality. Causal training (Wang, Sankaran &
Perdikaris, [2024](../about/references.md#wang2024causal)) splits the interval into time chunks and
weights each chunk's residual by how well the earlier chunks are already fitted. **Allen-Cahn 1D – Wight &
Zhao** with **Settings: Tuned** combines it with an MsFFN network, an exact initial condition and exact
periodicity on the whole time interval.

**What it costs.** Little extra computation per step (a residual per chunk). Training may need more
iterations, because later times only start to count once earlier ones are fitted. Settings: **Time
chunks:**, and either the ε schedule (**Raise ε step by step (0.01 → 100)**) or a **Fixed ε:**; with a fixed
ε, a suitable value depends on the size of the residuals.

**Limitations in PINNStudio.**

- Refused for a stationary problem and for an ODE; at least 2
  chunks and a positive ε.
- Only the PDE residual at the training points is weighted; during L-BFGS the weights are held fixed.
- Allowed with per-point weights, but the page warns against it (see above).

**Read more.** [Causal training](../reference/causal.md) ·
[Adaptive loss weighting](../theory/loss-weighting.md#causal-training) ·
[Tutorial 12: Causal and time-window training](../tutorials/12-causal-windows.md).

## Time windows

**What it targets.** Long time intervals and stiff or fast-changing evolutions that one network over the
whole interval follows poorly. **Time Adaptive Training** (**Adaptive training** page, **Method:**) splits
the interval into windows and trains them one after another, each starting from the previous window's
solution (Wight & Zhao, [2021](../about/references.md#wight2021)). Options:

- **Transfer learning**: each window starts from the previous window's weights instead of random weights.
- **Growing time range** (Wight & Zhao's "approach I"): each window trains on the whole range so far, with
  the true initial condition.
- **Backward compatibility (bc-PINN)** (Mattey & Ghosh, [2022](../about/references.md#mattey2022)):
  the solution on earlier windows is stored and later windows are held to it.
- **IC-guided learning**: each window first fits its starting profile (also Mattey & Ghosh).
- **Resampling rounds**: after each window's training, points of large residual are added and training
  continues (Wight & Zhao).

The examples cover the variants: **Allen-Cahn 2D – Wight & Zhao** with **Settings: Default** (a new network
per window), **Cahn-Hilliard 1D – Wight & Zhao** with **Settings: Default** (growing range with resampling
rounds), and the Mattey & Ghosh examples with **Settings: Default** (bc-PINN with transfer learning and
IC-guided learning).

**What it costs.** The training phases run in full in every window, so the training time grows roughly with
the number of windows. The solution is passed on through a grid of **IC grid resolution:** points per axis,
which is squared in 2D and cubed in 3D and can exhaust GPU memory in 3D. bc-PINN adds stored points
(**Times/window:** × **Points:**) for every earlier window; a growing range trains on more points in each
window; resampling rounds and IC-guided learning add iterations. This is the method with the most settings.
Errors made in one window are passed on to the next through its starting condition and can accumulate.

**Limitations in PINNStudio.**

- Offered for time-dependent forward problems only (not stationary problems, ODEs or inverse problems).
- **Growing time range** and **Backward compatibility** need **Enable transfer learning**
 ; bc-PINN needs at least one stored time and point.
- Resampling rounds need sensible counts and are refused with per-point weights.
- Refused with an exact initial condition, with **Scale by initial losses** and for
  `frac(...)` problems; the IC-guided learning fraction must be from 0 to below 1.
- The code generator also refuses extra training points (anchors) and an Operator BC at the initial time
  with time windows.
- Resampling rounds and IC-guided learning use a constant learning rate; IC-guided learning uses 10 time
  levels per window.

**Read more.** [Time windows and bc-PINN](../reference/time-windows.md) ·
[Time-dependent training](../theory/time-dependent.md) (including
[Choosing between the options](../theory/time-dependent.md#choosing-between-the-options)) ·
[Adaptive training](adaptive.md) · [Tutorial 12](../tutorials/12-causal-windows.md).

## IC pre-training

**What it targets.** A run that settles on a trivial solution (for example \( u \equiv 0 \)) because the
initial-condition term is fitted too slowly. **Enable IC-guided pre-training** (**Training** page) trains the
network on the initial-condition loss alone before the first training phase, or restores such a pre-trained
network from a file. No shipped example uses it, so there is no example setting to start from.

**What it costs.** Its own Adam iterations (20,000 by default) before the main training, and its own
settings: iterations, learning rate and decay, loss function, test and initial points.

**Limitations in PINNStudio.**

- Needs a time-dependent PDE and at least one initial condition given as
  a loss term; refused for `frac(...)` problems.
- **Restore from checkpoint** needs an existing `.pt` file.
- Adam only; it does not use the training callbacks. With time windows it runs before the first window
  only.

**Read more.** [IC pre-training and guided learning](../reference/pretraining.md) ·
[Time-dependent training](../theory/time-dependent.md#ic-pre-training).

## Loss weights

**What it targets.** Loss terms of very different size, or whose gradients differ greatly, let one term
dominate and the others are fitted poorly (Wang, Teng & Perdikaris,
[2021](../about/references.md#wang2021gradient), discussed in
[Scope and limitations](../theory/limitations.md)). The **Loss weights** page sets one weight per term by
hand, for all phases or per phase, or sets them from the initial losses with **Scale by initial losses**
(\( w_i = c / L_i(0) \)), as in DeepXDE's wave example (**Wave 1D – space-time multi-scale Fourier
features**). A new time-dependent problem starts with an initial-condition weight of 100; the shipped inverse
examples use 100 for the measured data. Gradient-norm balancing as in Wang, Teng & Perdikaris is not offered
in PINNStudio.

**What it costs.** No extra training time. Hand-set weights are a choice per term (and per phase), which is
what makes them hard to tune; **Scale by initial losses** costs one evaluation of the loss before training.

**Limitations in PINNStudio.**

- **Scale by initial losses** needs \( c > 0 \) and is refused with time-adaptive training
 . Its weights are computed once and do not adapt.
- A loss weight can be swept only while **Use same weights for all phases** is on.
- RAR rounds use the weights of the last first-order and the last L-BFGS phase; time windows map the weights
  onto each window's terms.

**Read more.** [Loss weights reference](../reference/loss-weights.md) ·
[Loss weights guide](loss-weights.md) ·
[Adaptive loss weighting](../theory/loss-weighting.md#scaling-by-the-initial-losses).

## Combinations that are refused

PINNStudio checks the settings before every Solve (and every run of a sweep) and refuses these combinations,
with a message that says what to change. The combinations that concern this page:

- Causal training on a stationary problem or an ODE
- Growing time range or bc-PINN without transfer learning
- Time-window resampling rounds with per-point weights
- Per-point weights with the Point Resampling callback or a batched Point Set condition
- IC pre-training with no initial condition as a loss term
- IC pre-training on a stationary problem or an ODE
- Exact initial condition on a stationary problem or an ODE
- Exact initial condition with time-adaptive training
- Exact initial condition from a file, or not as a formula in x, y, z
- Exact initial condition with your own output-transform formula
- Exact periodicity without a periodic condition for every output, or with the input transform
- `frac(...)` with RAR, time-adaptive training or IC pre-training
- **Scale by initial losses** with time-adaptive training

The messages themselves are quoted on the reference pages and in [Troubleshooting](../help/troubleshooting.md).

## Compare before you choose

Because the effect of a method depends on the problem, compare it with the plain run on the same problem,
with the same budget where that is possible, and judge it by the error against reference data:

- A [Parameter sweep](sweeps.md) can switch methods on and off: its **Training Methods** category offers
  **Per-point weights**, **Causal training**, **Exact periodicity**, **Exact initial condition** and
  **Time-adaptive training**, and the **Training** category **Random seed** and
  **L-BFGS phase at the end: iterations (0 = none)**. Every run's settings are checked as Solve checks them;
  a combination that is refused is skipped and marked *invalid*. RAR and IC pre-training cannot be switched
  on and off in a sweep; run them separately and compare the runs in **History** with **Compare**.
- Set up [Error Analysis](error-analysis.md) first, so that every run reports its relative L₂ error. With
  **Bayesian search**, minimise the **L2 error** rather than the final loss when the loss terms differ
  between the runs.
- A single run does not show how much a result varies. When two settings give similar errors, repeat them
  with other seeds (**Random seed** in the sweep) before concluding that one is better.
- Prefer the simpler setup when two give about the same error: it has fewer settings to justify and usually
  trains faster.

[Tutorial 13: Comparing configurations](../tutorials/13-compare.md) walks through such a comparison, and
[Sweeps and Bayesian optimization](../reference/sweeps.md) lists every sweepable setting.
