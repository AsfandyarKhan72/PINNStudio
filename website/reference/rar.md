---
search:
  boost: 2
---

# RAR

## What it does

Residual-based adaptive refinement (RAR) adds collocation points where the equation is fitted worst. After
the training phases, each round samples many random candidate points in the domain, evaluates the PDE
residual of the current network there, adds the points with the largest residual to the training points
(DeepXDE's `data.add_anchors`), and trains again — as in DeepXDE's Burgers-RAR example
([Lu et al., 2021](../about/references.md#lu2021deepxde)); see also Wu et al.
([2023](../about/references.md#wu2023rar)) for a comparison of residual-based sampling methods.

## When to use it

When the error is concentrated in small regions (shocks, fronts, boundary layers) that uniformly sampled
points resolve poorly.

## How to enable it

**Training** → **Adaptive training** page → **Method:** **Residual-based Adaptive Refinement (RAR)**.
**About RAR** opens a short description.

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **RAR training rounds:** | Number of refinement rounds | 3 | 1–1000 | `rar_cycles` |
| **Residual sampling points:** | Random candidate points per round | 50,000 | 1000–10,000,000 | `rar_candidates` |
| **Points added per round:** | Candidates with the largest residual added each round | 500 | 1–10,000 | `rar_add_points` |
| **Adam iterations:** | First-order training after each round's points are added | 20,000 | 0–100,000,000 | `rar_adam_iters` |
| **L-BFGS iterations:** | L-BFGS training after the Adam step of each round (0 = none) | 10,000 | 0–100,000,000 | `rar_lbfgs_iters` |
| **Stop below mean residual:** | End the rounds once the mean \|residual\| of the candidates is below this | off (0) | ≥ 0 | `rar_tolerance` |
| **Points from:** | Rank candidates by all equations' residuals together, or by one equation's | All outputs (combined) | All outputs (combined), Output 1, … | `rar_output_selector` (−1 = all, else the equation index) |

The method itself is `adapt_method = "RAR"`.

## Formulation

After the training phases, round \( r = 1, \dots, R \):

1. Draw \( N_c \) candidates \( \mathbf{x}_c \) uniformly in the domain (`geomtime.random_points`, inside the
   real geometry for any shape).
2. Compute the residual score
   \( s(\mathbf{x}_c) = \sum_k |r_k(\mathbf{x}_c)| \) (all equations), or \( |r_k(\mathbf{x}_c)| \) for the
   equation chosen in **Points from:**.
3. If **Stop below mean residual** is set and \( \bar{s} < \) tolerance, stop.
4. Add the \( N_a \) candidates with the largest \( s \) to the training points.
5. Train with the optimizer, learning rate, loss function and loss weights of the **last first-order
   training phase** for the **Adam iterations:**, then (if > 0) with L-BFGS for the **L-BFGS iterations:**,
   with the loss function and weights of the last L-BFGS phase.

The training callbacks, the inverse unknowns, causal training and the training monitors are carried into
the rounds. Each round saves diagnostics under `solution_results/rar_rounds/round_NN/` (a solution
snapshot, a plot of the collocation points with the new ones marked, and — with Error Analysis reference
data — a comparison and its relative L2 error), and the models of its Adam and L-BFGS steps.

## Defaults and limitations

- **The rounds use a constant learning rate**: the last first-order phase's learning rate, without its
  decay schedule.
- If the last first-order phase is NNCG, the round uses the last Adam/AdamW phase instead (NNCG takes no
  learning rate).
- The candidates are sampled uniformly; there is no probability-weighted sampling (RAR-D).
- With L-BFGS iterations > 0 the run uses L-BFGS, so weight decay must be 0.

!!! info "New in v1.6.0 (in development)"
    RAR rounds train with the loss function and loss weights of the Training Phases (they used the hidden
    single-optimizer setting, MSE, and the Loss Weights panel's weights), in both scripts. With causal
    training, only the training points are causally weighted, so the candidates are ranked by their plain
    residual.

## Supported problem types

ODE, 1D, 2D and 3D; stationary and time-dependent (RAR needs no time axis); forward and inverse.

## Compatibility with other features

Refused by `validate()`:

> frac(...) problems can't use RAR, Time-Adaptive training or IC pre-training (DeepXDE's fractional data
> keeps its own collocation points).

RAR and time windows are alternative **Method:** choices. With per-point weights, the weights start over
whenever new points are added (the weights belong to fixed points). RAR's L-BFGS step counts as L-BFGS in
the weight-decay check.

## Common mistakes and troubleshooting

- **Rounds add points but the error does not improve**: the round's Adam step may be too short, or the
  learning rate of the last first-order phase too large without its decay; lower it or add L-BFGS
  iterations.
- **Long run times**: every round evaluates the residual at all candidates and trains again; reduce
  **Residual sampling points:** or the iterations per round.

## Example

**Burgers 1D – RAR** in **Examples** (DeepXDE's Burgers-RAR problem, with 3 rounds of the 100 points
of largest residual out of 100,000 candidates and no tolerance: DeepXDE's own script adds one point per
round until the mean residual is below 0.005, which here stops after a single point), and
**Allen-Cahn 1D – Wight & Zhao** with **Settings: Default** (Wight & Zhao's resampling trial: 6 rounds of
200 points). See [Tutorial 10: RAR](../tutorials/10-rar.md).

## See also

- [Theory › Adaptive sampling](../theory/adaptive-sampling.md#residual-based-adaptive-refinement-rar)
- [Tutorial 10: RAR](../tutorials/10-rar.md)
- Examples that use it: [Allen-Cahn 1D – Wight & Zhao](../examples/1d/allen-cahn-1d-wight-zhao.md), [Burgers 1D – RAR](../examples/1d/burgers-1d-rar.md)
