---
search:
  boost: 2
---

# Loss weights

## What it does

The training loss is a weighted sum of terms: one per equation, one per boundary-condition row, one per
initial condition given as a loss term, and (inverse problems) one per measured-data file. The
**Loss weights** page sets these weights by hand, the same for every training phase or different per
phase, or sets them automatically from the initial losses (\( w_i = c / L_i(0) \)), as in DeepXDE's wave
example. Per-point weights (RBA, SA-PINN) and causal training, which weight *inside* the PDE term, are on
the same page under **Automatic weighting** and have their own pages ([RBA and SA-PINN](point-weights.md),
[Causal training](causal.md)).

## When to use it

When the loss terms differ by orders of magnitude, or one condition is fitted poorly. A larger
initial-condition weight is common for time-dependent problems (a new problem starts with 100). Use
**Scale by initial losses** when the terms start at very different sizes and hand-tuning is difficult.

## How to enable it

**Training** → **Loss weights** page:

- **Use same weights for all phases** (on): one row per term (**PDE 1 (u):**, **BC 1 (…):**,
  **IC 1 (u):**). Unticked, every training phase gets its own block (**── Phase *n* weights ──**, rows
  ending in **P*n***).
- **Scale by initial losses:  wᵢ = c / Lᵢ(0),  c =** with the value of \( c \) (5 by default; the field is
  enabled once the box is ticked). The manual rows are then replaced by a note.
- Measured-data weights are on the **Inverse problem** page (**Loss weight:** of each data file).

## Settings

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **PDE *n* (name):** | Weight of equation *n* | 1 | any number | `loss_weights_multi`, each phase's `weights` |
| **BC *n* (type, output):** | Weight of boundary-condition row *n* | 1 | any number | same |
| **IC *n* (name):** | Weight of output *n*'s initial condition | 100 (new problem) | any number | same |
| **Loss weight:** (Inverse problem page) | Weight of a measured-data file | 100 | any number | `inverse_obs_files_json` (`weight`) |
| **Use same weights for all phases** | One weight list for all phases | on | — | `scheduler_same_weights` |
| **Scale by initial losses … c =** | Automatic weights \( c / L_i(0) \) | off (c = 5) | \( c > 0 \) | `loss_weights_from_initial` (0 = off) |

## Formulation

The weights are given to DeepXDE's `model.compile(..., loss_weights=[...])` in this order:

\[
\big[\,w^{\mathrm{PDE}}_1, \dots, w^{\mathrm{PDE}}_K,\;
w^{\mathrm{BC}}_1, \dots, w^{\mathrm{BC}}_J,\;
w^{\mathrm{IC}}_{i_1}, \dots,\;
w^{\mathrm{data}}_1, \dots, w^{\mathrm{data}}_M\,\big],
\]

with one IC weight for each output whose initial condition is a loss term (an exact initial condition, or
an unticked **IC for …**, has none; a stationary problem has none). The loss is
\( \mathcal{L} = \sum_j w_j\,\ell(e_j) \) with the phase's loss function \( \ell \).

**Scale by initial losses.** Before training, the script compiles the model with the first phase's loss
function, evaluates every term once without training (`model.train(iterations=0)`) and sets

\[
w_i = \frac{c}{\max\big(L_i(0),\,10^{-30}\big)},
\]

so every term starts at size \( c \). These weights replace the manual ones in every training phase. With
IC pre-training, the initial losses are those of the pre-trained network.

## Defaults and limitations

- The automatic weights are computed once, at the start; they do not adapt during training.
- With time windows, the weights set for the first window are mapped onto the later windows' terms: the
  equation and boundary weights as set, each output's previous-solution term the weight of its initial
  condition (1 for an output without one), and the bc-PINN terms the **Weight:** of backward compatibility
  (see [Time windows](time-windows.md)).
- RAR rounds use the weights of the last first-order phase (Adam round) and of the last L-BFGS phase
  (L-BFGS round).
- A loss weight can be swept only while **Use same weights for all phases** is on (see
  [Sweeps](sweeps.md)).

## Supported problem types

All. **Scale by initial losses** is not available with time windows.

## Compatibility with other features

Refused by `validate()`:

> 'Scale by initial losses' isn't available with Time-Adaptive training -- turn one of them off.

> 'Scale by initial losses' needs c > 0.

Per-point weights and causal training combine with the term weights (they act inside the PDE term). The
page warns when causal training and per-point weights are both on: "Causal training and per-point weights
both reweight the same residuals; together they did worse than either alone
(benchmarks/training_methods.md). Pick one."

## Common mistakes and troubleshooting

- **A weight list of the wrong length** (e.g. a hand-edited file): the script prints a warning and uses
  weights of 1 for that phase.
- **The data term of an inverse problem is too weak**: raise its **Loss weight:**; the shipped inverse
  examples use 100.

## Example

**Wave 1D – space-time multi-scale Fourier features** (Scale by initial losses, \( c = 5 \), as in
DeepXDE's wave example) and **Black-Scholes 1D – American put** (hand-set weights for three equations,
two boundary conditions and the initial condition) in **Examples**. See
[Tutorial 13: Comparing configurations](../tutorials/13-compare.md) and the
[Loss weights guide page](../guide/loss-weights.md).

## Related

- **Tutorial:** [13. Comparing training configurations](../tutorials/13-compare.md)
- **Examples:** [Wave 1D – space-time multi-scale Fourier features](../examples/1d/wave-1d-space-time-multi-scale-fourier-features.md), [Black-Scholes 1D – American put](../examples/1d/black-scholes-1d-american-put.md)
- **Theory:** [Adaptive loss weighting](../theory/loss-weighting.md)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/loss-weights.md"
