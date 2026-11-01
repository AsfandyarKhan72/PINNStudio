---
search:
  boost: 2
---

# IC pre-training and guided learning

PINNStudio has two ways of fitting the network to the initial state before the main training:

- **IC pre-training** (Training page): once, before the first training phase, the network is trained on the
  initial-condition loss alone.
- **IC-guided learning** (time windows, Adaptive training page): before each time window, the network is
  fitted to the window's starting profile, held constant over the window (Mattey & Ghosh,
  [2022](../about/references.md#mattey2022)).

## What it does

**IC pre-training** builds a separate DeepXDE dataset with a zero residual, no domain or boundary points
and only initial-condition points, trains the network on it, and then starts the training phases from the
pre-trained weights. A pre-trained network can also be restored from a file instead of trained again.

**IC-guided learning** (ICGL) fits, for a chosen fraction of the first Adam phase's iterations, the
network on the window \( [t_0, t_1] \) to the profile it should start from — the initial condition in the
first window, the previous window's solution at \( t_0 \) afterwards — repeated at 10 time levels across the
window. The window's own training phases follow.

## When to use it

IC pre-training: when the main training settles on a trivial solution (e.g. \( u = 0 \)) because the
initial-condition term is fitted too slowly. ICGL: with time windows, especially Mattey & Ghosh's bc-PINN,
where each window should start close to its initial profile.

## How to enable it

- **IC pre-training**: **Training** page → **Initial-condition pre-training** box →
  **Enable IC-guided pre-training**, then **Train from start** (with the fields below) or
  **Restore from checkpoint** (a path to a `.pt` file and **Browse**).
- **IC-guided learning**: **Adaptive training** page → **Method:** **Time Adaptive Training** →
  **IC-guided learning:** (a fraction; 0 = off).

## Settings

### IC pre-training

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Enable IC-guided pre-training** | Pre-train on the initial conditions | off | — | `ic_pretrain` |
| **Train from start** / **Restore from checkpoint** | Train, or load saved weights and skip training | Train from start | — | `ic_pretrain_restore`, `ic_pretrain_restore_path` |
| **Optimizer:** | Adam only | Adam | Adam | `ic_pretrain_optimizer` |
| **Iterations:** | Pre-training iterations | 20,000 | 100–100,000,000 | `ic_pretrain_iterations` |
| **Learning rate:** | Its own learning rate | 0.001 | 10⁻¹²–10 | `ic_pretrain_lr` |
| **Learning rate decay:** | As in a training phase | None | None, Step, Cosine, Exponential, Inverse time | `ic_pretrain_decay_type`, `_decay_p1`, `_decay_p2` |
| **Loss function:** | As in a training phase | MSE | MSE, MAE | `ic_pretrain_loss` |
| **Test points:** | Test points of the pre-training dataset | 10,000 | 100–100,000 | `ic_pretrain_num_test` |
| **Initial points:** | Initial points (hidden when the IC comes from a file) | 1000 | 1–10,000 | `ic_pretrain_num_initial` |

### IC-guided learning (time windows)

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **IC-guided learning:** | Fraction of the first Adam (or AdamW) phase's iterations spent on ICGL in each window | 0.00 (off) | 0–0.9 | `ta_icgl_fraction` |

## Formulation

**IC pre-training.** With the initial conditions \( u_{0,i} \) given as loss terms (or the IC file for
output 0):

\[
\min_\theta\; \sum_i 1000\;\ell\big(u_{\theta,i}(\mathbf{x}, t_{\min}) - u_{0,i}(\mathbf{x})\big),
\]

with weight 0 on the (zero) PDE terms and 1000 on every IC term, on the pre-training's initial points. The
model is saved under the save folder as `ic_pretrain/ic_pretrain_model-<iterations>.pt` (a temporary
folder without a save folder). **Restore from checkpoint** loads such a file (`model_state_dict`) into the
network and skips the pre-training. With time windows, pre-training runs before the first window only.

**IC-guided learning.** In window \( [t_0, t_1] \) with starting profile \( v_k(\mathbf{x}) \) for output
\( k \), on the window's spatial grid (the **IC grid resolution:** per axis) and the 10 times
\( \tau_q = \) `linspace(t0, t1, 10)`:

\[
\min_\theta\; \sum_k \frac{1}{10\,N}\sum_{q=1}^{10}\sum_{n=1}^{N}\big(u_{\theta,k}(\mathbf{x}_n, \tau_q) - v_k(\mathbf{x}_n)\big)^2,
\]

for \( \mathrm{round}(f \cdot I_{\mathrm{Adam}}) \) iterations (at least 1), where \( f \) is the fraction
and \( I_{\mathrm{Adam}} \) the iterations of the first Adam/AdamW phase, with that phase's learning rate
(constant) and PDE weight 0. In the first window only the outputs with an initial condition are fitted
(output 0 from the IC file when there is one); in later windows every output is fitted to the previous
window's solution. With a growing time range the profile is the initial condition at \( t_{\text{start}} \).

## Defaults and limitations

- **ICGL uses 10 time levels** per window; this is not configurable.
- ICGL trains with a constant learning rate.
- IC pre-training uses Adam only, and only an Adam-trained pre-training checkpoint can be restored.
- **Restore** is the only way PINNStudio continues from saved weights: a run cannot be resumed from a
  model saved by the training phases (see [Checkpoints](checkpoints.md)).
- The exported script fits only the outputs whose initial condition is active, as Solve's does.

!!! info "New in v1.6.0 (in development)"
    - ICGL takes its iterations and learning rate from the first Adam **or AdamW** phase (an AdamW phase
      was ignored), and fits only the outputs that have an initial condition (an output without one, such
      as Cahn-Hilliard's \( \mu \), was pulled towards a placeholder; an IC file was ignored).
    - IC pre-training with no initial condition as a loss term is refused (it trained on an identically
      zero loss). With time windows, Solve's script pre-trains only the outputs whose IC is on, as the
      exported script does.

## Supported problem types

IC pre-training: time-dependent 1D, 2D and 3D problems (the option is hidden for a stationary problem),
forward and inverse. It is refused for an ODE (an ODE's initial values are boundary rows of the problem
PINNStudio solves, so there is no initial-condition term to pre-train) and for a stationary problem
opened from a file. ICGL: time windows (forward problems).

## Compatibility with other features

Refused by `validate()`:

> IC pre-training fits the network to the initial conditions given as loss terms, but no output has one
> (an exact initial condition needs no pre-training) -- untick IC pre-training.

> IC pre-training needs a time-dependent PDE: an ODE's initial values are conditions at t0, not
> initial-condition loss terms -- untick IC pre-training. (For a stationary problem: "… a steady problem
> has no initial condition …")

> IC pre-training: 'Restore' is chosen but no saved pre-training model is given -- browse to the .pt file,
> or choose to pre-train.

> IC pre-training: the saved pre-training model '…' does not exist -- check the path, or choose to
> pre-train.

> frac(...) problems can't use RAR, Time-Adaptive training or IC pre-training …

> Time-adaptive training: the initial-condition guided learning fraction must be from 0 (off) to below 1.

ICGL needs no transfer learning, but it is meant for one network carried through the windows (bc-PINN).
**Scale by initial losses** uses the losses of the pre-trained network.

## Common mistakes and troubleshooting

- **"Restore from checkpoint" is refused**: the path is empty or the file does not exist; Solve stops
  before training with one of the messages above.
- **ICGL fraction too large**: the iterations are taken from the first Adam phase and come *on top of* the
  window's phases; 0.1–0.25 is what the Mattey & Ghosh settings use.

## Example

**Cahn-Hilliard 1D – Mattey & Ghosh** (ICGL 0.1), **Allen-Cahn 2D – Mattey & Ghosh** and
**Cahn-Hilliard 2D – Mattey & Ghosh** with **Settings: Default** (ICGL 0.25 and 0.2, with bc-PINN) in
**Examples**. No shipped example uses IC pre-training. See
[Tutorial 12: Causal and time-window training](../tutorials/12-causal-windows.md).

## See also

- [Theory › IC pre-training](../theory/time-dependent.md#ic-pre-training)
- [Theory › Initial-condition guided learning](../theory/time-dependent.md#initial-condition-guided-learning)
- Examples that use it: [Cahn-Hilliard 1D – Mattey & Ghosh](../examples/1d/cahn-hilliard-1d-mattey-ghosh.md), [Allen-Cahn 2D – Mattey & Ghosh](../examples/2d/allen-cahn-2d-mattey-ghosh.md), [Cahn-Hilliard 2D – Mattey & Ghosh](../examples/2d/cahn-hilliard-2d-mattey-ghosh.md)
