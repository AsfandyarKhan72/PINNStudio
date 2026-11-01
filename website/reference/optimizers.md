# Optimizers and training phases

## What it does

Training runs as a list of **training phases**, one after another on the same network: each phase compiles
the DeepXDE model with its own optimizer, iterations, learning rate and decay, loss function and (optionally)
loss weights, and trains it (`model.compile(...)`, `model.train(...)`). The usual pattern is Adam followed by
L-BFGS. This page also covers the L-BFGS options, the floating-point precision, the training callbacks and
the random seed.

## When to use it

Always: every run is defined by its phases. Change them to train longer, to add or remove an L-BFGS phase,
to use a learning-rate schedule, or to change the loss function.

## How to enable it

**Training** page → **Training phases** box: one block per phase (**── Phase *n* ──**); **➕ Add Phase**
adds an Adam phase (50,000 iterations, learning rate 0.001), **✕** removes one. The callbacks are in the
**Callbacks** box (**Show Training Callbacks**). L-BFGS options: **Settings › L-BFGS Options**. Precision:
**Precision:** under the phases, or **Settings › Float Precision...**. Seed: **Settings › Random Seed...**.

## Settings

### Training phases

| Label | Meaning | Default | Range | Config field (`scheduler_phases` entry) |
|---|---|---|---|---|
| **Optimizer:** | Adam, AdamW, L-BFGS or NNCG | phase 1 Adam, phase 2 L-BFGS | — | `optimizer` (`adam`, `adamw`, `lbfgs`, `nncg`) |
| **Iterations:** | Iterations of this phase (L-BFGS: its maximum) | 10,000 and 10,000 | 0–100,000,000 | `iterations` |
| **Learning rate:** | Initial learning rate (hidden for L-BFGS and NNCG) | 0.001 | 10⁻¹²–10 | `lr` |
| **Learning rate decay:** | None, Step, Cosine, Exponential, Inverse time (hidden for L-BFGS and NNCG) | None | — | `decay_type` |
| **Step size:** / **Gamma:** (Step) | Multiply the rate by γ every *s* iterations | 5000 / 0.9 | ≥ 1 / ≥ 0 | `decay_p1` / `decay_p2` |
| **T_max (iters):** / **Eta min:** (Cosine) | Cosine decay to η_min over T_max iterations | the values in the fields (5000 / 0.9 on a new phase) | ≥ 1 / ≥ 0 | `decay_p1` / `decay_p2` |
| **Gamma:** (Exponential) | Multiply the rate by γ every iteration | 0.9999 | 0–1 | `decay_p1` |
| **Decay steps:** / **Decay rate:** (Inverse time) | \( \eta_0/(1 + r\,k/s) \) | the values in the fields (5000 / 0.9 on a new phase; 1000 when switching from Exponential) | ≥ 1 / ≥ 0 | `decay_p1` / `decay_p2` |
| **Loss function:** | Per-term loss | MSE | MSE, MAE, mean l2 relative error, mean absolute percentage error, softplus | `loss` |
| **Use same weights for all phases** (Loss weights page) | One weight list for every phase, or one per phase | on | — | `scheduler_same_weights`, `weights` |
| **Precision:** | float32, or float64 for the whole run | float32 | float32, float64 | `lbfgs_float_type` (and `float_type`) |

The training phases are always on (`optimizer_scheduler = true`); a configuration saved with the older
single-optimizer fields is converted to the equivalent phases.

### L-BFGS options (Settings › L-BFGS Options)

| Label | Meaning | Recommended value | Config field |
|---|---|---|---|
| **Use recommended L-BFGS settings** | Use DeepXDE's defaults below | on | `lbfgs_use_default` |
| **maxcor:** | Number of stored correction pairs | 100 | `lbfgs_maxcor` |
| **ftol:** | Tolerance on the relative change of the loss | 0 | `lbfgs_ftol` |
| **gtol:** | Tolerance on the gradient | 10⁻⁸ | `lbfgs_gtol` |
| **maxfun:** | Maximum loss evaluations | 1.25 × the phase's Iterations | `lbfgs_maxfun` |
| **maxls:** | Maximum line-search steps per iteration | 50 | `lbfgs_maxls` |

The number of L-BFGS iterations is always the phase's **Iterations** (`maxiter` is not set separately).
With the recommended settings, every L-BFGS run sets `maxfun = int(1.25 × Iterations)`, as DeepXDE does
when `maxfun` is unset; with them unticked, your own `maxfun` applies to every L-BFGS run.

### Callbacks (Training page → Callbacks → Show Training Callbacks)

| Group / label | Meaning | Default | Config field |
|---|---|---|---|
| **Early Stopping** → **Stop training when loss stops improving** | DeepXDE `EarlyStopping` | off | `cb_early_stopping` |
| **Min. delta:** | Smallest change that counts as an improvement | 0 | `cb_early_stopping_min_delta` |
| **Patience (iters):** | Iterations without improvement before stopping | 2000 (1–1,000,000) | `cb_early_stopping_patience` |
| **Baseline loss:** | Stop if the loss does not get below this | (none) | `cb_early_stopping_baseline` |
| **Monitor:** | Training loss or Testing loss | Training loss | `cb_early_stopping_monitor` |
| **Start after (iters):** | Ignore the first iterations (needs DeepXDE ≥ 1.12) | 0 | `cb_early_stopping_start_from` |
| **Point Resampling** → **Periodically resample collocation points** | DeepXDE `PDEPointResampler` | off | `cb_point_resampler` |
| **Resample every (iters):** | Period | 100 (1–1,000,000) | `cb_point_resampler_period` |
| **Resample PDE (domain) points** / **Also resample boundary-condition points** | Which points | on / off | `cb_point_resampler_pde_points` / `_bc_points` |
| **Model Checkpoint** | See [Checkpoints](checkpoints.md) | off | `cb_model_checkpoint`, … |
| **Training Timer** → **Stop training after a time budget** | DeepXDE `Timer` | off | `cb_timer` |
| **Available time (minutes):** | Budget | 60 (0.5–100,000) | `cb_timer_minutes` |
| **Training Monitors** → **Watch expressions during training (diagnostic only)** | DeepXDE `OperatorPredictor` at fixed points, every *n* iterations | off | `training_monitors_enabled`, `training_monitors` |

### Random seed (Settings › Random Seed...)

**Fix random seed for reproducible runs** (on) and **Seed:** (2026, 0–2,147,483,647) call
`dde.config.set_random_seed(seed)` at the start of the script, which seeds NumPy, PyTorch and Python's
`random` (config fields `use_random_seed`, `random_seed`). GPU runs are not made bit-for-bit deterministic.

## Formulation

Phase \( p \) minimizes \( \mathcal{L}_p(\theta) = \sum_j w_{p,j}\,\ell_p(e_j) \) starting from the
parameters the previous phase ended with. Learning-rate schedules are PyTorch's (DeepXDE's `decay=`):

| Decay | Learning rate at iteration \( k \) |
|---|---|
| Step | \( \eta_0\,\gamma^{\lfloor k/s\rfloor} \) (`StepLR`) |
| Cosine | \( \eta_{\min} + \tfrac12(\eta_0 - \eta_{\min})\big(1 + \cos(\pi k / T_{\max})\big) \) (`CosineAnnealingLR`; it rises again after \( T_{\max} \), so set \( T_{\max} \) to the phase's iterations) |
| Exponential | \( \eta_0\,\gamma^{k} \) (`ExponentialLR`) |
| Inverse time | \( \eta_0 / (1 + r\,k/s) \) |

L-BFGS is PyTorch's `torch.optim.LBFGS` as configured by DeepXDE's `set_LBFGS_options`; the generated
scripts keep DeepXDE's internal L-BFGS chunk size and evaluation budget in step with these options (they are
otherwise fixed when DeepXDE is imported). NNCG (DeepXDE ≥ 1.13; Rathore et al., [2024](../about/references.md#rathore2024)) runs with
DeepXDE's default NNCG options; it takes no learning rate.

## Defaults and limitations

- Each phase saves its model when a save folder is set (`model_adam-phase1-….pt`, `model_lbfgs-phase2-….pt`;
  see [Checkpoints](checkpoints.md)).
- Callbacks apply to the training phases (fresh instances in each time window). Early stopping and the
  timer stop the current `model.train()` call: early stopping is reset at the start of every phase; the
  timer's budget counts from the start of the first phase, so once it is used up every later phase stops
  at its first check.
- IC pre-training does not use the callbacks.
- **Precision float64** runs the whole training in double precision (slower; L-BFGS usually converges
  further).

## Supported problem types

All.

## Compatibility with other features

Refused before Solve (`_validate_optimizer_settings`):

> ⚠️ Weight decay is set > 0 (Settings > Optimizer Settings), but this run uses L-BFGS and/or NNCG
> somewhere (Training Phases, or RAR's optional L-BFGS polish) — both reject any nonzero weight decay.

> ⚠️ A Training Phase is set to use AdamW, which requires a nonzero weight decay. Set one in Settings >
> Optimizer Settings, or switch that phase to Adam instead.

> ⚠️ A Training Phase is set to use NNCG, which requires deepxde>=1.13.0 …

Refused by `validate()`: an empty phase list ("Optimizer Scheduler is enabled but has no phases
configured …") or every phase at 0 iterations ("Every configured Optimizer Scheduler phase has 0
iterations …"). Per-point weights cannot be combined with the Point Resampling callback (see
[RBA and SA-PINN](point-weights.md)).

RAR rounds, time-window resampling rounds and IC-guided learning take their optimizer and learning rate
from the phases (the last Adam/AdamW phase for rounds, the first Adam/AdamW phase for IC-guided learning)
and train with a **constant** learning rate (no decay).

!!! info "New in v1.6.0 (in development)"
    A problem opened from a file keeps each phase's own loss function (it used to train every phase with
    MSE). RAR and resampling rounds use the phases' loss function and weights, and skip an NNCG phase when
    choosing their optimizer.

## Common mistakes and troubleshooting

- **L-BFGS stops early**: check that `gtol`/`ftol` are not too large (untick the recommended settings to
  see them), or train in float64.
- **Cosine decay rises again**: \( T_{\max} \) is shorter than the phase.
- **Cosine decay with the default Eta min**: a new phase switched to Cosine keeps 0.9 in **Eta min:**,
  which is larger than a typical learning rate (the rate would then *rise* towards 0.9). Set **Eta min:**
  below the learning rate, e.g. 0 or 10⁻⁶.

## Example

**Klein-Gordon 1D** and **Poisson 1D – hard BCs** (an Adam phase with inverse-time decay),
**Burgers 1D – Raissi et al.** with **Settings: Default** (L-BFGS only, Raissi et al.'s settings), and
**Heat 1D – resampling** (the Point Resampling callback) in **Examples**. See the
[Training guide page](../guide/training.md).
