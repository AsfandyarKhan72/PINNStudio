# Training phases

The **Training** page sets how the network is trained. It has three boxes: **Training phases**,
**Initial-condition pre-training** and **Callbacks**. Two child pages hang under it in the tree:
[Loss weights](loss-weights.md) and [Adaptive training](adaptive.md).

![The Training page of a new 1D problem: Phase 1 with Optimizer Adam, Iterations 10000, Learning rate 0.001, Learning rate decay None, Loss function MSE; Phase 2 with Optimizer L-BFGS, Iterations 10000, Loss function MSE; Add Phase; Precision float32; the Initial-condition pre-training and Callbacks boxes below](../assets/screenshots/guide_training_default.png){ .screenshot loading=lazy }

Background: [Optimizers and training phases](../reference/optimizers.md),
[IC pre-training and guided learning](../reference/pretraining.md).

## Set up the phases

Training runs in **phases**, one after another; each phase continues from the network the previous one
left. A new problem starts with two: Adam for 10,000 iterations at learning rate 0.001, then L-BFGS for
10,000 iterations. This is the usual recipe: Adam gets close, L-BFGS refines.

For each phase:

- **Optimizer:** **Adam**, **AdamW**, **L-BFGS** or **NNCG**.
- **Iterations:** how many steps (0 to 100,000,000). L-BFGS may stop earlier by itself, once the loss no
  longer improves.
- **Learning rate:** (Adam and AdamW only) the step size.
- **Learning rate decay:** (Adam and AdamW only) lowers the learning rate during the phase:

    | Choice | Its numbers |
    |---|---|
    | **None** | |
    | **Step** | **Step size:** (iterations between drops), **Gamma:** (factor per drop) |
    | **Cosine** | **T_max (iters):** (iterations to fall to the minimum; set it to the phase's iterations), **Eta min:** (the lowest learning rate) |
    | **Exponential** | **Gamma:** (factor per iteration, below 1) |
    | **Inverse time** | **Decay steps:**, **Decay rate:** — DeepXDE's `lr / (1 + rate * step / steps)` |

- **Loss function:** how the residuals are turned into a loss: **MSE** (suits almost every problem) or
  **MAE**. Relative and percentage losses are not offered, because the residual targets are 0; a problem
  saved by an older version with one of them is refused by **Solve** until you choose MSE or MAE.

**Add Phase** adds an Adam phase (50,000 iterations, learning rate 0.001); the ✕ of a phase removes it.
Right-clicking **Training** in the tree also offers **Add training phase**.

!!! note "Eta min of a cosine decay"
    When you switch a phase's decay to **Cosine**, **Eta min** starts at 0.0 (the second number's 0.9 is
    the **Gamma** of a step decay; switching back to **Step** or **Inverse time** sets 0.9 again). If you
    change **Eta min**, keep it below the phase's learning rate, or the learning rate will rise instead of
    fall.

## How the phases connect to other settings

- **Loss weights.** With **Use same weights for all phases** ticked (the default) every phase uses one
  set of weights; unticked, each phase gets its own set on the [Loss weights](loss-weights.md) page.
- **RAR.** Each RAR round trains with the last Adam/AdamW phase's loss function and weights for its Adam
  step and the last L-BFGS phase's for its L-BFGS step ([Adaptive training](adaptive.md)).
- **Time-adaptive training** runs all the phases in every time window. Its IC-guided learning takes its
  learning rate and iterations from the first Adam or AdamW phase; its resampling rounds use the
  phases' learning rate and loss function.
- **Scale by initial losses** (Loss weights page) measures the initial losses with the first phase's
  loss function.
- **Per-point weights** (RBA, SA-PINN) act during the Adam phases; **During L-BFGS** chooses whether
  L-BFGS keeps them.

!!! info "New in v1.6.0 (in development)"
    RAR rounds, time-window guided learning and resampling, and *Scale by initial losses* now use the
    Training phases' optimizers, learning rates, loss functions and weights. Before, they used a hidden
    single-optimizer setting (MSE, the Loss weights page's flat weights, a fixed learning rate).

## Weight decay, AdamW and NNCG

Weight decay (L2 regularization) is set in **Settings › Optimizer Settings...** and applies to the whole
run. L-BFGS and NNCG cannot be combined with a non-zero weight decay, and **AdamW** needs one: **Solve**
refuses either mismatch with an explanation. NNCG needs DeepXDE 1.13.0 or newer.

## L-BFGS options and precision

**Use recommended L-BFGS settings** uses DeepXDE's defaults (maxcor = 100, ftol = 0, gtol = 1e-8,
maxls = 50); each L-BFGS phase runs for its own **Iterations**. To set the options yourself, untick it
(here, when the page shows it, or in **Settings › L-BFGS Options**) and edit **maxcor**, **ftol**,
**gtol**, **maxfun** and **maxls**.

**Precision:** applies to the whole run. **float32** is faster, above all on a GPU; **float64** is more
accurate and lets L-BFGS go further when very small errors are needed. This box and
**Settings › Float Precision...** are one setting: changing either one changes the other.

## Initial-condition pre-training

Tick **Enable IC-guided pre-training** to fit the network to the initial conditions alone before the
phases start; this can help a time-dependent problem start well. Choose:

- **Train from start** — **Optimizer:** (Adam), **Iterations:**, **Learning rate:**,
  **Learning rate decay:**, **Loss function:**, **Test points:** and **Initial points:** (hidden when the
  initial condition comes from a file);
- **Restore from checkpoint** — load an Adam-trained pre-training model (`.pt`) instead of training.

The box is hidden for a stationary problem. Pre-training needs a time-dependent PDE: for an ODE
(whose initial values are conditions at the start of the time interval, not initial-condition loss
terms) **Solve** refuses it with an explanation.

!!! info "New in v1.6.0 (in development)"
    Pre-training fits the initial conditions given as loss terms. When every initial condition is exact
    (built into the network) there is nothing to fit, and **Solve** now refuses with an explanation, as it
    does for pre-training on an ODE or stationary problem.

## Callbacks

Tick **Show Training Callbacks** to show the optional extras. They act during the training phases and
the training steps of RAR rounds, not during IC pre-training:

| Group | Check box | Its settings |
|---|---|---|
| **Early Stopping** | **Stop training when loss stops improving** | **Min. delta:**, **Patience (iters):**, **Baseline loss:**, **Monitor:** (Training loss / Testing loss), **Start after (iters):** |
| **Point Resampling** | **Periodically resample collocation points** | **Resample every (iters):**, **Resample PDE (domain) points**, **Also resample boundary-condition points** |
| **Model Checkpoint** | **Periodically save the model during training** | **Check every (iters):**, **Only save when the monitored loss improves**, **Monitor:** |
| **Training Timer** | **Stop training after a time budget** | **Available time (minutes):** |
| **Training Monitors** | **Watch expressions during training (diagnostic only)** | per monitor: a label, the points (e.g. `(0.3, 0.0); (0.5, 0.0)` in the input order x, y, z, t), the expressions (e.g. `u, du_x`), **Every (iterations):**; **Add Monitor** |

Training monitors only record values (shown afterwards on the Results Overview); they never change
training. Point resampling cannot be combined with per-point weights (RBA, SA-PINN), which belong to
fixed points.

!!! info "New in v1.6.0 (in development)"
    The exported script no longer stops with `NameError: _sol_dir` when **Model Checkpoint** is on, and
    per-point weights together with a point resampler are refused with an explanation (the resampling
    used to reset the weights, silently turning the method off).

## Settings on this page

The phases are saved together as a JSON list in `scheduler_phases`, one object per phase with the keys
`optimizer`, `iterations`, `lr`, `loss`, `weights`, `decay_type`, `decay_p1`, `decay_p2`.

| GUI label | Meaning | Default | Saved as |
|---|---|---|---|
| **Optimizer:** | Phase optimizer | Adam (phase 1), L-BFGS (phase 2) | `optimizer` (`adam`, `adamw`, `lbfgs`, `nncg`) |
| **Iterations:** | Phase length | 10000 | `iterations` |
| **Learning rate:** | Adam/AdamW step size (1e-12–10) | 0.001 | `lr` |
| **Learning rate decay:** | None, Step, Cosine, Exponential, Inverse time | None | `decay_type` |
| **Step size:** / **T_max (iters):** / **Gamma:** / **Decay steps:** | First decay number | 5000 (Step), 0.9999 (Exponential), 1000 (Inverse time) | `decay_p1` |
| **Gamma:** / **Eta min:** / **Decay rate:** | Second decay number | 0.9 (Step, Inverse time), 0.0 (Cosine) | `decay_p2` |
| **Loss function:** | MSE or MAE | MSE | `loss` |
| **Use recommended L-BFGS settings** | DeepXDE's L-BFGS defaults | on | `lbfgs_use_default` |
| **maxcor:**, **ftol:**, **gtol:**, **maxfun:**, **maxls:** | L-BFGS options | 100, 0, 1e-8, 18750, 50 | `lbfgs_maxcor`, `lbfgs_ftol`, `lbfgs_gtol`, `lbfgs_maxfun`, `lbfgs_maxls` |
| **Precision:** | float32 or float64 for the run | float32 | `lbfgs_float_type` |
| **Enable IC-guided pre-training** | Fit the initial conditions first | off | `ic_pretrain` |
| **Train from start** / **Restore from checkpoint** | Pre-training mode | Train from start | `ic_pretrain_restore` |
| **Iterations:** (pre-training) | 100–100,000,000 | 20000 | `ic_pretrain_iterations` |
| **Learning rate:** (pre-training) | | 0.001 | `ic_pretrain_lr` |
| **Learning rate decay:** (pre-training) | as for a phase | None | `ic_pretrain_decay_type`, `ic_pretrain_decay_p1`, `ic_pretrain_decay_p2` |
| **Loss function:** (pre-training) | MSE or MAE | MSE | `ic_pretrain_loss` |
| **Test points:** (pre-training) | 100–100,000 | 10000 | `ic_pretrain_num_test` |
| **Initial points:** (pre-training) | 1–10,000 | 1000 | `ic_pretrain_num_initial` |
| (checkpoint path) | Adam-trained pre-training model | blank | `ic_pretrain_restore_path` |
| **Stop training when loss stops improving** | Early stopping | off | `cb_early_stopping` |
| **Min. delta:** / **Patience (iters):** / **Baseline loss:** / **Monitor:** / **Start after (iters):** | | 0 / 2000 / none / Training loss / 0 | `cb_early_stopping_min_delta`, `_patience`, `_baseline`, `_monitor`, `_start_from` |
| **Periodically resample collocation points** | Point resampling | off | `cb_point_resampler` |
| **Resample every (iters):** / **Resample PDE (domain) points** / **Also resample boundary-condition points** | | 100 / on / off | `cb_point_resampler_period`, `_pde_points`, `_bc_points` |
| **Periodically save the model during training** | Model checkpoint | off | `cb_model_checkpoint` |
| **Check every (iters):** / **Only save when the monitored loss improves** / **Monitor:** | | 1000 / on / Training loss | `cb_checkpoint_period`, `cb_checkpoint_save_better_only`, `cb_checkpoint_monitor` |
| **Stop training after a time budget** / **Available time (minutes):** | Training timer | off / 60 | `cb_timer`, `cb_timer_minutes` |
| **Watch expressions during training (diagnostic only)** | Training monitors | off | `training_monitors_enabled`, `training_monitors` (JSON) |
