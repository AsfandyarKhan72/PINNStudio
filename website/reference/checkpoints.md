---
search:
  boost: 2
---

# Checkpoints and restoring

## What it does

A run with a save folder saves the trained network after every training phase, together with the
configuration it was trained with. The **Model Checkpoint** callback can also save the network during
training, every *n* iterations or only when the monitored loss improves. **Restore & visualize** loads a
saved model and draws its plots (and Error Analysis) again without retraining.

PINNStudio does **not** continue training from a saved model: Restore only predicts and plots. The one
place saved weights are loaded into training is **IC pre-training → Restore from checkpoint**, which loads
a saved IC pre-training model instead of pre-training again (see
[IC pre-training](pretraining.md)).

## When to use it

- Model Checkpoint: long runs where the best network is not the last one (the loss rises again), or to keep
  intermediate states.
- Restore & visualize: to draw other plots, times or outputs of a finished run, or Error Analysis against
  new reference data, without training again.

## How to enable it

- **Training** page → **Training Callbacks** → **Show Training Callbacks** → **Model Checkpoint** →
  **Periodically save the model during training**.
- **Results** → **Restore & visualize** page → **Trained model** box: choose **Forward Model** or
  **Inverse Model**, **Browse** for the **Model file:** (`.pt`); the **Config:** is found next to it (or
  browse for it); set **Plot type:**, **Output:** and **Save to:**; press **Restore & Visualize**.

## Settings

### Model Checkpoint

| Label | Meaning | Default | Range | Config field |
|---|---|---|---|---|
| **Periodically save the model during training** | DeepXDE `ModelCheckpoint` | off | — | `cb_model_checkpoint` |
| **Check every (iters):** | Period of the check | 1000 | 1–1,000,000 | `cb_checkpoint_period` |
| **Only save when the monitored loss improves** | Save only an improvement | on | — | `cb_checkpoint_save_better_only` |
| **Monitor:** | Training loss or Testing loss | Training loss | — | `cb_checkpoint_monitor` (`train loss`, `test loss`) |

### Restore & visualize

| Label | Meaning | Default |
|---|---|---|
| mode | **Forward Model** or **Inverse Model** (set from the config when it names unknowns) | Forward Model |
| **Model file:** | The saved `.pt` file | — |
| **Config:** | The model's configuration: `<model>.json`, else `model_config.json`, else (time windows) the step's `step_config.json` | found automatically |
| **Combine all detected steps into one full-domain animation** | For a model inside a time-window run: use every window | on when steps are found |
| **Optimizer used for this model:** | Adam or L-BFGS (how the model is compiled before restoring) | from the config |
| **Plot type:** | Surface, Line (time steps), Animation Line (GIF), Animation Surface (GIF); inverse: Parameter Convergence, Parameter Convergence (GIF) | Surface (Parameter Convergence for an inverse model) |
| **Output:** | An output, or **Custom...** with an expression and label | Output 1 |
| **Parameter names:**, convergence files (**➕ Add parameter file**), **Figures:**, **Log-scale y-axis** | Inverse models: the `<name>_convergence.txt` files, found in the run folder | found automatically |
| **Save to:** | Folder for the new figures | — |

## Formulation

Saved files in a run folder (DeepXDE's `model.save` appends `-<iteration>.pt`):

| File | Written by |
|---|---|
| `solution_results/model_adam-phase<p>-<it>.pt`, `model_lbfgs-phase<p>-<it>.pt`, `model_nncg-phase<p>-<it>.pt` | each training phase \( p \) |
| `solution_results/model_config.json` | the configuration the run trained with (used by Restore) |
| `solution_results/checkpoint_best-<it>.pt` | Model Checkpoint (`ModelCheckpoint(..., save_better_only, period, monitor)`) |
| `solution_results/checkpoint_step<NNN>_best-<it>.pt` | Model Checkpoint with time windows: one per window \( NNN \) (001, 002, …), each window starting from a fresh best value |
| `solution_results/rar_rounds/round_NN/model_adam-….pt`, `model_lbfgs-….pt` | RAR rounds |
| `time_adaptive_steps/step_NNN_t<t0>_to_t<t1>/model_adam-….pt`, `model_lbfgs-phase<p>-….pt`, `model_resampled-….pt`, `step_config.json` | each time window (`model_resampled` only with resampling rounds: the window's final network, which Error Analysis and Restore use) |
| `ic_pretrain/ic_pretrain_model-<it>.pt` | IC pre-training |
| `<name>_convergence.txt` | inverse unknowns (iteration, value) |

Restore builds a script that recreates the network from the config (the same network code, transforms and
Fourier matrices as training), compiles the model with the chosen optimizer, calls `model.restore(...)` and
predicts on the plot grids.

## Defaults and limitations

- **No continuation of training** from a phase's model or a checkpoint; a new Solve trains from new random
  weights (or from an IC pre-training checkpoint).
- Model Checkpoint applies to the training phases (it is in the same callback list as the other callbacks).
- Without a save folder, no model is saved (plots go to a temporary folder).
- A model's config must match it: restoring with another network size fails in `model.restore`.

!!! info "New in v1.6.0 (in development)"
    The exported script no longer stops with `NameError: _sol_dir` when Model Checkpoint is on.

## Supported problem types

All; time-window runs are restored window by window (or combined).

## Compatibility with other features

No refusals specific to checkpoints.

## Common mistakes and troubleshooting

- **"No config found"**: the `.pt` file was moved without its `model_config.json` (or `step_config.json`);
  browse for the config.
- **Restoring fails with an optimizer error**: set **Optimizer used for this model:** to the optimizer the
  file was saved with (its name says it: `model_adam…` or `model_lbfgs…`).

## Example

Any example run with a save folder, e.g. **Burgers 1D – Raissi et al.** See
[Tutorial 15: Restoring a trained model](../tutorials/15-restore.md) and the
[Restore & visualize guide page](../guide/restore.md).

## Related

- **Tutorial:** [15. Restoring a trained model](../tutorials/15-restore.md)
- **Examples:** [Burgers 1D – Raissi et al.](../examples/1d/burgers-1d-raissi-et-al.md)
- **Known limitations:** [Defaults and limitations](#defaults-and-limitations) and [Compatibility with other features](#compatibility-with-other-features) on this page

### Implementation and source code

--8<-- "_snippets/source/checkpoints.md"
