# Architecture and code generation

PINNStudio never trains inside the window. Every action that trains or evaluates a model generates a Python
script and runs it as a separate process. The window only builds the configuration, starts the process,
streams its output and shows the files it writes.

```text
 widgets ──_build_config()──▶ PINNConfig ──validate()──▶ generate_script(config) ──▶ temp .py file
    ▲                            │                                                     │
    │                            │ generate_clean_script(config)                       ▼
 _apply_config()                 └──▶ Export as DeepXDE Script... / Script tab     subprocess (run_pinn)
    ▲                                                                                  │ stdout
 .pinn.json ◀── Save Problem                                     Training Log ◀────────┤
                                                                 Results tabs ◀── files in the run folder
```

## From widgets to configuration

`MainWindow._build_config()` (`pinnstudio/ui/main_window.py`) reads every widget and returns a `PINNConfig`
([Configuration schema](config.md)). `MainWindow._apply_config(config)` is its inverse: it sets every widget
from a configuration, and is used to open a saved problem and to load an example. **Save Problem** writes

```json
{"pinnstudio_problem_format": 1, "problem_name": "...", "saved_with_pinnstudio_version": "...",
 "saved_at": "...", "config": { ...every PINNConfig field... }, "template": "..."}
```

with reference-data paths rewritten as `@REF/...` by `core/portable_paths.py`, so that a problem file works on
another machine.

## Solve

`MainWindow._on_solve()`:

1. checks that the geometry type has code-generation support, checks optimizer settings, and calls
   `config.validate()`; any problem is listed in the Training Log and the run does not start;
2. with a sweep enabled, hands the configuration to `SweepThread` (`core/sweep_runner.run_sweep`), which repeats
   the steps below for every run, each in its own folder;
3. otherwise starts `SolverThread(config)`, a `QThread` that calls `core/runner.run_pinn(config)`.

`run_pinn` calls `generate_script(config)`, writes the script to a temporary file, and runs it with the Python
chosen by `core/pyexe.py` (in UTF-8 mode, with `DDE_BACKEND=pytorch`). Its standard output is buffered and
passed to the window every 0.5 s (the Training Log and the live loss read it); the temporary file is deleted at
the end. **Stop** terminates the process, and kills it if it has not exited after 3 seconds.

The run folder is `<Save to>/<label>__<YYYY-MM-DD_HH-MM-SS>`, computed by the window once per Solve
(`_timestamped_save_dir`). The script saves a checkpoint after every Training Phase in `solution_results/`
(`model_adam-phase1-<step>.pt`, `model_lbfgs-phase2-<step>.pt`, ...; RAR rounds and time windows in their own
folders), `solution_results/model_config.json` (what Restore needs to rebuild the network), the figures, the loss history and, with Error Analysis set up, `error_analysis/`. When the process ends,
`MainWindow._on_done()` shows the run's figures (`_display_run_result_plots`) and the Error analysis table
from the files of that run folder.

A config-level error raised by the generator itself (for example an impossible boundary-condition combination)
is caught in `run_pinn` and reported as `ERROR: ...` without starting a process.

## The two generators

Both live in `pinnstudio/core/codegen.py` and take a `PINNConfig`:

| | `generate_script(config)` | `generate_clean_script(config)` |
|---|---|---|
| Used by | **Solve**, sweeps | **Export as DeepXDE Script...**, the **Script** tab |
| Style | one long script that reads many settings at run time (`if {config.time_adaptive}: ...`) | short, tutorial-style: only the configured features are written, as plain DeepXDE calls |
| Extras | live progress output, every plot type, inline Error Analysis, per-round and per-window diagnostics, parameter history | the problem, training, the main plots and Error Analysis |

The two must train the same model. Shared helpers keep them in step: the network construction
(`_net_construction_helper_code`), the transforms (`_net_transforms_code`), the exact constraints, the training
methods (`_training_methods_defs`, used by both), the callbacks, the IC pre-training block
(`_ic_pretrain_train_code`), the RAR phase selection (`_rar_optimizer`, `_rar_phase`), the formula reference
(`_formula_reference_code`), and the error-metric columns. `tests/test_export_parity.py` checks the generated
code, and `tools/check_export_equivalence.py` trains every example with both scripts and compares the losses
([Export equivalence](../reproducibility/export-equivalence.md)).

Where the short script does not implement a feature yet, it says so: for time windows with backward
compatibility, a growing time range, initial-condition guided learning or resampling in each window
(`_ta_clean_unsupported`), the export is the output of `generate_script()` with a header comment.

### Preprocessing: `_effective_config`

Both generators start with `config = _effective_config(config)`, which returns a copy (never modifying the
caller's object) that the code below can read without special cases:

- time-adaptive training is switched off for a steady problem;
- for a non-box shape, the domain box fields are replaced by the shape's bounding box (`_shape_bbox`);
- named constants are substituted by their values in every expression, including the JSON lists of boundary
  conditions, measured-data expressions and training monitors;
- a `t` in an initial-condition formula is replaced by \(t_{\min}\);
- exact initial conditions and exact periodicity become output- and input-transform formulas
  (`_exact_constraints`);
- with **Scale by initial losses**, the phases' own weights are removed;
- `frac(` becomes the generated `_frac(` helper; an ODE becomes a 1D problem in the variable `t`
  (`_ode_to_1d`); `integral(...)` terms are rewritten for quadrature;
- a formula reference becomes ordinary reference-file entries (`_formula_reference_entries`);
- with an input-feature formula, the first layer width becomes the number of features.

A new feature that can be expressed as one of the existing ones (as the exact constraints are) belongs here.

## Restore & Visualize

The **Restore & visualize** page also generates scripts and runs them in a subprocess (`_RestoreThread`). The
builders are in `main_window.py`: `_build_restore_script` (one model), `_build_restore_script_ta` (one model per
time window), `_build_restore_ea_script` (Error Analysis of a restored model) and `_build_restore_param_script`
(the parameter convergence plot from a saved history). They rebuild the network with the same
`_net_construction_helper_code` and transforms as training, from `model_config.json` (or `step_config.json` of
a time window), then load the checkpoint; a network shaped differently would fail to load, which is why the
construction code is shared.

## Sweeps

`core/sweep_runner.run_sweep` builds the runs (`build_runs_detailed`, or `BayesRuns` for Bayesian search), sets
each run's values through `core/sweep_registry` (which knows, for example, that the learning rate lives in the
Training Phases, not in the legacy field), validates the run, and calls `run_pinn` for it. After each run it
reads the final loss from the log and the metrics from the run's `error_analysis/error_metrics*.txt`, and at the
end writes `sweep_summary.csv`, `sweep_manifest.json` and the summary plots in the sweep's folder.
