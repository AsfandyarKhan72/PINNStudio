<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_training_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Training page
- `MainWindow._add_scheduler_phase` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): one row of Training phases

**Helpers**

- `MainWindow._validate_optimizer_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): checks before Solve
- `_clean_sched_phases` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): the phases a run actually trains
- `_clean_phase_train_lines` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): compile and train calls for one phase
- `_lbfgs_maxfun` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): L-BFGS iteration limit
- `_build_train_cbs_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): callbacks: early stopping, point resampling, checkpoints, timer
- `phases_from_legacy` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/config.py): older saved problems to training phases

**Tests**

- [`tests/test_round78_example_phases.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round78_example_phases.py): examples train through the phases shown
- [`tests/test_lbfgs_steps.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_lbfgs_steps.py): L-BFGS steps
- [`tests/test_training_callbacks_visibility.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_training_callbacks_visibility.py): callbacks

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
