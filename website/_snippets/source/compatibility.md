<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `PINNConfig.validate` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): the combinations refused before every Solve and sweep run
- `MainWindow._validate_optimizer_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): weight decay with L-BFGS or NNCG, AdamW, NNCG's DeepXDE version
- `MainWindow._on_steady_state_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): a stationary problem disables Time Adaptive Training
- `MainWindow._on_problem_type_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Inverse mode removes Time Adaptive Training

**Helpers**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the warning for causal training with per-point weights
- `_check_anchor_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): extra training points refused with time windows
- `_bc_check_initial_operator_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Operator BC at the initial time
- `_bc_check_periodic_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Periodic BC only on box shapes
- `_bc_check_interface2d_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Interface 2D BC requirements
- `_effective_config` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): settings the generators switch off (time windows on a stationary problem)
- `_ode_to_1d` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): an ODE as a 1D problem in t (time windows switched off)

**Tests**

- [`tests/test_audit_v160.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_audit_v160.py): refused and generated combinations of the v1.6.0 audit
- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): causal training with RBA, exact constraints, validation
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): time-window options and their checks
- [`tests/test_lbfgs_steps.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_lbfgs_steps.py): network and method checks; RAR in an inverse problem
- [`tests/test_steady_state_rar_gating.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_rar_gating.py): RAR on a stationary problem
- [`tests/test_round77_fractional.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round77_fractional.py): frac(...) problems and their checks

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
