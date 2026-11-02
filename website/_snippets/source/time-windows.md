<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._add_ta_step_group` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): time-window groups on the Adaptive training page
- `MainWindow._build_ta_step_groups_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): groups to the saved settings
- `_gs_time_adaptive_loop_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the time-window loop: transfer learning, bc-PINN, growing range, resampling

**Helpers**

- `_bc_rng_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): random generator for bc-PINN's stored points
- `_ta_clean_unsupported` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): time-window options the exported script does not write

**Tests**

- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): one network through all windows, bc-PINN
- [`tests/test_ta_growing_2d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_growing_2d.py): growing time range in 2D
- [`tests/test_export_time_window_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_time_window_options.py): export trains the same windows as Solve

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
