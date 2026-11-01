<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `generate_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the script Solve runs (shown in the Script tab)
- `generate_clean_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Export as DeepXDE Script: the short standalone script
- `MainWindow._export_deepxde_script` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the export command

**Helpers**

- `MainWindow._refresh_script_tab` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the Script tab
- `run_pinn` in [`pinnstudio/core/runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/runner.py): runs a generated script in its own process
- [`tools/check_export_equivalence.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/check_export_equivalence.py): checks that both scripts train the same model

**Tests**

- [`tests/test_export_parity.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_parity.py): export and restore parity
- [`tests/test_export_time_window_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_export_time_window_options.py): export trains the same model as Solve
- [`tests/test_round55_generator_naming.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round55_generator_naming.py): the two generators

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
