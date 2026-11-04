<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_domain_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Initial conditions page: one row per output, formula or file

**Helpers**

- `MainWindow._ic_file_anchors_checked` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): an initial condition read from a file: also check the equation there
- `_clean_active_ic_outputs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): which outputs get an initial-condition term
- `_bc_initial_operator_rows` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): operator conditions applied at the initial time
- [`pinnstudio/core/portable_paths.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/portable_paths.py): data paths saved as @REF/... so problems open anywhere

**Tests**

- [`tests/test_ic_file_choice.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_ic_file_choice.py): formula or file, never both
- [`tests/test_round64_initial_operator_decay.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round64_initial_operator_decay.py): operator condition at the initial time
- [`tests/test_portable_paths.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_portable_paths.py): portable data paths

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
