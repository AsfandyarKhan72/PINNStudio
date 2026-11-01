<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Adaptive training page
- `_gs_rar_loop_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the RAR loop of the Solve script

**Helpers**

- `_rar_phase` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): loss and weights of an RAR round
- `_rar_tolerance_check` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): stop on a residual tolerance
- `_clean_rar_diagnostics_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): RAR in the exported script, with per-round figures

**Tests**

- [`tests/test_rar_round_diagnostics.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_rar_round_diagnostics.py): per-round diagnostics
- [`tests/test_round66_rar_tolerance.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round66_rar_tolerance.py): residual tolerance
- [`tests/test_steady_state_rar_gating.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_rar_gating.py): RAR for steady problems

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
