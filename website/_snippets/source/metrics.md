<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._show_ea_dialog` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Error Analysis dialog
- `_gs_ea_setup_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): Error Analysis in the Solve script

**Helpers**

- `_ea_metrics_row_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): one row of error_metrics.txt
- `_ea_metrics_csv_header` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): metric columns (L2, MSE, max, mean absolute)
- `_formula_reference_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): reference from an exact-solution formula
- `auto_line_slice` in [`pinnstudio/core/line_slice.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/line_slice.py): where line comparisons cut a 2D/3D domain
- [`pinnstudio/core/ea_view3d.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/ea_view3d.py): 3D view of the error

**Tests**

- [`tests/test_ea_fixes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_ea_fixes.py): reference files and metrics
- [`tests/test_ea_spacetime.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_ea_spacetime.py): relative L2 over all reference points
- [`tests/test_round67_formula_reference.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round67_formula_reference.py): reference from a formula

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
