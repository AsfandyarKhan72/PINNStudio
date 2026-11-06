<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._add_custom_bc_entry` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): one row of the Boundary conditions page
- `MainWindow._build_custom_bc_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): rows to the saved settings (custom_bc_json)

**Helpers**

- `_clean_bc_row_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): one row to DeepXDE boundary-condition code
- `_bc_operator_def_lines` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): Operator and Point Set Operator conditions
- `_bc_sides` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): side names (left, right, ...) in Where expressions
- `_bc_check_periodic_preconditions` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): checks before a periodic condition

**Tests**

- [`tests/test_round68_bc_where_names.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round68_bc_where_names.py): Where expressions and side names
- [`tests/test_round71_pointset_points.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round71_pointset_points.py): Point Set conditions without a file
- [`tests/test_sketch_edges.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_sketch_edges.py): sketch edges as boundary conditions

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
