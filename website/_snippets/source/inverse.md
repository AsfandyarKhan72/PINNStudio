<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_inverse_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Inverse page
- `MainWindow._add_inverse_var_row` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): one unknown parameter (initial and true value)

**Helpers**

- `_parse_inverse_variables` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): unknown parameters to DeepXDE variables
- `inv_record_period` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): how often the parameter values are recorded
- `_param_axis_helper_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): parameter convergence plots
- `MainWindow.INVERSE_AUTO_VARS` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): the templates' unknown parameters in Inverse mode

**Tests**

- [`tests/test_added_inverse_modes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_added_inverse_modes.py): Inverse modes added to DeepXDE forward examples
- [`tests/test_burgers2d_inverse_setup.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_burgers2d_inverse_setup.py): an inverse template set up end to end
- [`tests/test_round49_param_axis_shared.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round49_param_axis_shared.py): parameter plots

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
