<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_setup_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Problem page: dimension, forward or inverse, steady or in time
- `MainWindow._on_dim_changed` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): switching between ODE, 1D, 2D and 3D
- `generate_script` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the script Solve runs, for every problem type

**Helpers**

- `_ode_to_1d` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): an ODE problem rewritten as a 1D problem in t
- `_effective_config` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the settings as the generators read them

**Tests**

- [`tests/test_round74_ode.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round74_ode.py): ODE problems
- [`tests/test_templates.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_templates.py): every template in Forward and Inverse mode

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
