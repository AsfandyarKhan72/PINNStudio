<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `PINNConfig` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): every setting of a problem (saved as .pinn.json)
- `PINNConfig.validate` in [`pinnstudio/core/config.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/config.py): checks run before Solve and on a loaded file

**Helpers**

- `MainWindow._build_config` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the window's widgets to a PINNConfig
- `MainWindow._apply_config` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): a PINNConfig back into the widgets
- `MainWindow._save_problem` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Save Problem
- `MainWindow._open_problem` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Open Saved Problem
- `MainWindow._on_gpu_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): GPU / Hardware settings
- [`tools/docs/make_config_reference.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tools/docs/make_config_reference.py): the configuration reference page

**Tests**

- [`tests/test_config_options.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_config_options.py): GPU and random-seed settings
- [`tests/test_template_copies.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_template_copies.py): saving never changes an example
- [`tests/test_audit_v160.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_audit_v160.py): validation checks of the v1.6.0 audit

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
