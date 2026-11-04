<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._display_run_result_plots` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Results tabs after a run
- `MainWindow._on_plot_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Plot settings dialog

**Helpers**

- `generate_clean_script._result_plot_parts` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): result plots of the exported script
- `_restore_surface_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): surface plots
- `_restore_line_plot_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): line plots over time steps
- `_restore_anim_surface_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): animations
- `_loss_plot_runtime_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): saved loss figure
- `MainWindow._on_export_settings` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): Export Solution (solution data)
- `FigureStyleDialog` in [`pinnstudio/ui/figure_style.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/figure_style.py): figure style
- `LiveLossPlot` in [`pinnstudio/ui/results.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/results.py): live loss during training
- `RunCompare` in [`pinnstudio/ui/results.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/results.py): comparing runs
- [`pinnstudio/ui/export_tools.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/export_tools.py): copying and saving figures and tables

**Tests**

- [`tests/test_figure_style.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_figure_style.py): one Plot settings dialog for every figure
- [`tests/test_plot_settings_more.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_plot_settings_more.py): plot settings reach every script
- [`tests/test_round69_output_named_plots.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_round69_output_named_plots.py): plots of multi-output problems

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
