<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._add_inverse_data_row` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): one measured-data file on the Inverse page
- `MainWindow._build_inverse_obs_files_json` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/ui/main_window.py): data rows to the saved settings

**Helpers**

- `_parse_inverse_obs_files` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): data files to DeepXDE point-set conditions
- `obs_reduce_args` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/codegen.py): the Use: choice (all points, grid, random subset)
- [`pinnstudio/core/obs_reduce.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/pinnstudio/core/obs_reduce.py): reading and thinning the data points

**Tests**

- [`tests/test_obs_reduce.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_obs_reduce.py): Use: all points, grid, random subset
- [`tests/test_large_data_note.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/v1.6.0/tests/test_large_data_note.py): note for large data files

Files on release `v1.6.0` of the repository; each link names the function or class to look for in the file.
