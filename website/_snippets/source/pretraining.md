<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_training_panel` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): IC pre-training options on the Training page
- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): IC-guided learning in the time-window options
- `_ic_pretrain_train_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): pre-training on the initial condition, in both scripts

**Helpers**

- `_icgl_outputs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): outputs IC-guided learning fits in each window
- `_ta_icgl_iterations` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): IC-guided learning iterations per window

**Tests**

- [`tests/test_round47_ic_pretrain_shared_block.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round47_ic_pretrain_shared_block.py): pre-training block shared by both scripts
- [`tests/test_round40_ic_pretrain_restore_parity.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round40_ic_pretrain_restore_parity.py): pre-training and restoring
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): IC-guided learning with bc-PINN windows

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
