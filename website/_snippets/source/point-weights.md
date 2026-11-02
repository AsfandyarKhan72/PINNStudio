<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_adaptive_and_sweep_panels` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): the controls on the Loss weights page (Automatic weighting), built with the Adaptive training page
- `_training_methods_defs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): runtime code of RBA, SA-PINN and causal training

**Helpers**

- `_point_weights_mode` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): which point weights a run uses
- `_training_methods_cbs` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the callbacks that update the weights

**Tests**

- [`tests/test_training_methods.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_training_methods.py): causal training, RBA and SA-PINN
- [`tests/test_ta_bcpinn.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_ta_bcpinn.py): point weights inside time windows

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
