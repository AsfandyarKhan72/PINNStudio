<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._build_weight_inputs` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Loss weights page: one weight per term

**Helpers**

- `_clean_loss_weights` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): the weight list in term order
- `_clean_phase_weights` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): weights per training phase
- `_init_loss_weights_code` in [`pinnstudio/core/codegen.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/codegen.py): Scale by initial losses

**Tests**

- [`tests/test_steady_state_ic_weights.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_steady_state_ic_weights.py): no initial-condition weight without time
- [`tests/test_round75_wave_helmholtz_bayes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round75_wave_helmholtz_bayes.py): weights from the initial losses (wave example)
- [`tests/test_entered_values.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_entered_values.py): weights used as entered

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
