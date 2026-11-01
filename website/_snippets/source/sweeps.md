<!-- Written by tools/docs/make_source_links.py from tools/docs/source_map.py: do not edit. -->

**Entry points**

- `MainWindow._start_sweep` in [`pinnstudio/ui/main_window.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/ui/main_window.py): Parameter sweep page: start a sweep or a Bayesian search
- `run_sweep` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): runs the configurations one after another

**Helpers**

- `BayesRuns` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): Bayesian search: next run from the results so far
- `BayesOptimizer` in [`pinnstudio/core/bayes_opt.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/bayes_opt.py): Gaussian process with expected improvement
- `available_params` in [`pinnstudio/core/sweep_registry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_registry.py): the sweepable parameters
- `write_sweep_summary_plot` in [`pinnstudio/core/sweep_runner.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/pinnstudio/core/sweep_runner.py): summary table and plot

**Tests**

- [`tests/test_sweep_registry.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sweep_registry.py): sweepable parameters
- [`tests/test_sweep_execution.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_sweep_execution.py): a sweep run end to end
- [`tests/test_round75_wave_helmholtz_bayes.py`](https://github.com/AsfandyarKhan72/PINNStudio/blob/main/tests/test_round75_wave_helmholtz_bayes.py): Bayesian search (Helmholtz example)

Files on the `main` branch of the repository; each link names the function or class to look for in the file.
