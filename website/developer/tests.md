# Tests

## Layout

All tests are in `tests/`, one file per feature or per fixed issue (`test_*.py`, about 130 files). Most build a
`MainWindow` offscreen or a `PINNConfig` directly, generate scripts with `generate_script` /
`generate_clean_script` and check the generated code, the validation messages or the widgets; some train small
models in their own processes. Most files also run on their own as plain scripts
(`python tests/test_<name>.py`), which is how the smoke-test workflow runs them. `tests/local/` (ignored by git)
holds tests you keep for yourself; the workflows skip it.

Two environment variables are set by the tests themselves when missing, and should be set when running a
single file by hand: `QT_QPA_PLATFORM=offscreen` (no display needed) and `DDE_BACKEND=pytorch`.

## Slow tests

`tests/conftest.py` lists in `SLOW_FILES` the test files that train real models (minutes each on a CI machine)
and gives every test in them the `slow` marker. It also loads PyTorch before PySide6 on Windows (otherwise
PyTorch fails to load its DLLs) and runs the scripts started by tests in Python's UTF-8 mode, as the app does.

```bash
pip install pytest pytest-xdist
pytest tests -m "not slow" -n auto --ignore=tests/local   # everything except the training tests, on all cores
pytest tests -n auto --ignore=tests/local                 # the full suite
pytest tests/test_training_methods.py -q                  # one file
```

`-n auto` (pytest-xdist) spreads the tests over the machine's cores. Run the full suite before a change is
merged; the training tests take long on a CPU.

## Continuous integration

| Workflow | When | What |
|---|---|---|
| `platform-tests.yml` | push to `main`, `cleanup`, `integration/**`; pull requests to `main`; by hand | the full suite (`pytest -n auto`) on the latest Windows, macOS and Ubuntu with Python 3.12 and CPU PyTorch; failures as annotations (`tools/junit_annotations.py`); the window end to end (`tools/smoke_gui.py`: train, show results, restore); the desktop shortcut; the one-command installer on Windows and macOS |
| `platform-matrix.yml` | a push whose commit message contains `[matrix]`, or by hand | the suite without slow tests and the window check on older and newer versions of each system (Ubuntu 22.04/24.04 including ARM, Windows 2022/2025 and Windows 11 ARM, macOS 14/15 and Intel) and Python 3.9 to 3.14; the installer on several systems. Run before a release |
| `smoke-test.yml` | push to `main`, `cleanup`, `integration/**`; pull requests to `main` | boots `MainWindow` offscreen and runs a list of test files as scripts, with PySide6 (below 6.12 on its Python 3.11), NumPy, Matplotlib and pandas only (parts that need PyTorch or DeepXDE report themselves skipped) |
| `docs.yml` | pull requests and pushes that change the site | `mkdocs build --strict`, a link check and a downloadable preview; deployment only by hand from `main` after approval ([Contributing and documentation](contributing.md#build-and-preview)) |

## Checks outside pytest

- `tools/check_export_equivalence.py` trains every example with Solve's script and with the exported script
  and compares them ([Export equivalence](../reproducibility/export-equivalence.md)).
- `tools/run_deepxde_examples.py` trains the DeepXDE examples (`--smoke` for a few hundred iterations each).
- `tools/all_examples_check.py` loads every example, forward and inverse, as the window loads it, and runs it.
- `tools/check_benchmarks_figures.py` checks that every figure the benchmark pages need exists.

## Writing a regression test

A fixed bug gets a test that fails before the fix and passes after it. `tests/test_audit_v160.py` is the model:

- the module docstring lists, numbered, each problem the file guards against, in one or two sentences;
- each test builds the smallest configuration that shows the problem, usually from a shipped example
  (`deepxde_examples.build_config(dx.BY_NAME[name], REF)` and `dataclasses.replace(...)` for the setting under
  test);
- it checks the generated code of **both** generators, or the result of `validate()`, rather than training:

```python
def test_per_point_weights_with_a_point_resampler_are_refused():
    base = _dx("Burgers equation with RAR")
    for pw in ("RBA", "SA-PINN"):
        errs = dataclasses.replace(base, point_weights=pw, cb_point_resampler=True).validate()
        assert any("Point Resampler" in e for e in errs), errs
```

- behaviour that only shows in a run (losses, saved files) goes in a separate file with short runs, like
  `tests/test_audit_v160_runs.py`, which is listed in `SLOW_FILES`.

Checking strings of generated code is brittle when the generator's formatting changes; check the smallest
fragment that shows the behaviour, and also `compile(script, ...)` or `ast.parse(script)` so that the script is
at least valid Python.
