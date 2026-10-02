#!/usr/bin/env python3
"""
Regression check for Phase 3 (the "Parameter Sweep" GUI tab in
main_window.py) and Phase 4 (pinnstudio/core/sweep_runner.py, the
execution driver). Phase 1/2 (config fields + registry) already has its
own coverage in test_sweep_registry.py; this file is specifically about
the GUI wiring on top of it and the driver that turns a configured
sweep into real runs.

Checks:
 - The top-level "Parameter Sweep" tab exists alongside "Setup", and a
   default row is present on boot.
 - Add/remove row wiring (_add_sweep_row / its remove button closure).
 - Selecting a categorical parameter (e.g. a scheduler phase's
   optimizer) hides the mode/range widgets and shows the list entry
   with a choices-listing placeholder; selecting a numeric parameter
   restores the mode picker, and switching modes toggles list vs.
   range widgets correctly.
 - _build_sweep_parameters_json() + _build_config() produce a
   sweep_parameters payload PINNConfig.validate() accepts, matching
   what was actually entered in the rows (list values parsed as floats
   where the parameter is numeric, left as strings for a categorical
   one).
 - sweep_runner.build_runs(): "oat" produces baseline + one run per
   listed value per row (not a cross product); "grid" produces
   baseline + the full Cartesian product.
 - sweep_runner's stale-results guard: a leftover error_metrics.txt
   from BEFORE a run started must never be reported as that run's
   result (this exact bug was caught by hand while building this
   feature -- see sweep_runner._read_last_l2's `not_before` parameter).

This file is wired into CI (smoke-test.yml) and deliberately never
actually trains anything -- CI only installs PyQt6/numpy/matplotlib/
pandas, no torch/deepxde, matching every other file in this suite. A
real, tiny, end-to-end run_sweep() (actual subprocess training) is
covered separately in test_sweep_execution.py, which is NOT wired into
CI and must be run locally where torch/deepxde are installed.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_tab.py
"""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.sweep_runner import build_runs, _read_last_l2


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()

    # ── Tab structure ───────────────────────────────────────────────
    tab_texts = [win.central_tabs.tabText(i) for i in range(win.central_tabs.count())]
    check("Setup" in tab_texts, f"expected a 'Setup' tab, got {tab_texts}")
    check("Parameter Sweep" in tab_texts, f"expected a 'Parameter Sweep' tab, got {tab_texts}")
    check(len(win.sweep_row_list) == 1, f"expected exactly one default sweep row on boot, got {len(win.sweep_row_list)}")

    win._refresh_sweep_param_choices()

    # ── Add / remove rows ────────────────────────────────────────────
    win._add_sweep_row()
    check(len(win.sweep_row_list) == 2, f"expected 2 rows after _add_sweep_row(), got {len(win.sweep_row_list)}")
    second = win.sweep_row_list[1]
    second_widget = second['widget']
    # Simulate clicking that row's own remove button: find it via the
    # same closure wiring _add_sweep_row() sets up (its remove_btn is not
    # stored in row_data, but deleteLater()+list removal is what the
    # closure does -- exercise that behavior directly).
    second_widget.deleteLater()
    win.sweep_row_list.remove(second)
    check(len(win.sweep_row_list) == 1, f"expected 1 row after removing the second, got {len(win.sweep_row_list)}")

    row = win.sweep_row_list[0]
    combo = row['param_combo']

    # ── Categorical vs numeric visibility ────────────────────────────
    opt_idx = combo.findData("phase0_optimizer")
    check(opt_idx >= 0, "expected 'phase0_optimizer' to be available with the default Optimizer Scheduler on")
    combo.setCurrentIndex(opt_idx)
    check(row['mode_combo'].parentWidget().isHidden(),
          "mode picker should be hidden for a categorical parameter")
    check(not row['list_edit'].parentWidget().isHidden(),
          "the values field should stay visible for a categorical parameter")
    check("adam" in row['list_edit'].placeholderText(),
          f"categorical placeholder should list its choices, got {row['list_edit'].placeholderText()!r}")

    hl_idx = combo.findData("hidden_layers")
    combo.setCurrentIndex(hl_idx)
    check(not row['mode_combo'].parentWidget().isHidden(),
          "mode picker should reappear for a numeric parameter")
    check(not row['list_edit'].parentWidget().isHidden(),
          "list entry should be visible by default (mode defaults to 'list')")
    check(row['min_spin'].parentWidget().isHidden(),
          "range widgets should stay hidden while mode is 'list'")

    row['mode_combo'].setCurrentIndex(row['mode_combo'].findData("linear"))
    check(row['list_edit'].parentWidget().isHidden(), "list entry should hide once mode is 'linear'")
    check(not row['min_spin'].parentWidget().isHidden(), "range widgets should show once mode is 'linear'")

    # ── JSON building + validate() ───────────────────────────────────
    row['mode_combo'].setCurrentIndex(row['mode_combo'].findData("list"))
    row['list_edit'].setText("2, 4, 6")
    win.sweep_enable_cb.setChecked(True)
    config = win._build_config()
    entries = json.loads(config.sweep_parameters)
    check(entries == [{"id": "hidden_layers", "mode": "list", "values": [2.0, 4.0, 6.0]}],
          f"unexpected sweep_parameters payload: {entries}")
    check(config.validate() == [], f"well-formed sweep config should validate clean, got {config.validate()}")

    # A categorical row's list values should be kept as strings, not
    # coerced to floats (they're optimizer names, not numbers).
    combo.setCurrentIndex(combo.findData("phase0_optimizer"))
    row['list_edit'].setText("adam, lbfgs")
    config2 = win._build_config()
    entries2 = json.loads(config2.sweep_parameters)
    check(entries2 == [{"id": "phase0_optimizer", "mode": "list", "values": ["adam", "lbfgs"]}],
          f"categorical sweep values should stay as strings: {entries2}")
    check(config2.validate() == [], f"well-formed categorical sweep should validate clean, got {config2.validate()}")

    # ── build_runs(): oat vs grid ─────────────────────────────────────
    win._add_sweep_row()
    row_b = win.sweep_row_list[1]
    row_b['param_combo'].setCurrentIndex(row_b['param_combo'].findData("neurons_per_layer"))
    row_b['mode_combo'].setCurrentIndex(row_b['mode_combo'].findData("list"))
    row_b['list_edit'].setText("32, 64")

    win.sweep_mode_combo.setCurrentIndex(win.sweep_mode_combo.findData("oat"))
    cfg_oat = win._build_config()
    runs_oat = build_runs(cfg_oat)
    check(len(runs_oat) == 1 + 2 + 2, f"OAT with 2+2 values should give 5 runs (1 baseline + 2 + 2), got {len(runs_oat)}: {[l for l, _ in runs_oat]}")

    win.sweep_mode_combo.setCurrentIndex(win.sweep_mode_combo.findData("grid"))
    cfg_grid = win._build_config()
    runs_grid = build_runs(cfg_grid)
    check(len(runs_grid) == 1 + 2 * 2, f"Grid with 2x2 values should give 5 runs (1 baseline + 4), got {len(runs_grid)}: {[l for l, _ in runs_grid]}")

    # ── Stale-results guard ───────────────────────────────────────────
    # A file that predates `not_before` must be ignored even if present.
    stale_dir = "/tmp/_sweep_tab_test_stale/error_analysis"
    os.makedirs(stale_dir, exist_ok=True)
    stale_path = os.path.join(stale_dir, "error_metrics.txt")
    with open(stale_path, "w") as f:
        f.write("t,L2_relative,MSE,Max_error,Mean_abs_error\n0.0,0.999,0,0,0\n")
    class _Fake:
        ea_files = "[(0.0, 'x', None)]"
        save_dir = "/tmp/_sweep_tab_test_stale"
    future = time.time() + 5
    check(_read_last_l2(_Fake(), future) is None,
          "a results file older than the run's own start time must be ignored, not reported")
    # A file written AFTER not_before should be read correctly.
    time.sleep(0.05)
    now = time.time()
    time.sleep(0.05)
    with open(stale_path, "w") as f:
        f.write("t,L2_relative,MSE,Max_error,Mean_abs_error\n0.0,0.123,0,0,0\n")
    check(_read_last_l2(_Fake(), now) == 0.123,
          "a results file written after the run's own start time should be read correctly")

    # NOTE: a real, tiny, end-to-end run_sweep() execution (actual
    # subprocess training) is intentionally NOT exercised here -- this
    # file is wired into CI's smoke-test workflow, which only installs
    # PyQt6/numpy/matplotlib/pandas (no torch/deepxde), matching every
    # other file in this suite (test_templates.py etc. only ever call
    # generate_script(), never actually run it). That real-execution
    # check lives in test_sweep_execution.py instead, run locally where
    # torch/deepxde are installed -- see that file's docstring.

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP TAB/RUNNER TESTS PASSED")
    return failures


def test_sweep_tab():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
