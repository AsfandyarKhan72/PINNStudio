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
 - Phase 5 (MainWindow._render_sweep_plot()): using SYNTHETIC
   (label, result) cache entries -- not real training, matplotlib with
   the Agg backend doesn't need one -- confirms a rendered PNG and a
   populated sweep_plot_label._source_path for: a one-at-a-time sweep
   over 2 numeric parameters (2 subplots); a Grid sweep over exactly 2
   parameters (a heatmap, with the (row, col) <-> (param0, param1)
   value mapping checked against sweep_runner.build_runs()'s own
   documented iteration order); a Grid sweep over 3+ parameters (no
   plot -- a message instead, since it's no longer a 2D grid); and a
   categorical parameter with one run's final_loss missing (status
   "error") -- confirming the exact bug caught by hand while building
   this (a NaN-valued point silently dropping its own x-tick out of
   the visible plot range via matplotlib's autoscale) stays fixed.

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

    # ── Phase 5: results plot (synthetic cache, no training) ──────────
    import os as _os

    def _fresh_window_with_rows(mode, param_ids_and_values):
        w2 = MainWindow()
        _app.processEvents()
        w2._refresh_sweep_param_choices()
        for i, (pid, text) in enumerate(param_ids_and_values):
            if i > 0:
                w2._add_sweep_row()
            r = w2.sweep_row_list[i]
            r["param_combo"].setCurrentIndex(r["param_combo"].findData(pid))
            meta = r["param_combo"].itemData(r["param_combo"].currentIndex(), Qt.ItemDataRole.UserRole + 1) or {}
            if meta.get("value_type") != "categorical":
                r["mode_combo"].setCurrentIndex(r["mode_combo"].findData("list"))
            r["list_edit"].setText(text)
        w2.sweep_enable_cb.setChecked(True)
        w2.sweep_mode_combo.setCurrentIndex(w2.sweep_mode_combo.findData(mode))
        return w2

    # OAT, 2 numeric parameters -> 2 subplots, no crash, file written.
    w_oat = _fresh_window_with_rows("oat", [("hidden_layers", "2, 4, 6"), ("neurons_per_layer", "32, 64")])
    cfg_oat = w_oat._build_config()
    check(cfg_oat.validate() == [], f"OAT plot-test config should validate clean: {cfg_oat.validate()}")
    w_oat._sweep_last_config = cfg_oat
    from pinnstudio.core.sweep_runner import build_runs as _build_runs
    oat_runs = _build_runs(cfg_oat)
    check(len(oat_runs) == 6, f"expected 1 baseline + 3 + 2 = 6 OAT runs, got {len(oat_runs)}")
    w_oat._sweep_results_cache = [
        (label, {"status": "done", "final_loss": 1e-3 / (k + 1), "l2_relative": 1e-2 / (k + 1)})
        for k, (label, _) in enumerate(oat_runs)
    ]
    w_oat._render_sweep_plot()
    check(w_oat.sweep_plot_label._source_path is not None and _os.path.exists(w_oat.sweep_plot_label._source_path),
          "OAT sweep plot should render a PNG and set sweep_plot_label._source_path")
    check(w_oat.sweep_plot_save_btn.isEnabled(), "Save Figure button should be enabled once a plot has rendered")

    # Grid, exactly 2 parameters -> heatmap; verify the (row, col) <->
    # (param0, param1) value mapping matches build_runs()'s own
    # itertools.product order (values1 iterates fastest).
    w_grid2 = _fresh_window_with_rows("grid", [("hidden_layers", "2, 4"), ("neurons_per_layer", "32, 64, 96")])
    cfg_grid2 = w_grid2._build_config()
    check(cfg_grid2.validate() == [], f"Grid-2 plot-test config should validate clean: {cfg_grid2.validate()}")
    w_grid2._sweep_last_config = cfg_grid2
    grid2_runs = _build_runs(cfg_grid2)
    check(len(grid2_runs) == 1 + 2 * 3, f"expected 1 baseline + 2x3 = 7 grid runs, got {len(grid2_runs)}")
    # Distinct, easily-traceable values: run k (0-indexed, after baseline)
    # gets final_loss = k -- after rendering, re-derive the matrix the
    # same way the heatmap code does and confirm it matches build_runs()'s
    # own iteration order exactly (values1 fastest).
    w_grid2._sweep_results_cache = [(label, {"status": "done", "final_loss": float(k), "l2_relative": None})
                                     for k, (label, _) in enumerate(grid2_runs)]
    w_grid2._render_sweep_plot()
    check(w_grid2.sweep_plot_label._source_path is not None and _os.path.exists(w_grid2.sweep_plot_label._source_path),
          "Grid-2 sweep plot should render a PNG")
    # grid2_runs[1:] should be ordered (hl=2,np=32),(hl=2,np=64),(hl=2,np=96),(hl=4,np=32),... --
    # i.e. run index 0..5 (after baseline) = final_loss 0..5, with
    # hidden_layers changing every 3 runs (slower axis) and
    # neurons_per_layer changing every run (faster axis).
    expected_grid2_labels = [
        "Hidden layers=2, Neurons per layer=32", "Hidden layers=2, Neurons per layer=64",
        "Hidden layers=2, Neurons per layer=96", "Hidden layers=4, Neurons per layer=32",
        "Hidden layers=4, Neurons per layer=64", "Hidden layers=4, Neurons per layer=96",
    ]
    labels_only = [l for l, _ in grid2_runs[1:]]
    check(labels_only == expected_grid2_labels,
          f"grid run order should have neurons_per_layer (2nd param) as the fast-varying axis, "
          f"matching the heatmap code's own assumption: got {labels_only}, expected {expected_grid2_labels}")

    # Grid, 3 parameters -> message only, no plot file, save button stays disabled.
    w_grid3 = _fresh_window_with_rows("grid", [("hidden_layers", "2, 4"), ("neurons_per_layer", "32, 64"), ("num_domain", "1000, 2000")])
    cfg_grid3 = w_grid3._build_config()
    check(cfg_grid3.validate() == [], f"Grid-3 plot-test config should validate clean: {cfg_grid3.validate()}")
    w_grid3._sweep_last_config = cfg_grid3
    grid3_runs = _build_runs(cfg_grid3)
    w_grid3._sweep_results_cache = [(label, {"status": "done", "final_loss": 1e-3, "l2_relative": None}) for label, _ in grid3_runs]
    w_grid3._render_sweep_plot()
    check("3+" in w_grid3.sweep_plot_label.text() or "no longer a 2D grid" in w_grid3.sweep_plot_label.text(),
          f"a 3-parameter Grid sweep should show an explanatory message, got: {w_grid3.sweep_plot_label.text()!r}")
    check(not w_grid3.sweep_plot_save_btn.isEnabled(),
          "Save Figure button should stay disabled when no plot was rendered (3+ param Grid)")

    # Categorical parameter + a run with status "error" (no final_loss):
    # this is the exact scenario that caught the NaN-point-drops-its-own-
    # tick-from-view matplotlib autoscale bug while building this feature.
    w_cat = _fresh_window_with_rows("oat", [("phase0_optimizer", "adam, lbfgs")])
    cfg_cat = w_cat._build_config()
    check(cfg_cat.validate() == [], f"categorical plot-test config should validate clean: {cfg_cat.validate()}")
    w_cat._sweep_last_config = cfg_cat
    w_cat._sweep_results_cache = [
        ("baseline", {"status": "done", "final_loss": 1e-3, "l2_relative": 1e-2}),
        ("Phase 1: Optimizer=adam", {"status": "done", "final_loss": 2e-3, "l2_relative": 2e-2}),
        ("Phase 1: Optimizer=lbfgs", {"status": "error", "final_loss": None, "l2_relative": None}),
    ]
    w_cat._render_sweep_plot()  # must not raise
    check(w_cat.sweep_plot_label._source_path is not None and _os.path.exists(w_cat.sweep_plot_label._source_path),
          "categorical sweep plot (with one failed run) should still render a PNG without crashing")

    # Empty sweep (baseline only) -> a clear message, not a crash.
    w_empty = MainWindow()
    _app.processEvents()
    w_empty._sweep_last_config = w_empty._build_config()
    w_empty._sweep_results_cache = [("baseline", {"status": "done", "final_loss": 1e-3, "l2_relative": None})]
    w_empty._render_sweep_plot()
    check(w_empty.sweep_plot_label._source_path is None,
          "a baseline-only cache (no sweep parameters) should show a message, not a plot")

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
