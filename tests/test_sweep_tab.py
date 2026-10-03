#!/usr/bin/env python3
"""
Regression check for Phase 3 (the Parameter Sweep panel in
main_window.py, inline in the Setup tab's right side since v70 -- see
its own "Enable Parameter Sweep" toggle in the left training panel,
right after Adaptive Training) and Phase 4 (pinnstudio/core/
sweep_runner.py, the execution driver). Phase 1/2 (config fields +
registry) already has its own coverage in test_sweep_registry.py; this
file is specifically about the GUI wiring on top of it and the driver
that turns a configured sweep into real runs.

Checks:
 - The sweep panel's own widgets exist (Enable checkbox in the left
   panel, off by default; sweep_panel_widget hidden/normal_results_
   widget shown by default; toggling the checkbox swaps the two and
   disables the main Solve button), and a default row is present on
   boot.
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

    # ── v70: tab merge -- sweep lives inline in the Setup tab now ─────
    tab_texts = [win.central_tabs.tabText(i) for i in range(win.central_tabs.count())]
    check(tab_texts == ["Setup"], f"expected only a single 'Setup' tab (no separate 'Parameter Sweep' tab anymore), got {tab_texts}")
    check(not win.central_tabs.tabBar().isVisible() or win.central_tabs.count() <= 1,
          "the tab bar should stay hidden with only one tab")
    check(hasattr(win, "sweep_enable_cb"), "expected an Enable Parameter Sweep checkbox")
    check(not win.sweep_enable_cb.isChecked(), "Parameter Sweep should be OFF by default")
    check(hasattr(win, "normal_results_widget") and hasattr(win, "sweep_panel_widget"),
          "expected normal_results_widget and sweep_panel_widget as the two swappable right-panel containers")
    check(win.sweep_panel_widget.isHidden(), "the sweep panel should be hidden by default (sweep is off by default)")
    check(not win.normal_results_widget.isHidden(), "the normal results view should be visible by default")
    check(win.solve_btn.isEnabled(), "Solve should be enabled by default (sweep is off)")

    win.sweep_enable_cb.setChecked(True)
    check(win.sweep_panel_widget.isHidden() is False, "enabling the sweep should reveal the sweep panel")
    check(win.normal_results_widget.isHidden(), "enabling the sweep should hide the normal results view")
    check(not win.solve_btn.isEnabled(), "Solve should be disabled while Parameter Sweep is enabled")
    win.sweep_enable_cb.setChecked(False)
    check(win.normal_results_widget.isHidden() is False, "disabling the sweep should restore the normal results view")
    check(win.sweep_panel_widget.isHidden(), "disabling the sweep should hide the sweep panel again")
    check(win.solve_btn.isEnabled(), "Solve should be re-enabled once the sweep is disabled")
    # Leave it enabled for the rest of this test, same as before (most of
    # what follows configures/exercises the sweep itself).
    win.sweep_enable_cb.setChecked(True)

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

    # "zip" ("Specified combinations"): row A (phase0_optimizer, 2 values)
    # and row B (neurons_per_layer, 2 values) -- zip() pairs them up (NOT
    # a cross product), so this should give 1 baseline + 2 runs, not 1 + 4
    # like grid mode above.
    win.sweep_mode_combo.setCurrentIndex(win.sweep_mode_combo.findData("zip"))
    cfg_zip = win._build_config()
    check(cfg_zip.sweep_mode == "zip", "config should record sweep_mode='zip' once selected")
    check(cfg_zip.validate() == [], f"equal-length zip sweep should validate clean, got {cfg_zip.validate()}")
    runs_zip = build_runs(cfg_zip)
    check(len(runs_zip) == 1 + 2, f"zip with 2+2 equal-length values should give 3 runs (1 baseline + 2 paired), "
                                   f"got {len(runs_zip)}: {[l for l, _ in runs_zip]}")
    zip_labels = [l for l, _ in runs_zip[1:]]
    check(zip_labels[0].endswith("=adam, Neurons per layer=32") and zip_labels[1].endswith("=lbfgs, Neurons per layer=64"),
          f"zip mode should pair each row's Nth value together (not cross them), got {zip_labels}")

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

    # ── v68: adjustable split between sweep setup and results ────────
    from PyQt6.QtWidgets import QSplitter
    check(isinstance(win.sweep_splitter, QSplitter), "expected a QSplitter between sweep setup and results")
    check(win.sweep_splitter.orientation() == Qt.Orientation.Vertical,
          "the sweep setup/results split should be vertical (stacked top/bottom), like the Setup tab's own splitter")
    check(not win.sweep_splitter.childrenCollapsible(),
          "childrenCollapsible should be disabled so a drag can't snap a pane fully shut")
    check(win.sweep_splitter.count() == 2, f"expected exactly 2 panes (setup, results), got {win.sweep_splitter.count()}")
    default_sizes = win.sweep_splitter.sizes()
    check(len(default_sizes) == 2 and abs(default_sizes[0] - default_sizes[1]) <= max(default_sizes) * 0.05,
          f"default split should be roughly half/half, got {default_sizes}")
    # Confirm it's actually adjustable (not fixed): asking for a bigger top
    # pane should always yield a bigger top pane than asking for a smaller
    # one, even though Qt clamps the exact pixel values against each
    # pane's own minimum size in this headless/offscreen window (the
    # splitter's absolute sizes() here aren't directly comparable to a
    # real, properly-sized on-screen window -- only the relative
    # direction is a meaningful thing to assert under offscreen Qt).
    win.show()
    _app.processEvents()
    win.sweep_splitter.setSizes([600, 100])
    _app.processEvents()
    top_requested_big = win.sweep_splitter.sizes()[0]
    win.sweep_splitter.setSizes([100, 600])
    _app.processEvents()
    top_requested_small = win.sweep_splitter.sizes()[0]
    check(top_requested_big > top_requested_small,
          f"the splitter should respond to setSizes() -- requesting a bigger top pane ({top_requested_big}) "
          f"should beat requesting a smaller one ({top_requested_small})")

    # ── v68: sweep mode combo now has 3 COMSOL-style options ──────────
    mode_data = [win.sweep_mode_combo.itemData(i) for i in range(win.sweep_mode_combo.count())]
    check(mode_data == ["oat", "grid", "zip"],
          f"expected exactly 3 sweep modes in order [oat, grid, zip], got {mode_data}")

    # ── v69: Sweep Parameters as its own panel, one compact row per
    # parameter, no nested/height-capped inner scroll area ────────────
    from PyQt6.QtWidgets import QGroupBox, QScrollArea
    params_groups = [w for w in win.sweep_panel_widget.findChildren(QGroupBox)
                      if w.title() == "Sweep Parameters"]
    check(len(params_groups) == 1, f"expected exactly one 'Sweep Parameters' QGroupBox panel, got {len(params_groups)}")
    if params_groups:
        # Walk sweep_rows_widget's own parent chain up to confirm it's
        # nested inside that panel (rather than, say, a sibling further
        # up the tab).
        ancestor = win.sweep_rows_widget.parentWidget()
        found = False
        while ancestor is not None:
            if ancestor is params_groups[0]:
                found = True
                break
            ancestor = ancestor.parentWidget()
        check(found, "the sweep parameter rows should live inside the 'Sweep Parameters' panel")
    # No more inner QScrollArea wrapping just the rows (that's what caused
    # the "locked at a fixed height, has to scroll inside a tiny box even
    # though the splitter has room" complaint) -- the rows sit directly in
    # the panel, so the row list grows with its content and only the ONE
    # outer setup_scroll (or dragging sweep_splitter) handles overflow,
    # same pattern as the Setup tab's own scheduler-phase list.
    check(not isinstance(win.sweep_rows_widget.parentWidget(), QScrollArea),
          "sweep_rows_widget's direct parent should NOT be a QScrollArea -- "
          "no more nested, nested height-capped scroll area around just the parameter rows")

    # A fresh row must show exactly one of (Values field / Min-Max-Steps
    # range fields) visible, never both and never neither -- this is the
    # exact bug caught while redesigning this tab: _refresh_sweep_param_
    # choices() populates the Parameter combo with blockSignals(True), so
    # _update_value_widgets() previously never ran for a brand new row
    # until the user happened to touch a combo, leaving BOTH value
    # widgets visible at once (glaringly obvious once they sit on the
    # same line instead of further down a stacked layout).
    win2_fresh = MainWindow()
    _app.processEvents()
    win2_fresh.show()
    _app.processEvents()
    # Enabling the sweep (rather than switching tabs -- there's no
    # separate tab anymore) is what actually shows the sweep panel.
    win2_fresh.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    fresh_row = win2_fresh.sweep_row_list[0]
    list_visible = fresh_row['list_edit'].parentWidget().isVisible()
    range_visible = fresh_row['min_spin'].parentWidget().isVisible()
    check(list_visible != range_visible,
          f"a freshly added row should show exactly one of the Values field / range fields, "
          f"not both or neither -- got list_visible={list_visible}, range_visible={range_visible}")

    # Each parameter row is now a single compact line (num + Parameter +
    # Sweep as + Values/range + remove), not 3-4 stacked lines -- check
    # the row's own layout is a QHBoxLayout (not the old per-row QVBoxLayout).
    from PyQt6.QtWidgets import QHBoxLayout as _QHBoxLayout
    check(isinstance(row['widget'].layout(), _QHBoxLayout),
          "each sweep parameter row should be laid out as a single horizontal line")

    # A phase-scoped entry's own label already embeds "Phase N: " --
    # the dropdown display text should show that once, not doubled up
    # as "Phase 1: Phase 1: Optimizer".
    win3_fresh = MainWindow()
    _app.processEvents()
    row3_fresh = win3_fresh.sweep_row_list[0]
    phase_idx = row3_fresh['param_combo'].findData("phase0_optimizer")
    if phase_idx >= 0:
        text = row3_fresh['param_combo'].itemText(phase_idx)
        check(text.count("Phase 1:") == 1,
              f"a phase-scoped entry's label already says 'Phase 1: ...' -- the dropdown display "
              f"text shouldn't prepend the category a second time, got {text!r}")

    # ── v68: "Save Sweep Results To" field + Browse button ────────────
    check(hasattr(win, "sweep_save_dir_input"), "expected a dedicated sweep_save_dir_input field")
    check(win.sweep_save_dir_input.text() == "", "sweep save dir should start blank (falls back to a default path at run time)")
    expected_placeholder = os.path.join(os.path.expanduser("~"), "PINNStudio_Results", "parameter_sweep_results")
    check(win.sweep_save_dir_input.placeholderText() == expected_placeholder,
          f"placeholder should hint at the real default path, got {win.sweep_save_dir_input.placeholderText()!r}")
    check(callable(win._on_browse_sweep_save_dir), "expected a working Browse handler for the sweep save dir")
    win.sweep_save_dir_input.setText("/tmp/_v68_custom_sweep_dir")
    cfg_savedir = win._build_config()
    check(cfg_savedir.sweep_save_dir == "/tmp/_v68_custom_sweep_dir",
          f"_build_config() should carry through the custom sweep save dir, got {cfg_savedir.sweep_save_dir!r}")
    win.sweep_save_dir_input.setText("")

    # ── v68: per-sweep export/figure override (default vs custom) ────
    win.show()  # a top-level window must actually be shown for isVisible()
    # to report real values under offscreen Qt -- a widget whose window
    # was never shown reports isVisible()==False no matter what
    # setVisible() calls were made on it or its ancestors (confirmed
    # against the pre-existing, known-correct plot_custom_expr_input
    # toggle, which shows the identical false negative without .show()).
    # The sweep panel itself also needs to actually be enabled (sweep_
    # enable_cb checked) for its own widgets to report real isVisible()
    # values -- same "ancestor not really shown" flavor as the window-
    # shown check above, just one level further up the widget tree.
    win.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    check(win.sweep_export_mode_combo.currentData() == "same_as_setup",
          "export override should default to 'same as Setup tab', preserving today's existing behavior")
    check(not win.sweep_export_custom_widget.isVisible(),
          "the custom export-settings widgets should be hidden by default")
    win.sweep_export_mode_combo.setCurrentIndex(win.sweep_export_mode_combo.findData("custom"))
    _app.processEvents()
    check(win.sweep_export_custom_widget.isVisible(),
          "selecting 'Custom for this sweep' should reveal the plot-type/output/t-steps widgets")
    win.sweep_export_mode_combo.setCurrentIndex(win.sweep_export_mode_combo.findData("same_as_setup"))
    _app.processEvents()
    check(not win.sweep_export_custom_widget.isVisible(),
          "switching back to 'Same as Setup tab' should hide the custom widgets again")

    plot_type_items = [win.sweep_plot_type_combo.itemText(i) for i in range(win.sweep_plot_type_combo.count())]
    check(plot_type_items == ["Surface", "Line (time steps)", "Line Animation (GIF)", "Surface Animation (GIF)"],
          f"sweep plot-type choices should mirror the Setup tab's own 4 options, got {plot_type_items}")
    check(win.sweep_export_tsteps_spin.value() == 11 and win.sweep_export_tsteps_spin.minimum() == 2,
          "export t-steps spinner should default to 11 (today's existing default) with a sane minimum")

    # Output combo refresh + Custom... expr/label visibility toggle.
    # (These fields live inside sweep_export_custom_widget, which is only
    # shown in "custom" export mode -- switch there first.)
    win.sweep_export_mode_combo.setCurrentIndex(win.sweep_export_mode_combo.findData("custom"))
    _app.processEvents()
    win.radio_1d.setChecked(True)
    win.quick_examples_combo.setCurrentText("1D Heat")
    win._refresh_sweep_export_output_choices()
    out_items = [win.sweep_plot_output_combo.itemText(i) for i in range(win.sweep_plot_output_combo.count())]
    check(out_items[-1] == "Custom...", f"output combo should always end with a 'Custom...' entry, got {out_items}")
    check(len(out_items) >= 2, f"1D Heat (1 output) should give >=2 entries (1 output + Custom...), got {out_items}")
    win.sweep_plot_output_combo.setCurrentIndex(win.sweep_plot_output_combo.findText(out_items[0]))
    _app.processEvents()
    check(not win.sweep_plot_custom_expr_input.isVisible() and not win.sweep_plot_custom_label_input.isVisible(),
          "custom expr/label fields should be hidden while a real output is selected")
    win.sweep_plot_output_combo.setCurrentIndex(win.sweep_plot_output_combo.findText("Custom..."))
    _app.processEvents()
    check(win.sweep_plot_custom_expr_input.isVisible() and win.sweep_plot_custom_label_input.isVisible(),
          "custom expr/label fields should appear once 'Custom...' is selected")

    # ── v68: _build_config() round-trip for all 7 new export fields ──
    # (export mode is "custom" at this point -- confirm the custom
    # plot-type/expr/label/t-steps fields all make it into the config.)
    win.sweep_plot_type_combo.setCurrentText("Line (time steps)")
    win.sweep_plot_custom_expr_input.setText("sqrt(u**2)")
    win.sweep_plot_custom_label_input.setText("|u|")
    win.sweep_export_tsteps_spin.setValue(21)
    cfg_export2 = win._build_config()
    check(cfg_export2.sweep_export_mode == "custom", "export mode should round-trip to 'custom'")
    check(cfg_export2.sweep_plot_type == "Line (time steps)", f"sweep_plot_type should round-trip, got {cfg_export2.sweep_plot_type!r}")
    check(cfg_export2.sweep_plot_custom_expr == "sqrt(u**2)" and cfg_export2.sweep_plot_custom_label == "|u|",
          "custom expr/label should round-trip once 'Custom...' output is selected")
    check(cfg_export2.sweep_export_t_steps == 21, f"sweep_export_t_steps should round-trip, got {cfg_export2.sweep_export_t_steps}")
    # Selecting a real (non-Custom) output should blank out the expr/label
    # that get sent to the config, even if the text fields still hold stale text.
    win.sweep_plot_output_combo.setCurrentIndex(win.sweep_plot_output_combo.findText(out_items[0]))
    cfg_export3 = win._build_config()
    check(cfg_export3.sweep_plot_custom_expr == "" and cfg_export3.sweep_plot_custom_label == "",
          "custom expr/label should be blanked in the built config once a real output (not Custom...) is selected")
    win.sweep_export_mode_combo.setCurrentIndex(win.sweep_export_mode_combo.findData("same_as_setup"))
    cfg_export4 = win._build_config()
    check(cfg_export4.sweep_export_mode == "same_as_setup", "export mode should round-trip back to 'same_as_setup'")

    # ── v68: naming helpers (sweep_root_dir / run_folder_name / _slugify) ──
    from pinnstudio.core.sweep_runner import sweep_root_dir, run_folder_name, _slugify, _apply_run_output_settings
    import datetime as _dt
    cfg_names = win._build_config()
    cfg_names.sweep_save_dir = ""
    stamp = _dt.datetime(2026, 1, 2, 3, 4, 5)
    root_default = sweep_root_dir(cfg_names, started_at=stamp)
    check(root_default == os.path.join(os.path.expanduser("~"), "PINNStudio_Results",
                                        "parameter_sweep_results", "sweep_20260102_030405"),
          f"blank sweep_save_dir should fall back to the documented default path, got {root_default!r}")
    cfg_names.sweep_save_dir = "/tmp/_v68_custom_root"
    root_custom = sweep_root_dir(cfg_names, started_at=stamp)
    check(root_custom == "/tmp/_v68_custom_root/sweep_20260102_030405",
          f"a custom sweep_save_dir should be used as-is with a timestamped subfolder, got {root_custom!r}")
    check(run_folder_name(0, "baseline") == "run_000_baseline", "run 0 labeled 'baseline' should get a fixed, predictable folder name")
    check(run_folder_name(3, "Hidden layers=4, Neurons per layer=64") == "run_003_Hidden_layers=4_Neurons_per_layer=64",
          f"run folder names should be zero-padded, prefixed, and slugified, got {run_folder_name(3, 'Hidden layers=4, Neurons per layer=64')!r}")
    check(_slugify("BC 2 (dirichlet, Output 0)=7.5") == "BC_2_dirichlet_Output_0_=7.5",
          f"_slugify should strip filesystem-unsafe characters while keeping the run identifiable, got {_slugify('BC 2 (dirichlet, Output 0)=7.5')!r}")
    check(_slugify("") == "run", "an empty label should fall back to a safe non-empty default")

    # _apply_run_output_settings(): same_as_setup leaves plot_type/output/
    # expr/label/t_steps untouched (today's exact behavior); custom mode
    # overrides them from the sweep tab's own settings; save_dir is always
    # overridden to the given per-run folder either way.
    base_same = win._build_config()
    base_same.sweep_export_mode = "same_as_setup"
    base_same.plot_type = "Surface"
    base_same.export_t_steps = 7
    import copy as _copy
    run_cfg_same = _copy.deepcopy(base_same)
    _apply_run_output_settings(run_cfg_same, base_same, "/tmp/_v68_run_dir_same")
    check(run_cfg_same.save_dir == "/tmp/_v68_run_dir_same", "save_dir should always be overridden to the run's own folder")
    check(run_cfg_same.plot_type == "Surface" and run_cfg_same.export_t_steps == 7,
          "same_as_setup mode must leave plot_type/export_t_steps exactly as the base config had them")

    base_custom = win._build_config()
    base_custom.sweep_export_mode = "custom"
    base_custom.sweep_plot_type = "Line"
    base_custom.sweep_plot_output_idx = 2
    base_custom.sweep_plot_custom_expr = "u+v"
    base_custom.sweep_plot_custom_label = "u+v"
    base_custom.sweep_export_t_steps = 31
    run_cfg_custom = _copy.deepcopy(base_custom)
    _apply_run_output_settings(run_cfg_custom, base_custom, "/tmp/_v68_run_dir_custom")
    check(run_cfg_custom.save_dir == "/tmp/_v68_run_dir_custom", "save_dir should be overridden in custom mode too")
    check(run_cfg_custom.plot_type == "Line" and run_cfg_custom.plot_output_idx == 2
          and run_cfg_custom.plot_custom_expr == "u+v" and run_cfg_custom.plot_custom_label == "u+v"
          and run_cfg_custom.export_t_steps == 31,
          f"custom mode should override every export setting from the sweep tab's own fields, got "
          f"plot_type={run_cfg_custom.plot_type!r} idx={run_cfg_custom.plot_output_idx!r} "
          f"expr={run_cfg_custom.plot_custom_expr!r} t_steps={run_cfg_custom.export_t_steps!r}")

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
