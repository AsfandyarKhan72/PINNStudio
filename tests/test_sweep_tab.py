#!/usr/bin/env python3
"""
Regression check for Phase 3 (the Parameter Sweep configuration in
main_window.py -- since v71, fully inline in the left training panel's
own "Parameter Sweep" group, right after Adaptive Training: an Enable
checkbox, Mode picker, and the parameter list itself, with no separate
tab, no separate results panel, and no dedicated Run/Cancel/Export UI
-- a sweep reuses the main Solve/Stop buttons and the main Training
Log, and each run's save location/plot-export settings are just the
Setup tab's own "Save to:"/plot settings) and Phase 4
(pinnstudio/core/sweep_runner.py, the execution driver). Phase 1/2
(config fields + registry) already has its own coverage in
test_sweep_registry.py; this file is specifically about the GUI wiring
on top of it and the driver that turns a configured sweep into real
runs.

Checks:
 - The sweep panel's own widgets exist in the left panel (Enable
   checkbox, off by default; Mode picker; the parameter row list and
   its Add/Refresh buttons), and a default row is present on boot.
 - Enabling the sweep does NOT touch solve_btn's enabled state or the
   right panel at all -- it only relabels Solve to "Run Sweep" (see
   _on_sweep_enabled_toggled) -- and _on_solve()/_on_stop() route to
   the sweep (_start_sweep/cancelling the sweep thread) vs. a normal
   single run purely based on config.sweep_enabled, with no separate
   Run Sweep/Cancel buttons anymore.
 - Add/remove row wiring (_add_sweep_row / its remove button closure);
   each row is a stacked block (QVBoxLayout: header + Parameter +
   Sweep-as + Values/range), the same layout convention Training
   Phases already use in this same narrow column -- not v69's single-
   line-per-row table layout, which needed far more width than this
   column has.
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
 - A sweep no longer has its own save-location or export-settings
   fields: _build_config() always sends sweep_save_dir from the main
   save_dir_input (the same folder a normal Solve uses) and
   sweep_export_mode="same_as_setup" unconditionally.
 - sweep_runner.build_runs(): "oat" produces baseline + one run per
   listed value per row (not a cross product); "grid" produces
   baseline + the full Cartesian product; "zip" pairs rows instead of
   crossing them.
 - sweep_runner's stale-results guard: a leftover error_metrics.txt
   from BEFORE a run started must never be reported as that run's
   result (this exact bug was caught by hand while building this
   feature -- see sweep_runner._read_last_l2's `not_before` parameter).
 - sweep_runner naming helpers (sweep_root_dir / run_folder_name /
   _slugify / _apply_run_output_settings) -- unaffected by the GUI
   relocation, still exercised directly here since they're what
   _start_sweep's SweepThread actually calls.

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

    # ── Single "Setup" tab (no separate "Parameter Sweep" tab) ───────
    tab_texts = [win.central_tabs.tabText(i) for i in range(win.central_tabs.count())]
    check(tab_texts == ["Setup"], f"expected only a single 'Setup' tab (no separate 'Parameter Sweep' tab), got {tab_texts}")
    check(not win.central_tabs.tabBar().isVisible() or win.central_tabs.count() <= 1,
          "the tab bar should stay hidden with only one tab")

    # ── Enable toggle lives in the left panel; no right-panel takeover ─
    check(hasattr(win, "sweep_enable_cb"), "expected an Enable Parameter Sweep checkbox")
    check(not win.sweep_enable_cb.isChecked(), "Parameter Sweep should be OFF by default")
    check(not hasattr(win, "sweep_panel_widget") and not hasattr(win, "normal_results_widget"),
          "there should be no separate sweep results panel/right-panel swap anymore -- "
          "the sweep lives entirely in the left panel and reuses the normal results view")
    check(win.solve_btn.isEnabled(), "Solve should start enabled")
    check(win.solve_btn.text().strip() == "▶  Solve", f"Solve button should read '▶  Solve' by default, got {win.solve_btn.text()!r}")

    win.sweep_enable_cb.setChecked(True)
    check(win.solve_btn.isEnabled(), "Solve should STAY enabled once Parameter Sweep is enabled -- it's reused to run the sweep, not disabled")
    check(win.solve_btn.text().strip() == "▶  Run Sweep", f"Solve button should relabel to '▶  Run Sweep' once enabled, got {win.solve_btn.text()!r}")
    win.sweep_enable_cb.setChecked(False)
    check(win.solve_btn.text().strip() == "▶  Solve", "Solve button should relabel back once disabled")
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

    # ── _on_solve()/_on_stop() route by config.sweep_enabled, with no
    # separate Run Sweep/Cancel buttons anymore -- exercised now that
    # the row above makes for a config that actually validates clean,
    # since _on_solve() refuses to start anything otherwise. ─────────
    _sweep_calls = []
    win._start_sweep = lambda cfg: _sweep_calls.append(cfg)
    win._on_solve()
    check(len(_sweep_calls) == 1, "clicking Solve while Parameter Sweep is enabled should route to _start_sweep(), not a normal single run")
    check(_sweep_calls[0].sweep_enabled is True, "the config handed to _start_sweep should itself have sweep_enabled set")
    check(win.solve_btn.text().strip() != "⏳  Solving...",
          "routing to a sweep must not touch the normal single-run 'Solving...' button state")
    del win._start_sweep  # restore the real bound method for the rest of this test

    class _FakeSweepThread:
        def __init__(self):
            self.stopped = False
        def isRunning(self):
            return True
        def stop(self):
            self.stopped = True
        def wait(self, _ms):
            pass

    fake_thread = _FakeSweepThread()
    win.sweep_thread = fake_thread
    win.log_box.clear()
    win._on_stop()
    check(fake_thread.stopped, "Stop should hard-stop an in-progress sweep thread (kills the current run's subprocess, not a graceful finish-then-stop)")
    check("Stopped by user" in win.log_box.toPlainText(), "Stop should log the same message a normal single-run Stop does")
    check(win.solve_btn.isEnabled() and "Run Sweep" in win.solve_btn.text(),
          "Stop should immediately re-enable Solve/relabel it back to Run Sweep, not wait for _on_sweep_finished")
    check(not win.stop_btn.isEnabled(), "Stop should immediately disable the Stop button itself")
    del win.sweep_thread

    # A categorical row's list values should be kept as strings, not
    # coerced to floats (they're optimizer names, not numbers).
    combo.setCurrentIndex(combo.findData("phase0_optimizer"))
    row['list_edit'].setText("adam, lbfgs")
    config2 = win._build_config()
    entries2 = json.loads(config2.sweep_parameters)
    check(entries2 == [{"id": "phase0_optimizer", "mode": "list", "values": ["adam", "lbfgs"]}],
          f"categorical sweep values should stay as strings: {entries2}")
    check(config2.validate() == [], f"well-formed categorical sweep should validate clean, got {config2.validate()}")

    # ── No more separate sweep save-location/export settings -- reuse
    # the Setup tab's own "Save to:" field and plot/export settings ──
    check(not hasattr(win, "sweep_save_dir_input"), "there should be no dedicated sweep save-dir field anymore")
    check(not hasattr(win, "sweep_export_mode_combo"), "there should be no dedicated sweep export-mode combo anymore")
    win.save_dir_input.setText("/tmp/_v71_shared_save_dir")
    cfg_savedir = win._build_config()
    check(cfg_savedir.sweep_save_dir == "/tmp/_v71_shared_save_dir",
          f"a sweep should reuse the main Save to: folder, got {cfg_savedir.sweep_save_dir!r}")
    check(cfg_savedir.sweep_export_mode == "same_as_setup",
          f"sweep export mode should always be 'same_as_setup' now (no GUI override), got {cfg_savedir.sweep_export_mode!r}")

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

    # ── v68: sweep mode combo still has the 3 COMSOL-style options ───
    mode_data = [win.sweep_mode_combo.itemData(i) for i in range(win.sweep_mode_combo.count())]
    check(mode_data == ["oat", "grid", "zip"],
          f"expected exactly 3 sweep modes in order [oat, grid, zip], got {mode_data}")

    # ── Sweep Parameters live inline in the left panel's "Parameter
    # Sweep" group, no nested/height-capped inner scroll area ────────
    from PyQt6.QtWidgets import QGroupBox, QScrollArea
    sweep_groups = [w for w in win.findChildren(QGroupBox) if w.title() == "Parameter Sweep"]
    check(len(sweep_groups) == 1, f"expected exactly one 'Parameter Sweep' QGroupBox panel, got {len(sweep_groups)}")
    if sweep_groups:
        # Walk sweep_rows_widget's own parent chain up to confirm it's
        # nested inside that one panel (rather than, say, a sibling
        # further up the left column).
        ancestor = win.sweep_rows_widget.parentWidget()
        found = False
        while ancestor is not None:
            if ancestor is sweep_groups[0]:
                found = True
                break
            ancestor = ancestor.parentWidget()
        check(found, "the sweep parameter rows should live inside the 'Parameter Sweep' panel")
    check(not isinstance(win.sweep_rows_widget.parentWidget(), QScrollArea),
          "sweep_rows_widget's direct parent should NOT be a QScrollArea -- "
          "no nested, height-capped scroll area around just the parameter rows")

    # A fresh row must show exactly one of (Values field / Min-Max-Steps
    # range fields) visible, never both and never neither.
    win2_fresh = MainWindow()
    _app.processEvents()
    win2_fresh.show()
    _app.processEvents()
    win2_fresh.sweep_enable_cb.setChecked(True)
    _app.processEvents()
    fresh_row = win2_fresh.sweep_row_list[0]
    list_visible = fresh_row['list_edit'].parentWidget().isVisible()
    range_visible = fresh_row['min_spin'].parentWidget().isVisible()
    check(list_visible != range_visible,
          f"a freshly added row should show exactly one of the Values field / range fields, "
          f"not both or neither -- got list_visible={list_visible}, range_visible={range_visible}")

    # Each parameter row is now a stacked block (header + Parameter +
    # Sweep as + Values/range), the same convention Training Phases use
    # in this same narrow column -- not v69's single-line table layout.
    from PyQt6.QtWidgets import QVBoxLayout as _QVBoxLayout
    check(isinstance(row['widget'].layout(), _QVBoxLayout),
          "each sweep parameter row should be laid out as a stacked (vertical) block, matching "
          "the Training Phases' own per-row layout in this narrow column")

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

    # _apply_run_output_settings(): same_as_setup (now the GUI's only
    # option) leaves plot_type/output/expr/label/t_steps untouched;
    # save_dir is always overridden to the given per-run folder. The
    # "custom" branch itself is exercised directly here too (it's still
    # reachable code in sweep_runner.py, just no longer GUI-selectable)
    # so a future change there doesn't silently go untested.
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
          f"custom mode (still reachable in sweep_runner.py, just unused by the GUI now) should override "
          f"every export setting, got plot_type={run_cfg_custom.plot_type!r} idx={run_cfg_custom.plot_output_idx!r} "
          f"expr={run_cfg_custom.plot_custom_expr!r} t_steps={run_cfg_custom.export_t_steps!r}")

    # NOTE: a real, tiny, end-to-end run_sweep() execution (actual
    # subprocess training) is intentionally NOT exercised here -- this
    # file is wired into CI's smoke-test workflow, which only installs
    # PyQt6/numpy/matplotlib/pandas (no torch/deepxde), matching every
    # other file in this suite (test_templates.py etc. only ever call
    # generate_script(), never actually run it). That real-execution
    # check lives in test_sweep_execution.py instead, run locally where
    # torch/deepxde are installed -- see that file's docstring.

    # ── Once a sweep finishes, the right panel should show the LAST
    # successfully-completed run's own loss/solution figure -- the same
    # way a normal single Solve already does (_display_run_result_plots,
    # shared between _on_done() and _on_sweep_finished()). Exercised here
    # with synthetic result dicts/fake PNG files (no real training, to
    # stay CI-safe) by driving the exact signal handlers SweepThread
    # would fire, in order. ─────────────────────────────────────────────
    import tempfile
    tmp_root = tempfile.mkdtemp(prefix="v72_sweep_plot_")
    try:
        run0_dir = os.path.join(tmp_root, "run_000_baseline", "solution_results")
        run1_dir = os.path.join(tmp_root, "run_001_hidden_layers_2", "solution_results")
        run2_dir = os.path.join(tmp_root, "run_002_hidden_layers_4_FAILED")  # no solution_results at all
        os.makedirs(run0_dir)
        os.makedirs(run1_dir)
        os.makedirs(run2_dir)
        from PIL import Image
        for d in (run0_dir, run1_dir):
            Image.new("RGB", (4, 4)).save(os.path.join(d, "loss_plot.png"))
            Image.new("RGB", (4, 4)).save(os.path.join(d, "solution_plot.png"))

        sweep_cfg = win._build_config()
        sweep_cfg.plot_type = "Surface"
        win._sweep_run_count = 0
        win._sweep_run_root = tmp_root
        win._sweep_base_config = sweep_cfg
        win._sweep_last_run_dir = None
        win._sweep_last_run_label = None

        win._on_sweep_run_done(0, 3, "baseline", {"status": "done", "final_loss": 1.0, "l2_relative": None, "save_dir": os.path.dirname(run0_dir)})
        win._on_sweep_run_done(1, 3, "Hidden layers=2", {"status": "done", "final_loss": 0.5, "l2_relative": None, "save_dir": os.path.dirname(run1_dir)})
        # The LAST run in the sequence errors out -- the panel should
        # still show run 1's figure (the last one that actually
        # succeeded), not nothing and not a crash trying to load files
        # that were never written.
        win._on_sweep_run_done(2, 3, "Hidden layers=4", {"status": "error", "final_loss": None, "l2_relative": None, "save_dir": run2_dir})

        check(win._sweep_last_run_dir == os.path.dirname(run1_dir),
              f"the last SUCCESSFUL run's folder should be remembered even though a later run errored, got {win._sweep_last_run_dir}")

        win._on_sweep_finished()
        check(win.loss_label._source_path == os.path.join(run1_dir, "loss_plot.png"),
              f"the right panel should show the last successful run's loss figure, got {win.loss_label._source_path!r}")
        check(win.solution_label._source_path == os.path.join(run1_dir, "solution_plot.png"),
              f"the right panel should show the last successful run's solution figure, got {win.solution_label._source_path!r}")
        check("last completed run" in win.log_box.toPlainText(),
              "the log should mention that it's showing the last completed run's figure")
    finally:
        import shutil as _shutil
        _shutil.rmtree(tmp_root, ignore_errors=True)

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
