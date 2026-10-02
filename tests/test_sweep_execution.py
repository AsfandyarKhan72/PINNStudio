#!/usr/bin/env python3
"""
Real, end-to-end Parameter Sweep execution check: actually runs
pinnstudio.core.sweep_runner.run_sweep() against a live config -- real
subprocess training, not mocked -- on 1D Heat with iterations shrunk
down purely for test speed (not training quality). This is the one
sweep test that needs torch/deepxde actually installed, which is why
it's kept out of test_sweep_tab.py and NOT wired into the CI
smoke-test workflow (that workflow only installs PyQt6/numpy/
matplotlib/pandas). Run this locally instead, wherever PINNStudio's own
install.sh/install.bat environment (or an equivalent one with torch +
deepxde) is available.

Checks:
 - A tiny real sweep (baseline + one hidden_layers variant) on 1D Heat
   runs to completion, both runs reporting status "done" with a real,
   distinct parsed final_loss, and l2_relative=None (1D Heat has no
   Error Analysis reference data configured by default, so nothing
   should ever be reported there).
 - The same sweep on 3D Poisson (Sphere) -- which DOES have Error
   Analysis reference data configured by default -- reports a real,
   distinct l2_relative for each run, confirming sweep_runner reads
   back each run's own freshly-written error_metrics.txt rather than a
   stale one left over from an earlier run (the exact bug caught by
   hand while building this feature).
 - Phase 5's MainWindow._render_sweep_plot() renders a real PNG from
   each of the above REAL run_sweep() results (not synthetic data --
   that's covered in test_sweep_tab.py; this just confirms the real
   data shape flows into it correctly).
 - A real sweep over a loss weight (sweep_registry.py's "SCOPE (v2)"
   entries) on 1D Heat: pushing the Initial Condition weight to an
   extreme value (100000, vs. the default 100) for a handful of
   iterations measurably changes the real, parsed final_loss relative
   to the baseline. test_sweep_registry.py already checks -- without
   training -- that the swept value reaches every scheduler phase's
   embedded weights string in the generated script; this is the one
   check that it also reaches an ACTUAL training run's real result, not
   just the script text.

Run directly (this trains for real, so budget roughly a minute or two
depending on the machine):
    QT_QPA_PLATFORM=offscreen python3 tests/test_sweep_execution.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.sweep_runner import run_sweep


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── 1D Heat: no Error Analysis data configured ───────────────────
    win = MainWindow()
    _app.processEvents()
    for ph in win.sched_phase_list:
        ph["iters"].setValue(15)  # shrunk purely for test speed
    row = win.sweep_row_list[0]
    row["param_combo"].setCurrentIndex(row["param_combo"].findData("hidden_layers"))
    row["mode_combo"].setCurrentIndex(row["mode_combo"].findData("list"))
    row["list_edit"].setText("2")
    win.sweep_enable_cb.setChecked(True)
    config = win._build_config()
    check(config.validate() == [], f"1D Heat sweep config should validate clean: {config.validate()}")
    check(config.ea_files in ("[]", []), "1D Heat should have no Error Analysis data configured by default")

    results = run_sweep(config)
    check(len(results) == 2, f"expected 2 runs (baseline + hidden_layers=2), got {len(results)}")
    for label, result in results:
        check(result["status"] == "done", f"[1D Heat] run {label!r} should finish 'done', got {result}")
        check(result["final_loss"] is not None, f"[1D Heat] run {label!r} should report a parsed final_loss, got {result}")
        check(result["l2_relative"] is None,
              f"[1D Heat] run {label!r} should report l2_relative=None (no EA data configured), got {result}")

    # Phase 5: the results plot should render from these REAL results
    # (not synthetic data -- test_sweep_tab.py already covers the
    # rendering logic itself with synthetic cache entries; this is the
    # one check that real run_sweep() output flows into it correctly).
    win._sweep_last_config = config
    win._sweep_results_cache = results
    win._render_sweep_plot()
    check(win.sweep_plot_label._source_path is not None and os.path.exists(win.sweep_plot_label._source_path),
          "[1D Heat] sweep results plot should render from real run_sweep() output")

    # ── 3D Poisson (Sphere): DOES have Error Analysis data configured ─
    win2 = MainWindow()
    _app.processEvents()
    win2.radio_3d.setChecked(True)
    win2.quick_examples_combo.setCurrentText("3D Poisson (Sphere)")
    _app.processEvents()
    for ph in win2.sched_phase_list:
        ph["iters"].setValue(15)
    row2 = win2.sweep_row_list[0]
    row2["param_combo"].setCurrentIndex(row2["param_combo"].findData("hidden_layers"))
    row2["mode_combo"].setCurrentIndex(row2["mode_combo"].findData("list"))
    row2["list_edit"].setText("2")
    win2.sweep_enable_cb.setChecked(True)
    config2 = win2._build_config()
    check(config2.validate() == [], f"3D Sphere sweep config should validate clean: {config2.validate()}")
    check(config2.ea_files not in ("[]", []), "3D Poisson (Sphere) should have Error Analysis data configured by default")

    results2 = run_sweep(config2)
    check(len(results2) == 2, f"expected 2 runs (baseline + hidden_layers=2), got {len(results2)}")
    l2s = []
    for label, result in results2:
        check(result["status"] == "done", f"[3D Sphere] run {label!r} should finish 'done', got {result}")
        check(result["l2_relative"] is not None,
              f"[3D Sphere] run {label!r} should report a real l2_relative (EA data IS configured), got {result}")
        l2s.append(result["l2_relative"])
    check(len(set(l2s)) == len(l2s),
          f"each 3D Sphere run should read back its OWN freshly-written error_metrics.txt, not a shared/stale one -- got identical values: {l2s}")

    win2._sweep_last_config = config2
    win2._sweep_results_cache = results2
    win2._render_sweep_plot()
    check(win2.sweep_plot_label._source_path is not None and os.path.exists(win2.sweep_plot_label._source_path),
          "[3D Sphere] sweep results plot (with real L2 data) should render from real run_sweep() output")

    # ── Loss weight sweep: a real training run, not just generated-
    # script text (test_sweep_registry.py already checks the text) ────
    win3 = MainWindow()
    _app.processEvents()
    win3.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()
    for ph in win3.sched_phase_list:
        ph["iters"].setValue(20)
    row3 = win3.sweep_row_list[0]
    row3["param_combo"].setCurrentIndex(row3["param_combo"].findData("weight_ic_0"))
    row3["mode_combo"].setCurrentIndex(row3["mode_combo"].findData("list"))
    row3["list_edit"].setText("100000")  # the default IC weight is 100 -- a 1000x jump
    win3.sweep_enable_cb.setChecked(True)
    config3 = win3._build_config()
    check(config3.validate() == [], f"1D Heat weight-sweep config should validate clean: {config3.validate()}")

    results3 = run_sweep(config3)
    check(len(results3) == 2, f"expected 2 runs (baseline + weight_ic_0=100000), got {len(results3)}")
    losses3 = [r.get("final_loss") for _, r in results3]
    check(all(v is not None for v in losses3),
          f"[1D Heat weight sweep] both runs should report a parsed final_loss, got {losses3}")
    if all(v is not None for v in losses3):
        baseline_loss3, swept_loss3 = losses3
        check(abs(swept_loss3 - baseline_loss3) > 1e-6 * max(abs(baseline_loss3), 1e-12),
              f"[1D Heat weight sweep] pushing the IC weight to 100000 should measurably change "
              f"the real final_loss vs. baseline -- got baseline={baseline_loss3}, "
              f"swept={swept_loss3} (if these are too close, the swept weight may not actually "
              "be reaching training)")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP EXECUTION TESTS PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
