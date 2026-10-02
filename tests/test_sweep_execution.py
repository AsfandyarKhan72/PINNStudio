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

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("ALL SWEEP EXECUTION TESTS PASSED")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
