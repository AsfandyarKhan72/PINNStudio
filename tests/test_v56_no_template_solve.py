#!/usr/bin/env python3
"""
Reproduces the exact bug report: open the GUI, don't select any Quick
Examples template, fill in a custom PDE (the README's own Fisher-KPP
walkthrough) with its IC and BC, leave the optimizer scheduler on its
default (Adam then L-BFGS, both with real iteration counts, as shown in
the GUI by default) and click Solve. Before this fix, config.validate()
unconditionally required config.iterations > 0 -- read from the hidden,
legacy "Phase 1 iterations" spinbox, which stays at its own unrelated
default of 0 unless a template happens to set it as a side effect of
loading -- so a from-scratch config failed with "Phase 1 iterations must
be positive" even though the scheduler phases actually driving training
were completely valid.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_v56_no_template_solve.py
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


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    # The default Adam/L-BFGS scheduler phases are populated via a
    # QTimer.singleShot(0, ...) in __init__, which only fires once the Qt
    # event loop actually turns -- pump it once here, same as it would on
    # the very first real event-loop iteration after the window opens.
    _app.processEvents()
    # Deliberately do NOT touch win.quick_examples_combo -- this is the
    # "no template selected" path the bug report is about. Confirm that
    # precondition explicitly so this test fails loudly if MainWindow ever
    # starts auto-selecting a template.
    check(getattr(win, "_current_template_type", "") == "",
          "precondition failed: a template_type is already set at startup")

    # Fill in the README's own Fisher-KPP walkthrough, verbatim.
    win.pde_inputs[0].setText("du_t - 0.01*du_xx - 1.0*u*(1 - u)")
    win.ic_inputs[0].setText("exp(-50*(x-0.5)**2)")
    win.bc_left_types[0].setCurrentText("Neumann")
    win.bc_left_vals[0].setValue(0)
    win.bc_right_types[0].setCurrentText("Neumann")
    win.bc_right_vals[0].setValue(0)
    win.x_min.setValue(0); win.x_max.setValue(1)
    win.t_min.setValue(0); win.t_max.setValue(1)

    # This is exactly what the GUI shows by default without any template:
    # scheduler on, two phases (Adam, L-BFGS) with real iteration counts,
    # legacy iter1_spin/iter2_spin hidden and left at 0.
    check(win.sched_cb.isChecked() is True,
          "precondition failed: optimizer scheduler isn't on by default")
    check(win.iter1_spin.value() == 0,
          "precondition failed: hidden legacy iter1_spin isn't 0 by default "
          "-- test no longer reproduces the reported scenario")
    check(len(win.sched_phase_list) == 2,
          f"precondition failed: expected 2 default scheduler phases, got "
          f"{len(win.sched_phase_list)}")

    config = win._build_config()
    check(config.optimizer_scheduler is True,
          "built config should have optimizer_scheduler=True (matches GUI default)")
    check(config.iterations == 0,
          "built config should still carry iterations=0 from the hidden spinbox "
          "(this is expected and fine -- it's just unused)")

    errors = config.validate()
    check(not any("phase 1 iterations" in e.lower() for e in errors),
          f"BUG REPRODUCED: validate() still rejects a from-scratch config over "
          f"the unused legacy iterations field. Errors: {errors}")
    check(errors == [],
          f"a from-scratch Fisher-KPP config with default scheduler phases "
          f"should validate cleanly; got: {errors}")

    # Exactly what _on_solve() does before launching the SolverThread.
    ok2, msg2 = win._validate_optimizer_settings(config)
    check(ok2, f"_validate_optimizer_settings rejected a valid config: {msg2}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OK: from-scratch Fisher-KPP config (no template) validates "
              "cleanly with the default Adam/L-BFGS scheduler phases.")
    return failures


def test_v56_no_template_solve():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
