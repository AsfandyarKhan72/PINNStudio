#!/usr/bin/env python3
"""
RAR (Residual-based Adaptive Refinement) was wrongly gated to time-
dependent problems only: _on_steady_state_changed() hid the ENTIRE
"Adaptive Training" group box the moment Steady-state was checked,
even though only Time Adaptive Training (which trains one model per
time step) actually needs a time axis -- RAR just resamples
collocation points toward wherever the PDE residual is currently
largest, which works identically whether or not the domain has a time
dimension. The practical symptom (reported by a user): the Adaptive
Training panel disappeared entirely for a steady-state problem like a
Poisson equation, with no way to turn RAR on at all.

Fix: _on_steady_state_changed() no longer hides adapt_group. Instead,
it disables (not removes) the "Time Adaptive Training" combo item --
mirroring, but kept independent of, the existing suspend/restore
bookkeeping Inverse mode uses to remove this same item for a different
reason (_current_ta_cfg/_ta_suspended_for_inverse) -- and only forces
the current selection back to "None" if Time Adaptive Training was the
one active when Steady-state toggles on. A RAR (or "None") selection
is left untouched across the toggle.

Checks:
 - adapt_group stays visible when Steady-state is checked (previously
   hidden entirely).
 - The "Time Adaptive Training" combo item is disabled while Steady-
   state is checked, and re-enabled when it's unchecked again.
 - Selecting "Time Adaptive Training" and then checking Steady-state
   forces the selection back to "None" (unchanged behavior -- Time
   Adaptive genuinely doesn't apply to a steady problem).
 - Selecting "Residual-based Adaptive Refinement (RAR)" and then
   checking Steady-state leaves RAR selected (this is the actual fix --
   previously this also got silently reset to "None" along with
   everything else the old unconditional reset touched).
 - With RAR selected and Steady-state checked, the RAR settings widget
   (rar_widget) is still shown -- the panel is fully usable, not just
   visible-but-empty.
 - A real, tiny RAR training run with steady_state=True and
   adapt_method="RAR" (built via _build_config() from this exact live
   GUI state) actually produces a per-round solution_plot.png --
   confirming the GUI's own config construction, not just codegen.py in
   isolation, produces a config that trains successfully.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_steady_state_rar_gating.py
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.core.codegen import generate_script


def _ta_item_enabled(win):
    idx = win.adapt_combo.findText("Time Adaptive Training")
    item = win.adapt_combo.model().item(idx)
    return item.isEnabled()


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── adapt_group stays visible for a steady-state problem ───────
    win = MainWindow()
    win.show()
    _app.processEvents()
    check(not win.steady_state_check.isChecked(), "sanity: steady-state should start unchecked")

    win.steady_state_check.setChecked(True)
    _app.processEvents()
    check(win.adapt_group.isVisible(),
          "the Adaptive Training group should stay visible for a steady-state problem "
          "(RAR applies here even though Time Adaptive Training doesn't)")

    # ── Time Adaptive Training combo item is disabled while steady ──
    check(not _ta_item_enabled(win),
          "the 'Time Adaptive Training' combo item should be disabled while Steady-state is checked")
    win.steady_state_check.setChecked(False)
    _app.processEvents()
    check(_ta_item_enabled(win),
          "the 'Time Adaptive Training' combo item should be re-enabled after unchecking Steady-state")

    # ── Selecting Time Adaptive, then checking steady -> forced to None ──
    win2 = MainWindow()
    _app.processEvents()
    win2.adapt_combo.setCurrentText("Time Adaptive Training")
    _app.processEvents()
    win2.steady_state_check.setChecked(True)
    _app.processEvents()
    check(win2.adapt_combo.currentText() == "None",
          f"checking Steady-state while 'Time Adaptive Training' was selected should reset it to "
          f"'None', got {win2.adapt_combo.currentText()!r}")

    # ── Selecting RAR, then checking steady -> RAR stays selected ──
    win3 = MainWindow()
    win3.show()
    _app.processEvents()
    win3.adapt_combo.setCurrentText("Residual-based Adaptive Refinement (RAR)")
    _app.processEvents()
    win3.steady_state_check.setChecked(True)
    _app.processEvents()
    check(win3.adapt_combo.currentText() == "Residual-based Adaptive Refinement (RAR)",
          f"checking Steady-state while RAR was selected should leave RAR selected (RAR works for "
          f"steady problems too), got {win3.adapt_combo.currentText()!r}")
    check(win3.rar_widget.isVisible(),
          "the RAR settings widget should still be visible/usable with RAR selected under Steady-state")

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("Partial run (UI-only): deepxde/torch not installed in this environment, "
              "skipping the real-training check below.")
        if failures:
            print("FAILURES:")
            for f in failures:
                print(f"  - {f}")
        return failures

    # ── Real, tiny RAR run from the GUI's own _build_config() output ──
    win4 = MainWindow()
    _app.processEvents()
    win4.radio_1d.setChecked(True)
    win4._on_dim_changed()
    win4.steady_state_check.setChecked(True)
    win4.adapt_combo.setCurrentText("Residual-based Adaptive Refinement (RAR)")
    win4.pde_inputs[0].setText("du_xx + 1")
    win4.layers_spin.setValue(1)
    win4.neurons_spin.setValue(20)
    win4.rar_cycles.setValue(1)
    win4.rar_candidates.setValue(150)
    win4.rar_add_points.setValue(8)
    win4.rar_adam_iters.setValue(5)
    win4.rar_lbfgs_iters.setValue(0)
    win4.iter1_spin.setValue(5)
    win4.iter2_spin.setValue(0)
    _app.processEvents()

    with tempfile.TemporaryDirectory() as tmpdir:
        win4.save_dir_input.setText(tmpdir)
        cfg = win4._build_config()
        check(cfg.steady_state is True, "sanity: built config should have steady_state=True")
        check(cfg.adapt_method == "RAR", f"sanity: built config should have adapt_method='RAR', got {cfg.adapt_method!r}")

        script = generate_script(cfg)
        sp = os.path.join(tmpdir, "gui_rar_steady_script.py")
        with open(sp, "w") as f:
            f.write(script)
        proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=120)
        check(proc.returncode == 0,
              f"a real RAR run built from live GUI state (steady-state) should exit cleanly, "
              f"got {proc.returncode}: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

        # cfg.save_dir is a per-run timestamped subfolder of tmpdir
        # (see MainWindow._run_results_dir()), not tmpdir itself.
        sol = os.path.join(cfg.save_dir, "solution_results", "rar_rounds", "round_01", "solution_plot.png")
        check(os.path.exists(sol) and os.path.getsize(sol) > 0,
              "the GUI-built steady-state RAR run should produce a non-empty round_01/solution_plot.png")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OK: RAR stays available (and the Adaptive Training panel stays visible) for steady-state "
              "problems, Time Adaptive Training is correctly disabled instead, and a real GUI-built "
              "steady-state RAR config trains successfully end to end.")
    return failures


def test_steady_state_rar_gating():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
