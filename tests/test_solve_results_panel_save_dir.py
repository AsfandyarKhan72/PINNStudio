#!/usr/bin/env python3
"""
Reported bug: after a normal (non-sweep, non-restore) Solve run finished,
the right-side Results panel sometimes showed a stale/unrelated plot --
e.g. running a 1D Heat problem showed what looked like a leftover 3D Heat
surface plot -- instead of the figure the just-finished run actually
produced for whatever output (raw network output or a derivative-aware
Custom expression) was configured.

Root cause: _on_done() (main_window.py) decided what to show using TWO
wrong sources instead of the run it just finished:

    self._last_config = self._build_config()          # (1)
    save_dir = self.save_dir_input.text().strip()      # (2)
    self._display_run_result_plots(self._last_config, save_dir)

(1) _build_config() calls _run_results_dir()/_timestamped_save_dir(),
which stamps a BRAND NEW "<label>__<timestamp>" folder name every time
it's called (see test_run_results_folder_naming.py). Calling it again
here -- after training already finished -- computes a different
timestamp than the one SolverThread actually trained with, so even
_last_config.save_dir would point at a folder that was never created.

(2) self.save_dir_input.text() is just the Setup tab's PARENT "Save to:"
location (e.g. ".../PINNStudio_Results"), not this run's own unique
timestamped subfolder. _display_run_result_plots()'s "older runs"
fallback then looks directly in that parent folder for loss_plot.png/
solution_plot.png -- so if an unrelated earlier run (e.g. a 3D Heat run,
or anything predating the per-run-folder feature) ever left a plot
sitting directly in that parent folder, THAT stale file gets shown for
every subsequent run, regardless of problem/output.

This is the exact same bug class the Restore tab already had and fixed
(see test_restore_display_refresh.py / _last_restore_save_dir) -- a
stale plot sitting in a shared/parent location winning over the just-
finished run's own figure -- just never applied to the plain Solve path.

Fix: _on_done() now reuses self.thread.config -- the EXACT PINNConfig
object SolverThread actually trained with (see run_pinn()/generate_script(),
which embed config.save_dir literally into the subprocess's own script)
-- and reads save_dir from THAT config, instead of rebuilding a fresh
config or reading the raw Setup-tab widget text.

Checks (synthetic fake PNGs / a fake thread object, no real training --
same CI-safe style as test_sweep_tab.py's v72 check, since this file has
no torch/deepxde dependency):
 - A stale loss_plot.png/solution_plot.png sitting directly in the
   configured parent "Save to:" folder (simulating an unrelated earlier
   run) is left alone -- _on_done("DONE") must NOT display it.
 - The right panel shows the just-finished run's OWN loss_plot.png/
   solution_plot.png, read from self.thread.config.save_dir's own
   solution_results/ subfolder, not from a freshly-recomputed (and
   never-created) timestamped folder and not from the parent.
 - self._last_config is the SAME object as self.thread.config (not a
   fresh self._build_config() rebuild).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_solve_results_panel_save_dir.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.main_window import MainWindow


class _FakeSolverThread:
    """Stand-in for the real SolverThread -- _on_done only ever reads
    `.config` off of self.thread, so this is all that's needed to drive
    it without spawning a real background thread/subprocess."""
    def __init__(self, config):
        self.config = config


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    win = MainWindow()
    _app.processEvents()

    parent_dir = tempfile.mkdtemp(prefix="v_results_panel_parent_")
    try:
        # Stale files sitting directly in the PARENT save-to folder --
        # simulating a leftover from an unrelated earlier run (e.g. a 3D
        # Heat run, or anything saved before per-run folders existed).
        stale_loss = os.path.join(parent_dir, "loss_plot.png")
        stale_sol = os.path.join(parent_dir, "solution_plot.png")
        with open(stale_loss, "wb") as f:
            f.write(b"STALE loss plot from an unrelated earlier run")
        with open(stale_sol, "wb") as f:
            f.write(b"STALE solution plot from an unrelated earlier run (e.g. 3D Heat)")

        # This run's OWN unique timestamped subfolder (what
        # _run_results_dir() would actually have produced when Solve was
        # originally clicked) -- the real files THIS run wrote.
        run_dir = os.path.join(parent_dir, "1D_Heat__2026-10-06_23-08-09")
        sol_results_dir = os.path.join(run_dir, "solution_results")
        os.makedirs(sol_results_dir)
        real_loss = os.path.join(sol_results_dir, "loss_plot.png")
        real_sol = os.path.join(sol_results_dir, "solution_plot.png")
        with open(real_loss, "wb") as f:
            f.write(b"this run's own real loss plot")
        with open(real_sol, "wb") as f:
            f.write(b"this run's own real solution plot")

        # Setup tab still shows the raw PARENT text (exactly as it would
        # between Solve and the run finishing) -- _on_done must NOT read
        # save_dir from this widget.
        win.save_dir_input.setText(parent_dir)

        config = win._build_config()
        config.save_dir = run_dir          # the folder this (fake) run actually trained into
        config.plot_type = "Surface"

        win.thread = _FakeSolverThread(config)
        win._on_done("DONE")

        check(win._last_config is config,
              "_on_done should reuse self.thread.config as-is (the exact config the run trained "
              "with), not rebuild a fresh one via self._build_config()")

        loss_shown = getattr(win.loss_label, "_source_path", None)
        sol_shown = getattr(win.solution_label, "_source_path", None)
        check(loss_shown == real_loss,
              f"loss panel should show THIS run's own loss_plot.png ({real_loss!r}), got {loss_shown!r} "
              f"(BUG: stale/parent-folder plot shown instead)")
        check(sol_shown == real_sol,
              f"solution panel should show THIS run's own solution_plot.png ({real_sol!r}), got {sol_shown!r} "
              f"(BUG: stale/parent-folder plot shown instead, e.g. an unrelated 3D Heat surface)")
        check(loss_shown != stale_loss and sol_shown != stale_sol,
              "the stale files sitting directly in the parent save-to folder must be ignored")

        # "Results saved to:" log line must point at the run's own
        # folder too, not the bare parent text.
        log_text = win.log_box.toPlainText()
        check(run_dir in log_text and parent_dir.rstrip("/") + "\n" not in log_text,
              f"log should report this run's own folder ({run_dir!r}), not just the parent, got: {log_text!r}")
    finally:
        import shutil
        shutil.rmtree(parent_dir, ignore_errors=True)

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("OK: Results panel shows the just-finished run's OWN plots from its own "
              "unique save folder, not a stale/parent-folder leftover.")
    return failures


def test_solve_results_panel_uses_own_run_save_dir():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
