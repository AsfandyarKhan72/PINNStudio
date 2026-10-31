#!/usr/bin/env python3
"""Screenshots for the user guide (website/guide/), from the real interface.

    QT_QPA_PLATFORM=offscreen python tools/docs/shots_guide.py website/assets/screenshots [--run-folder DIR]

Modelled on tools/docs/make_screenshots.py: opens the real MainWindow offscreen, sets widgets the
way a user would (New problem, typing, ticking, picking from lists), selects pages of the model tree
and saves the window or the open dialog. Nothing is trained and nothing is drawn on the images.

--run-folder DIR: a finished run folder (the folder Solve writes, with solution_results/ and
error_analysis/). It is copied into the neutral home folder's PINNStudio_Results so the History tab
and the Restore & visualize page can be shown with a real saved run.
Output files are named guide_*.png.
"""
import os
import shutil
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ["HOME"] = os.environ.get("PINNSTUDIO_DOCS_HOME", "/tmp/home/user")
os.makedirs(os.environ["HOME"], exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from PyQt6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

W, H = 1600, 1000
OUT = None


def settle(s=0.3):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def pick(w, category, key):
    w.quick_examples_category_combo.setCurrentText(category)
    settle(0.2)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    if i < 0:
        raise SystemExit(f"example not found: {category} / {key}")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    settle(1.0)
    w.log_box.clear()
    settle(0.1)


def new(w, dim):
    w._new_problem(dim)
    settle(0.8)
    w.log_box.clear()
    settle(0.1)


def page(w, key):
    w.workspace.select(key)
    settle(0.4)


def save(w, name):
    p = os.path.join(OUT, name)
    w.grab().save(p)
    print("saved", p, flush=True)


def grab_dialog(name, delay_ms=900):
    """Grab the next modal dialog that opens, then close it."""
    def _do():
        d = QApplication.activeModalWidget()
        if d is not None:
            d.grab().save(os.path.join(OUT, name))
            print("saved", name, flush=True)
            d.reject()
    QTimer.singleShot(delay_ms, _do)


def shot(fn):
    """Run one screenshot step; a failure is reported and the others go on.
    --only NAME runs just the step NAME."""
    if "--only" in sys.argv and sys.argv[sys.argv.index("--only") + 1] != fn.__name__:
        return
    try:
        fn()
    except Exception as e:      # noqa: BLE001
        print("FAILED", fn.__name__, repr(e), flush=True)


def main():
    global OUT
    OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
                          else "website/assets/screenshots")
    os.makedirs(OUT, exist_ok=True)
    run_src = None
    if "--run-folder" in sys.argv:
        run_src = sys.argv[sys.argv.index("--run-folder") + 1]
    results = os.path.join(os.environ["HOME"], "PINNStudio_Results")
    os.makedirs(results, exist_ok=True)
    run_dst = None
    if run_src and os.path.isdir(run_src):
        run_dst = os.path.join(results, os.path.basename(os.path.normpath(run_src)))
        if not os.path.isdir(run_dst):
            shutil.copytree(run_src, run_dst)

    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(W, H)
    w.show()
    settle(1.5)

    # ── a new 1D problem: training phases as PINNStudio starts them ──
    def training_default():
        new(w, "1D")
        page(w, "training")
        save(w, "guide_training_default.png")
    shot(training_default)

    def weights_per_phase():
        w.sched_same_weights_cb.setChecked(False)
        settle(0.3)
        page(w, "weights")
        save(w, "guide_weights_per_phase.png")
        w.sched_same_weights_cb.setChecked(True)
        settle(0.2)
    shot(weights_per_phase)

    # ── boundary conditions: a Dirichlet row on the left, a Neumann row on the right ──
    def bc_rows():
        new(w, "1D")
        w.add_custom_bc_btn.click()
        settle(0.3)
        w.add_custom_bc_btn.click()
        settle(0.3)
        e1, e2 = w.custom_bc_list[0], w.custom_bc_list[1]
        e1["location"].setText("left")
        e2["type"].setCurrentIndex(e2["type"].findData("neumann"))
        e2["location"].setText("right")
        e2["value"].setText("1")
        settle(0.5)
        page(w, "bcs")
        save(w, "guide_bcs_list.png")
        node = e2.get("_node")
        if node:
            page(w, node)
            save(w, "guide_bc_neumann.png")
        page(w, "weights")
        save(w, "guide_weights_with_bcs.png")
    shot(bc_rows)

    # ── an output name that is refused (v1.6.0) ──
    def bad_name():
        new(w, "1D")
        w.output_name_inputs[0].setText("y")
        settle(0.3)
        w._check_problems()
        settle(0.3)
        page(w, "equations")
        save(w, "guide_output_name_refused.png")
        w.output_name_inputs[0].setText("u")
        settle(0.2)
        w._check_problems()
    shot(bad_name)

    # ── two outputs: names, equations and IC rows follow ──
    def two_outputs():
        new(w, "1D")
        w.num_outputs_spin.setValue(2)
        settle(0.5)
        page(w, "equations")
        save(w, "guide_equations_two_outputs.png")
        page(w, "ic")
        save(w, "guide_ic_two_outputs.png")
    shot(two_outputs)

    # ── ODE ──
    def ode():
        new(w, "ODE")
        page(w, "equations")
        w.pde_hint_toggle.setChecked(True)
        settle(0.3)
        save(w, "guide_ode_equations.png")
        w.pde_hint_toggle.setChecked(False)
    shot(ode)

    # ── 2D geometries ──
    def disk():
        new(w, "2D")
        w.geometry_type_combo.setCurrentText("Disk")
        settle(0.5)
        page(w, "geometry")
        w.right_tabs.setCurrentIndex(w._domain_tab_index)
        settle(1.0)
        save(w, "guide_geometry_disk.png")
    shot(disk)

    def custom():
        new(w, "2D")
        w.geometry_type_combo.setCurrentText("Custom")
        settle(0.5)
        w._add_custom_geom_shape("Disk", "subtract", {"cx": 0.5, "cy": 0.5, "r": 0.2})
        settle(0.6)
        page(w, "geometry")
        w.right_tabs.setCurrentIndex(w._domain_tab_index)
        settle(1.0)
        save(w, "guide_geometry_custom.png")
    shot(custom)

    def sketch():
        new(w, "2D")
        w.geometry_type_combo.setCurrentText("Sketch")
        settle(1.0)
        page(w, "geometry")
        settle(0.6)
        save(w, "guide_geometry_sketch.png")
    shot(sketch)

    # ── adaptive training and automatic weighting ──
    def adaptive():
        new(w, "1D")
        w.adapt_combo.setCurrentText("Time Adaptive Training")
        settle(0.4)
        page(w, "adaptive")
        sizes = w.main_splitter.sizes()
        w.main_splitter.setSizes([230, 860, 510])   # the step-group row is wide
        settle(0.5)
        save(w, "guide_adaptive_time.png")
        w.main_splitter.setSizes(sizes)
        settle(0.3)
        w.adapt_combo.setCurrentText("Residual-based Adaptive Refinement (RAR)")
        settle(0.4)
        page(w, "adaptive")
        save(w, "guide_adaptive_rar.png")
        w.adapt_combo.setCurrentText("None")
        settle(0.2)
    shot(adaptive)

    def auto_weights():
        new(w, "1D")
        w._set_combo_data(w.point_weights_combo, "RBA")
        settle(0.3)
        page(w, "weights")
        save(w, "guide_weights_rba.png")
        w._set_combo_data(w.point_weights_combo, "None")
    shot(auto_weights)

    # ── a parameter sweep with one row ──
    def sweep():
        new(w, "1D")
        w.sweep_enable_cb.setChecked(True)
        settle(0.5)
        row = w.sweep_row_list[0]
        w._refresh_sweep_param_choices()
        i = row["param_combo"].findData("phase0_lr")
        if i >= 0:
            row["param_combo"].setCurrentIndex(i)
        row["mode_combo"].setCurrentIndex(row["mode_combo"].findData("list"))
        row["list_edit"].setText("0.001, 0.0005, 0.0001")
        settle(0.4)
        page(w, "sweep")
        save(w, "guide_sweep_row.png")
        w.sweep_enable_cb.setChecked(False)
        settle(0.2)
    shot(sweep)

    # ── an example: the Script tab, the network page with a transform, dialogs ──
    def script_tab():
        pick(w, "Forward", "1D Burgers")
        page(w, "results")
        w.bottom_tabs.setCurrentIndex(w._script_tab_index)
        settle(2.0)
        w._refresh_script_tab()
        settle(0.5)
        save(w, "guide_script_tab.png")
        w.bottom_tabs.setCurrentIndex(0)
        settle(0.2)
    shot(script_tab)

    def dialogs():
        pick(w, "Forward", "1D Burgers")
        page(w, "results")
        for name, fn in (("guide_dialog_error_analysis.png", w._on_error_analysis_btn),
                         ("guide_dialog_export_solution.png", w._on_export_settings),
                         ("guide_dialog_plot_settings.png",
                          lambda: w._on_figure_style(context="solve", tab="General")),
                         ("guide_dialog_gpu.png", w._on_gpu_settings),
                         ("guide_dialog_lbfgs.png", w._on_lbfgs_settings),
                         ("guide_dialog_optimizer.png", w._on_optimizer_settings)):
            try:
                grab_dialog(name)
                fn()
                settle(0.6)
            except Exception as e:      # noqa: BLE001
                print("FAILED", name, repr(e), flush=True)
    shot(dialogs)

    # ── a saved run: History and Restore & visualize ──
    def saved_run():
        if not run_dst:
            print("no --run-folder: History / Restore shots skipped", flush=True)
            return
        w.save_dir_input.setText(results)
        settle(0.3)
        pick(w, "Forward", "1D Burgers")
        w.save_dir_input.setText(results)
        page(w, "results")
        w.right_tabs.setCurrentWidget(w.results_tabs)
        w.results_tabs.setCurrentWidget(w.run_history)
        w.run_history.refresh()
        settle(1.0)
        save(w, "guide_results_history.png")
        if w.run_history.runs:
            w._show_saved_run(w.run_history.runs[0])
            settle(1.5)
            save(w, "guide_results_overview_saved.png")
            w.results_tabs.setCurrentWidget(w.ea_panel)
            settle(1.0)
            save(w, "guide_results_error_analysis_saved.png")
        sol = os.path.join(run_dst, "solution_results")
        pts = sorted(f for f in os.listdir(sol) if f.endswith(".pt"))
        if pts:
            w.log_box.clear()
            page(w, "restore")
            w._on_restore_model_chosen(os.path.join(sol, pts[-1]))
            settle(0.8)
            save(w, "guide_restore_model.png")
    shot(saved_run)

    w.close()
    os._exit(0)


if __name__ == "__main__":
    main()
