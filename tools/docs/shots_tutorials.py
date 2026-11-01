#!/usr/bin/env python3
"""Screenshots for the step-by-step tutorials (website/tutorials/).

    QT_QPA_PLATFORM=offscreen python tools/docs/shots_tutorials.py website/assets/screenshots \
        [--solve] [--only 01,04] [--model <a model .pt saved by Solve, e.g. of 1D Burgers>]

Opens the real MainWindow (offscreen) and makes the settings of each tutorial through the same widgets
a user edits (New problem from the Start tab, output names, equation boxes, Add constant, Add Boundary
Condition rows, Add Phase, ...), then saves the window as tut<NN>_<what>.png. Dialogs are grabbed while
open. With --solve, tutorial 1 is also solved for a short time (CPU, a few minutes) to capture the
Results tab; tutorial 15 restores the 1D Burgers model given with --model (from a Solve of tutorial 2). Nothing is drawn or edited afterwards.
Modelled on tools/docs/make_screenshots.py.
"""
import os
import tempfile
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
# a neutral home folder, so the status bar shows ~/PINNStudio_Results-style paths, not the build machine's
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _docs_home import private_home  # noqa: E402
private_home()
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from PyQt6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

W, H = 1600, 1000
OUT = None
OBS_FILE = os.path.join(REPO, "website", "assets", "data", "tut07_observations.txt")


def settle(s=0.3):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def new_window():
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(W, H)
    w.show()
    settle(1.0)
    return w


def pick(w, category, key, mode="Default"):
    w.quick_examples_settings_combo.setCurrentText(mode)
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


def page(w, key):
    w.workspace.select(key)
    settle(0.4)


def save(w, name):
    p = os.path.join(OUT, name)
    w.grab().save(p)
    print("saved", p, flush=True)


def grab_dialog(name, delay_ms=900, prepare=None):
    """Grab the next modal dialog that opens, then close it."""
    def _do():
        d = QApplication.activeModalWidget()
        if d is not None:
            if prepare:
                prepare(d)
                QCoreApplication.processEvents()
            d.adjustSize()
            QCoreApplication.processEvents()
            d.grab().save(os.path.join(OUT, name))
            print("saved", name, flush=True)
            d.reject()
    QTimer.singleShot(delay_ms, _do)


def texts(edits, vals):
    for e, v in zip(edits, vals):
        e.setText(v)
        e.editingFinished.emit()
    settle(0.1)


def outputs(w, names):
    w.num_outputs_spin.setValue(len(names))
    settle(0.2)
    texts(w.output_name_inputs, names)


def add_bc(w, type_label, output, where_label, value=None, custom_where=None):
    w.add_custom_bc_btn.click()
    settle(0.3)
    e = w.custom_bc_list[-1]
    e['type'].setCurrentIndex(e['type'].findText(type_label))
    settle(0.1)
    e['component'].setValue(output - 1)
    pk = e['location_picker']
    j = pk.findText(where_label)
    pk.setCurrentIndex(j)
    pk.activated.emit(j)
    if custom_where is not None:
        e['location'].setText(custom_where)
        e['location'].editingFinished.emit()
    if value is not None:
        e['value'].setText(value)
        e['value'].editingFinished.emit()
    settle(0.2)
    return e


def bc_page(w, n):
    keys = [k for k in w.workspace.order if str(k).startswith("bc:")]
    page(w, keys[n - 1])


def set_phase(w, k, opt=None, iters=None, lr=None):
    while len(w.sched_phase_list) < k:
        w._add_scheduler_phase()
        settle(0.1)
    ph = w.sched_phase_list[k - 1]
    if opt:
        ph['opt'].setCurrentIndex(ph['opt'].findData(opt))
        settle(0.1)
    if iters is not None:
        ph['iters'].setValue(iters)
    if lr is not None:
        ph['lr'].setValue(lr)
    settle(0.1)


def ea_formula_dialog(w, name, formulas, times=""):
    """Open Results > Error Analysis, choose Exact solution formula, fill it in, grab."""
    def prep(d):
        w._ea_ref_mode_combo.setCurrentIndex(w._ea_ref_mode_combo.findData("formula"))
        for e, f in zip(w._ea_formula_edits, formulas):
            e.setText(f)
        w._ea_ref_times_edit.setText(times)
    grab_dialog(name, prepare=prep)
    w._on_error_analysis_btn()
    settle(1.5)


def wait_thread(w, attr="thread", limit=3600):
    t0 = time.time()
    while time.time() - t0 < limit:
        settle(1.0)
        th = getattr(w, attr, None)
        if th is not None and not th.isRunning() and time.time() - t0 > 5:
            break
    settle(3.0)


# ── tutorial set-ups (the same steps the pages describe) ──────────────

def wide(w):
    """Drag the splitter so the settings column shows wide pages without a scroll bar (as a user would)."""
    w.main_splitter.setSizes([225, 800, 575])
    settle(0.3)


def ode_problem(w):
    w._new_problem("ODE")
    settle(0.4)
    w.t_min.setValue(0.0)
    w.t_max.setValue(5.0)
    outputs(w, ["y1", "y2"])
    w._add_constant_row("zeta", "0.1")
    w._add_constant_row("om", "2")
    settle(0.2)
    texts(w.pde_inputs, ["dy1_t - y2", "dy2_t + 2*zeta*om*y2 + om**2*y1"])
    texts(w.ic_inputs, ["1", "0"])


ODE_EXACT = ["exp(-zeta*om*t)*(cos(sqrt(om**2*(1 - zeta**2))*t) + zeta*om/sqrt(om**2*(1 - zeta**2))"
             "*sin(sqrt(om**2*(1 - zeta**2))*t))",
             "-om**2/sqrt(om**2*(1 - zeta**2))*exp(-zeta*om*t)*sin(sqrt(om**2*(1 - zeta**2))*t)"]


def tut01(w, solve):
    ode_problem(w)
    for key, name in (("problem", "problem"), ("equations", "equations"), ("geometry", "geometry"),
                      ("ic", "ic"), ("training", "training")):
        page(w, key)
        save(w, f"tut01_{name}.png")
    page(w, "results")
    ea_formula_dialog(w, "tut01_ea_dialog.png", ODE_EXACT)
    if solve:
        # a short run for the screenshot (the page recommends the default phases)
        set_phase(w, 1, iters=3000)
        set_phase(w, 2, iters=1000)
        w._ea_settings = {'files': [], 'ref_mode': 'formula', 'ref_formula': "|".join(ODE_EXACT), 'ref_times': '',
                          'ref_grid': 0, 'do_line': True, 'do_surface': True, 'do_l2': True, 'do_mse': False,
                          'do_max': True}
        w.save_dir_input.setText(os.path.join(os.environ["HOME"], "PINNStudio_Results"))
        w._on_solve()
        wait_thread(w)
        w.results_tabs.setCurrentIndex(0)
        settle(1.0)
        save(w, "tut01_results.png")
        for i in range(w.results_tabs.count()):
            if "rror" in w.results_tabs.tabText(i):
                w.results_tabs.setCurrentIndex(i)
                settle(1.0)
                save(w, "tut01_results_error.png")
        w.bottom_tabs.setCurrentIndex(0)


def tut02(w):
    pick(w, "Forward", "1D Burgers")
    page(w, "training")
    save(w, "tut02_training.png")
    page(w, "results")
    grab_dialog("tut02_ea_dialog.png")
    w._on_error_analysis_btn()
    settle(1.5)


def tut03(w):
    pick(w, "Forward", "2D Heat")
    page(w, "bcs")
    save(w, "tut03_bcs.png")


def tut04(w):
    w._new_problem("1D")
    settle(0.4)
    outputs(w, ["u", "v"])
    w._add_constant_row("D", "0.1")
    settle(0.2)
    texts(w.pde_inputs, ["du_t - D*du_xx + u - v", "dv_t - D*dv_xx - u + v"])
    texts(w.ic_inputs, ["sin(pi*x)", "0"])
    add_bc(w, "Dirichlet BC", 1, "Whole boundary", "0")
    add_bc(w, "Dirichlet BC", 2, "Whole boundary", "0")
    page(w, "equations")
    save(w, "tut04_equations.png")
    page(w, "ic")
    save(w, "tut04_ic.png")
    bc_page(w, 2)
    save(w, "tut04_bc2.png")
    page(w, "weights")
    save(w, "tut04_weights.png")


def tut05(w):
    pick(w, "Forward", "1D Allen-Cahn (whole interval)")
    for key in ("equations", "network"):
        page(w, key)
        save(w, f"tut05_{key}.png")
    wide(w)
    page(w, "adaptive")
    save(w, "tut05_adaptive.png")
    w.main_splitter.setSizes([265, 660, 920])
    settle(0.3)
    bc_page(w, 1)
    save(w, "tut05_bc1.png")


def tut06(w):
    pick(w, "Inverse", "Heat equation")
    page(w, "inverse")
    save(w, "tut06_inverse.png")
    page(w, "problem")
    save(w, "tut06_problem.png")


def tut07(w):
    wide(w)
    w._new_problem("1D")
    settle(0.4)
    w.radio_inverse.setChecked(True)
    settle(0.3)
    outputs(w, ["u"])
    texts(w.pde_inputs, ["du_t - a*du_xx + b*u"])
    texts(w.ic_inputs, ["sin(pi*x) + 0.5*sin(3*pi*x)"])
    add_bc(w, "Dirichlet BC", 1, "Whole boundary", "0")
    r0 = w.inv_var_rows[0]
    r0['name'].setText("a")
    r0['init'].setValue(0.2)
    w._add_inverse_var_row()
    settle(0.2)
    r1 = w.inv_var_rows[1]
    r1['name'].setText("b")
    r1['init'].setValue(1.5)
    r0['true'].setValue(0.05)
    r1['true'].setValue(0.5)
    # the file as a user would have it after downloading it from the tutorial page
    import shutil
    dl = os.path.join(os.environ["HOME"], "Downloads")
    os.makedirs(dl, exist_ok=True)
    obs = shutil.copy(OBS_FILE, dl)
    w.inv_data_rows[0]['path'].setText(obs)
    w.inv_data_rows[0]['path'].editingFinished.emit()
    settle(1.5)
    page(w, "inverse")
    save(w, "tut07_inverse.png")
    page(w, "equations")
    save(w, "tut07_equations.png")


def tut08(w):
    w._new_problem("2D")
    settle(0.4)
    w.radio_stationary.setChecked(True)
    settle(0.2)
    w.geometry_type_combo.setCurrentText("Polygon")
    settle(0.3)
    w.geom_polygon_verts_input.setText("0,0; 2,0; 2,1; 1,1.5; 0,1")
    w.geom_polygon_verts_input.editingFinished.emit()
    settle(0.3)
    outputs(w, ["u"])
    texts(w.pde_inputs, ["du_xx + du_yy"])
    add_bc(w, "Neumann BC", 1, "Bottom side (y = y_min)", "-exp(x)")
    add_bc(w, "Dirichlet BC", 1, "Custom expression…", "exp(x)*sin(y)", custom_where="y > 1e-8")
    page(w, "geometry")
    w.right_tabs.setCurrentIndex(w._domain_tab_index)
    settle(1.5)
    save(w, "tut08_geometry.png")
    bc_page(w, 2)
    settle(1.0)
    save(w, "tut08_bc2.png")


def tut09(w):
    pick(w, "Forward", "Diffusion equation with hard IC and BC")
    page(w, "network")
    save(w, "tut09_network.png")
    page(w, "bcs")
    save(w, "tut09_bcs.png")
    w._new_problem("1D")
    settle(0.4)
    texts(w.pde_inputs, ["du_t - 0.1*du_xx"])
    texts(w.ic_inputs, ["sin(pi*x)"])
    w.ic_hard[0].setChecked(True)
    settle(0.2)
    page(w, "ic")
    save(w, "tut09_ic_exact.png")


def tut10(w):
    pick(w, "Forward", "Burgers equation with RAR")
    wide(w)
    page(w, "adaptive")
    save(w, "tut10_adaptive.png")


def tut11(w):
    pick(w, "Forward", "1D Burgers")
    set_phase(w, 1, opt="adam", iters=10000, lr=1e-3)
    set_phase(w, 2, opt="lbfgs", iters=10000)
    page(w, "training")
    save(w, "tut11_training.png")
    w.point_weights_combo.setCurrentIndex(1)
    settle(0.3)
    page(w, "weights")
    save(w, "tut11_rba.png")
    w.point_weights_combo.setCurrentIndex(2)
    settle(0.3)
    save(w, "tut11_sa.png")


def tut12(w):
    pick(w, "Forward", "1D Allen-Cahn (whole interval)")
    w.adapt_combo.setCurrentIndex(w.adapt_combo.findText("None"))
    settle(0.2)
    w.causal_cb.setChecked(True)
    settle(0.3)
    page(w, "weights")
    save(w, "tut12_causal.png")
    w.causal_cb.setChecked(False)
    w.adapt_combo.setCurrentIndex(w.adapt_combo.findText("Time Adaptive Training"))
    settle(0.3)
    w.ta_group_rows[0]["steps"].setValue(4)
    w.ta_transfer_cb.setChecked(True)
    settle(0.3)
    wide(w)
    page(w, "adaptive")
    save(w, "tut12_windows.png")


def ode_ea(w):
    """The Tutorial 1 Error Analysis settings (Exact solution formula), as the dialog stores them."""
    w._ea_settings = {'files': [], 'ref_mode': 'formula', 'ref_formula': "|".join(ODE_EXACT), 'ref_times': '',
                      'ref_grid': 0, 'do_line': True, 'do_surface': True, 'do_l2': True, 'do_mse': False,
                      'do_max': False}


def tut13(w):
    ode_problem(w)
    ode_ea(w)
    wide(w)
    w.sweep_enable_cb.setChecked(True)
    settle(0.3)
    row = w.sweep_row_list[0]
    row['param_combo'].setCurrentIndex(row['param_combo'].findText("Network: Neurons per layer"))
    settle(0.2)
    row['list_edit'].setText("16, 64")
    settle(1.5)
    page(w, "sweep")
    save(w, "tut13_sweep.png")
    w.quick_examples_settings_combo.setCurrentText("Tuned")
    w.quick_examples_category_combo.setCurrentText("Forward")
    settle(0.2)
    i = w.quick_examples_combo.findData("Heat equation")
    w.quick_examples_combo.setCurrentIndex(i)
    w.quick_examples_combo.activated.emit(i)
    settle(1.0)          # the log keeps what the Tuned settings change
    page(w, "training")
    w.bottom_tabs.setCurrentIndex(0)
    settle(0.3)
    save(w, "tut13_tuned.png")
    w.quick_examples_settings_combo.setCurrentText("Default")
    settle(0.5)


def tut14(w):
    ode_problem(w)
    ode_ea(w)
    wide(w)
    w.sweep_enable_cb.setChecked(True)
    settle(0.3)
    w.sweep_mode_combo.setCurrentIndex(w.sweep_mode_combo.findText("Bayesian search"))
    settle(0.3)
    w.sweep_bayes_calls_spin.setValue(5)
    w.sweep_bayes_init_spin.setValue(2)
    w.sweep_bayes_obj_combo.setCurrentIndex(w.sweep_bayes_obj_combo.findText("L2 error"))
    row = w.sweep_row_list[0]
    row['param_combo'].setCurrentIndex(row['param_combo'].findText("Phase 1: Learning rate"))
    settle(0.2)
    row['mode_combo'].setCurrentIndex(row['mode_combo'].findText("Log"))
    settle(0.2)
    row['min_spin'].setValue(1e-4)
    row['max_spin'].setValue(1e-2)
    settle(1.5)
    page(w, "sweep")
    save(w, "tut14_bayes.png")


def tut15(w, model=None):
    # the Burgers example of tutorial 2 provides the Error Analysis reference files for the restore
    pick(w, "Forward", "1D Burgers")
    page(w, "restore")
    if model and os.path.isfile(model):
        w._on_restore_model_chosen(model)
        settle(0.5)
        w.restore_viz_combo.setCurrentText("Line (time steps)")
        settle(0.3)
        save(w, "tut15_restore.png")
        w._on_restore()
        t0 = time.time()
        while time.time() - t0 < 600:
            settle(1.0)
            if w.restore_btn.isEnabled() and time.time() - t0 > 5:
                break
        settle(3.0)
        w.right_tabs.setCurrentWidget(w.results_tabs)
        settle(1.0)
        save(w, "tut15_restored.png")
    else:
        print("tutorial 15 needs --model <path to a model .pt saved by Solve>", flush=True)


def tut16(w):
    ode_problem(w)
    page(w, "results")
    for i in range(w.bottom_tabs.count()):
        if "Script" in w.bottom_tabs.tabText(i):
            w.bottom_tabs.setCurrentIndex(i)
    settle(1.5)
    save(w, "tut16_script_tab.png")
    w.bottom_tabs.setCurrentIndex(0)


def main():
    global OUT
    OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
                          else "website/assets/screenshots")
    os.makedirs(OUT, exist_ok=True)
    solve = "--solve" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    steps = [("01", lambda w: tut01(w, solve)), ("02", tut02), ("03", tut03), ("04", tut04), ("05", tut05),
             ("06", tut06), ("07", tut07), ("08", tut08), ("09", tut09), ("10", tut10), ("11", tut11),
             ("12", tut12), ("13", tut13), ("14", tut14), ("16", tut16)]
    for num, fn in steps:
        if only and num not in only:
            continue
        w = new_window()
        fn(w)
        w.close()
        settle(0.3)
    if not only or "15" in only:
        w = new_window()
        model = None
        if "--model" in sys.argv:
            model = sys.argv[sys.argv.index("--model") + 1]
        tut15(w, model)
        w.close()
    os._exit(0)


if __name__ == "__main__":
    main()
