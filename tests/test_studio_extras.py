#!/usr/bin/env python3
"""Start tab, Script tab, Save / Open / recent files and the unsaved-changes
marker, the model tree's right-click menus, the typeset equations, sketch
undo/redo and Results -> History.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_studio_extras.py
"""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.ui.equation import to_latex, unknown_names, render
from pinnstudio.ui.results import scan_runs

app = QApplication.instance() or QApplication(sys.argv)


def _settle(secs=0.3):
    t = time.time()
    while time.time() - t < secs:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _until(cond, secs=10):
    """Wait for the preview's delayed update (slow CI runners take longer)."""
    t = time.time()
    while not cond() and time.time() - t < secs:
        _settle(0.1)
    return cond()


def _window():
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    _settle()
    # the starting problem ("unchanged") is recorded a few event-loop passes
    # after the window opens; on a slow machine that can take longer than
    # the settle above, and a change made before it would count as the start
    t = time.time()
    while getattr(w, "_clean_key", None) is None and time.time() - t < 10:
        _settle(0.05)
    return w


def test_equation_preview():
    assert to_latex("du_t - 0.4*du_xx", ["u"]) == \
        r"\frac{\partial u}{\partial t} - 0.4 \cdot \frac{\partial^{2} u}{\partial x^{2}}" or \
        r"\frac{\partial^{2} u}{\partial x^{2}}" in to_latex("du_t - 0.4*du_xx", ["u"])
    assert r"\sin" in to_latex("sin(pi*x)", ["u"]) and r"\pi" in to_latex("sin(pi*x)", ["u"])
    assert r"\frac{\partial^{4} u}{\partial x^{4}}" in to_latex("du_xxxx", ["u"])
    assert render(to_latex("du_t + u*du_x", ["u"])) is not None
    known = {"outputs": {"u"}, "other": {"nu"}, "coords": "xt"}
    assert unknown_names("du_t + u*du_x - nu*du_xx + sin(pi*x)", known) == []
    assert unknown_names("du_t - Re*du_xx", known) == ["Re"]
    assert unknown_names("du_t - dv_xx", known) == ["dv_xx"], "not an output's derivative"
    assert unknown_names("du_t - (du_xx", known) is None

    w = _window()
    w.workspace.select("equations")
    _settle()
    prev = w.pde_inputs[0]._preview
    assert prev.math.pixmap() is not None and not prev.math.pixmap().isNull()
    assert not prev.note.isVisible()
    w.pde_inputs[0].setText("du_t - Re*du_xx")
    assert _until(lambda: prev.note.isVisible() and "Re" in prev.note.text())
    w._add_constant_row("Re", "20")
    assert _until(lambda: not prev.note.isVisible()), "a named constant is known"


def test_start_files_and_unsaved_marker():
    w = _window()
    assert w.right_tabs.currentIndex() == w._start_tab_index, "opens on the Start tab"
    assert not w.isWindowModified()
    w.x_max.setValue(2.0)
    w._check_problems()
    assert w.isWindowModified(), "a changed setting marks the window"
    d = tempfile.mkdtemp()
    path = os.path.join(d, "mine.pinn.json")
    assert w._save_problem(path)
    assert not w.isWindowModified() and "mine" in w.windowTitle()
    assert w._recent_files()[0] == os.path.abspath(path)
    assert len(w.start_page.recent_buttons) >= 1 and w.recent_menu.isEnabled()
    # Ctrl+S saves to the same file, no dialog
    w.x_max.setValue(3.0)
    w._check_problems()
    assert w.isWindowModified()
    assert w._save_problem()
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["config"]["x_max"] == 3.0
    # New 2D problem from the Start tab
    w.start_page.new_cards["2D"].click()
    _settle()
    assert w.radio_2d.isChecked() and w.x_max.value() == 1.0 and not w.isWindowModified()
    assert w.workspace.current_key() == "equations"
    # open the saved one again from the recent list
    w.start_page.recent_buttons[0].findChildren(type(w.solve_btn))[0].click()
    _settle()
    assert w.radio_1d.isChecked() and w.x_max.value() == 3.0 and not w.isWindowModified()
    # an example from the gallery
    key = next(iter(w.start_page.example_cards))
    w.start_page.example_cards[key].click()
    _settle()
    assert w.quick_examples_combo.currentText() == key and not w.isWindowModified()
    # shortcuts are on the actions
    assert w.solve_action.shortcuts() and w.stop_action.shortcut().toString() == "Shift+F5"


def test_script_tab():
    w = _window()
    w.bottom_tabs.setCurrentIndex(w._script_tab_index)
    _settle()
    text = w.script_view.toPlainText()
    assert "import deepxde" in text or "deepxde" in text
    w.x_max.setValue(2.5)
    w._check_problems()
    assert "2.5" in w.script_view.toPlainText(), "the script follows the settings"


def test_tree_menus():
    w = _window()
    ws = w.workspace

    def act(key, text):
        m = ws.build_menu(key)
        return [a for a in m.actions() if a.text() == text][0]

    n = len(w.custom_bc_list)
    act("bcs", "Add boundary condition").trigger()
    _settle()
    e = w.custom_bc_list[-1]
    assert len(w.custom_bc_list) == n + 1 and ws.current_key() == e['_node']
    e['location'].setText("right")
    act(e['_node'], "Duplicate").trigger()
    _settle()
    d = w.custom_bc_list[-1]
    assert len(w.custom_bc_list) == n + 2 and d['location'].text() == "right"
    act(d['_node'], "Delete").trigger()
    _settle()
    assert len(w.custom_bc_list) == n + 1
    w.workspace.select("training")
    _settle()
    act("training", "Fold all sections").trigger()
    folded = [b._collapser.folded for b in ws.pages["training"]["widgets"] if getattr(b, "_collapser", None)
              and not b.isHidden()]
    assert folded and all(folded)
    act("training", "Unfold all sections").trigger()
    assert "New 2D problem" in [a.text() for a in ws.build_menu("problem").actions()]


def test_sketch_undo_redo():
    w = _window()
    w.radio_2d.setChecked(True)
    _settle()
    w.geometry_type_combo.setCurrentText("Sketch")
    _settle()
    sp = w.sketch_panel
    r = {"type": "Rectangle", "params": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1}}
    c = {"type": "Disk", "params": {"cx": 1, "cy": 0.5, "r": 0.2}, "op": "difference"}
    sp.canvas.shapes = [r]
    sp.canvas.changed.emit()
    sp.canvas.shapes = [r, c]
    sp.canvas.changed.emit()
    _settle()
    assert len(w._custom_geom_shapes_list()) == 2
    QTest.keyClick(sp.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    _settle()
    assert len(sp.canvas.shapes) == 1 and len(w._custom_geom_shapes_list()) == 1, "undo reaches the settings"
    sp.redo()
    _settle()
    assert len(sp.canvas.shapes) == 2 and len(w._custom_geom_shapes_list()) == 2
    sp.clear()
    sp.undo()
    assert len(sp.canvas.shapes) == 2


def test_run_history():
    base = tempfile.mkdtemp()
    run = os.path.join(base, "1D_Forward__2026-10-01_12-30-00")
    os.makedirs(os.path.join(run, "solution_results"))
    os.makedirs(os.path.join(run, "error_analysis"))
    with open(os.path.join(run, "solution_results", "training_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("Step 0: [np.float32(1.0), np.float32(2.0)]\nStep 100: [np.float32(0.001), np.float32(0.002)]\n")
    with open(os.path.join(run, "error_analysis", "error_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write("t,L2_relative,Mean_abs_error\n0.0,0.01,0.001\n1.0,0.03,0.002\n")
    os.makedirs(os.path.join(base, "not_a_run"))
    runs = scan_runs(base)
    assert len(runs) == 1
    r = runs[0]
    assert abs(r["loss"] - 0.003) < 1e-9 and abs(r["l2"] - 0.02) < 1e-12 and r["when"] == "2026-10-01 12:30"

    w = _window()
    w.save_dir_input.setText(base)
    w.right_tabs.setCurrentWidget(w.results_tabs)
    w.results_tabs.setCurrentWidget(w.run_history)
    _settle()
    assert w.run_history.table.rowCount() == 1
    w.run_history.table.selectRow(0)
    w.run_history.show_btn.click()
    _settle()
    assert w.results_tabs.currentIndex() == 0
    assert w.ea_panel.table.rowCount() == 2



def _fake_run(base, name, losses, l2s):
    run = os.path.join(base, name)
    os.makedirs(os.path.join(run, "solution_results"))
    os.makedirs(os.path.join(run, "error_analysis"))
    with open(os.path.join(run, "solution_results", "training_log.txt"), "w", encoding="utf-8") as fh:
        for i, v in enumerate(losses):
            fh.write(f"Step {i * 100}: [np.float32({v}), np.float32({v / 10})]\n")
    with open(os.path.join(run, "error_analysis", "error_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write("t,L2_relative,Mean_abs_error\n")
        for i, e in enumerate(l2s):
            fh.write(f"{i * 0.5},{e},0.001\n")
    return run


def test_compare_runs():
    from pinnstudio.ui.results import read_loss_history
    base = tempfile.mkdtemp()
    a = _fake_run(base, "run_00__baseline", [1.0, 0.1, 0.01], [0.01, 0.02])
    _fake_run(base, "run_01__Hidden-layers_4", [2.0, 0.2, 0.05], [0.03, 0.04])
    hist = read_loss_history(a)
    assert [h[0] for h in hist] == [0, 100, 200] and abs(hist[-1][1] - 0.011) < 1e-9
    w = _window()
    w.save_dir_input.setText(base)
    w.right_tabs.setCurrentWidget(w.results_tabs)
    w.results_tabs.setCurrentWidget(w.run_history)
    _settle()
    h = w.run_history
    assert h.table.rowCount() == 2
    h.table.selectRow(0)
    _settle()
    assert not h.compare_btn.isEnabled() and h.show_btn.isEnabled()
    h.table.selectAll()
    _settle()
    assert h.compare_btn.isEnabled() and not h.show_btn.isEnabled()
    h.compare_btn.click()
    _settle()
    assert h.compare.isVisible() and len(h.compare.runs) == 2
    axes = h.compare._fig.axes
    assert len(axes) == 2, "loss and error side by side"
    assert len(axes[0].lines) == 2 and len(axes[1].lines) == 2
    labels = {t.get_text() for t in axes[1].get_legend().get_texts()}
    assert labels == {"baseline", "Hidden-layers 4"}, labels


def test_help_links_and_start_first():
    w = _window()
    assert w.right_tabs.indexOf(w.start_page) == 0, "Start is the first tab"
    opened = []
    w.workspace.open_url = opened.append
    w.workspace.select("weights")
    _settle()
    assert w.workspace.help_btn.isVisible()
    w.workspace.help_btn.click()
    assert opened == [w.DOCS_URL + "#training"], "a child node uses its parent's section"
    assert "Help" in [a.text() for a in w.workspace.build_menu("sweep").actions()]
    # opening a saved problem keeps Save-to at the folder, not the old run's folder
    assert w._save_root("/r/1D_Forward__2026-10-14_22-02-08") == "/r"
    assert w._save_root("/r/mine") == "/r/mine"



def test_buttons_fit_and_restore_visible():
    from PyQt6.QtWidgets import QPushButton
    from pinnstudio.ui import theme
    w = _window()
    for key in ("restore", "ic", "results"):
        w.workspace.select(key)
        _settle()
    for b in w.findChildren(QPushButton):
        if b.text().strip() and b.minimumWidth() == b.maximumWidth() and b.maximumWidth() > 48:
            need = b.fontMetrics().horizontalAdvance(b.text().replace("&&", "&")) + 24
            assert b.width() >= need, (b.text(), b.width(), need)
    assert w.restore_btn.objectName() == "primaryButton"
    # white text on a background that becomes an ordinary light button turns dark
    css = theme.remap_css("QPushButton { background: #1a5c3a; color: white; }", theme.tokens("Studio Light"))
    assert "white" not in css and theme.tokens("Studio Light")["text"] in css


def test_live_loss_per_time_step_and_ea_cleared():
    from types import SimpleNamespace
    from pinnstudio.ui.results import RunProgress
    p = RunProgress(SimpleNamespace(iterations=1000, optimizer="adam", optimizer2="none", iterations2=0))
    for line in ("--- Time step 1/4: t = 0.0000 to 0.2500 ---", "=== Scheduler Phase 1: adam 1000 iters ===",
                 "0  [1.0, 2.0]  [1.0, 2.0]  []", "1000  [0.1, 0.2]  [0.1, 0.2]  []",
                 "--- Time step 2/4: t = 0.2500 to 0.5000 ---", "0  [3.0, 1.0]  [3.0, 1.0]  []"):
        p.feed(line)
    assert p.time_step == 2 and p.n_time_steps == 4
    assert [h[0] for h in p.history] == [0], "the new step's loss starts over"
    assert abs(p.fraction() - 0.25) < 1e-9
    w = _window()
    w.ea_panel.show_run([(0.0, 0.01, 0.001)], "")
    assert w.ea_panel.table.rowCount() == 1
    from pinnstudio.core.config import PINNConfig
    w._start_progress(PINNConfig())
    assert w.ea_panel.table.rowCount() == 0, "a new run clears the old error analysis"
    for line in ("--- Time step 3/4: t = 0.5 to 0.75 ---", "0  [1.0]  [1.0]  []"):
        w._progress_feed(line)
    w.live_loss._redraw()
    assert "time step 3 of 4" in w.live_loss.title.text()
    w._finish_progress("Stopped", ok=False)


def test_copy_and_save():
    from PyQt6.QtGui import QGuiApplication, QImage
    from pinnstudio.ui import export_tools as ex
    d = tempfile.mkdtemp()
    png = os.path.join(d, "fig.png")
    img = QImage(40, 30, QImage.Format.Format_RGB32)
    img.fill(0x336699)
    img.save(png)
    assert ex.copy_image_file(png) and not QGuiApplication.clipboard().image().isNull()
    assert ex.save_image_file(png, target=os.path.join(d, "copy")) == os.path.join(d, "copy.png")
    assert ex.save_image_file(png, target=os.path.join(d, "c.jpg")) and os.path.getsize(os.path.join(d, "c.jpg"))
    menu = ex.image_menu(None, png, lambda: None)
    acts = [a.text() for a in menu.actions()]
    assert acts == ["Copy image", "Save image as…", "Open in viewer", "Show in folder"]
    w = _window()
    w.ea_panel.show_run([(0.0, 0.0123456789, 0.001), (0.5, 0.02, None)], "")
    text = ex.copy_table(w.ea_panel.table)
    assert text.splitlines()[0].startswith("time\t") and "0.0123456789" in text, "full precision"
    csv_path = ex.export_table_csv(w.ea_panel.table, target=os.path.join(d, "err"))
    with open(csv_path, encoding="utf-8", errors="replace") as fh:
        rows = fh.read().splitlines()
    assert len(rows) == 3 and rows[2].endswith(","), "an empty cell stays empty"
    from matplotlib.figure import Figure
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    assert ex.copy_figure(fig)
    for ext in ("png", "pdf", "svg"):
        out = ex.save_figure(fig, target=os.path.join(d, f"f.{ext}"))
        assert os.path.getsize(out) > 0



def test_fresh_window_and_new_problem_are_clean_and_valid():
    w = _window()
    _settle(0.5)
    w._check_problems()
    assert not w.isWindowModified(), "an untouched window has nothing to save"
    for dim in ("1D", "2D", "3D", "ODE"):
        w._new_problem(dim)
        _settle(0.3)
        assert w._build_config().validate() == [], dim
        w._check_problems()
        assert not w.isWindowModified(), dim


def test_error_analysis_shows_only_this_runs_folder():
    from pinnstudio.core.config import PINNConfig
    w = _window()
    old = tempfile.mkdtemp()
    os.makedirs(os.path.join(old, "error_analysis"))
    from PyQt6.QtGui import QImage
    img = QImage(10, 10, QImage.Format.Format_RGB32)
    img.save(os.path.join(old, "error_analysis", "line_comparison.png"))
    w._last_config = PINNConfig(save_dir=old)          # a previous run's
    new = tempfile.mkdtemp()                            # this run: no error analysis
    w._start_progress(PINNConfig(save_dir=new))
    w._finish_progress("Finished", ok=True, folder=new)
    assert w.ea_panel.figs_lay.count() <= 1, "no figures from the previous run"
    # Stop, then the run's own "done": the status keeps saying Stopped
    w._start_progress(PINNConfig())
    w._stopped = True
    w._finish_progress("Stopped", ok=False)
    assert w.status_text.text().startswith("Stopped") and w._progress is None


if __name__ == "__main__":
    test_equation_preview()
    test_start_files_and_unsaved_marker()
    test_script_tab()
    test_tree_menus()
    test_sketch_undo_redo()
    test_run_history()
    test_compare_runs()
    test_help_links_and_start_first()
    test_buttons_fit_and_restore_visible()
    test_live_loss_per_time_step_and_ea_cleared()
    test_copy_and_save()
    test_fresh_window_and_new_problem_are_clean_and_valid()
    test_error_analysis_shows_only_this_runs_folder()
    print("PASSED")
