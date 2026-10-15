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


def _window():
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    _settle()
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
    _settle(0.6)
    assert prev.note.isVisible() and "Re" in prev.note.text()
    w._add_constant_row("Re", "20")
    _settle(0.6)
    assert not prev.note.isVisible(), "a named constant is known"


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
    with open(os.path.join(run, "solution_results", "training_log.txt"), "w") as fh:
        fh.write("Step 0: [np.float32(1.0), np.float32(2.0)]\nStep 100: [np.float32(0.001), np.float32(0.002)]\n")
    with open(os.path.join(run, "error_analysis", "error_metrics.txt"), "w") as fh:
        fh.write("t,L2_relative,Mean_abs_error\n0.0,0.01,0.001\n1.0,0.03,0.002\n")
    os.makedirs(os.path.join(base, "not_a_run"))
    runs = scan_runs(base)
    assert len(runs) == 1
    r = runs[0]
    assert abs(r["loss"] - 0.003) < 1e-9 and abs(r["l2"] - 0.02) < 1e-12 and r["when"] == "2026-10-01 12:30"

    w = _window()
    w.save_dir_input.setText(base)
    w.right_tabs.setCurrentIndex(0)
    w.results_tabs.setCurrentWidget(w.run_history)
    _settle()
    assert w.run_history.table.rowCount() == 1
    w.run_history.table.selectRow(0)
    w.run_history.show_btn.click()
    _settle()
    assert w.results_tabs.currentIndex() == 0
    assert w.ea_panel.table.rowCount() == 2


if __name__ == "__main__":
    test_equation_preview()
    test_start_files_and_unsaved_marker()
    test_script_tab()
    test_tree_menus()
    test_sketch_undo_redo()
    test_run_history()
    print("PASSED")
