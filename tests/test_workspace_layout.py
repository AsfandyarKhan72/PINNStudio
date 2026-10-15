#!/usr/bin/env python3
"""The three-panel workspace: model tree | settings | graphics + log.

Checks that every panel landed on its tree node's page, that nodes follow
their panels' visibility (Inverse problem only in Inverse mode, Initial
conditions not for a stationary problem), that each BC row is its own node
(added, renumbered and removed with the rows), that the Solve/Stop buttons
and Quick Examples are in the toolbar, that settings which would stop Solve
mark their node, and that with a sketched domain a BC node edits its edges
by clicking them.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_workspace_layout.py
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.ui.sketch_edges import split_where

app = QApplication.instance() or QApplication(sys.argv)


def _settle(secs=0.3):
    t = time.time()
    while time.time() - t < secs:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _bc_keys(w):
    return [k for k in w.workspace.order if k.startswith("bc:")]


def test_workspace_layout():
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    _settle()
    ws = w.workspace

    # panels on their pages
    for key, widget in (("equations", w.pde_group), ("geometry", w.domain_group), ("bcs", w.custom_bc_group),
                        ("weights", w.weights_group), ("adaptive", w.adapt_group), ("results", w.results_controls),
                        ("geometry", w.geom_shape_box), ("training", w.train_callbacks_box)):
        assert ws.key_of(widget) == key, (key, ws.key_of(widget))
    # toolbar
    for b in (w.solve_btn, w.stop_btn, w.quick_examples_combo):
        assert b.parentWidget() is w.app_toolbar
    # tree follows panel visibility
    assert ws.pages["inverse"]["item"].isHidden()
    w.radio_inverse.setChecked(True)
    _settle()
    assert not ws.pages["inverse"]["item"].isHidden()
    w.radio_forward.setChecked(True)
    w.radio_stationary.setChecked(True)
    _settle()
    assert ws.pages["inverse"]["item"].isHidden() and ws.pages["geometry"]["item"].isHidden() is False
    assert w.geom_time_box.isHidden(), "no time interval for a stationary problem"
    w.radio_time_dependent.setChecked(True)
    _settle()

    # BC rows are nodes
    n0 = len(w.custom_bc_list)
    assert len(_bc_keys(w)) == n0
    w.add_custom_bc_btn.click()
    w.add_custom_bc_btn.click()
    _settle()
    keys = _bc_keys(w)
    assert len(keys) == n0 + 2
    assert ws.key_of(w.custom_bc_list[-1]['widget']) == keys[-1]
    labels = [ws.pages[k]["label"] for k in keys]
    assert labels[-1].startswith(f"BC {n0 + 2}"), labels
    # remove the first new one: the other is renumbered
    first_new = w.custom_bc_list[n0]
    rm = [b for b in first_new['widget'].findChildren(type(w.solve_btn)) if b.text() == "✕"][0]
    rm.click()
    _settle()
    keys = _bc_keys(w)
    assert len(keys) == n0 + 1 and ws.pages[keys[-1]]["label"].startswith(f"BC {n0 + 1}")
    # a changed type renames the node
    w.custom_bc_list[-1]['type'].setCurrentIndex(1)
    _settle()
    assert "Neumann" in ws.pages[keys[-1]]["label"]

    # settings that would stop Solve mark their node
    w.x_max.setValue(-1.0)
    w._check_problems()
    assert ws.problems.get("geometry") and w.problems_btn.isVisible()
    w._goto_first_problem()
    assert ws.current_key() == "geometry" and w.workspace.banner.isVisible()
    w.x_max.setValue(1.0)
    w._check_problems()
    assert not ws.problems and not w.problems_btn.isVisible()


def test_bc_node_edits_sketch_edges():
    w = MainWindow()
    w.resize(1700, 1000)
    w.show()
    _settle()
    w.radio_2d.setChecked(True)
    _settle()
    w.geometry_type_combo.setCurrentText("Sketch")
    _settle()
    shapes = [{"type": "Rectangle", "params": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1}}]
    w._on_sketch_edited(shapes)
    sp, c = w.sketch_panel, w.sketch_panel.canvas
    sp.set_shapes(shapes, fit=True)
    w.workspace.select("bcs")
    _settle()
    assert sp.canvas.tool == "Edges" and w.right_tabs.currentIndex() == w._sketch_tab_index
    assert w.sketch_selection_box.isVisible(), "the sketch's selection controls are on the BC page"
    E = c.edges()

    def click(k):
        QTest.mouseClick(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         c.to_screen(QPointF(*E[k]["mid"])).toPoint())
        _settle(0.1)

    left = next(i for i, e in enumerate(E) if e["desc"].startswith("x = 0"))
    right = next(i for i, e in enumerate(E) if e["desc"].startswith("x = 2"))
    click(left)
    sp.new_bc_btn.click()
    _settle()
    entry = w.custom_bc_list[-1]
    assert w.workspace.current_key() == entry['_node'], "the new BC's node is selected"
    assert sp.active_bc == len(w.custom_bc_list) - 1
    click(right)      # add
    assert split_where(entry['location'].text()) == [E[left]["expr"], E[right]["expr"]]
    assert "E" in entry['sketch_lbl'].text() and entry['sketch_box'].isVisible()
    click(left)       # take out
    assert entry['location'].text() == E[right]["expr"]
    click(right)      # the last edge stays
    assert entry['location'].text() == E[right]["expr"]
    w.workspace.select("problem")
    _settle()
    assert sp.active_bc is None


if __name__ == "__main__":
    test_workspace_layout()
    test_bc_node_edits_sketch_edges()
    print("PASSED")
