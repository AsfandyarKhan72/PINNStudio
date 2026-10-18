"""Boxes are drawn whole and the settings forms stay aligned.

* A box given a fixed height for an older style (24-26 px) is drawn by the
  theme at 28-30 px; its row was laid out for the smaller height, so the
  box's bottom edge was cut off (Add constant, Extra training points'
  Count, the BC rows, the training phases). No box on any page may be cut.
* Rows added after a page is opened (a training phase, a BC, the loss
  weights) get the page's label column, and every row has the same spacing,
  so the controls of a group start at the same place.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QWidget)

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def _pump(sec=0.3):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()


def _cut_boxes(w):
    """Visible boxes whose bottom edge is hidden by the row around them (a
    box cut only by the scrolled page's edge is fine)."""
    bad = []
    for c in w.findChildren(QWidget):
        if not isinstance(c, (QLineEdit, QAbstractSpinBox, QComboBox, QPushButton)) or not c.isVisible():
            continue
        vis, r = c.visibleRegion().boundingRect(), c.rect()
        if vis.isEmpty() or vis.height() >= r.height() or vis.top() != 0:
            continue
        p = c.parentWidget()
        while p is not None and p.objectName() != "qt_scrollarea_viewport":
            p = p.parentWidget()
        if p is not None and c.mapTo(p, r.bottomLeft()).y() >= p.height() - 1:
            continue          # at the bottom of the scrolled page
        text = c.text() if hasattr(c, "text") else (c.currentText() if isinstance(c, QComboBox) else "")
        bad.append(f"{type(c).__name__} {text!r}: {vis.height()} of {r.height()} px shown")
    return bad


def _misaligned(w):
    """Group boxes whose label rows start their controls at different places."""
    def rows(lay, out):
        for i in range(lay.count()):
            it = lay.itemAt(i)
            if it.layout() is not None:
                rows(it.layout(), out)
            elif it.widget() is not None and it.widget().layout() is not None:
                rows(it.widget().layout(), out)
        if isinstance(lay, QHBoxLayout) and lay.count() >= 2:
            first = lay.itemAt(0).widget()
            if isinstance(first, QLabel) and first.isVisible() and not first.wordWrap() \
                    and first.text().rstrip().endswith(":"):
                ctl = next((lay.itemAt(j).widget() for j in range(1, lay.count())
                            if lay.itemAt(j).widget() is not None and lay.itemAt(j).widget().isVisible()), None)
                if ctl is not None:
                    out.append((first, ctl))
    bad = []
    for gb in w.findChildren(QGroupBox):
        if not gb.isVisible():
            continue
        out, by_label_x = [], {}
        rows(gb.layout(), out)
        for lbl, ctl in out:
            lx = lbl.mapTo(gb, lbl.rect().topLeft()).x()
            by_label_x.setdefault(lx, set()).add(ctl.mapTo(gb, ctl.rect().topLeft()).x())
        bad += [f"{gb.title()}: controls at {sorted(xs)}" for xs in by_label_x.values() if len(xs) > 1]
    return bad


def test_boxes_whole_and_forms_aligned(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    for name in ("question", "warning", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w = MainWindow()
    w.resize(1500, 900)
    w.show()
    _pump(0.5)
    w.quick_examples_combo.setCurrentText("Navier-Stokes 2D – Kovasznay flow")
    _pump(0.5)
    cut, off = [], []
    pages = ["equations", "points", "training", "weights", w.custom_bc_list[0]["_node"]]
    for key in pages:
        w.workspace.select(key)
        _pump(0.3)
        if key == "points":
            w.anchors_mode_combo.setCurrentText("Random in a box")     # Count and Ranges
        if key == "training":
            w._add_scheduler_phase()                                   # a phase added by the user
        _pump(0.4)
        cut += [f"{key}: {b}" for b in _cut_boxes(w)]
        off += [f"{key}: {b}" for b in _misaligned(w)]
    # the Inverse page, with a second data file added
    w.quick_examples_category_combo.setCurrentText("Inverse")
    _pump(0.3)
    w.quick_examples_combo.setCurrentText("Allen-Cahn 3D – Wight & Zhao")
    _pump(0.5)
    w.workspace.select("inverse")
    _pump(0.3)
    w._add_inverse_data_row()
    w._add_inverse_var_row()
    _pump(0.5)
    cut += [f"inverse: {b}" for b in _cut_boxes(w)]
    off += [f"inverse: {b}" for b in _misaligned(w)]
    heads = [r["init"].mapTo(w.inverse_group, r["init"].rect().topLeft()).x() for r in w.inv_var_rows]
    assert len(set(heads)) == 1, f"parameter rows' Initial guess columns differ: {heads}"
    assert not cut, "\n".join(cut)
    assert not off, "\n".join(off)
    w.close()
    w.deleteLater()
    _pump(0.2)
