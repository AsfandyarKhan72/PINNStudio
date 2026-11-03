"""Every settings row reads as one form: its control starts right after the
label column, every row label ends in a colon and fits the column. A
fixed-width field with no stretch after it left the spare room around the
label (Min. delta, the loss-weight rows), and two RAR labels were too long
or had no colon, so their fields sat apart from the rest."""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])
HEADERS = ("──", "Automatic weighting")      # section titles, not form rows


def _pump(sec=0.3):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()


def _slots(lay):
    """{index: widget} and {index: nested layout} of one layout, through the QObject tree (QLayout.itemAt
    would leave PySide6 item wrappers behind in the test process; see workspace._row_slots)."""
    from pinnstudio.ui.workspace import _row_slots, layout_sublayouts
    widgets, _nested = _row_slots(lay)
    return widgets, {lay.indexOf(sub): sub for sub in layout_sublayouts(lay)}


def _rows(w, key):
    page = w.workspace.pages[key]["page"]
    out = []

    def walk(lay):
        widgets, subs = _slots(lay)
        for i in range(lay.count()):
            if i in subs:
                walk(subs[i])
            elif i in widgets and widgets[i].layout() is not None:
                walk(widgets[i].layout())
        if not isinstance(lay, QHBoxLayout) or lay.count() < 2:
            return
        first = widgets.get(0)
        if not isinstance(first, QLabel) or not first.isVisibleTo(page) or first.wordWrap():
            return
        ctrl = next((widgets[j] for j in range(1, lay.count())
                     if j in widgets and widgets[j].isVisibleTo(page)), None)
        if ctrl is None or isinstance(ctrl, QLabel) or first.text().startswith(HEADERS):
            return
        gap = ctrl.mapTo(page, ctrl.rect().topLeft()).x() - first.mapTo(page, first.rect().topRight()).x()
        out.append((first, gap))
    walk(page.layout())
    return out


def test_rows_line_up(monkeypatch, tmp_path):
    (tmp_path / ".pinnstudio").mkdir()
    (tmp_path / ".pinnstudio" / "display_settings.json").write_text(json.dumps({"style_version": 2, "ui_scale": 100}))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QDialog, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    _pump()
    w.quick_examples_combo.setCurrentText("Heat 2D")
    _pump(0.5)
    for cb in w.findChildren(QCheckBox):        # the optional sections, shown
        if cb.text().startswith(("Show Training Callbacks", "Stop training", "Periodically")):
            cb.setChecked(True)
    bad = []
    for method in ("Residual-based Adaptive Refinement (RAR)", "Time Adaptive Training"):
        w.adapt_combo.setCurrentText(method)
        _pump()
        for key in w.workspace.order:
            if w.workspace.pages[key]["item"].isHidden():
                continue
            w.workspace.select(key)
            _pump(0.3)
            for label, gap in _rows(w, key):
                t = label.text().strip()
                if not t.endswith(":"):
                    bad.append(f"{key}: row label without a colon: {t!r}")
                elif gap > 20:
                    bad.append(f"{key}: {t!r} field {gap} px from its label")
                elif label.maximumWidth() > 1000:
                    bad.append(f"{key}: {t!r} too long for the label column")
    w.close()
    assert not bad, "\n".join(sorted(set(bad)))
