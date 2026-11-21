"""No text is cut off: every button, checkbox, label and drop-down on the
settings pages and in the settings dialogs is as wide as its text needs, at
the default text size and at 175% (Windows' wider text, display scaling).

Fixed pixel widths (a 65 px Browse, a 60 px Reset, a 520 px dialog) cut
their text off once the text was larger; the theme now widens them."""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QLabel, QMessageBox, QPushButton,
                             QRadioButton, QWidget)

from pinnstudio.ui.main_window import ExampleCombo, MainWindow

app = QApplication.instance() or QApplication([])

DIALOGS = ("_on_float_settings", "_on_gpu_settings", "_on_seed_settings", "_on_lbfgs_settings",
           "_on_optimizer_settings", "_on_display_settings", "_on_export_settings", "_on_loss_plot_settings",
           "_on_error_analysis_btn", "_on_plot_settings")


def _pump(sec=0.3):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()


def _cut(root, where):
    out = []
    for c in root.findChildren(QWidget):
        if not c.isVisible() or c.width() < 4 or isinstance(c, ExampleCombo):
            continue      # (ExampleCombo shows a label, not its currentText())
        if isinstance(c, (QPushButton, QCheckBox, QRadioButton)) and c.text().strip():
            need, text = c.sizeHint().width(), c.text()
        elif isinstance(c, QLabel) and c.text().strip() and not c.wordWrap() and "<" not in c.text():
            need, text = c.sizeHint().width(), c.text()
        elif isinstance(c, QComboBox):
            need, text = c.fontMetrics().horizontalAdvance(c.currentText()) + 34, c.currentText()
        else:
            continue
        if need > c.width() + 3:
            out.append(f"{where}: {type(c).__name__} {text[:40]!r} is {c.width()} px, needs {need}")
    return out


@pytest.mark.parametrize("ui_scale", [100, 175])
def test_no_text_cut_off(monkeypatch, tmp_path, ui_scale):
    (tmp_path / ".pinnstudio").mkdir()
    (tmp_path / ".pinnstudio" / "display_settings.json").write_text(
        json.dumps({"style_version": 2, "ui_scale": ui_scale}))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    cut = []

    def _exec(dlg):
        dlg.show()
        _pump(0.3)
        cut.extend(_cut(dlg, "dialog " + dlg.windowTitle()))
        dlg.close()
        return 0
    monkeypatch.setattr(QDialog, "exec", _exec)

    w = MainWindow()
    w.resize(1500, 950)
    w.show()
    _pump(0.5)
    w.quick_examples_combo.setCurrentText("Navier-Stokes 2D – Kovasznay flow")
    _pump(0.5)
    for key in w.workspace.order:
        if not w.workspace.pages[key]["item"].isHidden():
            w.workspace.select(key)
            _pump(0.2)
            cut += _cut(w, key)
    # the Network tab: the palette, the canvas and a block's settings in the inspector
    w.network_type_combo.setCurrentIndex(w.network_type_combo.findText("Designed"))
    _pump(0.3)
    w.right_tabs.setCurrentIndex(w._network_tab_index)
    _pump(0.5)
    stack = next(b["id"] for b in w.design_state.level()["blocks"] if b["kind"] == "stack")
    w.designer.select(stack)
    _pump(0.5)
    panel = w.designer_panel
    cut += _cut(panel.inspector, "network inspector")
    canvas = panel.canvas
    assert canvas.width() >= 4 * panel.palette.width() // 3 or canvas.width() >= 300, \
        f"the canvas is squeezed: {canvas.width()} px beside a {panel.palette.width()} px palette"
    # the cards grow with the UI scale (by zooming), and are drawn in the scene's own font
    assert canvas.text_scale() == pytest.approx(1.0, abs=0.35) if ui_scale == 100 else canvas.text_scale() > 1.5
    from PySide6.QtGui import QImage, QPainter
    item = canvas.blocks[stack]
    img = QImage(400, 300, QImage.Format.Format_ARGB32)
    seen = []
    pt = QPainter(img)
    pt.setFont(canvas.font())                       # as the viewport's (stylesheet-scaled) font would be
    item.paint(pt, None, None)
    seen.append(pt.fontMetrics().height())
    pt.end()
    from PySide6.QtGui import QFontMetrics
    assert seen[0] <= QFontMetrics(canvas.scene_.font()).height() + 1, "a card's text follows the viewport font"
    for name in DIALOGS:
        getattr(w, name)()
    assert not cut, "\n".join(sorted(set(cut)))
    w.close()
    w.deleteLater()
    _pump(0.2)
