"""Edit › Undo and Redo: the problem's settings step back and forward (a change is recorded once it has
settled), the sketch and the network design take the keys when they have the focus, and a new problem
starts a new history."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def _window():
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    for _ in range(3):
        app.processEvents()
    w._new_problem("1D")
    w.show()
    app.processEvents()
    return w


def test_settings_undo_and_redo():
    w = _window()
    try:
        w._check_problems()
        assert not w.settings_history.can_undo()
        w.neurons_spin.setValue(40)
        w._check_problems()
        w.pde_inputs[0].setText("du_t - 0.2*du_xx")
        w._check_problems()
        assert w.settings_history.can_undo()
        w.menuBar().setFocus()
        w.undo_action.trigger()
        assert w.pde_inputs[0].text() != "du_t - 0.2*du_xx" and w.neurons_spin.value() == 40
        w._check_problems()                       # settling after an undo records nothing new
        assert w.settings_history.can_redo()
        w.undo_action.trigger()
        assert w.neurons_spin.value() == 64
        w.redo_action.trigger()
        w.redo_action.trigger()
        assert w.neurons_spin.value() == 40 and w.pde_inputs[0].text() == "du_t - 0.2*du_xx"
        w._new_problem("2D")
        assert not w.settings_history.can_undo(), "a new problem starts a new history"
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_the_designer_takes_undo_when_it_has_the_focus():
    w = _window()
    try:
        w.network_type_combo.setCurrentText("Designed")
        app.processEvents()
        n = len(w.design_state.design["blocks"])
        w.designer.add("stack", "dense", 0, 300)
        assert len(w.design_state.design["blocks"]) == n + 1
        w.designer_panel.palette.setFocus()
        app.processEvents()
        w.undo_action.trigger()
        assert len(w.design_state.design["blocks"]) == n
        w.designer_panel.canvas.setFocus()
        app.processEvents()
        QTest.keyClick(w.designer_panel.canvas, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        assert len(w.design_state.design["blocks"]) == n + 1
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()
