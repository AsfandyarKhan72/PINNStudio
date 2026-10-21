"""GUI workflow fixes: picking the example already shown loads it again;
Plot settings OK with nothing changed leaves the problem unchanged; a
Forward-list template loads in Forward after an Inverse example; a problem
file with a value of the wrong type is refused with a message; suggested
file names have no "/" or ":"; a Stop is not reported as an error."""
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow, _safe_file_stem


def _settle(s=0.3):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _window():
    w = MainWindow()
    _settle()
    return w


def _pick(w, category, key):
    w.quick_examples_category_combo.setCurrentText(category)
    _settle(0.1)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    assert i >= 0, key
    combo.setCurrentIndex(i)
    combo.activated.emit(i)       # what a click in the list sends after the change
    _settle(0.3)
    return i


def test_picking_the_shown_example_again_reloads_it():
    w = _window()
    w._maybe_save_changes = lambda: True
    i = _pick(w, "Forward", "1D Heat")
    x0 = w._build_config().x_max
    w.x_max.setValue(x0 + 2)
    assert w._build_config().x_max != x0
    w.quick_examples_combo.activated.emit(i)     # picked again, index unchanged
    _settle(0.3)
    assert w._build_config().x_max == x0
    w.close()


def test_plot_settings_ok_leaves_problem_unchanged():
    from pinnstudio.ui.figure_style import FigureStyleDialog
    w = _window()
    w._maybe_save_changes = lambda: True
    _pick(w, "Forward", "2D Heat")
    before = MainWindow._config_key(w._build_config())
    d = FigureStyleDialog(w, "results", "Line")
    d._ok()
    _settle(0.1)
    assert MainWindow._config_key(w._build_config()) == before
    w.close()


def test_forward_template_after_inverse_example():
    w = _window()
    w._maybe_save_changes = lambda: True
    _pick(w, "Inverse", "1D Burgers")
    assert w.radio_inverse.isChecked()
    _pick(w, "Forward", "2D Heat")
    assert w.radio_forward.isChecked()
    w.radio_inverse.setChecked(True)       # Inverse chosen by the user stays
    _pick(w, "Forward", "1D Heat")
    assert w.radio_inverse.isChecked()
    _pick(w, "Inverse", "1D Burgers")
    w._load_example_from_start("2D Heat")  # the Start page loads in Forward
    assert w.radio_forward.isChecked()
    w.close()


def test_wrong_type_value_in_file():
    w = _window()
    d = tempfile.mkdtemp()
    p = os.path.join(d, "bad.pinn.json")
    w._current_path = None
    data = {"config": {"num_domain": "2000"}}
    with open(p, "w") as f:
        json.dump(data, f)
    w._open_problem(p)
    assert "a value has the wrong type" in w.log_box.toPlainText()
    w.close()


def test_suggested_file_names():
    assert _safe_file_stem("Poisson 1D – Dirichlet/Neumann (copy)") == "Poisson 1D – Dirichlet - Neumann (copy)"
    assert _safe_file_stem("Inverse: Lorenz system") == "Inverse - Lorenz system"
    assert _safe_file_stem("Heat 1D") == "Heat 1D"


def test_stop_is_not_an_error():
    w = _window()
    w._stopped = True
    w.log_box.clear()
    w._on_done("ERROR")
    assert "Error during training" not in w.log_box.toPlainText()
    w._stopped = False
    w._on_done("ERROR")
    assert "Error during training" in w.log_box.toPlainText()
    w.close()
