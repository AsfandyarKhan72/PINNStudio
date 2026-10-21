"""The initial condition comes from a formula or from a file, never both:
with "Load IC from file" ticked, the formula's "IC for" box is off and
can't be ticked again (the file always won, so a ticked formula did
nothing). The file only sets the first output's IC, so the option is shown
for the first output only (it used to show under every output and load
into the first one anyway)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def test_ic_formula_or_file():
    w = MainWindow()
    w.num_outputs_spin.setValue(2)
    app.processEvents()
    assert w.ic_from_file[0] is not None and w.ic_from_file[1] is None
    cb, act, hard = w.ic_from_file[0], w.ic_active[0], w.ic_hard[0]
    hard.setChecked(True)
    cb.setChecked(True)
    assert not act.isChecked() and not act.isEnabled()
    assert not hard.isChecked() and not hard.isEnabled()
    w.ic_file_paths[0].setText("/some/ic.txt")
    c = w._build_config()
    assert c.forward_ic_from_file and c.forward_ic_file == "/some/ic.txt"
    assert c.ic_active.split(",")[0].strip() != "True"
    cb.setChecked(False)
    assert act.isChecked() and act.isEnabled() and hard.isEnabled()
    assert not w._build_config().forward_ic_from_file
    w.close()
