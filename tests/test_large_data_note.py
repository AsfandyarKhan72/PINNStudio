"""Solve notes in the log when an inverse problem's measured data has many
points (every one is used at every training step); small data, or a forward
problem, gets no note."""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def test_large_data_note(tmp_path):
    w = MainWindow()
    big, small = tmp_path / "big.txt", tmp_path / "small.txt"
    big.write_text("x t u\n" + "0 0 0\n" * 25000)
    small.write_text("x t u\n" + "0 0 0\n" * 100)
    c = PINNConfig()
    c.problem_type = "Inverse"
    for path, noted in ((big, True), (small, False)):
        w.log_box.clear()
        c.inverse_obs_files_json = json.dumps([{"path": str(path), "output_idx": 0, "weight": 100.0}])
        w._note_large_data(c)
        assert ("25,000 points" in w.log_box.toPlainText()) == noted
    c.problem_type = "Forward"
    c.inverse_obs_files_json = json.dumps([{"path": str(big)}])
    w.log_box.clear()
    w._note_large_data(c)
    assert not w.log_box.toPlainText()
    w.close()


def test_measured_points_as_anchors_option():
    """Measured data -> "Check the equation at the measured points too":
    on by default; off keeps the points only in the data loss."""
    from pinnstudio.core.codegen import generate_script, generate_clean_script
    w = MainWindow()
    w.quick_examples_category_combo.setCurrentText("Inverse")
    app.processEvents()
    w.quick_examples_combo.setCurrentText("Heat 2D")
    for _ in range(5):
        app.processEvents()
    assert w.inv_obs_anchors_cb.isChecked()
    c = w._build_config()
    assert c.inverse_obs_anchors
    assert "_obs_anchors = np.vstack(" in generate_script(c)
    w.inv_obs_anchors_cb.setChecked(False)
    c = w._build_config()
    assert not c.inverse_obs_anchors
    assert "_obs_anchors = None" in generate_script(c)
    assert "obs_anchors = None" in generate_clean_script(c)
    w.inv_obs_anchors_cb.setChecked(True)
    w._apply_config(c)                       # a saved problem with it off opens with it off
    app.processEvents()
    assert not w.inv_obs_anchors_cb.isChecked()
    w.close()
