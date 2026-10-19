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
