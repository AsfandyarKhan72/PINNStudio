"""Train on: CPU (GPU / Hardware settings). The scripts then hide the GPU from
PyTorch before importing it; with the default (the GPU when there is one)
they are unchanged. The choice is part of the saved settings."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QPushButton

from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])
LINE = 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"'


def test_scripts():
    c = PINNConfig()
    assert c.train_device == "auto" and not c.validate()
    for gen in (generate_script, generate_clean_script):
        assert LINE not in gen(c)
    c.train_device = "cpu"
    assert not c.validate()
    for gen in (generate_script, generate_clean_script):
        text = gen(c)
        assert text.index(LINE) < text.index("import torch"), "set before PyTorch is imported"
        compile(text, gen.__name__, "exec")
    c.train_device = "gpu"
    assert any("Train on" in e for e in c.validate())


def test_dialog_and_saved_settings(monkeypatch):
    w = MainWindow()

    def _exec(dlg):
        combo = [c for c in dlg.findChildren(QComboBox) if c.findData("cpu") >= 0][0]
        combo.setCurrentIndex(combo.findData("cpu"))
        [b for b in dlg.findChildren(QPushButton) if b.text().replace("&", "") == "OK"][0].click()
        return 0
    monkeypatch.setattr(QDialog, "exec", _exec)
    w._on_gpu_settings()
    cfg = w._build_config()
    assert cfg.train_device == "cpu"
    w2 = MainWindow()
    w2._apply_config(cfg)
    assert w2._build_config().train_device == "cpu", "kept when a saved problem is opened"
    w.close()
    w2.close()
