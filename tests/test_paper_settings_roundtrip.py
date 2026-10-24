"""The papers' settings (paper_settings.py) reach training as written when the
window loads them: the loss weights and each training phase's loss function."""
import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core import paper_settings as ps  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402
import tuning_study as ts  # noqa: E402


def _loaded(cfg):
    w = MainWindow()
    w._apply_config(cfg)
    for _ in range(10):
        app.processEvents()
    got = w._build_config()
    w.close()
    return got


@pytest.mark.parametrize("key,variant", [("1D Burgers", "plain"), ("2D Allen-Cahn (Wight & Zhao)", "plain"),
                                         ("2D Burgers (Mathias)", "paper_mathias")])
def test_paper_settings_load_unchanged(key, variant):
    cfg = ps.apply(ts.template_config(key, False), variant, key)
    got = _loaded(cfg)
    want = [(p["weights"], p["loss"]) for p in json.loads(cfg.scheduler_phases)]
    have = [(p["weights"], p["loss"]) for p in json.loads(got.scheduler_phases)]
    assert have == want      # the base PINN's equal weights, Mathias et al.'s MAE loss
