"""GUI regression checks for the v1.6.0 audit (P2 group): settings that the window must offer, keep through
Save/Open, and pass to both generated scripts."""
import dataclasses
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.core import codegen  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def _settle(s=0.2):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _window():
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    _settle(0.3)
    return w


def test_ic_file_anchors_has_a_checkbox_and_survives_save_and_open():
    """P2: 'ic_file_anchors' was used by both generators but had no widget, so the GUI reset it to False and a
    problem saved with it on lost it."""
    w = _window()
    w.radio_1d.setChecked(True)
    _settle()
    assert w._build_config().ic_file_anchors is False
    w.ic_from_file[0].setChecked(True)
    w.ic_file_paths[0].setText("/tmp/ic.txt")
    w.ic_file_anchor_cb.setChecked(True)
    c = w._build_config()
    assert c.forward_ic_from_file and c.ic_file_anchors
    for s in (codegen.generate_script(c), codegen.generate_clean_script(c)):
        assert "anchors=_ic_xt" in s or "anchors=_ic_xyt" in s
    w2 = _window()
    w2._apply_config(c)
    _settle(0.3)
    assert w2._build_config().ic_file_anchors is True
    w2.ic_from_file[0].setChecked(False)              # no IC file: the setting has no meaning
    assert w2._build_config().ic_file_anchors is False
    w.close(); w2.close()
