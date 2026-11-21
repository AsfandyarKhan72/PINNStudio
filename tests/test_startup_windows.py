"""Starting PINNStudio shows one window: a widget shown before it has a parent becomes a top-level window of
its own (a small empty one appeared before the main window)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QEvent, QObject  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

app = QApplication.instance() or QApplication([])


class _Spy(QObject):
    def __init__(self):
        super().__init__()
        self.windows = []

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            self.windows.append(f"{type(obj).__name__} {obj.width()}x{obj.height()}")
        return False


def test_only_the_main_window_is_shown_at_start(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from pinnstudio.ui.main_window import MainWindow
    spy = _Spy()
    app.installEventFilter(spy)
    try:
        w = MainWindow()
        w.show()
        app.processEvents()
    finally:
        app.removeEventFilter(spy)
    try:
        assert spy.windows == [f"MainWindow {w.width()}x{w.height()}"], spy.windows
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()
