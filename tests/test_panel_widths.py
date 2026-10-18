"""The divider between the settings and the graphics column moves freely:
the graphics column's minimum width follows the tab that is showing, not the
widest page of all its tabs (Script note, Sketch toolbar, History buttons),
which on Windows stopped the divider early with Error analysis open."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSplitter

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


def _pump(n=20):
    for _ in range(n):
        app.processEvents()


def test_graphics_column_minimum_follows_visible_tab():
    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    _pump()
    sp = [s for s in w.findChildren(QSplitter)
          if s.orientation() == Qt.Orientation.Horizontal and s.isVisible() and s.count() == 3][0]
    graphics = sp.widget(2)
    w.right_tabs.setCurrentWidget(w.results_tabs)
    w.results_tabs.setCurrentWidget(w.ea_panel)
    w.bottom_tabs.setCurrentIndex(0)
    _pump()
    ea_min = graphics.minimumSizeHint().width()
    w.bottom_tabs.setCurrentIndex(1)            # Script tab below the plots
    _pump()
    script_min = graphics.minimumSizeHint().width()
    w.results_tabs.setCurrentWidget(w.run_history)
    _pump()
    hist_min = graphics.minimumSizeHint().width()
    w.results_tabs.setCurrentWidget(w.ea_panel)
    w.bottom_tabs.setCurrentIndex(0)
    _pump()
    assert ea_min <= 350, ea_min                # was 576 (Linux), wider on Windows
    assert script_min <= 450, script_min        # was 738
    assert graphics.minimumSizeHint().width() == ea_min   # goes back down
    assert hist_min > ea_min                    # History still needs room for its buttons
    w.close()
    w.deleteLater()
    _pump()
