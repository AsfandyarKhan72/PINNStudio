#!/usr/bin/env python3
"""2D Sketch domain: shapes drawn on the canvas become Custom-geometry rows
(with the chosen combine op), editing a row redraws the canvas, and the
config round-trips back to the Sketch choice."""
import json
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from pinnstudio.ui.main_window import MainWindow
from pinnstudio.ui.sketch_canvas import combined_path, polygon_shape

app = QApplication.instance() or QApplication(sys.argv)


def _settle(secs=0.3):
    t = time.time()
    while time.time() - t < secs:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def test_polygon_shape_picks_the_type_deepxde_accepts():
    assert polygon_shape([(0, 0), (1, 0), (0, 1)])["type"] == "Triangle"
    assert polygon_shape([(0, 0), (2, 0), (2, 1), (0, 1)])["type"] == "Rectangle"
    assert polygon_shape([(0, 0), (1, 0), (1, 1), (0.5, 1.5), (0, 1)])["type"] == "Polygon"


def test_sketch_draws_rows_and_rows_redraw_sketch():
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    app.processEvents()
    w.radio_2d.setChecked(True)
    app.processEvents()
    w.geometry_type_combo.setCurrentText("Sketch")
    app.processEvents()
    assert w.right_tabs.currentIndex() == w._sketch_tab_index
    sp = w.sketch_panel
    c = sp.canvas

    def at(x, y):
        return c.to_screen(QPointF(x, y)).toPoint()

    sp.clear()
    sp.set_tool("Rectangle")
    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=at(0, 0))
    QTest.mouseMove(c, at(1, 0.8))
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=at(1, 0.8))
    sp.op_combo.setCurrentIndex(1)  # subtract
    sp.set_tool("Circle")
    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=at(0.5, 0.4))
    QTest.mouseMove(c, at(0.7, 0.4))
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=at(0.7, 0.4))
    _settle()

    rows = json.loads(w._build_custom_geom_shapes_json())
    assert [r["type"] for r in rows] == ["Rectangle", "Disk"], rows
    assert rows[1].get("op") == "subtract"
    assert abs(rows[1]["params"]["r"] - 0.2) < 1e-9
    assert abs(rows[0]["params"]["y_max"] - 0.8) < 1e-9
    # the hole is cut out of the combined region
    path = combined_path(rows)
    assert path.contains(QPointF(0.1, 0.1)) and not path.contains(QPointF(0.5, 0.4))

    cfg = w._build_config()
    assert cfg.geometry_type == "Custom" and cfg.geom_sketch
    assert not cfg.validate()

    w.custom_geom_shape_rows[1]["disk_r"].setValue(0.15)
    _settle(0.5)
    assert abs(c.shapes[1]["params"]["r"] - 0.15) < 1e-9

    w2 = MainWindow()
    app.processEvents()
    w2._apply_config(cfg)
    app.processEvents()
    assert w2.geometry_type_combo.currentText() == "Sketch"
    w.right_tabs.setCurrentWidget(w.results_tabs)


if __name__ == "__main__":
    test_polygon_shape_picks_the_type_deepxde_accepts()
    test_sketch_draws_rows_and_rows_redraw_sketch()
    print("PASSED")
