"""Standalone reproducer (PySide6 only, no PINNStudio): stale QLayoutItem wrappers.

PySide6 keeps the Python wrapper of every QLayoutItem returned by QLayout.itemAt() alive (PyQt6 does not).
When Qt itself deletes such an item -- here because the widget whose layout holds it is deleted -- PySide6
is not notified (QLayoutItem is not a QObject), so the wrapper stays registered at the freed address and
reports isValid() == True.  Repeating "walk the layouts with itemAt(), delete the rows, build new rows"
accumulates stale wrappers; when their addresses are reused the process eventually aborts with heap
corruption (malloc_consolidate(): unaligned fastbin chunk detected / free(): invalid pointer) or segfaults.

    python pyside_layoutitem_repro.py [ITERATIONS]        (QT_QPA_PLATFORM=offscreen is fine)

Prints the number of live QWidgetItem/QSpacerItem wrappers every 50 iterations.
"""
import faulthandler
import gc
import os
import sys
import time

faulthandler.enable()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,  # noqa: E402
                               QLabel, QLayout, QLayoutItem, QLineEdit, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
app = QApplication([])


def walk(lay):
    """Read-only, like a form-alignment helper: visit every item of every nested layout."""
    for i in range(lay.count()):
        it = lay.itemAt(i)
        if it.layout() is not None:
            walk(it.layout())
        elif it.widget() is not None and it.widget().layout() is not None:
            walk(it.widget().layout())
        elif it.spacerItem() is not None:
            it.spacerItem().sizeHint()


def make_row(i):
    box = QGroupBox(f"Condition {i}")
    v = QVBoxLayout(box)
    for j, cls in enumerate((QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox)):
        h = QHBoxLayout()
        h.addWidget(QLabel(f"field {j}:"))
        h.addStretch()
        w = cls()
        if isinstance(w, QComboBox):
            w.addItems(["a", "b", "c"])
        h.addWidget(w)
        v.addLayout(h)
    inner = QWidget()
    il = QHBoxLayout(inner)
    il.addWidget(QLabel("extra:"))
    il.addStretch()
    il.addWidget(QLineEdit())
    v.addWidget(inner)
    return box


stack = QStackedWidget()
stack.resize(600, 800)
pages = []
for p in range(8):
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.addStretch()
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(page)
    stack.addWidget(sc)
    pages.append(page)
stack.show()


def settle(s):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def live_item_wrappers():
    gc.collect()
    return sum(1 for x in gc.get_objects() if isinstance(x, QLayoutItem) and not isinstance(x, QLayout))


for k in range(N):
    for p, page in enumerate(pages):
        lay = page.layout()
        # delete the old rows with deleteLater only (valid Qt: when a widget is destroyed its layout drops
        # the widget's item and Qt deletes that item -- PySide6 is not told)
        for i in reversed(range(lay.count() - 1)):
            w = lay.itemAt(i).widget()
            if w is not None:
                w.deleteLater()
        for i in range(2 + (k + p) % 5):
            lay.insertWidget(lay.count() - 1, make_row(i))
    settle(0.005)
    for p, page in enumerate(pages):        # show each page and walk it
        stack.setCurrentIndex(p)
        if os.environ.get("WALK", "1") == "1":
            walk(page.layout())
        settle(0.002)
    if k % 50 == 0:
        print(f"iteration {k}: live QWidgetItem/QSpacerItem wrappers: {live_item_wrappers()}", flush=True)
print("finished", N, "iterations without a crash", flush=True)
os._exit(0)
