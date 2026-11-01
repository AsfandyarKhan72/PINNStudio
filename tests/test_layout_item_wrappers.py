"""PySide6 keeps a Python wrapper for every layout item QLayout.itemAt() returns; Qt deletes such items
itself (a QLayoutItem is not a QObject, so PySide6 is not told) and the wrapper stays registered at the
freed address. workspace.align_forms used to walk every row with itemAt() on each page switch: after a few
example loads about 1200 such wrappers were alive and the heap was eventually corrupted (about 1 crash in
15 loads, tools/gui_stress.py). align_forms now works on the QObject tree only; this checks that switching
examples and pages no longer leaves layout-item wrappers behind."""
import gc
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QLayout, QLayoutItem  # noqa: E402

app = QApplication.instance() or QApplication([])
from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def _settle(s):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def _item_wrappers():
    gc.collect()
    return sum(1 for x in shiboken6.getAllValidWrappers()
               if isinstance(x, QLayoutItem) and not isinstance(x, QLayout))


def test_switching_examples_and_pages_leaves_no_layout_item_wrappers():
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.show()
    _settle(0.3)
    counts = []
    for _ in range(2):
        for name in ("Heat 3D", "Allen-Cahn 1D – Wight & Zhao (time-adaptive)"):
            w.quick_examples_combo.setCurrentIndex(w.quick_examples_combo.findText(name))
            _settle(0.5)
            for key in list(w.workspace.order):
                w.workspace.select(key)
                _settle(0.02)
        counts.append(_item_wrappers())
    w.close()
    w.deleteLater()
    _settle(0.3)
    # before the fix: ~1200 and growing; after: the few made while the window is built
    assert counts[-1] < 60, counts
    assert counts[-1] <= counts[0] + 5, counts


def test_align_forms_is_idempotent_and_aligns_labels():
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget
    from pinnstudio.ui.workspace import align_forms
    page = QWidget()
    v = QVBoxLayout(page)
    for text in ("Short:", "A longer label:"):
        h = QHBoxLayout()
        h.addWidget(QLabel(text))
        h.addStretch()
        e = QLineEdit()
        e.setFixedWidth(80)
        h.addWidget(e)
        v.addLayout(h)
    page.show()
    _settle(0.1)
    align_forms(page)
    counts = [h.count() for h in page.findChildren(QHBoxLayout)]
    align_forms(page)
    assert [h.count() for h in page.findChildren(QHBoxLayout)] == counts      # nothing added twice
    labels = page.findChildren(QLabel)
    assert len({l.minimumWidth() for l in labels}) == 1 and labels[0].minimumWidth() > 0
    # the control sits right after its label (the old stretch takes no room)
    for h in page.findChildren(QHBoxLayout):
        assert h.stretch(1) == 0
    page.close()


def test_no_layout_itemat_or_takeat_in_the_interface():
    """QLayout.itemAt/takeAt create the layout-item wrappers PySide6 keeps (see above); the interface finds
    a layout's widgets through the QObject tree instead (workspace.layout_widgets / layout_sublayouts)."""
    import re
    ui = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pinnstudio", "ui")
    bad = []
    for name in sorted(os.listdir(ui)):
        if name.endswith(".py"):
            for n, line in enumerate(open(os.path.join(ui, name), encoding="utf-8"), 1):
                code = line.split("#", 1)[0]
                if re.search(r"\.(itemAt|takeAt)\(", code) and "tree.itemAt" not in code:
                    bad.append(f"{name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)
