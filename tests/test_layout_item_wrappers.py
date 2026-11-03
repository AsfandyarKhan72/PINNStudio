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
    for _ in range(3):
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
    assert counts[-1] <= counts[0] + 5, counts
    if not _binding_wraps_items_on_add_layout():
        assert counts[-1] < 60, counts


def _binding_wraps_items_on_add_layout():
    """PySide6 6.11 and older wrap every item of a layout passed to addLayout() (their ownership code
    walks it) and track those wrappers themselves, so after loading examples thousands are alive, a
    constant number that does not grow (9741 here with 6.11.3 and 6.10.3). 6.12 makes none. Only the
    growth check above applies then."""
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
    before = _item_wrappers()
    host = QWidget()
    outer = QVBoxLayout(host)
    row = QHBoxLayout()
    row.addWidget(QLabel("probe"))
    outer.addLayout(row)
    made = _item_wrappers() - before
    host.deleteLater()
    _settle(0.05)
    return made > 0


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
    """QLayout.itemAt/takeAt create the layout-item wrappers PySide6 keeps (see above); the package finds
    a layout's widgets through the QObject tree instead (workspace.layout_widgets / layout_sublayouts)."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    # the package, and the tests and tools that run in the same process as the window (a wrapper left by a
    # test walking a layout with itemAt stays in the test process and skews the counts above)
    files = [os.path.join(d, f) for top in ("pinnstudio", "tests", "tools")
             for d, _s, fs in os.walk(os.path.join(root, top)) for f in fs if f.endswith(".py")]
    for path in sorted(files):
        name = os.path.relpath(path, root)
        if name == os.path.join("tests", "test_layout_item_wrappers.py"):
            continue
        if True:
            for n, line in enumerate(open(path, encoding="utf-8"), 1):
                code = line.split("#", 1)[0]
                if re.search(r"\.(itemAt|takeAt)\(", code) and "tree.itemAt" not in code:
                    bad.append(f"{name}:{n}: {line.strip()}")
    assert not bad, "\n".join(bad)


def test_closing_the_window_releases_all_its_widgets():
    """Settings kept as widgets but never placed in the window had no parent; PySide6 keeps the closed window's
    Python object alive (its signal connections hold it), so 18 controls (64 widgets) outlived every closed
    window. They are now children of the window (hidden) and go with it."""
    before = len(app.allWidgets())
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.show()
    _settle(0.3)
    w.close()
    w.deleteLater()
    del w
    _settle(0.5)
    gc.collect()
    _settle(0.2)
    assert len(app.allWidgets()) <= before, len(app.allWidgets()) - before


def test_example_loads_do_not_accumulate_styled_widget_entries():
    """Every example load rebuilds the condition rows; their entries in the Display Settings registry
    (with a style lambda each) used to stay until the next Display Settings change (about 50 per load)."""
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.show()
    _settle(0.3)
    sizes = []
    for _ in range(12):
        for name in ("Heat 3D", "Allen-Cahn 1D – Wight & Zhao (time-adaptive)"):
            w.quick_examples_combo.setCurrentIndex(w.quick_examples_combo.findText(name))
            _settle(0.3)
        sizes.append(len(w._styled_widgets))
    w.close()
    w.deleteLater()
    _settle(0.3)
    assert max(sizes[4:]) <= 2 * max(256, sizes[3]), sizes
