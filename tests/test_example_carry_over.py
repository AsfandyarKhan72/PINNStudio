"""A template loaded after a DeepXDE example has the same settings as when
it is loaded first. Templates set only what differs between templates, so
the example's output transform formula, constants, initializer and loss
weights used to carry over (Allen-Cahn 1D, then the whole-interval
Allen-Cahn template: Solve refused it, the exact initial condition clashing
with Allen-Cahn 1D's formula)."""
import dataclasses
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from pinnstudio.ui.main_window import MainWindow

app = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_dialogs(monkeypatch):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QDialog, "exec", lambda self: 0)
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))


def _load(w, category, name):
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    i = w.quick_examples_combo.findText(name)
    assert i >= 0, name
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(10):
        app.processEvents()
    cfg = dataclasses.asdict(w._build_config())
    cfg.pop("save_dir")
    return cfg


def _free(w):
    """Close and really delete a window: about 80 of them otherwise stay in
    memory and the test runs out of it on a 7 GB machine."""
    import gc
    from PyQt6.QtCore import QCoreApplication, QEvent
    w.close()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()
    gc.collect()


@pytest.mark.parametrize("target", [("Forward", "Allen-Cahn 1D – Wight & Zhao (whole interval)"),
                                    ("Inverse", "Heat 2D"), ("Forward", "Heat 3D")])
def test_template_after_deepxde_example(target):
    first = MainWindow()
    fresh = _load(first, *target)
    first.close()
    w = MainWindow()
    _load(w, "Forward", "Allen-Cahn 1D")
    got = _load(w, *target)
    w.close()
    diff = {k: (fresh[k], got[k]) for k in fresh if fresh[k] != got[k] and k != "y_min"}
    assert not diff, diff
    assert not w._build_config().validate()


@pytest.mark.parametrize("category", ["Inverse", "Forward"])
def test_examples_in_a_row(category):
    """Every example loaded one after another has the same settings as when
    loaded first. Found: Burgers 2D's second measured-data file stayed with
    Heat 2D (whose alpha then came out at 2e-5 instead of 0.4), 3D Allen-Cahn
    after 3D Poisson kept sin(pi*x) as its initial condition, time-step groups
    and the sphere's centre carried from one template to the next, and the
    hyperparameter search's sweep rows stayed for every later example."""
    w = MainWindow()
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    names = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
    names = [n for n in names if n != "None" and not n.startswith("──")]
    bad = []
    prev = None
    for n in names:
        got = _load(w, category, n)
        fresh_w = MainWindow()
        fresh = _load(fresh_w, category, n)
        _free(fresh_w)
        diff = sorted(k for k in fresh if fresh[k] != got[k] and k != "y_min")
        if diff:
            bad.append(f"{n} (after {prev}): {diff}")
        prev = n
    w.close()
    assert not bad, "\n".join(bad)
