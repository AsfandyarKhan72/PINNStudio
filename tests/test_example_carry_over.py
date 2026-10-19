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


def test_inverse_examples_in_a_row():
    """Every Inverse example loaded one after another has the same unknowns
    and measured-data files as when loaded first. Burgers 2D's second file
    (v) stayed as Heat 2D's second measured-data file, so Heat 2D's alpha
    came out at 2e-5 instead of 0.4."""
    import json
    w = MainWindow()
    w.quick_examples_category_combo.setCurrentText("Inverse")
    app.processEvents()
    names = [w.quick_examples_combo.itemText(i) for i in range(w.quick_examples_combo.count())]
    names = [n for n in names if n != "None" and not n.startswith("──")]
    bad = []
    for n in names:
        got = _load(w, "Inverse", n)
        fresh_w = MainWindow()
        fresh = _load(fresh_w, "Inverse", n)
        fresh_w.close()
        for k in ("inverse_obs_files_json", "inverse_variables_json"):
            if json.loads(got[k] or "[]") != json.loads(fresh[k] or "[]"):
                bad.append(f"{n}: {k}")
    w.close()
    assert not bad, bad
