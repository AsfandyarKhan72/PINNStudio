"""Settings: Tuned stays chosen across examples.

An example without separate Tuned settings (its Default is already PINNStudio's best) loads with its
Default settings, but the user's choice must stay Tuned, so the next example loads with its Tuned
settings. (It used to switch the box to Default, and every later example then loaded Default.)
"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from pinnstudio.core import example_catalog as cat  # noqa: E402
from pinnstudio.core import example_settings as es  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)


def _load(w, key, category):
    e = next(x for x in cat.ENTRIES if x.key == key)
    w.quick_examples_settings_combo.setCurrentText(es.TUNED)
    w.quick_examples_category_combo.setCurrentText(category)
    app.processEvents()
    i = w.quick_examples_combo.findText(e.name)
    assert i >= 0
    w.quick_examples_combo.setCurrentIndex(-1)
    app.processEvents()
    w.quick_examples_combo.setCurrentIndex(i)
    for _ in range(30):
        app.processEvents()
    return w._build_config()


def test_tuned_choice_survives_an_example_without_tuned_settings():
    no_tuned, tuned = "Inverse: Lorenz system", "Wave propagation – space-time multi-scale Fourier features"
    assert not es.has_tuned(no_tuned, False) and es.has_tuned(tuned, False)
    w = MainWindow()
    _load(w, no_tuned, "Inverse")
    assert w.quick_examples_settings_combo.currentText() == es.TUNED
    c = _load(w, tuned, "Forward")
    phases = json.loads(c.scheduler_phases)
    assert [p["iterations"] for p in phases] == [30000, 15000], phases     # the Tuned settings, not DeepXDE's 10,000
