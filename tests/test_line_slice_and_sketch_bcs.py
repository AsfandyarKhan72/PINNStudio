"""Line plots of 2D/3D problems cut the domain where something happens, and
leave a gap where the line crosses a hole; the geometry view shows every
boundary condition where it applies.

Helmholtz 2D with a hole: u = sin(k0 x) sin(k0 y) is exactly zero on the
middle line y = 0, which also runs through the hole, so the line
comparison showed the reference as a flat zero and the PINN's noise, drawn
straight across the hole. The automatic line now moves when the middle
shows (almost) nothing; it stays in the middle otherwise (Heat 2D).

Its inner-circle condition's Where: uses the constant R, which the
geometry view could not evaluate, so the circle was never highlighted; and
boundaries an output transform fixes (Helmholtz 2D Dirichlet) showed as
"no BC yet"."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtWidgets import QApplication

from pinnstudio.core import codegen
from pinnstudio.core.line_slice import auto_line_slice
from pinnstudio.ui.main_window import MainWindow
from pinnstudio.ui.theme import EXACT_BC_INDEX

app = QApplication.instance() or QApplication([])


def _load(w, name):
    w.quick_examples_combo.setCurrentText(name)
    for _ in range(5):
        app.processEvents()
    return w._build_config()


def test_auto_line_and_sketch_bcs():
    w = MainWindow()
    c = _load(w, "Helmholtz 2D – square with hole")
    y, _z = auto_line_slice(c)
    assert abs(y) > 0.1                                   # not the zero line through the hole
    c.line_slice_y_auto, c.line_slice_y = False, 0.0
    assert auto_line_slice(c)[0] == 0.0                   # a value set by hand is kept
    w._refresh_domain_view()
    app.processEvents()
    bc_of = w._gpanel().canvas.edge_bcs()
    assert any(0 in lst for lst in bc_of)                 # BC 1 (the circle, uses R) picks an edge

    c = _load(w, "Heat 2D")
    assert auto_line_slice(c)[0] == (c.y_min + c.y_max) / 2.0

    _load(w, "Helmholtz 2D – Dirichlet")
    w._refresh_domain_view()
    app.processEvents()
    rows = w._sketch_bc_rows()
    assert rows and rows[-1]["index"] == EXACT_BC_INDEX and "output transform" in rows[-1]["label"]
    assert all(EXACT_BC_INDEX in lst for lst in w._gpanel().canvas.edge_bcs())
    w.close()


def test_line_gaps():
    ns = {}
    exec(codegen._PLOT_TEXT_HELPERS, ns)
    x = np.linspace(0, 1, 11)
    keep = (x < 0.3) | (x > 0.6)
    gx, gy = ns["_line_gaps"](x[keep], x[keep] ** 2)
    assert np.isnan(gx).sum() == 1 and np.isnan(gy).sum() == 1
    gx, gy = ns["_line_gaps"](x, x)                       # nothing taken out: unchanged
    assert not np.isnan(gx).any() and len(gx) == 11
