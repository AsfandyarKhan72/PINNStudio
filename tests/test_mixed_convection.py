"""The mixed-convection template (Soomro et al., 2020): listed as a PINNStudio
template, loads in Forward and Inverse mode, generates a valid script, and
ships its reference data."""
import ast
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication

from pinnstudio.core import example_catalog as cat
from pinnstudio.core import pinnstudio_examples as ps
from pinnstudio.core.codegen import generate_script, generate_clean_script

KEY = "2D Mixed convection (Soomro et al.)"
REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_data", "2D",
                   "mixed_convection")
_app = QApplication.instance() or QApplication([])


def _settle():
    for _ in range(5):
        _app.processEvents()


def test_catalog_entry():
    e = cat.BY_KEY[KEY]
    assert e.source == cat.PINNSTUDIO and e.name == "Mixed convection 2D – Soomro et al."
    assert e.dim == "2D" and e.regime == "Steady" and e.problem == cat.BOTH
    assert e.geometry == "Triangle − Disk" and cat.PAPERS[e.reference].startswith("https://doi.org/")
    assert KEY in [x.key for x in cat.for_category(cat.FORWARD)]
    assert KEY in [x.key for x in cat.for_category(cat.INVERSE)]


def test_reference_data():
    for name in ("u", "v", "p", "T"):
        d = np.loadtxt(os.path.join(REF, f"solution_{name}.txt"))
        assert d.shape[1] == 3 and len(d) > 5000 and np.isfinite(d).all(), name
        x, y = d[:, 0], d[:, 1]
        # every point inside the triangle and outside the cylinder
        assert (y >= 1 - np.sqrt(3) * np.minimum(x, 1 - x) - 1e-9).all()
        assert (np.hypot(x - 0.5, y - 0.7) >= 0.1 - 1e-9).all()
    T = np.loadtxt(os.path.join(REF, "solution_T.txt"))[:, 2]
    assert T.min() > -0.02 and T.max() < 1.02
    for name in ("u", "v"):
        assert os.path.isfile(os.path.join(REF, f"obs_{name}.txt"))


def test_gui_forward_and_inverse():
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    _settle()
    w.quick_examples_combo.setCurrentText(KEY)
    _settle()
    c = w._build_config()
    assert c.output_names == "u,v,p,T" and c.steady_state and c.geometry_type == "Custom"
    assert c.problem_type == "Forward" and c.validate() == [], c.validate()
    assert "Ha**2/Re*v" in c.pde_expressions
    for gen in (generate_script, generate_clean_script):
        ast.parse(gen(c))
    assert "Template loaded" in w.log_box.toPlainText()
    w.quick_examples_category_combo.setCurrentText(cat.INVERSE)
    _settle()
    w.quick_examples_combo.setCurrentText(KEY)
    _settle()
    c = w._build_config()
    assert c.problem_type == "Inverse" and '"Ha"' in c.inverse_variables_json and c.validate() == []
    ast.parse(generate_script(c))
    # back to the Forward list (the category is remembered for the next window)
    w.quick_examples_category_combo.setCurrentText(cat.FORWARD)
    _settle()
    w.close()
    w.deleteLater()
    _settle()


def test_settings_match_the_paper():
    c = ps.build_config(ps.MIXED_CONVECTION, "/r")
    consts = {d["name"]: d["value"] for d in __import__("json").loads(c.constants)}
    assert consts == {"Re": "300", "Ri": "0.1", "Ha": "10", "Pr": "6.2"}
    assert len(c.pde_expressions.split("|")) == 4
