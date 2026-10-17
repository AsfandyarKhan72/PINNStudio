#!/usr/bin/env python3
"""
Round 68 -- simpler boundary-condition "Where" expressions.

 - Named sides of the domain box: left/right (x = x_min/x_max), bottom/top
   (y, 2D/3D), front/back (z, 3D), compared with a tolerance against the
   problem's own domain limits, in both generators.
 - The domain limits themselves (x_min, x_max, y_min, ..., t_min, t_max)
   are names in every BC expression, Value included ("x <= x_min" now works
   as the location hint always said).
 - A "Where:" picker on each BC row (Whole boundary / Left / Right / ... /
   Custom expression) that writes the expression and follows typed text.
 - Built-in templates and the DeepXDE examples use the side names; the
   DeepXDE examples write constant boundary values as numbers.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round68_bc_where_names.py
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import (_bc_domain_values, _clean_bc_row_code, generate_clean_script,
                                     generate_script)
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _cfg(**kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                num_initial=0, pde_expressions="du_xx - 2", layers=[1, 20, 20, 1], iterations=300,
                optimizer2="none", iterations2=0, optimizer_scheduler=False, num_domain=16, num_boundary=2,
                num_test=50,
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "left", "value": "0", "component": 0},
                                           {"type": "neumann", "location": "right", "value": "4", "component": 0}]),
                ea_ref_mode="formula", ea_ref_formula="(x + 1)**2")
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    # 1) Export: side names and limits defined from the domain, readable.
    c2 = _cfg(problem_dim="2D", geometry_type="Rectangle", x_min=0.0, x_max=2.0, y_min=-1.0, y_max=3.0)
    dom = _bc_domain_values(c2)
    check(dom == {"x_min": 0.0, "x_max": 2.0, "y_min": -1.0, "y_max": 3.0},
          f"steady 2D domain names {dom}")
    check("t_min" in _bc_domain_values(_cfg(steady_state=False, t_min=0.0, t_max=2.0)), "t_min/t_max when time-dependent")
    d, _a = _clean_bc_row_code(0, {"type": "dirichlet", "location": "left or top", "value": "x_max - x"},
                               True, False, 2, True, domain=dom)
    src = "\n".join(d)
    check("left = bool(np.isclose(x, 0.0))  # x = x_min" in src and "top = bool(np.isclose(y, 3.0))  # y = y_max" in src,
          "export defines only the side names used, against the domain limits")
    check("right =" not in src and "bottom =" not in src, "unused side names are not defined")
    check("    x_max = 2.0" in src, "domain limit used in Value is defined")
    ns = {"np": np}
    exec(src, ns)
    loc, val = ns["_bc0_loc"], ns["_bc0_val"]
    pts = {(0.0, 1.0): True, (2.0, 1.0): False, (1.0, 3.0): True, (1.0, -1.0): False, (1e-12, 0.5): True}
    check(all(loc(np.array(p), True) == want for p, want in pts.items()), "left or top picks the right points")
    check(not loc(np.array([0.0, 1.0]), False), "points off the boundary are never picked")
    check(np.allclose(val(np.array([[0.5, 0.0], [2.0, 1.0]])).ravel(), [1.5, 0.0]), "x_max in a value")
    d, _a = _clean_bc_row_code(1, {"type": "neumann", "location": "x <= x_min", "value": "4"},
                               False, False, 1, True, domain={"x_min": -1.0, "x_max": 1.0})
    ns = {"np": np}
    exec("\n".join(d), ns)
    check(ns["_bc1_loc"](np.array([-1.0]), True) and not ns["_bc1_loc"](np.array([1.0]), True),
          "x <= x_min works literally")

    # 2) Live script: the same names, from the same domain.
    live = generate_script(c2)
    m = re.search(r"^_BC_DOMAIN = (\{.*\})$", live, re.M)
    check(m and ast.literal_eval(m.group(1)) == dom, "live script defines _BC_DOMAIN from the domain")
    m = re.search(r"^_BC_SIDES = (\[.*\])$", live, re.M)
    check(m and ast.literal_eval(m.group(1)) == [("left", 0, 0.0), ("right", 0, 2.0), ("bottom", 1, -1.0), ("top", 1, 3.0)],
          "live script defines the 2D sides")
    c3 = _cfg(problem_dim="3D", geometry_type="Cuboid", y_min=0.0, y_max=1.0, z_min=0.0, z_max=5.0)
    m = re.search(r"^_BC_SIDES = (\[.*\])$", generate_script(c3), re.M)
    check(m and ("back", 2, 5.0) in ast.literal_eval(m.group(1)), "3D has front/back on z")
    for gen in (generate_script, generate_clean_script):
        try:
            ast.parse(gen(_cfg()))
            ok = True
        except SyntaxError:
            ok = False
        check(ok, f"{gen.__name__}: valid Python")

    # 3) GUI picker.
    w = MainWindow()
    w.radio_1d.setChecked(True)
    e = w._add_custom_bc_entry(location="right")
    pick = e["location_picker"]
    labels = [pick.itemText(i) for i in range(pick.count())]
    check(labels == ["Whole boundary", "Left side (x = x_min)", "Right side (x = x_max)", "Left and right sides",
                     "Custom expression…"], f"1D choices {labels}")
    check(pick.currentText() == "Right side (x = x_max)", "picker follows the expression")
    # a named choice says it in words: its code ("right", "True") stays hidden
    check(e["location"].isHidden(), "expression box hidden for a named choice")
    e["location"].setText("x <= x_min")
    check(pick.currentText() == "Custom expression…", "typed expression -> Custom")
    check(not e["location"].isHidden(), "expression box shown for a custom expression")
    e["location"].setText("")
    check(pick.currentText() == "Whole boundary", "empty Where = whole boundary")
    check(e["location"].isHidden(), "whole boundary: no 'True' shown")
    pick.setCurrentIndex(pick.findData("left or right"))
    w._on_bc_where_picked(pick, e["location"])
    check(e["location"].text() == "left or right", "picking writes the expression")
    # Custom expression…: the box opens and stays open while typing, even
    # through text that matches a named choice
    pick.setCurrentIndex(pick.count() - 1)
    w._on_bc_where_picked(pick, e["location"])
    check(not e["location"].isHidden(), "Custom expression… opens the box")
    e["location"].setText("left")
    check(not e["location"].isHidden() and pick.currentText() == "Custom expression…",
          "typing a named choice's code keeps the custom box open")
    pick.setCurrentIndex(pick.findData("True"))
    w._on_bc_where_picked(pick, e["location"])
    check(e["location"].isHidden(), "picking a named choice closes it again")
    w.radio_3d.setChecked(True)
    e3 = w._add_custom_bc_entry(location="back")
    check(e3["location_picker"].currentText() == "Back side (z = z_max)", "3D offers front/back")

    # periodic rows: the axis says which sides are tied, so Where: is set from it and hidden
    ep = w._add_custom_bc_entry(bc_type="periodic", location="True", axis="y")
    loc_w = ep["location"].parentWidget()
    check(ep["location"].text() == "True" and loc_w.isHidden(),
          "periodic: whole boundary (DeepXDE's own examples) is kept as it is, and hidden")
    epb = w._add_custom_bc_entry(bc_type="periodic", location="bottom", axis="y")
    epb["axis"].setCurrentText("z")
    check(epb["location"].text() == "front" and epb["location"].parentWidget().isHidden(),
          "periodic: changing the axis moves the side")
    epm = w._add_custom_bc_entry(bc_type="periodic", location="bottom", axis="x")
    check(epm["location"].text() == "bottom" and not epm["location"].parentWidget().isHidden(),
          "periodic: a side that doesn't fit the axis is kept and shown, never rewritten on load")
    ep2 = w._add_custom_bc_entry(bc_type="periodic", location="right", axis="x")
    check(ep2["location"].text() == "right" and ep2["location"].parentWidget().isHidden(),
          "periodic: either side of the pair is kept as it is")
    ep3 = w._add_custom_bc_entry(bc_type="periodic", location="x <= 0.5 and left", axis="x")
    check(ep3["location"].text() == "x <= 0.5 and left" and not ep3["location"].parentWidget().isHidden(),
          "periodic with a custom Where: kept and shown")

    # 4) Built-in templates use side names.
    w.quick_examples_category_combo.setCurrentText("Forward")
    w.quick_examples_combo.setCurrentText("1D Heat")
    locs = [r["location"] for r in json.loads(w._build_config().custom_bc_json)]
    check(locs and set(locs) <= {"left", "right"}, f"1D Heat BC rows use left/right: {locs}")
    w.quick_examples_combo.setCurrentText("3D Heat")
    locs = [r["location"] for r in json.loads(w._build_config().custom_bc_json)]
    check(set(locs) == {"left", "right", "bottom", "top", "front", "back"}, f"3D Heat faces: {locs}")

    # 5) DeepXDE examples: side names; np.isclose only where no side fits.
    for ex in dx.EXAMPLES:
        for row in json.loads(ex.settings.get("custom_bc_json", "[]")):
            loc = row.get("location", "")
            if "isclose" in loc:
                check(ex.name in ("Helmholtz 2D with a hole", "Helmholtz sound-hard scattering (absorbing BC)"),
                      f"{ex.name}: only the holes need np.isclose ({loc})")
    rows = json.loads(dx.BY_NAME["Poisson 1D – Dirichlet/Neumann"].settings["custom_bc_json"])
    check([(r["location"], r["value"]) for r in rows] == [("left", "0"), ("right", "4")],
          "Dirichlet/Neumann example: left 0, right 4")

    # 6) End to end (needs deepxde): both scripts train and agree.
    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        l2 = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            c = _cfg(save_dir=d)
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(c))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            l2[gen.__name__] = re.findall(r"L2=(\S+),", p.stdout)
        check(l2["generate_script"] and l2["generate_script"] == l2["generate_clean_script"],
              f"live and export give the same result: {l2}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 68 BC WHERE-NAME TESTS PASSED")
    return 0


def test_round68_bc_where_names():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
