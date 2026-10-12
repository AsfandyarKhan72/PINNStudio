#!/usr/bin/env python3
"""
Round 71 -- Point Set / Point Set Operator BCs without a file.

Points can come from a file (as before), a typed list ("1", "0.2, 0.5",
"0,0; 1,0.5") or be generated (N random points on the part of the boundary
picked by Where, or in the domain); target values from the file or a
formula in x, y, z, t ('|' between components). Both generators; the GUI
row round-trips; the DeepXDE PointSetOperator example needs no file.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round71_pointset_points.py
"""
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
from pinnstudio.core.codegen import _clean_bc_row_code, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _cfg(rows, d="."):
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                     num_initial=0, pde_expressions="du_xx - 2", layers=[1, 30, 30, 1], iterations=1500,
                     optimizer2="none", iterations2=0, optimizer_scheduler=False, num_domain=16, num_boundary=2,
                     num_test=50, custom_bc_json=json.dumps(rows), ea_ref_mode="formula", ea_ref_formula="(x + 1)**2",
                     save_dir=d, plot_resolution=40).items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


LEFT = {"type": "dirichlet", "location": "left", "value": "0", "component": 0}
PSO = {"type": "pointset_operator", "location": "True", "value": "du_x", "component": 0,
       "points_mode": "list", "points_list": "1", "values_mode": "formula", "values_formula": "4"}
GEN = {"type": "pointset", "location": "True", "value": "0", "component": 0, "points_mode": "generate",
       "points_count": 12, "points_where": "domain", "values_mode": "formula", "values_formula": "(x + 1)**2"}


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    d, a = _clean_bc_row_code(1, PSO, False, False, 1, True)
    src = "\n".join(d + a)
    check("_pts1 = np.array([[1.0]], dtype=float)" in src and "_pvals1 = np.hstack([_bc1_pval0(_pts1)])" in src,
          "export: typed point and formula values")
    ns = {"np": np}
    exec("\n".join(d), ns)
    check(np.allclose(ns["_bc1_pval0"](np.array([[1.0]])), 4.0), "formula value evaluates")
    d, a = _clean_bc_row_code(2, dict(GEN, points_where="boundary", location="right"), False, False, 1, True)
    src = "\n".join(d + a)
    check("geomtime.random_boundary_points" in src and "_bc2_ploc(p, True)" in src, "export: generated on the boundary, filtered by Where")
    d, a = _clean_bc_row_code(3, {"type": "pointset", "component": 0, "points_mode": "list",
                                  "points_list": "0,0; 1,0.5", "values_mode": "formula", "values_formula": "x + y"},
                              True, False, 2, True)
    check("np.array([[0.0, 0.0], [1.0, 0.5]], dtype=float)" in "\n".join(a), "2D typed points")
    try:
        _clean_bc_row_code(4, {"type": "pointset", "points_mode": "list", "points_list": "1; 0.5", "values_mode": "formula",
                               "values_formula": "0"}, True, False, 2, True)
        check(False, "a point with the wrong number of coordinates is rejected")
    except ValueError as e:
        check("coordinate" in str(e), f"bad typed point rejected: {e}")
    live = generate_script(_cfg([LEFT, PSO]))
    check("def _bc_points(" in live and "_bc_points(_be, _blabel, 1, geomtime)" in live, "live: rows go through _bc_points")

    ex = dx.BY_NAME["Poisson 1D – Dirichlet/PointSetOperator"]
    rows = json.loads(ex.settings["custom_bc_json"])
    check(rows[1].get("points_list") == "1" and "points_file" not in rows[1], "DeepXDE example: no file")

    w = MainWindow()
    e = w._add_custom_bc_entry(bc_type="pointset_operator", value="du_x", points_mode="list", points_list="1",
                               values_mode="formula", values_formula="4")
    check(e["points_list"].isVisibleTo(e["widget"]) and not e["points_file"].isVisibleTo(e["widget"]),
          "list mode shows the list field, hides the file field")
    row = json.loads(w._build_custom_bc_json())[-1]
    check((row["points_mode"], row["points_list"], row["values_mode"], row["values_formula"]) == ("list", "1", "formula", "4"),
          f"row saved: {row}")
    w._apply_custom_bc_json(json.dumps([row]))
    e2 = w.custom_bc_list[-1]
    check(e2["points_mode"].currentData() == "list" and e2["values_formula"].text() == "4", "row restored")
    e2["points_mode"].setCurrentIndex(e2["points_mode"].findData("generate"))
    check(e2["values_mode"].currentData() == "formula", "non-file points force formula values")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        for rows, tag in (([LEFT, PSO], "pso"), ([LEFT, GEN], "gen")):
            res = {}
            for gen in (generate_script, generate_clean_script):
                dd = os.path.join(tmp, tag + gen.__name__)
                os.makedirs(dd)
                with open(os.path.join(dd, "run.py"), "w") as fh:
                    fh.write(gen(_cfg(rows, dd)))
                p = subprocess.run([sys.executable, "run.py"], cwd=dd, capture_output=True, text=True, timeout=600)
                check(p.returncode == 0, f"{tag} {gen.__name__} ran:\n{p.stderr[-1500:]}")
                res[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
            check(res["generate_script"] and res["generate_script"][0] < 0.05, f"{tag}: trains to the exact solution {res}")
            if tag == "pso":
                check(res["generate_script"] == res["generate_clean_script"], f"{tag}: live and export agree {res}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 71 POINT-SET TESTS PASSED")
    return 0


def test_round71_pointset_points():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
