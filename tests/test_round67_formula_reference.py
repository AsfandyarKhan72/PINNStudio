#!/usr/bin/env python3
"""
Round 67 -- Error Analysis can use an exact-solution formula instead of
reference files (most DeepXDE examples have one).

The Error Analysis dialog gets "Reference: Reference files | Exact solution
formula": one formula per output in x, y, z, t (numpy math names plus erf
and norm_cdf -- Black-Scholes needs the normal CDF), comparison times
(blank = 5 evenly spaced) and a grid size. PINNConfig ea_ref_mode /
ea_ref_formula / ea_ref_times / ea_ref_grid.

The generated scripts write the formula's values as ordinary reference
files (<save dir>/reference_from_formula/, on a grid masked to the
geometry, in the format Error Analysis already reads) and list them as
ea_files (codegen._effective_config), so the comparison runs through the
unchanged file-based Error Analysis in both generators. Restore & Visualize
can use the files written there.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round67_formula_reference.py
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

from PyQt6.QtWidgets import QApplication, QDialog, QPushButton

_app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

from pinnstudio.core.codegen import (_effective_config, _formula_reference_entries, generate_clean_script,
                                     generate_script)
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _heat(d, it, **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=False, geometry_type="Interval", x_min=0.0, x_max=1.0,
                t_min=0.0, t_max=1.0, pde_expressions="du_t - 0.4*du_xx", ic_expressions="sin(pi*x)",
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}]),
                num_domain=300, num_boundary=40, num_initial=40, num_test=100, layers=[2, 20, 20, 1],
                iterations=it, optimizer2="none", iterations2=0, optimizer_scheduler=False, save_dir=d,
                ea_ref_mode="formula", ea_ref_formula="exp(-0.4*pi**2*t)*sin(pi*x)", ea_ref_times="0.5, 1",
                plot_resolution=30)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    c.ic_expression = c.ic_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── entries / effective config ───────────────────────────────────────
    ents = _formula_reference_entries(_heat("/runs/a", 10))
    check([(t, os.path.basename(p), i) for (t, p, i, _e) in ents] == [(0.5, "u_t_0.5.txt", 0), (1.0, "u_t_1.txt", 0)], ents)
    eff = _effective_config(_heat("/runs/a", 10))
    check(ast.literal_eval(eff.ea_files) == [(0.5, "/runs/a/reference_from_formula/u_t_0.5.txt", 0),
                                              (1.0, "/runs/a/reference_from_formula/u_t_1.txt", 0)], eff.ea_files)
    check(len(_formula_reference_entries(_heat("/r", 10, ea_ref_times=""))) == 5, "blank times -> 5 evenly spaced")
    check(_formula_reference_entries(_heat("/r", 10, ea_ref_mode="files")) == [], "files mode -> no formula files")
    for gen in (generate_script, generate_clean_script):
        s = gen(_heat("/tmp/x", 10))
        ast.parse(s)
        check("_write_formula_reference()" in s, f"{gen.__name__}: writes the reference files")

    # ── dialog + save/load ────────────────────────────────────────────────
    win = MainWindow()
    real_exec = QDialog.exec

    def fake_exec(dlg):
        win._set_combo_data(win._ea_ref_mode_combo, "formula")
        win._ea_formula_edits[0].setText("x*cos(t)")
        win._ea_ref_times_edit.setText("0, 1")
        for b in dlg.findChildren(QPushButton):
            if "Run Error Analysis" in b.text():
                b.click()
                break
        return 1

    QDialog.exec = fake_exec
    try:
        win.save_dir_input.setText(tempfile.gettempdir())
        win._on_error_analysis_btn()
    finally:
        QDialog.exec = real_exec
    c1 = win._build_config()
    check((c1.ea_ref_mode, c1.ea_ref_formula, c1.ea_ref_times) == ("formula", "x*cos(t)", "0, 1"),
          f"dialog -> config: {(c1.ea_ref_mode, c1.ea_ref_formula, c1.ea_ref_times)}")
    win2 = MainWindow()
    win2._apply_config(c1)
    c2 = win2._build_config()
    check((c2.ea_ref_mode, c2.ea_ref_formula, c2.ea_ref_times) == (c1.ea_ref_mode, c1.ea_ref_formula, c1.ea_ref_times),
          "formula reference survives save/load")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    with tempfile.TemporaryDirectory() as tmp:
        # 1D heat: both generators run Error Analysis against the formula, identically
        metrics = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                fh.write(gen(_heat(d, 300)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} failed:\n{p.stderr[-1500:]}")
            metrics[gen.__name__] = re.findall(r"t=1\.0000\s+\S+\s+L2=(\S+),", p.stdout)
            check(os.path.isfile(os.path.join(d, "reference_from_formula", "u_t_0.5.txt")), "reference file written")
            check(os.path.isfile(os.path.join(d, "error_analysis", "error_metrics.txt")), "Error Analysis ran")
        check(metrics["generate_script"] and metrics["generate_script"] == metrics["generate_clean_script"],
              f"live and export should report the same L2: {metrics}")

        # 2D steady disk: the reference grid is masked to the geometry
        d = os.path.join(tmp, "disk")
        os.makedirs(d)
        c = _heat(d, 30, problem_dim="2D", steady_state=True, geometry_type="Disk", geom_center_x=0.0,
                  geom_center_y=0.0, geom_radius=1.0, x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0,
                  pde_expressions="-du_xx - du_yy - 1", ea_ref_formula="(1 - x**2 - y**2)/4", ea_ref_grid=41,
                  layers=[2, 16, 1])
        c.pde_expression = c.pde_expressions
        with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
            fh.write(generate_script(c))
        p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
        check(p.returncode == 0, f"disk failed:\n{p.stderr[-1500:]}")
        f = os.path.join(d, "reference_from_formula", "u.txt")
        if os.path.isfile(f):
            ref = np.loadtxt(f)
            check(ref.shape[1] == 3 and len(ref) < 41 * 41 and np.all(ref[:, 0] ** 2 + ref[:, 1] ** 2 <= 1 + 1e-9),
                  f"steady 2D reference: x, y, u columns, inside the disk only: {ref.shape}")
            check(np.allclose(ref[:, 2], (1 - ref[:, 0] ** 2 - ref[:, 1] ** 2) / 4), "values follow the formula")
        else:
            check(False, "steady 2D reference file missing")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 67 FORMULA-REFERENCE TESTS PASSED")
    return 0


def test_round67_formula_reference():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
