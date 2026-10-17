#!/usr/bin/env python3
"""
Round 66 -- RAR can stop on a residual tolerance, as DeepXDE's Burgers-RAR
example does (it keeps adding points until the mean residual is below
0.005). "Stop when mean residual <" in the RAR settings (0 = off: every
round runs, as before; otherwise "RAR training rounds" is the cap). The
RAR limits were also widened for that recipe: one point per round, up to
1000 rounds. Shared by both generators via codegen._rar_tolerance_check;
nothing changes in a script when the tolerance is off.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round66_rar_tolerance.py
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _rar(d, tol):
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                     pde_expressions="-du_xx - pi**2*sin(pi*x)",
                     custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "sin(pi*x)", "component": 0}]),
                     num_domain=16, num_boundary=2, num_initial=0, num_test=50, layers=[1, 16, 16, 1],
                     iterations=50, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                     adapt_method="RAR", rar_cycles=3, rar_candidates=1000, rar_add_points=1,
                     rar_adam_iters=20, rar_lbfgs_iters=0, rar_tolerance=tol,
                     save_dir=d, ea_files="[]").items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    for gen in (generate_script, generate_clean_script):
        on, off = gen(_rar("/tmp/x", 10.0)), gen(_rar("/tmp/x", 0.0))
        ast.parse(on)
        check("is below the tolerance 10.0" in on and "below the tolerance" not in off,
              f"{gen.__name__}: tolerance check only when set")

    win = MainWindow()
    check(win.rar_cycles.maximum() >= 1000 and win.rar_add_points.minimum() == 1, "RAR limits widened")
    win.rar_tolerance.setValue(0.005)
    c = win._build_config()
    win2 = MainWindow()
    win2._apply_config(c)
    check(abs(win2._build_config().rar_tolerance - 0.005) < 1e-12, "tolerance survives save/load")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    with tempfile.TemporaryDirectory() as tmp:
        for gen in (generate_script, generate_clean_script):
            for tol, stops in ((10.0, True), (0.0, False)):
                d = os.path.join(tmp, f"{gen.__name__}_{tol}")
                os.makedirs(d)
                with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                    fh.write(gen(_rar(d, tol)))
                p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
                check(p.returncode == 0, f"{gen.__name__} tol={tol} failed:\n{p.stderr[-1200:]}")
                check(("refinement done" in p.stdout) == stops,
                      f"{gen.__name__} tol={tol}: early stop expected={stops}\n{p.stdout[-800:]}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 66 RAR-TOLERANCE TESTS PASSED")
    return 0


def test_round66_rar_tolerance():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
