#!/usr/bin/env python3
"""
Round 74 -- ODE problems (Problem dimension "ODE (t)").

An ODE problem has t as its only variable: derivatives du_t, du_tt, initial
conditions at t₀, BC rows at "start"/"end". Both generators solve it as a
1D steady problem in x = t (initial conditions become Dirichlet rows at the
start). Checks: the expression mapping, the mapped config, validation, the
GUI radio round-trip, the five DeepXDE ODE examples build, and both
generators train the ODE system to the same result.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round74_ode.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import _effective_config, _ode_text, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

ODE_NAMES = ["ODE system", "Lotka-Volterra equation", "Second-order ODE", "Inverse: Lorenz system",
             "Inverse: Lorenz system with exogenous input"]


def _cfg(d=".", **kw):
    c = PINNConfig()
    base = dict(problem_dim="ODE", steady_state=False, geometry_type="Interval", t_min=0.0, t_max=5.0,
                num_outputs=2, output_names="y1,y2", pde_expressions="dy1_t - y2|dy2_t + y1",
                ic_expressions="0|1", ic_active="True,True", custom_bc_json="[]", layers=[1, 32, 32, 2],
                iterations=3000, optimizer2="none", iterations2=0, optimizer_scheduler=False, num_domain=40,
                num_boundary=2, num_test=60, ea_ref_mode="formula", ea_ref_formula="sin(t)|cos(t)",
                save_dir=d, plot_resolution=60)
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions.split("|")[0]
    c.ic_expression = c.ic_expressions.split("|")[0]
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    check(_ode_text("dy1_tt + t*dy1_t - sin(t_max - t)") == "dy1_xx + x*dy1_x - sin(x_max - x)", "derivatives and t map to x")
    check(_ode_text("start") == "left" and _ode_text("end") == "right", "start/end map to left/right")
    check(_ode_text("theta + t2") == "theta + t2", "names containing t are left alone")

    c = _cfg()
    check(c.validate() == [], f"an ODE config validates {c.validate()}")
    check(any("x" in e for e in _cfg(output_names="x,y2").validate()), "an output named x is rejected")
    e = _effective_config(c)
    check((e.problem_dim, e.steady_state, e.x_min, e.x_max) == ("1D", True, 0.0, 5.0), "solved as 1D steady in x = t")
    rows = json.loads(e.custom_bc_json)
    check([(r["location"], r["value"], r["component"]) for r in rows] == [("left", "0", 0), ("left", "1", 1)],
          f"initial conditions become Dirichlet rows at the start: {rows}")
    check(e.pde_expressions == "dy1_x - y2|dy2_x + y1" and e.ea_ref_formula == "sin(x)|cos(x)", "expressions mapped")
    op = _cfg(custom_bc_json=json.dumps([{"type": "operator", "value": "dy1_t - 1", "apply_at": "initial",
                                          "location": "True", "component": 0}]))
    r0 = json.loads(_effective_config(op).custom_bc_json)[0]
    check((r0["location"], r0["value"]) == ("left", "dy1_x - 1"), f"an operator at the initial time -> at the start: {r0}")

    w = MainWindow()
    w.radio_ode.setChecked(True)
    check(not w.radio_stationary.isEnabled() and not w.x_row_widget.isVisibleTo(w), "ODE mode: time-dependent, no x row")
    cg = w._build_config()
    check(cg.problem_dim == "ODE" and not cg.steady_state, "GUI -> config")
    w.radio_1d.setChecked(True)
    w._apply_config(cg)
    check(w.radio_ode.isChecked(), "config -> GUI")
    w.radio_1d.setChecked(True)
    check(w.radio_stationary.isEnabled(), "leaving ODE mode re-enables Stationary")

    for n in ODE_NAMES:
        ex = dx.build_config(dx.BY_NAME[n], "/r")
        check(ex.problem_dim == "ODE" and ex.validate() == [], f"DeepXDE example builds and validates: {n}")
        check(bool(generate_script(ex)) and bool(generate_clean_script(ex)), f"both generators write {n}")

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
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(_cfg(d)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            l2[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
        check(l2["generate_script"] and max(l2["generate_script"]) < 0.05
              and l2["generate_script"] == l2["generate_clean_script"], f"ODE system trains, live and export agree: {l2}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 74 ODE TESTS PASSED")
    return 0


def test_round74_ode():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
