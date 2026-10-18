#!/usr/bin/env python3
"""
Round 64 -- two DeepXDE features from its Klein-Gordon example, plus an
export parity fix found on the way.

  - Operator BC "Applies on: the initial time (t = t0)": a derivative
    condition at t = t_min, e.g. a wave-type equation's second initial
    condition du_t = 0. (DeepXDE's own Klein-Gordon example passes
    `lambda _, on_initial: on_initial` to an OperatorBC, but OperatorBC
    filters with the *spatial* boundary test, so there it actually lands on
    x = +-1, where the exact solution's du_t = -x*sin(t) is not 0. This
    implements what the example intends.) Rejected for a steady problem
    and for Time-Adaptive training, with a clear message.
  - "Inverse time" learning-rate decay in the Optimizer Scheduler
    (DeepXDE's ("inverse time", decay_steps, decay_rate)).
  - Fix: the exported script's boundary-condition functions had no `t`,
    so a time-dependent value like x*cos(t) was a NameError there (the
    live script supported it).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round64_initial_operator_decay.py
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

from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow

_BCS = [{"type": "dirichlet", "location": "True", "value": "x*cos(t)", "component": 0},
        {"type": "operator", "location": "True", "value": "du_t", "apply_at": "initial"}]


def _kg(d, it, **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=False, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                t_min=0.0, t_max=10.0,
                pde_expressions="du_tt - du_xx + u**2 + x*cos(t) - x**2*cos(t)**2", ic_expressions="x",
                custom_bc_json=json.dumps(_BCS), num_domain=300, num_boundary=40, num_initial=40, num_test=100,
                layers=[2, 20, 20, 1], iterations=it, optimizer2="none", iterations2=0,
                optimizer_scheduler=True,
                scheduler_phases=json.dumps([{"optimizer": "adam", "iterations": it, "lr": 1e-3,
                                              "decay_type": "inverse time", "decay_p1": 3000, "decay_p2": 0.9,
                                              "weights": "1,1,1,1"}]),
                save_dir=d, ea_files="[]", plot_resolution=20)
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

    # ── preconditions ─────────────────────────────────────────────────────
    for gen in (generate_script, generate_clean_script):
        for kw, why in ((dict(steady_state=True), "steady"),
                        (dict(time_adaptive=True, adapt_method="Time Adaptive Training"), "Time-Adaptive")):
            try:
                gen(_kg("/tmp/x", 10, **kw))
                check(False, f"{gen.__name__}: initial-time Operator BC with {why} should be rejected")
            except ValueError as e:
                check("initial time" in str(e), f"{gen.__name__}: clear message expected, got {e}")

    # ── generated code ────────────────────────────────────────────────────
    live, export = generate_script(_kg("/tmp/x", 10)), generate_clean_script(_kg("/tmp/x", 10))
    for name, s in (("live", live), ("export", export)):
        ast.parse(s)
        check("'inverse time'" in s, f"{name}: inverse-time decay should reach the compile call")
    ns = {"np": np}
    m = re.search(r"(def _bc1_loc\(X, on_boundary\):\n(?:    .*\n)+)", export)
    check(m is not None, "export should define the initial-time location function")
    if m:
        exec(m.group(1), ns)
        f = ns["_bc1_loc"]
        check(f(np.array([0.3, 0.0]), False) and not f(np.array([0.3, 0.5]), False) and not f(np.array([1.0, 2.0]), True),
              "initial-time location: t = t_min only, independent of the spatial boundary flag")
    check("    t = X[:, -1]" in export, "export: a BC value using t unpacks it")

    # ── GUI round trip ────────────────────────────────────────────────────
    win = MainWindow()
    win._apply_custom_bc_json(json.dumps(_BCS))
    saved = json.loads(win._build_custom_bc_json())
    check([e.get("apply_at") for e in saved] == ["boundary", "initial"], f"apply_at should round-trip: {saved}")
    check(win.custom_bc_list[1]['apply_at'].isVisibleTo(win.custom_bc_list[1]['widget']) and
          not win.custom_bc_list[0]['apply_at'].isVisibleTo(win.custom_bc_list[0]['widget']),
          "'Applies on' shown for Operator rows only")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── Klein-Gordon trains identically through both generators ──────────
    with tempfile.TemporaryDirectory() as tmp:
        rows = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as fh:
                fh.write(gen(_kg(d, 200)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} failed:\n{p.stdout[-600:]}\n{p.stderr[-1200:]}")
            r = [l for l in p.stdout.splitlines() if re.match(r"^\s*200\s+\[", l)]
            rows[gen.__name__] = r[-1].split("]")[0] if r else None
        check(rows["generate_script"] is not None and rows["generate_script"].count(",") == 3,
              f"4 loss terms expected (PDE, boundary, initial-time operator, IC): {rows}")
        check(rows["generate_script"] == rows["generate_clean_script"], f"live and export should match: {rows}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 64 INITIAL-OPERATOR / DECAY TESTS PASSED")
    return 0


def test_round64_initial_operator_decay():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
