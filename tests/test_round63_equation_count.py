#!/usr/bin/env python3
"""
Round 63 -- the number of equations is independent of the number of
outputs, as in DeepXDE (its American-put example has three residuals for
one output; its unknown-field inverse problems have one for two outputs).

  - PDE panel: "Outputs" (now up to 10, was 4 -- DeepXDE's elasticity
    example needs 5) and a new "Equations" count, which follows Outputs
    unless changed. PINNConfig.num_equations (0 = one per output, as every
    config saved before this round).
  - Every place that assumed one equation per output now reads
    codegen._n_equations: both generators' PDE function and loss weights,
    the Loss Weights rows (plain and per-phase), the saved weight strings,
    the sweep registry's weight slots and RAR's "Points from:" list.
  - relu/sigmoid are available in PDE expressions.
  - Fix found here: the exported script always expected a time column in
    a point-set data file, so a steady problem's measured data couldn't be
    loaded (the live script was right).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round63_equation_count.py
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

from pinnstudio.core.codegen import _n_equations, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.sweep_registry import _weight_slot_descriptors
from pinnstudio.ui.main_window import MainWindow


def _field(d, it, obs):
    """Two outputs (u, q), one equation, measured u -- DeepXDE's elliptic inverse-field shape."""
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                     num_outputs=2, output_names="u,q", num_equations=1, pde_expressions="-du_xx + q",
                     custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "sin(pi*x)", "component": 0},
                                                {"type": "pointset", "points_file": obs, "component": 0}]),
                     loss_weights_multi="1,100,1000", network_type="PFNN",
                     num_domain=100, num_boundary=2, num_initial=0, num_test=50, layers=[1, 16, 16, 2],
                     iterations=it, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                     save_dir=d, ea_files="[]").items():
        setattr(c, k, v)
    c.pde_expression = "-du_xx + q"
    return c


def _multi(d, it):
    """One output, two equations (one a relu inequality penalty)."""
    c = PINNConfig()
    for k, v in dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                     num_outputs=1, output_names="u", num_equations=2,
                     pde_expressions="-du_xx - pi**2*sin(pi*x)|relu(-2 - u)",
                     custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "sin(pi*x)", "component": 0}]),
                     num_domain=32, num_boundary=2, num_initial=0, num_test=50, layers=[1, 16, 16, 1],
                     iterations=it, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                     save_dir=d, ea_files="[]").items():
        setattr(c, k, v)
    c.pde_expression = "-du_xx - pi**2*sin(pi*x)"
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── GUI ───────────────────────────────────────────────────────────────
    win = MainWindow()
    check(win.num_outputs_spin.maximum() >= 5, "outputs should allow at least 5 (DeepXDE elasticity)")
    win.num_equations_spin.setValue(3)
    _app.processEvents()
    check(len(win.pde_inputs) == 3 and len(win.output_name_inputs) == 1, "3 PDE rows for 1 output")
    for i, t in enumerate(["-du_xx", "relu(u)", "u - 1"]):
        win.pde_inputs[i].setText(t)
    check(sorted(k for k in win.weight_widgets if k.startswith("pde")) == ["pde_0", "pde_1", "pde_2"], "one weight per equation")
    check(win.rar_output_combo.count() == 4 and win.rar_output_combo.itemText(3) == "Equation 3", "RAR lists equations")
    c = win._build_config()
    check(c.num_equations == 3 and c.pde_expressions == "-du_xx|relu(u)|u - 1", f"saved: {c.num_equations} {c.pde_expressions!r}")
    check(len(c.loss_weights_multi.split(",")) >= 3, "weights string carries 3 PDE weights")
    win2 = MainWindow()
    win2._apply_config(c)
    c2 = win2._build_config()
    check((c2.num_equations, c2.pde_expressions, c2.loss_weights_multi) == (c.num_equations, c.pde_expressions, c.loss_weights_multi),
          "equations survive save/load")
    win2.num_outputs_spin.setValue(2)
    _app.processEvents()
    check(win2.num_equations_spin.value() == 2 and win2._build_config().num_equations == 0,
          "changing Outputs resets Equations to match (saved as 0 = one per output)")

    # ── codegen / registry ────────────────────────────────────────────────
    check(_n_equations(PINNConfig()) == 1, "default: one equation per output")
    m = _multi("/tmp/x", 10)
    check(_n_equations(m) == 2, "num_equations wins over num_outputs")
    check("_n_eq = 2" in generate_script(m), "live PDE function uses the equation count")
    check(len([s for s in _weight_slot_descriptors(m) if s["key"].startswith("pde_")]) == 2, "sweep: one PDE slot per equation")
    for gen in (generate_script, generate_clean_script):
        for cfg in (m, _field("/tmp/x", 10, "/tmp/obs.txt")):
            try:
                ast.parse(gen(cfg))
            except SyntaxError as e:
                check(False, f"{gen.__name__}: script should parse: {e}")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── train: equations != outputs, both generators, same losses ─────────
    with tempfile.TemporaryDirectory() as tmp:
        obs = os.path.join(tmp, "obs.txt")
        xs = np.linspace(-1, 1, 40)
        np.savetxt(obs, np.column_stack([xs, np.sin(np.pi * xs)]))
        for tag, make, n_terms in (("field", lambda d: _field(d, 200, obs), 3), ("multi", lambda d: _multi(d, 200), 3)):
            losses = {}
            for gen in (generate_script, generate_clean_script):
                d = os.path.join(tmp, f"{tag}_{gen.__name__}")
                os.makedirs(d)
                with open(os.path.join(d, "run.py"), "w") as f:
                    f.write(gen(make(d)))
                p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
                check(p.returncode == 0, f"{tag} via {gen.__name__} failed:\n{p.stdout[-600:]}\n{p.stderr[-1200:]}")
                rows = [l for l in p.stdout.splitlines() if re.match(r"^\s*200\s+\[", l)]
                losses[gen.__name__] = rows[-1].split("]")[0] if rows else None
                if rows:
                    check(rows[-1].split("]")[0].count(",") == n_terms - 1, f"{tag}: expected {n_terms} loss terms: {rows[-1]}")
            check(losses["generate_script"] == losses["generate_clean_script"],
                  f"{tag}: live and export should train identically: {losses}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 63 EQUATION-COUNT TESTS PASSED")
    return 0


def test_round63_equation_count():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
