#!/usr/bin/env python3
"""
Round 76 -- integro-differential equations: integral(...) in an ODE.

integral(f) is the integral of f(s, t, outputs at s) for s from t₀ to t,
evaluated by Gauss-Legendre quadrature of the network (as DeepXDE's
dde.data.IDE, which supports only the TensorFlow/Paddle backends). Checks
the rewrite, validation, the GUI field, and that both DeepXDE IDE examples
train to their exact solutions with the live and exported scripts agreeing.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round76_integral.py
"""
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
from pinnstudio.core.codegen import _effective_config, _integral_text, generate_clean_script, generate_script
from pinnstudio.ui.main_window import MainWindow

NAMES = ["Integro-differential equation", "Volterra integro-differential equation"]


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    check(_integral_text("du_x + u - integral(exp(s - x)*(u + 1))", ["u"], 0.0)
          == "du_x + u - _integral(lambda s, u: (exp(s - x)*(u + 1)), x, 0.0)", "rewrite keeps nested parentheses")
    check(_integral_text("integral(a*b)", ["a", "b"], 1.0) == "_integral(lambda s, a, b: (a*b), x, 1.0)",
          "every output is a lambda argument")
    try:
        _integral_text("integral(u", ["u"], 0.0)
        check(False, "unbalanced integral( rejected")
    except ValueError:
        check(True, "unbalanced integral( rejected")

    cfg = dx.build_config(dx.BY_NAME["Volterra integro-differential equation"], "/r")
    check(cfg.validate() == [], "Volterra example validates")
    e = _effective_config(cfg)
    check("_integral(lambda s, u: (exp(s - x)*u), x, 0.0)" in e.pde_expressions, f"mapped: {e.pde_expressions}")
    bad = dx.build_config(dx.BY_NAME["Volterra integro-differential equation"], "/r")
    bad.problem_dim, bad.steady_state = "2D", True
    check(any("integral(" in m for m in bad.validate()), "integral rejected outside ODE/steady 1D")
    bad = dx.build_config(dx.BY_NAME["Volterra integro-differential equation"], "/r")
    bad.output_names = "s"
    check(any("'s'" in m for m in bad.validate()), "an output named s rejected")
    for gen in (generate_script, generate_clean_script):
        src = gen(cfg)
        check("np.polynomial.legendre.leggauss(20)" in src, f"{gen.__name__}: 20-point quadrature")

    w = MainWindow()
    w._apply_deepxde_example("Integro-differential equation")
    w.workspace.select("equations")
    check(w.integral_points_spin.isVisibleTo(w) and w.integral_points_spin.value() == 16, "GUI shows the quadrature points")
    w.integral_points_spin.setValue(24)
    check(w._build_config().integral_points == 24, "GUI -> config")
    w.radio_1d.setChecked(True)
    check(not w.integral_points_spin.isVisibleTo(w), "hidden outside ODE mode")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        for name, it, tol in ((NAMES[0], 2000, 0.05), (NAMES[1], 300, 0.02)):
            res = {}
            for gen in (generate_script, generate_clean_script):
                d = os.path.join(tmp, re.sub(r"\W+", "_", name) + gen.__name__)
                os.makedirs(d)
                c = dx.build_config(dx.BY_NAME[name], os.path.join(_REPO_ROOT, "reference_data"), keep={"save_dir": d})
                c.iterations = min(c.iterations, it)
                import json
                ph = json.loads(c.scheduler_phases) if c.optimizer_scheduler else []
                for p in ph:
                    p["iterations"] = min(int(p["iterations"]), it)
                if ph:
                    c.scheduler_phases = json.dumps(ph)
                with open(os.path.join(d, "run.py"), "w") as fh:
                    fh.write(gen(c))
                p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900)
                check(p.returncode == 0, f"{name} {gen.__name__} ran:\n{p.stderr[-1500:]}")
                res[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
            check(res["generate_script"] and res["generate_script"][0] < tol
                  and res["generate_script"] == res["generate_clean_script"],
                  f"{name}: trains to the exact solution, live and export agree {res}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 76 INTEGRAL TESTS PASSED")
    return 0


def test_round76_integral():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
