#!/usr/bin/env python3
"""
Round 70 -- named constants ("Re = 20", "nu = 1/Re", ...) in the PDE panel,
usable by name in every expression of a problem.

 - core/constants.py: evaluation in order (math functions, earlier
   constants), clear errors for bad names, clashes and bad expressions;
   substitution replaces whole names only.
 - Both generators substitute the values (codegen._effective_config) into
   the PDE, ICs, BC Where/Value, transforms, the exact-solution reference,
   custom plot fields and monitors; the exported script lists the
   constants at its top. PINNConfig.validate reports invalid constants.
 - GUI: Constants rows with the computed value shown, saved and restored.
 - DeepXDE examples use the constants their scripts define.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round70_constants.py
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

from pinnstudio.core import constants as cm
from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core.codegen import _effective_config, generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _consts(*pairs):
    return json.dumps([{"name": n, "value": v} for n, v in pairs])


def _cfg(d=".", **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0, num_initial=0,
                constants=_consts(("k", "pi"), ("A", "2")),
                pde_expressions="-du_xx - A*k**2*sin(k*x)",
                custom_bc_json=json.dumps([{"type": "dirichlet", "location": "True", "value": "A*sin(k*x)", "component": 0}]),
                layers=[1, 20, 20, 1], iterations=300, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                num_domain=32, num_boundary=2, num_test=50, ea_ref_mode="formula", ea_ref_formula="A*sin(k*x)",
                save_dir=d, plot_resolution=40)
    base.update(kw)
    for key, v in base.items():
        setattr(c, key, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    vals = cm.evaluate(_consts(("Re", "20"), ("nu", "1/Re"), ("l", "1/(2*nu) - sqrt(1/(4*nu**2) + 4*pi**2)")))
    check([n for n, _e, _v in vals] == ["Re", "nu", "l"] and abs(vals[2][2] + 1.8100981200139667) < 1e-12,
          "constants evaluate in order, using earlier ones and math functions")
    for bad, why in ((_consts(("x", "1")), "already used"), (_consts(("2a", "1")), "not a valid name"),
                     (_consts(("a", "b + 1")), "can't evaluate"), (_consts(("a", "1"), ("a", "2")), "twice"),
                     (_consts(("u", "1")), "already used"), (_consts(("left", "1")), "already used")):
        try:
            cm.evaluate(bad)
            check(False, f"{bad} should be rejected")
        except ValueError as e:
            check(why in str(e), f"rejected with a clear message: {e}")
    try:
        cm.evaluate(_consts(("v", "1")), other_names=["u", "v"])
        check(False, "a constant can't share an output's name")
    except ValueError:
        check(True, "a constant can't share an output's name")
    sub = cm.substitute("du_Re + Re*np.e + 1e-3*Re2 + Re", [("Re", "20", 20.0)])
    check(sub == "du_Re + (20.0)*np.e + 1e-3*Re2 + (20.0)", f"whole names only: {sub}")

    c = _cfg()
    eff = _effective_config(c)
    check("k" not in re.findall(r"[A-Za-z_]\w*", eff.pde_expressions.replace("sin", "")) and "3.14159" in eff.pde_expressions,
          f"PDE gets the values: {eff.pde_expressions}")
    check("A" not in json.loads(eff.custom_bc_json)[0]["value"], "BC value gets the values")
    check(eff.constants == "", "constants consumed")
    exp = generate_clean_script(c)
    check("# Constants (their values are substituted" in exp and "#   k = pi  = 3.14159" in exp, "export lists them")
    check(not generate_script(c) is None, "live script generates")
    check(_cfg(constants=_consts(("k", "nope"))).validate() != [], "validate reports a bad constant")
    check(_cfg().validate() == [], "a good config validates")

    w = MainWindow()
    w._add_constant_row("Re", "20")
    e = w._add_constant_row("nu", "1/Re")
    check(e["result"].text() == "= 0.05", f"computed value shown: {e['result'].text()!r}")
    bad = w._add_constant_row("x", "1")
    check(bad["result"].text() == "⚠", "a bad row is flagged")
    bad["name"].setText("Pe")
    cfg = w._build_config()
    check(json.loads(cfg.constants) == [{"name": "Re", "value": "20"}, {"name": "nu", "value": "1/Re"},
                                         {"name": "Pe", "value": "1"}], f"saved: {cfg.constants}")
    w._set_constants("")
    w._apply_config(cfg)
    check([r["name"].text() for r in w.constant_rows] == ["Re", "nu", "Pe"], "restored")

    kov = dx.build_config(dx.BY_NAME["Kovasznay flow"], "/r")
    check("Re" in kov.constants and "1/Re" in kov.pde_expressions, "Kovasznay uses Re, nu, l")

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
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-1500:]}")
            l2[gen.__name__] = re.findall(r"L2=(\S+),", p.stdout)
        check(l2["generate_script"] and l2["generate_script"] == l2["generate_clean_script"],
              f"live and export agree: {l2}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 70 CONSTANTS TESTS PASSED")
    return 0


def test_round70_constants():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
