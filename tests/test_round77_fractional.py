#!/usr/bin/env python3
"""
Round 77 -- fractional equations: frac(u) in an equation.

frac(u) is DeepXDE's discretized fractional operator (dde.data.FPDE /
TimeFPDE, Grunwald-Letnikov) applied to output u. DeepXDE's own fractional
scripts support only TensorFlow/Paddle; PINNStudio applies the FPDE matrix
with PyTorch. Checks: the data call and residual in both generators,
validation, numbers folded inside gamma(...), the GUI fields, the six
DeepXDE examples, and end-to-end training (1D forward and inverse alpha).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round77_fractional.py
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
from pinnstudio.core.codegen import _simplify_pde_expr, generate_clean_script, generate_script
from pinnstudio.ui.main_window import MainWindow

NAMES = ["Fractional Poisson equation 1D", "Fractional Poisson equation 2D", "Fractional Poisson equation 3D",
         "Fractional diffusion equation", "Inverse: fractional Poisson equation 1D",
         "Inverse: fractional Poisson equation 2D"]


def _short(name, d, iters):
    c = dx.build_config(dx.BY_NAME[name], os.path.join(_REPO_ROOT, "reference_data"), keep={"save_dir": d})
    c.iterations = iters
    if c.optimizer_scheduler:
        ph = json.loads(c.scheduler_phases)
        for p in ph:
            p["iterations"] = min(int(p["iterations"]), iters)
        c.scheduler_phases = json.dumps(ph)
    return c


def run():
    failures = []

    def check(cond, msg):
        print(("ok: " if cond else "FAIL: ") + msg)
        if not cond:
            failures.append(msg)

    check(_simplify_pde_expr("gamma((1 - (1.8))/2)*u") == "(-3.7229806220320425)*u", "gamma of a number folded")
    check(_simplify_pde_expr("lgamma((1 - alpha)/2)") == "torch.lgamma((1 - alpha)/2)", "lgamma of a variable -> torch")

    for n in NAMES:
        c = dx.build_config(dx.BY_NAME[n], "/r")
        check(c.validate() == [], f"validates: {n} {c.validate()}")
    c1 = dx.build_config(dx.BY_NAME[NAMES[0]], "/r")
    live, exp = generate_script(c1), generate_clean_script(c1)
    for name, src in (("live", live), ("export", exp)):
        check("dde.data.FPDE(" in src and "[101], meshtype='static'" in src and "num_test=None" in src,
              f"{name}: FPDE data, static mesh, no test points")
        check("def _frac_apply(" in src, f"{name}: fractional helper")
    check("def pde(x, y, int_mat=None):" in exp and "_res[:_nr]" in exp, "export residual cut to the FPDE rows")
    cd = dx.build_config(dx.BY_NAME["Fractional diffusion equation"], "/r")
    check("dde.data.TimeFPDE(" in generate_script(cd) and "dde.data.TimeFPDE(" in generate_clean_script(cd),
          "time-dependent -> TimeFPDE")
    ci = dx.build_config(dx.BY_NAME[NAMES[4]], "/r")
    check(re.search(r"FPDE\(\s*geomtime, pde, alpha, ", generate_clean_script(ci)) is not None, "inverse: alpha is the trainable variable")

    bad = dx.build_config(dx.BY_NAME[NAMES[1]], "/r")
    bad.fractional_resolution = "100"
    check(any("resolution" in e for e in bad.validate()), "2D needs two resolution numbers")
    bad = dx.build_config(dx.BY_NAME[NAMES[1]], "/r")
    bad.fractional_meshtype = "static"
    check(any("static" in e for e in bad.validate()), "static mesh only on a 1D interval")
    bad = dx.build_config(dx.BY_NAME[NAMES[0]], "/r")
    bad.fractional_alpha = "beta"
    check(any("alpha" in e for e in bad.validate()), "unknown alpha name rejected")
    bad = dx.build_config(dx.BY_NAME[NAMES[0]], "/r")
    bad.adapt_method = "RAR"
    check(any("RAR" in e for e in bad.validate()), "RAR rejected")

    w = MainWindow()
    w._apply_deepxde_example(NAMES[1])
    w.workspace.select("equations")
    check(w.frac_widget.isVisibleTo(w) and w.frac_res_edit.text() == "8, 100", "GUI shows the fractional settings")
    g = w._build_config()
    check((g.fractional_alpha, g.fractional_resolution, g.fractional_meshtype) == ("alpha", "8, 100", "dynamic"),
          "GUI -> config")
    w.pde_inputs[0].setText("-du_xx - du_yy - 1")
    check(not w.frac_widget.isVisibleTo(w), "hidden without frac(...)")

    try:
        import deepxde  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures
    with tempfile.TemporaryDirectory() as tmp:
        res = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, "f1" + gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(_short(NAMES[0], d, 3000)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900)
            check(p.returncode == 0, f"1D {gen.__name__} ran:\n{p.stderr[-1500:]}")
            res[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
        check(res["generate_script"] and res["generate_script"][0] < 0.3 and res["generate_script"] == res["generate_clean_script"],
              f"fractional Poisson 1D trains, live and export agree {res}")
        d = os.path.join(tmp, "inv")
        os.makedirs(d)
        with open(os.path.join(d, "run.py"), "w") as fh:
            fh.write(generate_script(_short(NAMES[4], d, 600)))
        p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900)
        check(p.returncode == 0, f"inverse 1D ran:\n{p.stderr[-1500:]}")
        a = re.findall(r"=== alpha final value: (\S+) ===", p.stdout)
        check(a and abs(float(a[-1]) - 1.8) < 0.1, f"inverse 1D learns alpha towards 1.8: {a}")
    return failures


def main():
    failures = run()
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 77 FRACTIONAL TESTS PASSED")
    return 0


def test_round77_fractional():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
