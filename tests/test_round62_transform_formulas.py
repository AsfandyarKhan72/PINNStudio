#!/usr/bin/env python3
"""
Round 62 -- input/output transform formulas (DeepXDE's apply_feature_transform
and apply_output_transform with an arbitrary function), and Restore now
re-attaches transforms.

  - "Enable input transform" / "Enable output transform" gain a Mode:
    Scale/shift (as before) or Formula. Input formula: the network's input
    features, ';'-separated, in x, y, z, t (e.g. "x*sin(y); x*cos(y)" --
    DeepXDE's Laplace-on-a-disk example); the first layer is sized to the
    feature count. Output formula: one expression per output in the
    coordinates and the raw outputs by name -- a hard constraint, e.g.
    "(1-x**2)*u" makes u = 0 at x = +-1 exactly.
  - One generator, codegen._net_transforms_code, writes the transform
    function for the live script, the exported script and both Restore &
    Visualize builders. Before this round the restore builders re-attached
    NO transform at all, so restoring any model trained with the existing
    scale/shift transform predicted wrong values; model_config.json now
    records the transform settings (an older one without them restores as
    before).

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round62_transform_formulas.py
"""
import ast
import glob
import json
import os
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

from pinnstudio.core.codegen import (_effective_config, _net_transforms_code, generate_clean_script,
                                     generate_script)
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow


def _cfg(save_dir, iters, **kw):
    c = PINNConfig()
    base = dict(problem_dim="1D", steady_state=True, geometry_type="Interval", x_min=-1.0, x_max=1.0,
                pde_expressions="-du_xx - pi**2*sin(pi*x)", custom_bc_json="[]",
                output_transform_enabled=True, output_transform_mode="formula", output_transform_formula="(1-x**2)*u",
                num_domain=32, num_boundary=2, num_initial=0, num_test=50, layers=[1, 20, 20, 1],
                iterations=iters, optimizer2="none", iterations2=0, optimizer_scheduler=False,
                save_dir=save_dir, ea_files="[]")
    base.update(kw)
    for k, v in base.items():
        setattr(c, k, v)
    c.pde_expression = c.pde_expressions
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── code generation ───────────────────────────────────────────────────
    plain = PINNConfig()
    check(_net_transforms_code(plain).strip().endswith("return net") and "net.apply_" not in _net_transforms_code(plain),
          "no transform configured -> the function just returns the net")
    feat = _cfg("/tmp/x", 10, problem_dim="2D", geometry_type="Rectangle", output_transform_enabled=False,
                input_transform_enabled=True, input_transform_mode="formula",
                input_feature_formula="x*sin(y); x*cos(y); x", layers=[2, 20, 1])
    check(_effective_config(feat).layers[0] == 3, "first layer sized to the 3 input features")
    code = _net_transforms_code(_effective_config(feat))
    check("torch.cat([x*torch.sin(y), x*torch.cos(y), x], dim=1)" in code, code)
    for gen in (generate_script, generate_clean_script):
        for c in (_cfg("/tmp/x", 10), feat):
            try:
                ast.parse(gen(c))
            except SyntaxError as e:
                check(False, f"{gen.__name__}: script should parse: {e}")

    # ── GUI round trip ────────────────────────────────────────────────────
    win = MainWindow()
    win.input_transform_cb.setChecked(True)
    win._set_combo_data(win.input_transform_mode_combo, "formula")
    win.input_feature_formula_edit.setText("x*sin(y); x*cos(y)")
    win.output_transform_cb.setChecked(True)
    win._set_combo_data(win.output_transform_mode_combo, "formula")
    win.output_formula_edits[0].setText("x*(1-x)*u")
    c1 = win._build_config()
    win2 = MainWindow()
    win2._apply_config(c1)
    c2 = win2._build_config()
    for k in ("input_transform_mode", "input_feature_formula", "output_transform_mode", "output_transform_formula"):
        check(getattr(c1, k) == getattr(c2, k), f"{k} should survive save/load: {getattr(c1, k)!r} vs {getattr(c2, k)!r}")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── train (live + export), then Restore must re-attach the constraint ─
    with tempfile.TemporaryDirectory() as tmp:
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as f:
                f.write(gen(_cfg(d, 300)))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=600)
            check(p.returncode == 0, f"{gen.__name__}: hard-constraint run failed:\n{p.stderr[-1500:]}")
        sol = os.path.join(tmp, "generate_script", "solution_results")
        cfg = json.load(open(os.path.join(sol, "model_config.json"), encoding="utf-8", errors="replace"))
        check(cfg.get("output_transform_formula") == "(1-x**2)*u", "model_config.json should record the transform")
        model = sorted(glob.glob(os.path.join(sol, "model_*.pt")))[-1]
        s = win._build_restore_script(model, cfg, "adam", "Surface", 0, 3, os.path.join(tmp, "restore"))
        s += "\nimport numpy as _np\nprint('PRED', *[float(v) for v in model.predict(_np.array([[-1.0], [1.0]]))[:, 0]])\n"
        rp = os.path.join(tmp, "restore.py")
        with open(rp, "w", encoding="utf-8") as f:
            f.write(s)
        p = subprocess.run([sys.executable, rp], cwd=tmp, capture_output=True, text=True, timeout=300)
        pred = [l for l in p.stdout.splitlines() if l.startswith("PRED")]
        check(p.returncode == 0 and pred, f"restore script should run:\n{p.stdout[-800:]}\n{p.stderr[-1200:]}")
        if pred:
            vals = [abs(float(v)) for v in pred[0].split()[1:]]
            check(max(vals) < 1e-6, f"restored model must keep u = 0 at x = +-1 (transform re-attached), got {vals}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 62 TRANSFORM-FORMULA TESTS PASSED")
    return 0


def test_round62_transform_formulas():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
