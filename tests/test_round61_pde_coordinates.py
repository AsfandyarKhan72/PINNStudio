#!/usr/bin/env python3
"""
Round 61 -- PDE expressions can use the coordinates and math functions.

DeepXDE's examples are full of source terms like sin(pi*x) or
exp(-t)*sin(pi*x)*sin(pi*y). PINNStudio couldn't express them: inside the
PDE, x was the whole input tensor and y the network output, and sin(...)
became np.sin(...), which crashes on the network's grad-tracking tensors
(no built-in template used either, so it never showed). Now:

  - math functions in a PDE become torch calls (sin -> torch.sin,
    arcsin -> torch.asin, ...); np./torch.-qualified calls are untouched;
  - x, y, z, t used in a PDE mean the coordinate columns (y/z only in
    2D/3D, t only when time-dependent), bound only when the equation
    actually uses them -- so every existing template generates
    byte-identical scripts (checked for 216 live + 168 exported scripts);
  - the exported script's boundary-condition expressions get numpy math
    names too (sin(pi*x) in a Dirichlet value used to be a NameError there;
    the live script already had them).

Run directly:
    python3 tests/test_round61_pde_coordinates.py
"""
import ast
import json
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pinnstudio.core.codegen import (_np_math_expr, _pde_coord_columns, _pde_coord_names,
                                     _simplify_pde_expr, generate_clean_script, generate_script)
from pinnstudio.core.config import PINNConfig


def _poisson_1d(save_dir, iters):
    c = PINNConfig()
    c.problem_dim = "1D"; c.steady_state = True; c.geometry_type = "Interval"
    c.x_min, c.x_max = -1.0, 1.0
    c.num_outputs = 1; c.output_names = "u"
    c.pde_expressions = "-du_xx - pi**2*sin(pi*x)"; c.pde_expression = c.pde_expressions
    c.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "sin(pi*x)", "component": 0}])
    c.num_domain = 16; c.num_boundary = 2; c.num_initial = 0; c.num_test = 100
    c.layers = [1, 20, 20, 1]; c.iterations = iters; c.optimizer2 = "none"; c.iterations2 = 0
    c.optimizer_scheduler = False; c.adapt_method = "None"; c.save_dir = save_dir; c.ea_files = "[]"
    return c


def _heat_2d_source(save_dir, iters):
    c = PINNConfig()
    c.problem_dim = "2D"; c.steady_state = False; c.geometry_type = "Rectangle"
    c.x_min, c.x_max, c.y_min, c.y_max, c.t_min, c.t_max = 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    c.num_outputs = 1; c.output_names = "u"
    c.pde_expressions = "du_t - (du_xx + du_yy) - exp(-t)*sin(pi*x)*sin(pi*y)*(2*pi**2 - 1)"
    c.pde_expression = c.pde_expressions
    c.ic_expressions = "sin(pi*x)*sin(pi*y)"; c.ic_expression = c.ic_expressions
    c.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    c.num_domain = 200; c.num_boundary = 40; c.num_initial = 40; c.num_test = 100
    c.layers = [3, 20, 20, 1]; c.iterations = iters; c.optimizer2 = "none"; c.iterations2 = 0
    c.optimizer_scheduler = False; c.adapt_method = "None"; c.save_dir = save_dir; c.ea_files = "[]"
    c.plot_resolution = 20
    return c


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── conversions ───────────────────────────────────────────────────────
    check(_simplify_pde_expr("-du_xx - pi**2*sin(pi*x)") == "-du_xx - np.pi**2*torch.sin(np.pi*x)",
          _simplify_pde_expr("-du_xx - pi**2*sin(pi*x)"))
    check(_simplify_pde_expr("arcsin(u) + np.sin(x) + torch.exp(t)") == "torch.asin(u) + np.sin(x) + torch.exp(t)",
          "qualified calls must be left alone; arcsin -> torch.asin")
    check(_np_math_expr("sin(pi*x) + np.cos(y)") == "np.sin(np.pi*x) + np.cos(y)", _np_math_expr("sin(pi*x) + np.cos(y)"))
    check(_pde_coord_names(["du_t - du_xx + x*t"]) == {"x", "t"}, "du_t/du_xx must not count as coordinates")
    check(_pde_coord_names(["du_t - 0.4*du_xx"]) == set(), "no coordinates in a plain heat equation")
    check(_pde_coord_columns({"x", "y", "t"}, True, False, False) == [("x", 0), ("y", 1), ("t", 2)], "2D columns")
    check(_pde_coord_columns({"y", "t"}, False, False, True) == [], "no y in 1D, no t when steady")

    # ── generated scripts ─────────────────────────────────────────────────
    plain = _poisson_1d("/tmp/x", 10)
    plain.pde_expressions = plain.pde_expression = "-du_xx - 1"
    for gen in (generate_script, generate_clean_script):
        check("coordinates used by the equation itself" not in gen(plain),
              f"{gen.__name__}: no coordinate binding for an equation without coordinates")
    live, export = generate_script(_poisson_1d("/tmp/x", 10)), generate_clean_script(_poisson_1d("/tmp/x", 10))
    for name, s in (("live", live), ("export", export)):
        try:
            ast.parse(s)
        except SyntaxError as e:
            check(False, f"{name} script should parse: {e}")
    check('_eval_ns[\'x\'] = x[:, 0:1]' in live, "live: x bound as the first input column")
    check("    x = _X[:, 0:1]" in export, "export: x rebound as a column inside pde()")
    check("np.sin(np.pi*x)" in export, "export: Dirichlet value gets numpy math names")

    try:
        import deepxde as _dde_probe  # noqa: F401
    except ImportError:
        print("SKIPPED (end-to-end part): deepxde not installed.")
        return failures

    # ── they actually train ───────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        for tag, make in (("poisson1d", _poisson_1d), ("heat2d_source", _heat_2d_source)):
            for gen in (generate_script, generate_clean_script):
                d = os.path.join(tmp, f"{tag}_{gen.__name__}")
                os.makedirs(d)
                sp = os.path.join(d, "run.py")
                with open(sp, "w", encoding="utf-8") as f:
                    f.write(gen(make(d, 300)))
                p = subprocess.run([sys.executable, sp], cwd=d, capture_output=True, text=True, timeout=600)
                check(p.returncode == 0, f"{tag} via {gen.__name__} should run, got {p.returncode}:\n"
                                         f"{p.stdout[-800:]}\n{p.stderr[-1200:]}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 61 PDE-COORDINATE TESTS PASSED")
    return 0


def test_round61_pde_coordinates():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
