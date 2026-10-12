#!/usr/bin/env python3
"""
Round 58 -- generate_script()'s Error Analysis setup (normalize ea_files to
(time, path, selector) 3-tuples, keep the ones in [t_min, t_max], group
them by output, define _ea_group_label / _ea_extract) was hand-kept twice:
once for the Standard path and once for the Time-Adaptive path, with small
drift (log-line order; TA's _ea_extract required the model argument while
Standard's defaulted to the module-level model). Both now emit
_gs_ea_setup_code(config, banner, indent).

Before the commit, 216 generated live scripts (12 templates x Forward/
Inverse x 9 variants, including Standard and Time-Adaptive Error Analysis
with mixed selectors) differed only in those two expected ways; the
end-to-end Error Analysis tests (Rounds 41 and 50, the TA line-comparison
test) passed. This file checks the single-source property.

Run directly:
    python3 tests/test_round58_ea_setup_shared.py
"""
import ast
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pinnstudio.core.codegen import _gs_ea_setup_code, generate_script
from pinnstudio.core.config import PINNConfig


def _cfg(ta):
    cfg = PINNConfig()
    cfg.problem_dim = "1D"; cfg.steady_state = False
    cfg.num_outputs = 1; cfg.output_names = "u"; cfg.layers = [2, 8, 1]
    cfg.pde_expressions = "du_t - 0.1*du_xx"; cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "sin(pi*x)"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = "[]"
    cfg.time_adaptive = ta
    cfg.adapt_method = "Time Adaptive Training" if ta else "None"
    cfg.ea_files = repr([(0.5, "/r/a.txt"), (0.25, "/r/b.txt", 0), (0.75, "/r/c.txt", ["u**2", "sq"]), (9.0, "/r/out.txt")])
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # generate_script() is runtime-generic: every live script carries both
    # the Standard and the Time-Adaptive branch, so both setup blocks are
    # emitted -- each must be exactly the shared helper's output.
    for ta in (False, True):
        cfg = _cfg(ta)
        script = generate_script(cfg)
        try:
            ast.parse(script)
        except SyntaxError as e:
            check(False, f"ta={ta}: generated script should parse: {e}")
        check(script.count("def _ea_group_label(") == 2 and script.count("def _ea_extract(") == 2,
              f"ta={ta}: one setup block per path (Standard + Time-Adaptive) expected")
        check(script.count("def _ea_extract(_ea_grid, _ea_sel, _ea_model=None):") == 2,
              f"ta={ta}: both paths' _ea_extract should take an optional model")
        for banner, indent in (("Error Analysis", "            "), ("Time-Adaptive Error Analysis", "        ")):
            block = _gs_ea_setup_code(cfg, banner, indent)
            check(block in script, f"ta={ta}: the {banner} setup block should be the shared helper's output")
            check(0 <= block.find(f"=== Running {banner} ===") < block.find("Filtering to t=["),
                  f"{banner}: banner should print before the filtering line")

    # The emitted setup runs: grouping, the t-range filter and labels behave.
    code = _gs_ea_setup_code(_cfg(False), "Error Analysis", "")
    ns = {"_plot_output_names_list": ["u"], "print": lambda *a, **k: None}
    exec(code, ns)
    check(len(ns["_ea_files_norm"]) == 3, "the t=9.0 file is outside [t_min, t_max] and should be dropped")
    check(ns["_ea_multi_output"] and len(ns["_ea_groups"]) == 3, "three distinct selectors -> three groups")
    check([ns["_ea_group_label"](s) for s in (None, 0, ["u**2", "sq"], ["x", ""])] == ["", "u", "sq", "custom"],
          "group labels: default/raw/custom-with-label/custom")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 58 EA-SETUP TESTS PASSED")
    return 0


def test_round58_ea_setup_shared():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
