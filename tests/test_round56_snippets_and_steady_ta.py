#!/usr/bin/env python3
"""
Round 56 -- two small cleanups plus one real fix.

1. The _DTypeSafeGeom adapter class (4 hand-typed copies: live Solve,
   export, both restore builders -- two with a docstring, two without) and
   the eval() math-function dict (7 copies) now come from one helper each:
   _dtype_safe_geom_class_code() and _math_namespace_code(var, lib, indent).
   Generated scripts were checked byte-for-byte identical before the
   commit (168 live, 168 export, 400 restore), apart from the class
   docstring now appearing in every copy.

2. PINNConfig documents time_adaptive as ignored for a steady-state
   problem, but neither generator enforced it: generate_script() took its
   Time-Adaptive branch for a steady problem and generate_clean_script()
   emitted a script that didn't parse (12 of 168 template variants with
   both flags set, e.g. from a hand-edited save). Both generators now apply
   _effective_config() first.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round56_snippets_and_steady_ta.py
"""
import ast
import dataclasses
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from pinnstudio.core import codegen
from pinnstudio.core.config import PINNConfig


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # helper output is valid Python and has the right entries
    for lib in ("torch", "np"):
        src = codegen._math_namespace_code("_NS", lib, "    ")
        check(all(ln.startswith("    ") for ln in src.split("\n")), f"{lib}: every line should carry the indent")
        tree = ast.parse("\n".join(ln[4:] for ln in src.split("\n")))
        keys = [k.value for k in tree.body[0].value.keys]
        check(keys == ["sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
                       "exp", "log", "log10", "sqrt", "abs", "ceil", "floor", "pi"], f"{lib}: unexpected keys {keys}")
    check("asin" in codegen._math_namespace_code("_NS", "torch") and "np.arcsin" in codegen._math_namespace_code("_NS", "np"),
          "torch uses torch.asin/acos/atan, numpy uses np.arcsin/...")
    ns = {"np": np}
    class _Geom:
        def random_points(self, n, random="pseudo"):
            return np.zeros((n, 2), dtype=np.float64)
        bbox = "passthrough"
    class _dde:
        class config:
            @staticmethod
            def real(_np):
                return np.float32
    ns["dde"] = _dde
    exec(codegen._dtype_safe_geom_class_code(), ns)
    g = ns["_DTypeSafeGeom"](_Geom())
    check(g.random_points(3).dtype == np.float32 and g.bbox == "passthrough",
          "_DTypeSafeGeom should cast sampled points and pass other attributes through")

    # no hand-typed copies left
    for path in ("pinnstudio/core/codegen.py", "pinnstudio/ui/main_window.py"):
        text = open(os.path.join(_REPO_ROOT, path), encoding="utf-8").read()
        check(len(re.findall(r'"arcsin": (?:torch\.asin|np\.arcsin)', text)) == 0, f"{path}: hand-typed math dict left")
        check(text.count("def uniform_boundary_points(self, n):\n        return self._geom.uniform_boundary_points") <= (1 if "codegen" in path else 0),
              f"{path}: hand-typed _DTypeSafeGeom copy left")

    # steady + time_adaptive: time_adaptive is ignored
    base = PINNConfig()
    base.steady_state = True
    base.problem_dim = "2D"
    base.ic_expressions = "0.0"
    cfg = dataclasses.replace(base, time_adaptive=True, ta_num_steps=2)
    clean = codegen.generate_clean_script(cfg)
    try:
        ast.parse(clean)
    except SyntaxError as e:
        check(False, f"export for steady + time_adaptive should parse: {e}")
    check("step_i" not in clean, "export for a steady problem should not contain the Time-Adaptive loop")
    live = codegen.generate_script(cfg)
    ast.parse(live)
    check("if False:\n    print(\"\\n=== Starting Time-Adaptive" in live,
          "live script for a steady problem should switch the Time-Adaptive branch off")
    check(codegen._effective_config(base) is base, "a config without the conflict should be returned unchanged")
    check(cfg.time_adaptive is True, "the caller's config must not be mutated")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 56 TESTS PASSED")
    return 0


def test_round56_snippets_and_steady_ta():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
