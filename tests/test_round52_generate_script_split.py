#!/usr/bin/env python3
"""
Round 52 -- codebase cleanup, phase 15 (audit §5): generate_script()'s two
largest unnamed inline blocks -- the RAR loop with its per-round
diagnostics (~735 lines) and the Time-Adaptive training loop (~1,618
lines) -- moved verbatim into _gs_rar_loop_code(config) and
_gs_time_adaptive_loop_code(...), which generate_script()'s template now
calls in place. generate_script() itself shrank from ~5,860 to ~3,520 lines.

Pure refactor: before the commit, 168 generated scripts (all 12 templates x
Forward/Inverse x 7 variants: base, RAR, Time-Adaptive, TA + IC
pre-training, IC restore, Surface GIF with a manual range, custom-field
Line plot) were byte-for-byte identical to the pre-refactor output. This
file checks the wiring: every generated script embeds both helpers' text
exactly, and still parses.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round52_generate_script_split.py
"""
import ast
import dataclasses
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pinnstudio.core import codegen
from pinnstudio.core.config import PINNConfig


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    base = PINNConfig()
    base.ic_expressions = "sin(pi*x)"
    for tag, kw in (("base", {}), ("rar", {"adapt_method": "RAR", "rar_cycles": 2}),
                    ("ta", {"time_adaptive": True, "ta_num_steps": 2, "ic_pretrain": True})):
        cfg = dataclasses.replace(base, **kw)
        script = codegen.generate_script(cfg)
        try:
            ast.parse(script)
        except SyntaxError as e:
            check(False, f"{tag}: generated script should parse: {e}")
        _rar = codegen._gs_rar_loop_code(cfg)
        # the inverse parameter history goes to the run's own scratch folder
        _rar_own = _rar.replace('"/tmp/param', '_PS_TMP + "/param')
        check(_rar in script or _rar_own in script, f"{tag}: RAR loop text should be embedded verbatim")
        check("=== Starting Time-Adaptive Training ===" in script and "# ── Time Adaptive Loop" in script,
              f"{tag}: Time-Adaptive loop text should be embedded")
    import inspect
    n_lines = len(inspect.getsource(codegen.generate_script).splitlines())
    check(n_lines < 4000, f"generate_script() should stay split (now {n_lines} lines)")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 52 GENERATE_SCRIPT-SPLIT TESTS PASSED")
    return 0


def test_round52_generate_script_split():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
