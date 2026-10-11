#!/usr/bin/env python3
"""
Round 55 -- codebase cleanup, audit §4 (naming). The two script generators
used different generation-time names for the same things -- generate_script()
had _line_slice_y/_line_slice_z/_sol_ext/_inv_vars_parsed/_obs_files_parsed/
_net_helper_code/_uses_nncg (and read config.steady_state inline, with no
is_steady flag), where generate_clean_script() had line_slice_y/.../is_steady;
generate_clean_script() in turn had _clean_bc_forced_uniform/
_clean_bc_op_uses_X/_clean_safe_num_test for generate_script()'s
_bc_forced_uniform/_bc_op_uses_X/_safe_num_test. Porting a fix between the two
kept tripping over this (it was a standing-rule gotcha). Both now use one
spelling per concept, and so do main_window.py's restore builders
(_line_slice_y/_z -> line_slice_y/_z, like _build_restore_ea_script).

Only generation-time Python names changed (renamed by ast position, never by
text search), so every generated script is byte-for-byte unchanged -- checked
before the commit on 168 live, 168 exported and 400 restore scripts. Runtime names inside
the live Solve script (_is_steady, _is_2d, _save_dir, ...) deliberately keep
their underscore: every runtime variable there uses it.

This guard fails if an old spelling is reintroduced in either generator.

Run directly:
    python3 tests/test_round55_generator_naming.py
"""
import ast
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODEGEN = os.path.join(_REPO_ROOT, "pinnstudio", "core", "codegen.py")
_MAIN_WINDOW = os.path.join(_REPO_ROOT, "pinnstudio", "ui", "main_window.py")

_OLD = {"_line_slice_y", "_line_slice_z", "_sol_ext", "_inv_vars_parsed", "_obs_files_parsed",
        "_net_helper_code", "_uses_nncg", "_clean_bc_forced_uniform", "_clean_bc_op_uses_X",
        "_clean_safe_num_test"}
_GENERATORS = ("generate_script", "_gs_rar_loop_code", "_gs_time_adaptive_loop_code", "generate_clean_script")


def run():
    failures = []
    tree = ast.parse(open(_CODEGEN, encoding="utf-8").read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in _GENERATORS:
        fn = fns.get(name)
        if fn is None:
            failures.append(f"{name} not found in codegen.py")
            continue
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {a.arg for a in fn.args.args}
        bad = sorted(used & _OLD)
        if bad:
            failures.append(f"{name}: old generation-time spelling(s) back in use: {bad}")
    gs_names = {t.id for s in ast.walk(fns["generate_script"]) if isinstance(s, ast.Assign)
                for t in s.targets if isinstance(t, ast.Name)}
    for flag in ("is_2d", "is_3d", "is_steady", "line_slice_y", "line_slice_z", "sol_ext"):
        if flag not in gs_names:
            failures.append(f"generate_script() should define the shared generation-time name {flag!r}")
    # The restore builders in main_window.py follow the same spelling
    # (_build_restore_ea_script already did; the other two used _line_slice_*).
    mw = ast.parse(open(_MAIN_WINDOW, encoding="utf-8").read())
    cls = [n for n in mw.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow"][0]
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name.startswith("_build_restore"):
            used = {n.id for n in ast.walk(f) if isinstance(n, ast.Name)}
            bad = sorted(used & {"_line_slice_y", "_line_slice_z"})
            if bad:
                failures.append(f"MainWindow.{f.name}: old spelling(s) back in use: {bad}")
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 55 GENERATOR-NAMING TESTS PASSED")
    return 0


def test_round55_generator_naming():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
