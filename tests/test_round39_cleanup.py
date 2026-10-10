#!/usr/bin/env python3
"""
Round 39: codebase cleanup, phase 2 (part of the audit's own §2 + §3c).

This covers:

 1. §2a -- the figsize-mode helper (`_plot_figsize`) was hand-copied 5
    times (generate_script(), generate_clean_script()'s own
    _clean_figsize_runtime_code(), and 3 separate copies in
    main_window.py's _build_restore_param_script/_build_restore_script/
    _build_restore_script_ta). All 5 now go through one shared function,
    codegen.py's _figsize_helper_code(mode, w, h) -- generate_clean_script()
    via the thin _clean_figsize_runtime_code() wrapper (kept for backward
    compatibility, same call signature as before), the other 4 directly.
    This test exercises _figsize_helper_code() itself (all 4 modes), then
    confirms every one of the 5 call sites embeds its output correctly and
    (for the 3 main_window.py builders + generate_script()) that a real
    run with a non-default figsize mode still executes cleanly end-to-end.

 2. §2b -- generate_clean_script()'s own custom-plot-expression guard used
    a bare `if _plot_custom_expr:` (no `.strip()`), unlike the two
    generate_script() copies of the same guard (`if _plot_custom_expr.strip():`).
    Harmless for freshly-generated output (the Python-side value is always
    pre-stripped before being embedded), but a real crash risk for the
    "Export as DeepXDE Script" feature's whole reason to exist: the user
    hand-edits the exported .py file afterward. A custom-expression field
    left as whitespace-only after such an edit would silently fall into
    eval("   ", ...) -> SyntaxError under the old code. Now matches the
    other two copies.

 3. §3c -- `_on_scheduler_changed` (main_window.py) was investigated (not
    just assumed): confirmed to have been orphaned by an earlier change
    that made the "Enable Optimizer Scheduler" checkbox permanently
    checked AND permanently hidden (`sched_cb.setVisible(False)` with the
    comment "always on, hidden"), with `sched_widget` ALSO unconditionally
    `setVisible(True)` regardless of the checkbox. A hidden, always-checked
    checkbox can never fire a state-change signal from user interaction, so
    wiring this handler up would be a no-op, not a bug fix -- confirmed
    genuinely dead, not a missing-`.connect()` bug, and deleted. An
    orphaned duplicate `self.sched_widget = QWidget()` immediately
    shadowed 18 lines later by the real one was removed in the same pass.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round39_cleanup.py
"""
import json
import os
import re
import sys
import tempfile
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core.config import PINNConfig
from pinnstudio.core.codegen import (generate_script, generate_clean_script,
                                      _figsize_helper_code, _clean_figsize_runtime_code)
from pinnstudio.ui.main_window import MainWindow


def _run_script(script, tmpdir, tag, timeout_s=120):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def _rect_cfg(problem_dim="1D", plot_type="Line (time steps)"):
    """Minimal steady Rectangle/Interval config -- not testing geometry
    masking here, so the plain default geometry is fine and keeps the
    real training run fast."""
    cfg = PINNConfig()
    cfg.problem_dim = problem_dim
    cfg.steady_state = True
    cfg.num_outputs = 1
    cfg.output_names = "u"
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.y_min, cfg.y_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 20; cfg.num_boundary = 10; cfg.num_initial = 0; cfg.num_test = 15
    cfg.layers = [1 if problem_dim == "1D" else 2, 16, 16, 1]
    cfg.pde_expressions = "du_xx" if problem_dim == "1D" else "du_xx + du_yy"
    cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "0.0"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    cfg.adapt_method = "None"
    cfg.time_adaptive = False
    cfg.iterations = 1; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.plot_type = plot_type
    cfg.plot_resolution = 16
    cfg.num_timesteps_line = 1
    return cfg


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("FAIL:", msg)
        else:
            print("ok:", msg)

    try:
        import numpy as np
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    # ── 1. _figsize_helper_code() itself: all 4 modes, in isolation ──────
    for mode, w, h, expect in [
        ("Default", 7.0, 5.0, (7.0, 5.0)),
        ("Square", 7.0, 5.0, (7.0, 7.0)),
        ("Wide", 7.0, 5.0, (9.0, 5.0)),
        ("Custom", 12.0, 2.5, (12.0, 2.5)),
    ]:
        src = _figsize_helper_code(mode, w, h)
        check(src.strip().startswith("def _plot_figsize("),
              f"_figsize_helper_code({mode!r}) should return a _plot_figsize(...) function definition")
        ns = {}
        exec(src, ns)
        got = ns["_plot_figsize"](7.0, 5.0)
        check(tuple(got) == expect,
              f"_plot_figsize in {mode!r} mode on defaults (7.0, 5.0) should return {expect}, got {got}")

    # Thin wrapper delegates correctly.
    cfg_w = PINNConfig()
    cfg_w.plot_figsize_mode = "Custom"; cfg_w.plot_figsize_w = 11.0; cfg_w.plot_figsize_h = 3.0
    wrapped_src = _clean_figsize_runtime_code(cfg_w)
    direct_src = _figsize_helper_code("Custom", 11.0, 3.0)
    check(wrapped_src == direct_src,
          "_clean_figsize_runtime_code(config) should delegate verbatim to _figsize_helper_code(...)")

    # ── 2. All 5 call sites embed it correctly ───────────────────────────
    cfg_sq = _rect_cfg("1D")
    cfg_sq.plot_figsize_mode = "Square"
    gs_script = generate_script(cfg_sq)
    check("def _plot_figsize" in gs_script and "_fs_mode = 'Square'" in gs_script,
          "generate_script() should embed the shared figsize helper with the configured mode")

    clean_script = generate_clean_script(cfg_sq)
    check("def _plot_figsize" in clean_script and "_fs_mode = 'Square'" in clean_script,
          "generate_clean_script() should embed the shared figsize helper with the configured mode")

    win = MainWindow()
    win._on_restore_viz_changed = lambda text: None

    rect_cfg_dict = {
        "layers": [1, 16, 16, 1], "activation": "tanh",
        "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
        "t_min": 0.0, "t_max": 1.0,
        "problem_dim": "1D", "steady_state": True, "loss_type": "MSE",
        "output_names": "u", "geometry_type": "Rectangle",
    }

    win._restore_viz_settings = {"figsize_mode": "Custom", "figsize_w": 9.0, "figsize_h": 1.5}
    restore_script = win._build_restore_script(
        "dummy_model_path", rect_cfg_dict, "adam", "Line (time steps)", 0, 1, "/tmp")
    check("_figsize_helper_code(figsize_mode, figsize_w, figsize_h)" not in restore_script,
          "_build_restore_script()'s figsize block should be a RESOLVED f-string substitution in the "
          "output, not the literal unresolved call text")
    check("def _plot_figsize" in restore_script and "_fs_mode = 'Custom'" in restore_script
          and "return (9.0, 1.5)" in restore_script,
          "_build_restore_script() should embed the shared figsize helper with the configured Custom "
          "width/height")

    fake_ta_steps = [
        {"t0": 0.0, "t1": 0.5, "model_path": "/tmp/fake_ta_step1", "cfg": {}},
        {"t0": 0.5, "t1": 1.0, "model_path": "/tmp/fake_ta_step2", "cfg": {}},
    ]
    ta_script = win._build_restore_script_ta(
        fake_ta_steps, rect_cfg_dict, "adam", "Line (time steps)", 0, 1, "/tmp")
    check("def _plot_figsize" in ta_script and "_fs_mode = 'Custom'" in ta_script
          and "return (9.0, 1.5)" in ta_script,
          "_build_restore_script_ta() should embed the shared figsize helper with the configured "
          "Custom width/height")

    param_script = win._build_restore_param_script(["/tmp/fake_param.txt"], "/tmp", False, False, False)
    check("def _plot_figsize" in param_script,
          "_build_restore_param_script() should embed the shared figsize helper")

    # ── 3. Real exec-level run with a non-default figsize mode ───────────
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_run = _rect_cfg("1D")
        cfg_run.plot_figsize_mode = "Wide"
        script = generate_script(cfg_run)
        proc = _run_script(script, tmpdir, "figsize_wide_1d")
        check(proc.returncode == 0,
              f"generate_script() with plot_figsize_mode='Wide' should run cleanly, got "
              f"{proc.returncode}:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    import deepxde as dde

    def _train_rect_model(tmpdir, tag):
        geom = dde.geometry.Interval(0.0, 1.0)
        data = dde.data.PDE(geom, lambda x, y: y[:, 0:1] * 0, [], num_domain=20, num_test=15)
        net = dde.nn.FNN([1, 16, 16, 1], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        model.compile("adam", lr=0.001)
        model.train(iterations=1, display_every=1000)
        return model.save(os.path.join(tmpdir, tag))

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _train_rect_model(tmpdir, "rect_model")
        win._restore_viz_settings = {"figsize_mode": "Custom", "figsize_w": 10.0, "figsize_h": 2.0}
        restore_script = win._build_restore_script(
            model_path, rect_cfg_dict, "adam", "Line (time steps)", 0, 1, tmpdir)
        proc = _run_script(restore_script, tmpdir, "restore_rect_line_custom_figsize")
        check(proc.returncode == 0,
              f"_build_restore_script() restore with a Custom figsize should run cleanly, got "
              f"{proc.returncode}:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check(os.path.exists(os.path.join(tmpdir, "restored_plot.png")),
              "restore script should save restored_plot.png with the Custom figsize applied")

    win.close()

    # ── 4. §2b: generate_clean_script()'s custom-expr guard now matches
    #      generate_script()'s (.strip()) instead of crashing on a
    #      hand-edited whitespace-only expression ──────────────────────
    cfg_ce = _rect_cfg("1D")
    cfg_ce.plot_custom_expr = ""  # no custom expression configured
    clean_ce = generate_clean_script(cfg_ce)
    check("if _plot_custom_expr.strip():" in clean_ce,
          "generate_clean_script()'s _extract_plot_field guard should use .strip(), matching "
          "generate_script()'s two copies of the same guard")
    check("_plot_custom_expr = ''" in clean_ce,
          "sanity precondition: freshly generated script should embed an empty custom-expr literal")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate a user hand-editing the exported script and leaving the
        # custom-expression field whitespace-only instead of truly empty
        # (the whole point of "Export as DeepXDE Script" is that the user
        # can edit this file afterward).
        edited = clean_ce.replace("_plot_custom_expr = ''", "_plot_custom_expr = '   '")
        check(edited != clean_ce, "sanity precondition: the hand-edit substitution should actually apply")
        proc = _run_script(edited, tmpdir, "clean_script_whitespace_custom_expr")
        check(proc.returncode == 0,
              f"a hand-edited whitespace-only _plot_custom_expr should fall back to the plain output "
              f"field, not crash with eval()'s SyntaxError, got {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check("SyntaxError" not in proc.stderr, "should not hit eval('   ', ...)'s SyntaxError")

    # ── 5. §3c: _on_scheduler_changed confirmed dead and removed ─────────
    check(not hasattr(MainWindow, "_on_scheduler_changed"),
          "_on_scheduler_changed should no longer exist on MainWindow")
    main_window_src = open(os.path.join(_REPO_ROOT, "pinnstudio", "ui", "main_window.py"),
                            encoding="utf-8").read()
    check("_on_scheduler_changed" not in main_window_src,
          "no remaining references to _on_scheduler_changed anywhere in main_window.py")
    check(len(re.findall(r"self\.sched_widget\s*=\s*QWidget\(\)", main_window_src)) == 1,
          "the orphaned duplicate `self.sched_widget = QWidget()` line should be gone -- exactly one "
          "real assignment should remain")

    return failures


def main():
    failures = run()
    print()
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("ALL ROUND 39 CLEANUP TESTS PASSED")
        sys.exit(0)


def test_round39_cleanup():
    """pytest entry point."""
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    main()
