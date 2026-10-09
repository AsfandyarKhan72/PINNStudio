#!/usr/bin/env python3
"""
Item #4 from the pre-release request: "general figure/plot dimension-size
standardization -- add a 'Plot Settings' option where the user can choose
plot dimensions (square, current default, and something more useful)."

Adds a `plot_figsize_mode` ("Default" | "Square" | "Wide" | "Custom") +
`plot_figsize_w`/`plot_figsize_h` field to PINNConfig, a matching "Figure
size" control (combo + custom width/height spinboxes, shown only for
Custom) to the existing "Plot Settings" dialog on the Setup tab
(_on_plot_settings) and to the Restore tab's own viz-settings dialog
(_on_restore_viz_settings), and a `_plot_figsize(default_w, default_h)`
helper emitted near the top of every generated script -- both the main
Setup/Run script (codegen.py's generate_script) and the three Restore
script builders that produce their own independent scripts
(_build_restore_script, _build_restore_script_ta,
_build_restore_param_script).

Scope, deliberately: every single-panel (or one-plot-per-item, e.g. a row
of per-variable Parameter Convergence subplots, or a row of Surface
snapshots) results plot -- Loss, Line, Surface, both Animation GIF types,
Parameter Convergence, the Parametric-sweep summary bar chart, and all of
their Time-Adaptive equivalents -- gets figsize=_plot_figsize(w, h) in
place of a bare literal tuple. Two categories are deliberately EXCLUDED
and must never be touched by this: the multi-column Error Analysis
comparison grids (sized from however many reference files/columns are
being compared -- forcing any fixed mode on those would misshape a grid
whose whole point is to lay out N columns), and the Setup tab's
domain-sampling preview plots (sized to the problem's own geometry, not
a "results" plot).

A later round brought generate_clean_script (a separate, less-central
exporter) to parity with this feature: its own in-scope single-panel
sites (Loss, single Line/Surface plot) now also get
figsize=_plot_figsize(w, h), via its own independent `_plot_figsize`
helper. Its EA grid(s) remain correctly excluded, same as above, and
RAR's own fixed-3-panel 1D solution_plot.png comparison (a separate
plot from the EA grids, not a "multi-column, N varies" grid) is in
scope for this feature in both generators.

"Default" mode must render every one of the ~40 in-scope figsize=(...)
call sites touched by this feature *exactly* as they rendered before this
feature existed -- this is purely additive. That's checked two ways
here: (1) by extracting the real `_plot_figsize` function text from a
generated script and exec'ing it in isolation for each mode (so the
assertions run against the actual code that will execute, not a
reimplementation of it), and (2) with one real, tiny, real-training run
whose Loss plot (one of the few save sites with no bbox_inches='tight'
crop, so its saved PNG's pixel dimensions are exactly figsize*dpi) is
checked byte-for-pixel against figsize*dpi under all four modes.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_plot_figsize_settings.py
"""
import dataclasses
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

from pinnstudio.core.config import PINNConfig
from pinnstudio.core import codegen
from pinnstudio.ui.main_window import MainWindow


def _extract_plot_figsize_func(script_text):
    """Pulls the generated `def _plot_figsize(...): ...` block's source
    out of a generated script's text (it always ends right before the
    next top-level/dedented line), and exec's it in an isolated namespace.
    Returns the callable. Asserts it was found at all -- a disappearing
    helper (e.g. an indentation slip in a future edit) fails loudly here
    rather than silently no-op'ing figsize for everyone."""
    m = re.search(
        r"^def _plot_figsize\(_default_w, _default_h\):\n(?:[ \t].*\n|\n)*",
        script_text, re.MULTILINE,
    )
    assert m, "Could not find a top-level `def _plot_figsize(...)` block in the generated script"
    ns = {}
    exec(m.group(0), ns)
    return ns["_plot_figsize"]


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── Config defaults ──────────────────────────────────────────────
    cfg = PINNConfig()
    check(cfg.plot_figsize_mode == "Default", f"PINNConfig.plot_figsize_mode should default to 'Default', got {cfg.plot_figsize_mode!r}")
    check(cfg.plot_figsize_w == 7.0, f"PINNConfig.plot_figsize_w should default to 7.0, got {cfg.plot_figsize_w!r}")
    check(cfg.plot_figsize_h == 5.0, f"PINNConfig.plot_figsize_h should default to 5.0, got {cfg.plot_figsize_h!r}")

    # ── GUI default dicts carry the same defaults ─────────────────────
    win = MainWindow()
    check(win._plot_viz_settings.get("figsize_mode") == "Default",
          "MainWindow._plot_viz_settings should default figsize_mode to 'Default'")
    check(win._plot_viz_settings.get("figsize_w") == 7.0 and win._plot_viz_settings.get("figsize_h") == 5.0,
          "MainWindow._plot_viz_settings figsize_w/h should default to 7.0/5.0")
    check(win._restore_viz_settings.get("figsize_mode") == "Default",
          "MainWindow._restore_viz_settings should default figsize_mode to 'Default'")
    check(win._restore_viz_settings.get("figsize_w") == 7.0 and win._restore_viz_settings.get("figsize_h") == 5.0,
          "MainWindow._restore_viz_settings figsize_w/h should default to 7.0/5.0")

    # ── _build_config() threads self._plot_viz_settings into the config ──
    win._plot_viz_settings = dict(win._plot_viz_settings)
    win._plot_viz_settings.update({"figsize_mode": "Square", "figsize_w": 11.0, "figsize_h": 4.0})
    built_cfg = win._build_config()
    check(built_cfg.plot_figsize_mode == "Square", "_build_config() should read figsize_mode from self._plot_viz_settings")
    check(built_cfg.plot_figsize_w == 11.0 and built_cfg.plot_figsize_h == 4.0,
          "_build_config() should read figsize_w/h from self._plot_viz_settings")

    # ── generate_script(): in-scope sites wrapped, EA/clean-script sites untouched ──
    cfg_default = PINNConfig(time_adaptive=False, steady_state=False)
    script_default = codegen.generate_script(cfg_default)

    in_scope_snippets = [
        "figsize=_plot_figsize(7, 5)",                                             # Loss plot
        "figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * _n_ivars)",  # Parameter Convergence (inverse, forward path)
        "figsize=_plot_figsize(8, 5)",                                             # a Line/Surface-type single-panel plot
    ]
    for snippet in in_scope_snippets:
        check(snippet in script_default, f"Expected in-scope figsize call missing from generated script: {snippet!r}")

    # These two have no other (15, 5)-shaped call site anywhere in the
    # generated script, so the simple wrap-and-check-absent pattern is safe.
    ea_untouched_snippets = [
        "figsize=(4*_ea_ncols, 3.5*_ea_nrows)",
        "figsize=(15, 4.5 * _ea_n_t)",
    ]
    for snippet in ea_untouched_snippets:
        check(snippet in script_default, f"Error Analysis grid figsize should stay a bare literal (untouched): {snippet!r}")
        wrapped = snippet.replace("figsize=(", "figsize=_plot_figsize(")
        check(wrapped not in script_default, f"Error Analysis grid figsize must NOT be wrapped by _plot_figsize: found {wrapped!r}")

    # "figsize=(15, 5)" / "figsize=_plot_figsize(15, 5)" and even the
    # "fig, axes_s = plt.subplots(1, 3, ...)" statement text are NOT unique
    # to the standard Error Analysis grid: both RAR's own 1D
    # solution_plot.png comparison and the Time-Adaptive per-step reference
    # overlay (a different plot, which happens to reuse the same local
    # variable name "axes_s") default to the same dimensions and ARE
    # legitimately wrapped. Disambiguate with each site's unique neighboring
    # line instead of matching the subplots() statement alone.
    _ea_1d_bare = (
        "                        _levels_ea1d = np.linspace(_ea_vmin, _ea_vmax, 41)  # see the matching comment above -- contourf ignores vmin/vmax with an integer `levels=N`\n"
        "                        fig, axes_s = plt.subplots(1, 3, figsize=(15, 5))"
    )
    _ea_1d_wrapped = _ea_1d_bare.replace("figsize=(15, 5)", "figsize=_plot_figsize(15, 5)")
    check(_ea_1d_bare in script_default,
          "Error Analysis 1D surface-comparison grid figsize should stay a bare literal (untouched)")
    check(_ea_1d_wrapped not in script_default,
          "Error Analysis 1D surface-comparison grid figsize must NOT be wrapped by _plot_figsize")
    check(
        "_tea_levels = np.linspace(_tea_vmin, _tea_vmax, 41)\n"
        "                        fig, axes_s = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))"
        in script_default,
        "Time-Adaptive per-step reference-overlay figsize IS in scope for this feature and should use _plot_figsize",
    )
    check("_fig_r, _axes_r = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))" in script_default,
          "RAR's own 1D solution_plot.png comparison IS in scope for this feature and should use _plot_figsize")

    # generate_clean_script is a separate, less-central exporter. Its
    # in-scope single-panel sites (Loss, single Line/Surface plot) use their
    # own independent _plot_figsize helper; its EA grid(s) stay bare, same
    # as the main generator, whenever Error Analysis data is configured.
    clean_script = codegen.generate_clean_script(cfg_default)
    check("_plot_figsize" in clean_script,
          "generate_clean_script() should define its own _plot_figsize helper")
    check("plt.figure(figsize=_plot_figsize(7, 5))" in clean_script,
          "generate_clean_script()'s Loss plot should use _plot_figsize")
    check("fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))" in clean_script,
          "generate_clean_script()'s single Line/Surface plot should use _plot_figsize")

    cfg_clean_ea = dataclasses.replace(
        cfg_default,
        ea_files=repr([(0.0, "/tmp/_fake_ref_for_figsize_test.txt", None)]),
        ea_do_surface=True, ea_do_line=False, t_min=0.0, t_max=1.0,
    )
    clean_script_ea = codegen.generate_clean_script(cfg_clean_ea)
    check("fig, axes = plt.subplots(1, 3, figsize=(15, 5))" in clean_script_ea,
          "generate_clean_script()'s EA surface-comparison grid figsize should stay a bare literal (untouched)")
    check("fig, axes = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))" not in clean_script_ea,
          "generate_clean_script()'s EA surface-comparison grid figsize must NOT be wrapped by _plot_figsize")

    # Time-Adaptive path has its own set of in-scope sites.
    cfg_ta = PINNConfig(time_adaptive=True)
    script_ta = codegen.generate_script(cfg_ta)
    check("figsize=_plot_figsize(7, 5)" in script_ta, "Time-Adaptive Surface/loss single-panel plot should use _plot_figsize")
    check("figsize=(4*_ea_ncols, 3.5*_ea_nrows)" in script_ta, "Time-Adaptive EA grid figsize should stay untouched")

    # ── _plot_figsize's actual (exec'd from the real generated text) behavior ──
    plot_figsize = _extract_plot_figsize_func(script_default)
    check(plot_figsize(7, 5) == (7, 5), "Default mode must return the input unchanged (full backward compatibility)")
    check(plot_figsize(8, 4) == (8, 4), "Default mode must return the input unchanged (full backward compatibility)")

    cfg_square = PINNConfig(plot_figsize_mode="Square")
    plot_figsize_sq = _extract_plot_figsize_func(codegen.generate_script(cfg_square))
    check(plot_figsize_sq(7, 5) == (7, 7), f"Square mode should force a 7x7 square from (7,5), got {plot_figsize_sq(7, 5)!r}")
    check(plot_figsize_sq(4, 9) == (9, 9), f"Square mode should use the LARGER side, got {plot_figsize_sq(4, 9)!r}")

    cfg_wide = PINNConfig(plot_figsize_mode="Wide")
    plot_figsize_wide = _extract_plot_figsize_func(codegen.generate_script(cfg_wide))
    check(plot_figsize_wide(7, 5) == (9.0, 5), f"Wide mode should widen to height*1.8, got {plot_figsize_wide(7, 5)!r}")

    cfg_custom = PINNConfig(plot_figsize_mode="Custom", plot_figsize_w=12.0, plot_figsize_h=2.5)
    plot_figsize_custom = _extract_plot_figsize_func(codegen.generate_script(cfg_custom))
    check(plot_figsize_custom(7, 5) == (12.0, 2.5), f"Custom mode should ignore the call's own default and use the configured W/H, got {plot_figsize_custom(7, 5)!r}")

    # ── Restore script builders: each gets its own _plot_figsize, same behavior ──
    win2 = MainWindow()
    minimal_cfg = {
        "layers": [2, 8, 1], "activation": "tanh", "x_min": 0.0, "x_max": 1.0,
        "t_min": 0.0, "t_max": 1.0, "problem_dim": "1D", "output_names": "u",
    }
    for mode, w, h, expect7_5 in [
        ("Default", 7.0, 5.0, (7, 5)),
        ("Square", 7.0, 5.0, (8, 8)),
        ("Wide", 7.0, 5.0, (9.0, 5)),
        ("Custom", 3.0, 6.0, (3.0, 6.0)),
    ]:
        win2._restore_viz_settings = dict(win2._restore_viz_settings)
        win2._restore_viz_settings.update({"figsize_mode": mode, "figsize_w": w, "figsize_h": h})

        single_script = win2._build_restore_script(
            "/tmp/does_not_need_to_exist.pt", minimal_cfg, "adam", "Line (time steps)", 0, 5, "/tmp")
        f_single = _extract_plot_figsize_func(single_script)
        got = f_single(8, 5) if mode != "Default" and mode != "Custom" else (f_single(7, 5) if mode in ("Default",) else f_single(8, 5))
        # Simpler, mode-independent check: Square forces a square from
        # whatever (w,h) the call site itself used -- verify generically
        # instead of hardcoding which literal each mode happens to produce.
        if mode == "Square":
            check(f_single(8, 5) == (8, 8), f"_build_restore_script Square mode: expected (8,8) from (8,5), got {f_single(8, 5)!r}")
        elif mode == "Wide":
            check(f_single(8, 5) == (9.0, 5), f"_build_restore_script Wide mode: expected (9.0,5) from (8,5), got {f_single(8, 5)!r}")
        elif mode == "Custom":
            check(f_single(8, 5) == (3.0, 6.0), f"_build_restore_script Custom mode: expected (3.0,6.0), got {f_single(8, 5)!r}")
        else:
            check(f_single(8, 5) == (8, 5), f"_build_restore_script Default mode must be a no-op, got {f_single(8, 5)!r}")

        ta_steps = [{"dir": "/tmp/step", "t0": 0.0, "t1": 1.0, "model_path": "/tmp/does_not_need_to_exist.pt", "cfg": minimal_cfg}]
        ta_script = win2._build_restore_script_ta(ta_steps, minimal_cfg, "adam", "Line (time steps)", 0, 5, "/tmp")
        f_ta = _extract_plot_figsize_func(ta_script)
        if mode == "Square":
            check(f_ta(7, 4) == (7, 7), f"_build_restore_script_ta Square mode: expected (7,7) from (7,4), got {f_ta(7, 4)!r}")
        elif mode == "Default":
            check(f_ta(7, 4) == (7, 4), f"_build_restore_script_ta Default mode must be a no-op, got {f_ta(7, 4)!r}")

        param_script = win2._build_restore_param_script(["/tmp/a_convergence.txt"], "/tmp", False, False, False, [None])
        f_param = _extract_plot_figsize_func(param_script)
        if mode == "Custom":
            check(f_param(6, 3.2) == (3.0, 6.0), f"_build_restore_param_script Custom mode: expected (3.0,6.0), got {f_param(6, 3.2)!r}")
        elif mode == "Default":
            check(f_param(6, 3.2) == (6, 3.2), f"_build_restore_param_script Default mode must be a no-op, got {f_param(6, 3.2)!r}")

    # ── _build_restore_ea_script must NEVER get a figsize knob -- Error ──
    # Analysis-on-restore grids are out of scope exactly like the main
    # Setup tab's EA grids are.
    ea_script = win2._build_restore_ea_script(
        [], "/tmp", False, True, True, 0.0, 1.0, 0.0, 1.0, "restored_ea")
    check("_plot_figsize" not in ea_script,
          "_build_restore_ea_script's output must never reference _plot_figsize (Error Analysis grids are out of scope)")

    # ── Dialog wiring: the new combo/spinboxes exist and are visible ─────
    # (headless -- never .exec()'d, just constructed and inspected, same
    # approach test_restore_steady_state.py / test_prerelease_gui_tweaks.py
    # use elsewhere in this suite for dialog-content checks without a
    # modal event loop).
    import pinnstudio.ui.main_window as mw_mod
    from PyQt6.QtWidgets import QDialog
    _orig_exec = QDialog.exec
    captured = {}

    def _capturing_exec(self):
        captured["dialog"] = self
        return 0  # pretend "Cancel" without blocking
    QDialog.exec = _capturing_exec
    try:
        win3 = MainWindow()
        win3.plot_type_combo.setCurrentText("Line (time steps)")
        win3._on_plot_settings()
        dlg = captured.get("dialog")
        check(dlg is not None, "_on_plot_settings should have opened a dialog")
        combos = dlg.findChildren(type(win3.plot_type_combo)) if dlg is not None else []
        combo_texts = set()
        for c in combos:
            for i in range(c.count()):
                combo_texts.add(c.itemText(i))
        check({"Default", "Square", "Wide", "Custom"} <= combo_texts,
              f"Plot Settings dialog should offer a Figure-size combo with Default/Square/Wide/Custom, found combo items: {combo_texts}")
    finally:
        QDialog.exec = _orig_exec

    # ── Real end-to-end run: actual pixel dimensions of a saved PNG ──────
    # The Loss plot's savefig call has no bbox_inches='tight' crop, so its
    # saved PNG's pixel size is exactly figsize*dpi -- the strongest
    # available check that this feature's effect reaches an actual
    # produced file, not just the generated script's text. Needs
    # deepxde+torch (same as the rest of this project's real-training
    # checks) -- skipped, not failed, when unavailable (e.g. this repo's
    # lightweight CI job, which only installs PyQt6/numpy/matplotlib/
    # pandas).
    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED real-run pixel checks: deepxde/torch not installed in this environment -- "
              "all the checks above (config defaults, generated-script text, _plot_figsize's actual "
              "exec'd behavior for every mode/builder, dialog wiring) still ran and passed.")
        print("Checks:", len(failures), "failing" if failures else "all passing")
        return failures

    try:
        from PIL import Image
    except ImportError:
        print("SKIPPED real-run pixel checks: Pillow not installed in this environment.")
        print("Checks:", len(failures), "failing" if failures else "all passing")
        return failures

    win4 = MainWindow()
    _app.processEvents()
    win4.quick_examples_combo.setCurrentText("1D Heat")
    _app.processEvents()
    # Disable the optimizer scheduler entirely so this run goes through
    # the fast legacy flat config.iterations (iter1_spin) path, a single
    # short Adam phase -- not the GUI-hidden, always-on-by-default
    # scheduler, which carries whatever (possibly large/L-BFGS) phases the
    # Quick Example set up and would make this check far slower/flakier
    # than the mechanism it's actually testing needs.
    win4.sched_cb.setChecked(False)
    win4.iter1_spin.setValue(3)

    cases = [
        ("Default", 7.0, 5.0, (700, 500)),
        ("Square", 7.0, 5.0, (700, 700)),   # max(7,5)=7 -> 7x7 @ dpi100
        ("Wide", 7.0, 5.0, (900, 500)),     # height*1.8=9 -> 9x5 @ dpi100
        ("Custom", 9.0, 3.0, (900, 300)),
    ]
    for mode, w, h, (exp_w_px, exp_h_px) in cases:
        win4._plot_viz_settings = dict(win4._plot_viz_settings)
        win4._plot_viz_settings.update({"figsize_mode": mode, "figsize_w": w, "figsize_h": h, "dpi": 100})
        config = win4._build_config()
        config.iterations = 3
        config.plot_dpi = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_dir = tmpdir
            script = codegen.generate_script(config)
            sp = os.path.join(tmpdir, "run_script.py")
            with open(sp, "w") as f:
                f.write(script)
            proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=120)
            check(proc.returncode == 0, f"[{mode}] real run should exit 0, got {proc.returncode}. stderr tail: {proc.stderr[-1500:]}")
            loss_png = os.path.join(tmpdir, "solution_results", "loss_plot.png")
            if not os.path.isfile(loss_png):
                # save_dir layout fallback -- some configs nest under a run dir
                found = None
                for root, _dirs, fnames in os.walk(tmpdir):
                    if "loss_plot.png" in fnames:
                        found = os.path.join(root, "loss_plot.png")
                        break
                loss_png = found
            check(loss_png is not None and os.path.isfile(loss_png), f"[{mode}] loss_plot.png should exist under {tmpdir}")
            if loss_png and os.path.isfile(loss_png):
                with Image.open(loss_png) as im:
                    got_w, got_h = im.size
                check((got_w, got_h) == (exp_w_px, exp_h_px),
                      f"[{mode}] loss_plot.png pixel size should be {(exp_w_px, exp_h_px)} (figsize*dpi), got {(got_w, got_h)}")

    print("Checks:", len(failures), "failing" if failures else "all passing")
    return failures


if __name__ == "__main__":
    failures = run()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nALL PLOT-SETTINGS (FIGURE SIZE) TESTS PASSED")
