#!/usr/bin/env python3
"""
Standalone (no separate .patch file needed) version of the fix for:
restoring a Time-Adaptive run's COMBINED models (>=2 step checkpoints
auto-detected and stitched together) with Error Analysis reference files
configured crashed every time, for every viz type:

    NameError: name 'model' is not defined

(in _ea_u_pinns.append(_extract_restore_field(model.predict(_xt)).flatten()))

Root cause: _build_restore_ea_script() in pinnstudio/ui/main_window.py
always generated a bare model.predict(...) in its 3 prediction call
sites, assuming whichever restore script ran before it left a plain
`model` bound. A Time-Adaptive combined restore never does -- it routes
through _ta_model_for_t(tv) instead.

This script edits pinnstudio/ui/main_window.py directly (exact string
replacements, not a git patch -- more robust to hand-transfer onto a
machine your downloads don't reach) and writes
tests/test_restore_ta_error_analysis.py. Safe to re-run: each edit checks
whether it is already present first.

Usage:
    python3 apply_ta_restore_ea_fix_standalone.py [path-to-pinnstudio-repo]
"""
import os
import sys

TEST_FILE_CONTENT = '#!/usr/bin/env python3\n"""\nUser report: restoring a Time-Adaptive run\'s combined models (>=2 step\ncheckpoints auto-detected and stitched together -- see\n_detect_restore_ta_steps/_build_restore_script_ta) with Error Analysis\nreference files configured crashed every time, for every viz type\n(Animation Surface, Animation Line, Line (time steps) all reproduced it\nin the user\'s own logs):\n\n    File "<script>", line NNN, in <module>\n      _ea_u_pinns.append(_extract_restore_field(model.predict(_xt)).flatten())\n    NameError: name \'model\' is not defined\n    Restore failed -- check architecture matches saved model.\n\nThe restore itself (RESTORE_DONE, the actual plot/gif) always completed\nfine -- the crash only hit the separate Error-Analysis-on-restore pass\nthat runs right after, confirming it: the restore script and the EA\nscript are two pieces of text concatenated together, and the SECOND one\nis what breaks.\n\nRoot cause: _build_restore_ea_script() -- the shared EA-on-restore script\nbuilder _on_restore() appends after EITHER _build_restore_script()\n(single model) OR _build_restore_script_ta() (Time-Adaptive combined,\nmultiple per-step models, never a single `model`) -- always generated\n`model.predict(...)` in its 3 prediction call sites (_ea_u_pinns loop,\nand the 2D/1D-3D surface-comparison branches), unconditionally assuming\nwhichever restore script ran before it left a plain `model` variable\nbound. _build_restore_script_ta never does -- it restores each detected\nstep\'s own model lazily and routes by time range through\n_ta_model_for_t(tv) (defined unconditionally near the top of its\ngenerated script, before any viz-type branching), which is also exactly\nright for the EA pass: the reference times being compared can span more\nthan one TA step\'s own time window, so no single model would even be\nvalid for all of them.\n\nFix: _build_restore_ea_script() takes a new `is_ta` parameter. When\nTrue, all 3 prediction call sites use `_ta_model_for_t(_tv).predict(...)`\n(each already inside a `for _i, _tv in enumerate(_ea_times):` loop, so\n_tv is in scope) instead of a bare `model.predict(...)`. _on_restore()\nnow passes `is_ta=_use_ta_restore` -- the same flag it already uses to\ndecide which restore-script builder to call in the first place. The\nsingle-model case (is_ta=False, the default) renders byte-identical text\nto before this fix -- `_predict_call` resolves to the literal string\n"model.predict" -- so test_export_parity.py\'s existing\n_build_restore_ea_script checks are unaffected.\n\nChecks:\n - A real Time-Adaptive restore (2 tiny trained step models, each in its\n   own time_adaptive_steps/step_NNN_t.../ folder, auto-detected via the\n   real _detect_restore_ta_steps()) with Error Analysis reference files\n   spanning BOTH steps\' time windows runs end to end with no crash,\n   through _on_restore\'s actual is_ta=_use_ta_restore wiring reproduced\n   here directly against _build_restore_ea_script.\n - Negative control: appending the EA script with the OLD call\n   convention (omitting is_ta, i.e. is_ta=False) onto that same\n   Time-Adaptive main script reproduces the user\'s exact crash --\n   "NameError: name \'model\' is not defined" -- proving this test\'s setup\n   genuinely exercises the reported bug and isn\'t passing vacuously.\n - test_export_parity.py\'s pre-existing _build_restore_ea_script checks\n   (single-model case, is_ta omitted) still pass unchanged.\n\nRun directly:\n    QT_QPA_PLATFORM=offscreen python3 tests/test_restore_ta_error_analysis.py\n"""\nimport json\nimport os\nimport subprocess\nimport sys\nimport tempfile\n\nos.environ.setdefault("QT_QPA_PLATFORM", "offscreen")\nos.environ.setdefault("DDE_BACKEND", "pytorch")\n\n_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nif _REPO_ROOT not in sys.path:\n    sys.path.insert(0, _REPO_ROOT)\n\nfrom PyQt6.QtWidgets import QApplication\n\n_app = QApplication.instance() or QApplication(sys.argv)\n\nfrom pinnstudio.ui.main_window import MainWindow\n\n\ndef _train_and_save_ta_step(step_dir, t0, t1, tag):\n    """Tiny real 1D transient model trained and saved as if it were one\n    Time-Adaptive step\'s own checkpoint -- "model_lbfgs-<N>.pt" naming\n    matches _detect_restore_ta_steps()\'s own glob pattern\n    ("model_lbfgs-*.pt"), and a sibling step_config.json gives it its own\n    t_min/t_max (exactly the layout generate_script()\'s real Time-\n    Adaptive loop writes -- see that method\'s docstring)."""\n    import deepxde as dde\n\n    os.makedirs(step_dir, exist_ok=True)\n    geom = dde.geometry.Interval(0, 1)\n    timedomain = dde.geometry.TimeDomain(t0, t1)\n    geomtime = dde.geometry.GeometryXTime(geom, timedomain)\n    data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=20,\n                             num_test=20, num_initial=5)\n    net = dde.nn.FNN([2, 16, 16, 1], "tanh", "Glorot uniform")\n    model = dde.Model(data, net)\n    model.compile("adam", lr=0.001)\n    model.train(iterations=1, display_every=1000)\n    # Saved as "model_lbfgs..." purely to match _detect_restore_ta_steps()\'s\n    # own glob pattern ("model_lbfgs-*.pt" is tried first) -- what actually\n    # trained it doesn\'t matter for this test (only the EA predict()-routing\n    # fix is under test here, not training quality), and a real L-BFGS run\n    # against this trivial always-zero-residual dummy PDE never converges\n    # in a bounded number of steps the way Adam\'s `iterations=` does.\n    model_path = model.save(os.path.join(step_dir, "model_lbfgs"))\n\n    with open(os.path.join(step_dir, "step_config.json"), "w") as f:\n        json.dump({\n            "t_min": t0, "t_max": t1, "problem_dim": "1D", "steady_state": False,\n            "layers": [2, 16, 16, 1], "activation": "tanh", "output_names": "u",\n            "x_min": 0.0, "x_max": 1.0,\n        }, f)\n    return model_path\n\n\ndef _run_script(script, tmpdir, tag):\n    sp = os.path.join(tmpdir, f"{tag}_script.py")\n    with open(sp, "w") as f:\n        f.write(script)\n    proc = subprocess.run(\n        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=120,\n    )\n    return proc\n\n\ndef run():\n    failures = []\n\n    def check(cond, msg):\n        if not cond:\n            failures.append(msg)\n            print("FAIL:", msg)\n        else:\n            print("ok:", msg)\n\n    try:\n        import deepxde as _dde_probe  # noqa: F401\n        import torch as _torch_probe  # noqa: F401\n    except ImportError:\n        print("SKIPPED: deepxde/torch not installed in this environment.")\n        return failures\n\n    import numpy as np\n\n    win = MainWindow()\n    win._on_restore_viz_changed = lambda text: None  # skip the "Line Plot Settings"\n    # dialog.exec() side effect -- unrelated to this fix, blocks under headless\n    # test runs with no one to close it. Same guard test_restore_steady_state.py\n    # already uses for the same reason.\n\n    with tempfile.TemporaryDirectory() as tmpdir:\n        ta_root = os.path.join(tmpdir, "time_adaptive_steps")\n        step1_dir = os.path.join(ta_root, "step_001_t0.0000_to_t0.5000")\n        step2_dir = os.path.join(ta_root, "step_002_t0.5000_to_t1.0000")\n        step1_model = _train_and_save_ta_step(step1_dir, 0.0, 0.5, "step1")\n        _train_and_save_ta_step(step2_dir, 0.5, 1.0, "step2")\n\n        ta_steps = win._detect_restore_ta_steps(step1_model)\n        check(len(ta_steps) == 2, f"expected 2 auto-detected Time-Adaptive steps, got {len(ta_steps)}")\n\n        cfg = {\n            "layers": [2, 16, 16, 1], "activation": "tanh",\n            "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,\n            "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,\n            "problem_dim": "1D", "steady_state": False, "loss_type": "MSE",\n            "output_names": "u",\n        }\n        main_script = win._build_restore_script_ta(\n            ta_steps, cfg, "adam", "Line (time steps)", 0, 5, tmpdir)\n        check("_ta_model_for_t" in main_script,\n              "precondition: the Time-Adaptive restore script should define/use _ta_model_for_t")\n        check("\\nmodel =" not in main_script and not main_script.strip().startswith("model ="),\n              "precondition: a combined Time-Adaptive restore script should not bind a plain top-level "\n              "`model` the way a single-model restore does (that\'s the whole reason this bug exists)")\n\n        # Reference times deliberately span BOTH steps\' own windows (0.1 is\n        # step 1\'s, 0.6 is step 2\'s) -- exactly the scenario that makes a\n        # single bare `model` wrong even if one happened to be bound.\n        ref1 = os.path.join(tmpdir, "ref_t0.1.txt")\n        ref2 = os.path.join(tmpdir, "ref_t0.6.txt")\n        xs = np.linspace(0, 1, 11)\n        np.savetxt(ref1, np.column_stack([xs, np.full_like(xs, 0.1), np.sin(xs)]))\n        np.savetxt(ref2, np.column_stack([xs, np.full_like(xs, 0.6), np.cos(xs)]))\n        files = [(0.1, ref1), (0.6, ref2)]\n\n        # ── Negative control: the OLD call convention (is_ta omitted) ──\n        # should still reproduce the user\'s exact crash on this Time-\n        # Adaptive main script, proving this test setup is real.\n        ea_script_old = win._build_restore_ea_script(\n            files, tmpdir, is_2d=False, do_line=True, do_surface=True,\n            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="u",\n            is_3d=False, output_idx=0, output_names="u", is_steady=False,\n        )\n        proc_old = _run_script(main_script + ea_script_old, tmpdir, "ta_ea_old_buggy")\n        check(proc_old.returncode != 0 and "NameError: name \'model\' is not defined" in proc_old.stderr,\n              f"negative control: appending the EA script WITHOUT is_ta=True onto a Time-Adaptive "\n              f"restore script should reproduce the user\'s exact NameError (proves this test\'s setup "\n              f"is real) -- got exit {proc_old.returncode}:\\n{proc_old.stdout[-1000:]}\\n{proc_old.stderr[-1000:]}")\n\n        # ── The fix: is_ta=True (what _on_restore now actually passes) ──\n        ea_script_fixed = win._build_restore_ea_script(\n            files, tmpdir, is_2d=False, do_line=True, do_surface=True,\n            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="u",\n            is_3d=False, output_idx=0, output_names="u", is_steady=False,\n            is_ta=True,\n        )\n        check("_ta_model_for_t(_tv).predict" in ea_script_fixed,\n              "the fixed EA script should route predictions through _ta_model_for_t(_tv), not a bare model")\n        check("model.predict(" not in ea_script_fixed,\n              "the fixed (is_ta=True) EA script should not reference a bare `model` at all")\n\n        proc_fixed = _run_script(main_script + ea_script_fixed, tmpdir, "ta_ea_fixed")\n        check(proc_fixed.returncode == 0,\n              f"Time-Adaptive restore + Error Analysis should run cleanly with the fix, got exit "\n              f"{proc_fixed.returncode}:\\n{proc_fixed.stdout[-2000:]}\\n{proc_fixed.stderr[-2000:]}")\n        check("RESTORE_DONE" in proc_fixed.stdout, "restore itself should still finish (RESTORE_DONE)")\n        check("Restore Error Analysis Complete" in proc_fixed.stdout\n              or "error_metrics_restore.txt" in " ".join(os.listdir(os.path.join(tmpdir, "error_analysis")))\n              if os.path.isdir(os.path.join(tmpdir, "error_analysis")) else False,\n              f"Error Analysis should actually complete and save its metrics file, stdout:\\n{proc_fixed.stdout[-1500:]}")\n        check(os.path.exists(os.path.join(tmpdir, "error_analysis", "error_metrics_restore.txt")),\n              "error_metrics_restore.txt should exist after a successful Time-Adaptive restore + EA run")\n\n    # ── test_export_parity.py\'s existing single-model EA checks should ──\n    # still pass unchanged (is_ta omitted/False renders identical text).\n    files2 = [(0.5, "/tmp/fake_1d.txt")]\n    s_1d = win._build_restore_ea_script(\n        files2, "/tmp/save", is_2d=False, do_line=True, do_surface=True,\n        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="v",\n        is_3d=False, output_idx=1, output_names="u,v",\n    )\n    check("model.predict(_xt)[:, 0]" not in s_1d and "model.predict(_xt_c)[:, 0]" not in s_1d,\n          "single-model EA script should still not hardcode output column 0 (regression check)")\n    check("pred[:, 1]" in s_1d, "single-model EA script should still use the selected output_idx")\n    check("model.predict(_xt)" in s_1d,\n          "single-model (is_ta=False/default) EA script should render the same bare model.predict() as before this fix")\n\n    win.close()\n    return failures\n\n\ndef main():\n    failures = run()\n    for f in failures:\n        print("FAIL:", f)\n    if failures:\n        print(f"\\n{len(failures)} FAILURE(S)")\n        return 1\n    print("\\nALL RESTORE TIME-ADAPTIVE ERROR-ANALYSIS TESTS PASSED")\n    return 0\n\n\ndef test_restore_ta_error_analysis():\n    failures = run()\n    assert not failures, "\\n".join(failures)\n\n\nif __name__ == "__main__":\n    sys.exit(main())\n'


def looks_like_inner_package_folder(path):
    return (
        os.path.basename(os.path.normpath(path)) == "pinnstudio"
        and os.path.isdir(os.path.join(path, "ui"))
        and os.path.isdir(os.path.join(path, "core"))
        and not os.path.isdir(os.path.join(path, "pinnstudio"))
    )


EDITS = [
    (
        "                        custom_expr=custom_expr,\n"
        "                        output_names=cfg.get('output_names', 'u'),\n"
        "                        is_steady=cfg.get('steady_state', False),\n"
        "                    )",
        "                        custom_expr=custom_expr,\n"
        "                        output_names=cfg.get('output_names', 'u'),\n"
        "                        is_steady=cfg.get('steady_state', False),\n"
        "                        is_ta=_use_ta_restore,\n"
        "                    )",
    ),
    (
        "    def _build_restore_ea_script(self, files, save_dir, is_2d, do_line, do_surface,\n"
        "                                  x_min, x_max, y_min, y_max, out_name, viz_settings=None,\n"
        '                                  is_3d=False, output_idx=0, custom_expr="", output_names="u",\n'
        "                                  is_steady=False):",
        "    def _build_restore_ea_script(self, files, save_dir, is_2d, do_line, do_surface,\n"
        "                                  x_min, x_max, y_min, y_max, out_name, viz_settings=None,\n"
        '                                  is_3d=False, output_idx=0, custom_expr="", output_names="u",\n'
        "                                  is_steady=False, is_ta=False):",
    ),
    (
        "        _vmin     = viz_settings.get('vmin', -1.0)\n"
        "        _vmax     = viz_settings.get('vmax', 1.0)\n"
        "        files_repr = repr(files)\n",
        "        _vmin     = viz_settings.get('vmin', -1.0)\n"
        "        _vmax     = viz_settings.get('vmax', 1.0)\n"
        "        files_repr = repr(files)\n"
        "        # Which model to call .predict() on, per reference time _tv: a\n"
        "        # Time-Adaptive combined restore (_build_restore_script_ta) never\n"
        "        # defines a plain `model` -- it restores each detected step's own\n"
        "        # model lazily and routes by time range via _ta_model_for_t(tv)\n"
        "        # (defined unconditionally near the top of that generated script,\n"
        "        # before any viz-type branching -- see that method). Calling this\n"
        "        # per _tv instead of once is required here anyway: the reference\n"
        "        # times being compared can span more than one TA step's own\n"
        "        # window, so no single restored model is even valid for all of\n"
        "        # them. A plain single-model restore (_build_restore_script) still\n"
        "        # just uses its own `model` exactly as before.\n"
        '        _predict_call = "_ta_model_for_t(_tv).predict" if is_ta else "model.predict"\n',
    ),
    (
        "    _ea_u_pinns.append(_extract_restore_field(model.predict(_xt)).flatten())",
        "    _ea_u_pinns.append(_extract_restore_field({_predict_call}(_xt)).flatten())",
    ),
    (
        "            _u_pinn_g = _extract_restore_field(model.predict(_xyt_g)).reshape(_res_ea, _res_ea)",
        "            _u_pinn_g = _extract_restore_field({_predict_call}(_xyt_g)).reshape(_res_ea, _res_ea)",
    ),
    (
        "            _U_pinn[_i] = _extract_restore_field(model.predict(_xt_c)).flatten()",
        "            _U_pinn[_i] = _extract_restore_field({_predict_call}(_xt_c)).flatten()",
    ),
]

CI_OLD = "          python tests/test_ta_loss_history.py\n"
CI_NEW = "          python tests/test_ta_loss_history.py\n          python tests/test_restore_ta_error_analysis.py\n"

MARKER = '_predict_call = "_ta_model_for_t(_tv).predict" if is_ta else "model.predict"'


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    repo = os.path.abspath(repo)

    if looks_like_inner_package_folder(repo):
        parent = os.path.dirname(repo)
        print(f"\u274c {repo!r} looks like the INNER pinnstudio package folder, not the repo root.")
        print(f"   Run this from the repo root instead (the folder that CONTAINS pinnstudio/):")
        print(f"     cd {parent!r} && python3 {os.path.basename(__file__)}")
        return 1

    mw_path = os.path.join(repo, "pinnstudio", "ui", "main_window.py")
    if not os.path.isfile(mw_path):
        print(f"\u274c Could not find pinnstudio/ui/main_window.py under {repo!r}.")
        print(f"   Run this from your pinnstudio repo root, or pass its path as an argument:")
        print(f"     python3 {os.path.basename(__file__)} /path/to/pinnstudio")
        return 1

    with open(mw_path, "r") as f:
        content = f.read()

    if MARKER in content:
        print("\u2705 Already applied -- main_window.py already routes Restore Error Analysis "
              "through _ta_model_for_t() for Time-Adaptive combined restores.")
    else:
        for old, new in EDITS:
            if new in content:
                continue
            if old not in content:
                print(f"\u274c Could not find expected text in main_window.py for one of the edits:")
                print("---- expected to find ----")
                print(old[:300])
                print("---------------------------")
                print("Your main_window.py doesn't match what this script expects (maybe it's "
                      "already been modified, or is a different version). Stopping without making "
                      "any changes so nothing gets half-applied.")
                return 1
            content = content.replace(old, new, 1)
        with open(mw_path, "w") as f:
            f.write(content)
        print("\u2705 pinnstudio/ui/main_window.py updated.")

    tests_dir = os.path.join(repo, "tests")
    test_path = os.path.join(tests_dir, "test_restore_ta_error_analysis.py")
    if os.path.isfile(test_path):
        print("\u2705 tests/test_restore_ta_error_analysis.py already exists.")
    else:
        os.makedirs(tests_dir, exist_ok=True)
        with open(test_path, "w") as f:
            f.write(TEST_FILE_CONTENT)
        print("\u2705 tests/test_restore_ta_error_analysis.py created.")

    ci_path = os.path.join(repo, ".github", "workflows", "smoke-test.yml")
    if os.path.isfile(ci_path):
        with open(ci_path, "r") as f:
            ci_content = f.read()
        if "test_restore_ta_error_analysis.py" in ci_content:
            print("\u2705 Already registered in .github/workflows/smoke-test.yml.")
        elif CI_OLD in ci_content:
            ci_content = ci_content.replace(CI_OLD, CI_NEW, 1)
            with open(ci_path, "w") as f:
                f.write(ci_content)
            print("\u2705 .github/workflows/smoke-test.yml updated.")
        else:
            print("\u26a0\ufe0f  Could not find the expected CI line to update smoke-test.yml -- "
                  "skipping that (harmless).")
    else:
        print("\u2139\ufe0f  No .github/workflows/smoke-test.yml found -- skipping CI registration.")

    print()
    print("Next steps:")
    print("  1. Review the changes:  git diff")
    print("  2. Run the test (needs PyQt6 + torch + deepxde installed, same as the app itself --")
    print("     this one actually trains 2 tiny real models, so it takes a minute or two):")
    print("     QT_QPA_PLATFORM=offscreen python3 tests/test_restore_ta_error_analysis.py")
    print('  3. git add -A && git commit -m "..." && git push')
    return 0


if __name__ == "__main__":
    sys.exit(main())
