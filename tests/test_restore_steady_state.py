#!/usr/bin/env python3
"""
Restore Model crashed for a steady-state problem: restoring a trained
Soomro_Paper-style 2D steady model and generating a Surface plot raised
"RuntimeError: mat1 and mat2 shapes cannot be multiplied (40000x3 and
2x64)". Root cause: _build_restore_script() in main_window.py -- the
script-builder behind the Restore panel's "Restore && Visualize" button --
never checked cfg.get("steady_state", False) anywhere. A steady-state
config's network has one fewer input (no time axis -- see
_on_steady_state_changed and generate_script()'s own _is_steady branch in
codegen.py, both already correct), but every model.predict(...) call this
function built always appended a time column regardless, so the restored
network's first layer (sized correctly from cfg["layers"], e.g. [2,64,...]
for 2D steady) was fed one column too many.

Fix:
 - _build_restore_script() now threads is_steady through its geometry/data
   restore scaffold (geomtime = geom, dde.data.PDE instead of TimePDE,
   same convention as codegen.py) and every viz_type branch's prediction
   arrays (Surface for 1D/2D/3D -- the actually-reachable, fixed paths;
   "Line (time steps)" and the two Animation types get a defensive
   steady-state fallback too, even though they're now hidden from the
   dropdown for a steady-state restore -- see next bullet).
 - _build_restore_ea_script() (the separate Error Analysis-on-restore
   script-builder) got the same is_steady treatment for its reference-file
   loading and prediction arrays, matching generate_script()'s own
   steady-state reference-file format (x,y,u / x,y,z,u -- no time column).
 - The Restore panel's Visualization type dropdown now hides "Line (time
   steps)", "Animation Line (GIF)", and "Animation Surface (GIF)" when the
   config about to be restored is steady-state (_restore_forward_viz_items/
   _refresh_restore_viz_options), since stepping through or animating over
   time doesn't apply with no time axis at all -- same reasoning
   _on_steady_state_changed already applies to Time Adaptive Training on
   the main Setup tab. Only "Surface" (plus, in Inverse mode, the two
   Parameter Convergence options, which never touch the model/time axis)
   stay available.

Checks:
 - _restore_forward_viz_items()/_refresh_restore_viz_options() filter the
   dropdown correctly for steady vs. non-steady configs, with and without
   Inverse mode.
 - _build_restore_script()'s generated script actually restores a real
   trained steady-state model and produces a Surface plot without
   crashing, for 1D/2D/3D -- the exact scenario that crashed (2D) plus its
   siblings.
 - The non-steady (transient) 2D Surface path still works unchanged
   (regression check -- this fix must not touch the already-correct case).
 - _build_restore_ea_script()'s generated script runs cleanly against a
   steady-state reference file (x, y, u -- no time column) and a restored
   steady 2D model, matching the user's actual Error-Analysis-on-restore
   use case for their corrected Soomro_Paper problem.

SECOND BUG, found when the user re-tried restoring after the first fix
above and hit the *same* crash again: the checks above all hand-built the
`cfg` dict directly with "steady_state" already set, which never exercised
the actual save path at all -- so they missed that codegen.py's own
`_model_config` dict (what actually gets written to model_config.json by
a real training run) never included "steady_state" in the first place.
Every restored steady-state model was silently defaulting to "treat as
transient" (cfg.get("steady_state", False)) regardless of the fix above,
because the saved file never said otherwise. Confirmed directly against
the user's own model_config.json (grep for "steady_state" found nothing).

Second fix: codegen.py's `_model_config` dict (the one write site for
model_config.json, confirmed via grep -- the Time-Adaptive step_config.json
writer is separate and already unreachable for steady-state, since Time
Adaptive is hidden whenever Steady-state is checked) now includes
"steady_state": config.steady_state, plus "z_min"/"z_max" (present in the
Time-Adaptive step writer already, but missing here -- needed for a 3D
steady model's restore to use the right domain instead of silently
defaulting to [0,1]).

test_real_save_then_restore_steady_2d below closes exactly this gap: it
goes through the REAL pipeline end to end -- load the GUI's own "2D
Poisson (Disk)" steady-state Quick Example, build a real PINNConfig via
_build_config(), run codegen.generate_script()'s actual generated script
as a subprocess (so model_config.json is written by the real save code,
not assembled by the test), read that file back, and feed it into
_build_restore_script() to restore the real checkpoint that run produced.
Verified this exact test fails without the codegen.py fix (model_config.
json has no "steady_state" key -> restore script defaults to transient ->
the same shape-mismatch RuntimeError) and passes with it.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_restore_steady_state.py
"""
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

from pinnstudio.ui.main_window import MainWindow


def _train_and_save_model(tmpdir, is_3d, is_2d, steady, layers, tag):
    """Builds and trains (1 iteration -- only the saved architecture and
    checkpoint matter here, not solution quality) a tiny model matching
    the given dimension/steady-state combination, saves it, and returns
    the .pt path -- so the restore scripts below restore a REAL checkpoint
    rather than just being syntax-checked."""
    import deepxde as dde

    if is_3d:
        geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
    elif is_2d:
        geom = dde.geometry.Rectangle([0, 0], [1, 1])
    else:
        geom = dde.geometry.Interval(0, 1)

    if steady:
        data = dde.data.PDE(geom, lambda x, y: y[:, 0:1] * 0, [], num_domain=20, num_test=20)
    else:
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=20,
                                 num_test=20, num_initial=5)

    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1, display_every=1000)
    return model.save(os.path.join(tmpdir, tag))


def _run_script(script, tmpdir, tag):
    """Writes `script` to disk and runs it as a subprocess (so a crash
    inside the generated restore script -- e.g. the exact shape-mismatch
    RuntimeError this fix addresses -- fails the test the same way it
    failed for the user, instead of being silently swallowed)."""
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    proc = subprocess.run(
        [sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=120,
    )
    return proc


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # ── Dropdown filtering ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        win = MainWindow()
        win._on_restore_viz_changed = lambda text: None  # skip the pre-existing
        # settings-dialog side effect (dialog.exec()) -- unrelated to this fix,
        # and blocking under headless/offscreen test runs with no one to close it.

        steady_cfg_path = os.path.join(tmpdir, "steady_cfg.json")
        with open(steady_cfg_path, "w") as f:
            json.dump({"steady_state": True, "num_outputs": 1, "output_names": "u"}, f)
        trans_cfg_path = os.path.join(tmpdir, "trans_cfg.json")
        with open(trans_cfg_path, "w") as f:
            json.dump({"steady_state": False, "num_outputs": 1, "output_names": "u"}, f)

        win.restore_config_path.setText(steady_cfg_path)
        win._refresh_restore_viz_options()
        items_steady = [win.restore_viz_combo.itemText(i) for i in range(win.restore_viz_combo.count())]
        check(items_steady == ["Surface"],
              f"steady-state restore dropdown should show only 'Surface', got {items_steady}")

        win.restore_config_path.setText(trans_cfg_path)
        win._refresh_restore_viz_options()
        items_trans = [win.restore_viz_combo.itemText(i) for i in range(win.restore_viz_combo.count())]
        check(items_trans == ["Surface", "Line (time steps)", "Animation Line (GIF)", "Animation Surface (GIF)"],
              f"non-steady restore dropdown should keep all four options, got {items_trans}")

        win.restore_mode_combo.setCurrentText("Inverse Model")
        win.restore_config_path.setText(steady_cfg_path)
        win._refresh_restore_viz_options()
        items_inv_steady = [win.restore_viz_combo.itemText(i) for i in range(win.restore_viz_combo.count())]
        check(items_inv_steady == ["Surface", "Parameter Convergence Plot (PNG)", "Parameter Convergence Animation (GIF)"],
              f"inverse+steady restore dropdown should keep Surface + param options only, got {items_inv_steady}")

    # ── _build_restore_script / _build_restore_ea_script: real restore + ──
    # plot, no crash. These need deepxde+torch actually installed (the
    # app's own runtime dependencies -- see requirements.txt/setup.py) to
    # train, save, and restore a real checkpoint -- unlike the rest of this
    # repo's smoke-test suite, which only ever exercises GUI/config-
    # building logic and deliberately keeps CI's install step to
    # PyQt6/numpy/matplotlib/pandas. Skipped (not failed) when they aren't
    # available, e.g. this lightweight CI job -- every developer's and
    # user's real environment has them (the app can't run PINNs without
    # torch/deepxde either way), so this still gets fully exercised there.
    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment -- "
              "the dropdown-filtering checks above still ran and passed; "
              "the real restore/Error-Analysis script checks need deepxde+"
              "torch (see requirements.txt) and are skipped here.")
        return failures

    win = MainWindow()
    win._restore_viz_settings = dict(win._restore_viz_settings or {})
    win._restore_viz_settings["resolution"] = 20  # keep the grid small -- speed only

    _dim_cases = [
        # (is_3d, is_2d, steady, layers, problem_dim)
        (False, True, True, [2, 16, 16, 4], "2D"),   # the exact reported crash
        (True, False, True, [3, 16, 16, 2], "3D"),
        (False, False, True, [1, 16, 16, 1], "1D"),
        (False, True, False, [3, 16, 16, 4], "2D"),  # regression: transient unaffected
    ]
    for is_3d, is_2d, steady, layers, dim in _dim_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            tag = f"{dim}_{'steady' if steady else 'transient'}"
            model_path = _train_and_save_model(tmpdir, is_3d, is_2d, steady, layers, tag)
            cfg = {
                "layers": layers, "activation": "tanh",
                "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
                "problem_dim": dim, "steady_state": steady, "loss_type": "MSE",
                "output_names": ",".join(f"u{i}" for i in range(layers[-1])),
            }
            script = win._build_restore_script(model_path, cfg, "adam", "Surface", 0, 5, tmpdir)
            proc = _run_script(script, tmpdir, tag)
            check(proc.returncode == 0,
                  f"restore script for {tag} should run cleanly, got exit {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
            check("RESTORE_DONE" in proc.stdout, f"restore script for {tag} should finish (RESTORE_DONE), got:\n{proc.stdout}")
            check(os.path.exists(os.path.join(tmpdir, "restored_plot.png")),
                  f"restore script for {tag} should save restored_plot.png")

    # ── _build_restore_ea_script: steady 2D Error Analysis, no crash ──
    with tempfile.TemporaryDirectory() as tmpdir:
        import numpy as np
        layers = [2, 16, 16, 4]
        model_path = _train_and_save_model(tmpdir, False, True, True, layers, "ea_steady_2d")
        cfg = {
            "layers": layers, "activation": "tanh",
            "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
            "problem_dim": "2D", "steady_state": True, "loss_type": "MSE",
            "output_names": "u,v,w,p",
        }
        main_script = win._build_restore_script(model_path, cfg, "adam", "Surface", 2, 5, tmpdir)

        # Steady 2D reference format: x, y, u -- no time column, matching
        # generate_script()'s own inline Error Analysis convention for a
        # steady config (see codegen.py).
        ref_path = os.path.join(tmpdir, "ref_steady_2d.txt")
        xs = np.linspace(0, 1, 6); ys = np.linspace(0, 1, 6)
        X, Y = np.meshgrid(xs, ys)
        U = np.sin(X) * np.cos(Y)
        np.savetxt(ref_path, np.column_stack([X.ravel(), Y.ravel(), U.ravel()]))

        ea_script = win._build_restore_ea_script(
            files=[(0.0, ref_path)], save_dir=tmpdir, is_2d=True,
            do_line=True, do_surface=True,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            out_name="w", viz_settings={}, is_3d=False, output_idx=2,
            custom_expr="", output_names="u,v,w,p", is_steady=True,
        )
        full_script = main_script + ea_script
        proc = _run_script(full_script, tmpdir, "ea_steady_2d")
        check(proc.returncode == 0,
              f"restore+EA script for steady 2D should run cleanly, got exit {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
        check("Restore Error Analysis Complete" in proc.stdout,
              f"restore+EA script for steady 2D should finish Error Analysis, got:\n{proc.stdout}")
        check(os.path.exists(os.path.join(tmpdir, "error_analysis", "error_metrics_restore.txt")),
              "restore+EA script for steady 2D should save error_metrics_restore.txt")

    # ── Real end-to-end pipeline: GUI template -> _build_config() -> ──
    # generate_script() -> actual training subprocess writes a REAL
    # model_config.json -> read it back -> _build_restore_script() restores
    # the REAL checkpoint that run produced. This is the only check in this
    # file that exercises codegen.py's own model_config.json writer rather
    # than a hand-built cfg dict -- which is exactly what let the second bug
    # (missing "steady_state" key in that writer) slip past every check
    # above.
    with tempfile.TemporaryDirectory() as tmpdir:
        from pinnstudio.core.codegen import generate_script

        win2 = MainWindow()
        win2.radio_2d.setChecked(True)
        win2._on_dim_changed()  # repopulates quick_examples_combo with the 2D list
        win2.quick_examples_combo.setCurrentText("2D Poisson (Disk)")
        _app.processEvents()
        check(win2.steady_state_check.isChecked(),
              "precondition failed: '2D Poisson (Disk)' should load as steady-state")
        # Trim every phase down to near-nothing -- only the real save-time
        # behavior matters here, not solution quality.
        for ph in win2.sched_phase_list:
            ph['iters'].setValue(1)
        win2.iter1_spin.setValue(1)
        win2.iter2_spin.setValue(0)
        win2.save_dir_input.setText(tmpdir)

        config = win2._build_config()
        check(config.steady_state is True, "built config should have steady_state=True")
        # _build_config() now points save_dir at its own fresh
        # "<label>__<timestamp>" subfolder under whatever "Save to:" path
        # was configured (see MainWindow._run_results_dir) rather than
        # that path directly -- so the real training run's actual output
        # location is config.save_dir, not the bare tmpdir this test
        # configured save_dir_input with.
        run_dir = config.save_dir
        check(run_dir and run_dir != tmpdir and os.path.dirname(run_dir) == tmpdir,
              f"config.save_dir should be its own fresh subfolder under {tmpdir!r}, got {run_dir!r}")

        train_script = generate_script(config)
        sp = os.path.join(tmpdir, "train_script.py")
        with open(sp, "w") as f:
            f.write(train_script)
        proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=180)
        check(proc.returncode == 0,
              f"real training run for '2D Poisson (Disk)' should complete, got exit {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

        mc_path = os.path.join(run_dir, "solution_results", "model_config.json")
        check(os.path.exists(mc_path), f"real training run should write {mc_path}")
        if os.path.exists(mc_path):
            with open(mc_path) as f:
                saved_cfg = json.load(f)
            check(saved_cfg.get("steady_state") is True,
                  f"model_config.json written by a real steady-state training run should have "
                  f"\"steady_state\": true, got {saved_cfg.get('steady_state')!r} (full keys: {sorted(saved_cfg)})")

            pt_candidates = [f for f in os.listdir(os.path.join(run_dir, "solution_results")) if f.endswith(".pt")]
            check(bool(pt_candidates), "real training run should save at least one .pt checkpoint")
            if pt_candidates:
                pt_name = max(pt_candidates, key=lambda f: os.path.getmtime(
                    os.path.join(run_dir, "solution_results", f)))
                pt_path = os.path.join(run_dir, "solution_results", pt_name)
                # Match the Restore panel's own "Optimizer used for this model"
                # selector to whichever phase actually produced this checkpoint
                # (its optimizer_state_dict only loads correctly into the same
                # optimizer type it was saved from) -- same as a real user
                # would pick after checking which phase's file they browsed to.
                restore_optimizer = "lbfgs" if "lbfgs" in pt_name else "adam"
                restore_script = win2._build_restore_script(pt_path, saved_cfg, restore_optimizer, "Surface", 0, 5, tmpdir)
                rproc = _run_script(restore_script, tmpdir, "real_pipeline_restore")
                check(rproc.returncode == 0,
                      f"restoring the real checkpoint from model_config.json should work, got exit "
                      f"{rproc.returncode}:\n{rproc.stdout}\n{rproc.stderr}")
                check("RESTORE_DONE" in rproc.stdout,
                      f"real-pipeline restore should finish (RESTORE_DONE), got:\n{rproc.stdout}")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL RESTORE STEADY-STATE TESTS PASSED")
    return 0


def test_restore_steady_state():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
