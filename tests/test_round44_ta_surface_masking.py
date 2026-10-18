#!/usr/bin/env python3
"""
Round 44 -- codebase cleanup, phase 6 (audit §2e follow-up): the
Time-Adaptive "Surface" plot never masked its prediction to the real
(possibly non-rectangular/CSG) geometry, unlike its single-model
sibling, which always has (see test_restore_geometry_masking.py). This
gap existed in BOTH places the Time-Adaptive Surface plot is built:

 1. main_window.py's `_build_restore_script_ta` (Restore && Visualize),
    both the 3D (6 bounding-box faces via plot_surface) and 2D
    (contourf) branches -- confirmed via direct inspection that its
    single-model sibling `_build_restore_script` has always called
    `geom.inside(...)` to mask here, while the Time-Adaptive branch
    built the same bounding-box grid/faces but never masked anything.

 2. codegen.py's `generate_script()` own live Time-Adaptive
    final-solution "Surface" plot (the 2D contourf branch specifically
    -- the 1D/3D "else" branch stitches together real per-step training
    data as an x-t slice, not a fresh unmasked grid, so it was not
    affected and is not touched here). The real reconstructed `geom`
    (built once at module scope, `geom = _build_geom()`, confirmed
    never reassigned before this section) was already in lexical scope
    at this point -- the per-step throwaway `dde.geometry.Rectangle(...)`
    built nearby only exists to satisfy DeepXDE's Model/TimePDE
    constructor API so a restored per-step checkpoint's `.predict()`
    works; it was never meant to represent the true domain and is
    unrelated to masking.

Fix: both locations now call `geom.inside(...)` once per panel's
spatial grid (3D: once per bounding-box face, 2D: once for the whole
grid) and use it exactly the way the already-correct single-model
paths do -- NaN out-of-domain predictions before `contourf` (2D), and
zero the `facecolor` alpha channel for out-of-domain quads before
`plot_surface` (3D), restricting `auto_range`'s min/max to in-domain
values only.

This is a real bug fix, not a pure refactor, so unlike Rounds 42/43 the
verification bar here is the standard one: real exec-level DeepXDE
subprocess runs, with a side-channel mask array written out (same
technique test_restore_geometry_masking.py already established for the
single-model Surface branch) to confirm the TA Surface masks are
genuinely MIXED (some grid points in-domain, some out) for a
non-rectangular geometry -- not a no-op that happens to mark everything
"inside" -- and that a plain Rectangle/Cuboid (regression) still marks
its own bounding box fully inside.

A Sphere's own bounding box is tangent to the sphere at a single point
per face -- a poor case for checking 3D FACE masking specifically (see
test_restore_geometry_masking.py's own comment on this). A Custom 3D
"cuboid with a corner notched out" has a genuinely mixed inside/outside
region covering a real fraction of 3 of its 6 bounding-box faces
instead, so that's used here for the 3D case, same as that file.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round44_ta_surface_masking.py
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


def _train_and_save_ta_step(tmpdir, dde_geom, t0, t1, layers, tag):
    import deepxde as dde

    timedomain = dde.geometry.TimeDomain(t0, t1)
    geomtime = dde.geometry.GeometryXTime(dde_geom, timedomain)
    data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=40,
                             num_test=40, num_initial=5)
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1, display_every=1000)
    return model.save(os.path.join(tmpdir, tag))


def _run_script(script, tmpdir, tag, timeout_s=240):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


# ---------------------------------------------------------------------
# Part 1: main_window.py's _build_restore_script_ta Surface branch
# (both 2D and 3D), via a real Time-Adaptive restore of real checkpoints.
# ---------------------------------------------------------------------

def _run_restore_ta_case(check, tag, dim, geom_builder, geom_fields, is_3d, expect_mixed):
    import deepxde as dde
    import numpy as np

    win = MainWindow()
    win._restore_viz_settings = dict(win._restore_viz_settings or {})
    win._restore_viz_settings["resolution"] = 16
    win._restore_viz_settings["n_2d_snapshots"] = 1
    win._restore_viz_settings["n_steps"] = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        geom = geom_builder()
        layers = [4, 16, 16, 1] if is_3d else [3, 16, 16, 1]
        step0 = _train_and_save_ta_step(tmpdir, geom, 0.0, 0.5, layers, "s0")
        step1 = _train_and_save_ta_step(tmpdir, geom, 0.5, 1.0, layers, "s1")
        ta_steps = [
            {"t0": 0.0, "t1": 0.5, "model_path": step0,
             "cfg": {"layers": layers, "activation": "tanh", "loss_type": "MSE", "is_lbfgs": False}},
            {"t0": 0.5, "t1": 1.0, "model_path": step1,
             "cfg": {"layers": layers, "activation": "tanh", "loss_type": "MSE", "is_lbfgs": False}},
        ]
        cfg = {"layers": layers, "activation": "tanh",
               "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0,
               "problem_dim": dim, "output_names": "u"}
        cfg.update(geom_fields)

        script = win._build_restore_script_ta(ta_steps, cfg, "adam", "Surface", 0, 3, tmpdir)
        check("geom.inside(" in script, f"{tag}: TA Surface restore script should call geom.inside(...) to mask")

        mask_path = os.path.join(tmpdir, "mask.npy")
        if is_3d:
            diag = (f"\nimport numpy as _np_check\n"
                    f"_np_check.save({mask_path!r}, np.concatenate([m.ravel() for m in _face_inside3]))\n")
        else:
            diag = f"\nimport numpy as _np_check\n_np_check.save({mask_path!r}, _inside2d)\n"

        proc = _run_script(script + diag, tmpdir, tag)
        check(proc.returncode == 0,
              f"{tag}: TA Surface restore should run cleanly, got exit {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check("RESTORE_DONE" in proc.stdout, f"{tag}: TA Surface restore should finish (RESTORE_DONE)")
        check(os.path.exists(os.path.join(tmpdir, "restored_plot.png")),
              f"{tag}: TA Surface restore should save restored_plot.png")

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            has_inside = bool(np.any(mask))
            has_outside = bool(np.any(~mask))
            check(has_inside, f"{tag}: mask should mark at least some grid points as inside the domain")
            if expect_mixed:
                check(has_outside,
                      f"{tag}: TA Surface mask should mark some bounding-box grid points OUTSIDE the "
                      f"real domain -- all-inside means the fix isn't actually discriminating anything")
            else:
                check(not has_outside,
                      f"{tag}: a plain Rectangle/Cuboid fills its own bounding box -- TA Surface masking "
                      f"should not blank out any grid point for it (regression check)")


# ---------------------------------------------------------------------
# Part 2: codegen.py's generate_script() own live TA final-solution
# Surface plot (2D branch), via a real end-to-end training run.
# ---------------------------------------------------------------------

def _ta_2d_live_cfg(save_dir, geometry_type, **geom_fields):
    from pinnstudio.core.config import PINNConfig

    cfg = PINNConfig()
    cfg.problem_dim = "2D"; cfg.steady_state = False
    cfg.num_outputs = 1; cfg.output_names = "u"
    cfg.geometry_type = geometry_type
    for k, v in geom_fields.items():
        setattr(cfg, k, v)
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.y_min, cfg.y_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 80; cfg.num_boundary = 20; cfg.num_initial = 20; cfg.num_test = 40
    cfg.layers = [3, 16, 16, 1]
    cfg.pde_expression = "du_t - 0.4*(du_xx + du_yy)"
    cfg.pde_expressions = cfg.pde_expression
    cfg.ic_expression = "0.0"; cfg.ic_expressions = cfg.ic_expression
    cfg.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    cfg.adapt_method = "None"
    cfg.time_adaptive = True
    cfg.ta_num_steps = 2
    cfg.ta_step_groups = ""
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.save_dir = save_dir
    cfg.plot_type = "Surface"
    cfg.plot_resolution = 20
    cfg.n_2d_snapshots = 1
    return cfg


def _run_live_ta_surface_case(check, tag, geometry_type, geom_fields, expect_mixed):
    import numpy as np
    from pinnstudio.core.codegen import generate_script

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "out")
        os.makedirs(save_dir, exist_ok=True)
        cfg = _ta_2d_live_cfg(save_dir, geometry_type, **geom_fields)
        script = generate_script(cfg)
        check("_inside_ta" in script,
              f"{tag}: live TA Surface plot should compute _inside_ta to mask the 2D contourf")

        mask_path = os.path.join(tmpdir, "mask_live.npy")
        diag = f"\nimport numpy as _np_check\n_np_check.save({mask_path!r}, _inside_ta)\n"
        proc = _run_script(script + diag, tmpdir, tag, timeout_s=240)
        check(proc.returncode == 0,
              f"{tag}: live TA training + Surface plot should run cleanly, got exit {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check(os.path.exists(os.path.join(save_dir, "solution_results", "solution_plot.png")),
              f"{tag}: live TA run should save solution_results/solution_plot.png")

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            has_inside = bool(np.any(mask))
            has_outside = bool(np.any(~mask))
            check(has_inside, f"{tag}: live TA mask should mark at least some grid points as inside the domain")
            if expect_mixed:
                check(has_outside,
                      f"{tag}: live TA Surface mask should mark some grid points OUTSIDE the real domain")
            else:
                check(not has_outside,
                      f"{tag}: a plain Rectangle fills its own bounding box -- live TA Surface masking "
                      f"should not blank out any grid point for it (regression check)")


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    import deepxde as dde

    # ── Part 1: _build_restore_script_ta (main_window.py) ───────────────
    _run_restore_ta_case(
        check, "restore_ta_disk_2d", "2D",
        lambda: dde.geometry.Disk([0.5, 0.5], 0.4),
        {"geometry_type": "Disk", "geom_center_x": 0.5, "geom_center_y": 0.5, "geom_radius": 0.4},
        is_3d=False, expect_mixed=True,
    )
    _run_restore_ta_case(
        check, "restore_ta_rect_2d", "2D",
        lambda: dde.geometry.Rectangle([0, 0], [1, 1]),
        {"geometry_type": "Rectangle"},
        is_3d=False, expect_mixed=False,
    )
    notched_json = json.dumps([
        {"type": "Cuboid", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}},
        {"type": "Cuboid", "params": {"x_min": 0.5, "x_max": 1.0, "y_min": 0.5, "y_max": 1.0, "z_min": 0.5, "z_max": 1.0},
         "op": "subtract"},
    ])
    _run_restore_ta_case(
        check, "restore_ta_notched_3d", "3D",
        lambda: dde.geometry.CSGDifference(dde.geometry.Cuboid([0, 0, 0], [1, 1, 1]),
                                            dde.geometry.Cuboid([0.5, 0.5, 0.5], [1, 1, 1])),
        {"geometry_type": "Custom", "geom_custom_shapes_json": notched_json},
        is_3d=True, expect_mixed=True,
    )
    _run_restore_ta_case(
        check, "restore_ta_cuboid_3d", "3D",
        lambda: dde.geometry.Cuboid([0, 0, 0], [1, 1, 1]),
        {"geometry_type": "Cuboid"},
        is_3d=True, expect_mixed=False,
    )

    # ── Part 2: generate_script()'s own live TA Surface plot (2D) ──────
    _run_live_ta_surface_case(
        check, "live_ta_disk_2d", "Disk",
        {"geom_center_x": 0.5, "geom_center_y": 0.5, "geom_radius": 0.4},
        expect_mixed=True,
    )
    _run_live_ta_surface_case(
        check, "live_ta_rect_2d", "Rectangle", {},
        expect_mixed=False,
    )

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 44 TA SURFACE MASKING TESTS PASSED")
    return 0


def test_round44_ta_surface_masking():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
