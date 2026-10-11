#!/usr/bin/env python3
"""
Round 46 -- codebase cleanup, phase 9 (audit §2e follow-up): the
"Surface Animation (GIF)" / "Animation Surface (GIF)" plot never masked
its prediction to the real (possibly non-rectangular/CSG) geometry, in ANY
of the three places it's generated -- unlike the static Surface plot,
which masks everywhere as of Round 44:

 1. codegen.py's _restore_anim_surface_code (shared by main_window.py's
    _build_restore_script and _build_restore_script_ta since Round 43):
    3D bounding-box faces and 2D contourf grid both unmasked.
 2. codegen.py's generate_script() live "Surface Animation (GIF)": same
    3D face / 2D grid approach, unmasked.
 3. codegen.py's generate_clean_script() exported "Surface Animation
    (GIF)": 2D grid and 3D z-mid-plane grid unmasked, even though the
    exported script's own static 2D/3D snapshot plots right below it
    already mask with geom.inside(...).

Fix: the same geom.inside(...) idiom the static Surface plots already use
-- NaN out-of-domain grid points before contourf (2D, and the exported
script's 3D z-mid-plane), alpha-0 out-of-domain bounding-box face quads
before plot_surface (3D), with auto-range min/max taken over in-domain
values only (np.nanmin/np.nanmax for the NaN-masked frames).

Verified the same way as Round 44: real exec-level DeepXDE subprocess
runs with a side-channel mask dump, asserting the mask is genuinely MIXED
for a non-rectangular geometry (Disk in 2D; a Custom "notched cuboid" in
3D -- a Sphere's bounding box is tangent-only, see Round 44's notes) and
fully inside for a plain Rectangle/Cuboid regression case.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round46_anim_surface_masking.py
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

_DISK = {"geom_center_x": 0.5, "geom_center_y": 0.5, "geom_radius": 0.4}


def _notched_json(z_lo):
    return json.dumps([
        {"type": "Cuboid", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}},
        {"type": "Cuboid", "params": {"x_min": 0.5, "x_max": 1.0, "y_min": 0.5, "y_max": 1.0, "z_min": z_lo, "z_max": 1.0},
         "op": "subtract"},
    ])


def _train_and_save(tmpdir, dde_geom, t0, t1, layers, tag):
    import deepxde as dde

    geomtime = dde.geometry.GeometryXTime(dde_geom, dde.geometry.TimeDomain(t0, t1))
    data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=40,
                             num_test=40, num_initial=5)
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1, display_every=1000)
    return model.save(os.path.join(tmpdir, tag))


def _run_script(script, tmpdir, tag, timeout_s=300):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def _check_mask(check, tag, mask_path, expect_mixed):
    import numpy as np

    check(os.path.exists(mask_path), f"{tag}: mask dump should have been written")
    if not os.path.exists(mask_path):
        return
    mask = np.load(mask_path)
    check(bool(np.any(mask)), f"{tag}: mask should mark some grid points inside the domain")
    if expect_mixed:
        check(bool(np.any(~mask)),
              f"{tag}: mask should mark some bounding-box grid points OUTSIDE the real domain")
    else:
        check(not bool(np.any(~mask)),
              f"{tag}: a plain Rectangle/Cuboid fills its own bounding box -- nothing should be masked")


def _restore_case(check, tag, dim, geom_builder, geom_fields, time_adaptive, expect_mixed):
    is_3d = dim == "3D"
    win = MainWindow()
    win._restore_viz_settings = dict(win._restore_viz_settings or {})
    win._restore_viz_settings["n_steps_anim"] = 3
    win._restore_viz_settings[win._viz_steps_key("Animation Surface (GIF)")] = 3
    layers = [4, 16, 1] if is_3d else [3, 16, 1]
    cfg = {"layers": layers, "activation": "tanh", "x_min": 0.0, "x_max": 1.0,
           "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
           "problem_dim": dim, "steady_state": False, "loss_type": "MSE", "output_names": "u"}
    cfg.update(geom_fields)
    with tempfile.TemporaryDirectory() as tmpdir:
        geom = geom_builder()
        if time_adaptive:
            steps = []
            for i, (t0, t1) in enumerate([(0.0, 0.5), (0.5, 1.0)]):
                steps.append({"t0": t0, "t1": t1, "model_path": _train_and_save(tmpdir, geom, t0, t1, layers, f"s{i}"),
                              "cfg": {"layers": layers, "activation": "tanh", "loss_type": "MSE"}})
            script = win._build_restore_script_ta(steps, cfg, "adam", "Animation Surface (GIF)", 0, 3, tmpdir)
        else:
            mp = _train_and_save(tmpdir, geom, 0.0, 1.0, layers, "m")
            script = win._build_restore_script(mp, cfg, "adam", "Animation Surface (GIF)", 0, 3, tmpdir)
        check("geom.inside(" in script, f"{tag}: Animation Surface restore script should mask with geom.inside")
        mask_path = os.path.join(tmpdir, "mask.npy")
        var = "np.concatenate([m.ravel() for m in _face_inside3a])" if is_3d else "_inside2da"
        proc = _run_script(script + f"\nnp.save({mask_path!r}, {var})\n", tmpdir, tag)
        check(proc.returncode == 0,
              f"{tag}: restore should run cleanly, got {proc.returncode}:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check("RESTORE_DONE" in proc.stdout, f"{tag}: restore should finish (RESTORE_DONE)")
        check(os.path.exists(os.path.join(tmpdir, "restored_animation.gif")),
              f"{tag}: restore should save restored_animation.gif")
        _check_mask(check, tag, mask_path, expect_mixed)


def _live_cfg(save_dir, dim, geometry_type, **geom_fields):
    from pinnstudio.core.config import PINNConfig

    cfg = PINNConfig()
    cfg.problem_dim = dim; cfg.steady_state = False
    cfg.num_outputs = 1; cfg.output_names = "u"
    cfg.geometry_type = geometry_type
    for k, v in geom_fields.items():
        setattr(cfg, k, v)
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.y_min, cfg.y_max = 0.0, 1.0
    cfg.z_min, cfg.z_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.num_domain = 80; cfg.num_boundary = 20; cfg.num_initial = 20; cfg.num_test = 40
    if dim == "3D":
        cfg.layers = [4, 16, 1]
        cfg.pde_expression = "du_t - 0.4*(du_xx + du_yy + du_zz)"
    else:
        cfg.layers = [3, 16, 1]
        cfg.pde_expression = "du_t - 0.4*(du_xx + du_yy)"
    cfg.pde_expressions = cfg.pde_expression
    cfg.ic_expression = "0.0"; cfg.ic_expressions = cfg.ic_expression
    cfg.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    cfg.adapt_method = "None"
    cfg.time_adaptive = False
    cfg.iterations = 2; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.save_dir = save_dir
    cfg.plot_type = "Surface Animation (GIF)"
    cfg.plot_resolution = 20
    cfg.num_timesteps_anim = 3
    return cfg


def _live_case(check, tag, generator, dim, geometry_type, geom_fields, mask_var, expect_mixed):
    from pinnstudio.core import codegen

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = os.path.join(tmpdir, "out")
        os.makedirs(save_dir, exist_ok=True)
        cfg = _live_cfg(save_dir, dim, geometry_type, **geom_fields)
        script = getattr(codegen, generator)(cfg)
        mask_name = "_face_inside3a" if "_face_inside3a" in mask_var else mask_var
        check(mask_name in script, f"{tag}: {generator} Surface Animation should compute {mask_name}")
        mask_path = os.path.join(tmpdir, "mask.npy")
        proc = _run_script(script + f"\nimport numpy as _np_chk\n_np_chk.save({mask_path!r}, {mask_var})\n",
                           tmpdir, tag)
        check(proc.returncode == 0,
              f"{tag}: {generator} run should exit cleanly, got {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        gifs = [os.path.join(r, f) for r, _, fs in os.walk(tmpdir) for f in fs if f.endswith(".gif")]
        check(bool(gifs), f"{tag}: {generator} run should save a solution GIF")
        _check_mask(check, tag, mask_path, expect_mixed)


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

    disk = lambda: dde.geometry.Disk([0.5, 0.5], 0.4)  # noqa: E731
    notched = lambda: dde.geometry.CSGDifference(  # noqa: E731
        dde.geometry.Cuboid([0, 0, 0], [1, 1, 1]), dde.geometry.Cuboid([0.5, 0.5, 0.5], [1, 1, 1]))
    cuboid = lambda: dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])  # noqa: E731

    # ── Part 1: restore builders, through _restore_anim_surface_code ────
    _restore_case(check, "restore_single_disk_2d", "2D", disk, {"geometry_type": "Disk", **_DISK},
                  time_adaptive=False, expect_mixed=True)
    _restore_case(check, "restore_ta_notched_3d", "3D", notched,
                  {"geometry_type": "Custom", "geom_custom_shapes_json": _notched_json(0.5)},
                  time_adaptive=True, expect_mixed=True)
    _restore_case(check, "restore_single_cuboid_3d", "3D", cuboid, {"geometry_type": "Cuboid"},
                  time_adaptive=False, expect_mixed=False)

    # ── Part 2: generate_script()'s live Surface Animation (GIF) ────────
    _live_case(check, "live_disk_2d", "generate_script", "2D", "Disk", _DISK,
               "_inside2d_gif", expect_mixed=True)
    _live_case(check, "live_notched_3d", "generate_script", "3D", "Custom",
               {"geom_custom_shapes_json": _notched_json(0.5)},
               "np.concatenate([m.ravel() for m in _face_inside3a])", expect_mixed=True)

    # ── Part 3: generate_clean_script()'s exported Surface Animation ────
    _live_case(check, "export_disk_2d", "generate_clean_script", "2D", "Disk", _DISK,
               "inside_a", expect_mixed=True)
    # z mid-plane is 0.5, so the notch must start below it to cut that plane.
    _live_case(check, "export_notched_3d", "generate_clean_script", "3D", "Custom",
               {"geom_custom_shapes_json": _notched_json(0.4)}, "inside_a", expect_mixed=True)
    _live_case(check, "export_rect_2d", "generate_clean_script", "2D", "Rectangle", {},
               "inside_a", expect_mixed=False)

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 46 ANIMATION-SURFACE MASKING TESTS PASSED")
    return 0


def test_round46_anim_surface_masking():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
