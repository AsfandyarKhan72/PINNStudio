#!/usr/bin/env python3
"""
Restore Model's Surface plot predicted and plotted over the full
rectangular BOUNDING BOX of a non-rectangular problem, instead of the
real domain. Reported by the user for a Custom-geometry "triangle with
cavity" (a Soomro_Paper-style steady 2D problem): after the two earlier
restore fixes (steady-state shape mismatch, then steady_state missing
from model_config.json) got "Restore && Visualize" running without
crashing, the resulting plot still showed the prediction smeared across
the whole rectangle the triangle happens to sit inside, not masked down
to the triangle (with its circular cavity cut out) itself -- and the user
asked for this to work for ANY geometry shape, not just rectangles.

Root cause, two layers (same pattern as the steady-state bug before it):

 1. SAVE side: codegen.py's `_model_config` dict (the sole writer of
    model_config.json) never saved geometry_type or any of the geom_*
    shape parameters (center/radius/semi-axes/angle/vertices/
    geom_custom_shapes_json) -- only the bounding box (x_min/x_max/...)
    ever made it into that file, for every geometry type, in every prior
    version. So even a perfect restore-side fix would have nothing to
    reconstruct the real shape from.

 2. RESTORE side: _build_restore_script() in main_window.py always built
    a plain Rectangle/Cuboid/Interval from the bounding box as `geom`,
    with no geometry-type dispatch at all, and never masked a prediction
    against the real domain -- every Surface plot (2D contourf, 3D
    6-face plot_surface) covered the full box regardless of shape.

Fix:
 - codegen.py's `_model_config` dict now also saves geometry_type,
   geom_center_x/y/z, geom_radius, geom_semi_major/minor, geom_angle,
   geom_triangle_vertices, geom_polygon_vertices, and
   geom_custom_shapes_json.
 - _build_restore_script() now reconstructs the REAL geometry from those
   fields by calling codegen.py's own `_clean_geom_line()` (the same
   reusable per-shape dispatcher "Export as DeepXDE Script" already
   uses, plus `_parse_vertex_list`/`_build_custom_geom_code` underneath
   it) instead of hardcoding a Rectangle/Cuboid/Interval -- so every
   geometry type (Disk/Ellipse/Triangle/Polygon/Sphere/Custom CSG combos,
   "that could be anything", per the user's request) is covered with no
   new per-shape code added here.
 - The Surface plot's 2D branch (contourf) and 3D branch (6-face
   plot_surface) both now mask the prediction against `geom.inside(...)`
   -- same NaN-for-outside-points convention codegen.py's own Surface
   plots already use for non-rectangular geometries -- so points inside
   the bounding box but outside the real domain are left blank instead
   of shown as if they were valid predictions.

Checks:
 - For a Triangle, a Custom "triangle with a circular cavity cut out"
   (matching the user's own problem), and a Disk, the generated restore
   script's own `geom = ...` line is NOT a plain Rectangle/Cuboid (i.e.
   the real shape was actually reconstructed, not just the bounding box).
 - Running the generated script against a REAL trained+saved checkpoint
   restores and plots cleanly (no crash, RESTORE_DONE printed,
   restored_plot.png saved) for 2D and 3D, steady and transient.
 - The inside/outside mask the script computes for masking (saved to a
   side-channel .npy file by this test, not part of the product code) is
   genuinely mixed -- some grid points inside the real domain, some
   outside -- proving the mask isn't a no-op that happens to mark
   everything "inside" (which would silently hide a masking bug: the
   bounding box of a Triangle/Disk/Custom shape by construction contains
   points outside the shape, e.g. its own corners).
 - A plain Rectangle/Cuboid config (regression) still restores and plots
   with every grid point marked "inside" -- masking a genuine box
   shouldn't blank out anything.
 - The real end-to-end pipeline (GUI Custom-geometry config ->
   _build_config() -> codegen.generate_script() -> real training
   subprocess -> real model_config.json -> _build_restore_script())
   round-trips geometry_type/geom_custom_shapes_json correctly.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_restore_geometry_masking.py
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


def _base_cfg(layers, problem_dim, steady, **geom_fields):
    cfg = {
        "layers": layers, "activation": "tanh",
        "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
        "problem_dim": problem_dim, "steady_state": steady, "loss_type": "MSE",
        "output_names": ",".join(f"u{i}" for i in range(layers[-1])),
        "geometry_type": "Rectangle",
        "geom_center_x": 0.5, "geom_center_y": 0.5, "geom_center_z": 0.5,
        "geom_radius": 0.3, "geom_semi_major": 0.4, "geom_semi_minor": 0.25,
        "geom_angle": 0.0,
        "geom_triangle_vertices": "0,0;1,0;0,1",
        "geom_polygon_vertices": "0,0;1,0;1,1;0,1",
        "geom_custom_shapes_json": "[]",
    }
    cfg.update(geom_fields)
    return cfg


class _DTypeSafeGeomForTraining:
    """Same float64-vs-network-dtype issue _DTypeSafeGeom (main_window.py's
    restore script / codegen.py) works around, applied here so this test's
    OWN training step (not the restore script under test) doesn't hit the
    unrelated "mat1 and mat2 must have the same dtype" crash for
    Disk/Triangle/Sphere geometries."""
    def __init__(self, geom):
        self._geom = geom
    def __getattr__(self, name):
        return getattr(self._geom, name)
    def random_points(self, n, random="pseudo"):
        import deepxde as dde, numpy as np
        return self._geom.random_points(n, random=random).astype(dde.config.real(np))
    def uniform_points(self, n, boundary=True):
        import deepxde as dde, numpy as np
        return self._geom.uniform_points(n, boundary=boundary).astype(dde.config.real(np))
    def random_boundary_points(self, n, random="pseudo"):
        import deepxde as dde, numpy as np
        return self._geom.random_boundary_points(n, random=random).astype(dde.config.real(np))
    def uniform_boundary_points(self, n):
        import deepxde as dde, numpy as np
        return self._geom.uniform_boundary_points(n).astype(dde.config.real(np))


def _train_and_save_model(tmpdir, dde_geom, is_3d, is_2d, steady, layers, tag):
    """Trains (1 iteration) and saves a real checkpoint for the given
    DeepXDE geometry object -- same approach as
    test_restore_steady_state.py, extended to take an actual (possibly
    non-rectangular) geometry rather than only Rectangle/Cuboid/Interval,
    so restore is tested against a real non-box-shaped model."""
    import deepxde as dde

    dde_geom = _DTypeSafeGeomForTraining(dde_geom)

    if steady:
        data = dde.data.PDE(dde_geom, lambda x, y: y[:, 0:1] * 0, [], num_domain=40, num_test=40)
    else:
        timedomain = dde.geometry.TimeDomain(0, 1)
        geomtime = dde.geometry.GeometryXTime(dde_geom, timedomain)
        data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=40,
                                 num_test=40, num_initial=5)

    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    model.train(iterations=1, display_every=1000)
    return model.save(os.path.join(tmpdir, tag))


def _run_script(script, tmpdir, tag):
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

    try:
        import deepxde as _dde_probe  # noqa: F401
        import torch as _torch_probe  # noqa: F401
    except ImportError:
        print("SKIPPED: deepxde/torch not installed in this environment.")
        return failures

    import deepxde as dde
    import numpy as np

    win = MainWindow()
    win._restore_viz_settings = dict(win._restore_viz_settings or {})
    win._restore_viz_settings["resolution"] = 24  # keep the grid small -- speed only

    # ── geom-line reconstruction: not a plain Rectangle/Cuboid ───────────
    tri_cfg = _base_cfg([2, 16, 16, 1], "2D", True,
                         geometry_type="Triangle", geom_triangle_vertices="0,0;1,0;0.5,1")
    tri_script = win._build_restore_script("/nonexistent.pt", tri_cfg, "adam", "Surface", 0, 5, "/tmp")
    check("dde.geometry.Triangle(" in tri_script,
          "Triangle config's restore script should build a real Triangle, not a Rectangle")
    check("dde.geometry.Rectangle(" not in tri_script.split("def pde")[0],
          "Triangle config's restore script should not fall back to Rectangle for geom")

    cavity_shapes = json.dumps([
        {"type": "Triangle", "params": {"vertices_text": "0,0;1,0;0.5,1"}},
        {"type": "Disk", "op": "subtract", "params": {"cx": 0.5, "cy": 0.35, "r": 0.15}},
    ])
    cavity_cfg = _base_cfg([2, 16, 16, 1], "2D", True,
                           geometry_type="Custom", geom_custom_shapes_json=cavity_shapes)
    cavity_script = win._build_restore_script("/nonexistent.pt", cavity_cfg, "adam", "Surface", 0, 5, "/tmp")
    check("CSGDifference(" in cavity_script and "dde.geometry.Triangle(" in cavity_script
          and "dde.geometry.Disk(" in cavity_script,
          "Custom 'triangle with cavity' config's restore script should build the real CSG combo")

    # ── real restore + masking, per shape ────────────────────────────────
    _shape_cases = [
        # (tag, dim, is_2d, is_3d, steady, layers, dde_geom_builder, cfg_geom_fields, expect_mixed_mask)
        ("triangle_2d_steady", "2D", True, False, True, [2, 16, 16, 1],
         lambda: dde.geometry.Triangle([0, 0], [1, 0], [0.5, 1]),
         {"geometry_type": "Triangle", "geom_triangle_vertices": "0,0;1,0;0.5,1"}, True),
        ("disk_2d_transient", "2D", True, False, False, [3, 16, 16, 1],
         lambda: dde.geometry.Disk([0.5, 0.5], 0.4),
         {"geometry_type": "Disk", "geom_center_x": 0.5, "geom_center_y": 0.5, "geom_radius": 0.4}, True),
        ("cavity_2d_steady", "2D", True, False, True, [2, 16, 16, 1],
         lambda: dde.geometry.CSGDifference(
             dde.geometry.Triangle([0, 0], [1, 0], [0.5, 1]), dde.geometry.Disk([0.5, 0.35], 0.15)),
         {"geometry_type": "Custom", "geom_custom_shapes_json": cavity_shapes}, True),
        # A Sphere's own bbox is exactly its circumscribing box, tangent to
        # every face at a single point -- a poor case for checking FACE
        # masking specifically (the inside region on any face is a single
        # point, which a discrete grid will almost never land on exactly).
        # A Custom 3D "cuboid with a corner notched out by a subtracted
        # sphere" has a genuinely mixed inside/outside region covering a
        # real fraction of 3 of its 6 bounding-box faces instead, and
        # still exercises the Custom/CSGDifference 3D code path.
        ("notched_cuboid_3d_steady", "3D", False, True, True, [3, 16, 16, 1],
         lambda: dde.geometry.CSGDifference(
             dde.geometry.Cuboid([0, 0, 0], [1, 1, 1]), dde.geometry.Sphere([0, 0, 0], 0.5)),
         {"geometry_type": "Custom", "geom_custom_shapes_json": json.dumps([
             {"type": "Cuboid", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
                                            "z_min": 0.0, "z_max": 1.0}},
             {"type": "Sphere", "op": "subtract", "params": {"cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 0.5}},
         ])}, True),
        ("rectangle_2d_steady", "2D", True, False, True, [2, 16, 16, 1],
         lambda: dde.geometry.Rectangle([0, 0], [1, 1]),
         {"geometry_type": "Rectangle"}, False),  # regression: box stays fully "inside"
    ]

    for tag, dim, is_2d, is_3d, steady, layers, geom_builder, geom_fields, expect_mixed in _shape_cases:
        with tempfile.TemporaryDirectory() as tmpdir:
            dde_geom = geom_builder()
            model_path = _train_and_save_model(tmpdir, dde_geom, is_3d, is_2d, steady, layers, tag)
            cfg = _base_cfg(layers, dim, steady, **geom_fields)
            script = win._build_restore_script(model_path, cfg, "adam", "Surface", 0, 5, tmpdir)

            mask_path = os.path.join(tmpdir, "mask.npy")
            if is_3d:
                diag = (f"\nimport numpy as _np_check\n"
                        f"_np_check.save({mask_path!r}, np.concatenate([m.ravel() for m in _face_inside3]))\n")
            else:
                diag = f"\nimport numpy as _np_check\n_np_check.save({mask_path!r}, _inside2d)\n"
            proc = _run_script(script + diag, tmpdir, tag)
            check(proc.returncode == 0,
                  f"restore script for {tag} should run cleanly, got exit {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
            check("RESTORE_DONE" in proc.stdout, f"restore script for {tag} should finish (RESTORE_DONE)")
            check(os.path.exists(os.path.join(tmpdir, "restored_plot.png")),
                  f"restore script for {tag} should save restored_plot.png")

            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                has_inside = bool(np.any(mask))
                has_outside = bool(np.any(~mask))
                check(has_inside, f"{tag}: mask should mark at least some grid points as inside the domain")
                if expect_mixed:
                    check(has_outside,
                          f"{tag}: mask should mark some bounding-box grid points OUTSIDE the real "
                          f"domain (a {geom_fields.get('geometry_type')} doesn't fill its own bounding "
                          f"box) -- all-inside means masking isn't actually discriminating anything")
                else:
                    check(not has_outside,
                          f"{tag}: a plain Rectangle fills its own bounding box -- masking should not "
                          f"blank out any grid point")

    # ── real end-to-end pipeline for a Custom geometry ───────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        from pinnstudio.core.codegen import generate_script

        win2 = MainWindow()
        win2.radio_2d.setChecked(True)
        win2._on_dim_changed()
        win2.steady_state_check.setChecked(True)
        # The default PDE text ("du_t - 0.4 * du_xx") is time-dependent --
        # steady_state=True drops du_t from the generated script entirely
        # (no time axis at all), so it must be replaced with a steady
        # (no-du_t) expression here, same as any user would have to do
        # when ticking "Steady-state" for a non-templated problem.
        if win2.pde_inputs:
            win2.pde_inputs[0].setText("du_xx + du_yy")
        win2._current_geometry_type = lambda: "Custom"
        if hasattr(win2, "geometry_type_combo"):
            idx = win2.geometry_type_combo.findText("Custom")
            if idx >= 0:
                win2.geometry_type_combo.setCurrentIndex(idx)
        win2._build_custom_geom_shapes_json = lambda: cavity_shapes
        for ph in win2.sched_phase_list:
            ph['iters'].setValue(1)
        win2.iter1_spin.setValue(1)
        win2.iter2_spin.setValue(0)
        win2.save_dir_input.setText(tmpdir)

        config = win2._build_config()
        config.geometry_type = "Custom"
        config.geom_custom_shapes_json = cavity_shapes

        train_script = generate_script(config)
        sp = os.path.join(tmpdir, "train_script.py")
        with open(sp, "w") as f:
            f.write(train_script)
        proc = subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=180)
        check(proc.returncode == 0,
              f"real training run for Custom geometry should complete, got exit {proc.returncode}:\n"
              f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

        mc_path = os.path.join(tmpdir, "solution_results", "model_config.json")
        if os.path.exists(mc_path):
            with open(mc_path) as f:
                saved_cfg = json.load(f)
            check(saved_cfg.get("geometry_type") == "Custom",
                  f"model_config.json from a real Custom-geometry run should save "
                  f"geometry_type=='Custom', got {saved_cfg.get('geometry_type')!r}")
            check(saved_cfg.get("geom_custom_shapes_json") == cavity_shapes,
                  f"model_config.json should round-trip geom_custom_shapes_json unchanged, got "
                  f"{saved_cfg.get('geom_custom_shapes_json')!r}")

            pt_candidates = [f for f in os.listdir(os.path.join(tmpdir, "solution_results")) if f.endswith(".pt")]
            if pt_candidates:
                pt_name = max(pt_candidates, key=lambda f: os.path.getmtime(
                    os.path.join(tmpdir, "solution_results", f)))
                pt_path = os.path.join(tmpdir, "solution_results", pt_name)
                restore_optimizer = "lbfgs" if "lbfgs" in pt_name else "adam"
                restore_script = win2._build_restore_script(pt_path, saved_cfg, restore_optimizer, "Surface", 0, 5, tmpdir)
                check("CSGDifference(" in restore_script,
                      "restoring the real Custom-geometry checkpoint should reconstruct the real CSG combo")
                rproc = _run_script(restore_script, tmpdir, "real_pipeline_custom_restore")
                check(rproc.returncode == 0,
                      f"restoring the real Custom-geometry checkpoint should work, got exit "
                      f"{rproc.returncode}:\n{rproc.stdout}\n{rproc.stderr}")
                check("RESTORE_DONE" in rproc.stdout,
                      f"real-pipeline Custom-geometry restore should finish (RESTORE_DONE)")

    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL RESTORE GEOMETRY-MASKING TESTS PASSED")
    return 0


def test_restore_geometry_masking():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
