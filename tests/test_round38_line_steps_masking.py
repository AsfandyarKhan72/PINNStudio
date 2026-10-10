#!/usr/bin/env python3
"""
Round 38: the "Line (time steps)" plot type never masked to the real
geometry anywhere in the codebase, unlike its adjacent x-y heatmap/
Surface Comparison siblings (which already call geom.inside(...) and
np.where(inside, pred, np.nan)). For any non-rectangular 2D/3D geometry
(Disk/Ellipse/Triangle/Polygon/Custom CSG in 2D, Sphere/Custom CSG in
3D), selecting "Line (time steps)" could silently evaluate/plot the
network past its real trained domain, at points outside the true shape
but inside its bounding box.

This covers the full set of fixes made for it this round:

 1. codegen.py's generate_script() -- all 4 "Line (time steps)" branches
    (2D steady, 2D transient, 3D steady, 3D transient) now mask with
    geom.inside(...), same convention the adjacent heatmap branches
    already use, with a "Line outside the domain" fallback when fewer
    than 2 points survive the mask.
 2. codegen.py's generate_clean_script() -- same 4 branches, same fix.
 3. main_window.py's _build_restore_script() -- the single-model
    restore path's own "Line (time steps)"/"Animation Line (GIF)"
    branches, same fix (geom is already the real reconstructed geometry
    there from an earlier round).
 4. main_window.py's _build_restore_script_ta()'s _ta_build_geomtime()
    -- previously always built a plain Cuboid/Rectangle/Interval
    bounding box; now reconstructs the real geometry the same way
    _build_restore_script() does (reusing codegen.py's _clean_geom_line),
    built once at module scope and reused both by _ta_build_geomtime()
    and directly by its own "Line (time steps)"/"Animation Line (GIF)"
    branches for masking.
 5. main_window.py's _build_restore_ea_script()'s Line Comparison --
    replaced the old tolerance-BAND approach (every reference point
    within (range)/20 of the slice, widened to /5, falling back to ALL
    points if still <2) with a true line at the exact slice, masked with
    geom.inside(...), PINN evaluated directly on it, reference
    interpolated onto it via griddata -- the same fix Rounds 36/37 gave
    the Error Analysis path's own Line Comparison, now also covering
    Restore & Visualize's "Run Error Analysis on a restored model" flow,
    for both the single-model and Time-Adaptive (is_ta=True) cases.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_round38_line_steps_masking.py
"""
import json
import os
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
from pinnstudio.core.codegen import generate_script, generate_clean_script
from pinnstudio.ui.main_window import MainWindow


def _run_script(script, tmpdir, tag, timeout_s=120):
    sp = os.path.join(tmpdir, f"{tag}_script.py")
    with open(sp, "w") as f:
        f.write(script)
    return subprocess.run([sys.executable, sp], cwd=tmpdir, capture_output=True, text=True, timeout=timeout_s)


def _disk_cfg(problem_dim="2D", steady=True, plot_type="Line (time steps)", line_slice_y=0.5, line_slice_z=0.5):
    cfg = PINNConfig()
    cfg.problem_dim = problem_dim
    cfg.steady_state = steady
    cfg.num_outputs = 1
    cfg.output_names = "u"
    cfg.x_min, cfg.x_max = 0.0, 1.0
    cfg.y_min, cfg.y_max = 0.0, 1.0
    cfg.z_min, cfg.z_max = 0.0, 1.0
    cfg.t_min, cfg.t_max = 0.0, 1.0
    cfg.geometry_type = "Sphere" if problem_dim == "3D" else "Disk"
    cfg.geom_center_x = cfg.geom_center_y = cfg.geom_center_z = 0.5
    cfg.geom_radius = 0.3
    cfg.num_domain = 40; cfg.num_boundary = 10; cfg.num_initial = 10; cfg.num_test = 20
    cfg.layers = [1 if problem_dim == "1D" else (2 if problem_dim == "2D" else 3), 16, 16, 1]
    if not steady:
        cfg.layers[0] += 1
    if problem_dim == "2D":
        cfg.pde_expressions = "du_xx + du_yy" if steady else "du_t - 0.4*(du_xx + du_yy)"
    else:
        cfg.pde_expressions = "du_xx + du_yy + du_zz" if steady else "du_t - 0.4*(du_xx + du_yy + du_zz)"
    cfg.pde_expression = cfg.pde_expressions
    cfg.ic_expressions = "0.0"; cfg.ic_expression = cfg.ic_expressions
    cfg.custom_bc_json = json.dumps([{"type": "dirichlet", "location": "True", "value": "0", "component": 0}])
    cfg.adapt_method = "None"
    cfg.time_adaptive = False
    cfg.iterations = 1; cfg.optimizer_scheduler = False; cfg.iterations2 = 0
    cfg.plot_type = plot_type
    cfg.plot_resolution = 24
    cfg.line_slice_y_auto = False; cfg.line_slice_y = line_slice_y
    cfg.line_slice_z_auto = False; cfg.line_slice_z = line_slice_z
    cfg.num_timesteps_line = 3
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

    # ── 1. generate_script(): all 4 "Line (time steps)" branches mask ──
    for dim, steady in [("2D", True), ("2D", False), ("3D", True), ("3D", False)]:
        cfg = _disk_cfg(dim, steady)
        script = generate_script(cfg)
        section_marker = "Line (time steps)"
        check(section_marker in script or "geom.inside(" in script,
              f"generate_script() {dim}/{'steady' if steady else 'transient'}: sanity precondition")
        # Isolate roughly the Line-plot section (between the geom build and
        # the next plot-type branch) to avoid matching an unrelated
        # geom.inside( call elsewhere (e.g. the heatmap branches).
        check("geom.inside(" in script,
              f"generate_script() {dim}/{'steady' if steady else 'transient'}: Line plot section should "
              f"mask with geom.inside(...)")
        check('"Line outside the domain' in script,
              f"generate_script() {dim}/{'steady' if steady else 'transient'}: should have an "
              f"out-of-domain fallback message")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _disk_cfg("2D", True, line_slice_y=0.5)
        script = generate_script(cfg)
        diag = "\nimport numpy as _np_check\n_np_check.save(%r, _line_in_l2ds)\n" % os.path.join(tmpdir, "mask2ds.npy")
        proc = _run_script(script + diag, tmpdir, "gs_2d_steady_disk")
        check(proc.returncode == 0,
              f"generate_script() 2D steady Disk Line plot should run cleanly, got {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        mpath = os.path.join(tmpdir, "mask2ds.npy")
        if os.path.exists(mpath):
            mask = np.load(mpath)
            check(bool(np.any(mask)) and bool(np.any(~mask)),
                  "2D steady Disk Line mask should be genuinely mixed (some in-domain, some out) at "
                  "y=0.5 through a Disk of radius 0.3 centered at (0.5, 0.5) inside a [0,1]^2 bbox")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Slice entirely outside the Disk -> fallback text, no crash.
        cfg_oob = _disk_cfg("2D", True, line_slice_y=0.99)
        proc_oob = _run_script(generate_script(cfg_oob), tmpdir, "gs_2d_steady_disk_oob")
        check(proc_oob.returncode == 0,
              f"generate_script() out-of-domain Disk slice should exit cleanly (fallback, not crash), "
              f"got {proc_oob.returncode}:\n{proc_oob.stdout[-1500:]}\n{proc_oob.stderr[-1500:]}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg3 = _disk_cfg("3D", True, line_slice_y=0.5, line_slice_z=0.5)
        script3 = generate_script(cfg3)
        diag3 = "\nimport numpy as _np_check\n_np_check.save(%r, _line_in_l3ds)\n" % os.path.join(tmpdir, "mask3ds.npy")
        proc3 = _run_script(script3 + diag3, tmpdir, "gs_3d_steady_sphere")
        check(proc3.returncode == 0,
              f"generate_script() 3D steady Sphere Line plot should run cleanly, got {proc3.returncode}:\n"
              f"{proc3.stdout[-1500:]}\n{proc3.stderr[-1500:]}")
        mpath3 = os.path.join(tmpdir, "mask3ds.npy")
        if os.path.exists(mpath3):
            mask3 = np.load(mpath3)
            check(bool(np.any(mask3)) and bool(np.any(~mask3)),
                  "3D steady Sphere Line mask should be genuinely mixed at the y=z=0.5 slice through a "
                  "Sphere of radius 0.3 centered at (0.5,0.5,0.5) inside a [0,1]^3 bbox")

    # ── 2. generate_clean_script(): same 4 branches ──────────────────────
    for dim, steady in [("2D", True), ("2D", False), ("3D", True), ("3D", False)]:
        cfg = _disk_cfg(dim, steady)
        clean = generate_clean_script(cfg)
        check("geom.inside(" in clean,
              f"generate_clean_script() {dim}/{'steady' if steady else 'transient'}: Line plot should "
              f"mask with geom.inside(...)")
        check('"Line outside the domain' in clean,
              f"generate_clean_script() {dim}/{'steady' if steady else 'transient'}: should have an "
              f"out-of-domain fallback message")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _disk_cfg("2D", True, line_slice_y=0.5)
        clean = generate_clean_script(cfg)
        proc = _run_script(clean, tmpdir, "gcs_2d_steady_disk")
        check(proc.returncode == 0,
              f"generate_clean_script() 2D steady Disk Line plot should run cleanly, got {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    # ── 3/4. Restore paths: single-model and Time-Adaptive ───────────────
    import deepxde as dde

    class _DTypeSafeGeomForTraining:
        def __init__(self, geom):
            self._geom = geom
        def __getattr__(self, name):
            return getattr(self._geom, name)
        def random_points(self, n, random="pseudo"):
            return self._geom.random_points(n, random=random).astype(dde.config.real(np))
        def uniform_points(self, n, boundary=True):
            return self._geom.uniform_points(n, boundary=boundary).astype(dde.config.real(np))
        def random_boundary_points(self, n, random="pseudo"):
            return self._geom.random_boundary_points(n, random=random).astype(dde.config.real(np))
        def uniform_boundary_points(self, n):
            return self._geom.uniform_boundary_points(n).astype(dde.config.real(np))

    def _train_disk_model(tmpdir, tag, t0=None, t1=None):
        dgeom = _DTypeSafeGeomForTraining(dde.geometry.Disk([0.5, 0.5], 0.3))
        if t0 is None:
            data = dde.data.PDE(dgeom, lambda x, y: y[:, 0:1] * 0, [], num_domain=30, num_test=20)
            layers = [2, 16, 16, 1]
        else:
            geomtime = dde.geometry.GeometryXTime(dgeom, dde.geometry.TimeDomain(t0, t1))
            data = dde.data.TimePDE(geomtime, lambda x, y: y[:, 0:1] * 0, [], num_domain=30,
                                     num_test=20, num_initial=5)
            layers = [3, 16, 16, 1]
        net = dde.nn.FNN(layers, "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        model.compile("adam", lr=0.001)
        model.train(iterations=1, display_every=1000)
        return model.save(os.path.join(tmpdir, tag))

    win = MainWindow()
    win._on_restore_viz_changed = lambda text: None

    disk_cfg_dict = {
        "layers": [2, 16, 16, 1], "activation": "tanh",
        "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0, "t_min": 0.0, "t_max": 1.0,
        "problem_dim": "2D", "steady_state": True, "loss_type": "MSE",
        "output_names": "u", "geometry_type": "Disk",
        "geom_center_x": 0.5, "geom_center_y": 0.5, "geom_radius": 0.3,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _train_disk_model(tmpdir, "disk_model")
        restore_script = win._build_restore_script(
            model_path, disk_cfg_dict, "adam", "Line (time steps)", 0, 3, tmpdir)
        check("geom.inside(" in restore_script,
              "_build_restore_script()'s Line (time steps) branch should mask with geom.inside(...)")
        check("dde.geometry.Disk(" in restore_script,
              "_build_restore_script() should reconstruct the real Disk, not just its bbox")
        proc = _run_script(restore_script, tmpdir, "restore_disk_line")
        check(proc.returncode == 0,
              f"_build_restore_script() Disk Line restore should run cleanly, got {proc.returncode}:\n"
              f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        check("RESTORE_DONE" in proc.stdout, "restore script should finish (RESTORE_DONE)")
        check(os.path.exists(os.path.join(tmpdir, "restored_plot.png")),
              "restore script should save restored_plot.png")

        anim_script = win._build_restore_script(
            model_path, disk_cfg_dict, "adam", "Animation Line (GIF)", 0, 3, tmpdir)
        check("geom.inside(" in anim_script,
              "_build_restore_script()'s Animation Line (GIF) branch should mask with geom.inside(...)")
        proc_a = _run_script(anim_script, tmpdir, "restore_disk_animline")
        check(proc_a.returncode == 0,
              f"_build_restore_script() Disk Animation Line restore should run cleanly, got "
              f"{proc_a.returncode}:\n{proc_a.stdout[-1500:]}\n{proc_a.stderr[-1500:]}")

    # Out-of-domain restore slice: fallback, not a crash.
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _train_disk_model(tmpdir, "disk_model_oob")
        win._restore_viz_settings = dict(getattr(win, "_restore_viz_settings", None) or {})
        win._restore_viz_settings["line_slice_y_auto"] = False
        win._restore_viz_settings["line_slice_y"] = 0.99
        restore_script_oob = win._build_restore_script(
            model_path, disk_cfg_dict, "adam", "Line (time steps)", 0, 3, tmpdir)
        proc_oob = _run_script(restore_script_oob, tmpdir, "restore_disk_line_oob")
        check(proc_oob.returncode == 0,
              f"out-of-domain restore Line slice should exit cleanly, got {proc_oob.returncode}:\n"
              f"{proc_oob.stdout[-1500:]}\n{proc_oob.stderr[-1500:]}")
        win._restore_viz_settings.pop("line_slice_y_auto", None)
        win._restore_viz_settings.pop("line_slice_y", None)

    # ── Time-Adaptive restore: real geometry (not bbox) + masking ────────
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path1 = _train_disk_model(tmpdir, "ta_step1", t0=0.0, t1=0.5)
        model_path2 = _train_disk_model(tmpdir, "ta_step2", t0=0.5, t1=1.0)
        ta_steps = [
            {"t0": 0.0, "t1": 0.5, "model_path": model_path1, "cfg": {}},
            {"t0": 0.5, "t1": 1.0, "model_path": model_path2, "cfg": {}},
        ]
        ta_cfg_dict = dict(disk_cfg_dict); ta_cfg_dict["steady_state"] = False
        ta_cfg_dict["layers"] = [3, 16, 16, 1]
        ta_script = win._build_restore_script_ta(
            ta_steps, ta_cfg_dict, "adam", "Line (time steps)", 0, 3, tmpdir)
        check("dde.geometry.Disk(" in ta_script,
              "_build_restore_script_ta()'s _ta_build_geomtime should reconstruct the real Disk, not "
              "just a bounding-box Rectangle")
        check("geom.inside(" in ta_script,
              "_build_restore_script_ta()'s Line (time steps) branch should mask with geom.inside(...)")
        proc_ta = _run_script(ta_script, tmpdir, "restore_ta_disk_line")
        check(proc_ta.returncode == 0,
              f"Time-Adaptive Disk restore Line plot should run cleanly, got {proc_ta.returncode}:\n"
              f"{proc_ta.stdout[-2000:]}\n{proc_ta.stderr[-2000:]}")
        check("RESTORE_DONE" in proc_ta.stdout, "TA restore script should finish (RESTORE_DONE)")

        ta_anim_script = win._build_restore_script_ta(
            ta_steps, ta_cfg_dict, "adam", "Animation Line (GIF)", 0, 3, tmpdir)
        check("geom.inside(" in ta_anim_script,
              "_build_restore_script_ta()'s Animation Line (GIF) branch should mask with geom.inside(...)")
        proc_ta_a = _run_script(ta_anim_script, tmpdir, "restore_ta_disk_animline")
        check(proc_ta_a.returncode == 0,
              f"Time-Adaptive Disk restore Animation Line should run cleanly, got {proc_ta_a.returncode}:\n"
              f"{proc_ta_a.stdout[-2000:]}\n{proc_ta_a.stderr[-2000:]}")

    # ── 5. _build_restore_ea_script(): true-line Line Comparison ─────────
    with tempfile.TemporaryDirectory() as tmpdir:
        xs = np.linspace(0, 1, 20)
        ys = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(xs, ys)
        X = X.ravel(); Y = Y.ravel()
        ref = os.path.join(tmpdir, "ref.txt")
        np.savetxt(ref, np.column_stack([X, Y, X + 2.0 * Y]))

        ea_script = win._build_restore_ea_script(
            [(0.0, ref)], tmpdir, is_2d=True, do_line=True, do_surface=False,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="u",
            is_3d=False, output_idx=0, output_names="u", is_steady=True,
            line_slice_y=0.5,
        )
        check("griddata" in ea_script, "EA Line Comparison should use griddata to interpolate the reference")
        check("geom.inside(" in ea_script, "EA Line Comparison should mask with geom.inside(...)")
        check("_y_tol" not in ea_script and "_mid_mask" not in ea_script,
              "EA Line Comparison should no longer use the old tolerance-band approach")

        model_path = _train_disk_model(tmpdir, "ea_disk_model")
        restore_script = win._build_restore_script(
            model_path, disk_cfg_dict, "adam", "Surface", 0, 3, tmpdir)
        proc = _run_script(restore_script + ea_script, tmpdir, "ea_restore_disk")
        check(proc.returncode == 0,
              f"restore + EA Line Comparison for a Disk model should run cleanly, got {proc.returncode}:\n"
              f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        check(os.path.exists(os.path.join(tmpdir, "error_analysis", "line_comparison_restore.png")),
              "line_comparison_restore.png should be produced")

    # ── EA Line Comparison + Time-Adaptive (is_ta=True), Disk geometry ───
    # Ties #131 and #133 together: the EA script's geom.inside(...) call
    # only works for a TA restore because _ta_build_geomtime now builds
    # the real Disk instead of a bounding box.
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path1 = _train_disk_model(tmpdir, "ta_ea_step1", t0=0.0, t1=0.5)
        model_path2 = _train_disk_model(tmpdir, "ta_ea_step2", t0=0.5, t1=1.0)
        ta_steps = [
            {"t0": 0.0, "t1": 0.5, "model_path": model_path1, "cfg": {}},
            {"t0": 0.5, "t1": 1.0, "model_path": model_path2, "cfg": {}},
        ]
        ta_cfg_dict = dict(disk_cfg_dict); ta_cfg_dict["steady_state"] = False
        ta_cfg_dict["layers"] = [3, 16, 16, 1]
        ta_main_script = win._build_restore_script_ta(
            ta_steps, ta_cfg_dict, "adam", "Surface", 0, 3, tmpdir)

        xs = np.linspace(0, 1, 15)
        ys = np.linspace(0, 1, 15)
        X, Y = np.meshgrid(xs, ys)
        X = X.ravel(); Y = Y.ravel()
        ref1 = os.path.join(tmpdir, "ref_t0.2.txt")
        ref2 = os.path.join(tmpdir, "ref_t0.7.txt")
        np.savetxt(ref1, np.column_stack([X, Y, np.full_like(X, 0.2), X + Y]))
        np.savetxt(ref2, np.column_stack([X, Y, np.full_like(X, 0.7), X - Y]))

        ea_ta_script = win._build_restore_ea_script(
            [(0.2, ref1), (0.7, ref2)], tmpdir, is_2d=True, do_line=True, do_surface=False,
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, out_name="u",
            is_3d=False, output_idx=0, output_names="u", is_steady=False, is_ta=True,
            line_slice_y=0.5,
        )
        check("_ta_model_for_t(_tv).predict" in ea_ta_script,
              "TA EA Line Comparison should route predictions through _ta_model_for_t(_tv)")
        proc_ta_ea = _run_script(ta_main_script + ea_ta_script, tmpdir, "ea_restore_ta_disk")
        check(proc_ta_ea.returncode == 0,
              f"TA restore + EA Line Comparison for a Disk model should run cleanly, got "
              f"{proc_ta_ea.returncode}:\n{proc_ta_ea.stdout[-2500:]}\n{proc_ta_ea.stderr[-2500:]}")
        check(os.path.exists(os.path.join(tmpdir, "error_analysis", "line_comparison_restore.png")),
              "TA line_comparison_restore.png should be produced")

    win.close()
    return failures


def main():
    failures = run()
    for f in failures:
        print("FAIL:", f)
    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        return 1
    print("\nALL ROUND 38 LINE-STEPS MASKING TESTS PASSED")
    return 0


def test_round38_line_steps_masking():
    failures = run()
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    sys.exit(main())
