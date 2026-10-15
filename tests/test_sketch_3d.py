#!/usr/bin/env python3
"""3D Sketch: boxes and spheres drawn on x–y / x–z / y–z views, shown as
the slice at a chosen depth, and boundary conditions put on faces.

Checks the faces of a few 3D sketches and that each face's Where: picks
it alone; that every boundary point DeepXDE samples on the CSG geometry is
picked by one face; the 3D panel (drawing by mouse and by typed numbers,
switching views, faces from the canvas and from the list, BC rows, config
round-trip); and a Laplace problem (u = x + y + z) on a sketched solid with
Dirichlet faces and a Neumann spherical hole, trained by the live and the
exported script.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sketch_3d.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.core.codegen import generate_clean_script, generate_script
from pinnstudio.core.config import PINNConfig
from pinnstudio.ui.main_window import MainWindow
from pinnstudio.ui.sketch3d import face_in_view, faces3, section_shapes, where_matches3
from pinnstudio.ui.sketch_edges import join_where, split_where


def C(a, b, c, d, e, f, op=None):
    s = {"type": "Cuboid", "params": {"x_min": a, "x_max": b, "y_min": c, "y_max": d, "z_min": e, "z_max": f}}
    if op:
        s["op"] = op
    return s


def S(x, y, z, r, op=None):
    s = {"type": "Sphere", "params": {"cx": x, "cy": y, "cz": z, "r": r}}
    if op:
        s["op"] = op
    return s


CASES = {
    "box, hole, box": ([C(0, 2, 0, 1, 0, 1), S(1, 0.5, 0.5, 0.3, "subtract"), C(2, 3, 0, 0.5, 0, 0.5, "union")], 12),
    "sphere cut by a box": ([S(0, 0, 0, 1), C(0, 2, -2, 2, -2, 2, "subtract")], 2),
    "intersection": ([C(-1, 1, -1, 1, -1, 1), S(0, 0, 0, 1.2, "intersect")], 7),
}


def _settle(secs=0.3):
    t = time.time()
    while time.time() - t < secs:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        _app.processEvents()


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("FAIL:", msg)

    for name, (shapes, n) in CASES.items():
        F = faces3(shapes)
        check(len(F) == n, f"{name}: {len(F)} faces, expected {n}")
        for i, f in enumerate(F):
            hits = [j for j, g in enumerate(F) if where_matches3(g["expr"], f["mid"])]
            check(hits == [i], f"{name}: F{i + 1} ({f['expr']}) picked by {hits}")
    sec = section_shapes(CASES["box, hole, box"][0], "xy", 0.75)
    check([s["type"] for s in sec] == ["Rectangle", "Disk", "Empty"], f"slice at z = 0.75: {sec}")
    check(abs(sec[1]["params"]["r"] - (0.3 ** 2 - 0.25 ** 2) ** 0.5) < 1e-9 and sec[2]["op"] == "union", "slice of the sphere, empty box keeps its op")
    F = faces3(CASES["box, hole, box"][0])
    x2 = next(f for f in F if f["desc"].startswith("x = 2 "))
    tr = face_in_view(x2, CASES["box, hole, box"][0], "xy", 0.25)
    check(len(tr) == 1 and abs(tr[0][0][1] - 0.5) < 1e-6 and abs(tr[0][1][1] - 1) < 1e-6,
          f"x = 2 face is covered by the second box below y = 0.5 at z = 0.25: {tr}")
    check(face_in_view(next(f for f in F if f["desc"].startswith("z = 1 ")), CASES["box, hole, box"][0], "xy", 0.25)
          == [], "a face parallel to the view has no trace")

    try:
        import numpy as np
        import deepxde as dde
    except Exception as exc:  # pragma: no cover
        print("deepxde not available, skipping:", exc)
    else:
        def geom(sh):
            p = sh["params"]
            if sh["type"] == "Cuboid":
                return dde.geometry.Cuboid([p["x_min"], p["y_min"], p["z_min"]], [p["x_max"], p["y_max"], p["z_max"]])
            return dde.geometry.Sphere([p["cx"], p["cy"], p["cz"]], p["r"])

        for name, (shapes, _n) in CASES.items():
            g = geom(shapes[0])
            for sh in shapes[1:]:
                h = geom(sh)
                g = g - h if sh["op"] == "subtract" else (g & h if sh["op"] == "intersect" else g | h)
            F = faces3(shapes)
            X = g.random_boundary_points(1500).astype(np.float32)
            X = X[g.on_boundary(X)]
            bad = [x.tolist() for x in X if sum(where_matches3(f["expr"], x) for f in F) != 1]
            check(len(X) > 1000 and len(bad) <= 2, f"{name}: {len(bad)} DeepXDE boundary points not on exactly one "
                                                    f"face, e.g. {bad[:3]}")

    # ── the 3D panel ────────────────────────────────────────────
    w = MainWindow()
    w.resize(1700, 1000)
    w.show()
    _settle()
    w.radio_3d.setChecked(True)
    _settle()
    w.geometry_type_combo.setCurrentText("Sketch")
    _settle()
    sp, c = w.sketch_panel, w.sketch_panel.canvas
    check(c.dim == 3 and w.right_tabs.currentIndex() == w._sketch_tab_index, "3D Sketch opens the 3D canvas")
    check(not sp.tool_buttons["Polygon"].isVisible() and sp.tool_buttons["Rectangle"].text().endswith("Box"),
          "3D tools are Box and Sphere")
    sp.set_tool("Rectangle")
    sp.coord_edit.setText("0, 0, 0, 2, 1, 1")
    sp._on_coords_entered()
    sp.op_combo.setCurrentIndex(1)
    sp.set_tool("Circle")
    sp.coord_edit.setText("1, 0.5, 0.5, 0.3")
    sp._on_coords_entered()
    sp.op_combo.setCurrentIndex(0)
    sp.set_tool("Rectangle")
    sp.new_d0.setValue(0)
    sp.new_d1.setValue(0.5)

    def at(x, y):
        return c.to_screen(QPointF(x, y)).toPoint()

    QTest.mousePress(c, Qt.MouseButton.LeftButton, pos=at(2, 0))
    QTest.mouseMove(c, at(3, 0.5))
    QTest.mouseRelease(c, Qt.MouseButton.LeftButton, pos=at(3, 0.5))
    _settle()
    rows = json.loads(w._build_custom_geom_shapes_json())
    check(rows == CASES["box, hole, box"][0], f"3D sketch -> shape rows {rows}")
    cfg = w._build_config()
    check(cfg.problem_dim == "3D" and cfg.geometry_type == "Custom" and cfg.geom_sketch and not cfg.validate(),
          f"3D sketch config {cfg.validate()}")
    sp.depth_spin.setValue(0.3)
    sp.set_tool("Edges")
    _settle()
    F = c.edges()
    check(len(F) == 12 and sp.face_list.count() == 12, f"12 faces listed, got {len(F)} / {sp.face_list.count()}")
    tr = next(t for t in c.traces() if F[t["idx"]]["desc"].startswith("x = 0 "))
    QTest.mouseClick(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     c.to_screen(QPointF(*tr["mid"])).toPoint())
    top = next(i for i, f in enumerate(F) if f["desc"].startswith("z = 1 "))
    sp.face_list.item(top).setSelected(True)
    _settle(0.1)
    check(c.sel_edges == [tr["idx"], top], f"canvas click + list pick {c.sel_edges}")
    n0 = len(w.custom_bc_list)
    sp.new_bc_btn.click()
    _settle()
    check(split_where(w.custom_bc_list[-1]["location"].text()) == [F[tr["idx"]]["expr"], F[top]["expr"]],
          f"3D BC Where: {w.custom_bc_list[-1]['location'].text()}")
    check(c.edge_bcs()[top] == [n0], "faces coloured by their BC")
    sp.view_combo.setCurrentIndex(1)  # x–z
    _settle()
    check(c.view == "xz" and abs(c.depth - 0.5) < 1e-9 and len(c.traces()) == 8, f"x–z view {c.depth} {len(c.traces())}")
    w2 = MainWindow()
    _settle()
    w2._apply_config(cfg)
    _settle()
    check(w2.geometry_type_combo.currentText() == "Sketch" and w2.sketch_panel.canvas.dim == 3, "3D sketch reloads")
    w.right_tabs.setCurrentWidget(w.results_tabs)
    w2.right_tabs.setCurrentWidget(w2.results_tabs)

    # ── training ────────────────────────────────────────────────
    shapes = CASES["box, hole, box"][0]
    F = faces3(shapes)
    planes = [f["expr"] for f in F if f["kind"] == "plane"]
    sphere = [f["expr"] for f in F if f["kind"] == "sphere"]
    bcs = [{"type": "dirichlet", "location": join_where(planes), "value": "x + y + z", "component": 0},
           {"type": "neumann", "location": sphere[0], "value": "-((x - 1) + (y - 0.5) + (z - 0.5))/0.3",
            "component": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        l2 = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            cfg = PINNConfig()
            for k, v in dict(problem_dim="3D", steady_state=True, geometry_type="Custom", geom_sketch=True,
                             geom_custom_shapes_json=json.dumps(shapes), x_min=0.0, x_max=3.0, y_min=0.0, y_max=1.0,
                             z_min=0.0, z_max=1.0, num_initial=0, pde_expressions="du_xx + du_yy + du_zz",
                             custom_bc_json=json.dumps(bcs), layers=[3, 32, 32, 1], iterations=3000,
                             optimizer2="none", iterations2=0, optimizer_scheduler=False, num_domain=1000,
                             num_boundary=600, num_test=500, ea_ref_mode="formula", ea_ref_formula="x + y + z",
                             save_dir=d, plot_resolution=16, seed=1).items():
                setattr(cfg, k, v)
            cfg.pde_expression = cfg.pde_expressions
            check(not cfg.validate(), f"training config valid: {cfg.validate()}")
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(cfg))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=1500)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-2000:]}")
            l2[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
        print("relative L2:", l2)
        for name, vals in l2.items():
            check(vals and vals[0] < 0.02, f"{name}: Laplace on the sketched solid trains, L2 {vals}")
    return failures


def test_sketch_3d():
    assert not run()


if __name__ == "__main__":
    f = run()
    print(f"\n{len(f)} FAILURE(S)" if f else "\nALL 3D SKETCH TESTS PASSED")
    sys.exit(1 if f else 0)
