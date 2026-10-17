"""3D view of a 3D domain (pinnstudio/ui/view3d.py) and its place in the
sketch panel: only surfaces that bound the domain are drawn, the view
switches in and out with the View dropdown, and periodic BC rows colour the
opposite face too."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pinnstudio.ui import sketch3d as s3
from pinnstudio.ui.view3d import face_label_point, face_patches, shaded_colours

BOX = {"type": "Cuboid", "params": {"x_min": 0, "x_max": 2, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}}
HOLE = {"type": "Sphere", "op": "subtract", "params": {"cx": 1, "cy": 0.5, "cz": 1, "r": 0.4}}


def _centre(q):
    return tuple(sum(p[k] for p in q) / 4 for k in range(3))


def test_whole_faces_are_one_quad():
    faces = s3.faces3([BOX])
    patches = face_patches([BOX], faces)
    assert len(patches) == 6 and all(len(p) == 1 for p in patches)


def test_cut_out_sphere_is_a_bowl():
    shapes = [BOX, HOLE]
    faces = s3.faces3(shapes)
    patches = face_patches(shapes, faces)
    top = next(k for k, f in enumerate(faces) if f["kind"] == "plane" and f["axis"] == 2 and f["value"] == 1)
    sph = next(k for k, f in enumerate(faces) if f["kind"] == "sphere")
    # the top face has a hole: it is drawn in strips, and none covers the
    # middle of the sphere's circle
    assert len(patches[top]) > 1
    for q in patches[top]:
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        assert not (min(xs) < 1 < max(xs) and min(ys) < 0.5 < max(ys))
    # only the lower half of the sphere bounds the domain (the bowl)
    assert patches[sph] and all(_centre(q)[2] < 1.0 for q in patches[sph])
    assert face_label_point(patches[sph])[2] < 1.0


def test_shared_inner_faces_are_not_drawn():
    a = {"type": "Cuboid", "params": {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}}
    b = {"type": "Cuboid", "op": "union",
         "params": {"x_min": 1, "x_max": 2, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}}
    faces = s3.faces3([a, b])
    patches = face_patches([a, b], faces)
    for f, p in zip(faces, patches):
        if f["kind"] == "plane" and f["axis"] == 0 and f["value"] == 1:
            assert not p                # the wall between the two boxes is inside the domain


def test_shading_darkens_faces_turned_away():
    q_up = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    q_side = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]
    (up,), (side,) = shaded_colours([q_up], "#808080"), shaded_colours([q_side], "#808080")
    assert up != side and all(0 <= v <= 1 for v in up + side)


def _settle(secs=0.6):
    from PyQt6.QtWidgets import QApplication
    t = time.time()
    while time.time() - t < secs:
        QApplication.processEvents()


def test_view_switches_in_the_sketch_panel():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.sketch_canvas import SketchPanel
    p = SketchPanel()
    p.resize(900, 900)
    p.set_bc_provider(lambda: [{"index": 0, "label": "BC 1", "expr": "np.isclose(x, 0)"}], lambda: {})
    p.set_dim(3)
    p.show()
    p.set_shapes([BOX, HOLE], fit=True)
    p.view_combo.setCurrentIndex(p.view_combo.findData("3d"))
    _settle()
    assert p.in_view3d() and p.canvas_stack.currentWidget() is p.view3d
    assert not any(b.isEnabled() for b in p.tool_buttons.values())
    assert p.hint.text() == p.HINT_VIEW3D
    assert p.view3d._patches and p.canvas.view in s3.VIEWS      # the flat view (slice) is kept
    p.view3d.set_orientation("Top")
    assert (p.view3d._ax.elev, p.view3d._ax.azim) == (90, -90)
    p.view_combo.setCurrentIndex(p.view_combo.findData("xz"))
    _settle(0.2)
    assert p.canvas_stack.currentWidget() is p.canvas and p.canvas.view == "xz"
    assert all(b.isEnabled() for b in p.tool_buttons.values())
    p.view_combo.setCurrentIndex(p.view_combo.findData("3d"))
    p.set_dim(2)                                                  # 2D: always the flat canvas
    assert p.view_combo.currentData() != "3d" and p.canvas_stack.currentWidget() is p.canvas
    p.close()


def test_periodic_rows_colour_the_opposite_face():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w.show()
    w.quick_examples_combo.setCurrentText("3D Allen-Cahn (Wight & Zhao)")
    _settle(1.2)
    bc_of = w.domain_view.canvas.edge_bcs()
    assert len(bc_of) == 6 and all(bc_of)                         # all six faces of the periodic box
    w.close()


def test_point_preview_stays_in_the_domain():
    import numpy as np
    from pinnstudio.ui.view3d import inside_mask, sample_points
    shapes = [BOX, HOLE]
    rng = np.random.default_rng(1)
    P = rng.random((2000, 3)) * [2, 1, 1]
    assert (inside_mask(shapes, P) == np.array([s3.in_region3(shapes, q) for q in P])).all()
    inner, bnd = sample_points(shapes, face_patches(shapes), n_domain=800, n_boundary=300)
    assert len(inner) == 800 and len(bnd) == 300
    assert inside_mask(shapes, inner).all()                          # none in the cut-out
    assert not (np.sum((inner - [1, 0.5, 1]) ** 2, axis=1) < 0.4 ** 2).any()
    again = sample_points(shapes, face_patches(shapes), n_domain=800, n_boundary=300)
    assert np.array_equal(inner, again[0])                          # fixed seed: no jumping on redraw


def test_points_checkbox_draws_points():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from pinnstudio.ui.sketch_canvas import SketchPanel
    p = SketchPanel()
    p.resize(900, 900)
    p.set_bc_provider(lambda: [], lambda: {})
    p.set_dim(3)
    p.show()
    p.set_shapes([BOX, HOLE], fit=True)
    p.view_combo.setCurrentIndex(p.view_combo.findData("3d"))
    _settle()
    assert p.view3d._points is None                                 # off by default
    p.view3d.points_cb.setChecked(True)
    _settle()
    assert p.view3d._points is not None and len(p.view3d._points[0]) > 0
    p.close()
