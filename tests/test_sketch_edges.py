#!/usr/bin/env python3
"""Sketch → edges → boundary conditions.

The sketched domain's boundary is split into edges, each with a "Where:"
expression (pinnstudio/ui/sketch_edges.py). Checks:
  * the edges of a few sketches (holes cutting an edge, unions, intersections,
    a rotated ellipse, negative coordinates, a large domain), and that each
    edge's expression picks that edge and no other;
  * against DeepXDE itself: every boundary point the CSG geometry samples
    (float32) is picked by exactly one edge;
  * the Edges tool in the GUI: a new BC from selected edges, adding edges to
    a BC, taking them out again, and the edge colouring;
  * a Laplace problem (u = x + y) on a sketched domain with Dirichlet edges
    and a Neumann hole, trained by both the live and the exported script.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sketch_edges.py
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
from pinnstudio.ui.sketch_edges import (add_to_where, boundary_edges, join_where, remove_from_where, split_where,
                                        where_matches)


def R(a, b, c, d, op=None):
    s = {"type": "Rectangle", "params": {"x_min": a, "x_max": b, "y_min": c, "y_max": d}}
    if op:
        s["op"] = op
    return s


def D(x, y, r, op=None):
    s = {"type": "Disk", "params": {"cx": x, "cy": y, "r": r}}
    if op:
        s["op"] = op
    return s


CASES = {
    "rect, hole, triangle": [R(0, 1, 0, 0.8), D(0.5, 0.4, 0.2, "subtract"),
                             {"type": "Triangle", "params": {"vertices_text": "1,0;1.3,0.4;1,0.8"}, "op": "union"}],
    "hole cutting an edge": [R(0, 2, 0, 1), D(1, 0, 0.3, "subtract")],
    "overlapping union": [R(0, 1, 0, 1), R(0.5, 2, 0, 0.5, "union")],
    "intersection": [D(0, 0, 1), R(0, 2, -2, 2, "intersect")],
    "rotated ellipse": [{"type": "Ellipse", "params": {"cx": 1, "cy": -1, "a": 2, "b": 1, "angle": 0.5}},
                        R(0, 3, -3, 0, "subtract")],
    "negative coordinates": [R(-1, 1, -1, 1),
                             {"type": "Polygon", "params": {"vertices_text": "-0.5,-0.5;0.5,-0.5;0,0.5"},
                              "op": "subtract"}],
    "large domain": [R(0, 200, 0, 100), D(100, 50, 20, "subtract")],
    "circle split in two": [R(0, 2, 0, 1), D(1, 0.5, 0.6, "subtract")],
}
N_EDGES = {"rect, hole, triangle": 6, "hole cutting an edge": 6, "overlapping union": 6, "intersection": 2,
           "rotated ellipse": 3, "negative coordinates": 7, "large domain": 5, "circle split in two": 8}


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

    # ── edges and their expressions ─────────────────────────────
    for name, shapes in CASES.items():
        edges = boundary_edges(shapes)
        check(len(edges) == N_EDGES[name], f"{name}: {len(edges)} edges, expected {N_EDGES[name]}")
        for i, e in enumerate(edges):
            hits = [j for j, f in enumerate(edges) if where_matches(f["expr"], *e["mid"])]
            check(hits == [i], f"{name}: E{i + 1} ({e['expr']}) is picked by edges {hits}")
    e = boundary_edges(CASES["hole cutting an edge"])
    bottoms = sorted(x["expr"] for x in e if x["expr"].startswith("np.isclose(y, 0)"))
    check(bottoms == ["np.isclose(y, 0) and 0 <= x <= 0.7", "np.isclose(y, 0) and 1.3 <= x <= 2"],
          f"the hole splits the bottom edge in two: {bottoms}")
    e = boundary_edges(CASES["overlapping union"])
    check("np.isclose(y, 0) and 0 <= x <= 2" in [x["expr"] for x in e], "collinear bottom edges merge into one")

    # Where: helpers
    w = join_where(["A and B", "C"])
    check(w == "(A and B) or (C)", w)
    check(split_where(w) == ["A and B", "C"], split_where(w))
    check(add_to_where("True", "C") == "C", "adding to the whole boundary replaces it")
    check(add_to_where("A and B", "C") == "(A and B) or (C)", add_to_where("A and B", "C"))
    check(remove_from_where(w, ["C"]) == "A and B", remove_from_where(w, ["C"]))
    check(remove_from_where("left or right", ["x"]) is None, "nothing to remove from a typed Where:")

    # ── against DeepXDE's own boundary points ───────────────────
    try:
        import numpy as np
        import deepxde as dde
    except Exception as exc:  # pragma: no cover
        print("deepxde not available, skipping the DeepXDE check:", exc)
    else:
        dde.config.set_default_float("float32")

        def geom(sh):
            t, p = sh["type"], sh["params"]
            if t == "Rectangle":
                return dde.geometry.Rectangle([p["x_min"], p["y_min"]], [p["x_max"], p["y_max"]])
            if t == "Disk":
                return dde.geometry.Disk([p["cx"], p["cy"]], p["r"])
            if t == "Ellipse":
                return dde.geometry.Ellipse([p["cx"], p["cy"]], p["a"], p["b"], p.get("angle", 0))
            pts = [[float(a) for a in c.split(",")] for c in p["vertices_text"].split(";")]
            return dde.geometry.Triangle(*pts) if len(pts) == 3 else dde.geometry.Polygon(pts)

        for name, shapes in CASES.items():
            g = geom(shapes[0])
            for sh in shapes[1:]:
                h = geom(sh)
                g = g - h if sh["op"] == "subtract" else (g & h if sh["op"] == "intersect" else g | h)
            edges = boundary_edges(shapes)
            X = g.random_boundary_points(1500).astype(np.float32)
            X = X[g.on_boundary(X)]
            counts = [sum(where_matches(e["expr"], x[0], x[1]) for e in edges) for x in X]
            bad = [X[i].tolist() for i, k in enumerate(counts) if k != 1]
            # (a point at a corner can sit on both edges within float32 precision)
            check(len(X) > 1000 and len(bad) <= 2,
                  f"{name}: {len(bad)} of {len(X)} DeepXDE boundary points not picked by exactly one edge, "
                  f"e.g. {bad[:3]}")

    # ── the Edges tool in the GUI ───────────────────────────────
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    _settle()
    w.radio_2d.setChecked(True)
    _settle()
    w.geometry_type_combo.setCurrentText("Sketch")
    _settle()
    shapes = [R(0, 2, 0, 1), D(0.6, 0.5, 0.25, "subtract")]
    w._on_sketch_edited(shapes)
    sp, c = w.sketch_panel, w.sketch_panel.canvas
    sp.set_shapes(shapes, fit=True)
    sp.set_tool("Edges")
    _settle()
    E = c.edges()
    check(len(E) == 5, f"GUI: 5 edges, got {len(E)}")

    def click(k, mod=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(c, Qt.MouseButton.LeftButton, mod, c.to_screen(QPointF(*E[k]["mid"])).toPoint())

    def find(prefix):
        return next(i for i, e in enumerate(E) if e["desc"].startswith(prefix))

    n0 = len(w.custom_bc_list)
    left, right, top = find("x = 0"), find("x = 2"), find("y = 1")
    hole = next(i for i, e in enumerate(E) if e["kind"] == "arc")
    click(left)
    click(right, Qt.KeyboardModifier.ControlModifier)
    check(c.sel_edges == [left, right], f"GUI: Ctrl-click adds to the selection {c.sel_edges}")
    sp.new_bc_btn.click()
    _settle()
    check(len(w.custom_bc_list) == n0 + 1, "GUI: New BC on selected adds a row")
    row = w.custom_bc_list[-1]
    check(split_where(row["location"].text()) == [E[left]["expr"], E[right]["expr"]],
          f"GUI: the new row's Where: {row['location'].text()!r}")
    bc_of = c.edge_bcs()
    check(bc_of[left] == [n0] and bc_of[right] == [n0] and not bc_of[top],
          f"GUI: edges coloured by their BC {bc_of}")
    w.workspace.select("bcs")   # (the new BC's node was selected: back to picking edges freely)
    _settle()
    click(top)
    sp.add_to_combo.setCurrentIndex(sp.add_to_combo.findData(n0))
    sp.add_to_btn.click()
    _settle()
    check(c.edge_bcs()[top] == [n0], "GUI: Add to puts the edge in that BC")
    w.workspace.select("bcs")
    _settle()
    click(left)
    sp.remove_btn.click()
    _settle()
    check(not c.edge_bcs()[left] and c.edge_bcs()[right] == [n0], "GUI: Remove from BC takes the edge out")
    w.workspace.select("bcs")
    _settle()
    click(hole)
    sp.new_bc_btn.click()
    _settle()
    check(c.edge_bcs()[hole] == [n0 + 1], "GUI: the hole gets its own BC")
    w.workspace.select("bcs")
    _settle()
    sp._select_free_edges()
    check(sorted(c.sel_edges) == sorted([left, find("y = 0")]), f"GUI: select edges without a BC {c.sel_edges}")
    cfg = w._build_config()
    check(cfg.geometry_type == "Custom" and cfg.geom_sketch and not cfg.validate(), f"GUI config {cfg.validate()}")
    w.right_tabs.setCurrentIndex(0)

    # ── training on a sketched domain with sketched BCs ─────────
    shapes = [R(0, 2, 0, 1), D(0.6, 0.5, 0.25, "subtract"),
              {"type": "Polygon", "params": {"vertices_text": "2,0;2.6,0.3;2.6,0.7;2,1"}, "op": "union"}]
    edges = boundary_edges(shapes)
    straight = [e["expr"] for e in edges if e["kind"] == "seg"]
    arc = [e["expr"] for e in edges if e["kind"] == "arc"]
    check(len(arc) == 1 and len(straight) == 6, f"training domain edges {len(straight)} + {len(arc)}")
    # u = x + y; on the hole the outward normal points to its centre: du/dn = -((x-0.6) + (y-0.5))/0.25
    bcs = [{"type": "dirichlet", "location": join_where(straight), "value": "x + y", "component": 0},
           {"type": "neumann", "location": arc[0], "value": "-((x - 0.6) + (y - 0.5))/0.25", "component": 0}]
    with tempfile.TemporaryDirectory() as tmp:
        l2 = {}
        for gen in (generate_script, generate_clean_script):
            d = os.path.join(tmp, gen.__name__)
            os.makedirs(d)
            cfg = PINNConfig()
            for k, v in dict(problem_dim="2D", steady_state=True, geometry_type="Custom", geom_sketch=True,
                             geom_custom_shapes_json=json.dumps(shapes), x_min=0.0, x_max=2.6, y_min=0.0, y_max=1.0,
                             num_initial=0, pde_expressions="du_xx + du_yy", custom_bc_json=json.dumps(bcs),
                             layers=[2, 32, 32, 1], iterations=3000, optimizer2="none", iterations2=0,
                             optimizer_scheduler=False, num_domain=600, num_boundary=300, num_test=300,
                             ea_ref_mode="formula", ea_ref_formula="x + y", save_dir=d, plot_resolution=40,
                             seed=1).items():
                setattr(cfg, k, v)
            cfg.pde_expression = cfg.pde_expressions
            check(not cfg.validate(), f"training config valid: {cfg.validate()}")
            with open(os.path.join(d, "run.py"), "w") as fh:
                fh.write(gen(cfg))
            p = subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True, text=True, timeout=900)
            check(p.returncode == 0, f"{gen.__name__} ran:\n{p.stderr[-2000:]}")
            l2[gen.__name__] = [float(v) for v in re.findall(r"L2=(\S+),", p.stdout)]
        print("relative L2:", l2)
        for name, vals in l2.items():
            check(vals and vals[0] < 0.02, f"{name}: Laplace on the sketched domain trains, L2 {vals}")
    return failures


def test_sketch_edges():
    assert not run()


if __name__ == "__main__":
    f = run()
    print(f"\n{len(f)} FAILURE(S)" if f else "\nALL SKETCH EDGE TESTS PASSED")
    sys.exit(1 if f else 0)
