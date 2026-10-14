#!/usr/bin/env python3
"""Sketch → Import DXF/SVG: closed outlines become Custom-geometry shapes
(circles stay Disks, boxes Rectangles, outlines inside others become holes),
lines and arcs drawn separately are joined up, arcs inside a polyline are
found again as single edges for boundary conditions, and every boundary
point DeepXDE samples on the result is picked by exactly one edge.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_sketch_import.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui.sketch_canvas import SketchPanel
from pinnstudio.ui.sketch_edges import arc_runs, boundary_edges, where_matches
from pinnstudio.ui.sketch_import import import_file


def _dxf(ents):
    out = ["0", "SECTION", "2", "HEADER", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for e in ents:
        out += e
    return "\n".join(out + ["0", "ENDSEC", "0", "EOF"]) + "\n"


def _line(a, b, c, d):
    return ["0", "LINE", "8", "0", "10", str(a), "20", str(b), "11", str(c), "21", str(d)]


# a 4 x 2 plate drawn as four separate LINEs, a round hole, a slot with
# rounded ends (LWPOLYLINE bulges), a stray arc that doesn't close, and text
PLATE_DXF = _dxf([
    _line(0, 0, 4, 0), _line(4, 2, 4, 0), _line(4, 2, 0, 2), _line(0, 0, 0, 2),
    ["0", "CIRCLE", "8", "0", "10", "1", "20", "1", "40", "0.4"],
    ["0", "LWPOLYLINE", "8", "0", "90", "4", "70", "1", "10", "2.6", "20", "0.7", "10", "3.2", "20", "0.7", "42", "1",
     "10", "3.2", "20", "1.3", "10", "2.6", "20", "1.3", "42", "1"],
    ["0", "ARC", "8", "0", "10", "9", "20", "9", "40", "1", "50", "0", "51", "90"],
    ["0", "TEXT", "8", "0", "10", "0", "20", "0", "1", "hello"],
])

PART_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200">
<g transform="translate(10,10)">
<path d="M0 0 H300 V150 H0 Z" fill="none" stroke="black"/>
<circle cx="80" cy="75" r="30"/>
<ellipse cx="200" cy="75" rx="40" ry="20" transform="rotate(30 200 75)"/>
<path d="M240 60 a 20 20 0 0 1 40 0 l 0 20 h -40 z"/>
<text x="5" y="5">label</text>
</g></svg>"""


def run():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)
            print("FAIL:", msg)

    with tempfile.TemporaryDirectory() as tmp:
        dxf = os.path.join(tmp, "plate.dxf")
        svg = os.path.join(tmp, "part.svg")
        with open(dxf, "w") as fh:
            fh.write(PLATE_DXF)
        with open(svg, "w") as fh:
            fh.write(PART_SVG)

        shapes, notes = import_file(dxf)
        check([(s["type"], s.get("op")) for s in shapes] ==
              [("Rectangle", None), ("Polygon", "subtract"), ("Disk", "subtract")],
              f"DXF shapes {[(s['type'], s.get('op')) for s in shapes]}")
        check(shapes[0]["params"] == {"x_min": 0.0, "x_max": 4.0, "y_min": 0.0, "y_max": 2.0},
              f"four LINEs join into the plate {shapes[0]['params']}")
        check(any("TEXT" in n for n in notes) and any("don't close" in n for n in notes), f"DXF notes {notes}")
        edges = boundary_edges(shapes)
        arcs = [e for e in edges if e["kind"] == "arc"]
        check(len(edges) == 9 and len(arcs) == 3, f"DXF edges: {len(edges)} with {len(arcs)} arcs (want 9, 3)")
        check(any("centre (3.2, 1)" in e["desc"] for e in arcs) and any("centre (2.6, 1)" in e["desc"] for e in arcs),
              f"the slot's rounded ends are one edge each {[e['desc'] for e in arcs]}")
        half, _ = import_file(dxf, 0.5)
        check(half[0]["params"]["x_max"] == 2.0 and half[2]["params"]["r"] == 0.2, "scale on import")

        shapes_svg, notes = import_file(svg)
        kinds = [(s["type"], s.get("op")) for s in shapes_svg]
        check(kinds == [("Rectangle", None), ("Disk", "subtract"), ("Ellipse", "subtract"), ("Polygon", "subtract")],
              f"SVG shapes {kinds}")
        check(shapes_svg[0]["params"] == {"x_min": 10.0, "x_max": 310.0, "y_min": -160.0, "y_max": -10.0},
              f"SVG y is flipped and the group translated {shapes_svg[0]['params']}")
        ell = shapes_svg[2]["params"]
        check(abs(ell["a"] - 40) < 1e-6 and abs(ell["b"] - 20) < 1e-6 and abs(abs(ell["angle"]) - 0.523599) < 1e-5,
              f"rotated SVG ellipse {ell}")
        e_svg = boundary_edges(shapes_svg)
        check(len(e_svg) == 10, f"SVG edges {len(e_svg)}")

        # a polyline circle is read back as a circle
        import math
        ring = [(math.cos(2 * math.pi * k / 72), math.sin(2 * math.pi * k / 72)) for k in range(72)]
        check(arc_runs(ring)[0][0] == "full", "a 72-gon is recognised as a circle")

        for name, shp in (("DXF", shapes), ("SVG", shapes_svg)):
            E = boundary_edges(shp)
            for i, e in enumerate(E):
                hits = [j for j, f in enumerate(E) if where_matches(f["expr"], *e["mid"])]
                check(hits == [i], f"{name}: E{i + 1} picked by {hits}")
        try:
            import numpy as np
            import deepxde as dde
        except Exception as exc:  # pragma: no cover
            print("deepxde not available, skipping:", exc)
        else:
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

            for name, shp in (("DXF", shapes), ("SVG", shapes_svg)):
                g = geom(shp[0])
                for sh in shp[1:]:
                    g = g - geom(sh) if sh["op"] == "subtract" else g | geom(sh)
                E = boundary_edges(shp)
                X = g.random_boundary_points(1500).astype(np.float32)
                X = X[g.on_boundary(X)]
                bad = [x.tolist() for x in X if sum(where_matches(e["expr"], x[0], x[1]) for e in E) != 1]
                check(len(X) > 1000 and not bad, f"{name}: {len(bad)} DeepXDE boundary points not on exactly one "
                                                  f"edge, e.g. {bad[:3]}")

        # the panel
        p = SketchPanel()
        got = []
        p.shapes_edited.connect(lambda lst: got.append(lst))
        check(p.import_drawing(dxf, scale=1.0) and len(got[-1]) == 3, "panel imports into an empty sketch")
        check(p.import_drawing(svg, scale=0.01, replace=False) and len(got[-1]) == 7 and got[-1][3]["op"] == "union",
              "panel adds a drawing to the sketch")
        check(p.import_drawing(svg, scale=0.01, replace=True) and len(got[-1]) == 4, "panel replaces the sketch")
    return failures


def test_sketch_import():
    assert not run()


if __name__ == "__main__":
    f = run()
    print(f"\n{len(f)} FAILURE(S)" if f else "\nALL SKETCH IMPORT TESTS PASSED")
    sys.exit(1 if f else 0)
