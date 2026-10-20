"""A non-box shape's own bounding box is the domain box the scripts use (the
named sides, plot areas, Error Analysis grids): the window's hidden box
fields don't follow a Polygon, Disk, Ellipse or Custom shape. DeepXDE 1.15's
Ellipse box ((center - a, center + b), whatever the rotation) is corrected."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnstudio.core import codegen as cg
from pinnstudio.core.config import PINNConfig


def test_shape_box_is_the_domain_box():
    c = PINNConfig()
    c.problem_dim, c.geometry_type = "2D", "Polygon"
    c.geom_polygon_vertices = "0,0;2,0;2,1;1,1;1,2;0,2"
    c.x_min = c.y_min = -1.0
    c.x_max = c.y_max = 1.0
    e = cg._effective_config(c)
    assert (e.x_min, e.x_max, e.y_min, e.y_max) == (0.0, 2.0, 0.0, 2.0)
    c.geometry_type = "Ellipse"
    c.geom_center_x = c.geom_center_y = 0.0
    c.geom_semi_major, c.geom_semi_minor, c.geom_angle = 1.0, 0.3, 0.0
    assert cg._shape_bbox(c) == {"x_min": -1.0, "x_max": 1.0, "y_min": -0.3, "y_max": 0.3}
    c.geometry_type = "Custom"
    c.geom_custom_shapes_json = json.dumps([
        {"type": "Rectangle", "params": {"x_min": 0, "x_max": 1, "y_min": -1, "y_max": 1}},
        {"type": "Disk", "op": "union", "params": {"cx": 2, "cy": 0, "r": 0.5}}])
    assert cg._shape_bbox(c) == {"x_min": 0.0, "x_max": 2.5, "y_min": -1.0, "y_max": 1.0}
    c.geometry_type = "Rectangle"
    assert cg._shape_bbox(c) is None


def test_ellipse_box_in_the_scripts():
    import numpy as np
    import deepxde as dde
    t = cg.generate_script(PINNConfig())
    i = t.index("def _ellipse_box_fix")
    j = t.index("\n_ellipse_box_fix()", i) + len("\n_ellipse_box_fix()")
    exec(t[i:j], {"dde": dde})
    box = dde.geometry.Ellipse([0, 0], 1, 0.3, np.pi / 2).bbox
    assert np.allclose(box[0], [-0.3, -1]) and np.allclose(box[1], [0.3, 1])
    pts = dde.geometry.CSGIntersection(dde.geometry.Rectangle([0, -1], [1, 1]),
                                       dde.geometry.Ellipse([0.2, 0], 0.4, 0.2, 0.3)).random_points(3000)
    assert pts[:, 0].max() > 0.5


def test_robin_and_operator_namespaces():
    """A Robin value using u is computed on tensors (u*x, t - u); an exported
    Operator BC has np, torch and the domain limits, as Solve's does."""
    c = PINNConfig()
    c.custom_bc_json = json.dumps([{"type": "robin", "location": "True", "value": "t - u", "component": 0},
                                   {"type": "operator", "location": "True", "value": "du_x - np.sin(x_max)"}])
    exp = cg.generate_clean_script(c)
    assert "t = Y.new_tensor(np.asarray(t, dtype=float)).reshape(-1, 1)" in exp
    assert "'np': np, 'torch': torch" in exp and "'x_max':" in exp
    live = cg.generate_script(c)
    assert "_BC_MATH_NS_T = {" in live and '_ns["t"] = _Xt[:, -1:]' in live
    compile(exp, "x", "exec")
    compile(live, "x", "exec")
