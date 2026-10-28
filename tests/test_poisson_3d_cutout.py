"""Poisson 3D – cube with spherical cut-out: a PINNStudio template whose domain is built from two primitives
(Custom geometry: a cube with a sphere subtracted), with a Dirichlet condition on the cube's faces and another on
the spherical surface, checked against the exact solution."""
import json

from pinnstudio.core import codegen, example_catalog as cat, pinnstudio_examples as ps

KEY = "3D Poisson (Cube with spherical cut-out)"


def test_listed_and_built_from_two_primitives():
    e = next(x for x in cat.ENTRIES if x.key == KEY)
    assert (e.dim, e.regime, e.source, e.hidden) == ("3D", "Steady", cat.PINNSTUDIO, False)
    assert e.geometry == "Cuboid − Sphere"
    cfg = ps.build_config(ps.BY_NAME[KEY], "/r")
    shapes = json.loads(cfg.geom_custom_shapes_json)
    assert [s["type"] for s in shapes] == ["Cuboid", "Sphere"] and shapes[1]["op"] == "subtract"
    assert not cfg.validate()


def test_scripts_use_the_combined_geometry_and_exact_solution():
    cfg = ps.build_config(ps.BY_NAME[KEY], "/r")
    for script in (codegen.generate_script(cfg), codegen.generate_clean_script(cfg)):
        assert "CSGDifference(dde.geometry.Cuboid(" in script and "dde.geometry.Sphere([0.0, 0.0, 1.0], 0.7)" in script
        compile(script, "run.py", "exec")


def test_exact_solution_satisfies_the_problem():
    import numpy as np
    import deepxde as dde
    u = lambda p: np.exp(-(p ** 2).sum(1))
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (200, 3))
    h = 1e-3   # -Δu = (6 - 4 r²) exp(-r²) (central differences)
    lap = sum((u(p + h * e) - 2 * u(p) + u(p - h * e)) / h ** 2 for e in np.eye(3))
    r2 = (p ** 2).sum(1)
    assert np.allclose(-lap, (6 - 4 * r2) * np.exp(-r2), atol=1e-4)
    # the Neumann value is du/dn with DeepXDE's outward normal of the domain on the spherical surface
    g = dde.geometry.CSGDifference(dde.geometry.Cuboid([-1, -1, -1], [1, 1, 1]), dde.geometry.Sphere([0, 0, 1], 0.7))
    q = g.random_boundary_points(3000)
    q = q[np.isclose(q[:, 0] ** 2 + q[:, 1] ** 2 + (q[:, 2] - 1) ** 2, 0.49)]
    x, y, z = q.T
    dn = ((-2 * q * u(q)[:, None]) * g.boundary_normal(q)).sum(1)
    assert len(q) > 100 and np.allclose(dn, 2 * u(q) * (x ** 2 + y ** 2 + z * (z - 1)) / 0.7, atol=1e-5)


def test_error_analysis_surface_leaves_out_quads_outside_the_domain():
    """The 3D Error Analysis surface plot draws the bounding box's faces; for a combined geometry the parts
    outside the domain are transparent and out of the colour range (they showed an extrapolated PINN against
    an interpolated reference before)."""
    cfg = ps.build_config(ps.BY_NAME[KEY], "/r")
    script = codegen.generate_script(cfg)
    assert "_inside_faces_ea.append(np.asarray(geom.inside(" in script
    assert "np.where(_inside_faces_ea[_face_i_ea], 1.0, 0.0)" in script
