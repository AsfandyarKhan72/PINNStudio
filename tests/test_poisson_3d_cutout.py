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
    q = _sphere_points_away_from_the_rim(_boundary_sample(0))
    x, y, z = q.T
    dn = ((-2 * q * u(q)[:, None]) * _geom().boundary_normal(q)).sum(1)
    assert len(q) > 100 and np.allclose(dn, 2 * u(q) * (x ** 2 + y ** 2 + z * (z - 1)) / 0.7, atol=1e-5)


# The sphere (centre (0, 0, 1), radius 0.7) meets the cube only in its top face, along the circle x^2 + y^2 = 0.49,
# z = 1. The outward normal is not defined on that rim. DeepXDE decides "on a face" and "on the sphere" with its
# isclose (float32: atol 1e-6 plus numpy's rtol 1e-5), so a sampled point within about 1e-5 of the top face counts as
# on both surfaces and CSGDifference.boundary_normal returns the sum of the two normals (not a unit vector). The
# checks of the normal therefore leave out a band |z - 1| < RIM around the rim -- far wider than that tolerance and a
# negligible part of the surface. (An unseeded run on macOS CI drew such a point; seed 257 below reproduces it.)
RIM = 1e-4
R2 = 0.49


def _geom():
    import deepxde as dde
    return dde.geometry.CSGDifference(dde.geometry.Cuboid([-1, -1, -1], [1, 1, 1]),
                                      dde.geometry.Sphere([0, 0, 1], 0.7))


def _boundary_sample(seed, n=3000):
    import deepxde as dde
    dde.config.set_random_seed(seed)     # DeepXDE samples with numpy's global generator
    return _geom().random_boundary_points(n)


def _on_sphere(q):
    import numpy as np
    return np.isclose(q[:, 0] ** 2 + q[:, 1] ** 2 + (q[:, 2] - 1) ** 2, R2)


def _sphere_points_away_from_the_rim(q):
    return q[_on_sphere(q) & (q[:, 2] < 1 - RIM)]


def test_boundary_points_lie_on_the_domain_boundary_with_the_outward_normal():
    """Every sampled boundary point is on a cube face outside the sphere or on the sphere inside the cube, and away
    from the rim its normal is the unit outward normal of the domain: the face normal on the faces, pointing into
    the sphere (towards its centre) on the cut-out surface."""
    import numpy as np
    g = _geom()
    for seed in (0, 1, 257):
        q = _boundary_sample(seed)
        on_sphere = _on_sphere(q)
        on_face = np.isclose(np.abs(q), 1, rtol=0, atol=1e-6).any(1)
        assert (on_sphere | on_face).all()
        assert (np.abs(q) <= 1 + 1e-6).all()                                   # inside the cube
        assert (q[on_face & ~on_sphere, 0] ** 2 + q[on_face & ~on_sphere, 1] ** 2
                + (q[on_face & ~on_sphere, 2] - 1) ** 2 >= R2).all()             # outside the sphere
        assert on_sphere.sum() > 100 and on_face.sum() > 100
        away = np.abs(q[:, 2] - 1) >= RIM
        n = g.boundary_normal(q)
        assert np.allclose(np.linalg.norm(n[away], axis=1), 1, atol=1e-5)
        s = on_sphere & away
        assert np.allclose(n[s], -(q[s] - [0, 0, 1]) / 0.7, atol=1e-5)
        f = on_face & ~on_sphere & away
        expected = np.where(np.isclose(np.abs(q[f]), 1, rtol=0, atol=1e-6), np.sign(q[f]), 0)
        assert np.allclose(n[f], expected, atol=1e-6)
        # on the rim the normal is not defined: what DeepXDE returns there is excluded above, not asserted


def test_seed_257_draws_a_rim_point_and_the_rim_band_excludes_it():
    """The macOS CI failure: a point on the sphere within DeepXDE's tolerance of the top face, where the returned
    normal is the sum of both surfaces' normals. The rim band used by the checks removes exactly such points."""
    import numpy as np
    q = _boundary_sample(257)
    rim = _on_sphere(q) & (q[:, 2] >= 1 - RIM)
    n = _geom().boundary_normal(q[rim])
    assert rim.any() and (np.linalg.norm(n, axis=1) > 1 + 1e-3).any()


def test_error_analysis_surface_leaves_out_quads_outside_the_domain():
    """The 3D Error Analysis surface plot draws the bounding box's faces; for a combined geometry the parts
    outside the domain are transparent and out of the colour range (they showed an extrapolated PINN against
    an interpolated reference before)."""
    cfg = ps.build_config(ps.BY_NAME[KEY], "/r")
    script = codegen.generate_script(cfg)
    assert "_inside_faces_ea.append(np.asarray(geom.inside(" in script
    assert "np.where(_inside_faces_ea[_face_i_ea], 1.0, 0.0)" in script
