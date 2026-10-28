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
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (200, 3))
    x, y, z = p.T
    u = lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    h = 1e-3   # -Δu = 3π² u (central differences)
    lap = sum((u(*(p + h * e).T) - 2 * u(x, y, z) + u(*(p - h * e).T)) / h ** 2 for e in np.eye(3))
    assert np.allclose(-lap, 3 * np.pi ** 2 * u(x, y, z), atol=1e-4)
    for c in range(3):          # zero on the cube's faces
        q = p.copy(); q[:, c] = 1.0
        assert np.allclose(u(*q.T), 0, atol=1e-12)
