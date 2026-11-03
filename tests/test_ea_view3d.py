"""Error Analysis' 3D view of a 3D box problem (pinnstudio/core/ea_view3d.py):
the level surface by marching tetrahedra, the cut-away box, and the code that
the generated script gets."""
import dataclasses

import numpy as np

from pinnstudio.core import ea_view3d as v3
from pinnstudio.core.config import PINNConfig


def _sphere(n=41, r=0.3):
    g = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return g, r - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)


def test_march_gives_the_sphere_with_outward_normals():
    g, V = _sphere()
    tri = v3._v3_march(V, g, g, g, 0.0)
    assert len(tri) > 1000
    r = np.linalg.norm(tri - 0.5, axis=2)
    assert abs(r.mean() - 0.3) < 2e-3                       # every vertex on the sphere
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1).sum()
    assert abs(area - 4 * np.pi * 0.3 ** 2) / (4 * np.pi * 0.3 ** 2) < 0.02
    out = (nrm * (tri.mean(1) - 0.5)).sum(1)
    assert (out > 0).all()                                   # V > level inside, normals point out


def test_march_without_the_level_gives_nothing():
    g, V = _sphere()
    assert len(v3._v3_march(V, g, g, g, 5.0)) == 0


def test_cut_quads_cover_the_visible_faces_and_the_cuts():
    g = np.linspace(0, 1, 11)
    E = np.random.default_rng(0).random((11, 11, 11))
    polys, vals = v3._v3_cut_quads(g, g, g, E)
    assert len(polys) == len(vals)
    # three faces with a quarter removed (3 x 3/4) and three cut quarters: 3 face-areas in all
    area = sum(np.linalg.norm(np.cross(np.subtract(q[1], q[0]), np.subtract(q[3], q[0]))) for q in polys)
    assert abs(area - 3.0) < 1e-9
    c = np.array([np.mean(q, 0) for q in polys])
    assert not ((c[:, 0] > 0.5 + 1e-9) & (c[:, 1] < 0.5 - 1e-9) & (c[:, 2] > 0.5 + 1e-9)).any()


def test_block_level_setting_and_compiles():
    cfg = dataclasses.replace(PINNConfig(), plot_iso_level="0.25", plot_resolution_3d=30)
    code = v3.view3d_block(cfg, 0)
    assert "_lvl_v3 = 0.25" in code and "_n_v3 = 30" in code
    compile(code, "<block>", "exec")
    auto = v3.view3d_block(dataclasses.replace(cfg, plot_iso_level=""), 0)
    assert "_lvl_v3 = None" in auto
    bad = v3.view3d_block(dataclasses.replace(cfg, plot_iso_level="zero"), 0)
    assert "_lvl_v3 = None" in bad and "is not a number" in bad


def test_generated_script_has_the_helpers_and_the_view():
    from pinnstudio.core.codegen import generate_script
    cfg = dataclasses.replace(PINNConfig(), problem_dim="3D")
    script = generate_script(cfg)
    assert "def _v3_march(" in script and "view3d_comparison" in script
    compile(script, "<script>", "exec")
