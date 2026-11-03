"""PINNStudio templates defined by their settings.

The older templates are set up widget by widget in the main window; the
ones here are plain PINNConfig settings, loaded the same way as DeepXDE's
examples (deepxde_examples.build_config, then MainWindow._apply_config), so
a template is one dictionary. They are listed as PINNStudio templates in the
Examples lists (example_catalog), with the paper they follow.

Templates you keep for yourself go in pinnstudio/core/local_examples.py
(ignored by git), with their reference data under reference_data/local/:
the file defines EXAMPLES, CATALOG ({name: (shown name, paper)}) and PAPERS
({paper: link}) in the same form, and is added to the lists when present.
"""
import json as _json

from pinnstudio.core import deepxde_examples as _dx

PAPERS = {}
EXAMPLES = []
# shown name and paper of each, for the example catalogue
CATALOG = {}

try:                                    # local templates, if there are any
    from pinnstudio.core import local_examples as _local
except ImportError:
    _local = None
if _local is not None:
    PAPERS.update(getattr(_local, "PAPERS", {}))
    EXAMPLES.extend(getattr(_local, "EXAMPLES", []))
    CATALOG.update(getattr(_local, "CATALOG", {}))

# Poisson in a cube with a spherical cut-out: a 3D domain built from two primitives (a cube with a sphere
# centred on its top face subtracted), with a Dirichlet condition on the cube's faces and a Neumann condition
# on the spherical surface. The exact solution u = exp(-(x^2 + y^2 + z^2)) gives both, and Error Analysis
# compares against it directly. On the spherical surface the outward normal of the domain points into the
# sphere, n = -(p - c)/R with c = (0, 0, 1), so du/dn = 2 u (x^2 + y^2 + z (z - 1)) / R.
EXAMPLES.append(_dx._ex(
    "3D Poisson (Cube with spherical cut-out)", _dx.FORWARD, "",
    dict(problem_dim="3D", steady_state=True, geometry_type="Custom",
         geom_custom_shapes_json=_json.dumps([
             {"type": "Cuboid", "params": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0,
                                           "z_min": -1.0, "z_max": 1.0}},
             {"type": "Sphere", "op": "subtract", "params": {"cx": 0.0, "cy": 0.0, "cz": 1.0, "r": 0.7}}]),
         x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=-1.0, z_max=1.0, num_initial=0,
         pde_expressions="-du_xx - du_yy - du_zz - (6 - 4*(x**2 + y**2 + z**2))*exp(-(x**2 + y**2 + z**2))",
         custom_bc_json=_dx._bcs(
             _dx._bc("dirichlet", "left or right or bottom or top or front or back", "exp(-(x**2 + y**2 + z**2))"),
             _dx._bc("neumann", "np.isclose(x**2 + y**2 + (z - 1)**2, R**2)",
                     "2*exp(-(x**2 + y**2 + z**2))*(x**2 + y**2 + z*(z - 1))/R")),
         num_domain=8000, num_boundary=2000, num_test=5000, plot_resolution=60),
    _dx._consts(("R", "0.7")),
    _dx._net([3, 64, 64, 64, 64, 1]), _dx._adam(15000), dict(optimizer2="lbfgs", iterations2=15000),
    _dx._formula_ref(["exp(-(x**2 + y**2 + z**2))"]),
    note="A cube [-1, 1]^3 with a sphere of radius R = 0.7 centred on its top face subtracted (Geometry -> "
         "Custom); u = exp(-(x^2 + y^2 + z^2)) on the cube's faces (Dirichlet, the named sides) and its normal "
         "derivative on the spherical surface (Neumann, a coordinate expression); compared with the exact solution."))
CATALOG["3D Poisson (Cube with spherical cut-out)"] = ("Poisson 3D – cube with spherical cut-out", "")

BY_NAME = {e.name: e for e in EXAMPLES}


def build_config(example, ref_root, keep=None, inverse=False):
    return _dx.build_config(example, ref_root, keep=keep, inverse=inverse)
