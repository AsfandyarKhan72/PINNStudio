"""PINNStudio templates defined by their settings.

The older templates are set up widget by widget in the main window; the
ones here are plain PINNConfig settings, loaded the same way as DeepXDE's
examples (deepxde_examples.build_config, then MainWindow._apply_config), so
a template is one dictionary. They are listed as PINNStudio templates in the
Examples lists and the README (example_catalog), with the paper they follow.
"""
import json
import math

from pinnstudio.core import deepxde_examples as _dx
from pinnstudio.core.deepxde_examples import Example, FORWARD, _bc, _bcs, _consts, _net

PAPERS = {
    "Soomro et al. (2020)": "https://doi.org/10.1016/j.est.2020.101702",
}

# Mixed convection in a lid-driven triangular cavity -------------------------
_H = math.sqrt(3) / 2
_Y0 = 1 - _H                    # apex of the equilateral triangle
_LID = "np.isclose(y, 1)"
_WALLS = "not np.isclose(y, 1)"                                      # walls and cylinder: no slip
_INCLINED = "not np.isclose(y, 1) and np.hypot(x - 0.5, y - 0.7) > 0.15"
_CYLINDER = "np.hypot(x - 0.5, y - 0.7) < 0.15"
_MC_REF = "@REF/2D/mixed_convection/"

MIXED_CONVECTION = Example(
    "2D Mixed convection (Soomro et al.)", FORWARD, PAPERS["Soomro et al. (2020)"],
    {
        **dict(problem_dim="2D", steady_state=True, geometry_type="Custom",
               geom_custom_shapes_json=json.dumps([
                   {"type": "Triangle", "params": {"vertices_text": f"0, 1; 0.5, {_Y0!r}; 1, 1"}},
                   {"type": "Disk", "op": "subtract", "params": {"cx": 0.5, "cy": 0.7, "r": 0.1}}]),
               x_min=0.0, x_max=1.0, y_min=_Y0, y_max=1.0, num_initial=0,
               num_outputs=4, output_names="u,v,p,T",
               pde_expressions="|".join([
                   "du_x + dv_y",
                   "u*du_x + v*du_y + dp_x - (du_xx + du_yy)/Re",
                   "u*dv_x + v*dv_y + dp_y - (dv_xx + dv_yy)/Re - Ri*T + Ha**2/Re*v",
                   "u*dT_x + v*dT_y - (dT_xx + dT_yy)/(Pr*Re)"]),
               custom_bc_json=_bcs(
                   _bc("dirichlet", _LID, "-1", 0), _bc("dirichlet", _LID, "0", 1),
                   _bc("dirichlet", _LID, "1", 3),
                   _bc("dirichlet", _WALLS, "0", 0), _bc("dirichlet", _WALLS, "0", 1),
                   _bc("neumann", _INCLINED, "0", 3), _bc("dirichlet", _CYLINDER, "0", 3)),
               num_domain=12000, num_boundary=3000, num_test=4000,
               ea_files=repr([(0.0, _MC_REF + "solution_u.txt", 0), (0.0, _MC_REF + "solution_v.txt", 1),
                              (0.0, _MC_REF + "solution_T.txt", 3)]),
               ea_do_line=False, ea_do_surface=True),
        **_consts(("Re", "300"), ("Ri", "0.1"), ("Ha", "10"), ("Pr", "6.2")),
        # a multi-scale Fourier feature network: the plain network settles on a
        # weak, viscous-looking flow (see benchmarks/training_methods.md)
        **_net([2, 64, 64, 64, 64, 64, 4], "Glorot normal", network_type="MsFFN"),
        **dict(msffn_sigmas="1, 5"),
        **dict(optimizer="adam", learning_rate=1e-3, iterations=30000, optimizer2="lbfgs", iterations2=20000),
    },
    note="Steady mixed convection with a magnetic field in a lid-driven equilateral triangle with a cold "
         "cylinder (Re = 300, Ri = 0.1, Ha = 10, Pr = 6.2). The lid moves to the left, as in the paper's "
         "figures. The cylinder's centre (0.5, 0.7) and radius 0.1 are read off the paper's figures, which "
         "do not state them. Reference: finite elements, tools/make_mixed_convection_reference.py.",
    inverse=dict(
        problem_type="Inverse",
        constants=json.dumps([{"name": "Re", "value": "300"}, {"name": "Ri", "value": "0.1"},
                              {"name": "Pr", "value": "6.2"}]),
        inverse_variables_json=json.dumps([{"name": "Ha", "init": 1.0, "true": 10.0}]),
        inverse_obs_files_json=json.dumps([
            {"path": _MC_REF + "obs_u.txt", "output_idx": 0, "weight": 100.0, "custom_expr": ""},
            {"path": _MC_REF + "obs_v.txt", "output_idx": 1, "weight": 100.0, "custom_expr": ""}]),
        inv_param_save="Every 1000 iters"),
    inverse_note="Learns the Hartmann number Ha (true value 10) from 400 measurements of u and v.",
)

EXAMPLES = [MIXED_CONVECTION]
BY_NAME = {e.name: e for e in EXAMPLES}
# shown name and paper of each, for the example catalogue
CATALOG = {
    MIXED_CONVECTION.name: ("Mixed convection 2D – Soomro et al.", "Soomro et al. (2020)"),
}


def build_config(example, ref_root, keep=None, inverse=False):
    return _dx.build_config(example, ref_root, keep=keep, inverse=inverse)
