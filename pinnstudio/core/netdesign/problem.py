"""What a problem gives a designed network: the names of its inputs and outputs.

The inputs are what the network's first layer sees, after PINNStudio's own transforms: the coordinates
(x, y, z, t as the problem has them; t alone for an ODE), or, with an input feature formula or exact
periodicity, one input per feature, named f1, f2, … (`feature_formulas` gives each one's formula). The
outputs are the problem's output names.
"""
from __future__ import annotations


def coordinate_names(problem_dim: str, steady: bool) -> list:
    if problem_dim == "ODE":
        return ["t"]
    space = {"1D": ["x"], "2D": ["x", "y"], "3D": ["x", "y", "z"]}.get(problem_dim, ["x"])
    return space + ([] if steady else ["t"])


def output_names(config) -> list:
    n = max(1, int(getattr(config, "num_outputs", 1) or 1))
    names = [s.strip() for s in (getattr(config, "output_names", "") or "u").split(",") if s.strip()]
    while len(names) < n:
        names.append(f"u{len(names) + 1}")
    return names[:n]


def names_effective(eff) -> tuple:
    """(input names, output names) of a config that has been through codegen._effective_config: the
    inputs are f1, f2, … for the features of an input formula or exact periodicity (stable names: the
    formulas change with the domain), else the coordinates (t alone for an ODE)."""
    from pinnstudio.core import codegen as _cg
    if _cg._input_formula_on(eff):
        ins = _feature_names(_cg._feature_formula_list(eff))
    elif getattr(eff, "_ode", False):
        ins = ["t"]
    else:
        ins = coordinate_names(getattr(eff, "problem_dim", "1D"), bool(getattr(eff, "steady_state", False)))
    return ins, output_names(eff)


def feature_formulas(config) -> dict:
    """{f1: its formula, …} when the network's inputs are features of a formula, else {}."""
    from pinnstudio.core import codegen as _cg
    eff = _cg._effective_config(config)
    if not _cg._input_formula_on(eff):
        return {}
    feats = _cg._feature_formula_list(eff)
    return dict(zip(_feature_names(feats), (f.strip() for f in feats)))


def _feature_names(feats) -> list:
    """A feature that is just a variable (x, t) keeps its name; a formula is f1, f2, … by its place."""
    out = []
    for i, f in enumerate(feats):
        f = f.strip()
        out.append(f if f.isidentifier() and f not in out else f"f{i + 1}")
    return out


def input_names(config) -> list:
    """The network's input names for `config` (a PINNConfig), as the generated scripts feed them."""
    from pinnstudio.core import codegen as _cg
    return names_effective(_cg._effective_config(config))[0]


def network_io(config):
    """(input names, output names) of a problem."""
    from pinnstudio.core import codegen as _cg
    return names_effective(_cg._effective_config(config))
