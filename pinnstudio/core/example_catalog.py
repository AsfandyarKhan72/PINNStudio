"""
One catalogue of every Quick Example -- PINNStudio's own templates and
DeepXDE's examples -- shared by the GUI's Quick Examples lists and the
README's Examples table, so both show the same names in the same order.

Names follow one style, "Equation Dimension – detail" (the detail only
where it is needed: boundary conditions, method, geometry, the paper, or
what an inverse example learns), and every name is unique, so the GUI
shows the name alone. The README adds where each example comes from: the
DeepXDE script (code_url) or the template's paper (PAPERS). Each entry keeps its internal key -- the template's or
DeepXDE example's original name -- which saved projects, tests and the
code that sets a template up still use; only the shown name changed.

Order: ODE, 1D, 2D, 3D; within each, steady before time-dependent and
forward before inverse, same problems next to each other, Allen-Cahn and
then Cahn-Hilliard first.
"""
import re
from dataclasses import dataclass

from pinnstudio.core import deepxde_examples as dx
from pinnstudio.core import pinnstudio_examples as ps

PINNSTUDIO = "PINNStudio"
DEEPXDE = "DeepXDE"
FORWARD = "Forward"
INVERSE = "Inverse"
CATEGORIES = (FORWARD, INVERSE)
BOTH = "Forward / Inverse"   # an example with both modes (every template, some DeepXDE examples)
DIMENSIONS = ("ODE", "1D", "2D", "3D")


@dataclass(frozen=True)
class Entry:
    key: str            # internal name (template name / DeepXDE example name)
    name: str           # shown name, "Equation Dimension – detail"
    source: str         # PINNSTUDIO or DEEPXDE
    dim: str            # ODE / 1D / 2D / 3D
    regime: str         # "Steady" / "Time-dependent"
    problem: str        # FORWARD, INVERSE or BOTH (has an Inverse mode)
    geometry: str
    reference: str      # paper or DeepXDE source URL
    hidden: bool = False  # not listed (a DeepXDE example covers the same problem); still loadable by key

    @property
    def label(self):
        """How the entry appears in the Quick Examples list (the name alone)."""
        return self.name


# PINNStudio templates: key, shown name, dimension, regime, geometry, paper.
_TEMPLATES = [
    ("1D Heat", "Heat 1D", "1D", "Time-dependent", "Interval", ""),
    ("1D Allen-Cahn", "Allen-Cahn 1D – Wight & Zhao (time-adaptive)", "1D", "Time-dependent", "Interval",
     "Wight & Zhao (2021)"),
    ("1D Allen-Cahn (whole interval)", "Allen-Cahn 1D – Wight & Zhao (whole interval)", "1D", "Time-dependent",
     "Interval", "Wight & Zhao (2021)"),
    ("1D Cahn-Hilliard", "Cahn-Hilliard 1D – Wight & Zhao", "1D", "Time-dependent", "Interval", "Wight & Zhao (2021)"),
    ("1D Cahn-Hilliard (Mattey & Ghosh)", "Cahn-Hilliard 1D – Mattey & Ghosh", "1D", "Time-dependent", "Interval",
     "Mattey & Ghosh (2022)"),
    ("1D Burgers", "Burgers 1D – Raissi et al.", "1D", "Time-dependent", "Interval", "Raissi et al. (2019)"),
    ("1D Schrödinger", "Schrödinger 1D – Raissi et al.", "1D", "Time-dependent", "Interval", "Raissi et al. (2019)"),
    ("2D Heat", "Heat 2D", "2D", "Time-dependent", "Rectangle", ""),
    ("2D Allen-Cahn (Mattey & Ghosh)", "Allen-Cahn 2D – Mattey & Ghosh", "2D", "Time-dependent", "Rectangle",
     "Mattey & Ghosh (2022)"),
    ("2D Allen-Cahn (Wight & Zhao)", "Allen-Cahn 2D – Wight & Zhao", "2D", "Time-dependent", "Rectangle",
     "Wight & Zhao (2021)"),
    ("2D Cahn-Hilliard (Mattey & Ghosh)", "Cahn-Hilliard 2D – Mattey & Ghosh", "2D", "Time-dependent", "Rectangle",
     "Mattey & Ghosh (2022)"),
    ("2D Cahn-Hilliard (Wight & Zhao)", "Cahn-Hilliard 2D – Wight & Zhao", "2D", "Time-dependent", "Rectangle",
     "Wight & Zhao (2021)"),
    ("2D Burgers (Mathias)", "Burgers 2D – Mathias et al.", "2D", "Time-dependent", "Rectangle", "Mathias et al. (2022)"),
    ("2D Poisson (L-Shape)", "Poisson 2D – L-shape, Lu et al.", "2D", "Steady", "Polygon (L-shape)", "Lu et al. (2021)"),
    ("2D Poisson (Disk)", "Poisson 2D – disk", "2D", "Steady", "Disk", ""),
    ("3D Heat", "Heat 3D", "3D", "Time-dependent", "Cuboid", ""),
    ("3D Allen-Cahn (Wight & Zhao)", "Allen-Cahn 3D – Wight & Zhao", "3D", "Time-dependent", "Cuboid",
     "Wight & Zhao (2021)"),
    ("3D Poisson (Sphere)", "Poisson 3D – sphere", "3D", "Steady", "Sphere", ""),
]
TEMPLATE_KEYS = tuple(t[0] for t in _TEMPLATES)

# Templates not listed in Quick Examples or the README because a DeepXDE
# example is the same problem (and has the same Inverse mode): 1D Heat is
# DeepXDE's Heat 1D – Dirichlet, 1D Schrödinger DeepXDE's Schrödinger 1D,
# 2D Poisson (L-Shape) DeepXDE's Poisson 2D – L-shape.
# They stay loadable by name, so saved projects and scripts keep working.
HIDDEN_TEMPLATES = ("1D Heat", "1D Schrödinger", "2D Poisson (L-Shape)")

# Phase-field problems lead their group, Allen-Cahn before Cahn-Hilliard;
# within each, DeepXDE's example, then Wight & Zhao, then Mattey & Ghosh.
_FAMILY_FIRST = ("allen-cahn", "cahn-hilliard")
_PAPER_ORDER = ("Wight & Zhao", "Mattey & Ghosh")

# The templates' papers (Entry.reference) -> DOI / arXiv link.
PAPERS = {
    "Raissi et al. (2019)": "https://doi.org/10.1016/j.jcp.2018.10.045",
    "Wight & Zhao (2021)": "https://doi.org/10.4208/cicp.OA-2020-0086",
    "Mattey & Ghosh (2022)": "https://doi.org/10.1016/j.cma.2021.114474",
    "Mathias et al. (2022)": "https://arxiv.org/abs/2301.07824",
    "Lu et al. (2021)": "https://doi.org/10.1137/19M1274067",
    **ps.PAPERS,
}

_DEEPXDE_CODE_ROOT = "https://github.com/lululxvi/deepxde/blob/master/examples/"
# DeepXDE documentation page -> the example's script in DeepXDE's repository.
_DOC_TO_SCRIPT = {
    "ode.system": "ode_system.py", "Lotka.Volterra": "Lotka_Volterra.py", "ode.2nd": "ode_2nd.py", "ide": "ide.py",
    "poisson.1d.dirichlet": "Poisson_Dirichlet_1d.py", "poisson.1d.neumanndirichlet": "Poisson_Neumann_1d.py",
    "poisson.1d.dirichletrobin": "Poisson_Robin_1d.py", "poisson.1d.dirichletperiodic": "Poisson_periodic_1d.py",
    "poisson.1d.pointsetoperator": "Poisson_PointSetOperator_1d.py", "poisson.Lshape": "Poisson_Lshape.py",
    "laplace.disk": "Laplace_disk.py", "eulerbeam": "Euler_beam.py", "elasticity.plate": "elasticity_plate.py",
    "helmholtz.2d.dirichlet": "Helmholtz_Dirichlet_2d.py", "helmholtz.2d.neumann.hole": "Helmholtz_Neumann_2d_hole.py",
    "Kovasznay.flow": "Kovasznay_flow.py", "burgers": "Burgers.py", "burgers.rar": "Burgers_RAR.py", "heat": "heat.py",
    "heat.resample": "heat_resample.py", "diffusion.1d": "diffusion_1d.py", "diffusion.1d.exactBC": "diffusion_1d_exactBC.py",
    "diffusion.1d.resample": "diffusion_1d_resample.py", "diffusion.reaction": "diffusion_reaction.py",
    "allen.cahn": "Allen_Cahn.py", "klein.gordon": "Klein_Gordon.py", "black_scholes": "black_scholes_call.py",
    "Lorenz.inverse": "Lorenz_inverse.py", "Lorenz.inverse.forced": "Lorenz_inverse_forced.py",
    "diffusion.1d.inverse": "diffusion_1d_inverse.py", "reaction.inverse": "reaction_inverse.py",
    "elliptic.inverse.field": "elliptic_inverse_field.py",
}


def code_url(source):
    """The DeepXDE example's script on GitHub, from its documented source
    (a documentation page or already the script)."""
    if "github.com" in source:
        return source
    folder, page = source.split("/demos/")[1].split("/")
    return _DEEPXDE_CODE_ROOT + folder + "/" + _DOC_TO_SCRIPT[page[:-len(".html")]]

# DeepXDE examples: internal name -> shown name.
DEEPXDE_NAMES = {
    "ODE system": "Linear ODE system",
    "Second-order ODE": "Second-order ODE",
    "Lotka-Volterra equation": "Lotka-Volterra ODE",
    "Integro-differential equation": "Integro-differential ODE",
    "Volterra integro-differential equation": "Integro-differential ODE – Volterra kernel",
    "Inverse: Lorenz system": "Lorenz ODE – unknown C₁, C₂, C₃",
    "Inverse: Lorenz system with exogenous input": "Lorenz ODE – unknown C₁, C₂, C₃, exogenous input",
    "Euler beam": "Euler beam 1D",
    "Fractional Poisson equation 1D": "Fractional Poisson 1D",
    "Poisson 1D – Dirichlet": "Poisson 1D – Dirichlet",
    "Poisson 1D – Dirichlet/Neumann": "Poisson 1D – Dirichlet/Neumann",
    "Poisson 1D – Dirichlet/Robin": "Poisson 1D – Dirichlet/Robin",
    "Poisson 1D – Dirichlet/Periodic": "Poisson 1D – Dirichlet/Periodic",
    "Poisson 1D – Dirichlet/PointSetOperator": "Poisson 1D – Dirichlet/point-set operator",
    "Poisson 1D – hard boundary conditions": "Poisson 1D – hard BCs",
    "Poisson 1D – multi-scale Fourier features": "Poisson 1D – multi-scale Fourier features",
    "Inverse: Poisson with unknown forcing field": "Poisson 1D – unknown forcing field",
    "Inverse: Poisson with unknown forcing field (batches)": "Poisson 1D – unknown forcing field (batches)",
    "Inverse: Brinkman-Forchheimer model": "Brinkman-Forchheimer 1D – unknown ν_e, K",
    "Inverse: diffusion-reaction with space-dependent rate": "Diffusion-reaction 1D – unknown rate field",
    "Inverse: fractional Poisson equation 1D": "Fractional Poisson 1D – unknown order α",
    "Allen-Cahn equation": "Allen-Cahn 1D",
    "Burgers equation with RAR": "Burgers 1D – RAR",
    "Heat equation": "Heat 1D – Dirichlet",
    "Heat equation with resampling": "Heat 1D – resampling",
    "Schrödinger equation": "Schrödinger 1D",
    "Diffusion equation": "Diffusion 1D",
    "Diffusion equation with hard IC and BC": "Diffusion 1D – hard IC and BCs",
    "Diffusion equation with resampling": "Diffusion 1D – resampling",
    "Diffusion-reaction equation": "Diffusion-reaction 1D",
    "Fractional diffusion equation": "Fractional diffusion 1D",
    "Klein-Gordon equation": "Klein-Gordon 1D",
    "Black-Scholes equation (call option)": "Black-Scholes 1D – European call",
    "American put option": "Black-Scholes 1D – American put",
    "Wave propagation – space-time multi-scale Fourier features": "Wave 1D – space-time multi-scale Fourier features",
    "Inverse: diffusion equation": "Diffusion 1D – unknown coefficient",
    "Inverse: diffusion-reaction system": "Diffusion-reaction 1D – unknown k_f, D",
    "Fractional Poisson equation 2D": "Fractional Poisson 2D – disk",
    "Helmholtz 2D (Dirichlet)": "Helmholtz 2D – Dirichlet",
    "Helmholtz 2D with a hole": "Helmholtz 2D – square with hole",
    "Helmholtz 2D – hyperparameter optimization": "Helmholtz 2D – hyperparameter search",
    "Helmholtz sound-hard scattering (absorbing BC)": "Helmholtz 2D – sound-hard scattering",
    "Kovasznay flow": "Navier-Stokes 2D – Kovasznay flow",
    "Laplace equation on a disk": "Laplace 2D – disk",
    "Linear elastostatic plate": "Linear elasticity 2D – plate",
    "Linear elastostatic plate (soft BCs)": "Linear elasticity 2D – plate, soft BCs",
    "Poisson over an L-shaped domain": "Poisson 2D – L-shape",
    "Inverse: fractional Poisson equation 2D": "Fractional Poisson 2D – unknown order α",
    "Fractional Poisson equation 3D": "Fractional Poisson 3D – sphere",
    "Beltrami flow": "Navier-Stokes 3D – Beltrami flow",
}

_DIM_ORDER = {d: i for i, d in enumerate(DIMENSIONS)}


def _family(name):
    """The equation part of a shown name ("Burgers 1D – RAR" -> "burgers")."""
    eq = re.split(r"\s+(?:ODE|1D|2D|3D)\b", name)[0]
    return eq.lower()


def _dx_geometry(cfg, key):
    if cfg.problem_dim == "ODE":
        return "— (t only)"
    if key == "Laplace equation on a disk":
        return "Rectangle in (r, θ)"
    g = cfg.geometry_type
    if g == "Custom":
        import json
        shapes = json.loads(cfg.geom_custom_shapes_json or "[]")
        out = shapes[0]["type"] if shapes else "Custom"
        for sh in shapes[1:]:
            out += {"subtract": " − ", "intersect": " ∩ "}.get(sh.get("op"), " + ") + sh["type"]
        return out
    return "Polygon (L-shape)" if g == "Polygon" else g


def _build():
    entries = []
    for key, name, dim, reg, geo, paper in _TEMPLATES:
        entries.append(Entry(key, name, PINNSTUDIO, dim, reg, BOTH, geo, paper, key in HIDDEN_TEMPLATES))
    for ex in ps.EXAMPLES:          # templates defined by their settings
        cfg = ps.build_config(ex, "/r")
        name, paper = ps.CATALOG[ex.name]
        entries.append(Entry(ex.name, name, PINNSTUDIO, cfg.problem_dim,
                             "Steady" if cfg.steady_state else "Time-dependent",
                             BOTH if ex.inverse else FORWARD, _dx_geometry(cfg, ex.name), paper))
    for ex in dx.EXAMPLES:
        cfg = dx.build_config(ex, "/r")
        problem = INVERSE if ex.category == dx.INVERSE else (BOTH if ex.inverse else FORWARD)
        entries.append(Entry(ex.name, DEEPXDE_NAMES.get(ex.name, ex.name), DEEPXDE, cfg.problem_dim,
                             "Steady" if cfg.steady_state else "Time-dependent", problem,
                             _dx_geometry(cfg, ex.name), ex.source))

    def _paper_rank(e):
        if e.source == DEEPXDE:
            return 0
        return next((i + 1 for i, p in enumerate(_PAPER_ORDER) if p in e.name), len(_PAPER_ORDER) + 1)

    def _key(e):
        fam = _family(e.name)
        lead = _FAMILY_FIRST.index(fam) if fam in _FAMILY_FIRST else len(_FAMILY_FIRST)
        return (_DIM_ORDER[e.dim], 0 if e.regime == "Steady" else 1, 1 if e.problem == INVERSE else 0,
                lead, fam, _paper_rank(e) if fam in _FAMILY_FIRST else 0, e.name.lower(), 0 if e.source == PINNSTUDIO else 1)
    return sorted(entries, key=_key)


ENTRIES = _build()
BY_KEY = {e.key: e for e in ENTRIES}


def display_name(key):
    """The shown name for an internal key (the key itself if unknown)."""
    e = BY_KEY.get(key)
    return e.name if e else key


def for_category(category, include_hidden=False):
    """Entries in one Quick Examples category, in catalogue order: Forward
    has every forward example; Inverse has DeepXDE's inverse examples and
    every example with an Inverse mode (every template, and DeepXDE
    examples with Example.inverse), which load in Inverse mode from there.
    Hidden templates are left out unless `include_hidden`."""
    if category == INVERSE:
        out = [e for e in ENTRIES if e.problem in (INVERSE, BOTH)]
    else:
        out = [e for e in ENTRIES if e.problem != INVERSE]
    return out if include_hidden else [e for e in out if not e.hidden]


def has_inverse_mode(key):
    """True for an example that is forward by default but has an Inverse mode."""
    e = BY_KEY.get(key)
    return bool(e) and e.problem == BOTH


def for_category_all(category):
    """for_category() with the hidden templates included (the Quick Examples
    list adds them as hidden rows, see ExampleCombo.set_entries)."""
    return for_category(category, include_hidden=True)
