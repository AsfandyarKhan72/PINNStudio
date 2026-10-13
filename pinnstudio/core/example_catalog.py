"""
One catalogue of every Quick Example -- PINNStudio's own templates and
DeepXDE's examples -- shared by the GUI's Quick Examples lists and the
README's Examples table, so both show the same names in the same order.

Names follow one style, "Equation Dimension – detail" (the detail only
where it is needed: boundary conditions, method, geometry, the paper, or
what an inverse example learns); the source (PINNStudio / DeepXDE) is a
separate tag. Each entry keeps its internal key -- the template's or
DeepXDE example's original name -- which saved projects, tests and the
code that sets a template up still use; only the shown name changed.

Order: ODE, 1D, 2D, 3D; within each, steady before time-dependent and
forward before inverse, same problems next to each other.
"""
import re
from dataclasses import dataclass

from pinnstudio.core import deepxde_examples as dx

PINNSTUDIO = "PINNStudio"
DEEPXDE = "DeepXDE"
FORWARD = "Forward"
INVERSE = "Inverse"
CATEGORIES = (FORWARD, INVERSE)
DIMENSIONS = ("ODE", "1D", "2D", "3D")


@dataclass(frozen=True)
class Entry:
    key: str            # internal name (template name / DeepXDE example name)
    name: str           # shown name, "Equation Dimension – detail"
    source: str         # PINNSTUDIO or DEEPXDE
    dim: str            # ODE / 1D / 2D / 3D
    regime: str         # "Steady" / "Time-dependent"
    problem: str        # "Forward / Inverse" (templates), "Forward" or "Inverse"
    geometry: str
    reference: str      # paper or DeepXDE source URL

    @property
    def label(self):
        """How the entry appears in the Quick Examples list."""
        return f"{self.name}  ·  {self.source}"


# PINNStudio templates: key, shown name, dimension, regime, geometry, paper.
_TEMPLATES = [
    ("1D Heat", "Heat 1D", "1D", "Time-dependent", "Interval", ""),
    ("1D Allen-Cahn", "Allen-Cahn 1D – Wight & Zhao", "1D", "Time-dependent", "Interval", "Wight & Zhao (2021)"),
    ("1D Burgers", "Burgers 1D – Raissi et al.", "1D", "Time-dependent", "Interval", "Raissi et al. (2019)"),
    ("1D Schrödinger", "Schrödinger 1D – Raissi et al.", "1D", "Time-dependent", "Interval", "Raissi et al. (2019)"),
    ("2D Heat", "Heat 2D", "2D", "Time-dependent", "Rectangle", ""),
    ("2D Allen-Cahn (Mattey & Ghosh)", "Allen-Cahn 2D – Mattey & Ghosh", "2D", "Time-dependent", "Rectangle",
     "Mattey & Ghosh (2022)"),
    ("2D Allen-Cahn (Wight & Zhao)", "Allen-Cahn 2D – Wight & Zhao", "2D", "Time-dependent", "Rectangle",
     "Wight & Zhao (2021)"),
    ("2D Burgers (Mathias)", "Burgers 2D – Mathias et al.", "2D", "Time-dependent", "Rectangle", "Mathias et al. (2022)"),
    ("2D Poisson (L-Shape)", "Poisson 2D – L-shape", "2D", "Steady", "Polygon (L-shape)", "Lu et al. (2021)"),
    ("2D Poisson (Disk)", "Poisson 2D – disk", "2D", "Steady", "Disk", ""),
    ("3D Heat", "Heat 3D", "3D", "Time-dependent", "Cuboid", ""),
    ("3D Poisson (Sphere)", "Poisson 3D – sphere", "3D", "Steady", "Sphere", ""),
]
TEMPLATE_KEYS = tuple(t[0] for t in _TEMPLATES)

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
    "Burgers equation": "Burgers 1D",
    "Burgers equation with RAR": "Burgers 1D – RAR",
    "Heat equation": "Heat 1D",
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
        entries.append(Entry(key, name, PINNSTUDIO, dim, reg, "Forward / Inverse", geo, paper))
    for ex in dx.EXAMPLES:
        cfg = dx.build_config(ex, "/r")
        entries.append(Entry(ex.name, DEEPXDE_NAMES.get(ex.name, ex.name), DEEPXDE, cfg.problem_dim,
                             "Steady" if cfg.steady_state else "Time-dependent",
                             INVERSE if ex.category == dx.INVERSE else FORWARD,
                             _dx_geometry(cfg, ex.name), ex.source))

    def _key(e):
        return (_DIM_ORDER[e.dim], 0 if e.regime == "Steady" else 1, 1 if e.problem == INVERSE else 0,
                _family(e.name), e.name.lower(), 0 if e.source == PINNSTUDIO else 1)
    return sorted(entries, key=_key)


ENTRIES = _build()
BY_KEY = {e.key: e for e in ENTRIES}


def display_name(key):
    """The shown name for an internal key (the key itself if unknown)."""
    e = BY_KEY.get(key)
    return e.name if e else key


def for_category(category):
    """Entries in one Quick Examples category, in catalogue order: Forward
    has every template and DeepXDE's forward examples; Inverse has DeepXDE's
    inverse examples and every template (loaded in Inverse mode)."""
    if category == INVERSE:
        return [e for e in ENTRIES if e.source == PINNSTUDIO or e.problem == INVERSE]
    return [e for e in ENTRIES if e.problem != INVERSE]
