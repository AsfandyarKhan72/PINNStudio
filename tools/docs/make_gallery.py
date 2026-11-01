#!/usr/bin/env python3
"""Generate the documentation site's example gallery from PINNStudio's own
example definitions, so the pages cannot drift from what the software loads.

    python tools/docs/make_gallery.py            # write website/examples/**
    python tools/docs/make_gallery.py --refresh  # also rebuild the template cache
    python tools/docs/make_gallery.py --check    # report only (README/code differences), write nothing

What is read:

* the Examples list: ``pinnstudio/core/example_catalog.py`` (``ENTRIES`` without the
  hidden ones, in list order -- the order and names ``ExampleCombo.set_entries`` shows
  under the toolbar's **Examples:** list, ``for_category_all``);
* DeepXDE examples and settings-defined templates: ``deepxde_examples.build_config`` /
  ``pinnstudio_examples.build_config`` (with ``inverse=`` for an Inverse mode), and their
  ``source``, ``note`` and ``inverse_note``;
* the older PINNStudio templates: ``tools/tuning_study.py: template_config(key, inverse)``,
  which loads the template in an offscreen ``MainWindow`` the way the Examples list does.
  These configurations are cached (``--cache``, default ``/tmp/pinnstudio_gallery_cache.json``)
  and rebuilt automatically when any of the source files they depend on changes;
* **Settings: Default / Tuned**: ``example_settings.settings_for`` and ``has_tuned`` on the
  loaded configuration, exactly as ``MainWindow._apply_example_settings_mode`` calls them;
* the README's per-example sections (display equations, the "where" lines and the source
  paper), ``benchmarks/problems/index.json`` (benchmark record per example),
  ``benchmarks/tuning/*.pinn.json`` (configuration files) and ``reference_data/``.

Images: only existing result images from the repository (``assets/Images``, ``docs/assets``),
copied to ``website/assets/images/examples/``; the list ``IMAGES`` below says which example
each one shows and where it comes from. No numerical results are written on the pages.
"""
import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

WEB = REPO / "website"
OUT = WEB / "examples"
IMG_DIR = WEB / "assets" / "images" / "examples"
REF_ROOT = REPO / "reference_data"
GH = "https://github.com/AsfandyarKhan72/PINNStudio"
GH_BLOB = GH + "/blob/main/"
GH_TREE = GH + "/tree/main/"
DIM_DIRS = {"ODE": "ode", "1D": "1d", "2D": "2d", "3D": "3d"}

# Files whose content decides what a template loads with (cache key).
_CACHE_SOURCES = ["pinnstudio/ui/main_window.py", "pinnstudio/core/example_catalog.py",
                  "pinnstudio/core/config.py", "pinnstudio/core/deepxde_examples.py",
                  "pinnstudio/core/pinnstudio_examples.py", "tools/tuning_study.py"]

# Existing result images (repository files) and the example each one shows.
# (source file, example key, file name on the site, caption, alt text)
IMAGES = [
    ("assets/Images/Top_GIFs/1D_Schrodinger.gif", "Schrödinger equation", "schrodinger-1d.gif",
     "Animation from PINNStudio's README (top of the page): the PINN solution |h| of this example over time.",
     "Animated line plot of the PINN solution |h|(x, t) of the 1D Schrödinger equation"),
    ("assets/Images/Top_GIFs/2D_Allen_Cahn_Wight.gif", "2D Allen-Cahn (Wight & Zhao)", "allen-cahn-2d-wight-zhao.gif",
     "Animation from PINNStudio's README (top of the page): the PINN solution of this example over time.",
     "Animated contour plot of the 2D Allen-Cahn PINN solution: a circular interface shrinking over time"),
    ("assets/Images/Top_GIFs/3D_Heat.gif", "3D Heat", "heat-3d.gif",
     "Animation from PINNStudio's README (top of the page): the PINN solution of this example over time.",
     "Animated 3D voxel plot of the heat-equation PINN solution in the unit cube"),
    ("assets/Images/1D-AllenCahn/param_convergence_animation.gif", "1D Allen-Cahn (whole interval)",
     "allen-cahn-1d-wight-zhao-inverse.gif",
     "Animation from PINNStudio's README (section \"Inverse parameters during training\"): this example in "
     "Inverse mode, the two coefficients gamma_1 and gamma_2 during training, with their true values as dashed lines.",
     "Animated plots of the inferred parameters gamma_1 and gamma_2 against the training iteration"),
    ("assets/Images/2D_AllenCahn-Wight/rar_collocation_points.png", "2D Allen-Cahn (Wight & Zhao)",
     "allen-cahn-2d-wight-zhao-rar-points.png",
     "Figure from PINNStudio's README (section on RAR): a run of this example with RAR switched on, round 1 — "
     "the collocation points added in that round. RAR is not part of the example's Default settings.",
     "Collocation points of a 2D Allen-Cahn run after RAR round 1, the added points near the interface"),
    ("assets/Images/2D_AllenCahn-Wight/rar_solution_plot.png", "2D Allen-Cahn (Wight & Zhao)",
     "allen-cahn-2d-wight-zhao-rar-solution.png",
     "Figure from PINNStudio's README (section on RAR): the same RAR run, PINN, reference and error after round 1.",
     "PINN solution, reference solution and error of a 2D Allen-Cahn run after RAR round 1"),
    ("assets/Images/1D-Burgers/rar_collocation_points.png", "Burgers equation with RAR", "burgers-1d-rar-points.png",
     "Figure from PINNStudio's README (section on RAR): a Burgers 1D run with RAR, round 2 — original points, "
     "points added in earlier rounds and points added in this round. The README does not state which example's "
     "settings the run used.",
     "Collocation points of a Burgers 1D RAR run after round 2, the added points along the shock at x = 0"),
    ("assets/Images/1D-Burgers/rar_solution_plot.png", "Burgers equation with RAR", "burgers-1d-rar-solution.png",
     "Figure from PINNStudio's README (section on RAR): the same Burgers 1D RAR run, PINN, reference and error after round 2.",
     "PINN solution, reference solution and error over (t, x) of a Burgers 1D RAR run after round 2"),
    ("assets/Images/3D_AllenCahn-Wight/isosurface.png", "3D Allen-Cahn (Wight & Zhao)", "allen-cahn-3d-wight-zhao-isosurface.png",
     "Figure from the repository's benchmark notes (benchmarks/paper.md): a run of this example in Inverse mode — "
     "the PINN interface phi = 0 at five times (top) and the error on the cube with one corner cut away (bottom).",
     "Spherical interface of the 3D Allen-Cahn PINN shrinking at five times, and the error inside the cube"),
    ("assets/Images/3D_AllenCahn-Wight/radius.png", "3D Allen-Cahn (Wight & Zhao)", "allen-cahn-3d-wight-zhao-radius.png",
     "Figure from the repository's benchmark notes (benchmarks/paper.md): the same Inverse-mode run, the drop's radius "
     "over time — PINN, reference and the sharp-interface limit.",
     "Radius of the 3D Allen-Cahn drop over time: PINN, reference and sharp-interface limit"),
    ("assets/Images/DeepXDE/Helmholtz_2D_hyperparameter_optimization/surface_comparison.png",
     "Helmholtz 2D – hyperparameter optimization", "helmholtz-2d-hpo-surface.png",
     "Figure from PINNStudio's README (this example's results): a run with DeepXDE's settings (Default) — "
     "PINN, reference and error.",
     "Heat maps of the PINN solution, the reference solution and the error of the 2D Helmholtz problem"),
]

# Index groups (title, anchor, description, predicate on the page record).
_FLUID = ("burgers", "navier-stokes", "brinkman-forchheimer")
_SOLID = ("euler beam", "linear elasticity")
_PHASE = ("allen-cahn", "cahn-hilliard")


# ----------------------------------------------------------------- helpers
def _slug(text):
    t = text.lower().replace("ö", "o").replace("α", "alpha").replace("ν", "nu")
    t = re.sub(r"[₀-₉]", lambda m: str(ord(m.group(0)) - 0x2080), t)
    t = re.sub(r"[^a-z0-9]+", "-", t)
    return t.strip("-")


def _rel_ref(path):
    """A path inside reference_data, relative to the repository ('' otherwise)."""
    p = str(path).replace("\\", "/")
    i = p.find("reference_data/")
    return p[i:] if i >= 0 else ""


def _md_math(text):
    """README Markdown math ($...$, $$...$$) -> arithmatex delimiters."""
    text = re.sub(r"\$`(.+?)`\$", r"$\1$", text)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: "\\[" + m.group(1).replace("\\Large", "").strip() + "\\]", text)
    return re.sub(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$", lambda m: "\\(" + m.group(1) + "\\)", text)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f) and abs(f) < 1e7:
        return str(int(f))
    return f"{f:g}"


def _code(s):
    s = str(s)
    tick = "``" if "`" in s else "`"
    return f"{tick}{s}{tick}" if s.strip() else "—"


def _cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _sha():
    h = hashlib.sha256()
    for f in _CACHE_SOURCES:
        h.update((REPO / f).read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------- configurations
def _template_configs(entries, cache_path, refresh):
    """{(key, inverse): asdict(PINNConfig)} for the GUI-built templates, cached."""
    from pinnstudio.core import example_catalog as cat
    want = []
    for e in entries:
        if e.key in cat.TEMPLATE_KEYS:
            if e.problem != cat.INVERSE:
                want.append((e.key, False))
            if e.problem in (cat.INVERSE, cat.BOTH):
                want.append((e.key, True))
    sha = _sha()
    data = {}
    if cache_path.exists() and not refresh:
        try:
            raw = json.loads(cache_path.read_text())
            if raw.get("sha") == sha and raw.get("repo") == str(REPO):
                data = raw["configs"]
        except (ValueError, KeyError):
            data = {}
    missing = [w for w in want if f"{w[0]}||{int(w[1])}" not in data]
    if missing:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import tuning_study
        for key, inv in missing:
            print(f"  building template configuration: {key} ({'Inverse' if inv else 'Forward'})", flush=True)
            cfg = tuning_study.template_config(key, inv)
            data[f"{key}||{int(inv)}"] = dataclasses.asdict(cfg)
        cache_path.write_text(json.dumps({"sha": sha, "repo": str(REPO), "configs": data}))
    return {(k.split("||")[0], bool(int(k.split("||")[1]))): v for k, v in data.items()}


def _config(entry, inverse, tcache):
    """The configuration the Examples list loads (before Settings: Default/Tuned)."""
    from pinnstudio.core import deepxde_examples as dx, pinnstudio_examples as pse
    from pinnstudio.core.config import PINNConfig
    if (entry.key, inverse) in tcache:
        d = tcache[(entry.key, inverse)]
        fields = {f.name for f in dataclasses.fields(PINNConfig)}
        return PINNConfig(**{k: v for k, v in d.items() if k in fields})
    ex = dx.BY_NAME.get(entry.key) or pse.BY_NAME[entry.key]
    # what MainWindow._on_quick_example_selected_impl passes: the Inverse-mode version only for a
    # forward example picked from the Inverse list
    return dx.build_config(ex, str(REF_ROOT), inverse=inverse and bool(ex.inverse))


def _phases(cfg):
    from pinnstudio.core import example_settings as es
    return es._phases(cfg)


# ----------------------------------------------------------------- README
def _readme_sections():
    """{anchor: lines of the README's per-example section, up to its results}."""
    text = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    out, cur, buf = {}, None, []
    for line in text:
        m = re.match(r'<details id="([^"]+)">', line)
        if m:
            if cur:
                out[cur] = buf
            cur, buf = m.group(1), []
            continue
        if cur is not None:
            if line.startswith("**Results") or line.startswith("<details>"):
                out[cur] = buf
                cur, buf = None, []
            else:
                buf.append(line)
    return out


def _readme_problem_column():
    """{shown name: Problem column} from the README's Examples table."""
    out = {}
    for line in (REPO / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| \d+ \| \[([^\]]+)\]\([^)]*\) \| [^|]+\| [^|]+\| ([^|]+)\|", line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def _readme_math(lines):
    """(display equations of the problem, inverse-mode equations, 'where' lines, source-paper line,
    'benchmark problem after' line, inverse-mode paragraph)."""
    eqs, inv_eqs, where, paper, after, inv_par = [], [], [], "", "", ""
    in_inv = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("*Inverse mode*"):
            in_inv = True
            inv_par = line
            continue
        if line.startswith("$$"):
            (inv_eqs if in_inv else eqs).append(_md_math(line))
        elif line.startswith("where "):
            where.append(_md_math(line))
        elif line.startswith("*Source paper:*"):
            paper = _md_math(line[len("*Source paper:*"):].strip())
        elif line.startswith("Benchmark problem after"):
            after = _md_math(line.split(". ")[0].rstrip(".") + ".")
    return eqs, inv_eqs, where, paper, after, _md_math(inv_par)


ANCHOR_PAGES = {}     # README anchor -> page path (filled in main)


def _fix_links(text, here):
    """README-relative links in text taken from the README -> site pages or GitHub."""
    def repl(m):
        label, target = m.group(1), m.group(2)
        if target.startswith("#"):
            page = ANCHOR_PAGES.get(target[1:])
            if page:
                return f"[{label}]({os.path.relpath(page, os.path.dirname(here))})"
            return label
        if re.match(r"https?://", target):
            return m.group(0)
        path = target.split("?")[0]
        base = GH_TREE if path.endswith("/") or "." not in os.path.basename(path) else GH_BLOB
        return f"[{label}]({base}{path})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", repl, text)


# ----------------------------------------------------------------- page parts
def _dim_text(cfg):
    if cfg.problem_dim == "ODE":
        return "ODE (t)"
    ax = {"1D": "x", "2D": "x, y", "3D": "x, y, z"}[cfg.problem_dim]
    return f"{cfg.problem_dim} ({ax}{'' if cfg.steady_state else ', t'})"


def _domain_lines(cfg, entry):
    g, out = cfg.geometry_type, []
    b = lambda lo, hi: f"[{_num(lo)}, {_num(hi)}]"
    if cfg.problem_dim == "ODE":
        pass
    elif g == "Interval":
        out.append(f"Interval: \\(x \\in {b(cfg.x_min, cfg.x_max)}\\)")
    elif g == "Rectangle":
        label = "Rectangle"
        if "(r, θ)" in entry.geometry:
            label = "Rectangle in polar coordinates (x is r, y is θ)"
        out.append(f"{label}: \\(x \\in {b(cfg.x_min, cfg.x_max)}\\), \\(y \\in {b(cfg.y_min, cfg.y_max)}\\)")
    elif g == "Cuboid":
        out.append(f"Cuboid: \\(x \\in {b(cfg.x_min, cfg.x_max)}\\), \\(y \\in {b(cfg.y_min, cfg.y_max)}\\), "
                   f"\\(z \\in {b(cfg.z_min, cfg.z_max)}\\)")
    elif g == "Disk":
        out.append(f"Disk: centre ({_num(cfg.geom_center_x)}, {_num(cfg.geom_center_y)}), radius {_num(cfg.geom_radius)}")
    elif g == "Sphere":
        out.append(f"Sphere: centre ({_num(cfg.geom_center_x)}, {_num(cfg.geom_center_y)}, {_num(cfg.geom_center_z)}), "
                   f"radius {_num(cfg.geom_radius)}")
    elif g == "Polygon":
        verts = "; ".join(f"({v.strip()})" for v in cfg.geom_polygon_vertices.split(";") if v.strip())
        out.append(f"Polygon with vertices {verts}")
    elif g == "Custom":
        shapes = json.loads(cfg.geom_custom_shapes_json or "[]")
        parts = []
        for i, sh in enumerate(shapes):
            p = ", ".join(f"{k} = {_num(v)}" for k, v in (sh.get("params") or {}).items())
            op = "" if i == 0 else {"subtract": "minus ", "intersect": "intersected with ", "union": "plus "}.get(
                sh.get("op", "union"), (sh.get("op") or "plus") + " ")
            parts.append(f"{op}{sh['type']} ({p})")
        out.append("Custom geometry (**Geometry** › Custom): " + ", ".join(parts))
    else:
        out.append(f"{g}")
    if not cfg.steady_state:
        out.append(f"Time interval: \\(t \\in {b(cfg.t_min, cfg.t_max)}\\)")
    return out


def _bc_rows(cfg, bc_labels):
    rows = json.loads(cfg.custom_bc_json or "[]")
    outs = [o.strip() for o in (cfg.output_names or "u").split(",")]
    table = []
    for r in rows:
        t = r.get("type", "")
        comp = int(r.get("component", 0) or 0)
        out_name = outs[comp] if comp < len(outs) else str(comp)
        extra = []
        value = r.get("value", "")
        if t == "periodic":
            extra.append(f"axis {r.get('axis', 'x')}, derivative order {r.get('deriv_order', 0)}")
            value = ""
        if t in ("operator", "pointset_operator"):
            out_name = "—"
        if r.get("apply_at") and r.get("apply_at") != "boundary":
            extra.append(f"applied at the {r['apply_at']} time" if r["apply_at"] == "initial" else f"applied at {r['apply_at']}")
        if t in ("pointset", "pointset_operator"):
            if r.get("points_mode") == "list":
                extra.append(f"points: {r.get('points_list')}")
            elif r.get("points_file"):
                extra.append(f"points from `{_rel_ref(r['points_file']) or os.path.basename(r['points_file'])}`")
            if t == "pointset_operator":
                if r.get("values_mode") == "formula" and r.get("values_formula"):
                    extra.append(f"operator `{value}` equals `{r['values_formula']}` at the points")
                else:
                    extra.append(f"operator `{value}` equals the values in the file")
            elif r.get("values_mode") == "formula" and r.get("values_formula"):
                value = r["values_formula"]
            elif r.get("points_file") and t == "pointset":
                value = "(values from the file)"
            if int(r.get("batch_size", 0) or 0) > 0:
                extra.append(f"batch size {r['batch_size']}")
        if t == "robin":
            extra.append("value is the right-hand side of du/dn = f(x, u)")
        where = r.get("location", "") if t not in ("pointset", "pointset_operator") else "—"
        if r.get("apply_at") == "initial":
            where = "initial time"
        elif where.strip() == "True":
            where = "True (the whole boundary)"
        table.append((bc_labels.get(t, t), where, out_name, value, "; ".join(extra)))
    return table


def _ic_rows(cfg):
    if cfg.steady_state:
        return []
    outs = [o.strip() for o in (cfg.output_names or "u").split(",")]
    ics = (cfg.ic_expressions or "").split("|")
    act = [a.strip() == "True" for a in (cfg.ic_active or "").split(",")]
    hard = [h.strip() == "True" for h in (cfg.ic_hard or "").split(",")]
    rows = []
    for i, o in enumerate(outs):
        expr = ics[i].strip() if i < len(ics) else ""
        a = i < len(act) and act[i]
        h = i < len(hard) and hard[i]
        if h:
            how = "exact (built into the network output)"
        elif a:
            how = "loss term"
        else:
            how = "not used as a loss term"
        rows.append((o, expr, how))
    return rows


def _network(cfg):
    lay = list(cfg.layers)
    hidden = lay[1:-1]
    if hidden and all(h == hidden[0] for h in hidden):
        size = f"{len(hidden)} × {hidden[0]}"
    else:
        size = " – ".join(str(h) for h in hidden)
    kind = {"MsFFN": "multi-scale Fourier-feature network (MsFFN)",
            "STMsFFN": "space-time multi-scale Fourier-feature network (STMsFFN)",
            "PFNN": "parallel FNN (PFNN)"}.get(cfg.network_type, cfg.network_type)
    if cfg.problem_dim == "ODE":
        coords = "t"
    else:
        coords = {"1D": "x", "2D": "x, y", "3D": "x, y, z"}[cfg.problem_dim] + ("" if cfg.steady_state else ", t")
    s = f"{kind}, hidden layers {size}, inputs ({coords}), {lay[-1]} output{'s' if lay[-1] != 1 else ''}; " \
        f"activation {cfg.activation}, initializer {cfg.kernel_initializer}"
    if cfg.network_type == "MsFFN":
        s += f"; Fourier-feature scales {cfg.msffn_sigmas}"
    elif cfg.network_type == "STMsFFN":
        s += f"; scales {cfg.msffn_sigmas} (space), {cfg.msffn_sigmas_t} (time)"
    return s


def _points(cfg):
    p = [f"{cfg.num_domain:,} domain", f"{cfg.num_boundary:,} boundary"]
    if not cfg.steady_state and cfg.problem_dim != "ODE":
        p.append(f"{cfg.num_initial:,} initial")
    p.append(f"{cfg.num_test:,} test")
    return ", ".join(p)


def _methods(cfg):
    """Training methods and exact constraints a configuration uses (from its fields)."""
    m = []
    if cfg.time_adaptive:
        try:
            groups = json.loads(cfg.ta_step_groups or "[]")
        except ValueError:
            groups = []
        n = sum(int(g.get("steps", 0)) for g in groups) if groups else int(cfg.ta_num_steps)
        s = f"time windows ({n} windows"
        s += ", transfer learning" if cfg.ta_transfer_learning else ""
        s += ")"
        m.append(s)
        if cfg.ta_backward:
            m.append("backward-compatible PINN (bc-PINN) across the windows")
        if cfg.ta_growing:
            m.append("growing time range")
        if float(cfg.ta_icgl_fraction or 0) > 0:
            m.append("initial-condition guided learning")
        if int(cfg.ta_resample_rounds or 0) > 0:
            m.append("adaptive resampling in each window")
    if cfg.adapt_method == "RAR":
        m.append(f"residual-based adaptive refinement (RAR, {cfg.rar_cycles} rounds of {cfg.rar_add_points} points)")
    if cfg.causal_training:
        m.append("causal training")
    if cfg.point_weights in ("RBA", "SA-PINN"):
        m.append(f"{cfg.point_weights} point weights")
    if cfg.network_type in ("MsFFN", "STMsFFN"):
        m.append("multi-scale Fourier features")
    if cfg.cb_point_resampler:
        m.append(f"collocation-point resampling every {cfg.cb_point_resampler_period} iterations")
    if any(h.strip() == "True" for h in (cfg.ic_hard or "").split(",")):
        m.append("exact initial condition")
    if cfg.periodic_inputs:
        m.append("exact periodicity (periodic input features)")
    if cfg.output_transform_enabled and cfg.output_transform_mode == "formula":
        m.append(f"output transform `{cfg.output_transform_formula}`")
    if cfg.input_transform_enabled and cfg.input_transform_mode == "formula":
        m.append(f"input features `{cfg.input_feature_formula}`")
    if cfg.ic_pretrain:
        m.append("initial-condition pre-training")
    if float(cfg.loss_weights_from_initial or 0) > 0:
        m.append("loss weights set from the initial losses")
    return m


# Each method named on an example page links to the reference page that explains it (pages are examples/<dim>/x.md).
_METHOD_PAGES = [("bc-PINN", "time-windows.md"), ("growing time range", "time-windows.md"),
                 ("resampling in each window", "time-windows.md"), ("time windows", "time-windows.md"),
                 ("guided learning", "pretraining.md"), ("pre-training", "pretraining.md"),
                 ("RAR", "rar.md"), ("causal training", "causal.md"), ("point weights", "point-weights.md"),
                 ("Fourier features", "networks.md"), ("output transform", "networks.md"),
                 ("input features", "networks.md"), ("exact initial condition", "exact-constraints.md"),
                 ("exact periodicity", "exact-constraints.md"), ("initial losses", "loss-weights.md")]


def _link_method(text):
    for key, page in _METHOD_PAGES:
        if key in text:
            url = "../../reference/" + page
            return f"[{text}]({url})" if "`" not in text else f"{text} ([reference]({url}))"
    return text


def _strategies(cfg):
    s = set()
    if cfg.time_adaptive:
        s.add("time windows")
    if cfg.adapt_method == "RAR":
        s.add("RAR")
    if cfg.causal_training:
        s.add("causal training")
    if cfg.point_weights in ("RBA", "SA-PINN"):
        s.add(cfg.point_weights)
    if cfg.network_type in ("MsFFN", "STMsFFN"):
        s.add("Fourier features")
    return s


def _phase_table(cfg):
    lines = ["| Phase | Optimizer | Iterations | Learning rate | Decay |", "|---|---|---|---|---|"]
    for i, p in enumerate(_phases(cfg), 1):
        opt = {"adam": "Adam", "lbfgs": "L-BFGS"}.get(str(p.get("optimizer", "")).lower(), p.get("optimizer"))
        lr = "—" if opt == "L-BFGS" else _num(p.get("lr"))
        dec = p.get("decay_type") or "none"
        if dec != "none":
            dec = f"{dec} ({_num(p.get('decay_p1'))}, {_num(p.get('decay_p2'))})"
        lines.append(f"| {i} | {opt} | {int(p.get('iterations', 0)):,} | {lr} | {dec} |")
    return lines


def _settings_block(cfg, title_note=""):
    out = [f"- **Network:** {_network(cfg)}", f"- **Points:** {_points(cfg)}"]
    ph = _phases(cfg)
    w = (ph[0].get("weights") if ph and ph[0].get("weights") else cfg.loss_weights_multi) or ""
    if w.strip():
        from pinnstudio.core import paper_settings as _ps
        n_terms = sum(_ps._n_terms(cfg)) if not cfg.steady_state else sum(_ps._n_terms(cfg)[:2])
        inv = len([x for x in w.split(",") if x.strip()]) > n_terms
        same = all((p.get("weights") or w) == w for p in ph)
        out.append(f"- **Loss weights:** `{w}` (equations, then boundary-condition rows"
                   + ("" if cfg.steady_state else ", then initial conditions")
                   + (", then measured data" if inv else "") + ")" + ("" if same else " in phase 1; see the phase table"))
    meth = _methods(cfg)
    if meth:
        out.append("- **Methods:** " + "; ".join(_link_method(x) for x in meth))
    out += ["", *_phase_table(cfg)]
    return out


# ----------------------------------------------------------------- one example
def _record(entry, tcache, readme, bench_index, tuned_files, bc_labels):
    from pinnstudio.core import example_catalog as cat, example_settings as es, paper_settings as ps
    from pinnstudio.core import deepxde_examples as dx, pinnstudio_examples as pse
    modes = []
    if entry.problem != cat.INVERSE:
        modes.append(False)
    if entry.problem in (cat.INVERSE, cat.BOTH):
        modes.append(True)
    rec = {"entry": entry, "modes": {}}
    for inv in modes:
        cfg = _config(entry, inv, tcache)
        gui_inverse = cfg.problem_type == "Inverse"       # the radio button the GUI passes to settings_for
        d_cfg, _ = es.settings_for(cfg, entry.key, gui_inverse, es.DEFAULT)
        has_t = es.has_tuned(entry.key, gui_inverse)
        t_cfg, t_msg = (es.settings_for(cfg, entry.key, gui_inverse, es.TUNED) if has_t else (None, ""))
        t_msg = re.sub(r"\s*\(benchmarks: [^)]*\)", "", t_msg or "")
        paper_default = (not gui_inverse and entry.key in ps.TEMPLATE_PAPER
                         and (entry.key, ps.TEMPLATE_PAPER[entry.key]) in ps.PAPER)
        tt = es.tuned_table().get(entry.key, {})
        unoffered = inv and not gui_inverse and not has_t and bool(tt.get("inverse"))
        rec["modes"][inv] = dict(cfg=cfg, default=d_cfg, tuned=t_cfg, tuned_msg=t_msg, has_tuned=has_t,
                                 tuned_unoffered=unoffered,
                                 paper_default=ps.PAPER_LABEL.get(ps.TEMPLATE_PAPER.get(entry.key)) if paper_default else "",
                                 gui_inverse=gui_inverse)
    ex = dx.BY_NAME.get(entry.key) or pse.BY_NAME.get(entry.key)
    rec["example"] = ex
    b = bench_index.get(entry.name)
    rec["readme"] = _readme_math(readme.get(b["anchor"], [])) if b else ([], [], [], "", "", "")
    rec["bench"] = b
    rec["tuned_files"] = tuned_files.get(entry.name, [])
    strategies = set()
    for m in rec["modes"].values():
        strategies |= _strategies(m["default"])
        if m["tuned"] is not None:
            strategies |= _strategies(m["tuned"])
    rec["strategies"] = strategies
    any_cfg = rec["modes"][modes[0]]["cfg"]
    rec["n_eq"] = len([e for e in (any_cfg.pde_expressions or "").split("|") if e.strip()])
    rec["family"] = cat._family(entry.name)
    rec["path"] = f"{DIM_DIRS[entry.dim]}/{_slug(entry.name)}.md"
    return rec


def _ref_folders(cfg):
    """Reference-data folders (relative) used by Error Analysis and by the measured data."""
    ea, obs = [], []
    try:
        for item in ast.literal_eval(cfg.ea_files or "[]"):
            r = _rel_ref(item[1])
            if r:
                ea.append(os.path.dirname(r))
    except (ValueError, SyntaxError):
        pass
    for o in json.loads(cfg.inverse_obs_files_json or "[]") if (cfg.inverse_obs_files_json or "").strip() else []:
        r = _rel_ref(o.get("path", ""))
        if r:
            obs.append(r)
    seen = []
    for f in ea:
        if f not in seen:
            seen.append(f)
    return seen, obs


def _gui_path(entry, inverse, tuned):
    cat_name = "Inverse" if inverse else "Forward"
    return (f"toolbar **Examples:** › **{cat_name}** › under the **── {entry.dim} ──** heading, "
            f"**{entry.name}** › **Default**" + (" or **Tuned**" if tuned else ""))


def _page(rec, images, root_rel):
    from pinnstudio.core import example_catalog as cat
    e, ex = rec["entry"], rec["example"]
    modes = rec["modes"]
    first = modes[min(modes)]           # forward if it exists
    cfg0 = first["cfg"]
    eqs, inv_eqs, where, paper_line, after_line, inv_par = rec["readme"]
    here = rec["path"]
    where = [_fix_links(w, here) for w in where]
    paper_line, after_line, inv_par = (_fix_links(x, here) for x in (paper_line, after_line, inv_par))
    L = ["---", "search:", "  boost: 0.5", "---", "", f"# {e.name}", ""]
    src = "PINNStudio template" if e.source == cat.PINNSTUDIO else "DeepXDE example"
    mode_txt = " and ".join("Inverse" if m else "Forward" for m in sorted(modes))
    if e.problem == cat.INVERSE:
        mode_txt = "Inverse (DeepXDE inverse example)"
    elif e.problem == cat.BOTH and e.source == cat.DEEPXDE:
        mode_txt += " (the Inverse mode is PINNStudio's addition)"
    L += ["| Example | |", "|---|---|",
          f"| **Source** | {src} |",
          f"| **Problem** | {mode_txt} |",
          f"| **Dimension** | {_dim_text(cfg0)} |",
          f"| **Regime** | {e.regime} |",
          f"| **Geometry** | {_cell(e.geometry)} |",
          f"| **Outputs** | {_cell(', '.join('`' + o.strip() + '`' for o in cfg0.output_names.split(',')))} |",
          f"| **Settings** | Default{', Tuned' if any(m['has_tuned'] for m in modes.values()) else ' (no separate Tuned preset)'} |",
          ""]
    for img in images:
        L += ['<figure markdown="span">',
              f"  ![{img[4]}]({root_rel}assets/images/examples/{img[2]}){{ loading=lazy }}",
              f"  <figcaption>{img[3]}</figcaption>",
              "</figure>", ""]

    # equations
    L += ["## Governing equations", ""]
    if eqs:
        L += ["Mathematical form (from the README):", ""]
        for q in eqs:
            L += [q, ""]
        for w in where:
            L += [w[0].upper() + w[1:], ""]
    for inv in sorted(modes):
        cfg = modes[inv]["cfg"]
        eq_list = [q.strip() for q in (cfg.pde_expressions or "").split("|") if q.strip()]
        if inv and False in modes and cfg.pde_expressions == modes[False]["cfg"].pde_expressions:
            continue
        label = "As typed in PINNStudio's **Equations** page" + (" (Inverse mode)" if inv and False in modes else "")
        L += [label + " (each expression is a residual, set to zero):", "", "```text"]
        L += eq_list + ["```", ""]
        consts = json.loads(cfg.constants or "[]") if (cfg.constants or "").strip() else []
        if consts:
            L += ["Constants: " + ", ".join(f"`{c['name']} = {c['value']}`" for c in consts), ""]
    if inv_eqs:
        L += ["Inverse-mode form (from the README):", ""]
        for q in inv_eqs:
            L += [q, ""]

    # domain
    L += ["## Domain", ""]
    dl = _domain_lines(cfg0, e)
    L += [f"- {d}" for d in dl] + [""]

    # initial and boundary conditions
    ic = _ic_rows(first["default"])
    if ic:
        L += ["## Initial conditions", "", "| Output | Initial value | How it is imposed (Settings: Default) |", "|---|---|---|"]
        L += [f"| `{o}` | {_cell(_code(x))} | {h} |" for o, x, h in ic] + [""]
        if cfg0.forward_ic_from_file and cfg0.forward_ic_file:
            L += [f"The initial condition is read from `{_rel_ref(cfg0.forward_ic_file)}`.", ""]
    L += ["## Boundary conditions", ""]
    bcs = _bc_rows(first["default"], rec["bc_labels"])
    if bcs:
        L += ["| Type | Where | Output | Value | Details |", "|---|---|---|---|---|"]
        L += [f"| {t} | {_cell(_code(w)) if w not in ('—', 'initial time') else w} | {_cell(o if o == '—' else '`' + o + '`')} | "
              f"{_cell(_code(v)) if v and not v.startswith('(') else (v or '—')} | {_cell(x) or '—'} |"
              for t, w, o, v, x in bcs] + [""]
    else:
        L += ["No boundary-condition rows: the boundary values are imposed exactly through the network's "
              "output transform (see **Methods** below), or the problem has no boundary terms.", ""]

    # inverse
    if True in modes:
        mi = modes[True]
        cfg = mi["cfg"]
        L += ["## Unknowns and measured data (Inverse)", ""]
        if ex is not None and ex.inverse_note and e.problem == cat.BOTH:
            pass
        vars_ = json.loads(cfg.inverse_variables_json or "[]") if (cfg.inverse_variables_json or "").strip() else []
        if vars_:
            L += ["| Trainable parameter | Initial value | True value |", "|---|---|---|"]
            L += [f"| `{v['name']}` | {_num(v.get('init'))} | {_num(v['true']) if v.get('true') not in (None, '') else '—'} |"
                  for v in vars_] + [""]
        else:
            L += ["This example learns an unknown **field** as an extra network output (it has no scalar "
                  "trainable parameters), from measurements given as a Point Set boundary condition; it runs as "
                  "a Forward problem, as DeepXDE's script does.", ""]
        _, obs = _ref_folders(cfg)
        if obs:
            outs = [o.strip() for o in cfg.output_names.split(",")]
            items = json.loads(cfg.inverse_obs_files_json)
            L += ["Measured data (**Inverse** page, measured-data files):", ""]
            for o in items:
                r = _rel_ref(o["path"])
                idx = int(o.get("output_idx", 0))
                L.append(f"- [`{r}`]({GH_BLOB}{r}) → output `{outs[idx] if idx < len(outs) else idx}`, "
                         f"weight {_num(o.get('weight', 1))}")
            L.append("")
        if inv_par:
            L += [inv_par.replace("*Inverse mode* — ", "From the README: "), ""]

    # settings
    L += ["## Settings", ""]
    for inv in sorted(modes):
        m = modes[inv]
        head = "Inverse mode" if inv and e.problem != cat.INVERSE else ("Forward mode" if not inv else "")
        if len(modes) > 1:
            L += [f"### {head}", ""]
        if m["paper_default"]:
            what = f"the paper's method ({m['paper_default']})"
        elif e.source == cat.DEEPXDE and (not inv or e.problem == cat.INVERSE):
            what = "DeepXDE's own settings"
        elif e.source == cat.DEEPXDE:
            what = "DeepXDE's settings with PINNStudio's Inverse-mode changes (see *Notes*)"
        else:
            what = "the template's own settings"
        L += [f"**Settings: Default** — {what}", ""]
        L += _settings_block(m["default"]) + [""]
        if m["has_tuned"] and m["tuned"] is not None:
            msg = m["tuned_msg"].replace("Tuned settings: ", "")
            base = ("the template's own settings (not the paper's method that Default loads)" if m["paper_default"]
                    else "the example's own settings")
            L += ['??? note "Settings: Tuned"', "",
                  f"    The Training Log reports: *Tuned settings: {msg.rstrip('.')}.*"
                  + ("" if msg.startswith("the template's own") else f" The change is applied to {base}."), ""]
            L += ["    " + x if x else "" for x in _settings_block(m["tuned"])] + [""]
        elif m["tuned_unoffered"]:
            L += ["No **Tuned** choice in the GUI: the **Tuned** item is greyed out for this example. "
                  "`pinnstudio/resources/tuned_settings.json` does hold a tuned setting for it (under its inverse "
                  "entry), but the example loads as a Forward problem, so the Examples list does not offer it. "
                  "The tuning-study configuration file is linked under *Configuration and benchmark records*.", ""]
        else:
            L += ["No separate **Tuned** preset: the **Tuned** item is greyed out for this example "
                  "(its Default settings are PINNStudio's best).", ""]

    # load
    L += ["## Loading it in PINNStudio", ""]
    for inv in sorted(modes):
        L.append(f"- {'Inverse' if inv else 'Forward'}: {_gui_path(e, inv, modes[inv]['has_tuned'])}, then **Solve**.")
    L += ["", "Loading an example fills in every setting; anything you change before **Solve** is listed in the "
          "Training Log. Your save folder, GPU, random-seed and plot-style settings are kept.", ""]

    # reference
    L += ["## Reference solution", ""]
    ref_lines = []
    for inv in sorted(modes):
        cfg = modes[inv]["cfg"]
        tag = ("Inverse mode: " if inv else "Forward mode: ") if len(modes) > 1 else ""
        if cfg.ea_ref_mode == "formula" and (cfg.ea_ref_formula or "").strip():
            outs = [o.strip() for o in cfg.output_names.split(",")]
            fs = cfg.ea_ref_formula.split("|")
            ref_lines.append(f"- {tag}exact solution, **Error Analysis** › reference from formula: " +
                             ", ".join(f"`{outs[i] if i < len(outs) else i} = {f.strip()}`" for i, f in enumerate(fs)))
        else:
            folders, _ = _ref_folders(cfg)
            if folders:
                ref_lines.append(f"- {tag}reference data files: " + ", ".join(
                    f"[`{f}/`]({GH_TREE}{f})" for f in folders))
            else:
                ref_lines.append(f"- {tag}none bundled; you can add your own files in the **Error Analysis** dialog.")
    bodies = [r.split(": ", 1)[1] if r.startswith(("- Forward mode: ", "- Inverse mode: ")) else r[2:]
              for r in ref_lines]
    if len(set(bodies)) == 1:
        b0 = bodies[0]
        ref_lines = ["- " + b0[0].upper() + b0[1:]]
    L += ref_lines + [""]

    # source and citation
    L += ["## Source and citation", ""]
    if e.source == cat.DEEPXDE:
        L.append(f"- DeepXDE script: [{cat.code_url(ex.source).rsplit('/', 1)[-1]}]({cat.code_url(ex.source)})")
        if "readthedocs" in ex.source:
            L.append(f"- DeepXDE documentation: [{ex.source.rsplit('/', 1)[-1]}]({ex.source})")
        if paper_line:
            L.append(f"- Source paper (from the README): {paper_line}")
    else:
        if e.reference:
            link = cat.PAPERS.get(e.reference, "")
            L.append(f"- Paper: [{e.reference}]({link})" if link else f"- Paper: {e.reference}")
            if after_line:
                L.append(f"- README: {after_line}")
        else:
            L.append("- PINNStudio's own problem (no source paper).")
    L.append("")

    # notes
    notes = []
    if ex is not None and ex.note:
        notes.append(ex.note)
    if ex is not None and ex.inverse_note and True in modes and e.problem == cat.BOTH:
        notes.append(ex.inverse_note)
    if e.key in cat.FORWARD_ONLY_TEMPLATES:
        notes.append("Forward only: the Inverse mode of this template is withdrawn for now because the parameters "
                     "were not found (pinnstudio/core/example_catalog.py, FORWARD_ONLY_TEMPLATES).")
    if e.key == "1D Allen-Cahn (whole interval)":
        notes.append("The time-adaptive version of this template is not listed in the Examples list; it still "
                     "loads by its name from saved problems (HIDDEN_TEMPLATES in example_catalog.py).")
    if e.key == "Poisson over an L-shaped domain":
        notes.append("PINNStudio's own L-shape template is not listed separately: this DeepXDE example is the same "
                     "problem (HIDDEN_TEMPLATES in example_catalog.py).")
    if notes:
        L += ["## Notes", ""] + [f"- {_md_math(n)}" for n in notes] + [""]

    # configuration and benchmark records
    L += ["## Configuration and benchmark records", ""]
    links = []
    for f in rec["tuned_files"]:
        links.append(f"- [`benchmarks/tuning/{f}`]({GH_BLOB}benchmarks/tuning/{urllib.parse.quote(f)}) — "
                     "a configuration tried in the tuning study, kept as a record (it can be opened with **Problem › Open Saved Problem...**; it is not a preset of the Examples list)")
    if rec["bench"]:
        links.append(f"- [Benchmark record]({GH_BLOB}benchmarks/problems/{rec['bench']['file']}) "
                     f"(`benchmarks/problems/{rec['bench']['file']}`)")
        links.append(f"- [Entry in the benchmark collection]({GH_BLOB}benchmarks/README.md)"
                     " (`benchmarks/README.md`)")
    if not links:
        links.append("- No configuration file or benchmark record for this example.")
    L += links + ["", "Verified error values are reported in the [Reproducibility](" + root_rel +
                  "reproducibility/benchmarks.md) section, not on this page.", ""]
    L += ["[← Back to the example gallery](../index.md)", ""]
    return "\n".join(L)


# ----------------------------------------------------------------- index
def _groups():
    return [
        ("ODEs", "odes", "Ordinary differential equations in t only.", lambda r: r["entry"].dim == "ODE"),
        ("1D PDEs", "1d-pdes", "One space dimension, steady or in time.", lambda r: r["entry"].dim == "1D"),
        ("2D PDEs", "2d-pdes", "Two space dimensions.", lambda r: r["entry"].dim == "2D"),
        ("3D PDEs", "3d-pdes", "Three space dimensions.", lambda r: r["entry"].dim == "3D"),
        ("Coupled systems", "coupled-systems", "More than one governing equation.", lambda r: r["n_eq"] > 1),
        ("Forward problems", "forward", "Listed under **Forward** in the Examples list.", lambda r: False in r["modes"]),
        ("Inverse problems", "inverse", "Listed under **Inverse**: DeepXDE's inverse examples and every example "
         "with an Inverse mode.", lambda r: True in r["modes"]),
        ("Phase-field", "phase-field", "Allen–Cahn and Cahn–Hilliard equations.", lambda r: r["family"] in _PHASE),
        ("Fluid mechanics", "fluid-mechanics", "Burgers, Navier–Stokes (Kovasznay, Beltrami) and Brinkman–Forchheimer.",
         lambda r: r["family"] in _FLUID),
        ("Solid mechanics and elasticity", "solid-mechanics", "Euler beam and linear elastostatic plate.",
         lambda r: r["family"] in _SOLID),
        ("Advanced training strategies", "advanced-training", "Default or Tuned settings that use time windows, "
         "RAR, causal training, RBA or SA-PINN point weights, or multi-scale Fourier features.",
         lambda r: bool(r["strategies"])),
    ]


def _index(records):
    from pinnstudio.core import example_catalog as cat
    n = len(records)
    n_t = sum(r["entry"].source == cat.PINNSTUDIO for r in records)
    n_d = n - n_t
    groups = [(t, a, d, [r for r in records if f(r)]) for t, a, d, f in _groups()]
    L = ["# Examples", "",
         f"PINNStudio ships **{n} examples**: {n_t} PINNStudio templates and {n_d} examples reproduced from DeepXDE's "
         "documentation. Each page below is generated from the example's own definition in the software "
         "(`pinnstudio/core/example_catalog.py`, `deepxde_examples.py`, `pinnstudio_examples.py` and the templates "
         "as the main window loads them), so the equations, conditions and settings shown are the ones PINNStudio "
         "loads.", "",
         "Every example is loaded from the toolbar's **Examples:** list: choose **Forward** or **Inverse**, pick the "
         "example under its **ODE**, **1D**, **2D** or **3D** heading, and choose **Default** or **Tuned** "
         "settings. *Default* is DeepXDE's own settings for a DeepXDE example, the paper's method for a template "
         "taken from a paper, and the template's own settings otherwise; *Tuned* is PINNStudio's best setting from "
         "the tuning study, where one exists.", "",
         "!!! note \"Development version\"",
         "    This gallery describes the development version that will become v1.6.0. The Examples list of the "
         "released v1.5.0 may differ.", "",
         "!!! warning \"Scope\"",
         "    An example is a starting point. PINNStudio sets up, trains and evaluates the PINN; it does not "
         "guarantee convergence or accuracy, and results vary from run to run. How the examples' errors are measured and recorded is described in "
         "[Reproducibility](../reproducibility/benchmarks.md).", "",
         "## Browse by group", "", '<div class="grid cards" markdown>', ""]
    for t, a, d, rs in groups:
        L += [f"-   **[{t}](#{a})** ({len(rs)})", "", "    ---", "", f"    {d}", ""]
    L += ["</div>", ""]
    for t, a, d, rs in groups:
        L += [f"## {t} {{ #{a} }}", "", f"{d} {len(rs)} example{'s' if len(rs) != 1 else ''}.", ""]
        extra = a == "advanced-training"
        L += ["| Example | Dimension | Regime | Modes | Tuned |" + (" Strategies |" if extra else ""),
              "|---|---|---|---|---|" + ("---|" if extra else "")]
        for r in rs:
            e = r["entry"]
            modes = " / ".join("Inverse" if m else "Forward" for m in sorted(r["modes"]))
            tuned = "yes" if any(m["has_tuned"] for m in r["modes"].values()) else "—"
            row = f"| [{_cell(e.name)}]({r['path']}) | {e.dim} | {e.regime} | {modes} | {tuned} |"
            if extra:
                row += " " + ", ".join(sorted(r["strategies"])) + " |"
            L.append(row)
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------- main
def _tuned_file_map(records_names):
    files = sorted(os.listdir(REPO / "benchmarks" / "tuning"))
    out = {}
    for name in records_names:
        stem = name.replace("/", " - ")
        out[name] = [f for f in files if f.endswith(".pinn.json")
                     and (f.startswith(stem + " (") or f.startswith(stem + ", inverse mode"))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="/tmp/pinnstudio_gallery_cache.json")
    ap.add_argument("--refresh", action="store_true", help="rebuild the template configurations")
    ap.add_argument("--check", action="store_true", help="only report README/code differences")
    args = ap.parse_args()

    from pinnstudio.core import example_catalog as cat
    from pinnstudio.ui.main_window import MainWindow
    entries = [e for e in cat.ENTRIES if not e.hidden]     # what ExampleCombo shows
    tcache = _template_configs(entries, Path(args.cache), args.refresh)
    readme = _readme_sections()
    bench_index = {b["name"]: b for b in json.loads((REPO / "benchmarks/problems/index.json").read_text())}
    tuned_files = _tuned_file_map([e.name for e in entries])
    bc_labels = {k: lbl for lbl, k in MainWindow.CUSTOM_BC_TYPES}
    records = []
    for e in entries:
        r = _record(e, tcache, readme, bench_index, tuned_files, bc_labels)
        r["bc_labels"] = bc_labels
        records.append(r)
        if r["bench"]:
            ANCHOR_PAGES[r["bench"]["anchor"]] = r["path"]

    # README / code differences
    issues = []
    col = _readme_problem_column()
    for r in records:
        e = r["entry"]
        if e.name not in col:
            issues.append(f"README table has no row for '{e.name}'")
        elif col[e.name] != e.problem:
            issues.append(f"README table says '{col[e.name]}' for '{e.name}', the catalog '{e.problem}'")
        if not r["bench"]:
            issues.append(f"no benchmark record for '{e.name}'")
        elif not r["readme"][0]:
            issues.append(f"README section of '{e.name}' has no display equation")
        for inv, m in r["modes"].items():
            if not inv and m["gui_inverse"]:
                issues.append(f"'{e.name}' Forward mode loads as an Inverse problem")
        from pinnstudio.core import example_settings as es
        tt = es.tuned_table().get(e.key, {})
        for side in ("forward", "inverse"):
            if tt.get(side) and not any(m["has_tuned"] for m in r["modes"].values()
                                        if m["gui_inverse"] == (side == "inverse")):
                issues.append(f"'{e.name}': tuned_settings.json has a '{side}' entry the GUI never offers "
                              f"(the example loads with problem_type {[m['cfg'].problem_type for m in r['modes'].values()]})")
    for i in issues:
        print("NOTE:", i)
    if args.check:
        return

    # images
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
    IMG_DIR.mkdir(parents=True)
    by_key = {}
    for img in IMAGES:
        shutil.copy2(REPO / img[0], IMG_DIR / img[2])
        by_key.setdefault(img[1], []).append(img)

    for d in DIM_DIRS.values():
        if (OUT / d).exists():
            shutil.rmtree(OUT / d)
    for r in records:
        p = OUT / r["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_page(r, by_key.get(r["entry"].key, []), "../../"), encoding="utf-8")
    (OUT / "index.md").write_text(_index(records), encoding="utf-8")
    counts = {}
    for r in records:
        counts[r["entry"].dim] = counts.get(r["entry"].dim, 0) + 1
    print(f"wrote {len(records)} example pages {counts} and examples/index.md; {len(IMAGES)} images")
    for t, a, d, f in _groups():
        print(f"  {t}: {sum(1 for r in records if f(r))}")


if __name__ == "__main__":
    main()
