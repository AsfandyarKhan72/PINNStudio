import re
from pinnstudio.core.line_slice import auto_line_slice
import os
os.environ["DDE_BACKEND"] = "pytorch"
# ── CUDA performance environment variables ────────────────────
# (No TORCH_CUDA_ARCH_LIST here -- that used to be pinned to a specific
# GPU model (RTX 4090 / sm_89) on the original dev machine, which only
# matters if something JIT-compiles a CUDA extension from source, but
# would silently restrict that compile to sm_89 on anyone else's card.
# Leaving it unset lets PyTorch/DeepXDE target whatever GPU is actually
# present, same as it already does for everyone via the ordinary prebuilt
# CUDA wheels install.sh/select_torch_index.py picks for their driver.)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"        # async CUDA launches
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import ast as _bc_ast
import json as _bc_json_std


def _bc_op_expr_uses_X(expr):
    """Does a user-written Operator/PointSetOperator BC "Value" expression
    reference the raw ``X`` argument (the BC function's own 3rd positional
    arg, exposed by ``_bc_operator_fn`` in the generated script)?

    DeepXDE's own OperatorBC/PointSetOperatorBC docstrings warn that if
    ``func`` uses ``X``, ``num_test`` must NOT be set on the owning
    ``dde.data.PDE``/``TimePDE`` (DeepXDE raises an error otherwise, since
    the X-usage path requires every collocation point to be evaluated, not
    a random test subsample). Detected with a real ``ast`` parse (not a
    substring/regex search) so identifiers like "Xavier" or a string
    literal containing the letter X are never mistaken for the argument.

    Malformed expressions are reported as "doesn't use X" here -- they will
    raise their own clear error later at the real ``compile()``/eval call
    inside the generated script, so there's nothing useful to flag yet.
    """
    if not expr or "X" not in expr:
        return False
    try:
        tree = _bc_ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in _bc_ast.walk(tree):
        if isinstance(node, _bc_ast.Name) and node.id == "X":
            return True
    return False


def _bc_entries_use_operator_X(bc_entries):
    """True if any ``operator``/``pointset_operator`` row in a parsed
    ``custom_bc_json`` entry list references the raw ``X`` argument in its
    "Value" expression (see ``_bc_op_expr_uses_X``)."""
    for entry in bc_entries or []:
        if entry.get("type") in ("operator", "pointset_operator"):
            if _bc_op_expr_uses_X(entry.get("value", "")):
                return True
    return False


def _bc_entries_use_batching(bc_entries):
    """True if any ``pointset``/``pointset_operator`` row in a parsed
    ``custom_bc_json`` entry list has a nonzero ``batch_size`` -- i.e.
    needs ``dde.callbacks.PDEPointResampler(bc_points=True)`` in the
    training callbacks for that batching to actually do anything (see
    ``_build_train_cbs_code``'s own comment)."""
    for entry in bc_entries or []:
        if entry.get("type") in ("pointset", "pointset_operator"):
            try:
                if int(entry.get("batch_size", 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _bc_entries_use_batching_json(custom_bc_json):
    """``_bc_entries_use_batching``, parsing ``custom_bc_json`` itself --
    convenience for call sites (like ``_build_train_cbs_code``) that only
    have the raw config string, not an already-parsed entry list."""
    try:
        entries = _bc_json_std.loads(custom_bc_json) if custom_bc_json else []
    except (ValueError, TypeError):
        entries = []
    return _bc_entries_use_batching(entries)


def _bc_entries_use_interface2d(bc_entries):
    """True if any row in a parsed ``custom_bc_json`` entry list is an
    Interface 2D BC -- used both by ``_bc_check_interface2d_preconditions``
    (the geometry check, which has no sensible auto-fix) and by both
    generators' own point-distribution override (see the comment at each
    ``_safe_point_distribution`` computation): unlike the geometry
    requirement, the "must be uniform" requirement always has exactly one
    valid answer whenever this BC type is used at all, so it's corrected
    automatically there instead of being enforced as a second hard error
    here."""
    return any(e.get("type") == "interface2d" for e in (bc_entries or []))


def _bc_check_interface2d_preconditions(bc_entries, geometry_type, is_2d=None, num_outputs=None):
    """DeepXDE's own real Interface2DBC requirements (see the class's own
    docstring and error() method in deepxde/icbc/boundary_conditions.py):
    (1) the problem must be 2D -- Rectangle/Polygon, the only geometries
    it accepts, only exist in 2D DeepXDE at all; (2) the network output
    must have exactly 2 elements -- error()'s "normal" mode dots the
    whole output against a 2-column normal vector, and its "tangent" mode
    hardcodes outputs[:, 0:1]/outputs[:, 1:2] by position; (3) the
    geometry itself must be Rectangle or Polygon, so the two sides it's
    told about are real, equal-length edges.

    The GUI's own Type dropdown already keeps (1) and (2) from being
    reachable in the first place -- "Interface 2D BC" simply isn't
    offered unless Dimension=2D and Output Size=2 (see
    _interface2d_eligible/_refresh_bc_interface2d_availability in
    main_window.py) -- so is_2d/num_outputs are checked here only as
    defense-in-depth for a hand-edited or legacy saved config that
    reaches codegen with a stale/corrupted combination (e.g. geometry
    left at "Rectangle" after a dimension was changed by hand-editing
    JSON). (3), the geometry shape itself, has no GUI equivalent at all
    -- there's no sensible geometry to auto-pick the way the companion
    "train_distribution must be uniform" requirement is auto-corrected
    for instead (see _bc_entries_use_interface2d's docstring) -- so it
    always raises a clear error here, at codegen time, rather than a
    confusing crash deep inside DeepXDE or Interface2DBC.error()'s own
    "different number of points on each edge" RuntimeError, right when
    Solve is clicked / the script is exported, before any GPU/training
    time is spent on a run that would be silently wrong.
    """
    if not _bc_entries_use_interface2d(bc_entries):
        return
    if is_2d is False:
        raise ValueError(
            "Interface 2D BC only applies to 2D problems (current "
            "Dimension setting isn't 2D). Switch Dimension to 2D, or "
            "remove the Interface 2D BC row."
        )
    if num_outputs is not None and num_outputs != 2:
        raise ValueError(
            "Interface 2D BC requires Output Size = 2 "
            f"(current Output Size: {num_outputs}). Set Output Size to 2, "
            "or remove the Interface 2D BC row."
        )
    if geometry_type not in ("Rectangle", "Polygon"):
        raise ValueError(
            "Interface 2D BC requires a Rectangle or Polygon geometry "
            f"(current geometry: {geometry_type!r}). Change the geometry, "
            "or remove the Interface 2D BC row."
        )


def _bc_entries_use_periodic(bc_entries):
    """True if any row in a parsed ``custom_bc_json`` entry list is a
    Periodic BC -- used by ``_bc_check_periodic_preconditions``."""
    return any(e.get("type") == "periodic" for e in (bc_entries or []))


def _bc_initial_operator_rows(bc_entries):
    """Operator BC rows set to apply at the initial time (Round 64)."""
    return [i for i, e in enumerate(bc_entries or [])
            if isinstance(e, dict) and e.get("type") == "operator" and e.get("apply_at") == "initial"]


def _bc_check_initial_operator_preconditions(bc_entries, is_steady, time_adaptive):
    """An Operator BC "at the initial time" (e.g. du_t = 0 at t = t_min, the
    second initial condition of a wave-type equation) needs a time axis, and
    isn't supported with Time-Adaptive training: only the first time window
    contains t = t_min, and later windows have no matching points."""
    rows = _bc_initial_operator_rows(bc_entries)
    if not rows:
        return
    labels = ", ".join(f"BC {i + 1}" for i in rows)
    if is_steady:
        raise ValueError(f"{labels}: an Operator BC set to apply at the initial time needs a "
                         "time-dependent problem (this one is steady-state).")
    if time_adaptive:
        raise ValueError(f"{labels}: an Operator BC at the initial time isn't supported with "
                         "Time-Adaptive training (only the first time window contains t = t_min). "
                         "Use Adaptation = None or RAR for this problem.")


def _check_anchor_preconditions(config):
    """Extra training points (Round 65) aren't supported with Time-Adaptive
    training -- each step only covers part of the time range."""
    if (getattr(config, "anchors_mode", "none") or "none") != "none" and getattr(config, "time_adaptive", False):
        raise ValueError("Extra training points (anchors) aren't supported with Time-Adaptive training "
                         "(each step covers only part of the time range). Use Adaptation = None or RAR.")


def _bc_check_periodic_preconditions(bc_entries, geometry_type):
    """DeepXDE's own PeriodicBC calls ``geom.periodic_point(x,
    component_x)``, which is only implemented for ``Interval`` (1D),
    the ``Hypercube`` base class shared by ``Rectangle`` (2D) and
    ``Cuboid`` (3D), and the three CSG combination classes (which
    delegate to their own ``geom1``/``geom2``) -- see
    deepxde/geometry/geometry.py's base ``Geometry.periodic_point``,
    which raises ``NotImplementedError`` for every other shape (``Disk``,
    ``Ellipse``, ``Triangle``, ``Polygon``, ``Sphere``).

    Unlike Interface 2D BC's codegen-time check, nothing here is an
    app-crash risk: this would only ever surface later, inside the
    generated script's own training subprocess, which ``run_pinn()``'s
    existing ``on_output``/returncode handling already reports as a
    normal (if confusing) training failure rather than crashing the GUI.
    It's checked here anyway for the same reason as every other
    precondition in this file: a clear PINNStudio message naming the
    actual requirement, right when Solve is clicked / the script is
    exported, beats a bare ``NotImplementedError`` with no explanation
    surfacing minutes into a training run.

    ``geometry_type in ("Interval", "Rectangle", "Cuboid")`` is the same
    "box-shaped" test already used elsewhere in this app (see
    main_window.py's own ``box_shape = geom_type in (...)`` checks) --
    each of those three strings only exists for exactly one dimension, so
    this alone is enough to cover 1D/2D/3D without a separate
    is_2d/is_3d check. "Custom" (CSG-combination) geometries are treated
    as ineligible here too: validating an arbitrary CSG tree's
    periodic_point support would mean walking its geom1/geom2 recursively
    to confirm every leaf is itself box-shaped, which is out of scope for
    this round -- a Custom-geometry user who needs this can still express
    the same periodicity by hand with Operator BC.
    """
    if not _bc_entries_use_periodic(bc_entries):
        return
    if geometry_type not in ("Interval", "Rectangle", "Cuboid"):
        raise ValueError(
            "Periodic BC requires an Interval (1D), Rectangle (2D), or "
            f"Cuboid (3D) geometry (current geometry: {geometry_type!r}); "
            "DeepXDE doesn't implement periodic boundary matching for "
            "other geometry shapes. Change the geometry, or remove the "
            "Periodic BC row."
        )


def _simplify_expr(expr, is_2d=False, is_3d=False):
    """Convert user-friendly math syntax to numpy syntax."""
    import re
    e = expr.strip()
    # Replace math functions with numpy equivalents
    for fn in ["sin","cos","tan","sinh","cosh","tanh","arcsin","arccos","arctan",
               "exp","log","log10","sqrt","abs","ceil","floor"]:
        e = re.sub(rf'\b{fn}\(', f'np.{fn}(', e)
    # Replace the pi constant. (There used to be a similar line here for a
    # bare Euler's-number "e" constant, but it was written as \bexp\b
    # instead of \be\b -- since \b is a boundary between a word character
    # and a non-word character, \bexp\b matches "exp" immediately before
    # "(" too (a word/non-word boundary either way), so it silently
    # clobbered every use of the exp() function into "np.e(", right after
    # the loop above had already correctly turned it into "np.exp(" --
    # e.g. "exp(-20*x)" -> "np.exp(-20*x)" -> "np.np.e(-20*x)". No shipped
    # template used exp() before this patch's Diffusion-Reaction and 2D
    # Burgers examples, so it went unnoticed. Removed rather than fixed to
    # \be\b, matching _simplify_pde_expr in main_window.py (used for the
    # same user-facing "type math like you'd write it" convention
    # elsewhere), which never had a bare-e substitution to begin with.)
    e = re.sub(r'\bpi\b', 'np.pi', e)
    # Replace x, y, z variables — must be done carefully to avoid replacing
    # inside words
    if is_3d:
        e = re.sub(r'\bz\b', '__Z__', e)
        e = re.sub(r'\by\b', '__Y__', e)
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
        e = e.replace('__Y__', 'x[:, 1]')
        e = e.replace('__Z__', 'x[:, 2]')
    elif is_2d:
        e = re.sub(r'\by\b', '__Y__', e)
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
        e = e.replace('__Y__', 'x[:, 1]')
    else:
        e = re.sub(r'\bx\b', 'x[:, 0]', e)
    return e

def _parse_vertex_list(text):
    """'x1,y1;x2,y2;...' -> [[x1,y1], [x2,y2], ...]. Skips malformed pairs --
    mirrors MainWindow._parse_vertices() exactly, so a Triangle/Polygon
    trains on the same vertices the domain preview already draws."""
    verts = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) != 2:
            continue
        try:
            verts.append([float(parts[0].strip()), float(parts[1].strip())])
        except ValueError:
            continue
    return verts

_CUSTOM_GEOM_OP_CTORS = {"union": "CSGUnion", "subtract": "CSGDifference", "intersect": "CSGIntersection"}
_CUSTOM_GEOM_BUGGY_LEAF_TYPES = ("Disk", "Ellipse", "Triangle", "Polygon", "Sphere")


def _custom_geom_leaf_literal(shape_type, params):
    """One Custom-geometry leaf shape -> its dde.geometry constructor call,
    as a literal string, from the params dict the GUI's
    _custom_geom_row_to_dict() produces (see config.py's
    geom_custom_shapes_json docs). Mirrors the per-type literal building
    the other geometry types already do above (_triangle_vertices_literal
    etc.) and in _clean_geom_line() below -- falls back to a sane default
    for a missing/malformed param rather than raising, since codegen must
    never crash on a stale or hand-edited config."""
    params = params or {}
    if shape_type == "Disk":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5); r = params.get("r", 0.5)
        return f"dde.geometry.Disk([{cx}, {cy}], {r})"
    if shape_type == "Ellipse":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5)
        a = params.get("a", 0.5); b = params.get("b", 0.3); angle = params.get("angle", 0.0)
        return f"dde.geometry.Ellipse([{cx}, {cy}], {a}, {b}, {angle})"
    if shape_type == "Triangle":
        verts = _parse_vertex_list(params.get("vertices_text", ""))
        if len(verts) != 3:
            verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        return f"dde.geometry.Triangle({verts[0]}, {verts[1]}, {verts[2]})"
    if shape_type == "Polygon":
        verts = _parse_vertex_list(params.get("vertices_text", ""))
        if len(verts) < 3:
            verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        return f"dde.geometry.Polygon({verts})"
    if shape_type == "Cuboid":
        x_min = params.get("x_min", 0.0); x_max = params.get("x_max", 1.0)
        y_min = params.get("y_min", 0.0); y_max = params.get("y_max", 1.0)
        z_min = params.get("z_min", 0.0); z_max = params.get("z_max", 1.0)
        if x_min >= x_max:
            x_min, x_max = 0.0, 1.0
        if y_min >= y_max:
            y_min, y_max = 0.0, 1.0
        if z_min >= z_max:
            z_min, z_max = 0.0, 1.0
        return f"dde.geometry.Cuboid([{x_min}, {y_min}, {z_min}], [{x_max}, {y_max}, {z_max}])"
    if shape_type == "Sphere":
        cx = params.get("cx", 0.5); cy = params.get("cy", 0.5); cz = params.get("cz", 0.5)
        r = params.get("r", 0.5)
        return f"dde.geometry.Sphere([{cx}, {cy}, {cz}], {r})"
    # Rectangle, and any unrecognized/missing type -- fall back to a unit
    # square Rectangle rather than crashing codegen over a stale entry.
    x_min = params.get("x_min", 0.0); x_max = params.get("x_max", 1.0)
    y_min = params.get("y_min", 0.0); y_max = params.get("y_max", 1.0)
    if x_min >= x_max:
        x_min, x_max = 0.0, 1.0
    if y_min >= y_max:
        y_min, y_max = 0.0, 1.0
    return f"dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])"


def _build_custom_geom_code(config):
    """Custom geometry: config.geom_custom_shapes_json -> (a dde.geometry
    constructor call chaining CSGUnion/CSGDifference/CSGIntersection in
    build order, wrap_needed). Shared by generate_script()'s _build_geom()
    template and generate_clean_script()'s _clean_geom_line() below --
    the same two call sites every other geometry type already has its own
    literal-building logic at (see _triangle_vertices_literal above vs
    tri_verts in _clean_geom_line). Falls back to a unit-square Rectangle
    (2D) or unit cube Cuboid (3D, per config.problem_dim) if the shape
    list is empty or unparseable, matching every other shape's
    fallback-to-sane-default philosophy in this file.

    wrap_needed mirrors _DTypeSafeGeom's existing single-shape rule
    (Disk/Ellipse/Triangle/Polygon/Sphere get wrapped, Rectangle/Cuboid
    don't): a CSG combination of 2+ shapes never needs it -- verified
    empirically against the installed DeepXDE version, in both 2D and 3D,
    CSGUnion/CSGDifference/CSGIntersection always normalize their sampled
    points to the network's own configured float dtype regardless of
    which leaf types feed into them, even when every leaf is individually
    "buggy" -- so wrapping is only applied for the degenerate single-shape
    Custom case, exactly like a plain (non-Custom) Disk/Ellipse/Triangle/
    Polygon/Sphere already gets."""
    import json as _json_cg
    try:
        entries = _json_cg.loads(config.geom_custom_shapes_json or "[]")
    except (ValueError, TypeError):
        entries = []
    if not isinstance(entries, list):
        entries = []
    entries = [e for e in entries if isinstance(e, dict)]
    if not entries:
        if getattr(config, "problem_dim", "2D") == "3D":
            entries = [{"type": "Cuboid", "params": {
                "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}}]
        else:
            entries = [{"type": "Rectangle", "params": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}}]

    code = None
    for entry in entries:
        leaf = _custom_geom_leaf_literal(entry.get("type"), entry.get("params"))
        if code is None:
            code = leaf
        else:
            ctor = _CUSTOM_GEOM_OP_CTORS.get(entry.get("op") or "union", "CSGUnion")
            code = f"dde.geometry.{ctor}({code}, {leaf})"

    wrap_needed = len(entries) == 1 and entries[0].get("type") in _CUSTOM_GEOM_BUGGY_LEAF_TYPES
    return code, wrap_needed


# PDE expressions are evaluated on the network's own tensors (with
# autograd), so their math functions must be torch's, not numpy's: numpy's
# would try to convert a grad-tracking tensor and crash. The names users
# type are numpy-style (arcsin, ...); the right side is torch's spelling.
_PDE_TORCH_FUNCS = (
    ("sin", "sin"), ("cos", "cos"), ("tan", "tan"),
    ("sinh", "sinh"), ("cosh", "cosh"), ("tanh", "tanh"),
    ("arcsin", "asin"), ("arccos", "acos"), ("arctan", "atan"),
    ("exp", "exp"), ("log", "log"), ("log10", "log10"), ("sqrt", "sqrt"),
    ("abs", "abs"), ("ceil", "ceil"), ("floor", "floor"),
    ("relu", "relu"), ("sigmoid", "sigmoid"), ("lgamma", "lgamma"),
)


def _fold_numeric_calls(e):
    """sqrt(3), exp(-1), gamma((1 - 1.8)/2), ... -> their values (Round 75):
    torch's functions take tensors, not numbers, so torch.sqrt(3) in a
    generated expression fails. Calls whose argument is only numbers, pi,
    arithmetic and parentheses are evaluated here (innermost first);
    anything else is left as written."""
    import math
    import re
    fns = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "sinh": math.sinh, "cosh": math.cosh,
           "tanh": math.tanh, "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan, "exp": math.exp,
           "log": math.log, "log10": math.log10, "sqrt": math.sqrt, "abs": abs, "ceil": math.ceil,
           "floor": math.floor, "gamma": math.gamma, "lgamma": math.lgamma}
    head = re.compile(r"(?<![\w.])(" + "|".join(sorted(fns, key=len, reverse=True)) + r")\(")
    numeric = re.compile(r"^(?:[0-9eE.+\-*/ ()]|(?<![\w])pi(?![\w]))*$")
    for _ in range(50):
        changed = False
        pos = 0
        while True:
            m = head.search(e, pos)
            if not m:
                break
            depth, j = 1, m.end()
            while j < len(e) and depth:
                depth += {"(": 1, ")": -1}.get(e[j], 0)
                j += 1
            if depth:
                break
            arg = e[m.end():j - 1]
            if arg.strip() and numeric.match(arg) and "e" not in re.sub(r"\d[eE][+-]?\d", "", arg).replace("pi", ""):
                try:
                    v = float(fns[m.group(1)](eval(arg, {"__builtins__": {}}, {"pi": math.pi})))
                except Exception:
                    pos = m.end()
                    continue
                e = e[:m.start()] + (repr(v) if v >= 0 else f"({v!r})") + e[j:]
                changed = True
                break
            pos = m.end()
        if not changed:
            break
    return e


def _simplify_pde_expr(expr):
    """Convert user-friendly math in a PDE expression: bare function calls
    become torch calls (sin(...) -> torch.sin(...)) and pi becomes np.pi.
    A call the user already qualified (np.sin, torch.sin) is left alone.
    (Before Round 61 these became np.sin(...), which crashed on the
    network's tensors -- no built-in template used one, so it never showed.)"""
    import re
    e = _fold_numeric_calls(expr.strip())
    for fn, tfn in _PDE_TORCH_FUNCS:
        e = re.sub(rf'(?<![a-zA-Z_.]){fn}\(', f'torch.{tfn}(', e)
    e = re.sub(r'(?<![a-zA-Z_.])pi(?![a-zA-Z_])', 'np.pi', e)
    return e


def _bc_domain_values(config):
    """{name: value} of the domain limits a Boundary Conditions expression
    can use by name (Round 68): x_min/x_max always, y_* in 2D/3D, z_* in
    3D, t_min/t_max when time-dependent."""
    dim = getattr(config, "problem_dim", "1D")
    names = ["x_min", "x_max"]
    if dim in ("2D", "3D"):
        names += ["y_min", "y_max"]
    if dim == "3D":
        names += ["z_min", "z_max"]
    if not getattr(config, "steady_state", False):
        names += ["t_min", "t_max"]
    return {n: float(getattr(config, n)) for n in names}


# Named sides of the domain box for "Where:" -> (coordinate column, limit).
BC_SIDE_NAMES = (("left", "x", "x_min"), ("right", "x", "x_max"),
                 ("bottom", "y", "y_min"), ("top", "y", "y_max"),
                 ("front", "z", "z_min"), ("back", "z", "z_max"))


def _bc_sides(config):
    """[(side name, column, limit value)] for this problem's dimension."""
    dom = _bc_domain_values(config)
    col = {"x": 0, "y": 1, "z": 2}
    return [(n, col[c], dom[lim]) for (n, c, lim) in BC_SIDE_NAMES if lim in dom]


def _bc_domain_literal(config):
    return repr(_bc_domain_values(config))


def _bc_sides_literal(config):
    return repr(_bc_sides(config))


def _expr_names(expr):
    """Every bare name an expression uses (ast; empty set if it doesn't parse)."""
    import ast as _ast_names
    try:
        tree = _ast_names.parse((expr or "").strip() or "0", mode="eval")
    except SyntaxError:
        return set()
    return {n.id for n in _ast_names.walk(tree) if isinstance(n, _ast_names.Name)}


def _clean_bc_name_lines(indent, expr, domain, is_location):
    """For the exported script: define, inside a BC helper function, only
    the domain names an expression uses -- e.g. `x_min = -1.0` and
    `left = np.isclose(x, -1.0)  # x = x_min` -- so the expression reads
    exactly as the user typed it (Round 68)."""
    used = _expr_names(expr)
    lines = [f"{indent}{n} = {v!r}" for n, v in domain.items() if n in used]
    if is_location:
        for n, c, lim in BC_SIDE_NAMES:
            if n in used and lim in domain:
                lines.append(f"{indent}{n} = bool(np.isclose({c}, {domain[lim]!r}))  # {c} = {lim}")
    return lines


def _np_math_expr(expr):
    """Bare math names in a boundary-condition expression -> numpy calls
    (sin(...) -> np.sin(...), pi -> np.pi), for the exported script, which
    pastes these expressions straight into small functions instead of
    eval'ing them with the live script's _BC_MATH_NS namespace. Same names
    as that namespace (_MATH_NS_ROWS); np.-qualified calls are left alone."""
    import re
    e = (expr or "").strip()
    for row in _MATH_NS_ROWS:
        for fn in row:
            e = re.sub(rf'(?<![a-zA-Z_.]){fn}\(', f'np.{fn}(', e)
    return re.sub(r'(?<![a-zA-Z_.])pi(?![a-zA-Z_0-9])', 'np.pi', e)


def _pde_coord_names(exprs):
    """The coordinate names (x, y, z, t) a set of PDE expressions actually
    uses -- found with ast, so du_x/du_t and the like don't count. Both
    generators bind only these, as one column each, so an equation without
    a source term generates exactly the code it always did."""
    import ast as _ast_pde
    used = set()
    for e in exprs:
        try:
            tree = _ast_pde.parse((e or "").strip() or "0", mode="eval")
        except SyntaxError:
            continue
        used |= {n.id for n in _ast_pde.walk(tree) if isinstance(n, _ast_pde.Name)}
    return used & {"x", "y", "z", "t"}


def _pde_coord_columns(used, is_2d, is_3d, is_steady):
    """[(name, column index)] for the coordinates in `used` that exist for
    this problem's dimension (y only in 2D/3D, z only in 3D, t only when
    time-dependent; t is always the last input column)."""
    n_space = 3 if is_3d else (2 if is_2d else 1)
    cols = {"x": 0}
    if is_2d or is_3d:
        cols["y"] = 1
    if is_3d:
        cols["z"] = 2
    if not is_steady:
        cols["t"] = n_space
    return [(n, cols[n]) for n in ("x", "y", "z", "t") if n in used and n in cols]


def _sanitize_python_identifier(name, fallback):
    """Coerce a free-typed string into a legal Python identifier. Inverse
    trainable-variable names come straight from a QLineEdit with no
    character restriction, but get spliced directly into generated code
    as a bare variable name (`{name} = dde.Variable(...)`), not inside a
    string literal -- so the repr()-escaping used for other free-text
    fields elsewhere in this file doesn't apply here; a name like
    "diffusion coefficient" would otherwise produce a SyntaxError in the
    generated script. An already-valid identifier (which covers every
    built-in template and any already-well-formed custom name) is
    returned completely unchanged -- this only rewrites the ones that
    would otherwise break."""
    import re as _re_id
    import keyword as _keyword_id
    _name = (name or "").strip()
    if _name.isidentifier() and not _keyword_id.iskeyword(_name):
        return _name
    _clean = _re_id.sub(r'[^A-Za-z0-9_]', '_', _name).strip('_')
    if not _clean:
        _clean = fallback
    elif _clean[0].isdigit():
        _clean = "_" + _clean
    if _keyword_id.iskeyword(_clean):
        _clean = _clean + "_var"
    return _clean


def _parse_inverse_variables(config):
    """Parse config.inverse_variables_json into an ordered list of
    (name, init, true) triples, one per trainable variable (variable 1
    first). `true` is the known ground-truth value for a built-in template
    whose PDE constant was substituted for this variable (see
    main_window.py's INVERSE_AUTO_VARS) -- None when not known (a manually
    added variable, or a config saved before this field existed), in which
    case downstream code (the parameter-convergence plot) simply skips
    drawing a true-value reference line for that variable. Falls back to a
    single entry built from the legacy inverse_param_name/inverse_param_init
    fields when the JSON is empty or unparseable -- covers configs saved
    before this feature existed, and stays perfectly single-variable-
    compatible: one variable named inverse_param_name, true=None, in that
    case."""
    import json
    raw = getattr(config, "inverse_variables_json", "") or ""
    variables = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = []
        for i, v in enumerate(parsed or []):
            name = str((v or {}).get("name") or f"trainable_variable_{i + 1}").strip()
            if not name:
                name = f"trainable_variable_{i + 1}"
            name = _sanitize_python_identifier(name, f"trainable_variable_{i + 1}")
            try:
                init = float((v or {}).get("init", 1.0))
            except (TypeError, ValueError):
                init = 1.0
            _true_raw = (v or {}).get("true", None)
            try:
                true = float(_true_raw) if _true_raw is not None else None
            except (TypeError, ValueError):
                true = None
            variables.append((name, init, true))
    if not variables:
        variables.append((
            _sanitize_python_identifier(config.inverse_param_name, "trainable_variable_1"),
            config.inverse_param_init, None))

    # Two different user-typed names can sanitize to the same identifier
    # (e.g. "D 1" and "D_1" both become "D_1") -- de-duplicate so the
    # generated script never defines the same variable name twice.
    _seen = {}
    _deduped = []
    for name, init, true in variables:
        if name in _seen:
            _seen[name] += 1
            name = f"{name}_{_seen[name]}"
        else:
            _seen[name] = 0
        _deduped.append((name, init, true))
    return _deduped

OBS_REDUCE_HELPER = """
def _obs_reduce(_d, _n_coord, _n_space, _mode, _n, _seed=0):
    # Fewer measured points (Inverse -> Measured data -> Use): "grid" keeps
    # _n evenly spaced values per space axis (edges kept), the same for
    # each time in the file; "random" keeps _n points.
    _n0 = len(_d)
    if _mode == "grid":
        _t_groups = ([np.where(_d[:, _n_coord - 1] == _tv)[0] for _tv in np.unique(_d[:, _n_coord - 1])]
                     if _n_coord > _n_space else [np.arange(_n0)])
        _grid = all(int(np.prod([len(np.unique(_d[_g, _k])) for _k in range(_n_space)])) == len(_g)
                    for _g in _t_groups)
        if not _grid:
            print(f"Measured data is not on a grid: keeping {_n ** _n_space} random points instead")
            _mode, _n = "random", _n ** _n_space
        else:
            _keep = []
            for _g in _t_groups:
                _m = np.ones(len(_g), dtype=bool)
                for _k in range(_n_space):
                    _vals = np.unique(_d[_g, _k])
                    if len(_vals) > _n:
                        _pick = _vals[np.unique(np.round(np.linspace(0, len(_vals) - 1, _n)).astype(int))]
                        _m &= np.isin(_d[_g, _k], _pick)
                _keep.append(_g[_m])
            _d = _d[np.sort(np.concatenate(_keep))]
            print(f"Measured data: {len(_d)} of {_n0} points kept ({_n} per axis)")
            return _d
    if _mode == "random" and _n < _n0:
        _d = _d[np.sort(np.random.default_rng(_seed).choice(_n0, _n, replace=False))]
        print(f"Measured data: {len(_d)} of {_n0} points kept (random)")
    return _d
"""


import textwrap as _textwrap_obs


def obs_reduce_args(config):
    """(mode, n, seed) for _obs_reduce, or None when every point is used."""
    mode = getattr(config, "inverse_obs_reduce", "all") or "all"
    if mode not in ("grid", "random"):
        return None
    n = int(getattr(config, "inverse_obs_grid_n", 51) if mode == "grid" else getattr(config, "inverse_obs_random_n", 1000))
    seed = int(config.random_seed) if getattr(config, "use_random_seed", True) else 0
    return mode, max(1 if mode == "random" else 2, n), seed


def _parse_inverse_obs_files(config):
    """Parse config.inverse_obs_files_json into an ordered list of
    {"path", "output_idx", "weight", "custom_expr"} dicts, one per
    measured-data file (file 1 first). "custom_expr" is non-empty only
    when the measured data corresponds to a *derived* field the raw
    network outputs don't have as a single column -- e.g. 1D
    Schrodinger's data is the magnitude |h| = sqrt(u**2+v**2), not
    output u or v alone. When set, the generated script matches this
    file against that expression (evaluated against ALL of this
    problem's outputs by name, via dde.icbc.PointSetOperatorBC) instead
    of a single output column (dde.icbc.PointSetBC(component=output_idx))
    -- mirrors the existing Results-panel "Custom..." plot-field option
    (see main_window.py's plot_custom_expr / codegen.py's
    _extract_plot_field), just for the *measured-data* side of an
    Inverse problem instead of the plotted solution. Falls back to a
    single entry built from the legacy inverse_data_file/
    inverse_obs_output_idx/loss_weight_obs fields when the JSON is empty
    or unparseable -- covers configs saved before this feature existed,
    and stays perfectly single-file-compatible: one observation file/
    constraint/weight, no custom expression, in that case, exactly like
    before. Unlike trainable variables, every entry here is *data*, not
    a Python identifier, so it needs no codegen-time unrolling -- the
    whole list is embedded as one repr() literal and looped over at
    runtime."""
    import json
    raw = getattr(config, "inverse_obs_files_json", "") or ""
    files = []
    if raw:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed = []
        for f in (parsed or []):
            f = f or {}
            path = str(f.get("path") or "").strip()
            try:
                output_idx = int(f.get("output_idx", 0))
            except (TypeError, ValueError):
                output_idx = 0
            try:
                weight = float(f.get("weight", 100.0))
            except (TypeError, ValueError):
                weight = 100.0
            custom_expr = str(f.get("custom_expr") or "").strip()
            if path:
                files.append({"path": path, "output_idx": output_idx,
                               "weight": weight, "custom_expr": custom_expr})
    if not files:
        files.append({
            "path": config.inverse_data_file or "",
            "output_idx": getattr(config, "inverse_obs_output_idx", 0),
            "weight": config.loss_weight_obs,
            "custom_expr": "",
        })
    return files



def _training_methods_defs(config):
    """Module-level source for causal training and per-point residual
    weights (empty when neither is on). Both wrap `pde` and scale each
    collocation point's residual, so the mean-squared PDE loss becomes the
    weighted one.

    Causal training (Wang, Sankaran & Perdikaris, CMAME 2024): a point in
    time chunk i is scaled by sqrt(w_i), w_i = exp(-eps * sum_{k<i} L_k),
    L_k the mean squared residual in chunk k (no gradient through w).

    Per-point weights lambda (one per training point and equation, the
    residual scaled by lambda):
      RBA, residual-based attention (Anagnostopoulos, Toscano, Stergiopulos
      & Karniadakis, CMAME 2024): lambda <- gamma*lambda + eta*|r|/max|r|
      after every Adam step (no gradient through lambda);
      SA-PINN, self-adaptive weights (McClenny & Braga-Neto, JCP 2023):
      lambda trained by gradient ascent (Adam, maximize) on the same loss
      the network minimizes.
    Only the training points are weighted (the test loss stays the plain
    one). L-BFGS needs a fixed objective: causal weights are held fixed;
    point weights are either held ("keep") or dropped, so L-BFGS refines on
    the plain loss ("plain", the default -- holding them made already-
    accurate problems such as 1D Burgers and Kovasznay flow 2-10x worse)."""
    causal = bool(getattr(config, "causal_training", False)) and not config.steady_state
    pw = _point_weights_mode(config)
    if not (causal or pw):
        return ""
    out = ["", "# ── Training methods: causal training / per-point weights ──────"]
    if causal:
        out.append(f"""_CAUSAL = {{"M": {int(config.causal_chunks)}, "eps": {float(config.causal_eps)!r}, "t0": {float(config.t_min)!r}, "t1": {float(config.t_max)!r},
           "w": None, "frozen": False, "n": 0,
           "sched": {[0.01, 0.1, 1.0, 10.0, 100.0] if getattr(config, "causal_anneal", False) else None!r}, "k": 0}}
if _CAUSAL["sched"]:
    _CAUSAL["eps"] = _CAUSAL["sched"][0]


def _causal_wrap(_f):
    def _weigh(x, _r):
        _rs = list(_r) if isinstance(_r, (list, tuple)) else [_r]
        if any(_ri.shape[0] != x.shape[0] for _ri in _rs):
            return _r                      # residual on a subset of points (fractional): unchanged
        _M = _CAUSAL["M"]
        _t = x[:, -1].detach()
        _t0, _t1 = float(_t.min()), float(_t.max())   # this training window (one Time-Adaptive step, or all of it)
        _CAUSAL["t0"], _CAUSAL["t1"] = _t0, _t1
        _idx = ((_t - _t0) / max(_t1 - _t0, 1e-12) * _M).floor().clamp(0, _M - 1).long()
        if _CAUSAL["frozen"] and _CAUSAL["w"] is not None:
            _w = _CAUSAL["w"].to(x.device)
        else:
            with torch.no_grad():
                _sq = sum(_ri.detach().reshape(-1) ** 2 for _ri in _rs)
                _L = torch.zeros(_M, dtype=_sq.dtype, device=x.device).scatter_add_(0, _idx, _sq)
                _c = torch.zeros(_M, dtype=_sq.dtype, device=x.device).scatter_add_(0, _idx, torch.ones_like(_sq))
                _L = _L / _c.clamp(min=1)
                _w = torch.exp(-_CAUSAL["eps"] * (torch.cumsum(_L, 0) - _L))
            if x.shape[0] >= _CAUSAL["n"]:     # the training points (not the smaller test set)
                _CAUSAL["w"], _CAUSAL["n"] = _w.detach(), x.shape[0]
        _s = _w[_idx].sqrt().unsqueeze(1).to(_rs[0].dtype)
        _out = [_ri * _s for _ri in _rs]
        return _out if isinstance(_r, (list, tuple)) else _out[0]
    import inspect as _inspect
    if len(_inspect.signature(_f).parameters) == 3:    # fractional: pde(x, y, int_mat)
        def _pde_causal(x, y, int_mat=None):
            return _weigh(x, _f(x, y, int_mat))
    else:
        def _pde_causal(x, y):
            return _weigh(x, _f(x, y))
    return _pde_causal


pde = _causal_wrap(pde)
print(f"Causal training: {{_CAUSAL['M']}} time chunks, epsilon = "
      + (f"{{_CAUSAL['sched']}} (next one once all weights > 0.99)" if _CAUSAL["sched"] else f"{{_CAUSAL['eps']}}"))


class _CausalMonitor(dde.callbacks.Callback):
    \"\"\"Prints how far in time the causal weights say the solution is
    settled, and holds the weights fixed during L-BFGS.\"\"\"
    def __init__(self, every=1000):
        super().__init__()
        self.every = every

    def on_train_begin(self):
        _CAUSAL["frozen"] = "L-BFGS" in str(getattr(self.model, "opt_name", ""))
        if _CAUSAL["sched"] and not _CAUSAL["frozen"]:     # a new Adam run (e.g. the next time-adaptive step): start the schedule over
            _CAUSAL["k"], _CAUSAL["eps"] = 0, _CAUSAL["sched"][0]

    def on_batch_end(self):
        st = self.model.train_state
        if (_CAUSAL["sched"] and not _CAUSAL["frozen"] and _CAUSAL["w"] is not None
                and _CAUSAL["k"] < len(_CAUSAL["sched"]) - 1 and float(_CAUSAL["w"].min()) > 0.99):
            _CAUSAL["k"] += 1
            _CAUSAL["eps"] = _CAUSAL["sched"][_CAUSAL["k"]]
            print(f"  causal: all weights > 0.99 at step {{st.step}}, epsilon -> {{_CAUSAL['eps']}}", flush=True)
        if st.step % self.every or _CAUSAL["w"] is None:
            return
        _w = _CAUSAL["w"].cpu().numpy()
        _ok = int((_w > 0.99).sum()) if (_w > 0.99).all() else int(np.argmax(_w <= 0.99))
        _tt = _CAUSAL["t0"] + (_CAUSAL["t1"] - _CAUSAL["t0"]) * _ok / _CAUSAL["M"]
        print(f"  causal weights (epsilon {{_CAUSAL['eps']}}): min {{_w.min():.3g}}, settled up to t = {{_tt:.3g}}"
              + (" (held during L-BFGS)" if _CAUSAL["frozen"] else ""), flush=True)
""")
    if pw:
        out.append(f"""_PW = {{"mode": {pw!r}, "gamma": {float(config.rba_gamma)!r}, "eta": {float(config.rba_eta)!r}, "lr": {float(config.sa_lr)!r},
       "norm": {bool(getattr(config, "sa_normalize", True))!r},
       "keep": {str(getattr(config, "point_weights_lbfgs", "plain")) == "keep"!r}, "lam": None, "fp": None, "frozen": False, "pending": False, "opt": None}}


def _pw_sums(X):
    # checksum of the whole point set: plain sum and a position-weighted one
    # (so a reordering or a few moved points change it too)
    _X = torch.as_tensor(X).detach().to(torch.float64).reshape(len(X), -1)
    _i = torch.arange(1, len(_X) + 1, dtype=torch.float64, device=_X.device).unsqueeze(1) / len(_X)
    return float(_X.sum()), float((_X * _i).sum())


def _pw_is_train(x):
    _fp = _PW["fp"]
    if _fp is None or x.shape[0] != _fp[0]:
        return False
    _a, _b = _pw_sums(x)
    return (abs(_a - _fp[1]) <= 1e-5 * (1 + abs(_fp[1]))
            and abs(_b - _fp[2]) <= 1e-5 * (1 + abs(_fp[2])))


def _pw_wrap(_f):
    def _weigh(x, _r):
        _rs = list(_r) if isinstance(_r, (list, tuple)) else [_r]
        if any(_ri.shape[0] != x.shape[0] for _ri in _rs) or not _pw_is_train(x):
            return _r                      # test points, or a residual on a subset of points: unchanged
        if _PW["frozen"] and not _PW["keep"]:
            return _r                      # L-BFGS on the plain loss
        _n, _k = x.shape[0], len(_rs)
        if _PW["lam"] is None or tuple(_PW["lam"].shape) != (_n, _k):
            _PW["lam"] = torch.ones(_n, _k, dtype=_rs[0].dtype, device=x.device,
                                    requires_grad=_PW["mode"] == "SA-PINN")
            _PW["opt"] = (torch.optim.Adam([_PW["lam"]], lr=_PW["lr"], maximize=True)
                          if _PW["mode"] == "SA-PINN" else None)
        _lam = _PW["lam"]
        if _PW["mode"] == "RBA" and _PW["pending"] and not _PW["frozen"]:
            with torch.no_grad():
                _a = torch.cat([_ri.detach().abs().reshape(-1, 1) for _ri in _rs], dim=1)
                _lam.mul_(_PW["gamma"]).add_(_PW["eta"] * _a / _a.max(dim=0, keepdim=True).values.clamp(min=1e-30))
            _PW["pending"] = False
        if _PW["frozen"] or _PW["mode"] == "RBA":
            _lam = _lam.detach()
        _out = [_ri * _lam[:, j:j + 1].to(_ri.dtype) for j, _ri in enumerate(_rs)]
        return _out if isinstance(_r, (list, tuple)) else _out[0]
    import inspect as _inspect
    if len(_inspect.signature(_f).parameters) == 3:    # fractional: pde(x, y, int_mat)
        def _pde_pw(x, y, int_mat=None):
            return _weigh(x, _f(x, y, int_mat))
    else:
        def _pde_pw(x, y):
            return _weigh(x, _f(x, y))
    return _pde_pw


pde = _pw_wrap(pde)
print(f"Per-point weights: {{_PW['mode']}}" + (f" (gamma = {{_PW['gamma']}}, eta = {{_PW['eta']}})" if _PW["mode"] == "RBA"
                                               else f" (learning rate {{_PW['lr']}}, gradient ascent)"))


class _PointWeights(dde.callbacks.Callback):
    \"\"\"Marks the training points, updates the weights once per Adam step
    (RBA) or takes the ascent step (SA-PINN), and holds them during L-BFGS.\"\"\"
    def __init__(self, every=1000):
        super().__init__()
        self.every = every

    def on_train_begin(self):
        _PW["frozen"] = "L-BFGS" in str(getattr(self.model, "opt_name", ""))
        self._mark()

    def _mark(self):
        X = self.model.train_state.X_train
        if X is not None and len(X):
            fp = (len(X),) + _pw_sums(X)
            if _PW["fp"] is not None and fp != _PW["fp"] and _PW["lam"] is not None:
                # new training points (a resampler, a RAR round, the next
                # time-adaptive step): weights belong to points -- start over
                _PW["lam"], _PW["opt"] = None, None
            _PW["fp"] = fp

    def on_batch_begin(self):
        if not _PW["frozen"]:
            self._mark()
            _PW["pending"] = True

    def on_batch_end(self):
        lam = _PW["lam"]
        if lam is None:
            return
        if _PW["mode"] == "SA-PINN" and not _PW["frozen"] and lam.grad is not None:
            _PW["opt"].step()
            _PW["opt"].zero_grad()
            with torch.no_grad():
                lam.clamp_(min=0.0)
                if _PW["norm"]:          # mean weight 1 per equation: the emphasis moves, the loss scale doesn't grow
                    lam.div_(lam.mean(dim=0, keepdim=True).clamp(min=1e-30))
        st = self.model.train_state
        if st.step % self.every == 0:
            l = lam.detach()
            print(f"  point weights: mean {{float(l.mean()):.3g}}, max {{float(l.max()):.3g}}"
                  + ((" (held during L-BFGS)" if _PW["keep"] else " (L-BFGS on the plain loss)")
                     if _PW["frozen"] else ""), flush=True)
""")
    return "\n".join(out) + "\n"


def _point_weights_mode(config):
    m = str(getattr(config, "point_weights", "None") or "None")
    return m if m in ("RBA", "SA-PINN") else ""


def _training_methods_cbs(config, var_name, pad):
    lines = []
    if bool(getattr(config, "causal_training", False)) and not config.steady_state:
        lines.append(f"{pad}{var_name}.append(_CausalMonitor(1000))")
    if _point_weights_mode(config):
        lines.append(f"{pad}{var_name}.append(_PointWeights(1000))")
    return lines


def _build_train_cbs_code(config, var_name="_train_cbs", indent=4):
    """Builds the literal Python source (already indented at `indent`
    spaces, matching whatever base indent level the generated script has
    at the call site this is embedded into -- 4 for the Standard path, 8
    for the Time-Adaptive per-step loop) that constructs the shared list
    of opt-in DeepXDE training callbacks (EarlyStopping, PDEPointResampler,
    ModelCheckpoint, Timer) from config.cb_*. All four are off by default,
    so an unconfigured run gets exactly `{var_name} = []`, appended into
    every mainline .train() call's existing callbacks list -- a no-op,
    byte-for-byte reproducing every pre-existing generated script. NOT
    threaded into IC pre-training or the RAR refinement sub-loop (see the
    call sites) -- those are short, purpose-built inner loops where early-
    stopping/point-resampling/checkpointing would fight their intent
    rather than help it. EarlyStopping's start_from_epoch needs
    deepxde>=1.12.0 (older than this app's minimum pinned 1.10.0) -- only
    attempted when the user set a nonzero value, and falls back gracefully
    with a printed note rather than crashing if the installed deepxde is
    too old for it."""
    pad = " " * indent
    pad2 = " " * (indent + 4)
    lines = [f"{pad}{var_name} = []"]
    if config.cb_early_stopping:
        baseline_src = "None"
        raw_baseline = str(config.cb_early_stopping_baseline or "").strip()
        if raw_baseline:
            try:
                baseline_src = repr(float(raw_baseline))
            except ValueError:
                baseline_src = "None"
        es_kwargs = (
            f"min_delta={config.cb_early_stopping_min_delta}, "
            f"patience={int(config.cb_early_stopping_patience)}, "
            f"baseline={baseline_src}, "
            f"monitor={config.cb_early_stopping_monitor!r}"
        )
        if config.cb_early_stopping_start_from > 0:
            lines.append(f"{pad}try:")
            lines.append(
                f"{pad2}{var_name}.append(dde.callbacks.EarlyStopping("
                f"{es_kwargs}, start_from_epoch={int(config.cb_early_stopping_start_from)}))"
            )
            lines.append(f"{pad}except TypeError:")
            lines.append(
                f"{pad2}print(\"Note: this deepxde version doesn't support "
                "EarlyStopping's start_from_epoch (added in 1.12.0) -- ignoring it.\")"
            )
            lines.append(f"{pad2}{var_name}.append(dde.callbacks.EarlyStopping({es_kwargs}))")
        else:
            lines.append(f"{pad}{var_name}.append(dde.callbacks.EarlyStopping({es_kwargs}))")
    if config.cb_point_resampler:
        lines.append(
            f"{pad}{var_name}.append(dde.callbacks.PDEPointResampler("
            f"period={int(config.cb_point_resampler_period)}, "
            f"pde_points={bool(config.cb_point_resampler_pde_points)}, "
            f"bc_points={bool(config.cb_point_resampler_bc_points)}))"
        )
    elif _bc_entries_use_batching_json(config.custom_bc_json):
        # A Point Set / Point Set Operator BC row has batch_size set (see
        # the Boundary Conditions panel), but the user never separately
        # turned on Training Callbacks' own Point Resampler (bc_points) --
        # without it, DeepXDE draws that BC's random batch once and keeps
        # training on the exact same points forever, which isn't what
        # "batch size" implies. Added automatically here instead, with
        # DeepXDE's own default period (not forced to any particular
        # value, so it stays correct if that default ever changes).
        lines.append(f"{pad}{var_name}.append(dde.callbacks.PDEPointResampler(bc_points=True))")
    if config.cb_model_checkpoint:
        lines.append(
            f'{pad}{var_name}.append(dde.callbacks.ModelCheckpoint('
            f'_os.path.join(_sol_dir, "checkpoint_best"), verbose=1, '
            f"save_better_only={bool(config.cb_checkpoint_save_better_only)}, "
            f"period={int(config.cb_checkpoint_period)}, "
            f"monitor={config.cb_checkpoint_monitor!r}))"
        )
    if config.cb_timer:
        lines.append(f"{pad}{var_name}.append(dde.callbacks.Timer(available_time={config.cb_timer_minutes}))")
    lines.extend(_training_methods_cbs(config, var_name, pad))
    return "\n".join(lines)


def _parse_sigmas(text):
    """"1, 10" -> [1.0, 10.0] (Fourier-feature scales, Round 73)."""
    out = []
    for p in str(text or "").replace(";", ",").split(","):
        p = p.strip()
        if p:
            out.append(float(p))
    return out


# Multi-scale Fourier feature networks (Wang, Wang & Perdikaris, CMAME 2021),
# as in DeepXDE's dde.nn.MsFFN / STMsFFN, written here so they work with the
# released DeepXDE (whose PyTorch backend doesn't have them yet) and keep one
# checkpoint format for Restore. The random Fourier matrices are buffers, so
# a saved model restores with exactly the features it was trained with.
_MSFFN_CODE = """
class _MsFFN(dde.nn.NN):
    # Input -> [cos(x B_k), sin(x B_k)] for each scale sigma_k (B_k ~ N(0, sigma_k^2),
    # fixed) -> the same hidden layers for every scale -> concatenated -> linear output.
    # Space-time version (sigmas_t given): x and t get their own features,
    # and every (space, time) pair of branches is multiplied point-wise.
    def __init__(self, layer_sizes, activation, kernel_initializer, sigmas, sigmas_t=None):
        super().__init__()
        self.activation = dde.nn.activations.get(activation)
        _init = dde.nn.initializers.get(kernel_initializer)
        _zero = dde.nn.initializers.get("zeros")
        self.space_time = bool(sigmas_t)
        _n_ff = layer_sizes[1] // 2
        _n_x = layer_sizes[0] - (1 if self.space_time else 0)
        self.n_x_scales = len(sigmas)
        for _k, _s in enumerate(sigmas):
            self.register_buffer(f"B_x{_k}", torch.randn(_n_x, _n_ff) * float(_s))
        self.n_t_scales = len(sigmas_t or [])
        for _k, _s in enumerate(sigmas_t or []):
            self.register_buffer(f"B_t{_k}", torch.randn(1, _n_ff) * float(_s))
        self.linears = torch.nn.ModuleList()
        for _i in range(2, len(layer_sizes) - 1):
            _lin = torch.nn.Linear(layer_sizes[_i - 1], layer_sizes[_i])
            _init(_lin.weight); _zero(_lin.bias)
            self.linears.append(_lin)
        _n_branches = self.n_x_scales * (self.n_t_scales if self.space_time else 1)
        self.dense = torch.nn.Linear(layer_sizes[-2] * _n_branches, layer_sizes[-1])
        _init(self.dense.weight); _zero(self.dense.bias)

    def _branch(self, _x, _B):
        _p = _x @ _B
        _y = torch.cat([torch.cos(_p), torch.sin(_p)], dim=1)
        for _lin in self.linears:
            _y = self.activation(_lin(_y))
        return _y

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        if self.space_time:
            _yx = [self._branch(x[:, :-1], getattr(self, f"B_x{_k}")) for _k in range(self.n_x_scales)]
            _yt = [self._branch(x[:, -1:], getattr(self, f"B_t{_k}")) for _k in range(self.n_t_scales)]
            _ys = [_a * _b for _a in _yx for _b in _yt]
        else:
            _ys = [self._branch(x, getattr(self, f"B_x{_k}")) for _k in range(self.n_x_scales)]
        y = self.dense(torch.cat(_ys, dim=1))
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y
"""


def _net_construction_helper_code(network_type, sigmas="1, 10", sigmas_t="1, 10"):
    """Builds the literal Python source for a `_make_net(_flat_layers,
    _activation, _kernel_init, _weight_decay=0.0)` helper, embedded once
    near the top of every generated/restore script and called at every
    network-construction site in place of a bare `dde.nn.FNN(...)`. This is
    the SINGLE place that decides network_type (FNN vs DeepXDE's own
    built-in PFNN class) -- shared verbatim (via this same function) by
    generate_script(), generate_clean_script(), AND every Model Restore /
    Error Analysis restore-script builder in main_window.py, so a training
    script and the script that later restores its checkpoint can never
    independently drift apart on how the network is shaped. That parity is
    the critical correctness requirement here: DeepXDE's model.restore()
    loads a raw state_dict by key/shape, so a restore-side net built with
    even a slightly different layer shape either crashes outright or
    silently loads nothing useful.

    layers[0] (raw input dim) and layers[-1] (num_outputs) always come from
    the caller's flat `_flat_layers` list unchanged -- only the hidden-layer
    shape (nested per-branch lists for PFNN) is computed here, purely
    locally, rather than ever being written back into a stored layers list
    -- deliberately avoiding a second place that needs the same shape
    derivation (see config.py's network_type field comment).

    PFNN has no `regularization` kwarg in DeepXDE (confirmed against its
    PyTorch backend) -- weight_decay is silently dropped for PFNN here
    rather than raising, since weight decay is a secondary training knob,
    not something that should block switching architectures."""
    _ms = network_type in ("MsFFN", "STMsFFN")
    _ms_code = _MSFFN_CODE if _ms else ""
    _ms_branch = (f'''    if _network_type in ("MsFFN", "STMsFFN"):
        return _MsFFN(_flat_layers, _activation, _kernel_init, {_parse_sigmas(sigmas)!r},
                      {(_parse_sigmas(sigmas_t) if network_type == "STMsFFN" else None)!r})
''' if _ms else "")
    return f'''# ── Network architecture: FNN, DeepXDE's built-in PFNN, or a multi-scale Fourier feature network ──
_network_type = {network_type!r}
{_ms_code}
def _make_net(_flat_layers, _activation, _kernel_init, _weight_decay=0.0):
    _flat_layers = list(_flat_layers)
    _hidden = list(_flat_layers[1:-1])
    _n_out = _flat_layers[-1]
{_ms_branch}    if _network_type == "PFNN":
        _eff_layers = [_flat_layers[0]] + [[_w] * _n_out for _w in _hidden] + [_n_out]
        _net = dde.nn.PFNN(_eff_layers, _activation, _kernel_init)
    else:
        if _weight_decay and _weight_decay > 0:
            _net = dde.nn.FNN(_flat_layers, _activation, _kernel_init, regularization=("l2", _weight_decay))
        else:
            _net = dde.nn.FNN(_flat_layers, _activation, _kernel_init)
    return _net
'''


def _loss_plot_runtime_code(config):
    """Builds the literal Python source for a module-level `_make_loss_plot(
    train_rows, test_rows, steps, save_path, title, xlabel="Iteration",
    extra_fn=None)` helper, embedded once near the top of the generated
    script (alongside `net_helper_code`/`_tm_runtime_code`, same pattern)
    and called at every loss-plotting site in generate_script() (the
    Standard/RAR path and the Time-Adaptive path) and
    generate_clean_script() -- honoring the Round 28 Loss Plot Settings
    (config.loss_plot_mode/loss_plot_log_y/loss_plot_linewidth).

    Defining this as a reusable FUNCTION at module level, rather than a
    block of statements spliced in at each (differently-indented) call
    site, sidesteps the indentation problem a multi-line splice would
    hit inside, e.g., the Standard path's `if not {{config.time_adaptive}}:`
    block -- every call site only needs ONE correctly-indented statement
    calling this function, exactly like `_make_net(...)` below it.

    `train_rows`/`test_rows` must be DeepXDE's own loss_history.loss_train/
    loss_test shape: one row per recorded step, each row a list of every
    individual loss TERM's value that step, PDE terms first -- one per
    output, in output order -- followed by every BC/IC term in whatever
    order codegen's own data.add_boundary_condition/IC-constraint-building
    added them.

    loss_plot_mode="train_test" (default) reproduces the exact plot every
    loss figure in the app already drew before this feature existed --
    summed Train + Test, semilogy -- so this is a verified no-op for any
    config that doesn't touch the new setting. "individual" additionally
    needs per-term labels: DeepXDE provides no per-term names at all, and
    reconstructing codegen's own full BC/IC term order statically (outside
    the generated script, which builds `_constraints`/`_constraints_i`
    from runtime-parsed custom_bc_json) would risk a label drifting out of
    sync with the real term it's attached to -- a mislabeled line is worse
    than an unlabeled one. So only the leading `config.num_outputs` PDE
    terms (a static, always-true fact -- see the matching comment in
    _rar_save_round_diagnostics/_ea_extract's own loss-term-adjacent code)
    get a real label ("PDE (u)", from config.output_names); every
    following term is labeled generically ("Constraint 1", "Constraint
    2", ...) by its position past the PDE terms. DeepXDE never tracks Test
    loss per-term (only the single summed total loss_history.loss_test
    itself), so "individual"/"all" only ever add TRAIN per-term lines.

    `extra_fn`, if given, is called with the Axes object AFTER the loss
    lines are drawn but BEFORE the figure is finalized/saved -- the one
    call site with its own extra (the Time-Adaptive accumulated loss
    plot's per-sub-domain-boundary `axvline` markers) passes a small
    closure here instead of this shared helper needing to know about
    Time-Adaptive sub-domain boundaries at all."""
    _pde_names = [n.strip() or f"u{i+1}" for i, n in enumerate((config.output_names or "u").split(","))]
    return f'''_LOSS_MODE = {config.loss_plot_mode!r}
_LOSS_LOG_Y = {config.loss_plot_log_y!r}
_LOSS_LW = {config.loss_plot_linewidth!r}
_LOSS_PDE_NAMES = {_pde_names!r}

def _loss_term_label(li):
    if li < len(_LOSS_PDE_NAMES):
        return f"PDE ({{_LOSS_PDE_NAMES[li]}})"
    return f"Constraint {{li - len(_LOSS_PDE_NAMES) + 1}}"

def _make_loss_plot(train_rows, test_rows, steps, save_path, title, xlabel="Iteration", extra_fn=None):
    train_rows = list(train_rows); steps = list(steps)
    total_train = [sum(r) for r in train_rows]
    total_test = [sum(r) for r in test_rows] if test_rows else []

    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    plotfn = ax.semilogy if _LOSS_LOG_Y else ax.plot
    if _LOSS_MODE in ("train_test", "all"):
        plotfn(steps, total_train, label="Train loss", color="#4dabf7", linewidth=_LOSS_LW)
        if total_test:
            plotfn(steps, total_test, label="Test loss", color="#ff8787", linestyle="--", linewidth=_LOSS_LW)
    if _LOSS_MODE in ("individual", "all") and train_rows:
        n_terms = len(train_rows[0])
        term_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(n_terms, 1)))
        for li in range(n_terms):
            plotfn(steps, [r[li] if li < len(r) else float("nan") for r in train_rows],
                   label=_loss_term_label(li), color=term_colors[li],
                   linewidth=max(_LOSS_LW - 0.5, 0.5), alpha=0.85)
    if extra_fn is not None:
        extra_fn(ax)
    ax.set_xlabel(xlabel, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("Loss", fontsize={config.plot_axis_label_fontsize})
    ax.set_title(title)
    ax.legend(fontsize=_lfs(8 if _LOSS_MODE != "train_test" else 10))
    _grid(ax, alpha=0.3, which="major")
    fig.tight_layout()
    fig.savefig(save_path, dpi={config.plot_dpi}); plt.close(fig)
'''


def _training_monitor_runtime_code():
    """Builds the literal Python source for the Training Monitors feature's
    shared, static derivative-building/expression-evaluation helpers --
    embedded once near the top of every generated script, exactly like
    _net_construction_helper_code() above.

    Deliberately NOT shared with _pde_standard()'s own derivative builder
    (the nested `_hess`/`_dvars` logic inside generate_script()'s PDE
    definition) even though the math is identical -- this is an
    intentionally separate, self-contained copy so a diagnostic-only
    feature can never perturb the already-tested PDE-residual codegen
    path. The duplication is the deliberate tradeoff (see Round 15's
    network-architecture lesson about avoiding a second place that needs
    the same derivation logic -- accepted here specifically to keep this
    feature's blast radius at zero for the PDE itself).

    `_tm_build_dvars(x, y, n_out, out_names, is_steady, dim, expr_check)`
    builds only the derivative entries actually referenced as substrings
    of `expr_check` (a Training Monitor's own expression text), using the
    same d{name}_x / d{name}_xx / d{name}_xy / d{name}_xxxx naming the
    Custom PDE expression box already uses -- so a user who already knows
    that syntax from writing PDEs in this app can use it here unchanged.
    `dim` is "1D"/"2D"/"3D"; higher-order terms referencing a "t" column
    that doesn't exist for a steady problem are only safe to write if the
    monitor's own `is_steady` is False, exactly mirroring the same
    unenforced assumption _pde_standard()'s PDE box already relies on
    (there's no "t" in a steady problem's vocabulary to begin with).

    `_tm_eval_terms(expr_list, x, y, dvars, out_names)` splits a monitor's
    comma-separated expression list, evaluates each term against the
    built `dvars` (plus x/y/dde/torch/np), and torch.cat's the results
    into one (batch, k) tensor -- the "beyond a single derivative pair"
    capability: one dde.callbacks.OperatorPredictor can then log several
    related quantities (e.g. "u, du_x, du_xx") together, in one file,
    against the same points, every call."""
    return '''# ── Training Monitors: shared derivative-building/eval helpers ──
# (see config.py's training_monitors_enabled/training_monitors docstring)
def _tm_hess(ys, xs, i, j, _oi, _n_out):
    if _n_out == 1:
        return dde.grad.hessian(ys, xs, i=i, j=j)
    return dde.grad.hessian(ys, xs, component=_oi, i=i, j=j)

def _tm_build_dvars(x, y, n_out, out_names, is_steady, dim, expr_check):
    dvars = {}
    for oi, oname in enumerate(out_names):
        dvars[oname] = y[:, oi:oi + 1]
        if dim == "3D":
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            dvars[f"d{oname}_y"] = dde.grad.jacobian(y, x, i=oi, j=1)
            dvars[f"d{oname}_z"] = dde.grad.jacobian(y, x, i=oi, j=2)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=3)
            if f"d{oname}_xx" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_yy" in expr_check:
                dvars[f"d{oname}_yy"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_zz" in expr_check:
                dvars[f"d{oname}_zz"] = _tm_hess(y, x, 2, 2, oi, n_out)
            if f"d{oname}_tt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 3, 3, oi, n_out)
            if f"d{oname}_xy" in expr_check:
                dvars[f"d{oname}_xy"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_xz" in expr_check:
                dvars[f"d{oname}_xz"] = _tm_hess(y, x, 0, 2, oi, n_out)
            if f"d{oname}_yz" in expr_check:
                dvars[f"d{oname}_yz"] = _tm_hess(y, x, 1, 2, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 3, oi, n_out)
            if f"d{oname}_yt" in expr_check:
                dvars[f"d{oname}_yt"] = _tm_hess(y, x, 1, 3, oi, n_out)
            if f"d{oname}_zt" in expr_check:
                dvars[f"d{oname}_zt"] = _tm_hess(y, x, 2, 3, oi, n_out)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_yyyy" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyyy"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=1, j=1)
            if f"d{oname}_zzzz" in expr_check and f"d{oname}_zz" in dvars:
                dvars[f"d{oname}_zzzz"] = dde.grad.hessian(dvars[f"d{oname}_zz"], x, i=2, j=2)
            if f"d{oname}_xxyy" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxyy"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_xxzz" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxzz"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=2, j=2)
            if f"d{oname}_yyzz" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyzz"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=2, j=2)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=3, j=3)
            if f"d{oname}_yytt" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yytt"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=3, j=3)
            if f"d{oname}_zztt" in expr_check and f"d{oname}_zz" in dvars:
                dvars[f"d{oname}_zztt"] = dde.grad.hessian(dvars[f"d{oname}_zz"], x, i=3, j=3)
        elif dim == "2D":
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            dvars[f"d{oname}_y"] = dde.grad.jacobian(y, x, i=oi, j=1)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=2)
            if f"d{oname}_xx" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_yy" in expr_check:
                dvars[f"d{oname}_yy"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_xy" in expr_check:
                dvars[f"d{oname}_xy"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_tt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 2, 2, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 2, oi, n_out)
            if f"d{oname}_yt" in expr_check:
                dvars[f"d{oname}_yt"] = _tm_hess(y, x, 1, 2, oi, n_out)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_yyyy" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yyyy"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=1, j=1)
            if f"d{oname}_xxyy" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxyy"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=2, j=2)
            if f"d{oname}_yytt" in expr_check and f"d{oname}_yy" in dvars:
                dvars[f"d{oname}_yytt"] = dde.grad.hessian(dvars[f"d{oname}_yy"], x, i=2, j=2)
        else:
            dvars[f"d{oname}_x"] = dde.grad.jacobian(y, x, i=oi, j=0)
            if not is_steady:
                dvars[f"d{oname}_t"] = dde.grad.jacobian(y, x, i=oi, j=1)
            if f"d{oname}_xx" in expr_check or f"d{oname}_xxx" in expr_check or f"d{oname}_xxxx" in expr_check or f"d{oname}_xxtt" in expr_check:
                dvars[f"d{oname}_xx"] = _tm_hess(y, x, 0, 0, oi, n_out)
            if f"d{oname}_tt" in expr_check or f"d{oname}_tttt" in expr_check or f"d{oname}_xxtt" in expr_check:
                dvars[f"d{oname}_tt"] = _tm_hess(y, x, 1, 1, oi, n_out)
            if f"d{oname}_xt" in expr_check:
                dvars[f"d{oname}_xt"] = _tm_hess(y, x, 0, 1, oi, n_out)
            if f"d{oname}_xxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxx"] = dde.grad.jacobian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_xxxx" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxxx"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=0, j=0)
            if f"d{oname}_xxtt" in expr_check and f"d{oname}_xx" in dvars:
                dvars[f"d{oname}_xxtt"] = dde.grad.hessian(dvars[f"d{oname}_xx"], x, i=1, j=1)
            if f"d{oname}_tttt" in expr_check and f"d{oname}_tt" in dvars:
                dvars[f"d{oname}_tttt"] = dde.grad.hessian(dvars[f"d{oname}_tt"], x, i=1, j=1)
    return dvars

def _tm_eval_terms(expr_list, x, y, dvars, out_names):
    _ns = dict(dvars)
    _ns["x"] = x
    _ns["y"] = y
    _ns["dde"] = dde
    _ns["torch"] = torch
    _ns["np"] = np
    _terms = [eval(_e.strip(), _ns) for _e in expr_list.split(",") if _e.strip()]
    return torch.cat(_terms, dim=1)
'''


def _build_training_monitors_code(config, cbs_var_name="_train_cbs", sol_dir_var="_sol_dir", indent=4):
    """Builds the literal source that constructs one dde.callbacks.
    OperatorPredictor per configured Training Monitor (see config.py's
    training_monitors_enabled/training_monitors) and appends each to the
    shared `{cbs_var_name}` list that _build_train_cbs_code() builds --
    so monitors ride along in every Standard-path .train() call exactly
    like EarlyStopping/PDEPointResampler/ModelCheckpoint/Timer already do,
    with zero per-call-site changes needed. NOT wired into the Time-
    Adaptive per-step loop (`_train_cbs_ta`): each TA step trains an
    independent model from scratch, so a monitor's log file would be
    truncated and restarted every single step rather than accumulating
    one continuous trace -- out of scope for this round, not silently
    wrong.

    Each monitor also gets a small `<name>.meta.json` sidecar (point
    coordinates + expression labels + period), written directly by this
    generated code via plain json.dump, separate from the .txt log file
    OperatorPredictor itself writes -- kept as two separate files
    specifically because OperatorPredictor's own __init__ opens its log
    file in "w" mode (truncating), so anything this code wrote into that
    same file first would simply be wiped the moment the callback is
    constructed. The Results panel reads both: the sidecar to know how
    many points/expressions there are and what to label them, the log to
    get the actual (iteration, value...) time series.

    Returns "" (a true no-op) when Training Monitors is off or no
    monitors are configured, same convention _build_train_cbs_code() uses."""
    if not config.training_monitors_enabled:
        return ""
    import json as _json_tm
    try:
        _monitors = _json_tm.loads(config.training_monitors or "[]")
    except (ValueError, TypeError):
        _monitors = []
    if not isinstance(_monitors, list) or not _monitors:
        return ""

    pad = " " * indent
    _out_names_list = [n.strip() for n in (config.output_names or "").split(",") if n.strip()]
    _dim = config.problem_dim
    _is_steady = bool(config.steady_state)
    _n_out = int(config.num_outputs)

    lines = []
    for _mi, _mon in enumerate(_monitors):
        if not isinstance(_mon, dict):
            continue
        _name = str(_mon.get("name") or f"monitor{_mi}").strip() or f"monitor{_mi}"
        _safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in _name)
        _points = _mon.get("points") or []
        _expr = str(_mon.get("expr") or "").strip()
        _period = int(_mon.get("period") or 1000)
        if not _points or not _expr:
            continue
        _expr_labels = [e.strip() for e in _expr.split(",") if e.strip()]

        lines.append(f"{pad}_tm_points_{_mi} = {_points!r}")
        lines.append(f"{pad}_tm_log_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.txt')")
        lines.append(f"{pad}_tm_meta_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.meta.json')")
        lines.append(f"{pad}with open(_tm_meta_path_{_mi}, 'w') as _tm_mf_{_mi}:")
        lines.append(
            f"{pad}    _json_tm_write.dump({{'name': {_name!r}, 'points': _tm_points_{_mi}, "
            f"'expressions': {_expr_labels!r}, 'period': {_period}}}, _tm_mf_{_mi})"
        )
        lines.append(
            f"{pad}def _tm_op_{_mi}(x, y, _expr={_expr!r}, _out_names={_out_names_list!r}, "
            f"_n_out={_n_out}, _is_steady={_is_steady!r}, _dim={_dim!r}):"
        )
        lines.append(f"{pad}    _dvars = _tm_build_dvars(x, y, _n_out, _out_names, _is_steady, _dim, _expr)")
        lines.append(f"{pad}    return _tm_eval_terms(_expr, x, y, _dvars, _out_names)")
        lines.append(
            f"{pad}{cbs_var_name}.append(dde.callbacks.OperatorPredictor(_tm_points_{_mi}, _tm_op_{_mi}, "
            f"period={_period}, filename=_tm_log_path_{_mi}, precision=8))"
        )
    if not lines:
        return ""
    return (f"{pad}import json as _json_tm_write\n") + "\n".join(lines)


def _build_training_monitors_plot_code(config, sol_dir_var="_sol_dir", indent=0):
    """Builds the literal source (placed near the very end of the
    generated script, just before the final "DONE" print) that reads
    each Training Monitor's own .txt log + .meta.json sidecar back in
    (the exact same files _build_training_monitors_code() wrote the
    OperatorPredictor callbacks to write) and renders one line plot per
    monitor via matplotlib (already a hard dependency of this app,
    imported as `plt` everywhere else in this script), saved as
    `monitor_<name>_plot.png` next to the log itself -- so the Results
    panel can display it exactly the way it already displays
    loss_plot.png/solution_plot.png (see main_window.py's
    _display_run_result_plots()), with zero new rendering logic needed
    on the GUI side.

    dde.callbacks.OperatorPredictor logs each line as
    "<iteration> [v0, v1, ...]" (space before the bracket, DeepXDE's own
    utils.list_to_str() format) with the flattened values in point-major,
    expression-minor order -- matching exactly how _tm_eval_terms()
    builds its (n_points, n_expr) tensor via torch.cat(..., dim=1) before
    flattening, confirmed by a real training run's logged output during
    this feature's own development, not just inferred from the source.

    Wrapped in its own try/except per monitor -- a plotting failure
    (e.g. an empty log because training was stopped at iteration 0)
    prints a warning and moves on rather than crashing a run that
    otherwise trained and saved successfully."""
    if not config.training_monitors_enabled:
        return ""
    import json as _json_tm2
    try:
        _monitors = _json_tm2.loads(config.training_monitors or "[]")
    except (ValueError, TypeError):
        _monitors = []
    if not isinstance(_monitors, list) or not _monitors:
        return ""

    pad = " " * indent
    lines = [f"{pad}import json as _json_tm_read"]
    _any = False
    for _mi, _mon in enumerate(_monitors):
        if not isinstance(_mon, dict):
            continue
        _name = str(_mon.get("name") or f"monitor{_mi}").strip() or f"monitor{_mi}"
        _safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in _name)
        if not (_mon.get("points") and _mon.get("expr")):
            continue
        _any = True
        lines.append(f"{pad}_tm_plot_log_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.txt')")
        lines.append(f"{pad}_tm_plot_meta_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}.meta.json')")
        lines.append(f"{pad}if _os.path.exists(_tm_plot_log_{_mi}) and _os.path.exists(_tm_plot_meta_{_mi}):")
        lines.append(f"{pad}    try:")
        lines.append(f"{pad}        with open(_tm_plot_meta_{_mi}) as _tm_mf_{_mi}:")
        lines.append(f"{pad}            _tm_meta_{_mi} = _json_tm_read.load(_tm_mf_{_mi})")
        lines.append(f"{pad}        _tm_npts_{_mi} = len(_tm_meta_{_mi}['points'])")
        lines.append(f"{pad}        _tm_nexpr_{_mi} = len(_tm_meta_{_mi}['expressions'])")
        lines.append(f"{pad}        _tm_iters_{_mi} = []")
        lines.append(f"{pad}        _tm_rows_{_mi} = []")
        lines.append(f"{pad}        with open(_tm_plot_log_{_mi}) as _tm_lf_{_mi}:")
        lines.append(f"{pad}            for _tm_line_{_mi} in _tm_lf_{_mi}:")
        lines.append(f"{pad}                _tm_line_{_mi} = _tm_line_{_mi}.strip()")
        lines.append(f"{pad}                if not _tm_line_{_mi}:")
        lines.append(f"{pad}                    continue")
        lines.append(f"{pad}                _tm_it_s_{_mi}, _tm_rest_{_mi} = _tm_line_{_mi}.split(' ', 1)")
        lines.append(f"{pad}                _tm_rows_{_mi}.append([float(_v) for _v in _tm_rest_{_mi}.strip('[]').split(',')])")
        lines.append(f"{pad}                _tm_iters_{_mi}.append(int(_tm_it_s_{_mi}))")
        lines.append(f"{pad}        if _tm_iters_{_mi} and _tm_npts_{_mi} > 0 and _tm_nexpr_{_mi} > 0:")
        lines.append(f"{pad}            _tm_arr_{_mi} = np.array(_tm_rows_{_mi}).reshape(-1, _tm_npts_{_mi}, _tm_nexpr_{_mi})")
        lines.append(f"{pad}            _tm_fig_{_mi}, _tm_ax_{_mi} = plt.subplots(figsize=_plot_figsize(7, 5))")
        lines.append(f"{pad}            for _tm_pi_{_mi} in range(_tm_npts_{_mi}):")
        lines.append(f"{pad}                for _tm_ei_{_mi} in range(_tm_nexpr_{_mi}):")
        lines.append(
            f"{pad}                    _tm_ax_{_mi}.plot(_tm_iters_{_mi}, "
            f"_tm_arr_{_mi}[:, _tm_pi_{_mi}, _tm_ei_{_mi}], marker='o', markersize=3, "
            f"label=f\"{{_tm_meta_{_mi}['expressions'][_tm_ei_{_mi}]}} @ pt{{_tm_pi_{_mi}}}\")"
        )
        lines.append(f"{pad}            _tm_ax_{_mi}.set_xlabel('Iteration', fontsize={config.plot_axis_label_fontsize}); _tm_ax_{_mi}.set_ylabel('Value', fontsize={config.plot_axis_label_fontsize})")
        lines.append(f"{pad}            _tm_ax_{_mi}.set_title({('Training Monitor: ' + _name)!r})")
        lines.append(f"{pad}            _tm_ax_{_mi}.legend(fontsize=_lfs(7), loc='best'); _tm_ax_{_mi}.grid(alpha=0.3)")
        lines.append(f"{pad}            _tm_plot_path_{_mi} = _os.path.join({sol_dir_var}, 'monitor_{_safe_name}_plot.png')")
        lines.append(f"{pad}            plt.tight_layout(); plt.savefig(_tm_plot_path_{_mi}, dpi={int(config.plot_dpi)}); plt.close(_tm_fig_{_mi})")
        lines.append(f"{pad}            print(f'  Training monitor plot saved: {{_tm_plot_path_{_mi}}}')")
        lines.append(f"{pad}    except Exception as _tm_plot_err_{_mi}:")
        lines.append(f'{pad}        print(f"  Warning: failed to plot training monitor {_name!r}: {{_tm_plot_err_{_mi}}}")')
    if not _any:
        return ""
    return "\n".join(lines)


def _ea_metrics_csv_header(config):
    """CSV column header for an error_metrics.txt file, gated by the
    "Error Metrics to Compute" settings (Round 31: config.ea_metric_l2/
    mse/max, wired from the Error Analysis dialog's own checkboxes --
    previously stored but never actually read by anything, so every
    metric was always computed and shown regardless of what was
    checked). "t" (the index column) and "Mean_abs_error" have no
    corresponding checkbox in that dialog, so both always show, same as
    before this feature existed -- only L2_relative/MSE/Max_error are
    gated.
    """
    cols = ["t"]
    if config.ea_metric_l2:
        cols.append("L2_relative")
    if config.ea_metric_mse:
        cols.append("MSE")
    if config.ea_metric_max:
        cols.append("Max_error")
    cols.append("Mean_abs_error")
    return ",".join(cols)


def _ea_metrics_row_code(config, tv, l2, mse, mx, ma):
    """One error_metrics.txt data row's literal runtime-f-string body
    (no surrounding f"..."/quotes) -- matches _ea_metrics_csv_header's
    column selection for the same config. tv/l2/mse/mx/ma are the exact
    local variable names used at the call site (e.g. "_ea_tv" or "tv"),
    so this can drop straight into generate_script()'s own f-string via
    a single-brace {..._row_code} substitution (its returned text's own
    single braces pass through untouched -- see _net_construction_helper_
    code's docstring for the same convention) or into generate_clean_
    script()'s plain (non-f) string fragments unchanged.
    """
    fields = [f"{{{tv}:.6f}}"]
    if config.ea_metric_l2:
        fields.append(f"{{{l2}:.6e}}")
    if config.ea_metric_mse:
        fields.append(f"{{{mse}:.6e}}")
    if config.ea_metric_max:
        fields.append(f"{{{mx}:.6e}}")
    fields.append(f"{{{ma}:.6e}}")
    return ",".join(fields)


def _ea_metrics_print_fragment(config, l2, mse, mx, ma):
    """The "L2=..., MSE=..., Max=..., MeanAbs=..." portion of the
    per-time-snapshot console print line, gated the same way as
    _ea_metrics_csv_header/_ea_metrics_row_code (MeanAbs always shows --
    no corresponding checkbox). The caller keeps its own "  t={tv:.4f} —
    " (or "--") prefix and print(f"...") wrapper; this only decides which
    of the three configurable metrics appear.
    """
    parts = []
    if config.ea_metric_l2:
        parts.append(f"L2={{{l2}:.4e}}")
    if config.ea_metric_mse:
        parts.append(f"MSE={{{mse}:.4e}}")
    if config.ea_metric_max:
        parts.append(f"Max={{{mx}:.4e}}")
    parts.append(f"MeanAbs={{{ma}:.4e}}")
    return ", ".join(parts)


def _gs_rar_loop_code(config):
    """generate_script()'s Residual-based Adaptive Refinement loop, including the
    per-round diagnostics (_rar_save_round_diagnostics) -- cleanup Round 52
    (audit §5): moved verbatim out of generate_script()'s template, where it
    was the largest unnamed inline block. Returns the exact same text the
    template used to contain (byte-for-byte; see the round's notes). Every
    parameter is a generation-time value the text interpolates."""
    return f"""    # ── RAR Loop ─────────────────────────────────────────────
    if "{config.adapt_method}" == "RAR" and not {config.time_adaptive}:
        # ── Per-round RAR diagnostics ──────────────────────────
        # For every RAR round: a static snapshot of whatever field is
        # configured above (raw output or derivative-aware Custom
        # expression -- same _extract_plot_field every other plot uses),
        # a lightweight comparison against one Error Analysis reference
        # file if any are configured (not the full multi-file/multi-time
        # sweep Inline Error Analysis runs once at the end -- just one
        # representative reference, so this doesn't meaningfully slow
        # RAR training down), and a 3-color collocation-points scatter
        # (original domain points vs. points added in EARLIER rounds vs.
        # points added THIS round) so sharp-residual/refinement patterns
        # across rounds are visible afterward, not just the final model.
        # Saved under this run's own solution_results/rar_rounds/
        # round_NN/ -- the same per-run-unique save_dir every other plot
        # already uses (see Round 19's _on_done() fix).
        _rar_rounds_dir = _os.path.join(_sol_dir, "rar_rounds")
        _rar_orig_pts = np.array(data.train_x_all, copy=True)
        _rar_added_so_far = np.empty((0, _rar_orig_pts.shape[1]))
        _rar_ea_entries = {config.ea_files}
        from scipy.interpolate import interp1d as _rar_interp1d

        def _rar_ref_extract(_rre_grid, _rre_sel, _rre_model):
            # Generalized reference-field extractor for RAR's own diagnostics
            # -- mirrors `_ea_extract` (Inline/Restore Error Analysis) and the
            # `_rar_ea_op`/`_rar_ea_op2` closures already used lower down in
            # this function, but as a single reusable helper: _rre_sel is
            # None (delegate to the default plot field), an int (a raw
            # output column), or an [expr, label] pair (a custom derived
            # field, derivative-aware via the same _tm_build_dvars mechanism).
            if _rre_sel is None:
                return _extract_plot_field(_rre_grid, _rre_model)
            if isinstance(_rre_sel, int):
                return _rre_model.predict(_rre_grid)[:, _rre_sel]
            _rre_expr = _rre_sel[0]

            def _rre_op(_rre_i, _rre_o):
                _rre_dv = _tm_build_dvars(_rre_i, _rre_o, _plot_n_out, _plot_output_names_list,
                                           _is_steady, _plot_dim, _rre_expr)
                _rre_ns = dict(_rre_dv)
                _rre_ns.update(_PLOT_TORCH_MATH_NS)
                _rre_ns["torch"] = torch
                return eval(_rre_expr, _rre_ns)

            return _rre_model.predict(_rre_grid, operator=_rre_op)[:, 0]

        def _rar_save_round_diagnostics(_rar_idx, _rar_new_pts):
            _rd = _os.path.join(_rar_rounds_dir, f"round_{{_rar_idx:02d}}")
            _os.makedirs(_rd, exist_ok=True)
            _res = {config.plot_resolution}

            # Per-round model checkpoint -- same convention Time-Adaptive's
            # per-step diagnostics already use (a model file saved
            # alongside that unit's own plots), closing a gap RAR's own
            # per-round diagnostics had since they were first added
            # (Round 20): this round's trained model is NOT saved anywhere
            # unless it happens to also be the FINAL round's model (saved
            # separately, after the whole RAR loop ends, as this run's one
            # overall model). Without this, every earlier round's model is
            # unrecoverable once RAR moves on -- only its plots/metrics
            # survive, not the weights themselves.
            #
            # Round 31: this save always runs at the END of the round,
            # after whichever phase(s) ran (Adam-only, or Adam+L-BFGS).
            # Standardized on "one checkpoint per optimizer phase"
            # everywhere, matching Time-Adaptive's own per-step
            # convention -- when this round also ran an L-BFGS phase, the
            # Adam-phase checkpoint is now saved separately (see
            # "model_adam" save right after the Adam train() call in the
            # round loop below), so THIS save represents the L-BFGS
            # phase's own final state and is named accordingly; when a
            # round has no L-BFGS phase, this is that round's only
            # checkpoint and keeps the "model_adam" name for the same
            # reason the plain Solve path's own Adam-only run does.
            _rar_final_model_name = "model_lbfgs" if {config.rar_lbfgs_iters} > 0 else "model_adam"
            if _use_save:
                try:
                    model.save(_os.path.join(_rd, _rar_final_model_name))
                    print(f"  [RAR round {{_rar_idx}}] {{_rar_final_model_name}} model saved: {{_os.path.join(_rd, _rar_final_model_name)}}.pt")
                except Exception as _rar_err_save:
                    print(f"  [RAR round {{_rar_idx}}] model save failed: {{_rar_err_save}}")

            if not _is_2d and not _is_3d:
                if _is_steady:
                    # Steady-state 1D (e.g. a Poisson equation): a single
                    # static u(x) curve -- no time axis at all, so neither
                    # the x-t surface nor the "earlier-time-steps" framing
                    # the time-dependent branch below uses applies here.
                    # Mirrors the plain (non-RAR) Solve path's own
                    # steady-1D solution plot further above in this file.
                    try:
                        _xr = np.linspace({config.x_min}, {config.x_max}, _res)
                        _field_r = _extract_plot_field(_xr.reshape(-1, 1))
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5))
                        _ax_r.plot(_xr, _field_r, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
                        _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("u", fontsize={config.plot_axis_label_fontsize})
                        _grid(_ax_r, alpha=0.3)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot")
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                        plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                        plt.close(_fig_r)
                        print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
                    except Exception as _rar_err1:
                        print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

                    try:
                        # Three stacked rows (no shared meaning on the y
                        # axis -- just one row per category) instead of the
                        # time-dependent branch's x-t scatter, since a
                        # steady 1D point has no second coordinate at all.
                        _fig_c, _ax_c = plt.subplots(figsize=_plot_figsize(8, 3.2))
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 0], np.full(len(_rar_orig_pts), 2.0), s=10, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 0], np.full(len(_rar_added_so_far), 1.0), s=14, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 0], np.full(len(_rar_new_pts), 0.0), s=18, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_yticks([]); _ax_c.set_ylim(-0.6, 2.6)
                        _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize})
                        _ax_c.legend(loc="best", fontsize=_lfs(8))
                        _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                        plt.tight_layout()
                        plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                        plt.close(_fig_c)
                        print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
                    except Exception as _rar_err2:
                        print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")

                    if _rar_ea_entries:
                        try:
                            # Steady-state: at most one reference file
                            # really means anything (there's no time axis
                            # to pick "the closest snapshot" by), so just
                            # use the first one configured -- format is
                            # x, u (2 columns, no time column), matching
                            # the steady 2D/3D reference-file convention
                            # (spatial columns then u, nothing else).
                            _best_r = _rar_ea_entries[0]
                            _ref_fp_r = _best_r[1]
                            _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                            _ref_d_r = np.loadtxt(_ref_fp_r)
                            if _ref_d_r.ndim == 1:
                                _ref_d_r = _ref_d_r.reshape(1, -1)
                            _ref_x_r = _ref_d_r[:, 0]; _ref_u_r = _ref_d_r[:, 1]
                            _ref_pred_r = _rar_ref_extract(_ref_x_r.reshape(-1, 1), _ref_sel_r, model)
                            _err_r = _ref_pred_r - _ref_u_r
                            _l2_r = float(np.linalg.norm(_err_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                            _mse_r = float(np.mean(_err_r ** 2))
                            print(f"  [RAR round {{_rar_idx}}] vs Reference: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                            _fig_e, _ax_e = plt.subplots(figsize=_plot_figsize(6.5, 5))
                            _ord_e = np.argsort(_ref_x_r)
                            _ref_plot(_ax_e, _ref_x_r[_ord_e], _ref_u_r[_ord_e], label="Reference", color=_PLOT_REF_COLOR)
                            _ax_e.plot(_ref_x_r[_ord_e], _ref_pred_r[_ord_e], label="PINN", color=_PLOT_PINN_COLOR, linestyle="--")
                            _ax_e.legend(loc="best", fontsize=_lfs(8))
                            _ax_e.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_e.set_ylabel("u", fontsize={config.plot_axis_label_fontsize})
                            _ax_e.set_title(f"RAR round {{_rar_idx}} -- PINN vs Reference  ({{_l2_txt(_l2_r)}})")
                            plt.tight_layout()
                            plt.savefig(_os.path.join(_rd, "error_compare.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                            plt.close(_fig_e)
                        except Exception as _rar_err3:
                            print(f"  [RAR round {{_rar_idx}}] error compare failed: {{_rar_err3}}")
                    return

                # -- 1D: own two-file structure (solution snapshot +
                # line-comparison against one reference, if configured).
                # Axis orientation matches the main Plot Settings "Swap
                # axes" choice -- the same convention the Surface plot type
                # and every Error Analysis comparison plot already use
                # (t on the x-axis, x on the y-axis, by default).
                try:
                    if len(_rar_ea_entries) >= 2:
                        # 1D solution snapshot merged with reference -- same
                        # spirit as the 2D/3D branch below (PINN | Reference
                        # | |Error|, sharing one color scale), built as a
                        # full x-t surface interpolated across every
                        # configured reference file's own time (same
                        # technique the Standard path's own Inline Error
                        # Analysis "1D: standard x vs t surface" uses --
                        # needs >= 2 distinct time snapshots to form a
                        # non-degenerate (time x space) grid). With 0 or 1
                        # reference file, falls back to the original
                        # PINN-only snapshot below -- the single-reference
                        # case is already covered by error_compare.png's own
                        # line comparison.
                        _rar_x_common = np.linspace({config.x_min}, {config.x_max}, _res)
                        _rar_times_m = sorted(set(_e[0] for _e in _rar_ea_entries))
                        _rar_U_pinn = np.zeros((len(_rar_times_m), len(_rar_x_common)))
                        _rar_U_ref = np.zeros((len(_rar_times_m), len(_rar_x_common)))
                        for _rmi, _rmtv in enumerate(_rar_times_m):
                            _rmentry = next(_e for _e in _rar_ea_entries if _e[0] == _rmtv)
                            _rmfp = _rmentry[1]
                            _rmsel = _rmentry[2] if len(_rmentry) >= 3 else None
                            _rar_xt_c = np.column_stack([_rar_x_common, np.full_like(_rar_x_common, _rmtv)])
                            _rar_U_pinn[_rmi, :] = _rar_ref_extract(_rar_xt_c, _rmsel, model).flatten()
                            _rd_ld = np.loadtxt(_rmfp)
                            if _rd_ld.ndim == 1:
                                _rd_ld = _rd_ld.reshape(1, -1)
                            _rd_ord = np.argsort(_rd_ld[:, 0])
                            _rd_fi = _rar_interp1d(_rd_ld[_rd_ord, 0], _rd_ld[_rd_ord, 2], kind="linear", fill_value="extrapolate")
                            _rar_U_ref[_rmi, :] = _rd_fi(_rar_x_common)
                        _rar_Xg, _rar_Tg = np.meshgrid(_rar_x_common, _rar_times_m)
                        _rar_U_err = np.abs(_rar_U_pinn - _rar_U_ref)
                        _rar_vmin = min(_rar_U_pinn.min(), _rar_U_ref.min())
                        _rar_vmax = max(_rar_U_pinn.max(), _rar_U_ref.max())
                        if _rar_vmax - _rar_vmin < 1e-12:
                            _rar_vmax = _rar_vmin + 1e-12
                        _rar_vmin, _rar_vmax = _vr(_rar_vmin, _rar_vmax)   # Plot settings -> Surface -> Colour range
                        _rar_levels = _cf_levels({config.plot_levels}, _rar_vmin, _rar_vmax)["levels"]
                        _fig_r, _axes_r = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))
                        _rar_cols = [
                            (_rar_U_pinn, "PINN", _rar_levels, "{config.plot_colormap}"),
                            (_rar_U_ref, "Reference", _rar_levels, "{config.plot_colormap}"),
                            (_rar_U_err, f"|Error|  max = {{_sci_txt(_rar_U_err.max())}}", {config.plot_levels}, "{err_cmap(config)}"),
                        ]
                        for _rci, (_rvals, _rttl, _rlv, _rcmap) in enumerate(_rar_cols):
                            if {config.plot_swap_xt}:
                                _rim = _axes_r[_rci].contourf(_rar_Tg, _rar_Xg, _rvals, levels=_rlv, cmap=_rcmap, extend=_vext(_rlv))
                                _axes_r[_rci].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); _axes_r[_rci].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                            else:
                                _rim = _axes_r[_rci].contourf(_rar_Xg.T, _rar_Tg.T, _rvals.T, levels=_rlv, cmap=_rcmap, extend=_vext(_rlv))
                                _axes_r[_rci].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes_r[_rci].set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
                            _axes_r[_rci].set_title(_rttl, fontsize={config.plot_subplot_title_fontsize})
                            _fig_r.colorbar(_rim, ax=_axes_r[_rci])
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        print(f"  [RAR round {{_rar_idx}}] solution snapshot merged with reference ({{len(_rar_times_m)}} time(s))")
                    else:
                        _xr = np.linspace({config.x_min}, {config.x_max}, _res)
                        _tr = np.linspace({config.t_min}, {config.t_max}, _res)
                        _Xr, _Tr = np.meshgrid(_xr, _tr)
                        _field_r = _extract_plot_field(np.column_stack([_Xr.ravel(), _Tr.ravel()])).reshape(_res, _res)
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5))
                        if {config.plot_swap_xt}:
                            _im_r = _ax_r.contourf(_Tr, _Xr, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                            _ax_r.set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                        else:
                            _im_r = _ax_r.contourf(_Xr, _Tr, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                            _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
                        _fig_r.colorbar(_im_r, ax=_ax_r)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot")
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                    plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_r)
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
                except Exception as _rar_err1:
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

                try:
                    _fig_c, _ax_c = plt.subplots(figsize=_plot_figsize(6.5, 5))
                    if {config.plot_swap_xt}:
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 1], _rar_orig_pts[:, 0], s=6, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 1], _rar_added_so_far[:, 0], s=10, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 1], _rar_new_pts[:, 0], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                    else:
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], s=6, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], s=10, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
                    _ax_c.legend(loc="best", fontsize=_lfs(8))
                    _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                    plt.tight_layout()
                    plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_c)
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
                except Exception as _rar_err2:
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")

                if _rar_ea_entries:
                    try:
                        _best_r = min(_rar_ea_entries, key=lambda _e: abs(_e[0] - {config.t_max}))
                        _ref_t_r, _ref_fp_r = _best_r[0], _best_r[1]
                        _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                        _ref_d_r = np.loadtxt(_ref_fp_r)
                        if _ref_d_r.ndim == 1:
                            _ref_d_r = _ref_d_r.reshape(1, -1)
                        _ref_xyz_r = _ref_d_r[:, :1]; _ref_u_r = _ref_d_r[:, 2]
                        _ref_grid_r = np.column_stack([_ref_xyz_r, np.full(len(_ref_d_r), _ref_t_r)])
                        if _ref_sel_r is None:
                            _ref_pred_r = _extract_plot_field(_ref_grid_r)
                        elif isinstance(_ref_sel_r, int):
                            _ref_pred_r = model.predict(_ref_grid_r)[:, _ref_sel_r]
                        else:
                            _ref_expr_r = _ref_sel_r[0]
                            def _rar_ea_op(_rea_i, _rea_o):
                                _rea_dv = _tm_build_dvars(_rea_i, _rea_o, _plot_n_out, _plot_output_names_list,
                                                           _is_steady, _plot_dim, _ref_expr_r)
                                _rea_ns = dict(_rea_dv); _rea_ns.update(_PLOT_TORCH_MATH_NS); _rea_ns["torch"] = torch
                                return eval(_ref_expr_r, _rea_ns)
                            _ref_pred_r = model.predict(_ref_grid_r, operator=_rar_ea_op)[:, 0]
                        _err_r = _ref_pred_r - _ref_u_r
                        _l2_r = float(np.linalg.norm(_err_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                        _mse_r = float(np.mean(_err_r ** 2))
                        print(f"  [RAR round {{_rar_idx}}] vs Reference t={{_ref_t_r:.3g}}: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                        _fig_e, _ax_e = plt.subplots(figsize=_plot_figsize(6.5, 5))
                        _ord_e = np.argsort(_ref_xyz_r[:, 0])
                        _ref_plot(_ax_e, _ref_xyz_r[_ord_e, 0], _ref_u_r[_ord_e], label="Reference", color=_PLOT_REF_COLOR)
                        _ax_e.plot(_ref_xyz_r[_ord_e, 0], _ref_pred_r[_ord_e], label="PINN", color=_PLOT_PINN_COLOR, linestyle="--")
                        _ax_e.legend(loc="best", fontsize=_lfs(8))
                        _ax_e.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_e.set_ylabel("u", fontsize={config.plot_axis_label_fontsize})
                        _ax_e.set_title(f"RAR round {{_rar_idx}} -- PINN vs Reference  t={{_fmt_num(_ref_t_r)}} ({{_l2_txt(_l2_r)}})")
                        plt.tight_layout()
                        plt.savefig(_os.path.join(_rd, "error_compare.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                        plt.close(_fig_e)
                    except Exception as _rar_err3:
                        print(f"  [RAR round {{_rar_idx}}] error compare failed: {{_rar_err3}}")
                return

            if _is_steady:
                # Steady-state 2D/3D (e.g. a Poisson equation): PINN |
                # Reference | |Error| when a reference file is configured,
                # just the PINN prediction alone otherwise -- no time
                # slicing anywhere (there's no t_max to evaluate "the
                # final time" at, unlike the time-dependent branch below).
                try:
                    _ref_ok_r = False
                    if _rar_ea_entries:
                        try:
                            # Steady-state: at most one reference file
                            # really means anything (no time axis to pick
                            # "the closest snapshot" by) -- format is
                            # x,y,u (2D) / x,y,z,u (3D), no time column.
                            _best_r = _rar_ea_entries[0]
                            _ref_fp_r = _best_r[1]
                            _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                            _ref_d_r = np.loadtxt(_ref_fp_r)
                            if _ref_d_r.ndim == 1:
                                _ref_d_r = _ref_d_r.reshape(1, -1)
                            if _is_3d:
                                _ref_xyz_r = _ref_d_r[:, :3]; _ref_u_r = _ref_d_r[:, 3]
                            else:
                                _ref_xyz_r = _ref_d_r[:, :2]; _ref_u_r = _ref_d_r[:, 2]
                            _ref_pred_pts_r = _rar_ref_extract(_ref_xyz_r, _ref_sel_r, model)
                            _err_pts_r = _ref_pred_pts_r - _ref_u_r
                            _l2_r = float(np.linalg.norm(_err_pts_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                            _mse_r = float(np.mean(_err_pts_r ** 2))
                            print(f"  [RAR round {{_rar_idx}}] vs Reference: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                            _ref_ok_r = True
                        except Exception as _rar_ref_err:
                            print(f"  [RAR round {{_rar_idx}}] reference compare failed, showing prediction only: {{_rar_ref_err}}")
                            _ref_ok_r = False

                    if _is_3d:
                        if _ref_ok_r:
                            _fig_r = plt.figure(figsize=_plot_figsize(15, 5))
                            _abs_err_r = np.abs(_err_pts_r)
                            _vmin3_r = min(_ref_pred_pts_r.min(), _ref_u_r.min())
                            _vmax3_r = max(_ref_pred_pts_r.max(), _ref_u_r.max())
                            _cols3_r = [
                                (_ref_pred_pts_r, f"PINN  {{_l2_txt(_l2_r)}}", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                                (_ref_u_r, "Reference", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                                (_abs_err_r, f"|Error|  max = {{_sci_txt(_abs_err_r.max())}}", None, None, "{err_cmap(config)}"),
                            ]
                            for _ci_r, (_vals3_r, _ttl3_r, _vmin_c3_r, _vmax_c3_r, _cmap_c3_r) in enumerate(_cols3_r):
                                _ax3_r = _fig_r.add_subplot(1, 3, _ci_r + 1, projection="3d")
                                _sc3_r = _ax3_r.scatter(_ref_xyz_r[:, 0], _ref_xyz_r[:, 1], _ref_xyz_r[:, 2], c=_vals3_r,
                                                         cmap=_cmap_c3_r, s=10, vmin=_vmin_c3_r, vmax=_vmax_c3_r)
                                _fig_r.colorbar(_sc3_r, ax=_ax3_r, shrink=0.6, pad=0.12)
                                _ax3_r.set_title(_ttl3_r, fontsize={config.plot_subplot_title_fontsize})
                                _ax3_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax3_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax3_r.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax3_r.xaxis, _ax3_r.yaxis, _ax3_r.zaxis)]
                            _fig_r.suptitle(f"RAR round {{_rar_idx}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        else:
                            _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                            _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                            _Xg, _Yg = np.meshgrid(_xp, _yp)
                            _z_mid_r = {auto_line_slice(config)[1]!r}   # Plot settings -> Line: z
                            _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _z_mid_r)])
                            _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                            _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                            _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                            _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_aspect("equal", adjustable="box")
                            _fig_r.colorbar(_im_r, ax=_ax_r, fraction=0.046, pad=0.04)
                            _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot (z={{_fmt_num(_z_mid_r)}})")
                    else:
                        _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                        _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                        _Xg, _Yg = np.meshgrid(_xp, _yp)
                        _inside_r = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res, _res)
                        if _ref_ok_r:
                            from scipy.interpolate import griddata as _gd_r
                            _grid_pinn_r = np.column_stack([_Xg.ravel(), _Yg.ravel()])
                            _u_pinn_g_r = _extract_plot_field(_grid_pinn_r).reshape(_res, _res)
                            _u_pinn_g_r = np.where(_inside_r, _u_pinn_g_r, np.nan)
                            _u_ref_g_r = _gd_r(_ref_xyz_r, _ref_u_r, (_Xg, _Yg), method="linear", fill_value=0.0)
                            _u_ref_g_r = np.where(_inside_r, _u_ref_g_r, np.nan)
                            _u_err_g_r = np.abs(_u_pinn_g_r - _u_ref_g_r)
                            _vmin2_r = np.nanmin([_u_pinn_g_r, _u_ref_g_r]); _vmax2_r = np.nanmax([_u_pinn_g_r, _u_ref_g_r])
                            if _vmax2_r - _vmin2_r < 1e-12:
                                _vmax2_r = _vmin2_r + 1e-12
                            _vmin2_r, _vmax2_r = _vr(_vmin2_r, _vmax2_r)   # Plot settings -> Surface -> Colour range
                            _levels2_r = _cf_levels({config.plot_levels}, _vmin2_r, _vmax2_r)["levels"]
                            _fig_r, _axes2_r = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))
                            _im0_r = _axes2_r[0].contourf(_Xg, _Yg, _u_pinn_g_r, levels=_levels2_r, cmap="{config.plot_colormap}", extend=_vext())
                            _axes2_r[0].set_title(f"PINN  {{_l2_txt(_l2_r)}}", fontsize={config.plot_subplot_title_fontsize})
                            _axes2_r[0].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[0].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[0].set_aspect("equal", adjustable="box")
                            _fig_r.colorbar(_im0_r, ax=_axes2_r[0], fraction=0.046, pad=0.04)
                            _im1_r = _axes2_r[1].contourf(_Xg, _Yg, _u_ref_g_r, levels=_levels2_r, cmap="{config.plot_colormap}", extend=_vext())
                            _axes2_r[1].set_title("Reference", fontsize={config.plot_subplot_title_fontsize})
                            _axes2_r[1].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[1].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[1].set_aspect("equal", adjustable="box")
                            _fig_r.colorbar(_im1_r, ax=_axes2_r[1], fraction=0.046, pad=0.04)
                            _im2_r = _axes2_r[2].contourf(_Xg, _Yg, _u_err_g_r, levels={config.plot_levels}, cmap="{err_cmap(config)}")
                            _axes2_r[2].set_title(f"|Error|  max = {{_sci_txt(np.nanmax(_u_err_g_r))}}", fontsize={config.plot_subplot_title_fontsize})
                            _axes2_r[2].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[2].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[2].set_aspect("equal", adjustable="box")
                            _fig_r.colorbar(_im2_r, ax=_axes2_r[2], fraction=0.046, pad=0.04)
                            _fig_r.suptitle(f"RAR round {{_rar_idx}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        else:
                            _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel()])
                            _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                            _field_r = np.where(_inside_r, _field_r, np.nan)
                            _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                            _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                            _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_aspect("equal", adjustable="box")
                            _fig_r.colorbar(_im_r, ax=_ax_r, fraction=0.046, pad=0.04)
                            _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot")
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                    plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_r)
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
                except Exception as _rar_err1:
                    print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

                # -- collocation points (steady: no time dimension at all,
                # so no alpha-for-time trick in 3D and no second "spatial-
                # only" side panel in 2D -- just the points themselves) --
                try:
                    if _is_3d:
                        from mpl_toolkits.mplot3d import Axes3D as _Axes3D_unused2  # noqa: F401 -- registers the 3D projection
                        _fig_c = plt.figure(figsize=_plot_figsize(7, 6))
                        _ax_c = _fig_c.add_subplot(111, projection="3d")
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], _rar_orig_pts[:, 2], s=5, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], _rar_added_so_far[:, 2], s=9, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], _rar_new_pts[:, 2], s=12, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax_c.xaxis, _ax_c.yaxis, _ax_c.zaxis)]
                        _ax_c.legend(loc="best", fontsize=_lfs(8))
                        _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                        plt.tight_layout()
                    else:
                        _fig_c, _ax_c = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                        if len(_rar_orig_pts):
                            _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], s=6, c="#adb5bd", alpha=0.6, label="Original points")
                        if len(_rar_added_so_far):
                            _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], s=10, c="#fd7e14", label="Added in earlier rounds")
                        _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                        _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_aspect("equal", adjustable="box")
                        _ax_c.legend(loc="best", fontsize=_lfs(8))
                        _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                        plt.tight_layout()
                    plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close(_fig_c)
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
                except Exception as _rar_err2:
                    print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")
                return

            # -- 2D/3D: solution snapshot merged with the error comparison --
            # One single file: PINN | Reference | |Error| -- all three
            # panels now share the SAME colormap (config.plot_colormap),
            # standardized in Round 31 (previously |Error| hardcoded its
            # own "inferno" colormap here, inconsistent with the other
            # two panels) -- whenever a reference file is configured;
            # just the PINN prediction alone, same as before, when it isn't.
            try:
                _ref_ok_r = False
                _ref_t_r = {config.t_max}
                if _rar_ea_entries:
                    try:
                        _best_r = min(_rar_ea_entries, key=lambda _e: abs(_e[0] - {config.t_max}))
                        _ref_t_r, _ref_fp_r = _best_r[0], _best_r[1]
                        _ref_sel_r = _best_r[2] if len(_best_r) >= 3 else None
                        _ref_d_r = np.loadtxt(_ref_fp_r)
                        if _ref_d_r.ndim == 1:
                            _ref_d_r = _ref_d_r.reshape(1, -1)
                        if _is_3d:
                            _ref_xyz_r = _ref_d_r[:, :3]; _ref_u_r = _ref_d_r[:, 4]
                        else:
                            _ref_xyz_r = _ref_d_r[:, :2]; _ref_u_r = _ref_d_r[:, 3]
                        _ref_grid_r = np.column_stack([_ref_xyz_r, np.full(len(_ref_d_r), _ref_t_r)])
                        if _ref_sel_r is None:
                            _ref_pred_pts_r = _extract_plot_field(_ref_grid_r)
                        elif isinstance(_ref_sel_r, int):
                            _ref_pred_pts_r = model.predict(_ref_grid_r)[:, _ref_sel_r]
                        else:
                            _ref_expr_r = _ref_sel_r[0]
                            def _rar_ea_op2(_rea_i, _rea_o):
                                _rea_dv = _tm_build_dvars(_rea_i, _rea_o, _plot_n_out, _plot_output_names_list,
                                                           _is_steady, _plot_dim, _ref_expr_r)
                                _rea_ns = dict(_rea_dv); _rea_ns.update(_PLOT_TORCH_MATH_NS); _rea_ns["torch"] = torch
                                return eval(_ref_expr_r, _rea_ns)
                            _ref_pred_pts_r = model.predict(_ref_grid_r, operator=_rar_ea_op2)[:, 0]
                        _err_pts_r = _ref_pred_pts_r - _ref_u_r
                        _l2_r = float(np.linalg.norm(_err_pts_r) / (np.linalg.norm(_ref_u_r) + 1e-12))
                        _mse_r = float(np.mean(_err_pts_r ** 2))
                        print(f"  [RAR round {{_rar_idx}}] vs Reference t={{_ref_t_r:.3g}}: L2 rel error = {{_l2_r:.4e}}, MSE = {{_mse_r:.4e}}")
                        _ref_ok_r = True
                    except Exception as _rar_ref_err:
                        print(f"  [RAR round {{_rar_idx}}] reference compare failed, showing prediction only: {{_rar_ref_err}}")
                        _ref_ok_r = False

                if _is_3d:
                    if _ref_ok_r:
                        _fig_r = plt.figure(figsize=_plot_figsize(15, 5))
                        _abs_err_r = np.abs(_err_pts_r)
                        _vmin3_r = min(_ref_pred_pts_r.min(), _ref_u_r.min())
                        _vmax3_r = max(_ref_pred_pts_r.max(), _ref_u_r.max())
                        _cols3_r = [
                            (_ref_pred_pts_r, f"PINN  t={{_fmt_num(_ref_t_r)}}  {{_l2_txt(_l2_r)}}", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                            (_ref_u_r, f"Reference  t={{_ref_t_r:.3g}}", _vmin3_r, _vmax3_r, "{config.plot_colormap}"),
                            (_abs_err_r, f"|Error|  max = {{_sci_txt(_abs_err_r.max())}}", None, None, "{err_cmap(config)}"),
                        ]
                        for _ci_r, (_vals3_r, _ttl3_r, _vmin_c3_r, _vmax_c3_r, _cmap_c3_r) in enumerate(_cols3_r):
                            _ax3_r = _fig_r.add_subplot(1, 3, _ci_r + 1, projection="3d")
                            _sc3_r = _ax3_r.scatter(_ref_xyz_r[:, 0], _ref_xyz_r[:, 1], _ref_xyz_r[:, 2], c=_vals3_r,
                                                     cmap=_cmap_c3_r, s=10, vmin=_vmin_c3_r, vmax=_vmax_c3_r)
                            _fig_r.colorbar(_sc3_r, ax=_ax3_r, shrink=0.6, pad=0.12)
                            _ax3_r.set_title(_ttl3_r, fontsize={config.plot_subplot_title_fontsize})
                            _ax3_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax3_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax3_r.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax3_r.xaxis, _ax3_r.yaxis, _ax3_r.zaxis)]
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                    else:
                        _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                        _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                        _Xg, _Yg = np.meshgrid(_xp, _yp)
                        _z_mid_r = {auto_line_slice(config)[1]!r}   # Plot settings -> Line: z
                        _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _z_mid_r), np.full(_Xg.size, {config.t_max})])
                        _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                        _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im_r, ax=_ax_r, fraction=0.046, pad=0.04)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot (z={{_fmt_num(_z_mid_r)}}, t={{_fmt_num({config.t_max!r})}})")
                else:
                    _xp = np.linspace({config.x_min}, {config.x_max}, _res)
                    _yp = np.linspace({config.y_min}, {config.y_max}, _res)
                    _Xg, _Yg = np.meshgrid(_xp, _yp)
                    _inside_r = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res, _res)
                    if _ref_ok_r:
                        from scipy.interpolate import griddata as _gd_r
                        _grid_pinn_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _ref_t_r)])
                        _u_pinn_g_r = _extract_plot_field(_grid_pinn_r).reshape(_res, _res)
                        _u_pinn_g_r = np.where(_inside_r, _u_pinn_g_r, np.nan)
                        _u_ref_g_r = _gd_r(_ref_xyz_r, _ref_u_r, (_Xg, _Yg), method="linear", fill_value=0.0)
                        _u_ref_g_r = np.where(_inside_r, _u_ref_g_r, np.nan)
                        _u_err_g_r = np.abs(_u_pinn_g_r - _u_ref_g_r)
                        _vmin2_r = np.nanmin([_u_pinn_g_r, _u_ref_g_r]); _vmax2_r = np.nanmax([_u_pinn_g_r, _u_ref_g_r])
                        if _vmax2_r - _vmin2_r < 1e-12:
                            _vmax2_r = _vmin2_r + 1e-12
                        _vmin2_r, _vmax2_r = _vr(_vmin2_r, _vmax2_r)   # Plot settings -> Surface -> Colour range
                        # contourf's own vmin/vmax kwargs do NOT restrict
                        # an integer `levels=N` -- it still auto-ranges
                        # each panel to that panel's own data, so PINN and
                        # Reference would silently get two different color
                        # scales despite passing the same vmin/vmax here.
                        # Passing the level BOUNDARIES explicitly (instead
                        # of a level count) is what actually makes both
                        # panels share one scale.
                        _levels2_r = _cf_levels({config.plot_levels}, _vmin2_r, _vmax2_r)["levels"]
                        _fig_r, _axes2_r = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))
                        _im0_r = _axes2_r[0].contourf(_Xg, _Yg, _u_pinn_g_r, levels=_levels2_r, cmap="{config.plot_colormap}", extend=_vext())
                        _axes2_r[0].set_title(f"PINN  t={{_fmt_num(_ref_t_r)}}  {{_l2_txt(_l2_r)}}", fontsize={config.plot_subplot_title_fontsize})
                        _axes2_r[0].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[0].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[0].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im0_r, ax=_axes2_r[0], fraction=0.046, pad=0.04)
                        _im1_r = _axes2_r[1].contourf(_Xg, _Yg, _u_ref_g_r, levels=_levels2_r, cmap="{config.plot_colormap}", extend=_vext())
                        _axes2_r[1].set_title(f"Reference  t={{_fmt_num(_ref_t_r)}}", fontsize={config.plot_subplot_title_fontsize})
                        _axes2_r[1].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[1].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[1].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im1_r, ax=_axes2_r[1], fraction=0.046, pad=0.04)
                        _im2_r = _axes2_r[2].contourf(_Xg, _Yg, _u_err_g_r, levels={config.plot_levels}, cmap="{err_cmap(config)}")
                        _axes2_r[2].set_title(f"|Error|  max = {{_sci_txt(np.nanmax(_u_err_g_r))}}", fontsize={config.plot_subplot_title_fontsize})
                        _axes2_r[2].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _axes2_r[2].set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _axes2_r[2].set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im2_r, ax=_axes2_r[2], fraction=0.046, pad=0.04)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                    else:
                        _grid_r = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, {config.t_max})])
                        _field_r = _extract_plot_field(_grid_r).reshape(_res, _res)
                        _field_r = np.where(_inside_r, _field_r, np.nan)
                        _fig_r, _ax_r = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                        _im_r = _ax_r.contourf(_Xg, _Yg, _field_r, levels={config.plot_levels}, cmap="{config.plot_colormap}")
                        _ax_r.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_r.set_aspect("equal", adjustable="box")
                        _fig_r.colorbar(_im_r, ax=_ax_r, fraction=0.046, pad=0.04)
                        _fig_r.suptitle(f"RAR round {{_rar_idx}} solution snapshot (t={{_fmt_num({config.t_max!r})}})")
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(_os.path.join(_rd, "solution_plot.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                plt.close(_fig_r)
                print(f"  [RAR round {{_rar_idx}}] solution snapshot saved: {{_os.path.join(_rd, 'solution_plot.png')}}")
            except Exception as _rar_err1:
                print(f"  [RAR round {{_rar_idx}}] solution snapshot failed: {{_rar_err1}}")

            # -- 2D/3D: collocation points (original vs. earlier rounds vs. this round) --
            try:
                from mpl_toolkits.mplot3d import Axes3D as _Axes3D_unused  # noqa: F401 -- registers the 3D projection
                if _is_3d:
                    _fig_c = plt.figure(figsize=_plot_figsize(7, 6))
                    _ax_c = _fig_c.add_subplot(111, projection="3d")
                    # 4 real coordinate dims (x,y,z,t) can't all be axes on
                    # one static 3D plot -- spatial (x,y,z) get the 3 plot
                    # axes, and t is folded into per-category alpha instead
                    # (lighter = earlier in the time domain) rather than
                    # dropped entirely.
                    _t_lo_r, _t_hi_r = {config.t_min}, {config.t_max}
                    _span_t_r = (_t_hi_r - _t_lo_r) or 1.0
                    def _rar_alpha(_pts):
                        return float(np.clip(0.25 + 0.65 * (np.mean(_pts[:, 3]) - _t_lo_r) / _span_t_r, 0.15, 0.95)) if len(_pts) else 1.0
                    _orig_alpha, _added_alpha = _rar_alpha(_rar_orig_pts), _rar_alpha(_rar_added_so_far)
                    if len(_rar_orig_pts):
                        _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], _rar_orig_pts[:, 2], s=5, c="#adb5bd", alpha=_orig_alpha, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], _rar_added_so_far[:, 2], s=9, c="#fd7e14", alpha=_added_alpha, label="Added in earlier rounds")
                    _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], _rar_new_pts[:, 2], s=12, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax_c.xaxis, _ax_c.yaxis, _ax_c.zaxis)]
                    _ax_c.legend(loc="best", fontsize=_lfs(8))
                    _ax_c.set_title(f"RAR round {{_rar_idx}} -- collocation points")
                    plt.tight_layout()
                else:
                    # 2D: the existing 3D (x, y, t) view, plus a plain 2D
                    # (x, y) view collapsing time -- side by side -- so the
                    # spatial concentration of refinement is readable at a
                    # glance without having to rotate a 3D plot.
                    _fig_c = plt.figure(figsize=_plot_figsize(13, 6))
                    _ax_c = _fig_c.add_subplot(1, 2, 1, projection="3d")
                    if len(_rar_orig_pts):
                        _ax_c.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], _rar_orig_pts[:, 2], s=5, c="#adb5bd", alpha=0.5, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], _rar_added_so_far[:, 2], s=9, c="#fd7e14", alpha=0.85, label="Added in earlier rounds")
                    _ax_c.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], _rar_new_pts[:, 2], s=12, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_c.set_zlabel("t", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax_c.xaxis, _ax_c.yaxis, _ax_c.zaxis)]
                    _ax_c.legend(loc="best", fontsize=_lfs(8))
                    _ax_c.set_title("3D view (x, y, t)")
                    _ax_c2 = _fig_c.add_subplot(1, 2, 2)
                    if len(_rar_orig_pts):
                        _ax_c2.scatter(_rar_orig_pts[:, 0], _rar_orig_pts[:, 1], s=6, c="#adb5bd", alpha=0.5, label="Original points")
                    if len(_rar_added_so_far):
                        _ax_c2.scatter(_rar_added_so_far[:, 0], _rar_added_so_far[:, 1], s=10, c="#fd7e14", alpha=0.85, label="Added in earlier rounds")
                    _ax_c2.scatter(_rar_new_pts[:, 0], _rar_new_pts[:, 1], s=14, c="#e03131", label=f"Added this round ({{len(_rar_new_pts)}})")
                    _ax_c2.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_c2.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax_c2.set_aspect("equal", adjustable="box")
                    _ax_c2.legend(loc="best", fontsize=_lfs(8))
                    _ax_c2.set_title("Spatial view (x, y) -- all times")
                    _fig_c.suptitle(f"RAR round {{_rar_idx}} -- collocation points", fontsize={config.plot_title_fontsize}, fontweight="bold")
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(_os.path.join(_rd, "collocation_points.png"), dpi={config.plot_dpi}, bbox_inches="tight")
                plt.close(_fig_c)
                print(f"  [RAR round {{_rar_idx}}] collocation-points plot saved: {{_os.path.join(_rd, 'collocation_points.png')}}")
            except Exception as _rar_err2:
                print(f"  [RAR round {{_rar_idx}}] collocation-points plot failed: {{_rar_err2}}")

        print("\\n=== Starting RAR Adaptive Refinement ===")
        for rar_cycle in range({config.rar_cycles}):
            print(f"\\n--- RAR Cycle {{rar_cycle+1}}/{config.rar_cycles} ---")
            # Candidate points come straight from the real geometry's own
            # random_points() -- geomtime (an alias for geom when the
            # problem is steady, see the plain dde.data.PDE built above;
            # otherwise a GeometryXTime wrapping geom and the time domain)
            # -- instead of independently uniform-sampling each axis over
            # the domain's bounding box, which used to be the case here.
            # The two agree for a Rectangle/Cuboid/Interval domain, but for
            # any non-box geometry (L-Shape, Disk, Polygon, Triangle, an
            # Annulus or other CSG shape with a hole, ...) the bounding box
            # is strictly larger than the real domain, so bounding-box
            # sampling put candidate points -- and then, since RAR just
            # keeps whichever candidates have the largest predicted PDE
            # residual, sometimes the actual added points too -- outside
            # the geometry entirely (e.g. inside the L-Shape's missing
            # quadrant), where the model's prediction is meaningless.
            # geom/geomtime's own random_points() is exactly what
            # dde.data.PDE/TimePDE already use to generate ordinary
            # training collocation points elsewhere in this same script,
            # and it performs real rejection sampling against the
            # geometry for every non-box shape (see e.g.
            # CSGDifference.random_points in DeepXDE's own geometry/csg.py),
            # so RAR's candidates are now guaranteed to lie inside the
            # actual domain for any geometry, not just a rectangular one.
            xt_cand = geomtime.random_points({config.rar_candidates})
            _rar_res = model.predict(xt_cand, operator=pde)
            if isinstance(_rar_res, list):
                # Restrict point selection to one equation's residual
                # (config.rar_output_selector >= 0 -- "Points from:" in the
                # RAR settings panel; equation order == output order for
                # every current template) instead of the combined sum --
                # -1 (the default) always falls through to the original,
                # unchanged combined behavior.
                if 0 <= {config.rar_output_selector} < len(_rar_res):
                    residuals = np.abs(_rar_res[{config.rar_output_selector}]).flatten()
                else:
                    residuals = np.sum([np.abs(r).flatten() for r in _rar_res], axis=0)
            else:
                residuals = np.abs(_rar_res).flatten(){_rar_tolerance_check(config, "            ")}
            top_idx    = np.argsort(residuals)[-{config.rar_add_points}:]
            new_points = xt_cand[top_idx]
            print(f"Max residual: {{residuals.max():.4e}}, Mean: {{residuals.mean():.4e}}")
            data.add_anchors(new_points)
            model.compile("{config.optimizer}", lr=_lr, loss="{config.loss_type}", loss_weights=_multi_weights)
            loss_history, train_state = model.train(iterations={config.rar_adam_iters}, display_every={config.loss_display_every})
            if {config.rar_lbfgs_iters} > 0:
                # Round 31: when this round also runs an L-BFGS phase, save
                # the Adam phase's own checkpoint here -- standardized on
                # "one checkpoint per optimizer phase" everywhere, matching
                # Time-Adaptive's own per-step convention (previously RAR
                # only ever saved ONE checkpoint per round, representing
                # whatever state the model was in after ALL of that
                # round's phases, so an Adam-phase-only snapshot was never
                # recoverable once L-BFGS started). The round's final
                # (L-BFGS) state is saved separately at the end of this
                # round via _rar_save_round_diagnostics, as "model_lbfgs".
                if _use_save:
                    _rar_rd_adam = _os.path.join(_rar_rounds_dir, f"round_{{rar_cycle+1:02d}}")
                    _os.makedirs(_rar_rd_adam, exist_ok=True)
                    try:
                        model.save(_os.path.join(_rar_rd_adam, "model_adam"))
                        print(f"  [RAR round {{rar_cycle+1}}] model_adam model saved: {{_os.path.join(_rar_rd_adam, 'model_adam')}}.pt")
                    except Exception as _rar_adam_save_err:
                        print(f"  [RAR round {{rar_cycle+1}}] model_adam save failed: {{_rar_adam_save_err}}")
                dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.rar_lbfgs_iters},
                    maxfun={_lbfgs_maxfun(config, config.rar_lbfgs_iters)}, maxls={config.lbfgs_maxls}
                )
                model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_multi_weights)
                loss_history, train_state = model.train(display_every={config.loss_display_every})
            _rar_save_round_diagnostics(rar_cycle + 1, new_points)
            _rar_added_so_far = np.vstack([_rar_added_so_far, new_points])
        print("\\n=== RAR Complete ===")"""


def _gs_ea_setup_code(config, banner, indent):
    """generate_script()'s Error Analysis setup, shared by the Standard and
    Time-Adaptive paths (cleanup Round 58 -- these were two hand-kept
    copies): normalize `ea_files` entries to (time, path, output_selector)
    3-tuples, keep the ones inside [t_min, t_max], group them by output,
    and define _ea_group_label / _ea_extract.

    output_selector is None (the Results panel's own plot field, via
    _extract_plot_field), an int (a raw output column), or an [expr, label]
    pair (a derivative-aware custom field, evaluated through Training
    Monitors' _tm_build_dvars). Older configs only wrote 2-tuples; those
    are padded with None. One group means every output filename stays
    unsuffixed. _ea_extract's model argument is optional: the Standard
    path falls back to the module-level `model`; Time-Adaptive always
    passes each step's restored model."""
    code = f'''_ea_files_norm = []
for _ea_entry in {config.ea_files}:
    if len(_ea_entry) >= 3:
        _ea_tv0, _ea_fp0, _ea_sel0 = _ea_entry[0], _ea_entry[1], _ea_entry[2]
    else:
        _ea_tv0, _ea_fp0 = _ea_entry[0], _ea_entry[1]
        _ea_sel0 = None
    if {config.t_min} - 1e-10 <= _ea_tv0 <= {config.t_max} + 1e-10:
        _ea_files_norm.append((_ea_tv0, _ea_fp0, _ea_sel0))
print("\\n=== Running {banner} ===")
print(f"  Filtering to t=[{config.t_min}, {config.t_max}]: {{len(_ea_files_norm)}} files")

_ea_groups = {{}}
for _ea_tv0, _ea_fp0, _ea_sel0 in _ea_files_norm:
    _ea_gkey = repr(_ea_sel0)
    if _ea_gkey not in _ea_groups:
        _ea_groups[_ea_gkey] = {{"sel": _ea_sel0, "files": []}}
    _ea_groups[_ea_gkey]["files"].append((_ea_tv0, _ea_fp0))
_ea_multi_output = len(_ea_groups) > 1

def _ea_group_label(_ea_sel):
    if _ea_sel is None:
        return ""
    if isinstance(_ea_sel, int):
        return (_plot_output_names_list[_ea_sel].strip()
                if 0 <= _ea_sel < len(_plot_output_names_list) else f"out{{_ea_sel}}")
    _ea_lbl = (_ea_sel[1] or "").strip() if len(_ea_sel) > 1 else ""
    return _ea_lbl if _ea_lbl else "custom"

def _ea_title_txt(_kind, _ea_sel):
    # "PINN vs Reference — ux — 2D Heatmaps": the compared field's name in
    # every Error Analysis title when there is more than one output.
    _lbl = _ea_group_label(_ea_sel)
    if not _lbl and len(_plot_output_names_list) > 1:
        _g = globals()
        _lbl = (_g.get("_plot_custom_label", "") if str(_g.get("_plot_custom_expr", "")).strip()
                else _plot_output_names_list[int(_g.get("_plot_idx", 0))].strip())
    return f"PINN vs Reference — {{_lbl}} — {{_kind}}" if _lbl else f"PINN vs Reference — {{_kind}}"

def _ea_field_txt(_ea_sel):
    # The compared field's name for panel titles ("PINN  mu(x,t)"), the
    # first output's name when nothing else is selected.
    _lbl = _ea_group_label(_ea_sel)
    if not _lbl:
        _g = globals()
        if str(_g.get("_plot_custom_expr", "")).strip():
            _lbl = str(_g.get("_plot_custom_label", "")).strip()
        elif _plot_output_names_list:
            _i = int(_g.get("_plot_idx", 0))
            _lbl = _plot_output_names_list[_i if 0 <= _i < len(_plot_output_names_list) else 0].strip()
    return _lbl or "u"

def _ea_extract(_ea_grid, _ea_sel, _ea_model=None):
    _ea_m = _ea_model if _ea_model is not None else model
    if _ea_sel is None:
        return _extract_plot_field(_ea_grid, _ea_m)
    if isinstance(_ea_sel, int):
        return _ea_m.predict(_ea_grid)[:, _ea_sel]
    _ea_expr = _ea_sel[0]

    def _ea_custom_op(_ea_inputs, _ea_outputs):
        _ea_dvars = _tm_build_dvars(_ea_inputs, _ea_outputs, _plot_n_out,
                                     _plot_output_names_list, _is_steady, _plot_dim,
                                     _ea_expr)
        _ea_ns = dict(_ea_dvars)
        _ea_ns.update(_PLOT_TORCH_MATH_NS)
        _ea_ns["torch"] = torch
        return eval(_ea_expr, _ea_ns)

    return _ea_m.predict(_ea_grid, operator=_ea_custom_op)[:, 0]'''
    return "\n".join((indent + ln) if ln else ln for ln in code.split("\n"))


def _gs_time_adaptive_loop_code(_ic_pretrain_block_ta, line_slice_y, line_slice_z, _safe_num_test, _ta_ic_init, _train_cbs_code_ta, config, plot_custom_expr_converted, _safe_point_distribution="Hammersley"):
    """generate_script()'s Time-Adaptive training loop (per-step models, IC
    pre-training, per-step previews, the combined final plot and the
    Time-Adaptive Error Analysis) -- cleanup Round 52 (audit §5): moved
    verbatim out of generate_script()'s template, where it was the largest
    unnamed inline block (~1,600 lines). Returns the exact same text the
    template used to contain (byte-for-byte; see the round's notes)."""
    return f"""# ── Time Adaptive Loop ────────────────────────────────────────
if {config.time_adaptive}:
    print("\\n=== Starting Time-Adaptive Training ===")

    # Clear old step directories to avoid stale models from previous runs
    import shutil as _ta_shutil
    _ta_steps_root = _os.path.join(_save_dir, "time_adaptive_steps")
    if _os.path.isdir(_ta_steps_root):
        _ta_shutil.rmtree(_ta_steps_root)
        print(f"  Cleared old time_adaptive_steps directory")
    _os.makedirs(_ta_steps_root, exist_ok=True)

    grid_size = {config.ta_grid_size}
    if _is_3d:
        # grid_size is a per-axis resolution, so this cubes it (e.g. 51 ->
        # ~132k points) -- much bigger than the 2D case's square. Left to
        # the user's own ta_grid_size choice (the GUI's smallest preset,
        # 11, keeps this to ~1.3k points) rather than silently overriding
        # it here.
        _xg_ta = np.linspace({config.x_min}, {config.x_max}, grid_size)
        _yg_ta = np.linspace({config.y_min}, {config.y_max}, grid_size)
        _zg_ta = np.linspace({config.z_min}, {config.z_max}, grid_size)
        _Xmesh, _Ymesh, _Zmesh = np.meshgrid(_xg_ta, _yg_ta, _zg_ta)
        x_grid = np.column_stack([_Xmesh.ravel(), _Ymesh.ravel(), _Zmesh.ravel()])
        x = x_grid  # (N*N*N, 3)
    elif _is_2d:
        _xg_ta = np.linspace({config.x_min}, {config.x_max}, grid_size)
        _yg_ta = np.linspace({config.y_min}, {config.y_max}, grid_size)
        _Xmesh, _Ymesh = np.meshgrid(_xg_ta, _yg_ta)
        x_grid = np.column_stack([_Xmesh.ravel(), _Ymesh.ravel()])
        x = x_grid  # (N*N, 2)
    else:
        x_grid = np.linspace({config.x_min}, {config.x_max}, grid_size)

    all_x = []; all_t = []; all_u = []

    if _is_2d or _is_3d:
        x = x_grid
    else:
        x = x_grid.reshape(-1, 1)
    {_ta_ic_init}
    _lr = {config.learning_rate}
    _prev_step_model_path = ""
    _prev_step_state = None

    # ── Build flat interval list from step groups ─────────────
    import json as _json_ta_groups
    _ta_groups = _json_ta_groups.loads({repr(config.ta_step_groups) if config.ta_step_groups else repr('[{"t_start":' + str(config.t_min) + ',"t_end":' + str(config.t_max) + ',"steps":' + str(config.ta_num_steps) + '}]')})
    _ta_flat_intervals = []
    for _grp in _ta_groups:
        _grp_dt = (_grp['t_end'] - _grp['t_start']) / _grp['steps']
        for _gi in range(_grp['steps']):
            _t0_g = _grp['t_start'] + _gi * _grp_dt
            _t1_g = _t0_g + _grp_dt
            _ta_flat_intervals.append((_t0_g, _t1_g))
    n_steps = len(_ta_flat_intervals)
    print(f"  Total time steps: {{n_steps}} from {{len(_ta_groups)}} group(s)")

    _plot_idx = {config.plot_output_idx}
    _plot_custom_expr = "{plot_custom_expr_converted}"
    _plot_output_names_list = {repr(config.output_names)}.split(",")
    _plot_n_out = len(_plot_output_names_list)
    _plot_dim = "3D" if _is_3d else ("2D" if _is_2d else "1D")
{_math_namespace_code('_PLOT_TORCH_MATH_NS', 'torch', '    ')}

    def _plot_custom_op(_pf_inputs, _pf_outputs):
        _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                     _plot_output_names_list, _is_steady, _plot_dim,
                                     _plot_custom_expr)
        _pf_ns = dict(_pf_dvars)
        _pf_ns.update(_PLOT_TORCH_MATH_NS)
        _pf_ns["torch"] = torch
        return eval(_plot_custom_expr, _pf_ns)

    def _extract_plot_field(_x_grid, _pf_model):
        # Time-Adaptive's own copy of the Standard-path helper of the same
        # name (out of scope here -- it's defined inside `if not
        # {config.time_adaptive}:`, a separate top-level block this one
        # never runs alongside). Same logic: a derived custom field (e.g.
        # |h| = sqrt(u**2+v**2) for 1D Schrodinger), now including
        # derivatives (du_x, du_xx, du_xy, ...) via the same
        # dde.Model.predict(x, operator=...) mechanism the Standard path
        # uses, when one is configured, else the single raw output column
        # -- so every prediction call in this Time-Adaptive loop plots/
        # animates the same field the Standard path and the main Results
        # panel would, instead of indexing a column that may not even
        # exist (e.g. plot_output_idx pointing past the last real output
        # when "Custom..." is selected). Unlike the Standard path there's
        # no single module-level `model` here -- each step has its own
        # restored model -- so the caller's model object is always passed
        # in explicitly rather than defaulted.
        if _plot_custom_expr.strip():
            return _pf_model.predict(_x_grid, operator=_plot_custom_op)[:, 0]
        return _pf_model.predict(_x_grid)[:, _plot_idx]

    # ── Reference-data matching for per-step comparison plots ────
    # Each entry is (time, path, output_selector) -- same convention RAR's
    # own `_rar_ea_entries` and the inline/Restore Error Analysis sections
    # use (output_selector: None = the default plot field, an int = a raw
    # output column, or an [expr, label] pair = a custom derived field).
    # A reference file is matched to the Time-Adaptive step whose own
    # [t0, t1) window contains that file's time value (closed on the right
    # only for the LAST step, so a file sitting exactly on a shared
    # boundary between two adjacent steps is matched to exactly one of
    # them, not both). This step's own model_i is already in scope live in
    # this loop, so matching only needs the time-range filter below --
    # unlike Restore's own TA Error Analysis, which has to reconstruct and
    # restore each step's model from the saved step directories after the
    # fact.
    from scipy.interpolate import interp1d as _ta_interp1d
    _ta_ea_entries = {config.ea_files}

    def _ta_ref_extract(_tea_grid, _tea_sel, _tea_model):
        if _tea_sel is None:
            return _extract_plot_field(_tea_grid, _tea_model)
        if isinstance(_tea_sel, int):
            return _tea_model.predict(_tea_grid)[:, _tea_sel]
        _tea_expr = _tea_sel[0]

        def _tea_ref_op(_tea_i, _tea_o):
            _tea_dv = _tm_build_dvars(_tea_i, _tea_o, _plot_n_out, _plot_output_names_list,
                                       _is_steady, _plot_dim, _tea_expr)
            _tea_ns = dict(_tea_dv)
            _tea_ns.update(_PLOT_TORCH_MATH_NS)
            _tea_ns["torch"] = torch
            return eval(_tea_expr, _tea_ns)

        return _tea_model.predict(_tea_grid, operator=_tea_ref_op)[:, 0]

    # Accumulates the loss history across EVERY optimizer phase of EVERY
    # time sub-domain, so the loss plot below (see "_ta_all_train_loss")
    # shows the full Time-Adaptive run instead of just the LAST phase of
    # the LAST sub-domain. lh_i/ts_i get overwritten by each new
    # model_i.train() call below -- there can be several per sub-domain
    # (one per optimizer-scheduler phase), plus one more for the legacy
    # Adam-then-L-BFGS path -- so without this, only whichever call
    # happened to run last for the very last step_i would ever reach the
    # plot. Each phase's own step counter restarts at 0 (DeepXDE's
    # LossHistory numbers every train() call's steps from scratch), so
    # _ta_loss_offset -- the cumulative iteration count so far across
    # every phase and every sub-domain -- is added to keep the x-axis
    # continuously increasing across the whole run. Same "steps[-1] added
    # to a running total" trick already used just for the printed
    # cumulative-iteration numbers in the scheduler-phase logging below
    # (see "_sp_iters"/"_sched_cum_iters" elsewhere in this file) --
    # applied here to the loss arrays themselves, not just a printed
    # number.
    _ta_all_train_loss = []
    _ta_all_test_loss = []
    _ta_all_steps = []
    _ta_loss_offset = 0
    _ta_step_boundaries = []  # cumulative-iteration marks where one time sub-domain ends and the next begins

    def _ta_accumulate_loss(_lh):
        global _ta_loss_offset
        _ta_all_steps.extend([_s + _ta_loss_offset for _s in _lh.steps])
        _ta_all_train_loss.extend(_lh.loss_train)
        _ta_all_test_loss.extend(_lh.loss_test)
        _ta_loss_offset += (_lh.steps[-1] if _lh.steps else 0)

    for step_i, (t0, t1) in enumerate(_ta_flat_intervals):
        print(f"\\n--- Time step {{step_i+1}}/{{n_steps}}: t = {{t0:.4f}} to {{t1:.4f}} ---")

        geom_i = _build_geom()
        time_i     = dde.geometry.TimeDomain(t0, t1)
        geomtime_i = dde.geometry.GeometryXTime(geom_i, time_i)

        _constraints_i = []
        # The Boundary Conditions panel (custom_bc_json, already parsed into
        # _custom_bc_entries above) is now the source of truth here too --
        # same as the main (non-time-adaptive) path. The legacy per-side
        # scheme below only runs for configs saved before the panel existed.
        if _custom_bc_entries or _bc_panel_data_present:
            for _bi_ta, _be_ta in enumerate(_custom_bc_entries):
                _btype_ta = _be_ta.get('type', 'dirichlet')
                _bcomp_ta = int(_be_ta.get('component', 0) or 0)
                _bloc_ta  = _be_ta.get('location', '')
                _bval_ta  = _be_ta.get('value', '0')
                _baxis_ta = _be_ta.get('axis', 'x')
                _bderiv_ta = min(max(int(_be_ta.get('deriv_order', 0) or 0), 0), 1)  # same clamp as the main path above
                _bpts_file_ta = _be_ta.get('points_file', '')
                _bloc2_ta = _be_ta.get('location2', '')
                _bdir_ta  = _be_ta.get('direction', 'normal')
                _blabel_ta = f"BC {{_bi_ta + 1}} ({{_btype_ta}})"
                # Same component-list/batching parsing as the main path above.
                _bcomponents_txt_ta = (_be_ta.get('components', '') or '').strip()
                _bcomponents_ta = [int(_c.strip()) for _c in _bcomponents_txt_ta.split(',') if _c.strip()] if _bcomponents_txt_ta else None
                _bbatch_ta = int(_be_ta.get('batch_size', 0) or 0)
                _bshuffle_ta = bool(_be_ta.get('shuffle', True))
                _bbatch_kwargs_ta = {{"batch_size": _bbatch_ta, "shuffle": _bshuffle_ta}} if _bbatch_ta > 0 else {{}}

                if _btype_ta == 'neumann':
                    _constraints_i.append(dde.icbc.NeumannBC(geomtime_i, _bc_val_fn(_bval_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))
                elif _btype_ta == 'robin':
                    _constraints_i.append(dde.icbc.RobinBC(geomtime_i, _bc_robin_fn(_bval_ta, _bcomp_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))
                elif _btype_ta == 'periodic':
                    _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, _axis_idx_map.get(_baxis_ta, 0), _bc_loc_fn(_bloc_ta),
                                                                derivative_order=_bderiv_ta, component=_bcomp_ta))
                elif _btype_ta == 'pointset':
                    _pts_ta, _pvals_ta = _bc_points(_be_ta, _blabel_ta, len(_bcomponents_ta) if _bcomponents_ta else 1, geomtime_i)
                    _constraints_i.append(dde.icbc.PointSetBC(_pts_ta, _pvals_ta, component=(_bcomponents_ta if _bcomponents_ta else _bcomp_ta), **_bbatch_kwargs_ta))
                elif _btype_ta == 'pointset_operator':
                    _pts_ta, _pvals_ta = _bc_points(_be_ta, _blabel_ta, 1, geomtime_i)
                    _constraints_i.append(dde.icbc.PointSetOperatorBC(_pts_ta, _pvals_ta, _bc_operator_fn(_bval_ta), **_bbatch_kwargs_ta))
                elif _btype_ta == 'operator':
                    _constraints_i.append(dde.icbc.OperatorBC(geomtime_i, _bc_operator_fn(_bval_ta), _bc_loc_fn(_bloc_ta)))
                elif _btype_ta == 'interface2d':
                    # Time-Adaptive steps are never steady-state (geomtime_i
                    # is always a real GeometryXTime here), so the adapter's
                    # trim is always correct -- passed explicitly (rather
                    # than relying on the default) to document that.
                    _constraints_i.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime_i, False), _bc_val_fn(_bval_ta),
                                                                    _bc_loc_fn(_bloc_ta), _bc_loc_fn(_bloc2_ta), direction=_bdir_ta))
                elif _btype_ta == 'dirichlet':
                    _constraints_i.append(dde.icbc.DirichletBC(geomtime_i, _bc_val_fn(_bval_ta), _bc_loc_fn(_bloc_ta), component=_bcomp_ta))
                else:
                    # Same loud failure as the main (non-Time-Adaptive) path
                    # above, instead of silently training with the wrong BC.
                    raise ValueError(
                        f"{{_blabel_ta}}: unrecognized boundary condition type {{_btype_ta!r}}. "
                        "Expected one of: dirichlet, neumann, robin, periodic, pointset, "
                        "pointset_operator, operator, interface2d."
                    )

            # IC (independent of the BC panel -- same per-step IC scheme as always)
            for _oi_ta in range({config.num_outputs}):
                _comp_ta = _oi_ta
                if step_i == 0:
                    if {config.forward_ic_from_file} and _oi_ta == 0:
                        _xyt_ic_anchor = _ic_xyt  # use the already-loaded IC points
                        _constraints_i.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                    elif _oi_ta < len(_ic_active_list) and _ic_active_list[_oi_ta].strip() == "True":
                        _ic_expr_ta = _ic_expressions[_oi_ta].strip() if _oi_ta < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                        def _mk_ic_ta(expr, comp):
                            def _ic_fn(x):
                                return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                            return dde.icbc.IC(geomtime_i, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                        _constraints_i.append(_mk_ic_ta(_ic_expr_ta, _comp_ta))
                else:
                    if _is_2d or _is_3d:
                        # x_grid is already an (N, 2) or (N, 3) array of
                        # spatial points either way, so one column_stack
                        # covers both.
                        _xt_ic_2d = np.column_stack([x_grid, np.full(len(x_grid), t0)])
                        _xyt_ic_anchor = _xt_ic_2d
                        _constraints_i.append(dde.icbc.PointSetBC(_xt_ic_2d, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
                    else:
                        xt_ic = np.column_stack([x_grid, np.full_like(x_grid.ravel(), t0)])
                        _xyt_ic_anchor = xt_ic
                        _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
        else:
            # Legacy per-side BCs, for configs saved before the Boundary
            # Conditions panel existed (custom_bc_json empty).
            for _oi_ta in range({config.num_outputs}):
                _comp_ta = _oi_ta
                _blt_ta = _bc_left_types[_oi_ta].strip()  if _oi_ta < len(_bc_left_types)  else "Dirichlet"
                _brt_ta = _bc_right_types[_oi_ta].strip() if _oi_ta < len(_bc_right_types) else "Dirichlet"
                _blv_ta = float(_bc_left_values[_oi_ta].strip())  if _oi_ta < len(_bc_left_values)  else 0.0
                _brv_ta = float(_bc_right_values[_oi_ta].strip()) if _oi_ta < len(_bc_right_values) else 0.0

                if _oi_ta < len(_bc_left_active) and _bc_left_active[_oi_ta].strip() == "True":
                    def _bl_on_ta(x, on_boundary):
                        return on_boundary and dde.utils.isclose(x[0], {config.x_min})
                    if _blt_ta == "Dirichlet":
                        def _mk_dbl_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_dbl_ta(_blv_ta, _comp_ta, _bl_on_ta))
                    elif _blt_ta == "Neumann":
                        def _mk_nbl_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.NeumannBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_nbl_ta(_blv_ta, _comp_ta, _bl_on_ta))
                    elif _blt_ta == "Periodic":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _bl_on_ta, derivative_order=0, component=_comp_ta))
                        if _oi_ta < len(_bc_left_deriv_list) and _bc_left_deriv_list[_oi_ta].strip() == "True":
                            _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _bl_on_ta, derivative_order=1, component=_comp_ta))

                if _oi_ta < len(_bc_right_active) and _bc_right_active[_oi_ta].strip() == "True":
                    def _br_on_ta(x, on_boundary):
                        return on_boundary and dde.utils.isclose(x[0], {config.x_max})
                    if _brt_ta == "Dirichlet":
                        def _mk_dbr_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_dbr_ta(_brv_ta, _comp_ta, _br_on_ta))
                    elif _brt_ta == "Neumann":
                        def _mk_nbr_ta(v, comp, on_bd):
                            def _vf(x): return np.full((len(x),1), v)
                            return dde.icbc.NeumannBC(geomtime_i, _vf, on_bd, component=comp)
                        _constraints_i.append(_mk_nbr_ta(_brv_ta, _comp_ta, _br_on_ta))
                    elif _brt_ta == "Periodic":
                        _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _br_on_ta, derivative_order=0, component=_comp_ta))
                        if _oi_ta < len(_bc_right_deriv_list) and _bc_right_deriv_list[_oi_ta].strip() == "True":
                            _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 0, _br_on_ta, derivative_order=1, component=_comp_ta))

                if _is_2d:
                    _bbt_ta = _bc_bottom_types[_oi_ta].strip() if _oi_ta < len(_bc_bottom_types) else "Dirichlet"
                    _btt_ta = _bc_top_types[_oi_ta].strip()    if _oi_ta < len(_bc_top_types)    else "Dirichlet"
                    _bbv_ta = float(_bc_bottom_values[_oi_ta].strip()) if _oi_ta < len(_bc_bottom_values) else 0.0
                    _btv_ta = float(_bc_top_values[_oi_ta].strip())    if _oi_ta < len(_bc_top_values)    else 0.0

                    if _oi_ta < len(_bc_bottom_active) and _bc_bottom_active[_oi_ta].strip() == "True":
                        def _bb_on_ta(x, on_boundary):
                            return on_boundary and dde.utils.isclose(x[1], {config.y_min})
                        if _bbt_ta == "Dirichlet":
                            def _mk_dbb_ta(v, comp, on_bd):
                                def _vf(x): return np.full((len(x),1), v)
                                return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                            _constraints_i.append(_mk_dbb_ta(_bbv_ta, _comp_ta, _bb_on_ta))
                        elif _bbt_ta == "Periodic":
                            _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 1, _bb_on_ta, derivative_order=0, component=_comp_ta))

                    if _oi_ta < len(_bc_top_active) and _bc_top_active[_oi_ta].strip() == "True":
                        def _bt_on_ta(x, on_boundary):
                            return on_boundary and dde.utils.isclose(x[1], {config.y_max})
                        if _btt_ta == "Dirichlet":
                            def _mk_dbt_ta(v, comp, on_bd):
                                def _vf(x): return np.full((len(x),1), v)
                                return dde.icbc.DirichletBC(geomtime_i, _vf, on_bd, component=comp)
                            _constraints_i.append(_mk_dbt_ta(_btv_ta, _comp_ta, _bt_on_ta))
                        elif _btt_ta == "Periodic":
                            _constraints_i.append(dde.icbc.PeriodicBC(geomtime_i, 1, _bt_on_ta, derivative_order=0, component=_comp_ta))

                if step_i == 0:
                    if {config.forward_ic_from_file} and _oi_ta == 0:
                        _xyt_ic_anchor = _ic_xyt  # use the already-loaded IC points
                        _constraints_i.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                    elif _oi_ta < len(_ic_active_list) and _ic_active_list[_oi_ta].strip() == "True":
                        _ic_expr_ta = _ic_expressions[_oi_ta].strip() if _oi_ta < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                        def _mk_ic_ta(expr, comp):
                            def _ic_fn(x):
                                return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                            return dde.icbc.IC(geomtime_i, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                        _constraints_i.append(_mk_ic_ta(_ic_expr_ta, _comp_ta))
                else:
                    if _is_2d or _is_3d:
                        _xt_ic_2d = np.column_stack([x_grid, np.full(len(x_grid), t0)])
                        _xyt_ic_anchor = _xt_ic_2d
                        _constraints_i.append(dde.icbc.PointSetBC(_xt_ic_2d, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))
                    else:
                        xt_ic = np.column_stack([x_grid, np.full_like(x_grid.ravel(), t0)])
                        _xyt_ic_anchor = xt_ic
                        _constraints_i.append(dde.icbc.PointSetBC(xt_ic, prev_u[:, _oi_ta:_oi_ta+1], component=_oi_ta))

        data_i = dde.data.TimePDE(
            geomtime_i, pde, _constraints_i,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={_safe_num_test!r},
            train_distribution="{_safe_point_distribution}",   # as the whole-run model (was DeepXDE's default)
            # forward_ic_from_file only matters for step 0 (loading the
            # true initial condition from a file there); every later step's
            # IC constraint comes from the previous window's own predicted
            # solution (prev_u) via _xyt_ic_anchor, regardless of that flag.
            # Gating this on the bare flag (rather than "step 0 AND the
            # flag") silently dropped the previous-window anchor points for
            # EVERY step after the first whenever forward_ic_from_file was
            # on, weakening (though not removing -- the PointSetBC loss
            # term above is still added either way) how strongly later
            # windows are forced to match their own true starting point.
            anchors=None if (step_i == 0 and {config.forward_ic_from_file}) else (_xyt_ic_anchor if (step_i > 0 and (_is_2d or _is_3d)) else None)
        )

        net_i   = _apply_net_transforms(_make_net({config.layers}, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay}))
        model_i = dde.Model(data_i, net_i)

        # ── Training callbacks (opt-in, fresh instances each step) ─
{_train_cbs_code_ta}

        # ── Transfer learning — warm start from previous step ─
        if {config.ta_transfer_learning} and step_i > 0 and (_prev_step_model_path or _prev_step_state):
            try:
                import torch as _torch
                # the saved checkpoint of the chosen phase; without a save
                # folder, the previous step's network kept in memory
                _state = (_torch.load(_prev_step_model_path, map_location='cpu')["model_state_dict"]
                          if _prev_step_model_path else _prev_step_state)
                # Extract ONLY weights and biases — nothing else
                _wb_only = {{
                    k: v for k, v in _state.items()
                    if k.endswith('.weight') or k.endswith('.bias')
                }}
                # Strict=False so geometry mismatch doesn't crash
                # but we verify shapes match before loading
                _cur_state = model_i.net.state_dict()
                _matched = {{}}
                _skipped = []
                for k, v in _wb_only.items():
                    if k in _cur_state and _cur_state[k].shape == v.shape:
                        _matched[k] = v
                    else:
                        _skipped.append(k)
                _cur_state.update(_matched)
                model_i.net.load_state_dict(_cur_state)
                print(f"  ✅ Transfer learning: {{len(_matched)}} weight/bias tensors transferred from step {{step_i}}")
                if _skipped:
                    print(f"  ⚠️  Skipped {{len(_skipped)}} tensors (shape mismatch): {{_skipped}}")
            except Exception as _te:
                print(f"  ⚠️ Transfer learning failed: {{_te}} — training from scratch")

        # IC pre-training for first step only
        if {config.ic_pretrain} and step_i == 0:
            print(f"  === IC Pre-Training: {config.ic_pretrain_iterations} iterations ===")
            # Build IC-only data: no domain/BC points, only IC points
            _ic_geom_pt = _build_geom()
            _ic_td_pt = dde.geometry.TimeDomain(t0, t1)
            _ic_gt_pt = dde.geometry.GeometryXTime(_ic_geom_pt, _ic_td_pt)
            # Only IC constraint
            _ic_constraints_pt = []
            if {config.forward_ic_from_file}:
                _ic_ta_xyt, _ic_ta_vals = _load_ic_from_file({repr(config.forward_ic_file)})
                _ic_constraints_pt.append(dde.icbc.PointSetBC(_ic_ta_xyt, _ic_ta_vals, component=0))
            else:
                for _oi_pt in range({config.num_outputs}):
                    _ic_expr_pt = _ic_expressions[_oi_pt].strip() if _oi_pt < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                    def _mk_ic_pt(expr, comp):
                        def _ic_fn(x):
                            return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                        return dde.icbc.IC(_ic_gt_pt, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                    _ic_constraints_pt.append(_mk_ic_pt(_ic_expr_pt, _oi_pt))
            # IC-only loss weights: PDE terms 0, every IC term 1000.
            _ic_only_weights = [0.0] * {config.num_outputs} + [1000.0] * len(_ic_constraints_pt)
{_ic_pretrain_block_ta}
            print("  === Starting Main Training ===")

        # ── Phase 1 (Adam) ──────────────────────────────────────
        # Mirrors the Standard (non-Time-Adaptive) path's own unconditional
        # Phase 1 training call above (model.train(iterations=_iters, ...)
        # under "if not _sched_active:") -- this per-step loop previously
        # had no equivalent: it only ever compiled/trained model_i inside
        # the scheduler-phases block below or the legacy optimizer2
        # L-BFGS-only branch's elif, with nothing to fall back on when
        # neither applied. A legacy/hand-edited config with the (GUI-
        # hidden, always-on) scheduler off, an empty scheduler_phases, and
        # optimizer2 left at "none" would then never compile or train
        # model_i at all for this step -- model_i.predict(...) ran
        # immediately after on an untrained, randomly-initialized network,
        # and that garbage became the next step's initial condition, with
        # no exception raised anywhere.
        if not ({config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}):
            model_i.compile("{config.optimizer}", lr={config.learning_rate},
                            loss="{config.loss_type}", loss_weights=_multi_weights)
            lh_i, ts_i = model_i.train(iterations={config.iterations}, display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
            _ta_accumulate_loss(lh_i)
            print(f"  Adam phase done. Steps: {{len(lh_i.steps)}}")
            if _use_save:
                _step_dir_adam = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                _os.makedirs(_step_dir_adam, exist_ok=True)
                _step_adam_path = _os.path.join(_step_dir_adam, "model_adam")
                model_i.save(_step_adam_path)
                print(f"Step Adam model saved: {{_step_adam_path}}.pt")

        # ── Optimizer Scheduler phases ────────────────────────
        if {config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}:
            import json as _json
            _sched_phases = _json.loads({repr(config.scheduler_phases)})
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                # A phase without its own weights (e.g. written by hand) uses
                # the Loss Weights panel's.
                _sp_weights = [float(w) for w in str(_sp.get('weights') or '').split(',') if w.strip()] or list(_multi_weights)
                # Same safety net as the main (non-Time-Adaptive) scheduler loop:
                # each step rebuilds its own _constraints_i, which can be a
                # different length than what the GUI's scheduler-phase weight
                # string was built from (e.g. a step boundary changing which IC
                # constraint applies) -- fall back to uniform weights of the
                # right length rather than crash mid-training.
                _sp_expected_len_i = {_n_equations(config)} + len(_constraints_i)
                if len(_sp_weights) != _sp_expected_len_i:
                    print(f"  ⚠️ Scheduler Phase {{_sp_i+1}} (step {{step_i+1}}): configured weights have "
                          f"{{len(_sp_weights)}} entries but {{_sp_expected_len_i}} loss terms are active this "
                          f"step ({{len(_constraints_i)}} constraints + {_n_equations(config)} PDE) — using "
                          f"uniform weights for this phase instead of crashing.")
                    _sp_weights = [1.0] * _sp_expected_len_i
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=(int(_sp['iterations']*1.25) if {bool(getattr(config, 'lbfgs_use_default', True))} else {config.lbfgs_maxfun}), maxls={config.lbfgs_maxls})
                    model_i.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                    loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
                    if _use_save:
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} L-BFGS model saved: {{_sp_save_path}}.pt")
                elif _sp['optimizer'] == 'nncg':
                    # See the matching NNCG branch in the Standard scheduler
                    # loop above for why: exact-case "NNCG" name, no lr=, no
                    # decay (NNCG_options is left at its defaults -- not
                    # exposed in this app's UI).
                    model_i.compile("NNCG", loss=_sp.get('loss', '{config.loss_type}'),
                                    loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
                    if _use_save:
                        _sp_iters = lh_i.steps[-1] if lh_i.steps else _sp['iterations']
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_nncg-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} NNCG model saved: {{_sp_save_path}}.pt")
                else:
                    _sp_decay = None
                    _sp_decay_type = _sp.get('decay_type', 'none')
                    if _sp_decay_type and _sp_decay_type != 'none':
                        _sp_p1 = _sp.get('decay_p1', 0) or 0
                        _sp_p2 = _sp.get('decay_p2', 0) or 0
                        if _sp_decay_type == 'step':
                            _sp_decay = ('step', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'cosine':
                            _sp_decay = ('cosine', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'inverse time':
                            _sp_decay = ('inverse time', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'exponential':
                            _sp_decay = ('exponential', float(_sp_p1))
                    model_i.compile(_sp['optimizer'], lr=_sp['lr'], decay=_sp_decay,
                                    loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights)
                    lh_i, ts_i = model_i.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
                    _ta_accumulate_loss(lh_i)
                    if _use_save:
                        _sp_iters = lh_i.steps[-1] if lh_i.steps else _sp['iterations']
                        _sp_step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                        _os.makedirs(_sp_step_dir, exist_ok=True)
                        _sp_save_path = _os.path.join(_sp_step_dir, f"model_adam-phase{{_sp_i+1}}")
                        model_i.save(_sp_save_path)
                        print(f"  Phase {{_sp_i+2}} Adam model saved: {{_sp_save_path}}.pt")
        elif "{config.optimizer2}" != "none":
            dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.iterations2},
                    maxfun={_lbfgs_maxfun(config, config.iterations2)}, maxls={config.lbfgs_maxls})
            model_i.compile("L-BFGS", loss="{config.loss_type}",
                            loss_weights=_multi_weights)
            lh_i, ts_i = model_i.train(display_every={config.loss_display_every}, callbacks=_train_cbs_ta)
            _ta_accumulate_loss(lh_i)
            print(f"  L-BFGS phase done. Steps: {{len(lh_i.steps)}}")

            if _use_save:
                _step_dir_lbfgs = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
                _os.makedirs(_step_dir_lbfgs, exist_ok=True)
                _step_lbfgs_path = _os.path.join(_step_dir_lbfgs, "model_lbfgs")
                model_i.save(_step_lbfgs_path)
                print(f"Step L-BFGS model saved: {{_step_lbfgs_path}}.pt")

        if _is_3d:
            # 3D: predict on x-y-z grid at t=t1, store as flat array for PointSetBC next step
            _x_pred = np.linspace({config.x_min}, {config.x_max}, grid_size)
            _y_pred = np.linspace({config.y_min}, {config.y_max}, grid_size)
            _z_pred = np.linspace({config.z_min}, {config.z_max}, grid_size)
            _Xp, _Yp, _Zp = np.meshgrid(_x_pred, _y_pred, _z_pred)
            _xyzt_pred = np.column_stack([_Xp.ravel(), _Yp.ravel(), _Zp.ravel(), np.full(_Xp.size, t1)])
            prev_u  = model_i.predict(_xyzt_pred)
            x_grid  = _xyzt_pred[:, :3]  # store (x,y,z) triples for next step's PointSetBC
        elif _is_2d:
            # 2D: predict on x-y grid at t=t1, store as flat array for PointSetBC next step
            _x_pred = np.linspace({config.x_min}, {config.x_max}, grid_size)
            _y_pred = np.linspace({config.y_min}, {config.y_max}, grid_size)
            _Xp, _Yp = np.meshgrid(_x_pred, _y_pred)
            _xyt_pred = np.column_stack([_Xp.ravel(), _Yp.ravel(), np.full(_Xp.size, t1)])
            prev_u  = model_i.predict(_xyt_pred)
            x_grid  = _xyt_pred[:, :2]  # store (x,y) pairs for next step's PointSetBC
        else:
            x_pred  = np.linspace({config.x_min}, {config.x_max}, grid_size)
            t_pred  = np.full_like(x_pred, t1)
            xt_pred = np.column_stack([x_pred, t_pred])
            prev_u  = model_i.predict(xt_pred)
            x_grid  = x_pred.reshape(-1, 1)

        if _is_3d:
            # 3D: no volumetric renderer for the stitched plot either (same
            # limit as the Standard path's own 3D solution plot) -- fix
            # BOTH y and z at their domain mid-points and store an x-t
            # slice, exactly like the 2D branch below fixes y alone. This
            # keeps all_x/all_t/all_u in the same shape regardless of
            # dimension, so the final combined plot needs no 3D-specific
            # branch of its own.
            x_plot = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
            t_plot = np.linspace(t0, t1, {config.plot_resolution})
            y_mid  = {line_slice_y}
            z_mid  = {line_slice_z}
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XYZTp  = np.column_stack([Xp.ravel(), np.full(Xp.size, y_mid), np.full(Xp.size, z_mid), Tp.ravel()])
            Up     = _extract_plot_field(XYZTp, model_i).reshape({config.plot_resolution}, {config.plot_resolution})
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)
        elif not _is_2d:
            x_plot = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
            t_plot = np.linspace(t0, t1, {config.plot_resolution})
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XTp    = np.vstack([Xp.ravel(), Tp.ravel()]).T
            Up     = _extract_plot_field(XTp, model_i).reshape({config.plot_resolution}, {config.plot_resolution})
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)
        else:
            # 2D: store mid-y slice for combined plot
            x_plot = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
            t_plot = np.linspace(t0, t1, {config.plot_resolution})
            y_mid  = {line_slice_y}
            Xp, Tp = np.meshgrid(x_plot, t_plot)
            XYTp   = np.column_stack([Xp.ravel(), np.full(Xp.size, y_mid), Tp.ravel()])
            Up     = _extract_plot_field(XYTp, model_i).reshape({config.plot_resolution}, {config.plot_resolution})
            all_x.append(Xp); all_t.append(Tp); all_u.append(Up)

        print(f"Step {{step_i+1}} done. Final train loss: {{sum(lh_i.loss_train[-1]):.4e}}")
        _ta_step_boundaries.append(_ta_loss_offset)

        # ── Save step plot & models ───────────────────────────
        if _use_save:
            _step_dir = _os.path.join(_save_dir, "time_adaptive_steps", f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}")
            _os.makedirs(_step_dir, exist_ok=True)
            # ── Reference matches for this step's own time range ──
            # Half-open [t0, t1) except the LAST step, whose upper bound is
            # inclusive -- avoids double-counting a reference file sitting
            # exactly on the shared boundary between two adjacent steps
            # (ta_step_groups splits time into contiguous sub-domains, so
            # t1 of step i always equals t0 of step i+1).
            _step_matches = sorted(
                [_tea for _tea in _ta_ea_entries
                 if (t0 - 1e-9) <= _tea[0] <= (t1 + 1e-9 if step_i == n_steps - 1 else t1 - 1e-9)],
                key=lambda _tea: _tea[0],
            )
            if _step_matches:
                print(f"  Step {{step_i+1}}: {{len(_step_matches)}} reference file(s) in t=[{{t0:.4f}}, {{t1:.4f}}]")

            if not _is_2d and not _is_3d:
                # ── 1D: always both a surface and a line comparison --
                # merged with reference when this step's time range has
                # matching reference file(s), else PINN-only (same as
                # before this feature existed).
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}_surface.png")
                try:
                    if len(_step_matches) >= 2:
                        # >= 2 distinct reference times in range -- build a
                        # real (time x space) comparison surface, same
                        # technique the Standard path's Inline Error
                        # Analysis "1D: standard x vs t surface" uses.
                        _tea_x_common = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                        _tea_times_m = [_m[0] for _m in _step_matches]
                        _tea_U_pinn = np.zeros((len(_step_matches), len(_tea_x_common)))
                        _tea_U_ref = np.zeros((len(_step_matches), len(_tea_x_common)))
                        for _mi, (_mtv, _mfp, _msel) in enumerate(_step_matches):
                            _tea_xt_c = np.column_stack([_tea_x_common, np.full_like(_tea_x_common, _mtv)])
                            _tea_U_pinn[_mi, :] = _ta_ref_extract(_tea_xt_c, _msel, model_i).flatten()
                            _tea_d = np.loadtxt(_mfp)
                            if _tea_d.ndim == 1:
                                _tea_d = _tea_d.reshape(1, -1)
                            _tea_ord = np.argsort(_tea_d[:, 0])
                            _tea_fi = _ta_interp1d(_tea_d[_tea_ord, 0], _tea_d[_tea_ord, 2], kind="linear", fill_value="extrapolate")
                            _tea_U_ref[_mi, :] = _tea_fi(_tea_x_common)
                        _tea_Xg, _tea_Tg = np.meshgrid(_tea_x_common, _tea_times_m)
                        _tea_U_err = np.abs(_tea_U_pinn - _tea_U_ref)
                        _tea_vmin = min(_tea_U_pinn.min(), _tea_U_ref.min())
                        _tea_vmax = max(_tea_U_pinn.max(), _tea_U_ref.max())
                        if _tea_vmax - _tea_vmin < 1e-12:
                            _tea_vmax = _tea_vmin + 1e-12
                        _tea_vmin, _tea_vmax = _vr(_tea_vmin, _tea_vmax)   # Plot settings -> Surface -> Colour range
                        _tea_levels = _cf_levels({config.plot_levels}, _tea_vmin, _tea_vmax)["levels"]
                        fig, axes_s = plt.subplots(1, 3, figsize=_plot_figsize(15, 5))
                        fig.suptitle(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        _tea_cols = [
                            (_tea_U_pinn, "PINN", _tea_levels, "{config.plot_colormap}"),
                            (_tea_U_ref, "Reference", _tea_levels, "{config.plot_colormap}"),
                            (_tea_U_err, f"|Error|  max = {{_sci_txt(_tea_U_err.max())}}", {config.plot_levels}, "{err_cmap(config)}"),
                        ]
                        for _tea_ci, (_tea_vals, _tea_ttl, _tea_lv, _tea_cmap) in enumerate(_tea_cols):
                            if {config.plot_swap_xt}:
                                _tea_im = axes_s[_tea_ci].contourf(_tea_Tg, _tea_Xg, _tea_vals, levels=_tea_lv, cmap=_tea_cmap, extend=_vext(_tea_lv))
                                axes_s[_tea_ci].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes_s[_tea_ci].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                            else:
                                _tea_im = axes_s[_tea_ci].contourf(_tea_Xg.T, _tea_Tg.T, _tea_vals.T, levels=_tea_lv, cmap=_tea_cmap, extend=_vext(_tea_lv))
                                axes_s[_tea_ci].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes_s[_tea_ci].set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
                            axes_s[_tea_ci].set_title(_tea_ttl, fontsize={config.plot_subplot_title_fontsize})
                            fig.colorbar(_tea_im, ax=axes_s[_tea_ci])
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    else:
                        # 0 or 1 reference match -- a surface needs >= 2
                        # distinct time snapshots to interpolate against;
                        # fall back to the plain PINN-only step preview
                        # (the line comparison below still covers a single
                        # match on its own).
                        _vmin_step = None if {config.plot_auto_range} else {config.plot_vmin}
                        _vmax_step = None if {config.plot_auto_range} else {config.plot_vmax}
                        _res_step = {config.plot_resolution}
                        _x_s2 = np.linspace({config.x_min}, {config.x_max}, _res_step)
                        _t_s2 = np.linspace(t0, t1, _res_step)
                        _Xs2, _Ts2 = np.meshgrid(_x_s2, _t_s2)
                        _XTs2 = np.vstack([_Xs2.ravel(), _Ts2.ravel()]).T
                        _Us2 = _extract_plot_field(_XTs2, model_i).reshape(_res_step, _res_step)
                        fig, ax = plt.subplots(figsize=_plot_figsize(7, 4))
                        if {config.plot_swap_xt}:
                            im = ax.contourf(_Ts2, _Xs2, _Us2, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_step, _vmax_step, _Us2))
                            ax.set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                        else:
                            im = ax.contourf(_Xs2, _Ts2, _Us2, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_step, _vmax_step, _Us2))
                            ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
                        if {config.plot_colorbar}:
                            fig.colorbar(im, ax=ax)
                        ax.set_title(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}}")
                        plt.tight_layout()
                    plt.savefig(_step_fname, dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close()
                    print(f"Step surface plot saved: {{_step_fname}}")
                except Exception as _tea_err_surf:
                    print(f"  Step {{step_i+1}} surface plot failed: {{_tea_err_surf}}")

                _step_line_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}_line.png")
                try:
                    if _step_matches:
                        # One subplot per matching reference time (showing
                        # ALL of them when there's more than one, since
                        # that's more informative) instead of the plain
                        # evenly-spaced time sampling used below.
                        _tea_ncols = min({max(1, int(getattr(config, 'plot_ea_cols', 4) or 4))}, len(_step_matches))
                        _tea_nrows = (len(_step_matches) + _tea_ncols - 1) // _tea_ncols
                        fig, axes = plt.subplots(_tea_nrows, _tea_ncols, figsize=_panel_figsize(4, 3.5, _tea_ncols, _tea_nrows), squeeze=False)
                        fig.suptitle(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        _tea_ax_flat = axes.flatten()
                        for _mi, (_mtv, _mfp, _msel) in enumerate(_step_matches):
                            _tea_d = np.loadtxt(_mfp)
                            if _tea_d.ndim == 1:
                                _tea_d = _tea_d.reshape(1, -1)
                            _tea_ord = np.argsort(_tea_d[:, 0])
                            _tea_xv = _tea_d[_tea_ord, 0]
                            _tea_uv = _tea_d[_tea_ord, 2]
                            _tea_xt_l = np.column_stack([_tea_xv, np.full_like(_tea_xv, _mtv)])
                            _tea_up = _ta_ref_extract(_tea_xt_l, _msel, model_i).flatten()
                            _tea_l2 = float(np.linalg.norm(_tea_up - _tea_uv) / (np.linalg.norm(_tea_uv) + 1e-10))
                            _ax = _tea_ax_flat[_mi]
                            _ref_plot(_ax, _tea_xv, _tea_uv, color=_PLOT_REF_COLOR, linewidth={config.plot_linewidth}, label="Reference")
                            _ax.plot(_tea_xv, _tea_up, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth}, linestyle="--", label="PINN")
                            _ax.set_title(f"t={{_fmt_num(_mtv)}}  {{_l2_txt(_tea_l2)}}", fontsize={config.plot_subplot_title_fontsize})
                            _ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax.set_ylabel("u", fontsize={config.plot_axis_label_fontsize}); _grid(_ax, alpha=0.3)
                        for _mj in range(len(_step_matches), len(_tea_ax_flat)):
                            _tea_ax_flat[_mj].set_visible(False)
                        _tea_h, _tea_l = _tea_ax_flat[0].get_legend_handles_labels()
                        fig.legend(_tea_h, _tea_l, loc="lower center", ncol=2, fontsize=_lfs(9), framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
                        plt.tight_layout(rect=[0, 0.06, 1, 1])
                    else:
                        n_steps_plot = {config.num_timesteps_line}
                        _x_l2 = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                        _t_line = np.linspace(t0, t1, n_steps_plot)
                        fig, ax = plt.subplots(figsize=_plot_figsize(8, 4))
                        colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                        for _ci, _tv in enumerate(_t_line):
                            _xt_line = np.column_stack([_x_l2, np.full_like(_x_l2, _tv)])
                            _u_line = _extract_plot_field(_xt_line, model_i).flatten()
                            ax.plot(_x_l2, _u_line, color=colors[_ci], linewidth={config.plot_linewidth}, label=f"t={{_fmt_num(_tv)}}")
                        ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("u", fontsize={config.plot_axis_label_fontsize})
                        ax.set_title(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}}")
                        ax.legend(loc="upper right", fontsize=_lfs(7)); _grid(ax, alpha=0.2)
                        plt.tight_layout()
                    plt.savefig(_step_line_fname, dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close()
                    print(f"Step line plot saved: {{_step_line_fname}}")
                except Exception as _tea_err_line:
                    print(f"  Step {{step_i+1}} line plot failed: {{_tea_err_line}}")
            else:
                # ── 2D/3D: surface only -- merged with reference (ALL
                # matching files, not just the closest one) when this
                # step's time range has matches, else the plain PINN-only
                # t0/t1 preview (unchanged from before this feature).
                _step_fname = _os.path.join(_step_dir, f"step_{{step_i+1:03d}}_t{{t0:.4f}}_to_t{{t1:.4f}}.png")
                try:
                    if _step_matches:
                        from mpl_toolkits.mplot3d import Axes3D as _tea_Axes3D_unused  # noqa: F401 -- registers the 3D projection
                        _tea_n = len(_step_matches)
                        if _is_3d:
                            fig = plt.figure(figsize=_plot_figsize(15, 4.5 * _tea_n))
                            for _mi, (_mtv, _mfp, _msel) in enumerate(_step_matches):
                                _tea_d = np.loadtxt(_mfp)
                                if _tea_d.ndim == 1:
                                    _tea_d = _tea_d.reshape(1, -1)
                                _tea_xyz = _tea_d[:, :3]
                                _tea_u = _tea_d[:, 4]
                                _tea_grid = np.column_stack([_tea_xyz, np.full(len(_tea_d), _mtv)])
                                _tea_pred = _ta_ref_extract(_tea_grid, _msel, model_i)
                                _tea_err = np.abs(_tea_pred - _tea_u)
                                _tea_l2 = float(np.linalg.norm(_tea_pred - _tea_u) / (np.linalg.norm(_tea_u) + 1e-12))
                                _tea_vmin = min(_tea_pred.min(), _tea_u.min())
                                _tea_vmax = max(_tea_pred.max(), _tea_u.max())
                                _tea_vmin, _tea_vmax = _vr(_tea_vmin, _tea_vmax)
                                _tea_cols3 = [
                                    (_tea_pred, f"PINN  t={{_fmt_num(_mtv)}}  {{_l2_txt(_tea_l2)}}", _tea_vmin, _tea_vmax, "{config.plot_colormap}"),
                                    (_tea_u, f"Reference  t={{_mtv:.4f}}", _tea_vmin, _tea_vmax, "{config.plot_colormap}"),
                                    (_tea_err, f"|Error|  max = {{_sci_txt(_tea_err.max())}}", None, None, "{err_cmap(config)}"),
                                ]
                                for _tea_ci, (_tea_vals, _tea_ttl, _tea_vmn, _tea_vmx, _tea_cmap) in enumerate(_tea_cols3):
                                    _tea_ax = fig.add_subplot(_tea_n, 3, _mi * 3 + _tea_ci + 1, projection="3d")
                                    _tea_sc = _tea_ax.scatter(_tea_xyz[:, 0], _tea_xyz[:, 1], _tea_xyz[:, 2], c=_tea_vals,
                                                               cmap=_tea_cmap, s=10, vmin=_tea_vmn, vmax=_tea_vmx)
                                    fig.colorbar(_tea_sc, ax=_tea_ax, shrink=0.6, pad=0.12)
                                    _tea_ax.set_title(_tea_ttl, fontsize={config.plot_subplot_title_fontsize})
                                    _tea_ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _tea_ax.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _tea_ax.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_tea_ax.xaxis, _tea_ax.yaxis, _tea_ax.zaxis)]
                            fig.suptitle(f"Step {{step_i+1}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                        else:
                            fig, axes = plt.subplots(_tea_n, 3, figsize=_panel_figsize(5, 4, 3, _tea_n), squeeze=False)
                            fig.suptitle(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}} -- PINN vs Reference", fontsize={config.plot_title_fontsize}, fontweight="bold")
                            _res_step = {config.plot_resolution}
                            _xg_s = np.linspace({config.x_min}, {config.x_max}, _res_step)
                            _yg_s = np.linspace({config.y_min}, {config.y_max}, _res_step)
                            _Xg_s, _Yg_s = np.meshgrid(_xg_s, _yg_s)
                            _inside_s = geom_i.inside(np.column_stack([_Xg_s.ravel(), _Yg_s.ravel()])).reshape(_res_step, _res_step)
                            from scipy.interpolate import griddata as _tea_gd
                            for _mi, (_mtv, _mfp, _msel) in enumerate(_step_matches):
                                _tea_d = np.loadtxt(_mfp)
                                if _tea_d.ndim == 1:
                                    _tea_d = _tea_d.reshape(1, -1)
                                _tea_xy = _tea_d[:, :2]
                                _tea_u = _tea_d[:, 3]
                                _tea_grid_pinn = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _mtv)])
                                _tea_u_pinn = _ta_ref_extract(_tea_grid_pinn, _msel, model_i).reshape(_res_step, _res_step)
                                _tea_u_pinn = np.where(_inside_s, _tea_u_pinn, np.nan)
                                _tea_u_ref = _tea_gd(_tea_xy, _tea_u, (_Xg_s, _Yg_s), method="linear", fill_value=0.0)
                                _tea_u_ref = np.where(_inside_s, _tea_u_ref, np.nan)
                                _tea_u_err = np.abs(_tea_u_pinn - _tea_u_ref)
                                _tea_vmin = np.nanmin([_tea_u_pinn, _tea_u_ref])
                                _tea_vmax = np.nanmax([_tea_u_pinn, _tea_u_ref])
                                if _tea_vmax - _tea_vmin < 1e-12:
                                    _tea_vmax = _tea_vmin + 1e-12
                                _tea_vmin, _tea_vmax = _vr(_tea_vmin, _tea_vmax)   # Plot settings -> Surface -> Colour range
                                _tea_levels = _cf_levels({config.plot_levels}, _tea_vmin, _tea_vmax)["levels"]
                                # Round 31: standardized on "|Error|  Max=..."
                                # here -- matching every other per-panel error
                                # label in the app (RAR's own round diagnostics,
                                # this same per-step plot's sibling 3D branch,
                                # and the formal Error Analysis feature) --
                                # this branch alone used to show RMSE instead,
                                # one of the inconsistent labels the user
                                # reported.
                                _tea_max_err = float(np.nanmax(_tea_u_err))
                                im0 = axes[_mi][0].contourf(_Xg_s, _Yg_s, _tea_u_pinn, levels=_tea_levels, cmap="{config.plot_colormap}", extend=_vext())
                                axes[_mi][0].set_title(f"PINN  t={{_fmt_num(_mtv)}}", fontsize={config.plot_subplot_title_fontsize})
                                axes[_mi][0].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_mi][0].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                fig.colorbar(im0, ax=axes[_mi][0])
                                im1 = axes[_mi][1].contourf(_Xg_s, _Yg_s, _tea_u_ref, levels=_tea_levels, cmap="{config.plot_colormap}", extend=_vext())
                                axes[_mi][1].set_title(f"Reference  t={{_fmt_num(_mtv)}}", fontsize={config.plot_subplot_title_fontsize})
                                axes[_mi][1].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_mi][1].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                fig.colorbar(im1, ax=axes[_mi][1])
                                im2 = axes[_mi][2].contourf(_Xg_s, _Yg_s, _tea_u_err, levels={config.plot_levels}, cmap="{err_cmap(config)}")
                                axes[_mi][2].set_title(f"|Error|  max = {{_sci_txt(_tea_max_err)}}", fontsize={config.plot_subplot_title_fontsize})
                                axes[_mi][2].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_mi][2].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                fig.colorbar(im2, ax=axes[_mi][2])
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    else:
                        _vmin_step = None if {config.plot_auto_range} else {config.plot_vmin}
                        _vmax_step = None if {config.plot_auto_range} else {config.plot_vmax}
                        _res_step = {config.plot_resolution}
                        _xg_s = np.linspace({config.x_min}, {config.x_max}, _res_step)
                        _yg_s = np.linspace({config.y_min}, {config.y_max}, _res_step)
                        _Xg_s, _Yg_s = np.meshgrid(_xg_s, _yg_s)
                        if _is_3d:
                            _z_mid_s = {line_slice_z}
                            fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(10, 4))
                            for _ai, _tv_s in enumerate([t0, t1]):
                                _xyt_s = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _z_mid_s), np.full(_Xg_s.size, _tv_s)])
                                _U_s = _extract_plot_field(_xyt_s, model_i).reshape(_res_step, _res_step)
                                im = axes[_ai].contourf(_Xg_s, _Yg_s, _U_s, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_step, _vmax_step, _U_s))
                                if {config.plot_colorbar}:
                                    fig.colorbar(im, ax=axes[_ai])
                                axes[_ai].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ai].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                axes[_ai].set_title(f"t = {{_fmt_num(_tv_s)}}  (z={{_fmt_num(_z_mid_s)}})")
                            fig.suptitle(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}}", fontsize={config.plot_title_fontsize})
                        else:
                            fig, axes = plt.subplots(1, 2, figsize=_plot_figsize(10, 4))
                            for _ai, _tv_s in enumerate([t0, t1]):
                                _xyt_s = np.column_stack([_Xg_s.ravel(), _Yg_s.ravel(), np.full(_Xg_s.size, _tv_s)])
                                _U_s = _extract_plot_field(_xyt_s, model_i).reshape(_res_step, _res_step)
                                im = axes[_ai].contourf(_Xg_s, _Yg_s, _U_s, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_step, _vmax_step, _U_s))
                                if {config.plot_colorbar}:
                                    fig.colorbar(im, ax=axes[_ai])
                                axes[_ai].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ai].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                axes[_ai].set_title(f"t = {{_fmt_num(_tv_s)}}")
                            fig.suptitle(f"Step {{step_i+1}}: t = {{_fmt_num(t0)}} → {{_fmt_num(t1)}}", fontsize={config.plot_title_fontsize})
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    plt.savefig(_step_fname, dpi={config.plot_dpi}, bbox_inches="tight")
                    plt.close()
                    print(f"Step plot saved: {{_step_fname}}")
                except Exception as _tea_err_23d:
                    print(f"  Step {{step_i+1}} plot failed: {{_tea_err_23d}}")

            # ── Final model already saved per-phase, just track for transfer learning ───
            _last_phase_opt = "{config.optimizer}"
            if {config.optimizer_scheduler} and {len(config.scheduler_phases) > 0}:
                import json as _json_lp
                _lp_phases = _json_lp.loads({repr(config.scheduler_phases)})
                if _lp_phases:
                    _last_phase_opt = _lp_phases[-1]["optimizer"]

            # Track for transfer learning — prefer lbfgs if chosen, else adam
            _prev_step_model_path = ""
            import glob as _tl_glob
            if "{config.ta_transfer_optimizer}" == "lbfgs":
                # Look for any lbfgs .pt file (may have iteration number suffix)
                _lbfgs_pts = sorted(_tl_glob.glob(_os.path.join(_step_dir, "model_lbfgs*.pt")))
                if _lbfgs_pts:
                    _prev_step_model_path = max(_lbfgs_pts, key=_os.path.getmtime)
                    print(f"  Transfer learning source (lbfgs): {{_os.path.basename(_prev_step_model_path)}}")
            if not _prev_step_model_path:
                # Fall back to adam
                _adam_pts = sorted(_tl_glob.glob(_os.path.join(_step_dir, "model_adam*.pt")))
                if _adam_pts:
                    _prev_step_model_path = max(_adam_pts, key=_os.path.getmtime)
                    print(f"  Transfer learning source (adam): {{_os.path.basename(_prev_step_model_path)}}")

            # without a save folder (no checkpoints): the network kept in memory
            _prev_step_state = {{_k: _v.detach().clone() for _k, _v in model_i.net.state_dict().items()}}

            # ── Save step config JSON ─────────────────────────
            # y_min/y_max/z_min/z_max added alongside x_min/x_max so a 2D/3D
            # Time-Adaptive step's own model_config-equivalent is complete
            # on its own (previously only x_min/x_max were saved here --
            # harmless for 1D, since restore already defaults those two,
            # but silently wrong for 2D/3D if this file were ever used as a
            # restore config directly, since the y/z bounds would fall back
            # to the [0,1] default instead of this run's real domain).
            _step_cfg = {{
                "step": step_i + 1,
                "t0": t0, "t1": t1,
                "layers": {config.layers},
                "activation": "{config.activation}",
                "x_min": {config.x_min}, "x_max": {config.x_max},
                "y_min": {config.y_min}, "y_max": {config.y_max},
                "z_min": {config.z_min}, "z_max": {config.z_max},
                "t_min": t0, "t_max": t1,
                "problem_dim": "{config.problem_dim}",
                "loss_type": "{config.loss_type}",
                "optimizer": "{config.optimizer}",
                "optimizer2": "{config.optimizer2}",
                "output_names": {repr(config.output_names)},
                "num_outputs": {config.num_outputs},
            }}
            with open(_os.path.join(_step_dir, "step_config.json"), "w") as _scf:
                _json.dump(_step_cfg, _scf, indent=2)
            print(f"Step config saved: {{_os.path.join(_step_dir, 'step_config.json')}}")

    X_full = np.vstack(all_x); T_full = np.vstack(all_t); U_full = np.vstack(all_u)
    print("\\n=== Time-Adaptive Training Complete ===")
    try:            # the last time step's losses, as a sweep reads them
        _fl_ta, _fl_ta_unw = _final_losses(model_i, lh_i)
        print(f"Final loss: {{_fl_ta:.4e}}")
        print(f"Final loss (unweighted): {{_fl_ta_unw:.4e}}")
    except Exception:
        pass

    _ta_solution_path = _os.path.join(_sol_dir, "solution_plot.png")
    _ta_loss_path     = _os.path.join(_sol_dir, "loss_plot.png")

    _plot_type_ta = "{config.plot_type}"
    if _plot_type_ta not in ("Surface", "Line (time steps)"):
        # Same fallback reasoning as the per-step preview above -- GIF
        # animations and Parameter Convergence aren't wired up for
        # Time-Adaptive's final solution plot yet.
        _plot_type_ta = "Surface"
    if _plot_type_ta == "Surface":
        _vmin_ta = None if {config.plot_auto_range} else {config.plot_vmin}
        _vmax_ta = None if {config.plot_auto_range} else {config.plot_vmax}
        if _is_2d:
            # 2D: x-y heatmaps at n snapshots using the last step model
            _n_snaps_ta = {config.plot_n_2d_snapshots}
            _t_snaps_ta = np.linspace({config.t_min}, {config.t_max}, _n_snaps_ta)
            _res_ta = {config.plot_resolution}
            _xp_ta = np.linspace({config.x_min}, {config.x_max}, _res_ta)
            _yp_ta = np.linspace({config.y_min}, {config.y_max}, _res_ta)
            _Xg_ta, _Yg_ta = np.meshgrid(_xp_ta, _yp_ta)
            # cleanup Round 44: masked to the real geometry with
            # geom.inside(...) -- same convention the Standard path's own
            # 2D Surface branch already uses (geom here is the real
            # reconstructed geometry built at module scope above, NOT the
            # plain Rectangle built a few lines below for each step's
            # throwaway dde.Model -- that Rectangle only exists to satisfy
            # DeepXDE's Model/TimePDE constructor so .predict() works, it
            # was never meant to represent the true domain). Spatial-only,
            # so computed once and reused for every snapshot panel.
            _inside_ta = np.asarray(geom.inside(np.column_stack([_Xg_ta.ravel(), _Yg_ta.ravel()]))).reshape(_res_ta, _res_ta)
            # Collect all step models to predict at each snapshot time
            import glob as _ta_glob_sol, json as _ta_json_sol
            _ta_sol_dirs = sorted([_sd for _sd in _ta_glob_sol.glob(_os.path.join(_ta_steps_root, "step_*")) if _os.path.isdir(_sd)])
            _ta_sol_intervals = []
            for _sd in _ta_sol_dirs:
                try:
                    _p = _os.path.basename(_sd).split("_")
                    _ta_sol_intervals.append((float(_p[2].replace("t","")), float(_p[4].replace("t","")), _sd))
                except Exception: pass
            fig, axes = plt.subplots(1, _n_snaps_ta, figsize=(_plot_figsize(5, 5)[0]*_n_snaps_ta, _plot_figsize(5, 5)[1]))
            if _n_snaps_ta == 1: axes = [axes]
            for _ai, _tv_ta in enumerate(_t_snaps_ta):
                # Find which step model covers this time
                _sd_for_t = _ta_sol_dirs[-1] if _ta_sol_dirs else ""
                for _t0s, _t1s, _sds in _ta_sol_intervals:
                    if _t0s <= _tv_ta <= _t1s + 1e-10:
                        _sd_for_t = _sds; break
                # Load step model
                _spt_ta = ""
                for _pat_ta in ["model_lbfgs-*.pt","model_lbfgs.pt","model_adam-*.pt","model_adam.pt"]:
                    _pts_ta = sorted(_ta_glob_sol.glob(_os.path.join(_sd_for_t, _pat_ta)))
                    if _pts_ta: _spt_ta = max(_pts_ta, key=_os.path.getmtime); break
                try:
                    with open(_os.path.join(_sd_for_t, "step_config.json")) as _scf_ta:
                        _sc_ta = _ta_json_sol.load(_scf_ta)
                except: _sc_ta = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}
                _sn_ta = _apply_net_transforms(_make_net(_sc_ta.get("layers",{config.layers}), _sc_ta.get("activation","{config.activation}"), "Glorot uniform"))
                _sg_ta = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
                _st_ta = dde.geometry.TimeDomain(_sc_ta.get("t_min",0), _sc_ta.get("t_max",1))
                _sgt_ta = dde.geometry.GeometryXTime(_sg_ta, _st_ta)
                _sd_ta = dde.data.TimePDE(_sgt_ta, lambda x,y: y[:,0:1]*0, [], num_domain=100, num_test=100)
                _sm_ta = dde.Model(_sd_ta, _sn_ta)
                if _spt_ta:
                    if "lbfgs" in _os.path.basename(_spt_ta):
                        dde.optimizers.set_LBFGS_options(maxiter=1)
                        _sm_ta.compile("L-BFGS", loss=_sc_ta.get("loss_type","{config.loss_type}"))
                    else:
                        _sm_ta.compile("adam", lr=0.001, loss=_sc_ta.get("loss_type","{config.loss_type}"))
                    _sm_ta.restore(_spt_ta, verbose=0)
                    _xyt_ta = np.column_stack([_Xg_ta.ravel(), _Yg_ta.ravel(), np.full(_Xg_ta.size, _tv_ta)])
                    _pred_ta = _sm_ta.predict(_xyt_ta)[:, {config.plot_output_idx}].reshape(_res_ta, _res_ta)
                else:
                    _pred_ta = np.zeros((_res_ta, _res_ta))
                _pred_ta = np.where(_inside_ta, _pred_ta, np.nan)
                im = axes[_ai].contourf(_Xg_ta, _Yg_ta, _pred_ta, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_ta, _vmax_ta, _pred_ta))
                axes[_ai].set_title(f"t = {{_fmt_num(_tv_ta)}}")
                axes[_ai].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ai].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
            fig.suptitle("Time-Adaptive PINN Solution", fontsize={config.plot_title_fontsize})
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
        else:
            # 1D, and 3D too (its per-step loop above already fixed y and z
            # at their domain mid-points and stitched an x-t slice into
            # X_full/T_full/U_full, same shape as the 1D case) -- no
            # volumetric renderer, so this mid-plane slice is the final
            # solution plot for 3D as well.
            fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
            # Same configurable axis orientation as the Standard path's 1D
            # "Surface" plot -- see the matching comment there.
            if {config.plot_swap_xt}:
                im = ax.contourf(T_full, X_full, U_full, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_ta, _vmax_ta, U_full))
                ax.set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
            else:
                im = ax.contourf(X_full, T_full, U_full, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_ta, _vmax_ta, U_full))
                ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("t", fontsize={config.plot_axis_label_fontsize})
            if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
            ax.set_title("Time-Adaptive PINN Solution")
            plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
    elif _plot_type_ta.startswith("Line"):
        n_ts   = {config.num_timesteps_line}
        t_vals = np.linspace(_ta_flat_intervals[0][0], _ta_flat_intervals[-1][1], n_ts)
        fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
        colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_ts))
        for ci, tv in enumerate(t_vals):
            idx = np.argmin(np.abs(T_full[:,0] - tv))
            ax.plot(X_full[idx,:], U_full[idx,:], color=colors[ci], linewidth={config.plot_linewidth}, label=f"t={{_fmt_num(tv)}}")
        _ylabel_ta_line = (f"u(x,y={{_fmt_num({line_slice_y!r})}},z={{_fmt_num({line_slice_z!r})}},t)" if _is_3d
                           else (f"u(x,y={line_slice_y:.3g},t)" if _is_2d else "u(x,t)"))
        ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_ylabel_ta_line, fontsize={config.plot_axis_label_fontsize})
        ax.set_title("Time-Adaptive PINN — Line Plot")
        ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
        plt.tight_layout(); plt.savefig(_ta_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

    # Full Time-Adaptive run -- every optimizer phase of every time
    # sub-domain, concatenated with a running iteration offset (see
    # "_ta_accumulate_loss" above) -- not just whichever phase happened
    # to run last for the very last sub-domain.
    train_loss_ta = _ta_all_train_loss; test_loss_ta = _ta_all_test_loss
    steps_ta = _ta_all_steps
    # Light vertical markers at each time sub-domain's boundary, so a
    # jump/kink in the loss (a brand-new network starting that
    # sub-domain, warm-started only via its IC) is visually distinguishable
    # from an actual optimizer-related loss spike. Skip the very last
    # boundary (the run's own end -- nothing follows it worth marking) and
    # don't bother with a boundary at 0 if the first sub-domain starts
    # right there.
    def _ta_loss_boundary_markers(_ax):
        for _tb in _ta_step_boundaries[:-1]:
            if _tb > 0:
                _ax.axvline(_tb, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    _make_loss_plot(train_loss_ta, test_loss_ta, steps_ta, _ta_loss_path,
                     f"Loss — All {{n_steps}} Time Sub-domain(s)",
                     xlabel="Iteration (cumulative across all time sub-domains)",
                     extra_fn=_ta_loss_boundary_markers)

    # ── Time Adaptive Error Analysis ──────────────────────────
    if {config.ea_files}:
        from scipy.interpolate import interp1d as _interp1d
        import glob as _ea_glob, json as _ea_json
        _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
        _os.makedirs(_ea_dir, exist_ok=True)

{_gs_ea_setup_code(config, "Time-Adaptive Error Analysis", "        ")}

        for _ea_group_key, _ea_group in _ea_groups.items():
            _ea_files = _ea_group["files"]
            _ea_sel = _ea_group["sel"]
            _ea_suffix = f"_{{_ea_group_label(_ea_sel) or 'default'}}" if _ea_multi_output else ""
            if _ea_multi_output:
                print(f"  ── Output group: {{_ea_group_label(_ea_sel) or 'default'}} ({{len(_ea_files)}} files) ──")

            # Load all ground truth files
            _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_z_refs = []; _ea_u_refs = []
            for _ea_tv, _ea_fp in _ea_files:
                _ea_d = np.loadtxt(_ea_fp)
                if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
                if _is_3d:
                    # 3D format: x, y, z, t, u -- same as the Standard path's
                    # own non-steady 3D Error Analysis loader.
                    _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(_ea_d[_ea_idx, 1])
                    _ea_z_refs.append(_ea_d[_ea_idx, 2])
                    _ea_u_refs.append(_ea_d[_ea_idx, 4])
                    _detected_t = float(_ea_d[0, 3])
                    _ea_times.append(_detected_t)
                    print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                elif _is_2d:
                    _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(_ea_d[_ea_idx, 1])
                    _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_u_refs.append(_ea_d[_ea_idx, 3])
                    _detected_t = float(_ea_d[0, 2])
                    _ea_times.append(_detected_t)
                    print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                else:
                    _ea_idx = np.argsort(_ea_d[:, 0])
                    _ea_x_refs.append(_ea_d[_ea_idx, 0])
                    _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                    _ea_u_refs.append(_ea_d[_ea_idx, 2])
                    _ea_times.append(float(_ea_tv))
                    print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
            # Sort by time
            _ea_sort_idx = np.argsort(_ea_times)
            _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
            _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
            _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
            _ea_z_refs = [_ea_z_refs[_i] for _i in _ea_sort_idx]
            _ea_u_refs = [_ea_u_refs[_i] for _i in _ea_sort_idx]
            _ea_n_t = len(_ea_times)
            _ea_u_pinns = [None] * _ea_n_t

            # Find all step directories
            _ta_step_dir = _os.path.join(_save_dir, "time_adaptive_steps")
            _ta_step_dirs = sorted([_sd for _sd in _ea_glob.glob(_os.path.join(_ta_step_dir, "step_*")) if _os.path.isdir(_sd)])
            _ta_intervals = []
            for _sd in _ta_step_dirs:
                _sd_name = _os.path.basename(_sd)
                try:
                    _parts = _sd_name.split("_")
                    _t0_str = _parts[2].replace("t","")
                    _t1_str = _parts[4].replace("t","")
                    _ta_intervals.append((float(_t0_str), float(_t1_str), _sd))
                except Exception as _pe:
                    print(f"  Could not parse step dir: {{_sd_name}}: {{_pe}}")
            print(f"  Found {{len(_ta_intervals)}} time-adaptive step models")

            for _si, (_t0_i, _t1_i, _sd_i) in enumerate(_ta_intervals):
                _is_last = (_si == len(_ta_intervals) - 1)
                _matching = []
                for _ei, _ea_tv in enumerate(_ea_times):
                    if _is_last:
                        _in_range = (_t0_i <= _ea_tv <= _t1_i + 1e-10)
                    else:
                        _in_range = (_t0_i <= _ea_tv < _t1_i - 1e-10) or \
                                    (abs(_ea_tv - _t1_i) < 1e-10)
                    if _in_range and _ea_u_pinns[_ei] is None:
                        _matching.append(_ei)

                if not _matching:
                    continue

                print(f"  Step {{_si+1}} [{{_t0_i:.4f}}→{{_t1_i:.4f}}]: t = {{[_ea_times[_ei] for _ei in _matching]}}")

                _step_cfg_path = _os.path.join(_sd_i, "step_config.json")
                try:
                    with open(_step_cfg_path) as _scf:
                        _step_cfg = _ea_json.load(_scf)
                except Exception:
                    _step_cfg = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}

                _step_layers = _step_cfg.get("layers", {config.layers})
                _step_act    = _step_cfg.get("activation", "{config.activation}")
                _step_loss   = _step_cfg.get("loss_type", "{config.loss_type}")

                if _is_3d:
                    _step_geom = dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}],
                                                       [{config.x_max}, {config.y_max}, {config.z_max}])
                elif _is_2d:
                    _step_geom = dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
                else:
                    _step_geom = dde.geometry.Interval({config.x_min}, {config.x_max})
                _step_td    = dde.geometry.TimeDomain(_t0_i, _t1_i)
                _step_gt    = dde.geometry.GeometryXTime(_step_geom, _step_td)
                def _step_pde(x, y): return y[:, 0:1] * 0
                _step_data  = dde.data.TimePDE(_step_gt, _step_pde, [], num_domain=100, num_test=100)
                _step_net   = _apply_net_transforms(_make_net(_step_layers, _step_act, "Glorot uniform"))
                _step_model = dde.Model(_step_data, _step_net)

                _step_pt = ""
                for _pat in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                    _step_pts = sorted(_ea_glob.glob(_os.path.join(_sd_i, _pat)))
                    if _step_pts:
                        _step_pt = max(_step_pts, key=_os.path.getmtime)
                        break

                if not _step_pt:
                    print(f"  ⚠️ No model found for step {{_si+1}}, skipping")
                    continue

                if "lbfgs" in _os.path.basename(_step_pt):
                    dde.optimizers.set_LBFGS_options(maxiter=1)
                    _step_model.compile("L-BFGS", loss=_step_loss)
                else:
                    _step_model.compile("adam", lr=0.001, loss=_step_loss)

                _step_model.restore(_step_pt, verbose=0)
                print(f"    Restored: {{_os.path.basename(_step_pt)}}")

                for _ei in _matching:
                    _ea_xf = _ea_x_refs[_ei]
                    _ea_tv = _ea_times[_ei]
                    if _is_3d:
                        _ea_yf = _ea_y_refs[_ei]; _ea_zf = _ea_z_refs[_ei]
                        _ea_xt = np.column_stack([_ea_xf, _ea_yf, _ea_zf, np.full_like(_ea_xf, _ea_tv)])
                    elif _is_2d:
                        _ea_yf = _ea_y_refs[_ei]
                        _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                    else:
                        _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                    _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, _step_model).flatten()
                    print(f"    Predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")

            for _ei in range(_ea_n_t):
                if _ea_u_pinns[_ei] is None:
                    print(f"  ⚠️ No prediction for t={{_ea_times[_ei]:.4f}} — zero fill")
                    _ea_u_pinns[_ei] = np.zeros_like(_ea_u_refs[_ei])

            # Metrics
            _ea_metrics = []
            for _ei, _ea_tv in enumerate(_ea_times):
                _up = _ea_u_pinns[_ei]; _uf = _ea_u_refs[_ei]
                _ea_abs = np.abs(_up - _uf)
                _ea_l2  = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
                _ea_mse = np.mean((_up - _uf)**2)
                _ea_mx  = np.max(_ea_abs)
                _ea_ma  = np.mean(_ea_abs)
                _ea_metrics.append((_ea_tv, _ea_l2, _ea_mse, _ea_mx, _ea_ma))
                print(f"  t={{_ea_tv:.4f}} — {_ea_metrics_print_fragment(config, '_ea_l2', '_ea_mse', '_ea_mx', '_ea_ma')}")

            _ea_metrics_path = _os.path.join(_ea_dir, f"error_metrics{{_ea_suffix}}.txt")
            with open(_ea_metrics_path, "w") as _emf:
                _emf.write("{_ea_metrics_csv_header(config)}\\n")
                for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                    _emf.write(f"{_ea_metrics_row_code(config, '_ea_tv', '_l2', '_mse', '_mx', '_ma')}\\n")
            print(f"  Metrics saved: {{_ea_metrics_path}}")

            # Line comparison
            if {config.ea_do_line}:
                _ea_ncols = min({max(1, int(getattr(config, 'plot_ea_cols', 4) or 4))}, _ea_n_t)
                _ea_nrows = (_ea_n_t + _ea_ncols - 1) // _ea_ncols
                fig, axes = plt.subplots(_ea_nrows, _ea_ncols, figsize=_panel_figsize(4, 3.5, _ea_ncols, _ea_nrows), squeeze=False)
                _ea_line_suptitle_ta = _ea_title_txt("Line Comparison", _ea_sel)
                if _is_3d:
                    _ea_line_suptitle_ta += f" (y={{_fmt_num({line_slice_y!r})}}, z={{_fmt_num({line_slice_z!r})}})"
                elif _is_2d:
                    _ea_line_suptitle_ta += f" (y={{_fmt_num({line_slice_y!r})}})"
                fig.suptitle(_ea_line_suptitle_ta.replace(" — Line Comparison", ""), fontsize={config.plot_subplot_title_fontsize} + 2)
                _ea_ax_flat = axes.flatten()
                if _is_2d or _is_3d:
                    # Same true-line-at-the-exact-slice fix as the Standard
                    # path's own Line Comparison above (see its comment for
                    # the full rationale) -- adapted for Time-Adaptive, which
                    # has no single `model`: each time snapshot's PINN curve
                    # must come from whichever step model actually covers
                    # that snapshot's time, not an unfiltered scatter of
                    # every reference point regardless of y(/z) (the old
                    # behavior here, which didn't even have the Standard
                    # path's tolerance-band narrowing -- it was strictly
                    # worse). `_ta_intervals` was already built above for
                    # the per-time prediction loop; reused here rather than
                    # re-scanning the step directories again.
                    from scipy.interpolate import griddata as _gd_line_ta
                    _xv_line_ta = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                    _y_mid_ta = {line_slice_y}
                    _z_mid_ta = {line_slice_z}
                    _ea_geom_line_ta = _build_geom()
                    _step_dir_map_line = {{}}
                    for _si3, (_t0_i3, _t1_i3, _sd_i3) in enumerate(_ta_intervals):
                        _is_last3 = (_si3 == len(_ta_intervals) - 1)
                        for _ei3 in range(_ea_n_t):
                            _tv3 = _ea_times[_ei3]
                            if _is_last3:
                                _in3 = (_t0_i3 <= _tv3 <= _t1_i3 + 1e-10)
                            else:
                                _in3 = (_t0_i3 <= _tv3 < _t1_i3 - 1e-10) or (abs(_tv3 - _t1_i3) < 1e-10)
                            if _in3:
                                _step_dir_map_line[_ei3] = _sd_i3
                for _ei in range(_ea_n_t):
                    ax = _ea_ax_flat[_ei]
                    if _is_2d or _is_3d:
                        if _is_3d:
                            _line_pts_ta = np.column_stack([_xv_line_ta, np.full_like(_xv_line_ta, _y_mid_ta), np.full_like(_xv_line_ta, _z_mid_ta)])
                        else:
                            _line_pts_ta = np.column_stack([_xv_line_ta, np.full_like(_xv_line_ta, _y_mid_ta)])
                        _line_in_ta = _ea_geom_line_ta.inside(_line_pts_ta)
                        _xv_s = _xv_line_ta[_line_in_ta]
                        _line_pts_ta_in = _line_pts_ta[_line_in_ta]
                        if len(_xv_s) < 2:
                            ax.text(0.5, 0.5, "Line outside the domain at this (y, z) slice" if _is_3d else "Line outside the domain at this y slice",
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.set_xticks([]); ax.set_yticks([])
                            continue
                        _sd_for_line = _step_dir_map_line.get(_ei, "")
                        _pinn_s = None
                        if _sd_for_line:
                            try:
                                with open(_os.path.join(_sd_for_line, "step_config.json")) as _scfl:
                                    _sc_l = _ea_json.load(_scfl)
                            except Exception:
                                _sc_l = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}
                            _net_l = _apply_net_transforms(_make_net(_sc_l.get("layers", {config.layers}), _sc_l.get("activation", "{config.activation}"), "Glorot uniform"))
                            if _is_3d:
                                _geom_l = dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}], [{config.x_max}, {config.y_max}, {config.z_max}])
                            else:
                                _geom_l = dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
                            _td_l = dde.geometry.TimeDomain(_sc_l.get("t_min", 0), _sc_l.get("t_max", 1))
                            _gt_l = dde.geometry.GeometryXTime(_geom_l, _td_l)
                            _data_l = dde.data.TimePDE(_gt_l, lambda x, y: y[:, 0:1] * 0, [], num_domain=100, num_test=100)
                            _model_l = dde.Model(_data_l, _net_l)
                            _pt_l = ""
                            for _pat_l in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                                _pts_l = sorted(_ea_glob.glob(_os.path.join(_sd_for_line, _pat_l)))
                                if _pts_l:
                                    _pt_l = max(_pts_l, key=_os.path.getmtime)
                                    break
                            if _pt_l:
                                if "lbfgs" in _os.path.basename(_pt_l):
                                    dde.optimizers.set_LBFGS_options(maxiter=1)
                                    _model_l.compile("L-BFGS", loss=_sc_l.get("loss_type", "{config.loss_type}"))
                                else:
                                    _model_l.compile("adam", lr=0.001, loss=_sc_l.get("loss_type", "{config.loss_type}"))
                                _model_l.restore(_pt_l, verbose=0)
                                _line_eval_ta = np.column_stack([_line_pts_ta_in, np.full(len(_xv_s), _ea_times[_ei])])
                                _pinn_s = _ea_extract(_line_eval_ta, _ea_sel, _model_l).flatten()
                        if _pinn_s is None:
                            _pinn_s = np.zeros(len(_xv_s))
                        _ref_pts_ta = (np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei], _ea_z_refs[_ei]]) if _is_3d
                                       else np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]))
                        _gt_s = _gd_line_ta(_ref_pts_ta, _ea_u_refs[_ei], _line_pts_ta_in, method='linear')
                        _gt_nan_ta = np.isnan(_gt_s)
                        if _gt_nan_ta.any():
                            _gt_s[_gt_nan_ta] = _gd_line_ta(_ref_pts_ta, _ea_u_refs[_ei], _line_pts_ta_in[_gt_nan_ta], method='nearest')
                    else:
                        _xv = _ea_x_refs[_ei]
                        _ea_sort = np.argsort(_xv)
                        _xv_s = _xv[_ea_sort]
                        _gt_s = _ea_u_refs[_ei][_ea_sort]
                        _pinn_s = _ea_u_pinns[_ei][_ea_sort]
                    _ea_tv, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                    if _is_2d or _is_3d:
                        _xv_s, _gt_s, _pinn_s = _line_gaps(_xv_s, _gt_s, _pinn_s)
                    _ref_plot(ax, _xv_s, _gt_s,   color=_PLOT_REF_COLOR, linewidth={config.plot_linewidth}, linestyle='-',  label='Reference')
                    ax.plot(_xv_s, _pinn_s, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth}, linestyle='--', label='PINN')
                    ax.set_title(f"t = {{_fmt_num(_ea_tv)}}  ({{_l2_txt(_l2)}})", fontsize={config.plot_subplot_title_fontsize})
                    ax.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("u(x,t)", fontsize={config.plot_axis_label_fontsize}); _grid(ax, alpha=0.3)
                for _ej in range(_ea_n_t, len(_ea_ax_flat)):
                    _ea_ax_flat[_ej].set_visible(False)
                _ea_ax_flat[0].legend(loc='best', fontsize=_lfs(10), framealpha=0.9)
                plt.tight_layout()
                _ea_lp = _os.path.join(_ea_dir, f"line_comparison{{_ea_suffix}}.png")
                plt.savefig(_ea_lp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                print(f"  Line comparison saved: {{_ea_lp}}")

            # Surface comparison
            if {config.ea_do_surface}:
                if _is_2d:
                    # 2D: PINN | Reference | Error heatmaps, one row per time snapshot
                    from scipy.interpolate import griddata as _gd
                    _res_ea = {config.plot_resolution}
                    _xg_ea = np.linspace({config.x_min}, {config.x_max}, _res_ea)
                    _yg_ea = np.linspace({config.y_min}, {config.y_max}, _res_ea)
                    _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
                    fig, axes = plt.subplots(_ea_n_t, 3, figsize=_panel_figsize(5, 4, 3, _ea_n_t), squeeze=False)
                    fig.suptitle(_ea_title_txt("2D Heatmaps", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                    # Need step models to predict on grid — collect from step dirs
                    import glob as _ea_glob2, json as _ea_json2
                    _ta_step_dir2 = _os.path.join(_save_dir, "time_adaptive_steps")
                    _ta_step_dirs2 = sorted([_sd for _sd in _ea_glob2.glob(_os.path.join(_ta_step_dir2, "step_*")) if _os.path.isdir(_sd)])
                    _ta_intervals2 = []
                    for _sd in _ta_step_dirs2:
                        try:
                            _parts = _os.path.basename(_sd).split("_")
                            _ta_intervals2.append((float(_parts[2].replace("t","")), float(_parts[4].replace("t","")), _sd))
                        except Exception: pass
                    # Map each time to its step model
                    _step_model_map = {{}}
                    for _si2, (_t0_i2, _t1_i2, _sd_i2) in enumerate(_ta_intervals2):
                        for _ei in range(_ea_n_t):
                            _tv = _ea_times[_ei]
                            _is_last2 = (_si2 == len(_ta_intervals2) - 1)
                            if _is_last2:
                                _in = (_t0_i2 <= _tv <= _t1_i2 + 1e-10)
                            else:
                                _in = (_t0_i2 <= _tv < _t1_i2 - 1e-10) or (abs(_tv - _t1_i2) < 1e-10)
                            if _in:
                                _step_model_map[_ei] = _sd_i2
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                        # PINN on grid using step model
                        _sd_for_ei = _step_model_map.get(_ei, "")
                        _xyt_grid = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _ea_tv)])
                        if _sd_for_ei:
                            try:
                                with open(_os.path.join(_sd_for_ei, "step_config.json")) as _scf2: _sc2 = _ea_json2.load(_scf2)
                            except: _sc2 = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}
                            _sn2 = _apply_net_transforms(_make_net(_sc2.get("layers",{config.layers}), _sc2.get("activation","{config.activation}"), "Glorot uniform"))
                            _sg2 = dde.geometry.Rectangle([{config.x_min},{config.y_min}],[{config.x_max},{config.y_max}])
                            _st2 = dde.geometry.TimeDomain(_sc2.get("t_min",0), _sc2.get("t_max",1))
                            _sgt2 = dde.geometry.GeometryXTime(_sg2, _st2)
                            _sd2 = dde.data.TimePDE(_sgt2, lambda x,y: y[:,0:1]*0, [], num_domain=100, num_test=100)
                            _sm2 = dde.Model(_sd2, _sn2)
                            _spt2 = ""
                            for _pat2 in ["model_lbfgs-*.pt","model_lbfgs.pt","model_adam-*.pt","model_adam.pt"]:
                                _pts2 = sorted(_ea_glob2.glob(_os.path.join(_sd_for_ei, _pat2)))
                                if _pts2: _spt2 = max(_pts2, key=_os.path.getmtime); break
                            if _spt2:
                                if "lbfgs" in _os.path.basename(_spt2):
                                    dde.optimizers.set_LBFGS_options(maxiter=1)
                                    _sm2.compile("L-BFGS", loss=_sc2.get("loss_type","{config.loss_type}"))
                                else:
                                    _sm2.compile("adam", lr=0.001, loss=_sc2.get("loss_type","{config.loss_type}"))
                                _sm2.restore(_spt2, verbose=0)
                                _u_pinn_grid = _ea_extract(_xyt_grid, _ea_sel, _sm2).reshape(_res_ea, _res_ea)
                            else:
                                _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                        else:
                            _u_pinn_grid = np.zeros((_res_ea, _res_ea))
                        _u_fem_grid = _gd(np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                                          _ea_u_refs[_ei], (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                        _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                        _vmin_ea = min(_u_pinn_grid.min(), _u_fem_grid.min())
                        _vmax_ea = max(_u_pinn_grid.max(), _u_fem_grid.max())
                        if _vmax_ea - _vmin_ea < 1e-12:
                            _vmax_ea = _vmin_ea + 1e-12
                        _vmin_ea, _vmax_ea = _vr(_vmin_ea, _vmax_ea)   # Plot settings -> Surface -> Colour range
                        _levels_ea2d = _cf_levels({config.plot_levels}, _vmin_ea, _vmax_ea)["levels"]  # contourf ignores vmin/vmax with an integer `levels=N` -- see the Standard-path comment on the same pattern above
                        im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels=_levels_ea2d, cmap='{config.plot_colormap}', extend=_vext())
                        axes[_ei][0].set_title(f"PINN  t={{_fmt_num(_ea_tv)}}  {{_l2_txt(_l2)}}", fontsize={config.plot_subplot_title_fontsize})
                        axes[_ei][0].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][0].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im0, ax=axes[_ei][0])
                        im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels=_levels_ea2d, cmap='{config.plot_colormap}', extend=_vext())
                        axes[_ei][1].set_title(f"Reference  t={{_fmt_num(_ea_tv)}}", fontsize={config.plot_subplot_title_fontsize})
                        axes[_ei][1].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][1].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im1, ax=axes[_ei][1])
                        im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='{err_cmap(config)}')
                        axes[_ei][2].set_title(f"|Error|  t={{_fmt_num(_ea_tv)}}  max = {{_sci_txt(_mx)}}", fontsize={config.plot_subplot_title_fontsize})
                        axes[_ei][2].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][2].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im2, ax=axes[_ei][2])
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                    _ea_did_surface_ta = True
                elif _is_3d:
                    # 3D: same "boundary-point scatter" convention as the
                    # Standard path's non-box-geometry 3D surface comparison
                    # -- no flat-face grid to heatmap onto in general, so
                    # compare real (x, y, z) points that lie on the
                    # geometry's own boundary. Reuses the per-time PINN
                    # predictions already computed above (_ea_u_pinns) --
                    # each one came from that time's own correctly-restored
                    # step model -- instead of reloading a step model again
                    # here just to predict on a dense grid, as the 2D branch
                    # above does.
                    _ea_geom_ta = _build_geom()
                    _ea_bbox_ta = np.asarray(_ea_geom_ta.bbox)
                    _bx0_ta, _by0_ta, _bz0_ta = _ea_bbox_ta[0]
                    _bx1_ta, _by1_ta, _bz1_ta = _ea_bbox_ta[1]
                    _corners_ta = [(_bx0_ta,_by0_ta,_bz0_ta),(_bx1_ta,_by0_ta,_bz0_ta),(_bx1_ta,_by1_ta,_bz0_ta),(_bx0_ta,_by1_ta,_bz0_ta),
                                    (_bx0_ta,_by0_ta,_bz1_ta),(_bx1_ta,_by0_ta,_bz1_ta),(_bx1_ta,_by1_ta,_bz1_ta),(_bx0_ta,_by1_ta,_bz1_ta)]
                    _edges_ta = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    fig = plt.figure(figsize=_panel_figsize(5, 4.5, 3, _ea_n_t))
                    fig.suptitle(_ea_title_txt("3D Comparison", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                        _bnd_ta = _ea_geom_ta.on_boundary(np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei], _ea_z_refs[_ei]]))
                        # A Cartesian reference grid only reliably intersects a
                        # FLAT boundary (a Cuboid's faces) in large numbers; a
                        # CURVED boundary (a Sphere) is essentially never
                        # landed on exactly by an axis-aligned grid, so
                        # on_boundary() can come back with only a handful of
                        # coincidental exact matches (e.g. a unit sphere's 6
                        # axis points) even when the dataset has thousands of
                        # valid points -- falling back whenever the boundary
                        # subset is a small fraction of the data (not just
                        # when it's nearly empty) catches that case too.
                        if _bnd_ta.sum() < 4 or _bnd_ta.sum() < 0.05 * len(_ea_x_refs[_ei]):
                            _bnd_ta = np.ones_like(_ea_x_refs[_ei], dtype=bool)
                        _bx_ta, _by_ta, _bz_ta = _ea_x_refs[_ei][_bnd_ta], _ea_y_refs[_ei][_bnd_ta], _ea_z_refs[_ei][_bnd_ta]
                        _gt_b_ta = _ea_u_refs[_ei][_bnd_ta]
                        _pinn_b_ta = _ea_u_pinns[_ei][_bnd_ta]
                        if len(_bx_ta) > 3000:
                            _ea_sub_ta = np.random.default_rng(0).choice(len(_bx_ta), size=3000, replace=False)
                            _bx_ta, _by_ta, _bz_ta = _bx_ta[_ea_sub_ta], _by_ta[_ea_sub_ta], _bz_ta[_ea_sub_ta]
                            _gt_b_ta, _pinn_b_ta = _gt_b_ta[_ea_sub_ta], _pinn_b_ta[_ea_sub_ta]
                        _err_b_ta = np.abs(_pinn_b_ta - _gt_b_ta)
                        _cols_ta = [(_pinn_b_ta, f"PINN  t={{_fmt_num(_ea_tv)}}  {{_l2_txt(_l2)}}"),
                                    (_gt_b_ta, f"Reference  t={{_ea_tv:.3f}}"),
                                    (_err_b_ta, f"|Error|  max = {{_sci_txt(_mx)}}")]
                        for _ci_ta, (_vals_ta, _ttl_ta) in enumerate(_cols_ta):
                            _ax3_ta = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _ci_ta + 1, projection='3d')
                            for _i3_ta, _j3_ta in _edges_ta:
                                _p0_ta, _p1_ta = _corners_ta[_i3_ta], _corners_ta[_j3_ta]
                                _ax3_ta.plot([_p0_ta[0], _p1_ta[0]], [_p0_ta[1], _p1_ta[1]], [_p0_ta[2], _p1_ta[2]],
                                             color='#888888', linewidth=0.8, alpha=0.6)
                            _sc3_ta = _ax3_ta.scatter(_bx_ta, _by_ta, _bz_ta, c=_vals_ta,
                                                       cmap=('{err_cmap(config)}' if _ci_ta == 2 else '{config.plot_colormap}'), s=14)
                            fig.colorbar(_sc3_ta, ax=_ax3_ta, shrink=0.6, pad=0.12)
                            _ax3_ta.set_title(_ttl_ta, fontsize={config.plot_subplot_title_fontsize})
                            _ax3_ta.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax3_ta.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax3_ta.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax3_ta.xaxis, _ax3_ta.yaxis, _ax3_ta.zaxis)]
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                    _ea_did_surface_ta = True
                elif len(_ea_times) < 2:
                    # Same fix as the Standard path's own copy of this
                    # section (see "need at least 2 time snapshots" above):
                    # a 1D x-t surface needs >= 2 distinct time snapshots to
                    # form a non-degenerate grid -- e.g. only one reference
                    # file falls within this Time-Adaptive run's range so
                    # far. This path didn't have the guard, so a single
                    # reference file crashed matplotlib's contourf on a
                    # (1, N) array instead of skipping gracefully (the line
                    # comparison above already covers this single snapshot).
                    _ea_did_surface_ta = False
                    print("  Skipping surface comparison — need at least 2 time snapshots for a 1D x-t surface plot")
                else:
                    _ea_x_common = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                    _ea_t_arr = np.array(_ea_times)
                    _ea_U_pinn = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                    _ea_U_fem  = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                    for _ei in range(_ea_n_t):
                        _ea_fi_p = _interp1d(_ea_x_refs[_ei], _ea_u_pinns[_ei], kind='linear', fill_value='extrapolate')
                        _ea_U_pinn[_ei, :] = _ea_fi_p(_ea_x_common)
                        _ea_fi_f = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                        _ea_U_fem[_ei, :] = _ea_fi_f(_ea_x_common)
                    _ea_Xg, _ea_Tg = np.meshgrid(_ea_x_common, _ea_t_arr)
                    _ea_U_err = np.abs(_ea_U_pinn - _ea_U_fem)
                    _ea_vmin = min(_ea_U_pinn.min(), _ea_U_fem.min())
                    _ea_vmax = max(_ea_U_pinn.max(), _ea_U_fem.max())
                    if _ea_vmax - _ea_vmin < 1e-12:
                        _ea_vmax = _ea_vmin + 1e-12
                    _ea_vmin, _ea_vmax = _vr(_ea_vmin, _ea_vmax)   # Plot settings -> Surface -> Colour range
                    _levels_ea1d = _cf_levels({config.plot_levels}, _ea_vmin, _ea_vmax)["levels"]  # contourf ignores vmin/vmax with an integer `levels=N` -- see the Standard-path comment on the same pattern above
                    fig, axes = plt.subplots(1, 3, figsize=_panel_figsize(5, 5, 3, 1))
                    fig.suptitle(_ea_title_txt("Surface Comparison", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                    im0 = axes[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels=_levels_ea1d, cmap='{config.plot_colormap}', extend=_vext())
                    axes[0].set_title(f"PINN  {{_ea_field_txt(_ea_sel)}}(x,t)"); axes[0].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[0].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                    fig.colorbar(im0, ax=axes[0])
                    im1 = axes[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels=_levels_ea1d, cmap='{config.plot_colormap}', extend=_vext())
                    axes[1].set_title(f"Reference  {{_ea_field_txt(_ea_sel)}}(x,t)"); axes[1].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[1].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                    fig.colorbar(im1, ax=axes[1])
                    im2 = axes[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='{err_cmap(config)}')
                    axes[2].set_title("Error  |PINN - Reference|"); axes[2].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[2].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                    fig.colorbar(im2, ax=axes[2])
                    plt.tight_layout(rect=[0, 0, 1, 0.93])
                    _ea_did_surface_ta = True
                if _ea_did_surface_ta:
                    _ea_sp = _os.path.join(_ea_dir, f"surface_comparison{{_ea_suffix}}.png")
                    plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                    print(f"  Surface comparison saved: {{_ea_sp}}")

            print(f"  Group '{{_ea_group_label(_ea_sel) or 'default'}}' analysis complete")
        print("=== Time-Adaptive Error Analysis Complete ===")"""


def _n_equations(config):
    """How many PDE residuals the problem has (Round 63): config.num_equations,
    or one per output when it's 0 (every config saved before this round).
    DeepXDE itself allows any number -- e.g. its American-put example has
    three for one output, its unknown-field inverse problems one for two."""
    return int(_cfg_get(config, "num_equations", 0) or 0) or int(_cfg_get(config, "num_outputs", 1) or 1)


def _cfg_get(cfg, key, default=None):
    """Attribute or dict access -- the transform helpers below take either a
    PINNConfig or a restored model_config.json dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _input_formula_on(cfg):
    return bool(_cfg_get(cfg, "input_transform_enabled", False)) and \
        _cfg_get(cfg, "input_transform_mode", "scale") == "formula" and bool(_feature_formula_list(cfg))


def _output_formula_on(cfg):
    return bool(_cfg_get(cfg, "output_transform_enabled", False)) and \
        _cfg_get(cfg, "output_transform_mode", "scale") == "formula" and \
        any(e.strip() for e in (_cfg_get(cfg, "output_transform_formula", "") or "").split("|"))


def _feature_formula_list(cfg):
    return [f.strip() for f in (_cfg_get(cfg, "input_feature_formula", "") or "").split(";") if f.strip()]


def _net_transforms_code(cfg, fn_name="_apply_net_transforms"):
    """Source for `<fn_name>(net)`, which attaches this model's input and
    output transforms to a freshly built network and returns it (cleanup
    Round 62). Shared by the live Solve script, the exported script and
    both Restore & Visualize builders: DeepXDE's apply_feature_transform /
    apply_output_transform attach a Python closure to the net instance,
    not anything saved in a checkpoint, so every script that rebuilds the
    net must re-attach the same transforms or its predictions are wrong
    (the restore builders didn't, before this round).

    Two modes per side: "scale" (x * scale + shift, one value per column)
    and "formula" (expressions in the coordinates x, y, z, t and, on the
    output side, the raw network outputs by name -- see PINNConfig).
    `cfg` is a PINNConfig or a restored model_config.json dict; a dict
    without these keys (an older save) means no transforms."""
    is_2d = _cfg_get(cfg, "problem_dim", "1D") == "2D"
    is_3d = _cfg_get(cfg, "problem_dim", "1D") == "3D"
    is_steady = bool(_cfg_get(cfg, "steady_state", False))
    n_out = int(_cfg_get(cfg, "num_outputs", 1) or 1)
    names = [n.strip() for n in (_cfg_get(cfg, "output_names", "u") or "u").split(",") if n.strip()]
    while len(names) < n_out:
        names.append(f"u{len(names)}")
    names = names[:n_out]

    L = [f"def {fn_name}(net):",
         '    """Attach this model\'s input/output transforms (none if not configured)."""']
    in_on = bool(_cfg_get(cfg, "input_transform_enabled", False))
    out_on = bool(_cfg_get(cfg, "output_transform_enabled", False))

    def _coord_lines(exprs, src):
        return [f"        {n} = {src}[:, {c}:{c + 1}]"
                for n, c in _pde_coord_columns(_pde_coord_names(exprs), is_2d, is_3d, is_steady)]

    if _input_formula_on(cfg):
        feats = [_simplify_pde_expr(f) for f in _feature_formula_list(cfg)]
        L += ["    def _input_transform(X):"] + _coord_lines(feats, "X")
        L.append("        return torch.cat([" + ", ".join(feats) + "], dim=1)")
        L.append("    net.apply_feature_transform(_input_transform)")
    elif in_on and _cfg_get(cfg, "input_transform_scale", None):
        L += [f"    in_scale = {list(_cfg_get(cfg, 'input_transform_scale'))}",
              f"    in_shift = {list(_cfg_get(cfg, 'input_transform_shift', []))}",
              "    def _input_transform(x):",
              "        sc = torch.tensor(in_scale, dtype=x.dtype, device=x.device)",
              "        sh = torch.tensor(in_shift, dtype=x.dtype, device=x.device)",
              "        return x * sc + sh",
              "    net.apply_feature_transform(_input_transform)"]

    if _output_formula_on(cfg):
        raw = (_cfg_get(cfg, "output_transform_formula", "") or "").split("|")
        exprs = [_simplify_pde_expr(raw[i]) if i < len(raw) and raw[i].strip() else names[i] for i in range(n_out)]
        L += ["    def _output_transform(X, N):"] + _coord_lines(exprs, "X")
        L += [f"        {nm} = N[:, {i}:{i + 1}]" for i, nm in enumerate(names)]
        if n_out == 1:
            L.append(f"        return {exprs[0]}")
        else:
            L.append("        return torch.cat([" + ", ".join(exprs) + "], dim=1)")
        L.append("    net.apply_output_transform(_output_transform)")
    elif out_on and _cfg_get(cfg, "output_transform_scale", None):
        L += [f"    out_scale = {list(_cfg_get(cfg, 'output_transform_scale'))}",
              f"    out_shift = {list(_cfg_get(cfg, 'output_transform_shift', []))}",
              "    def _output_transform(x, y):",
              "        sc = torch.tensor(out_scale, dtype=y.dtype, device=y.device)",
              "        sh = torch.tensor(out_shift, dtype=y.dtype, device=y.device)",
              "        return y * sc + sh",
              "    net.apply_output_transform(_output_transform)"]
    L.append("    return net")
    return "\n".join(L)


def _transform_config_literal(config):
    """The transform settings as model_config.json entries (a string of
    `"key": value,` pairs to splice into the dict literal the generated
    script writes), so Restore & Visualize can re-attach the same
    transforms -- see _net_transforms_code."""
    keys = ("input_transform_enabled", "input_transform_mode", "input_transform_scale",
            "input_transform_shift", "input_feature_formula",
            "output_transform_enabled", "output_transform_mode", "output_transform_scale",
            "output_transform_shift", "output_transform_formula")
    return " ".join(f"{k!r}: {getattr(config, k)!r}," for k in keys)


def _constant_values(config):
    """[(name, expression, value)] of the problem's named constants (Round
    70); [] when there are none. Invalid constants raise ValueError (the
    GUI checks them before Solve/Export, see PINNConfig.validate)."""
    text = getattr(config, "constants", "") or ""
    if not text.strip() or text.strip() == "[]":
        return []
    from pinnstudio.core import constants as _cmod
    names = [n for n in (config.output_names or "").split(",")]
    try:
        names += [v.get("name", "") for v in _bc_json_std.loads(config.inverse_variables_json or "[]")]
    except (ValueError, AttributeError):
        pass
    return _cmod.evaluate(text, names)


def _ode_text(expr):
    """An expression of an ODE problem (independent variable t) rewritten
    for the 1D machinery that solves it (variable x): du_t -> du_x,
    dy1_tt -> dy1_xx, t -> x, t_min/t_max -> x_min/x_max, start/end -> left/right."""
    import re as _re_ode
    if not isinstance(expr, str) or not expr:
        return expr
    e = _re_ode.sub(r"(?<![\w.])(d[A-Za-z]\w*?)_(t+)(?![\w])", lambda m: m.group(1) + "_" + "x" * len(m.group(2)), expr)
    e = _re_ode.sub(r"(?<![\w.])t_(min|max)(?![\w])", r"x_\1", e)
    e = _re_ode.sub(r"(?<![\w.])t(?![\w])", "x", e)
    e = _re_ode.sub(r"(?<![\w.])start(?![\w])", "left", e)
    e = _re_ode.sub(r"(?<![\w.])end(?![\w])", "right", e)
    return e


def _ode_to_1d(config):
    """Round 74: an ODE problem (problem_dim "ODE", independent variable t
    in [t_min, t_max]) solved as a 1D steady problem in x = t. Initial
    conditions become Dirichlet rows at the start (x = t_min), keeping
    their place in the loss-weight order; an Operator row "at the initial
    time" becomes one at the start."""
    import dataclasses as _dc
    import re as _re_ode
    n_out = max(1, int(config.num_outputs or 1))
    rows = []
    try:
        rows = _bc_json_std.loads(config.custom_bc_json or "[]")
    except ValueError:
        rows = []
    for r in rows:
        for k in ("location", "value", "location2", "values_formula", "points_list"):
            if isinstance(r.get(k), str):
                r[k] = _ode_text(r[k])
        if r.get("type") == "operator" and r.get("apply_at") == "initial":
            r["apply_at"] = "boundary"
            loc = (r.get("location") or "").strip()
            r["location"] = "left" if loc in ("", "True") else f"left and ({loc})"
    ic_exprs = (config.ic_expressions or "").split("|")
    ic_act = [a.strip() == "True" for a in (config.ic_active or "").split(",")]
    for i in range(n_out):
        if i < len(ic_act) and ic_act[i] and i < len(ic_exprs) and ic_exprs[i].strip():
            v = _re_ode.sub(r"x\[:,\s*0(:1)?\]", "x", _ode_text(ic_exprs[i].strip()))
            rows.append({"type": "dirichlet", "location": "left", "value": v, "component": i})
    tm = config.training_monitors
    try:
        _tml = _bc_json_std.loads(tm or "[]")
        for m in _tml:
            if isinstance(m, dict) and isinstance(m.get("expr"), str):
                m["expr"] = _ode_text(m["expr"])
        tm = _bc_json_std.dumps(_tml)
    except ValueError:
        pass
    obs = config.inverse_obs_files_json
    try:
        _ol = _bc_json_std.loads(obs or "[]")
        for o in _ol:
            if isinstance(o, dict) and isinstance(o.get("custom_expr"), str):
                o["custom_expr"] = _ode_text(o["custom_expr"])
        obs = _bc_json_std.dumps(_ol) if _ol else obs
    except ValueError:
        pass
    return _dc.replace(
        config, problem_dim="1D", steady_state=True, geometry_type="Interval",
        x_min=float(config.t_min), x_max=float(config.t_max), num_initial=0,
        layers=[1] + list(config.layers[1:]) if config.layers else config.layers,
        time_adaptive=False, adapt_method="None" if config.adapt_method == "Time Adaptive" else config.adapt_method,
        pde_expressions=_ode_text(config.pde_expressions), pde_expression=_ode_text(config.pde_expression),
        ic_expressions="|".join(["0"] * n_out), ic_expression="0", ic_active=",".join(["False"] * n_out),
        forward_ic_from_file=False,
        custom_bc_json=_bc_json_std.dumps(rows), training_monitors=tm, inverse_obs_files_json=obs,
        input_feature_formula=_ode_text(config.input_feature_formula),
        output_transform_formula=_ode_text(config.output_transform_formula),
        ea_ref_formula=_ode_text(config.ea_ref_formula), plot_custom_expr=_ode_text(config.plot_custom_expr),
        anchors_box=_ode_text(config.anchors_box))


_INTEGRAL_RE = None


def _integral_text(expr, out_names, x0):
    """Round 76: integro-differential terms. `integral(f)` in an equation
    of an ODE (or steady 1D) problem is the Volterra integral of f from the
    start of the range to the current point, f being an expression in the
    integration variable s, the current point (x, i.e. t) and the outputs
    evaluated at s -- e.g. integral(exp(s - x)*u). Rewritten to
    _integral(lambda s, <outputs>: (f), x, x0), which the generated script
    evaluates by Gauss-Legendre quadrature of the network (as DeepXDE's
    dde.data.IDE does)."""
    import re as _re_int
    if not isinstance(expr, str) or "integral(" not in expr:
        return expr
    out, i = [], 0
    pat = _re_int.compile(r"(?<![\w.])integral\(")
    names = ", ".join(["s"] + [n for n in out_names if n])
    while True:
        m = pat.search(expr, i)
        if not m:
            out.append(expr[i:])
            break
        out.append(expr[i:m.start()])
        depth, j = 1, m.end()
        while j < len(expr) and depth:
            depth += {"(": 1, ")": -1}.get(expr[j], 0)
            j += 1
        if depth:
            raise ValueError(f"integral( has no closing parenthesis in {expr!r}")
        inner = expr[m.end():j - 1]
        out.append(f"_integral(lambda {names}: ({inner}), x, {float(x0)!r})")
        i = j
    return "".join(out)


def _uses_integral(config):
    return "_integral(lambda" in (getattr(config, "pde_expressions", "") or "")


def _integral_helper_code(config):
    """Runtime helper for integral(...) terms (see _integral_text); "" when
    no equation uses one."""
    if not _uses_integral(config):
        return ""
    q = int(getattr(config, "integral_points", 20) or 20)
    return f'''# ── Integral terms: Gauss-Legendre quadrature of the network ──────────
_GL_xi, _GL_w = np.polynomial.legendre.leggauss({q})
def _integral(_f, _xc, _x0):
    """Integral of _f(s, outputs at s) for s from _x0 to _xc (one per point)."""
    _xc = _xc[:, 0:1]
    _xi = torch.as_tensor(_GL_xi, dtype=_xc.dtype, device=_xc.device)
    _wq = torch.as_tensor(_GL_w, dtype=_xc.dtype, device=_xc.device)
    _h = (_xc - _x0) / 2
    _s = _x0 + _h * (_xi[None, :] + 1)
    _ys = net(_s.reshape(-1, 1)).reshape(_s.shape[0], _s.shape[1], -1)
    _v = _f(_s, *[_ys[..., _k] for _k in range(_ys.shape[-1])])
    return (_v * _wq).sum(dim=1, keepdim=True) * _h
'''


_FRAC_RE_SRC = r"(?<![\w.])frac\("


def _uses_frac(config):
    """Round 77: an equation uses the fractional operator frac(...)."""
    import re as _re_fr
    return bool(_re_fr.search(_FRAC_RE_SRC, getattr(config, "pde_expressions", "") or ""))


def _frac_resolution(config):
    return [int(float(v)) for v in str(getattr(config, "fractional_resolution", "") or "").replace(";", ",").split(",")
            if v.strip()]


def _frac_alpha_expr(config):
    """The alpha passed to dde.data.FPDE: a number, or the name of an
    inverse variable (trainable alpha)."""
    a = str(getattr(config, "fractional_alpha", "") or "").strip()
    try:
        return repr(float(a.strip("() ")))
    except ValueError:
        return a


def _frac_data_text(text, config, geom, pde, cons):
    """dde.data.PDE/TimePDE(geom, pde, cons, ...) -> dde.data.FPDE/TimeFPDE(
    geom, pde, alpha, cons, resolution, meshtype=..., ...) for a fractional
    problem (Round 77). The loss is tested on the training points
    (num_test=None, as in DeepXDE's fractional scripts; a static mesh can't
    have test points at all)."""
    import re as _re_fr
    res = _frac_resolution(config)
    mesh = getattr(config, "fractional_meshtype", "dynamic") or "dynamic"
    head = _re_fr.compile(r"dde\.data\.(Time)?PDE\((\s*)" + _re_fr.escape(geom) + r", " + _re_fr.escape(pde) + r", "
                          + _re_fr.escape(cons) + r",")

    def _sub(m):
        return (f"dde.data.{m.group(1) or ''}FPDE({m.group(2)}{geom}, {pde}, {_frac_alpha_expr(config)}, {cons}, {res!r}, "
                f"meshtype={mesh!r},")
    out = head.sub(_sub, text)
    if True:   # fractional problems test on the training points (DeepXDE's num_test=None)
        out = _re_fr.sub(r"(dde\.data\.(?:Time)?FPDE\((?:(?!\)\n).)*?)num_test=[^,\n]+,", r"\1num_test=None,", out,
                         flags=_re_fr.S)
    return out


def _frac_helper_code(config):
    """Runtime helper for frac(...) terms: DeepXDE's FPDE hands the residual
    its fractional matrix (dense, or sparse as (indices, values, shape)),
    which this applies to an output column, zero-padded to all points; the
    residual is then cut to the matrix's rows (the FPDE points; the rest are
    DeepXDE's auxiliary points)."""
    if "_frac(" not in (getattr(config, "pde_expressions", "") or ""):
        return ""
    return '''# ── Fractional operator (DeepXDE FPDE matrix) ──────────────────────────
def _frac_rows(_im):
    return int(_im[2][0]) if isinstance(_im, (list, tuple)) and len(_im) == 3 else int(_im.shape[0])
def _frac_apply(_im, _v):
    """The fractional operator (Grunwald-Letnikov directional derivatives) applied to _v."""
    if _im is None:
        raise RuntimeError("frac(...) needs the fractional matrix: it is only defined while training (FPDE data)")
    if isinstance(_im, (list, tuple)) and len(_im) == 3:
        _idx, _vals, _shape = _im
        _idx = torch.as_tensor(np.asarray(_idx), dtype=torch.long, device=_v.device).reshape(-1, 2).T
        _vals = _vals.to(_v.dtype) if torch.is_tensor(_vals) else torch.as_tensor(np.asarray(_vals), dtype=_v.dtype, device=_v.device)
        _r = torch.sparse.mm(torch.sparse_coo_tensor(_idx, _vals.to(_v.device), tuple(int(_d) for _d in _shape)), _v)
    else:
        _m = _im if torch.is_tensor(_im) else torch.as_tensor(np.asarray(_im), dtype=_v.dtype, device=_v.device)
        _r = _m.to(_v.dtype) @ _v
    if _r.shape[0] < _v.shape[0]:
        _r = torch.cat([_r, _r.new_zeros((_v.shape[0] - _r.shape[0], _r.shape[1]))], dim=0)
    return _r
'''


def _init_loss_weights_code(config, indent, model, weights_var, lr, ext_vars):
    """Round 75: 'Scale by initial losses' -- evaluate every loss term once
    (no training) and set weight_i = c / L_i(0), as DeepXDE's wave example
    does (loss_weights = 5 / initial_losses). Empty when the option is off."""
    c = float(getattr(config, "loss_weights_from_initial", 0.0) or 0.0)
    if c <= 0:
        return ""
    lines = [
        f"# Loss weights from the initial losses: w_i = {c:g} / L_i(0)",
        f'{model}.compile("{config.optimizer}", lr={lr}, loss="{config.loss_type}", external_trainable_variables={ext_vars})',
        f"_lh0, _ = {model}.train(iterations=0, display_every=1)",
        f"{weights_var} = [{c!r} / max(float(_l), 1e-30) for _l in _lh0.loss_train[0]]",
        f'print("Loss weights from the initial losses:", ["%.4g" % _w for _w in {weights_var}])',
    ]
    return "\n".join(indent + ln for ln in lines)


def _num_lit(v):
    """A float as a short literal for a generated formula (2.0 -> "2")."""
    v = float(v)
    return repr(int(v)) if v == int(v) and abs(v) < 1e15 else repr(v)


def _exact_constraints(config):
    """Exact initial conditions (config.ic_hard) and exact periodicity
    (config.periodic_inputs), turned into the output / input transform
    formulas both generators already support. Returns the config unchanged
    when neither is on, else a copy:
      output i = IC_i(x) + tau * (network output i), tau = (t - t_min) /
      (t_max - t_min) -- exact at t = t_min (outputs without an exact IC keep
      their own scale/shift);
      inputs = cos(2 pi x / L), sin(2 pi x / L) for each periodic axis
      (L = that axis' domain length), the other coordinates as they are, t."""
    import dataclasses as _dc
    from pinnstudio.core.config import _periodic_axes
    names = [n.strip() for n in (config.output_names or "u").split(",") if n.strip()]
    n_out = max(1, int(getattr(config, "num_outputs", 1) or 1))
    while len(names) < n_out:
        names.append(f"u{len(names)}")
    hard = [h.strip() == "True" for h in (getattr(config, "ic_hard", "") or "").split(",")][:n_out]
    hard += [False] * (n_out - len(hard))
    per = bool(getattr(config, "periodic_inputs", False))
    if config.steady_state or config.problem_dim == "ODE":
        hard = [False] * n_out
    if not (any(hard) or per):
        return config
    upd = {}
    if any(hard):
        ics = (config.ic_expressions or "").split("|")
        t0, t1 = float(config.t_min), float(config.t_max)
        if t0 == 0.0:
            tau = "t" if t1 == 1.0 else f"(t/{_num_lit(t1)})"
        else:
            tau = f"((t - {_num_lit(t0)})/{_num_lit(t1 - t0)})"
        scale_on = bool(config.output_transform_enabled) and (config.output_transform_mode or "scale") == "scale"
        sc = list(config.output_transform_scale or []) if scale_on else []
        sh = list(config.output_transform_shift or []) if scale_on else []
        outs = []
        for i, nm in enumerate(names[:n_out]):
            base = nm
            a = float(sc[i]) if i < len(sc) else 1.0
            b = float(sh[i]) if i < len(sh) else 0.0
            if a != 1.0:
                base = f"{_num_lit(a)}*{base}"
            if b != 0.0:
                base = f"({base} + {_num_lit(b)})"
            if hard[i]:
                ic = ics[i].strip() if i < len(ics) else "0"
                outs.append(f"({ic}) + {tau}*{base}")
            else:
                outs.append(base)
        upd.update(output_transform_enabled=True, output_transform_mode="formula",
                   output_transform_formula="|".join(outs))
    if per:
        axes = _periodic_axes(config)
        coords = ["x"] + (["y"] if config.problem_dim in ("2D", "3D") else []) + (["z"] if config.problem_dim == "3D" else [])
        bounds = {"x": (config.x_min, config.x_max), "y": (getattr(config, "y_min", 0.0), getattr(config, "y_max", 1.0)),
                  "z": (getattr(config, "z_min", 0.0), getattr(config, "z_max", 1.0))}
        feats = []
        for c in coords:
            if c in axes:
                L = float(bounds[c][1]) - float(bounds[c][0])
                k = 2.0 / L
                arg = f"pi*{c}" if k == 1.0 else f"{_num_lit(k)}*pi*{c}"
                feats += [f"cos({arg})", f"sin({arg})"]
            else:
                feats.append(c)
        if not config.steady_state:
            feats.append("t")
        upd.update(input_transform_enabled=True, input_transform_mode="formula",
                   input_feature_formula="; ".join(feats))
    return _dc.replace(config, **upd)


def _effective_config(config):
    """The config as the generators should read it -- cleanup Round 56.
    PINNConfig documents time_adaptive as IGNORED for a steady-state problem
    (Time-Adaptive training splits a time axis a steady problem doesn't
    have), and the GUI hides it there -- but neither generator enforced
    that, so a hand-edited or older save with both set made
    generate_script() take its Time-Adaptive branch for a steady problem
    and made generate_clean_script() emit a script that doesn't even parse.
    Returns a copy with time_adaptive switched off in that one case; every
    other config is returned unchanged (the same object)."""
    import dataclasses as _dc
    if getattr(config, "steady_state", False) and getattr(config, "time_adaptive", False):
        config = _dc.replace(config, time_adaptive=False)
    # Round 70: named constants -> their values, in every expression.
    _consts = _constant_values(config)
    if _consts:
        from pinnstudio.core import constants as _cmod
        _sub = {f: _cmod.substitute(getattr(config, f), _consts)
                for f in ("pde_expressions", "pde_expression", "ic_expressions", "ic_expression",
                          "input_feature_formula", "output_transform_formula", "ea_ref_formula",
                          "plot_custom_expr", "sweep_plot_custom_expr", "anchors_box", "fractional_alpha")
                if isinstance(getattr(config, f, None), str)}
        _sub["custom_bc_json"] = _cmod.substitute_json(config.custom_bc_json, _consts, ("location", "value", "location2",
                                                                                      "points_list", "values_formula"))
        _sub["inverse_obs_files_json"] = _cmod.substitute_json(config.inverse_obs_files_json, _consts, ("custom_expr",))
        _sub["training_monitors"] = _cmod.substitute_json(config.training_monitors, _consts, ("expr",))
        config = _dc.replace(config, constants="", **_sub)
    config = _exact_constraints(config)
    if (getattr(config, "loss_weights_from_initial", 0) or 0) > 0 and getattr(config, "scheduler_phases", ""):
        # Round 75: weights from the initial losses replace every phase's own.
        try:
            _ph = _bc_json_std.loads(config.scheduler_phases)
            if any(isinstance(p, dict) and "weights" in p for p in _ph):
                for p in _ph:
                    if isinstance(p, dict):
                        p.pop("weights", None)
                config = _dc.replace(config, scheduler_phases=_bc_json_std.dumps(_ph))
        except ValueError:
            pass
    if _uses_frac(config):
        import re as _re_fr
        config = _dc.replace(config,
                             pde_expressions=_re_fr.sub(_FRAC_RE_SRC, "_frac(", config.pde_expressions),
                             pde_expression=_re_fr.sub(_FRAC_RE_SRC, "_frac(", config.pde_expression or ""))
    _was_ode = getattr(config, "problem_dim", "") == "ODE"
    if _was_ode:
        config = _ode_to_1d(config)
    if "integral(" in (config.pde_expressions or "") and config.problem_dim == "1D" and config.steady_state:
        _names = [n.strip() for n in (config.output_names or "u").split(",")][:max(1, int(config.num_outputs or 1))]
        config = _dc.replace(config,
                             pde_expressions=_integral_text(config.pde_expressions, _names, config.x_min),
                             pde_expression=_integral_text(config.pde_expression, _names, config.x_min))
    # Round 62: with an input feature formula the network's first layer
    # takes one input per feature, not one per coordinate.
    # Round 67: "reference from a formula" -> the files the scripts will
    # write (see _formula_reference_code), as ordinary ea_files entries.
    _ref = _formula_reference_entries(config)
    if _ref:
        config = _dc.replace(config, ea_files=repr([(t, p, i) for (t, p, i, _e) in _ref]))
    n_feat = len(_feature_formula_list(config)) if _input_formula_on(config) else 0
    if n_feat and config.layers and config.layers[0] != n_feat:
        config = _dc.replace(config, layers=[n_feat] + list(config.layers[1:]))
    if _was_ode:
        # Round 74: plots of an ODE name its variable t, not x (a fresh
        # copy, never the caller's config).
        config._ode = True
    return config




# On the CPU, PyTorch uses one thread per core. On processors with
# efficiency cores (recent Intel, Apple) the slow cores hold every step back
# and run times jump by 2-3 times between identical runs; a few threads
# (about one per performance core) were faster and steady in the timing
# study (README, "Speed"). OMP_NUM_THREADS, when set, still decides.
def _cpu_threads_code(config):
    """CPU threads (GPU / Hardware settings): a number of your own, or
    Automatic -- about one per performance core, at most 8."""
    n = int(getattr(config, "cpu_threads", 0) or 0)
    if n > 0:
        return f"""
# CPU training: {n} threads (GPU / Hardware settings -> CPU threads)
if not torch.cuda.is_available():
    torch.set_num_threads({n})
"""
    return """
# CPU training: about one thread per performance core is faster and steadier
# than PyTorch's one per core on processors with efficiency cores
# (GPU / Hardware settings -> CPU threads: Automatic). OMP_NUM_THREADS, when
# set, still decides.
if not torch.cuda.is_available() and "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2)))
"""

def _cpu_only_line(config):
    """Train on: CPU -- PyTorch then sees no GPU (set before it is imported)."""
    if getattr(config, "train_device", "auto") == "cpu":
        return 'os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Train on: CPU (PINNStudio GPU / Hardware settings)\n'
    return ""

def generate_script(config):
    config = _effective_config(config)
    # Naming convention (cleanup Round 55): generation-time Python names in
    # this function use the same plain spelling as generate_clean_script()'s
    # (is_2d, is_3d, is_steady, line_slice_y, sol_ext, ...), so a fix ported
    # between the two generators keeps its names. Names INSIDE the script
    # text this function writes (_is_steady, _is_2d, _save_dir, _plot_*, ...)
    # keep their leading underscore: that marks every runtime variable of
    # the live Solve script as private to the app, and it is applied
    # uniformly there, so it stays.
    is_2d = config.problem_dim == "2D"
    is_3d = config.problem_dim == "3D"
    is_steady = bool(config.steady_state)
    # Backward-compat / defense-in-depth: the GUI's point-distribution
    # dropdown and the Parametric Sweep registry used to offer the literal
    # string "pseudorandom" as a choice, but DeepXDE's own train_distribution
    # only ever accepts "pseudo" -- selecting "pseudorandom" raised
    # ValueError: pseudorandom sampling is not available. right as training
    # started. Both the dropdown and the sweep registry's choices list now
    # say "pseudo" directly, but this remap also covers any config saved
    # before that fix, so an old saved file with the bad literal still
    # works instead of crashing.
    _safe_point_distribution = "pseudo" if config.point_distribution == "pseudorandom" else config.point_distribution
    # DeepXDE's own OperatorBC/PointSetOperatorBC docstrings warn that if the
    # BC function uses the raw X argument, num_test must not be set on the
    # owning dde.data.PDE/TimePDE (DeepXDE raises an error otherwise). The
    # BC panel's "Value" expression for operator/pointset_operator rows is
    # free-form user text that can reference X (see _bc_operator_fn below),
    # so detect that here -- at codegen time, since custom_bc_json is
    # already a plain Python string available right now -- and silently
    # fall back to no test set (num_test=None) only when it's actually
    # needed, leaving every other problem's num_test exactly as configured.
    try:
        _bc_entries_for_numtest = _bc_json_std.loads(config.custom_bc_json) if config.custom_bc_json else []
    except (ValueError, TypeError):
        _bc_entries_for_numtest = []
    _bc_op_uses_X = _bc_entries_use_operator_X(_bc_entries_for_numtest)
    _safe_num_test = None if _bc_op_uses_X else config.num_test
    # Interface 2D BC's real DeepXDE "train_distribution must be uniform"
    # requirement (see _bc_entries_use_interface2d's docstring) has exactly
    # one valid answer whenever this BC type is active at all, so it's
    # corrected here automatically instead of making the user hunt down
    # and change the Point Distribution setting themselves -- the GUI's BC
    # panel already mirrors this (switching its own Point Distribution
    # combo to "uniform" the moment an Interface 2D row is selected), this
    # is the defense-in-depth codegen-level guarantee for any config that
    # reaches here with a stale/hand-edited value regardless. Only the
    # DISTRIBUTION half of the precondition is auto-fixed this way -- the
    # GEOMETRY half (Rectangle/Polygon) has no sensible value to guess, so
    # it still raises via _bc_check_interface2d_preconditions below.
    _bc_forced_uniform = (
        _bc_entries_use_interface2d(_bc_entries_for_numtest)
        and _safe_point_distribution != "uniform"
    )
    if _bc_forced_uniform:
        _safe_point_distribution = "uniform"
    # Interface 2D BC's real DeepXDE geometry precondition (Rectangle/
    # Polygon only) -- already a plain config value at this point, so
    # checked here immediately rather than letting a wrong geometry reach
    # DeepXDE's own far-less-clear error (or, worse, train silently wrong)
    # -- see the helper's own docstring.
    _bc_check_interface2d_preconditions(_bc_entries_for_numtest, config.geometry_type,
                                        is_2d=(config.problem_dim == "2D"), num_outputs=config.num_outputs)
    _bc_check_initial_operator_preconditions(_bc_entries_for_numtest, bool(config.steady_state),
                                             bool(config.time_adaptive))
    _check_anchor_preconditions(config)
    # Periodic BC's own real DeepXDE geometry precondition -- see
    # _bc_check_periodic_preconditions's docstring. Same "clear message
    # instead of a confusing one" rationale as Interface 2D BC's check
    # just above, though this one isn't an app-crash risk the way that
    # one was.
    _bc_check_periodic_preconditions(_bc_entries_for_numtest, config.geometry_type)
    # The "solution" output is a static PNG for every plot type except the
    # two GIF animations, where it's an actual animated .gif file instead
    # -- known now (at generation time) from the selected plot type, so the
    # right extension can be baked into every solution-path literal below
    # rather than guessed at runtime.
    # Steady-state problems have no time axis, so a GIF animation (which is
    # inherently a walk over time) never applies -- always .png there, even
    # if a stale config still has one of the GIF plot types selected.
    sol_ext = "gif" if ((not is_steady and config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"))
                        or (config.problem_type == "Inverse" and config.plot_type == "Parameter Convergence (GIF)")) else "png"
    # Convert user-friendly IC expressions
    ic_exprs_raw = config.ic_expressions.split("|")
    ic_exprs_converted = [_simplify_expr(e, is_2d, is_3d) for e in ic_exprs_raw]
    config_ic_expressions = "|".join(ic_exprs_converted)
    ta_ic_expr = _simplify_expr(config.ic_expression, is_2d, is_3d)

    # Convert user-friendly PDE expressions
    pde_exprs_raw = config.pde_expressions.split("|")
    pde_exprs_converted = [_simplify_pde_expr(e) for e in pde_exprs_raw]
    config_pde_expressions = "|".join(pde_exprs_converted)

    pde_expr_single = _simplify_pde_expr(config.pde_expression)
    # Extra training points (Round 65) -- see _anchor_points_code; the
    # text stays exactly as before when there are none.
    _has_user_anchors = bool(_anchor_points_code(config))
    _ic_anch = bool(getattr(config, "ic_file_anchors", False) and config.forward_ic_from_file
                    and not config.steady_state)
    _live_fwd_anchors = _anchor_join([n for n, on in (("_user_anchors", _has_user_anchors),
                                                    ("_ic_xyt", _ic_anch)) if on])
    _live_inv_anchors = _anchor_join(["_obs_anchors"] + [n for n, on in (("_user_anchors", _has_user_anchors),
                                                                        ("_ic_xyt", _ic_anch)) if on])
    # Coordinates the equation itself uses (a source term like sin(pi*x)):
    # bound as one column each, after the default x/y (inputs/outputs)
    # bindings, only when used -- see _pde_coord_names.
    _pde_cols = _pde_coord_columns(_pde_coord_names(pde_exprs_converted + [pde_expr_single]),
                                   config.problem_dim == "2D", config.problem_dim == "3D", bool(config.steady_state))
    pde_coord_ns_code = "".join(f"\n    _eval_ns[{n!r}] = x[:, {c}:{c + 1}]" for n, c in _pde_cols)
    if pde_coord_ns_code:
        pde_coord_ns_code = "\n    # coordinates used by the equation itself" + pde_coord_ns_code

    # Every 2D/3D line-type plot (Line (time steps), Line Animation (GIF),
    # the Time-Adaptive per-step preview, and the inline Error Analysis
    # line comparison) shows u vs x only, fixing y (and z, in 3D) at one
    # value -- previously each site hardcoded the domain midpoint
    # separately; now every site reads the same user-configurable value,
    # computed once here. _auto (the default) reproduces the previous
    # hardcoded behavior exactly.
    from pinnstudio.core.line_slice import auto_line_slice
    line_slice_y, line_slice_z = auto_line_slice(config)

    # The optional custom Results-panel plot expression (e.g. Schrodinger's
    # "sqrt(u**2+v**2)") is passed through UNCONVERTED, deliberately NOT
    # run through _simplify_pde_expr() the way PDE expressions are above.
    # PDE expressions are eval'd in the generated script's own MODULE
    # namespace (where `import numpy as np` and `import torch` already ran),
    # so the "torch."/"np.pi" prefixes added there resolve. This expression is
    # instead eval'd by _plot_custom_op/_extract_plot_field below inside a
    # small, ISOLATED namespace dict (_pf_ns) built from _PLOT_TORCH_MATH_NS
    # -- which already maps every one of the same bare names
    # (sin/cos/.../sqrt/pi, ...) straight to their torch equivalent, so a
    # raw "sqrt(...)" resolves correctly there. Running _simplify_pde_expr
    # on it rewrites "sqrt(" to "torch.sqrt(" instead, which that namespace
    # has no "np" binding for -- a real bug, caught via an actual exec-level
    # run of the 1D Schrodinger template's GIF export ("NameError: name
    # 'np' is not defined"), since _plot_custom_op's eval() only sees
    # whatever's in _pf_ns, not the module's real globals. Both
    # generate_clean_script()'s own _plot_custom_expr_val and every Restore
    # & Visualize script builder's _custom_expr_val already pass this value
    # through raw/unconverted -- this just brings generate_script() in
    # line with them.
    _plot_custom_expr_raw = (getattr(config, "plot_custom_expr", "") or "").strip()
    plot_custom_expr_converted = _plot_custom_expr_raw
    plot_custom_label_resolved = (getattr(config, "plot_custom_label", "") or "").strip() or _plot_custom_expr_raw

    # Optional title/axis-label overrides for the solution plot -- same
    # blank-means-default convention, and same reasoning, as Restore &
    # Visualize's own title_override/xlabel_override/ylabel_override in
    # _build_restore_script. Baked in as a plain string here (possibly
    # empty) and combined with each default title/label expression via
    # `_plot_title_override or <default>` INSIDE the generated script
    # itself (see the "Plot solution" dispatch below) -- unlike the
    # restore builders, out_name there is only computed at the generated
    # script's own runtime, not at codegen time, so the override can't be
    # resolved here already.
    plot_title_override = (getattr(config, "plot_title_override", "") or "").strip()
    plot_xlabel_override = (getattr(config, "plot_xlabel_override", "") or "").strip()
    plot_ylabel_override = (getattr(config, "plot_ylabel_override", "") or "").strip()
    v1 = "t" if getattr(config, "_ode", False) else "x"   # Round 74: an ODE's variable

    if config.forward_ic_from_file:
        _ta_ic_init = f"""_ic_ta_xt, _ic_ta_vals = _load_ic_from_file({repr(config.forward_ic_file)})
    prev_u = _ic_ta_vals"""
    else:
        _ta_ic_init = f"prev_u = np.reshape({ta_ic_expr}, (-1, 1))"
    _fecr_pde_block = ""

    # Boundary Conditions panel entries -- embedded via repr() (not a raw
    # f-string substitution) because the JSON text contains double quotes
    # and whatever punctuation the user typed into location/value fields;
    # repr() guarantees a syntactically valid, safely-escaped Python string
    # literal in the generated script no matter what's inside it.
    _custom_bc_json_literal = repr(config.custom_bc_json or "[]")
    # Whether this config actually has Boundary Conditions panel data at
    # all, checked BEFORE the "or '[]'" fallback above collapses the
    # distinction: a config built by the current GUI always has a real
    # (possibly empty-array) custom_bc_json string, since the panel is
    # always serialized -- only a config saved before this panel existed
    # would have this field genuinely blank. Used so an intentionally
    # emptied panel (any geometry) means "zero custom BCs", never a
    # silent fall-back to the legacy per-side scheme.
    _bc_panel_data_present_literal = repr(bool(config.custom_bc_json))

    # Geometry-type dispatch (Phase 2): the shape's own parameters, baked in
    # as literals the same way _custom_bc_json_literal is above. Triangle
    # needs exactly 3 vertices and Polygon at least 3 -- fall back to the
    # same defaults PINNConfig itself uses if parsing comes up short (a
    # malformed/edited-away vertex string shouldn't crash codegen).
    _geometry_type_literal = repr(config.geometry_type or "Rectangle")
    _triangle_verts_parsed = _parse_vertex_list(config.geom_triangle_vertices)
    if len(_triangle_verts_parsed) != 3:
        _triangle_verts_parsed = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    _polygon_verts_parsed = _parse_vertex_list(config.geom_polygon_vertices)
    if len(_polygon_verts_parsed) < 3:
        _polygon_verts_parsed = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    _triangle_vertices_literal = repr(_triangle_verts_parsed)
    _polygon_vertices_literal = repr(_polygon_verts_parsed)
    # Custom geometry: the CSG-chained constructor call is built once here
    # (at codegen time, not inside the generated script) and baked in as
    # a literal, same as every other shape's parameters above -- see
    # _build_custom_geom_code()'s docstring for the wrap_needed rule.
    _custom_geom_inner_code, _custom_geom_wrap_needed = _build_custom_geom_code(config)
    _custom_geom_final_literal = (
        f"_DTypeSafeGeom({_custom_geom_inner_code})" if _custom_geom_wrap_needed
        else _custom_geom_inner_code
    )

    # Inverse: one or more trainable variables (generalized from the single
    # trainable_variable this used to be limited to). Only the variable
    # *definitions* need to be unrolled per-variable here at generation
    # time -- every downstream use (external_trainable_variables,
    # VariableValue, print/save callbacks, the eval() namespace, the
    # convergence plot) operates generically over the _inv_vars /
    # _inv_var_names lists built right after these definitions run, so
    # none of that has to be unrolled.
    inv_vars_parsed = _parse_inverse_variables(config)
    _inv_var_def_lines = []
    for _iv_name, _iv_init, _iv_true in inv_vars_parsed:
        _inv_var_def_lines.append(f"    {_iv_name} = dde.Variable({_iv_init})")
        _inv_var_def_lines.append(
            f'    print(f"Inverse PINN: inferring {_iv_name}, init = {_iv_init}")')
    _inv_var_defs_code = "\n".join(_inv_var_def_lines)
    _inv_var_list_literal = "[" + ", ".join(n for n, _i, _t in inv_vars_parsed) + "]"
    # Recorded into model_config.json below so a later Model Restore knows
    # this run was Inverse and exactly which trainable variables it had --
    # restoring an Inverse checkpoint needs to recompile with the same
    # number of external_trainable_variables the optimizer was originally
    # given (init's own value doesn't matter for that -- only the count,
    # since neither the .pt file nor model.restore() ever stores/recovers a
    # trainable variable's actual value, only the network weights). "true"
    # is recorded too (None when not known) purely so a later Restore's
    # Parameter Convergence Plot/Animation can auto-fill the same true-value
    # dashed reference line the training-time plot already draws (see
    # main_window.py's _on_browse_restore_model auto-detect and
    # _build_restore_param_script) -- it plays no part in the restore
    # itself. Empty list for Forward configs, where none of this is
    # relevant.
    _mc_inv_vars_literal = (
        repr([{"name": n, "init": i, "true": t} for n, i, t in inv_vars_parsed])
        if config.problem_type == "Inverse" else "[]"
    )
    _inv_var_names_literal = repr([n for n, _i, _t in inv_vars_parsed])
    # One entry per trainable variable, the known ground-truth value for a
    # built-in template's auto-substituted PDE constant (see
    # main_window.py's INVERSE_AUTO_VARS), or None for a manually added
    # variable/legacy config with no known true value -- used only by the
    # parameter-convergence plot below to optionally draw a dashed
    # reference line at the true value, alongside the existing dashed
    # line at the run's own final inferred value.
    _inv_var_true_literal = repr([t for _n, _i, t in inv_vars_parsed])

    # Inverse: one or more measured-data files, each with its own output
    # column and its own loss weight (a separate observation loss term
    # per file). This is pure data (path/index/weight), not a Python
    # identifier, so -- unlike the variables above -- it needs no
    # per-entry unrolling: the whole parsed list is just one repr()
    # literal, looped over at runtime.
    obs_files_parsed = _parse_inverse_obs_files(config)
    _obs_files_literal = repr(obs_files_parsed)
    _obs_red = obs_reduce_args(config)
    _obs_reduce_def = _textwrap_obs.indent(OBS_REDUCE_HELPER.strip("\n"), "    ") + "\n" if _obs_red else ""
    _obs_reduce_call = (f"        _of_data = _obs_reduce(_of_data, _n_obs_coord_cols, _n_obs_coord_cols if _is_steady "
                        f"else _n_obs_coord_cols - 1, {_obs_red[0]!r}, {_obs_red[1]}, {_obs_red[2]})\n") if _obs_red else ""

    # Whether this run uses the NNCG optimizer anywhere (a Training Phase,
    # or the legacy phase-2 field for pre-scheduler configs) -- NNCG was
    # added in deepxde 1.13.0, but this app's minimum pinned version is
    # 1.10.0, so a version-guard block is only emitted into the generated
    # script when actually needed, printing a clear upgrade message instead
    # of letting an old deepxde installation hit a bare NotImplementedError
    # deep inside training. (The GUI itself also checks this at Solve time,
    # in _validate_optimizer_settings() -- this is the belt-and-suspenders
    # copy for a script saved and re-run standalone, possibly on a
    # different machine/environment than the one that generated it.)
    import json as _json_nncg
    try:
        _sched_phases_for_nncg = _json_nncg.loads(config.scheduler_phases) if config.scheduler_phases else []
    except (ValueError, TypeError):
        _sched_phases_for_nncg = []
    uses_nncg = (
        any(p.get("optimizer") == "nncg" for p in _sched_phases_for_nncg)
        or config.optimizer2 == "nncg"
    )
    _nncg_guard_code = ""
    if uses_nncg:
        _nncg_guard_code = '''
_dxde_ver_parts = dde.__version__.split(".")[:3]
try:
    _dxde_ver = tuple(int(_p) for _p in _dxde_ver_parts)
except ValueError:
    _dxde_ver = None
if _dxde_ver is not None and _dxde_ver < (1, 13, 0):
    print(f"ERROR: The NNCG optimizer requires deepxde>=1.13.0 "
          f"(you have {dde.__version__} installed). "
          f"Please upgrade: pip install --upgrade deepxde")
    raise SystemExit(1)
'''

    # Reproducibility: seed NumPy/PyTorch/random together via DeepXDE's own
    # helper, right before anything random happens (point sampling, network
    # weight init). Off entirely when use_random_seed is False -- emits
    # nothing, byte-for-byte the same as before this feature existed.
    # Deliberately NOT touching torch.backends.cudnn.deterministic below --
    # that would buy bit-for-bit GPU reproducibility too, at a real training-
    # speed cost that isn't worth it here.
    _seed_code = ""
    if config.use_random_seed:
        _seed_code = f'''dde.config.set_random_seed({config.random_seed})
'''

    # Weight decay (L2 regularization). 0.0 (the default) reproduces every
    # pre-existing generated script byte-for-byte in this section (an empty
    # regularization arg was never emitted before this feature existed).
    # Superseded by _make_net()'s own _weight_decay parameter (see
    # _net_construction_helper_code below), which additionally knows to
    # drop this entirely for PFNN (no `regularization` kwarg on DeepXDE's
    # PFNN) -- config.weight_decay is now passed straight through to
    # _make_net as a plain float at each call site instead of this being
    # pre-built into an arg-string splice.

    # Network architecture: FNN or DeepXDE's built-in PFNN class. See
    # _net_construction_helper_code()'s own docstring for the full design
    # -- this is the literal `_make_net(...)` helper's source, embedded
    # once near the top of the generated script below and called at every
    # network-construction site in place of a bare dde.nn.FNN(...).
    net_helper_code = _net_construction_helper_code(config.network_type, config.msffn_sigmas, config.msffn_sigmas_t)

    # Training Monitors' shared, static derivative-building/expression-
    # evaluation helpers -- see _training_monitor_runtime_code()'s own
    # docstring. Embedded once near the top of the generated script
    # (module level, alongside net_helper_code) regardless of whether
    # any monitors are actually configured -- cheap, and keeps this a
    # true no-op wiring point like _train_cbs_code below.
    _tm_runtime_code = _training_monitor_runtime_code()

    # Loss Plot Settings (Round 28) -- see _loss_plot_runtime_code()'s own
    # docstring. Embeds a single reusable `_make_loss_plot(...)` helper
    # once at module level (same pattern as net_helper_code/_make_net);
    # every loss-plotting call site (Standard/RAR path, Time-Adaptive
    # path, and generate_clean_script()'s own two analogous sites) then
    # just calls it with its own data/title/extra_fn, no further splicing.
    _loss_helper_code = _loss_plot_runtime_code(config)

    # Opt-in training callbacks (EarlyStopping/PDEPointResampler/
    # ModelCheckpoint/Timer/Training Monitors) -- one block for the
    # Standard path (built once, reused across every scheduler phase's
    # .train() call so state like Timer's running clock and
    # EarlyStopping's patience counter carry correctly across phases
    # within one run), one for the Time-Adaptive per-step loop (rebuilt
    # fresh each step, since each step is its own independent training
    # problem -- Training Monitors are deliberately NOT appended to this
    # one, see _build_training_monitors_code()'s docstring for why).
    _tm_cbs_code = _build_training_monitors_code(config, "_train_cbs", "_sol_dir", indent=4)
    _train_cbs_code = _build_train_cbs_code(config, "_train_cbs", indent=4) + ("\n" + _tm_cbs_code if _tm_cbs_code else "")
    _train_cbs_code_ta = _build_train_cbs_code(config, "_train_cbs_ta", indent=8)
    # Rendered at the very end of the script (see the {_tm_plot_code}
    # splice near the final "DONE" print below) -- reads each monitor's
    # own .txt/.meta.json back in and saves a line plot PNG, the same way
    # loss_plot.png/solution_plot.png already get saved, so the Results
    # panel can display it with no new rendering logic of its own.
    _tm_plot_code = _build_training_monitors_plot_code(config, "_sol_dir", indent=0)

    # IC pre-training needs at least 1 point sampled at the initial-time
    # slice, or DeepXDE's data construction fails outright (see the IC
    # pre-training blocks below for why). The GUI's own spinner is now
    # range-limited to >=1, but a config saved before that existed (or
    # hand-edited) could still carry 0 -- clamp here too as a second line
    # of defense. The field's own default (1000) is unaffected either way.
    _ic_pretrain_num_initial_safe = max(1, config.ic_pretrain_num_initial)
    # cleanup Round 47: the IC Pre-Training dataset/compile/restore-or-train
    # block, shared by both of this function's copies (Standard and
    # Time-Adaptive) and both of generate_clean_script()'s -- see
    # _ic_pretrain_train_code.
    _ic_pretrain_block_std = _ic_pretrain_train_code(
        config, "        ", "_ic_gt_pre", "_ic_ics_pre", _ic_pretrain_num_initial_safe,
        "_ic_only_weights", "net", f'_os.path.join({config.save_dir!r}, "ic_pretrain")',
        "_os", "  ", _safe_point_distribution)
    _ic_pretrain_block_ta = _ic_pretrain_train_code(
        config, "            ", "_ic_gt_pt", "_ic_constraints_pt", _ic_pretrain_num_initial_safe,
        "_ic_only_weights", "model_i.net", f'_os.path.join({config.save_dir!r}, "ic_pretrain")',
        "_os", "  ", _safe_point_distribution)

    script = f"""

import os
os.environ["DDE_BACKEND"] = "pytorch"
{_cpu_only_line(config)}
# ── Windows has no /tmp by default; make "/tmp/..." paths work there too ──
if os.name == "nt":
    _tmp_root = os.path.splitdrive(os.getcwd())[0] + os.sep + "tmp"
    os.makedirs(_tmp_root, exist_ok=True)

import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*cuBLAS.*")

# DeepXDE puts PyTorch on Apple's GPU (MPS) on a Mac. On an M1 it was never
# faster than the CPU for PINNs (2x slower on a small problem, even on 20-30k
# points; benchmarks/mac_gpu_check.py), and it has no float64, so PINNStudio
# trains on the CPU there.
if not torch.cuda.is_available() and str(getattr(torch, "get_default_device", lambda: "cpu")()).startswith("mps"):
    torch.set_default_device("cpu")
    print("  [Info] Training on the CPU (Apple GPU / MPS not used)")
{_cpu_threads_code(config)}

# ── Plot Settings: figure-size standardization ─────────────────
# Applied to every single-panel results plot below (Loss, Line, Surface,
# both Animation GIF types, Parameter Convergence, and their
# Time-Adaptive equivalents) via figsize=_plot_figsize(default_w,
# default_h) in place of a bare literal tuple -- "Default" mode (below)
# returns (default_w, default_h) unchanged, so every one of those call
# sites renders byte-identical output to before this existed unless the
# user has actually changed Plot Settings' Figure size option. NOT
# applied to the multi-column Error Analysis comparison grids (sized by
# however many files/columns are being compared) or the domain-sampling
# preview plots (sized to the problem's own geometry).
{_figsize_helper_code(config.plot_figsize_mode, config.plot_figsize_w, config.plot_figsize_h, getattr(config, "plot_grid", True), getattr(config, "plot_ref_style", "line"), number_settings(config), line_colors(config), font_sizes(config), anim_dpi(config), plot_extras(config))}

# Shared-color-scale contourf levels (see codegen.py's _contour_levels_helper_code).
{_contour_levels_helper_code()}

# Parameter-convergence axis drawing, shared with the exported script and
# Restore & Visualize (see codegen.py's _param_axis_helper_code).
{_param_axis_helper_code(config.plot_axis_label_fontsize, config.plot_subplot_title_fontsize, param_style_of(config))}
{_nncg_guard_code}

# ── Force GPU initialization ──────────────────────────────────
_effective_float = "{config.float_type}"
if "{config.lbfgs_float_type}" == "float64" and _effective_float == "float32":
    _effective_float = "float64"
    print("  [Info] Using float64 globally (required for L-BFGS float64 mode)")
dde.config.set_default_float(_effective_float)
{_seed_code}_gpu_device_index = {config.gpu_device_index}
_gpu_memory_fraction = {config.gpu_memory_fraction}
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(_gpu_device_index)

    # ── Maximize GPU memory usage ─────────────────────────────
    # Reserve up to gpu_memory_fraction of the selected device's memory
    # upfront (defaults: device 0, 95% -- unchanged from prior behavior;
    # both are configurable from the GUI's Hardware settings).
    torch.cuda.empty_cache()
    total_mem = torch.cuda.get_device_properties(_gpu_device_index).total_memory
    torch.cuda.set_per_process_memory_fraction(_gpu_memory_fraction, device=_gpu_device_index)

    # ── Performance settings ──────────────────────────────────
    _is_f64 = "{config.float_type}" == "float64"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = False

    # (No warm-up step here: a backward pass on a throwaway CUDA tensor before
    # training made every training step 5-10% slower on the GPU -- measured
    # with benchmarks/timing_ab.py; the first step warms CUDA up anyway.)

    _free, _total = torch.cuda.mem_get_info(_gpu_device_index)
    print(f"✅ GPU: {{torch.cuda.get_device_name(_gpu_device_index)}}")
    print(f"   Total memory: {{_total / 1e9:.1f}} GB")
    print(f"   Available:    {{_free / 1e9:.1f}} GB")
    print(f"   TF32:         False")
    print(f"   Float type:   {config.float_type}")
    print(f"   cuDNN bench:  False")
else:
    print("⚠️ No GPU found — running on CPU")

# ── Save directory setup ─────────────────────────────────────
import os as _os
_save_dir = {repr(config.save_dir)}.strip()
_use_save = bool(_save_dir)
if _use_save:
    _os.makedirs(_save_dir, exist_ok=True)
    import json as _json
    _model_config = {{
        "layers": {config.layers},
        "activation": {repr(config.activation)},
        # Network architecture -- read back by every Model Restore / Error
        # Analysis restore-script builder in main_window.py so a restored
        # net is reconstructed with the exact same shape it was trained
        # with (see _net_construction_helper_code() in this file). Absent
        # from a config saved before this feature existed; cfg.get(...) on
        # the restore side defaults to "FNN", i.e. the old behavior.
        "network_type": {repr(config.network_type)},
        "msffn_sigmas": {repr(config.msffn_sigmas)}, "msffn_sigmas_t": {repr(config.msffn_sigmas_t)},
        "num_outputs": {config.num_outputs},
        "output_names": {repr(config.output_names)},
        "x_min": {config.x_min}, "x_max": {config.x_max},
        "y_min": {config.y_min}, "y_max": {config.y_max},
        "z_min": {config.z_min}, "z_max": {config.z_max},
        "t_min": {config.t_min}, "t_max": {config.t_max},
        # 2D/3D Line-type plots' y/z slice -- read back by Restore &
        # Visualize's own "Line (time steps)"/"Animation Line (GIF)"
        # builders so a restored model's line plot uses the same slice
        # the run was trained/plotted with, not a different one chosen
        # later in the Restore panel. Absent from a config saved before
        # this feature existed; cfg.get(...) on the restore side defaults
        # to "auto" (domain midpoint), i.e. the old hardcoded behavior.
        "line_slice_y_auto": {config.line_slice_y_auto}, "line_slice_y": {config.line_slice_y},
        "line_slice_z_auto": {config.line_slice_z_auto}, "line_slice_z": {config.line_slice_z},
        "problem_dim": {repr(config.problem_dim)},
        # an ODE is solved as a 1D steady problem in x = t (_ode_to_1d):
        # Restore says ODE and plots it as one curve
        "ode": {bool(getattr(config, "_ode", False))},
        # Restoring a model has no other way to tell a steady-state
        # checkpoint from a transient one -- _build_restore_script()/
        # _build_restore_ea_script() in main_window.py both read this key
        # (defaulting to False, i.e. transient, when absent) to decide
        # whether model.predict(...)'s input arrays need a time column.
        # Missing here before this fix -- so EVERY restored steady-state
        # model was silently treated as transient, appending a time column
        # the restored network's first layer was never sized for, crashing
        # with "mat1 and mat2 shapes cannot be multiplied" the moment you
        # tried to plot or run Error Analysis against it.
        "steady_state": {config.steady_state},
        # Geometry definition -- without these, a restored model can only
        # ever be rebuilt against its rectangular/cuboid bounding box (see
        # _build_restore_script() in main_window.py), so a non-box geometry
        # (Disk/Ellipse/Triangle/Polygon/Sphere/Custom CSG combos) gets
        # predicted and plotted over the WRONG domain on restore -- the
        # actual shape, e.g. a triangular cavity, never gets reconstructed,
        # and nothing here lets the restored prediction be masked back down
        # to it either. Missing here before this fix.
        "geometry_type": {repr(config.geometry_type)},
        "geom_center_x": {config.geom_center_x}, "geom_center_y": {config.geom_center_y},
        "geom_center_z": {config.geom_center_z}, "geom_radius": {config.geom_radius},
        "geom_semi_major": {config.geom_semi_major}, "geom_semi_minor": {config.geom_semi_minor},
        "geom_angle": {config.geom_angle},
        "geom_triangle_vertices": {repr(config.geom_triangle_vertices)},
        "geom_polygon_vertices": {repr(config.geom_polygon_vertices)},
        "geom_custom_shapes_json": {repr(config.geom_custom_shapes_json)},
        "pde_expressions": {repr(config.pde_expressions)},
        "optimizer": {repr(config.optimizer)},
        "optimizer2": {repr(config.optimizer2)},
        "loss_type": {repr(config.loss_type)},
        "problem_type": {config.problem_type!r},
        "inverse_variables": {_mc_inv_vars_literal},
        {_transform_config_literal(config)}
    }}
    _os.makedirs(_os.path.join(_save_dir, "solution_results"), exist_ok=True)
    with open(_os.path.join(_save_dir, "solution_results", "model_config.json"), "w") as _mcf:
        _json.dump(_model_config, _mcf, indent=2)
    print(f"Model config saved to: {{_os.path.join(_save_dir, 'solution_results', 'model_config.json')}}")

# ── Solution results folder ───────────────────────────────────
_sol_dir = _os.path.join(_save_dir, "solution_results") if _use_save else "/tmp"
if _use_save:
    _os.makedirs(_sol_dir, exist_ok=True)
_loss_path     = _os.path.join(_sol_dir, "loss_plot.png")
_solution_path = _os.path.join(_sol_dir, "solution_plot.{sol_ext}")
_log_path      = _os.path.join(_sol_dir, "training_log.txt") if _use_save else None

# ── Problem dimension ─────────────────────────────────────────
_is_2d = "{config.problem_dim}" == "2D"
_is_3d = "{config.problem_dim}" == "3D"
# Steady-state (time-independent) problem, e.g. a Poisson equation -- no
# time axis on the network input, no GeometryXTime, no Initial Condition,
# a plain dde.data.PDE instead of dde.data.TimePDE. See _build_geom() and
# the "Data" section below for where this actually changes construction.
_is_steady = {str(is_steady)}
_v1d = {v1!r}

# Load an Initial Condition from a plain, header-less data file and
# return (xt, vals) ready for dde.icbc.PointSetBC -- xt at the domain's
# START time ({config.t_min}), since that's where an IC is defined by
# default. Column layout matches the problem's dimension, one
# coordinate column per spatial axis, then time, then value:
#   1D: x, t, c        2D: x, y, t, c        3D: x, y, z, t, c
# No header row. Rows are filtered to whichever ones already sit at
# t == {config.t_min} (small tolerance for floating-point data) --
# a file covering the whole time range works fine, only its t=t_min
# slice is used as the IC.
def _load_ic_from_file(_path):
    _rows = []
    with open(_path) as _fh:                 # commas or spaces; header/comment lines skipped
        for _line in _fh:
            try:
                _rows.append([float(_v) for _v in _line.replace(",", " ").split()])
            except ValueError:
                continue
    _raw = np.array([_r for _r in _rows if _r])
    _n_coord = 3 if _is_3d else (2 if _is_2d else 1)
    _t_col = _n_coord
    _v_col = _n_coord + 1
    _t_start = {config.t_min}
    _mask = np.abs(_raw[:, _t_col] - _t_start) < 1e-8
    _n_matched = int(_mask.sum())
    if _n_matched == 0:
        raise ValueError(
            f"IC file {{_path}} has no rows at t = {{_t_start}} (the domain's "
            f"start time) -- expected a header-less file with columns "
            f"{{'x, t, c' if not (_is_2d or _is_3d) else ('x, y, t, c' if _is_2d else 'x, y, z, t, c')}}, "
            f"with at least one row at t = {{_t_start}}."
        )
    _coords = _raw[_mask, :_n_coord]
    _t0 = np.full((_n_matched, 1), _t_start)
    _xt = np.hstack([_coords, _t0])
    _vals = _raw[_mask, _v_col:_v_col + 1]
    return _xt, _vals

# Optional input/output transform, applied to every net this script
# builds (the one that actually trains, and every per-step net rebuilt
# later purely to restore a checkpoint for plotting/error-analysis --
# restoring only loads weights, not a transform, since apply_feature_
# transform/apply_output_transform just attach a plain Python closure to
# the net instance rather than something saved in a .pt checkpoint, so a
# reloaded net needs the same transform re-applied to predict correctly).
{net_helper_code}

{_tm_runtime_code}

{_loss_helper_code}

{_net_transforms_code(config)}

# ── Parametric study setup ────────────────────────────────────
_parametric = {config.parametric_study}
_param_name  = "{config.parametric_param}"
_param_values_str = "{config.parametric_values}"

if _parametric:
    _param_values = [v.strip() for v in _param_values_str.split(",") if v.strip()]
    print(f"=== Parametric Study: {{_param_name}} = {{_param_values}} ===")
else:
    _param_values = [None]

# ── Inverse PINN setup ───────────────────────────────────────
_problem_type = "{config.problem_type}"

if _problem_type == "Inverse":
    import pandas as _pd

{_inv_var_defs_code}
    _inv_vars = {_inv_var_list_literal}
    _inv_var_names = {_inv_var_names_literal}
    _inv_var_trues = {_inv_var_true_literal}

    def _load_data(fpath):
        # Numbers separated by commas or spaces; a header line, comments
        # (# or %) and blank lines are skipped -- and only those (the first
        # line used to be dropped always, losing a point of a file without
        # a header).
        _rows = []
        with open(fpath) as _fh:
            for _line in _fh:
                _parts = _line.replace(",", " ").split()
                if not _parts or _parts[0][0] in "#%":
                    continue
                try:
                    _rows.append([float(_v) for _v in _parts])
                except ValueError:
                    continue
        return np.array(_rows)

    # Number of leading coordinate columns in a measured-data file (before
    # its value column): x/y/z plus a time column, EXCEPT for a steady-state
    # problem, which has no time axis at all -- same distinction
    # _n_coord_cols_bc already makes for BC-panel points below. Getting this
    # wrong for a steady 2D/3D template (assuming a "t" column that doesn't
    # exist) would silently read the value column as coordinates and vice
    # versa instead of raising, so this is computed generically rather than
    # copying the old is_3d/is_2d-only ladder that never checked _is_steady.
    _n_obs_coord_cols = ((3 if _is_3d else (2 if _is_2d else 1)) if _is_steady
                         else (4 if _is_3d else (3 if _is_2d else 2)))

    def _make_obs_func(_expr, _onames):
        # Builds a dde.icbc.PointSetOperatorBC-compatible func(inputs,
        # outputs, X) for a measured-data file whose value is a DERIVED
        # field (e.g. 1D Schrodinger's |h| = sqrt(u**2+v**2)) rather than
        # a single raw output column -- same expression syntax as the
        # Results panel's "Custom..." plot field (_extract_plot_field
        # below), but evaluated on live torch tensors during training
        # instead of a numpy array after predict(), so it uses real torch
        # ops (gradient-safe) rather than np.* (which errors or silently
        # detaches on a tensor that requires grad).
        _code_obs = compile(_expr, "<obs_custom_field>", "eval")
        def _f(inputs, outputs, X):
            _ns_obs = {{
                "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
                "arcsin": torch.arcsin, "arccos": torch.arccos, "arctan": torch.arctan,
                "exp": torch.exp, "log": torch.log, "log10": torch.log10,
                # +1e-12 baked into sqrt itself here (not just relying on a
                # default custom_expr string like "sqrt(u**2+v**2+1e-12)"
                # already spelling it out, e.g. 1D Schrodinger's |h|
                # default) -- unlike a plotting-only sqrt evaluation, this
                # namespace is what DeepXDE actually differentiates through
                # during training via PointSetOperatorBC, and
                # d/du sqrt(u**2+v**2) = u/sqrt(u**2+v**2) is genuinely
                # singular at u=v=0, which a freshly-initialized network can
                # (and does) land on exactly at some collocation points
                # early in training, producing an inf/nan gradient that
                # poisons every other loss term. Guarding the function
                # itself protects ANY custom expression using sqrt(...),
                # including ones a user hand-types without adding their own
                # epsilon, not just whichever default happens to spell it
                # out already. The shift is far below any physically
                # meaningful value for a sum-of-squares argument, so it
                # doesn't change what's being fit.
                "sqrt": lambda _v: torch.sqrt(_v + 1e-12), "abs": torch.abs, "ceil": torch.ceil,
                "floor": torch.floor, "pi": np.pi, "torch": torch,
            }}
            for _oi_ob, _on_ob in enumerate(_onames):
                _ns_obs[_on_ob.strip()] = outputs[:, _oi_ob:_oi_ob + 1]
            _r_ob = eval(_code_obs, _ns_obs)
            return _r_ob if _r_ob.dim() == 2 else _r_ob.reshape(-1, 1)
        return _f

    _obs_output_names_list = {repr(config.output_names)}.split(",")
    _obs_files = {_obs_files_literal}
    _obs_entries = []  # list of (xt, u, output_idx, weight, custom_expr), one per measured-data file
{_obs_reduce_def}    for _of in _obs_files:
        _of_data = _load_data(_of["path"])
{_obs_reduce_call}        _of_xt = _of_data[:, 0:_n_obs_coord_cols]
        _of_u  = _of_data[:, _n_obs_coord_cols:_n_obs_coord_cols + 1]
        _obs_entries.append((_of_xt, _of_u, _of.get("output_idx", 0),
                              _of.get("weight", 100.0), _of.get("custom_expr", "")))
        print(f"Loaded {{len(_of_xt)}} observation points from {{_of['path']}} (output {{_of.get('output_idx', 0)}}, weight {{_of.get('weight', 100.0)}})")
    # Kept as aliases to the first measured-data file's points, in case
    # anything downstream still expects the old single-file names.
    _obs_xt = _obs_entries[0][0] if _obs_entries else None
    _obs_u  = _obs_entries[0][1] if _obs_entries else None

    if "{config.inverse_ic_type}" == "File (x, t, u)":
        _ic_data = _load_data({repr(config.inverse_ic_file)})
        if _is_3d:
            _ic_xt = _ic_data[:, 0:4]  # x, y, z, t
            _ic_u  = _ic_data[:, 4:5]  # u
        elif _is_2d:
            _ic_xt = _ic_data[:, 0:3]  # x, y, t
            _ic_u  = _ic_data[:, 3:4]  # u
        else:
            _ic_xt = _ic_data[:, 0:2]  # x, t
            _ic_u  = _ic_data[:, 2:3]  # u
        print(f"Loaded {{len(_ic_xt)}} IC points from file")

    def _split_param_history_line(line):
        # Robustly split one dde.callbacks.VariableValue output line,
        # "<iter> [<v1>, <v2>, ...]", into (iter:int, raw_values_text:str)
        # without reparsing/reformatting the numeric values -- so the
        # precision the callback wrote is preserved exactly when this is
        # used to merge/append history files. Returns None if the line
        # doesn't look like a VariableValue line at all. Replaces the old
        # `.replace("[","").replace("]","").split()` approach, which broke
        # as soon as more than one comma-separated value appeared inside
        # the brackets (whitespace-split leaves a trailing comma on each
        # value, which float() rejects).
        line = line.strip()
        if not line or "[" not in line or "]" not in line:
            return None
        try:
            i0 = line.index("[")
            i1 = line.rindex("]")
            return int(float(line[:i0].strip())), line[i0 + 1:i1]
        except (ValueError, IndexError):
            return None

    class _PrintParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period=1000):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self._last_bucket = 0
        def set_offset(self, offset):
            # No-op, kept so older scheduler-loop code can still call it.
            # Real progress now comes from the model's own train_state.step,
            # which is already cumulative across every phase — no manual
            # offset bookkeeping needed. This also fixes L-BFGS: DeepXDE only
            # calls on_batch_end once per outer L-BFGS step (which can cover
            # hundreds of real iterations at once), not once per real
            # iteration the way Adam does, so counting callback firings
            # (the old approach) almost never reached the print period.
            pass
        def on_batch_end(self):
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
                print(f"  [{{self.name}}] Iter {{cur}}: {{val:.6g}}", flush=True)

    _print_cbs = [_PrintParamCallback(_iv, _in, period={max(1, int(config.loss_display_every))})
                  for _iv, _in in zip(_inv_vars, _inv_var_names)]

    # Parameter saving to text file -- one convergence file per variable,
    # all sharing the same save-period setting from the panel.
    _param_save_period = {inv_param_period(config)}   # Inverse -> Save parameters
    _param_save_paths = [
        _os.path.join(_save_dir if _use_save else "/tmp", f"{{_in}}_convergence.txt")
        for _in in _inv_var_names
    ] if _param_save_period > 0 else [None] * len(_inv_var_names)

    if _param_save_period > 0:
        for _in, _ip in zip(_inv_var_names, _param_save_paths):
            with open(_ip, "w") as _psf:
                _psf.write(f"iteration,{{_in}}\\n")
        print(f"Parameter convergence will be saved to: {{_param_save_paths}}")

    class _SaveParamCallback(dde.callbacks.Callback):
        def __init__(self, var, name, period, path):
            super().__init__()
            self.var = var
            self.name = name
            self.period = period
            self.path = path
            self._last_bucket = 0
            self._last_step = None
        def set_offset(self, offset):
            # No-op — see _PrintParamCallback.set_offset above.
            pass
        def _write(self, cur):
            val = self.var.detach().cpu().numpy().item() if hasattr(self.var, 'detach') else float(self.var.numpy())
            with open(self.path, "a") as _f:
                _f.write(f"{{cur}},{{val:.10g}}\\n")
            self._last_step = cur
        def on_batch_end(self):
            if self.period == 0: return
            cur = self.model.train_state.step
            bucket = cur // self.period
            if bucket > self._last_bucket:
                self._last_bucket = bucket
                self._write(cur)
        def on_train_end(self):
            # Also the value where training stopped (e.g. an L-BFGS phase that
            # converged before the next save period), so the saved curve
            # ends at the final value the run reports.
            if self.period == 0: return
            cur = self.model.train_state.step
            if cur != self._last_step:
                self._write(cur)

    _save_cbs = [
        _SaveParamCallback(_iv, _in, _param_save_period, _ip if _ip else f"/tmp/{{_in}}_param_save.txt")
        for _iv, _in, _ip in zip(_inv_vars, _inv_var_names, _param_save_paths)
    ]


{_fecr_pde_block}
{_integral_helper_code(config)}
{_frac_helper_code(config)}
_IS_FECR = False
# ── PDE definition (standard) ───────────────────────────────
# The equations are compiled once here; eval() of the text itself would
# parse it again at every training step.
_PDE_CODES = ([compile({repr(pde_expr_single)}, "<PDE>", "eval")] if {_n_equations(config)} == 1 else
              [compile(_e.strip(), "<PDE>", "eval")
               for _e in {repr(config_pde_expressions)}.split("|")[:{_n_equations(config)}]])
def _pde_standard(x, y{", _int_mat=None" if "_frac(" in (config.pde_expressions or "") else ""}):
    _n_out = {config.num_outputs}
    _n_eq = {_n_equations(config)}
    _out_names = [n.strip() for n in {repr(config.output_names)}.split(",")]
    _is_2d_pde = "{config.problem_dim}" == "2D"
    _is_3d_pde = "{config.problem_dim}" == "3D"

    def _hess(ys, xs, i, j):
        # DeepXDE 1.10.0 (the floor version pinned in setup.py) raises
        # "Do not use component for 1D y." if `component` is passed at all
        # for a single-output model, while multi-output models require it.
        # Newer deepxde releases accept it either way, so omitting `component`
        # only for the single-output case is safe across the whole supported range.
        if _n_out == 1:
            return dde.grad.hessian(ys, xs, i=i, j=j)
        return dde.grad.hessian(ys, xs, component=_oi, i=i, j=j)

    _dvars = {{}}
    for _oi, _oname in enumerate(_out_names):
        _dvars[_oname] = y[:, _oi:_oi+1]
        if _is_3d_pde:
            # 3D: inputs are (x, y, z, t) → j=0,1,2,3 -- or, for a steady
            # (time-independent) problem, just (x, y, z) → j=0,1,2, with no
            # time column at all, so the unconditional "_t" jacobian below
            # (which would otherwise index a column that doesn't exist) is
            # skipped entirely rather than computed and left unused.
            # Always compute first-order derivatives
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _dvars[f"d{{_oname}}_z"] = dde.grad.jacobian(y, x, i=_oi, j=2)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=3)
            # Only compute second-order derivatives if used in PDE (pure and
            # mixed). 4th-order pure/mixed terms (xxxx/yyyy/zzzz/xxyy/xxzz/
            # yyzz/xxtt/yytt/zztt) are also supported, mirroring the 2D/1D
            # cases below, computed as the hessian of the matching 2nd-order
            # term (which the substring check above already guarantees gets
            # built, since e.g. "xx" is a substring of "xxxx"/"xxyy"/"xxzz"/
            # "xxtt").
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_yy" in _pde_str_check:
                _dvars[f"d{{_oname}}_yy"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_zz" in _pde_str_check:
                _dvars[f"d{{_oname}}_zz"] = _hess(y, x, 2, 2)
            if f"d{{_oname_check}}_tt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 3, 3)
            if f"d{{_oname_check}}_xy" in _pde_str_check:
                _dvars[f"d{{_oname}}_xy"] = _hess(y, x, 0, 1)
            if f"d{{_oname_check}}_xz" in _pde_str_check:
                _dvars[f"d{{_oname}}_xz"] = _hess(y, x, 0, 2)
            if f"d{{_oname_check}}_yz" in _pde_str_check:
                _dvars[f"d{{_oname}}_yz"] = _hess(y, x, 1, 2)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 3)
            if f"d{{_oname_check}}_yt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yt"] = _hess(y, x, 1, 3)
            if f"d{{_oname_check}}_zt" in _pde_str_check:
                _dvars[f"d{{_oname}}_zt"] = _hess(y, x, 2, 3)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_yyyy" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=1, j=1)
            if f"d{{_oname_check}}_zzzz" in _pde_str_check and f"d{{_oname}}_zz" in _dvars:
                _dvars[f"d{{_oname}}_zzzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_zz"], x, i=2, j=2)
            if f"d{{_oname_check}}_xxyy" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxzz" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=2, j=2)
            if f"d{{_oname_check}}_yyzz" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyzz"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=2, j=2)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=3, j=3)
            if f"d{{_oname_check}}_yytt" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yytt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=3, j=3)
            if f"d{{_oname_check}}_zztt" in _pde_str_check and f"d{{_oname}}_zz" in _dvars:
                _dvars[f"d{{_oname}}_zztt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_zz"], x, i=3, j=3)
        elif _is_2d_pde:
            # 2D: inputs are (x, y, t) → j=0,1,2 -- or, for a steady
            # (time-independent) problem, just (x, y) → j=0,1, so the
            # unconditional "_t" jacobian below (which would otherwise
            # index a column that doesn't exist) is skipped entirely.
            # Always compute first-order derivatives
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            _dvars[f"d{{_oname}}_y"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=2)
            # Only compute higher-order derivatives if used in PDE
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_yy" in _pde_str_check or f"d{{_oname_check}}_yyyy" in _pde_str_check or f"d{{_oname_check}}_xxyy" in _pde_str_check or f"d{{_oname_check}}_yytt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yy"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_xy" in _pde_str_check:
                _dvars[f"d{{_oname}}_xy"] = _hess(y, x, 0, 1)
            if f"d{{_oname_check}}_tt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 2, 2)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 2)
            if f"d{{_oname_check}}_yt" in _pde_str_check:
                _dvars[f"d{{_oname}}_yt"] = _hess(y, x, 1, 2)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_yyyy" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yyyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxyy" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxyy"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=2, j=2)
            if f"d{{_oname_check}}_yytt" in _pde_str_check and f"d{{_oname}}_yy" in _dvars:
                _dvars[f"d{{_oname}}_yytt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_yy"], x, i=2, j=2)
        else:
            # 1D: inputs are (x, t) → j=0,1 -- or, for a steady
            # (time-independent) problem, just (x) → j=0, so the
            # unconditional "_t" jacobian below (which would otherwise
            # index a column that doesn't exist) is skipped entirely.
            _dvars[f"d{{_oname}}_x"] = dde.grad.jacobian(y, x, i=_oi, j=0)
            if not _is_steady:
                _dvars[f"d{{_oname}}_t"] = dde.grad.jacobian(y, x, i=_oi, j=1)
            _pde_str_check = {repr(config_pde_expressions)}
            _oname_check = _oname
            if f"d{{_oname_check}}_xx" in _pde_str_check or f"d{{_oname_check}}_xxx" in _pde_str_check or f"d{{_oname_check}}_xxxx" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xx"] = _hess(y, x, 0, 0)
            if f"d{{_oname_check}}_tt" in _pde_str_check or f"d{{_oname_check}}_tttt" in _pde_str_check or f"d{{_oname_check}}_xxtt" in _pde_str_check:
                _dvars[f"d{{_oname}}_tt"] = _hess(y, x, 1, 1)
            if f"d{{_oname_check}}_xt" in _pde_str_check:
                _dvars[f"d{{_oname}}_xt"] = _hess(y, x, 0, 1)
            # Third-order pure x-derivative (e.g. for the KdV equation's
            # u_xxx dispersive term): one more jacobian pass on top of the
            # already-computed u_xx, rather than a hessian (which would
            # give the 4th-order xxxx term instead -- see below).
            if f"d{{_oname_check}}_xxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxx"] = dde.grad.jacobian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_xxxx" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxxx"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=0, j=0)
            if f"d{{_oname_check}}_xxtt" in _pde_str_check and f"d{{_oname}}_xx" in _dvars:
                _dvars[f"d{{_oname}}_xxtt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_xx"], x, i=1, j=1)
            if f"d{{_oname_check}}_tttt" in _pde_str_check and f"d{{_oname}}_tt" in _dvars:
                _dvars[f"d{{_oname}}_tttt"] = dde.grad.hessian(_dvars[f"d{{_oname}}_tt"], x, i=1, j=1)

    _eval_ns = {{**_dvars}}
    _eval_ns["dde"] = dde
    _eval_ns["np"]  = np
    _eval_ns["x"]   = x
    _eval_ns["y"]   = y
    _eval_ns["torch"] = torch{pde_coord_ns_code}
    if "_integral" in globals():
        _eval_ns["_integral"] = _integral
    if "_frac_apply" in globals():
        _eval_ns["_frac"] = lambda _v: _frac_apply(_int_mat, _v)

    if _problem_type == "Inverse":
        for _iv_name, _iv_var in zip(_inv_var_names, _inv_vars):
            _eval_ns[_iv_name] = _iv_var

    if _n_eq == 1:
        _res = eval(_PDE_CODES[0], _eval_ns)
    else:
        _res = [eval(_c, _eval_ns) for _c in _PDE_CODES]
    if "_frac_apply" in globals() and _int_mat is not None:
        # fractional problem: the residual lives on the FPDE points only
        _nr = _frac_rows(_int_mat)
        _res = [_r[:_nr] for _r in _res] if isinstance(_res, list) else _res[:_nr]
    return _res
if not _IS_FECR:
    pde = _pde_standard
{_training_methods_defs(config)}
# ── Geometry & Time domain ──────────────────────────────────
# _build_geom() is the one place that turns the Geometry Type selector
# (Rectangle/Disk/Ellipse/Triangle/Polygon in 2D; Cuboid/Sphere in 3D;
# Interval for 1D) into an actual DeepXDE geometry object -- every other
# geometry-construction site in this script (IC pre-training, Time-Adaptive
# per-step) calls this same helper instead of re-deriving the shape, so
# there's exactly one place that needs to know each shape's constructor.
_geom_type = {_geometry_type_literal}
# The legacy per-side BC scheme below (bc_left/right/bottom/top) only ever
# meant "x = x_min" / "x = x_max" / etc, so it only makes sense for a box
# shape -- it's kept purely for configs saved before the Boundary
# Conditions panel existed, and every one of those was necessarily a box
# shape (Disk/Ellipse/Triangle/Polygon/Sphere didn't exist as options back
# then). So for a non-box shape, an empty panel should just mean "no
# custom BCs" (PDE + IC only) rather than silently reinterpreting hidden,
# always-on left/right/bottom/top widgets that don't correspond to
# anything the user can see or edit for that shape.
_box_shape_geom = _geom_type in ("Interval", "Rectangle", "Cuboid")

# Disk/Ellipse/Triangle/Polygon/Sphere -- in this installed DeepXDE version,
# some of these (observed on Ellipse.random_points, Triangle.random_points,
# Polygon.random_boundary_points) don't consistently respect
# dde.config.set_default_float(): they return float64 point arrays even
# when the network itself is float32, which crashes training the moment
# DeepXDE evaluates those points through the net (dtype mismatch error).
# Rather than force the whole run to float64 (worse memory/speed) just for
# these shapes, wrap them so every sampled point array is cast to whatever
# float type the run is actually using -- everything else (on_boundary,
# inside, bbox, dim, ...) passes straight through to the real geometry via
# __getattr__, same adapter pattern as _Interface2DGeomAdapter below.
{_dtype_safe_geom_class_code()}

def _build_geom():
    if _geom_type == "Disk":
        return _DTypeSafeGeom(dde.geometry.Disk([{config.geom_center_x}, {config.geom_center_y}], {config.geom_radius}))
    elif _geom_type == "Ellipse":
        return _DTypeSafeGeom(dde.geometry.Ellipse([{config.geom_center_x}, {config.geom_center_y}],
                                     {config.geom_semi_major}, {config.geom_semi_minor}, {config.geom_angle}))
    elif _geom_type == "Triangle":
        _tv = {_triangle_vertices_literal}
        return _DTypeSafeGeom(dde.geometry.Triangle(_tv[0], _tv[1], _tv[2]))
    elif _geom_type == "Polygon":
        return _DTypeSafeGeom(dde.geometry.Polygon({_polygon_vertices_literal}))
    elif _geom_type == "Sphere":
        return _DTypeSafeGeom(dde.geometry.Sphere([{config.geom_center_x}, {config.geom_center_y}, {config.geom_center_z}],
                                    {config.geom_radius}))
    elif _geom_type == "Custom":
        return {_custom_geom_final_literal}
    elif _is_2d:
        return dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])
    elif _is_3d:
        return dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}],
                                    [{config.x_max}, {config.y_max}, {config.z_max}])
    else:
        return dde.geometry.Interval({config.x_min}, {config.x_max})

geom = _build_geom()
if _is_steady:
    # No time axis at all -- every downstream BC constructor (DirichletBC,
    # NeumannBC, PeriodicBC, OperatorBC, ...) only ever needs a plain
    # Geometry (on_boundary/random_points/bbox/...), not specifically a
    # GeometryXTime, so aliasing geomtime straight to geom lets every BC-
    # construction call site below stay completely unchanged. Only the IC
    # section (which needs GeometryXTime's on_initial) and the "Data"
    # section (TimePDE vs plain PDE) below actually branch on _is_steady.
    geomtime = geom
else:
    timedomain = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)
{_formula_reference_code(config, "geom")}

# Bounding box of the ACTUAL shape (every DeepXDE geometry exposes this),
# used for plot/export grids so a Disk/Ellipse/Triangle/Polygon/Sphere gets
# plotted over its own true extent instead of the plain x_min/x_max/y_min/
# y_max fields, which only mean something for the box shapes (those fields
# are hidden in the UI for every other shape and can hold stale values).
_geom_bbox_arr = np.asarray(geom.bbox, dtype=float)
_plot_x_min, _plot_x_max = float(_geom_bbox_arr[0][0]), float(_geom_bbox_arr[1][0])
_plot_y_min = float(_geom_bbox_arr[0][1]) if _geom_bbox_arr.shape[1] > 1 else 0.0
_plot_y_max = float(_geom_bbox_arr[1][1]) if _geom_bbox_arr.shape[1] > 1 else 1.0
_plot_z_min = float(_geom_bbox_arr[0][2]) if _geom_bbox_arr.shape[1] > 2 else {config.z_min}
_plot_z_max = float(_geom_bbox_arr[1][2]) if _geom_bbox_arr.shape[1] > 2 else {config.z_max}

# ── Boundary & Initial conditions ────────────────────────────
_out_names_list  = {repr(config.output_names)}.split(",")
_bc_left_types   = "{config.bc_left_types}".split(",")
_bc_right_types  = "{config.bc_right_types}".split(",")
_bc_left_values  = "{config.bc_left_values}".split(",")
_bc_right_values = "{config.bc_right_values}".split(",")
_bc_left_active  = "{config.bc_left_active}".split(",")
_bc_right_active = "{config.bc_right_active}".split(",")
_bc_left_deriv_list  = "{config.bc_left_deriv}".split(",")
_bc_right_deriv_list = "{config.bc_right_deriv}".split(",")

_bc_bottom_types  = "{config.bc_bottom_types}".split(",")
_bc_top_types     = "{config.bc_top_types}".split(",")
_bc_bottom_values = "{config.bc_bottom_values}".split(",")
_bc_top_values    = "{config.bc_top_values}".split(",")
_bc_bottom_active = "{config.bc_bottom_active}".split(",")
_bc_top_active    = "{config.bc_top_active}".split(",")
_bc_bottom_deriv_list = "{config.bc_bottom_deriv}".split(",")
_bc_top_deriv_list    = "{config.bc_top_deriv}".split(",")

_ic_expressions = "{config_ic_expressions}".split("|")
_ic_active_list = "{config.ic_active}".split(",")

_constraints = []

# ── Boundary conditions ──────────────────────────────────────
# The Boundary Conditions panel (custom_bc_json) is now the single
# source of truth for BCs -- whatever the panel shows (hand-built,
# or pre-filled-but-editable from a template) is exactly what
# trains. The legacy per-side fields below are kept only as a
# fallback for configs saved before this panel existed.
import json as _bc_json_mod
_custom_bc_entries = _bc_json_mod.loads({_custom_bc_json_literal}) if {_custom_bc_json_literal} else []
_bc_panel_data_present = {_bc_panel_data_present_literal}
# One fewer coordinate column for a steady-state problem -- no time axis
# at all, so a BC point's last column is whatever spatial axis is last
# (z for 3D, y for 2D, x for 1D), not time.
_n_coord_cols_bc = (3 if _is_3d else (2 if _is_2d else 1)) if _is_steady else (4 if _is_3d else (3 if _is_2d else 2))


# Named math functions/constants available directly (unprefixed) in every
# Boundary Conditions panel expression field ("Where:"/"Value:"), matching
# the fields' own placeholder examples (e.g. "sin(pi*y)") -- unlike the
# PDE/IC expression boxes, these are NOT run through a "np." text-prefixing
# pass first (see _simplify_expr above), so without this dict "sin(pi*y)"
# or "exp(...)" would raise a plain NameError the first time anyone typed
# one, since eval()'s namespace otherwise only has x/y/z/t/np in it.
{_math_namespace_code('_BC_MATH_NS', 'np', '')}

# The domain's own limits, usable by name in every Boundary Conditions
# expression (e.g. "x <= x_min", "x_max - x"), and the named sides of the
# domain box usable in "Where:" (Round 68): left/right (x = x_min/x_max),
# bottom/top (y = y_min/y_max, 2D/3D), front/back (z = z_min/z_max, 3D).
_BC_DOMAIN = {_bc_domain_literal(config)}
_BC_SIDES = {_bc_sides_literal(config)}

def _bc_loc_fn(_expr):
    _src = _expr if _expr.strip() else "True"
    _code = compile(_src, "<bc_location>", "eval")
    _sides = [(_n, _c, _v) for (_n, _c, _v) in _BC_SIDES if _n in _code.co_names]
    def _f(_X, _on):
        if not _on:
            return False
        _ns = {{**_BC_MATH_NS, **_BC_DOMAIN, "x": _X[0], "np": np}}
        if not _is_steady:
            _ns["t"] = _X[_n_coord_cols_bc - 1]
        if _is_2d or _is_3d:
            _ns["y"] = _X[1]
        if _is_3d:
            _ns["z"] = _X[2]
        for _n, _c, _v in _sides:
            _ns[_n] = bool(np.isclose(_X[_c], _v))
        return bool(eval(_code, _ns))
    return _f

def _bc_init_loc_fn(_expr):
    # Operator BC applied at the initial time t = t_min (Round 64) instead
    # of on the spatial boundary -- e.g. a wave equation's du_t = 0 at t = 0.
    # TimePDE's training points include the initial-time points, so the
    # filter finds them; the location expression still applies on top.
    _inner = _bc_loc_fn(_expr)
    def _f(_X, _on):
        return bool(np.isclose(_X[_n_coord_cols_bc - 1], {config.t_min})) and _inner(_X, True)
    return _f

def _bc_val_fn(_expr):
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_value>", "eval")
    def _f(_Xb):
        _ns = {{**_BC_MATH_NS, **_BC_DOMAIN, "x": _Xb[:, 0], "np": np}}
        if not _is_steady:
            _ns["t"] = _Xb[:, _n_coord_cols_bc - 1]
        if _is_2d or _is_3d:
            _ns["y"] = _Xb[:, 1]
        if _is_3d:
            _ns["z"] = _Xb[:, 2]
        _r = np.asarray(eval(_code, _ns), dtype=float)
        return np.full((len(_Xb), 1), float(_r)) if _r.ndim == 0 else _r.reshape(-1, 1)
    return _f

def _bc_robin_fn(_expr, _comp):
    # Robin's func(X, y) is NOT wrapped by DeepXDE the way Dirichlet/Neumann
    # are (no return_tensor()) -- error() subtracts our return value directly
    # from a live torch tensor (normal_derivative(...)), so whatever this
    # returns has to already be a torch tensor on the same device/dtype as
    # _Yb, whether the expression references u (already a tensor) or is a
    # plain numpy/constant expression in x, y, z.
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_robin_value>", "eval")
    def _f(_Xb, _Yb):
        _ns = {{**_BC_MATH_NS, **_BC_DOMAIN, "x": _Xb[:, 0], "np": np, "torch": torch}}
        # Round 75: every output by name (e.g. -k0*u_im for a Helmholtz
        # absorbing condition on u_re), then u = this row's own output.
        for _oi, _on in enumerate(_out_names_list):
            if _on.strip():
                _ns[_on.strip()] = _Yb[:, _oi:_oi + 1]
        _ns["u"] = _Yb[:, _comp:_comp + 1]
        if _is_2d or _is_3d:
            _ns["y"] = _Xb[:, 1]
        if _is_3d:
            _ns["z"] = _Xb[:, 2]
        _r = eval(_code, _ns)
        if isinstance(_r, torch.Tensor):
            return _r if _r.dim() == 2 else _r.reshape(-1, 1)
        _arr = np.asarray(_r, dtype=float)
        if _arr.ndim == 0:
            return _Yb.new_full((len(_Xb), 1), float(_arr))
        return _Yb.new_tensor(_arr.reshape(-1, 1))
    return _f

def _bc_operator_fn(_expr):
    # DeepXDE's own real signature for OperatorBC/PointSetOperatorBC's
    # func (see each class's docstring): inputs/outputs are the network's
    # raw input/output tensors, X is the plain NumPy array of inputs --
    # all three already exposed below, unchanged. Additionally (new),
    # the same derivative shorthand the PDE box/Training Monitors/Custom
    # plot expressions already use (u, du_x, du_xx, ...) is available too,
    # via Training Monitors' own _tm_build_dvars helper (already defined
    # earlier in this script, unconditionally) -- so an expression can
    # use either vocabulary, or mix them, without hand-writing raw
    # dde.grad.jacobian/hessian calls.
    _src = _expr if _expr.strip() else "0"
    _code = compile(_src, "<bc_operator>", "eval")
    _bc_op_out_names = [_n.strip() for _n in {repr(config.output_names)}.split(",") if _n.strip()]
    _bc_op_n_out = len(_bc_op_out_names) or 1
    _bc_op_dim = "3D" if _is_3d else ("2D" if _is_2d else "1D")
    def _f(_inputs, _outputs, _X):
        _dvars = _tm_build_dvars(_inputs, _outputs, _bc_op_n_out, _bc_op_out_names,
                                  _is_steady, _bc_op_dim, _expr)
        _ns = dict(_dvars)
        _ns.update({{**_BC_DOMAIN, "inputs": _inputs, "outputs": _outputs, "X": _X, "np": np, "torch": torch}})
        return eval(_code, _ns)
    return _f

def _bc_load_points_file(_path, _bc_label, _n_values=1):
    if not _path or not _os.path.isfile(_path):
        raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}} needs a points file, but "
                            f"'{{_path}}' was not found. Browse for a valid file in that BC's row.")
    _data = np.loadtxt(_path, delimiter=",") if _path.lower().endswith(".csv") else np.loadtxt(_path)
    if _data.ndim == 1:
        _data = _data.reshape(1, -1)
    # _n_values > 1 is PointSetBC's multi-component case (component=[i, j,
    # ...]): one extra trailing value column per listed component, in the
    # same order, matching DeepXDE's own expectation that `values` has one
    # column per entry in a list-valued `component`.
    _n_have = _data.shape[1] - _n_coord_cols_bc
    if _n_have < _n_values:
        raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}} needs {{_n_values}} value "
                            f"column(s) after its {{_n_coord_cols_bc}} coordinate column(s), but "
                            f"'{{_path}}' only has {{_n_have}}.")
    return _data[:, :_n_coord_cols_bc], _data[:, _n_coord_cols_bc:_n_coord_cols_bc + _n_values]

def _bc_parse_points(_text, _bc_label):
    # Typed list of points (Round 71): ';' between points, ',' between
    # coordinates (x[, y[, z]][, t]); in 1D steady problems ',' also
    # separates points ("0.2, 0.5, 0.8").
    _txt = (_text or "").strip()
    if _n_coord_cols_bc == 1:
        _txt = _txt.replace(",", ";")
    _rows = [[float(_v) for _v in _p.replace("(", "").replace(")", "").split(",") if _v.strip()]
             for _p in _txt.split(";") if _p.strip()]
    if not _rows or any(len(_r) != _n_coord_cols_bc for _r in _rows):
        raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}}: each typed point needs "
                           f"{{_n_coord_cols_bc}} coordinate(s); got {{_text!r}}.")
    return np.array(_rows, dtype=float)

def _bc_points(_be, _bc_label, _n_values=1, _gt=None):
    # Points and target values of a Point Set / Point Set Operator row:
    # points from a file, a typed list or generated at random (on the part
    # of the boundary picked by "Where", or in the domain); values from the
    # file's value columns or a formula in x, y, z, t ('|' between
    # components) -- Round 71.
    _mode = _be.get('points_mode', 'file') or 'file'
    _vmode = _be.get('values_mode', 'file') or 'file'
    if _mode == 'file':
        _pts, _vals = _bc_load_points_file(_be.get('points_file', ''), _bc_label, _n_values if _vmode == 'file' else 0)
        if _vmode == 'file':
            return _pts, _vals
    elif _mode == 'list':
        _pts = _bc_parse_points(_be.get('points_list', ''), _bc_label)
    else:
        _n = int(_be.get('points_count', 10) or 10)
        if _be.get('points_where', 'boundary') == 'domain':
            _pts = _gt.random_points(_n)
        else:
            _loc = _bc_loc_fn(_be.get('location', '') or '')
            _cand = _gt.random_boundary_points(max(20 * _n, 200))
            _pts = np.array([_p for _p in _cand if _loc(_p, True)][:_n])
            if len(_pts) == 0:
                raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}}: no boundary points match its Where expression.")
    _formulas = [_f for _f in (_be.get('values_formula', '') or '0').split("|")]
    if len(_formulas) < _n_values:
        raise RuntimeError(f"Boundary Conditions panel: {{_bc_label}} needs {{_n_values}} value formula(s), "
                           f"separated by '|'.")
    _vals = np.hstack([_bc_val_fn(_f)(_pts) for _f in _formulas[:_n_values]])
    return _pts.astype(float), _vals

_axis_idx_map = {{"x": 0, "y": 1, "z": 2}}

class _Interface2DGeomAdapter:
    # Interface2DBC's error() dots the network output (one value per
    # spatial dimension -- e.g. 2 for a 2D problem) against
    # geom.boundary_normal(X); GeometryXTime.boundary_normal() appends an
    # extra always-zero "time normal" column, which breaks that dot
    # product (2-vector against a 3-vector). on_boundary() is unaffected
    # (GeometryXTime already strips the time column itself), so this
    # adapter only needs to trim boundary_normal's result back down to
    # the spatial dimensions Interface2DBC actually expects.
    #
    # For a steady-state problem geomtime is just geom itself (a plain
    # Rectangle/Polygon, aliased above -- see _is_steady), which has no
    # extra time-normal column to begin with, so the trim must be skipped
    # there -- otherwise it would wrongly chop off a real spatial normal
    # component instead, silently corrupting every steady-state Interface
    # 2D BC's error term.
    def __init__(self, gt, is_steady=False):
        self._gt = gt
        self._is_steady = is_steady
    def on_boundary(self, x):
        return self._gt.on_boundary(x)
    def boundary_normal(self, x):
        _n = self._gt.boundary_normal(x)
        return _n if self._is_steady else _n[:, :-1]

if _custom_bc_entries or _bc_panel_data_present:
    for _bi, _be in enumerate(_custom_bc_entries):
        _btype = _be.get('type', 'dirichlet')
        _bcomp = int(_be.get('component', 0) or 0)
        _bloc  = _be.get('location', '')
        _bval  = _be.get('value', '0')
        _baxis = _be.get('axis', 'x')
        # Defense-in-depth clamp (same idea as the point-distribution remap
        # above): the GUI's own spinbox already restricts this to 0-1, but
        # a hand-edited or legacy saved file with a bigger value would
        # otherwise reach dde.icbc.PeriodicBC raw and hit its own
        # NotImplementedError at training time with no clear explanation.
        _bderiv = min(max(int(_be.get('deriv_order', 0) or 0), 0), 1)
        _bpts_file = _be.get('points_file', '')
        _bloc2 = _be.get('location2', '')
        _bdir  = _be.get('direction', 'normal')
        _blabel = f"BC {{_bi + 1}} ({{_btype}})"
        # PointSetBC's component as a list of ints (DeepXDE's own support,
        # PyTorch backend): "Components" overrides the single "Output #:"
        # above when non-empty, one extra points-file value column per
        # entry. PointSetOperatorBC has no `component` at all, so this is
        # only ever consulted for 'pointset' below.
        _bcomponents_txt = (_be.get('components', '') or '').strip()
        _bcomponents = [int(_c.strip()) for _c in _bcomponents_txt.split(',') if _c.strip()] if _bcomponents_txt else None
        # PointSet/PointSetOperator minibatching (DeepXDE's own batch_size/
        # shuffle kwargs) -- 0 means off, unchanged from before this existed.
        _bbatch = int(_be.get('batch_size', 0) or 0)
        _bshuffle = bool(_be.get('shuffle', True))
        _bbatch_kwargs = {{"batch_size": _bbatch, "shuffle": _bshuffle}} if _bbatch > 0 else {{}}

        if _btype == 'neumann':
            _constraints.append(dde.icbc.NeumannBC(geomtime, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp))
        elif _btype == 'robin':
            _constraints.append(dde.icbc.RobinBC(geomtime, _bc_robin_fn(_bval, _bcomp), _bc_loc_fn(_bloc), component=_bcomp))
        elif _btype == 'periodic':
            _constraints.append(dde.icbc.PeriodicBC(geomtime, _axis_idx_map.get(_baxis, 0), _bc_loc_fn(_bloc),
                                                      derivative_order=_bderiv, component=_bcomp))
        elif _btype == 'pointset':
            _pts, _pvals = _bc_points(_be, _blabel, len(_bcomponents) if _bcomponents else 1, geomtime)
            _constraints.append(dde.icbc.PointSetBC(_pts, _pvals, component=(_bcomponents if _bcomponents else _bcomp), **_bbatch_kwargs))
        elif _btype == 'pointset_operator':
            _pts, _pvals = _bc_points(_be, _blabel, 1, geomtime)
            _constraints.append(dde.icbc.PointSetOperatorBC(_pts, _pvals, _bc_operator_fn(_bval), **_bbatch_kwargs))
        elif _btype == 'operator':
            _bloc_fn = _bc_init_loc_fn(_bloc) if _be.get('apply_at', 'boundary') == 'initial' else _bc_loc_fn(_bloc)
            _constraints.append(dde.icbc.OperatorBC(geomtime, _bc_operator_fn(_bval), _bloc_fn))
        elif _btype == 'interface2d':
            _constraints.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime, _is_steady), _bc_val_fn(_bval),
                                                         _bc_loc_fn(_bloc), _bc_loc_fn(_bloc2), direction=_bdir))
        elif _btype == 'dirichlet':
            _constraints.append(dde.icbc.DirichletBC(geomtime, _bc_val_fn(_bval), _bc_loc_fn(_bloc), component=_bcomp))
        else:
            # An unrecognized "type" used to fall back to Dirichlet
            # silently -- which trains and reports normally, giving no
            # hint that the saved config's boundary condition isn't what
            # was actually requested. Failing loudly here instead, right
            # where the problem is introduced, matches how a missing
            # points file is already handled elsewhere in this function.
            raise ValueError(
                f"{{_blabel}}: unrecognized boundary condition type {{_btype!r}}. "
                "Expected one of: dirichlet, neumann, robin, periodic, pointset, "
                "pointset_operator, operator, interface2d."
            )

    # ── IC (independent of the BC panel -- same IC config as always;
    # skipped entirely for a steady-state problem, which has no initial
    # time slice to speak of -- dde.icbc.IC()'s on_initial() would have
    # nothing to match against anyway once geomtime is just a plain
    # spatial geometry, see above) ──
    # v19: the Inverse panel's separate "IC type" (expression/file)
    # selector was removed -- confusing to have two places to set the
    # IC when the Initial Condition panel above already covers both
    # expression and file-based ICs, for every problem type and
    # dimension. Every problem -- inverse included -- goes through that
    # one panel's own logic just below. _ic_xt/_ic_u above are kept only
    # so a config saved before this change (inverse_ic_type could
    # still be "File (x, t, u)" in it) still loads without error -- this
    # function just never builds a constraint from them anymore (the
    # dead "elif False: PointSetBC(_ic_xt, _ic_u, ...)" branch that used
    # to sit here, from before this v19 change, was removed this round).
    if _is_steady:
        pass
    else:
        for _oi in range({config.num_outputs}):
            _comp = _oi
            if {config.forward_ic_from_file} and _oi == 0:
                # Load IC from file -- see _load_ic_from_file() above for
                # the expected column layout per dimension.
                _ic_xyt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})
                _constraints.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
                _ic_file_disp = {repr(config.forward_ic_file)}
                print(f"IC loaded from file: {{_ic_file_disp}} — {{len(_ic_xyt)}} points")
            elif _oi < len(_ic_active_list) and _ic_active_list[_oi].strip() == "True":
                _ic_expr = _ic_expressions[_oi].strip() if _oi < len(_ic_expressions) else "np.zeros_like(x[:,0])"
                def _make_ic(expr, comp):
                    def _ic_fn(x):
                        if x.ndim == 1:
                            x = x.reshape(-1, 1)
                        return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                    return dde.icbc.IC(geomtime, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                _constraints.append(_make_ic(_ic_expr, _comp))
else:
    # Legacy per-side BCs, for configs saved before the Boundary
    # Conditions panel existed (custom_bc_json empty).
    # v19: the Inverse panel's separate "IC type" (expression/file)
    # selector was removed -- confusing to have two places to set the
    # IC when the Initial Condition panel above already covers both
    # expression and file-based ICs, for every problem type and
    # dimension. Every problem -- inverse included -- goes through that
    # one panel's own logic just below. _ic_xt/_ic_u above are kept only
    # so a config saved before this change (inverse_ic_type could
    # still be "File (x, t, u)" in it) still loads without error -- this
    # branch just never builds a constraint from them anymore (the dead
    # "if False: PointSetBC(_ic_xt, _ic_u, ...)" wrapper that used to sit
    # here, from before this v19 change, was removed this round).
    for _oi in range({config.num_outputs}):
        _comp = _oi
        _blt = _bc_left_types[_oi].strip()  if _oi < len(_bc_left_types)  else "Dirichlet"
        _brt = _bc_right_types[_oi].strip() if _oi < len(_bc_right_types) else "Dirichlet"
        _blv = float(_bc_left_values[_oi].strip())  if _oi < len(_bc_left_values)  else 0.0
        _brv = float(_bc_right_values[_oi].strip()) if _oi < len(_bc_right_values) else 0.0

        # ── BC left (x = x_min) ──────────────────────────────
        if _oi < len(_bc_left_active) and _bc_left_active[_oi].strip() == "True":
            def _bl_on_boundary(x, on_boundary):
                return on_boundary and dde.utils.isclose(x[0], {config.x_min})
            if _blt == "Dirichlet":
                def _make_dbc_l(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_dbc_l(_blv, _comp, _bl_on_boundary))
            elif _blt == "Neumann":
                def _make_nbc_l(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_nbc_l(_blv, _comp, _bl_on_boundary))
            elif _blt == "Periodic":
                _constraints.append(dde.icbc.PeriodicBC(geomtime, 0, _bl_on_boundary, derivative_order=0, component=_comp))
                if _oi < len(_bc_left_deriv_list) and _bc_left_deriv_list[_oi].strip() == "True":
                    _constraints.append(dde.icbc.PeriodicBC(geomtime, 0, _bl_on_boundary, derivative_order=1, component=_comp))

        # ── BC right (x = x_max) ─────────────────────────────
        if _oi < len(_bc_right_active) and _bc_right_active[_oi].strip() == "True":
            def _br_on_boundary(x, on_boundary):
                return on_boundary and dde.utils.isclose(x[0], {config.x_max})
            if _brt == "Dirichlet":
                def _make_dbc_r(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_dbc_r(_brv, _comp, _br_on_boundary))
            elif _brt == "Neumann":
                def _make_nbc_r(v, comp, on_bd):
                    def _val_fn(x): return np.full((len(x), 1), v)
                    return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                _constraints.append(_make_nbc_r(_brv, _comp, _br_on_boundary))
            elif _brt == "Periodic":
                pass  # Periodic BC is handled by left side only — DeepXDE enforces both ends together

        # ── BC bottom (y = y_min) — 2D only ──────────────────
        if _is_2d:
            _bbt = _bc_bottom_types[_oi].strip()  if _oi < len(_bc_bottom_types)  else "Dirichlet"
            _btt = _bc_top_types[_oi].strip()     if _oi < len(_bc_top_types)     else "Dirichlet"
            _bbv = float(_bc_bottom_values[_oi].strip()) if _oi < len(_bc_bottom_values) else 0.0
            _btv = float(_bc_top_values[_oi].strip())    if _oi < len(_bc_top_values)    else 0.0

            if _oi < len(_bc_bottom_active) and _bc_bottom_active[_oi].strip() == "True":
                def _bb_on_boundary(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[1], {config.y_min})
                if _bbt == "Dirichlet":
                    def _make_dbc_b(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_dbc_b(_bbv, _comp, _bb_on_boundary))
                elif _bbt == "Neumann":
                    def _make_nbc_b(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_nbc_b(_bbv, _comp, _bb_on_boundary))
                elif _bbt == "Periodic":
                    _constraints.append(dde.icbc.PeriodicBC(geomtime, 1, _bb_on_boundary, derivative_order=0, component=_comp))
                    if _oi < len(_bc_bottom_deriv_list) and _bc_bottom_deriv_list[_oi].strip() == "True":
                        _constraints.append(dde.icbc.PeriodicBC(geomtime, 1, _bb_on_boundary, derivative_order=1, component=_comp))

            # ── BC top (y = y_max) ────────────────────────────
            if _oi < len(_bc_top_active) and _bc_top_active[_oi].strip() == "True":
                def _bt_on_boundary(x, on_boundary):
                    return on_boundary and dde.utils.isclose(x[1], {config.y_max})
                if _btt == "Dirichlet":
                    def _make_dbc_t(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.DirichletBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_dbc_t(_btv, _comp, _bt_on_boundary))
                elif _btt == "Neumann":
                    def _make_nbc_t(v, comp, on_bd):
                        def _val_fn(x): return np.full((len(x), 1), v)
                        return dde.icbc.NeumannBC(geomtime, _val_fn, on_bd, component=comp)
                    _constraints.append(_make_nbc_t(_btv, _comp, _bt_on_boundary))
                elif _btt == "Periodic":
                    pass  # Periodic BC handled by bottom side only

        # ── IC (skipped entirely for a steady-state problem, same
        # reasoning/fix as the Boundary-Conditions-panel-active branch
        # above: a steady-state geomtime is just a plain spatial
        # geometry with no on_initial() to match against, so building a
        # dde.icbc.IC() here crashed with "'Rectangle' object has no
        # attribute 'on_initial'" the moment any output's IC was
        # active -- caught via an actual exec-level steady-state
        # training run through this exact legacy branch, which only
        # the custom_bc_json-empty path reaches (any config built by
        # the current GUI always sends a non-empty custom_bc_json,
        # even "[]", routing to the other branch instead -- so this hit
        # only an old pre-Boundary-Conditions-panel config, loaded with
        # steady_state=True and one of its legacy ic_active entries
        # still "True") ──────────────────────────────────────────
        if _is_steady:
            pass
        elif {config.forward_ic_from_file} and _oi == 0:
            # Load IC from file -- see _load_ic_from_file() above for
            # the expected column layout per dimension.
            _ic_xyt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})
            _constraints.append(dde.icbc.PointSetBC(_ic_xyt, _ic_vals, component=0))
            _ic_file_disp = {repr(config.forward_ic_file)}
            print(f"IC loaded from file: {{_ic_file_disp}} — {{len(_ic_xyt)}} points")
        elif _oi < len(_ic_active_list) and _ic_active_list[_oi].strip() == "True":
            _ic_expr = _ic_expressions[_oi].strip() if _oi < len(_ic_expressions) else "np.zeros_like(x[:,0])"
            def _make_ic(expr, comp):
                def _ic_fn(x):
                    if x.ndim == 1:
                        x = x.reshape(-1, 1)
                    return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                return dde.icbc.IC(geomtime, _ic_fn, lambda x, on_initial: on_initial, component=comp)
            _constraints.append(_make_ic(_ic_expr, _comp))

# ── Loss weights ─────────────────────────────────────────────
_n_out_w = {config.num_outputs}
_ic_a  = "{config.ic_active}".split(",")
_wm_list = [float(v) for v in "{config.loss_weights_multi}".split(",") if v.strip()] if "{config.loss_weights_multi}".strip() else []

_pde_w = []
_wi = 0
for _ei_w in range({_n_equations(config)}):  # one per equation
    _pde_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1

if _custom_bc_entries or _bc_panel_data_present:
    _bc_entry_w = []
    for _bi_w in range(len(_custom_bc_entries)):
        _bc_entry_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1

    _ic_w = []
    for _oi_w in range(_n_out_w):
        if not _is_steady and ((_oi_w == 0 and {config.forward_ic_from_file}) or (_oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True")):
            _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _ic_w.append(None); _wi += 1 if (not _is_steady and _wi < len(_wm_list)) else 0

    _multi_weights = list(_pde_w) + list(_bc_entry_w)
    for _oi_w in range(_n_out_w):
        if _ic_w[_oi_w] is not None:
            _multi_weights.append(_ic_w[_oi_w])

    print(f"  PDE weights: {{_pde_w}}")
    print(f"  BC weights ({{len(_custom_bc_entries)}} rows from the Boundary Conditions panel): {{_bc_entry_w}}")
    print(f"  IC weights: {{[w for w in _ic_w if w is not None]}}")
else:
    # Legacy per-side weights, for configs saved before the Boundary
    # Conditions panel existed (custom_bc_json empty).
    _bc_la = "{config.bc_left_active}".split(",")
    _bc_ra = "{config.bc_right_active}".split(",")
    _bc_ba = "{config.bc_bottom_active}".split(",")
    _bc_ta = "{config.bc_top_active}".split(",")

    _bcl_w = []; _bcr_w = []; _bcb_w = []; _bct_w = []; _ic_w = []
    for _oi_w in range(_n_out_w):
        if _oi_w < len(_bc_la) and _bc_la[_oi_w].strip() == "True":
            _bcl_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcl_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
        _brt_check = "{config.bc_right_types}".split(",")
        _blt_check = "{config.bc_left_types}".split(",")
        _brt_is_periodic = _oi_w < len(_brt_check) and _brt_check[_oi_w].strip() == "Periodic"
        _blt_is_periodic = _oi_w < len(_blt_check) and _blt_check[_oi_w].strip() == "Periodic"
        if _oi_w < len(_bc_ra) and _bc_ra[_oi_w].strip() == "True" and not _brt_is_periodic and not _blt_is_periodic:
            _bcr_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _bcr_w.append(None)  # do NOT advance _wi — GUI sends no value for this slot
        _btt_check = "{config.bc_top_types}".split(",")
        _bbt_check = "{config.bc_bottom_types}".split(",")
        if _is_2d:
            if _oi_w < len(_bc_ba) and _bc_ba[_oi_w].strip() == "True":
                _bcb_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
            else:
                _bcb_w.append(None); _wi += 1 if _wi < len(_wm_list) else 0
            _btt_is_periodic = _oi_w < len(_btt_check) and _btt_check[_oi_w].strip() == "Periodic"
            _bbt_is_periodic = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
            if _oi_w < len(_bc_ta) and _bc_ta[_oi_w].strip() == "True" and not _btt_is_periodic and not _bbt_is_periodic:
                _bct_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
            else:
                _bct_w.append(None)  # do NOT advance _wi — GUI sends no value for this slot
        else:
            _bcb_w.append(None); _bct_w.append(None)
        # Same _is_steady guard as the constraints block above and as the
        # Boundary-Conditions-panel-active branch's own _ic_w loop just
        # above (which already checks "not _is_steady and (...)") -- a
        # steady-state problem builds no IC constraint at all now, so a
        # weight entry for one here (this legacy branch forgot the check
        # before this fix) would desync _multi_weights from the real
        # constraints list by one slot, and shift every weight after it
        # for any later output too.
        if (not _is_steady) and ((_oi_w == 0 and {config.forward_ic_from_file}) or (_oi_w < len(_ic_a) and _ic_a[_oi_w].strip() == "True")):
            _ic_w.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        else:
            _ic_w.append(None); _wi += 1 if (not _is_steady and _wi < len(_wm_list)) else 0

    _bc_left_deriv_active  = "{config.bc_left_deriv}".split(",")
    _bc_bottom_deriv_active = "{config.bc_bottom_deriv}".split(",")
    _multi_weights = list(_pde_w)
    for _oi_w in range(_n_out_w):
        if _bcl_w[_oi_w] is not None:
            _multi_weights.append(_bcl_w[_oi_w])
            # Add extra weight for derivative periodic BC if enabled -- this
            # must check the LEFT edge's own periodic/derivative flags, not
            # the bottom edge's (a copy-paste bug: this block was gated on
            # _bbt_check/_bc_bottom_deriv_active, the exact same variables
            # the bottom block below correctly uses for itself, so a
            # left-side periodic+derivative BC never got its extra weight
            # slot at all while a bottom-side one could spuriously trigger
            # this block instead, desyncing every loss weight after it).
            _blt_is_per = _oi_w < len(_blt_check) and _blt_check[_oi_w].strip() == "Periodic"
            _bld_is_active = _oi_w < len(_bc_left_deriv_active) and _bc_left_deriv_active[_oi_w].strip() == "True"
            if _blt_is_per and _bld_is_active:
                _multi_weights.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        if _bcr_w[_oi_w] is not None: _multi_weights.append(_bcr_w[_oi_w])
        if _bcb_w[_oi_w] is not None:
            _multi_weights.append(_bcb_w[_oi_w])
            # Add extra weight for derivative periodic BC if enabled -- same
            # fix as the left-edge block above: must actually consume the
            # next GUI-provided weight from _wm_list and advance the shared
            # _wi cursor, not hardcode 1.0. A hardcoded weight here silently
            # discards whatever weight the user set for this BC, AND (since
            # _wi was never advanced) desyncs every weight slot still to be
            # read after it for this or a later output.
            _bbt_is_per = _oi_w < len(_bbt_check) and _bbt_check[_oi_w].strip() == "Periodic"
            _bbd_is_active = _oi_w < len(_bc_bottom_deriv_active) and _bc_bottom_deriv_active[_oi_w].strip() == "True"
            if _bbt_is_per and _bbd_is_active:
                _multi_weights.append(_wm_list[_wi] if _wi < len(_wm_list) else 1.0); _wi += 1
        if _bct_w[_oi_w] is not None: _multi_weights.append(_bct_w[_oi_w])
        if _ic_w[_oi_w]  is not None: _multi_weights.append(_ic_w[_oi_w])

    print(f"  PDE weights: {{_pde_w}}")
    print(f"  BC left: {{[w for w in _bcl_w if w is not None]}}")
    print(f"  BC right: {{[w for w in _bcr_w if w is not None]}}")
    if _is_2d:
        print(f"  BC bottom: {{[w for w in _bcb_w if w is not None]}}")
        print(f"  BC top: {{[w for w in _bct_w if w is not None]}}")
    print(f"  IC weights: {{[w for w in _ic_w if w is not None]}}")

if _problem_type == "Inverse":
    # One observation-loss weight per measured-data file, same order as
    # the PointSetBC constraints appended just below.
    _multi_weights = _multi_weights + [_e[3] for _e in _obs_entries]

print(f"Loss weights: {{_multi_weights}} ({{len(_multi_weights)}} terms for {{len(_constraints)}} constraints)")
{"print('Note: num_test disabled (left at None) because an Operator or Point Set Operator BC Value expression references X -- DeepXDE requires this whenever a boundary condition function uses the raw input points.')" if _bc_op_uses_X else ""}
{"print('Note: Point Distribution was switched to uniform automatically because an Interface 2D boundary condition is active (DeepXDE requires evenly-spaced boundary points so Interface 2D BC can pair corresponding points correctly across its two edges).')" if _bc_forced_uniform else ""}

{_anchor_points_code(config)}
# ── Data ─────────────────────────────────────────────────────
# Steady-state problems have no time axis/Initial Condition, so they use a
# plain dde.data.PDE (spatial geometry only, no num_initial=) instead of
# dde.data.TimePDE -- geomtime is already just an alias for geom in that
# case (see the geometry-construction section above).
if _problem_type == "Inverse":
    _obs_bcs = [
        dde.icbc.PointSetOperatorBC(_e[0], _e[1], _make_obs_func(_e[4], _obs_output_names_list))
        if _e[4] else dde.icbc.PointSetBC(_e[0], _e[1], component=_e[2])
        for _e in _obs_entries
    ]
    _constraints.extend(_obs_bcs)
    _obs_anchors = {"np.vstack([_e[0] for _e in _obs_entries]) if _obs_entries else None" if getattr(config, "inverse_obs_anchors", True) else "None  # measured points only in the data loss, not anchors"}
    if _is_steady:
        data = dde.data.PDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_test={_safe_num_test!r},
            train_distribution="{_safe_point_distribution}",
            anchors={_live_inv_anchors}
        )
    else:
        data = dde.data.TimePDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={_safe_num_test!r},
            train_distribution="{_safe_point_distribution}",
            anchors={_live_inv_anchors}
        )
else:
    if _is_steady:
        data = dde.data.PDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_test={_safe_num_test!r},
            train_distribution="{_safe_point_distribution}",
            anchors={_live_fwd_anchors}
        )
    else:
        data = dde.data.TimePDE(
            geomtime, pde, _constraints,
            num_domain={config.num_domain}, num_boundary={config.num_boundary},
            num_initial={config.num_initial}, num_test={_safe_num_test!r},
            train_distribution="{_safe_point_distribution}",
            anchors={_live_fwd_anchors}
        )

# ── Parametric loop ──────────────────────────────────────────
_summary = []

for _pval in _param_values:
    if _parametric:
        print(f"\\n===== Running: {{_param_name}} = {{_pval}} =====")

    _lr           = {config.learning_rate}
    # NOTE: _loss_weights (and the four branches below that mutate it --
    # ic_weight/pde_weight/bc_left_weight/bc_right_weight) predate this
    # codebase's multi-output loss-weights system and are currently dead:
    # model.compile() below passes loss_weights=_multi_weights (the real,
    # general per-PDE/per-BC/per-IC weights list, built earlier in this
    # function), never _loss_weights -- so sweeping any of these four
    # parametric options would silently do nothing. Harmless today since
    # Parametric Study has no GUI controls at all yet (nothing can reach
    # this code path), but worth fixing for real -- route the swept value
    # into the matching slot of _multi_weights instead -- before ever
    # wiring up a GUI for it, rather than leaving a "parametric sweep"
    # feature that quietly does nothing for 4 of its 6 options.
    _loss_weights = list({config.loss_weights})
    _layers       = list({config.layers})

    if _parametric and _pval is not None:
        if _param_name == "learning_rate":
            _lr = float(_pval)
        elif _param_name == "ic_weight":
            _loss_weights[3] = float(_pval)
        elif _param_name == "pde_weight":
            _loss_weights[0] = float(_pval)
        elif _param_name == "bc_left_weight":
            _loss_weights[1] = float(_pval)
        elif _param_name == "bc_right_weight":
            _loss_weights[2] = float(_pval)
        elif _param_name == "hidden_layers":
            n = int(_pval)
            _layers = [{config.layers[0]}] + [{config.layers[1]}] * n + [{config.layers[-1]}]
        elif _param_name == "neurons_per_layer":
            _layers = [{config.layers[0]}] + [int(_pval)] * {len(config.layers) - 2} + [{config.layers[-1]}]

    net = _apply_net_transforms(_make_net(_layers, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay}))
    model = dde.Model(data, net)
{_init_loss_weights_code(config, "    ", "model", "_multi_weights", "_lr", "_inv_vars if _problem_type == 'Inverse' else None")}

    model.compile(
        "{config.optimizer}", lr=_lr, loss="{config.loss_type}",
        loss_weights=_multi_weights,
        external_trainable_variables=_inv_vars if _problem_type == "Inverse" else None
    )

    _iters = int(_pval) if (_parametric and _param_name == "phase1_iterations" and _pval is not None) else {config.iterations}

    # ── Training callbacks (opt-in: EarlyStopping/PDEPointResampler/
    # ModelCheckpoint/Timer) — see _build_train_cbs_code() in codegen.py
    # for exactly what's included and why; empty list when none are
    # enabled, which every mainline .train() call below appends as a no-op.
{_train_cbs_code}

    # ── IC Pre-Training ───────────────────────────────────────
    if {config.ic_pretrain} and not {config.time_adaptive}:
        print(f"\\n=== IC Pre-Training: {config.ic_pretrain_iterations} iterations (IC loss only) ===")
        # Build IC-only geometry and data — no domain/BC points, only IC
        _ic_geom_pre = _build_geom()
        _ic_td_pre  = dde.geometry.TimeDomain({config.t_min}, {config.t_max})
        _ic_gt_pre  = dde.geometry.GeometryXTime(_ic_geom_pre, _ic_td_pre)
        _ic_ics_pre = []
        if {config.forward_ic_from_file}:
            # Load IC from file for pre-training -- see _load_ic_from_file()
            # near the top of this script for the expected column layout.
            _ic_pre_xyt, _ic_pre_vals = _load_ic_from_file({repr(config.forward_ic_file)})
            _ic_ics_pre.append(dde.icbc.PointSetBC(_ic_pre_xyt, _ic_pre_vals, component=0))
        else:
            _ic_exprs_pre = "{config_ic_expressions}".split("|")
            # only outputs whose initial condition is on (as the exported script)
            for _oi_pre in {_clean_active_ic_outputs(config, config.num_outputs, (config.ic_active or "True").split(","))}:
                _ic_expr_pre = _ic_exprs_pre[_oi_pre].strip() if _oi_pre < len(_ic_exprs_pre) else "np.zeros_like(x[:,0])"
                def _mk_ic_pre(expr, comp):
                    def _ic_fn(x):
                        return np.reshape(eval(expr, {{"np": np, "x": x, "__builtins__": __builtins__}}), (-1, 1))
                    return dde.icbc.IC(_ic_gt_pre, _ic_fn, lambda x, on_initial: on_initial, component=comp)
                _ic_ics_pre.append(_mk_ic_pre(_ic_expr_pre, _oi_pre))
        
        # IC-only loss weights: PDE terms 0, every IC term 1000.
        _ic_only_weights = [0.0] * {config.num_outputs} + [1000.0] * len(_ic_ics_pre)
        print(f"  IC-only weights: {{_ic_only_weights}} -- dummy PDE, IC points only")
{_ic_pretrain_block_std}
        print("=== Starting Main Training ===\\n")

    if not {config.time_adaptive}:
        _sched_active = {config.optimizer_scheduler} and _os.path.exists('/tmp') and {len(config.scheduler_phases) > 0}
        if _problem_type == "Inverse":
            with open("/tmp/param_history.txt", "w") as _f:
                pass
            if not _sched_active:
                _var_cb = dde.callbacks.VariableValue(
                    _inv_vars, period={inv_param_period(config) or 1000}, filename="/tmp/param_history.txt",
                    precision=6
                )
                loss_history, train_state = model.train(iterations=_iters, display_every={config.loss_display_every}, callbacks=[_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                # Round 31: standardized on "save one checkpoint per
                # optimizer phase" everywhere (matching Time-Adaptive's own
                # established per-step convention, and the user's explicit
                # choice when this was flagged as an inconsistency) --
                # previously this Adam-phase checkpoint was only saved when
                # Adam was the run's SOLE/terminal phase (optimizer2
                # "none"), silently skipped whenever a Phase 2 was also
                # configured, since only Phase 2's own save used to run in
                # that case. Now always saved whenever the Adam phase
                # itself actually ran.
                if _use_save:
                    _adam_model_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}")
                    model.save(_adam_model_path)
                    _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                    with open(_adam_cfg_path, "w") as _acf:
                        _json.dump(_model_config, _acf, indent=2)
                    print(f"Adam model saved: {{_adam_model_path}}.pt")
                    print(f"Adam config saved to: {{_adam_cfg_path}}")
            else:
                # Scheduler phases below define all training — skip this
                # standalone _iters-iteration pass so training only runs
                # for the iteration counts configured in the scheduler.
                print(f"  (Scheduler enabled — skipping standalone {{_iters}}-iteration pass; "
                      f"training runs only for the phases below)")
        else:
            if not _sched_active:
                # Mirrors the Inverse branch above (which already trains its
                # Phase 1 here when the Training Phases scheduler is off) --
                # previously this Forward branch never called model.train()
                # under any condition, so a Forward problem with the
                # scheduler off either silently skipped Phase 1 entirely
                # (going straight into Phase 2 from a random, untrained
                # network, when optimizer2 was set) or left
                # loss_history/train_state unbound and crashed with a
                # confusing NameError at dde.saveplot() (when optimizer2
                # was also "none").
                loss_history, train_state = model.train(iterations=_iters, display_every={config.loss_display_every}, callbacks=_train_cbs)
            if _use_save and not _sched_active:
                # Round 31: standardized on "save one checkpoint per
                # optimizer phase" everywhere (matching Time-Adaptive's own
                # established per-step convention) -- previously this
                # Adam-phase checkpoint was only saved when Adam was the
                # run's SOLE/terminal phase (optimizer2 "none"), silently
                # skipped whenever a Phase 2 was also configured, since
                # only the scheduler-phase loop / Phase 2 branch below used
                # to save a checkpoint in that case. Now always saved
                # whenever the Adam phase itself actually ran (a plain
                # Adam-only run, with no Phase 2, is still covered -- this
                # is the only save it would otherwise ever get).
                _adam_model_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}")
                model.save(_adam_model_path)
                print(f"Adam model saved: {{_adam_model_path}}.pt")
                _adam_cfg_path = _os.path.join(_sol_dir, f"model_adam-{{_iters}}.json")
                with open(_adam_cfg_path, "w") as _acf:
                    _json.dump(_model_config, _acf, indent=2)
                print(f"Adam config saved to: {{_adam_cfg_path}}")

        # Optimizer Scheduler or Phase 2
        if _sched_active:
            import json as _json_sched
            _sched_phases = _json_sched.loads({repr(config.scheduler_phases)})
            _sched_cum_iters = 0
            for _sp_i, _sp in enumerate(_sched_phases):
                print(f"  === Scheduler Phase {{_sp_i+1}}: {{_sp['optimizer']}} {{_sp['iterations']}} iters ===")
                # A phase without its own weights (e.g. written by hand) uses
                # the Loss Weights panel's.
                _sp_weights = [float(w) for w in str(_sp.get('weights') or '').split(',') if w.strip()] or list(_multi_weights)
                # The GUI computes each phase's weight string from the Boundary
                # Conditions panel's row count at the time the phase was set up;
                # if the panel (or geometry) changed afterwards without the
                # scheduler panel being rebuilt, this can end up a different
                # length than the constraints actually being trained on, which
                # DeepXDE has no graceful way to handle (a hard crash multiplying
                # mismatched-length tensors). Fall back to the freshly computed,
                # always-correct-length weights rather than crash mid-training.
                _sp_expected_len = {_n_equations(config)} + len(_constraints)
                if len(_sp_weights) != _sp_expected_len:
                    print(f"  ⚠️ Scheduler Phase {{_sp_i+1}}: configured weights have {{len(_sp_weights)}} "
                          f"entries but {{_sp_expected_len}} loss terms are active "
                          f"({{len(_constraints)}} constraints + {_n_equations(config)} PDE) — the Boundary "
                          f"Conditions panel or geometry likely changed after this phase's weights were set. "
                          f"Using the freshly computed weights for this phase instead.")
                    _sp_weights = list(_multi_weights)
                _sp_ext_vars = _inv_vars if _problem_type == "Inverse" else None
                if _sp['optimizer'] == 'lbfgs':
                    dde.optimizers.set_LBFGS_options(
                        maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                        gtol={config.lbfgs_gtol}, maxiter=_sp['iterations'],
                        maxfun=(int(_sp['iterations']*1.25) if {bool(getattr(config, 'lbfgs_use_default', True))} else {config.lbfgs_maxfun}), maxls={config.lbfgs_maxls})
                    model.compile("L-BFGS", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period={min(inv_param_period(config) or 200, 200)},
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            display_every={config.loss_display_every}, callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_lbfgs-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} L-BFGS model saved: {{_nta_save_path}}.pt")
                elif _sp['optimizer'] == 'nncg':
                    # NNCG (deepxde>=1.13.0, version-guarded above): exact-case
                    # "NNCG" name required, no lr= (ignored with a warning --
                    # NysNewtonCG's own step size is configured via
                    # dde.optimizers.set_NNCG_options(), left at its defaults
                    # here since this app doesn't expose those knobs), and no
                    # LR decay (decay is meaningless for a Newton-CG method).
                    model.compile("NNCG", loss=_sp.get('loss', '{config.loss_type}'),
                                  loss_weights=_sp_weights, external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period={inv_param_period(config) or 1000},
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every={config.loss_display_every},
                            callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_nncg-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} NNCG model saved: {{_nta_save_path}}.pt")
                else:
                    _sp_decay = None
                    _sp_decay_type = _sp.get('decay_type', 'none')
                    if _sp_decay_type and _sp_decay_type != 'none':
                        _sp_p1 = _sp.get('decay_p1', 0) or 0
                        _sp_p2 = _sp.get('decay_p2', 0) or 0
                        if _sp_decay_type == 'step':
                            _sp_decay = ('step', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'cosine':
                            _sp_decay = ('cosine', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'inverse time':
                            _sp_decay = ('inverse time', int(_sp_p1), float(_sp_p2))
                        elif _sp_decay_type == 'exponential':
                            _sp_decay = ('exponential', float(_sp_p1))
                    model.compile(_sp['optimizer'], lr=_sp['lr'], decay=_sp_decay,
                                  loss=_sp.get('loss', '{config.loss_type}'), loss_weights=_sp_weights,
                                  external_trainable_variables=_sp_ext_vars)
                    if _problem_type == "Inverse":
                        for _icb in _print_cbs: _icb.set_offset(_sched_cum_iters)
                        for _icb in _save_cbs: _icb.set_offset(_sched_cum_iters)
                        _sp_var_cb = dde.callbacks.VariableValue(
                            _inv_vars, period={inv_param_period(config) or 1000},
                            filename=f"/tmp/param_history_sp{{_sp_i}}.txt", precision=6)
                        loss_history, train_state = model.train(
                            iterations=_sp['iterations'], display_every={config.loss_display_every},
                            callbacks=[_sp_var_cb] + _print_cbs + _save_cbs + _train_cbs)
                        try:
                            with open(f"/tmp/param_history_sp{{_sp_i}}.txt", "r") as _spf:
                                _sp_lines = [l.strip() for l in _spf if l.strip()]
                            with open("/tmp/param_history.txt", "a") as _fa:
                                for _spl in _sp_lines:
                                    _sp_parsed = _split_param_history_line(_spl)
                                    if _sp_parsed is not None:
                                        _sp_it, _sp_raw = _sp_parsed
                                        _fa.write(f"{{_sp_it}} [{{_sp_raw}}]\\n")
                        except Exception as _spe:
                            print(f"Could not merge phase {{_sp_i+1}} parameter history: {{_spe}}")
                        _sched_cum_iters = loss_history.steps[-1] if loss_history.steps else (_sched_cum_iters + _sp['iterations'])
                    else:
                        loss_history, train_state = model.train(iterations=_sp['iterations'], display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _nta_save_path = _os.path.join(_sol_dir, f"model_adam-phase{{_sp_i+1}}")
                        model.save(_nta_save_path)
                        print(f"  Phase {{_sp_i+1}} Adam model saved: {{_nta_save_path}}.pt")
        elif "{config.optimizer2}" != "none":
            _phase2_weights = _multi_weights
            if "{config.optimizer2}" == "lbfgs":
                dde.optimizers.set_LBFGS_options(
                    maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                    gtol={config.lbfgs_gtol}, maxiter={config.iterations2},
                    maxfun={_lbfgs_maxfun(config, config.iterations2)}, maxls={config.lbfgs_maxls}
                )
                if _problem_type == "Inverse":
                    _var_cb2 = dde.callbacks.VariableValue(
                        _inv_vars, period={min(inv_param_period(config) or 200, 200)}, filename="/tmp/param_history_phase2.txt",
                        precision=6
                    )
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights,
                                  external_trainable_variables=_inv_vars)
                    for _icb in _print_cbs: _icb.set_offset(_iters)
                    loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=[_var_cb2] + _print_cbs + _train_cbs)
                    # Append L-BFGS history to each variable's own convergence file
                    if _param_save_period > 0:
                        try:
                            with open("/tmp/param_history_phase2.txt", "r") as _lf:
                                for _ll in _lf:
                                    _l_parsed = _split_param_history_line(_ll)
                                    if _l_parsed is None:
                                        continue
                                    _lbfgs_iter, _lbfgs_raw = _l_parsed
                                    try:
                                        _lbfgs_vals = [float(v) for v in _lbfgs_raw.split(",") if v.strip()]
                                    except ValueError:
                                        continue
                                    for _lv_path, _lv_val in zip(_param_save_paths, _lbfgs_vals):
                                        if not _lv_path:
                                            continue
                                        with open(_lv_path, "a") as _pf:
                                            _pf.write(f"{{_lbfgs_iter}},{{_lv_val:.10g}}\\n")
                        except Exception as _le:
                            print(f"Could not save L-BFGS history: {{_le}}")
                    try:
                        with open("/tmp/param_history_phase2.txt", "r") as _f2:
                            _p2_lines = [l.strip() for l in _f2 if l.strip()]
                        with open("/tmp/param_history.txt", "a") as _fa:
                            for _p2l in _p2_lines:
                                _p2_parsed = _split_param_history_line(_p2l)
                                if _p2_parsed is not None:
                                    _new_iter, _p2_raw = _p2_parsed
                                    _fa.write(f"{{_new_iter}} [{{_p2_raw}}]\\n")
                    except Exception as _ae:
                        print(f"Could not append phase 2 history: {{_ae}}")
                else:
                    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights=_phase2_weights)
                    loss_history, train_state = model.train(display_every={config.loss_display_every}, callbacks=_train_cbs)
                    if _use_save:
                        _lbfgs_model_path = _os.path.join(_sol_dir, "model_lbfgs")
                        model.save(_lbfgs_model_path)
                        _lbfgs_iter = loss_history.steps[-1] if loss_history.steps else _iters
                        _lbfgs_cfg_path = _os.path.join(_sol_dir, f"model_lbfgs-{{_lbfgs_iter}}.json")
                        with open(_lbfgs_cfg_path, "w") as _lcf:
                            _json.dump(_model_config, _lcf, indent=2)
                        print(f"L-BFGS config saved to: {{_lbfgs_cfg_path}}")
            else:
                if _problem_type == "Inverse":
                    model.compile("{config.optimizer2}", lr=_lr, loss="{config.loss_type}",
                                  loss_weights=_phase2_weights,
                                  external_trainable_variables=_inv_vars)
                else:
                    model.compile("{config.optimizer2}", lr=_lr, loss="{config.loss_type}",
                                  loss_weights=_phase2_weights)
                loss_history, train_state = model.train(iterations={config.iterations2}, display_every={config.loss_display_every}, callbacks=_train_cbs)

    # ── Results-field extractor (_extract_plot_field) ──────────
    # Moved up from the "Plot solution" section below (same "not
    # time_adaptive" condition -- defined here, unconditionally on RAR,
    # instead of duplicated in both places) so each RAR round's own
    # solution snapshot can use it too; the final plot-type dispatch
    # down there still only runs once, at the very end, and now just
    # reuses this same definition instead of redefining it.
    if not {config.time_adaptive}:
        _plot_idx = {config.plot_output_idx}
        _plot_type = "{config.plot_type}"
        _plot_custom_expr = "{plot_custom_expr_converted}"
        _plot_custom_label = "{plot_custom_label_resolved}"
        _plot_title_override = {plot_title_override!r}
        _plot_xlabel_override = {plot_xlabel_override!r}
        _plot_ylabel_override = {plot_ylabel_override!r}
        _plot_output_names_list = {repr(config.output_names)}.split(",")
        _plot_n_out = len(_plot_output_names_list)
        _plot_dim = "3D" if _is_3d else ("2D" if _is_2d else "1D")
{_math_namespace_code('_PLOT_TORCH_MATH_NS', 'torch', '        ')}

        def _plot_custom_op(_pf_inputs, _pf_outputs):
            _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                         _plot_output_names_list, _is_steady, _plot_dim,
                                         _plot_custom_expr)
            _pf_ns = dict(_pf_dvars)
            _pf_ns.update(_PLOT_TORCH_MATH_NS)
            _pf_ns["torch"] = torch
            return eval(_plot_custom_expr, _pf_ns)

        def _extract_plot_field(_x_grid, _pf_model=None):
            # Pick the field to plot from a model.predict() call: either one
            # raw output column (_plot_idx, the pre-existing default), or --
            # when a custom expression is configured -- a derived field
            # built from ALL of this problem's outputs AND their
            # derivatives (du_x, du_xx, du_xy, ...), evaluated through
            # DeepXDE's own dde.Model.predict(x, operator=...) built-in.
            # _pf_model defaults to the module-level `model` but can be
            # overridden. Used here for each RAR round's own snapshot,
            # and below for the main Results panel plot and Inline Error
            # Analysis -- so every one of them shows/compares the exact
            # same field.
            _pf_m = _pf_model if _pf_model is not None else model
            if _plot_custom_expr.strip():
                return _pf_m.predict(_x_grid, operator=_plot_custom_op)[:, 0]
            return _pf_m.predict(_x_grid)[:, _plot_idx]

{_gs_rar_loop_code(config)}

    # ── Save paths ────────────────────────────────────────────
    if _parametric and _pval is not None:
        _run_dir = _os.path.join(_save_dir if _use_save else "/tmp", f"{{_param_name}}_{{_pval}}")
        _os.makedirs(_run_dir, exist_ok=True)
        _run_loss_path     = _os.path.join(_run_dir, "loss_plot.png")
        _run_solution_path = _os.path.join(_run_dir, "solution_plot.{sol_ext}")
        _run_log_path      = _os.path.join(_run_dir, "training_log.txt")
    else:
        _run_loss_path     = _loss_path
        _run_solution_path = _solution_path
        _run_log_path      = _log_path

    # ── Plot loss (Loss Plot Settings -- see _loss_plot_runtime_code) ────
    if not {config.time_adaptive}:
        dde.saveplot(loss_history, train_state, issave=False, isplot=False)
        train_loss = loss_history.loss_train
        test_loss  = loss_history.loss_test
        steps      = loss_history.steps

        title_str = f"Loss — {{_param_name}}={{_pval}}" if _parametric else "Training & Test Loss"
        _make_loss_plot(train_loss, test_loss, steps, _run_loss_path, title_str)

        if _run_log_path:
            with open(_run_log_path, "w") as f:
                for i, l in zip(steps, train_loss):
                    f.write(f"Step {{i}}: {{list(l)}}\\n")

        _final_loss, _final_loss_unw = _final_losses(model, loss_history)

        # Parameter history (inverse only) -- one subplot per trainable
        # variable, all sharing the single param_plot.png output path
        # (same filename as the single-variable version, for anything
        # downstream that expects it there).
        if _problem_type == "Inverse":
            try:
                _ph_iters = []
                _ph_vals_by_var = [[] for _ in _inv_var_names]
                with open("/tmp/param_history.txt", "r") as _phf:
                    for _line in _phf:
                        _parsed_line = _split_param_history_line(_line)
                        if _parsed_line is None:
                            continue
                        _it, _raw = _parsed_line
                        try:
                            _vals = [float(v) for v in _raw.split(",") if v.strip()]
                        except ValueError:
                            continue
                        if len(_vals) != len(_inv_var_names):
                            continue
                        _ph_iters.append(_it)
                        for _vi in range(len(_inv_var_names)):
                            _ph_vals_by_var[_vi].append(_vals[_vi])
                _ph_iters_arr = np.array(_ph_iters)
                _n_ivars = len(_inv_var_names)
                _fig, _iv_axes = plt.subplots(_n_ivars, 1, figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * _n_ivars), squeeze=False)
                for _vi, _vname in enumerate(_inv_var_names):
                    _true_val = _inv_var_trues[_vi] if _vi < len(_inv_var_trues) else None
                    _final_val = _draw_param_ax(
                        _iv_axes[_vi][0], _vname, _ph_iters_arr, _ph_vals_by_var[_vi], _true_val,
                        {config.inv_param_log_scale}, _plot_title_override,
                        _plot_xlabel_override, _plot_ylabel_override)
                    print(f"\\n=== {{_vname}} final value: {{_final_val:.6g}} ===")
                plt.tight_layout()
                plt.savefig("/tmp/param_plot.png", dpi={int(config.plot_dpi)})
                if _use_save:
                    import shutil as _sp
                    _sp.copy("/tmp/param_plot.png", _os.path.join(
                        _run_dir if (_parametric and _pval is not None) else _save_dir, "param_plot.png"))
                plt.close(_fig)
                for _vi, _vname in enumerate(_inv_var_names):
                    print(f"Final inferred {{_vname}}: {{_ph_vals_by_var[_vi][-1]:.6g}}")
                if "{config.plot_type}" == "Parameter Convergence (GIF)":
                    _param_animation(
                        [(_vn, _ph_iters_arr, _ph_vals_by_var[_vi],
                          _inv_var_trues[_vi] if _vi < len(_inv_var_trues) else None)
                         for _vi, _vn in enumerate(_inv_var_names)],
                        "/tmp/param_anim.gif", {bool(config.inv_param_log_scale)}, {int(config.plot_fps)},
                        {int(config.plot_dpi)}, _plot_title_override, _plot_xlabel_override, _plot_ylabel_override,
                        _plot_figsize(6, 3.2))
                # Plot settings -> Figures: one per parameter, as well
                _pf_dir = _run_dir if (_parametric and _pval is not None) else _save_dir
                if _use_save and (_PARAM_STYLE["plot_separate"] or _PARAM_STYLE["anim_separate"]):
                    for _vi, _vn in enumerate(_inv_var_names):
                        _tv = _inv_var_trues[_vi] if _vi < len(_inv_var_trues) else None
                        if _PARAM_STYLE["plot_separate"]:
                            _fig1, _ax1 = plt.subplots(figsize=_plot_figsize(6, 3.2))
                            _draw_param_ax(_ax1, _vn, _ph_iters_arr, _ph_vals_by_var[_vi], _tv,
                                           {bool(config.inv_param_log_scale)}, _plot_title_override,
                                           _plot_xlabel_override, _plot_ylabel_override)
                            plt.tight_layout()
                            plt.savefig(_os.path.join(_pf_dir, f"{{_vn}}_convergence_plot.png"), dpi={int(config.plot_dpi)})
                            plt.close(_fig1)
                        if _PARAM_STYLE["anim_separate"] and "{config.plot_type}" == "Parameter Convergence (GIF)":
                            _param_animation([(_vn, _ph_iters_arr, _ph_vals_by_var[_vi], _tv)],
                                             _os.path.join(_pf_dir, f"{{_vn}}_convergence_animation.gif"),
                                             {bool(config.inv_param_log_scale)}, {int(config.plot_fps)},
                                             {int(config.plot_dpi)}, _plot_title_override, _plot_xlabel_override,
                                             _plot_ylabel_override, _plot_figsize(6, 3.2))
            except Exception as _e:
                print(f"Could not plot parameter history: {{_e}}")

        _summary.append((_pval, _final_loss))
        print(f"Final loss: {{_final_loss:.4e}}")
        print(f"Final loss (unweighted): {{_final_loss_unw:.4e}}")

    # ── Plot solution ─────────────────────────────────────────
    # _plot_idx/_plot_type/_plot_custom_expr/_extract_plot_field etc. are
    # already defined above (see "Results-field extractor"), unconditionally
    # on "not time_adaptive" -- which this whole section already requires
    # too, so they're guaranteed to exist here without redefining them.
    if not {config.time_adaptive}:
        # Every output gets its own plot, named after it (Round 69): the
        # selected "Plot output" is still solution_plot.<ext>, and with more
        # than one output each one is also saved as solution_plot_<name>.<ext>.
        _plot_main_job = (_plot_idx, _run_solution_path)
        _plot_jobs = [_plot_main_job]
        if ({config.plot_all_outputs} and _plot_n_out > 1 and not _plot_custom_expr.strip()
                and not (_problem_type == "Inverse" and _plot_type.startswith("Parameter Convergence"))):
            _sol_root, _sol_ext = _os.path.splitext(_run_solution_path)
            for _pj, _pj_name in enumerate(_plot_output_names_list):
                _pj_safe = "".join(_c if (_c.isalnum() or _c in "-_") else "_" for _c in _pj_name.strip()) or f"out{{_pj}}"
                _plot_jobs.append((_pj, f"{{_sol_root}}_{{_pj_safe}}{{_sol_ext}}"))
        for _plot_idx, _run_solution_path in _plot_jobs:
            if (_plot_idx, _run_solution_path) != _plot_main_job and _plot_idx == _plot_main_job[0]:
                import shutil as _shutil_pj
                if _os.path.exists(_plot_main_job[1]):
                    _shutil_pj.copy(_plot_main_job[1], _run_solution_path)
                continue
            if _problem_type == "Inverse" and _plot_type == "Parameter Convergence":
                # Parameter Convergence isn't a spatial plot at all -- it's the
                # iteration-vs-inferred-value chart already built just above
                # (right after training) from this run's own parameter
                # history. Use that chart as the "solution" plot instead of
                # generating a spatial one, since that's what was asked to be
                # shown by default for an Inverse problem.
                import shutil as _shutil_pc
                if _os.path.exists("/tmp/param_plot.png"):
                    _shutil_pc.copy("/tmp/param_plot.png", _run_solution_path)
            elif _problem_type == "Inverse" and _plot_type == "Parameter Convergence (GIF)":
                import shutil as _shutil_pa
                if _os.path.exists("/tmp/param_anim.gif"):
                    _shutil_pa.copy("/tmp/param_anim.gif", _run_solution_path)

            elif (not _is_steady) and _plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"):
                import matplotlib.animation as _anim
                _n_frames = max(2, {config.num_timesteps_anim})
                _fps = {config.plot_fps}
                _t_frames = np.linspace({config.t_min}, {config.t_max}, _n_frames)
                _y_mid_gif = {line_slice_y} if (_is_2d or _is_3d) else 0.0
                _z_mid_gif = {line_slice_z} if _is_3d else 0.0

                if _plot_type == "Line Animation (GIF)":
                    _x_gif = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                    _all_u_gif = []
                    for _tv in _t_frames:
                        if _is_3d:
                            _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _y_mid_gif), np.full_like(_x_gif, _z_mid_gif), np.full_like(_x_gif, _tv)])
                        elif _is_2d:
                            _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _y_mid_gif), np.full_like(_x_gif, _tv)])
                        else:
                            _xt_gif = np.column_stack([_x_gif, np.full_like(_x_gif, _tv)])
                        _all_u_gif.append(_extract_plot_field(_xt_gif).flatten())
                    if _is_2d or _is_3d:
                        # only inside the domain (a hole, the L-shape's corner)
                        _in_gif = np.asarray(geom.inside(_xt_gif[:, :(3 if _is_3d else 2)]), dtype=bool)
                        _all_u_gif = [np.where(_in_gif, _u, np.nan) for _u in _all_u_gif]
                    _u_min_gif = min(np.nanmin(_u) for _u in _all_u_gif)
                    _u_max_gif = max(np.nanmax(_u) for _u in _all_u_gif)
                    _fig_gif, _ax_gif = plt.subplots(figsize=_plot_figsize(7, 5))
                    _ax_gif.set_xlim(_plot_x_min, _plot_x_max)
                    _ax_gif.set_ylim(_u_min_gif - 0.05*abs(_u_min_gif) - 1e-9, _u_max_gif + 0.05*abs(_u_max_gif) + 1e-9)
                    _gif_name = _plot_custom_label if _plot_custom_expr.strip() else _plot_output_names_list[_plot_idx].strip()
                    _ylabel_gif = (f"{{_gif_name}}(x,y={{_fmt_num(_y_mid_gif)}},z={{_fmt_num(_z_mid_gif)}},t)" if _is_3d else f"{{_gif_name}}(x,y={{_fmt_num(_y_mid_gif)}},t)") if (_is_2d or _is_3d) else f"{{_gif_name}}(x,t)"
                    _ax_gif.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); _ax_gif.set_ylabel(_plot_ylabel_override or _ylabel_gif, fontsize={config.plot_axis_label_fontsize})
                    if _plot_title_override:
                        _ax_gif.set_title(_plot_title_override)
                    _line_gif, = _ax_gif.plot([], [], color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth_anim})
                    # Positioned just above the axes (not set_title()) and
                    # styled to match it (black, centered, top) -- same visual
                    # place the 2D/3D Surface Animation branches below put their
                    # own per-frame "t = ..." via ax.set_title(). A real
                    # set_title() isn't used here because this animation runs
                    # with blit=True (see FuncAnimation below): blitting only
                    # redraws the specific artists returned by _update_gif, and
                    # a title isn't redrawn under blit, so it would never
                    # update frame to frame. A Text artist positioned in axes
                    # coordinates just above y=1.0 renders in the same place a
                    # title would, while still being blit-compatible. When a
                    # title override is also set (just above, a real ax.set_title()
                    # call that blit never touches again, so it's safe to combine
                    # with this blitted artist), that same "just above the axes"
                    # spot is exactly where the override title renders too -- so
                    # this moves inside the axes (top-left corner) instead,
                    # matching "Export as DeepXDE Script"'s own Line Animation
                    # GIF time-indicator placement, to avoid the two overlapping
                    # into illegible text.
                    _time_txt_x, _time_txt_y, _time_txt_ha = (0.02, 0.95, 'left') if _plot_title_override else (0.5, 1.02, 'center')
                    _time_txt_gif = _ax_gif.text(_time_txt_x, _time_txt_y, '', transform=_ax_gif.transAxes,
                                                  color='black', ha=_time_txt_ha, fontsize=11)
                    _grid(_ax_gif, alpha=0.2)
                    def _init_gif():
                        _line_gif.set_data([], []); _time_txt_gif.set_text(''); return _line_gif, _time_txt_gif
                    def _update_gif(i):
                        _line_gif.set_data(_x_gif, _all_u_gif[i])
                        _time_txt_gif.set_text(f"t = {{_fmt_num(_t_frames[i])}}")
                        return _line_gif, _time_txt_gif
                    _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, init_func=_init_gif, frames=_n_frames, interval=100, blit=True)
                    _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps, dpi=_PLOT_DPI_ANIM)
                    plt.close(_fig_gif)

                else:  # Surface Animation (GIF)
                    if _is_3d:
                        # Capped, unlike every other plot_resolution use in this
                        # file: this renders 6 faces via ax.plot_surface(rstride=1,
                        # cstride=1, ...) -- a per-quad renderer, not contourf --
                        # for EVERY animation frame. At plot_resolution's own
                        # default (200) that's ~31x more quads per face than the
                        # resolution (36) this used to be hardcoded to, and
                        # measured well over 10x slower wall-clock as a direct
                        # result (not from prediction cost, from matplotlib's own
                        # per-quad rendering). A user who explicitly lowers
                        # plot_resolution below this cap still gets an even
                        # coarser/faster 3D GIF, same as every other plot type.
                        _res3a = {res_3d(config)}   # Plot settings -> Surface -> 3D grid resolution
                        _bbox3a = np.asarray(geom.bbox)
                        _cx0a, _cy0a, _cz0a = _bbox3a[0]; _cx1a, _cy1a, _cz1a = _bbox3a[1]
                        _xg3a = np.linspace(_cx0a, _cx1a, _res3a)
                        _yg3a = np.linspace(_cy0a, _cy1a, _res3a)
                        _zg3a = np.linspace(_cz0a, _cz1a, _res3a)
                        _Xxy3a, _Yxy3a = np.meshgrid(_xg3a, _yg3a)
                        _Xxz3a, _Zxz3a = np.meshgrid(_xg3a, _zg3a)
                        _Yyz3a, _Zyz3a = np.meshgrid(_yg3a, _zg3a)
                        _faces3a = [
                            (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz0a)),
                            (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz1a)),
                            (_Xxz3a, np.full_like(_Xxz3a, _cy0a), _Zxz3a),
                            (_Xxz3a, np.full_like(_Xxz3a, _cy1a), _Zxz3a),
                            (np.full_like(_Yyz3a, _cx0a), _Yyz3a, _Zyz3a),
                            (np.full_like(_Yyz3a, _cx1a), _Yyz3a, _Zyz3a),
                        ]
                        # cleanup Round 46: mask every bounding-box face to the
                        # real geometry, same geom.inside(...) convention as the
                        # static 3D Surface plot -- spatial-only, computed once.
                        _face_inside3a = []
                        for _fX3a, _fY3a, _fZ3a in _faces3a:
                            _fspatial3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel()])
                            _face_inside3a.append(np.asarray(geom.inside(_fspatial3a)).reshape(_fX3a.shape))
                        _all_frames_gif = []
                        for _tv in _t_frames:
                            _frame_faces_gif = []
                            for _fX3a, _fY3a, _fZ3a in _faces3a:
                                _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel(), np.full(_fX3a.size, _tv)])
                                _frame_faces_gif.append(_extract_plot_field(_fpts3a).reshape(_fX3a.shape))
                            _all_frames_gif.append(_frame_faces_gif)
                        if {config.plot_auto_range}:
                            # In-domain values only (every value if no face
                            # touches the domain, e.g. a Sphere's bounding box).
                            _inside_vals3a = [_f[_m] for _frame in _all_frames_gif for _f, _m in zip(_frame, _face_inside3a) if _m.any()]
                            if _inside_vals3a:
                                _v_min_gif = min(_v.min() for _v in _inside_vals3a)
                                _v_max_gif = max(_v.max() for _v in _inside_vals3a)
                            else:
                                _v_min_gif = min(_f.min() for _frame in _all_frames_gif for _f in _frame)
                                _v_max_gif = max(_f.max() for _frame in _all_frames_gif for _f in _frame)
                        else:
                            _v_min_gif = {config.plot_vmin}
                            _v_max_gif = {config.plot_vmax}
                        _fig_gif = plt.figure(figsize=_plot_figsize(7, 5))
                        _ax_gif = _fig_gif.add_subplot(111, projection='3d')
                        _norm3a = plt.Normalize(vmin=_v_min_gif, vmax=_v_max_gif)
                        _cmap_obj3a = plt.get_cmap("{config.plot_colormap}")
                        _sm3a = plt.cm.ScalarMappable(cmap=_cmap_obj3a, norm=_norm3a)
                        if {config.plot_colorbar}: _fig_gif.colorbar(_sm3a, ax=_ax_gif, shrink=0.6, pad=0.12)
                        def _update_gif(i):
                            _ax_gif.cla()
                            for _fi3a, (_fX3a, _fY3a, _fZ3a) in enumerate(_faces3a):
                                _fc3a = np.array(_cmap_obj3a(_norm3a(_all_frames_gif[i][_fi3a])))
                                _fc3a[..., 3] = np.where(_face_inside3a[_fi3a], 1.0, 0.0)
                                _ax_gif.plot_surface(_fX3a, _fY3a, _fZ3a, facecolors=_fc3a,
                                                 rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
                            _ax_gif.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); _ax_gif.set_ylabel(_plot_ylabel_override or "y", fontsize={config.plot_axis_label_fontsize}); _ax_gif.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax_gif.xaxis, _ax_gif.yaxis, _ax_gif.zaxis)]
                            _ax_gif.set_title(_plot_title_override or f"t = {{_fmt_num(_t_frames[i])}}")
                            try:
                                _ax_gif.set_box_aspect((_cx1a - _cx0a, _cy1a - _cy0a, _cz1a - _cz0a))
                            except Exception:
                                pass
                        _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, frames=_n_frames, interval=150)
                        _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps, dpi=_PLOT_DPI_ANIM)
                        plt.close(_fig_gif)
                    else:
                        _res_anim_gif = {config.plot_resolution}
                        _x_anim_gif = np.linspace(_plot_x_min, _plot_x_max, _res_anim_gif)
                        _all_frames_gif = []
                        if _is_2d:
                            _y_anim_gif = np.linspace(_plot_y_min, _plot_y_max, _res_anim_gif)
                            _Xg_gif, _Yg_gif = np.meshgrid(_x_anim_gif, _y_anim_gif)
                            # cleanup Round 46: NaN out grid points outside the
                            # real geometry, same as the static 2D Surface plot.
                            _inside2d_gif = np.asarray(geom.inside(np.column_stack([_Xg_gif.ravel(), _Yg_gif.ravel()]))).reshape(_res_anim_gif, _res_anim_gif)
                            for _tv in _t_frames:
                                _xyt_gif = np.column_stack([_Xg_gif.ravel(), _Yg_gif.ravel(), np.full(_Xg_gif.size, _tv)])
                                _pred_gif = _extract_plot_field(_xyt_gif).reshape(_res_anim_gif, _res_anim_gif)
                                _pred_gif = np.where(_inside2d_gif, _pred_gif, np.nan)
                                _all_frames_gif.append((_Xg_gif, _Yg_gif, _pred_gif))
                        else:
                            # Same configurable axis orientation as the Standard
                            # path's static 1D "Surface" plot (Plot Settings ->
                            # "Swap axes") -- there's no real t-variation within
                            # a single frame (each frame is one instant, u(x) at
                            # that t, drawn as a color band against a cosmetic
                            # t-range axis for width), but the swap still needs
                            # honoring so a GIF frame's orientation matches the
                            # static Surface plot's, instead of always being
                            # x-on-horizontal/t-on-vertical regardless of the
                            # setting. Same no-reshape-of-Z trick: swapping which
                            # of _X_anim_gif/_T_anim_gif is stored first is
                            # enough, since contourf reads each array's own
                            # coordinate values rather than assuming an axis order.
                            _t_anim_gif = np.linspace({config.t_min}, {config.t_max}, _res_anim_gif)
                            _X_anim_gif, _T_anim_gif = np.meshgrid(_x_anim_gif, _t_anim_gif)
                            for _tv in _t_frames:
                                _xt_gif2 = np.vstack([_X_anim_gif.ravel(), np.full(_X_anim_gif.size, _tv)]).T
                                _pred_gif = _extract_plot_field(_xt_gif2).reshape(_res_anim_gif, _res_anim_gif)
                                if {config.plot_swap_xt}:
                                    _all_frames_gif.append((_T_anim_gif, _X_anim_gif, _pred_gif))
                                else:
                                    _all_frames_gif.append((_X_anim_gif, _T_anim_gif, _pred_gif))
                        if {config.plot_auto_range}:
                            _v_min_gif = min(np.nanmin(_f[2]) for _f in _all_frames_gif)
                            _v_max_gif = max(np.nanmax(_f[2]) for _f in _all_frames_gif)
                        else:
                            _v_min_gif = {config.plot_vmin}
                            _v_max_gif = {config.plot_vmax}
                        if _is_2d:
                            _xlabel_gif, _ylabel_gif = "x", "y"
                        elif {config.plot_swap_xt}:
                            _xlabel_gif, _ylabel_gif = "t", "x"
                        else:
                            _xlabel_gif, _ylabel_gif = "x", "t"
                        _fig_gif, _ax_gif = plt.subplots(figsize=_plot_figsize(7, 5))
                        from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable_gif
                        _div_gif = _make_axes_locatable_gif(_ax_gif)
                        _cax_gif = _div_gif.append_axes("right", size="5%", pad=0.1)
                        _sm_gif = plt.cm.ScalarMappable(cmap="{config.plot_colormap}", norm=plt.Normalize(vmin=_v_min_gif, vmax=_v_max_gif))
                        if {config.plot_colorbar}: _fig_gif.colorbar(_sm_gif, cax=_cax_gif)
                        def _update_gif(i):
                            _ax_gif.cla()
                            _Xp_gif, _Yp_gif, _Zp_gif = _all_frames_gif[i]
                            _ax_gif.contourf(_Xp_gif, _Yp_gif, _Zp_gif, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _v_min_gif, _v_max_gif, _Zp_gif))
                            _ax_gif.set_xlabel(_plot_xlabel_override or _xlabel_gif, fontsize={config.plot_axis_label_fontsize})
                            _ax_gif.set_ylabel(_plot_ylabel_override or _ylabel_gif, fontsize={config.plot_axis_label_fontsize})
                            _ax_gif.set_title(_plot_title_override or f"t = {{_fmt_num(_t_frames[i])}}")
                        _ani_gif = _anim.FuncAnimation(_fig_gif, _update_gif, frames=_n_frames, interval=150)
                        _ani_gif.save(_run_solution_path, writer='pillow', fps=_fps, dpi=_PLOT_DPI_ANIM)
                        plt.close(_fig_gif)

            elif _is_2d and _is_steady and _plot_type == "Line (time steps)":
                # Steady 2D "Line (time steps)": no time axis to step through,
                # so this is a single curve u(x) at the configured y slice --
                # same "one curve regardless of plot type" simplification the
                # steady-1D branch below already uses. Previously this
                # selection silently fell through to the x-y heatmap branch
                # just below (plot_type was never checked for 2D/3D at all).
                # Masked to the real geometry with geom.inside(...) -- same
                # pattern the x-y heatmap branch below already uses, and the
                # same fix Error Analysis's own Line Comparison plot got in
                # Rounds 36/37 -- so a non-rectangular domain (L-Shape, Disk,
                # ...) doesn't evaluate/plot the model somewhere it was never
                # trained, past the true boundary at this y slice.
                _x_l2ds = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                _xy_l2ds = np.column_stack([_x_l2ds, np.full_like(_x_l2ds, {line_slice_y})])
                _line_in_l2ds = geom.inside(_xy_l2ds)
                _x_l2ds = _x_l2ds[_line_in_l2ds]
                _xy_l2ds = _xy_l2ds[_line_in_l2ds]
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
                if len(_x_l2ds) < 2:
                    ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _u_l2ds = _extract_plot_field(_xy_l2ds).flatten()
                    ax.plot(*_line_gaps(_x_l2ds, _u_l2ds), color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
                    ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={{_fmt_num({line_slice_y!r})}})", fontsize={config.plot_axis_label_fontsize})
                    _grid(ax, alpha=0.2)
                ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={{_fmt_num({line_slice_y!r})}})", fontsize={config.plot_title_fontsize})
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_2d and _is_steady:
                # Steady-state 2D: a single static x-y heatmap of u(x,y) -- no
                # time axis, so no time snapshots and no time column on the
                # model input (matches the 2-column network input built for
                # steady 2D problems).
                _res_2d = {config.plot_resolution}
                _xp = np.linspace(_plot_x_min, _plot_x_max, _res_2d)
                _yp = np.linspace(_plot_y_min, _plot_y_max, _res_2d)
                _Xg, _Yg = np.meshgrid(_xp, _yp)
                _inside_2d = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res_2d, _res_2d)
                _vmin_2d = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_2d = None if {config.plot_auto_range} else {config.plot_vmax}

                _XY = np.column_stack([_Xg.ravel(), _Yg.ravel()])
                _pred = _extract_plot_field(_XY).reshape(_res_2d, _res_2d)
                _pred = np.where(_inside_2d, _pred, np.nan)
                fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                im = ax.contourf(_Xg, _Yg, _pred, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_2d, _vmax_2d, _pred))
                ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or "y", fontsize={config.plot_axis_label_fontsize})
                ax.set_aspect("equal", adjustable="box")
                if {config.plot_colorbar}: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y)", fontsize={config.plot_title_fontsize})
                plt.tight_layout()
                plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_2d and _plot_type == "Line (time steps)":
                # 2D "Line (time steps)": overlaid u(x) curves at several times,
                # all at the configured y slice -- same convention the 1D
                # "Line (time steps)" branch further below already uses
                # (num_timesteps_line curves, color-graded by time, legend by
                # t). Previously this selection silently fell through to the
                # x-y heatmap branch just below (plot_type was never checked
                # for 2D/3D at all). Masked to the real geometry with
                # geom.inside(...) -- the mask only depends on (x, y), not t,
                # so it's computed once outside the time loop -- same pattern
                # as the steady-2D Line branch above.
                _n_steps_l2d = {config.num_timesteps_line}
                _x_l2d = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                _xy_l2d = np.column_stack([_x_l2d, np.full_like(_x_l2d, {line_slice_y})])
                _line_in_l2d = geom.inside(_xy_l2d)
                _x_l2d = _x_l2d[_line_in_l2d]
                _t_steps_l2d = np.linspace({config.t_min}, {config.t_max}, _n_steps_l2d)
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
                if len(_x_l2d) < 2:
                    ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _colors_l2d = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, _n_steps_l2d))
                    for _i_l2d, _tv_l2d in enumerate(_t_steps_l2d):
                        _xyt_l2d = np.column_stack([_x_l2d, np.full_like(_x_l2d, {line_slice_y}), np.full_like(_x_l2d, _tv_l2d)])
                        _u_l2d = _extract_plot_field(_xyt_l2d).flatten()
                        ax.plot(*_line_gaps(_x_l2d, _u_l2d), color=_colors_l2d[_i_l2d], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(_tv_l2d)}}")
                    ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},t)", fontsize={config.plot_axis_label_fontsize})
                    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
                ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},t)", fontsize={config.plot_title_fontsize})
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_2d:
                # 2D: x-y heatmaps at user-selected number of time snapshots
                _n_snaps = {config.plot_n_2d_snapshots}
                _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
                _res_2d = {config.plot_resolution}
                _xp = np.linspace(_plot_x_min, _plot_x_max, _res_2d)
                _yp = np.linspace(_plot_y_min, _plot_y_max, _res_2d)
                _Xg, _Yg = np.meshgrid(_xp, _yp)
                # Outside the shape's own boundary (only differs from the bbox
                # for Disk/Ellipse/Triangle/Polygon), mask the grid so the plot
                # doesn't show extrapolated model output past the true domain.
                _inside_2d = geom.inside(np.column_stack([_Xg.ravel(), _Yg.ravel()])).reshape(_res_2d, _res_2d)
                _vmin_2d = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_2d = None if {config.plot_auto_range} else {config.plot_vmax}

                fig, axes = plt.subplots(1, _n_snaps, figsize=(_plot_figsize(5, 5)[0]*_n_snaps, _plot_figsize(5, 5)[1]))
                if _n_snaps == 1: axes = [axes]
                for _ai, _tv in enumerate(_t_snaps):
                    _XYT = np.column_stack([_Xg.ravel(), _Yg.ravel(), np.full(_Xg.size, _tv)])
                    _pred = _extract_plot_field(_XYT).reshape(_res_2d, _res_2d)
                    _pred = np.where(_inside_2d, _pred, np.nan)
                    im = axes[_ai].contourf(_Xg, _Yg, _pred, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_2d, _vmax_2d, _pred))
                    axes[_ai].set_title(f"t = {{_fmt_num(_tv)}}")
                    axes[_ai].set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); axes[_ai].set_ylabel(_plot_ylabel_override or "y", fontsize={config.plot_axis_label_fontsize})
                    if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,t)", fontsize={config.plot_title_fontsize})
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_3d and _is_steady and _plot_type == "Line (time steps)":
                # Steady 3D "Line (time steps)": no time axis, so a single
                # curve u(x) at the configured y and z slice -- same "one
                # curve" simplification the steady-2D Line branch above uses.
                # Masked to the real geometry with geom.inside(...) -- same
                # pattern as the steady-2D Line branch above.
                _x_l3ds = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                _xyz_l3ds = np.column_stack([_x_l3ds, np.full_like(_x_l3ds, {line_slice_y}), np.full_like(_x_l3ds, {line_slice_z})])
                _line_in_l3ds = geom.inside(_xyz_l3ds)
                _x_l3ds = _x_l3ds[_line_in_l3ds]
                _xyz_l3ds = _xyz_l3ds[_line_in_l3ds]
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
                if len(_x_l3ds) < 2:
                    ax.text(0.5, 0.5, "Line outside the domain at this y/z slice", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _u_l3ds = _extract_plot_field(_xyz_l3ds).flatten()
                    ax.plot(*_line_gaps(_x_l3ds, _u_l3ds), color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
                    ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},z={{_fmt_num({line_slice_z!r})}})", fontsize={config.plot_axis_label_fontsize})
                    _grid(ax, alpha=0.2)
                ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},z={{_fmt_num({line_slice_z!r})}})", fontsize={config.plot_title_fontsize})
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_3d and _is_steady:
                # Steady-state 3D: same "x-y heatmap at the z mid-plane"
                # simplification as the time-dependent 3D branch below, but a
                # single static plot -- no time axis, no time snapshots, and a
                # 3-column (x,y,z) model input rather than 4.
                _res_3d = {config.plot_resolution}
                _xp3 = np.linspace(_plot_x_min, _plot_x_max, _res_3d)
                _yp3 = np.linspace(_plot_y_min, _plot_y_max, _res_3d)
                _Xg3, _Yg3 = np.meshgrid(_xp3, _yp3)
                _z_mid = {line_slice_z!r}   # Plot settings -> Line: z (the middle by default)
                _inside_3d = geom.inside(np.column_stack(
                    [_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])).reshape(_res_3d, _res_3d)
                _vmin_3d = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_3d = None if {config.plot_auto_range} else {config.plot_vmax}

                _XYZ = np.column_stack([_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])
                _pred3 = _extract_plot_field(_XYZ).reshape(_res_3d, _res_3d)
                _pred3 = np.where(_inside_3d, _pred3, np.nan)
                fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
                im = ax.contourf(_Xg3, _Yg3, _pred3, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_3d, _vmax_3d, _pred3))
                ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or "y", fontsize={config.plot_axis_label_fontsize})
                ax.set_aspect("equal", adjustable="box")
                if {config.plot_colorbar}: fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,z={{_fmt_num(_z_mid)}})", fontsize={config.plot_title_fontsize})
                plt.tight_layout()
                plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_3d and _plot_type == "Line (time steps)":
                # 3D "Line (time steps)": overlaid u(x) curves at several
                # times, all at the configured y and z slice -- same
                # convention as the 2D Line branch above and the 1D one
                # further below. Previously this selection silently fell
                # through to the x-y heatmap branch just below. Masked to the
                # real geometry with geom.inside(...) -- the mask only depends
                # on (x, y, z), not t, so it's computed once outside the time
                # loop -- same pattern as the 2D time-dependent Line branch
                # above.
                _n_steps_l3d = {config.num_timesteps_line}
                _x_l3d = np.linspace(_plot_x_min, _plot_x_max, {config.plot_resolution})
                _xyz_l3d = np.column_stack([_x_l3d, np.full_like(_x_l3d, {line_slice_y}), np.full_like(_x_l3d, {line_slice_z})])
                _line_in_l3d = geom.inside(_xyz_l3d)
                _x_l3d = _x_l3d[_line_in_l3d]
                _t_steps_l3d = np.linspace({config.t_min}, {config.t_max}, _n_steps_l3d)
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
                if len(_x_l3d) < 2:
                    ax.text(0.5, 0.5, "Line outside the domain at this y/z slice", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    _colors_l3d = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, _n_steps_l3d))
                    for _i_l3d, _tv_l3d in enumerate(_t_steps_l3d):
                        _xyzt_l3d = np.column_stack([_x_l3d, np.full_like(_x_l3d, {line_slice_y}), np.full_like(_x_l3d, {line_slice_z}), np.full_like(_x_l3d, _tv_l3d)])
                        _u_l3d = _extract_plot_field(_xyzt_l3d).flatten()
                        ax.plot(*_line_gaps(_x_l3d, _u_l3d), color=_colors_l3d[_i_l3d], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(_tv_l3d)}}")
                    ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},z={{_fmt_num({line_slice_z!r})}},t)", fontsize={config.plot_axis_label_fontsize})
                    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
                ax.set_title(_plot_title_override or f"PINN Solution — {{out_name}}(x,y={{_fmt_num({line_slice_y!r})}},z={{_fmt_num({line_slice_z!r})}},t)", fontsize={config.plot_title_fontsize})
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_3d:
                # 3D: no volumetric renderer yet -- plot an x-y heatmap at the
                # domain's z mid-plane, at the same time snapshots the 2D case
                # uses, so the run finishes with a solution plot instead of
                # crashing. Full 3D visualization (multiple slices / isosurfaces)
                # is a possible future addition once there's a template that
                # needs it.
                _n_snaps = {config.plot_n_2d_snapshots}
                _t_snaps = np.linspace({config.t_min}, {config.t_max}, _n_snaps)
                _res_3d = {config.plot_resolution}
                _xp3 = np.linspace(_plot_x_min, _plot_x_max, _res_3d)
                _yp3 = np.linspace(_plot_y_min, _plot_y_max, _res_3d)
                _Xg3, _Yg3 = np.meshgrid(_xp3, _yp3)
                _z_mid = {line_slice_z!r}   # Plot settings -> Line: z (the middle by default)
                # Mask points outside the true 3D shape at this z-slice (only
                # differs from the bbox for Sphere) -- same reasoning as the 2D
                # branch above.
                _inside_3d = geom.inside(np.column_stack(
                    [_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid)])).reshape(_res_3d, _res_3d)
                _vmin_3d = None if {config.plot_auto_range} else {config.plot_vmin}
                _vmax_3d = None if {config.plot_auto_range} else {config.plot_vmax}

                fig, axes = plt.subplots(1, _n_snaps, figsize=(_plot_figsize(5, 5)[0]*_n_snaps, _plot_figsize(5, 5)[1]))
                if _n_snaps == 1: axes = [axes]
                for _ai, _tv in enumerate(_t_snaps):
                    _XYZT = np.column_stack([_Xg3.ravel(), _Yg3.ravel(), np.full(_Xg3.size, _z_mid), np.full(_Xg3.size, _tv)])
                    _pred3 = _extract_plot_field(_XYZT).reshape(_res_3d, _res_3d)
                    _pred3 = np.where(_inside_3d, _pred3, np.nan)
                    im = axes[_ai].contourf(_Xg3, _Yg3, _pred3, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_3d, _vmax_3d, _pred3))
                    axes[_ai].set_title(f"t = {{_fmt_num(_tv)}}, z = {{_fmt_num(_z_mid)}}")
                    axes[_ai].set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); axes[_ai].set_ylabel(_plot_ylabel_override or "y", fontsize={config.plot_axis_label_fontsize})
                    if {config.plot_colorbar}: fig.colorbar(im, ax=axes[_ai])
                out_name = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig.suptitle(_plot_title_override or f"PINN Solution — {{out_name}}(x,y,z={{_fmt_num(_z_mid)}},t)", fontsize={config.plot_title_fontsize})
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            elif _is_steady:
                # Steady-state 1D: a single static curve u(x) -- no time axis
                # at all, so neither of the time-dependent 1D plot types
                # ("Surface" over x,t / "Line (time steps)") apply.
                _res_1d = {config.plot_resolution}
                _x_1d = np.linspace({config.x_min}, {config.x_max}, _res_1d)
                _u_1d = _extract_plot_field(_x_1d.reshape(-1, 1)).flatten()
                _out_name_1d = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
                ax.plot(_x_1d, _u_1d, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
                ax.set_xlabel(_plot_xlabel_override or _v1d, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{_out_name_1d}}({{_v1d}})", fontsize={config.plot_axis_label_fontsize})
                ax.set_title(_plot_title_override or (f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric
                              else (f"PINN Solution — {{_out_name_1d}}({{_v1d}})" if _plot_n_out > 1 or _plot_custom_expr.strip() else "PINN Solution")),
                             fontsize={config.plot_title_fontsize})
                _grid(ax, alpha=0.2)
                plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
            else:
                # 1D plot (_plot_type already computed above)
                if _plot_type == "Surface":
                    _res = {config.plot_resolution}
                    _x_s = np.linspace({config.x_min}, {config.x_max}, _res)
                    _t_s = np.linspace({config.t_min}, {config.t_max}, _res)
                    _Xs, _Ts = np.meshgrid(_x_s, _t_s)
                    _XTs = np.vstack([_Xs.ravel(), _Ts.ravel()]).T
                    _u_s = _extract_plot_field(_XTs).reshape(_res, _res)
                    _out_name_1dt = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                    _vmin_s = None if {config.plot_auto_range} else {config.plot_vmin}
                    _vmax_s = None if {config.plot_auto_range} else {config.plot_vmax}
                    print(f"Plot settings: cmap={config.plot_colormap}, levels={config.plot_levels}, dpi={config.plot_dpi}, res={config.plot_resolution}")
                    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
                    # Axis orientation is configurable (Plot Settings ->
                    # "Swap axes"); contourf just needs its three arrays to
                    # line up element-for-element, so swapping which of
                    # _Xs/_Ts is passed first -- with no reshape/transpose of
                    # _u_s -- is enough to flip which one lands on the x-axis.
                    if {config.plot_swap_xt}:
                        im = ax.contourf(_Ts, _Xs, _u_s, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_s, _vmax_s, _u_s))
                        ax.set_xlabel(_plot_xlabel_override or "t", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or "x", fontsize={config.plot_axis_label_fontsize})
                    else:
                        im = ax.contourf(_Xs, _Ts, _u_s, cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, _vmin_s, _vmax_s, _u_s))
                        ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or "t", fontsize={config.plot_axis_label_fontsize})
                    if {config.plot_colorbar}: fig.colorbar(im, ax=ax)
                    ax.set_title(_plot_title_override or (f"PINN Solution — {{_out_name_1dt}}(x,t) — {{_param_name}}={{_pval}}" if _parametric
                                  else f"PINN Solution — {{_out_name_1dt}}(x,t)"), fontsize={config.plot_title_fontsize})
                    plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

                elif _plot_type == "Line (time steps)":
                    n_steps_plot = {config.num_timesteps_line}
                    _x_l = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                    t_steps_plot = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
                    fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
                    colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
                    _out_name_line = _plot_custom_label if _plot_custom_expr.strip() else {repr(config.output_names)}.split(",")[_plot_idx].strip()
                    for i, t_val in enumerate(t_steps_plot):
                        xt = np.column_stack([_x_l, np.full_like(_x_l, t_val)])
                        u_line = _extract_plot_field(xt).flatten()
                        ax.plot(_x_l, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(t_val)}}")
                    ax.set_xlabel(_plot_xlabel_override or "x", fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel(_plot_ylabel_override or f"{{_out_name_line}}(x,t)", fontsize={config.plot_axis_label_fontsize})
                    ax.set_title(_plot_title_override or (f"PINN Solution — {{_param_name}}={{_pval}}" if _parametric
                                  else (f"PINN Solution — {{_out_name_line}}(x,t)" if _plot_n_out > 1 or _plot_custom_expr.strip() else "PINN Solution")),
                                 fontsize={config.plot_title_fontsize})
                    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
                    plt.tight_layout(); plt.savefig(_run_solution_path, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()

        _plot_idx, _run_solution_path = _plot_main_job
        import shutil as _shutil
        # the same file can be spelled two ways on Windows (/tmp\\x.png, /tmp/x.png)
        _same_file = lambda _a, _b: _os.path.normcase(_os.path.abspath(_a)) == _os.path.normcase(_os.path.abspath(_b))
        if not _same_file(_run_loss_path, "/tmp/loss_plot.png"):
            _shutil.copy(_run_loss_path, "/tmp/loss_plot.png")
        _tmp_solution_path = "/tmp/solution_plot.{sol_ext}"
        if not _same_file(_run_solution_path, _tmp_solution_path) and _os.path.exists(_run_solution_path):
            _shutil.copy(_run_solution_path, _tmp_solution_path)

        # ── Inline Error Analysis ─────────────────────────────
        if {config.ea_files}:
            from scipy.interpolate import interp1d as _interp1d
            _ea_dir = _os.path.join(_save_dir if _use_save else "/tmp", "error_analysis")
            _os.makedirs(_ea_dir, exist_ok=True)

{_gs_ea_setup_code(config, "Error Analysis", "            ")}

            for _ea_group_key, _ea_group in _ea_groups.items():
                _ea_files = _ea_group["files"]
                _ea_sel = _ea_group["sel"]
                _ea_suffix = f"_{{_ea_group_label(_ea_sel) or 'default'}}" if _ea_multi_output else ""
                if _ea_multi_output:
                    print(f"  ── Output group: {{_ea_group_label(_ea_sel) or 'default'}} ({{len(_ea_files)}} files) ──")

                # Load all ground truth files
                _ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_z_refs = []; _ea_u_refs = []
                for _ea_tv, _ea_fp in _ea_files:
                    _ea_d = np.loadtxt(_ea_fp)
                    if _ea_d.ndim == 1: _ea_d = _ea_d.reshape(1, -1)
                    if _is_steady and _is_3d:
                        # Steady 3D format: x, y, z, u — no time column at all.
                        _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(_ea_d[_ea_idx, 2])
                        _ea_u_refs.append(_ea_d[_ea_idx, 3])
                        _ea_times.append(0.0)
                        print(f"  Loaded ground truth (steady-state): {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_steady and _is_2d:
                        # Steady 2D format: x, y, u — no time column at all.
                        _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 2])
                        _ea_times.append(0.0)
                        print(f"  Loaded ground truth (steady-state): {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_3d:
                        # 3D format: x, y, z, t, u — sort by x, y, z
                        _ea_idx = np.lexsort((_ea_d[:, 2], _ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(_ea_d[_ea_idx, 2])
                        _ea_u_refs.append(_ea_d[_ea_idx, 4])
                        _detected_t = float(_ea_d[0, 3])
                        _ea_times.append(_detected_t)
                        print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_2d:
                        # 2D format: x, y, t, u — sort by x then y
                        _ea_idx = np.lexsort((_ea_d[:, 1], _ea_d[:, 0]))
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(_ea_d[_ea_idx, 1])
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 3])
                        _detected_t = float(_ea_d[0, 2])
                        _ea_times.append(_detected_t)
                        print(f"  Loaded ground truth t={{_detected_t:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    elif _is_steady:
                        # Steady 1D format: x, u — no time column at all
                        # (matches generate_clean_script()'s own steady-1D
                        # EA loader, and the same no-time-column convention
                        # already used above for steady 2D/3D). This branch
                        # was missing here -- the "else" below unconditionally
                        # assumed the time-dependent 3-column (x, t, u)
                        # layout, so a steady 1D reference file crashed with
                        # an out-of-bounds IndexError reading a nonexistent
                        # 3rd column.
                        _ea_idx = np.argsort(_ea_d[:, 0])
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 1])
                        _ea_times.append(0.0)
                        print(f"  Loaded ground truth (steady-state): {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                    else:
                        # 1D format: x, t, u — sort by x
                        _ea_idx = np.argsort(_ea_d[:, 0])
                        _ea_x_refs.append(_ea_d[_ea_idx, 0])
                        _ea_y_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_z_refs.append(np.zeros_like(_ea_d[_ea_idx, 0]))
                        _ea_u_refs.append(_ea_d[_ea_idx, 2])
                        _ea_times.append(float(_ea_tv))
                        print(f"  Loaded ground truth t={{_ea_tv:.4f}}: {{len(_ea_d)}} pts from {{_os.path.basename(_ea_fp)}}")
                # Drop reference points with no solution value (NaN) -- e.g. grid
                # points outside a non-rectangular geometry like L-Shape's missing
                # quadrant or Disk/Sphere's bounding-box corners. A no-op for
                # reference files that don't have any (the usual case).
                for _ei in range(len(_ea_u_refs)):
                    _ea_valid = ~np.isnan(_ea_u_refs[_ei])
                    if not _ea_valid.all():
                        _n_dropped = int((~_ea_valid).sum())
                        _ea_x_refs[_ei] = _ea_x_refs[_ei][_ea_valid]
                        _ea_y_refs[_ei] = _ea_y_refs[_ei][_ea_valid]
                        _ea_z_refs[_ei] = _ea_z_refs[_ei][_ea_valid]
                        _ea_u_refs[_ei] = _ea_u_refs[_ei][_ea_valid]
                        print(f"  Dropped {{_n_dropped}} NaN reference point(s) outside the geometry")
                # Sort all loaded data by time value — outside the loop
                _ea_sort_idx = np.argsort(_ea_times)
                _ea_times  = [_ea_times[_i]  for _i in _ea_sort_idx]
                _ea_x_refs = [_ea_x_refs[_i] for _i in _ea_sort_idx]
                _ea_y_refs = [_ea_y_refs[_i] for _i in _ea_sort_idx]
                _ea_z_refs = [_ea_z_refs[_i] for _i in _ea_sort_idx]
                _ea_u_refs = [_ea_u_refs[_i] for _i in _ea_sort_idx]
                _ea_n_t = len(_ea_times)
                _ea_u_pinns = [None] * _ea_n_t
                _ea_metrics = [None] * _ea_n_t

                if not {config.time_adaptive}:
                    # ── Non-adaptive: use single model ───────────────
                    for _ei, _ea_tv in enumerate(_ea_times):
                        _ea_xf = _ea_x_refs[_ei]
                        if _is_steady and _is_3d:
                            _ea_xt = np.column_stack([_ea_xf, _ea_y_refs[_ei], _ea_z_refs[_ei]])
                        elif _is_steady and _is_2d:
                            _ea_xt = np.column_stack([_ea_xf, _ea_y_refs[_ei]])
                        elif _is_3d:
                            _ea_yf = _ea_y_refs[_ei]
                            _ea_zf = _ea_z_refs[_ei]
                            _ea_xt = np.column_stack([_ea_xf, _ea_yf, _ea_zf, np.full_like(_ea_xf, _ea_tv)])
                        elif _is_2d:
                            _ea_yf = _ea_y_refs[_ei]
                            _ea_xt = np.column_stack([_ea_xf, _ea_yf, np.full_like(_ea_xf, _ea_tv)])
                        elif _is_steady:
                            # Steady 1D: no time column -- this case was
                            # missing (only the 2D/3D steady variants were
                            # handled above), so it fell through to the
                            # time-dependent "else" below and fed the
                            # network a 2-column (x, t) input instead of
                            # the 1 column it was actually trained on.
                            _ea_xt = _ea_xf.reshape(-1, 1)
                        else:
                            _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                        _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, model).flatten()
                        if _is_steady:
                            print(f"  PINN predicted (steady-state): {{len(_ea_xf)}} points")
                        else:
                            print(f"  PINN predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")
                else:
                    # ── Time adaptive: match each GT file to correct step model ──
                    # Reconstruct step intervals from saved models
                    import glob as _ea_glob, json as _ea_json
                    _ta_step_dir = _os.path.join(_save_dir, "time_adaptive_steps")
                    _ta_step_dirs = sorted([_sd for _sd in _ea_glob.glob(_os.path.join(_ta_step_dir, "step_*")) if _os.path.isdir(_sd)])
                    # Parse t0, t1 from each step directory name
                    # Format: step_NNN_tX.XXXX_to_tY.YYYY
                    _ta_intervals = []
                    for _sd in _ta_step_dirs:
                        _sd_name = _os.path.basename(_sd)
                        try:
                            _parts = _sd_name.split("_")
                            _t0_str = _parts[2].replace("t","")
                            _t1_str = _parts[4].replace("t","")
                            _ta_intervals.append((float(_t0_str), float(_t1_str), _sd))
                        except Exception as _pe:
                            print(f"  Could not parse step dir: {{_sd_name}}: {{_pe}}")

                    print(f"  Found {{len(_ta_intervals)}} time-adaptive step models")

                    # Load each step model once and predict for all GT files in its interval
                    for _si, (_t0_i, _t1_i, _sd_i) in enumerate(_ta_intervals):
                        # Find GT files whose time falls in [t0, t1]
                        # For the last step include t1, for others use t0 <= t < t1
                        # except t0 of first step includes t=t_min
                        _is_last = (_si == len(_ta_intervals) - 1)
                        _matching = []
                        for _ei, _ea_tv in enumerate(_ea_times):
                            if _is_last:
                                _in_range = (_t0_i <= _ea_tv <= _t1_i + 1e-10)
                            else:
                                _in_range = (_t0_i <= _ea_tv < _t1_i - 1e-10) or \
                                            (abs(_ea_tv - _t1_i) < 1e-10)  # boundary goes to this step
                            if _in_range and _ea_u_pinns[_ei] is None:
                                _matching.append(_ei)

                        if not _matching:
                            continue

                        print(f"  Step {{_si+1}} [{{_t0_i:.4f}}→{{_t1_i:.4f}}]: predicting for t = {{[_ea_times[_ei] for _ei in _matching]}}")

                        # Load step model
                        _step_cfg_path = _os.path.join(_sd_i, "step_config.json")
                        try:
                            with open(_step_cfg_path) as _scf:
                                _step_cfg = _ea_json.load(_scf)
                        except Exception:
                            _step_cfg = {{"layers": {config.layers}, "activation": "{config.activation}", "loss_type": "{config.loss_type}"}}

                        _step_layers = _step_cfg.get("layers", {config.layers})
                        _step_act    = _step_cfg.get("activation", "{config.activation}")
                        _step_loss   = _step_cfg.get("loss_type", "{config.loss_type}")

                        # Build minimal geometry for this step
                        _step_geom = dde.geometry.Interval({config.x_min}, {config.x_max})
                        _step_td   = dde.geometry.TimeDomain(_t0_i, _t1_i)
                        _step_gt   = dde.geometry.GeometryXTime(_step_geom, _step_td)
                        def _step_pde(x, y): return y[:, 0:1] * 0
                        _step_data  = dde.data.TimePDE(_step_gt, _step_pde, [], num_domain=100, num_test=100)
                        _step_net   = _apply_net_transforms(_make_net(_step_layers, _step_act, "Glorot uniform"))
                        _step_model = dde.Model(_step_data, _step_net)

                        # Find best saved model for this step (lbfgs preferred)
                        _step_pt = ""
                        for _pat in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                            _step_pts = sorted(_ea_glob.glob(_os.path.join(_sd_i, _pat)))
                            if _step_pts:
                                _step_pt = max(_step_pts, key=_os.path.getmtime)
                                break

                        if not _step_pt:
                            print(f"  ⚠️ No model found for step {{_si+1}}, skipping")
                            continue

                        # Compile and restore
                        if "lbfgs" in _os.path.basename(_step_pt):
                            dde.optimizers.set_LBFGS_options(maxiter=1)
                            _step_model.compile("L-BFGS", loss=_step_loss)
                        else:
                            _step_model.compile("adam", lr=0.001, loss=_step_loss)

                        _step_model.restore(_step_pt, verbose=0)
                        print(f"    Restored: {{_os.path.basename(_step_pt)}}")

                        # Predict for each matching GT file
                        for _ei in _matching:
                            _ea_xf = _ea_x_refs[_ei]
                            _ea_tv = _ea_times[_ei]
                            _ea_xt = np.column_stack([_ea_xf, np.full_like(_ea_xf, _ea_tv)])
                            _ea_u_pinns[_ei] = _ea_extract(_ea_xt, _ea_sel, _step_model).flatten()
                            print(f"    Predicted at t={{_ea_tv:.4f}}: {{len(_ea_xf)}} points")

                    # Fill any unmatched with zeros (safety)
                    for _ei in range(_ea_n_t):
                        if _ea_u_pinns[_ei] is None:
                            print(f"  ⚠️ No prediction for t={{_ea_times[_ei]:.4f}} — skipping")
                            _ea_u_pinns[_ei] = np.zeros_like(_ea_u_refs[_ei])

                # ── Compute metrics ───────────────────────────────────
                for _ei, _ea_tv in enumerate(_ea_times):
                    _up = _ea_u_pinns[_ei]; _uf = _ea_u_refs[_ei]
                    _ea_abs = np.abs(_up - _uf)
                    _ea_l2  = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
                    _ea_mse = np.mean((_up - _uf)**2)
                    _ea_mx  = np.max(_ea_abs)
                    _ea_ma  = np.mean(_ea_abs)
                    _ea_metrics[_ei] = (_ea_tv, _ea_l2, _ea_mse, _ea_mx, _ea_ma)
                    print(f"  t={{_ea_tv:.4f}} — {_ea_metrics_print_fragment(config, '_ea_l2', '_ea_mse', '_ea_mx', '_ea_ma')}")

                _ea_metrics_path = _os.path.join(_ea_dir, f"error_metrics{{_ea_suffix}}.txt")
                with open(_ea_metrics_path, "w") as _emf:
                    _emf.write("{_ea_metrics_csv_header(config)}\\n")
                    for _ea_tv, _l2, _mse, _mx, _ma in _ea_metrics:
                        _emf.write(f"{_ea_metrics_row_code(config, '_ea_tv', '_l2', '_mse', '_mx', '_ma')}\\n")
                print(f"  Metrics saved: {{_ea_metrics_path}}")

                # ── Line comparison ───────────────────────────────────
                if {config.ea_do_line}:
                    _ea_ncols = min({max(1, int(getattr(config, 'plot_ea_cols', 4) or 4))}, _ea_n_t)
                    _ea_nrows = (_ea_n_t + _ea_ncols - 1) // _ea_ncols
                    # One panel: a 6 x 4.5 figure whose only title carries the
                    # error; several times: one panel each, one figure title.
                    _ea_single = _ea_n_t == 1
                    fig, axes = plt.subplots(_ea_nrows, _ea_ncols, squeeze=False,
                                             figsize=_panel_figsize(6, 4.5, 1, 1) if _ea_single else _panel_figsize(4.5, 3.8, _ea_ncols, _ea_nrows))
                    _ea_line_suptitle = _ea_title_txt("Line Comparison", _ea_sel)
                    _ea_line_head = _ea_line_suptitle.replace(" — Line Comparison", "")
                    if _is_3d:
                        _ea_line_head += f" (y={line_slice_y:.3g}, z={line_slice_z:.3g})"
                    elif _is_2d:
                        _ea_line_head += f" (y={line_slice_y:.3g})"
                    if not _ea_single:
                        fig.suptitle(_ea_line_head, fontsize={config.plot_subplot_title_fontsize} + 2)
                    _ea_ax_flat = axes.flatten()
                    if _is_2d or _is_3d:
                        # A true line at the exact slice (PINNConfig.
                        # line_slice_y/_z), not a scatter of whichever
                        # reference-file points happened to fall within a
                        # tolerance BAND around it. The PINN curve is
                        # evaluated directly on that line (same convention
                        # the Surface Comparison above already uses: PINN
                        # exact, reference interpolated), and the reference
                        # is interpolated onto it from its own scattered
                        # points via griddata (nearest-neighbor fallback
                        # for any point outside its convex hull), masked to
                        # the real geometry so a non-rectangular domain
                        # (e.g. the L-Shape) doesn't interpolate straight
                        # across its own missing region. The previous
                        # tolerance-band approach plotted every reference
                        # point within (range)/20 of the slice (widened to
                        # /5 if fewer than 5 matched) at that point's own x
                        # -- for any problem where u genuinely varies with
                        # y (or z) within that band, several different u
                        # values landed on top of each other at the same x,
                        # smearing the curve into an unreadable band instead
                        # of a clean line. That's especially likely on a
                        # non-box geometry, where the band is asymmetric
                        # near the slice (e.g. the L-Shape only has y<=0
                        # past x=0), which both shrinks the matched-point
                        # count (triggering the wider /5 fallback) and
                        # concentrates it right where RAR already found the
                        # sharpest residuals.
                        from scipy.interpolate import griddata as _gd_line
                        _xv_line = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                        _y_mid = {line_slice_y}
                        _z_mid = {line_slice_z}
                    for _ei in range(_ea_n_t):
                        ax = _ea_ax_flat[_ei]
                        if _is_3d:
                            _line_pts3 = np.column_stack([_xv_line, np.full_like(_xv_line, _y_mid), np.full_like(_xv_line, _z_mid)])
                            _line_in3 = geom.inside(_line_pts3)
                            _xv_s = _xv_line[_line_in3]
                            _line_pts3_in = _line_pts3[_line_in3]
                            if len(_xv_s) < 2:
                                ax.text(0.5, 0.5, "Line outside the domain at this (y, z) slice", ha='center', va='center', transform=ax.transAxes)
                                ax.set_xticks([]); ax.set_yticks([])
                                continue
                            _line_eval3 = (_line_pts3_in if _is_steady else
                                           np.column_stack([_line_pts3_in, np.full(len(_xv_s), _ea_times[_ei])]))
                            _pinn_s = _ea_extract(_line_eval3, _ea_sel, model).flatten()
                            _ref_pts3 = np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei], _ea_z_refs[_ei]])
                            _gt_s = _gd_line(_ref_pts3, _ea_u_refs[_ei], _line_pts3_in, method='linear')
                            _gt_nan3 = np.isnan(_gt_s)
                            if _gt_nan3.any():
                                _gt_s[_gt_nan3] = _gd_line(_ref_pts3, _ea_u_refs[_ei], _line_pts3_in[_gt_nan3], method='nearest')
                        elif _is_2d:
                            _line_pts2 = np.column_stack([_xv_line, np.full_like(_xv_line, _y_mid)])
                            _line_in2 = geom.inside(_line_pts2)
                            _xv_s = _xv_line[_line_in2]
                            _line_pts2_in = _line_pts2[_line_in2]
                            if len(_xv_s) < 2:
                                ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha='center', va='center', transform=ax.transAxes)
                                ax.set_xticks([]); ax.set_yticks([])
                                continue
                            _line_eval2 = (_line_pts2_in if _is_steady else
                                           np.column_stack([_line_pts2_in, np.full(len(_xv_s), _ea_times[_ei])]))
                            _pinn_s = _ea_extract(_line_eval2, _ea_sel, model).flatten()
                            _ref_pts2 = np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]])
                            _gt_s = _gd_line(_ref_pts2, _ea_u_refs[_ei], _line_pts2_in, method='linear')
                            _gt_nan2 = np.isnan(_gt_s)
                            if _gt_nan2.any():
                                _gt_s[_gt_nan2] = _gd_line(_ref_pts2, _ea_u_refs[_ei], _line_pts2_in[_gt_nan2], method='nearest')
                        else:
                            _xv = _ea_x_refs[_ei]
                            _ea_sort = np.argsort(_xv)
                            _xv_s   = _xv[_ea_sort]
                            _gt_s   = _ea_u_refs[_ei][_ea_sort]
                            _pinn_s = _ea_u_pinns[_ei][_ea_sort]
                        _ea_tv, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                        if _is_2d or _is_3d:
                            _xv_s, _gt_s, _pinn_s = _line_gaps(_xv_s, _gt_s, _pinn_s)
                        _ref_plot(ax, _xv_s, _gt_s,   color=_PLOT_REF_COLOR, linewidth={config.plot_linewidth}, linestyle='-',  label='Reference')
                        ax.plot(_xv_s, _pinn_s, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth}, linestyle='--', label='PINN')
                        # Steady-state (e.g. a Poisson equation) has no time
                        # axis at all -- every reference file is really just
                        # a single snapshot at a placeholder t=0, so showing
                        # "t = 0.000" here would be meaningless noise rather
                        # than a real time coordinate.
                        _ea_lt = (f"({{_l2_txt(_l2)}})" if _is_steady else f"t = {{_fmt_num(_ea_tv)}}  ({{_l2_txt(_l2)}})")
                        ax.set_title(f"{{_ea_line_head}}     {{_ea_lt}}" if _ea_single else _ea_lt,
                                     fontsize={config.plot_subplot_title_fontsize} + (2 if _ea_single else 0))
                        ax.set_xlabel(_v1d, fontsize={config.plot_axis_label_fontsize}); _ea_fn = (_ea_line_suptitle.split(" — ") + ["", ""])[1] if _ea_line_suptitle.count(" — ") >= 2 else "u"; ax.set_ylabel(f"{{_ea_fn}}({{_v1d}})" if _is_steady else f"{{_ea_fn}}(x,t)", fontsize={config.plot_axis_label_fontsize}); _grid(ax, alpha=0.3)
                    for _ej in range(_ea_n_t, len(_ea_ax_flat)):
                        _ea_ax_flat[_ej].set_visible(False)
                    # Legend inside the (first) panel, in its emptiest corner.
                    _ea_ax_flat[0].legend(loc='best', fontsize=_lfs(10), framealpha=0.9)
                    plt.tight_layout()
                    _ea_lp = _os.path.join(_ea_dir, f"line_comparison{{_ea_suffix}}.png")
                    plt.savefig(_ea_lp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                    print(f"  Line comparison saved: {{_ea_lp}}")

                # ── Surface comparison ────────────────────────────────
                if {config.ea_do_surface}:
                    _ea_did_surface = True
                    if _is_3d and _geom_type != "Sphere":
                        # Box-shaped 3D geometry (Cuboid): a genuine smooth
                        # surface (PINN | Reference | Error), like a COMSOL
                        # surface plot. Each of the geometry's 6 flat faces
                        # (from geom.bbox) is predicted on a fine regular grid;
                        # ground truth is interpolated (griddata) from reference
                        # points near that face onto the same grid; and each
                        # face is drawn with plot_surface's per-quad facecolors
                        # -- unlike a scatter of discrete points, adjacent
                        # same-ish-colored grid quads blend into a continuous-
                        # looking colored surface, matching the smoothness of
                        # the 2D contourf plots above.
                        from scipy.interpolate import griddata as _gd3
                        fig = plt.figure(figsize=_panel_figsize(5, 4.5, 3, _ea_n_t))
                        fig.suptitle(_ea_title_txt("3D Surface Comparison", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                        _geom_bbox_ea = np.asarray(geom.bbox)
                        _bx0, _by0, _bz0 = _geom_bbox_ea[0]
                        _bx1, _by1, _bz1 = _geom_bbox_ea[1]
                        # Capped the same way, and for the same reason, as
                        # the Surface Animation GIF's _res3a above: per-face
                        # griddata + plot_surface(rstride=1, cstride=1, ...)
                        # at plot_resolution's own default (200) measured
                        # well over 10x slower than this used to be hardcoded
                        # to (36) -- a per-quad renderer, not contourf, so it
                        # doesn't scale the way the 2D Error Analysis surface
                        # comparison (which does use contourf, and already
                        # read plot_resolution directly) does.
                        _res3_ea = {res_3d(config)}   # Plot settings -> Surface -> 3D grid resolution
                        _xg3_ea = np.linspace(_bx0, _bx1, _res3_ea)
                        _yg3_ea = np.linspace(_by0, _by1, _res3_ea)
                        _zg3_ea = np.linspace(_bz0, _bz1, _res3_ea)
                        _Xxy_ea, _Yxy_ea = np.meshgrid(_xg3_ea, _yg3_ea)   # z-faces (free: x,y)
                        _Xxz_ea, _Zxz_ea = np.meshgrid(_xg3_ea, _zg3_ea)   # y-faces (free: x,z)
                        _Yyz_ea, _Zyz_ea = np.meshgrid(_yg3_ea, _zg3_ea)   # x-faces (free: y,z)
                        # (fixed axis idx into x/y/z, fixed value, X, Y, Z grids, free-axis idx pair)
                        _faces3_ea = [
                            (2, _bz0, _Xxy_ea, _Yxy_ea, np.full_like(_Xxy_ea, _bz0), (0, 1)),
                            (2, _bz1, _Xxy_ea, _Yxy_ea, np.full_like(_Xxy_ea, _bz1), (0, 1)),
                            (1, _by0, _Xxz_ea, np.full_like(_Xxz_ea, _by0), _Zxz_ea, (0, 2)),
                            (1, _by1, _Xxz_ea, np.full_like(_Xxz_ea, _by1), _Zxz_ea, (0, 2)),
                            (0, _bx0, np.full_like(_Yyz_ea, _bx0), _Yyz_ea, _Zyz_ea, (1, 2)),
                            (0, _bx1, np.full_like(_Yyz_ea, _bx1), _Yyz_ea, _Zyz_ea, (1, 2)),
                        ]
                        _face_tol_ea = (
                            max((_bx1 - _bx0) * 0.02, 1e-6),
                            max((_by1 - _by0) * 0.02, 1e-6),
                            max((_bz1 - _bz0) * 0.02, 1e-6),
                        )
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            _bxr_ea = _ea_x_refs[_ei]; _byr_ea = _ea_y_refs[_ei]; _bzr_ea = _ea_z_refs[_ei]
                            _ur_ea = _ea_u_refs[_ei]
                            _all_coords_ea = (_bxr_ea, _byr_ea, _bzr_ea)
                            _pinn_faces_ea = []
                            _gt_faces_ea = []
                            for _fax_ea, _fval_ea, _fX_ea, _fY_ea, _fZ_ea, _free_idx_ea in _faces3_ea:
                                # Steady-state 3D has no time axis at all --
                                # the network only takes 3 (x,y,z) inputs,
                                # so appending a 4th placeholder column here
                                # (as the time-dependent case needs) crashed
                                # every steady 3D box-geometry (Cuboid)
                                # surface comparison with a matmul shape
                                # mismatch the moment Error Analysis was
                                # configured for one.
                                if _is_steady:
                                    _fpts_ea = np.column_stack([_fX_ea.ravel(), _fY_ea.ravel(), _fZ_ea.ravel()])
                                else:
                                    _fpts_ea = np.column_stack([_fX_ea.ravel(), _fY_ea.ravel(), _fZ_ea.ravel(), np.full(_fX_ea.size, _ea_tv)])
                                _fpinn_ea = _ea_extract(_fpts_ea, _ea_sel, model).reshape(_fX_ea.shape)
                                _near_mask_ea = np.abs(_all_coords_ea[_fax_ea] - _fval_ea) < _face_tol_ea[_fax_ea]
                                if _near_mask_ea.sum() < 4:
                                    _near_mask_ea = np.ones_like(_bxr_ea, dtype=bool)
                                _free0_ea = _all_coords_ea[_free_idx_ea[0]][_near_mask_ea]
                                _free1_ea = _all_coords_ea[_free_idx_ea[1]][_near_mask_ea]
                                _u_near_ea = _ur_ea[_near_mask_ea]
                                _face_arrs_ea = (_fX_ea, _fY_ea, _fZ_ea)
                                _gridA_ea = _face_arrs_ea[_free_idx_ea[0]]
                                _gridB_ea = _face_arrs_ea[_free_idx_ea[1]]
                                _fgt_ea = _gd3(np.column_stack([_free0_ea, _free1_ea]), _u_near_ea, (_gridA_ea, _gridB_ea), method='linear')
                                _nan_mask_ea = np.isnan(_fgt_ea)
                                if _nan_mask_ea.any():
                                    _fgt_nn_ea = _gd3(np.column_stack([_free0_ea, _free1_ea]), _u_near_ea, (_gridA_ea, _gridB_ea), method='nearest')
                                    _fgt_ea[_nan_mask_ea] = _fgt_nn_ea[_nan_mask_ea]
                                _pinn_faces_ea.append(_fpinn_ea)
                                _gt_faces_ea.append(_fgt_ea)
                            _vmin3_ea = min(min(_f.min() for _f in _pinn_faces_ea), min(_f.min() for _f in _gt_faces_ea))
                            _vmax3_ea = max(max(_f.max() for _f in _pinn_faces_ea), max(_f.max() for _f in _gt_faces_ea))
                            _vmin3_ea, _vmax3_ea = _vr(_vmin3_ea, _vmax3_ea)
                            _err_faces_ea = [np.abs(_pf_ea - _gf_ea) for _pf_ea, _gf_ea in zip(_pinn_faces_ea, _gt_faces_ea)]
                            _vmax_err_ea = max(_f.max() for _f in _err_faces_ea)
                            # See the matching comment on the Line comparison
                            # title above -- steady-state has no time axis,
                            # so "t=..." is dropped from every column title.
                            _cols_ea = [
                                (_pinn_faces_ea, (f"PINN  {{_l2_txt(_l2)}}" if _is_steady else f"PINN  t={{_fmt_num(_ea_tv)}}  {{_l2_txt(_l2)}}"), _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_gt_faces_ea,   ("Reference" if _is_steady else f"Reference  t={{_ea_tv:.3f}}"),         _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_err_faces_ea,  (f"|Error|  max = {{_sci_txt(_mx)}}" if _is_steady else f"|Error|  t={{_fmt_num(_ea_tv)}}  max = {{_sci_txt(_mx)}}"), 0.0, _vmax_err_ea, '{err_cmap(config)}'),
                            ]
                            for _col_ea, (_face_vals_ea, _ttl_ea, _vmin_c_ea, _vmax_c_ea, _cmap_c_ea) in enumerate(_cols_ea):
                                _ax3_ea = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _col_ea + 1, projection='3d')
                                _norm3_ea = plt.Normalize(vmin=_vmin_c_ea, vmax=_vmax_c_ea)
                                _cmap_obj_ea = plt.get_cmap(_cmap_c_ea)
                                for _face_i_ea, (_fax_ea, _fval_ea, _fX_ea, _fY_ea, _fZ_ea, _free_idx_ea) in enumerate(_faces3_ea):
                                    _ax3_ea.plot_surface(
                                        _fX_ea, _fY_ea, _fZ_ea,
                                        facecolors=_cmap_obj_ea(_norm3_ea(_face_vals_ea[_face_i_ea])),
                                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False,
                                    )
                                _sm3_ea = plt.cm.ScalarMappable(cmap=_cmap_obj_ea, norm=_norm3_ea)
                                fig.colorbar(_sm3_ea, ax=_ax3_ea, shrink=0.6, pad=0.12)
                                _ax3_ea.set_title(_ttl_ea, fontsize={config.plot_subplot_title_fontsize})
                                _ax3_ea.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax3_ea.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax3_ea.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax3_ea.xaxis, _ax3_ea.yaxis, _ax3_ea.zaxis)]
                                try:
                                    _ax3_ea.set_box_aspect((_bx1 - _bx0, _by1 - _by0, _bz1 - _bz0))
                                except Exception:
                                    pass  # older matplotlib without set_box_aspect -- cosmetic only, safe to skip
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    elif _is_3d:
                        # Non-box 3D geometry (Sphere): no flat-face
                        # parameterization to grid-interpolate onto, so fall
                        # back to a genuine boundary-point scatter -- real
                        # (x, y, z) points that already lie on the geometry's
                        # own boundary (geom.on_boundary()), predicted and
                        # compared directly (no interpolation needed), with a
                        # wireframe box from geom.bbox for spatial context.
                        fig = plt.figure(figsize=_panel_figsize(5, 4.5, 3, _ea_n_t))
                        fig.suptitle(_ea_title_txt("3D Surface Comparison", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                        _geom_bbox_ea = np.asarray(geom.bbox)
                        _bx0, _by0, _bz0 = _geom_bbox_ea[0]
                        _bx1, _by1, _bz1 = _geom_bbox_ea[1]
                        _corners3_ea = [(_bx0,_by0,_bz0),(_bx1,_by0,_bz0),(_bx1,_by1,_bz0),(_bx0,_by1,_bz0),
                                         (_bx0,_by0,_bz1),(_bx1,_by0,_bz1),(_bx1,_by1,_bz1),(_bx0,_by1,_bz1)]
                        _edges3_ea = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            _bxr_ea = _ea_x_refs[_ei]; _byr_ea = _ea_y_refs[_ei]; _bzr_ea = _ea_z_refs[_ei]
                            _bnd_mask_ea = geom.on_boundary(np.column_stack([_bxr_ea, _byr_ea, _bzr_ea]))
                            # A Cartesian reference grid only reliably
                            # intersects a FLAT boundary (a Cuboid's faces,
                            # handled by the box-face branch above) in large
                            # numbers; a CURVED boundary is essentially never
                            # landed on exactly by an axis-aligned grid, so
                            # on_boundary() can come back with only a handful
                            # of coincidental exact matches (e.g. a unit
                            # sphere's 6 axis points) even when the dataset
                            # has thousands of valid points. Falling back
                            # whenever the boundary subset is a small
                            # fraction of the data -- not just when it's
                            # nearly empty -- catches that case too.
                            if _bnd_mask_ea.sum() < 4 or _bnd_mask_ea.sum() < 0.05 * len(_bxr_ea):
                                _bnd_mask_ea = np.ones_like(_bxr_ea, dtype=bool)  # shape has no clean boundary subset -- use all ref points
                            _bx_ea = _bxr_ea[_bnd_mask_ea]; _by_ea = _byr_ea[_bnd_mask_ea]; _bz_ea = _bzr_ea[_bnd_mask_ea]
                            _gt_b_ea = _ea_u_refs[_ei][_bnd_mask_ea]
                            if len(_bx_ea) > 3000:
                                _ea_sub = np.random.default_rng(0).choice(len(_bx_ea), size=3000, replace=False)
                                _bx_ea, _by_ea, _bz_ea, _gt_b_ea = _bx_ea[_ea_sub], _by_ea[_ea_sub], _bz_ea[_ea_sub], _gt_b_ea[_ea_sub]
                            _pinn_b_pts_ea = (np.column_stack([_bx_ea, _by_ea, _bz_ea]) if _is_steady else
                                              np.column_stack([_bx_ea, _by_ea, _bz_ea, np.full_like(_bx_ea, _ea_tv)]))
                            _pinn_b_ea = _ea_extract(_pinn_b_pts_ea, _ea_sel, model).flatten()
                            _err_b_ea = np.abs(_pinn_b_ea - _gt_b_ea)
                            _vmin3_ea = min(_pinn_b_ea.min(), _gt_b_ea.min())
                            _vmax3_ea = max(_pinn_b_ea.max(), _gt_b_ea.max())
                            # See the matching comment on the Line comparison
                            # title above -- steady-state has no time axis,
                            # so "t=..." is dropped from every column title.
                            _cols_ea = [
                                (_pinn_b_ea, (f"PINN  {{_l2_txt(_l2)}}" if _is_steady else f"PINN  t={{_fmt_num(_ea_tv)}}  {{_l2_txt(_l2)}}"), _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_gt_b_ea,   ("Reference" if _is_steady else f"Reference  t={{_ea_tv:.3f}}"),         _vmin3_ea, _vmax3_ea, '{config.plot_colormap}'),
                                (_err_b_ea,  (f"|Error|  max = {{_sci_txt(_mx)}}" if _is_steady else f"|Error|  t={{_fmt_num(_ea_tv)}}  max = {{_sci_txt(_mx)}}"), None, None, '{err_cmap(config)}'),
                            ]
                            for _col_ea, (_vals_ea, _ttl_ea, _vmin_c_ea, _vmax_c_ea, _cmap_c_ea) in enumerate(_cols_ea):
                                _ax3_ea = fig.add_subplot(_ea_n_t, 3, _ei * 3 + _col_ea + 1, projection='3d')
                                for _i3_ea, _j3_ea in _edges3_ea:
                                    _p0_ea, _p1_ea = _corners3_ea[_i3_ea], _corners3_ea[_j3_ea]
                                    _ax3_ea.plot([_p0_ea[0], _p1_ea[0]], [_p0_ea[1], _p1_ea[1]], [_p0_ea[2], _p1_ea[2]],
                                                 color='#888888', linewidth=0.8, alpha=0.6)
                                _sc3_ea = _ax3_ea.scatter(_bx_ea, _by_ea, _bz_ea, c=_vals_ea, cmap=_cmap_c_ea, s=14,
                                                           vmin=_vmin_c_ea, vmax=_vmax_c_ea)
                                fig.colorbar(_sc3_ea, ax=_ax3_ea, shrink=0.6, pad=0.12)
                                _ax3_ea.set_title(_ttl_ea, fontsize={config.plot_subplot_title_fontsize})
                                _ax3_ea.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax3_ea.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); _ax3_ea.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (_ax3_ea.xaxis, _ax3_ea.yaxis, _ax3_ea.zaxis)]
                                try:
                                    _ax3_ea.set_box_aspect((_bx1 - _bx0, _by1 - _by0, _bz1 - _bz0))
                                except Exception:
                                    pass  # older matplotlib without set_box_aspect -- cosmetic only, safe to skip
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    elif _is_2d:
                        # 2D: 3 columns (PINN | Reference | Error), one row per time snapshot
                        fig, axes = plt.subplots(_ea_n_t, 3,
                                                 figsize=_panel_figsize(5, 4, 3, _ea_n_t), squeeze=False)
                        fig.suptitle(_ea_title_txt("2D Heatmaps", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                        _res_ea = {config.plot_resolution}
                        _xg_ea = np.linspace({config.x_min}, {config.x_max}, _res_ea)
                        _yg_ea = np.linspace({config.y_min}, {config.y_max}, _res_ea)
                        _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
                        # Outside the shape's own boundary (only differs from
                        # the bbox for Disk/Ellipse/Triangle/Polygon), mask the
                        # grid to NaN so neither the PINN prediction nor the
                        # griddata-interpolated ground truth shows fabricated
                        # values there -- griddata's Delaunay triangulation
                        # otherwise happily interpolates straight across a
                        # concave notch (e.g. the L-Shape's missing quadrant),
                        # matching the same masking the main solution plot uses.
                        _inside_2d_ea = geom.inside(np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel()])).reshape(_res_ea, _res_ea)
                        from scipy.interpolate import griddata as _gd
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            # PINN prediction on grid
                            _xy_grid = (np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel()]) if _is_steady else
                                        np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _ea_tv)]))
                            _u_pinn_grid = _ea_extract(_xy_grid, _ea_sel, model).reshape(_res_ea, _res_ea)
                            _u_pinn_grid = np.where(_inside_2d_ea, _u_pinn_grid, np.nan)
                            # Reference interpolated onto same grid
                            _u_fem_grid = _gd(
                                np.column_stack([_ea_x_refs[_ei], _ea_y_refs[_ei]]),
                                _ea_u_refs[_ei],
                                (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
                            _u_fem_grid = np.where(_inside_2d_ea, _u_fem_grid, np.nan)
                            _u_err_grid = np.abs(_u_pinn_grid - _u_fem_grid)
                            _vmin_ea = np.nanmin([_u_pinn_grid, _u_fem_grid])
                            _vmax_ea = np.nanmax([_u_pinn_grid, _u_fem_grid])
                            if _vmax_ea - _vmin_ea < 1e-12:
                                _vmax_ea = _vmin_ea + 1e-12
                            _vmin_ea, _vmax_ea = _vr(_vmin_ea, _vmax_ea)   # Plot settings -> Surface -> Colour range
                            # contourf's own vmin/vmax kwargs do NOT constrain
                            # an integer `levels=N` -- each panel still
                            # auto-ranges to its own data, so PINN and Ground
                            # Truth silently got two different color scales
                            # despite passing the same vmin/vmax here. Passing
                            # explicit level BOUNDARIES instead of a count is
                            # what actually makes both panels share one scale.
                            _levels_ea2d = _cf_levels({config.plot_levels}, _vmin_ea, _vmax_ea)["levels"]
                            # Column 0: PINN
                            im0 = axes[_ei][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_grid, levels=_levels_ea2d,
                                                         cmap='{config.plot_colormap}', extend=_vext())
                            # Steady-state has no time axis -- see the
                            # matching comment on the Line comparison title
                            # above -- so "t=..." is dropped from every
                            # column title here too.
                            axes[_ei][0].set_title((f"PINN  {{_l2_txt(_l2)}}" if _is_steady else f"PINN  t={{_fmt_num(_ea_tv)}}  {{_l2_txt(_l2)}}"), fontsize={config.plot_subplot_title_fontsize})
                            axes[_ei][0].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][0].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                            fig.colorbar(im0, ax=axes[_ei][0])
                            # Column 1: Reference
                            im1 = axes[_ei][1].contourf(_Xg_ea, _Yg_ea, _u_fem_grid, levels=_levels_ea2d,
                                                         cmap='{config.plot_colormap}', extend=_vext())
                            axes[_ei][1].set_title(("Reference" if _is_steady else f"Reference  t={{_fmt_num(_ea_tv)}}"), fontsize={config.plot_subplot_title_fontsize})
                            axes[_ei][1].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][1].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                            fig.colorbar(im1, ax=axes[_ei][1])
                            # Column 2: Absolute error
                            im2 = axes[_ei][2].contourf(_Xg_ea, _Yg_ea, _u_err_grid, levels={config.plot_levels}, cmap='{err_cmap(config)}')
                            axes[_ei][2].set_title((f"|Error|  max = {{_sci_txt(_mx)}}" if _is_steady else f"|Error|  t={{_fmt_num(_ea_tv)}}  max = {{_sci_txt(_mx)}}"), fontsize={config.plot_subplot_title_fontsize})
                            axes[_ei][2].set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); axes[_ei][2].set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                            fig.colorbar(im2, ax=axes[_ei][2])
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    elif len(_ea_times) < 2:
                        # A 1D x-t surface needs at least 2 distinct time
                        # snapshots to form a non-degenerate grid -- e.g. only
                        # one reference file provided so far. Skip gracefully
                        # instead of crashing matplotlib's contourf on a (1, N)
                        # array; the line comparison above already covers this
                        # single snapshot.
                        _ea_did_surface = False
                        if _is_steady:
                            print("  Skipping surface comparison — steady-state 1D has no time axis to build an x-t surface from; see the line comparison above")
                        else:
                            print("  Skipping surface comparison — need at least 2 time snapshots for a 1D x-t surface plot")
                    else:
                        # 1D: standard x vs t surface
                        _ea_x_common = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
                        _ea_t_arr = np.array(_ea_times)
                        _ea_U_pinn = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                        _ea_U_fem  = np.zeros((len(_ea_t_arr), len(_ea_x_common)))
                        if not {config.time_adaptive}:
                            for _ei, _ea_tv in enumerate(_ea_times):
                                _ea_xt_c = np.column_stack([_ea_x_common, np.full_like(_ea_x_common, _ea_tv)])
                                _ea_U_pinn[_ei, :] = _ea_extract(_ea_xt_c, _ea_sel, model).flatten()
                                _ea_fi = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                                _ea_U_fem[_ei, :] = _ea_fi(_ea_x_common)
                        else:
                            for _ei in range(_ea_n_t):
                                _ea_fi_pinn = _interp1d(_ea_x_refs[_ei], _ea_u_pinns[_ei], kind='linear', fill_value='extrapolate')
                                _ea_U_pinn[_ei, :] = _ea_fi_pinn(_ea_x_common)
                                _ea_fi_fem = _interp1d(_ea_x_refs[_ei], _ea_u_refs[_ei], kind='linear', fill_value='extrapolate')
                                _ea_U_fem[_ei, :] = _ea_fi_fem(_ea_x_common)
                        _ea_Xg, _ea_Tg = np.meshgrid(_ea_x_common, _ea_t_arr)
                        _ea_U_err = np.abs(_ea_U_pinn - _ea_U_fem)
                        _ea_vmin = min(_ea_U_pinn.min(), _ea_U_fem.min())
                        _ea_vmax = max(_ea_U_pinn.max(), _ea_U_fem.max())
                        if _ea_vmax - _ea_vmin < 1e-12:
                            _ea_vmax = _ea_vmin + 1e-12
                        _ea_vmin, _ea_vmax = _vr(_ea_vmin, _ea_vmax)   # Plot settings -> Surface -> Colour range
                        _levels_ea1d = _cf_levels({config.plot_levels}, _ea_vmin, _ea_vmax)["levels"]  # see the matching comment above -- contourf ignores vmin/vmax with an integer `levels=N`
                        fig, axes_s = plt.subplots(1, 3, figsize=_panel_figsize(5, 5, 3, 1))
                        fig.suptitle(_ea_title_txt("Surface Comparison", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                        im0 = axes_s[0].contourf(_ea_Tg, _ea_Xg, _ea_U_pinn, levels=_levels_ea1d, cmap='{config.plot_colormap}', extend=_vext())
                        axes_s[0].set_title(f"PINN  {{_ea_field_txt(_ea_sel)}}(x,t)"); axes_s[0].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes_s[0].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im0, ax=axes_s[0])
                        im1 = axes_s[1].contourf(_ea_Tg, _ea_Xg, _ea_U_fem, levels=_levels_ea1d, cmap='{config.plot_colormap}', extend=_vext())
                        axes_s[1].set_title(f"Reference  {{_ea_field_txt(_ea_sel)}}(x,t)"); axes_s[1].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes_s[1].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im1, ax=axes_s[1])
                        im2 = axes_s[2].contourf(_ea_Tg, _ea_Xg, _ea_U_err, levels={config.plot_levels}, cmap='{err_cmap(config)}')
                        axes_s[2].set_title("Error  |PINN - Reference|"); axes_s[2].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes_s[2].set_ylabel("x", fontsize={config.plot_axis_label_fontsize})
                        fig.colorbar(im2, ax=axes_s[2])
                        plt.tight_layout(rect=[0, 0, 1, 0.93])
                    if _ea_did_surface:
                        _ea_sp = _os.path.join(_ea_dir, f"surface_comparison{{_ea_suffix}}.png")
                        plt.savefig(_ea_sp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                        print(f"  Surface comparison saved: {{_ea_sp}}")
                    if _ea_did_surface and _is_3d and _geom_type != "Sphere":
                        # Mid-plane slice z = const (Settings' line slice z, the
                        # box's middle by default): the face plot above shows only
                        # the box's outside, so anything inside it -- a sphere in
                        # a box, say -- is only visible here (as a circle).
                        from scipy.interpolate import griddata as _gd_sl
                        _bb_sl = np.asarray(geom.bbox)
                        _zs_sl = {line_slice_z}
                        _res_sl = min({config.plot_resolution}, 120)
                        _Xs_sl, _Ys_sl = np.meshgrid(np.linspace(_bb_sl[0][0], _bb_sl[1][0], _res_sl),
                                                     np.linspace(_bb_sl[0][1], _bb_sl[1][1], _res_sl))
                        fig, _axs_sl = plt.subplots(_ea_n_t, 3, figsize=_panel_figsize(5, 4.2, 3, _ea_n_t), squeeze=False)
                        fig.suptitle(_ea_title_txt(f"Slice z = {{_fmt_num(_zs_sl)}}", _ea_sel), fontsize={config.plot_title_fontsize}, fontweight='bold')
                        for _ei, _ea_tv in enumerate(_ea_times):
                            _ea_tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_ei]
                            _cols_sl = [_Xs_sl.ravel(), _Ys_sl.ravel(), np.full(_Xs_sl.size, _zs_sl)]
                            if not _is_steady:
                                _cols_sl.append(np.full(_Xs_sl.size, _ea_tv))
                            _p_sl = np.asarray(_ea_extract(np.column_stack(_cols_sl), _ea_sel, model)).reshape(_Xs_sl.shape)
                            # reference: its grid plane nearest the slice, or points close to it
                            _zr_sl = _ea_z_refs[_ei]
                            _zn_sl = _zr_sl[np.argmin(np.abs(_zr_sl - _zs_sl))]
                            _m_sl = np.abs(_zr_sl - _zn_sl) < 1e-9 * (1.0 + abs(_zn_sl))
                            if _m_sl.sum() < 10:
                                _m_sl = np.abs(_zr_sl - _zs_sl) < 0.02 * max(_bb_sl[1][2] - _bb_sl[0][2], 1e-12)
                            _xy_sl = np.column_stack([_ea_x_refs[_ei][_m_sl], _ea_y_refs[_ei][_m_sl]])
                            _g_sl = _gd_sl(_xy_sl, _ea_u_refs[_ei][_m_sl], (_Xs_sl, _Ys_sl), method='linear')
                            if np.isnan(_g_sl).any():
                                _g_sl[np.isnan(_g_sl)] = _gd_sl(_xy_sl, _ea_u_refs[_ei][_m_sl], (_Xs_sl, _Ys_sl), method='nearest')[np.isnan(_g_sl)]
                            _e_sl = np.abs(_p_sl - _g_sl)
                            _lo_sl = min(_p_sl.min(), _g_sl.min()); _hi_sl = max(_p_sl.max(), _g_sl.max(), _lo_sl + 1e-12)
                            _lo_sl, _hi_sl = _vr(_lo_sl, _hi_sl)
                            _lv_sl = _cf_levels({config.plot_levels}, _lo_sl, _hi_sl)["levels"]
                            _tt_sl = "" if _is_steady else f"t={{_ea_tv:.3f}}  "
                            for _c_sl, (_v_sl, _ttl_sl, _lev_sl) in enumerate((
                                    (_p_sl, f"PINN  {{_tt_sl}}{{_l2_txt(_l2)}}", _lv_sl),
                                    (_g_sl, f"Reference  {{_tt_sl}}".rstrip(), _lv_sl),
                                    (_e_sl, f"|Error|  {{_tt_sl}}max = {{_sci_txt(float(_e_sl.max()))}}", {config.plot_levels}))):
                                _ax_sl = _axs_sl[_ei][_c_sl]
                                _im_sl = _ax_sl.contourf(_Xs_sl, _Ys_sl, _v_sl, levels=_lev_sl, extend=_vext(_lev_sl),
                                                         cmap=('{err_cmap(config)}' if _c_sl == 2 else '{config.plot_colormap}'))
                                fig.colorbar(_im_sl, ax=_ax_sl)
                                _ax_sl.set_title(_ttl_sl, fontsize={config.plot_subplot_title_fontsize})
                                _ax_sl.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); _ax_sl.set_ylabel("y", fontsize={config.plot_axis_label_fontsize})
                                _ax_sl.set_aspect('equal')
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        _ea_slp = _os.path.join(_ea_dir, f"slice_comparison{{_ea_suffix}}.png")
                        plt.savefig(_ea_slp, dpi={config.plot_dpi}, bbox_inches='tight'); plt.close()
                        print(f"  Slice comparison saved: {{_ea_slp}}")
                print(f"  Group '{{_ea_group_label(_ea_sel) or 'default'}}' analysis complete")
            print("=== Error Analysis Complete ===")

        # ── Export solution data ──────────────────────────────
        # Opt-in (config.export_enabled, default False) -- this used to run
        # unconditionally on every non-Inverse run even if the user never
        # opened the Export Solution Data dialog.
        if _problem_type != "Inverse" and {config.export_enabled}:
            _data_dir = _os.path.join(
                _run_dir if (_parametric and _pval is not None)
                else (_sol_dir if _use_save else "/tmp"),
                "solution_data"
            )
            _os.makedirs(_data_dir, exist_ok=True)
            _x_export = np.linspace(_plot_x_min, _plot_x_max, {config.export_grid_size})
            _t_export = np.linspace({config.t_min}, {config.t_max}, {config.export_t_steps})

            if _is_steady and _is_2d:
                # No time axis -- one export, x,y columns only (matches the
                # 2-column network input built for steady 2D problems).
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _Xe, _Ye = np.meshgrid(_x_export, _y_export)
                _XY_exp = np.column_stack([_Xe.ravel(), _Ye.ravel()])
                _u_exp = model.predict(_XY_exp)
                _header = "x,y," + ",".join(_out_names_list)
                _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _u_exp])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_steady and _is_3d:
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _z_export = np.linspace(_plot_z_min, _plot_z_max, {config.export_grid_size})
                _Xe, _Ye, _Ze = np.meshgrid(_x_export, _y_export, _z_export)
                _XYZ_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel()])
                _u_exp = model.predict(_XYZ_exp)
                _header = "x,y,z," + ",".join(_out_names_list)
                _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), _u_exp])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_steady:
                _u_exp_all = model.predict(_x_export.reshape(-1, 1))
                _header_cols = "x," + ",".join(_out_names_list)
                _out = np.column_stack([_x_export, _u_exp_all])
                _fname = _os.path.join(_data_dir, "solution.txt")
                np.savetxt(_fname, _out, header=_header_cols, delimiter=",", comments="")
            elif _is_2d:
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                for _t_val in _t_export:
                    _Xe, _Ye = np.meshgrid(_x_export, _y_export)
                    _XYT_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val)])
                    _u_exp = model.predict(_XYT_exp)
                    _header = "x,y,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), np.full(_Xe.size, _t_val), _u_exp])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            elif _is_3d:
                # Same "x,y,z,t,<outputs>" column convention _auto_configure_ea
                # (main_window.py) already expects when scanning a template's
                # ref_dir for t_*.txt ground-truth files -- so a COMSOL
                # export dropped in with matching columns will line up with
                # this automatically once it's provided.
                _y_export = np.linspace(_plot_y_min, _plot_y_max, {config.export_grid_size})
                _z_export = np.linspace(_plot_z_min, _plot_z_max, {config.export_grid_size})
                for _t_val in _t_export:
                    _Xe, _Ye, _Ze = np.meshgrid(_x_export, _y_export, _z_export)
                    _XYZT_exp = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), np.full(_Xe.size, _t_val)])
                    _u_exp = model.predict(_XYZT_exp)
                    _header = "x,y,z,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_Xe.ravel(), _Ye.ravel(), _Ze.ravel(), np.full(_Xe.size, _t_val), _u_exp])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header, delimiter=",", comments="")
            else:
                for _t_val in _t_export:
                    _xt_exp = np.column_stack([_x_export, np.full_like(_x_export, _t_val)])
                    _u_exp_all = model.predict(_xt_exp)
                    _header_cols = "x,t," + ",".join(_out_names_list)
                    _out = np.column_stack([_x_export, np.full_like(_x_export, _t_val), _u_exp_all])
                    _fname = _os.path.join(_data_dir, f"solution_t{{_t_val:.4f}}.txt")
                    np.savetxt(_fname, _out, header=_header_cols, delimiter=",", comments="")

            dim_str = "3D" if _is_3d else ("2D" if _is_2d else "1D")
            _steps_desc = "steady (single export)" if _is_steady else f"t_steps={config.export_t_steps}"
            print(f"Solution data saved to: {{_data_dir}} ({{dim_str}}, grid={config.export_grid_size}, {{_steps_desc}})")

# ── End parametric loop ───────────────────────────────────────

# ── Parametric summary plot ───────────────────────────────────
if _parametric and len(_summary) > 0:
    labels = [str(v) for v, _ in _summary]
    losses = [l for _, l in _summary]
    fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
    bars = ax.bar(labels, losses, color="#4dabf7")
    ax.set_yscale("log")
    ax.set_xlabel(_param_name, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel("Final Loss", fontsize={config.plot_axis_label_fontsize})
    ax.set_title(f"Parametric Study — Effect of {{_param_name}}")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, loss, _sci_txt(loss), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _summary_path = _os.path.join(_save_dir if _use_save else "/tmp", "parametric_summary.png")
    plt.savefig(_summary_path, dpi={int(config.plot_dpi)}); plt.close()
    import shutil as _shutil2
    _shutil2.copy(_summary_path, "/tmp/solution_plot.png")
    _shutil2.copy(_summary_path, "/tmp/loss_plot.png")
    print(f"\\n=== Parametric Study Complete ===")
    print(f"Summary saved to: {{_summary_path}}")

{_gs_time_adaptive_loop_code(_ic_pretrain_block_ta, line_slice_y, line_slice_z, _safe_num_test, _ta_ic_init, _train_cbs_code_ta, config, plot_custom_expr_converted, _safe_point_distribution)}

{_tm_plot_code}
print("DONE")
"""
    if "_frac(" in (config.pde_expressions or ""):
        script = _frac_data_text(script, config, "geomtime", "pde", "_constraints")
    return _own_param_scratch(script)


_PS_HEADER_OLD = (
    'if os.name == "nt":\n'
    '    _tmp_root = os.path.splitdrive(os.getcwd())[0] + os.sep + "tmp"\n'
    '    os.makedirs(_tmp_root, exist_ok=True)\n')
_PS_HEADER_NEW = _PS_HEADER_OLD + (
    'import tempfile as _tf_ps\n'
    '_PS_TMP = _tf_ps.mkdtemp(prefix="pinnstudio_run_").replace(os.sep, "/")   # this run\'s own scratch folder\n')


def _own_param_scratch(script):
    """The inverse parameter history and plots go to a scratch folder of
    this run's own, not /tmp, where two runs at once (two windows, or the
    test suite in parallel) read each other's parameter values."""
    out = (script.replace('f"/tmp/param', '_PS_TMP + f"/param').replace('"/tmp/param', '_PS_TMP + "/param')
           .replace('f"/tmp/{_in}_param_save.txt"', '_PS_TMP + f"/{_in}_param_save.txt"'))
    if out == script or _PS_HEADER_OLD not in out:
        return script
    return out.replace(_PS_HEADER_OLD, _PS_HEADER_NEW, 1)


# =====================================================================
# generate_clean_script -- a SECOND, independent code generator.
#
# generate_script() above is what Solve actually runs: deliberately
# generic/defensive, built to work for any possible GUI configuration via
# runtime "if" branches over every option this app exposes, with app
# bookkeeping (training_log.txt, model_config.json, per-phase JSON dumps)
# for the Restore & Visualization panel. It must stay exactly as-is.
#
# This function is for "Export as DeepXDE Script" instead: a short,
# tutorial-style, plain DeepXDE/PyTorch + matplotlib script containing
# ONLY the pieces actually configured for THIS problem -- an unticked
# feature (RAR, IC pre-training, Training Callbacks, weight decay,
# input/output transforms, Error Analysis, Inverse) is left out of the
# file entirely, not hidden behind a False branch. Everything that is
# fully known at export time (geometry type, which PDE derivative terms
# are referenced, the BC/IC list, loss weights, the training-phase list)
# is resolved right here in Python and written into the script as plain,
# literal DeepXDE calls instead of runtime dispatch code, so the result
# reads like a hand-written example: something to read, run, and edit.
# =====================================================================

import ast as _clean_ast
import json as _clean_json


def _clean_needed_derivs(oname, oi, n_out, pde_check, is_2d, is_3d, is_steady=False):
    """Host-time equivalent of _pde_standard's runtime substring checks
    above -- returns an ordered list of (varname, rhs_code) for exactly
    the derivative terms this output's PDE expression(s) actually
    reference, resolved now instead of re-checked every time the
    generated script's pde() function runs."""

    def has(suffix):
        return f"d{oname}_{suffix}" in pde_check

    def hess(i, j):
        if n_out == 1:
            return f"dde.grad.hessian(y, x, i={i}, j={j})"
        return f"dde.grad.hessian(y, x, component={oi}, i={i}, j={j})"

    out = []
    if is_3d:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        out.append((f"d{oname}_y", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        out.append((f"d{oname}_z", f"dde.grad.jacobian(y, x, i={oi}, j=2)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=3)"))
        second = [("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2), ("tt", 3, 3),
                  ("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2),
                  ("xt", 0, 3), ("yt", 1, 3), ("zt", 2, 3)]
        have2 = set()
        for nm, i, j in second:
            if has(nm):
                out.append((f"d{oname}_{nm}", hess(i, j)))
                have2.add(nm)
        fourth = [("xxxx", "xx", 0, 0), ("yyyy", "yy", 1, 1), ("zzzz", "zz", 2, 2),
                  ("xxyy", "xx", 1, 1), ("xxzz", "xx", 2, 2), ("yyzz", "yy", 2, 2),
                  ("xxtt", "xx", 3, 3), ("yytt", "yy", 3, 3), ("zztt", "zz", 3, 3)]
        for nm, base, i, j in fourth:
            if has(nm) and base in have2:
                out.append((f"d{oname}_{nm}", f"dde.grad.hessian(d{oname}_{base}, x, i={i}, j={j})"))
    elif is_2d:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        out.append((f"d{oname}_y", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=2)"))
        need_xx = has("xx") or has("xxxx") or has("xxyy") or has("xxtt")
        need_yy = has("yy") or has("yyyy") or has("xxyy") or has("yytt")
        if need_xx:
            out.append((f"d{oname}_xx", hess(0, 0)))
        if need_yy:
            out.append((f"d{oname}_yy", hess(1, 1)))
        if has("xy"):
            out.append((f"d{oname}_xy", hess(0, 1)))
        if has("tt"):
            out.append((f"d{oname}_tt", hess(2, 2)))
        if has("xt"):
            out.append((f"d{oname}_xt", hess(0, 2)))
        if has("yt"):
            out.append((f"d{oname}_yt", hess(1, 2)))
        if has("xxxx") and need_xx:
            out.append((f"d{oname}_xxxx", f"dde.grad.hessian(d{oname}_xx, x, i=0, j=0)"))
        if has("yyyy") and need_yy:
            out.append((f"d{oname}_yyyy", f"dde.grad.hessian(d{oname}_yy, x, i=1, j=1)"))
        if has("xxyy") and need_xx:
            out.append((f"d{oname}_xxyy", f"dde.grad.hessian(d{oname}_xx, x, i=1, j=1)"))
        if has("xxtt") and need_xx:
            out.append((f"d{oname}_xxtt", f"dde.grad.hessian(d{oname}_xx, x, i=2, j=2)"))
        if has("yytt") and need_yy:
            out.append((f"d{oname}_yytt", f"dde.grad.hessian(d{oname}_yy, x, i=2, j=2)"))
    else:
        out.append((f"d{oname}_x", f"dde.grad.jacobian(y, x, i={oi}, j=0)"))
        if not is_steady:
            out.append((f"d{oname}_t", f"dde.grad.jacobian(y, x, i={oi}, j=1)"))
        need_xx = has("xx") or has("xxx") or has("xxxx") or has("xxtt")
        need_tt = has("tt") or has("tttt") or has("xxtt")
        if need_xx:
            out.append((f"d{oname}_xx", hess(0, 0)))
        if need_tt:
            out.append((f"d{oname}_tt", hess(1, 1)))
        if has("xt"):
            out.append((f"d{oname}_xt", hess(0, 1)))
        # Third-order pure x-derivative (KdV's dispersive u_xxx term): one
        # more jacobian pass on top of u_xx, not a hessian (that would give
        # the 4th-order xxxx term below instead).
        if has("xxx") and need_xx:
            out.append((f"d{oname}_xxx", f"dde.grad.jacobian(d{oname}_xx, x, i=0, j=0)"))
        if has("xxxx") and need_xx:
            out.append((f"d{oname}_xxxx", f"dde.grad.hessian(d{oname}_xx, x, i=0, j=0)"))
        if has("xxtt") and need_xx:
            out.append((f"d{oname}_xxtt", f"dde.grad.hessian(d{oname}_xx, x, i=1, j=1)"))
        if has("tttt") and need_tt:
            out.append((f"d{oname}_tttt", f"dde.grad.hessian(d{oname}_tt, x, i=1, j=1)"))
    return out


def _clean_geom_line(config, is_2d, is_3d, tri_verts, poly_verts):
    """One literal geom = dde.geometry.XXX(...) line -- geometry type is
    fixed per-problem, so no runtime dispatch is needed."""
    inner, wrap_needed = _clean_geom_inner(config, is_2d, is_3d, tri_verts, poly_verts)
    if wrap_needed:
        return f"geom = _DTypeSafeGeom({inner})", True
    return f"geom = {inner}", False


def _clean_geom_inner(config, is_2d, is_3d, tri_verts, poly_verts):
    """The bare dde.geometry.XXX(...) constructor call behind
    _clean_geom_line(), plus whether it needs the _DTypeSafeGeom wrapper.
    Split out in cleanup Round 48 so the GUI's Domain Preview builds its
    geometry with exactly the constructor the exported script trains on,
    instead of its own hand-typed per-shape copy."""
    gt = config.geometry_type or "Rectangle"
    wrap_needed = gt in ("Disk", "Ellipse", "Triangle", "Polygon", "Sphere")
    if gt == "Custom":
        inner, wrap_needed = _build_custom_geom_code(config)
    elif gt == "Disk":
        inner = f"dde.geometry.Disk([{config.geom_center_x}, {config.geom_center_y}], {config.geom_radius})"
    elif gt == "Ellipse":
        inner = (f"dde.geometry.Ellipse([{config.geom_center_x}, {config.geom_center_y}], "
                  f"{config.geom_semi_major}, {config.geom_semi_minor}, {config.geom_angle})")
    elif gt == "Triangle":
        inner = f"dde.geometry.Triangle({tri_verts[0]}, {tri_verts[1]}, {tri_verts[2]})"
    elif gt == "Polygon":
        inner = f"dde.geometry.Polygon({poly_verts})"
    elif gt == "Sphere":
        inner = (f"dde.geometry.Sphere([{config.geom_center_x}, {config.geom_center_y}, "
                  f"{config.geom_center_z}], {config.geom_radius})")
    elif is_2d:
        inner = f"dde.geometry.Rectangle([{config.x_min}, {config.y_min}], [{config.x_max}, {config.y_max}])"
    elif is_3d:
        inner = (f"dde.geometry.Cuboid([{config.x_min}, {config.y_min}, {config.z_min}], "
                  f"[{config.x_max}, {config.y_max}, {config.z_max}])")
    else:
        inner = f"dde.geometry.Interval({config.x_min}, {config.x_max})"
    return inner, wrap_needed


def _clean_coord_unpack(indent, is_batch, is_2d, is_3d, is_steady=True, used=None):
    """`x = X[:, 0]` (batch) or `x = X[0]` (single point), plus y/z when
    the problem has them -- shared by every BC location/value helper
    below so an expression like "x" or "np.isclose(x, 0)" the user typed
    means exactly what it looks like. A time-dependent problem also gets
    t (the last column) when the expression uses it -- Round 64; before,
    a value like x*cos(t) was a NameError in the exported script while
    the live script supported it."""
    lines = []
    idx = ":, 0" if is_batch else "0"
    lines.append(f"{indent}x = X[{idx}]")
    if is_2d or is_3d:
        idx = ":, 1" if is_batch else "1"
        lines.append(f"{indent}y = X[{idx}]")
    if is_3d:
        idx = ":, 2" if is_batch else "2"
        lines.append(f"{indent}z = X[{idx}]")
    if not is_steady and used is not None and "t" in used:
        idx = ":, -1" if is_batch else "-1"
        lines.append(f"{indent}t = X[{idx}]")
    return lines


def _bc_operator_def_lines(name, expr, is_steady, dim_str, out_names_list, n_out):
    """def lines for an Operator BC / Point Set Operator BC row's func, in
    DeepXDE's own real signature -- func(inputs, outputs, X), see each
    class's docstring. Unlike every other BC row's helper (whose
    expression is spliced directly into the function body for a plain,
    readable script), this one goes through Training Monitors' own
    _tm_build_dvars + eval() so the same derivative shorthand already
    available in the PDE box/Training Monitors/Custom plot expressions
    (u, du_x, du_xx, ...) also works here, instead of forcing raw
    dde.grad.jacobian/hessian calls -- the tradeoff accepted specifically
    for these two "advanced" BC types (already raw-tensor-level, unlike
    every other BC's plain x/y/u expressions)."""
    return [
        f"def {name}(inputs, outputs, X):",
        f"    _dvars = _tm_build_dvars(inputs, outputs, {n_out}, {out_names_list!r}, "
        f"{is_steady!r}, {dim_str!r}, {expr!r})",
        "    _ns = dict(_dvars)",
        "    _ns['inputs'] = inputs; _ns['outputs'] = outputs; _ns['X'] = X",
        f"    return eval({expr!r}, _ns)",
    ]


def _clean_bc_row_code(i, entry, is_2d, is_3d, n_coord_cols, is_steady=False, n_out=1, out_names="", t_min=0.0,
                       domain=None):
    """Everything needed for one Boundary Conditions panel row: the small
    helper function(s) its location/value expression needs (the
    expression text itself is spliced straight into the function body --
    no eval() indirection, since the whole point is a script someone can
    read and edit -- except Operator BC/Point Set Operator BC's func,
    which needs the derivative-shorthand machinery; see
    _bc_operator_def_lines) plus the one dde.icbc.XxxBC(...) call that
    adds it to `constraints`. Returns (def_lines, append_lines)."""
    _out_names_list = [n.strip() for n in (out_names or "").split(",") if n.strip()] or ["u"]
    _dim_str = "3D" if is_3d else ("2D" if is_2d else "1D")
    btype = entry.get('type', 'dirichlet')
    comp = int(entry.get('component', 0) or 0)
    loc = _np_math_expr((entry.get('location', '') or '').strip() or 'True')
    val_raw = (entry.get('value', '') or '').strip() or '0'
    val = _np_math_expr(val_raw)
    axis = entry.get('axis', 'x')
    # Same defense-in-depth clamp as generate_script() -- see its comment.
    deriv = min(max(int(entry.get('deriv_order', 0) or 0), 0), 1)
    pts_file = entry.get('points_file', '')
    loc2 = _np_math_expr((entry.get('location2', '') or '').strip() or 'True')
    direction = entry.get('direction', 'normal')
    axis_idx_map = {"x": 0, "y": 1, "z": 2}
    # PointSetBC's component as a list of ints (see generate_script()'s
    # matching comment) -- resolved here directly since it's known at
    # codegen time too. PointSetOperatorBC has no `component` at all, so
    # `components`/`comp` are irrelevant there.
    _components_txt = (entry.get('components', '') or '').strip()
    components = [int(c.strip()) for c in _components_txt.split(',') if c.strip()] if _components_txt else None
    n_values = len(components) if components else 1
    # Same batching kwargs as generate_script() -- baked as a literal here
    # since batch_size/shuffle are already known at codegen time.
    batch_size = int(entry.get('batch_size', 0) or 0)
    shuffle = bool(entry.get('shuffle', True))
    batch_kwargs_str = f", batch_size={batch_size}, shuffle={shuffle!r}" if batch_size > 0 else ""

    loc_name = f"_bc{i}_loc"
    loc2_name = f"_bc{i}_loc2"
    val_name = f"_bc{i}_val"
    defs, append = [], []

    def _clean_points_values(i, entry, n_values):
        """Points and target values of a Point Set (Operator) row (Round 71):
        from a file, a typed list or generated; values from the file or a
        formula. Returns (def_lines, append_lines) defining _pts{i}/_pvals{i}."""
        mode = entry.get('points_mode', 'file') or 'file'
        vmode = entry.get('values_mode', 'file') or 'file'
        if mode != 'file':
            vmode = 'formula'
        D, A = [], []
        if mode == 'file':
            A.append(f'_pts{i}, _pvals{i} = _load_bc_points({pts_file!r}, {n_coord_cols}, {n_values if vmode == "file" else 0})')
        elif mode == 'list':
            txt = (entry.get('points_list', '') or '').strip()
            if n_coord_cols == 1:
                txt = txt.replace(",", ";")
            rows = [[float(v) for v in p.replace("(", "").replace(")", "").split(",") if v.strip()]
                    for p in txt.split(";") if p.strip()]
            if not rows or any(len(r) != n_coord_cols for r in rows):
                raise ValueError(f"BC {i + 1}: each typed point needs {n_coord_cols} coordinate(s); "
                                 f"got {entry.get('points_list', '')!r}.")
            A.append(f"_pts{i} = np.array({rows!r}, dtype=float)")
        else:
            npts = int(entry.get('points_count', 10) or 10)
            if entry.get('points_where', 'boundary') == 'domain':
                A.append(f"_pts{i} = geomtime.random_points({npts})")
            else:
                ploc = _np_math_expr((entry.get('location', '') or '').strip() or 'True')
                D += loc_fn(f"_bc{i}_ploc", ploc)
                A.append(f"_cand{i} = geomtime.random_boundary_points({max(20 * npts, 200)})")
                A.append(f"_pts{i} = np.array([p for p in _cand{i} if _bc{i}_ploc(p, True)][:{npts}])")
        if vmode == 'formula':
            forms = [(f.strip() or "0") for f in (entry.get('values_formula', '') or '0').split("|")]
            if len(forms) < n_values:
                raise ValueError(f"BC {i + 1} needs {n_values} value formula(s), separated by '|'.")
            for k, f in enumerate(forms[:n_values]):
                D += val_fn(f"_bc{i}_pval{k}", _np_math_expr(f))
            A.append(f"_pvals{i} = np.hstack([" + ", ".join(f"_bc{i}_pval{k}(_pts{i})" for k in range(n_values)) + "])")
        return D, A

    domain = domain or {}

    def loc_fn(name, expr):
        L = [f"def {name}(X, on_boundary):", "    if not on_boundary:", "        return False"]
        L += _clean_coord_unpack("    ", False, is_2d, is_3d, is_steady, _pde_coord_names([expr]))
        L += _clean_bc_name_lines("    ", expr, domain, True)
        L.append(f"    return bool({expr})")
        return L

    def val_fn(name, expr):
        L = [f"def {name}(X):"]
        L += _clean_coord_unpack("    ", True, is_2d, is_3d, is_steady, _pde_coord_names([expr]))
        L += _clean_bc_name_lines("    ", expr, domain, False)
        L.append(f"    _r = np.asarray({expr}, dtype=float)")
        L.append("    return np.full((len(X), 1), float(_r)) if _r.ndim == 0 else _r.reshape(-1, 1)")
        return L

    if btype == 'neumann':
        defs += loc_fn(loc_name, loc) + val_fn(val_name, val)
        append.append(f"constraints.append(dde.icbc.NeumannBC(geomtime, {val_name}, {loc_name}, component={comp}))")
    elif btype == 'robin':
        rob_name = f"_bc{i}_robin"
        L = [f"def {rob_name}(X, Y):"]
        L += _clean_coord_unpack("    ", True, is_2d, is_3d, is_steady, _pde_coord_names([val]))
        L += _clean_bc_name_lines("    ", val, domain, False)
        _used = _expr_names(val)
        for _oi, _on in enumerate(n.strip() for n in (out_names or "").split(",")):
            if _on and _on in _used and _on != "u":
                L.append(f"    {_on} = Y[:, {_oi}:{_oi + 1}]")
        L.append(f"    u = Y[:, {comp}:{comp}+1]")
        L.append(f"    _r = {val}")
        L.append("    if isinstance(_r, torch.Tensor):")
        L.append("        return _r if _r.dim() == 2 else _r.reshape(-1, 1)")
        L.append("    _arr = np.asarray(_r, dtype=float)")
        L.append("    if _arr.ndim == 0:")
        L.append("        return Y.new_full((len(X), 1), float(_arr))")
        L.append("    return Y.new_tensor(_arr.reshape(-1, 1))")
        defs += L + loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.RobinBC(geomtime, {rob_name}, {loc_name}, component={comp}))")
    elif btype == 'periodic':
        defs += loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.PeriodicBC(geomtime, {axis_idx_map.get(axis, 0)}, "
                       f"{loc_name}, derivative_order={deriv}, component={comp}))")
    elif btype == 'pointset':
        _pd, _pa = _clean_points_values(i, entry, n_values)
        defs += _pd
        append += _pa
        _comp_arg = components if components else comp
        append.append(f"constraints.append(dde.icbc.PointSetBC(_pts{i}, _pvals{i}, component={_comp_arg!r}{batch_kwargs_str}))")
    elif btype == 'pointset_operator':
        op_name = f"_bc{i}_op"
        defs += _bc_operator_def_lines(op_name, val_raw, is_steady, _dim_str, _out_names_list, n_out)
        _pd, _pa = _clean_points_values(i, entry, 1)
        defs += _pd
        append += _pa
        append.append(f"constraints.append(dde.icbc.PointSetOperatorBC(_pts{i}, _pvals{i}, {op_name}{batch_kwargs_str}))")
    elif btype == 'operator':
        op_name = f"_bc{i}_op"
        if entry.get('apply_at') == 'initial' and not is_steady:
            # At the initial time t = t_min (Round 64), not on the boundary --
            # the initial-time training points pass this filter.
            init_loc = [f"def {loc_name}(X, on_boundary):",
                        f"    if not np.isclose(X[-1], {t_min!r}):",
                        "        return False"]
            init_loc += _clean_coord_unpack("    ", False, is_2d, is_3d, is_steady, _pde_coord_names([loc]))
            init_loc += _clean_bc_name_lines("    ", loc, domain, True)
            init_loc.append(f"    return bool({loc})")
            defs += _bc_operator_def_lines(op_name, val_raw, is_steady, _dim_str, _out_names_list, n_out) + init_loc
        else:
            defs += _bc_operator_def_lines(op_name, val_raw, is_steady, _dim_str, _out_names_list, n_out) + loc_fn(loc_name, loc)
        append.append(f"constraints.append(dde.icbc.OperatorBC(geomtime, {op_name}, {loc_name}))")
    elif btype == 'interface2d':
        defs += loc_fn(loc_name, loc) + loc_fn(loc2_name, loc2) + val_fn(val_name, val)
        # Steady-state: geomtime is just geom (no extra time-normal column),
        # so the adapter must be told not to trim boundary_normal()'s
        # result -- see _Interface2DGeomAdapter's own docstring/comment.
        append.append(f"constraints.append(dde.icbc.Interface2DBC(_Interface2DGeomAdapter(geomtime, {is_steady!r}), {val_name}, "
                       f"{loc_name}, {loc2_name}, direction={direction!r}))")
    elif btype == 'dirichlet':
        defs += loc_fn(loc_name, loc) + val_fn(val_name, val)
        append.append(f"constraints.append(dde.icbc.DirichletBC(geomtime, {val_name}, {loc_name}, component={comp}))")
    else:
        # Same loud failure as generate_script()'s matching fallback --
        # raised here at export time (rather than baked silently into the
        # exported script as a wrong Dirichlet BC) so the problem surfaces
        # immediately as an "Export failed" error instead of training
        # quietly with the wrong boundary condition forever after.
        raise ValueError(
            f"BC {i + 1}: unrecognized boundary condition type {btype!r}. "
            "Expected one of: dirichlet, neumann, robin, periodic, pointset, "
            "pointset_operator, operator, interface2d."
        )

    return defs, append


def _clean_legacy_bc_code(config, is_2d, n_out):
    """Codegen-time port of generate_script()'s legacy per-side BC fallback
    (bc_left_types/bc_right_types/bc_bottom_types/bc_top_types and their
    matching values/active/deriv fields), for a config saved before the
    Boundary Conditions panel existed (custom_bc_json empty/absent).

    Without this, generate_clean_script() had no fallback at all for that
    case -- bc_entries came back as an empty list and the exported script
    silently ended up with zero boundary conditions, no warning, nothing.
    Mirrors generate_script()'s runtime `else:` branch, but resolved here
    directly in real Python since every one of these fields is already a
    plain string on `config`, known at codegen time. Returns (defs,
    appends) in the same shape _clean_bc_row_code uses."""

    def _split(value):
        return (value or "").split(",")

    def _get(lst, i, default):
        return lst[i].strip() if i < len(lst) and lst[i].strip() else default

    def _getf(lst, i, default=0.0):
        try:
            return float(_get(lst, i, str(default)))
        except ValueError:
            return default

    bc_left_types   = _split(config.bc_left_types)
    bc_right_types  = _split(config.bc_right_types)
    bc_left_values  = _split(config.bc_left_values)
    bc_right_values = _split(config.bc_right_values)
    bc_left_active  = _split(config.bc_left_active)
    bc_right_active = _split(config.bc_right_active)
    bc_left_deriv   = _split(config.bc_left_deriv)
    bc_right_deriv  = _split(config.bc_right_deriv)
    bc_bottom_types   = _split(config.bc_bottom_types)
    bc_top_types      = _split(config.bc_top_types)
    bc_bottom_values  = _split(config.bc_bottom_values)
    bc_top_values     = _split(config.bc_top_values)
    bc_bottom_active  = _split(config.bc_bottom_active)
    bc_top_active     = _split(config.bc_top_active)
    bc_bottom_deriv   = _split(config.bc_bottom_deriv)
    bc_top_deriv      = _split(config.bc_top_deriv)

    defs, append = [], []

    def _side(prefix, oi, comp, btype, bval, on_bd_name, on_bd_body, deriv_list, deriv_axis, is_second_side):
        defs.append(f"def {on_bd_name}(x, on_boundary):")
        defs.append(f"    return on_boundary and {on_bd_body}")
        if btype == "Dirichlet":
            append.append(f"constraints.append(dde.icbc.DirichletBC(geomtime, lambda x: np.full((len(x), 1), {bval!r}), {on_bd_name}, component={comp}))")
        elif btype == "Neumann":
            append.append(f"constraints.append(dde.icbc.NeumannBC(geomtime, lambda x: np.full((len(x), 1), {bval!r}), {on_bd_name}, component={comp}))")
        elif btype == "Periodic":
            if is_second_side:
                pass  # Periodic BC is handled by the first side only -- DeepXDE enforces both ends together
            else:
                append.append(f"constraints.append(dde.icbc.PeriodicBC(geomtime, {deriv_axis}, {on_bd_name}, derivative_order=0, component={comp}))")
                if _get(deriv_list, oi, "False") == "True":
                    append.append(f"constraints.append(dde.icbc.PeriodicBC(geomtime, {deriv_axis}, {on_bd_name}, derivative_order=1, component={comp}))")

    for oi in range(n_out):
        comp = oi
        blt = _get(bc_left_types, oi, "Dirichlet")
        brt = _get(bc_right_types, oi, "Dirichlet")
        blv = _getf(bc_left_values, oi)
        brv = _getf(bc_right_values, oi)

        if _get(bc_left_active, oi, "False") == "True":
            _side("l", oi, comp, blt, blv, f"_leg{oi}_l_on_bd",
                  f"dde.utils.isclose(x[0], {config.x_min})", bc_left_deriv, 0, False)
        if _get(bc_right_active, oi, "False") == "True":
            _side("r", oi, comp, brt, brv, f"_leg{oi}_r_on_bd",
                  f"dde.utils.isclose(x[0], {config.x_max})", bc_right_deriv, 0, True)

        if is_2d:
            bbt = _get(bc_bottom_types, oi, "Dirichlet")
            btt = _get(bc_top_types, oi, "Dirichlet")
            bbv = _getf(bc_bottom_values, oi)
            btv = _getf(bc_top_values, oi)

            if _get(bc_bottom_active, oi, "False") == "True":
                _side("b", oi, comp, bbt, bbv, f"_leg{oi}_b_on_bd",
                      f"dde.utils.isclose(x[1], {config.y_min})", bc_bottom_deriv, 1, False)
            if _get(bc_top_active, oi, "False") == "True":
                _side("t", oi, comp, btt, btv, f"_leg{oi}_t_on_bd",
                      f"dde.utils.isclose(x[1], {config.y_max})", bc_top_deriv, 1, True)

    return defs, append


def _clean_legacy_bc_weights(config, is_2d, n_out, wm_list, wi):
    """One loss-weight value per constraint ``_clean_legacy_bc_code`` will
    actually append, in the exact same order (per output: left [+ extra
    for a periodic+derivative left side], right, [2D] bottom [+ extra],
    top) -- so ``_clean_loss_weights``'s weight list stays the same length
    as the real constraints list it's multiplied against. Without this,
    a legacy config (bc_entries empty, so len(bc_entries) == 0) got ZERO
    BC weight slots here while the legacy per-side fallback it now gets
    (see _clean_legacy_bc_code) can add several real constraints --
    DeepXDE's own `losses *= torch.as_tensor(self.loss_weights)` then
    crashes with a tensor-size mismatch the moment training starts.

    Mirrors generate_script()'s own legacy weight-building loop (same
    per-side active/periodic/derivative checks), consuming from the same
    shared ``wm_list``/``wi`` cursor used for PDE weights before this and
    IC weights after it. Returns (weights, new_wi)."""

    def _split(value):
        return (value or "").split(",")

    def _get(lst, i, default):
        return lst[i].strip() if i < len(lst) and lst[i].strip() else default

    bc_left_types   = _split(config.bc_left_types)
    bc_right_types  = _split(config.bc_right_types)
    bc_left_active  = _split(config.bc_left_active)
    bc_right_active = _split(config.bc_right_active)
    bc_left_deriv   = _split(config.bc_left_deriv)
    bc_bottom_types   = _split(config.bc_bottom_types)
    bc_top_types      = _split(config.bc_top_types)
    bc_bottom_active  = _split(config.bc_bottom_active)
    bc_top_active     = _split(config.bc_top_active)
    bc_bottom_deriv   = _split(config.bc_bottom_deriv)

    weights = []

    def _next():
        nonlocal wi
        w = wm_list[wi] if wi < len(wm_list) else 1.0
        wi += 1
        return w

    for oi in range(n_out):
        blt = _get(bc_left_types, oi, "Dirichlet")
        brt = _get(bc_right_types, oi, "Dirichlet")

        if _get(bc_left_active, oi, "False") == "True":
            weights.append(_next())
            if blt == "Periodic" and _get(bc_left_deriv, oi, "False") == "True":
                weights.append(_next())
        if _get(bc_right_active, oi, "False") == "True" and brt != "Periodic":
            weights.append(_next())

        if is_2d:
            bbt = _get(bc_bottom_types, oi, "Dirichlet")
            btt = _get(bc_top_types, oi, "Dirichlet")
            if _get(bc_bottom_active, oi, "False") == "True":
                weights.append(_next())
                if bbt == "Periodic" and _get(bc_bottom_deriv, oi, "False") == "True":
                    weights.append(_next())
            if _get(bc_top_active, oi, "False") == "True" and btt != "Periodic":
                weights.append(_next())

    return weights, wi


def _clean_active_ic_outputs(config, n_out, ic_active_list):
    """Which output indices actually get an Initial Condition, in order --
    used both to build the IC constraints and to know how many IC weight
    slots the loss-weight list needs."""
    active = []
    for oi in range(n_out):
        if oi == 0 and config.forward_ic_from_file:
            active.append(oi)
        elif oi < len(ic_active_list) and ic_active_list[oi].strip() == "True":
            active.append(oi)
    return active


def _clean_loss_weights(config, n_out, bc_entries, ic_active_list, obs_weights, is_steady=False,
                          is_2d=False, bc_panel_data_present=True):
    """Fully resolves the loss-weight list at export time -- same slot
    ordering as the running app's runtime reconstruction (PDE per output,
    then one per Boundary Conditions panel row, then one per active IC,
    then one per Inverse observation file), but computed once now instead
    of on every run. Returns (weights, expected_len) -- expected_len is
    how many terms a Training Phase's own weight string must have to be
    used as-is (see _clean_phase_weights below). A steady-state problem
    has no Initial Condition at all, so it never gets an IC weight slot.

    bc_panel_data_present=False (custom_bc_json empty/absent -- a config
    saved before the Boundary Conditions panel existed) switches the BC
    weight slots from "one per custom_bc_json row" (bc_entries is empty
    in that case) to one per constraint the legacy per-side fallback
    actually builds (see _clean_legacy_bc_code/_clean_legacy_bc_weights)
    -- otherwise this list stayed empty while that fallback could still
    add several real BC constraints, and DeepXDE's own
    `losses *= torch.as_tensor(self.loss_weights)` crashed with a
    tensor-size mismatch the moment training started."""
    wm_list = [float(v) for v in (config.loss_weights_multi or "").split(",") if v.strip()]
    wi = 0
    pde_w = []
    for _ in range(_n_equations(config)):  # one per equation
        pde_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
        wi += 1
    if bc_entries or bc_panel_data_present:
        bc_w = []
        for _ in range(len(bc_entries)):
            bc_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
            wi += 1
    else:
        bc_w, wi = _clean_legacy_bc_weights(config, is_2d, n_out, wm_list, wi)
    ic_w = []
    if not is_steady:
        for oi in range(n_out):
            active = (oi == 0 and config.forward_ic_from_file) or (oi < len(ic_active_list) and ic_active_list[oi].strip() == "True")
            if active:
                ic_w.append(wm_list[wi] if wi < len(wm_list) else 1.0)
                wi += 1
            elif wi < len(wm_list):
                wi += 1
    weights = pde_w + bc_w + ic_w + list(obs_weights)
    expected_len = len(pde_w) + len(bc_w) + len(ic_w) + len(obs_weights)
    return weights, expected_len


def _clean_phase_weights(phase, expected_len, fallback):
    try:
        w = [float(x) for x in str(phase.get('weights', '')).split(',') if x.strip()]
    except (TypeError, ValueError):
        w = []
    if len(w) != expected_len:
        w = list(fallback)
    return w


def _clean_sched_phases(config, fallback_weights):
    """The literal, ordered list of training phases this run actually
    uses -- straight from the Training Phases scheduler when it's set up
    (the normal case for anything built in the GUI), or synthesized from
    the older single/second-optimizer fields for a config saved before
    the scheduler existed."""
    try:
        phases = _clean_json.loads(config.scheduler_phases) if config.scheduler_phases else []
    except (ValueError, TypeError):
        phases = []
    if not phases:
        w_str = ",".join(str(w) for w in fallback_weights)
        phases = [{"optimizer": config.optimizer, "iterations": config.iterations,
                   "lr": config.learning_rate, "loss": config.loss_type, "weights": w_str}]
        if config.optimizer2 and config.optimizer2 != "none":
            phases.append({"optimizer": config.optimizer2, "iterations": config.iterations2,
                            "lr": config.learning_rate, "loss": config.loss_type, "weights": w_str})
    return phases


def _clean_phase_train_lines(sp, w, ext_vars_kw, cbs_kw, model_name, data_name, config, indent=""):
    """The literal model.compile()/model.train() call(s) for exactly one
    training phase -- the "literal sequential calls" style: read top to
    bottom as exactly what happens, no phases-list loop at runtime."""
    opt = sp.get('optimizer', 'adam')
    iters = int(sp.get('iterations', 1000) or 0)
    loss = sp.get('loss') or config.loss_type
    L = []
    if opt == 'lbfgs':
        L.append(f"dde.optimizers.set_LBFGS_options(maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol}, "
                  f"gtol={config.lbfgs_gtol}, maxiter={iters}, maxfun={_lbfgs_maxfun(config, iters)}, maxls={config.lbfgs_maxls})")
        L.append(f"{model_name}.compile(\"L-BFGS\", loss={loss!r}, loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(display_every={config.loss_display_every}{cbs_kw})")
    elif opt == 'nncg':
        L.append(f"{model_name}.compile(\"NNCG\", loss={loss!r}, loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(iterations={iters}, display_every={config.loss_display_every}{cbs_kw})")
    else:
        decay = None
        dtype = sp.get('decay_type', 'none')
        if dtype and dtype != 'none':
            p1 = sp.get('decay_p1', 0) or 0
            p2 = sp.get('decay_p2', 0) or 0
            if dtype == 'step':
                decay = ('step', int(p1), float(p2))
            elif dtype == 'cosine':
                decay = ('cosine', int(p1), float(p2))
            elif dtype == 'inverse time':
                decay = ('inverse time', int(p1), float(p2))
            elif dtype == 'exponential':
                decay = ('exponential', float(p1))
        lr = sp.get('lr', config.learning_rate)
        L.append(f"{model_name}.compile({opt!r}, lr={lr}, decay={decay!r}, loss={loss!r}, "
                  f"loss_weights={w}{ext_vars_kw})")
        L.append(f"loss_history, train_state = {model_name}.train(iterations={iters}, display_every={config.loss_display_every}{cbs_kw})")
    return [f"{indent}{ln}" for ln in L]


def _clean_loss_series_lines(config, train_rows_expr, test_rows_expr, steps_expr):
    """Builds the plain `plt.<semilogy|plot>(...)` lines for one "Export as
    DeepXDE Script" loss plot, honoring the Round 28 Loss Plot Settings
    (config.loss_plot_mode/loss_plot_log_y/loss_plot_linewidth) -- the same
    settings generate_script()'s _make_loss_plot applies, but resolved HERE,
    directly into fixed plt calls at codegen time, since generate_clean_script
    deliberately bakes every option into plain literal calls rather than a
    shared runtime helper/dispatch (see its own docstring and the
    plot_title_override precedent above). The surrounding xlabel/title/
    legend/savefig lines at each call site are unaffected and stay as they
    were; only the data-series lines themselves are replaced by this.

    Mirrors _loss_plot_runtime_code's own term-labeling rule: only the
    leading len(output_names) PDE terms get a real name ("PDE (u)"), every
    later term (BC/IC, in whatever order they were added) is labeled
    generically as "Constraint N" -- DeepXDE never tracks per-term Test
    loss, so "individual"/"all" only ever add TRAIN per-term lines.

    Keeps the pre-existing `train_loss = [sum(l) for l in <rows>]` /
    `test_loss = [...]` intermediate-variable assignment (rather than
    summing inline in the plt call) so the default "train_test" mode's
    output is byte-for-byte identical to the pre-Round-28 code -- a
    verified no-op for any config that doesn't touch the new setting,
    and what tests/test_save_and_export_fixes.py's own literal
    "plt.semilogy(..., train_loss" substring check still expects."""
    mode = config.loss_plot_mode
    plotfn = "semilogy" if config.loss_plot_log_y else "plot"
    lw = config.loss_plot_linewidth
    lines = [
        f"train_loss = [sum(l) for l in {train_rows_expr}]",
        f"test_loss = [sum(l) for l in {test_rows_expr}]",
    ]
    if mode in ("train_test", "all"):
        lines.append(f'plt.{plotfn}({steps_expr}, train_loss, label="Train loss", color="#4dabf7", linewidth={lw})')
        lines.append(f'plt.{plotfn}({steps_expr}, test_loss, label="Test loss", color="#ff8787", linestyle="--", linewidth={lw})')
    if mode in ("individual", "all"):
        pde_names = [n.strip() or f"u{i+1}" for i, n in enumerate((config.output_names or "u").split(","))]
        lines.append(f"_loss_pde_names = {pde_names!r}")
        lines.append(f"_loss_term_rows = {train_rows_expr}")
        lines.append("_loss_n_terms = len(_loss_term_rows[0]) if _loss_term_rows else 0")
        lines.append('_loss_term_colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(_loss_n_terms, 1)))')
        lines.append("for _li in range(_loss_n_terms):")
        lines.append('    _loss_lbl = f"PDE ({_loss_pde_names[_li]})" if _li < len(_loss_pde_names) else f"Constraint {_li - len(_loss_pde_names) + 1}"')
        lines.append(f'    plt.{plotfn}({steps_expr}, [r[_li] if _li < len(r) else float("nan") for r in _loss_term_rows], '
                      f'label=_loss_lbl, color=_loss_term_colors[_li], linewidth=max({lw} - 0.5, 0.5), alpha=0.85)')
    return lines


_DTYPE_SAFE_GEOM_CLASS = '''class _DTypeSafeGeom:
    """Some DeepXDE geometries (Disk/Ellipse/Triangle/Polygon/Sphere on
    this installed version) return float64 points even when the network
    is float32 -- cast every sampled point array to the configured float
    type so training doesn't hit a dtype-mismatch error."""
    def __init__(self, geom):
        self._geom = geom
    def __getattr__(self, name):
        return getattr(self._geom, name)
    def random_points(self, n, random="pseudo"):
        return self._geom.random_points(n, random=random).astype(dde.config.real(np))
    def uniform_points(self, n, boundary=True):
        return self._geom.uniform_points(n, boundary=boundary).astype(dde.config.real(np))
    def random_boundary_points(self, n, random="pseudo"):
        return self._geom.random_boundary_points(n, random=random).astype(dde.config.real(np))
    def uniform_boundary_points(self, n):
        return self._geom.uniform_boundary_points(n).astype(dde.config.real(np))'''


def _dtype_safe_geom_class_code():
    """The _DTypeSafeGeom adapter class every generated script that needs it
    embeds -- cleanup Round 56: it used to be hand-typed four times (live
    Solve, export, both restore builders), two with a docstring and two
    without; now one copy."""
    return _DTYPE_SAFE_GEOM_CLASS


_MATH_NS_ROWS = (("sin", "cos", "tan"), ("sinh", "cosh", "tanh"), ("arcsin", "arccos", "arctan"),
                 ("exp", "log", "log10"), ("sqrt", "abs", "ceil", "floor"))


def _math_namespace_code(var, lib, indent=""):
    """`<var> = {"sin": ..., ..., "pi": np.pi}` -- the dict of math functions
    a generated script's eval() namespace offers user expressions, for
    lib "torch" (custom plot fields, evaluated on tensors) or "np" (BC/EA
    expressions, evaluated on arrays). cleanup Round 56: this dict was
    hand-typed seven times across the generators and restore builders."""
    alias = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"} if lib == "torch" else {}
    lines = [f"{var} = {{"]
    for row in _MATH_NS_ROWS:
        lines.append("    " + ", ".join(f'"{f}": {lib}.{alias.get(f, f)}' for f in row) + ",")
    lines.append('    "pi": np.pi,')
    lines.append("}")
    return "\n".join(indent + ln for ln in lines)


def number_settings(src):
    """(significant digits, scientific notation, style) for the numbers
    written in figures (Plot settings -> General -> Numbers), from a config
    or a Plot settings dict."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    try:
        sig = int(g("plot_sig_digits", None) or g("sig_digits", None) or 4)
    except (TypeError, ValueError):
        sig = 4
    sci = g("plot_sci_notation", None) or g("sci_notation", None) or "auto"
    style = g("plot_sci_style", None) or g("sci_style", None) or "times"
    return (max(1, min(10, sig)), sci if sci in ("auto", "always", "never") else "auto",
            style if style in ("times", "e") else "times")


def fmt_num(v, numbers=None, digits=None):
    """The scripts' _fmt_num, for a number written into a label while the
    script is generated (a slice position, t_max): the same text, so every
    number in a figure follows Plot settings -> General -> Numbers."""
    import math
    sig, sci, style = numbers or (4, "auto", "times")
    d = int(digits or sig)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(v) or v == 0:
        return f"{v:g}"
    a = abs(v)
    if sci == "always" or (sci == "auto" and (a < 1e-2 or a >= 1e5)):
        m, e = f"{v:.{max(0, d - 1)}e}".split("e")
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        e = int(e)
        if style == "e":
            return f"{m}e{e:+03d}"
        return f"{m} $\\times$ 10$^{{{e}}}$" if e else m
    txt = f"{v:.{d}g}"
    if "e" in txt:
        txt = f"{v:.{max(0, d - 1 - int(math.floor(math.log10(a))))}f}"
    return txt


def err_cmap(src):
    """The colormap of |Error| panels (Plot settings -> Surface): its own,
    or the solution's when left at "Same as the solution"."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    return (g("plot_err_colormap", None) or g("err_colormap", None)
            or g("plot_colormap", None) or g("colormap", None) or "jet")


def line_colors(src):
    """(PINN colour, Reference colour) for the line figures (Plot settings
    -> Line), from a config or a Plot settings dict."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    return (g("plot_pinn_color", None) or g("pinn_color", None) or "#ff6b6b",
            g("plot_ref_color", None) or g("ref_color", None) or "#4dabf7")


def font_sizes(src):
    """(title, panel title, axis label) font sizes, from a config or a Plot
    settings dict."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    def f(*keys, d):
        for k in keys:
            v = g(k, None)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return d
    return (f("plot_title_fontsize", "title_fontsize", d=13.0),
            f("plot_subplot_title_fontsize", "subplot_title_fontsize", d=10.0),
            f("plot_axis_label_fontsize", "axis_label_fontsize", d=10.0))


def plot_extras(src):
    """(legend font size, fixed colour range or None) for the scripts'
    helpers, from a config or a Plot settings dict."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    try:
        lfs = float(g("plot_legend_fontsize", None) or g("legend_fontsize", None) or 0.0)
    except (TypeError, ValueError):
        lfs = 0.0
    auto = g("plot_auto_range", None)
    if auto is None:
        auto = g("auto_range", True)
    vr = None
    if not auto:
        try:
            lo = float(g("plot_vmin", None) if g("plot_vmin", None) is not None else g("vmin", -1.0))
            hi = float(g("plot_vmax", None) if g("plot_vmax", None) is not None else g("vmax", 1.0))
            if hi > lo:
                vr = (lo, hi)
        except (TypeError, ValueError):
            vr = None
    return (lfs, vr)


def res_3d(src):
    """Points per side of the 3D surfaces (Plot settings -> Surface)."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    try:
        return max(10, int(g("plot_resolution_3d", None) or g("resolution_3d", None) or 40))
    except (TypeError, ValueError):
        return 40


def anim_dpi(src):
    """Resolution of the saved animations (Plot settings -> Animation)."""
    g = (src.get if isinstance(src, dict) else (lambda k, d=None: getattr(src, k, d)))
    try:
        return max(20, int(g("plot_dpi_anim", None) or g("dpi_anim", None) or 150))
    except (TypeError, ValueError):
        return 150


def _figsize_helper_code(mode, w, h, grid=True, ref_style="line", numbers=None, colors=None, fonts=None, dpi_anim=150,
                         extras=None):
    """Builds the literal Python source for a module-level `_plot_figsize(
    default_w, default_h)` helper -- the Round 31 Plot Settings figure-size
    standardization, called at every single-panel plot site in place of a
    bare figsize=(W, H) literal. "Default" mode returns (default_w,
    default_h) unchanged, so every wrapped call site renders byte-identical
    output to before this existed unless the user has actually changed
    Plot Settings' Figure size option.

    This is the ONE shared definition of that helper's source text, used by
    every code path that embeds it in a generated script: generate_script()
    (inline, below), generate_clean_script() (via _clean_figsize_runtime_code()
    right below this function), and main_window.py's three restore/GIF
    script builders (_build_restore_param_script, _build_restore_script,
    _build_restore_script_ta -- imported from here rather than hand-copied,
    cleanup Round 39). Previously this exact block existed as 5 separate
    hand-typed copies; this is now the only one.

    The multi-panel Error Analysis grids use _panel_figsize(), which scales
    each panel the way this scales a single 7 x 5 figure. The domain-sampling
    preview plots keep their own size (sized to the problem's geometry)."""
    return f'''def _plot_figsize(_default_w, _default_h):
    _fs_mode = {mode!r}
    if _fs_mode == "Square":
        _fs_s = max(_default_w, _default_h)
        return (_fs_s, _fs_s)
    elif _fs_mode == "Wide":
        return (_default_h * 1.8, _default_h)
    elif _fs_mode == "Custom":
        return ({w}, {h})
    return (_default_w, _default_h)

_PLOT_GRID = {bool(grid)!r}
_PLOT_REF_DOTS = {ref_style == "dots"!r}
_PLOT_SIG, _PLOT_SCI, _PLOT_SCI_STYLE = {tuple(numbers or (4, "auto", "times"))!r}
_PLOT_PINN_COLOR, _PLOT_REF_COLOR = {tuple(colors or ("#ff6b6b", "#4dabf7"))!r}
_PLOT_FONTS = {tuple(fonts or (13.0, 10.0, 10.0))!r}
_PLOT_DPI_ANIM = {int(dpi_anim)}
_PLOT_LEGEND_FS, _PLOT_VRANGE = {tuple(extras or (0.0, None))!r}

def _lfs(_d):
    # legend font size: the one set in Plot settings, or this figure's own
    return _PLOT_LEGEND_FS if _PLOT_LEGEND_FS else _d

def _vr(_lo, _hi):
    # colour range of the PINN and reference panels: fixed in Plot settings,
    # or the data's own
    return tuple(_PLOT_VRANGE) if _PLOT_VRANGE else (_lo, _hi)

def _vext(_levels=None):
    # contour colours beyond a fixed colour range: arrows on the colorbar
    # (an error panel, given a number of levels, keeps its own range)
    return "both" if (_PLOT_VRANGE and not isinstance(_levels, int)) else "neither"

def _panel_figsize(_pw, _ph, _ncols, _nrows):
    # a figure of _ncols x _nrows panels of _pw x _ph inches, scaled as the
    # Figure size setting scales a single 7 x 5 figure
    _fw, _fh = _plot_figsize(7.0, 5.0)
    return (_pw * _ncols * _fw / 7.0, _ph * _nrows * _fh / 5.0)''' + _PLOT_TEXT_HELPERS


_PLOT_TEXT_HELPERS = """

def _grid(_ax, **_kw):
    # Grid lines on a line-type figure, as set in Plot settings -> General.
    if globals().get("_PLOT_GRID", True):
        _ax.grid(True, **_kw)
    else:
        _ax.grid(False)


def _ref_plot(_ax, _x, _y, **_kw):
    # The reference in a line comparison: a line, or dots (Plot settings -> Line).
    if globals().get("_PLOT_REF_DOTS", False):
        _kw = {k: v for k, v in _kw.items() if k not in ("linestyle", "linewidth", "ls", "lw")}
        _n = len(_x)
        _ax.plot(_x, _y, linestyle="none", marker="o", markersize=4,
                 markevery=max(1, _n // 50), **_kw)
    else:
        _ax.plot(_x, _y, **_kw)

def _line_gaps(_x, *_ys):
    # A line along x masked to the domain: NaN where points were taken out
    # (a hole, the L-shape's missing corner), so the curve stops there
    # instead of a straight segment across the gap.
    import numpy as _np_g
    _x = _np_g.asarray(_x, dtype=float).ravel()
    _ys = tuple(_np_g.asarray(_y, dtype=float).ravel() for _y in _ys)
    _d = _np_g.diff(_x)
    _pos = _d[_d > 0]
    if _x.size < 3 or _pos.size == 0:
        return (_x,) + _ys
    _cut = _np_g.flatnonzero(_d > 1.5 * _pos.min()) + 1
    if _cut.size == 0:
        return (_x,) + _ys
    return (_np_g.insert(_x, _cut, _np_g.nan),) + tuple(_np_g.insert(_y, _cut, _np_g.nan) for _y in _ys)

def _fmt_num(_v, _d=None, _sci=None):
    # A number written in a figure (a time, an error, a parameter value):
    # _d significant digits (Plot settings -> General -> Numbers), and
    # scientific notation below 0.01 or from 100,000 up ("Automatic"),
    # always, or never -- as "1.23 x 10^-4" in mathtext or as "1.23e-04".
    import math as _m_fn
    _d = int(_d or globals().get("_PLOT_SIG", 4))
    try:
        _v = float(_v)
    except (TypeError, ValueError):
        return str(_v)
    if not _m_fn.isfinite(_v) or _v == 0:
        return f"{_v:g}"
    _mode = _sci or globals().get("_PLOT_SCI", "auto")
    _a = abs(_v)
    if _mode == "always" or (_mode == "auto" and (_a < 1e-2 or _a >= 1e5)):
        _m, _e = f"{_v:.{max(0, _d - 1)}e}".split("e")
        if "." in _m:
            _m = _m.rstrip("0").rstrip(".")
        _e = int(_e)
        if globals().get("_PLOT_SCI_STYLE", "times") == "e":
            return f"{_m}e{_e:+03d}"
        return f"{_m} $\\\\times$ 10$^{{{_e}}}$" if _e else _m
    _s = f"{_v:.{_d}g}"
    if "e" in _s:      # %g switched to an exponent: write the digits out
        _s = f"{_v:.{max(0, _d - 1 - int(_m_fn.floor(_m_fn.log10(_a))))}f}"
    return _s

def _final_losses(_model, _lh):
    # (weighted, unweighted) total of the last recorded training losses; the
    # unweighted one divides each term by its loss weight, so runs with
    # different weights (a loss-weight sweep) can be compared.
    _l = [float(_v) for _v in _lh.loss_train[-1]]
    _w = getattr(_model, "loss_weights", None)
    try:
        _w = [float(_x) for _x in _w] if _w is not None else None
    except TypeError:
        _w = None
    _unw = (sum(_li / _wi for _li, _wi in zip(_l, _w) if _wi) if _w and len(_w) == len(_l) else sum(_l))
    return sum(_l), _unw

def _sci_txt(_v, _d=None):
    # An error value for a plot title: with a power of ten ("4.79 x 10^-3")
    # when notation is Automatic, as errors are usually written.
    _mode = globals().get("_PLOT_SCI", "auto")
    return _fmt_num(_v, _d, "always" if _mode == "auto" else _mode)

def _l2_txt(_v, _d=None):
    # The relative L2 error as it is written in papers: italic L, subscript 2.
    return "$L_2$ = " + _sci_txt(_v, _d)

def _num_txt(_v, _d=None):
    # A parameter value for a plot label: two digits more than other numbers
    # (6 by default), so a value close to the true one shows how close.
    return _fmt_num(_v, (_d or globals().get("_PLOT_SIG", 4) + 2))

# Axis and colorbar numbers: scientific notation as set in Plot settings ->
# General -> Numbers (by default below 1e-2 or from 1e5 up, with one
# "x 10^n" factor in mathtext) instead of long decimals such as 0.000225.
import matplotlib as _mpl_sci
_mpl_sci.rcParams["axes.formatter.use_mathtext"] = globals().get("_PLOT_SCI_STYLE", "times") == "times"
_mpl_sci.rcParams["axes.formatter.limits"] = {"always": (0, 0), "never": (-1000, 1000)}.get(
    globals().get("_PLOT_SCI", "auto"), (-2, 5))
# Titles and labels that don't set a size follow Plot settings -> General
# (figure title, panel title, axis label font sizes).
_fs_t, _fs_p, _fs_a = globals().get("_PLOT_FONTS", (13.0, 10.0, 10.0))
_mpl_sci.rcParams["figure.titlesize"] = _fs_t
_mpl_sci.rcParams["axes.titlesize"] = _fs_p
_mpl_sci.rcParams["axes.labelsize"] = _fs_a"""


PARAM_STYLE_DEFAULTS = {
    "param_plot_color": "#e03131", "param_plot_linewidth": 3.0, "param_plot_markers": False,
    "param_plot_show_true": True, "param_true_color": "#212529", "param_true_style": "--",
    "param_plot_show_final": True, "param_final_color": "#868e96", "param_final_style": ":",
    "param_anim_color": "#e03131", "param_anim_linewidth": 3.0, "param_anim_frames": 60,
    "param_plot_figures": "combined", "param_anim_figures": "combined",
    "param_plot_draw": "line", "param_plot_marker": "o", "param_plot_marker_size": 4.0,
    "param_anim_draw": "line", "param_anim_head": True,
}


def param_style_of(cfg):
    """The parameter convergence plot and animation style (Plot settings ->
    Line / Animation) from a config object or a dict of its fields."""
    g = (cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d)))

    def v(k):
        x = g(k, None)
        return PARAM_STYLE_DEFAULTS[k] if x is None or x == "" else x
    return {"color": str(v("param_plot_color")), "linewidth": float(v("param_plot_linewidth")),
            "markers": bool(v("param_plot_markers")),
            "show_true": bool(v("param_plot_show_true")), "true_color": str(v("param_true_color")),
            "true_style": str(v("param_true_style")),
            "show_final": bool(v("param_plot_show_final")), "final_color": str(v("param_final_color")),
            "final_style": str(v("param_final_style")),
            "anim_color": str(v("param_anim_color")), "anim_linewidth": float(v("param_anim_linewidth")),
            "anim_frames": max(2, int(v("param_anim_frames"))),
            "plot_separate": v("param_plot_figures") == "separate",
            "anim_separate": v("param_anim_figures") == "separate",
            # the older on/off markers setting, when the newer one is left as it was
            "draw": ("line_markers" if (v("param_plot_draw") == "line" and bool(v("param_plot_markers")))
                     else str(v("param_plot_draw"))),
            "marker": str(v("param_plot_marker")), "marker_size": float(v("param_plot_marker_size")),
            "anim_draw": str(v("param_anim_draw")), "anim_head": bool(v("param_anim_head"))}


def _param_axis_helper_code(axis_label_fontsize, subplot_title_fontsize, style=None):
    """Runtime source for _draw_param_ax(...), the one "draw a single
    inferred parameter's convergence curve on an axis" function, and
    _param_animation(...), the convergence animation of every parameter --
    used by Solve's script, the exported script and Restore, so the three
    draw the parameter plots the same way with the same style (`style`,
    from param_style_of).

    Log scale needs every value > 0; otherwise it falls back to linear and
    says so."""
    style = dict(param_style_of({}), **(style or {}))
    return f'''def _param_label_num(_v):
    # A parameter value for the legend: two digits more than the other
    # numbers in figures (Plot settings -> General -> Numbers; 6 by default).
    if "_num_txt" in globals():
        return _num_txt(_v)
    _f = float(_v)
    if not np.isfinite(_f) or _f == 0 or 1e-2 <= abs(_f) < 1e5:
        return f"{{_f:.6g}}"
    _e = int(np.floor(np.log10(abs(_f))))
    return f"{{_f / 10 ** _e:.5f}} $\\\\times$ 10$^{{{{{{_e}}}}}}$"

_PARAM_STYLE = {style!r}

if "_grid" not in globals():
    def _grid(_ax, **_kw):
        # Grid lines (Plot settings -> General), when the figure helpers are not loaded.
        if globals().get("_PLOT_GRID", True):
            _ax.grid(True, **_kw)
        else:
            _ax.grid(False)

def _param_ref_lines(ax, final_val, true_val):
    if _PARAM_STYLE["show_final"]:
        ax.axhline(y=final_val, color=_PARAM_STYLE["final_color"], linestyle=_PARAM_STYLE["final_style"],
                   linewidth=1.5, label=f"Final = {{_param_label_num(final_val)}}")
    if true_val is not None and _PARAM_STYLE["show_true"]:
        ax.axhline(y=true_val, color=_PARAM_STYLE["true_color"], linestyle=_PARAM_STYLE["true_style"],
                   linewidth=1.5, label=f"True = {{_param_label_num(true_val)}}")

def _draw_param_ax(ax, name, iters, vals, true_val=None, log_scale=False,
                   title_override="", xlabel_override="", ylabel_override=""):
    vals = np.asarray(vals, dtype=float)
    final_val = vals[-1] if len(vals) else float("nan")
    # a log axis as long as the result (and the true value) is positive; values
    # that dipped to zero or below on the way are left out of the curve
    use_log = (bool(log_scale) and len(vals) > 0 and vals[-1] > 0
               and (true_val is None or true_val > 0))
    if log_scale and not use_log:
        print(f"⚠️ {{name}}: log scale requested but the value isn't positive -- using linear scale instead.")
    if use_log and not np.all(vals > 0):
        vals = np.where(vals > 0, vals, np.nan)
    _draw = _PARAM_STYLE.get("draw", "line_markers" if _PARAM_STYLE.get("markers") else "line")
    _kw = dict(color=_PARAM_STYLE["color"], linewidth=_PARAM_STYLE["linewidth"],
               marker=_PARAM_STYLE.get("marker", "o") if _draw != "line" else None,
               markersize=_PARAM_STYLE.get("marker_size", 4.0),
               markevery=max(1, len(iters) // 40),   # about 40 markers, however often values were saved
               linestyle="none" if _draw == "markers" else "-")
    # (a log axis shows the values themselves, so the label is the name)
    if use_log:
        ax.semilogy(iters, vals, **_kw)
    else:
        ax.plot(iters, vals, **_kw)
    ax.set_ylabel(ylabel_override or name, fontsize={axis_label_fontsize})
    _param_ref_lines(ax, final_val, true_val)
    ax.set_xlabel(xlabel_override or "Iteration", fontsize={axis_label_fontsize})
    ax.set_title(title_override or f"Inferred Parameter: {{name}}", fontsize={subplot_title_fontsize})
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _grid(ax, alpha=0.3)
    return final_val

def _param_animation(series, out_path, log_scale=False, fps=10, dpi=100,
                     title_override="", xlabel_override="", ylabel_override="", figsize=(6, 3.2)):
    """series: [(name, iters, vals, true_val or None)] -> one GIF, a panel per
    parameter, each curve growing to its last saved value."""
    import matplotlib.animation as _p_anim
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), squeeze=False)
    n_frames = min(max(len(s[1]) for s in series), _PARAM_STYLE["anim_frames"])
    lines = []
    for i, (name, iters, vals, true_val) in enumerate(series):
        ax = axes[i][0]
        iters = np.asarray(iters, dtype=float); vals = np.asarray(vals, dtype=float)
        _rv = vals if true_val is None else np.append(vals, true_val)
        use_log = (bool(log_scale) and len(vals) > 0 and vals[-1] > 0
                   and (true_val is None or true_val > 0))
        if use_log:
            # values that dipped to zero or below on the way are left out
            vals = np.where(vals > 0, vals, np.nan)
            _rv = _rv[_rv > 0]
        ax.set_ylabel(ylabel_override or name, fontsize={axis_label_fontsize})
        x_hi = iters.max() if iters.max() > iters.min() else iters.min() + 1
        ax.set_xlim(iters.min(), x_hi)
        lo, hi = _rv.min(), _rv.max()
        # The curve starts empty, so the axis can't take its range from the
        # data it draws: set it to the whole curve (and the true value) now.
        # On a log axis the margin is a factor -- an added margin there left
        # the range to the two reference lines alone, and the curve off it.
        if use_log:
            ax.set_yscale("log")
            pad = (hi / lo) ** 0.05 if hi > lo else 1.5
            ax.set_ylim(lo / pad, hi * pad)
        else:
            pad = 0.05 * (abs(hi - lo) if hi != lo else (abs(hi) + 1))
            ax.set_ylim(lo - pad, hi + pad)
        _param_ref_lines(ax, vals[-1], true_val)
        ax.set_xlabel(xlabel_override or "Iteration", fontsize={axis_label_fontsize})
        ax.set_title(title_override or f"Inferred Parameter: {{name}}", fontsize={subplot_title_fontsize})
        if ax.get_legend_handles_labels()[0]:
            # away from where the curve ends (the legend is placed before the
            # curve is drawn, so "best" can't see it)
            _end = vals[np.isfinite(vals)][-1] if np.isfinite(vals).any() else hi
            _mid = (np.sqrt(lo * hi) if use_log else 0.5 * (lo + hi))
            ax.legend(loc="lower right" if _end >= _mid else "upper right")
        _grid(ax, alpha=0.3)
        _adraw = _PARAM_STYLE.get("anim_draw", "line")
        line, = ax.plot([], [], color=_PARAM_STYLE["anim_color"], linewidth=_PARAM_STYLE["anim_linewidth"],
                        marker=_PARAM_STYLE.get("marker", "o") if _adraw != "line" else None,
                        markersize=_PARAM_STYLE.get("marker_size", 4.0),
                        markevery=max(1, len(iters) // 40),
                        linestyle="none" if _adraw == "markers" else "-")
        # a dot on the curve's tip: where the parameter is now
        head = (ax.plot([], [], linestyle="none", marker="o", color=_PARAM_STYLE["anim_color"],
                        markersize=max(6.0, 2.2 * _PARAM_STYLE["anim_linewidth"]),
                        markeredgecolor="white", markeredgewidth=1.0, zorder=5)[0]
                if _PARAM_STYLE.get("anim_head", True) else None)
        lines.append((line, iters, vals, head))
    plt.tight_layout()

    # Each frame moves on by the same number of iterations, not of saved
    # values: the values are saved more often in some phases than others
    # (L-BFGS logs less often than Adam), and stepping through them one
    # share at a time made the curve speed up in the sparser phase.
    _i0 = min(float(np.min(_it)) for _l, _it, _v, _h in lines)
    _i1 = max(float(np.max(_it)) for _l, _it, _v, _h in lines)

    def _update(frame):
        cut = _i0 + (frame + 1) / n_frames * (_i1 - _i0)
        out = []
        for line, iters, vals, head in lines:
            k = max(1, int(np.searchsorted(iters, cut, side="right")))
            line.set_data(iters[:k], vals[:k])
            out.append(line)
            if head is not None:
                _ok = np.flatnonzero(np.isfinite(vals[:k]))
                head.set_data(iters[_ok[-1:]], vals[_ok[-1:]]) if len(_ok) else head.set_data([], [])
                out.append(head)
        return out
    ani = _p_anim.FuncAnimation(fig, _update, frames=n_frames, interval=1000 / max(1, fps))
    ani.save(out_path, writer="pillow", fps=fps, dpi=globals().get("_PLOT_DPI_ANIM", dpi))
    plt.close(fig)
    print(f"Parameter convergence animation saved: {{out_path}}")'''


def _rar_tolerance_check(config, indent):
    """RAR early stop (Round 66): leave the refinement loop once the mean
    |residual| over the candidate points is below config.rar_tolerance --
    "" (no code) when it's 0, so existing scripts are unchanged."""
    tol = float(getattr(config, "rar_tolerance", 0.0) or 0.0)
    if tol <= 0:
        return ""
    return (f"\n{indent}if residuals.mean() < {tol!r}:"
            f"\n{indent}    print(f\"RAR: mean residual {{residuals.mean():.4e}} is below the tolerance {tol!r} -- refinement done.\")"
            f"\n{indent}    break")


def _formula_reference_dir(config):
    import os as _os_ref, tempfile as _tf_ref
    base = (getattr(config, "save_dir", "") or "").strip() or _os_ref.path.join(_tf_ref.gettempdir(), "pinnstudio_run")
    return _os_ref.path.join(base, "reference_from_formula")


def _formula_reference_entries(config):
    """[(t, path, output index, expression)] for "reference from a formula"
    Error Analysis (Round 67) -- one reference file per output with a
    formula and per comparison time (one per output when steady)."""
    if (getattr(config, "ea_ref_mode", "files") or "files") != "formula":
        return []
    import os as _os_ref
    n_out = int(getattr(config, "num_outputs", 1) or 1)
    names = [n.strip() for n in (config.output_names or "u").split(",") if n.strip()]
    while len(names) < n_out:
        names.append(f"u{len(names)}")
    exprs = (getattr(config, "ea_ref_formula", "") or "").split("|")
    if config.steady_state:
        times = [float(config.t_min)]
    else:
        txt = (getattr(config, "ea_ref_times", "") or "").strip()
        if txt:
            times = [float(v) for v in txt.replace(";", ",").split(",") if v.strip()]
        else:
            times = [config.t_min + k * (config.t_max - config.t_min) / 4.0 for k in range(5)]
    d = _formula_reference_dir(config)
    out = []
    for i in range(n_out):
        e = exprs[i].strip() if i < len(exprs) else ""
        if not e:
            continue
        for t in times:
            fname = f"{names[i]}.txt" if config.steady_state else f"{names[i]}_t_{t:.6g}.txt"
            out.append((float(t), _os_ref.path.join(d, fname), i, e))
    return out


def _formula_reference_code(config, geom_ref="geom"):
    """Source that writes the formula reference files listed by
    _formula_reference_entries, in exactly the format Error Analysis reads
    (x[, y[, z]][, t], value), on a grid over the geometry's bounding box
    masked to the geometry -- or "" when not in formula mode. Shared by both
    generators, so the comparison itself goes through the unchanged
    file-based Error Analysis code."""
    entries = _formula_reference_entries(config)
    if not entries:
        return ""
    is_2d, is_3d, steady = config.problem_dim == "2D", config.problem_dim == "3D", bool(config.steady_state)
    n = int(getattr(config, "ea_ref_grid", 0) or 0) or (31 if is_3d else (101 if is_2d else 201))
    rows = [(t, p, _np_math_expr(e)) for (t, p, _i, e) in entries]
    L = ["# ── Reference solution from a formula (Error Analysis) ──────────",
         "def _write_formula_reference():",
         "    import os as _ref_os",
         "    from scipy.special import erf as _ref_erf",
         _math_namespace_code("_ns", "np", "    "),
         "    _ns['np'] = np",
         "    _ns['erf'] = _ref_erf",
         "    _ns['norm_cdf'] = lambda _z: 0.5 * (1.0 + _ref_erf(np.asarray(_z) / np.sqrt(2.0)))",
         f"    _ref_os.makedirs({_formula_reference_dir(config)!r}, exist_ok=True)",
         f"    _g = {geom_ref}",
         "    _lo, _hi = np.asarray(_g.bbox[0], dtype=float), np.asarray(_g.bbox[1], dtype=float)",
         f"    _axes = [np.linspace(_lo[k], _hi[k], {n}) for k in range(len(_lo))]",
         "    _pts = np.stack([a.ravel() for a in np.meshgrid(*_axes, indexing='ij')], axis=1)",
         "    _pts = _pts[_g.inside(_pts)] if _pts.shape[1] > 1 else _pts",
         "    _cols = {'x': _pts[:, 0]}"]
    if is_2d or is_3d:
        L.append("    _cols['y'] = _pts[:, 1]")
    if is_3d:
        L.append("    _cols['z'] = _pts[:, 2]")
    L.append(f"    for _t, _path, _expr in {rows!r}:")
    L.append("        _u = np.broadcast_to(np.asarray(eval(_expr, dict(_ns, t=_t, **_cols)), dtype=float), (len(_pts),))")
    if steady:
        L.append("        np.savetxt(_path, np.column_stack([_pts, _u]))")
    else:
        L.append("        np.savetxt(_path, np.column_stack([_pts, np.full(len(_pts), _t), _u]))")
    L.append(f"    print('Reference from formula: wrote {len(rows)} file(s) to ' + {_formula_reference_dir(config)!r})")
    L.append("_write_formula_reference()")
    return "\n".join(L)


def _anchor_columns(config):
    """[(name, lo, hi)] per network input column -- x[, y[, z]][, t] --
    with the domain range as the default (anchors box)."""
    cols = [("x", config.x_min, config.x_max)]
    if config.problem_dim in ("2D", "3D"):
        cols.append(("y", config.y_min, config.y_max))
    if config.problem_dim == "3D":
        cols.append(("z", config.z_min, config.z_max))
    if not config.steady_state:
        cols.append(("t", config.t_min, config.t_max))
    return cols


def _anchor_join(names, var="_a"):
    """The anchors= text for a list of array names (any may be None at run
    time): "None", the one name, or a vstack of the ones that are set."""
    if not names:
        return "None"
    if len(names) == 1:
        return names[0]
    return f"np.vstack([{var} for {var} in ({', '.join(names)}) if {var} is not None])"


def _anchor_points_code(config):
    """Source defining `_user_anchors` (an (N, n_inputs) array) for the
    extra training points the user asked for (Round 65), or "" when none
    -- shared by both generators. "box": uniformly random points in the
    given per-column ranges (blank = the column's domain range); "file":
    a points file with one column per input."""
    mode = getattr(config, "anchors_mode", "none") or "none"
    cols = _anchor_columns(config)
    if mode == "file" and (getattr(config, "anchors_file", "") or "").strip():
        path = config.anchors_file.strip()
        return "\n".join([
            "# Extra training points (anchors) from a file -- one column per input:",
            f"# {', '.join(c[0] for c in cols)}",
            f"_user_anchors = np.loadtxt({path!r}, delimiter=',' if {path!r}.lower().endswith('.csv') else None, ndmin=2)",
            f"if _user_anchors.shape[1] < {len(cols)}:",
            f"    raise RuntimeError(f'Anchor points file needs {len(cols)} columns ({', '.join(c[0] for c in cols)}), "
            "got {_user_anchors.shape[1]}.')",
            f"_user_anchors = _user_anchors[:, :{len(cols)}]",
        ])
    if mode == "box":
        parts = [p.strip() for p in (getattr(config, "anchors_box", "") or "").split(";")]
        ranges = []
        for i, (name, lo, hi) in enumerate(cols):
            if i < len(parts) and parts[i]:
                a, b = [float(v) for v in parts[i].split(",")[:2]]
                ranges.append((a, b))
            else:
                ranges.append((float(lo), float(hi)))
        n = int(getattr(config, "anchors_count", 1000) or 0)
        lines = [f"# Extra training points (anchors): {n} random points in a box --"]
        lines.append("# " + ", ".join(f"{c[0]} in [{r[0]:g}, {r[1]:g}]" for c, r in zip(cols, ranges)))
        lines.append("_user_anchors = np.hstack([")
        for (a, b) in ranges:
            lines.append(f"    np.random.uniform({a!r}, {b!r}, ({n}, 1)),")
        lines.append("])")
        return "\n".join(lines)
    return ""


def _contour_levels_helper_code():
    """Runtime source for _cf_levels(n, vmin, vmax, *data) -- cleanup Round
    50 (open item from Round 22). matplotlib's contourf IGNORES vmin/vmax
    when choosing its level boundaries if `levels` is a plain integer count:
    the fill colors still go through the vmin/vmax norm, but the bands and
    the colorbar auto-range to each panel's own data. So two panels meant to
    share one color scale (PINN vs Reference) showed different colorbars,
    and a manual color range set in Plot Settings never showed up on the
    colorbar. A few comparison plots already worked around this with a
    hardcoded _cf_levels({config.plot_levels}, vmin, vmax)["levels"] (40 bands, ignoring the configured
    level count); everything now goes through this one function instead.

    Returns contourf keyword arguments: with no range (vmin or vmax None),
    just levels=n, exactly as before; with a range, ~n "nice" band
    boundaries covering [vmin, vmax] -- matplotlib's own MaxNLocator rule,
    the same one contourf uses for an integer count, so colorbar ticks stay
    round numbers -- plus extend="min"/"max"/"both" when any of `data` falls
    outside those boundaries (so out-of-range values are still filled, as
    they were with an integer level count) -- and plain levels=n plus the
    norm range for a degenerate range (vmin >= vmax, e.g. a constant
    field)."""
    return '''def _cf_levels(n, vmin=None, vmax=None, *data):
    if vmin is None or vmax is None:
        return {"levels": n}
    try:
        lo, hi = float(vmin), float(vmax)
    except (TypeError, ValueError):
        return {"levels": n}
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return {"levels": n, "vmin": vmin, "vmax": vmax}
    from matplotlib.ticker import MaxNLocator as _MNL
    lv = _MNL(int(n) + 1).tick_values(lo, hi)
    below = above = False
    for _d in data:
        _d = np.asarray(_d, dtype=float)
        if _d.size and np.isfinite(_d).any():
            below = below or bool(np.nanmin(_d) < lv[0])
            above = above or bool(np.nanmax(_d) > lv[-1])
    ext = "both" if (below and above) else ("min" if below else ("max" if above else "neither"))
    return {"levels": lv, "vmin": lo, "vmax": hi, "extend": ext}'''


def _clean_figsize_runtime_code(config):
    """Thin wrapper around _figsize_helper_code() for generate_clean_script()'s
    own call sites -- kept as its own name/signature since generate_clean_script()
    already calls it this way (see Round 31). See _figsize_helper_code()'s own
    docstring for the full rationale; this just extracts the 3 relevant fields
    from a PINNConfig."""
    return _figsize_helper_code(config.plot_figsize_mode, config.plot_figsize_w, config.plot_figsize_h,
                                getattr(config, "plot_grid", True), getattr(config, "plot_ref_style", "line"),
                                number_settings(config), line_colors(config), font_sizes(config), anim_dpi(config), plot_extras(config))


def _restore_line_plot_code(x_min, x_max, resolution, t_min, t_max, n_steps, colormap,
                             line_slice_y, line_slice_z, linewidth, xlabel, ylabel,
                             title, subplot_title_fontsize, axis_label_fontsize, save_dir, dpi,
                             is_time_adaptive):
    """Builds the "Line (time steps)" static-plot branch shared between
    main_window.py's single-model restore (_build_restore_script) and its
    Time-Adaptive counterpart (_build_restore_script_ta) -- cleanup Round
    42 (audit §2e, the first of the "six mini script generators" to get a
    shared helper). Before this round these were two independently
    hand-typed ~50-line copies, identical except for (a) whether a
    steady-state restore is even possible at all (only the single-model
    path supports it -- Time-Adaptive training always has a real time
    axis, so its own copy never had the extra is_steady branching) and
    (b) which model object .predict() is called on for a given time value
    -- a fixed `model` for the single-model path, vs. a per-time-value
    restored step model (`_ta_model_for_t(tv)`) for the Time-Adaptive
    path. Verified (before this function existed) to render byte-for-byte
    identical to both of the original hand-typed copies across a 200-trial
    randomized fuzz check, not just a couple of hand-picked examples.

    is_time_adaptive selects between the two -- everything else needed to
    tell them apart (the geometry-mask comment wording, whether a model
    needs resolving per time value, the extra predict() argument, the
    is_steady defensive branches, and the plot legend label) follows from
    it alone; no second flag is needed."""
    if is_time_adaptive:
        geom_mask_comment = (
            "# Masked to the real geometry with geom.inside(...) -- same convention as\n"
            "# generate_script()'s own (already-fixed) Time-Adaptive-aware Error\n"
            "# Analysis Line Comparison, and the single-model restore path's own\n"
            "# Line (time steps) branch. The mask only depends on (x[, y, z]), not t,\n"
            "# so it's computed once before the time loop. 1D has no non-Interval\n"
            "# geometry, so no masking is needed there.\n"
        )
        pre_xt_lines = "        _ta_m = _ta_model_for_t(tv)\n"
        predict_model_arg = ", _ta_m"
        line_label_stmt = ""
        line_label_arg = 'label=f"t={_fmt_num(tv)}"'
        xt_branch = (
            "        if is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])\n"
            "        elif is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])\n"
            "        else:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])\n"
        )
    else:
        geom_mask_comment = (
            "# Masked to the real geometry with geom.inside(...) -- same convention\n"
            "# the Surface branch above already uses. The mask only depends on\n"
            "# (x[, y, z]), not t, so it's computed once before the time loop. 1D has\n"
            "# no non-Interval geometry, so no masking is needed there.\n"
        )
        pre_xt_lines = (
            "        # Steady-state has no time axis -- \"time steps\" don't exist, so every\n"
            "        # iteration predicts the same single steady solution (this viz type is\n"
            "        # hidden from the Restore panel's dropdown for a steady-state config;\n"
            "        # this is just a defensive fallback so it can't crash if ever reached).\n"
        )
        predict_model_arg = ""
        line_label_stmt = '\n        _line_label = "steady-state" if is_steady else f"t={tv:.3f}"'
        line_label_arg = "label=_line_label"
        xt_branch = (
            "        if is_steady and is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid)])\n"
            "        elif is_steady and is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid)])\n"
            "        elif is_steady:\n"
            "            xt = x_vals.reshape(-1, 1)\n"
            "        elif is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])\n"
            "        elif is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])\n"
            "        else:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])\n"
        )

    return f"""x_vals = np.linspace({x_min}, {x_max}, {resolution})
t_steps_vals = np.linspace({t_min}, {t_max}, {n_steps})
y_mid = {line_slice_y}
z_mid = {line_slice_z}
{geom_mask_comment}if is_3d:
    _line_in = geom.inside(np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid)]))
    x_vals = x_vals[_line_in]
elif is_2d:
    _line_in = geom.inside(np.column_stack([x_vals, np.full_like(x_vals, y_mid)]))
    x_vals = x_vals[_line_in]
fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
if len(x_vals) < 2:
    ax.text(0.5, 0.5, "Line outside the domain at this slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    colors = plt.get_cmap("{colormap}")(np.linspace(0, 1, {n_steps}))
    for i, tv in enumerate(t_steps_vals):
{pre_xt_lines}{xt_branch}        u_line = _extract_plot_field(xt{predict_model_arg}).flatten(){line_label_stmt}
        ax.plot(*_line_gaps(x_vals, u_line), color=colors[i], linewidth={linewidth}, {line_label_arg})
    ax.set_xlabel({xlabel!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel!r}, fontsize={axis_label_fontsize})
    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
ax.set_title({title!r}, fontsize={subplot_title_fontsize})
plt.tight_layout()
out_path = os.path.join({save_dir!r}, "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Line plot saved to: {{out_path}}")"""


def _restore_line_animation_code(x_min, x_max, t_min, t_max, n_steps, line_slice_y, line_slice_z,
                                  linewidth, xlabel, ylabel, title_stmt, tt_x, tt_y, tt_ha,
                                  axis_label_fontsize, fps, save_dir, dpi, is_time_adaptive):
    """Builds the "Animation Line (GIF)" branch shared between
    main_window.py's single-model restore (_build_restore_script) and its
    Time-Adaptive counterpart (_build_restore_script_ta) -- cleanup Round
    42 (audit §2e), the Animation-Line sibling of _restore_line_plot_code
    right above (see its docstring for the full rationale; is_time_adaptive
    means the same thing here). Also verified byte-for-byte identical to
    both original hand-typed copies across a 200-trial randomized fuzz
    check before this function replaced them."""
    if is_time_adaptive:
        pre_xt_lines = "        _ta_m = _ta_model_for_t(tv)\n"
        predict_model_arg = ", _ta_m"
        xt_branch = (
            "        if is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])\n"
            "        elif is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])\n"
            "        else:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])\n"
        )
    else:
        pre_xt_lines = (
            "        # Defensive fallback for steady-state (see the matching comment in the\n"
            "        # \"Line (time steps)\" branch) -- this viz type is hidden from the\n"
            "        # dropdown for a steady-state config, so every \"frame\" below predicts\n"
            "        # the same single steady solution rather than crashing.\n"
        )
        predict_model_arg = ""
        xt_branch = (
            "        if is_steady and is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid)])\n"
            "        elif is_steady and is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid)])\n"
            "        elif is_steady:\n"
            "            xt = x_vals.reshape(-1, 1)\n"
            "        elif is_3d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])\n"
            "        elif is_2d:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])\n"
            "        else:\n"
            "            xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])\n"
        )

    return f"""import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
y_mid = {line_slice_y}
z_mid = {line_slice_z}
# Masked to the real geometry with geom.inside(...) -- same convention as
# the "Line (time steps)" branch above. If the slice leaves fewer than 2
# in-domain points, there's nothing to animate -- save a static fallback
# note as restored_plot.png instead (the path _on_restore's result
# display already falls back to when no .gif was produced) rather than
# crashing or silently animating a line outside the real domain.
if is_3d:
    _line_in = geom.inside(np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid)]))
    x_vals = x_vals[_line_in]
elif is_2d:
    _line_in = geom.inside(np.column_stack([x_vals, np.full_like(x_vals, y_mid)]))
    x_vals = x_vals[_line_in]
if len(x_vals) < 2:
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 4))
    ax.text(0.5, 0.5, "Line outside the domain at this slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    out_path = os.path.join({save_dir!r}, "restored_plot.png")
    plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
    print(f"⚠️ Line outside the domain at this slice -- no animation produced; static note saved to: {{out_path}}")
else:
    all_u = []
    for tv in t_frames:
{pre_xt_lines}{xt_branch}        all_u.append(_extract_plot_field(xt{predict_model_arg}).flatten())
    u_min = min(u.min() for u in all_u)
    u_max = max(u.max() for u in all_u)
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 4))
    ax.set_xlim({x_min}, {x_max})
    ax.set_ylim(u_min - 0.05*abs(u_min), u_max + 0.05*abs(u_max))
    ax.set_xlabel({xlabel!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel!r}, fontsize={axis_label_fontsize})
    {title_stmt}
    line, = ax.plot([], [], color=_PLOT_PINN_COLOR, linewidth={linewidth})
    time_txt = ax.text({tt_x}, {tt_y}, '', transform=ax.transAxes, color='black', ha={tt_ha!r}, fontsize=11)
    _grid(ax, alpha=0.2)
    def init():
        line.set_data([], []); time_txt.set_text(''); return line, time_txt
    def update(i):
        line.set_data(*_line_gaps(x_vals, all_u[i]))
        time_txt.set_text(f"t = {{_fmt_num(t_frames[i])}}")
        return line, time_txt
    ani = _anim.FuncAnimation(fig, update, init_func=init, frames={n_steps}, interval=100, blit=True)
    out_path = os.path.join({save_dir!r}, "restored_animation.gif")
    ani.save(out_path, writer='pillow', fps={fps}, dpi=_PLOT_DPI_ANIM)
    plt.close()
    print(f"Animation saved to: {{out_path}}")"""


def _restore_anim_surface_code(x_min, x_max, y_min, y_max, t_min, t_max, n_steps, colormap,
                                show_colorbar, levels, swap_xt, axis_label_fontsize, fps, save_dir,
                                xlabel_animsurf3d, ylabel_animsurf3d, animsurf_title_line_3d,
                                xlabel_animsurf_else, ylabel_animsurf_else, animsurf_title_line_else,
                                is_time_adaptive, resolution=80, res3d=40):
    """Builds the "Animation Surface (GIF)" branch shared between
    main_window.py's single-model restore (_build_restore_script) and its
    Time-Adaptive counterpart (_build_restore_script_ta) -- cleanup Round
    43 (audit §2e, sub-round 2). Unlike the Line/Animation-Line pair
    (Round 42), the "Surface" branch itself turned out to be only ~47%
    textually similar between the two builders (real structural
    differences under Time-Adaptive routing -- left for a separate
    investigation, see that round's notes) -- but THIS branch,
    "Animation Surface (GIF)", measured ~82% similar, concentrated in
    large matching blocks, and is in fact a clean is_time_adaptive-only
    difference exactly like Round 42's pair: which bbox source feeds the
    3D face grid (geom.bbox vs. a geomtime helper's own bbox), whether a
    per-frame Time-Adaptive model needs resolving before each predict
    call (and the extra model argument that goes with it), the
    is_steady-only-possible-on-the-single-model-path defensive branches,
    and (1D case only) whether the frame is predicted as one batched
    res-independent grid (single model, cheap to over-predict) or as one
    row's worth of points per frame then tiled (Time-Adaptive, where
    re-resolving + re-predicting 80x more points than needed per frame
    would be wasteful). All of it follows from one is_time_adaptive flag.
    Verified byte-for-byte identical to both original hand-typed copies
    across a 300-trial randomized fuzz check before this function
    replaced them. cleanup Round 46 then added geometry masking (3D:
    alpha-0 out-of-domain face quads; 2D: NaN out-of-domain grid points),
    which neither original copy had -- the same geom.inside(...)
    convention the static Surface plot (_restore_surface_code) uses."""
    res = max(10, int(resolution))     # Plot settings -> Surface -> Grid resolution
    res3 = max(10, int(res3d))         # Plot settings -> Surface -> 3D grid resolution
    if is_time_adaptive:
        comment_3d = ""
        bbox3a_expr = "_ta_build_geomtime(t_frames[0], t_frames[0]).geometry.bbox"
        loop_3d = (
            "    for tv in t_frames:\n"
            "        _ta_m = _ta_model_for_t(tv)\n"
            "        _frame_faces = []\n"
            "        for _fX3a, _fY3a, _fZ3a in _faces3a:\n"
            "            _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel(), np.full(_fX3a.size, tv)])\n"
            "            _frame_faces.append(_extract_plot_field(_fpts3a, _ta_m).reshape(_fX3a.shape))\n"
            "        all_frames.append(_frame_faces)\n"
        )
        loop_2d = (
            "        for tv in t_frames:\n"
            "            _ta_m = _ta_model_for_t(tv)\n"
            "            XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])\n"
            "            pred = _extract_plot_field(XYT, _ta_m).reshape({res}, {res})\n"
            "            pred = np.where(_inside2da, pred, np.nan)\n"
            "            all_frames.append((Xg, Yg, pred))\n"
        ).format(res=res)
        comment_else = ""
        loop_else = (
            "        for tv in t_frames:\n"
            "            _ta_m = _ta_model_for_t(tv)\n"
            "            XT = np.column_stack([x_anim, np.full_like(x_anim, tv)])\n"
            "            pred_row = _extract_plot_field(XT, _ta_m).flatten()\n"
            "            pred = np.tile(pred_row, (80, 1))\n"
            "            if {swap_xt}:\n"
            "                all_frames.append((T_anim, X_anim, pred))\n"
            "            else:\n"
            "                all_frames.append((X_anim, T_anim, pred))\n"
        ).format(swap_xt=swap_xt, res=res)
    else:
        comment_3d = (
            "    # Genuine smooth 3D surface animated over time: same 6-flat-face\n"
            "    # exact-prediction approach as the static 3D Surface option above,\n"
            "    # repeated once per frame (the face grids are fixed across frames,\n"
            "    # only the predicted color values change per t) and rendered with\n"
            "    # plot_surface's per-quad facecolors for a continuous-looking colored\n"
            "    # surface instead of a scatter of discrete points.\n"
        )
        bbox3a_expr = "geom.bbox"
        loop_3d = (
            "    for tv in t_frames:\n"
            "        _frame_faces = []\n"
            "        for _fX3a, _fY3a, _fZ3a in _faces3a:\n"
            "            # Defensive fallback for steady-state (see the matching comment\n"
            "            # in \"Line (time steps)\") -- this viz type is hidden from the\n"
            "            # dropdown for a steady-state config, so every frame below\n"
            "            # predicts the same single steady solution rather than crashing.\n"
            "            if is_steady:\n"
            "                _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel()])\n"
            "            else:\n"
            "                _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel(), np.full(_fX3a.size, tv)])\n"
            "            _frame_faces.append(_extract_plot_field(_fpts3a).reshape(_fX3a.shape))\n"
            "        all_frames.append(_frame_faces)\n"
        )
        loop_2d = (
            "        for tv in t_frames:\n"
            "            # Defensive fallback for steady-state, same as the 3D branch\n"
            "            # above -- hidden from the dropdown for a steady-state config.\n"
            "            if is_steady:\n"
            "                XYT = np.column_stack([Xg.ravel(), Yg.ravel()])\n"
            "            else:\n"
            "                XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])\n"
            "            pred = _extract_plot_field(XYT).reshape({res}, {res})\n"
            "            pred = np.where(_inside2da, pred, np.nan)\n"
            "            all_frames.append((Xg, Yg, pred))\n"
        ).format(res=res)
        comment_else = (
            "        # Same no-reshape-of-pred swap as the static Surface option above\n"
            "        # -- storing T_anim/X_anim in swapped order per frame is enough,\n"
            "        # since contourf reads each array's own coordinate values rather\n"
            "        # than assuming a fixed axis order.\n"
        )
        loop_else = (
            "        for tv in t_frames:\n"
            "            XT = np.vstack([X_anim.ravel(), np.full(X_anim.size, tv)]).T\n"
            "            pred = _extract_plot_field(XT).reshape({res}, {res})\n"
            "            if {swap_xt}:\n"
            "                all_frames.append((T_anim, X_anim, pred))\n"
            "            else:\n"
            "                all_frames.append((X_anim, T_anim, pred))\n"
        ).format(swap_xt=swap_xt, res=res)

    return f"""import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
x_anim = np.linspace({x_min}, {x_max}, {res})
all_frames = []
if is_3d:
{comment_3d}    _res3a = {res3}
    _bbox3a = np.asarray({bbox3a_expr})
    _cx0a, _cy0a, _cz0a = _bbox3a[0]; _cx1a, _cy1a, _cz1a = _bbox3a[1]
    _xg3a = np.linspace(_cx0a, _cx1a, _res3a)
    _yg3a = np.linspace(_cy0a, _cy1a, _res3a)
    _zg3a = np.linspace(_cz0a, _cz1a, _res3a)
    _Xxy3a, _Yxy3a = np.meshgrid(_xg3a, _yg3a)
    _Xxz3a, _Zxz3a = np.meshgrid(_xg3a, _zg3a)
    _Yyz3a, _Zyz3a = np.meshgrid(_yg3a, _zg3a)
    _faces3a = [
        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz0a)),
        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz1a)),
        (_Xxz3a, np.full_like(_Xxz3a, _cy0a), _Zxz3a),
        (_Xxz3a, np.full_like(_Xxz3a, _cy1a), _Zxz3a),
        (np.full_like(_Yyz3a, _cx0a), _Yyz3a, _Zyz3a),
        (np.full_like(_Yyz3a, _cx1a), _Yyz3a, _Zyz3a),
    ]
    # cleanup Round 46: mask every bounding-box face to the real geometry
    # with geom.inside(...), same convention as the static 3D Surface plot
    # -- a Sphere/Custom-3D model would otherwise animate its full
    # bounding-box faces. Spatial-only, so computed once for every frame.
    _face_inside3a = []
    for _fX3a, _fY3a, _fZ3a in _faces3a:
        _fspatial3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel()])
        _face_inside3a.append(np.asarray(geom.inside(_fspatial3a)).reshape(_fX3a.shape))
{loop_3d}    # Color range from in-domain values only (every value if no face
    # touches the domain at all, e.g. a Sphere's tangent-only bounding box).
    _inside_vals3a = [_f[_m] for _frame in all_frames for _f, _m in zip(_frame, _face_inside3a) if _m.any()]
    if _inside_vals3a:
        v_min = min(_v.min() for _v in _inside_vals3a)
        v_max = max(_v.max() for _v in _inside_vals3a)
    else:
        v_min = min(_f.min() for _frame in all_frames for _f in _frame)
        v_max = max(_f.max() for _frame in all_frames for _f in _frame)
    fig = plt.figure(figsize=_plot_figsize(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    _norm3a = plt.Normalize(vmin=v_min, vmax=v_max)
    _cmap_obj3a = plt.get_cmap("{colormap}")
    _sm3a = plt.cm.ScalarMappable(cmap=_cmap_obj3a, norm=_norm3a)
    if {show_colorbar}: fig.colorbar(_sm3a, ax=ax, shrink=0.6, pad=0.12)
    def update(i):
        ax.cla()
        for _fi3a, (_fX3a, _fY3a, _fZ3a) in enumerate(_faces3a):
            _fc3a = np.array(_cmap_obj3a(_norm3a(all_frames[i][_fi3a])))
            # Outside-the-domain quads get alpha=0, same as the static plot.
            _fc3a[..., 3] = np.where(_face_inside3a[_fi3a], 1.0, 0.0)
            ax.plot_surface(_fX3a, _fY3a, _fZ3a, facecolors=_fc3a,
                             rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel({xlabel_animsurf3d!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel_animsurf3d!r}, fontsize={axis_label_fontsize}); ax.set_zlabel("z", fontsize={axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (ax.xaxis, ax.yaxis, ax.zaxis)]
        {animsurf_title_line_3d}
        try:
            ax.set_box_aspect((_cx1a - _cx0a, _cy1a - _cy0a, _cz1a - _cz0a))
        except Exception:
            pass
    ani = _anim.FuncAnimation(fig, update, frames={n_steps}, interval=150)
    out_path = os.path.join({save_dir!r}, "restored_animation.gif")
    ani.save(out_path, writer='pillow', fps={fps}, dpi=_PLOT_DPI_ANIM)
    plt.close()
    print(f"3D surface animation saved to: {{out_path}}")
else:
    if is_2d:
        y_anim = np.linspace({y_min}, {y_max}, {res})
        Xg, Yg = np.meshgrid(x_anim, y_anim)
        # cleanup Round 46: NaN out grid points outside the real geometry
        # (contourf leaves them unfilled), same as the static 2D Surface plot.
        _inside2da = np.asarray(geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()]))).reshape({res}, {res})
{loop_2d}    else:
{comment_else}        t_anim = np.linspace({t_min}, {t_max}, {res})
        X_anim, T_anim = np.meshgrid(x_anim, t_anim)
{loop_else}    v_min = min(np.nanmin(f[2]) for f in all_frames)
    v_max = max(np.nanmax(f[2]) for f in all_frames)
    fig, ax = plt.subplots(figsize=_plot_figsize(8, 6))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    _div = make_axes_locatable(ax)
    _cax = _div.append_axes("right", size="5%", pad=0.1)
    _sm = plt.cm.ScalarMappable(cmap="{colormap}", norm=plt.Normalize(vmin=v_min, vmax=v_max))
    if {show_colorbar}: fig.colorbar(_sm, cax=_cax)
    def update(i):
        ax.cla()
        Xp, Yp, Zp = all_frames[i]
        ax.contourf(Xp, Yp, Zp, cmap="{colormap}", **_cf_levels({levels}, v_min, v_max, Zp))
        ax.set_xlabel({xlabel_animsurf_else!r}, fontsize={axis_label_fontsize})
        ax.set_ylabel({ylabel_animsurf_else!r}, fontsize={axis_label_fontsize})
        {animsurf_title_line_else}
    ani = _anim.FuncAnimation(fig, update, frames={n_steps}, interval=150)
    out_path = os.path.join({save_dir!r}, "restored_animation.gif")
    ani.save(out_path, writer='pillow', fps={fps}, dpi=_PLOT_DPI_ANIM)
    plt.close()
    print(f"Surface animation saved to: {{out_path}}")"""


def _restore_surface_code(resolution, x_min, x_max, y_min, y_max, t_min, t_max, n_2d_snapshots,
                          colormap, show_colorbar, levels, swap_xt, auto_range, vmin_val, vmax_val,
                          linewidth, title_fontsize, subplot_title_fontsize, axis_label_fontsize,
                          save_dir, dpi, xlabel_3d, ylabel_3d, title_3d, xlabel_2d, ylabel_2d,
                          title_2d, xlabel_1d, ylabel_1d, title_1d, xlabel_1d_steady,
                          ylabel_1d_steady, title_1d_steady, is_time_adaptive, res3d=40):
    """Builds the static "Surface" branch shared between main_window.py's
    single-model restore (_build_restore_script) and its Time-Adaptive
    counterpart (_build_restore_script_ta) -- cleanup Round 45 (audit §2e,
    sub-round 3). Round 43 measured this pair at only ~47% similar and left
    it alone, partly because the Time-Adaptive copy never masked to the real
    geometry; Round 44 fixed that, and with masking now in parity the two
    copies differ only in things that follow from one is_time_adaptive
    flag, same as Rounds 42/43's pairs:
      - which bbox source feeds the 3D face grid (geom.bbox vs. the
        Time-Adaptive script's _ta_build_geomtime(...).geometry.bbox --
        the same box either way);
      - whether each snapshot panel resolves its own step model first
        (_ta_model_for_t) and passes it to _extract_plot_field;
      - the is_steady branches, only reachable on the single-model path
        (a Time-Adaptive run is never steady-state, and its script never
        even defines is_steady): 1 panel instead of n_2d_snapshots, no
        time column, the steady per-panel title instead of "t = ...", no
        suptitle, and the steady-1D u(x) line plot;
      - 1D: one batched (x, t) grid predict (single model) vs. one row per
        t, each through whichever step model covers that t (Time-Adaptive);
      - comment wording, kept per side so generated scripts stay
        byte-for-byte what they were before this extraction.
    xlabel_1d_steady/ylabel_1d_steady/title_1d_steady and linewidth are only
    used on the single-model path (the steady-1D line plot); pass None for
    them from the Time-Adaptive builder. Verified byte-for-byte identical to
    both original hand-typed copies across a randomized fuzz check plus a
    full generated-script diff before this function replaced them."""
    cf_range = "None, None" if auto_range else f"{vmin_val}, {vmax_val}"
    if is_time_adaptive:
        suptitle_note = (
            "# Round 31: only the 3D/2D branches below add a fig.suptitle() above\n"
            "# their own per-panel subplot titles; the final 1D branch is single-panel\n"
            "# with no suptitle -- see the matching comment in _build_restore_script.\n"
        )
        comment_3d = (
            "    # Several evenly-spaced time snapshots, each its own 3D subplot side\n"
            "    # by side -- same convention as _build_restore_script's own 3D Surface\n"
            "    # branch (see its comment). Which step's restored model applies can\n"
            "    # differ snapshot to snapshot (see _ta_model_for_t/_ta_pick_step\n"
            "    # above), so each panel resolves its own model, unlike the rest of\n"
            "    # this plot's geometry (x_min/x_max/... are a whole-run choice, not\n"
            "    # per-step -- see this function's own docstring -- so the bounding box\n"
            "    # itself only needs computing once, not per panel).\n"
        )
        bbox3_expr = f"_ta_build_geomtime({t_min}, {t_max}).geometry.bbox"
        mesh_xy_note = mesh_xz_note = mesh_yz_note = ""
        mask_comment_3d = (
            "    # cleanup Round 44: masked to the real geometry with geom.inside(...) --\n"
            "    # same convention _build_restore_script's own 3D Surface branch already\n"
            "    # uses (geom here is the real reconstructed geometry built above, not\n"
            "    # just its bounding box -- a Sphere/Custom-3D model would otherwise show\n"
            "    # its full bounding-box face). Spatial-only, so computed once and reused\n"
            "    # for every snapshot panel, same as the single-model path.\n"
        )
        n_snaps_prefix = ""
        t_snaps_prefix = ""
        model3_line = "        _model3 = _ta_model_for_t(_tv3)\n"
        pts3_lines = (
            "            _fpts3 = np.column_stack([_fX3.ravel(), _fY3.ravel(), _fZ3.ravel(), np.full(_fX3.size, _tv3)])\n"
            "            _face_preds3.append(_extract_plot_field(_fpts3, _model3).reshape(_fX3.shape))\n"
        )
        range_comment_3d = (
            "        # Auto-range off only the in-domain values, same as the\n"
            "        # single-model path -- otherwise meaningless bounding-box-only\n"
            "        # predictions could skew the color scale.\n"
        )
        title3_expr = 'f"t = {_fmt_num(_tv3)}"'
        pass_note = ""
        suptitle3 = (
            f"    fig.suptitle({title_3d!r}, fontsize={title_fontsize})\n"
            "    _had_suptitle = True\n"
        )
        comment_2d = "    # Same multi-snapshot convention as the 3D branch above.\n"
        mask_comment_2d = (
            "    # cleanup Round 44: masked to the real geometry with geom.inside(...) --\n"
            "    # same convention _build_restore_script's own 2D Surface branch already\n"
            "    # uses. Spatial-only, so computed once and reused for every panel.\n"
        )
        xyt_lines = (
            "        _model2 = _ta_model_for_t(_tv2)\n"
            "        XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, _tv2)])\n"
            "        pred = _extract_plot_field(XYT, _model2).reshape(res, res)\n"
        )
        title2_expr = 'f"t = {_fmt_num(_tv2)}"'
        suptitle2 = (
            f"    fig.suptitle({title_2d!r}, fontsize={title_fontsize})\n"
            "    _had_suptitle = True\n"
        )
        steady_1d_block = ""
        body_1d = (
            "    # 1D + time already fully uses (x, t) as the two plot axes, so unlike\n"
            "    # the 2D/3D branches above there's no separate \"snapshot\" concept --\n"
            "    # this covers the FULL combined [t_min, t_max] range, evaluating each\n"
            "    # row of the grid (one fixed t) with whichever step's model actually\n"
            "    # covers that t, same shape/orientation as the single-model\n"
            "    # _build_restore_script's own 1D Surface branch.\n"
            f"    t_vals = np.linspace({t_min}, {t_max}, res)\n"
            "    X, T = np.meshgrid(x_vals, t_vals)\n"
            "    pred = np.empty_like(X)\n"
            "    for _ri, _tv in enumerate(t_vals):\n"
            "        _ta_m = _ta_model_for_t(_tv)\n"
            "        XT_row = np.column_stack([x_vals, np.full_like(x_vals, _tv)])\n"
            "        pred[_ri, :] = _extract_plot_field(XT_row, _ta_m).flatten()\n"
            "    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))\n"
        )
    else:
        suptitle_note = (
            "# Round 31: only the 3D/2D multi-snapshot branches below ever add a\n"
            "# fig.suptitle() above their own per-panel subplot titles -- the steady-1D\n"
            "# and 1D-surface branches are single-panel with no suptitle at all -- so\n"
            "# the shared plt.tight_layout() after this if/elif chain only needs to\n"
            "# reserve top margin for the suptitle when one was actually drawn.\n"
        )
        comment_3d = (
            "    # Genuine smooth 3D surface: each of the bounding box's 6 flat faces\n"
            "    # (from geom.bbox) is predicted directly on a fine regular grid -- no\n"
            "    # slicing or interpolation needed, it's an exact prediction at every\n"
            "    # grid point -- and drawn with plot_surface's per-quad facecolors.\n"
            "    # Unlike a scatter of discrete points, adjacent same-ish-colored grid\n"
            "    # quads blend into a continuous-looking colored surface. geom here\n"
            "    # may be a Cuboid/Sphere/Custom-3D CSG combo (not always a Cuboid),\n"
            "    # so every face is also masked down to the real domain below --\n"
            "    # otherwise a Sphere/Custom-3D model would show its full bounding-box\n"
            "    # face, not just the part that's actually inside the true geometry.\n"
            "    #\n"
            "    # Non-steady: several evenly-spaced time snapshots, each its own 3D\n"
            "    # subplot side by side in one figure -- matching the Setup tab's own\n"
            "    # \"Plot output\" -> Surface -> 3D non-steady convention (see\n"
            "    # codegen.py's \"elif _is_3d:\" Surface branch), not a single\n"
            "    # user-picked time the way this used to work. Steady: no time axis at\n"
            "    # all, always exactly one panel, unaffected by n_2d_snapshots.\n"
        )
        bbox3_expr = "geom.bbox"
        mesh_xy_note = "   # z-faces (free: x,y)"
        mesh_xz_note = "   # y-faces (free: x,z)"
        mesh_yz_note = "   # x-faces (free: y,z)"
        mask_comment_3d = (
            "    # Spatial-only inside() check -- the plain geom (not geomtime), since\n"
            "    # whether a point lies in the domain never depends on t; computed once\n"
            "    # and reused for every snapshot panel.\n"
        )
        n_snaps_prefix = "1 if is_steady else "
        t_snaps_prefix = "[None] if is_steady else "
        model3_line = ""
        pts3_lines = (
            "            _fspatial3 = np.column_stack([_fX3.ravel(), _fY3.ravel(), _fZ3.ravel()])\n"
            "            _fpts3 = _fspatial3 if is_steady else np.column_stack([_fspatial3, np.full(_fX3.size, _tv3)])\n"
            "            _face_preds3.append(_extract_plot_field(_fpts3).reshape(_fX3.shape))\n"
        )
        range_comment_3d = (
            "        # Each panel normalizes its own color scale independently when\n"
            "        # auto-range is on (matching the 2D branch below, and the Setup\n"
            "        # tab's own multi-snapshot convention: contourf with vmin=vmax=None\n"
            "        # auto-ranges per panel) -- only an explicit manual vmin/vmax is\n"
            "        # shared across every panel.\n"
        )
        title3_expr = f'{title_3d!r} if is_steady else f"t = {{_fmt_num(_tv3)}}"'
        pass_note = "  # older matplotlib without set_box_aspect -- cosmetic only"
        suptitle3 = (
            "    if not is_steady:\n"
            f"        fig.suptitle({title_3d!r}, fontsize={title_fontsize})\n"
            "        _had_suptitle = True\n"
        )
        comment_2d = (
            "    # Same multi-snapshot convention as the 3D branch above -- several\n"
            "    # evenly-spaced x-y heatmaps side by side for a non-steady restore\n"
            "    # (matching the Setup tab's own \"elif _is_2d:\" Surface branch), one\n"
            "    # plain single heatmap for a steady restore.\n"
        )
        mask_comment_2d = (
            "    # Mask the prediction down to the real problem domain -- geom is the\n"
            "    # actual (possibly non-rectangular / CSG-combined) geometry built\n"
            "    # above, not just its bounding box, so a grid point that falls in\n"
            "    # this box but outside the true domain (e.g. outside a triangular\n"
            "    # cavity) is blanked out (NaN -> contourf leaves it unfilled) rather\n"
            "    # than plotted as if it were a valid prediction there. Same masking\n"
            "    # convention codegen.py's own Surface plots already use for every\n"
            "    # non-rectangular geometry; computed once and reused for every panel.\n"
        )
        xyt_lines = (
            "        XYT = (np.column_stack([Xg.ravel(), Yg.ravel()]) if is_steady\n"
            "               else np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, _tv2)]))\n"
            "        pred = _extract_plot_field(XYT).reshape(res, res)\n"
        )
        title2_expr = f'{title_2d!r} if is_steady else f"t = {{_fmt_num(_tv2)}}"'
        suptitle2 = (
            "    if not is_steady:\n"
            f"        fig.suptitle({title_2d!r}, fontsize={title_fontsize})\n"
            "        _had_suptitle = True\n"
        )
        steady_1d_block = (
            "elif is_steady:\n"
            "    # Steady 1D: no time axis and no second spatial axis either -- there's\n"
            "    # nothing left to make a \"surface\" out of, just the one curve u(x).\n"
            "    pred = _extract_plot_field(x_vals.reshape(-1, 1)).flatten()\n"
            "    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))\n"
            f"    ax.plot(x_vals, pred, color=_PLOT_PINN_COLOR, linewidth={linewidth})\n"
            "    _grid(ax, alpha=0.2)\n"
            f"    ax.set_xlabel({xlabel_1d_steady!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel_1d_steady!r}, fontsize={axis_label_fontsize})\n"
            f"    ax.set_title({title_1d_steady!r}, fontsize={subplot_title_fontsize})\n"
        )
        body_1d = (
            f"    t_vals = np.linspace({t_min}, {t_max}, res)\n"
            "    X, T = np.meshgrid(x_vals, t_vals)\n"
            "    XT   = np.vstack([X.ravel(), T.ravel()]).T\n"
            "    pred = _extract_plot_field(XT).reshape(res, res)\n"
            "    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))\n"
            "    # Swapping which of X/T is passed first -- no reshape of pred needed,\n"
            "    # see the matching comment on the main Results panel's own 1D Surface\n"
            "    # plot for why that alone is enough to flip which one is on the x-axis.\n"
        )

    return f"""res = {resolution}
x_vals = np.linspace({x_min}, {x_max}, res)
y_vals = np.linspace({y_min}, {y_max}, res)
{suptitle_note}_had_suptitle = False
if is_3d:
{comment_3d}    _res3 = {max(10, int(res3d))}   # Plot settings -> Surface -> 3D grid resolution
    _bbox3 = np.asarray({bbox3_expr})
    _cx0, _cy0, _cz0 = _bbox3[0]; _cx1, _cy1, _cz1 = _bbox3[1]
    _xg3 = np.linspace(_cx0, _cx1, _res3)
    _yg3 = np.linspace(_cy0, _cy1, _res3)
    _zg3 = np.linspace(_cz0, _cz1, _res3)
    _Xxy3, _Yxy3 = np.meshgrid(_xg3, _yg3){mesh_xy_note}
    _Xxz3, _Zxz3 = np.meshgrid(_xg3, _zg3){mesh_xz_note}
    _Yyz3, _Zyz3 = np.meshgrid(_yg3, _zg3){mesh_yz_note}
    _faces3 = [
        (_Xxy3, _Yxy3, np.full_like(_Xxy3, _cz0)),
        (_Xxy3, _Yxy3, np.full_like(_Xxy3, _cz1)),
        (_Xxz3, np.full_like(_Xxz3, _cy0), _Zxz3),
        (_Xxz3, np.full_like(_Xxz3, _cy1), _Zxz3),
        (np.full_like(_Yyz3, _cx0), _Yyz3, _Zyz3),
        (np.full_like(_Yyz3, _cx1), _Yyz3, _Zyz3),
    ]
{mask_comment_3d}    _face_inside3 = []
    for _fX3, _fY3, _fZ3 in _faces3:
        _fspatial3 = np.column_stack([_fX3.ravel(), _fY3.ravel(), _fZ3.ravel()])
        _face_inside3.append(np.asarray(geom.inside(_fspatial3)).reshape(_fX3.shape))
    _n_snaps3 = {n_snaps_prefix}{n_2d_snapshots}
    _t_snaps3 = {t_snaps_prefix}np.linspace({t_min}, {t_max}, _n_snaps3)
    _cmap_obj3 = plt.get_cmap("{colormap}")
    _panel_w3, _panel_h3 = _plot_figsize(8, 6.5)
    fig = plt.figure(figsize=(_panel_w3 * _n_snaps3, _panel_h3))
    for _ai3, _tv3 in enumerate(_t_snaps3):
{model3_line}        _face_preds3 = []
        for _fX3, _fY3, _fZ3 in _faces3:
{pts3_lines}{range_comment_3d}        _inside_vals3 = [f[m] for f, m in zip(_face_preds3, _face_inside3) if m.any()]
        if {auto_range}:
            if _inside_vals3:
                _pv_min3 = min(v.min() for v in _inside_vals3)
                _pv_max3 = max(v.max() for v in _inside_vals3)
            else:
                _pv_min3 = min(_f.min() for _f in _face_preds3)
                _pv_max3 = max(_f.max() for _f in _face_preds3)
        else:
            _pv_min3, _pv_max3 = {vmin_val}, {vmax_val}
        ax = fig.add_subplot(1, _n_snaps3, _ai3 + 1, projection='3d')
        _norm3 = plt.Normalize(vmin=_pv_min3, vmax=_pv_max3)
        for _fi3, (_fX3, _fY3, _fZ3) in enumerate(_faces3):
            _fc3 = np.array(_cmap_obj3(_norm3(_face_preds3[_fi3])))
            # Outside-the-domain quads get alpha=0 -- invisible rather than
            # plotted as if they were a valid prediction on the bounding box.
            _fc3[..., 3] = np.where(_face_inside3[_fi3], 1.0, 0.0)
            ax.plot_surface(_fX3, _fY3, _fZ3, facecolors=_fc3,
                             rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        if {show_colorbar}:
            _sm3 = plt.cm.ScalarMappable(cmap=_cmap_obj3, norm=_norm3)
            fig.colorbar(_sm3, ax=ax, shrink=0.6, pad=0.12)
        ax.set_xlabel({xlabel_3d!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel_3d!r}, fontsize={axis_label_fontsize}); ax.set_zlabel("z", fontsize={axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (ax.xaxis, ax.yaxis, ax.zaxis)]
        ax.set_title({title3_expr}, fontsize={subplot_title_fontsize})
        try:
            ax.set_box_aspect((_cx1 - _cx0, _cy1 - _cy0, _cz1 - _cz0))
        except Exception:
            pass{pass_note}
{suptitle3}elif is_2d:
{comment_2d}    Xg, Yg = np.meshgrid(x_vals, y_vals)
{mask_comment_2d}    _inside2d = np.asarray(geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()]))).reshape(res, res)
    _n_snaps2 = {n_snaps_prefix}{n_2d_snapshots}
    _t_snaps2 = {t_snaps_prefix}np.linspace({t_min}, {t_max}, _n_snaps2)
    _panel_w2, _panel_h2 = _plot_figsize(7, 5)
    fig, axes2d = plt.subplots(1, _n_snaps2, figsize=(_panel_w2 * _n_snaps2, _panel_h2))
    if _n_snaps2 == 1: axes2d = [axes2d]
    for _ai2, _tv2 in enumerate(_t_snaps2):
{xyt_lines}        pred = np.where(_inside2d, pred, np.nan)
        im = axes2d[_ai2].contourf(Xg, Yg, pred, cmap="{colormap}", **_cf_levels({levels}, {cf_range}, pred))
        if {show_colorbar}: fig.colorbar(im, ax=axes2d[_ai2])
        axes2d[_ai2].set_xlabel({xlabel_2d!r}, fontsize={axis_label_fontsize}); axes2d[_ai2].set_ylabel({ylabel_2d!r}, fontsize={axis_label_fontsize})
        axes2d[_ai2].set_title({title2_expr}, fontsize={subplot_title_fontsize})
{suptitle2}{steady_1d_block}else:
{body_1d}    if {swap_xt}:
        im = ax.contourf(T, X, pred, cmap="{colormap}", **_cf_levels({levels}, {cf_range}, pred))
    else:
        im = ax.contourf(X, T, pred, cmap="{colormap}", **_cf_levels({levels}, {cf_range}, pred))
    if {show_colorbar}: fig.colorbar(im, ax=ax)
    ax.set_xlabel({xlabel_1d!r}, fontsize={axis_label_fontsize}); ax.set_ylabel({ylabel_1d!r}, fontsize={axis_label_fontsize})
    ax.set_title({title_1d!r}, fontsize={subplot_title_fontsize})
plt.tight_layout(rect=[0, 0, 1, 0.93] if _had_suptitle else None)
out_path = os.path.join({save_dir!r}, "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Surface plot saved to: {{out_path}}")"""


def inv_param_period(config):
    """Inverse -> Save parameters as a number of iterations: "No" -> 0,
    "Every 100 iters" -> 100, and any number typed ("Every 250 iters",
    "250"). Both scripts record the parameters this often."""
    text = str(getattr(config, "inv_param_save", "Every 100 iters") or "")
    if text.strip().lower() in ("", "no", "off", "none"):
        return 0
    m = re.search(r"\d+", text.replace(",", ""))
    return max(1, int(m.group(0))) if m else 100


def _lbfgs_maxfun(config, iterations):
    """L-BFGS maxfun for a run of `iterations`: 1.25 times them with the
    recommended settings (DeepXDE's own rule when maxfun is unset), the
    user's own maxfun otherwise."""
    if getattr(config, "lbfgs_use_default", True):
        return int(int(iterations) * 1.25)
    return config.lbfgs_maxfun


def _ic_pretrain_train_code(config, indent, geomtime_var, constraints_var, num_initial, weights_expr,
                            net_expr, save_dir_expr, os_name, log_prefix, distribution):
    """Builds the IC Pre-Training "dummy-PDE dataset + compile + restore-or-
    train" block -- cleanup Round 47 (audit §2b/§4). This block used to be
    hand-typed four times (generate_script()'s Standard and Time-Adaptive
    paths, generate_clean_script()'s Standard path and its Time-Adaptive
    per-step loop), under four different temp-model names (_model_pre,
    _model_pt, _model_pre again, _model_pre_i) and with small drifts between
    the copies (one never printed the final IC loss, two passed anchors=None
    and batch_size=None explicitly while two relied on DeepXDE's identical
    defaults). Each caller still builds its own IC constraints its own way
    (runtime eval in generate_script(), static appends in
    generate_clean_script()) and passes the resulting variable names in;
    everything from the dummy PDE onward is now this one block, with one set
    of names everywhere.

    Why the dataset looks like this (unchanged from the original copies):
    the dummy PDE returns a zero residual per output, and the dataset has
    no domain or boundary points at all -- real BCs are left out entirely
    rather than zero-weighted, because an empty-match BC term's MSE is NaN
    and 0 * NaN is still NaN. num_initial must stay >= 1 or DeepXDE fails
    at construction with an unrelated-looking IndexError.

    indent: leading whitespace for every emitted line. os_name: "_os"
    (generate_script()) or "os" (generate_clean_script()). log_prefix: the
    caller's own console-message indent ("  " or "")."""
    p = log_prefix
    rp = repr(config.ic_pretrain_restore_path)
    _dt = getattr(config, "ic_pretrain_decay_type", "none") or "none"
    _d1 = getattr(config, "ic_pretrain_decay_p1", 0) or 0
    _d2 = getattr(config, "ic_pretrain_decay_p2", 0) or 0
    if _dt in ("step", "cosine", "inverse time"):
        _ic_decay = (_dt, int(_d1), float(_d2))
    elif _dt == "exponential":
        _ic_decay = ("exponential", float(_d1))
    else:
        _ic_decay = None
    _ic_decay_kw = f", decay={_ic_decay!r}" if _ic_decay else ""
    body = f'''def _pde_dummy_pre(x, y):
    return [y[:, _oi_d:_oi_d + 1] * 0 for _oi_d in range({config.num_outputs})]
_data_pre = dde.data.TimePDE(
    {geomtime_var}, _pde_dummy_pre, {constraints_var},
    num_domain=0, num_boundary=0,
    num_initial={num_initial}, num_test={config.ic_pretrain_num_test},
    train_distribution="{distribution}",
    anchors=None,
)
_model_pre = dde.Model(_data_pre, {net_expr})
_model_pre.compile("{config.ic_pretrain_optimizer}", lr={config.ic_pretrain_lr}{_ic_decay_kw},
                   loss="{config.ic_pretrain_loss}", loss_weights={weights_expr})
# Restore-from-checkpoint vs. train-from-scratch -- the two options the GUI's
# IC Pre-Training panel offers. A checkpoint saved by any copy of this block
# (live Solve or exported script) restores in any other.
_ic_pretrain_save_dir = {save_dir_expr}
{os_name}.makedirs(_ic_pretrain_save_dir, exist_ok=True)
if {config.ic_pretrain_restore} and {rp} and {os_name}.path.exists({rp}):
    print("{p}Restoring IC pre-train model from: " + {rp})
    _ic_ckpt = torch.load({rp}, map_location="cpu")
    _model_pre.net.load_state_dict(_ic_ckpt.get("model_state_dict", _ic_ckpt))
    print("{p}IC pre-train model restored -- skipping training.")
else:
    _ic_lh, _ = _model_pre.train(iterations={config.ic_pretrain_iterations},
                                 display_every=10000, batch_size=None,
                                 model_save_path={os_name}.path.join(_ic_pretrain_save_dir, "ic_pretrain_model"))
    print(f"{p}IC pre-training done. Final IC loss: {{sum(_ic_lh.loss_train[-1]):.4e}}")
    print(f"{p}IC pre-train model saved to: {{_ic_pretrain_save_dir}}")'''
    return "\n".join((indent + ln) if ln else ln for ln in body.split("\n"))


def _clean_rar_diagnostics_code(config, is_2d, is_3d, is_steady, out_name):
    """generate_clean_script()'s RAR per-round diagnostics -- cleanup Round 53
    (long-standing open item: the export had none). For every RAR round,
    under <sol_dir>/rar_rounds/round_NN/: model.pt (that round's weights, in
    the same {"model_state_dict": ...} format the Restore tab loads),
    collocation_points.png (original points gray, earlier rounds' additions
    orange, this round's red) and solution_plot.png (the configured plot
    field after that round, masked to the real geometry). The RAR loop above
    only records each round's weights and added points in memory; this block
    replays them once _extract_plot_field exists, then restores the final
    weights. The live Solve path additionally merges each round's snapshot
    with Error Analysis reference data; the export keeps that comparison to
    its single end-of-run Error Analysis section."""
    res = config.plot_resolution
    tmax = config.t_max
    if is_3d:
        scatter = ('''    fig = plt.figure(figsize=_plot_figsize(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for pts, col, lab, sz in groups:
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=sz, c=col, label=lab)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (ax.xaxis, ax.yaxis, ax.zaxis)]''')
        zmid = auto_line_slice(config)[1]
        time_col = "" if is_steady else f", np.full(Xs.size, {tmax})"
        snap = (f'''    xs = np.linspace({config.x_min}, {config.x_max}, {res}); ys = np.linspace({config.y_min}, {config.y_max}, {res})
    Xs, Ys = np.meshgrid(xs, ys)
    pts_s = np.column_stack([Xs.ravel(), Ys.ravel(), np.full(Xs.size, {zmid}){time_col}])
    inside_s = geom.inside(pts_s[:, :3]).reshape({res}, {res})
    Us = np.where(inside_s, _extract_plot_field(pts_s).reshape({res}, {res}), np.nan)
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    im = ax.contourf(Xs, Ys, Us, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{out_name} after RAR round {{ri}} (z={{_fmt_num({zmid!r})}}{"" if is_steady else f", t={{_fmt_num({tmax!r})}}"})")''')
    elif is_2d:
        scatter = ('''    fig, ax = plt.subplots(figsize=_plot_figsize(7, 6))
    for pts, col, lab, sz in groups:
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=sz, c=col, label=lab)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal", adjustable="box")''')
        time_col = "" if is_steady else f", np.full(Xs.size, {tmax})"
        snap = (f'''    xs = np.linspace({config.x_min}, {config.x_max}, {res}); ys = np.linspace({config.y_min}, {config.y_max}, {res})
    Xs, Ys = np.meshgrid(xs, ys)
    pts_s = np.column_stack([Xs.ravel(), Ys.ravel(){time_col}])
    inside_s = geom.inside(pts_s[:, :2]).reshape({res}, {res})
    Us = np.where(inside_s, _extract_plot_field(pts_s).reshape({res}, {res}), np.nan)
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    im = ax.contourf(Xs, Ys, Us, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{out_name} after RAR round {{ri}}{"" if is_steady else f" (t={{_fmt_num({tmax!r})}})"}")''')
    elif is_steady:
        scatter = ('''    fig, ax = plt.subplots(figsize=_plot_figsize(7, 3))
    for pts, col, lab, sz in groups:
        if len(pts):
            ax.scatter(pts[:, 0], np.zeros(len(pts)), s=sz, c=col, label=lab)
    ax.set_xlabel("x"); ax.set_yticks([])''')
        snap = (f'''    xs = np.linspace({config.x_min}, {config.x_max}, {res})
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    ax.plot(xs, _extract_plot_field(xs.reshape(-1, 1)), color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
    ax.set_xlabel("x"); ax.set_ylabel({out_name!r}); _grid(ax, alpha=0.2)
    ax.set_title(f"{out_name} after RAR round {{ri}}")''')
    else:
        swap = bool(config.plot_swap_xt)
        scatter = (f'''    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    for pts, col, lab, sz in groups:
        if len(pts):
            if {swap}:
                ax.scatter(pts[:, 1], pts[:, 0], s=sz, c=col, label=lab)
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=sz, c=col, label=lab)
    ax.set_xlabel({"t" if swap else "x"!r}); ax.set_ylabel({"x" if swap else "t"!r})''')
        snap = (f'''    xs = np.linspace({config.x_min}, {config.x_max}, {res}); ts = np.linspace({config.t_min}, {config.t_max}, {res})
    Xs, Ts = np.meshgrid(xs, ts)
    Us = _extract_plot_field(np.column_stack([Xs.ravel(), Ts.ravel()])).reshape({res}, {res})
    fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
    if {swap}:
        im = ax.contourf(Ts, Xs, Us, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    else:
        im = ax.contourf(Xs, Ts, Us, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel({"t" if swap else "x"!r}); ax.set_ylabel({"x" if swap else "t"!r})
    ax.set_title(f"{out_name}(x,t) after RAR round {{ri}}")''')
    return f'''# ── RAR per-round diagnostics (rar_rounds/round_NN/) ──
rar_final_state = copy.deepcopy(model.net.state_dict())
rar_added = np.empty((0, rar_orig_pts.shape[1]))
for ri, (rar_state, rar_new) in enumerate(rar_rounds, start=1):
    rd = os.path.join(sol_dir, "rar_rounds", f"round_{{ri:02d}}")
    os.makedirs(rd, exist_ok=True)
    model.net.load_state_dict(rar_state)
    if save_dir:
        torch.save({{"model_state_dict": rar_state}}, os.path.join(rd, "model.pt"))
    groups = [(rar_orig_pts, "#adb5bd", f"Original ({{len(rar_orig_pts)}})", 4),
              (rar_added, "#ffa94d", f"Earlier rounds ({{len(rar_added)}})", 10),
              (rar_new, "#e03131", f"Round {{ri}} ({{len(rar_new)}})", 16)]
{scatter}
    ax.set_title(f"RAR collocation points -- round {{ri}}")
    ax.legend(fontsize=_lfs(8))
    plt.tight_layout(); plt.savefig(os.path.join(rd, "collocation_points.png"), dpi={config.plot_dpi}); plt.close()
{snap}
    plt.tight_layout(); plt.savefig(os.path.join(rd, "solution_plot.png"), dpi={config.plot_dpi}); plt.close()
    rar_added = np.vstack([rar_added, rar_new])
model.net.load_state_dict(rar_final_state)
print(f"RAR per-round diagnostics saved under: {{os.path.join(sol_dir, 'rar_rounds')}}")'''


def generate_clean_script(config):
    """Builds the "Export as DeepXDE Script" file: a short, plain,
    tutorial-style DeepXDE/PyTorch + matplotlib script for exactly this
    problem -- only the features actually configured (weight decay, RAR,
    Training Callbacks, IC Pre-Training, Inverse variables/observations,
    input/output transforms, Time-Adaptive stepping, Error Analysis) are
    written in at all, each as plain DeepXDE calls rather than runtime
    "if" dispatch over every option this app exposes. See generate_script()
    above for the generator Solve itself actually runs -- that one stays
    untouched; this is a second, independent generator."""
    from pinnstudio.core import constants as _cmod_hdr
    _const_header = _cmod_hdr.header_lines(_constant_values(config))
    config = _effective_config(config)
    is_2d = config.problem_dim == "2D"
    is_3d = config.problem_dim == "3D"
    # See generate_script()'s matching comment -- backward-compat remap for
    # the old "pseudorandom" literal DeepXDE never actually accepted.
    _safe_point_distribution = "pseudo" if config.point_distribution == "pseudorandom" else config.point_distribution
    # Steady-state (time-independent) problem -- see generate_script()'s
    # _is_steady for the full rationale; this generator mirrors the same
    # dispatch (plain dde.data.PDE, no GeometryXTime/Initial Condition/
    # time-derivative terms) for the "Export as DeepXDE Script" output.
    is_steady = bool(config.steady_state)
    problem_type = config.problem_type
    n_out = config.num_outputs
    out_names = [n.strip() for n in config.output_names.split(",")]
    while len(out_names) < n_out:
        out_names.append(f"u{len(out_names)}")
    n_coord = 3 if is_3d else (2 if is_2d else 1)      # spatial dims
    # Coordinate columns in a points file: + time only when time-dependent
    # (matches the live script's _n_coord_cols_bc -- a steady problem's
    # points files have no time column; this used to always expect one).
    n_coord_cols = n_coord + (0 if is_steady else 1)
    sol_ext = "gif" if ((not is_steady and config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"))
                        or (config.problem_type == "Inverse" and config.plot_type == "Parameter Convergence (GIF)")) else "png"
    is_inverse = problem_type == "Inverse"

    # Same y/z slice-value precomputation as generate_script() above -- see
    # the matching comment there. Every line-type plot in this generator
    # reads it instead of hardcoding the domain midpoint independently.
    from pinnstudio.core.line_slice import auto_line_slice
    line_slice_y, line_slice_z = auto_line_slice(config)

    # Optional title/axis-label overrides -- same blank-means-default
    # convention as generate_script() and the Restore & Visualize script
    # builders. Unlike generate_script()'s runtime `_plot_title_override
    # or <default>` (where out_name is only known at the generated
    # script's own runtime), out_name below is a plain Python variable
    # already known here at codegen time -- same as the restore builders
    # -- so the override is resolved immediately via `title_override or
    # <default>` in this function's own Python, baking in the final
    # fixed string.
    plot_title_override = (getattr(config, "plot_title_override", "") or "").strip()
    plot_xlabel_override = (getattr(config, "plot_xlabel_override", "") or "").strip()
    plot_ylabel_override = (getattr(config, "plot_ylabel_override", "") or "").strip()
    v1 = "t" if getattr(config, "_ode", False) else "x"   # Round 74: an ODE's variable

    # ---- PDE expressions (host-resolved derivative terms) --------------
    pde_exprs = [_simplify_pde_expr(e) for e in config.pde_expressions.split("|")]
    pde_check_str = "|".join(pde_exprs)
    pde_body = []
    for oi, oname in enumerate(out_names[:n_out]):
        pde_body.append(f"    {oname} = y[:, {oi}:{oi + 1}]")
        for varname, rhs in _clean_needed_derivs(oname, oi, n_out, pde_check_str, is_2d, is_3d, is_steady):
            pde_body.append(f"    {varname} = {rhs}")
    n_eq = _n_equations(config)
    pde_cols = _pde_coord_columns(_pde_coord_names(pde_exprs[:n_eq]), is_2d, is_3d, is_steady)
    if pde_cols:
        pde_body.append("    # coordinates used by the equation itself (one column each)")
        pde_body.append("    _X = x")
        for _cn, _cc in pde_cols:
            pde_body.append(f"    {_cn} = _X[:, {_cc}:{_cc + 1}]")
    uses_frac = "_frac(" in pde_check_str
    if uses_frac:
        pde_body.append("    _frac = lambda _v: _frac_apply(int_mat, _v)   # fractional operator")
    ret = "_res = " if uses_frac else "return "
    if n_eq == 1:
        pde_body.append(f"    {ret}{pde_exprs[0].strip()}")
    else:
        exprs = [e.strip() for e in pde_exprs[:n_eq]]
        pde_body.append(f"    {ret}[")
        for e in exprs:
            pde_body.append(f"        {e},")
        pde_body.append("    ]")
    if uses_frac:
        pde_body.append("    # the residual lives on the FPDE points only (the rest are auxiliary points)")
        pde_body.append("    _nr = _frac_rows(int_mat) if int_mat is not None else len(x)")
        pde_body.append("    return " + ("[_r[:_nr] for _r in _res]" if n_eq > 1 else "_res[:_nr]"))
    pde_block = ("def pde(x, y, int_mat=None):\n" if uses_frac else "def pde(x, y):\n") + "\n".join(pde_body)

    # ---- Geometry --------------------------------------------------------
    tri_verts = _parse_vertex_list(config.geom_triangle_vertices)
    if len(tri_verts) != 3:
        tri_verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    poly_verts = _parse_vertex_list(config.geom_polygon_vertices)
    if len(poly_verts) < 3:
        poly_verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    geom_line, needs_dtype_wrap = _clean_geom_line(config, is_2d, is_3d, tri_verts, poly_verts)

    # ---- Boundary conditions (custom_bc_json is the live source) --------
    # bc_panel_data_present distinguishes "the panel is active but has zero
    # rows" (custom_bc_json == "[]", a non-empty string -- the user's
    # intentional choice of no BCs) from "custom_bc_json was never set at
    # all" (a config saved before the Boundary Conditions panel existed),
    # matching generate_script()'s own `_custom_bc_entries or
    # _bc_panel_data_present` runtime check. Without this distinction, the
    # legacy fallback below would also fire (wrongly) for an intentionally
    # empty panel, or -- as this generator did before this fix -- never
    # fire at all, silently exporting zero boundary conditions for any
    # pre-panel config.
    bc_panel_data_present = bool(config.custom_bc_json)
    try:
        bc_entries = _clean_json.loads(config.custom_bc_json) if config.custom_bc_json else []
    except (ValueError, TypeError):
        bc_entries = []
    bc_used_types = {e.get('type', 'dirichlet') for e in bc_entries}
    needs_points_loader = bool({'pointset', 'pointset_operator'} & bc_used_types)
    needs_interface_adapter = 'interface2d' in bc_used_types
    # Operator BC / Point Set Operator BC's func now goes through Training
    # Monitors' own _tm_build_dvars/_tm_hess helpers (see
    # _bc_operator_def_lines) so its expression can use the same u/du_x/
    # du_xx shorthand as everywhere else, not just raw inputs/outputs/X --
    # needs that helper embedded even if Training Monitors itself is off
    # and there's no custom plot expression either.
    needs_bc_operator_dvars = bool({'operator', 'pointset_operator'} & bc_used_types)
    # Same num_test/X rule as generate_script() (see _bc_op_expr_uses_X's
    # docstring): resolved here at codegen time too, since bc_entries is
    # already parsed real Python at this point. The legacy per-side
    # fallback below never exposes X (no free-form Operator BC there), so
    # it's always left out of this check.
    _bc_op_uses_X = _bc_entries_use_operator_X(bc_entries)
    _safe_num_test = None if _bc_op_uses_X else config.num_test
    # Same Interface 2D BC "train_distribution must be uniform" auto-fix as
    # generate_script() above -- see that generator's own matching comment
    # for the full rationale (only the distribution half is auto-corrected;
    # the geometry half has no sensible value to guess).
    _bc_forced_uniform = (
        _bc_entries_use_interface2d(bc_entries) and _safe_point_distribution != "uniform"
    )
    if _bc_forced_uniform:
        _safe_point_distribution = "uniform"
    # Same Interface 2D BC geometry precondition check as generate_script()
    # -- see _bc_check_interface2d_preconditions's docstring. Raised here
    # at export time so a broken combination is caught immediately instead
    # of being baked into the exported script.
    _bc_check_interface2d_preconditions(bc_entries, config.geometry_type,
                                        is_2d=(config.problem_dim == "2D"), num_outputs=config.num_outputs)
    _bc_check_initial_operator_preconditions(bc_entries, is_steady, bool(config.time_adaptive))
    _check_anchor_preconditions(config)
    # Same Periodic BC geometry precondition as generate_script() above --
    # see _bc_check_periodic_preconditions's docstring.
    _bc_check_periodic_preconditions(bc_entries, config.geometry_type)

    bc_defs, bc_appends = [], []
    if bc_entries or bc_panel_data_present:
        for i, entry in enumerate(bc_entries):
            d, a = _clean_bc_row_code(i, entry, is_2d, is_3d, n_coord_cols, is_steady,
                                       n_out=n_out, out_names=config.output_names, t_min=config.t_min,
                                       domain=_bc_domain_values(config))
            bc_defs += d
            bc_appends += a
    else:
        bc_defs, bc_appends = _clean_legacy_bc_code(config, is_2d, n_out)

    # ---- Initial condition(s) --------------------------------------------
    ic_exprs = [_simplify_expr(e, is_2d, is_3d) for e in config.ic_expressions.split("|")]
    ic_active_list = config.ic_active.split(",")
    active_ic_outputs = _clean_active_ic_outputs(config, n_out, ic_active_list)
    needs_ic_file_loader = bool(config.forward_ic_from_file)

    ic_defs, ic_appends = [], []
    for oi in (range(n_out) if not is_steady else []):
        if oi == 0 and config.forward_ic_from_file:
            ic_appends.append(f'_ic_xt, _ic_vals = _load_ic_from_file({repr(config.forward_ic_file)})')
            ic_appends.append("constraints.append(dde.icbc.PointSetBC(_ic_xt, _ic_vals, component=0))")
        elif oi in active_ic_outputs:
            expr = ic_exprs[oi].strip() if oi < len(ic_exprs) else "np.zeros_like(x[:, 0])"
            ic_defs.append(f"def _ic{oi}_fn(x):")
            ic_defs.append(f"    return np.reshape({expr}, (-1, 1))")
            ic_appends.append(f"constraints.append(dde.icbc.IC(geomtime, _ic{oi}_fn, "
                               f"lambda x, on_initial: on_initial, component={oi}))")

    # ---- Inverse: trainable variables + observation data -----------------
    inv_vars_parsed = _parse_inverse_variables(config) if is_inverse else []
    obs_files_parsed = _parse_inverse_obs_files(config) if is_inverse else []
    obs_weights = [e['weight'] for e in obs_files_parsed] if is_inverse else []

    # ---- Loss weights (fully resolved now, not reconstructed at runtime) -
    multi_weights, expected_len = _clean_loss_weights(config, n_out, bc_entries, ic_active_list, obs_weights, is_steady,
                                                       is_2d, bc_panel_data_present)

    # ---- Training-phase list, resolved to literal per-phase weights ------
    sched_phases = _clean_sched_phases(config, multi_weights)
    for sp in sched_phases:
        sp['_w'] = _clean_phase_weights(sp, expected_len, multi_weights)
    if (config.loss_weights_from_initial or 0) > 0:
        # Round 75: weights computed at runtime from the initial losses.
        multi_weights = "loss_weights"
        for sp in sched_phases:
            sp['_w'] = "loss_weights"
    uses_nncg = any(sp.get('optimizer') == 'nncg' for sp in sched_phases)
    total_iters = sum(int(sp.get('iterations', 0) or 0) for sp in sched_phases)

    # ---- Training callbacks (opt-in only) --------------------------------
    # Also turned on by BC batching even if the user never opened Training
    # Callbacks themselves -- _build_train_cbs_code() auto-adds
    # PDEPointResampler(bc_points=True) in that case (see its own comment),
    # so this block must actually run for that line to make it into the
    # script at all.
    any_cbs = bool(config.cb_early_stopping or config.cb_point_resampler
                   or config.cb_model_checkpoint or config.cb_timer
                   or config.training_monitors_enabled
                   or bool(_training_methods_cbs(config, "train_cbs", ""))
                   or _bc_entries_use_batching(bc_entries))
    cbs_code = ""
    if any_cbs:
        _tm_clean_cbs_code = _build_training_monitors_code(config, "train_cbs", "(save_dir or '.')", indent=0)
        cbs_code = (
            _build_train_cbs_code(config, "train_cbs", indent=0)
            + ("\n" + _tm_clean_cbs_code if _tm_clean_cbs_code else "")
        ).replace("_os.path.join", "os.path.join")

    var_cb_period = inv_param_period(config) or (max(1, total_iters // 200) if total_iters else 100)
    callbacks_parts = []
    if is_inverse:
        callbacks_parts.append("[var_cb]")
    if any_cbs:
        callbacks_parts.append("train_cbs")
    if callbacks_parts:
        cbs_kw = ", callbacks=" + " + ".join(callbacks_parts)
    else:
        cbs_kw = ""
    ext_vars_kw = ", external_trainable_variables=inv_vars" if is_inverse else ""

    # ---- Network architecture: FNN or DeepXDE's built-in PFNN class -------
    # Same shared helper as generate_script() (see
    # _net_construction_helper_code()'s docstring) -- _make_net(...)
    # replaces every bare dde.nn.FNN(...) call below, including the
    # weight-decay arg this used to splice in directly (now passed through
    # as a plain float; _make_net silently drops it for PFNN).
    net_helper_code = _net_construction_helper_code(config.network_type, config.msffn_sigmas, config.msffn_sigmas_t)

    # ---- Plot Settings: figure-size standardization (Round 31) -----------
    # Same shared-helper pattern as net_helper_code just above -- see
    # _clean_figsize_runtime_code()'s own docstring.
    figsize_helper_code = _clean_figsize_runtime_code(config) + "\n\n" + _contour_levels_helper_code()

    # ---- Optional input/output transforms --------------------------------
    has_in_transform = bool(config.input_transform_enabled and config.input_transform_scale) or _input_formula_on(config)
    has_out_transform = bool(config.output_transform_enabled and config.output_transform_scale) or _output_formula_on(config)
    needs_transform_helper = has_in_transform or has_out_transform

    # ---- RAR / IC Pre-Training / Time-Adaptive flags ----------------------
    use_rar = (config.adapt_method == "RAR") and not config.time_adaptive
    use_ic_pretrain = bool(config.ic_pretrain)
    use_ta = bool(config.time_adaptive)

    # ---- Error Analysis ----------------------------------------------------
    try:
        ea_files = _clean_ast.literal_eval(config.ea_files) if config.ea_files else []
    except (ValueError, SyntaxError):
        ea_files = []
    # ea_files entries are (time, path, output_selector) 3-tuples since the
    # v40 multi-output Error Analysis feature (see the matching
    # normalization in generate_script()'s two Error Analysis paths above);
    # older saved configs only ever wrote 2-tuples (no selector), padded
    # with None here the same way, so they keep behaving exactly as before.
    # Unpacking as a plain 2-tuple here (as before v40) raised "too many
    # values to unpack" for any template with reference data the moment
    # "Export as DeepXDE Script" was used.
    _plot_custom_expr_val = (config.plot_custom_expr or "").strip()

    _ea_files_norm = []
    for _entry in ea_files:
        if len(_entry) >= 3:
            _tv0, _fp0, _sel0 = _entry[0], _entry[1], _entry[2]
        else:
            _tv0, _fp0, _sel0 = _entry[0], _entry[1], None
        if config.t_min - 1e-10 <= _tv0 <= config.t_max + 1e-10:
            _ea_files_norm.append((_tv0, _fp0, _sel0))

    # Group reference files by which model output they belong to -- same
    # grouping key convention as generate_script()'s own two Error
    # Analysis copies (repr() of the raw selector: None for the implicit
    # default field, an int for a raw output column, or a [expr, label]
    # pair for a per-group custom derived field). Each group gets its own
    # fully unrolled metrics/line/surface report below, with filenames
    # suffixed by the group's own output label when there's more than
    # one group. This fixes the gap flagged when this generator first
    # grew Error Analysis support (v40): until now, every ea_files entry
    # that didn't belong to the one field this script happens to be
    # plotting was silently DROPPED rather than compared against its own
    # output -- a template with reference files for two different
    # outputs only ever error-analyzed one of them, with no warning the
    # other was ignored.
    #
    # Unlike generate_script() (which groups at RUNTIME inside the
    # exported/live script, since that generator emits one generic,
    # config-driven script body), generate_clean_script() groups HERE,
    # at generation time, consistent with its own "decide the shape
    # once, emit only that one static code path" design: the Python
    # `for` loop over `_ea_groups.items()` below (not a loop in the
    # generated script) runs once per group while THIS FUNCTION builds
    # the script, so a single-group problem (the common case) still
    # gets byte-for-byte the same unsuffixed code as before this round.
    _ea_groups = {}
    for _tv0, _fp0, _sel0 in _ea_files_norm:
        _gkey = repr(_sel0)
        if _gkey not in _ea_groups:
            _ea_groups[_gkey] = {"sel": _sel0, "files": []}
        _ea_groups[_gkey]["files"].append((_tv0, _fp0))
    _ea_multi_output = len(_ea_groups) > 1

    def _ea_group_label(_sel):
        if _sel is None:
            return ""
        if isinstance(_sel, int):
            return (out_names[_sel].strip() if 0 <= _sel < len(out_names) else f"out{_sel}")
        _lbl = (_sel[1] or "").strip() if len(_sel) > 1 else ""
        return _lbl if _lbl else "custom"

    # Whether any group needs a custom derived field of its own (as
    # opposed to a raw output column or the implicit default field) --
    # if so, Training Monitors' _tm_build_dvars helper must be embedded
    # even when the main plot and Training Monitors themselves don't
    # otherwise need it (see the embedding condition below).
    _ea_needs_custom_op = any(isinstance(_g["sel"], (list, tuple)) for _g in _ea_groups.values())

    use_ea = bool(_ea_groups) and (config.ea_do_line or config.ea_do_surface)

    use_save = bool((config.save_dir or "").strip())

    # =====================================================================
    # Assemble the script
    # =====================================================================
    parts = []

    dim_label = "3D" if is_3d else ("2D" if is_2d else "1D")
    title = f"{dim_label} {problem_type} PINN" + (" (Time-Adaptive)" if use_ta else "")
    parts.append(f'''"""
{title}, exported from PINNStudio.

Outputs: {", ".join(out_names[:n_out])}
Domain:  x in [{config.x_min}, {config.x_max}]''' +
                 (f", y in [{config.y_min}, {config.y_max}]" if is_2d or is_3d else "") +
                 (f", z in [{config.z_min}, {config.z_max}]" if is_3d else "") +
                 ("" if is_steady else f'''
Time:    t in [{config.t_min}, {config.t_max}]''') +
                 f'''

Edit anything below and re-run -- this is a plain script, not tied to
the PINNStudio GUI.
"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
{_cpu_only_line(config)}
import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
# DeepXDE puts PyTorch on Apple's GPU (MPS) on a Mac. On an M1 it was never
# faster than the CPU for PINNs (2x slower on a small problem, even on 20-30k
# points; benchmarks/mac_gpu_check.py), and it has no float64, so PINNStudio
# trains on the CPU there.
if not torch.cuda.is_available() and str(getattr(torch, "get_default_device", lambda: "cpu")()).startswith("mps"):
    torch.set_default_device("cpu")
    print("  [Info] Training on the CPU (Apple GPU / MPS not used)")
{_cpu_threads_code(config)}''')

    if uses_nncg:
        parts.append('''_dxde_ver_parts = dde.__version__.split(".")[:3]
try:
    _dxde_ver = tuple(int(p) for p in _dxde_ver_parts)
except ValueError:
    _dxde_ver = None
if _dxde_ver is not None and _dxde_ver < (1, 13, 0):
    raise SystemExit(
        f"The NNCG optimizer requires deepxde>=1.13.0 (you have {dde.__version__}). "
        f"Please upgrade: pip install --upgrade deepxde"
    )''')

    # L-BFGS in float64 runs the whole training in float64, as in Solve's script
    _eff_float = "float64" if (config.lbfgs_float_type == "float64" or config.float_type == "float64") else config.float_type
    parts.append(f'dde.config.set_default_float("{_eff_float}")')

    if config.use_random_seed:
        parts.append(f'dde.config.set_random_seed({config.random_seed})')

    save_dir_repr = repr(config.save_dir) if use_save else repr("")
    parts.append(f'''save_dir = {save_dir_repr}
if save_dir:
    os.makedirs(save_dir, exist_ok=True)''')

    if needs_ic_file_loader:
        parts.append(f'''def _load_ic_from_file(path):
    """Loads an Initial Condition from a plain, header-less file: one
    coordinate column per spatial axis, then time, then value -- e.g.
    "x, t, u" in 1D. Only the rows already at t = {config.t_min} (the
    domain's start time) are used."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue                     # a header or comment line
    raw = np.array([r for r in rows if r])
    n_coord = {n_coord}
    t_col, v_col = n_coord, n_coord + 1
    mask = np.abs(raw[:, t_col] - {config.t_min}) < 1e-8
    if not mask.any():
        raise ValueError(f"IC file {{path}} has no rows at t = {config.t_min} (the domain's start time)")
    coords = raw[mask, :n_coord]
    t0 = np.full((int(mask.sum()), 1), {config.t_min})
    return np.hstack([coords, t0]), raw[mask, v_col:v_col + 1]''')

    if needs_points_loader:
        parts.append(f'''def _load_bc_points(path, n_coord_cols={n_coord_cols}, n_values=1):
    """Loads a points file for a PointSetBC / PointSetOperatorBC row:
    n_coord_cols coordinate (+ time) columns, then n_values target value
    column(s) -- n_values > 1 is PointSetBC's multi-component case
    (component=[i, j, ...]), one extra trailing column per listed
    component, in the same order DeepXDE expects."""
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"Boundary condition points file not found: {{path!r}}")
    data = np.loadtxt(path, delimiter=",") if path.lower().endswith(".csv") else np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_have = data.shape[1] - n_coord_cols
    if n_have < n_values:
        raise RuntimeError(f"Boundary condition points file {{path!r}} needs {{n_values}} value "
                            f"column(s) after its {{n_coord_cols}} coordinate column(s), but only has {{n_have}}.")
    return data[:, :n_coord_cols], data[:, n_coord_cols:n_coord_cols + n_values]''')

    if needs_interface_adapter:
        parts.append('''class _Interface2DGeomAdapter:
    """Interface2DBC expects boundary_normal() without GeometryXTime's
    extra (always-zero) time-normal column. For a steady-state problem
    geomtime is just geom (no such extra column), so the trim must be
    skipped there -- otherwise it would wrongly chop off a real spatial
    normal component instead."""
    def __init__(self, gt, is_steady=False):
        self._gt = gt
        self._is_steady = is_steady
    def on_boundary(self, x):
        return self._gt.on_boundary(x)
    def boundary_normal(self, x):
        _n = self._gt.boundary_normal(x)
        return _n if self._is_steady else _n[:, :-1]''')

    if needs_dtype_wrap:
        parts.append(_dtype_safe_geom_class_code())

    parts.append(net_helper_code)
    parts.append(figsize_helper_code)
    # Training Monitors' own _tm_build_dvars/_tm_hess helpers are also what
    # the custom plot-field expression (_extract_plot_field, below) routes
    # through once configured -- same derivative-aware mechanism Round 17
    # wired into generate_script()/the Restore scripts, now also reaching
    # this exporter. Embed the helper whenever any of these needs it, not
    # just for Training Monitors, so a custom expression works even if
    # Training Monitors itself is off -- including a multi-output Error
    # Analysis group whose own selector is a custom derived field rather
    # than the main plot's.
    if (any_cbs and config.training_monitors_enabled) or _plot_custom_expr_val or needs_bc_operator_dvars or _ea_needs_custom_op:
        parts.append(_training_monitor_runtime_code())

    if needs_transform_helper:
        parts.append(_net_transforms_code(config, "_apply_transforms"))

    if _uses_integral(config):
        parts.append(_integral_helper_code(config))
    if "_frac(" in (config.pde_expressions or ""):
        parts.append(_frac_helper_code(config))
    parts.append(pde_block)
    _tm_defs = _training_methods_defs(config)
    if _tm_defs:
        parts.append(_tm_defs.strip("\n"))

    if bc_defs:
        parts.append("\n".join(bc_defs))

    if ic_defs:
        parts.append("\n".join(ic_defs))

    if is_inverse:
        inv_lines = []
        for name, init, _true in inv_vars_parsed:
            inv_lines.append(f"{name} = dde.Variable({init})")
        inv_lines.append("inv_vars = [" + ", ".join(n for n, _i, _t in inv_vars_parsed) + "]")
        inv_lines.append('''
def _load_obs_data(path):
    # numbers separated by commas or spaces; header, comment and blank lines skipped
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if not parts or parts[0][0] in "#%":
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    return np.array(rows)
''')
        # Number of leading coordinate columns in a measured-data file --
        # x/y/z plus time, EXCEPT for a steady-state problem, which has no
        # time axis at all (same distinction the BC-panel machinery above
        # already makes). The pre-existing is_3d/is_2d-only ladder never
        # checked is_steady, so a steady 2D/3D template's observed-data
        # file (x, y, u -- no time column) would have been sliced as if
        # its value column were "t" and there were no value column at
        # all. Known at generation time here, so this resolves to a
        # literal slice, not a runtime branch.
        _n_obs_cols = (3 if is_3d else (2 if is_2d else 1)) if is_steady else (4 if is_3d else (3 if is_2d else 2))
        obs_lines = ["obs_entries = []  # (xt, u, output_idx, weight, custom_expr), one per measured-data file"]
        _obs_red = obs_reduce_args(config)
        if _obs_red:
            obs_lines.insert(0, OBS_REDUCE_HELPER.strip("\n") + "\n")
        _n_obs_space = _n_obs_cols if is_steady else _n_obs_cols - 1
        for _oi_f, of in enumerate(obs_files_parsed):
            # of['path']!r (not r"...") -- same Windows-trailing-backslash
            # SyntaxError this file's other path sites were already fixed
            # for; this one is generate_clean_script()'s own separate
            # Inverse observation-file loading path and was missed then.
            obs_lines.append(f"_of_data = _load_obs_data({of['path']!r})")
            if _obs_red:
                obs_lines.append(f"_of_data = _obs_reduce(_of_data, {_n_obs_cols}, {_n_obs_space}, "
                                 f"{_obs_red[0]!r}, {_obs_red[1]}, {_obs_red[2]})")
            obs_lines.append(f"_of_xt, _of_u = _of_data[:, 0:{_n_obs_cols}], _of_data[:, {_n_obs_cols}:{_n_obs_cols + 1}]")
            obs_lines.append(f"obs_entries.append((_of_xt, _of_u, {of['output_idx']}, {of['weight']}, {of.get('custom_expr', '')!r}))")
            if of.get('custom_expr'):
                # This measured-data file's value is a DERIVED field (e.g.
                # 1D Schrodinger's |h| = sqrt(u**2+v**2)), not a single raw
                # output column -- same expression syntax as the Results
                # panel's "Custom..." plot field, evaluated here on live
                # torch tensors (gradient-safe) rather than a numpy array
                # after predict(), via dde.icbc.PointSetOperatorBC instead
                # of PointSetBC(component=...).
                # n.strip()!r (not a manually-quoted f-string) so an output
                # name containing a quote can't break this dict literal --
                # same repr()-escaping convention used for free-text fields
                # elsewhere in this file.
                _bindings = ", ".join(f'{n.strip()!r}: outputs[:, {_i}:{_i + 1}]' for _i, n in enumerate(out_names))
                obs_lines.append(f'''def _obs_func_{_oi_f}(inputs, outputs, X):
    _ns_obs = {{"sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
                "arcsin": torch.arcsin, "arccos": torch.arccos, "arctan": torch.arctan,
                "exp": torch.exp, "log": torch.log, "log10": torch.log10,
                # +1e-12 guard baked into sqrt itself -- same fix and
                # rationale as generate_script()'s own _make_obs_func (this
                # is generate_clean_script()'s separate copy of the same
                # training-time observation-matching function, which
                # DeepXDE differentiates through via PointSetOperatorBC;
                # the same u=v=0 singularity applies here identically).
                "sqrt": lambda _v: torch.sqrt(_v + 1e-12), "abs": torch.abs, "ceil": torch.ceil,
                "floor": torch.floor, "pi": np.pi, "torch": torch,
                {_bindings}}}
    _r = eval({of['custom_expr']!r}, _ns_obs)
    return _r if _r.dim() == 2 else _r.reshape(-1, 1)''')
        # Built as literal per-entry statements (not a generic runtime
        # dispatch) so the exported script stays plain, readable code --
        # matching this generator's own "resolve everything at export
        # time" convention -- rather than looking up each _obs_func_N by
        # constructed name at runtime.
        obs_lines.append("obs_bcs = []")
        for _oi_f, of in enumerate(obs_files_parsed):
            if of.get('custom_expr'):
                obs_lines.append(f"obs_bcs.append(dde.icbc.PointSetOperatorBC("
                                  f"obs_entries[{_oi_f}][0], obs_entries[{_oi_f}][1], _obs_func_{_oi_f}))")
            else:
                obs_lines.append(f"obs_bcs.append(dde.icbc.PointSetBC("
                                  f"obs_entries[{_oi_f}][0], obs_entries[{_oi_f}][1], "
                                  f"component=obs_entries[{_oi_f}][2]))")
        if getattr(config, "inverse_obs_anchors", True):
            obs_lines.append("obs_anchors = np.vstack([e[0] for e in obs_entries]) if obs_entries else None")
        else:
            obs_lines.append("obs_anchors = None  # measured points only in the data loss, not anchors")
        parts.append("\n".join(inv_lines))
        parts.append("\n".join(obs_lines))

    net_line = (f'net = _make_net({list(config.layers)}, "{config.activation}", '
                f'"{config.kernel_initializer}", {config.weight_decay})')
    if needs_transform_helper:
        net_line += "\nnet = _apply_transforms(net)"

    if not use_ta:
        # =================================================================
        # Standard (non-Time-Adaptive) training path
        # =================================================================
        if is_steady:
            # No time axis at all -- geomtime is just an alias for geom (every
            # dde.icbc.XxxBC constructor accepts any Geometry, not specifically
            # a GeometryXTime), so every BC-construction line above is reused
            # unchanged. See generate_script()'s _is_steady for the same trick.
            geom_lines = [geom_line, "geomtime = geom"]
        else:
            geom_lines = [geom_line,
                          f"timedomain = dde.geometry.TimeDomain({config.t_min}, {config.t_max})",
                          "geomtime = dde.geometry.GeometryXTime(geom, timedomain)"]
        parts.append("\n".join(geom_lines))
        if _formula_reference_code(config):
            parts.append(_formula_reference_code(config, "geom"))

        constraints_lines = ["constraints = []"] + bc_appends + ic_appends
        if is_inverse:
            constraints_lines.append("constraints.extend(obs_bcs)")
        parts.append("\n".join(constraints_lines))
        if _bc_op_uses_X:
            parts.append(
                "# num_test disabled (left at None) because an Operator or Point Set\n"
                "# Operator BC's Value expression references X -- DeepXDE requires this\n"
                "# whenever a boundary condition function uses the raw input points.")
        if _bc_forced_uniform:
            parts.append(
                "# Point Distribution was switched to uniform automatically because an\n"
                "# Interface 2D boundary condition is active (DeepXDE requires evenly-\n"
                "# spaced boundary points so Interface 2D BC can pair corresponding\n"
                "# points correctly across its two edges).")

        user_anchor_code = _anchor_points_code(config)
        if user_anchor_code:
            parts.append(user_anchor_code)
        _ic_anch = bool(getattr(config, "ic_file_anchors", False) and config.forward_ic_from_file and not is_steady)
        anchors_arg = _anchor_join((["obs_anchors"] if is_inverse else [])
                                   + (["_user_anchors"] if user_anchor_code else [])
                                   + (["_ic_xt"] if _ic_anch else []), var="a")
        if is_steady:
            data_lines = [f'''data = dde.data.PDE(
    geomtime, pde, constraints,
    num_domain={config.num_domain}, num_boundary={config.num_boundary},
    num_test={_safe_num_test!r},
    train_distribution="{_safe_point_distribution}",
    anchors={anchors_arg},
)''']
        else:
            data_lines = [f'''data = dde.data.TimePDE(
    geomtime, pde, constraints,
    num_domain={config.num_domain}, num_boundary={config.num_boundary},
    num_initial={config.num_initial}, num_test={_safe_num_test!r},
    train_distribution="{_safe_point_distribution}",
    anchors={anchors_arg},
)''']
        if "_frac(" in (config.pde_expressions or ""):
            data_lines = [_frac_data_text(t, config, "geomtime", "pde", "constraints") for t in data_lines]
        parts.append("\n".join(data_lines))
        parts.append(net_line)

        if use_ic_pretrain and not is_steady:
            ic_pretrain_lines = [f'''# ── IC Pre-Training: {config.ic_pretrain_iterations} iterations, IC loss only ──
# A dummy zero-residual PDE and an IC-only dataset (no domain/boundary
# points at all -- an empty-match BC term's loss is NaN even at weight
# 0, so real BCs are left out of this dataset entirely, not zero-weighted).
_ic_pre_geomtime = dde.geometry.GeometryXTime({geom_line.split("=", 1)[1].strip()}, dde.geometry.TimeDomain({config.t_min}, {config.t_max}))
_ic_pre_constraints = []''']
            # Re-use the same IC construction, bound to the pre-training geomtime.
            pretrain_ic_appends = [ln.replace("geomtime", "_ic_pre_geomtime") if "geomtime" in ln else ln
                                    for ln in ic_appends]
            pretrain_ic_appends = [ln.replace("constraints.append", "_ic_pre_constraints.append")
                                    for ln in pretrain_ic_appends]
            ic_pretrain_lines[0] += "\n" + "\n".join(pretrain_ic_appends)
            ic_only_weights = [0.0] * n_out + [1000.0] * len(active_ic_outputs)
            ic_pretrain_lines.append(_ic_pretrain_train_code(
                config, "", "_ic_pre_geomtime", "_ic_pre_constraints",
                max(1, config.ic_pretrain_num_initial), repr(ic_only_weights), "net",
                'os.path.join(save_dir, "ic_pretrain") if save_dir else "ic_pretrain"',
                "os", "", _safe_point_distribution))
            parts.append("\n".join(ic_pretrain_lines))

        parts.append("model = dde.Model(data, net)")
        if (config.loss_weights_from_initial or 0) > 0:
            parts.append(_init_loss_weights_code(config, "", "model", "loss_weights", repr(config.learning_rate),
                                                 "inv_vars" if is_inverse else "None"))
        if cbs_code:
            parts.append(cbs_code)
        if is_inverse:
            hist_path = 'os.path.join(save_dir, "param_history.txt") if save_dir else "/tmp/param_history.txt"'
            parts.append(f'''var_history_path = {hist_path}
var_cb = dde.callbacks.VariableValue(inv_vars, period={var_cb_period}, filename=var_history_path, precision=8)''')

        phase_lines = []
        for pi, sp in enumerate(sched_phases):
            phase_lines.append(f"# ── Phase {pi + 1}: {sp.get('optimizer', 'adam')}, "
                                f"{int(sp.get('iterations', 0) or 0)} iterations ──")
            phase_lines += _clean_phase_train_lines(sp, sp['_w'], ext_vars_kw, cbs_kw, "model", "data", config)
        parts.append("\n".join(phase_lines))

        if use_rar:
            rar_lines = [f'''# ── RAR: residual-based adaptive refinement ──
# Each round's weights and newly added points are kept in memory so the
# per-round diagnostics further down (after _extract_plot_field exists) can
# replay them -- same rar_rounds/round_NN/ outputs as the live Solve path.
import copy
rar_orig_pts = np.array(data.train_x_all, copy=True)
rar_rounds = []
for rar_cycle in range({config.rar_cycles}):''']
            # Candidate points come straight from the real geometry's own
            # random_points() -- geomtime (an alias for geom when the
            # problem is steady; otherwise a GeometryXTime wrapping geom
            # and the time domain) -- instead of independently
            # uniform-sampling each axis over the domain's bounding box.
            # The two agree for a Rectangle/Cuboid/Interval domain, but for
            # any non-box geometry (L-Shape, Disk, Polygon, Triangle, an
            # Annulus or other CSG shape with a hole, ...) the bounding box
            # is strictly larger than the real domain, so bounding-box
            # sampling could put candidate points -- and then, since RAR
            # keeps whichever candidates have the largest predicted PDE
            # residual, sometimes the actual added points too -- outside
            # the geometry entirely. geom/geomtime's own random_points()
            # is exactly what dde.data.PDE/TimePDE already use for
            # ordinary training collocation points elsewhere in this same
            # script, and it performs real rejection sampling against the
            # geometry for every non-box shape, so RAR's candidates are
            # now guaranteed to lie inside the actual domain for any
            # geometry, not just a rectangular one.
            rar_lines.append(f'''    xt_cand = geomtime.random_points({config.rar_candidates})''')
            # "Points from:" (rar_output_selector): restrict selection to one
            # equation's residual, same as the live Solve path; -1 (default)
            # combines every equation.
            if 0 <= config.rar_output_selector < _n_equations(config) and _n_equations(config) > 1:
                _rar_resid_line = f"    residuals = np.abs(res[{config.rar_output_selector}]).flatten()"
            else:
                _rar_resid_line = "    residuals = np.sum([np.abs(r).flatten() for r in res], axis=0) if isinstance(res, list) else np.abs(res).flatten()"
            rar_lines.append(f'''    res = model.predict(xt_cand, operator=pde)
{_rar_resid_line}{_rar_tolerance_check(config, "    ")}
    top_idx = np.argsort(residuals)[-{config.rar_add_points}:]
    print(f"RAR round {{rar_cycle + 1}}: max residual {{residuals.max():.4e}}, mean {{residuals.mean():.4e}}")
    data.add_anchors(xt_cand[top_idx])
    model.compile("{config.optimizer}", lr={config.learning_rate}, loss="{config.loss_type}", loss_weights={multi_weights})
    loss_history, train_state = model.train(iterations={config.rar_adam_iters}, display_every={config.loss_display_every})''')
            if config.rar_lbfgs_iters > 0:
                rar_lines.append(f'''    dde.optimizers.set_LBFGS_options(maxcor={config.lbfgs_maxcor}, ftol={config.lbfgs_ftol},
                                     gtol={config.lbfgs_gtol}, maxiter={config.rar_lbfgs_iters},
                                     maxfun={_lbfgs_maxfun(config, config.rar_lbfgs_iters)}, maxls={config.lbfgs_maxls})
    model.compile("L-BFGS", loss="{config.loss_type}", loss_weights={multi_weights})
    loss_history, train_state = model.train(display_every={config.loss_display_every})''')
            rar_lines.append("    rar_rounds.append((copy.deepcopy(model.net.state_dict()), xt_cand[top_idx]))")
            parts.append("\n".join(rar_lines))

        if use_save:
            parts.append('model.save(os.path.join(save_dir, "model"))')

    else:
        # =================================================================
        # Time-Adaptive training path -- the domain's time range is split
        # into sequential sub-intervals; each one trains its own model,
        # seeded from the previous interval's predicted solution as its
        # Initial Condition (and, if Transfer Learning is on, from its
        # network weights too). Matches this app's real Time-Adaptive
        # support: 1D, 2D, and 3D (a 3D solution is visualized the same way
        # as the Standard path's own 3D plot -- an x-t slice at the domain's
        # y/z mid-points, since there's no volumetric renderer here).
        # =================================================================
        try:
            ta_groups = _clean_json.loads(config.ta_step_groups) if config.ta_step_groups else []
        except (ValueError, TypeError):
            ta_groups = []
        if not ta_groups:
            ta_groups = [{"t_start": config.t_min, "t_end": config.t_max, "steps": config.ta_num_steps}]
        intervals = []
        for grp in ta_groups:
            dt = (grp['t_end'] - grp['t_start']) / grp['steps']
            for gi in range(grp['steps']):
                t0 = grp['t_start'] + gi * dt
                intervals.append((t0, t0 + dt))
        parts.append(f"intervals = {intervals}  # (t0, t1) per Time-Adaptive step")

        geom_expr = geom_line.split("=", 1)[1].strip()
        ta_lines = [f'''def _build_geom():
    return {geom_expr}
''']
        if _formula_reference_code(config):
            ta_lines.append(_formula_reference_code(config, "_build_geom()"))
        if is_3d:
            # grid_size is a per-axis resolution, so this cubes it.
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
_xg = np.linspace({config.x_min}, {config.x_max}, grid_size)
_yg = np.linspace({config.y_min}, {config.y_max}, grid_size)
_zg = np.linspace({config.z_min}, {config.z_max}, grid_size)
_Xg, _Yg, _Zg = np.meshgrid(_xg, _yg, _zg)
x_grid = np.column_stack([_Xg.ravel(), _Yg.ravel(), _Zg.ravel()])''')
        elif is_2d:
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
_xg = np.linspace({config.x_min}, {config.x_max}, grid_size)
_yg = np.linspace({config.y_min}, {config.y_max}, grid_size)
_Xg, _Yg = np.meshgrid(_xg, _yg)
x_grid = np.column_stack([_Xg.ravel(), _Yg.ravel()])''')
        else:
            ta_lines.append(f'''grid_size = {config.ta_grid_size}
x_grid = np.linspace({config.x_min}, {config.x_max}, grid_size).reshape(-1, 1)''')

        if config.forward_ic_from_file:
            ta_lines.append(f'_ic_ta_xt, prev_u = _load_ic_from_file({repr(config.forward_ic_file)})')
        else:
            # ic_exprs[0] already expects spatial columns only (x, or x & y) --
            # exactly what x_grid holds, so no time column is needed here.
            ic0_expr = ic_exprs[0].strip() if ic_exprs else "np.zeros_like(x[:, 0])"
            ta_lines.append(f'''def _prev_u0(x):
    return np.reshape({ic0_expr}, (-1, 1))
prev_u = _prev_u0(x_grid)''')

        ta_lines.append('''
all_x, all_t, all_u = [], [], []   # accumulated per-step (X, T, U) grids for the final stitched plot
ta_step_models = []                # [(t0, t1, model_i)] -- used for Error Analysis, if configured
prev_net = None
_prev_state = None

# Accumulates the loss history across EVERY optimizer phase of EVERY time
# sub-domain, so the final loss plot shows the full Time-Adaptive run
# instead of just the last phase of the last sub-domain -- loss_history
# gets overwritten by each new model_i.train() call below (there can be
# several per sub-domain, one per scheduler phase), so without this only
# whichever call happened to run last for the very last step would ever
# reach the plot. Each phase's own step counter restarts at 0, so
# _ta_loss_offset (the cumulative iteration count so far) is added to keep
# the x-axis continuously increasing across the whole run.
_ta_all_steps, _ta_all_train_loss, _ta_all_test_loss = [], [], []
_ta_loss_offset = 0''')
        parts.append("\n".join(ta_lines))
        if cbs_code:
            parts.append(cbs_code)

        loop_lines = [f"def _ta_weights(w, constraints):",
                      f"    n = {_n_equations(config)} + len(constraints)   # PDE terms + this step's conditions",
                      "    return list(w) if len(w) == n else [1.0] * n",
                      "",
                      "",
                      "for step_i, (t0, t1) in enumerate(intervals):",
                      "    geom_i = _build_geom()",
                      "    timedomain_i = dde.geometry.TimeDomain(t0, t1)",
                      "    geomtime_i = dde.geometry.GeometryXTime(geom_i, timedomain_i)",
                      ""]
        # Reuse the shared BC construction, bound to this step's geomtime.
        step_bc_appends = [ln.replace("geomtime", "geomtime_i") for ln in bc_appends]
        loop_lines.append("    constraints_i = []")
        for ln in step_bc_appends:
            loop_lines.append(f"    {ln.replace('constraints.append', 'constraints_i.append')}")
        loop_lines.append("")
        loop_lines.append("    _xt_ic_i = None")
        loop_lines.append("    if step_i == 0:")
        step0_ic_appends = [ln.replace("geomtime", "geomtime_i").replace("constraints.append", "constraints_i.append")
                             for ln in ic_appends]
        for ln in step0_ic_appends:
            loop_lines.append(f"        {ln}")
        if not step0_ic_appends:
            loop_lines.append("        pass   # no initial condition as a loss term (an exact one, say)")
        loop_lines.append("    else:")
        loop_lines.append("        _xt_ic_i = np.column_stack([x_grid, np.full(len(x_grid), t0)])")
        for _oi_ic in range(n_out):
            loop_lines.append(
                f"        constraints_i.append(dde.icbc.PointSetBC(_xt_ic_i, "
                f"prev_u[:, {_oi_ic}:{_oi_ic + 1}], component={_oi_ic}))"
            )
        loop_lines.append("")
        # as in Solve's script: from the second step on (2D/3D), the points
        # where the previous window's solution is given are anchors too
        anchors_i = ("obs_anchors" if is_inverse
                     else ("_xt_ic_i" if (is_2d or is_3d) else "None"))
        loop_lines.append(f'''    data_i = dde.data.TimePDE(
        geomtime_i, pde, constraints_i,
        num_domain={config.num_domain}, num_boundary={config.num_boundary},
        num_initial={config.num_initial}, num_test={_safe_num_test!r},
        train_distribution="{_safe_point_distribution}",
        anchors={anchors_i},
    )
    net_i = _make_net({list(config.layers)}, "{config.activation}", "{config.kernel_initializer}", {config.weight_decay})''')
        if needs_transform_helper:
            loop_lines.append("    net_i = _apply_transforms(net_i)")
        if config.ta_transfer_learning:
            loop_lines.append('''    if _prev_state is not None:
        # transfer learning: the previous step's weights and biases (after the
        # phase chosen under Transfer from), as Solve's script does
        _cur = net_i.state_dict()
        _cur.update({k: v for k, v in _prev_state.items()
                     if (k.endswith(".weight") or k.endswith(".bias")) and k in _cur and _cur[k].shape == v.shape})
        net_i.load_state_dict(_cur)''')

        if use_ic_pretrain:
            ic_only_weights = [0.0] * n_out + [1000.0] * len(active_ic_outputs)
            step0_pretrain_ic = [ln.replace("geomtime", "_ic_pre_geomtime_i")
                                  .replace("constraints.append", "_ic_pre_constraints_i.append")
                                  for ln in ic_appends]
            loop_lines.append(f'''    if step_i == 0:
        _ic_pre_geomtime_i = dde.geometry.GeometryXTime(_build_geom(), dde.geometry.TimeDomain(t0, t1))
        _ic_pre_constraints_i = []''')
            for ln in step0_pretrain_ic:
                loop_lines.append(f"        {ln}")
            loop_lines.append(_ic_pretrain_train_code(
                config, "        ", "_ic_pre_geomtime_i", "_ic_pre_constraints_i",
                max(1, config.ic_pretrain_num_initial), repr(ic_only_weights), "net_i",
                'os.path.join(save_dir, "ic_pretrain") if save_dir else "ic_pretrain"',
                "os", "", _safe_point_distribution))

        loop_lines.append("    model_i = dde.Model(data_i, net_i)")
        loop_lines.append("    _adam_state_i = None")
        if is_inverse:
            loop_lines.append("    if step_i == 0:")
            hist_path = 'os.path.join(save_dir, "param_history.txt") if save_dir else "/tmp/param_history.txt"'
            loop_lines.append(f'''        var_history_path = {hist_path}
        var_cb = dde.callbacks.VariableValue(inv_vars, period={var_cb_period}, filename=var_history_path, precision=8)''')
        for pi, sp in enumerate(sched_phases):
            loop_lines.append(f"    # ── Phase {pi + 1}: {sp.get('optimizer', 'adam')}, "
                               f"{int(sp.get('iterations', 0) or 0)} iterations ──")
            # a later step pins every output to the previous step's solution,
            # which can be more loss terms than step 0 had (an output with no
            # initial condition): _ta_weights then uses equal weights, as
            # Solve does, instead of crashing on the length mismatch
            loop_lines += _clean_phase_train_lines(sp, f"_ta_weights({sp['_w']}, constraints_i)", ext_vars_kw,
                                                     cbs_kw, "model_i", "data_i", config, indent="    ")
            if sp.get('optimizer', 'adam') != 'lbfgs':
                loop_lines.append("    _adam_state_i = {k: v.detach().clone() for k, v in net_i.state_dict().items()}")
            loop_lines.append(
                "    _ta_all_steps.extend([_s + _ta_loss_offset for _s in loss_history.steps]); "
                "_ta_all_train_loss.extend(loss_history.loss_train); "
                "_ta_all_test_loss.extend(loss_history.loss_test); "
                "_ta_loss_offset += (loss_history.steps[-1] if loss_history.steps else 0)"
            )
        loop_lines.append("    ta_step_models.append((t0, t1, model_i))")
        loop_lines.append("    prev_net = net_i")
        loop_lines.append(f"    _prev_state = (_adam_state_i if ({config.ta_transfer_optimizer!r} == 'adam' and _adam_state_i is not None) "
                          f"else {{k: v.detach().clone() for k, v in net_i.state_dict().items()}})")
        if is_3d:
            loop_lines.append('''    _xyzt_pred = np.column_stack([x_grid, np.full(len(x_grid), t1)])
    prev_u = model_i.predict(_xyzt_pred)
    _x_plot = np.linspace({0}, {1}, {7})
    _t_plot = np.linspace(t0, t1, {7})
    _y_mid = ({2} + {3}) / 2.0
    _z_mid = ({4} + {5}) / 2.0
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XYZTp = np.column_stack([_Xp.ravel(), np.full(_Xp.size, _y_mid), np.full(_Xp.size, _z_mid), _Tp.ravel()])
    _Up = model_i.predict(_XYZTp)[:, {6}].reshape({7}, {7})
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, line_slice_y, line_slice_y, line_slice_z, line_slice_z, config.plot_output_idx, config.plot_resolution))
        elif is_2d:
            loop_lines.append('''    _xyt_pred = np.column_stack([x_grid, np.full(len(x_grid), t1)])
    prev_u = model_i.predict(_xyt_pred)
    _x_plot = np.linspace({0}, {1}, {5})
    _t_plot = np.linspace(t0, t1, {5})
    _y_mid = ({2} + {3}) / 2.0
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XYTp = np.column_stack([_Xp.ravel(), np.full(_Xp.size, _y_mid), _Tp.ravel()])
    _Up = model_i.predict(_XYTp)[:, {4}].reshape({5}, {5})
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, line_slice_y, line_slice_y, config.plot_output_idx, config.plot_resolution))
        else:
            loop_lines.append('''    _xt_pred = np.column_stack([x_grid.ravel(), np.full(grid_size, t1)])
    prev_u = model_i.predict(_xt_pred)
    _x_plot = np.linspace({0}, {1}, {3})
    _t_plot = np.linspace(t0, t1, {3})
    _Xp, _Tp = np.meshgrid(_x_plot, _t_plot)
    _XTp = np.vstack([_Xp.ravel(), _Tp.ravel()]).T
    _Up = model_i.predict(_XTp)[:, {2}].reshape({3}, {3})
    all_x.append(_Xp); all_t.append(_Tp); all_u.append(_Up)'''.format(
                config.x_min, config.x_max, config.plot_output_idx, config.plot_resolution))
        loop_lines.append(f'    print(f"Step {{step_i + 1}}/{{len(intervals)}} done. '
                           f'Final train loss: {{sum(loss_history.loss_train[-1]):.4e}}")')
        parts.append("\n".join(loop_lines))

        parts.append('''model = ta_step_models[-1][2]  # last step's model, for Error Analysis / ad-hoc predict() calls''')
        if use_save:
            parts.append('model.save(os.path.join(save_dir, "model"))')

    # =========================================================================
    # Plots
    # =========================================================================
    parts.append(f'''sol_dir = save_dir if save_dir else "/tmp"
loss_path = os.path.join(sol_dir, "loss_plot.png")
solution_path = os.path.join(sol_dir, "solution_plot.{sol_ext}")''')

    if use_ta:
        # Full Time-Adaptive run -- every optimizer phase of every time
        # sub-domain, concatenated with a running iteration offset (see
        # "_ta_all_steps"/"_ta_loss_offset" above) -- not just whichever
        # phase happened to run last for the very last sub-domain. This used
        # to plot only loss_history (the last phase's own, restarting-at-0
        # history), which is what the old "last Time-Adaptive step's loss
        # curve" comment here was flagging as a known limitation.
        loss_comment = "  # full run: every phase of every time sub-domain, stitched together"
        _loss_series = "\n".join(_clean_loss_series_lines(config, "_ta_all_train_loss", "_ta_all_test_loss", "_ta_all_steps"))
        parts.append(f'''# ── Loss plot ──{loss_comment}
plt.figure(figsize=_plot_figsize(7, 5))
{_loss_series}
plt.xlabel("Iteration (cumulative across all time sub-domains)"); plt.ylabel("Loss")
plt.title("Training & Test Loss — All Time Sub-domains")
plt.legend(); plt.tight_layout()
plt.savefig(loss_path, dpi={config.plot_dpi})
plt.close()
print(f"Loss plot saved: {{loss_path}}")''')
    else:
        _loss_series = "\n".join(_clean_loss_series_lines(config, "loss_history.loss_train", "loss_history.loss_test", "loss_history.steps"))
        parts.append(f'''# ── Loss plot ──
plt.figure(figsize=_plot_figsize(7, 5))
{_loss_series}
plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Training & Test Loss")
plt.legend(); plt.tight_layout()
plt.savefig(loss_path, dpi={config.plot_dpi})
plt.close()
print(f"Loss plot saved: {{loss_path}}")''')

    if is_inverse:
        parts.append(f'''# ── Parameter convergence plot ──
param_iters, param_vals = [], [[] for _ in inv_vars]
with open(var_history_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or "[" not in line:
            continue
        it_str, rest = line.split(None, 1)
        vals = [float(v) for v in rest.strip("[]").split(",") if v.strip()]
        if len(vals) != len(inv_vars):
            continue
        param_iters.append(int(it_str))
        for vi, v in enumerate(vals):
            param_vals[vi].append(v)
param_iters = np.array(param_iters)
_inv_true_vals = {[t for _n, _i, t in inv_vars_parsed]!r}

{_param_axis_helper_code(config.plot_axis_label_fontsize, config.plot_subplot_title_fontsize, param_style_of(config))}

fig, axes = plt.subplots(len(inv_vars), 1, figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * len(inv_vars)), squeeze=False)
for vi, name in enumerate({[n for n, _i, _t in inv_vars_parsed]!r}):
    true_val = _inv_true_vals[vi] if vi < len(_inv_true_vals) else None
    final_val = _draw_param_ax(axes[vi][0], name, param_iters, param_vals[vi], true_val,
                               {bool(config.inv_param_log_scale)}, {plot_title_override!r},
                               {plot_xlabel_override!r}, {plot_ylabel_override!r})
    print(f"Final inferred {{name}}: {{final_val:.6g}}")
plt.tight_layout()
param_plot_path = os.path.join(sol_dir, "param_convergence.png")
plt.savefig(param_plot_path, dpi={int(config.plot_dpi)})
plt.close(fig)
# Plot settings -> Figures: one per parameter, as well
for vi, name in enumerate({[n for n, _i, _t in inv_vars_parsed]!r}):
    true_val = _inv_true_vals[vi] if vi < len(_inv_true_vals) else None
    if _PARAM_STYLE["plot_separate"]:
        fig1, ax1 = plt.subplots(figsize=_plot_figsize(6, 3.2))
        _draw_param_ax(ax1, name, param_iters, param_vals[vi], true_val, {bool(config.inv_param_log_scale)},
                       {plot_title_override!r}, {plot_xlabel_override!r}, {plot_ylabel_override!r})
        plt.tight_layout()
        plt.savefig(os.path.join(sol_dir, f"{{name}}_convergence_plot.png"), dpi={int(config.plot_dpi)})
        plt.close(fig1)
    if _PARAM_STYLE["anim_separate"] and {config.plot_type == "Parameter Convergence (GIF)"}:
        _param_animation([(name, param_iters, param_vals[vi], true_val)],
                         os.path.join(sol_dir, f"{{name}}_convergence_animation.gif"),
                         {bool(config.inv_param_log_scale)}, {int(config.plot_fps)}, {int(config.plot_dpi)},
                         {plot_title_override!r}, {plot_xlabel_override!r}, {plot_ylabel_override!r}, _plot_figsize(6, 3.2))''')

    # ── Result plot -- exactly one of these, matching this problem's
    # dimension / Time-Adaptive status / selected Plot Type. No runtime
    # dispatch: which branch applies is already known now.
    plot_idx = config.plot_output_idx
    out_name = out_names[plot_idx] if plot_idx < len(out_names) else out_names[0]
    if _plot_custom_expr_val:
        out_name = (config.plot_custom_label or "").strip() or "custom"

    # ── Field to plot/analyze: one raw output column, or -- when a custom
    # expression is configured, e.g. |h| = sqrt(u**2+v**2) for 1D
    # Schrodinger's complex-valued u,v outputs, or a derivative like du_x --
    # a derived scalar field built from ALL of this problem's outputs AND
    # their derivatives (same d{{name}}_x syntax the Custom PDE box, Training
    # Monitors, and the Restore tab/live Results panel's own custom plotting
    # already use), evaluated through DeepXDE's own dde.Model.predict(x,
    # operator=...) built-in, reusing Training Monitors' own _tm_build_dvars
    # (embedded above whenever a custom expression is configured). Mirrors
    # generate_script()'s own _extract_plot_field helper (minus its multi-
    # output Error-Analysis routing, which this single-field generator
    # doesn't have) -- without this, an exported script for a custom-field
    # template always plotted and error-analyzed the raw first output
    # instead of the derived field the GUI itself shows, and (before this
    # round) couldn't reference derivatives at all -- Round 17 added
    # derivative support to the Restore tab and the live Results panel but
    # explicitly left this exporter on the old NumPy-only path; this closes
    # that gap.
    _plot_dim_clean = "3D" if is_3d else ("2D" if is_2d else "1D")
    parts.append(f'''_plot_custom_expr = {_plot_custom_expr_val!r}
_plot_idx = {plot_idx}
_plot_output_names = {out_names!r}
_plot_n_out = len(_plot_output_names)
_plot_is_steady = {is_steady}
_plot_dim = {_plot_dim_clean!r}
{_math_namespace_code("_PLOT_TORCH_MATH_NS", "torch", "")}
def _plot_custom_op(_pf_inputs, _pf_outputs):
    _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                 _plot_output_names, _plot_is_steady, _plot_dim,
                                 _plot_custom_expr)
    _pf_ns = dict(_pf_dvars)
    _pf_ns.update(_PLOT_TORCH_MATH_NS)
    _pf_ns["torch"] = torch
    return eval(_plot_custom_expr, _pf_ns)

def _extract_plot_field(_x_grid, _pf_model=None):
    _pf_m = _pf_model if _pf_model is not None else model
    if _plot_custom_expr.strip():
        return _pf_m.predict(_x_grid, operator=_plot_custom_op)[:, 0]
    return _pf_m.predict(_x_grid)[:, _plot_idx]''')

    if use_rar:
        parts.append(_clean_rar_diagnostics_code(config, is_2d, is_3d, is_steady, out_name))

    def _result_plot_parts(out_name):
        """The result-plot code for one plotted field (Round 69: called once
        for the selected output, then once per output when there are several)."""
        if use_ta:
            parts.append(f'''# ── Result plot: stitched Time-Adaptive solution ──
Xg = np.concatenate([a for a in all_x], axis=0)
Tg = np.concatenate([a for a in all_t], axis=0)
Ug = np.concatenate([a for a in all_u], axis=0)
plt.figure(figsize=_plot_figsize(7, 5))
if {config.plot_swap_xt}:
    plt.contourf(Tg, Xg, Ug, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    plt.xlabel("t"); plt.ylabel("x")
else:
    plt.contourf(Xg, Tg, Ug, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    plt.xlabel("x"); plt.ylabel("t")
if {config.plot_colorbar}:
    plt.colorbar(label="{out_name}")
plt.title("PINN Solution (Time-Adaptive)")
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_inverse and config.plot_type == "Parameter Convergence":
            parts.append('''# ── Result plot: parameter convergence (Inverse default) ──
import shutil
shutil.copy(param_plot_path, solution_path)
print(f"Solution plot saved: {solution_path}")''')

        elif is_inverse and config.plot_type == "Parameter Convergence (GIF)":
            parts.append(f'''# ── Result plot: parameter convergence animation ──
_param_animation([(name, param_iters, param_vals[vi], _inv_true_vals[vi] if vi < len(_inv_true_vals) else None)
                  for vi, name in enumerate({[n for n, _i, _t in inv_vars_parsed]!r})],
                 solution_path, {bool(config.inv_param_log_scale)}, {int(config.plot_fps)}, {int(config.plot_dpi)},
                 {plot_title_override!r}, {plot_xlabel_override!r}, {plot_ylabel_override!r}, _plot_figsize(6, 3.2))''')

        elif is_2d and is_steady and config.plot_type == "Line (time steps)":
            # Steady 2D "Line (time steps)": no time axis, so a single curve
            # u(x) at the configured y slice -- previously this selection
            # silently fell through to the x-y heatmap branch just below
            # (plot_type was never checked for 2D/3D at all in this
            # generator either). Masked to the real geometry with
            # geom.inside(...) -- same pattern generate_script()'s own
            # equivalent branch already uses.
            parts.append(f'''# ── Result plot: line at y={line_slice_y:.3g} (steady, no time axis) ──
res = {config.plot_resolution}
x_l2ds = np.linspace({config.x_min}, {config.x_max}, res)
xy_l2ds = np.column_stack([x_l2ds, np.full_like(x_l2ds, {line_slice_y})])
line_in_l2ds = geom.inside(xy_l2ds)
x_l2ds = x_l2ds[line_in_l2ds]
xy_l2ds = xy_l2ds[line_in_l2ds]
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
if len(x_l2ds) < 2:
    ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    u_l2ds = _extract_plot_field(xy_l2ds).flatten()
    ax.plot(*_line_gaps(x_l2ds, u_l2ds), color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))})")!r}, fontsize={config.plot_axis_label_fontsize})
    _grid(ax, alpha=0.2)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_2d and is_steady:
            parts.append(f'''# ── Result plot: steady-state 2D heatmap (no time axis) ──
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(res, res)
xy = np.column_stack([Xg.ravel(), Yg.ravel()])
pred = _extract_plot_field(xy).reshape(res, res)
pred = np.where(inside, pred, np.nan)
fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
im = ax.contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_aspect("equal", adjustable="box")
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle({(plot_title_override or f"PINN Solution — {out_name}(x, y)")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_3d and is_steady and config.plot_type == "Line (time steps)":
            # Steady 3D "Line (time steps)": single curve u(x) at the
            # configured y and z slice. Masked to the real geometry with
            # geom.inside(...) -- same pattern as the steady-2D Line block
            # above.
            parts.append(f'''# ── Result plot: line at y={line_slice_y:.3g}, z={line_slice_z:.3g} (steady, no time axis) ──
res = {config.plot_resolution}
x_l3ds = np.linspace({config.x_min}, {config.x_max}, res)
xyz_l3ds = np.column_stack([x_l3ds, np.full_like(x_l3ds, {line_slice_y}), np.full_like(x_l3ds, {line_slice_z})])
line_in_l3ds = geom.inside(xyz_l3ds)
x_l3ds = x_l3ds[line_in_l3ds]
xyz_l3ds = xyz_l3ds[line_in_l3ds]
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
if len(x_l3ds) < 2:
    ax.text(0.5, 0.5, "Line outside the domain at this y/z slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    u_l3ds = _extract_plot_field(xyz_l3ds).flatten()
    ax.plot(*_line_gaps(x_l3ds, u_l3ds), color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))})")!r}, fontsize={config.plot_axis_label_fontsize})
    _grid(ax, alpha=0.2)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_3d and is_steady:
            parts.append(f'''# ── Result plot: steady-state 3D heatmap at the z mid-plane (no time axis) ──
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
z_mid = {line_slice_z!r}   # Plot settings -> Line: z (the middle by default)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])).reshape(res, res)
xyz = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])
pred = _extract_plot_field(xyz).reshape(res, res)
pred = np.where(inside, pred, np.nan)
fig, ax = plt.subplots(figsize=_plot_figsize(6.5, 5.5))
im = ax.contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_aspect("equal", adjustable="box")
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle({plot_title_override!r} if {bool(plot_title_override)} else f"PINN Solution — {out_name}(x, y, z={{_fmt_num(z_mid)}})")
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_steady:
            # Steady-state 1D: a single static curve u(x) -- no time axis.
            parts.append(f'''# ── Result plot: steady-state 1D curve (no time axis) ──
res = {config.plot_resolution}
x_1d = np.linspace({config.x_min}, {config.x_max}, res)
u_1d = _extract_plot_field(x_1d.reshape(-1, 1)).flatten()
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
ax.plot(x_1d, u_1d, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth})
ax.set_xlabel({(plot_xlabel_override or v1)!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}({v1})")!r}, fontsize={config.plot_axis_label_fontsize})
ax.set_title({(plot_title_override or "PINN Solution")!r})
_grid(ax, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)"):
            n_frames = max(2, config.num_timesteps_anim)
            anim_kind = "line" if config.plot_type == "Line Animation (GIF)" else "surface"
            vrange = ("v_min, v_max = None, None" if config.plot_auto_range
                      else f"v_min, v_max = {config.plot_vmin}, {config.plot_vmax}")
            if anim_kind == "line":
                # 2D/3D previously fell through to the "else" (surface
                # animation) branch below instead -- plot_type was only
                # checked against ("Line Animation (GIF)", "Surface Animation
                # (GIF)") as a pair, never distinguishing anim_kind for 2D/3D,
                # so picking "Line Animation (GIF)" on a 2D/3D problem
                # silently produced a Surface Animation GIF instead. Now
                # handled directly, at the configured y/z slice (same
                # PINNConfig.line_slice_y/_z generate_script() uses).
                if is_3d:
                    _xt_build = (f'xt = np.column_stack([x_line, np.full_like(x_line, {line_slice_y}), '
                                 f'np.full_like(x_line, {line_slice_z}), np.full_like(x_line, tv)])')
                    _ylabel_anim = f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))}, t)"
                elif is_2d:
                    _xt_build = f'xt = np.column_stack([x_line, np.full_like(x_line, {line_slice_y}), np.full_like(x_line, tv)])'
                    _ylabel_anim = f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, t)"
                else:
                    _xt_build = 'xt = np.column_stack([x_line, np.full_like(x_line, tv)])'
                    _ylabel_anim = f"{out_name}(x, t)"
                parts.append(f'''# ── Result plot: line animation (GIF) ──
import matplotlib.animation as animation
t_frames = np.linspace({config.t_min}, {config.t_max}, {n_frames})
x_line = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
frames_u = []
for tv in t_frames:
    {_xt_build}
    frames_u.append(_extract_plot_field(xt).flatten()){chr(10) + "_in_line = np.asarray(geom.inside(xt[:, :" + ("3" if is_3d else "2") + "]), dtype=bool)" + chr(10) + "frames_u = [np.where(_in_line, u, np.nan) for u in frames_u]  # only inside the domain" if (is_2d or is_3d) else ""}
u_min = min(np.nanmin(u) for u in frames_u); u_max = max(np.nanmax(u) for u in frames_u)
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
ax.set_xlim({config.x_min}, {config.x_max})
ax.set_ylim(u_min - 0.05 * abs(u_min) - 1e-9, u_max + 0.05 * abs(u_max) + 1e-9)
ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or _ylabel_anim)!r}, fontsize={config.plot_axis_label_fontsize}); _grid(ax, alpha=0.2)
{f"ax.set_title({plot_title_override!r})" if plot_title_override else "# (no title override set)"}
line, = ax.plot([], [], color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth_anim})
time_txt = ax.text({0.02 if plot_title_override else 0.5}, {0.95 if plot_title_override else 1.02},
                   "", transform=ax.transAxes, color="black", fontsize=11,
                   ha={"left" if plot_title_override else "center"!r})   # as in Solve's animation
def _update(i):
    line.set_data(x_line, frames_u[i])
    time_txt.set_text(f"t = {{_fmt_num(t_frames[i])}}")
    return line, time_txt
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=100, blit=True)
ani.save(solution_path, writer="pillow", fps={config.plot_fps}, dpi=_PLOT_DPI_ANIM)
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
            else:
                parts.append(f'''# ── Result plot: surface animation (GIF) ──
import matplotlib.animation as animation
t_frames = np.linspace({config.t_min}, {config.t_max}, {n_frames})
res = {config.plot_resolution}
x_a = np.linspace({config.x_min}, {config.x_max}, res)
frames = []''')
                if is_3d:
                    parts.append(f'''# 3D: same "x-y heatmap at the z mid-plane" convention as the static
# 3D solution plot below -- no volumetric GIF renderer.
y_a = np.linspace({config.y_min}, {config.y_max}, res)
z_mid_a = ({config.z_min} + {config.z_max}) / 2.0
Xa, Ya = np.meshgrid(x_a, y_a)
# Blank out points outside the real geometry (same mask as the static plot).
inside_a = geom.inside(np.column_stack([Xa.ravel(), Ya.ravel(), np.full(Xa.size, z_mid_a)])).reshape(res, res)
for tv in t_frames:
    xyzt = np.column_stack([Xa.ravel(), Ya.ravel(), np.full(Xa.size, z_mid_a), np.full(Xa.size, tv)])
    frames.append(np.where(inside_a, _extract_plot_field(xyzt).reshape(res, res), np.nan))
{vrange}
if v_min is None:
    v_min = min(np.nanmin(f) for f in frames); v_max = max(np.nanmax(f) for f in frames)
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
def _update(i):
    ax.cla()
    ax.contourf(Xa, Ya, frames[i], cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, v_min, v_max, frames[i]))
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{_fmt_num(t_frames[i])}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps}, dpi=_PLOT_DPI_ANIM)
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
                elif is_2d:
                    parts.append(f'''y_a = np.linspace({config.y_min}, {config.y_max}, res)
Xa, Ya = np.meshgrid(x_a, y_a)
# Blank out points outside the real geometry (same mask as the static plot).
inside_a = geom.inside(np.column_stack([Xa.ravel(), Ya.ravel()])).reshape(res, res)
for tv in t_frames:
    xyt = np.column_stack([Xa.ravel(), Ya.ravel(), np.full(Xa.size, tv)])
    frames.append(np.where(inside_a, _extract_plot_field(xyt).reshape(res, res), np.nan))
{vrange}
if v_min is None:
    v_min = min(np.nanmin(f) for f in frames); v_max = max(np.nanmax(f) for f in frames)
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
def _update(i):
    ax.cla()
    ax.contourf(Xa, Ya, frames[i], cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, v_min, v_max, frames[i]))
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{_fmt_num(t_frames[i])}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps}, dpi=_PLOT_DPI_ANIM)
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')
                else:
                    parts.append(f'''t_a = np.linspace({config.t_min}, {config.t_max}, res)
Xa, Ta = np.meshgrid(x_a, t_a)
for tv in t_frames:
    xt = np.vstack([Xa.ravel(), np.full(Xa.size, tv)]).T
    frames.append(_extract_plot_field(xt).reshape(res, res))
{vrange}
if v_min is None:
    v_min = min(f.min() for f in frames); v_max = max(f.max() for f in frames)
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
# Same configurable axis orientation as the static "Surface" plot above
# (Plot Settings -> "Swap axes") -- no reshape of frames[i] needed, just
# swapping which of Xa/Ta is passed first to contourf (see the static
# Surface block's comment for why that alone is enough to flip axes).
def _update(i):
    ax.cla()
    if {config.plot_swap_xt}:
        ax.contourf(Ta, Xa, frames[i], cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, v_min, v_max, frames[i]))
        ax.set_xlabel({(plot_xlabel_override or "t")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize})
    else:
        ax.contourf(Xa, Ta, frames[i], cmap="{config.plot_colormap}", **_cf_levels({config.plot_levels}, v_min, v_max, frames[i]))
        ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "t")!r}, fontsize={config.plot_axis_label_fontsize})
    ax.set_title({plot_title_override!r} if {bool(plot_title_override)} else f"t = {{_fmt_num(t_frames[i])}}")
ani = animation.FuncAnimation(fig, _update, frames={n_frames}, interval=150)
ani.save(solution_path, writer="pillow", fps={config.plot_fps}, dpi=_PLOT_DPI_ANIM)
plt.close(fig)
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_2d and config.plot_type == "Line (time steps)":
            # 2D "Line (time steps)": overlaid u(x) curves at several times,
            # all at the configured y slice -- same convention as the 1D
            # "Line (time steps)" branch further below, and as
            # generate_script()'s own already-fixed equivalent. Previously
            # this selection silently fell through to the x-y heatmap branch
            # just below. Masked to the real geometry with geom.inside(...)
            # -- the mask only depends on (x, y), not t, so it's computed
            # once outside the time loop -- same pattern as the steady-2D
            # Line block above.
            parts.append(f'''# ── Result plot: line, several time steps, at y={line_slice_y:.3g} ──
n_steps_l2d = {config.num_timesteps_line}
x_l2d = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
xy_l2d = np.column_stack([x_l2d, np.full_like(x_l2d, {line_slice_y})])
line_in_l2d = geom.inside(xy_l2d)
x_l2d = x_l2d[line_in_l2d]
t_steps_l2d = np.linspace({config.t_min}, {config.t_max}, n_steps_l2d)
fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
if len(x_l2d) < 2:
    ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_l2d))
    for i, tv in enumerate(t_steps_l2d):
        xyt = np.column_stack([x_l2d, np.full_like(x_l2d, {line_slice_y}), np.full_like(x_l2d, tv)])
        u_line = _extract_plot_field(xyt).flatten()
        ax.plot(*_line_gaps(x_l2d, u_line), color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(tv)}}")
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, t)")!r}, fontsize={config.plot_axis_label_fontsize})
    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_2d:
            parts.append(f'''# ── Result plot: 2D snapshots ──
n_snaps = {config.plot_n_2d_snapshots}
t_snaps = np.linspace({config.t_min}, {config.t_max}, n_snaps)
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(res, res)
fig, axes = plt.subplots(1, n_snaps, figsize=(_plot_figsize(5, 5)[0] * n_snaps, _plot_figsize(5, 5)[1]))
if n_snaps == 1:
    axes = [axes]
for ai, tv in enumerate(t_snaps):
    xyt = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
    pred = _extract_plot_field(xyt).reshape(res, res)
    pred = np.where(inside, pred, np.nan)
    im = axes[ai].contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    axes[ai].set_title(f"t = {{_fmt_num(tv)}}"); axes[ai].set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); axes[ai].set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize})
    if {config.plot_colorbar}:
        fig.colorbar(im, ax=axes[ai])
fig.suptitle({(plot_title_override or f"PINN Solution — {out_name}(x, y, t)")!r})
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_3d and config.plot_type == "Line (time steps)":
            # 3D "Line (time steps)": overlaid u(x) curves at several times,
            # at the configured y and z slice. Masked to the real geometry
            # with geom.inside(...) -- the mask only depends on (x, y, z),
            # not t, so it's computed once outside the time loop -- same
            # pattern as the 2D time-dependent Line block above.
            parts.append(f'''# ── Result plot: line, several time steps, at y={line_slice_y:.3g}, z={line_slice_z:.3g} ──
n_steps_l3d = {config.num_timesteps_line}
x_l3d = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
xyz_l3d = np.column_stack([x_l3d, np.full_like(x_l3d, {line_slice_y}), np.full_like(x_l3d, {line_slice_z})])
line_in_l3d = geom.inside(xyz_l3d)
x_l3d = x_l3d[line_in_l3d]
t_steps_l3d = np.linspace({config.t_min}, {config.t_max}, n_steps_l3d)
fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
if len(x_l3d) < 2:
    ax.text(0.5, 0.5, "Line outside the domain at this y/z slice", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
else:
    colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_l3d))
    for i, tv in enumerate(t_steps_l3d):
        xyzt = np.column_stack([x_l3d, np.full_like(x_l3d, {line_slice_y}), np.full_like(x_l3d, {line_slice_z}), np.full_like(x_l3d, tv)])
        u_line = _extract_plot_field(xyzt).flatten()
        ax.plot(*_line_gaps(x_l3d, u_line), color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(tv)}}")
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))}, t)")!r}, fontsize={config.plot_axis_label_fontsize})
    ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif is_3d:
            parts.append(f'''# ── Result plot: 3D snapshots at the domain's z mid-plane ──
n_snaps = {config.plot_n_2d_snapshots}
t_snaps = np.linspace({config.t_min}, {config.t_max}, n_snaps)
res = {config.plot_resolution}
xp = np.linspace({config.x_min}, {config.x_max}, res)
yp = np.linspace({config.y_min}, {config.y_max}, res)
Xg, Yg = np.meshgrid(xp, yp)
z_mid = {line_slice_z!r}   # Plot settings -> Line: z (the middle by default)
inside = geom.inside(np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid)])).reshape(res, res)
fig, axes = plt.subplots(1, n_snaps, figsize=(_plot_figsize(5, 5)[0] * n_snaps, _plot_figsize(5, 5)[1]))
if n_snaps == 1:
    axes = [axes]
for ai, tv in enumerate(t_snaps):
    xyzt = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, z_mid), np.full(Xg.size, tv)])
    pred = _extract_plot_field(xyzt).reshape(res, res)
    pred = np.where(inside, pred, np.nan)
    im = axes[ai].contourf(Xg, Yg, pred, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    axes[ai].set_title(f"t = {{_fmt_num(tv)}}, z = {{_fmt_num(z_mid)}}"); axes[ai].set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); axes[ai].set_ylabel({(plot_ylabel_override or "y")!r}, fontsize={config.plot_axis_label_fontsize})
    if {config.plot_colorbar}:
        fig.colorbar(im, ax=axes[ai])
fig.suptitle({plot_title_override!r} if {bool(plot_title_override)} else f"PINN Solution — {out_name}(x, y, z={{_fmt_num(z_mid)}}, t)")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        elif config.plot_type == "Line (time steps)":
            parts.append(f'''# ── Result plot: line, several time steps ──
n_steps_plot = {config.num_timesteps_line}
x_l = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
t_steps = np.linspace({config.t_min}, {config.t_max}, n_steps_plot)
fig, ax = plt.subplots(figsize=_plot_figsize(8, 5))
colors = plt.get_cmap("{config.plot_colormap}")(np.linspace(0, 1, n_steps_plot))
for i, tv in enumerate(t_steps):
    xt = np.column_stack([x_l, np.full_like(x_l, tv)])
    u_line = _extract_plot_field(xt).flatten()
    ax.plot(x_l, u_line, color=colors[i], linewidth={config.plot_linewidth}, label=f"t = {{_fmt_num(tv)}}")
ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or f"{out_name}(x, t)")!r}, fontsize={config.plot_axis_label_fontsize})
ax.set_title({(plot_title_override or "PINN Solution")!r})
ax.legend(loc="upper right", fontsize=_lfs(8)); _grid(ax, alpha=0.2)
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

        else:  # "Surface" -- 1D x-t heatmap
            parts.append(f'''# ── Result plot: surface (x-t heatmap) ──
res = {config.plot_resolution}
x_s = np.linspace({config.x_min}, {config.x_max}, res)
t_s = np.linspace({config.t_min}, {config.t_max}, res)
Xs, Ts = np.meshgrid(x_s, t_s)
xts = np.vstack([Xs.ravel(), Ts.ravel()]).T
u_s = _extract_plot_field(xts).reshape(res, res)
fig, ax = plt.subplots(figsize=_plot_figsize(7, 5))
if {config.plot_swap_xt}:
    im = ax.contourf(Ts, Xs, u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    ax.set_xlabel({(plot_xlabel_override or "t")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize})
else:
    im = ax.contourf(Xs, Ts, u_s, levels={config.plot_levels}, cmap="{config.plot_colormap}")
    ax.set_xlabel({(plot_xlabel_override or "x")!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({(plot_ylabel_override or "t")!r}, fontsize={config.plot_axis_label_fontsize})
if {config.plot_colorbar}:
    fig.colorbar(im, ax=ax)
ax.set_title({(plot_title_override or "PINN Solution")!r})
plt.tight_layout()
plt.savefig(solution_path, dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print(f"Solution plot saved: {{solution_path}}")''')

    _result_plot_parts(out_name)
    if (config.plot_all_outputs and len(out_names) > 1 and not _plot_custom_expr_val and not use_ta
            and not (is_inverse and config.plot_type.startswith("Parameter Convergence"))):
        _ext_all = "gif" if (not is_steady and config.plot_type in ("Line Animation (GIF)", "Surface Animation (GIF)")) else "png"
        parts.append("# ── Every output's own plot, named after it ──\n_main_solution_path = solution_path")
        for _oi, _oname in enumerate(out_names):
            _osafe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in _oname.strip()) or f"out{_oi}"
            parts.append(f'_plot_idx = {_oi}\nsolution_path = os.path.join(sol_dir, "solution_plot_{_osafe}.{_ext_all}")')
            _result_plot_parts(_oname.strip())
        parts.append(f"solution_path = _main_solution_path\n_plot_idx = {plot_idx}")

    # =========================================================================
    # Error Analysis (only emitted when reference/ground-truth files are
    # configured for this problem)
    # =========================================================================
    if use_ea:
        # Shared, per-output-group extraction helper -- generated once,
        # reused by every group's own unrolled EA code block below. Mirrors
        # generate_script()'s own runtime _ea_extract closure: sel=None
        # delegates to this script's one default field (_extract_plot_field,
        # defined above), an int picks a raw output column, and a
        # [expr, label] pair builds its own derivative-aware custom field
        # via Training Monitors' _tm_build_dvars, reusing the exact same
        # operator-based mechanism _plot_custom_op (above) already uses for
        # the main plot's own custom expression.
        _ea_shared_lines = ['''def _ea_extract_group(_ea_grid, _ea_m, _ea_sel):
    if _ea_sel is None:
        return _extract_plot_field(_ea_grid, _ea_m)
    if isinstance(_ea_sel, int):
        return _ea_m.predict(_ea_grid)[:, _ea_sel]
    _ea_expr = _ea_sel[0]
    def _ea_custom_op(_pf_inputs, _pf_outputs):
        _pf_dvars = _tm_build_dvars(_pf_inputs, _pf_outputs, _plot_n_out,
                                     _plot_output_names, _plot_is_steady, _plot_dim,
                                     _ea_expr)
        _pf_ns = dict(_pf_dvars)
        _pf_ns.update(_PLOT_TORCH_MATH_NS)
        _pf_ns["torch"] = torch
        return eval(_ea_expr, _pf_ns)
    return _ea_m.predict(_ea_grid, operator=_ea_custom_op)[:, 0]''']
        if use_ta:
            _ea_shared_lines.append('''def _predict_at_time(xt_no_time, tv, sel=None):
    for t0, t1, m in ta_step_models:
        if t0 - 1e-9 <= tv <= t1 + 1e-9:
            return _ea_extract_group(np.column_stack([xt_no_time, np.full(len(xt_no_time), tv)]), m, sel).flatten()
    return _ea_extract_group(
        np.column_stack([xt_no_time, np.full(len(xt_no_time), tv)]), ta_step_models[-1][2], sel).flatten()''')
        parts.append("\n".join(_ea_shared_lines))

        # Loop over output groups AT GENERATION TIME (not a runtime loop in
        # the exported script) -- see the comment above _ea_groups (near the
        # top of this function) for why this generator groups here rather
        # than at runtime the way generate_script() does. Each iteration
        # emits one fully unrolled metrics/line/surface report, with
        # unsuffixed filenames for the (usual) single-group case -- byte-
        # for-byte the same code this function already produced before
        # multi-output grouping existed -- and a per-group suffix (e.g.
        # "_v", "_custom") whenever more than one distinct output has its
        # own reference files.
        for _ea_gkey, _ea_group in _ea_groups.items():
            _ea_sel = _ea_group["sel"]
            _ea_group_files = _ea_group["files"]
            _ea_suffix = f"_{_ea_group_label(_ea_sel) or 'default'}" if _ea_multi_output else ""
            _ea_header_note = f" (output: {_ea_group_label(_ea_sel) or 'default'})" if _ea_multi_output else ""
            ea_lines = [f'''# ── Error Analysis: compare against reference data{_ea_header_note} ──
ea_dir = os.path.join(sol_dir, "error_analysis")
os.makedirs(ea_dir, exist_ok=True)
ea_files = {_ea_group_files}
ea_times, ea_x_refs, ea_y_refs, ea_z_refs, ea_u_refs = [], [], [], [], []
for tv, fp in ea_files:
    d = np.loadtxt(fp)
    if d.ndim == 1:
        d = d.reshape(1, -1)''']
            # Same distinction as _n_obs_coord_cols/_n_coord_cols_bc elsewhere in
            # this file: a steady-state reference file (PINNStudio's own
            # "solution.txt", written by _auto_configure_ea's is_steady branch
            # in main_window.py) has no time column at all -- x,y,u for 2D /
            # x,y,z,u for 3D -- one fewer column than the time-dependent layout,
            # with every column after the coordinates shifted down by one.
            # Getting this wrong doesn't raise anywhere near the mistake: it
            # reads a steady 2D file's u column as if it were a time column and
            # crashes with an out-of-bounds IndexError on the (nonexistent)
            # column after it instead.
            if is_3d:
                if is_steady:
                    ea_lines.append('''    idx = np.lexsort((d[:, 2], d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(d[idx, 2])
    ea_u_refs.append(d[idx, 3]); ea_times.append(float(tv))''')
                else:
                    ea_lines.append('''    idx = np.lexsort((d[:, 2], d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(d[idx, 2])
    ea_u_refs.append(d[idx, 4]); ea_times.append(float(d[0, 3]))''')
            elif is_2d:
                if is_steady:
                    ea_lines.append('''    idx = np.lexsort((d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 2]); ea_times.append(float(tv))''')
                else:
                    ea_lines.append('''    idx = np.lexsort((d[:, 1], d[:, 0]))
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(d[idx, 1]); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 3]); ea_times.append(float(d[0, 2]))''')
            else:
                if is_steady:
                    ea_lines.append('''    idx = np.argsort(d[:, 0])
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(np.zeros_like(d[idx, 0])); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 1]); ea_times.append(float(tv))''')
                else:
                    ea_lines.append('''    idx = np.argsort(d[:, 0])
    ea_x_refs.append(d[idx, 0]); ea_y_refs.append(np.zeros_like(d[idx, 0])); ea_z_refs.append(np.zeros_like(d[idx, 0]))
    ea_u_refs.append(d[idx, 2]); ea_times.append(float(tv))''')
            # Drop reference points with no solution value (NaN) -- e.g. grid
            # points outside a non-rectangular geometry like L-Shape's missing
            # quadrant or Disk's bounding-box corners. A no-op for reference
            # files that don't have any (the usual case).
            ea_lines.append('''for _ei in range(len(ea_u_refs)):
    _ea_valid = ~np.isnan(ea_u_refs[_ei])
    if not _ea_valid.all():
        _n_dropped = int((~_ea_valid).sum())
        ea_x_refs[_ei] = ea_x_refs[_ei][_ea_valid]
        ea_y_refs[_ei] = ea_y_refs[_ei][_ea_valid]
        ea_z_refs[_ei] = ea_z_refs[_ei][_ea_valid]
        ea_u_refs[_ei] = ea_u_refs[_ei][_ea_valid]
        print(f"  Dropped {_n_dropped} NaN reference point(s) outside the geometry")''')
            ea_lines.append('''order = np.argsort(ea_times)
ea_times  = [ea_times[i] for i in order]
ea_x_refs = [ea_x_refs[i] for i in order]
ea_y_refs = [ea_y_refs[i] for i in order]
ea_z_refs = [ea_z_refs[i] for i in order]
ea_u_refs = [ea_u_refs[i] for i in order]
n_t = len(ea_times)''')

            if use_ta:
                # _predict_at_time itself is defined once, shared across all
                # output groups, in the pre-loop setup appended just before
                # this per-group loop begins (see _ea_shared_lines above) --
                # only the loop header is needed here.
                ea_lines.append('''ea_u_pinns = []
for i, tv in enumerate(ea_times):''')
                if is_3d:
                    ea_lines.append(f'''    coords = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]])
    ea_u_pinns.append(_predict_at_time(coords, tv, {_ea_sel!r}))''')
                elif is_2d:
                    ea_lines.append(f'''    coords = np.column_stack([ea_x_refs[i], ea_y_refs[i]])
    ea_u_pinns.append(_predict_at_time(coords, tv, {_ea_sel!r}))''')
                else:
                    ea_lines.append(f'''    coords = ea_x_refs[i].reshape(-1, 1)
    ea_u_pinns.append(_predict_at_time(coords, tv, {_ea_sel!r}))''')
            else:
                ea_lines.append("ea_u_pinns = []")
                ea_lines.append("for i, tv in enumerate(ea_times):")
                # A steady-state model was trained on (x[, y[, z]]) alone -- no
                # time input at all -- so predict() must not be fed a time
                # column here either (same distinction as the ground-truth
                # loader above); doing so is what produced the earlier
                # "mat1 and mat2 shapes cannot be multiplied" crash.
                if is_3d:
                    if is_steady:
                        ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]])
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')
                    else:
                        ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')
                elif is_2d:
                    if is_steady:
                        ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i]])
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')
                    else:
                        ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], ea_y_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')
                else:
                    if is_steady:
                        ea_lines.append(f'''    xt = ea_x_refs[i].reshape(-1, 1)
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')
                    else:
                        ea_lines.append(f'''    xt = np.column_stack([ea_x_refs[i], np.full_like(ea_x_refs[i], tv)])
    ea_u_pinns.append(_ea_extract_group(xt, model, {_ea_sel!r}).flatten())''')

            ea_lines.append(f'''
# ── Metrics ──
ea_metrics = []
for i, tv in enumerate(ea_times):
    up, uf = ea_u_pinns[i], ea_u_refs[i]
    l2 = np.linalg.norm(up - uf) / (np.linalg.norm(uf) + 1e-10)
    mse = np.mean((up - uf) ** 2)
    mx = np.max(np.abs(up - uf))
    ma = np.mean(np.abs(up - uf))
    ea_metrics.append((tv, l2, mse, mx, ma))
    print(f"  t={{tv:.4f}} -- {_ea_metrics_print_fragment(config, 'l2', 'mse', 'mx', 'ma')}")
with open(os.path.join(ea_dir, "error_metrics{_ea_suffix}.txt"), "w") as f:
    f.write("{_ea_metrics_csv_header(config)}\\n")
    for tv, l2, mse, mx, ma in ea_metrics:
        f.write(f"{_ea_metrics_row_code(config, 'tv', 'l2', 'mse', 'mx', 'ma')}\\n")''')

            if config.ea_do_line:
                if is_2d or is_3d:
                    # A true line at the exact slice (PINNConfig.line_slice_y/_z),
                    # not a scatter of whichever reference-file points happened
                    # to fall within a tolerance BAND around it -- that approach
                    # smeared multiple different-y (or y,z) reference/PINN values
                    # together at the same x whenever u varies meaningfully with
                    # y within the band, which is exactly what made this plot
                    # look like a dense cloud instead of a clean line for cases
                    # like the L-Shape Poisson problem (worse right near its
                    # reentrant-corner singularity). Mirrors the same fix already
                    # made in generate_script()'s copy of this block, and reuses
                    # the exact-eval + griddata-interpolate + geom.inside()-mask
                    # pattern the (untouched) 2D/3D Surface Comparison blocks
                    # already use elsewhere in this function: the PINN is
                    # evaluated directly on the exact line (no interpolation
                    # needed -- the model can be queried anywhere), the Reference
                    # is interpolated onto that same line from its own scattered
                    # points via scipy's griddata, and both are masked with
                    # geom.inside(...) so a non-rectangular domain's excluded
                    # region (e.g. L-Shape's missing quadrant) is never plotted
                    # straight across.
                    _ea_geom_ref = "_build_geom()" if use_ta else "geom"
                    if use_ta:
                        _line_pred_line = f"    pn = _predict_at_time(_line_pts_in, tv, {_ea_sel!r})"
                    elif is_steady:
                        _line_pred_line = f"    pn = _ea_extract_group(_line_pts_in, model, {_ea_sel!r}).flatten()"
                    else:
                        _line_pred_line = f"    pn = _ea_extract_group(np.column_stack([_line_pts_in, np.full(len(xv), tv)]), model, {_ea_sel!r}).flatten()"
                    if is_3d:
                        mid_slice = f'''
    _line_geom = {_ea_geom_ref}
    xv = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
    _line_pts = np.column_stack([xv, np.full_like(xv, {line_slice_y}), np.full_like(xv, {line_slice_z})])
    _line_in = _line_geom.inside(_line_pts)
    xv = xv[_line_in]
    _line_pts_in = _line_pts[_line_in]
    if len(xv) < 2:
        ax.text(0.5, 0.5, "Line outside the domain at this (y, z) slice", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        continue
{_line_pred_line}
    _ref_pts_line = np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]])
    gt = griddata(_ref_pts_line, ea_u_refs[i], _line_pts_in, method="linear")
    _gt_nan_line = np.isnan(gt)
    if _gt_nan_line.any():
        gt[_gt_nan_line] = griddata(_ref_pts_line, ea_u_refs[i], _line_pts_in[_gt_nan_line], method="nearest")'''
                    else:
                        mid_slice = f'''
    _line_geom = {_ea_geom_ref}
    xv = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
    _line_pts = np.column_stack([xv, np.full_like(xv, {line_slice_y})])
    _line_in = _line_geom.inside(_line_pts)
    xv = xv[_line_in]
    _line_pts_in = _line_pts[_line_in]
    if len(xv) < 2:
        ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        continue
{_line_pred_line}
    _ref_pts_line = np.column_stack([ea_x_refs[i], ea_y_refs[i]])
    gt = griddata(_ref_pts_line, ea_u_refs[i], _line_pts_in, method="linear")
    _gt_nan_line = np.isnan(gt)
    if _gt_nan_line.any():
        gt[_gt_nan_line] = griddata(_ref_pts_line, ea_u_refs[i], _line_pts_in[_gt_nan_line], method="nearest")'''
                    sort_line = ""
                    _ea_line_needs_griddata = True
                else:
                    mid_slice = ""
                    sort_line = "    order_i = np.argsort(ea_x_refs[i]); xv, gt, pn = ea_x_refs[i][order_i], ea_u_refs[i][order_i], ea_u_pinns[i][order_i]"
                    _ea_line_needs_griddata = False
                # Steady-state (e.g. a Poisson equation) has no time axis at
                # all -- every reference file is really just a single snapshot
                # at a placeholder t=0, so "t = 0.000" in the title would be
                # meaningless noise rather than a real time coordinate.
                _ea_line_title = 'f"({_l2_txt(l2)})"' if is_steady else 'f"t = {_fmt_num(tv)}  ({_l2_txt(l2)})"'
                if is_3d:
                    _ea_line_ylabel = repr(f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))})") if is_steady else repr(f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))}, t)")
                    _ea_line_suptitle = f" (y={fmt_num(line_slice_y, number_settings(config))}, z={fmt_num(line_slice_z, number_settings(config))})"
                elif is_2d:
                    _ea_line_ylabel = repr(f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))})") if is_steady else repr(f"{out_name}(x, y={fmt_num(line_slice_y, number_settings(config))}, t)")
                    _ea_line_suptitle = f" (y={fmt_num(line_slice_y, number_settings(config))})"
                else:
                    _ea_line_ylabel = repr(f"{out_name}({v1})") if is_steady else repr(f"{out_name}(x, t)")
                    _ea_line_suptitle = ""
                # One panel: a 6 x 4.5 figure whose only title carries the
                # error; several times: one panel each, one figure title.
                _ea_line_suptitle = "PINN vs Reference — " + out_name + _ea_line_suptitle
                ea_lines.append(f'''
# ── Line comparison ──{chr(10) + "from scipy.interpolate import griddata" if _ea_line_needs_griddata else ""}
ncols = min({max(1, int(getattr(config, 'plot_ea_cols', 4) or 4))}, n_t); nrows = (n_t + ncols - 1) // ncols
single = n_t == 1
fig, axes = plt.subplots(nrows, ncols, figsize=_panel_figsize(6, 4.5, 1, 1) if single else _panel_figsize(4.5, 3.8, ncols, nrows), squeeze=False)
line_head = {_ea_line_suptitle!r}
if not single:
    fig.suptitle(line_head, fontsize={config.plot_subplot_title_fontsize} + 2)
axf = axes.flatten()
for i, tv in enumerate(ea_times):
    ax = axf[i]{mid_slice}
{sort_line}
    tv_r, l2, mse, mx, ma = ea_metrics[i]{chr(10) + "    xv, gt, pn = _line_gaps(xv, gt, pn)" if (is_2d or is_3d) else ""}
    _ref_plot(ax, xv, gt, color=_PLOT_REF_COLOR, linewidth={config.plot_linewidth}, label="Reference")
    ax.plot(xv, pn, color=_PLOT_PINN_COLOR, linewidth={config.plot_linewidth}, linestyle="--", label="PINN")
    ax.set_title(line_head + "     " + {_ea_line_title} if single else {_ea_line_title},
                 fontsize={config.plot_subplot_title_fontsize} + (2 if single else 0))
    ax.set_xlabel({v1!r}, fontsize={config.plot_axis_label_fontsize}); ax.set_ylabel({_ea_line_ylabel}, fontsize={config.plot_axis_label_fontsize}); _grid(ax, alpha=0.3)
for j in range(n_t, len(axf)):
    axf[j].set_visible(False)
axf[0].legend(loc="best", fontsize=_lfs(10), framealpha=0.9)   # inside the panel, in its emptiest corner
plt.tight_layout()
plt.savefig(os.path.join(ea_dir, "line_comparison{_ea_suffix}.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Line comparison saved.")''')

            if config.ea_do_surface and not (is_2d or is_3d):
                # A 1D x-t surface needs >= 2 distinct time snapshots to form a
                # non-degenerate grid -- unguarded, a single reference file (the
                # steady-state case always has exactly one meaningful file,
                # since there's no time axis to vary over) crashed matplotlib's
                # contourf with "Input z must be at least a (2, 2) shaped
                # array" instead of skipping gracefully the way the live Solve
                # path's own copy of this section already does (the line
                # comparison above already covers a single snapshot either way).
                _skip_msg_1d = ("steady-state 1D has no time axis to build an x-t surface from; see the line comparison above"
                                 if is_steady else "need at least 2 time snapshots for a 1D x-t surface plot")
                ea_lines.append(f'''
# ── Surface comparison (1D: x-t) ──
if len(ea_times) < 2:
    print("  Skipping surface comparison -- {_skip_msg_1d}")
else:
    from scipy.interpolate import interp1d
    x_common = np.linspace({config.x_min}, {config.x_max}, {config.plot_resolution})
    t_arr = np.array(ea_times)
    U_pinn = np.zeros((len(t_arr), len(x_common)))
    U_ref  = np.zeros((len(t_arr), len(x_common)))
    for i, tv in enumerate(ea_times):
        fi_p = interp1d(ea_x_refs[i], ea_u_pinns[i], kind="linear", fill_value="extrapolate")
        fi_r = interp1d(ea_x_refs[i], ea_u_refs[i], kind="linear", fill_value="extrapolate")
        U_pinn[i, :] = fi_p(x_common); U_ref[i, :] = fi_r(x_common)
    Xg, Tg = np.meshgrid(x_common, t_arr)
    U_err = np.abs(U_pinn - U_ref)
    # PINN and Reference share one color scale (same as the live Solve path).
    _ea_cf = _cf_levels({config.plot_levels}, *_vr(np.nanmin([U_pinn, U_ref]), np.nanmax([U_pinn, U_ref])), U_pinn, U_ref)
    fig, axes = plt.subplots(1, 3, figsize=_panel_figsize(5, 5, 3, 1))
    fig.suptitle("PINN vs Reference -- Surface Comparison", fontsize={config.plot_title_fontsize}, fontweight="bold")
    im0 = axes[0].contourf(Tg, Xg, U_pinn, cmap="{config.plot_colormap}", **_ea_cf)
    axes[0].set_title("PINN"); axes[0].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[0].set_ylabel("x", fontsize={config.plot_axis_label_fontsize}); fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(Tg, Xg, U_ref, cmap="{config.plot_colormap}", **_ea_cf)
    axes[1].set_title("Reference"); axes[1].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[1].set_ylabel("x", fontsize={config.plot_axis_label_fontsize}); fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].contourf(Tg, Xg, U_err, levels={config.plot_levels}, cmap="{err_cmap(config)}")
    axes[2].set_title("|Error|"); axes[2].set_xlabel("t", fontsize={config.plot_axis_label_fontsize}); axes[2].set_ylabel("x", fontsize={config.plot_axis_label_fontsize}); fig.colorbar(im2, ax=axes[2])
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(ea_dir, "surface_comparison{_ea_suffix}.png"), dpi={config.plot_dpi}, bbox_inches="tight")
    plt.close()
    print("  Surface comparison saved.")''')
            elif config.ea_do_surface and is_2d:
                # See the matching comment on the Line comparison title above --
                # steady-state has no time axis, so "t=..." is dropped here too.
                _ea_pinn_title = 'f"PINN  {_l2_txt(l2)}"' if is_steady else 'f"PINN  t={_fmt_num(tv)}  {_l2_txt(l2)}"'
                _ea_gt_title = '"Reference"' if is_steady else 'f"Reference  t={_fmt_num(tv)}"'
                # Same geometry source as the 3D block below: Time-Adaptive
                # never keeps a bare module-level `geom` around.
                _ea_geom2d_ref = "_build_geom()" if use_ta else "geom"
                ea_lines.append(f'''
# ── Surface comparison (2D heatmaps) ──
from scipy.interpolate import griddata
res_ea = {config.plot_resolution}
xg_ea = np.linspace({config.x_min}, {config.x_max}, res_ea)
yg_ea = np.linspace({config.y_min}, {config.y_max}, res_ea)
Xg_ea, Yg_ea = np.meshgrid(xg_ea, yg_ea)
# Blank out grid points outside the real geometry (same as the live Solve
# path): no extrapolated PINN values and no fabricated reference values there.
inside_ea = {_ea_geom2d_ref}.inside(np.column_stack([Xg_ea.ravel(), Yg_ea.ravel()])).reshape(res_ea, res_ea)
fig, axes = plt.subplots(n_t, 3, figsize=_panel_figsize(5, 4, 3, n_t), squeeze=False)
fig.suptitle("PINN vs Reference -- 2D Heatmaps", fontsize={config.plot_title_fontsize}, fontweight="bold")
for i, tv in enumerate(ea_times):
    tv_r, l2, mse, mx, ma = ea_metrics[i]
    xyt_grid = np.column_stack([Xg_ea.ravel(), Yg_ea.ravel()]{"" if is_steady else " + [np.full(Xg_ea.size, tv)]"})
    u_pinn_grid = np.where(inside_ea, _ea_extract_group(xyt_grid, model, {_ea_sel!r}).reshape(res_ea, res_ea), np.nan)
    u_ref_grid = np.where(inside_ea, griddata(np.column_stack([ea_x_refs[i], ea_y_refs[i]]), ea_u_refs[i], (Xg_ea, Yg_ea), method="linear", fill_value=np.nan), np.nan)
    u_err_grid = np.abs(u_pinn_grid - u_ref_grid)
    _ea_cf = _cf_levels({config.plot_levels}, *_vr(np.nanmin([u_pinn_grid, u_ref_grid]), np.nanmax([u_pinn_grid, u_ref_grid])), u_pinn_grid, u_ref_grid)
    im0 = axes[i][0].contourf(Xg_ea, Yg_ea, u_pinn_grid, cmap="{config.plot_colormap}", **_ea_cf)
    axes[i][0].set_title({_ea_pinn_title}); fig.colorbar(im0, ax=axes[i][0])
    im1 = axes[i][1].contourf(Xg_ea, Yg_ea, u_ref_grid, cmap="{config.plot_colormap}", **_ea_cf)
    axes[i][1].set_title({_ea_gt_title}); fig.colorbar(im1, ax=axes[i][1])
    im2 = axes[i][2].contourf(Xg_ea, Yg_ea, u_err_grid, levels={config.plot_levels}, cmap="{err_cmap(config)}")
    axes[i][2].set_title(f"|Error|  max = {{_sci_txt(mx)}}"); fig.colorbar(im2, ax=axes[i][2])
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(ea_dir, "surface_comparison{_ea_suffix}.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Surface comparison saved.")''')
            elif config.ea_do_surface and is_3d:
                # Same steady-vs-time-dependent distinction as the metrics
                # block above (see the "mat1 and mat2 shapes cannot be
                # multiplied" comment there): a steady-state model takes no
                # time input at all, so predict() must not be fed a time
                # column here either. This also now routes through
                # _extract_plot_field like every other prediction site in
                # this function, instead of a raw [:, plot_idx] column --
                # previously a custom derived plot field (e.g. a future
                # 3D template's own |h|-style field) would have been silently
                # ignored in this one plot while the metrics/line-comparison
                # plots above already used it correctly.
                _ea_3d_predict_line = (
                    f"    pinn_b = _ea_extract_group(np.column_stack([bx, by, bz]), model, {_ea_sel!r}).flatten()"
                    if is_steady else
                    f"    pinn_b = _ea_extract_group(np.column_stack([bx, by, bz, np.full_like(bx, tv)]), model, {_ea_sel!r}).flatten()"
                )
                # Time-Adaptive never keeps a bare `geom` variable around (it only
                # ever builds per-step geometry via _build_geom()/geom_i, scoped
                # to that step's loop iteration) -- the Standard path's own
                # geom_line assignment is what defines `geom` here.
                _ea_geom_ref = "_build_geom()" if use_ta else "geom"
                # See the matching comment on the Line comparison title above --
                # steady-state has no time axis, so "t=..." is dropped here too.
                _ea_pinn_title_3d = 'f"PINN  {_l2_txt(l2)}"' if is_steady else 'f"PINN  t={_fmt_num(tv)}  {_l2_txt(l2)}"'
                _ea_gt_title_3d = '"Reference"' if is_steady else 'f"Reference  t={_fmt_num(tv)}"'
                ea_lines.append(f'''
# ── Surface comparison (3D: boundary-point scatter vs reference) ──
fig = plt.figure(figsize=_panel_figsize(5, 4.5, 3, n_t))
fig.suptitle("PINN vs Reference -- 3D Comparison", fontsize={config.plot_title_fontsize}, fontweight="bold")
_ea_geom3d = {_ea_geom_ref}
for i, tv in enumerate(ea_times):
    tv_r, l2, mse, mx, ma = ea_metrics[i]
    bnd = _ea_geom3d.on_boundary(np.column_stack([ea_x_refs[i], ea_y_refs[i], ea_z_refs[i]]))
    # A Cartesian reference grid only reliably intersects a FLAT boundary
    # (e.g. a Cuboid's 6 faces, which are themselves subsets of the grid's
    # own axis-aligned planes) in large numbers. A CURVED boundary (e.g. a
    # Sphere) is essentially never landed on exactly by an axis-aligned
    # grid -- for a unit sphere, only the 6 axis points satisfy x^2+y^2+z^2=1
    # exactly -- so on_boundary() can come back with a handful of
    # coincidental exact matches even when the dataset has thousands of
    # valid points. The old "< 4 points" fallback didn't catch this (6
    # matches is already >= 4), so it silently plotted a near-empty
    # scatter instead of a real comparison. Falling back whenever the
    # boundary subset is a small fraction of the available data -- not
    # just when it's almost empty -- fixes curved geometries without
    # changing anything for flat ones (a Cuboid's face points are normally
    # a large share of its grid, well above this threshold).
    if bnd.sum() < 4 or bnd.sum() < 0.05 * len(ea_x_refs[i]):
        bnd = np.ones_like(ea_x_refs[i], dtype=bool)
    bx, by, bz = ea_x_refs[i][bnd], ea_y_refs[i][bnd], ea_z_refs[i][bnd]
    gt_b = ea_u_refs[i][bnd]
    # Cap how many points actually get scattered -- falling back to a
    # dense volumetric grid (tens of thousands of points) is slow to
    # render and visually saturates a 3D scatter; a few thousand points
    # is still plenty to read the solution's shape clearly.
    if len(bx) > 3000:
        _ea_sub = np.random.default_rng(0).choice(len(bx), size=3000, replace=False)
        bx, by, bz, gt_b = bx[_ea_sub], by[_ea_sub], bz[_ea_sub], gt_b[_ea_sub]
{_ea_3d_predict_line}
    err_b = np.abs(pinn_b - gt_b)
    cols = [(pinn_b, {_ea_pinn_title_3d}), (gt_b, {_ea_gt_title_3d}), (err_b, f"|Error|  max = {{_sci_txt(mx)}}")]
    _ea_v3 = _vr(float(np.nanmin([pinn_b, gt_b])), float(np.nanmax([pinn_b, gt_b])))
    for ci, (vals, ttl) in enumerate(cols):
        ax3 = fig.add_subplot(n_t, 3, i * 3 + ci + 1, projection="3d")
        # PINN and Reference (ci 0/1) share one scale; |Error| keeps its own.
        _ea_vr = dict(vmin=_ea_v3[0], vmax=_ea_v3[1]) if ci < 2 else {{}}
        sc = ax3.scatter(bx, by, bz, c=vals, cmap=("{err_cmap(config)}" if ci == 2 else "{config.plot_colormap}"), s=14, **_ea_vr)
        fig.colorbar(sc, ax=ax3, shrink=0.6, pad=0.12)
        ax3.set_title(ttl, fontsize={config.plot_subplot_title_fontsize})
        ax3.set_xlabel("x", fontsize={config.plot_axis_label_fontsize}); ax3.set_ylabel("y", fontsize={config.plot_axis_label_fontsize}); ax3.set_zlabel("z", fontsize={config.plot_axis_label_fontsize}); [_tk.set_ticks(plt.matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(*_tk.get_data_interval())) for _tk in (ax3.xaxis, ax3.yaxis, ax3.zaxis)]
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(ea_dir, "surface_comparison{_ea_suffix}.png"), dpi={config.plot_dpi}, bbox_inches="tight")
plt.close()
print("  Surface comparison saved.")''')

            parts.append("\n".join(ea_lines))

    if any_cbs and config.training_monitors_enabled:
        _tm_clean_plot_code = _build_training_monitors_plot_code(config, "(save_dir or '.')", indent=0).replace("_os.path.join", "os.path.join")
        if _tm_clean_plot_code:
            parts.append(_tm_clean_plot_code)

    parts.append('print("DONE")')
    _script = "\n\n".join(parts) + "\n"
    if _const_header:
        # After the module docstring, if the script starts with one.
        _cut = _script.find('"""', 3) + 3 if _script.startswith('"""') else 0
        _script = _script[:_cut] + ("\n" if _cut else "") + "\n".join(_const_header) + "\n" + _script[_cut:]
    return _script
