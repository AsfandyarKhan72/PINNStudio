"""Where the line plots of a 2D/3D problem cut the domain.

Line plots show an output along x at one y (and z, in 3D). The automatic
choice used to be the middle of the domain, which can be a line where
nothing happens: Helmholtz 2D (u = sin(k0 x) sin(k0 y)) is exactly zero on
y = 0, and with a hole in the middle the line also crosses it. The middle
is still used whenever it shows the solution reasonably well; otherwise the
line moves to where the reference solution varies most along x and stays
inside the domain.
"""
import json
import os

import numpy as np

KEEP_MIDDLE = 0.05      # keep the middle unless it shows under 5% of the best line's variation (a line where nothing happens)
N_CAND = 21             # candidate lines per axis (the ends are skipped)

_cache = {}


def _reference_points(config):
    """(points, values) of the plotted output's reference at the last time
    (columns x, y[, z]), or None when there is none to read."""
    is_3d = config.problem_dim == "3D"
    n_sp = 3 if is_3d else 2
    mode = (getattr(config, "ea_ref_mode", "files") or "files")
    if mode == "formula":
        f = _formula_fn(config)
        return None if f is None else ("formula", f)
    files = config.ea_files
    if isinstance(files, str):
        try:
            files = json.loads(files)
        except ValueError:
            try:
                import ast
                files = ast.literal_eval(files)
            except (ValueError, SyntaxError):
                return None
    oi = int(getattr(config, "plot_output_idx", 0) or 0)
    paths = []
    for f in files or []:
        if isinstance(f, dict):
            p, grp = f.get("path"), f.get("output")
        elif isinstance(f, (list, tuple)) and len(f) > 1:
            p, grp = f[1], (f[2] if len(f) > 2 else None)
        else:
            continue
        if grp not in (None, oi, str(oi)) and not isinstance(grp, str):
            continue
        if p and os.path.isfile(p):
            paths.append(p)
    if not paths:
        return None
    path = paths[-1]
    key = (path, os.path.getmtime(path))
    if key not in _cache:
        try:
            a = np.loadtxt(path, delimiter="," if path.lower().endswith(".csv") else None, ndmin=2)
        except (ValueError, OSError):
            try:
                a = np.loadtxt(path, delimiter="," if path.lower().endswith(".csv") else None, ndmin=2, skiprows=1)
            except (ValueError, OSError):
                a = None
        _cache.clear()
        _cache[key] = a
    a = _cache[key]
    if a is None or a.shape[1] < n_sp + 1:
        return None
    return "points", (a[:, :n_sp], a[:, -1])


def _formula_fn(config):
    """u(points) from the plotted output's reference formula at the last
    comparison time, or None."""
    from pinnstudio.core.codegen import _formula_reference_entries, _np_math_expr, _constant_values, _MATH_NS_ROWS
    from pinnstudio.core import constants as _cmod
    ents = _formula_reference_entries(config)
    if not ents:
        return None
    oi = int(getattr(config, "plot_output_idx", 0) or 0)
    mine = [e for e in ents if e[2] == oi] or ents
    t, _p, _i, expr = mine[-1]
    try:
        consts = _constant_values(config)
        if consts:
            expr = _cmod.substitute(expr, consts)
    except Exception:
        pass
    code = compile(_np_math_expr(expr), "<reference>", "eval")
    base = {f: getattr(np, f) for row in _MATH_NS_ROWS for f in row if hasattr(np, f)}
    base.update(np=np, pi=np.pi, t=t)
    try:
        from scipy.special import erf
        base["erf"] = erf
    except ImportError:
        pass

    def f(pts):
        ns = dict(base, x=pts[:, 0], y=pts[:, 1])
        if pts.shape[1] > 2:
            ns["z"] = pts[:, 2]
        return np.broadcast_to(np.asarray(eval(code, ns), dtype=float), (len(pts),))
    return f


def _inside_fn(config):
    """inside(points) -> bool array for the problem's domain (its bounding
    box when the shape can't be read)."""
    try:
        if config.problem_dim == "3D":
            from pinnstudio.ui.sketch3d import in_region3
            shapes = _shapes3(config)
            return lambda pts: np.array([in_region3(shapes, tuple(q)) for q in pts], dtype=bool)
        from pinnstudio.ui.sketch_edges import in_region
        shapes = _shapes2(config)
        return lambda pts: np.array([in_region(shapes, float(q[0]), float(q[1])) for q in pts], dtype=bool)
    except Exception:
        return lambda pts: np.ones(len(pts), dtype=bool)


def _shapes2(config):
    t = config.geometry_type
    if t == "Custom":
        return json.loads(config.geom_custom_shapes_json or "[]")
    if t == "Disk":
        return [{"type": "Disk", "params": {"cx": config.geom_center_x, "cy": config.geom_center_y, "r": config.geom_radius}}]
    if t == "Ellipse":
        return [{"type": "Ellipse", "params": {"cx": config.geom_center_x, "cy": config.geom_center_y,
                                               "a": config.geom_semi_major, "b": config.geom_semi_minor,
                                               "angle": config.geom_angle}}]
    if t == "Triangle":
        return [{"type": "Triangle", "params": {"vertices_text": config.geom_triangle_vertices}}]
    if t == "Polygon":
        return [{"type": "Polygon", "params": {"vertices_text": config.geom_polygon_vertices}}]
    return [{"type": "Rectangle", "params": {"x_min": config.x_min, "x_max": config.x_max,
                                             "y_min": config.y_min, "y_max": config.y_max}}]


def _shapes3(config):
    t = config.geometry_type
    if t == "Custom":
        return json.loads(config.geom_custom_shapes_json or "[]")
    if t == "Sphere":
        return [{"type": "Sphere", "params": {"cx": config.geom_center_x, "cy": config.geom_center_y,
                                              "cz": config.geom_center_z, "r": config.geom_radius}}]
    return [{"type": "Cuboid", "params": {"x_min": config.x_min, "x_max": config.x_max, "y_min": config.y_min,
                                          "y_max": config.y_max, "z_min": config.z_min, "z_max": config.z_max}}]


def _score(pts, u, sel, x_lo, x_hi):
    """How well the points in `sel` show the solution along x: its spread
    (2nd to 98th percentile) times the share of x the line covers."""
    if sel.sum() < 4:
        return 0.0
    v = u[sel]
    spread = float(np.percentile(v, 98) - np.percentile(v, 2))
    bins = np.unique(np.clip(((pts[sel, 0] - x_lo) / max(x_hi - x_lo, 1e-12) * 40).astype(int), 0, 39))
    return spread * len(bins) / 40.0


def auto_line_slice(config):
    """(y, z) for the line plots: the user's values when set, else the
    middle of the domain, or a better line when the middle shows little."""
    y_mid = (config.y_min + config.y_max) / 2.0
    z_mid = (config.z_min + config.z_max) / 2.0
    y = y_mid if getattr(config, "line_slice_y_auto", True) else config.line_slice_y
    z = z_mid if getattr(config, "line_slice_z_auto", True) else config.line_slice_z
    if config.problem_dim not in ("2D", "3D"):
        return y, z
    is_3d = config.problem_dim == "3D"
    auto_y = getattr(config, "line_slice_y_auto", True)
    auto_z = is_3d and getattr(config, "line_slice_z_auto", True)
    if not (auto_y or auto_z):
        return y, z
    try:
        ref = _reference_points(config)
    except Exception:
        ref = None
    if ref is None:
        return y, z
    kind, ref = ref
    hy = (config.y_max - config.y_min) / (N_CAND - 1) / 2.0
    hz = (config.z_max - config.z_min) / (N_CAND - 1) / 2.0
    ys = np.linspace(config.y_min, config.y_max, N_CAND)[1:-1] if auto_y else np.array([y])
    zs = (np.linspace(config.z_min, config.z_max, N_CAND)[1:-1] if auto_z else np.array([z])) if is_3d else [None]
    if kind == "points":
        pts, u = ref
        ok = np.isfinite(u)
        pts, u = pts[ok], u[ok]
        if len(u) < 20 or np.ptp(u) == 0:
            return y, z

        def score_at(yc, zc):
            s = np.abs(pts[:, 1] - yc) <= hy * 1.0001
            if zc is not None:
                s &= np.abs(pts[:, 2] - zc) <= hz * 1.0001
            return _score(pts, u, s, config.x_min, config.x_max)
    else:
        f, inside = ref, _inside_fn(config)
        xs = np.linspace(config.x_min, config.x_max, 81)

        def score_at(yc, zc):
            cols = [xs, np.full_like(xs, yc)] + ([np.full_like(xs, zc)] if zc is not None else [])
            line = np.column_stack(cols)
            m = inside(line)
            if m.sum() < 4:
                return 0.0
            try:
                v = f(line[m])
            except Exception:
                return 0.0
            v = v[np.isfinite(v)]
            if len(v) < 4:
                return 0.0
            return float(np.percentile(v, 98) - np.percentile(v, 2)) * m.mean()

    mid = score_at(y, z if is_3d else None)
    best, best_d, best_yz = -1.0, float("inf"), (y, z)
    for yc in ys:
        for zc in zs:
            s = score_at(yc, zc)
            # the line nearest the middle wins a tie
            d = abs(yc - y_mid) + (abs(zc - z_mid) if zc is not None else 0.0)
            if s > best * (1 + 1e-9) or (abs(s - best) <= best * 1e-9 and d < best_d):
                best, best_d, best_yz = s, d, (float(yc), z if zc is None else float(zc))
    if best <= 0 or mid >= KEEP_MIDDLE * best:
        return y, z
    return round(best_yz[0], 10), round(best_yz[1], 10)
