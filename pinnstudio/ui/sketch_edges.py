"""Boundary edges of a sketched 2D domain, for assigning boundary conditions.

The domain is a Custom-geometry shape list (see sketch_canvas): primitives
combined in order by union, subtraction or intersection. Its boundary is
made of pieces of the primitives' own outlines -- straight segments of
rectangles and polygons, arcs of circles and ellipses. This module finds
those pieces exactly (membership is tested on the primitives themselves,
not on a drawn approximation), numbers them, and writes for each one a
"Where:" expression in x and y that is true on that piece only, e.g.

    np.isclose(y, 0) and 0 <= x <= 2
    abs(0.4*x - 0.3*y - 0.4) < 5e-06 and 1 <= x <= 1.3
    abs(np.hypot(x - 0.5, y - 0.4) - 0.2) < 1e-05

The expressions only use x, y and np, so they mean the same in the live
script and in an exported one. The BC row's location function is already
restricted to boundary points (on_boundary), so an expression only has to
tell this piece apart from the rest of the boundary.

No Qt in here: the canvas draws what this returns, and the tests check it
directly.
"""
import math

N_SEG = 256      # samples along each straight edge
N_ARC = 720      # samples around each circle / ellipse
REFINE = 48      # bisection steps for the ends of a visible piece


def fmt(v):
    """A number as a person would type it: 0.3, not 0.30000000000000004."""
    v = round(float(v), 9)
    s = f"{v:.6g}"
    return "0" if s in ("-0", "") else s


def parse_vertices(text):
    pts = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            x, y = (float(c) for c in chunk.split(","))
        except ValueError:
            return []
        pts.append((x, y))
    return pts


# ── primitives ───────────────────────────────────────────────────────────
def _polygon_inside(pts, x, y):
    inside = False
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xc:
                inside = not inside
    return inside


def shape_inside(shape, x, y):
    """Is (x, y) strictly inside the primitive? (Only ever asked for points
    a little off its outline, so the outline itself doesn't matter.)"""
    t, p = shape.get("type"), shape.get("params") or {}
    try:
        if t == "Rectangle":
            x0, x1 = sorted((float(p["x_min"]), float(p["x_max"])))
            y0, y1 = sorted((float(p["y_min"]), float(p["y_max"])))
            return x0 < x < x1 and y0 < y < y1
        if t == "Disk":
            return math.hypot(x - float(p["cx"]), y - float(p["cy"])) < float(p["r"])
        if t == "Ellipse":
            cx, cy, a, b = float(p["cx"]), float(p["cy"]), float(p["a"]), float(p["b"])
            # DeepXDE turns a positive angle clockwise
            ang = -float(p.get("angle", 0.0))
            c, s = math.cos(ang), math.sin(ang)
            u = (x - cx) * c + (y - cy) * s
            v = -(x - cx) * s + (y - cy) * c
            return (u / a) ** 2 + (v / b) ** 2 < 1.0
        if t in ("Triangle", "Polygon"):
            pts = parse_vertices(p.get("vertices_text", ""))
            return len(pts) >= 3 and _polygon_inside(pts, x, y)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return False


def in_region(shapes, x, y):
    """Membership in the combined domain, folding the list in order the way
    DeepXDE's CSG classes do."""
    inside = None
    for i, sh in enumerate(shapes):
        here = shape_inside(sh, x, y)
        if i == 0 or inside is None:
            inside = here
            continue
        op = sh.get("op", "union")
        if op == "subtract":
            inside = inside and not here
        elif op == "intersect":
            inside = inside and here
        else:
            inside = inside or here
    return bool(inside)


def _bbox(shapes):
    xs, ys = [], []
    for sh in shapes:
        t, p = sh.get("type"), sh.get("params") or {}
        try:
            if t == "Rectangle":
                xs += [float(p["x_min"]), float(p["x_max"])]
                ys += [float(p["y_min"]), float(p["y_max"])]
            elif t == "Disk":
                r = float(p["r"])
                xs += [float(p["cx"]) - r, float(p["cx"]) + r]
                ys += [float(p["cy"]) - r, float(p["cy"]) + r]
            elif t == "Ellipse":
                a = float(p["a"])
                xs += [float(p["cx"]) - a, float(p["cx"]) + a]
                ys += [float(p["cy"]) - a, float(p["cy"]) + a]
            elif t in ("Triangle", "Polygon"):
                for x, y in parse_vertices(p.get("vertices_text", "")):
                    xs.append(x)
                    ys.append(y)
        except (KeyError, TypeError, ValueError):
            pass
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _curves(shape):
    """The primitive's outline as curves: ("seg", p0, p1), ("circle", cx, cy, r)
    or ("ellipse", cx, cy, a, b, angle)."""
    t, p = shape.get("type"), shape.get("params") or {}
    try:
        if t == "Rectangle":
            x0, x1 = sorted((float(p["x_min"]), float(p["x_max"])))
            y0, y1 = sorted((float(p["y_min"]), float(p["y_max"])))
            c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            return [("seg", c[i], c[(i + 1) % 4]) for i in range(4)]
        if t == "Disk":
            return [("circle", float(p["cx"]), float(p["cy"]), float(p["r"]))]
        if t == "Ellipse":
            # stored counter-clockwise (DeepXDE turns a positive angle clockwise)
            return [("ellipse", float(p["cx"]), float(p["cy"]), float(p["a"]), float(p["b"]),
                     -float(p.get("angle", 0.0)))]
        if t in ("Triangle", "Polygon"):
            pts = parse_vertices(p.get("vertices_text", ""))
            return [("seg", pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))
                    if pts[i] != pts[(i + 1) % len(pts)]]
    except (KeyError, TypeError, ValueError):
        return []
    return []


# ── one curve -> its visible pieces ──────────────────────────────────────
def _point_normal(curve, s):
    """Point and (unnormalised) normal of the curve at parameter s in [0, 1]."""
    if curve[0] == "seg":
        (x0, y0), (x1, y1) = curve[1], curve[2]
        return (x0 + s * (x1 - x0), y0 + s * (y1 - y0)), (y1 - y0, x0 - x1)
    th = 2 * math.pi * s
    if curve[0] == "circle":
        _, cx, cy, r = curve
        return (cx + r * math.cos(th), cy + r * math.sin(th)), (math.cos(th), math.sin(th))
    _, cx, cy, a, b, ang = curve
    c, sn = math.cos(ang), math.sin(ang)
    u, v = a * math.cos(th), b * math.sin(th)
    nu, nv = math.cos(th) / a, math.sin(th) / b
    return (cx + u * c - v * sn, cy + u * sn + v * c), (nu * c - nv * sn, nu * sn + nv * c)


def _on_boundary(shapes, curve, s, eps):
    (x, y), (nx, ny) = _point_normal(curve, s)
    n = math.hypot(nx, ny) or 1.0
    nx, ny = nx / n * eps, ny / n * eps
    return in_region(shapes, x + nx, y + ny) != in_region(shapes, x - nx, y - ny)


def _refine(shapes, curve, s_off, s_on, eps):
    """Where the curve goes from off the boundary (at s_off) to on it (s_on)."""
    for _ in range(REFINE):
        m = 0.5 * (s_off + s_on)
        if _on_boundary(shapes, curve, m, eps):
            s_on = m
        else:
            s_off = m
    return s_on


def _visible_pieces(shapes, curve, eps, L=1.0):
    """[(s0, s1)] parameter ranges of the curve that are domain boundary.
    For closed curves a range may wrap (s1 > 1)."""
    closed = curve[0] != "seg"
    if closed:
        n = N_ARC
    else:
        (x0, y0), (x1, y1) = curve[1], curve[2]
        n = min(N_SEG, max(12, int(N_SEG * math.hypot(x1 - x0, y1 - y0) / L)))
    ss = [(k + 0.5) / n for k in range(n)]
    on = [_on_boundary(shapes, curve, s, eps) for s in ss]
    if not any(on):
        return []
    if all(on):
        return [(0.0, 1.0)]
    pieces = []
    if closed:
        # start the scan just after an "off" sample so no run wraps the seam
        k0 = next(k for k in range(n) if not on[k])
        order = [(k0 + 1 + j) % n for j in range(n)]
    else:
        order = list(range(n))
    j = 0
    while j < len(order):
        if not on[order[j]]:
            j += 1
            continue
        j0 = j
        while j + 1 < len(order) and on[order[j + 1]]:
            j += 1
        k_first, k_last = order[j0], order[j]
        # start of the run
        if not closed and k_first == 0:
            s0 = 0.0
        else:
            s0 = _refine(shapes, curve, ss[k_first] - 1.0 / n, ss[k_first], eps)
        # end of the run
        if not closed and k_last == n - 1:
            s1 = 1.0
        else:
            s1 = _refine(shapes, curve, ss[k_last] + 1.0 / n, ss[k_last], eps)
        if closed and s1 < s0:
            s1 += 1.0
        pieces.append((s0, s1))
        j += 1
    return pieces


# ── expressions ──────────────────────────────────────────────────────────
def _tol(v):
    return f"{float(f'{v:.0e}'):.0e}"


def _lin(a, b, c):
    """'a*x + b*y - c' written the way a person would."""
    terms = []
    for coef, name in ((a, "x"), (b, "y")):
        if abs(coef) < 1e-12:
            continue
        mag = fmt(abs(coef))
        t = name if mag == "1" else f"{mag}*{name}"
        terms.append(("- " if coef < 0 else "+ ") + t)
    if abs(c) > 1e-12:
        terms.append(("+ " if -c > 0 else "- ") + fmt(abs(c)))
    s = " ".join(terms)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def _segment_edge(p0, p1, L):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    tiny = 1e-9 * L
    if abs(dy) <= tiny:
        lo, hi = sorted((x0, x1))
        yv = 0.5 * (y0 + y1)
        expr = f"np.isclose(y, {fmt(yv)}) and {fmt(lo)} <= x <= {fmt(hi)}"
        desc = f"y = {fmt(yv)},  {fmt(lo)} ≤ x ≤ {fmt(hi)}"
    elif abs(dx) <= tiny:
        lo, hi = sorted((y0, y1))
        xv = 0.5 * (x0 + x1)
        expr = f"np.isclose(x, {fmt(xv)}) and {fmt(lo)} <= y <= {fmt(hi)}"
        desc = f"x = {fmt(xv)},  {fmt(lo)} ≤ y ≤ {fmt(hi)}"
    else:
        a, b = dy, -dx
        if a < 0:
            a, b = -a, -b
        c = a * x0 + b * y0
        tol = 1e-5 * L * math.hypot(a, b)
        if abs(dx) >= abs(dy):
            lo, hi = sorted((x0, x1))
            rng = f"{fmt(lo)} <= x <= {fmt(hi)}"
        else:
            lo, hi = sorted((y0, y1))
            rng = f"{fmt(lo)} <= y <= {fmt(hi)}"
        expr = f"abs({_lin(a, b, c)}) < {_tol(tol)} and {rng}"
        desc = f"line ({fmt(x0)}, {fmt(y0)}) → ({fmt(x1)}, {fmt(y1)})"
    return expr, desc


def _curve_expr(curve, L):
    if curve[0] == "circle":
        _, cx, cy, r = curve
        dxs = "x" if abs(cx) < 1e-12 else f"x - {fmt(cx)}" if cx > 0 else f"x + {fmt(-cx)}"
        dys = "y" if abs(cy) < 1e-12 else f"y - {fmt(cy)}" if cy > 0 else f"y + {fmt(-cy)}"
        return (f"abs(np.hypot({dxs}, {dys}) - {fmt(r)}) < {_tol(1e-5 * L)}",
                f"circle, centre ({fmt(cx)}, {fmt(cy)}), r = {fmt(r)}")
    _, cx, cy, a, b, ang = curve
    dxs = "x" if abs(cx) < 1e-12 else f"(x - {fmt(cx)})" if cx > 0 else f"(x + {fmt(-cx)})"
    dys = "y" if abs(cy) < 1e-12 else f"(y - {fmt(cy)})" if cy > 0 else f"(y + {fmt(-cy)})"
    c, s = math.cos(ang), math.sin(ang)
    if abs(s) < 1e-12:
        u, v = dxs, dys
    elif abs(c) < 1e-12:
        u, v = dys, dxs
    else:
        u = f"({fmt(c)}*{dxs} + {fmt(s)}*{dys})"
        v = f"({fmt(c)}*{dys} - {fmt(s)}*{dxs})"
    return (f"abs(({u}/{fmt(a)})**2 + ({v}/{fmt(b)})**2 - 1) < 1e-04",
            f"ellipse, centre ({fmt(cx)}, {fmt(cy)}), a = {fmt(a)}, b = {fmt(b)}")


# ── all edges ────────────────────────────────────────────────────────────
def boundary_edges(shapes):
    """The domain's boundary as numbered edges. Each edge is a dict:

        expr    "Where:" expression true on this edge only
        desc    short description for the user ("y = 0,  0 ≤ x ≤ 2")
        points  polyline to draw, in world coordinates
        mid     a point on the edge (for its label, and to test BC rows)
        out     unit outward normal at mid
        kind    "seg" or "arc"
    """
    shapes = [s for s in shapes if _curves(s)]
    if not shapes:
        return []
    x0, x1, y0, y1 = _bbox(shapes)
    L = max(x1 - x0, y1 - y0, 1e-9)
    eps = 1e-7 * L

    segs, arcs = [], []
    for sh in shapes:
        for curve in _curves(sh):
            pieces = _visible_pieces(shapes, curve, eps, L)
            if not pieces:
                continue
            if curve[0] == "seg":
                for s0, s1 in pieces:
                    a, _ = _point_normal(curve, s0)
                    b, _ = _point_normal(curve, s1)
                    if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6 * L:
                        segs.append((a, b))
            else:
                arcs.append((curve, pieces))

    # straight pieces on one line, touching or overlapping -> one edge
    merged = []
    for a, b in segs:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        if ux < -1e-12 or (abs(ux) <= 1e-12 and uy < 0):
            ux, uy, a, b = -ux, -uy, b, a
        placed = False
        for m in merged:
            mux, muy = m["u"]
            if abs(mux * uy - muy * ux) > 1e-9:
                continue
            # same line? distance of a from m's line
            ax, ay = m["a"]
            if abs((a[0] - ax) * muy - (a[1] - ay) * mux) > 1e-7 * L:
                continue
            t_a = (a[0] - ax) * mux + (a[1] - ay) * muy
            t_b = (b[0] - ax) * mux + (b[1] - ay) * muy
            lo, hi = m["t"]
            if t_a <= hi + 1e-7 * L and t_b >= lo - 1e-7 * L:
                m["t"] = (min(lo, t_a), max(hi, t_b))
                placed = True
                break
        if not placed:
            merged.append({"a": a, "u": (ux, uy), "t": (0.0, n)})

    edges = []
    for m in merged:
        (ax, ay), (ux, uy), (t0, t1) = m["a"], m["u"], m["t"]
        p0 = (ax + t0 * ux, ay + t0 * uy)
        p1 = (ax + t1 * ux, ay + t1 * uy)
        p0 = (round(p0[0], 9), round(p0[1], 9))
        p1 = (round(p1[0], 9), round(p1[1], 9))
        expr, desc = _segment_edge(p0, p1, L)
        mid = (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
        edges.append({"kind": "seg", "expr": expr, "desc": desc, "points": [p0, p1], "mid": mid,
                      "normal": (uy, -ux)})

    for curve, pieces in arcs:
        expr, desc = _curve_expr(curve, L)
        for s0, s1 in pieces:
            n = max(8, int(160 * (s1 - s0)))
            pts = [_point_normal(curve, (s0 + (s1 - s0) * k / n) % 1.0)[0] for k in range(n + 1)]
            mid, nrm = _point_normal(curve, (0.5 * (s0 + s1)) % 1.0)
            e_expr, e_desc = expr, desc
            if len(pieces) > 1:
                # the same circle shows more than once: tell its arcs apart
                dense = [_point_normal(curve, (s0 + (s1 - s0) * k / 2000) % 1.0)[0] for k in range(2001)]
                xs = [p[0] for p in dense]
                ys = [p[1] for p in dense]
                pad = 1e-5 * L
                e_expr = (f"{expr} and {fmt(min(xs) - pad)} <= x <= {fmt(max(xs) + pad)}"
                          f" and {fmt(min(ys) - pad)} <= y <= {fmt(max(ys) + pad)}")
                e_desc = f"{desc} (arc)"
            elif (s1 - s0) < 1.0:
                e_desc = f"{desc} (arc)"
            edges.append({"kind": "arc", "expr": e_expr, "desc": e_desc, "points": pts, "mid": mid,
                          "normal": nrm})

    # outward normal at each edge's midpoint, for its label
    for e in edges:
        (mx, my), (nx, ny) = e["mid"], e["normal"]
        n = math.hypot(nx, ny) or 1.0
        nx, ny = nx / n, ny / n
        if in_region(shapes, mx + nx * 1e-4 * L, my + ny * 1e-4 * L):
            nx, ny = -nx, -ny
        e["out"] = (nx, ny)
        del e["normal"]
    return edges


def where_namespace(x, y, domain=None):
    """Names a BC "Where:" expression can use, as in the generated script:
    x, y, np, math functions, pi, the domain limits and the side names."""
    import numpy as np
    ns = {"np": np, "x": x, "y": y, "pi": np.pi}
    for f in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
              "exp", "log", "log10", "sqrt", "abs", "ceil", "floor"):
        ns[f] = getattr(np, f)
    domain = domain or {}
    ns.update(domain)
    for name, c, lim in (("left", "x", "x_min"), ("right", "x", "x_max"),
                         ("bottom", "y", "y_min"), ("top", "y", "y_max")):
        if lim in domain:
            ns[name] = bool(np.isclose(x if c == "x" else y, domain[lim]))
    ns["t"] = domain.get("t_min", 0.0)
    return ns


def where_matches(expr, x, y, domain=None):
    """Does a BC row's "Where:" expression pick the boundary point (x, y)?
    False when it can't be evaluated."""
    src = (expr or "").strip() or "True"
    try:
        return bool(eval(src, {"__builtins__": {"abs": abs, "min": min, "max": max}},
                         where_namespace(x, y, domain)))
    except Exception:
        return False


def join_where(exprs):
    """Several edges' expressions as one "Where:" ("(...) or (...)")."""
    exprs = [e for e in exprs if e]
    if len(exprs) == 1:
        return exprs[0]
    return " or ".join(f"({e})" for e in exprs)


def _strip_parens(e):
    e = e.strip()
    while e.startswith("(") and e.endswith(")"):
        depth = 0
        for i, ch in enumerate(e):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(e) - 1:
                return e  # "(a) or (b)": the first "(" closes early
        e = e[1:-1].strip()
    return e


def split_where(expr):
    """A "Where:" joined with top-level "or" -> its parts, outer parentheses
    removed: "(A) or (B)" -> ["A", "B"]."""
    parts, depth, start = [], 0, 0
    e = (expr or "").strip()
    i = 0
    while i < len(e):
        ch = e[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and e.startswith(" or ", i):
            parts.append(e[start:i])
            i += 4
            start = i
            continue
        i += 1
    parts.append(e[start:])
    return [_strip_parens(p) for p in parts if p.strip()]


def add_to_where(current, extra):
    """Where: `current` plus the edges in `extra`."""
    cur = (current or "").strip()
    if cur in ("", "True"):
        return extra
    parts = split_where(cur)
    for p in split_where(extra):
        if p not in parts:
            parts.append(p)
    return join_where(parts)


def remove_from_where(current, exprs):
    """Where: `current` without the parts equal to one of `exprs`; None when
    nothing in it matches (the user wrote it differently)."""
    parts = split_where(current)
    keep = [p for p in parts if p not in set(exprs)]
    if len(keep) == len(parts):
        return None
    return join_where(keep)
