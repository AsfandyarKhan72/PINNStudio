"""3D sketching: boxes and spheres drawn on three orthographic views.

A 3D sketch is the same Custom-geometry shape list (Cuboid / Sphere,
combined in order by union, subtraction or intersection). The canvas shows
it on one of three views -- x–y (seen from +z), x–z (from -y), y–z (from
+x) -- as the slice of the domain at a chosen depth: the slice of a CSG
combination is the same combination of the slices, so the 2D machinery
(sketch_edges) draws and analyses it unchanged.

Boundary conditions in 3D go on faces: each side of a box and each
sphere's surface that actually bounds the domain is a numbered face F1,
F2, ..., with a "Where:" in x, y, z true on that face only:

    np.isclose(z, 1) and 0 <= x <= 2 and 0 <= y <= 1
    abs(np.sqrt((x - 1)**2 + (y - 1)**2 + (z - 1)**2) - 0.5) < 2e-05

A face seen edge-on in the current view (perpendicular to it) shows up in
the slice as a line or a circle, and can be clicked there; every face is
also listed, so faces parallel to the view can be picked too.
"""
import math

from pinnstudio.ui.sketch_edges import _segment_edge, _tol, _visible_pieces, fmt, in_region as in_region2

VIEWS = {  # name: (horizontal axis, vertical axis, depth axis), axes 0=x 1=y 2=z
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
VIEW_LABELS = {"xy": "x–y (top)", "xz": "x–z (front)", "yz": "y–z (side)"}
AX = "xyz"


def _box(p):
    return [sorted((float(p["x_min"]), float(p["x_max"]))), sorted((float(p["y_min"]), float(p["y_max"]))),
            sorted((float(p["z_min"]), float(p["z_max"])))]


def _sph(p):
    return (float(p["cx"]), float(p["cy"]), float(p["cz"])), float(p["r"])


def shape_inside3(shape, q):
    t, p = shape.get("type"), shape.get("params") or {}
    try:
        if t == "Cuboid":
            b = _box(p)
            return all(b[k][0] < q[k] < b[k][1] for k in range(3))
        if t == "Sphere":
            c, r = _sph(p)
            return math.dist(c, q) < r
    except (KeyError, TypeError, ValueError):
        return False
    return False


def in_region3(shapes, q):
    inside = None
    for i, sh in enumerate(shapes):
        here = shape_inside3(sh, q)
        if i == 0:
            inside = here
            continue
        op = sh.get("op", "union")
        inside = (inside and not here) if op == "subtract" else (inside and here) if op == "intersect" \
            else (inside or here)
    return bool(inside)


def bbox3(shapes):
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for sh in shapes:
        try:
            if sh.get("type") == "Cuboid":
                b = _box(sh["params"])
                for k in range(3):
                    lo[k], hi[k] = min(lo[k], b[k][0]), max(hi[k], b[k][1])
            elif sh.get("type") == "Sphere":
                c, r = _sph(sh["params"])
                for k in range(3):
                    lo[k], hi[k] = min(lo[k], c[k] - r), max(hi[k], c[k] + r)
        except (KeyError, TypeError, ValueError):
            pass
    if lo[0] == math.inf:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    return lo, hi


# ── views ────────────────────────────────────────────────────────────────
def section_shapes(shapes, view, depth):
    """The slice of every shape at depth (along the view's depth axis), as
    2D shapes in the view's (horizontal, vertical) coordinates. A shape that
    misses the slice becomes {"type": "Empty"} so the combination order
    holds."""
    h, v, d = VIEWS[view]
    out = []
    for sh in shapes:
        t, p = sh.get("type"), sh.get("params") or {}
        s2 = {"type": "Empty", "params": {}}
        try:
            if t == "Cuboid":
                b = _box(p)
                if b[d][0] <= depth <= b[d][1]:
                    s2 = {"type": "Rectangle", "params": {"x_min": b[h][0], "x_max": b[h][1],
                                                          "y_min": b[v][0], "y_max": b[v][1]}}
            elif t == "Sphere":
                c, r = _sph(p)
                rr = r * r - (depth - c[d]) ** 2
                if rr > 0:
                    s2 = {"type": "Disk", "params": {"cx": c[h], "cy": c[v], "r": math.sqrt(rr)}}
        except (KeyError, TypeError, ValueError):
            pass
        if sh.get("op"):
            s2["op"] = sh["op"]
        out.append(s2)
    return out


def projection_shape(shape, view):
    """The shape's outline seen along the view's depth axis (for drawing)."""
    h, v, _d = VIEWS[view]
    t, p = shape.get("type"), shape.get("params") or {}
    try:
        if t == "Cuboid":
            b = _box(p)
            return {"type": "Rectangle", "params": {"x_min": b[h][0], "x_max": b[h][1], "y_min": b[v][0], "y_max": b[v][1]}}
        if t == "Sphere":
            c, r = _sph(p)
            return {"type": "Disk", "params": {"cx": c[h], "cy": c[v], "r": r}}
    except (KeyError, TypeError, ValueError):
        pass
    return {"type": "Empty", "params": {}}


def translate3(shape, view, dh, dv):
    h, v, _d = VIEWS[view]
    t, p = shape["type"], dict(shape["params"])
    if t == "Cuboid":
        for k, dd in ((h, dh), (v, dv)):
            for e in ("min", "max"):
                key = f"{AX[k]}_{e}"
                p[key] = round(float(p[key]) + dd, 10)
    elif t == "Sphere":
        for k, dd in ((h, dh), (v, dv)):
            key = "c" + AX[k]
            p[key] = round(float(p[key]) + dd, 10)
    out = dict(shape)
    out["params"] = p
    return out


def box_from_view(view, h0, h1, v0, v1, d0, d1):
    h, v, d = VIEWS[view]
    rng = {h: sorted((h0, h1)), v: sorted((v0, v1)), d: sorted((d0, d1))}
    p = {}
    for k in range(3):
        p[f"{AX[k]}_min"] = round(rng[k][0], 10)
        p[f"{AX[k]}_max"] = round(rng[k][1], 10)
    return {"type": "Cuboid", "params": p}


def sphere_from_view(view, ch, cv, depth, r):
    h, v, d = VIEWS[view]
    c = [0.0, 0.0, 0.0]
    c[h], c[v], c[d] = ch, cv, depth
    return {"type": "Sphere", "params": {"cx": round(c[0], 10), "cy": round(c[1], 10), "cz": round(c[2], 10),
                                         "r": round(r, 10)}}


# ── faces ────────────────────────────────────────────────────────────────
def _face_samples(face, n=14):
    if face["kind"] == "plane":
        a, val, b = face["axis"], face["value"], face["bounds"]
        o = [k for k in range(3) if k != a]
        for i in range(n):
            for j in range(n):
                q = [0.0, 0.0, 0.0]
                q[a] = val
                q[o[0]] = b[o[0]][0] + (i + 0.5) / n * (b[o[0]][1] - b[o[0]][0])
                q[o[1]] = b[o[1]][0] + (j + 0.5) / n * (b[o[1]][1] - b[o[1]][0])
                n_vec = [0.0, 0.0, 0.0]
                n_vec[a] = 1.0
                yield q, n_vec
    else:
        c, r = face["center"], face["r"]
        m = n * n * 2
        golden = math.pi * (3 - math.sqrt(5))
        for k in range(m):
            zz = 1 - 2 * (k + 0.5) / m
            rad = math.sqrt(1 - zz * zz)
            th = golden * k
            u = (rad * math.cos(th), rad * math.sin(th), zz)
            yield [c[0] + r * u[0], c[1] + r * u[1], c[2] + r * u[2]], list(u)


def faces3(shapes):
    """The faces bounding the 3D domain, each a dict:

        expr    "Where:" true on this face only
        desc    for the user ("z = 1 (box 1)")
        kind    "plane" (axis, value, bounds) or "sphere" (center, r)
        mid     a point of the face on the domain boundary (to test BC rows)
        out     the outward axis direction, for planes (+1 / -1)
    """
    lo, hi = bbox3(shapes)
    L = max(hi[k] - lo[k] for k in range(3)) or 1.0
    eps = 1e-6 * L
    cands = []
    for i, sh in enumerate(shapes):
        t, p = sh.get("type"), sh.get("params") or {}
        try:
            if t == "Cuboid":
                b = _box(p)
                for a in range(3):
                    for side, val in ((-1, b[a][0]), (1, b[a][1])):
                        cands.append({"kind": "plane", "axis": a, "value": val, "bounds": b, "shape": i,
                                      "side": side})
            elif t == "Sphere":
                c, r = _sph(p)
                cands.append({"kind": "sphere", "center": c, "r": r, "shape": i})
        except (KeyError, TypeError, ValueError):
            continue
    faces = []
    for f in cands:
        mid = None
        for q, nv in _face_samples(f):
            qa = [q[k] + eps * nv[k] for k in range(3)]
            qb = [q[k] - eps * nv[k] for k in range(3)]
            if in_region3(shapes, qa) != in_region3(shapes, qb):
                mid = q
                break
        if mid is None:
            continue
        f["mid"] = tuple(mid)
        num = f["shape"] + 1
        if f["kind"] == "plane":
            a, val, b = f["axis"], f["value"], f["bounds"]
            o = [k for k in range(3) if k != a]
            f["expr"] = (f"np.isclose({AX[a]}, {fmt(val)}) and {fmt(b[o[0]][0])} <= {AX[o[0]]} <= {fmt(b[o[0]][1])}"
                         f" and {fmt(b[o[1]][0])} <= {AX[o[1]]} <= {fmt(b[o[1]][1])}")
            f["desc"] = (f"{AX[a]} = {fmt(val)}  ({fmt(b[o[0]][0])} ≤ {AX[o[0]]} ≤ {fmt(b[o[0]][1])}, "
                         f"{fmt(b[o[1]][0])} ≤ {AX[o[1]]} ≤ {fmt(b[o[1]][1])}) — box {num}")
        else:
            c, r = f["center"], f["r"]
            terms = []
            for k in range(3):
                ck = c[k]
                terms.append(f"{AX[k]}**2" if abs(ck) < 1e-12 else
                             f"({AX[k]} - {fmt(ck)})**2" if ck > 0 else f"({AX[k]} + {fmt(-ck)})**2")
            f["expr"] = f"abs(np.sqrt({' + '.join(terms)}) - {fmt(r)}) < {_tol(1e-5 * L)}"
            f["desc"] = f"sphere surface, centre ({fmt(c[0])}, {fmt(c[1])}, {fmt(c[2])}), r = {fmt(r)} — sphere {num}"
        faces.append(f)
    # coplanar faces of one box pair that touch and point the same way -> keep both (different boxes)
    return faces


def face_in_view(face, shapes, view, depth):
    """The face's visible trace in the view's slice, as 2D polylines (in
    the view's coordinates), or [] if it is parallel to the view or misses
    the slice."""
    h, v, d = VIEWS[view]
    sec = section_shapes(shapes, view, depth)
    lo, hi = bbox3(shapes)
    L = max(hi[k] - lo[k] for k in range(3)) or 1.0
    eps = 1e-7 * L
    if face["kind"] == "plane":
        a, val, b = face["axis"], face["value"], face["bounds"]
        if a == d or not (b[d][0] <= depth <= b[d][1]):
            return []
        if a == h:
            curve = ("seg", (val, b[v][0]), (val, b[v][1]))
        else:
            curve = ("seg", (b[h][0], val), (b[h][1], val))
        pieces = _visible_pieces(sec, curve, eps, L)
        out = []
        for s0, s1 in pieces:
            (x0, y0), (x1, y1) = curve[1], curve[2]
            out.append([(x0 + s0 * (x1 - x0), y0 + s0 * (y1 - y0)), (x0 + s1 * (x1 - x0), y0 + s1 * (y1 - y0))])
        return out
    c, r = face["center"], face["r"]
    rr = r * r - (depth - c[d]) ** 2
    if rr <= 0:
        return []
    curve = ("circle", c[h], c[v], math.sqrt(rr))
    out = []
    for s0, s1 in _visible_pieces(sec, curve, eps, L):
        n = max(8, int(160 * (s1 - s0)))
        out.append([(curve[1] + curve[3] * math.cos(2 * math.pi * (s0 + (s1 - s0) * k / n)),
                     curve[2] + curve[3] * math.sin(2 * math.pi * (s0 + (s1 - s0) * k / n))) for k in range(n + 1)])
    return out


def where_matches3(expr, q, domain=None):
    import numpy as np
    from pinnstudio.ui.sketch_edges import where_namespace
    src = (expr or "").strip() or "True"
    ns = where_namespace(q[0], q[1], domain)
    ns["z"] = q[2]
    domain = domain or {}
    for name, lim in (("front", "z_min"), ("back", "z_max")):
        if lim in domain:
            ns[name] = bool(np.isclose(q[2], domain[lim]))
    try:
        return bool(eval(src, {"__builtins__": {"abs": abs, "min": min, "max": max}}, ns))
    except Exception:
        return False


# for the canvas: 2D helpers re-exported
__all__ = ["VIEWS", "VIEW_LABELS", "section_shapes", "projection_shape", "faces3", "face_in_view", "in_region3",
           "where_matches3", "translate3", "box_from_view", "sphere_from_view", "bbox3", "in_region2",
           "_segment_edge"]
