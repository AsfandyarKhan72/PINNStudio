"""Import a 2D outline drawing (DXF or SVG) into the Sketch.

The drawing's closed outlines become Custom-geometry shapes: circles stay
Disks, ellipses Ellipses, axis-aligned boxes Rectangles, everything else a
Polygon (arcs and curves in a polyline or path are approximated by short
straight pieces). Lines and arcs drawn separately are joined end to end
into closed outlines first.

Holes are found by nesting: shapes are added from the largest to the
smallest, and a shape inside an odd number of bigger ones is subtracted
(a hole), inside an even number added (an island in a hole) -- which is
how a CAD outline with holes is meant.

Only plain geometry is read: no blocks/inserts, splines, text or
dimensions (DXF), and no clipping, masks or strokes (SVG). SVG's y axis
points down, so it is flipped. No Qt in here.
"""
import math
import re
import xml.etree.ElementTree as ET

from pinnstudio.ui.sketch_edges import arc_runs, fmt

ARC_STEP = math.radians(5.0)   # approximation step for arcs inside outlines


class ImportResult:
    def __init__(self):
        self.items = []      # ("circle", cx, cy, r) | ("ellipse", cx, cy, a, b, ccw angle) | ("poly", [(x, y), ...])
        self.open = []       # open polylines, to be chained
        self.notes = []      # what was skipped, for the user

    def skipped(self, what):
        self.notes.append(what)


# ── geometry helpers ─────────────────────────────────────────────────────
def _arc_points(cx, cy, r, a0, a1, rx=None, rot=0.0):
    """Points of a counter-clockwise arc from angle a0 to a1 (radians)."""
    rx = r if rx is None else rx
    while a1 <= a0:
        a1 += 2 * math.pi
    n = max(2, int(math.ceil((a1 - a0) / ARC_STEP)))
    c, s = math.cos(rot), math.sin(rot)
    pts = []
    for k in range(n + 1):
        t = a0 + (a1 - a0) * k / n
        u, v = rx * math.cos(t), r * math.sin(t)
        pts.append((cx + u * c - v * s, cy + u * s + v * c))
    return pts


def _bulge_points(p0, p1, bulge):
    """A DXF polyline segment with a bulge (tan of a quarter of its angle)."""
    if abs(bulge) < 1e-12:
        return [p0, p1]
    (x0, y0), (x1, y1) = p0, p1
    theta = 4 * math.atan(bulge)
    chord = math.hypot(x1 - x0, y1 - y0)
    if chord < 1e-15:
        return [p0, p1]
    r = chord / (2 * math.sin(abs(theta) / 2))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    d = math.sqrt(max(r * r - (chord / 2) ** 2, 0.0))
    ux, uy = (x1 - x0) / chord, (y1 - y0) / chord
    sgn = 1 if (bulge > 0) == (abs(theta) < math.pi) else -1
    cx, cy = mx - sgn * uy * d, my + sgn * ux * d
    a0 = math.atan2(y0 - cy, x0 - cx)
    n = max(2, int(math.ceil(abs(theta) / ARC_STEP)))
    return [(cx + r * math.cos(a0 + theta * k / n), cy + r * math.sin(a0 + theta * k / n)) for k in range(n + 1)]


def _area(item):
    if item[0] == "circle":
        return math.pi * item[3] ** 2
    if item[0] == "ellipse":
        return math.pi * item[3] * item[4]
    pts = item[1]
    return abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                   for i in range(len(pts)))) / 2


def _inside(item, x, y):
    if item[0] == "circle":
        return math.hypot(x - item[1], y - item[2]) < item[3]
    if item[0] == "ellipse":
        _, cx, cy, a, b, ang = item
        c, s = math.cos(ang), math.sin(ang)
        u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
        return (u / a) ** 2 + (v / b) ** 2 < 1
    pts, inside = item[1], False
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _probe(item):
    """A point just inside the item, to test which outlines contain it."""
    if item[0] in ("circle", "ellipse"):
        return item[1], item[2]
    pts = item[1]
    # centroid if it is inside, else a point near the first vertex
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    if _inside(item, cx, cy):
        return cx, cy
    for i in range(len(pts)):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        mx, my = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
        if _inside(item, mx, my):
            return mx, my
    return pts[0]


def _clean_poly(pts, tol):
    out = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol:
            out.append(p)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()
    # drop corners in the middle of a straight run
    changed = True
    while changed and len(out) > 3:
        changed = False
        for i in range(len(out)):
            (x0, y0), (x1, y1), (x2, y2) = out[i - 1], out[i], out[(i + 1) % len(out)]
            if abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) <= tol * max(math.hypot(x2 - x0, y2 - y0), tol):
                del out[i]
                changed = True
                break
    return out


def _chain(open_paths, tol):
    """Join open polylines end to end; return (closed loops, number left open)."""
    paths = [list(p) for p in open_paths if len(p) >= 2]
    loops, left = [], 0
    close = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) <= tol
    while paths:
        cur = paths.pop(0)
        grew = True
        while grew and not close(cur[0], cur[-1]):
            grew = False
            for i, p in enumerate(paths):
                if close(cur[-1], p[0]):
                    cur += p[1:]
                elif close(cur[-1], p[-1]):
                    cur += p[-2::-1]
                elif close(cur[0], p[-1]):
                    cur = p[:-1] + cur
                elif close(cur[0], p[0]):
                    cur = p[:0:-1] + cur
                else:
                    continue
                paths.pop(i)
                grew = True
                break
        if len(cur) >= 4 and close(cur[0], cur[-1]):
            loops.append(cur[:-1])
        else:
            left += 1
    return loops, left


def to_shapes(res, scale=1.0):
    """ImportResult -> (Custom-geometry shape list, notes)."""
    from pinnstudio.ui.sketch_canvas import polygon_shape
    items = list(res.items)
    xs, ys = [], []
    for it in items:
        if it[0] == "poly":
            xs += [p[0] for p in it[1]]
            ys += [p[1] for p in it[1]]
        else:
            xs += [it[1] - it[3], it[1] + it[3]]
            ys += [it[2] - it[3], it[2] + it[3]]
    for p in res.open:
        xs += [q[0] for q in p]
        ys += [q[1] for q in p]
    extent = max((max(xs) - min(xs)) if xs else 1.0, (max(ys) - min(ys)) if ys else 1.0, 1e-12)
    tol = 1e-6 * extent
    loops, left = _chain(res.open, max(tol, 1e-9))
    items += [("poly", lp) for lp in loops]
    notes = list(res.notes)
    if left:
        notes.append(f"{left} outline(s) that don't close were left out")
    cleaned = []
    for it in items:
        if it[0] == "poly":
            pts = _clean_poly(it[1], tol)
            if len(pts) < 3:
                continue
            it = ("poly", pts)
            runs = arc_runs(pts)
            if runs and runs[0][0] == "full":   # a circle drawn as a polyline
                it = ("circle",) + tuple(runs[0][1:])
        if _area(it) > 1e-12 * extent * extent:
            cleaned.append(it)
    cleaned.sort(key=_area, reverse=True)
    shapes = []
    for i, it in enumerate(cleaned):
        px, py = _probe(it)
        depth = sum(1 for big in cleaned[:i] if _inside(big, px, py))
        k = scale
        if it[0] == "circle":
            sh = {"type": "Disk", "params": {"cx": _r(it[1] * k), "cy": _r(it[2] * k), "r": _r(it[3] * k)}}
        elif it[0] == "ellipse":
            _, cx, cy, a, b, ang = it
            if b > a:
                a, b, ang = b, a, ang + math.pi / 2
            ang = math.atan2(math.sin(ang), math.cos(ang))
            if ang > math.pi / 2:
                ang -= math.pi
            elif ang < -math.pi / 2:
                ang += math.pi
            # stored with DeepXDE's sign: a positive angle turns clockwise
            sh = {"type": "Ellipse", "params": {"cx": _r(cx * k), "cy": _r(cy * k), "a": _r(a * k), "b": _r(b * k),
                                                "angle": _r(-ang)}}
        else:
            sh = polygon_shape([(_r(x * k), _r(y * k)) for x, y in it[1]])
        if i:
            sh["op"] = "subtract" if depth % 2 else "union"
        shapes.append(sh)
    return shapes, notes


def _r(v):
    return float(fmt(v))


# ── DXF ──────────────────────────────────────────────────────────────────
def read_dxf(text):
    res = ImportResult()
    lines = text.splitlines()
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        try:
            pairs.append((int(lines[i].strip()), lines[i + 1].strip()))
        except ValueError:
            continue
    # entities section only
    start = next((i for i in range(len(pairs) - 1)
                  if pairs[i] == (0, "SECTION") and pairs[i + 1] == (2, "ENTITIES")), None)
    if start is None:
        raise ValueError("No ENTITIES section found -- is this an ASCII DXF file?")
    ents, cur = [], None
    for code, val in pairs[start + 2:]:
        if code == 0:
            if cur is not None:
                ents.append(cur)
            if val == "ENDSEC":
                cur = None
                break
            cur = [val, []]
        elif cur is not None:
            cur[1].append((code, val))
    if cur is not None:
        ents.append(cur)

    def f(groups, code, default=0.0):
        for c, v in groups:
            if c == code:
                return float(v)
        return default

    skipped = {}
    i = 0
    while i < len(ents):
        kind, g = ents[i]
        if kind == "LINE":
            res.open.append([(f(g, 10), f(g, 20)), (f(g, 11), f(g, 21))])
        elif kind == "CIRCLE":
            res.items.append(("circle", f(g, 10), f(g, 20), f(g, 40)))
        elif kind == "ARC":
            res.open.append(_arc_points(f(g, 10), f(g, 20), f(g, 40), math.radians(f(g, 50)), math.radians(f(g, 51))))
        elif kind == "ELLIPSE":
            cx, cy, mx, my, ratio = f(g, 10), f(g, 20), f(g, 11), f(g, 21), f(g, 40, 1.0)
            t0, t1 = f(g, 41, 0.0), f(g, 42, 2 * math.pi)
            a = math.hypot(mx, my)
            ang = math.atan2(my, mx)
            if abs((t1 - t0) - 2 * math.pi) < 1e-6 or (abs(t0) < 1e-9 and abs(t1) < 1e-9):
                res.items.append(("ellipse", cx, cy, a, a * ratio, ang))
            else:
                res.open.append(_arc_points(cx, cy, a * ratio, t0, t1, rx=a, rot=ang))
        elif kind == "LWPOLYLINE":
            closed = int(f(g, 70, 0)) & 1
            verts, bulges = [], []
            for c, v in g:
                if c == 10:
                    verts.append([float(v), None])
                    bulges.append(0.0)
                elif c == 20 and verts:
                    verts[-1][1] = float(v)
                elif c == 42 and verts:
                    bulges[-1] = float(v)
            verts = [tuple(v) for v in verts if v[1] is not None]
            _add_polyline(res, verts, bulges, closed)
        elif kind == "POLYLINE":
            closed = int(f(g, 70, 0)) & 1
            verts, bulges = [], []
            i += 1
            while i < len(ents) and ents[i][0] == "VERTEX":
                vg = ents[i][1]
                verts.append((f(vg, 10), f(vg, 20)))
                bulges.append(f(vg, 42, 0.0))
                i += 1
            _add_polyline(res, verts, bulges, closed)
            continue  # i already points past SEQEND's predecessor
        elif kind in ("SEQEND", "VERTEX"):
            pass
        else:
            skipped[kind] = skipped.get(kind, 0) + 1
        i += 1
    for k, n in sorted(skipped.items()):
        res.skipped(f"{n} {k} entit{'y' if n == 1 else 'ies'} not read")
    return res


def _add_polyline(res, verts, bulges, closed):
    if len(verts) < 2:
        return
    pts = [verts[0]]
    n = len(verts)
    for k in range(n if closed else n - 1):
        seg = _bulge_points(verts[k], verts[(k + 1) % n], bulges[k] if k < len(bulges) else 0.0)
        pts += seg[1:]
    if closed:
        res.items.append(("poly", pts[:-1] if pts[0] == pts[-1] else pts))
    else:
        res.open.append(pts)


# ── SVG ──────────────────────────────────────────────────────────────────
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _mat_mul(a, b):
    # affine matrices as (a, b, c, d, e, f): x' = a x + c y + e, y' = b x + d y + f
    return (a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
            a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5])


def _parse_transform(text):
    m = (1, 0, 0, 1, 0, 0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        v = [float(x) for x in _NUM.findall(args)]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate":
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale":
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0])
            t = (math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)
            if len(v) == 3:
                t = _mat_mul(_mat_mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = _mat_mul(m, t)
    return m


def _apply(m, x, y):
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def _similarity(m):
    """(uniform scale, rotation) if m keeps circles circles, else None."""
    a, b, c, d = m[:4]
    if abs(a - d) < 1e-9 and abs(b + c) < 1e-9:
        return math.hypot(a, b), math.atan2(b, a)
    if abs(a + d) < 1e-9 and abs(b - c) < 1e-9:   # with a reflection
        return math.hypot(a, b), math.atan2(b, a)
    return None


def _svg_arc(x0, y0, rx, ry, phi, large, sweep, x1, y1):
    """SVG elliptical arc (endpoint form) -> points, F.6.5 of the SVG spec."""
    if rx == 0 or ry == 0:
        return [(x0, y0), (x1, y1)]
    rx, ry = abs(rx), abs(ry)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x0 - x1) / 2, (y0 - y1) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(num / den, 0.0)) if den else 0.0
    if large == sweep:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x0 + x1) / 2
    cy = sp * cxp + cp * cyp + (y0 + y1) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(2, int(math.ceil(abs(dt) / ARC_STEP)))
    pts = []
    for k in range(n + 1):
        t = t1 + dt * k / n
        pts.append((cx + rx * math.cos(t) * cp - ry * math.sin(t) * sp, cy + rx * math.cos(t) * sp + ry * math.sin(t) * cp))
    return pts


def _bezier(p0, p1, p2, p3=None, n=16):
    pts = []
    for k in range(1, n + 1):
        t = k / n
        if p3 is None:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        else:
            x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t * t * p2[0]
                 + t ** 3 * p3[0])
            y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t * t * p2[1]
                 + t ** 3 * p3[1])
        pts.append((x, y))
    return pts


def _path_subpaths(d):
    """SVG path data -> [(points, closed)] in the path's own coordinates."""
    toks = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|" + _NUM.pattern, d or "")
    out, pts, closed = [], [], False
    cur = start = (0.0, 0.0)
    last_ctrl, last_cmd = None, ""
    i, cmd = 0, None

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    def flush():
        nonlocal pts, closed
        if len(pts) >= 2:
            out.append((pts, closed))
        pts, closed = [], False

    while i < len(toks):
        if re.match(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                closed = True
                cur = start
                flush()
                last_cmd = cmd
                continue
        elif cmd is None:
            i += 1
            continue
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = cur if rel else (0.0, 0.0)
        if C == "M":
            flush()
            cur = (ox + num(), oy + num())
            start = cur
            pts = [cur]
            cmd = "l" if rel else "L"   # further pairs are line-tos
        elif C == "L":
            cur = (ox + num(), oy + num())
            pts.append(cur)
        elif C == "H":
            cur = ((cur[0] if rel else 0.0) + num(), cur[1])
            pts.append(cur)
        elif C == "V":
            cur = (cur[0], (cur[1] if rel else 0.0) + num())
            pts.append(cur)
        elif C == "C":
            p1 = (ox + num(), oy + num())
            p2 = (ox + num(), oy + num())
            p3 = (ox + num(), oy + num())
            pts += _bezier(cur, p1, p2, p3)
            last_ctrl, cur = p2, p3
        elif C == "S":
            p1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in "CcSs" and last_ctrl else cur
            p2 = (ox + num(), oy + num())
            p3 = (ox + num(), oy + num())
            pts += _bezier(cur, p1, p2, p3)
            last_ctrl, cur = p2, p3
        elif C == "Q":
            p1 = (ox + num(), oy + num())
            p2 = (ox + num(), oy + num())
            pts += _bezier(cur, p1, p2)
            last_ctrl, cur = p1, p2
        elif C == "T":
            p1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in "QqTt" and last_ctrl else cur
            p2 = (ox + num(), oy + num())
            pts += _bezier(cur, p1, p2)
            last_ctrl, cur = p1, p2
        elif C == "A":
            rx, ry, rot, large, sweep = num(), num(), num(), num(), num()
            end = (ox + num(), oy + num())
            pts += _svg_arc(cur[0], cur[1], rx, ry, math.radians(rot), int(large), int(sweep), end[0], end[1])[1:]
            cur = end
        last_cmd = cmd
    flush()
    return out


def read_svg(text):
    res = ImportResult()
    root = ET.fromstring(text)

    def tag(el):
        return el.tag.split("}")[-1]

    def attr(el, k, default=0.0):
        v = el.get(k)
        if v is None:
            return default
        m = _NUM.match(v.strip())
        return float(m.group()) if m else default

    flip = (1, 0, 0, -1, 0, 0)  # y down -> y up
    skipped = {}

    def poly_out(m, pts, closed):
        w = [_apply(m, x, y) for x, y in pts]
        if closed:
            res.items.append(("poly", w))
        else:
            res.open.append(w)

    def walk(el, m):
        m = _mat_mul(m, _parse_transform(el.get("transform")))
        t = tag(el)
        if t in ("defs", "clipPath", "mask", "symbol", "marker", "pattern", "title", "desc", "metadata", "style",
                 "text"):
            return
        if t == "rect":
            x, y, w_, h = attr(el, "x"), attr(el, "y"), attr(el, "width"), attr(el, "height")
            if w_ > 0 and h > 0:
                poly_out(m, [(x, y), (x + w_, y), (x + w_, y + h), (x, y + h)], True)
        elif t in ("circle", "ellipse"):
            cx, cy = attr(el, "cx"), attr(el, "cy")
            rx = attr(el, "r") if t == "circle" else attr(el, "rx")
            ry = attr(el, "r") if t == "circle" else attr(el, "ry")
            sim = _similarity(m)
            if rx > 0 and ry > 0:
                if sim is not None:
                    k, rot = sim
                    X, Y = _apply(m, cx, cy)
                    if abs(rx - ry) < 1e-12:
                        res.items.append(("circle", X, Y, k * rx))
                    else:
                        res.items.append(("ellipse", X, Y, k * rx, k * ry, rot if m[0] * m[3] - m[1] * m[2] > 0 else -rot))
                else:
                    poly_out(m, _arc_points(cx, cy, ry, 0, 2 * math.pi, rx=rx)[:-1], True)
        elif t in ("polygon", "polyline"):
            v = [float(x) for x in _NUM.findall(el.get("points", ""))]
            pts = list(zip(v[0::2], v[1::2]))
            if len(pts) >= 2:
                poly_out(m, pts, t == "polygon" or pts[0] == pts[-1])
        elif t == "line":
            poly_out(m, [(attr(el, "x1"), attr(el, "y1")), (attr(el, "x2"), attr(el, "y2"))], False)
        elif t == "path":
            for pts, closed in _path_subpaths(el.get("d")):
                poly_out(m, pts, closed)
        elif t not in ("svg", "g", "a", "switch"):
            skipped[t] = skipped.get(t, 0) + 1
        for ch in el:
            walk(ch, m)

    walk(root, flip)
    for k, n in sorted(skipped.items()):
        res.skipped(f"{n} <{k}> element(s) not read")
    return res


def import_file(path, scale=1.0):
    """DXF or SVG file -> (shapes, notes)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    low = path.lower()
    if low.endswith(".dxf"):
        res = read_dxf(text)
    elif low.endswith(".svg"):
        res = read_svg(text)
    else:
        raise ValueError("Only .dxf and .svg drawings can be imported.")
    shapes, notes = to_shapes(res, scale)
    if not shapes:
        raise ValueError("No closed outlines found in the drawing." + (" " + "; ".join(notes) if notes else ""))
    return shapes, notes


def drawing_extent(shapes):
    from pinnstudio.ui.sketch_edges import _bbox
    x0, x1, y0, y1 = _bbox(shapes)
    return x1 - x0, y1 - y0
