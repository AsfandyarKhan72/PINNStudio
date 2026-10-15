"""Sketch canvas for 2D geometry (Domain → Shape → Sketch).

A drawing surface in the problem's own x–y coordinates: a grid with labelled
axes, zoom and pan, snapping, and tools to draw rectangles, circles, ellipses
and polygons, each combined with the shapes before it by union, subtraction
or intersection. The canvas edits the same shape list as the Custom geometry
builder (config.geom_custom_shapes_json):

    [{"type": "Rectangle", "params": {"x_min":.., "x_max":.., "y_min":.., "y_max":..}},
     {"type": "Disk", "params": {"cx":.., "cy":.., "r":..}, "op": "subtract"}, ...]

so a sketched domain is trained by exactly the code that trains a Custom
geometry -- the canvas only draws and edits that list.

Controls: pick a tool and drag on the grid (Polygon: click each corner, then
double-click, press Enter or click the first corner to close; Esc cancels).
Select: click a shape to select it, drag to move it, Delete to remove it.
Mouse wheel zooms around the cursor; right- or middle-drag pans.

Edges → BCs: the domain's boundary is split into numbered edges (see
sketch_edges); click one (Ctrl/Shift-click for more) and make a boundary
condition for it, or add it to an existing one. Each edge is drawn in the
colour of the BC row whose "Where:" picks it, grey when none does.

Shapes can also be typed (the Type: field) or imported from a DXF/SVG
drawing (sketch_import). In 3D the canvas draws boxes and spheres on one
of three views and shows the domain's slice at a chosen depth; boundary
conditions go on faces (sketch3d).
"""
import json
import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF,
                         QTransform)
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QVBoxLayout,
                             QWidget)

from pinnstudio.ui.sketch_edges import boundary_edges, in_region, join_where, where_matches
from pinnstudio.ui import sketch3d as s3

TOOLS = ("Select", "Rectangle", "Circle", "Ellipse", "Polygon", "Edges")
OPS = (("Union (add)", "union"), ("Subtract (cut hole)", "subtract"), ("Intersect (keep overlap)", "intersect"))
OP_COLORS = {None: QColor("#1f6fd1"), "union": QColor("#1f6fd1"),
             "subtract": QColor("#d9480f"), "intersect": QColor("#2b8a3e")}
# one colour per Boundary Conditions row (cycled), and edges without one
from pinnstudio.ui.theme import BC_COLORS  # noqa: E402  (colour-blind-safe palette)
NO_BC_COLOR = "#868e96"


def _fmt(v):
    """A coordinate as typed by a person: 0.3, not 0.30000000000000004."""
    v = round(float(v), 10)
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


def vertices_text(pts):
    return ";".join(f"{_fmt(x)},{_fmt(y)}" for x, y in pts)


def shape_path(shape):
    """A shape dict -> QPainterPath in world coordinates (y up)."""
    t, p = shape.get("type"), shape.get("params") or {}
    path = QPainterPath()
    try:
        if t == "Rectangle":
            x0, x1 = sorted((float(p["x_min"]), float(p["x_max"])))
            y0, y1 = sorted((float(p["y_min"]), float(p["y_max"])))
            path.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        elif t == "Disk":
            r = float(p["r"])
            path.addEllipse(QPointF(float(p["cx"]), float(p["cy"])), r, r)
        elif t == "Ellipse":
            cx, cy = float(p["cx"]), float(p["cy"])
            a, b, ang = float(p["a"]), float(p["b"]), float(p.get("angle", 0.0))
            e = QPainterPath()
            e.addEllipse(QPointF(0, 0), a, b)
            tr = QTransform()
            tr.translate(cx, cy)
            tr.rotateRadians(-ang)  # DeepXDE turns a positive angle clockwise
            path = tr.map(e)
        elif t in ("Triangle", "Polygon"):
            pts = parse_vertices(p.get("vertices_text", ""))
            if len(pts) >= 3:
                path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
                path.closeSubpath()
    except (KeyError, TypeError, ValueError):
        return QPainterPath()
    return path


def combined_path(shapes, transform=None):
    """The CSG result of the shape list, in list order (DeepXDE's own order).
    Qt flattens curves when it combines paths, with a tolerance in the
    path's own units -- so for drawing, pass the world-to-screen transform
    and the combination is done in pixels (a hole stays round)."""
    result = None
    for i, sh in enumerate(shapes):
        p = shape_path(sh)
        if transform is not None:
            p = transform.map(p)
        if result is None:
            result = p
            continue
        op = sh.get("op", "union")
        if op == "subtract":
            result = result.subtracted(p)
        elif op == "intersect":
            result = result.intersected(p)
        else:
            result = result.united(p)
    return result if result is not None else QPainterPath()


def translate_shape(shape, dx, dy):
    t, p = shape["type"], dict(shape["params"])
    if t == "Rectangle":
        for k in ("x_min", "x_max"):
            p[k] = round(float(p[k]) + dx, 10)
        for k in ("y_min", "y_max"):
            p[k] = round(float(p[k]) + dy, 10)
    elif t in ("Disk", "Ellipse"):
        p["cx"] = round(float(p["cx"]) + dx, 10)
        p["cy"] = round(float(p["cy"]) + dy, 10)
    elif t in ("Triangle", "Polygon"):
        p["vertices_text"] = vertices_text([(x + dx, y + dy) for x, y in parse_vertices(p["vertices_text"])])
    out = dict(shape)
    out["params"] = p
    return out


def polygon_shape(pts):
    """Closed polygon corners -> the shape DeepXDE accepts for them: a
    Triangle for three corners, a Rectangle for an axis-aligned box (DeepXDE
    refuses both as a Polygon), a Polygon otherwise."""
    if len(pts) == 3:
        return {"type": "Triangle", "params": {"vertices_text": vertices_text(pts)}}
    if len(pts) == 4:
        xs = sorted({round(x, 10) for x, _ in pts})
        ys = sorted({round(y, 10) for _, y in pts})
        if len(xs) == 2 and len(ys) == 2 and {(round(x, 10), round(y, 10)) for x, y in pts} == \
                {(x, y) for x in xs for y in ys}:
            return {"type": "Rectangle", "params": {"x_min": xs[0], "x_max": xs[1], "y_min": ys[0], "y_max": ys[1]}}
    return {"type": "Polygon", "params": {"vertices_text": vertices_text(pts)}}


class _Canvas(QWidget):
    """The drawing surface itself (painted by hand, world y pointing up)."""
    changed = pyqtSignal()           # the shape list was edited (on mouse release)
    selection_changed = pyqtSignal(int)
    cursor_moved = pyqtSignal(float, float)
    edges_selected = pyqtSignal()    # Edges tool: the selected edges changed
    edge_hovered = pyqtSignal(int)   # Edges tool: edge under the cursor (-1: none)

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.shapes = []
        self.selected = -1
        self.tool = "Select"
        self.grid = 0.1
        self.snap = True
        self.scale = 300.0                 # pixels per unit
        self.center = QPointF(0.5, 0.5)    # world point at the widget centre
        self._drag = None                  # (kind, start world point, extra)
        self._poly = []                    # polygon corners being placed
        self._cursor = None
        self._edges, self._edges_key = [], None   # boundary edges of the current shapes
        self._bc_of, self._bc_key = [], None      # per edge: the BC rows that pick it
        self.sel_edges = []                       # selected edge indices, in click order
        self.hover_edge = -1
        # 3D: shapes are Cuboids/Spheres, shown as the slice at `depth` on one view
        self.dim = 2
        self.view = "xy"
        self.depth = 0.5
        self.new_depth = (0.0, 1.0)               # depth range of a new box
        self._traces, self._traces_key = [], None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── coordinates ──────────────────────────────────────────
    def to_screen(self, p):
        return QPointF(self.width() / 2 + (p.x() - self.center.x()) * self.scale,
                       self.height() / 2 - (p.y() - self.center.y()) * self.scale)

    def to_world(self, q):
        return QPointF(self.center.x() + (q.x() - self.width() / 2) / self.scale,
                       self.center.y() - (q.y() - self.height() / 2) / self.scale)

    def world_transform(self):
        t = QTransform()
        t.translate(self.width() / 2, self.height() / 2)
        t.scale(self.scale, -self.scale)
        t.translate(-self.center.x(), -self.center.y())
        return t

    def snapped(self, w):
        if not self.snap or self.grid <= 0:
            return w
        g = self.grid
        return QPointF(round(w.x() / g) * g, round(w.y() / g) * g)

    def shapes2d(self):
        """What the canvas shows: the shapes (2D), or their slice (3D)."""
        return self.shapes if self.dim == 2 else s3.section_shapes(self.shapes, self.view, self.depth)

    def outline_shapes(self):
        """Each shape's own outline: itself (2D), its projection on the view (3D)."""
        return self.shapes if self.dim == 2 else [s3.projection_shape(sh, self.view) for sh in self.shapes]

    def axis_names(self):
        if self.dim == 2:
            return "x", "y", ""
        h, v, d = s3.VIEWS[self.view]
        return s3.AX[h], s3.AX[v], s3.AX[d]

    def fit(self):
        """Zoom to the shapes (or to [0, 1]² when there are none)."""
        r = QRectF()
        for sh in self.outline_shapes():
            r = r.united(shape_path(sh).boundingRect())
        if r.width() <= 0 or r.height() <= 0:
            r = QRectF(0, 0, 1, 1)
        pad = 0.15 * max(r.width(), r.height())
        r = r.adjusted(-pad, -pad, pad, pad)
        self.center = r.center()
        w, h = max(self.width(), 50), max(self.height(), 50)
        self.scale = min(w / r.width(), h / r.height())
        self.update()

    # ── edges (Edges → BCs tool) ─────────────────────────────
    def edges(self):
        """The domain's boundary edges (2D) or faces (3D), recomputed only
        when the shapes change."""
        key = json.dumps([self.dim, self.shapes], sort_keys=True)
        if key != self._edges_key:
            self._edges = boundary_edges(self.shapes) if self.dim == 2 else s3.faces3(self.shapes)
            self._edges_key = key
            self.sel_edges = []
            self.hover_edge = -1
        return self._edges

    def traces(self):
        """What to draw and click for each edge/face in the current view:
        [{"idx", "points", "mid", "out"}] in view coordinates. A 3D face
        shows only where it crosses the slice edge-on."""
        edges = self.edges()
        if self.dim == 2:
            return [{"idx": k, "points": e["points"], "mid": e["mid"], "out": e["out"]} for k, e in enumerate(edges)]
        key = (self._edges_key, self.view, self.depth)
        if key != self._traces_key:
            sec = self.shapes2d()
            out = []
            for k, f in enumerate(edges):
                for pts in s3.face_in_view(f, self.shapes, self.view, self.depth):
                    if len(pts) == 2:
                        mid = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
                        nx, ny = pts[1][1] - pts[0][1], pts[0][0] - pts[1][0]
                    else:
                        mid = pts[len(pts) // 2]
                        a, b = pts[len(pts) // 2 - 1], pts[len(pts) // 2 + 1]
                        nx, ny = b[1] - a[1], a[0] - b[0]
                    n = math.hypot(nx, ny) or 1.0
                    nx, ny = nx / n, ny / n
                    L = max(abs(pts[0][0] - pts[-1][0]), abs(pts[0][1] - pts[-1][1]), 1e-6)
                    if in_region(sec, mid[0] + nx * 1e-3 * L, mid[1] + ny * 1e-3 * L):
                        nx, ny = -nx, -ny
                    out.append({"idx": k, "points": pts, "mid": mid, "out": (nx, ny)})
            self._traces, self._traces_key = out, key
        return self._traces

    def edge_bcs(self):
        """Per edge (face), the indices of the BC rows whose "Where:" picks it."""
        edges = self.edges()
        rows = self.owner.bc_rows()
        dom = self.owner.domain()
        key = (self._edges_key, tuple((b["index"], b["expr"]) for b in rows), tuple(sorted(dom.items())))
        if key != self._bc_key:
            self._bc_key = key
            if self.dim == 2:
                self._bc_of = [[b["index"] for b in rows if where_matches(b["expr"], e["mid"][0], e["mid"][1], dom)]
                               for e in edges]
            else:
                self._bc_of = [[b["index"] for b in rows if s3.where_matches3(b["expr"], e["mid"], dom)]
                               for e in edges]
        return self._bc_of

    def _edge_at(self, q, px=8.0):
        best, best_d = -1, px
        for tr in self.traces():
            k = tr["idx"]
            pts = [self.to_screen(QPointF(x, y)) for x, y in tr["points"]]
            for a, b in zip(pts, pts[1:]):
                dx, dy = b.x() - a.x(), b.y() - a.y()
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((q.x() - a.x()) * dx + (q.y() - a.y()) * dy) / L2))
                d = math.hypot(q.x() - (a.x() + t * dx), q.y() - (a.y() + t * dy))
                if d < best_d:
                    best, best_d = k, d
        return best

    def _paint_edges(self, qp):
        edges = self.edges()
        bc_of = self.edge_bcs()
        if not edges:
            qp.setPen(QColor("#8a96a3"))
            qp.drawText(self.rect().adjusted(0, 0, 0, -self.height() // 3), Qt.AlignmentFlag.AlignCenter,
                        "Draw the domain first, then its " + ("edges" if self.dim == 2 else "faces") +
                        " show up here")
            return
        prefix = "E" if self.dim == 2 else "F"
        traces = self.traces()
        f = QFont(self.font())
        f.setPointSizeF(9.0)
        f.setBold(True)
        qp.setFont(f)
        fm = qp.fontMetrics()
        for e in traces:
            k = e["idx"]
            poly = QPolygonF([self.to_screen(QPointF(x, y)) for x, y in e["points"]])
            color = QColor(BC_COLORS[bc_of[k][0] % len(BC_COLORS)]) if bc_of[k] else QColor(NO_BC_COLOR)
            if k in self.sel_edges or k == self.hover_edge:
                halo = QColor("#ffd43b") if k in self.sel_edges else QColor(255, 212, 59, 110)
                qp.setPen(QPen(halo, 11, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                qp.drawPolyline(poly)
            qp.setPen(QPen(color, 4.0 if bc_of[k] else 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                           Qt.PenJoinStyle.RoundJoin))
            qp.drawPolyline(poly)
        # labels on top of every edge
        for e in traces:
            k = e["idx"]
            color = QColor(BC_COLORS[bc_of[k][0] % len(BC_COLORS)]) if bc_of[k] else QColor(NO_BC_COLOR)
            m = self.to_screen(QPointF(*e["mid"]))
            ox, oy = e["out"]
            text = f"{prefix}{k + 1}"
            w, h = fm.horizontalAdvance(text) + 8, fm.height() + 2
            c = QPointF(m.x() + ox * (w / 2 + 8), m.y() - oy * (h / 2 + 8))
            box = QRectF(c.x() - w / 2, c.y() - h / 2, w, h)
            qp.setPen(QPen(color, 1.4))
            qp.setBrush(QColor("#fff3bf") if k in self.sel_edges else QColor("white"))
            qp.drawRoundedRect(box, 4, 4)
            qp.setPen(color.darker(130))
            qp.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        qp.setBrush(Qt.BrushStyle.NoBrush)
        # legend: which colour is which BC row
        rows = self.owner.bc_rows()
        used = {i for lst in bc_of for i in lst}
        items = [(QColor(BC_COLORS[b["index"] % len(BC_COLORS)]), b["label"]) for b in rows if b["index"] in used]
        if any(not lst for lst in bc_of):
            items.append((QColor(NO_BC_COLOR), "no BC yet"))
        if items:
            f.setBold(False)
            f.setPointSizeF(9.5)
            qp.setFont(f)
            fm = qp.fontMetrics()
            w = max(fm.horizontalAdvance(t) for _, t in items) + 34
            h = len(items) * (fm.height() + 3) + 8
            box = QRectF(self.width() - w - 10, 10, w, h)
            qp.setPen(QPen(QColor("#ced4da"), 1))
            qp.setBrush(QColor(255, 255, 255, 230))
            qp.drawRoundedRect(box, 5, 5)
            y = box.top() + 4
            for col, t in items:
                qp.setPen(QPen(col, 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                qp.drawLine(QPointF(box.left() + 8, y + fm.height() / 2 + 1), QPointF(box.left() + 22, y + fm.height() / 2 + 1))
                qp.setPen(QColor("#343a40"))
                qp.drawText(QPointF(box.left() + 28, y + fm.ascent()), t)
                y += fm.height() + 3
            qp.setBrush(Qt.BrushStyle.NoBrush)

    # ── painting ─────────────────────────────────────────────
    def _nice_step(self):
        """Grid spacing on screen: the snapping grid, coarsened until the
        lines are at least ~12 px apart."""
        g = self.grid if self.grid > 0 else 0.1
        while g * self.scale < 12:
            m = g / 10 ** math.floor(math.log10(g))
            g = g / m * (2 if m < 1.5 else 5 if m < 3.5 else 10)
        return g

    def paintEvent(self, _ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor("#fbfcfd"))
        tl, br = self.to_world(QPointF(0, 0)), self.to_world(QPointF(self.width(), self.height()))
        x0, x1, y0, y1 = tl.x(), br.x(), br.y(), tl.y()
        g = self._nice_step()
        major = g * 5
        # grid
        for step, color in ((g, QColor("#e9edf2")), (major, QColor("#cfd6de"))):
            qp.setPen(QPen(color, 1))
            i0, i1 = math.floor(x0 / step), math.ceil(x1 / step)
            for i in range(i0, i1 + 1):
                sx = self.to_screen(QPointF(i * step, 0)).x()
                qp.drawLine(QPointF(sx, 0), QPointF(sx, self.height()))
            j0, j1 = math.floor(y0 / step), math.ceil(y1 / step)
            for j in range(j0, j1 + 1):
                sy = self.to_screen(QPointF(0, j * step)).y()
                qp.drawLine(QPointF(0, sy), QPointF(self.width(), sy))
        # axes
        o = self.to_screen(QPointF(0, 0))
        qp.setPen(QPen(QColor("#7a8794"), 1.4))
        qp.drawLine(QPointF(o.x(), 0), QPointF(o.x(), self.height()))
        qp.drawLine(QPointF(0, o.y()), QPointF(self.width(), o.y()))
        # tick labels on the major lines (bottom and left edges)
        qp.setPen(QColor("#5b6773"))
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        qp.setFont(f)
        for i in range(math.floor(x0 / major), math.ceil(x1 / major) + 1):
            sx = self.to_screen(QPointF(i * major, 0)).x()
            qp.drawText(QPointF(sx + 3, self.height() - 4), _fmt(i * major))
        for j in range(math.floor(y0 / major), math.ceil(y1 / major) + 1):
            sy = self.to_screen(QPointF(0, j * major)).y()
            qp.drawText(QPointF(4, sy - 3), _fmt(j * major))
        hname, vname, dname = self.axis_names()
        qp.drawText(QPointF(self.width() - 14, o.y() - 5), hname)
        qp.drawText(QPointF(o.x() + 5, 12), vname)
        if self.dim == 3:
            f2 = QFont(f)
            f2.setPointSizeF(9.5)
            f2.setBold(True)
            qp.setFont(f2)
            txt = f"slice at {dname} = {_fmt(self.depth)}"
            box = QRectF(40, 8, qp.fontMetrics().horizontalAdvance(txt) + 14, qp.fontMetrics().height() + 4)
            qp.setPen(QPen(QColor("#ced4da"), 1))
            qp.setBrush(QColor(255, 255, 255, 230))
            qp.drawRoundedRect(box, 4, 4)
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QColor("#495057"))
            qp.drawText(box, Qt.AlignmentFlag.AlignCenter, txt)
            qp.setFont(f)

        wt = self.world_transform()
        # the resulting domain, filled
        dom = combined_path(self.shapes2d(), wt)
        qp.setPen(Qt.PenStyle.NoPen)
        qp.setBrush(QBrush(QColor(31, 111, 209, 55)))
        qp.drawPath(dom)
        qp.setPen(QPen(QColor("#0b3d91"), 2.2))
        qp.setBrush(Qt.BrushStyle.NoBrush)
        qp.drawPath(dom)
        if self.tool == "Edges":
            self._paint_edges(qp)
            qp.end()
            return
        # each shape's own outline (dashed), coloured by how it is combined
        for i, sh in enumerate(self.outline_shapes()):
            op = sh.get("op") if i else None
            pen = QPen(OP_COLORS.get(op, OP_COLORS[None]), 2.6 if i == self.selected else 1.2,
                       Qt.PenStyle.SolidLine if i == self.selected else Qt.PenStyle.DashLine)
            qp.setPen(pen)
            qp.drawPath(wt.map(shape_path(sh)))
            if i == self.selected:
                r = wt.map(shape_path(sh)).boundingRect()
                qp.setPen(QPen(QColor("#1f6fd1"), 1, Qt.PenStyle.DotLine))
                qp.drawRect(r.adjusted(-3, -3, 3, 3))
        # shape being drawn
        qp.setPen(QPen(QColor("#e8590c"), 1.6, Qt.PenStyle.DashLine))
        prev = self._preview_shape()
        if prev is not None:
            if self.dim == 3:
                prev = s3.projection_shape(prev, self.view)
            qp.drawPath(wt.map(shape_path(prev)))
        if self._poly:
            pts = [self.to_screen(QPointF(x, y)) for x, y in self._poly]
            if self._cursor is not None:
                pts.append(self.to_screen(self._cursor))
            for a, b in zip(pts, pts[1:]):
                qp.drawLine(a, b)
            qp.setBrush(QColor("#e8590c"))
            for x, y in self._poly:
                qp.drawEllipse(self.to_screen(QPointF(x, y)), 3.5, 3.5)
            qp.setBrush(Qt.BrushStyle.NoBrush)
        # snapped cursor
        if self._cursor is not None and self.tool not in ("Select", "Edges"):
            c = self.to_screen(self._cursor)
            qp.setPen(QPen(QColor("#e8590c"), 1))
            qp.drawLine(QPointF(c.x() - 6, c.y()), QPointF(c.x() + 6, c.y()))
            qp.drawLine(QPointF(c.x(), c.y() - 6), QPointF(c.x(), c.y() + 6))
        if not self.shapes and not self._poly and self._drag is None:
            qp.setPen(QColor("#8a96a3"))
            f.setPointSizeF(10.5)
            qp.setFont(f)
            qp.drawText(self.rect().adjusted(0, 0, 0, -self.height() // 3), Qt.AlignmentFlag.AlignCenter,
                        "Pick a tool above and draw on the grid" if self.dim == 2 else
                        "Pick Box or Sphere above and draw on the grid")
        qp.end()

    # ── drawing tools ────────────────────────────────────────
    def _new_op(self):
        return self.owner.op_combo.currentData() if self.shapes else None

    def _preview_shape(self):
        if self._drag is None or self._drag[0] != "draw" or self._cursor is None:
            return None
        a, b = self._drag[1], self._cursor
        return self._shape_from_drag(a, b)

    def _shape_from_drag(self, a, b):
        if self.dim == 3:
            if self.tool == "Rectangle":
                if abs(a.x() - b.x()) < 1e-12 or abs(a.y() - b.y()) < 1e-12:
                    return None
                return s3.box_from_view(self.view, a.x(), b.x(), a.y(), b.y(), *self.new_depth)
            if self.tool == "Circle":
                r = math.hypot(b.x() - a.x(), b.y() - a.y())
                if r < 1e-12:
                    return None
                if self.snap and self.grid > 0:
                    r = max(self.grid, round(r / self.grid) * self.grid)
                return s3.sphere_from_view(self.view, a.x(), a.y(), self.depth, r)
            return None
        if self.tool == "Rectangle":
            if abs(a.x() - b.x()) < 1e-12 or abs(a.y() - b.y()) < 1e-12:
                return None
            return {"type": "Rectangle", "params": {
                "x_min": round(min(a.x(), b.x()), 10), "x_max": round(max(a.x(), b.x()), 10),
                "y_min": round(min(a.y(), b.y()), 10), "y_max": round(max(a.y(), b.y()), 10)}}
        if self.tool == "Circle":
            r = math.hypot(b.x() - a.x(), b.y() - a.y())
            if r < 1e-12:
                return None
            if self.snap and self.grid > 0:
                r = max(self.grid, round(r / self.grid) * self.grid)
            return {"type": "Disk", "params": {"cx": round(a.x(), 10), "cy": round(a.y(), 10), "r": round(r, 10)}}
        if self.tool == "Ellipse":
            w, h = abs(b.x() - a.x()) / 2, abs(b.y() - a.y()) / 2
            if w < 1e-12 or h < 1e-12:
                return None
            cx, cy = (a.x() + b.x()) / 2, (a.y() + b.y()) / 2
            # DeepXDE's Ellipse: semi-major >= semi-minor, rotated by angle
            ang = 0.0 if w >= h else math.pi / 2
            return {"type": "Ellipse", "params": {"cx": round(cx, 10), "cy": round(cy, 10),
                                                  "a": round(max(w, h), 10), "b": round(min(w, h), 10),
                                                  "angle": round(ang, 10)}}
        return None

    def _add(self, shape):
        op = self._new_op()
        if op:
            shape["op"] = op
        self.shapes.append(shape)
        self.selected = len(self.shapes) - 1
        self.selection_changed.emit(self.selected)
        self.changed.emit()

    def _close_polygon(self):
        pts = list(self._poly)
        if len(pts) >= 2 and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 1e-9:
            pts.pop()
        self._poly = []
        if len(pts) >= 3:
            self._add(polygon_shape(pts))
        self.update()

    def _hit(self, w):
        outlines = self.outline_shapes()
        for i in range(len(outlines) - 1, -1, -1):
            p = shape_path(outlines[i])
            if p.contains(w):
                return i
            # near the outline counts too (thin shapes, shapes cut away)
            s = QPainterPathStroker()
            s.setWidth(6.0 / self.scale)
            if s.createStroke(p).contains(w):
                return i
        return -1

    # ── mouse / keys ─────────────────────────────────────────
    def mousePressEvent(self, ev):
        w = self.to_world(ev.position())
        if ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = ("pan", ev.position(), QPointF(self.center))
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        sw = self.snapped(w)
        if self.tool == "Edges":
            k = self._edge_at(ev.position())
            if getattr(self.owner, "active_bc", None) is not None:
                # a BC is being edited: a click puts the edge in it or takes it out
                if k >= 0:
                    self.owner.toggle_edge_in_active_bc(k)
                return
            if ev.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                if k >= 0:
                    if k in self.sel_edges:
                        self.sel_edges.remove(k)
                    else:
                        self.sel_edges.append(k)
            else:
                self.sel_edges = [] if k < 0 or self.sel_edges == [k] else [k]
            self.edges_selected.emit()
            self.update()
            return
        if self.tool == "Select":
            i = self._hit(w)
            self.selected = i
            self.selection_changed.emit(i)
            if i >= 0:
                self._drag = ("move", sw, dict(self.shapes[i]))
            self.update()
        elif self.tool == "Polygon":
            if len(self._poly) >= 3:
                fx, fy = self._poly[0]
                first = self.to_screen(QPointF(fx, fy))
                if math.hypot(first.x() - ev.position().x(), first.y() - ev.position().y()) < 9:
                    self._close_polygon()
                    return
            if not self._poly or (sw.x(), sw.y()) != self._poly[-1]:
                self._poly.append((round(sw.x(), 10), round(sw.y(), 10)))
            self.update()
        else:
            self._drag = ("draw", sw, None)

    def mouseDoubleClickEvent(self, ev):
        if self.tool == "Polygon" and len(self._poly) >= 3:
            self._close_polygon()

    def mouseMoveEvent(self, ev):
        w = self.to_world(ev.position())
        free = self.tool in ("Select", "Edges")
        self._cursor = w if free else self.snapped(w)
        shown = self._cursor
        if self.tool == "Edges" and self._drag is None:
            k = self._edge_at(ev.position())
            if k != self.hover_edge:
                self.hover_edge = k
                self.edge_hovered.emit(k)
        self.cursor_moved.emit(shown.x(), shown.y())
        if self._drag is not None:
            kind = self._drag[0]
            if kind == "pan":
                d = ev.position() - self._drag[1]
                self.center = QPointF(self._drag[2].x() - d.x() / self.scale, self._drag[2].y() + d.y() / self.scale)
            elif kind == "move" and self.selected >= 0:
                sw = self.snapped(w)
                dx, dy = sw.x() - self._drag[1].x(), sw.y() - self._drag[1].y()
                self.shapes[self.selected] = (translate_shape(self._drag[2], dx, dy) if self.dim == 2 else
                                              s3.translate3(self._drag[2], self.view, dx, dy))
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is None:
            return
        kind = self._drag[0]
        if kind == "draw":
            shape = self._shape_from_drag(self._drag[1], self.snapped(self.to_world(ev.position())))
            self._drag = None
            if shape is not None:
                self._add(shape)
        elif kind == "move":
            moved = self.shapes[self.selected] != self._drag[2] if self.selected >= 0 else False
            self._drag = None
            if moved:
                self.changed.emit()
        else:
            self._drag = None
        self.update()

    def resizeEvent(self, ev):
        """Keep the drawing in view when the canvas gets smaller (e.g. when
        the Edges tool's bar appears below it)."""
        super().resizeEvent(ev)
        if not self.shapes or ev.oldSize().width() <= 0:
            return
        r = QRectF()
        for sh in self.outline_shapes():
            r = r.united(shape_path(sh).boundingRect())
        tl, br = self.to_world(QPointF(0, 0)), self.to_world(QPointF(self.width(), self.height()))
        view = QRectF(QPointF(tl.x(), br.y()), QPointF(br.x(), tl.y()))
        if not view.contains(r):
            self.fit()

    def leaveEvent(self, _ev):
        self._cursor = None
        self.update()

    def wheelEvent(self, ev):
        before = self.to_world(ev.position())
        factor = 1.15 ** (ev.angleDelta().y() / 120)
        self.scale = min(max(self.scale * factor, 5.0), 1e6)
        after = self.to_world(ev.position())
        self.center = QPointF(self.center.x() + before.x() - after.x(), self.center.y() + before.y() - after.y())
        self.update()

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.tool == "Polygon":
            self._close_polygon()
        elif k == Qt.Key.Key_Escape:
            self._poly = []
            self._drag = None
            if self.sel_edges:
                self.sel_edges = []
                self.edges_selected.emit()
            self.update()
        elif k == Qt.Key.Key_A and ev.modifiers() & Qt.KeyboardModifier.ControlModifier and self.tool == "Edges":
            self.sel_edges = list(range(len(self.edges())))
            self.edges_selected.emit()
            self.update()
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.tool != "Edges":
            self.owner.delete_selected()
        else:
            super().keyPressEvent(ev)


class SketchPanel(QWidget):
    """Toolbar + canvas + status line. `shapes_edited(list)` fires whenever
    the user changes the sketch; `set_shapes(list)` shows a shape list
    edited elsewhere (the Custom geometry rows) without firing it."""
    shapes_edited = pyqtSignal(list)
    new_bc_requested = pyqtSignal(str)            # "Where:" of the selected edges
    add_to_bc_requested = pyqtSignal(int, str)    # BC row index, "Where:" to add
    remove_from_bcs_requested = pyqtSignal(list)  # edge expressions to take out of their BC rows
    toggle_edge_bc = pyqtSignal(int, int, bool)   # BC row index, edge index, add (else remove)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        labels = {"Select": "⬚ Select", "Rectangle": "▭ Rectangle", "Circle": "◯ Circle",
                  "Ellipse": "⬭ Ellipse", "Polygon": "⬠ Polygon", "Edges": "⌁ Edges → BCs"}
        for t in TOOLS:
            if t == "Edges":
                bar.addSpacing(10)
            b = QPushButton(labels[t])
            b.setCheckable(True)
            b.setFixedHeight(26)
            self.tool_group.addButton(b)
            self.tool_buttons[t] = b
            bar.addWidget(b)
            b.clicked.connect(lambda _c, tool=t: self.set_tool(tool))
        bar.addStretch()
        lay.addLayout(bar)

        bar2 = QHBoxLayout()
        bar2.setSpacing(4)
        self.op_label = QLabel("New shape:")
        bar2.addWidget(self.op_label)
        self.op_combo = QComboBox()
        for label, val in OPS:
            self.op_combo.addItem(label, val)
        self.op_combo.setFixedHeight(26)
        self.op_combo.setToolTip("How the next shape you draw is combined with the shapes before it "
                                 "(the first shape is always the starting domain)")
        bar2.addWidget(self.op_combo)
        bar2.addSpacing(10)
        bar2.addWidget(QLabel("Grid:"))
        self.grid_spin = QDoubleSpinBox()
        self.grid_spin.setDecimals(3)
        self.grid_spin.setRange(0.0001, 1000.0)
        self.grid_spin.setValue(0.1)
        self.grid_spin.setSingleStep(0.05)
        self.grid_spin.setFixedHeight(26)
        self.grid_spin.setFixedWidth(95)
        bar2.addWidget(self.grid_spin)
        self.snap_cb = QCheckBox("Snap to grid")
        self.snap_cb.setChecked(True)
        bar2.addWidget(self.snap_cb)
        bar2.addStretch()
        lay.addLayout(bar2)

        bar2 = QHBoxLayout()   # row 3: edit the drawing
        bar2.setSpacing(4)
        self.delete_btn = QPushButton("✕ Delete shape")
        self.delete_btn.setFixedHeight(26)
        bar2.addWidget(self.delete_btn)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setFixedHeight(26)
        bar2.addWidget(self.clear_btn)
        self.import_btn = QPushButton("📂 Import DXF/SVG…")
        self.import_btn.setFixedHeight(26)
        self.import_btn.setToolTip("Read the closed outlines of a 2D drawing (DXF or SVG): circles, ellipses, "
                                   "polylines, paths, and lines/arcs that join up. Outlines inside others "
                                   "become holes.")
        bar2.addWidget(self.import_btn)
        self.fit_btn = QPushButton("⤢ Fit view")
        self.fit_btn.setFixedHeight(26)
        bar2.addWidget(self.fit_btn)
        bar2.addStretch()
        lay.addLayout(bar2)

        bar2 = QHBoxLayout()   # row 4: typed numbers, cursor position
        bar2.setSpacing(4)
        # exact numbers instead of the mouse
        self.coord_entry_label = QLabel("Type:")
        bar2.addWidget(self.coord_entry_label)
        self.coord_edit = QLineEdit()
        self.coord_edit.setFixedHeight(26)
        self.coord_edit.setMinimumWidth(230)
        self.coord_edit.setToolTip("Type the shape's numbers and press Enter to add it "
                                   "(combined as set in New shape)")
        bar2.addWidget(self.coord_edit, 1)
        bar2.addSpacing(10)
        self.coord_label = QLabel("")
        self.coord_label.setMinimumWidth(150)
        bar2.addWidget(self.coord_label)
        lay.addLayout(bar2)

        # 3D only: which view, where the slice is, and how deep a new box is
        self.view_bar = QWidget()
        b3 = QHBoxLayout(self.view_bar)
        b3.setContentsMargins(0, 0, 0, 0)
        b3.setSpacing(4)
        b3.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        for key, label in s3.VIEW_LABELS.items():
            self.view_combo.addItem(label, key)
        self.view_combo.setFixedHeight(26)
        b3.addWidget(self.view_combo)
        b3.addSpacing(8)
        self.depth_label = QLabel("Slice at z =")
        b3.addWidget(self.depth_label)
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setDecimals(3)
        self.depth_spin.setRange(-1e6, 1e6)
        self.depth_spin.setSingleStep(0.05)
        self.depth_spin.setValue(0.5)
        self.depth_spin.setFixedHeight(26)
        self.depth_spin.setFixedWidth(100)
        self.depth_spin.setToolTip("The canvas shows the domain cut at this depth; a new sphere is centred on it")
        b3.addWidget(self.depth_spin)
        b3.addSpacing(8)
        self.new_depth_label = QLabel("New box: z from")
        b3.addWidget(self.new_depth_label)
        self.new_d0 = QDoubleSpinBox()
        self.new_d1 = QDoubleSpinBox()
        for sp_, v in ((self.new_d0, 0.0), (self.new_d1, 1.0)):
            sp_.setDecimals(3)
            sp_.setRange(-1e6, 1e6)
            sp_.setSingleStep(0.1)
            sp_.setValue(v)
            sp_.setFixedHeight(26)
            sp_.setFixedWidth(90)
        b3.addWidget(self.new_d0)
        b3.addWidget(QLabel("to"))
        b3.addWidget(self.new_d1)
        b3.addStretch()
        lay.addWidget(self.view_bar)
        self.view_bar.setVisible(False)

        self.canvas = _Canvas(self)
        lay.addWidget(self.canvas, 1)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        # Edges → BCs: what to do with the selected edges
        self.edge_bar = QWidget()
        eb = QVBoxLayout(self.edge_bar)
        eb.setContentsMargins(0, 0, 0, 0)
        eb.setSpacing(4)
        r1 = QHBoxLayout()
        r1.setSpacing(4)
        self.edge_sel_label = QLabel("No edge selected")
        self.edge_sel_label.setMinimumWidth(100)
        self.edge_sel_label.setWordWrap(True)
        r1.addWidget(self.edge_sel_label)
        self.new_bc_btn = QPushButton("＋ New BC on selected")
        self.new_bc_btn.setToolTip("Add a Boundary Conditions row (Dirichlet, value 0) whose Where: picks "
                                   "the selected edges; set its type and value in the panel on the left")
        self.add_to_combo = QComboBox()
        self.add_to_combo.setMinimumWidth(120)
        self.add_to_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.add_to_combo.setMinimumContentsLength(10)
        self.add_to_btn = QPushButton("Add to")
        self.add_to_btn.setToolTip("Add the selected edges to this existing BC row's Where:")
        self.remove_btn = QPushButton("Remove from BC")
        self.remove_btn.setToolTip("Take the selected edges out of the BC rows that pick them "
                                   "(only where Where: lists the edge as the sketch wrote it)")
        self.sel_free_btn = QPushButton("Select edges without a BC")
        for b in (self.new_bc_btn, self.add_to_btn, self.remove_btn, self.sel_free_btn, self.add_to_combo):
            b.setFixedHeight(26)
        r1.addStretch()
        eb.addLayout(r1)
        r1b = QHBoxLayout()
        r1b.setSpacing(4)
        r1b.addWidget(self.new_bc_btn)
        r1b.addWidget(self.remove_btn)
        r1b.addStretch()
        eb.addLayout(r1b)
        r1c = QHBoxLayout()
        r1c.setSpacing(4)
        r1c.addWidget(self.add_to_btn)
        r1c.addWidget(self.add_to_combo, 1)
        eb.addLayout(r1c)
        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addWidget(QLabel("Where:"))
        self.where_edit = QLineEdit()
        self.where_edit.setReadOnly(True)
        self.where_edit.setPlaceholderText("the selection's expression appears here")
        self.where_edit.setFixedHeight(26)
        r2.addWidget(self.where_edit, 1)
        eb.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(self.sel_free_btn)
        r3.addStretch()
        eb.addLayout(r3)
        self.face_list = QListWidget()
        self.face_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_list.setMaximumHeight(120)
        self.face_list.setToolTip("Every face of the 3D domain -- select here the ones parallel to the view, "
                                  "which the slice doesn't show")
        eb.addWidget(self.face_list)
        self.face_list.setVisible(False)
        lay.addWidget(self.edge_bar)

        self._bc_rows_fn = lambda: []
        self._domain_fn = lambda: {}

        self.grid_spin.valueChanged.connect(self._on_grid)
        self.snap_cb.toggled.connect(self._on_snap)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.clear)
        self.import_btn.clicked.connect(lambda: self.import_drawing())
        self.fit_btn.clicked.connect(self.canvas.fit)
        self.coord_edit.returnPressed.connect(self._on_coords_entered)
        self.canvas.changed.connect(lambda: self.shapes_edited.emit([dict(s) for s in self.canvas.shapes]))
        self.canvas.cursor_moved.connect(self._show_cursor)
        self.canvas.edges_selected.connect(self._update_edge_bar)
        self.face_list.itemSelectionChanged.connect(self._on_face_list_selection)
        self.view_combo.currentIndexChanged.connect(self._on_view)
        self.depth_spin.valueChanged.connect(self._on_depth)
        self.new_d0.valueChanged.connect(self._on_new_depth)
        self.new_d1.valueChanged.connect(self._on_new_depth)
        self.canvas.edge_hovered.connect(self._on_edge_hovered)
        self.new_bc_btn.clicked.connect(lambda: self._emit_selected(self.new_bc_requested))
        self.add_to_btn.clicked.connect(self._on_add_to)
        self.remove_btn.clicked.connect(self._on_remove)
        self.sel_free_btn.clicked.connect(self._select_free_edges)
        self.set_tool("Rectangle")

    HINTS = {
        "Select": "Click a shape to select it, drag to move it, Delete to remove it. "
                  "Wheel zooms, right-drag pans.",
        "Rectangle": "Drag from one corner to the opposite corner.",
        "Circle": "Press at the centre and drag out the radius.",
        "Ellipse": "Drag the box the ellipse fits in.",
        "Polygon": "Click each corner; double-click, press Enter or click the first corner to close. Esc cancels.",
        "Edges": "Click an edge to select it (Ctrl- or Shift-click for more, Ctrl+A for all), then give it a "
                 "boundary condition below. Each edge takes the colour of the BC row whose Where: picks it.",
    }

    HINTS_3D = {
        "Select": "Click a shape to select it, drag to move it in this view, Delete to remove it. "
                  "Wheel zooms, right-drag pans.",
        "Rectangle": "Drag a box's two corners in this view; its depth is set by “New box … from … to”.",
        "Circle": "Press at the sphere's centre and drag out its radius; the centre's depth is the slice.",
        "Edges": "Click a face where the slice shows it edge-on (Ctrl/Shift-click for more), or pick faces in "
                 "the list -- faces parallel to the view are only in the list. Then give them a BC below.",
    }

    COORD_FORMATS = {
        "Rectangle": "two corners: x0, y0, x1, y1",
        "Circle": "centre and radius: cx, cy, r",
        "Ellipse": "cx, cy, a, b  (or cx, cy, a, b, angle)",
        "Polygon": "corners: x,y; x,y; x,y …  (or one corner x, y)",
    }
    COORD_FORMATS_3D = {
        "Rectangle": "two corners: x0, y0, z0, x1, y1, z1",
        "Circle": "centre and radius: cx, cy, cz, r",
    }
    LABELS_2D = {"Select": "⬚ Select", "Rectangle": "▭ Rectangle", "Circle": "◯ Circle", "Ellipse": "⬭ Ellipse",
                 "Polygon": "⬠ Polygon", "Edges": "⌁ Edges → BCs"}
    LABELS_3D = {"Select": "⬚ Select", "Rectangle": "▣ Box", "Circle": "◉ Sphere", "Edges": "⌁ Faces → BCs"}

    def hints(self):
        return self.HINTS if self.canvas.dim == 2 else {**self.HINTS, **self.HINTS_3D}

    def coord_formats(self):
        return self.COORD_FORMATS if self.canvas.dim == 2 else self.COORD_FORMATS_3D

    # ── 3D ───────────────────────────────────────────────────
    def set_dim(self, dim):
        """2: rectangles, circles, ellipses, polygons. 3: boxes and spheres on
        three views."""
        dim = 3 if dim == 3 else 2
        if dim == self.canvas.dim:
            return
        self.canvas.dim = dim
        labels = self.LABELS_3D if dim == 3 else self.LABELS_2D
        for t, b in self.tool_buttons.items():
            b.setVisible(t in labels)
            b.setText(labels.get(t, b.text()))
        self.view_bar.setVisible(dim == 3)
        self.face_list.setVisible(dim == 3)
        self.import_btn.setVisible(dim == 2)
        if self.canvas.tool not in labels:
            self.set_tool("Rectangle")
        else:
            self.set_tool(self.canvas.tool)
        self._sync_axis_labels()

    def _sync_axis_labels(self):
        h, v, d = self.canvas.axis_names()
        if self.canvas.dim == 3:
            self.depth_label.setText(f"Slice at {d} =")
            self.new_depth_label.setText(f"New box: {d} from")

    def _on_view(self, _i=None):
        self.canvas.view = self.view_combo.currentData()
        self._sync_axis_labels()
        self._centre_depth()
        self.canvas.fit()
        self.canvas.update()

    def _centre_depth(self):
        """Put the slice through the middle of the shapes along the depth axis."""
        if not self.canvas.shapes:
            return
        lo, hi = s3.bbox3(self.canvas.shapes)
        d = s3.VIEWS[self.canvas.view][2]
        self.depth_spin.blockSignals(True)
        self.depth_spin.setValue(round((lo[d] + hi[d]) / 2, 6))
        self.depth_spin.blockSignals(False)
        self.canvas.depth = self.depth_spin.value()
        self.new_d0.blockSignals(True)
        self.new_d1.blockSignals(True)
        self.new_d0.setValue(lo[d])
        self.new_d1.setValue(hi[d])
        self.new_d0.blockSignals(False)
        self.new_d1.blockSignals(False)
        self.canvas.new_depth = (lo[d], hi[d])

    def _on_depth(self, v):
        self.canvas.depth = v
        self.canvas.update()

    def _on_new_depth(self, _v=None):
        self.canvas.new_depth = (self.new_d0.value(), self.new_d1.value())

    def _show_cursor(self, x, y):
        h, v, d = self.canvas.axis_names()
        txt = f"{h} = {_fmt(x)},  {v} = {_fmt(y)}"
        if self.canvas.dim == 3:
            txt += f"   ({d} = {_fmt(self.canvas.depth)})"
        self.coord_label.setText(txt)

    def _fill_face_list(self):
        if self.canvas.dim != 3:
            return
        edges = self.canvas.edges()
        self.face_list.blockSignals(True)
        if self.face_list.count() != len(edges) or any(
                self.face_list.item(k).data(Qt.ItemDataRole.UserRole) != e["expr"] for k, e in enumerate(edges)):
            self.face_list.clear()
            for k, e in enumerate(edges):
                it = QListWidgetItem(f"F{k + 1}:  {e['desc']}")
                it.setData(Qt.ItemDataRole.UserRole, e["expr"])
                self.face_list.addItem(it)
        bc_of = self.canvas.edge_bcs()
        labels = {b["index"]: b["label"] for b in self.bc_rows()}
        for k in range(self.face_list.count()):
            it = self.face_list.item(k)
            on = ", ".join(labels.get(i, f"BC {i + 1}").split(":")[0] for i in bc_of[k]) if k < len(bc_of) else ""
            it.setText(f"F{k + 1}:  {edges[k]['desc']}" + (f"   [{on}]" if on else ""))
            it.setForeground(QColor(BC_COLORS[bc_of[k][0] % len(BC_COLORS)]) if k < len(bc_of) and bc_of[k]
                             else QColor(NO_BC_COLOR))
            it.setSelected(k in self.canvas.sel_edges)
        self.face_list.blockSignals(False)

    def _on_face_list_selection(self):
        sel = [self.face_list.row(it) for it in self.face_list.selectedItems()]
        keep = [k for k in self.canvas.sel_edges if k in sel]
        self.canvas.sel_edges = keep + [k for k in sorted(sel) if k not in keep]
        self._update_edge_bar(from_list=True)
        self.canvas.update()

    def _on_coords_entered(self):
        """Add the shape typed in the Type: field (or, for a polygon, one corner)."""
        tool = self.canvas.tool
        text = self.coord_edit.text().strip()
        if not text:
            if tool == "Polygon" and len(self.canvas._poly) >= 3:
                self.canvas._close_polygon()
                self.hint.setText(self.hints()[tool])
            return
        try:
            shape = self._typed_shape(tool, text) if self.canvas.dim == 2 else self._typed_shape3(tool, text)
        except ValueError as exc:
            self.hint.setText(f"<span style='color:#e03131'>{exc}</span>")
            return
        if shape is None:      # a polygon corner was added
            self.coord_edit.clear()
            self.canvas.update()
            return
        self.canvas._add(shape)
        self.coord_edit.clear()
        self.hint.setText(self.hints()[tool])
        c = self.canvas
        tl, br = c.to_world(QPointF(0, 0)), c.to_world(QPointF(c.width(), c.height()))
        view = QRectF(QPointF(tl.x(), br.y()), QPointF(br.x(), tl.y()))
        outline = shape if c.dim == 2 else s3.projection_shape(shape, c.view)
        if not view.contains(shape_path(outline).boundingRect()):
            c.fit()

    def _typed_shape3(self, tool, text):
        fmt_hint = f"Expected {self.COORD_FORMATS_3D.get(tool, '')}."
        try:
            v = [float(t) for t in text.replace(";", ",").split(",") if t.strip()]
        except ValueError:
            raise ValueError(f"Couldn't read the numbers. {fmt_hint}")
        if tool == "Rectangle" and len(v) == 6:
            if v[0] == v[3] or v[1] == v[4] or v[2] == v[5]:
                raise ValueError("The two corners must differ in x, y and z.")
            p = {}
            for k, a in enumerate("xyz"):
                p[f"{a}_min"], p[f"{a}_max"] = min(v[k], v[k + 3]), max(v[k], v[k + 3])
            return {"type": "Cuboid", "params": p}
        if tool == "Circle" and len(v) == 4:
            if v[3] <= 0:
                raise ValueError("The radius must be positive.")
            return {"type": "Sphere", "params": {"cx": v[0], "cy": v[1], "cz": v[2], "r": v[3]}}
        raise ValueError(fmt_hint)

    def _typed_shape(self, tool, text):
        fmt_hint = f"Expected {self.COORD_FORMATS.get(tool, '')}."
        if tool == "Polygon" and ";" in text:
            pts = parse_vertices(text)
            if len(pts) < 3:
                raise ValueError(f"A polygon needs at least 3 corners, written x,y; x,y; x,y.")
            return polygon_shape(pts)
        try:
            v = [float(t) for t in text.replace(";", ",").split(",") if t.strip()]
        except ValueError:
            raise ValueError(f"Couldn't read the numbers. {fmt_hint}")
        if tool == "Rectangle" and len(v) == 4:
            if v[0] == v[2] or v[1] == v[3]:
                raise ValueError("The two corners must differ in both x and y.")
            return {"type": "Rectangle", "params": {"x_min": min(v[0], v[2]), "x_max": max(v[0], v[2]),
                                                    "y_min": min(v[1], v[3]), "y_max": max(v[1], v[3])}}
        if tool == "Circle" and len(v) == 3:
            if v[2] <= 0:
                raise ValueError("The radius must be positive.")
            return {"type": "Disk", "params": {"cx": v[0], "cy": v[1], "r": v[2]}}
        if tool == "Ellipse" and len(v) in (4, 5):
            if v[2] <= 0 or v[3] <= 0:
                raise ValueError("The semi-axes must be positive.")
            a, b, ang = v[2], v[3], (v[4] if len(v) == 5 else 0.0)
            if b > a:   # DeepXDE wants the semi-major axis first
                a, b, ang = b, a, ang + math.pi / 2
            return {"type": "Ellipse", "params": {"cx": v[0], "cy": v[1], "a": a, "b": b, "angle": round(ang, 10)}}
        if tool == "Polygon" and len(v) == 2:
            self.canvas._poly.append((v[0], v[1]))
            self.hint.setText(f"Corner {len(self.canvas._poly)} added; type the next one, or press Enter on an "
                              "empty field to close the polygon.")
            return None
        raise ValueError(fmt_hint)

    def set_tool(self, tool):
        self.canvas.tool = tool
        self.canvas._poly = []
        self.tool_buttons[tool].setChecked(True)
        self.hint.setText(self.hints()[tool])
        free = tool in ("Select", "Edges")
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor if free else Qt.CursorShape.CrossCursor)
        edges = tool == "Edges"
        typed = tool in self.coord_formats()
        self.coord_entry_label.setVisible(typed)
        self.coord_edit.setVisible(typed)
        self.coord_edit.setPlaceholderText(self.coord_formats().get(tool, ""))
        self.edge_bar.setVisible(edges)
        for w in (self.op_label, self.op_combo, self.delete_btn, self.clear_btn):
            w.setVisible(not edges)
        if edges:
            self.refresh_bcs()
        self.canvas.update()

    # ── Edges → BCs ──────────────────────────────────────────
    def set_bc_provider(self, rows_fn, domain_fn):
        """rows_fn() -> [{"index", "label", "expr"}] for the BC rows that have
        a Where:; domain_fn() -> {"x_min": .., ...} for the names Where: can use."""
        self._bc_rows_fn = rows_fn
        self._domain_fn = domain_fn

    def bc_rows(self):
        try:
            return self._bc_rows_fn() or []
        except Exception:
            return []

    def domain(self):
        try:
            return self._domain_fn() or {}
        except Exception:
            return {}

    active_bc = None   # index of the BC row being edited by clicking edges (None: plain selection)

    def set_active_bc(self, index, label=""):
        """Edit one BC row by clicking edges: its edges are shown selected,
        and a click adds an edge to it or takes it out. None: back to
        selecting edges freely."""
        self.active_bc = index
        self._active_label = label
        self.edge_bar.setEnabled(index is None)
        self._sync_active_selection()
        if index is not None:
            what = "edge" if self.canvas.dim == 2 else "face"
            self._active_hint = (f"Editing {label}: click an {what} to add it to this BC, or click one of its "
                                 f"{what}s to take it out.")
            self.hint.setText(self._active_hint)
        elif self.canvas.tool == "Edges":
            self.hint.setText(self.hints()["Edges"])
        self.canvas.update()

    def _sync_active_selection(self):
        if self.active_bc is not None:
            bc_of = self.canvas.edge_bcs()
            self.canvas.sel_edges = [k for k, lst in enumerate(bc_of) if self.active_bc in lst]
            self._update_edge_bar()

    def toggle_edge_in_active_bc(self, k):
        if self.active_bc is None:
            return
        bc_of = self.canvas.edge_bcs()
        inside = k < len(bc_of) and self.active_bc in bc_of[k]
        self.toggle_edge_bc.emit(self.active_bc, k, not inside)

    def refresh_bcs(self):
        """The BC rows changed: recolour the edges, refresh "Add to"."""
        if not self.edge_bar.isVisible() and self.canvas.tool != "Edges":
            return
        cur = self.add_to_combo.currentData()
        self.add_to_combo.blockSignals(True)
        self.add_to_combo.clear()
        for b in self.bc_rows():
            self.add_to_combo.addItem(b["label"], b["index"])
        i = self.add_to_combo.findData(cur)
        self.add_to_combo.setCurrentIndex(i if i >= 0 else 0)
        self.add_to_combo.blockSignals(False)
        self._update_edge_bar()
        self._sync_active_selection()
        self.canvas.update()

    def selected_where(self):
        edges = self.canvas.edges()
        return join_where([edges[k]["expr"] for k in self.canvas.sel_edges if k < len(edges)])

    def _update_edge_bar(self, from_list=False):
        sel = self.canvas.sel_edges
        n = len(sel)
        pre, what = ("E", "edge") if self.canvas.dim == 2 else ("F", "face")
        self.sel_free_btn.setText(f"Select {what}s without a BC")
        self.edge_sel_label.setText(f"No {what} selected" if not n else
                                    "Selected: " + ", ".join(f"{pre}{k + 1}" for k in sel))
        if not from_list:
            self._fill_face_list()
        self.where_edit.setText(self.selected_where())
        self.where_edit.setCursorPosition(0)
        has_rows = self.add_to_combo.count() > 0
        self.new_bc_btn.setEnabled(n > 0)
        self.add_to_btn.setEnabled(n > 0 and has_rows)
        self.add_to_combo.setEnabled(has_rows)
        bc_of = self.canvas.edge_bcs()
        self.remove_btn.setEnabled(any(k < len(bc_of) and bc_of[k] for k in sel))
        self.sel_free_btn.setEnabled(any(not lst for lst in bc_of))

    def _on_edge_hovered(self, k):
        edges = self.canvas.edges()
        if k < 0 or k >= len(edges):
            self.hint.setText(self._active_hint if self.active_bc is not None else self.hints()["Edges"])
            return
        bc_of = self.canvas.edge_bcs()
        labels = {b["index"]: b["label"] for b in self.bc_rows()}
        on = "; ".join(labels.get(i, f"BC {i + 1}") for i in bc_of[k]) if bc_of[k] else "no BC yet"
        pre = "E" if self.canvas.dim == 2 else "F"
        self.hint.setText(f"{pre}{k + 1}:  {edges[k]['desc']}   —   {on}")

    def _emit_selected(self, signal):
        where = self.selected_where()
        if where:
            signal.emit(where)

    def _on_add_to(self):
        where = self.selected_where()
        idx = self.add_to_combo.currentData()
        if where and idx is not None:
            self.add_to_bc_requested.emit(int(idx), where)

    def _on_remove(self):
        edges = self.canvas.edges()
        exprs = [edges[k]["expr"] for k in self.canvas.sel_edges if k < len(edges)]
        if exprs:
            self.remove_from_bcs_requested.emit(exprs)

    def _select_free_edges(self):
        bc_of = self.canvas.edge_bcs()
        self.canvas.sel_edges = [k for k, lst in enumerate(bc_of) if not lst]
        self._update_edge_bar()
        self.canvas.update()

    def _on_grid(self, v):
        self.canvas.grid = v
        self.canvas.update()

    def _on_snap(self, on):
        self.canvas.snap = on
        self.canvas.update()

    def delete_selected(self):
        c = self.canvas
        if 0 <= c.selected < len(c.shapes):
            del c.shapes[c.selected]
            if c.shapes and "op" in c.shapes[0]:
                c.shapes[0] = {k: v for k, v in c.shapes[0].items() if k != "op"}
            c.selected = -1
            c.update()
            self.shapes_edited.emit([dict(s) for s in c.shapes])

    def import_drawing(self, path=None, scale=None, replace=None):
        """Import a DXF/SVG drawing (asks for the file, the scale and, when
        the sketch isn't empty, whether to replace it)."""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
        from pinnstudio.ui.sketch_import import drawing_extent, import_file
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import a 2D drawing", "",
                                                  "Drawings (*.dxf *.svg);;DXF (*.dxf);;SVG (*.svg)")
            if not path:
                return False
        try:
            shapes, notes = import_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import drawing", f"Couldn't import {path}:\n{exc}")
            return False
        if scale is None:
            w, h = drawing_extent(shapes)
            scale, ok = QInputDialog.getDouble(
                self, "Import drawing",
                f"The drawing is {_fmt(w)} × {_fmt(h)} in its own units.\n"
                "Multiply its coordinates by (e.g. 0.001 for mm → m):", 1.0, 1e-9, 1e9, 6)
            if not ok:
                return False
        if scale != 1.0:
            shapes, notes = import_file(path, scale)
        if replace is None and self.canvas.shapes:
            box = QMessageBox(QMessageBox.Icon.Question, "Import drawing",
                              "Replace the current sketch with the drawing, or add the drawing to it?",
                              parent=self)
            rep_btn = box.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
            add_btn = box.addButton("Add", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() not in (rep_btn, add_btn):
                return False
            replace = box.clickedButton() is rep_btn
        if replace or not self.canvas.shapes:
            new = shapes
        else:
            new = list(self.canvas.shapes)
            for sh in shapes:
                sh = dict(sh)
                sh.setdefault("op", "union")
                new.append(sh)
        self.canvas.shapes = [dict(sh) for sh in new]
        self.canvas.selected = -1
        self.canvas.fit()
        self.shapes_edited.emit([dict(sh) for sh in self.canvas.shapes])
        msg = f"Imported {len(shapes)} shape(s) from {path.replace(chr(92), '/').split('/')[-1]}"
        self.hint.setText(msg + (" -- " + "; ".join(notes) if notes else "."))
        return True

    def clear(self):
        if self.canvas.shapes:
            self.canvas.shapes = []
            self.canvas.selected = -1
            self.canvas.update()
            self.shapes_edited.emit([])

    def set_shapes(self, shapes, fit=False):
        self.canvas.shapes = [dict(s) for s in shapes]
        if self.canvas.selected >= len(self.canvas.shapes):
            self.canvas.selected = -1
        if fit:
            if self.canvas.dim == 3:
                self._centre_depth()
            self.canvas.fit()
        if self.canvas.tool == "Edges":
            self._update_edge_bar()
        self.canvas.update()

    def shapes(self):
        return [dict(s) for s in self.canvas.shapes]
