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
"""
import json
import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF,
                         QTransform)
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from pinnstudio.ui.sketch_edges import boundary_edges, join_where, where_matches

TOOLS = ("Select", "Rectangle", "Circle", "Ellipse", "Polygon", "Edges")
OPS = (("Union (add)", "union"), ("Subtract (cut hole)", "subtract"), ("Intersect (keep overlap)", "intersect"))
OP_COLORS = {None: QColor("#1f6fd1"), "union": QColor("#1f6fd1"),
             "subtract": QColor("#d9480f"), "intersect": QColor("#2b8a3e")}
# one colour per Boundary Conditions row (cycled), and edges without one
BC_COLORS = ("#1971c2", "#e8590c", "#2f9e44", "#9c36b5", "#c2255c", "#0c8599", "#f08c00", "#5c940d",
             "#3b5bdb", "#a61e4d")
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


def combined_path(shapes):
    """The CSG result of the shape list, in list order (DeepXDE's own order)."""
    result = None
    for i, sh in enumerate(shapes):
        p = shape_path(sh)
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

    def fit(self):
        """Zoom to the shapes (or to [0, 1]² when there are none)."""
        r = combined_path(self.shapes).boundingRect()
        for sh in self.shapes:
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
        """The domain's boundary edges (recomputed only when the shapes change)."""
        key = json.dumps(self.shapes, sort_keys=True)
        if key != self._edges_key:
            self._edges = boundary_edges(self.shapes)
            self._edges_key = key
            self.sel_edges = []
            self.hover_edge = -1
        return self._edges

    def edge_bcs(self):
        """Per edge, the indices of the BC rows whose "Where:" picks it."""
        edges = self.edges()
        rows = self.owner.bc_rows()
        dom = self.owner.domain()
        key = (self._edges_key, tuple((b["index"], b["expr"]) for b in rows), tuple(sorted(dom.items())))
        if key != self._bc_key:
            self._bc_key = key
            self._bc_of = [[b["index"] for b in rows if where_matches(b["expr"], e["mid"][0], e["mid"][1], dom)]
                           for e in edges]
        return self._bc_of

    def _edge_at(self, q, px=8.0):
        best, best_d = -1, px
        for k, e in enumerate(self.edges()):
            pts = [self.to_screen(QPointF(x, y)) for x, y in e["points"]]
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
                        "Draw the domain first, then its edges show up here")
            return
        f = QFont(self.font())
        f.setPointSizeF(9.0)
        f.setBold(True)
        qp.setFont(f)
        fm = qp.fontMetrics()
        for k, e in enumerate(edges):
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
        for k, e in enumerate(edges):
            color = QColor(BC_COLORS[bc_of[k][0] % len(BC_COLORS)]) if bc_of[k] else QColor(NO_BC_COLOR)
            m = self.to_screen(QPointF(*e["mid"]))
            ox, oy = e["out"]
            text = f"E{k + 1}"
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
        qp.drawText(QPointF(self.width() - 14, o.y() - 5), "x")
        qp.drawText(QPointF(o.x() + 5, 12), "y")

        wt = self.world_transform()
        # the resulting domain, filled
        dom = wt.map(combined_path(self.shapes))
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
        for i, sh in enumerate(self.shapes):
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
                        "Pick a tool above and draw on the grid")
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
        for i in range(len(self.shapes) - 1, -1, -1):
            p = shape_path(self.shapes[i])
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
                self.shapes[self.selected] = translate_shape(self._drag[2], dx, dy)
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
        bar.addSpacing(8)
        self.op_label = QLabel("New shape:")
        bar.addWidget(self.op_label)
        self.op_combo = QComboBox()
        for label, val in OPS:
            self.op_combo.addItem(label, val)
        self.op_combo.setFixedHeight(26)
        self.op_combo.setToolTip("How the next shape you draw is combined with the shapes before it "
                                 "(the first shape is always the starting domain)")
        bar.addWidget(self.op_combo)
        bar.addStretch()
        lay.addLayout(bar)

        bar2 = QHBoxLayout()
        bar2.setSpacing(4)
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
        bar2.addSpacing(8)
        self.delete_btn = QPushButton("✕ Delete shape")
        self.delete_btn.setFixedHeight(26)
        bar2.addWidget(self.delete_btn)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.setFixedHeight(26)
        bar2.addWidget(self.clear_btn)
        self.fit_btn = QPushButton("⤢ Fit view")
        self.fit_btn.setFixedHeight(26)
        bar2.addWidget(self.fit_btn)
        bar2.addStretch()
        self.coord_label = QLabel("")
        self.coord_label.setMinimumWidth(150)
        bar2.addWidget(self.coord_label)
        lay.addLayout(bar2)

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
        self.edge_sel_label.setMinimumWidth(150)
        r1.addWidget(self.edge_sel_label)
        self.new_bc_btn = QPushButton("＋ New BC on selected")
        self.new_bc_btn.setToolTip("Add a Boundary Conditions row (Dirichlet, value 0) whose Where: picks "
                                   "the selected edges; set its type and value in the panel on the left")
        self.add_to_combo = QComboBox()
        self.add_to_combo.setMinimumWidth(170)
        self.add_to_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.add_to_btn = QPushButton("Add to")
        self.add_to_btn.setToolTip("Add the selected edges to this existing BC row's Where:")
        self.remove_btn = QPushButton("Remove from BC")
        self.remove_btn.setToolTip("Take the selected edges out of the BC rows that pick them "
                                   "(only where Where: lists the edge as the sketch wrote it)")
        self.sel_free_btn = QPushButton("Select edges without a BC")
        for b in (self.new_bc_btn, self.add_to_btn, self.remove_btn, self.sel_free_btn, self.add_to_combo):
            b.setFixedHeight(26)
        r1.addWidget(self.new_bc_btn)
        r1.addSpacing(6)
        r1.addWidget(self.add_to_btn)
        r1.addWidget(self.add_to_combo)
        r1.addWidget(self.remove_btn)
        r1.addStretch()
        r1.addWidget(self.sel_free_btn)
        eb.addLayout(r1)
        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addWidget(QLabel("Where:"))
        self.where_edit = QLineEdit()
        self.where_edit.setReadOnly(True)
        self.where_edit.setPlaceholderText("the selected edges' expression appears here")
        self.where_edit.setFixedHeight(26)
        r2.addWidget(self.where_edit, 1)
        eb.addLayout(r2)
        lay.addWidget(self.edge_bar)

        self._bc_rows_fn = lambda: []
        self._domain_fn = lambda: {}

        self.grid_spin.valueChanged.connect(self._on_grid)
        self.snap_cb.toggled.connect(self._on_snap)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn.clicked.connect(self.clear)
        self.fit_btn.clicked.connect(self.canvas.fit)
        self.canvas.changed.connect(lambda: self.shapes_edited.emit([dict(s) for s in self.canvas.shapes]))
        self.canvas.cursor_moved.connect(lambda x, y: self.coord_label.setText(f"x = {_fmt(x)},  y = {_fmt(y)}"))
        self.canvas.edges_selected.connect(self._update_edge_bar)
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

    def set_tool(self, tool):
        self.canvas.tool = tool
        self.canvas._poly = []
        self.tool_buttons[tool].setChecked(True)
        self.hint.setText(self.HINTS[tool])
        free = tool in ("Select", "Edges")
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor if free else Qt.CursorShape.CrossCursor)
        edges = tool == "Edges"
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
        self.canvas.update()

    def selected_where(self):
        edges = self.canvas.edges()
        return join_where([edges[k]["expr"] for k in self.canvas.sel_edges if k < len(edges)])

    def _update_edge_bar(self):
        sel = self.canvas.sel_edges
        n = len(sel)
        self.edge_sel_label.setText("No edge selected" if not n else
                                    "Selected: " + ", ".join(f"E{k + 1}" for k in sel))
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
            self.hint.setText(self.HINTS["Edges"])
            return
        bc_of = self.canvas.edge_bcs()
        labels = {b["index"]: b["label"] for b in self.bc_rows()}
        on = "; ".join(labels.get(i, f"BC {i + 1}") for i in bc_of[k]) if bc_of[k] else "no BC yet"
        self.hint.setText(f"E{k + 1}:  {edges[k]['desc']}   —   {on}")

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
            self.canvas.fit()
        self.canvas.update()

    def shapes(self):
        return [dict(s) for s in self.canvas.shapes]
