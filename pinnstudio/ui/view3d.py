"""3D view of a 3D domain (boxes and spheres combined in order).

The flat views of the sketch show one slice at a time; this one shows the
whole solid, shaded, so a union or a cut-out reads at a glance. Only the
surfaces that actually bound the domain are drawn: each face from
sketch3d.faces3() is split into small patches, and a patch is kept only if
the domain is on one side of it and not the other (so a sphere cut out of a
box shows as a hole, not as two overlapping objects). Each face takes the
colour of its boundary condition and is labelled F1, F2, ... like the face
list; the slice the flat views show is drawn as a translucent plane.

The view is for looking and checking. Faces are still picked in the flat
views or the face list: clicking a face in a matplotlib 3D plot isn't
reliable enough to build on.
"""
import json
import math

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from pinnstudio.ui import sketch3d as s3

# (elevation, azimuth) of the standard orientations, as in COMSOL's view buttons
ORIENTATIONS = {"Iso": (24, -58), "Top": (90, -90), "Front": (0, -90), "Side": (0, 0)}


# ── geometry → patches (no Qt; tested on its own) ───────────────────────
def _plane_patches(face, shapes, n, eps):
    a, val, b = face["axis"], face["value"], face["bounds"]
    o = [k for k in range(3) if k != a]
    (u0, u1), (v0, v1) = b[o[0]], b[o[1]]
    du, dv = (u1 - u0) / n, (v1 - v0) / n

    def bounds_here(i, j):
        q = [0.0, 0.0, 0.0]
        q[a], q[o[0]], q[o[1]] = val, u0 + (i + 0.5) * du, v0 + (j + 0.5) * dv
        qa, qb = list(q), list(q)
        qa[a] += eps
        qb[a] -= eps
        return s3.in_region3(shapes, qa) != s3.in_region3(shapes, qb)

    def quad(ua, ub, va, vb):
        out = []
        for uu, vv in ((ua, va), (ub, va), (ub, vb), (ua, vb)):
            p = [0.0, 0.0, 0.0]
            p[a], p[o[0]], p[o[1]] = val, uu, vv
            out.append(tuple(p))
        return out

    keep = [[bounds_here(i, j) for i in range(n)] for j in range(n)]
    if all(all(row) for row in keep):          # nothing cuts this face: draw it whole (no seams)
        return [quad(u0, u1, v0, v1)]
    out = []
    for j, row in enumerate(keep):             # each row's runs of kept cells as one strip
        i = 0
        while i < n:
            if not row[i]:
                i += 1
                continue
            i0 = i
            while i < n and row[i]:
                i += 1
            out.append(quad(u0 + i0 * du, u0 + i * du, v0 + j * dv, v0 + (j + 1) * dv))
    return out


def _sphere_patches(face, shapes, n_lat, n_lon, eps):
    c, r = face["center"], face["r"]

    def pt(th, ph, rr=r):
        return (c[0] + rr * math.sin(th) * math.cos(ph), c[1] + rr * math.sin(th) * math.sin(ph),
                c[2] + rr * math.cos(th))

    out = []
    for i in range(n_lat):
        t0, t1 = math.pi * i / n_lat, math.pi * (i + 1) / n_lat
        for j in range(n_lon):
            p0, p1 = 2 * math.pi * j / n_lon, 2 * math.pi * (j + 1) / n_lon
            tm, pm = (t0 + t1) / 2, (p0 + p1) / 2
            if s3.in_region3(shapes, pt(tm, pm, r + eps)) == s3.in_region3(shapes, pt(tm, pm, r - eps)):
                continue
            out.append([pt(t0, p0), pt(t1, p0), pt(t1, p1), pt(t0, p1)])
    return out


def face_patches(shapes, faces=None, n=24, n_lat=24, n_lon=40):
    """Per face of the domain (same order as sketch3d.faces3), the list of
    quadrilaterals (4 xyz points each) of that face that bound the domain."""
    faces = s3.faces3(shapes) if faces is None else faces
    lo, hi = s3.bbox3(shapes)
    L = max(hi[k] - lo[k] for k in range(3)) or 1.0
    eps = 1e-6 * L
    out = []
    for f in faces:
        if f["kind"] == "plane":
            out.append(_plane_patches(f, shapes, n, eps))
        else:
            out.append(_sphere_patches(f, shapes, n_lat, n_lon, eps))
    return out


_LIGHT = (-0.45, -0.55, 0.70)          # from the front-left, above


def shaded_colours(quads, colour):
    """One colour per quad: the face colour, darker the more the quad turns
    away from the light (both sides lit, since faces are seen from either)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(colour)
    ln = math.sqrt(sum(x * x for x in _LIGHT))
    out = []
    for q in quads:
        u = [q[2][k] - q[0][k] for k in range(3)]
        v = [q[3][k] - q[1][k] for k in range(3)]
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        nn = math.sqrt(sum(x * x for x in n)) or 1.0
        f = 0.62 + 0.38 * abs(sum(n[k] * _LIGHT[k] for k in range(3))) / (nn * ln)
        out.append((r * f, g * f, b * f))
    return out


def face_label_point(patches):
    """Where to put a face's label: the patch centre nearest the mean of all
    its patch centres (so the label sits on the face, even with a hole)."""
    if not patches:
        return None
    cs = [tuple(sum(p[k] for p in q) / 4 for k in range(3)) for q in patches]
    m = tuple(sum(c[k] for c in cs) / len(cs) for k in range(3))
    return min(cs, key=lambda c: sum((c[k] - m[k]) ** 2 for k in range(3)))


def inside_mask(shapes, pts):
    """in_region3 for many points at once (numpy): True where the point is
    inside the domain."""
    import numpy as np
    pts = np.asarray(pts, dtype=float)
    inside = None
    for i, sh in enumerate(shapes):
        t, p = sh.get("type"), sh.get("params") or {}
        here = np.zeros(len(pts), dtype=bool)
        try:
            if t == "Cuboid":
                b = s3._box(p)
                here = np.all([(pts[:, k] > b[k][0]) & (pts[:, k] < b[k][1]) for k in range(3)], axis=0)
            elif t == "Sphere":
                c, r = s3._sph(p)
                here = np.sum((pts - np.asarray(c)) ** 2, axis=1) < r * r
        except (KeyError, TypeError, ValueError):
            pass
        if i == 0:
            inside = here
            continue
        op = sh.get("op", "union")
        inside = (inside & ~here) if op == "subtract" else (inside & here) if op == "intersect" else (inside | here)
    return inside if inside is not None else np.zeros(len(pts), dtype=bool)


def sample_points(shapes, patches, n_domain=1500, n_boundary=600, seed=0):
    """A preview of where training points fall: n_domain random points inside
    the domain (drawn in its bounding box, kept if inside) and n_boundary on
    its boundary (on the drawn face patches, by area). Uniform random, as
    DeepXDE's default sampler; not the run's exact points, which are drawn
    when training starts."""
    import numpy as np
    rng = np.random.default_rng(seed)
    lo, hi = (np.asarray(v, dtype=float) for v in s3.bbox3(shapes))
    inner = np.empty((0, 3))
    for _ in range(20):                       # rejection sampling; stops early for a full box
        cand = lo + (hi - lo) * rng.random((max(n_domain, 256) * 2, 3))
        inner = np.vstack([inner, cand[inside_mask(shapes, cand)]])
        if len(inner) >= n_domain:
            break
    inner = inner[:n_domain]
    quads = [np.asarray(q, dtype=float) for face in patches for q in face]
    if not quads:
        return inner, np.empty((0, 3))
    areas = np.array([0.5 * (np.linalg.norm(np.cross(q[1] - q[0], q[3] - q[0]))
                             + np.linalg.norm(np.cross(q[1] - q[2], q[3] - q[2]))) for q in quads])
    pick = rng.choice(len(quads), size=n_boundary, p=areas / areas.sum())
    u, v = rng.random(n_boundary), rng.random(n_boundary)
    Q = np.asarray(quads)[pick]
    bnd = ((1 - u)[:, None] * (1 - v)[:, None] * Q[:, 0] + u[:, None] * (1 - v)[:, None] * Q[:, 1]
           + u[:, None] * v[:, None] * Q[:, 2] + (1 - u)[:, None] * v[:, None] * Q[:, 3])
    return inner, bnd


# ── the widget ───────────────────────────────────────────────────────────
class Domain3DView(QWidget):
    """Rotatable 3D view of the sketch canvas's shapes. `canvas` is the
    sketch's _Canvas (shapes, face BCs, view and slice depth come from it)."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas_src = canvas
        self._fig = self._ax = self._mpl = None
        self._key = None
        self._mesh_key = None
        self._patches = []
        self._points = None
        self._view_angles = ORIENTATIONS["Iso"]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.orient_buttons = {}
        for name in ORIENTATIONS:
            b = QPushButton(name)
            b.setFixedHeight(26)
            b.setToolTip({"Iso": "Look at the domain from an angle",
                          "Top": "Look down the z axis (x–y)",
                          "Front": "Look along the y axis (x–z)",
                          "Side": "Look along the x axis (y–z)"}[name])
            b.clicked.connect(lambda _c, n=name: self.set_orientation(n))
            bar.addWidget(b)
            self.orient_buttons[name] = b
        bar.addSpacing(12)
        self.points_cb = QCheckBox("Show points")
        self.points_cb.setToolTip("A preview of where training points fall: random points inside the domain and on "
                                  "its boundary, as DeepXDE samples them. Check that none land in a cut-out. "
                                  "These are not the run's exact points, which are drawn when training starts.")
        self.points_cb.toggled.connect(lambda _on: self.refresh())
        bar.addWidget(self.points_cb)
        bar.addStretch()
        lay.addLayout(bar)
        self.holder = QVBoxLayout()
        self.holder.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self.holder, 1)
        self.empty = QLabel("The 3D view needs matplotlib.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.hide()
        self.holder.addWidget(self.empty)
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self.refresh)

    # shown / hidden with the "3D" view
    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
        self._timer.start()

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def _ensure_canvas(self):
        if self._mpl is not None:
            return True
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
        except Exception:
            self.empty.show()
            return False
        self._fig = Figure(figsize=(6, 5), dpi=100)
        self._mpl = FigureCanvasQTAgg(self._fig)
        self._mpl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.holder.addWidget(self._mpl, 1)
        return True

    def set_orientation(self, name):
        self._view_angles = ORIENTATIONS[name]
        if self._ax is not None:
            self._ax.view_init(*self._view_angles)
            self._mpl.draw_idle()

    def _state(self):
        """Everything the picture depends on, to redraw only on a change."""
        from pinnstudio.ui import theme as _theme
        c = self.canvas_src
        bc_of = c.edge_bcs() if c.shapes else []
        rows = c.owner.bc_rows() if c.shapes else []
        return json.dumps([c.shapes, bc_of, [(b["index"], b["label"]) for b in rows],
                           c.view, round(c.depth, 9), _theme.current().get("card"), self.points_cb.isChecked()],
                          sort_keys=True, default=str)

    def refresh(self):
        if not self.isVisible():
            return
        key = self._state()
        if key == self._key:
            return
        self._key = key
        if self._ensure_canvas():
            try:
                self._draw()
            except Exception as exc:          # a drawing problem must never take the window down
                self._ax.clear()
                self._ax.set_axis_off()
                self._ax.text2D(0.5, 0.5, f"Could not draw the 3D view:\n{exc}", transform=self._ax.transAxes,
                                ha="center", va="center")
                self._mpl.draw_idle()

    def _draw(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from pinnstudio.ui import theme as _theme
        from pinnstudio.ui.sketch_canvas import NO_BC_COLOR
        from pinnstudio.ui.theme import BC_COLORS
        t = _theme.current()
        c = self.canvas_src
        ax, fig = self._ax, self._fig
        # keep the user's rotation across redraws
        if ax.has_data():
            self._view_angles = (ax.elev, ax.azim)
        ax.clear()
        fig.set_facecolor(t["card"])
        ax.set_facecolor(t["card"])
        shapes = c.shapes
        if not shapes:
            ax.set_axis_off()
            ax.text2D(0.5, 0.5, "No 3D domain yet", transform=ax.transAxes, ha="center", va="center",
                      color=t["text_muted"])
            self._mpl.draw_idle()
            return
        faces = c.edges()
        mkey = json.dumps(shapes, sort_keys=True)
        if mkey != self._mesh_key:
            self._patches = face_patches(shapes, faces)
            self._points = None
            self._mesh_key = mkey
        bc_of = c.edge_bcs()
        for k, quads in enumerate(self._patches):
            if not quads:
                continue
            col = BC_COLORS[bc_of[k][0] % len(BC_COLORS)] if k < len(bc_of) and bc_of[k] else NO_BC_COLOR
            poly = Poly3DCollection(quads, facecolors=shaded_colours(quads, col), edgecolors="none",
                                    alpha=0.22 if self.points_cb.isChecked() else 0.6,
                                    linewidths=0, antialiased=False)
            ax.add_collection3d(poly)
            lp = face_label_point(quads)
            if lp is not None:
                ax.text(*lp, f"F{k + 1}", color=t["text"], fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc=t["card"], ec=col, lw=0.8, alpha=0.9))
        n_inner = n_bnd = 0
        if self.points_cb.isChecked():
            if self._points is None:
                self._points = sample_points(shapes, self._patches)
            inner, bnd = self._points
            n_inner, n_bnd = len(inner), len(bnd)
            if n_inner:
                ax.scatter(inner[:, 0], inner[:, 1], inner[:, 2], s=2, c=t["text_muted"], alpha=0.45,
                           depthshade=False, linewidths=0)
            if n_bnd:
                ax.scatter(bnd[:, 0], bnd[:, 1], bnd[:, 2], s=3, c=t["text"], alpha=0.7, depthshade=False,
                           linewidths=0)
        lo, hi = s3.bbox3(shapes)
        # the slice the flat views show, as a translucent plane
        if c.view in s3.VIEWS:
            h, v, d = s3.VIEWS[c.view]
            if lo[d] <= c.depth <= hi[d]:
                corners = []
                for uh, uv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = [0.0, 0.0, 0.0]
                    p[d] = c.depth
                    p[h] = lo[h] + uh * (hi[h] - lo[h])
                    p[v] = lo[v] + uv * (hi[v] - lo[v])
                    corners.append(tuple(p))
                ax.add_collection3d(Poly3DCollection([corners], facecolors=t["accent"], edgecolors=t["accent"],
                                                     alpha=0.12, linewidths=0.8))
        # equal scale on all three axes, with a little margin
        span = max(hi[k] - lo[k] for k in range(3)) or 1.0
        mid = [(lo[k] + hi[k]) / 2 for k in range(3)]
        for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(mid[k] - 0.55 * span, mid[k] + 0.55 * span)
        ax.set_box_aspect((1, 1, 1))
        for name, setter in (("x", ax.set_xlabel), ("y", ax.set_ylabel), ("z", ax.set_zlabel)):
            setter(name, color=t["text_muted"], labelpad=2)
        ax.tick_params(colors=t["text_muted"], labelsize=7, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1, 1, 1, 0))
            axis.line.set_color(t["border_strong"])
            axis._axinfo["grid"]["color"] = t["border"]
        # legend: which colour is which BC row
        rows = c.owner.bc_rows()
        used = {i for lst in bc_of for i in lst}
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=BC_COLORS[b["index"] % len(BC_COLORS)], label=b["label"])
                   for b in rows if b["index"] in used]
        if any(not lst for lst in bc_of):
            handles.append(Patch(facecolor=NO_BC_COLOR, label="no BC yet"))
        if n_inner or n_bnd:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], ls="none", marker="o", ms=3, color=t["text_muted"],
                                  label=f"domain points (preview, {n_inner})"))
            handles.append(Line2D([], [], ls="none", marker="o", ms=3, color=t["text"],
                                  label=f"boundary points (preview, {n_bnd})"))
        if handles:
            leg = ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
            leg.get_frame().set_facecolor(t["card"])
            leg.get_frame().set_edgecolor(t["border"])
            for txt in leg.get_texts():
                txt.set_color(t["text"])
        ax.view_init(*self._view_angles)
        self._mpl.draw_idle()
