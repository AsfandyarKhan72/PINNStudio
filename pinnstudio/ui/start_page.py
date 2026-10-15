"""The Start tab: begin a new problem, open a saved one or a recent one, or
load an example from the gallery."""
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
                             QVBoxLayout, QWidget)

from pinnstudio.core import example_catalog as _cat
from pinnstudio.ui import icons as _icons

NEW_KINDS = (("1D", "1D problem", "x and t, e.g. heat, Burgers"),
             ("2D", "2D problem", "x, y and t, any 2D shape"),
             ("3D", "3D problem", "x, y, z and t, any 3D shape"),
             ("ODE", "ODE system", "t only, e.g. Lotka-Volterra"))


class _Card(QPushButton):
    """A button with a title line and a muted line under it."""

    def __init__(self, title, meta, icon=None, parent=None):
        super().__init__(parent)
        self.setObjectName("startCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(10)
        if icon:
            ic = QLabel()
            self._icon, self._icon_label = icon, ic
            ic.setPixmap(_icons.pixmap(icon, _icons._color("accent"), 22))
            ic.setStyleSheet("background: transparent;")
            ic.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            lay.addWidget(ic, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = QLabel(title)
        self.title.setObjectName("startCardTitle")
        self.meta = QLabel(meta)
        self.meta.setObjectName("startCardMeta")
        self.meta.setWordWrap(True)
        for lbl in (self.title, self.meta):
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            col.addWidget(lbl)
        lay.addLayout(col, 1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def sizeHint(self):
        return self.layout().sizeHint()

    def minimumSizeHint(self):
        h = self.layout().minimumSize()
        h.setWidth(140)
        return h


class StartPage(QScrollArea):
    new_requested = pyqtSignal(str)        # "1D" / "2D" / "3D" / "ODE"
    open_requested = pyqtSignal()
    open_path = pyqtSignal(str)
    example_requested = pyqtSignal(str)    # an example key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("startScroll")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("startPage")
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(6)

        title = QLabel("PINNStudio")
        title.setObjectName("startTitle")
        sub = QLabel("Physics-informed neural networks for forward and inverse problems -- "
                     "set a problem up on the left, then Solve.")
        sub.setObjectName("startSubtitle")
        sub.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(sub)

        lay.addWidget(self._section("NEW"))
        grid = QGridLayout()
        grid.setSpacing(8)
        self.new_cards = {}
        for i, (dim, name, meta) in enumerate(NEW_KINDS):
            c = _Card(name, meta, "add")
            c.clicked.connect(lambda _c=False, d=dim: self.new_requested.emit(d))
            self.new_cards[dim] = c
            grid.addWidget(c, i // 2, i % 2)
        self.open_card = _Card("Open a saved problem…", "A .pinn.json file saved from PINNStudio", "open")
        self.open_card.clicked.connect(self.open_requested.emit)
        grid.addWidget(self.open_card, 2, 0, 1, 2)
        lay.addLayout(grid)

        self.recent_header = self._section("RECENT")
        lay.addWidget(self.recent_header)
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(0)
        lay.addLayout(self.recent_box)
        self.recent_buttons = []

        lay.addWidget(self._section("EXAMPLES"))
        self.example_cards = {}
        last = None
        grid = None
        n = 0
        for e in _cat.for_category(_cat.FORWARD):
            if e.dim != last:
                dl = QLabel(e.dim)
                dl.setObjectName("startCardMeta")
                dl.setStyleSheet("font-weight: 600; padding-top: 6px;")
                lay.addWidget(dl)
                grid = QGridLayout()
                grid.setSpacing(8)
                lay.addLayout(grid)
                last, n = e.dim, 0
            geo = "" if (e.geometry or "").startswith("—") else e.geometry
            meta = " · ".join(x for x in (e.regime, geo,
                                          "DeepXDE" if e.source == _cat.DEEPXDE else "PINNStudio") if x)
            c = _Card(e.name, meta)
            c.setToolTip(e.reference or "")
            c.clicked.connect(lambda _c=False, k=e.key: self.example_requested.emit(k))
            self.example_cards[e.key] = c
            grid.addWidget(c, n // 2, n % 2)
            n += 1
        lay.addStretch(1)
        self.set_recent([])

    def retheme(self):
        """Redraw the card icons in the current theme's accent colour."""
        for c in list(self.new_cards.values()) + [self.open_card]:
            if getattr(c, "_icon_label", None) is not None:
                c._icon_label.setPixmap(_icons.pixmap(c._icon, _icons._color("accent"), 22))

    @staticmethod
    def _section(text):
        lbl = QLabel(text)
        lbl.setObjectName("startSection")
        return lbl

    def set_recent(self, paths):
        for b in self.recent_buttons:
            b.setParent(None)
            b.deleteLater()
        self.recent_buttons = []
        for p in paths:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)
            name = os.path.basename(p)
            if name.endswith(".pinn.json"):
                name = name[:-len(".pinn.json")]
            link = QPushButton(name)
            link.setObjectName("recentLink")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.setToolTip(p)
            link.clicked.connect(lambda _c=False, q=p: self.open_path.emit(q))
            where = QLabel(os.path.dirname(p))
            where.setObjectName("recentPath")
            h.addWidget(link)
            h.addWidget(where, 1)
            self.recent_box.addWidget(row)
            self.recent_buttons.append(row)
        self.recent_header.setVisible(bool(paths))
