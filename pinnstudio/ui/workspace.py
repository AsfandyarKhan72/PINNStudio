"""The main window's three-panel workspace, after COMSOL's Model Builder.

    ┌ model tree ┐┌ settings of the selected node ┐┌ graphics (plots, sketch) ┐
    │ Problem    ││ Title                          ││                          │
    │ Equations  ││ one-line description           ││                          │
    │ Geometry   ││ [the node's panels]            │├──────────────────────────┤
    │ ...        ││                                ││ Training Log             │
    └────────────┘└────────────────────────────────┘└──────────────────────────┘

The model tree lists the steps of setting up a problem; selecting one shows
its panels in the settings stack. The panels themselves are the window's
existing QGroupBoxes, moved onto pages -- every widget keeps its name and
its signals, so building and loading a configuration works unchanged.

A node is hidden when every panel on its page is hidden (e.g. "Inverse
problem" outside Inverse mode), and shown again when one is shown.
"""
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (QFrame, QLabel, QScrollArea, QSizePolicy, QStackedWidget, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)


def node_icon(glyph, color, size=18):
    """A small round badge with a glyph -- the tree's node icons."""
    pm = QPixmap(size * 2, size * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    qp = QPainter(pm)
    qp.setRenderHint(QPainter.RenderHint.Antialiasing)
    qp.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    c = QColor(color)
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(c)
    qp.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), 5, 5)
    f = QFont()
    f.setPixelSize(int(size * 0.62))
    f.setBold(True)
    qp.setFont(f)
    qp.setPen(QColor("#0b1a22") if c.lightnessF() > 0.55 else QColor("white"))
    qp.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, glyph)
    qp.end()
    return QIcon(pm)


class ModelWorkspace(QObject):
    """Model tree + settings stack. `tree_panel` and `settings_panel` are
    the two widgets to put in the window's splitter."""
    node_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ── tree ──
        self.tree_panel = QWidget()
        self.tree_panel.setObjectName("treePanel")
        tl = QVBoxLayout(self.tree_panel)
        tl.setContentsMargins(6, 6, 2, 6)
        tl.setSpacing(4)
        head = QLabel("MODEL")
        head.setObjectName("panelHeader")
        tl.addWidget(head)
        self.tree = QTreeWidget()
        self.tree.setObjectName("modelTree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setFrameShape(QFrame.Shape.NoFrame)
        self.tree.setMinimumWidth(170)
        self.tree.currentItemChanged.connect(self._on_current)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.menu_provider = None   # key -> [(label, callback) or None for a separator]
        tl.addWidget(self.tree, 1)

        # ── settings ──
        self.settings_panel = QWidget()
        self.settings_panel.setObjectName("settingsPanel")
        sl = QVBoxLayout(self.settings_panel)
        sl.setContentsMargins(4, 6, 4, 4)
        sl.setSpacing(2)
        head = QLabel("SETTINGS")
        head.setObjectName("panelHeader")
        sl.addWidget(head)
        self.title = QLabel("")
        self.title.setObjectName("pageTitle")
        from PyQt6.QtWidgets import QHBoxLayout, QToolButton
        trow = QHBoxLayout()
        trow.setContentsMargins(0, 0, 4, 0)
        trow.addWidget(self.title)
        trow.addStretch()
        self.help_btn = QToolButton()
        self.help_btn.setObjectName("helpButton")
        self.help_btn.setText("?")
        self.help_btn.setAutoRaise(True)
        self.help_btn.setFixedSize(22, 22)
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.clicked.connect(self._open_help)
        self.help_btn.hide()
        trow.addWidget(self.help_btn)
        sl.addLayout(trow)
        self.help_urls = {}   # key -> URL of its section in the documentation
        self.open_url = None  # callable(url); QDesktopServices when None
        self.desc = QLabel("")
        self.desc.setObjectName("pageDesc")
        self.desc.setWordWrap(True)
        sl.addWidget(self.desc)
        self.banner = QLabel("")
        self.banner.setObjectName("problemBanner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        sl.addWidget(self.banner)
        line = QFrame()
        line.setObjectName("pageRule")
        line.setFixedHeight(1)
        sl.addWidget(line)
        self.stack = QStackedWidget()
        sl.addWidget(self.stack, 1)

        self.pages = {}      # key -> {"item", "scroll", "page", "layout", "label", "desc", "widgets"}
        self.order = []
        self._watched = {}   # widget -> key
        self._vis_pending = False

    # ── building ─────────────────────────────────────────────
    def add_node(self, key, label, glyph, color, desc="", parent=None, widgets=()):
        item = QTreeWidgetItem([label])
        item.setData(0, Qt.ItemDataRole.UserRole, key)
        item.setIcon(0, node_icon(glyph, color))
        if parent:
            self.pages[parent]["item"].addChild(item)
            self.pages[parent]["item"].setExpanded(True)
        else:
            self.tree.addTopLevelItem(item)
        page = QWidget()
        page.setObjectName("settingsPage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(2, 6, 10, 10)
        lay.setSpacing(10)
        lay.addStretch()
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)
        page.installEventFilter(self)      # rows added later are aligned too
        self.stack.addWidget(scroll)
        self.pages[key] = {"item": item, "scroll": scroll, "page": page, "layout": lay, "label": label,
                           "desc": desc, "widgets": []}
        self.order.append(key)
        for w in widgets:
            self.add_widget(key, w)
        return item

    def add_widget(self, key, widget):
        from PyQt6.QtWidgets import QGroupBox
        if isinstance(widget, QGroupBox):
            make_collapsible(widget)
        info = self.pages[key]
        lay = info["layout"]
        lay.insertWidget(lay.count() - 1, widget)   # before the stretch
        info["widgets"].append(widget)
        self._watched[widget] = key
        widget.installEventFilter(self)
        self._schedule_visibility()

    def remove_node(self, key):
        info = self.pages.pop(key, None)
        if info is None:
            return
        self.order.remove(key)
        for w in info["widgets"]:
            self._watched.pop(w, None)
        item = info["item"]
        was_current = self.tree.currentItem() is item
        parent = item.parent()
        if parent is not None:
            parent.removeChild(item)
        else:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        self.stack.removeWidget(info["scroll"])
        info["scroll"].deleteLater()
        if was_current:
            self.tree.setCurrentItem(parent if parent is not None else self.tree.topLevelItem(0))

    def set_label(self, key, label, glyph=None, color=None):
        info = self.pages[key]
        info["label"] = label
        info["item"].setText(0, label)
        if glyph is not None:
            self.set_icon(key, glyph, color)
        if self.current_key() == key:
            self.title.setText(label)

    def set_desc(self, key, desc):
        info = self.pages[key]
        info["desc"] = desc
        if self.current_key() == key:
            self.desc.setText(desc)
            self.desc.setVisible(bool(desc))

    # ── navigation ───────────────────────────────────────────
    def current_key(self):
        it = self.tree.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it else None

    def select(self, key):
        if key in self.pages:
            self.tree.setCurrentItem(self.pages[key]["item"])

    def _on_current(self, item, _prev):
        if item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        info = self.pages[key]
        from PyQt6.QtWidgets import QGroupBox
        for w in info["widgets"]:
            if isinstance(w, QGroupBox):
                make_collapsible(w)
        align_forms(info["page"])
        self.stack.setCurrentWidget(info["scroll"])
        self.title.setText(info["label"])
        url = self.help_url(key)
        self.help_btn.setVisible(bool(url))
        self.help_btn.setToolTip(f"Help: open the documentation for {info['label']}" if url else "")
        self.desc.setText(info["desc"])
        self.desc.setVisible(bool(info["desc"]))
        self._show_banner()
        self.node_selected.emit(key)

    def help_url(self, key):
        """The documentation for node `key`: its own, else its parent's."""
        if key in self.help_urls:
            return self.help_urls[key]
        info = self.pages.get(key)
        parent = info["item"].parent() if info else None
        if parent is not None:
            return self.help_urls.get(parent.data(0, Qt.ItemDataRole.UserRole))
        return None

    def _open_help(self):
        url = self.help_url(self.current_key())
        if not url:
            return
        if self.open_url is not None:
            self.open_url(url)
            return
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))

    def build_menu(self, key):
        """The right-click menu of node `key`: the window's actions for it,
        then folding the page's sections."""
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self.tree)
        entries = []
        if self.menu_provider is not None:
            try:
                entries = self.menu_provider(key) or []
            except Exception:
                entries = []
        for e in entries:
            if e is None:
                menu.addSeparator()
            else:
                label, cb = e[0], e[1]
                act = menu.addAction(label)
                act.triggered.connect(lambda _c=False, f=cb: f())
                if len(e) > 2 and e[2] is not None:
                    from pinnstudio.ui import icons as _icons
                    act.setIcon(_icons.icon(e[2]))
        if self.help_url(key):
            if entries:
                menu.addSeparator()
            h = menu.addAction("Help")
            h.triggered.connect(lambda: (self.select(key), self._open_help()))
        boxes = [getattr(w, "_collapser", None) for w in self.pages.get(key, {}).get("widgets", [])]
        boxes = [c for c in boxes if c is not None and not c.box.isHidden()]
        if boxes:
            if not menu.isEmpty():
                menu.addSeparator()
            fold = menu.addAction("Fold all sections")
            fold.triggered.connect(lambda: [c.toggle() for c in boxes if not c.folded])
            unfold = menu.addAction("Unfold all sections")
            unfold.triggered.connect(lambda: [c.toggle() for c in boxes if c.folded])
        return menu

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        self.select(key)
        menu = self.build_menu(key)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _show_banner(self):
        msgs = (getattr(self, "problems", {}) or {}).get(self.current_key()) or []
        self.banner.setText("\n".join("⚠  " + m for m in msgs))
        self.banner.setVisible(bool(msgs))

    def key_of(self, widget):
        """The node whose page holds `widget` (None if none does)."""
        pages = {info["page"]: k for k, info in self.pages.items()}
        w = widget
        while w is not None:
            if w in pages:
                return pages[w]
            w = w.parentWidget()
        return None

    def reveal(self, widget, margin=60):
        """Select the node holding `widget` and scroll its page to it."""
        key = self.key_of(widget)
        if key is None:
            return False
        self.select(key)
        scroll = self.pages[key]["scroll"]
        QTimer.singleShot(30, lambda: scroll.ensureWidgetVisible(widget, 0, margin))
        return True

    # ── node visibility follows its panels ───────────────────
    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent) and obj in self._watched:
            self._schedule_visibility()
        elif ev.type() == QEvent.Type.LayoutRequest and obj.objectName() == "settingsPage":
            self._schedule_align()
        return False

    def _schedule_align(self):
        """Rows added or shown on the open page (a training phase, a BC,
        the loss weights ...) get the page's label column too."""
        if not getattr(self, "_align_pending", False):
            self._align_pending = True
            QTimer.singleShot(0, self._align_current)

    def _align_current(self):
        self._align_pending = False
        key = self.current_key()
        if key in self.pages:
            align_forms(self.pages[key]["page"])

    def _schedule_visibility(self):
        if not self._vis_pending:
            self._vis_pending = True
            QTimer.singleShot(0, self.refresh_visibility)

    def refresh_visibility(self):
        self._vis_pending = False
        for key, info in self.pages.items():
            ws = info["widgets"]
            if not ws:
                continue
            hidden = all(w.isHidden() for w in ws)
            if info["item"].isHidden() != hidden:
                info["item"].setHidden(hidden)
                if hidden and self.current_key() == key:
                    parent = info["item"].parent()
                    self.tree.setCurrentItem(parent if parent is not None else self.tree.topLevelItem(0))

    def set_problems(self, problems):
        """problems: {key: [messages]}. A node with problems gets a warning
        badge and the messages as its tooltip; the rest get their own icon."""
        for key, info in self.pages.items():
            msgs = problems.get(key) or []
            item = info["item"]
            if "icon" not in info:
                info["icon"] = item.icon(0)
            if msgs:
                item.setIcon(0, node_icon("!", "#ff922b"))
                item.setToolTip(0, "\n".join("• " + m for m in msgs))
            else:
                item.setIcon(0, info["icon"])
                item.setToolTip(0, "")
        self.problems = problems
        self._show_banner()

    def set_icon(self, key, glyph, color):
        info = self.pages[key]
        info["icon"] = node_icon(glyph, color)
        if not (getattr(self, "problems", {}) or {}).get(key):
            info["item"].setIcon(0, info["icon"])

    def visible_keys(self):
        return [k for k in self.order if not self.pages[k]["item"].isHidden()]


class VisibilityFollower(QObject):
    """Keep `container` shown only while at least one of `children` is
    (explicitly) shown -- e.g. a "Shape" box around a row that the window
    hides in 1D."""

    def __init__(self, container, children):
        super().__init__(container)
        self.container = container
        self.children_ = list(children)
        for c in self.children_:
            c.installEventFilter(self)
        self.sync()

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            QTimer.singleShot(0, self.sync)
        return False

    def sync(self):
        try:
            self.container.setVisible(any(not c.isHidden() for c in self.children_))
        except RuntimeError:
            pass


# ── page forms: one label column, controls next to their labels ──────────
ROW_SPACING = 8      # between a row's label and its control
MAX_LABEL_CHARS = 26


def align_forms(root, max_label=190):
    """Rows built as  [label][stretch][control]  -- control pushed to the far
    right -- become  [label][control][stretch], and every row label on the
    page gets the same width, so a page reads as one aligned form.
    Idempotent: safe to call again after rows are added."""
    from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QSpinBox, QDoubleSpinBox
    labels = []

    def walk(lay):
        for i in range(lay.count()):
            it = lay.itemAt(i)
            if it.layout() is not None:
                walk(it.layout())
            elif it.widget() is not None and it.widget().layout() is not None:
                walk(it.widget().layout())
        if not isinstance(lay, QHBoxLayout) or lay.count() < 2:
            return
        first = lay.itemAt(0).widget()
        if not isinstance(first, QLabel) or not first.text().rstrip().endswith(":") or first.wordWrap():
            return
        if lay.itemAt(1).spacerItem() is not None and lay.count() >= 3:
            lay.takeAt(1)
            last = lay.itemAt(lay.count() - 1).widget()
            if not isinstance(last, QLineEdit):
                lay.addStretch(1)
        if first.property("_pinn_col") or first.minimumWidth() != first.maximumWidth() \
                or first.maximumWidth() > 1000:      # (a width the code fixed itself is kept)
            labels.append(first)
            if lay.spacing() != ROW_SPACING:   # rows built in different ways had 3-6 px here
                lay.setSpacing(ROW_SPACING)

    if root.layout() is not None:
        walk(root.layout())
    shown = [l for l in labels if l.isVisibleTo(root)]
    if not shown:
        return
    # The column is as wide as the widest label of up to MAX_LABEL_CHARS
    # characters (counted in characters, not pixels, so it grows with the
    # font: Windows' and display scaling's wider text no longer broke it). A
    # longer label keeps its own width and does not widen every row.
    col = [l for l in shown if len(l.text()) <= MAX_LABEL_CHARS] or shown
    w = max(l.sizeHint().width() for l in col)
    for l in shown:
        own = l not in col
        if l.minimumWidth() != (0 if own else w):   # only when it changes: setting it re-lays the page
            l.setMinimumWidth(0 if own else w)
        # not wider either, so the control next to it starts at the same place
        cap = 16777215 if own else w
        if l.maximumWidth() != cap:
            l.setMaximumWidth(cap)
        l.setProperty("_pinn_col", True)


class _Collapser(QObject):
    """A chevron at a group box's top-right corner folds it to its title
    (by limiting its height, so nothing inside changes state)."""

    def __init__(self, box):
        super().__init__(box)
        from PyQt6.QtWidgets import QToolButton
        self.box = box
        self.btn = QToolButton(box)
        self.btn.setObjectName("collapseButton")
        self.btn.setText("⌄")
        self.btn.setToolTip("Fold / unfold this section")
        self.btn.setAutoRaise(True)
        self.btn.setFixedSize(22, 18)
        self.btn.clicked.connect(self.toggle)
        self.folded = False
        box.installEventFilter(self)
        self._place()

    def _place(self):
        self.btn.move(self.box.width() - self.btn.width() - 8, 0)
        self.btn.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Resize:
            self._place()
        elif ev.type() == QEvent.Type.MouseButtonDblClick and ev.position().y() < 22:
            self.toggle()
            return True
        return False

    def toggle(self):
        self.folded = not self.folded
        if self.folded:
            self.box.setMaximumHeight(self.box.fontMetrics().height() + 22)
            self.btn.setText("›")
        else:
            self.box.setMaximumHeight(16777215)
            self.btn.setText("⌄")


def make_collapsible(box):
    if getattr(box, "_collapser", None) is None:
        box._collapser = _Collapser(box)
    return box._collapser
