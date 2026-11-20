"""The command palette (Ctrl+K): type a few letters to go to a page of the model tree, a section on a
page, or to run any menu command, without hunting through the tree and the menus.

Entries are collected when the palette opens, from the window itself: the model tree's visible pages
(with their descriptions), the section boxes on each page, and every enabled menu action. Matching is
by words: every typed word must start a word of the entry (or occur in it), so "lbfgs", "loss w" and
"sav as" all find what they mean.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QDialog, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout


def _words(text):
    import re
    return [w for w in re.split(r"[^0-9a-zA-Z]+", text.lower()) if w]


def score(query, text):
    """0 when `query` does not match `text`; higher for better matches (word starts beat substrings,
    earlier words beat later ones)."""
    q = _words(query)
    if not q:
        return 1
    words = _words(text)
    low = text.lower()
    compact = "".join(words)          # "L-BFGS Options" -> "lbfgsoptions", so "lbfgs" finds it
    total = 0
    for part in q:
        starts = [i for i, w in enumerate(words) if w.startswith(part)]
        if starts:
            total += 10 - min(starts[0], 9)
        elif part in low or part in compact:
            total += 2
        else:
            return 0
    return total


def collect(window):
    """[(title, detail, kind, callback)] for the window: pages, sections, then menu commands."""
    entries = []
    ws = getattr(window, "workspace", None)
    if ws is not None:
        for key in ws.order:
            info = ws.pages.get(key) or {}
            item = info.get("item")
            if item is None or item.isHidden():
                continue
            label = info.get("label") or key
            entries.append((label, info.get("desc") or "", "Page", lambda k=key: ws.select(k)))
            for w in info.get("widgets", []):
                if isinstance(w, QGroupBox) and not w.isHidden() and w.title().strip():
                    title = w.title().replace("&&", "&")
                    entries.append((f"{label} › {title}", "", "Section",
                                    lambda k=key, box=w: _show_section(ws, k, box)))
    menubar = window.menuBar() if window is not None else None

    def walk(menu, path):
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible():
                continue
            if act.menu() is not None:
                walk(act.menu(), path + [act.text().replace("&", "")])
                continue
            text = act.text().replace("&", "").strip()
            if not text or not act.isEnabled():
                continue
            keys = ", ".join(k.toString(QKeySequence.SequenceFormat.NativeText) for k in act.shortcuts()
                             if not k.isEmpty())
            entries.append((text, " › ".join(path) + (f"   {keys}" if keys else ""), "Command",
                            lambda a=act: a.trigger()))
    if menubar is not None:
        for top in menubar.actions():
            if top.menu() is not None:
                walk(top.menu(), [top.text().replace("&", "")])
    return entries


def _show_section(ws, key, box):
    ws.select(key)
    collapser = getattr(box, "_collapser", None)
    if collapser is not None and getattr(collapser, "folded", False):
        collapser.toggle()
    scroll = (ws.pages.get(key) or {}).get("scroll")
    if scroll is not None:
        scroll.ensureWidgetVisible(box, 0, 40)


class CommandPalette(QDialog):
    MAX_ROWS = 60

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("commandPalette")
        self.setWindowTitle("Go to page or command")
        self.setModal(True)
        self.resize(560, 420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 8)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Type a page, a section or a command (for example: loss weights, L-BFGS, export)")
        self.edit.setClearButtonEnabled(True)
        lay.addWidget(self.edit)
        self.list = QListWidget()
        self.list.setObjectName("paletteList")
        self.list.setUniformItemSizes(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideRight)
        lay.addWidget(self.list, 1)
        hint = QLabel("↑ ↓ to choose · Enter to go · Esc to close")
        hint.setObjectName("pageDesc")
        lay.addWidget(hint)
        self.entries = collect(window)
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._run_current)
        self.list.itemActivated.connect(lambda _i: self._run_current())
        self.edit.installEventFilter(self)
        self._filter("")

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.edit and ev.type() == QEvent.Type.KeyPress and ev.key() in (
                Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
            self.list.setFocus()
            self.list.keyPressEvent(ev)
            self.edit.setFocus()
            return True
        return super().eventFilter(obj, ev)

    def _filter(self, text):
        self.list.clear()
        ranked = []
        for i, (title, detail, kind, cb) in enumerate(self.entries):
            sc = score(text, f"{title} {detail}") if text.strip() else 1
            if sc:
                # pages first when nothing is typed; otherwise the best match, then the original order
                ranked.append((-sc, i, title, detail, kind, cb))
        ranked.sort()
        for _s, _i, title, detail, kind, cb in ranked[:self.MAX_ROWS]:
            it = QListWidgetItem(f"{title}" + (f"    —  {detail}" if detail else ""))
            it.setData(Qt.ItemDataRole.UserRole, cb)
            it.setToolTip(kind)
            self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _run_current(self):
        it = self.list.currentItem()
        if it is None:
            return
        cb = it.data(Qt.ItemDataRole.UserRole)
        self.accept()
        if callable(cb):
            cb()
