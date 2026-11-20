"""The designer's palette: the blocks to drag onto the canvas (or double-click to add in the middle), the
ready-made designs, and the blocks saved in the library (*My blocks*)."""
from __future__ import annotations

import json

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

from pinnstudio.core.netdesign.layers import LAYERS
from pinnstudio.ui.designer.canvas import MIME

ROLE = Qt.ItemDataRole.UserRole

SECTIONS = (
    ("Blocks", (
        ("Input", "input", "", "Reads the network's inputs (x, y, z, t, or the input transform's features)"),
        ("Stack", "stack", "dense", "Layers run in order, top first"),
        ("Merge", "merge", "concatenate", "Combines what reaches it: concatenate, add, multiply or average"),
        ("Head", "head", "", "Gives outputs by name (a linear layer to that many values)"),
        ("Group", "group", "", "A design of its own used as one block (double-click to open)"),
        ("Shared", "shared", "", "Calls the selected stack, group or saved module again, with the same weights"),
        ("Saved module", "saved", "", "A TorchScript module from a file, frozen or fine-tuned"),
        ("Port in", "port_in", "", "Inside a group: one of the group's inputs"),
        ("Port out", "port_out", "", "Inside a group: its output"),
    )),
    ("Layers", tuple(
        (lt.title, "stack", name, lt.description) for name, lt in LAYERS.items())),
    ("Merges", tuple(
        (f"Merge: {m}", "merge", m, "") for m in ("concatenate", "add", "multiply", "average"))),
)


class BlockPalette(QListWidget):
    add_requested = Signal(str, str)        # kind, preset: add in the middle of the canvas
    preset_requested = Signal(str)          # a ready-made design, in place of this one
    library_requested = Signal(str)         # a saved block file, added in the middle

    def __init__(self, presets: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("designPalette")
        self.setAccessibleName("Block palette")
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.presets = presets
        self.itemActivated.connect(self._activated)
        self.set_library([])

    def changeEvent(self, ev):
        super().changeEvent(ev)
        from PySide6.QtCore import QEvent
        if ev.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._fit_to_font()

    def _fit_to_font(self):
        """Section titles bold in the list's font (the theme's size, which follows the UI scale), and the
        list as wide as its longest entry needs, up to 18 characters."""
        from PySide6.QtGui import QFont, QFontMetrics
        bold = QFont(self.font())
        bold.setBold(True)
        fm = QFontMetrics(self.font())
        widest = 0
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() == Qt.ItemFlag.NoItemFlags and it.text().isupper():
                it.setFont(bold)            # (section titles may be cut: their entries say enough)
            elif it.flags() != Qt.ItemFlag.NoItemFlags:
                widest = max(widest, fm.horizontalAdvance(it.text()))
        ch = fm.averageCharWidth()
        self.setMinimumWidth(min(widest + ch * 4, ch * 22))

    def _header(self, text):
        it = QListWidgetItem(text.upper())
        it.setFlags(Qt.ItemFlag.NoItemFlags)
        self.addItem(it)            # bold in the list's own font: _fit_to_font

    def _entry(self, label, data, tip=""):
        it = QListWidgetItem(label)
        it.setData(ROLE, data)
        if tip:
            it.setToolTip(tip)
        self.addItem(it)

    def set_library(self, blocks) -> None:
        """Rebuild the list, with the library's [(name, path)] under My blocks."""
        self.clear()
        for title, entries in SECTIONS:
            self._header(title)
            for label, kind, preset, tip in entries:
                self._entry(label, {"kind": kind, "preset": preset}, tip)
        self._header("Ready-made designs")
        for key, label in self.presets.items():
            self._entry(label, {"design": key}, "Puts this design in place of the current one (Undo brings it back)")
        self._header("My blocks")
        if not blocks:
            it = QListWidgetItem("(Save block… keeps a block here)")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.addItem(it)
        for name, path in blocks:
            self._entry(name, {"library": path}, path)
        self._fit_to_font()

    def _activated(self, item):
        data = item.data(ROLE) or {}
        if "kind" in data:
            self.add_requested.emit(data["kind"], data["preset"])
        elif "design" in data:
            self.preset_requested.emit(data["design"])
        elif "library" in data:
            self.library_requested.emit(data["library"])

    def mimeData(self, items):
        data = (items[0].data(ROLE) or {}) if items else {}
        md = QMimeData()
        if "kind" in data:
            md.setData(MIME, json.dumps(data).encode("utf-8"))
        elif "library" in data:
            md.setData(MIME, json.dumps({"kind": "library", "preset": data["library"]}).encode("utf-8"))
        return md

    def startDrag(self, actions):
        item = self.currentItem()
        data = (item.data(ROLE) or {}) if item else {}
        if "kind" not in data and "library" not in data:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData([item]))
        drag.exec(Qt.DropAction.CopyAction)
