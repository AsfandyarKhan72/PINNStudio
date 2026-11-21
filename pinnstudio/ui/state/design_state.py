"""The network designer's state: the design, the level on view, the selection, undo and redo.

Every change is a whole new design dict (`set_design`), kept as one snapshot, so one edit is one step of
Undo. The widths and problems of the design are worked out here, by the domain (`netdesign.shapes`),
whenever the design or the problem's inputs and outputs change, so every view sees the same.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import QObject, Signal

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign import shapes as S

UNDO_DEPTH = 200

# The methods that change the state: only controllers may call them (tests/test_mvc.py, rule R5).
MUTATORS = ("set_design", "reset", "undo", "redo", "set_path", "select", "set_io", "set_message")


class DesignState(QObject):
    design_changed = Signal()        # the design (or the widths and problems with it)
    path_changed = Signal()          # the level on view: the top, or inside a group
    selection_changed = Signal(str)  # a block id, "src>dst@port" for a connection, or ""
    message = Signal(str)            # a short note for the user (a refused edit and why)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._design = D.empty()
        self._undo, self._redo = [], []
        self._path = ()
        self._selection = ""
        self._inputs, self._outputs = ["x", "t"], ["u"]
        self._widths, self._problems = {}, []
        self._analyse()

    # ── queries (views may read these) ─────────────────────────────
    @property
    def design(self) -> dict:
        """A deep copy of the design: changing it changes nothing here."""
        return copy.deepcopy(self._design)

    @property
    def path(self) -> tuple:
        return self._path

    @property
    def selection(self) -> str:
        return self._selection

    @property
    def inputs(self) -> list:
        return list(self._inputs)

    @property
    def outputs(self) -> list:
        return list(self._outputs)

    @property
    def widths(self) -> dict:
        return dict(self._widths)

    @property
    def problems(self) -> list:
        return list(self._problems)

    def level(self) -> dict:
        """A copy of the level on view."""
        return copy.deepcopy(D.level_of(self._design, self._path))

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def is_empty(self) -> bool:
        return not self._design["blocks"]

    # ── mutators (controllers only) ────────────────────────────────
    def set_design(self, d: dict) -> bool:
        """Make `d` (checked) the design, as one step of Undo. False when nothing changed."""
        if d == self._design:
            return False
        self._undo.append(self._design)
        del self._undo[:-UNDO_DEPTH]
        self._redo.clear()
        self._design = copy.deepcopy(d)
        self._after_change()
        return True

    def reset(self, d: dict) -> None:
        """A design from elsewhere (a problem opened): no undo across it."""
        self._design = copy.deepcopy(d)
        self._undo.clear()
        self._redo.clear()
        self._path = ()
        self._selection = ""
        self._after_change()
        self.path_changed.emit()
        self.selection_changed.emit("")

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._design)
        self._design = self._undo.pop()
        self._after_change()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._design)
        self._design = self._redo.pop()
        self._after_change()
        return True

    def set_path(self, path) -> None:
        path = tuple(path)
        if path != self._path:
            self._path = path
            self._selection = ""
            self.path_changed.emit()
            self.selection_changed.emit("")
            self.design_changed.emit()

    def select(self, key: str) -> None:
        if key != self._selection:
            self._selection = key
            self.selection_changed.emit(key)

    def set_io(self, inputs, outputs) -> None:
        inputs, outputs = list(inputs), list(outputs)
        if (inputs, outputs) != (self._inputs, self._outputs):
            self._inputs, self._outputs = inputs, outputs
            self._analyse()
            self.design_changed.emit()

    def set_message(self, text: str) -> None:
        self.message.emit(text)

    # ── inside ─────────────────────────────────────────────────────
    def _after_change(self) -> None:
        try:                              # the level on view may have gone (an undo removed its group)
            D.level_of(self._design, self._path)
        except Exception:
            self._path = ()
            self.path_changed.emit()
        if self._selection and not self._still_there(self._selection):
            self._selection = ""
            self.selection_changed.emit("")
        self._analyse()
        self.design_changed.emit()

    def _still_there(self, key: str) -> bool:
        level = D.level_of(self._design, self._path)
        if ">" in key:
            src, rest = key.split(">", 1)
            dst, _, port = rest.partition("@")
            return any(e["src"] == src and e["dst"] == dst and e["port"] == (port or "in") for e in level["edges"])
        return any(b["id"] == key for b in level["blocks"])

    def _analyse(self) -> None:
        try:
            self._widths, self._problems = S.infer(self._design, self._inputs, self._outputs)
        except Exception as exc:          # a design that cannot even be walked: say so, draw what there is
            self._widths, self._problems = {}, [((), "", f"the design cannot be read: {exc}")]
