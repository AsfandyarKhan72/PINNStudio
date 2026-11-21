"""The problem's settings as they were after each change: Edit › Undo and Redo.

A snapshot is a PINNConfig taken when the settings have settled (the window's 1.5 s check), so typing
a number is one step, not one per key. Opening or starting a problem begins a new history.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import QObject, Signal

DEPTH = 100
MUTATORS = ("record", "reset", "step_back", "step_forward", "adopt_key")


class SettingsHistory(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []          # [(key, config)]
        self._pos = -1

    def can_undo(self) -> bool:
        return self._pos > 0

    def can_redo(self) -> bool:
        return 0 <= self._pos < len(self._items) - 1

    def current_key(self):
        return self._items[self._pos][0] if self._pos >= 0 else None

    def record(self, key, config) -> bool:
        if self._pos >= 0 and self._items[self._pos][0] == key:
            return False
        del self._items[self._pos + 1:]
        self._items.append((key, copy.deepcopy(config)))
        del self._items[:-DEPTH]
        self._pos = len(self._items) - 1
        self.changed.emit()
        return True

    def reset(self, key=None, config=None) -> None:
        self._items = [(key, copy.deepcopy(config))] if key is not None else []
        self._pos = len(self._items) - 1
        self.changed.emit()

    def step_back(self):
        if not self.can_undo():
            return None
        self._pos -= 1
        self.changed.emit()
        return copy.deepcopy(self._items[self._pos][1])

    def step_forward(self):
        if not self.can_redo():
            return None
        self._pos += 1
        self.changed.emit()
        return copy.deepcopy(self._items[self._pos][1])

    def adopt_key(self, key) -> None:
        """The settings, applied from a snapshot, compare as `key`: the snapshot is that state."""
        if self._pos >= 0:
            self._items[self._pos] = (key, self._items[self._pos][1])
