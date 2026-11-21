"""Undo and redo of the problem's settings (Edit › Undo, Edit › Redo).

The window records the settings when they have settled and asks for the settings to go back or forward
to; it applies what it gets. Applying a snapshot is not itself recorded.
"""
from __future__ import annotations

from PySide6.QtCore import QObject

from pinnstudio.ui.state.settings_history import SettingsHistory


class SettingsController(QObject):
    def __init__(self, history: SettingsHistory, parent=None):
        super().__init__(parent)
        self.history = history
        self._applying = False

    def settled(self, key, config) -> None:
        """The settings have settled at `config` (with its comparison key)."""
        if not self._applying:
            self.history.record(key, config)

    def start(self, key, config) -> None:
        """A new or opened problem: its history begins here."""
        self.history.reset(key, config)

    def undo(self, apply) -> bool:
        return self._go(self.history.step_back(), apply)

    def redo(self, apply) -> bool:
        return self._go(self.history.step_forward(), apply)

    def _go(self, config, apply) -> bool:
        if config is None:
            return False
        self._applying = True
        try:
            key = apply(config)          # the window applies it and says how the result compares
        finally:
            self._applying = False
        if key is not None:
            self.history.adopt_key(key)
        return True

    def applying(self) -> bool:
        return self._applying
