"""Crash recovery: keeping unsaved work in an autosave file, and offering it back after a crash.

The window calls `keep` on a timer while its problem has unsaved changes and `forget` when the problem
is saved, replaced or closed; at start it calls `offer`, which asks through the Prompter and returns the
problem to open, if any (see pinnstudio.core.recovery).
"""
from __future__ import annotations

from PySide6.QtCore import QObject

from pinnstudio.core import problem_file as PF
from pinnstudio.core import recovery as RC
from pinnstudio.ui.controllers.prompter import Prompter


class RecoveryController(QObject):
    def __init__(self, prompter: Prompter | None = None, version: str = "", enabled: bool = True, parent=None):
        super().__init__(parent)
        self.prompter = prompter or Prompter()
        self.version = version
        self.enabled = enabled

    def keep(self, config, original_path, problem_name) -> bool:
        if not self.enabled:
            return False
        try:
            RC.write(config, original_path, problem_name, self.version)
            return True
        except OSError:
            return False

    def forget(self) -> None:
        if self.enabled:
            RC.discard()

    def offer(self):
        """(config, original path, problem name) of a problem to recover, or None. Asks about the newest
        orphan; a discarded one is removed, one left for later stays."""
        if not self.enabled:
            return None
        items = RC.orphans()
        if not items:
            return None
        item = items[0]
        answer = self.prompter.ask_recover(item, len(items))
        if answer == "discard":
            RC.discard(item["path"])
            return None
        if answer != "recover":
            return None
        try:
            config = PF.read(item["path"])[0]
        except PF.ProblemFileError as exc:
            self.prompter.inform("Recover unsaved work", f"The autosave cannot be read: {exc}")
            return None
        RC.discard(item["path"])
        return config, item.get("original_path") or None, item.get("problem_name") or ""
