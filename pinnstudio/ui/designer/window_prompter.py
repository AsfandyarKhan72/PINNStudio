"""The window's answers to the controllers' questions, as dialogs.

Implements the `Prompter` port (pinnstudio/ui/controllers/prompter.py) method for method; it does not
import it, since among the views only the main window imports controllers at run time (rule R8).
"""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QLabel, QMessageBox,
                               QPlainTextEdit, QVBoxLayout)


class WindowPrompter:
    def __init__(self, parent):
        self.parent = parent

    def ask_open_path(self, title, file_filter):
        path, _f = QFileDialog.getOpenFileName(self.parent, title, "", file_filter)
        return path or ""

    def ask_save_path(self, title, default_name, file_filter):
        path, _f = QFileDialog.getSaveFileName(self.parent, title, default_name, file_filter)
        return path or ""

    def ask_text(self, title, label, text=""):
        value, ok = QInputDialog.getText(self.parent, title, label, text=text)
        return value if ok else None

    def inform(self, title, text):
        QMessageBox.information(self.parent, title, text)

    def confirm_trust(self, items):
        dlg = QDialog(self.parent)
        dlg.setObjectName("trustDialog")
        dlg.setWindowTitle("Trust the network design's code and files?")
        lay = QVBoxLayout(dlg)
        intro = QLabel("This network design runs code of its own or loads a file when the network is built. "
                       "Code runs with your rights: trust it only if you wrote it or know where it comes from. "
                       "The decision is kept for exactly this code and these files; a change asks again.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        for it in items:
            head = QLabel(f"<b>{it['name']}</b> ({it['kind']}, {it['size']} bytes, SHA-256 {it['sha256'][:16]}…)"
                          + (f"<br>{it['path']}" if it.get("path") else ""))
            head.setWordWrap(True)
            lay.addWidget(head)
            if it.get("text"):
                view = QPlainTextEdit(it["text"])
                view.setReadOnly(True)
                view.setObjectName("scriptView")
                lay.addWidget(view, 1)
        buttons = QDialogButtonBox()
        trust = buttons.addButton("Trust and use", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Keep inactive", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        trust.setDefault(False)
        dlg.resize(640, 480)
        ok = dlg.exec() == QDialog.DialogCode.Accepted
        dlg.deleteLater()
        return ok
