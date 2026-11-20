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

    def ask_recover(self, item, count):
        import time as _time
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(float(item.get("time") or 0)))
        name = item.get("problem_name") or "an untitled problem"
        box = QMessageBox(self.parent)
        box.setObjectName("recoverDialog")
        box.setWindowTitle("Recover unsaved work")
        box.setIcon(QMessageBox.Icon.Question)
        more = f" ({count - 1} more autosave{'s' if count > 2 else ''} will be offered next time)" if count > 1 else ""
        box.setText(f"PINNStudio closed without saving {name} (last kept {when}). Recover it?{more}")
        if item.get("original_path"):
            box.setInformativeText(f"It was saved before as {item['original_path']}; recovering does not change "
                                   "that file until you save.")
        rec = box.addButton("Recover", QMessageBox.ButtonRole.AcceptRole)
        dis = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(rec)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()
        return "recover" if clicked is rec else "discard" if clicked is dis else "later"

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
