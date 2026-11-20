"""The Network tab: the palette, the canvas and the inspector, with the designer's toolbar.

A view: it reads the state (`DesignState`) and redraws on its signals, and hands every request to the
controller (`DesignerController`) it is given. It builds neither of them (rule R8 of tests/test_mvc.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMenu, QPushButton, QSplitter, QToolButton, QVBoxLayout,
                               QWidget)

from pinnstudio.core.netdesign import design as D
from pinnstudio.ui.designer.canvas import DesignCanvas
from pinnstudio.ui.designer.inspector import BlockInspector
from pinnstudio.ui.designer.palette import BlockPalette

if TYPE_CHECKING:
    from pinnstudio.ui.controllers.designer import DesignerController
    from pinnstudio.ui.state.design_state import DesignState


class DesignerPanel(QWidget):
    def __init__(self, state: "DesignState", controller: "DesignerController", presets: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("designerPanel")
        self.state, self.ctl = state, controller
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        bar = QHBoxLayout()
        self.up_btn = QPushButton("Up")
        self.up_btn.setToolTip("Back out of the group")
        self.up_btn.clicked.connect(controller.up)
        self.crumb = QLabel("")
        self.crumb.setObjectName("pageDesc")
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo the last change to the design (Ctrl+Z on the canvas)")
        self.undo_btn.clicked.connect(controller.undo)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setToolTip("Redo (Ctrl+Y on the canvas)")
        self.redo_btn.clicked.connect(controller.redo)
        self.layout_btn = QPushButton("Lay out")
        self.layout_btn.setToolTip("Inputs on the left, heads on the right (Ctrl+L)")
        self.layout_btn.clicked.connect(self._layout)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("The whole design in view (F)")
        self.from_form_btn = QPushButton("From the form")
        self.from_form_btn.setToolTip("Replace the design with the network the Neural network page's settings "
                                      "make (FNN, PFNN, MsFFN, STMsFFN); Undo brings this one back")
        self.from_form_btn.clicked.connect(lambda: controller.from_settings())
        self.more_btn = QToolButton()
        self.more_btn.setText("More ▾")
        self.more_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_btn.setAccessibleName("More designer actions")
        menu = QMenu(self.more_btn)
        for text, fn in (("Save block…", lambda: controller.save_block()),
                         ("Save design…", controller.save_design),
                         ("Load block or design…", self._load),
                         (None, None),
                         ("Export network as Python code…", lambda: controller.export_code()),
                         ("Trust the design's code and files…", controller.check_trust)):
            if text is None:
                menu.addSeparator()
            else:
                menu.addAction(text).triggered.connect(lambda _c=False, f=fn: f())
        self.more_btn.setMenu(menu)
        for w in (self.up_btn, self.crumb):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (self.undo_btn, self.redo_btn, self.layout_btn, self.fit_btn, self.from_form_btn, self.more_btn):
            bar.addWidget(w)
        lay.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.palette = BlockPalette(presets)
        self.canvas = DesignCanvas()
        self.inspector = BlockInspector(state, controller)
        split.addWidget(self.palette)
        split.addWidget(self.canvas)
        split.addWidget(self.inspector)
        split.setStretchFactor(1, 1)
        split.setSizes([170, 640, 290])
        lay.addWidget(split, 1)
        self.status = QLabel("")
        self.status.setObjectName("pageDesc")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        # the canvas asks, the controller decides
        c = self.canvas
        c.legality = controller.legal
        c.drag_started.connect(controller.drag_started)
        c.add_requested.connect(self._add_at)
        c.connect_requested.connect(controller.connect)
        c.move_requested.connect(controller.move)
        c.delete_requested.connect(controller.delete)
        c.duplicate_requested.connect(controller.duplicate)
        c.layout_requested.connect(self._layout)
        c.enter_requested.connect(controller.enter)
        c.select_requested.connect(controller.select)
        c.undo_requested.connect(controller.undo)
        c.redo_requested.connect(controller.redo)
        self.fit_btn.clicked.connect(c.fit)
        self.palette.add_requested.connect(self._add_in_middle)
        self.palette.preset_requested.connect(self._preset)
        self.palette.library_requested.connect(self._library_in_middle)

        state.design_changed.connect(self.redraw)
        state.selection_changed.connect(lambda _k: self.redraw())
        state.path_changed.connect(self._path_changed)
        state.message.connect(self.status.setText)
        self.refresh_library()
        self.redraw()

    # ── requests that need the canvas's position ──────────────────
    def _add_at(self, kind, preset, x, y):
        if kind == "library":
            self.ctl.load_file(preset, x, y)
        else:
            self.ctl.add(kind, preset, x, y)
        self.canvas.setFocus()

    def _add_in_middle(self, kind, preset):
        p = self.canvas.center_scene_pos()
        self.ctl.add(kind, preset, round(p.x() / 20) * 20 - 80, round(p.y() / 20) * 20 - 30)

    def _library_in_middle(self, path):
        p = self.canvas.center_scene_pos()
        self.ctl.load_file(path, p.x() - 80, p.y() - 30)

    def _preset(self, name):
        if self.ctl.apply_preset(name):
            self.canvas.refit()

    def _layout(self):
        self.ctl.auto_layout()
        self.canvas.fit()

    def _load(self):
        p = self.canvas.center_scene_pos()
        self.ctl.load_file("", p.x() - 80, p.y() - 30)
        self.refresh_library()

    def refresh_library(self):
        self.palette.set_library(self.ctl.library_blocks())

    # ── drawing ────────────────────────────────────────────────────
    def _path_changed(self):
        self.redraw()
        self.canvas.refit()

    def redraw(self):
        level = self.state.level()
        path = self.state.path
        errors = {}
        for p, bid, msg in self.state.problems:
            if tuple(p) == tuple(path) and bid and bid not in errors:
                errors[bid] = msg
        self.canvas.show_level(level, self.state.widths, errors, self.state.selection)
        self.up_btn.setVisible(bool(path))
        names = []
        for gid in path:
            _p, g = D.find(self.state.design, gid)
            names.append(D.label(g) if g else gid)
        self.crumb.setText(("Design › " + " › ".join(names)) if path else "")
        self.undo_btn.setEnabled(self.state.can_undo())
        self.redo_btn.setEnabled(self.state.can_redo())
        design_level = [m for p, bid, m in self.state.problems if not bid]
        if self.state.is_empty():
            self.status.setText("Empty: drag blocks from the palette, or double-click a ready-made design.")
        elif design_level:
            self.status.setText("⚠ " + design_level[0])
        elif self.state.problems:
            n = len(self.state.problems)
            self.status.setText(f"⚠ {n} block{'s' if n != 1 else ''} with a problem (outlined in red)")
        else:
            self.status.setText(f"Ready: {', '.join(self.state.inputs)} → {', '.join(self.state.outputs)}")
