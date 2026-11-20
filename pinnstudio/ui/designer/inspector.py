"""The designer's inspector: the selected block's settings, or, with nothing selected, the design's
problems and its own code.

The forms are built from the parameter specs (`netdesign.params`, `netdesign.layers`). What is typed is a
draft: after a pause in typing it goes to the controller, which checks it and changes the design, or
refuses and says why (and nothing changes). The inspector never changes the state itself.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QScrollArea, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign.layers import LAYERS
from pinnstudio.core.netdesign.params import Bool, Choice, Float, Int, JsonDict, Text

DEBOUNCE_MS = 350


class BlockInspector(QWidget):
    def __init__(self, state, controller, parent=None):
        super().__init__(parent)
        self.setObjectName("designInspector")
        self.state, self.ctl = state, controller
        self._bid = ""
        self._draft = None          # the block's settings being edited
        self._layer_row = -1
        self._own_edit = False      # the design change under way is this form's own field edit
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._apply)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(8, 8, 8, 8)
        # the form scrolls rather than claim the canvas's width or the window's height (large text)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)   # never clip a field
        self.lay.addWidget(self.scroll)
        self.body = None
        state.selection_changed.connect(lambda _k: self.refresh())
        state.design_changed.connect(self._design_changed)
        self.refresh()

    # ── building ───────────────────────────────────────────────────
    def _clear(self):
        if self.body is not None:
            self.scroll.takeWidget()
            self.body.hide()                 # now: deleteLater only runs when the event loop gets to it
            self.body.deleteLater()
        self.body = QWidget()
        self.scroll.setWidget(self.body)
        return self.body

    def _design_changed(self):
        if self._timer.isActive():          # an edit of ours is waiting: keep its draft
            return
        if self._own_edit:                  # our own field edit arrived: keep the form and its cursor
            level = self.state.level()
            now = next((b for b in level["blocks"] if b["id"] == self._bid), None)
            if now is not None:
                self._draft = copy.deepcopy(now)
                return
        self.refresh()

    def refresh(self):
        self._timer.stop()
        key = self.state.selection
        level = self.state.level()
        block = next((b for b in level["blocks"] if b["id"] == key), None) if key and ">" not in key else None
        if block is None:
            self._bid, self._draft = "", None
            self._design_form()
            return
        if block["id"] != self._bid:
            self._layer_row = 0 if block.get("layers") else -1
        self._bid = block["id"]
        self._draft = copy.deepcopy(block)
        self._block_form(block, level)

    def _design_form(self):
        body = self._clear()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Network design")
        title.setObjectName("pageTitle")
        title.setWordWrap(True)
        v.addWidget(title)
        problems = self.state.problems
        if self.state.is_empty():
            v.addWidget(self._hint("Drag blocks from the palette, or pick a ready-made design. Select a block "
                                   "to see its settings here."))
        elif problems:
            lst = QListWidget()
            lst.setObjectName("designProblems")
            for path, bid, msg in problems:
                where = (" › ".join(path) + " › " if path else "") + (bid or "design")
                lst.addItem(f"{where}: {msg}")
            v.addWidget(self._text(f"{len(problems)} problem{'s' if len(problems) != 1 else ''} to fix before Solve:"))
            v.addWidget(lst, 1)
        else:
            v.addWidget(self._hint(f"Ready: inputs {', '.join(self.state.inputs)} → outputs "
                                   f"{', '.join(self.state.outputs)}."))
        missing = self.ctl.missing_outputs()
        if missing:
            fix = QPushButton(f"Add a head for {', '.join(missing)}")
            fix.setObjectName("addHeadsButton")
            fix.setToolTip("A head for each output no head gives, fed like the heads already there")
            fix.clicked.connect(self.ctl.add_missing_heads)
            v.addWidget(fix)
        v.addWidget(self._text("The design's code (classes for Custom layers):"))
        code = QPlainTextEdit(self.state.design.get("code", ""))
        code.setObjectName("designCode")
        code.setPlaceholderText("class MyLayer(torch.nn.Module):\n    def forward(self, x):\n        return x")
        v.addWidget(code, 2)
        row = QHBoxLayout()
        apply_btn = QPushButton("Apply code")
        apply_btn.clicked.connect(lambda: self.ctl.set_code(code.toPlainText()))
        row.addStretch()
        row.addWidget(apply_btn)
        v.addLayout(row)
        v.addWidget(self._hint("Code runs only after you trust it (Solve asks). torch, dde, np and json are "
                               "available."))

    @staticmethod
    def _text(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        return lbl

    @staticmethod
    def _hint(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setObjectName("pageDesc")
        return lbl

    def _block_form(self, b, level):
        body = self._clear()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        head = QLabel(f"{b['kind'].replace('_', ' ').capitalize()} {b['id']}")
        head.setObjectName("pageTitle")
        head.setWordWrap(True)
        v.addWidget(head)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)   # labels above the fields when narrow
        v.addLayout(form)
        title = QLineEdit(b.get("title", ""))
        title.setPlaceholderText(b["id"])
        title.textEdited.connect(lambda t: self._set("title", t))
        form.addRow("Title:", title)
        k = b["kind"]
        if k == "input":
            e = QLineEdit(", ".join(b["inputs"]))
            e.setToolTip(f"@all, or some of: {', '.join(self.state.inputs)}")
            e.textEdited.connect(lambda t: self._set("inputs", [x.strip() for x in t.split(",") if x.strip()]))
            form.addRow("Inputs:", e)
            v.addWidget(self._hint(f"The network's inputs: {', '.join(self.state.inputs)}."))
        if k == "head":
            e = QLineEdit(", ".join(b["outputs"]))
            e.setToolTip(f"@all, or some of: {', '.join(self.state.outputs)}")
            e.textEdited.connect(lambda t: self._set("outputs", [x.strip() for x in t.split(",") if x.strip()]))
            form.addRow("Outputs:", e)
            cb = QCheckBox("Values as they come (no layer)")
            cb.setChecked(bool(b["direct"]))
            cb.toggled.connect(lambda on: self._set("direct", on))
            form.addRow("", cb)
        if "merge" in b and k != "merge":
            form.addRow("Several inputs:", self._choice(D.MERGES, b["merge"], lambda x: self._set("merge", x)))
        if k == "merge":
            form.addRow("Merge:", self._choice(D.MERGES, b["merge"], lambda x: self._set("merge", x)))
        if k in ("stack", "group"):
            cb = QCheckBox("Trainable")
            cb.setChecked(bool(b["trainable"]))
            cb.toggled.connect(lambda on: self._set("trainable", on))
            form.addRow("", cb)
        if k == "shared":
            refs = [x["id"] for x in level["blocks"] if x["kind"] in ("stack", "group", "saved")]
            form.addRow("Calls:", self._choice(refs, b["ref"], lambda x: self._set("ref", x)))
        if k == "port_in":
            e = QLineEdit(b["port"])
            e.textEdited.connect(lambda t: self._set("port", t))
            form.addRow("Port name:", e)
        if k == "saved":
            path = QLineEdit(b["path"])
            path.setReadOnly(True)
            form.addRow("File:", path)
            for key, lbl in (("width_in", "Takes:"), ("width_out", "Gives:")):
                sp = QSpinBox()
                sp.setRange(1, 4096)
                sp.setValue(int(b[key]))
                sp.valueChanged.connect(lambda val, kk=key: self._set(kk, int(val)))
                form.addRow(lbl, sp)
            cb = QCheckBox("Fine-tune (train its weights too)")
            cb.setChecked(bool(b["fine_tune"]))
            cb.toggled.connect(lambda on: self._set("fine_tune", on))
            form.addRow("", cb)
        if k == "group":
            btn = QPushButton("Open the group")
            btn.clicked.connect(lambda: self.ctl.enter(b["id"]))
            v.addWidget(btn)
        if k == "stack":
            self._layers_editor(v)
        if D.has_out(k):
            self._connect_row(v, b["id"])
        width = self.state.widths.get(b["id"])
        errs = [m for p, bid, m in self.state.problems if bid == b["id"]]
        v.addWidget(self._hint("⚠ " + errs[0] if errs else (f"Gives {width} value(s)." if width else "")))
        v.addStretch(1)

    def _connect_row(self, v, bid):
        """Connect this block's output from the keyboard: the in-ports it may go to now."""
        row = QHBoxLayout()
        targets = QComboBox()
        targets.setObjectName("connectTargets")
        targets.setAccessibleName("Connect the output to")
        targets.addItem("Connect output to…", None)
        for dst, port, text in self.ctl.legal_targets(bid):
            targets.addItem(text, (dst, port))
        go = QPushButton("Connect")
        go.setEnabled(targets.count() > 1)
        go.clicked.connect(lambda: targets.currentData() and self.ctl.connect(bid, *targets.currentData()))
        row.addWidget(targets, 1)
        row.addWidget(go)
        v.addLayout(row)

    def _choice(self, options, current, on_change):
        c = QComboBox()
        c.addItems(list(options))
        if current in options:
            c.setCurrentText(current)
        c.currentTextChanged.connect(on_change)
        return c

    def _layers_editor(self, v):
        v.addWidget(self._text("Layers (top runs first):"))
        self.layer_list = QListWidget()
        self.layer_list.setObjectName("layerList")
        for sp in self._draft["layers"]:
            self.layer_list.addItem(LAYERS[sp["type"]].title if sp["type"] in LAYERS else sp["type"])
        v.addWidget(self.layer_list)
        row = QHBoxLayout()
        add = QComboBox()
        add.addItem("Add layer…")
        for name, lt in LAYERS.items():
            add.addItem(lt.title, name)
        add.activated.connect(lambda i: i > 0 and self._layer_op("add", add.itemData(i)))
        row.addWidget(add, 1)
        for text, op in (("▲", "up"), ("▼", "down"), ("✕", "remove")):
            btn = QPushButton(text)
            btn.setFixedWidth(30)
            btn.setToolTip({"up": "Move up", "down": "Move down", "remove": "Remove the layer"}[op])
            btn.setAccessibleName(btn.toolTip())
            btn.clicked.connect(lambda _c=False, o=op: self._layer_op(o))
            row.addWidget(btn)
        v.addLayout(row)
        self.layer_form = QFormLayout()
        v.addLayout(self.layer_form)
        if 0 <= self._layer_row < len(self._draft["layers"]):
            self.layer_list.setCurrentRow(self._layer_row)
            self._layer_params()
        self.layer_list.currentRowChanged.connect(self._layer_selected)

    def _layer_selected(self, row):
        self._layer_row = row
        while self.layer_form.rowCount():
            self.layer_form.removeRow(0)
        if 0 <= row < len(self._draft["layers"]):
            self._layer_params()

    def _layer_params(self):
        spec = self._draft["layers"][self._layer_row]
        lt = LAYERS.get(spec["type"])
        if lt is None:
            return
        for p in lt.params:
            value = spec["params"].get(p.name, p.default)
            w = self._param_widget(p, value, lambda val, name=p.name: self._set_layer_param(name, val))
            w.setToolTip(p.help)
            self.layer_form.addRow(p.title() + ":", w)
        if spec["type"] == "custom":
            names = D.class_names(self.state.design.get("code", ""))
            self.layer_form.addRow("", self._hint("Classes in the design's code: " + (", ".join(names) or "none yet")))

    @staticmethod
    def _param_widget(p, value, on_change):
        if isinstance(p, Bool):
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(on_change)
            return w
        if isinstance(p, Choice):
            w = QComboBox()
            w.addItems([str(o) for o in p.options])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(on_change)
            return w
        if isinstance(p, Int):
            w = QSpinBox()
            w.setRange(max(p.min, -10 ** 9), min(p.max, 10 ** 9))
            w.setValue(int(value))
            w.valueChanged.connect(on_change)
            return w
        w = QLineEdit("" if value is None else (f"{value:g}" if isinstance(p, Float) and isinstance(value, float)
                                                else str(value)))
        if isinstance(p, (Text, JsonDict)) or isinstance(p, Float):
            w.textEdited.connect(on_change)
        return w

    # ── drafts to the controller ───────────────────────────────────
    def _set(self, key, value):
        if self._draft is None:
            return
        self._draft[key] = value
        self._timer.start()

    def _set_layer_param(self, name, value):
        if self._draft is None or not (0 <= self._layer_row < len(self._draft["layers"])):
            return
        self._draft["layers"][self._layer_row]["params"][name] = value
        self._timer.start()

    def _layer_op(self, op, layer=None):
        if self._draft is None:
            return
        self._flush()
        layers = copy.deepcopy(self._draft["layers"])
        r = self._layer_row
        if op == "add":
            layers.insert(r + 1 if r >= 0 else len(layers), {"type": layer, "params": {}})
            self._layer_row = r + 1 if r >= 0 else len(layers) - 1
        elif op == "remove" and 0 <= r < len(layers):
            layers.pop(r)
            self._layer_row = min(r, len(layers) - 1)
        elif op in ("up", "down") and 0 <= r < len(layers):
            j = r - 1 if op == "up" else r + 1
            if not 0 <= j < len(layers):
                return
            layers[r], layers[j] = layers[j], layers[r]
            self._layer_row = j
        self.ctl.update_block(self._bid, {"layers": layers})

    def _flush(self):
        if self._timer.isActive():
            self._timer.stop()
            self._apply()

    def _apply(self):
        if self._draft is None:
            return
        level = self.state.level()
        current = next((b for b in level["blocks"] if b["id"] == self._bid), None)
        if current is None:
            return
        values = {k: v for k, v in self._draft.items()
                  if k in ("title", "merge", "inputs", "outputs", "direct", "trainable", "ref", "port",
                           "width_in", "width_out", "fine_tune", "layers") and current.get(k) != v}
        if values:
            self._own_edit = True
            try:
                self.ctl.update_block(self._bid, values)
            finally:
                self._own_edit = False
