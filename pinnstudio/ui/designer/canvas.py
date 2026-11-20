"""The designer's canvas: blocks, their ports and the connections between them.

Plain Qt graphics (QGraphicsScene and QGraphicsView). The canvas keeps no design of its own: it draws the
level it is given (`show_level`) and asks for every change through its signals, which the panel hands to
the controller. Interactions: drop a block from the palette; drag from a port to another to connect (the
port under the cursor turns green, or red with the reason, asked once per port per drag); drag a block
past a small dead zone to move it; click a connection to select it; drag on empty space to pan; the wheel
zooms. Keys: Delete, Ctrl+D duplicate, Ctrl+L lay out, F fit, Ctrl+Z / Ctrl+Y undo and redo, and
double-click a group to open it.
"""
from __future__ import annotations

import json

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QKeySequence, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsObject, QGraphicsPathItem,
                               QGraphicsScene, QGraphicsView, QToolTip)

from pinnstudio.core.netdesign import design as D
from pinnstudio.core.netdesign.layers import LAYERS
from pinnstudio.ui import theme as _theme

MIME = "application/x-pinnstudio-block"
WIDTH, HEADER, LINE = 180.0, 28.0, 16.0
PORT_R = 6.0
GRID = 20.0
DEAD_ZONE = 4.0
MIN_FIT = 0.65      # Fit never shrinks blocks below this: their text must stay readable
KIND_COLOR = {"input": 0, "stack": 2, "merge": 4, "head": 1, "shared": 2, "group": 3, "saved": 5,
              "port_in": 0, "port_out": 1}
KIND_TEXT = {"input": "Input", "stack": "Stack", "merge": "Merge", "head": "Head", "shared": "Shared",
             "group": "Group", "saved": "Saved module", "port_in": "Port in", "port_out": "Port out"}


def kind_color(kind: str) -> QColor:
    return QColor(_theme.BC_COLORS[KIND_COLOR.get(kind, 6) % len(_theme.BC_COLORS)])


def summary(block: dict) -> list:
    """The lines a block card shows under its title."""
    k = block["kind"]
    if k == "input":
        return ["all inputs" if block["inputs"] == [D.ALL] else ", ".join(block["inputs"])]
    if k == "head":
        names = "all outputs" if block["outputs"] == [D.ALL] else ", ".join(block["outputs"])
        return [names + (" (as they come)" if block.get("direct") else "")]
    if k == "stack":
        lines = []
        for sp in block["layers"][:5]:
            lt = LAYERS.get(sp["type"])
            p = sp["params"]
            if sp["type"] == "dense":
                lines.append(f"Dense {p['units']}" + ("" if p["activation"] == "inherit" else f" {p['activation']}"))
            elif sp["type"] == "fourier":
                lines.append(f"Fourier {p['features']}, σ={p['sigma']:g}")
            elif sp["type"] == "custom":
                lines.append(f"{p['class_name']} (code)")
            else:
                lines.append(lt.title if lt else sp["type"])
        if len(block["layers"]) > 5:
            lines.append(f"… {len(block['layers']) - 5} more")
        return lines or ["no layers (passes values on)"]
    if k == "merge":
        return [block["merge"]]
    if k == "shared":
        return [f"calls {block['ref']}"]
    if k == "group":
        n = len((block.get("inner") or {}).get("blocks", []))
        return [f"{n} blocks inside", "double-click to open"]
    if k == "saved":
        return [block.get("path", "").replace("\\", "/").split("/")[-1] or "no file",
                "fine-tuned" if block.get("fine_tune") else "frozen"]
    if k == "port_in":
        return [f"port {block['port']}"]
    return []


class PortItem(QGraphicsEllipseItem):
    def __init__(self, block_item, side, port="in"):
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, block_item)
        self.block_item, self.side, self.port = block_item, side, port
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        self.setToolTip("Drag from here to connect" if side == "out"
                        else ("Connections arrive here" if port == "in" else f"Port {port}: connections arrive here"))
        self.restyle()

    def restyle(self, verdict=""):
        t = _theme.current()
        fill = t["success"] if verdict == "ok" else t["danger"] if verdict == "bad" else t["card"]
        self.setBrush(QBrush(QColor(fill)))
        self.setPen(QPen(kind_color(self.block_item.kind), 2.0))


class BlockItem(QGraphicsObject):
    def __init__(self, block: dict, width: int | None, error: str, selected: bool):
        super().__init__()
        self.block_id, self.kind = block["id"], block["kind"]
        self.title = D.label(block)
        self.lines = summary(block)
        self.width_text = f"→ {width} value{'s' if width != 1 else ''}" if width else ""
        self.error = error
        self.selected_ = selected
        self.setZValue(1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setAcceptHoverEvents(True)
        self.in_ports = {p: PortItem(self, "in", p) for p in D.ports_of(block)}
        self.out_port = PortItem(self, "out") if D.has_out(self.kind) else None
        self.edges = []
        self.setPos(block["x"], block["y"])
        self._place_ports()
        tip = error
        if self.kind == "group":
            tip = (tip + "\n" if tip else "") + "Double-click to open the group"
        self.setToolTip(tip)

    def height(self) -> float:
        rows = max(1, len(self.lines)) + (1 if (self.width_text or self.error) else 0)
        rows = max(rows, len(self.in_ports))
        return HEADER + LINE * rows + 10

    def _place_ports(self):
        h = self.height()
        n = len(self.in_ports)
        for k, port in enumerate(self.in_ports.values()):
            port.setPos(0, h / 2 if n == 1 else HEADER + (h - HEADER) * (k + 1) / (n + 1))
        if self.out_port is not None:
            self.out_port.setPos(WIDTH, h / 2)

    def boundingRect(self):
        return QRectF(-PORT_R - 2, -2, WIDTH + 2 * PORT_R + 4, self.height() + 4)

    def shape(self):
        p = QPainterPath()
        p.addRoundedRect(QRectF(0, 0, WIDTH, self.height()), 8, 8)
        return p

    def paint(self, painter: QPainter, option, widget=None):
        t = _theme.current()
        strong = kind_color(self.kind)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, WIDTH, self.height())
        border = QColor(t["danger"]) if self.error else QColor(t["accent"]) if self.selected_ else QColor(t["border_strong"])
        pen = QPen(border, 2.0 if (self.error or self.selected_) else 1.0)
        if self.kind == "shared":
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(t["card"])))
        painter.drawRoundedRect(rect, 8, 8)
        band = QPainterPath()
        band.addRoundedRect(QRectF(0, 0, WIDTH, HEADER), 8, 8)
        band.addRect(QRectF(0, HEADER / 2, WIDTH, HEADER / 2))
        soft = QColor(strong)
        soft.setAlpha(60 if t.get("dark") else 38)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(soft))
        painter.drawPath(band.simplified())
        painter.setBrush(QBrush(strong))
        painter.drawRect(QRectF(0, 7, 3, HEADER - 10))
        # the scene's font, a fixed size: the painter's is the viewport's, which the UI scale enlarges,
        # while the card's rows are fixed. The canvas zooms by the UI scale instead (text_scale)
        base = self.scene().font() if self.scene() is not None else QFont()
        font = QFont(base)
        font.setFamily(base.family())           # set explicitly: unset attributes would be taken from the
        if base.pixelSize() > 0:                # viewport's font when the painter resolves this one
            font.setPixelSize(base.pixelSize())
        else:
            font.setPointSizeF(base.pointSizeF())
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(t["text"])))
        fm = painter.fontMetrics()
        kind = KIND_TEXT.get(self.kind, self.kind)
        painter.drawText(QRectF(10, 0, WIDTH - 18, HEADER), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         fm.elidedText(self.title, Qt.TextElideMode.ElideRight, int(WIDTH - 24 - fm.horizontalAdvance(kind))))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor(t["text_muted"])))
        painter.drawText(QRectF(10, 0, WIDTH - 18, HEADER), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, kind)
        y = HEADER + 2
        fm = painter.fontMetrics()
        painter.setPen(QPen(QColor(t["text"])))
        for ln in self.lines:
            painter.drawText(QRectF(10, y, WIDTH - 18, LINE), Qt.AlignmentFlag.AlignVCenter,
                             fm.elidedText(ln, Qt.TextElideMode.ElideRight, int(WIDTH - 20)))
            y += LINE
        if self.error:
            painter.setPen(QPen(QColor(t["danger"])))
            painter.drawText(QRectF(10, y, WIDTH - 18, LINE), Qt.AlignmentFlag.AlignVCenter,
                             fm.elidedText("⚠ " + self.error, Qt.TextElideMode.ElideRight, int(WIDTH - 20)))
        elif self.width_text:
            painter.setPen(QPen(QColor(t["accent_text"])))
            painter.drawText(QRectF(10, y, WIDTH - 18, LINE), Qt.AlignmentFlag.AlignVCenter, self.width_text)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged:
            for e in self.edges:
                e.refresh()
        return super().itemChange(change, value)


def _curve(a: QPointF, b: QPointF) -> QPainterPath:
    p = QPainterPath(a)
    dx = max(40.0, abs(b.x() - a.x()) / 2)
    p.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return p


class EdgeItem(QGraphicsPathItem):
    def __init__(self, src: BlockItem, dst: BlockItem, port: str, selected: bool):
        super().__init__()
        self.src, self.dst, self.port = src, dst, port
        self.key = f"{src.block_id}>{dst.block_id}@{port}"
        self.selected_ = selected
        self.setZValue(0)
        src.edges.append(self)
        dst.edges.append(self)
        self.refresh()

    def refresh(self):
        t = _theme.current()
        a = self.src.out_port.scenePos()
        b = self.dst.in_ports[self.port].scenePos()
        self.setPath(_curve(a, b))
        self.setPen(QPen(QColor(t["accent"] if self.selected_ else t["text_muted"]), 2.4 if self.selected_ else 1.6))

    def shape(self):
        s = QPainterPathStroker()
        s.setWidth(10)
        return s.createStroke(self.path())


class DesignCanvas(QGraphicsView):
    add_requested = Signal(str, str, float, float)
    connect_requested = Signal(str, str, str)
    move_requested = Signal(str, float, float)
    delete_requested = Signal(str)
    duplicate_requested = Signal(str)
    layout_requested = Signal()
    enter_requested = Signal(str)
    select_requested = Signal(str)
    drag_started = Signal()
    undo_requested = Signal()
    redo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("designCanvas")
        self.setAccessibleName("Network design canvas")
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.legality = lambda src, dst, port: (True, "")
        self.blocks: dict[str, BlockItem] = {}
        self.edges: dict[str, EdgeItem] = {}
        self.selection = ""
        self._press = None          # ("port", PortItem) | ("block", BlockItem, start pos) | ("pan", pos)
        self._band = None
        self._target = None
        self.verdicts = {}
        self._fitted = False

    # ── drawing what it is given ───────────────────────────────────
    def show_level(self, level: dict, widths: dict, errors: dict, selection: str) -> None:
        """Draw `level` (blocks and edges) with each block's width or first problem."""
        # A drag under way (a press on a block selects it, and the selection redraws the canvas) holds items
        # that clear() deletes: remember them by name, and take the new items after drawing.
        held = self._held_by_name()
        self.scene_.clear()
        self.blocks, self.edges = {}, {}
        self._band = None
        self._target = None
        self.selection = selection
        t = _theme.current()
        self.setBackgroundBrush(QBrush(QColor(t["bg"])))
        for b in level["blocks"]:
            item = BlockItem(b, widths.get(b["id"]), errors.get(b["id"], ""), selection == b["id"])
            self.scene_.addItem(item)
            self.blocks[b["id"]] = item
        for e in level["edges"]:
            s, d = self.blocks.get(e["src"]), self.blocks.get(e["dst"])
            if s is None or d is None or s.out_port is None or e["port"] not in d.in_ports:
                continue
            key = f"{e['src']}>{e['dst']}@{e['port']}"
            item = EdgeItem(s, d, e["port"], key == selection)
            self.scene_.addItem(item)
            self.edges[key] = item
        names = {b["id"]: D.label(b) for b in level["blocks"]}
        self.setAccessibleDescription(
            f"{len(level['blocks'])} blocks: " + "; ".join(f"{D.label(b)} ({b['kind']})" for b in level["blocks"])
            + ". Connections: " + "; ".join(f"{names.get(e['src'], e['src'])} to {names.get(e['dst'], e['dst'])}"
                                            for e in level["edges"]))
        self._hold_again(held)
        rect = self.scene_.itemsBoundingRect().adjusted(-400, -300, 400, 300)
        self.scene_.setSceneRect(rect)
        if not self._fitted and self.blocks:
            self.fit()

    def text_scale(self) -> float:
        """How much larger the text around the canvas is than the cards' own (the scene's) font: the UI
        scale reaches widgets through the theme's stylesheet, not the scene. The canvas zooms by it, so the
        cards grow with the rest of the window."""
        app_px = QFontMetricsF(self.scene_.font()).height()
        own_px = QFontMetricsF(self.font()).height()
        return max(1.0, own_px / app_px) if app_px > 0 else 1.0

    def fit(self) -> None:
        if not self.blocks:
            return
        r = QRectF()
        for b in self.blocks.values():
            r = r.united(b.sceneBoundingRect())
        self.fitInView(r.adjusted(-60, -60, 60, 60), Qt.AspectRatioMode.KeepAspectRatio)
        s, ts = self.transform().m11(), self.text_scale()
        if s > ts:
            self.resetTransform()
            self.scale(ts, ts)
            self.centerOn(r.center())
        elif s < MIN_FIT * ts:                 # text stays readable: scroll to see the rest
            z = MIN_FIT * ts
            self.resetTransform()
            self.scale(z, z)
            first = min(self.blocks.values(), key=lambda b: b.pos().x())
            self.centerOn(QPointF(first.sceneBoundingRect().left() + self.viewport().width() / (2 * z) - 40,
                                  r.center().y()))
        self._fitted = True

    def refit(self) -> None:
        """Fit the design in view now (a new design, or another level of it)."""
        self._fitted = False
        self.fit()

    def center_scene_pos(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def _held_by_name(self):
        press = self._press
        if not press:
            return None
        try:
            if press[0] == "block":
                return ("block", press[1].block_id) + tuple(press[2:])
            if press[0] == "port":
                p = press[1]
                return ("port", p.block_item.block_id, p.side, p.port)
        except RuntimeError:             # already gone
            return None
        return press                     # ("pan", ...) holds no item

    def _hold_again(self, held):
        """The drag's items, drawn anew; a drag whose block has gone ends."""
        if held is None:
            self._press = None
            return
        if held[0] == "block":
            item = self.blocks.get(held[1])
            self._press = ("block", item) + tuple(held[2:]) if item is not None else None
        elif held[0] == "port":
            item = self.blocks.get(held[1])
            port = None
            if item is not None:
                port = item.out_port if held[2] == "out" else item.in_ports.get(held[3])
            self._press = ("port", port) if port is not None else None
        else:
            self._press = held
        if self._press is None:
            self._press_moved = False

    # ── mouse ──────────────────────────────────────────────────────
    def _port_at(self, scene_pos, side=None):
        best, dist = None, 1.6 * PORT_R
        for b in self.blocks.values():
            ports = list(b.in_ports.values()) + ([b.out_port] if b.out_port else [])
            for p in ports:
                if side and p.side != side:
                    continue
                d = (p.scenePos() - scene_pos).manhattanLength()
                if d <= dist:
                    best, dist = p, d
        return best

    def mousePressEvent(self, ev):
        pos = self.mapToScene(ev.position().toPoint())
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(ev)
        port = self._port_at(pos)
        if port is not None:
            self._press = ("port", port)
            self.verdicts = {}
            self.drag_started.emit()
            return
        item = self.itemAt(ev.position().toPoint())
        while item is not None and not isinstance(item, (BlockItem, EdgeItem)):
            item = item.parentItem()
        if isinstance(item, BlockItem):
            self._press = ("block", item, pos, item.pos())
            self.select_requested.emit(item.block_id)
            return
        if isinstance(item, EdgeItem):
            self._press = None
            self.select_requested.emit(item.key)
            return
        self._press = ("pan", ev.position())
        self.select_requested.emit("")

    def mouseMoveEvent(self, ev):
        if not self._press:
            return super().mouseMoveEvent(ev)
        pos = self.mapToScene(ev.position().toPoint())
        kind = self._press[0]
        if kind == "port":
            self._drag_band(self._press[1], pos)
        elif kind == "block":
            item, start, origin = self._press[1], self._press[2], self._press[3]
            delta = pos - start
            if self._press_moved or delta.manhattanLength() > DEAD_ZONE:
                self._press_moved = True
                item.setPos(origin + delta)
        elif kind == "pan":
            d = ev.position() - self._press[1]
            self._press = ("pan", ev.position())
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))

    _press_moved = False

    def mouseReleaseEvent(self, ev):
        press, self._press = self._press, None
        moved, self._press_moved = self._press_moved, False
        if not press:
            return super().mouseReleaseEvent(ev)
        if press[0] == "port":
            target, self._target = self._target, None
            if self._band is not None:
                self.scene_.removeItem(self._band)
                self._band = None
            for b in self.blocks.values():
                for p in list(b.in_ports.values()) + ([b.out_port] if b.out_port else []):
                    p.restyle()
            QToolTip.hideText()
            if target is not None:
                src, dst = self._pair(press[1], target)
                key = (src.block_item.block_id, dst.block_item.block_id, dst.port)
                if self.verdicts.get(key, (False, ""))[0]:
                    self.connect_requested.emit(*key)
        elif press[0] == "block" and moved:
            item = press[1]
            x = round(item.pos().x() / GRID) * GRID
            y = round(item.pos().y() / GRID) * GRID
            self.move_requested.emit(item.block_id, float(x), float(y))

    @staticmethod
    def _pair(a, b):
        """(out-port, in-port) of a drag that may start at either end."""
        return (a, b) if a.side == "out" else (b, a)

    def _verdict(self, src_port, dst_port):
        key = (src_port.block_item.block_id, dst_port.block_item.block_id, dst_port.port)
        if key not in self.verdicts:
            self.verdicts[key] = self.legality(*key)
        return self.verdicts[key]

    def _drag_band(self, start, pos):
        t = _theme.current()
        if self._band is None:
            self._band = QGraphicsPathItem()
            self._band.setZValue(5)
            self.scene_.addItem(self._band)
        want = "in" if start.side == "out" else "out"
        target = self._port_at(pos, want)
        if self._target is not None and self._target is not target:
            self._target.restyle()
        self._target = target
        colour = t["text_muted"]
        if target is not None and target.block_item is not start.block_item:
            src, dst = self._pair(start, target)
            ok, why = self._verdict(src, dst)
            target.restyle("ok" if ok else "bad")
            colour = t["success"] if ok else t["danger"]
            if not ok:
                QToolTip.showText(self.mapToGlobal(self.mapFromScene(pos)), why, self)
            else:
                QToolTip.hideText()
        elif target is not None:
            self._target = None
        a = start.scenePos()
        path = _curve(a, pos) if start.side == "out" else _curve(pos, a)
        pen = QPen(QColor(colour), 2.0, Qt.PenStyle.DashLine)
        self._band.setPen(pen)
        self._band.setPath(path)

    def mouseDoubleClickEvent(self, ev):
        item = self.itemAt(ev.position().toPoint())
        while item is not None and not isinstance(item, BlockItem):
            item = item.parentItem()
        if isinstance(item, BlockItem) and item.kind == "group":
            self.enter_requested.emit(item.block_id)
            return
        super().mouseDoubleClickEvent(ev)

    def wheelEvent(self, ev):
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.15 ** steps
        s = self.transform().m11() * factor
        if 0.25 <= s <= 3.0 * self.text_scale():
            self.scale(factor, factor)

    # ── drop from the palette ──────────────────────────────────────
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):
        if not ev.mimeData().hasFormat(MIME):
            return super().dropEvent(ev)
        try:
            data = json.loads(bytes(ev.mimeData().data(MIME)).decode("utf-8"))
        except ValueError:
            return
        pos = self.mapToScene(ev.position().toPoint())
        x = round((pos.x() - WIDTH / 2) / GRID) * GRID
        y = round((pos.y() - HEADER) / GRID) * GRID
        self.add_requested.emit(str(data.get("kind", "")), str(data.get("preset", "")), float(x), float(y))
        ev.acceptProposedAction()

    # ── keys ───────────────────────────────────────────────────────
    _KEYS = (
        (QKeySequence.StandardKey.Delete, "delete"), (QKeySequence("Backspace"), "delete"),
        (QKeySequence("Ctrl+D"), "duplicate"), (QKeySequence("Ctrl+L"), "layout"), (QKeySequence("F"), "fit"),
        (QKeySequence.StandardKey.Undo, "undo"), (QKeySequence.StandardKey.Redo, "redo"),
        (QKeySequence("Ctrl+Y"), "redo"),
    )

    def _action_for(self, ev):
        combo = QKeySequence(ev.keyCombination())
        for seq, action in self._KEYS:
            seq = QKeySequence(seq) if not isinstance(seq, QKeySequence) else seq
            if combo.matches(seq) == QKeySequence.SequenceMatch.ExactMatch:
                return action
        return None

    def event(self, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Type.ShortcutOverride and self._action_for(ev):
            ev.accept()      # the canvas's keys win over the window's while it has the focus
            return True
        return super().event(ev)

    def keyPressEvent(self, ev):
        action = self._action_for(ev)
        if action == "delete" and self.selection:
            self.delete_requested.emit(self.selection)
        elif action == "duplicate" and self.selection and ">" not in self.selection:
            self.duplicate_requested.emit(self.selection)
        elif action == "layout":
            self.layout_requested.emit()
            self._fitted = False
        elif action == "fit":
            self.fit()
        elif action == "undo":
            self.undo_requested.emit()
        elif action == "redo":
            self.redo_requested.emit()
        else:
            super().keyPressEvent(ev)
