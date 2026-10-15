"""Following a run while it trains, and looking at what it produced.

* `RunProgress` reads the training script's output line by line: which
  phase is running, the step, the train/test loss rows DeepXDE prints, and
  the Error Analysis results at the end. The window's status bar and the
  live loss plot are fed from it.
* `LiveLossPlot`: the loss history as it arrives (matplotlib, with zoom and
  pan), one line per loss term or their sum, phases marked.
* `ErrorAnalysisPanel`: the run's error metrics as a table, and its
  comparison figures.
* `ImageViewer`: any saved figure at full resolution, with wheel zoom and
  drag to pan.
"""
import json
import math
import os
import re
import time

from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from pinnstudio.ui import theme as _theme


# ── what a run is going to do ────────────────────────────────────────────
def planned_phases(config):
    """[(optimizer name, iterations)] the run will train, in order."""
    phases = []
    if getattr(config, "optimizer_scheduler", False) and (getattr(config, "scheduler_phases", "") or "").strip():
        try:
            for p in json.loads(config.scheduler_phases):
                phases.append((str(p.get("optimizer", "")), int(p.get("iterations", 0) or 0)))
        except (ValueError, TypeError):
            phases = []
    if not phases:
        it1 = int(getattr(config, "iterations", 0) or 0)
        if it1:
            phases.append((str(getattr(config, "optimizer", "adam")), it1))
        opt2 = str(getattr(config, "optimizer2", "none") or "none")
        it2 = int(getattr(config, "iterations2", 0) or 0)
        if opt2.lower() != "none" and it2:
            phases.append((opt2, it2))
    return phases


_PHASE = re.compile(r"=== (?:Scheduler )?Phase (\d+)[^:]*: *([\w-]+) +(\d+) iters")
_ROW = re.compile(r"^\s*(\d+)\s+\[([^\]]*)\]\s+\[([^\]]*)\]")
_FINAL = re.compile(r"Final loss:\s*([-\d.eE+]+)")


def _floats(txt):
    out = []
    for v in txt.split(","):
        v = v.strip()
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            return []
    return out


class RunProgress:
    """State of one training run, from its printed output."""

    def __init__(self, config=None):
        self.phases = planned_phases(config) if config is not None else []
        self.total = sum(n for _, n in self.phases)
        self.phase = 0               # 1-based, 0 before training starts
        self.phase_name = ""
        self.step = 0
        self.history = []            # (step, train terms, test terms)
        self.phase_starts = []       # (step, label)
        self.metrics = []            # (t or None, L2, MeanAbs or None)
        self.final_loss = None
        self.figures = []            # figure files the run reported saving
        self.started = time.time()
        self._phase_step0 = 0
        self._phase_done = 0         # iterations of finished phases

    # ── feeding ──
    def feed(self, line):
        m = _PHASE.search(line)
        if m:
            if self.phase and self.history:
                self._phase_done = max(self._phase_done, self.step)
            self.phase = int(m.group(1))
            self.phase_name = m.group(2)
            self._phase_step0 = self.step
            self.phase_starts.append((self.step, f"{self.phase_name}"))
            return
        m = _ROW.match(line)
        if m:
            tr, te = _floats(m.group(2)), _floats(m.group(3))
            if tr:
                step = int(m.group(1))
                if self.phase == 0:
                    self.phase = 1
                    self.phase_starts.append((step, self.phases[0][0] if self.phases else ""))
                self.step = step
                if not self.history or self.history[-1][0] != step or len(self.history[-1][1]) != len(tr):
                    self.history.append((step, tr, te))
            return
        if "MeanAbs=" in line:
            tm = re.search(r"t=([-\d.eE+]+)", line)
            l2 = re.search(r"L2=([-\d.eE+]+)", line)
            ma = re.search(r"MeanAbs=([-\d.eE+]+)", line)
            try:
                self.metrics.append((float(tm.group(1)) if tm else None,
                                     float(l2.group(1)) if l2 else None,
                                     float(ma.group(1)) if ma else None))
            except ValueError:
                pass
            return
        if " saved: " in line and line.rstrip().lower().endswith(".png"):
            path = line.split(" saved: ", 1)[1].strip()
            if path not in self.figures:
                self.figures.append(path)
        m = _FINAL.search(line)
        if m:
            try:
                self.final_loss = float(m.group(1))
            except ValueError:
                pass

    # ── reading ──
    def fraction(self):
        if not self.total:
            return None
        return max(0.0, min(1.0, self.step / self.total))

    def train_loss(self):
        return sum(self.history[-1][1]) if self.history else None

    def test_loss(self):
        return sum(self.history[-1][2]) if self.history and self.history[-1][2] else None

    def elapsed(self):
        return time.time() - self.started

    def eta(self):
        f = self.fraction()
        if not f or f <= 0.02 or f >= 1:
            return None
        return self.elapsed() * (1 - f) / f


def fmt_seconds(s):
    if s is None:
        return "—"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ── live loss plot ───────────────────────────────────────────────────────
class LiveLossPlot(QWidget):
    """The loss history while a run trains: the train and test totals (or
    each term), log scale, phases marked, with matplotlib's zoom/pan bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self._canvas = None
        self._progress = None
        self._dirty = False
        bar = QHBoxLayout()
        self.title = QLabel("Loss during training")
        self.title.setObjectName("plotTitle")
        bar.addWidget(self.title)
        bar.addStretch()
        self.terms_cb = QCheckBox("Each loss term")
        self.terms_cb.setToolTip("One line per loss term (PDE residuals, each BC, IC...) instead of the totals")
        self.terms_cb.toggled.connect(self._redraw)
        bar.addWidget(self.terms_cb)
        lay.addLayout(bar)
        self.holder = QVBoxLayout()
        lay.addLayout(self.holder, 1)
        self.empty = QLabel("The loss curve appears here while a run trains.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setObjectName("plotPlaceholder")
        self.holder.addWidget(self.empty)
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._tick)

    def _ensure_canvas(self):
        if self._canvas is not None:
            return True
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure
        except Exception:
            self.empty.setText("Live loss needs matplotlib's Qt backend.")
            return False
        self._fig = Figure(figsize=(6, 4), dpi=100, layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self.empty.hide()
        self.holder.addWidget(self._toolbar)
        self.holder.addWidget(self._canvas, 1)
        return True

    def start(self, progress):
        self._progress = progress
        self._dirty = True
        if self._ensure_canvas():
            self._redraw()
        self._timer.start()

    def update_from(self, progress):
        self._progress = progress
        self._dirty = True

    def stop(self):
        self._timer.stop()
        self._dirty = True
        self._tick()

    def _tick(self):
        if self._dirty:
            self._dirty = False
            self._redraw()

    def _redraw(self, *_a):
        p = self._progress
        if p is None or not self._ensure_canvas():
            return
        t = _theme.current()
        ax, fig = self._ax, self._fig
        ax.clear()
        fig.set_facecolor(t["card"])
        ax.set_facecolor(t["card"])
        for s in ax.spines.values():
            s.set_color(t["border_strong"])
        ax.tick_params(colors=t["text_muted"], labelsize=9)
        ax.xaxis.label.set_color(t["text_muted"])
        ax.yaxis.label.set_color(t["text_muted"])
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, which="major", color=t["border"], linewidth=0.6)
        hist = p.history
        if hist:
            steps = [h[0] for h in hist]
            if self.terms_cb.isChecked():
                n = max(len(h[1]) for h in hist)
                cols = _theme.BC_COLORS
                for k in range(n):
                    ys = [h[1][k] if k < len(h[1]) else math.nan for h in hist]
                    ax.plot(steps, ys, color=cols[k % len(cols)], lw=1.4, label=f"term {k + 1}")
            else:
                ax.plot(steps, [sum(h[1]) for h in hist], color=t["accent"], lw=1.8, label="train")
                if any(h[2] for h in hist):
                    ax.plot(steps, [sum(h[2]) if h[2] else math.nan for h in hist], color=t["warning"],
                            lw=1.4, ls="--", label="test")
            ax.set_yscale("log")
            for s0, label in p.phase_starts[1:]:
                ax.axvline(s0, color=t["text_faint"], lw=0.8, ls=":")
                ax.text(s0, 1, f" {label}", transform=ax.get_xaxis_transform(), va="top", ha="left",
                        fontsize=8, color=t["text_muted"])
            leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
            leg.get_frame().set_facecolor(t["card"])
            leg.get_frame().set_edgecolor(t["border"])
            for txt in leg.get_texts():
                txt.set_color(t["text"])
            if p.total:
                ax.set_xlim(0, max(p.total, steps[-1]) * 1.01)
        self._canvas.draw_idle()


# ── error analysis results ───────────────────────────────────────────────
class ImageViewer(QDialog):
    """A saved figure at full size: wheel to zoom, drag to pan, double-click to fit."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(os.path.basename(path))
        self.resize(1100, 800)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.view = _ZoomView(path)
        lay.addWidget(self.view)
        hint = QLabel("Wheel to zoom · drag to pan · double-click to fit")
        hint.setObjectName("pageDesc")
        lay.addWidget(hint)


class _ZoomView(QGraphicsView):
    def __init__(self, path):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem(QPixmap(path))
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        QTimer.singleShot(0, self.fit)

    def fit(self):
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, ev):
        f = 1.2 ** (ev.angleDelta().y() / 120)
        self.scale(f, f)

    def mouseDoubleClickEvent(self, _ev):
        self.fit()


class _Thumb(QLabel):
    def __init__(self, path, width=320):
        super().__init__()
        self.path = path
        pm = QPixmap(path)
        if not pm.isNull():
            self.setPixmap(pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation))
        self.setToolTip(f"{os.path.basename(path)} -- click to open at full size")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("thumb")

    def mousePressEvent(self, _ev):
        ImageViewer(self.path, self.window()).exec()


class ErrorAnalysisPanel(QWidget):
    """The last run's error metrics as a table, and its comparison figures."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.summary = QLabel("Run a problem with Error Analysis set up to see its errors here.")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["time", "relative L₂ error", "mean |error|"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        split.addWidget(self.table)
        self.figs = QWidget()
        self.figs_lay = QVBoxLayout(self.figs)
        self.figs_lay.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.figs)
        split.addWidget(scroll)
        split.setSizes([160, 400])
        lay.addWidget(split, 1)

    def show_run(self, metrics, folder, figures=()):
        self.table.setRowCount(0)
        for t, l2, ma in metrics:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate((t, l2, ma)):
                item = QTableWidgetItem("—" if v is None else (f"{v:g}" if c == 0 else f"{v:.3e}"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, item)
        l2s = [m[1] for m in metrics if m[1] is not None]
        if l2s:
            self.summary.setText(f"Relative L₂ error: mean {sum(l2s) / len(l2s):.3e}, worst {max(l2s):.3e} "
                                 f"over {len(l2s)} reference time{'s' if len(l2s) != 1 else ''}.")
        else:
            self.summary.setText("No error analysis in the last run (set it up on the Results page).")
        while self.figs_lay.count():
            w = self.figs_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        paths = [f for f in figures if "error_analysis" in f.replace("\\", "/") and os.path.exists(f)]
        if not paths and folder and os.path.isdir(folder):
            paths = [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.lower().endswith(".png")]
        for path in paths:
            cap = QLabel(os.path.basename(path))
            cap.setObjectName("pageDesc")
            self.figs_lay.addWidget(cap)
            self.figs_lay.addWidget(_Thumb(path, 520))
        self.figs_lay.addStretch()
