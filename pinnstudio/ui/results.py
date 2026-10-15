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

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, QTimer, pyqtSignal
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
_TSTEP = re.compile(r"--- Time step (\d+)/(\d+)")


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
        self.term_names = []         # what each loss term is (PDE, BC 1, IC, ...), from the run's output
        self._groups = {}
        self.time_step = 0           # time-adaptive training: step k of n (0 = not time-adaptive)
        self.n_time_steps = 0

    # ── feeding ──
    def _count(self, line):
        try:
            inner = line[line.rindex("[") + 1:line.rindex("]")]
        except ValueError:
            return 0
        return len([v for v in inner.split(",") if v.strip()])

    def _names_from_weights(self, line):
        g = self._groups
        m = re.search(r"\((\d+) terms", line)
        if not m:
            return
        k = int(m.group(1))
        n_pde, n_bc, n_ic = g.get("pde", 0), g.get("bc", 0), g.get("ic", 0)
        names = (["PDE"] if n_pde == 1 else [f"PDE {i + 1}" for i in range(n_pde)])
        names += [f"BC {i + 1}" for i in range(n_bc)]
        names += (["IC"] if n_ic == 1 else [f"IC {i + 1}" for i in range(n_ic)])
        extra = k - len(names)
        if extra > 0:
            names += ["data"] if extra == 1 else [f"data {i + 1}" for i in range(extra)]
        if len(names) == k:
            self.term_names = names

    def feed(self, line):
        s = line.strip()
        if s.startswith("PDE weights:"):
            self._groups = {"pde": self._count(s)}
            return
        if s.startswith("BC weights"):
            self._groups["bc"] = self._count(s)
            return
        if s.startswith("IC weights:"):
            self._groups["ic"] = self._count(s)
            return
        if s.startswith("Loss weights:") and self._groups:
            self._names_from_weights(s)
            return
        m = _TSTEP.search(line)
        if m:
            # a new time window trains from iteration 0: its own loss history
            self.time_step, self.n_time_steps = int(m.group(1)), int(m.group(2))
            self.history = []
            self.phase_starts = []
            self.phase = 0
            self.phase_name = ""
            self.step = 0
            self._phase_step0 = 0
            self._phase_done = 0
            return
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
        f = max(0.0, min(1.0, self.step / self.total))
        if self.n_time_steps:   # each time window runs the planned phases
            f = (self.time_step - 1 + f) / self.n_time_steps
        return f

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
        from pinnstudio.ui.export_tools import attach_figure_menu
        attach_figure_menu(self._canvas, lambda: self._fig, "live_loss.png")
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
        self.title.setText(f"Loss during training -- time step {p.time_step} of {p.n_time_steps}"
                           if p.n_time_steps else "Loss during training")
        hist = p.history
        if hist:
            steps = [h[0] for h in hist]
            if self.terms_cb.isChecked():
                n = max(len(h[1]) for h in hist)
                cols = _theme.BC_COLORS
                names = p.term_names if len(p.term_names) == n else [f"term {k + 1}" for k in range(n)]
                for k in range(n):
                    ys = [h[1][k] if k < len(h[1]) else math.nan for h in hist]
                    ax.plot(steps, ys, color=cols[k % len(cols)], lw=1.4, label=names[k])
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
        hint = QLabel("Wheel to zoom · drag to pan · double-click to fit · right-click to copy or save")
        hint.setObjectName("pageDesc")
        lay.addWidget(hint)


class _ZoomView(QGraphicsView):
    def __init__(self, path):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.path = path
        from pinnstudio.ui.export_tools import attach_image_menu
        attach_image_menu(self, lambda: self.path)
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

        from pinnstudio.ui.export_tools import attach_image_menu
        attach_image_menu(self, lambda: self.path, lambda: ImageViewer(self.path, self.window()).exec())

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ImageViewer(self.path, self.window()).exec()


class ErrorAnalysisPanel(QWidget):
    """The last run's error metrics as a table, and its comparison figures."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.summary = QLabel("Run a problem with Error Analysis set up to see its errors here.")
        self.summary.setWordWrap(True)
        bar = QHBoxLayout()
        bar.addWidget(self.summary, 1)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy the table (or the selected rows) -- it pastes into a spreadsheet")
        self.export_btn = QPushButton("Export CSV…")
        bar.addWidget(self.copy_btn)
        bar.addWidget(self.export_btn)
        lay.addLayout(bar)
        split = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["time", "relative L₂ error", "mean |error|"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        from pinnstudio.ui.export_tools import attach_table_menu, copy_table, export_table_csv
        attach_table_menu(self.table, "error_metrics.csv")
        self.copy_btn.clicked.connect(lambda: copy_table(self.table))
        self.export_btn.clicked.connect(lambda: export_table_csv(self.table, self, "error_metrics.csv"))
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

    def clear(self, message="The error analysis of this run appears here when it finishes."):
        """A new run started: nothing from the last one stays."""
        self.table.setRowCount(0)
        self.summary.setText(message)
        while self.figs_lay.count():
            w = self.figs_lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def show_run(self, metrics, folder, figures=()):
        self.table.setRowCount(0)
        for t, l2, ma in metrics:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate((t, l2, ma)):
                item = QTableWidgetItem("—" if v is None else (f"{v:g}" if c == 0 else f"{v:.3e}"))
                if v is not None:
                    item.setData(Qt.ItemDataRole.UserRole, float(v))
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


# ── Run history ──────────────────────────────────────────────────────────
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


def read_run(folder):
    """One finished run's summary from its folder (None if it isn't one):
    when, final training loss, relative L2 error, plot files."""
    sol = os.path.join(folder, "solution_results")
    if not os.path.isdir(sol):
        return None
    run = {"folder": folder, "name": os.path.basename(folder.rstrip("/\\")), "loss": None, "l2": None,
           "metrics": [], "plot_type": "", "time": None}
    m = _STAMP.search(run["name"]) or _STAMP.search(os.path.basename(os.path.dirname(folder)))
    if m:
        run["when"] = f"{m.group(1)} {m.group(2)}:{m.group(3)}"
    try:
        run["time"] = os.path.getmtime(sol)
    except OSError:
        run["time"] = 0
    if "when" not in run:
        run["when"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(run["time"] or 0))
    label = _STAMP.sub("", run["name"]).rstrip("_ ")
    run["label"] = label or run["name"]
    try:
        with open(os.path.join(sol, "training_log.txt"), encoding="utf-8", errors="replace") as fh:
            last = None
            for line in fh:
                if line.strip().startswith("Step"):
                    last = line
        if last:
            vals = _log_values(last)
            if vals:
                run["loss"] = sum(vals)
    except OSError:
        pass
    try:
        with open(os.path.join(folder, "error_analysis", "error_metrics.txt"), encoding="utf-8") as fh:
            rows = [ln.strip().split(",") for ln in fh if ln.strip()]
        for r in rows[1:]:
            try:
                run["metrics"].append((float(r[0]), float(r[1]), float(r[2]) if len(r) > 2 else None))
            except (ValueError, IndexError):
                pass
        l2s = [x[1] for x in run["metrics"]]
        if l2s:
            run["l2"] = sum(l2s) / len(l2s)
    except OSError:
        pass
    try:
        with open(os.path.join(sol, "model_config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        run["plot_type"] = cfg.get("plot_type", "") or ""
        run["dim"] = cfg.get("problem_dim", "")
    except (OSError, ValueError, AttributeError):
        pass
    return run


def _log_values(line):
    body = re.sub(r"(?:np\.)?float\d*\(|tensor\(|array\(", "", line.split(":", 1)[1])
    return _floats(re.sub(r"[\[\]()]", "", body))


def read_loss_history(folder):
    """[(step, total training loss)] from a run's training_log.txt."""
    out = []
    try:
        with open(os.path.join(folder, "solution_results", "training_log.txt"), encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                line = line.strip()
                m = re.match(r"Step\s+(\d+)\s*:", line)
                if not m:
                    continue
                vals = _log_values(line)
                if vals:
                    out.append((int(m.group(1)), sum(vals)))
    except OSError:
        pass
    return out


def scan_runs(base, max_depth=3, limit=500):
    """Every run folder under `base` (a run's own folder, or a sweep's runs),
    newest first."""
    out = []
    if not base or not os.path.isdir(base):
        return out

    def walk(d, depth):
        if len(out) >= limit:
            return
        try:
            names = os.listdir(d)
        except OSError:
            return
        if "solution_results" in names:
            r = read_run(d)
            if r:
                out.append(r)
            return
        if depth >= max_depth:
            return
        for n in names:
            p = os.path.join(d, n)
            if os.path.isdir(p) and not n.startswith("."):
                walk(p, depth + 1)

    walk(base, 0)
    # newest first; the runs of one sweep (same date) in their own order
    out.sort(key=lambda r: r["folder"])
    out.sort(key=lambda r: r.get("when") or "", reverse=True)
    return out


def short_label(run):
    """A run's name for a legend: without the sweep's "run_01__" prefix."""
    return re.sub(r"^run_\d+__", "", run.get("label") or run.get("name") or "run").replace("_", " ")


class RunCompare(QWidget):
    """Several runs on one figure: training loss against iteration, and
    the relative L2 error against time (when the runs have error analysis)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._canvas = None
        self.runs = []
        self.empty = QLabel("Select two or more runs and press Compare.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setObjectName("plotPlaceholder")
        lay.addWidget(self.empty)
        self.holder = lay

    def _ensure_canvas(self):
        if self._canvas is not None:
            return True
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure
        except Exception:
            self.empty.setText("Comparing runs needs matplotlib's Qt backend.")
            return False
        self._fig = Figure(figsize=(8, 3.4), dpi=100, layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        from pinnstudio.ui.export_tools import attach_figure_menu
        attach_figure_menu(self._canvas, lambda: self._fig, "run_comparison.png")
        self.empty.hide()
        self.holder.addWidget(self._toolbar)
        self.holder.addWidget(self._canvas, 1)
        return True

    def set_runs(self, runs):
        self.runs = list(runs)
        if self._ensure_canvas():
            self.redraw()

    def redraw(self):
        if self._canvas is None:
            return
        t = _theme.current()
        fig = self._fig
        fig.clear()
        fig.set_facecolor(t["card"])
        with_err = [r for r in self.runs if r.get("metrics")]
        axes = fig.subplots(1, 2 if with_err else 1, squeeze=False)[0]
        for ax in axes:
            ax.set_facecolor(t["card"])
            for sp in ax.spines.values():
                sp.set_color(t["border_strong"])
            ax.tick_params(colors=t["text_muted"], labelsize=9)
            ax.xaxis.label.set_color(t["text_muted"])
            ax.yaxis.label.set_color(t["text_muted"])
            ax.title.set_color(t["text"])
            ax.grid(True, which="major", color=t["border"], linewidth=0.6)
        cols = _theme.BC_COLORS
        ax = axes[0]
        any_loss = False
        for i, r in enumerate(self.runs):
            hist = read_loss_history(r["folder"])
            if hist:
                any_loss = True
                ax.plot([h[0] for h in hist], [h[1] for h in hist], color=cols[i % len(cols)], lw=1.6,
                        label=short_label(r))
        ax.set_xlabel("iteration")
        ax.set_ylabel("training loss")
        ax.set_title("Loss", fontsize=10)
        if any_loss:
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "No training log saved for these runs", transform=ax.transAxes, ha="center",
                    va="center", color=t["text_muted"], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        if with_err:
            ax2 = axes[1]
            for i, r in enumerate(self.runs):
                m = r.get("metrics") or []
                if m:
                    ts = [x[0] for x in m]
                    style = "o-" if len(m) > 1 else "o"
                    ax2.plot(ts, [x[1] for x in m], style, color=cols[i % len(cols)], lw=1.6, ms=4,
                             label=short_label(r))
            ax2.set_xlabel("t")
            ax2.set_ylabel("relative L₂ error")
            ax2.set_title("Error", fontsize=10)
            ax2.set_yscale("log")
        leg = axes[-1].legend(loc="best", fontsize=8, frameon=True) if self.runs else None
        if leg is not None:
            leg.get_frame().set_facecolor(t["card"])
            leg.get_frame().set_edgecolor(t["border"])
            for txt in leg.get_texts():
                txt.set_color(t["text"])
        self._canvas.draw_idle()


class RunHistory(QWidget):
    """The runs saved in the Save-to folder: when, which, final loss and
    error. Double-click (or Show) brings a run's figures back up."""
    run_activated = pyqtSignal(dict)

    def __init__(self, base_fn, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QPushButton
        self.base_fn = base_fn
        self.runs = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.summary = QLabel("")
        self.summary.setObjectName("pageDesc")
        self.summary.setWordWrap(True)
        bar.addWidget(self.summary, 1)
        self.show_btn = QPushButton("Show")
        self.show_btn.setToolTip("Show this run's figures in Overview and its errors in Error analysis")
        self.show_btn.clicked.connect(self._activate_current)
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setToolTip("Select two or more runs (Ctrl/Shift-click): their loss and error on one plot")
        self.compare_btn.clicked.connect(self.compare_selected)
        self.folder_btn = QPushButton("Open folder")
        self.folder_btn.clicked.connect(self._open_folder)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.setToolTip("Every run's name, date, final loss and error, as a CSV file")
        for b in (self.show_btn, self.compare_btn, self.folder_btn, self.export_btn, self.refresh_btn):
            bar.addWidget(b)
        lay.addLayout(bar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["run", "when", "final loss", "relative L₂ error"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(False)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._activate(r))
        from pinnstudio.ui.export_tools import attach_table_menu, export_table_csv
        attach_table_menu(self.table, "runs.csv")
        self.export_btn.clicked.connect(lambda: export_table_csv(self.table, self, "runs.csv"))
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(self.table)
        self.compare = RunCompare()
        self.compare.setVisible(False)
        self.split.addWidget(self.compare)
        self.split.setChildrenCollapsible(False)
        lay.addWidget(self.split, 1)
        self._update_buttons()

    stale = True
    _scanned = None

    def refresh(self):
        base = ""
        try:
            base = (self.base_fn() or "").strip()
        except Exception:
            pass
        self._scanned, self.stale = base, False
        self.runs = scan_runs(base)
        self.table.setRowCount(0)
        for r in self.runs:
            i = self.table.rowCount()
            self.table.insertRow(i)
            rel = os.path.relpath(r["folder"], base) if base else r["folder"]
            name = QTableWidgetItem(r["label"] if os.sep not in rel else rel.replace(os.sep, " / "))
            name.setToolTip(r["folder"])
            self.table.setItem(i, 0, name)
            for c, v, full in ((1, r["when"], None), (2, "—" if r["loss"] is None else f"{r['loss']:.3e}", r["loss"]),
                               (3, "—" if r["l2"] is None else f"{r['l2']:.3e}", r["l2"])):
                it = QTableWidgetItem(v)
                if full is not None:
                    it.setData(Qt.ItemDataRole.UserRole, float(full))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, c, it)
        if not base:
            self.summary.setText("Runs are listed from the Save-to folder (Results page); it is empty.")
        elif not self.runs:
            self.summary.setText(f"No saved runs yet in {base}.")
        else:
            self.summary.setText(f"{len(self.runs)} run{'s' if len(self.runs) != 1 else ''} in {base}  "
                                 "-- double-click one to see its figures.")
        self._update_buttons()

    def selected_rows(self):
        sm = self.table.selectionModel()
        rows = sorted(i.row() for i in sm.selectedRows()) if sm else []
        return [r for r in rows if 0 <= r < len(self.runs)]

    def _row(self):
        rows = self.selected_rows()
        return rows[0] if rows else -1

    def _update_buttons(self):
        n = len(self.selected_rows())
        self.show_btn.setEnabled(n == 1)
        self.folder_btn.setEnabled(n == 1)
        self.compare_btn.setEnabled(n >= 2)

    def compare_selected(self):
        rows = self.selected_rows()
        if len(rows) < 2:
            return
        self.compare.setVisible(True)
        self.compare.set_runs([self.runs[r] for r in rows])
        self.split.setSizes([max(120, self.split.height() // 3), max(240, 2 * self.split.height() // 3)])

    def _activate(self, row):
        if 0 <= row < len(self.runs):
            self.run_activated.emit(self.runs[row])

    def _activate_current(self):
        self._activate(self._row())

    def _open_folder(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        r = self._row()
        if 0 <= r < len(self.runs):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.runs[r]["folder"]))

    def showEvent(self, ev):
        # rescanning a large folder on every visit would freeze the window:
        # only when the folder changed or a run has finished since
        super().showEvent(ev)
        try:
            base = (self.base_fn() or "").strip()
        except Exception:
            base = ""
        if self.stale or base != self._scanned:
            self.refresh()


class FitImage(QObject):
    """Keeps a figure label's picture (its `_source_path`) fitted to the
    label as the panels are resized or the tab is first shown -- a figure
    loaded while its tab was hidden would otherwise stay thumbnail-sized."""

    def __init__(self, label):
        super().__init__(label)
        self.label = label
        self._cache = (None, None)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.refit)
        label.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._timer.start()
        return False

    def refit(self):
        lab = self.label
        try:
            path = getattr(lab, "_source_path", None)
            if not path or path.lower().endswith(".gif") or lab.movie() is not None:
                return
            pm = lab.pixmap()
            if pm is None or pm.isNull() or not os.path.exists(path):
                return
            key = (path, os.path.getmtime(path))
            if self._cache[0] != key:
                self._cache = (key, QPixmap(path))
            src = self._cache[1]
            if src.isNull():
                return
            w, h = max(10, lab.width() - 4), max(10, lab.height() - 4)
            if abs(pm.width() - min(w, src.width() * h / max(1, src.height()))) < 3 and pm.height() <= h:
                return
            lab.setPixmap(src.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
        except RuntimeError:
            pass
