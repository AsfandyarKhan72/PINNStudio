"""Figure style: one dialog for how every saved figure looks.

Solve and Restore & Visualize draw with the same style (colours, sizes,
fonts, line widths ...), kept in the window's _plot_viz_settings and
_loss_plot_settings and saved with the problem. The dialog has a tab per
kind of plot; the Style... button next to a Plot type list opens it on that
plot's tab, and Settings -> Figure style opens it on General.

What depends on the problem being plotted -- the number of time steps or
frames, and where a line plot slices a 2D/3D domain -- is kept per place:
Solve's in its own settings, Restore's with the model being restored
(`context` "solve" or "restore").
"""
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

TABS = ("General", "Surface", "Line", "Animation", "Loss", "Parameters")

COLORMAPS = ["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"]
PARAM_COLORS = [("Green", "#69db7c"), ("Blue", "#1c7ed6"), ("Black", "#212529"), ("Red", "#e03131"),
                ("Orange", "#f08c00"), ("Purple", "#7048e8")]


def tab_for(viz_type):
    """The tab a plot type's Style... button opens."""
    v = viz_type or ""
    if "Parameter" in v:
        return "Parameters"
    if "Animation" in v:
        return "Animation"
    if v.startswith("Line"):
        return "Line"
    if v.startswith("Surface"):
        return "Surface"
    if v == "Loss":
        return "Loss"
    return "General"


def _form(tab):
    f = QFormLayout(tab)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    f.setLabelAlignment(f.labelAlignment())
    f.setVerticalSpacing(8)
    return f


def _combo(items, value, width=None):
    c = QComboBox()
    for it in items:
        if isinstance(it, tuple):
            c.addItem(it[0], it[1])
        else:
            c.addItem(it, it)
    i = c.findData(value)
    if i < 0:
        i = c.findText(str(value))
    if i >= 0:
        c.setCurrentIndex(i)
    if width:
        c.setMinimumWidth(width)
    return c


def _spin(lo, hi, value, step=None, double=False, width=90):
    s = QDoubleSpinBox() if double else QSpinBox()
    s.setRange(lo, hi)
    if step:
        s.setSingleStep(step)
    s.setValue(value)
    s.setMinimumWidth(width)
    return s


def _hint(text):
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setObjectName("figureStyleHint")
    lbl.setStyleSheet("color: #868e96;")
    return lbl


class FigureStyleDialog(QDialog):
    def __init__(self, win, context="solve", tab="General"):
        super().__init__(win)
        self.win = win
        self.context = context
        self.setWindowTitle("Figure Style")
        self.setMinimumWidth(540)
        self.style = dict(win._plot_viz_settings)
        self.loss = dict(win._loss_plot_settings)
        self.view = dict(getattr(win, "_restore_viz_settings", {}) or {})
        self._problem()

        lay = QVBoxLayout(self)
        lay.addWidget(_hint("Used by every figure PINNStudio saves -- after training, in the exported script and "
                            "in Restore & Visualize -- and saved with the problem."))
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._general()
        self._surface()
        self._line()
        self._animation()
        self._loss_tab()
        self._param()
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        if tab in names:
            self.tabs.setCurrentIndex(names.index(tab))

        row = QHBoxLayout()
        reset = QPushButton("Reset to defaults")
        reset.setToolTip("The figure style PINNStudio starts with (this tab only)")
        reset.clicked.connect(self._reset_tab)
        row.addWidget(reset)
        row.addStretch()
        ok = QPushButton("OK")
        ok.setDefault(True)
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self._ok)
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)

    # ── the problem the plots are for ──
    def _problem(self):
        w = self.win
        self.is_inverse = False
        self.y_rng, self.z_rng = (0.0, 1.0), (0.0, 1.0)
        if self.context == "restore":
            import json
            cfg = {}
            try:
                with open(w.restore_config_path.text().strip(), encoding="utf-8", errors="replace") as f:
                    cfg = json.load(f)
            except Exception:
                pass
            self.dim = cfg.get("problem_dim", "1D")
            self.steady = bool(cfg.get("steady_state", False))
            self.is_inverse = cfg.get("problem_type") == "Inverse" or bool(getattr(w, "_restore_is_inverse", False))
            self.y_rng = (cfg.get("y_min", 0.0), cfg.get("y_max", 1.0))
            self.z_rng = (cfg.get("z_min", 0.0), cfg.get("z_max", 1.0))
        else:
            self.dim = "1D" if w.radio_1d.isChecked() else "2D" if w.radio_2d.isChecked() else \
                "3D" if w.radio_3d.isChecked() else "ODE"
            self.steady = bool(getattr(w, "steady_state_check", None) and w.steady_state_check.isChecked())
            self.is_inverse = w.radio_inverse.isChecked()

    # ── tabs ──
    def _page(self, name):
        page = QWidget()
        self.tabs.addTab(page, name)
        return _form(page)

    def _general(self):
        s = self.style
        f = self._page("General")
        self.dpi = _combo(["100", "150", "200", "300", "600"], str(s.get("dpi", 100)), 90)
        f.addRow("Resolution (DPI):", self.dpi)
        self.figsize = _combo(["Default", "Square", "Wide", "Custom"], s.get("figsize_mode", "Default"), 110)
        f.addRow("Figure size:", self.figsize)
        self.fig_w = _spin(2.0, 30.0, s.get("figsize_w", 7.0), 0.5, True, 80)
        self.fig_h = _spin(2.0, 30.0, s.get("figsize_h", 5.0), 0.5, True, 80)
        wh = QWidget()
        whl = QHBoxLayout(wh)
        whl.setContentsMargins(0, 0, 0, 0)
        whl.addWidget(self.fig_w)
        whl.addWidget(QLabel("×"))
        whl.addWidget(self.fig_h)
        whl.addWidget(QLabel("inches"))
        f.addRow("Width × height:", wh)
        self._wh_row = wh

        def _sync(_t=None):
            wh.setEnabled(self.figsize.currentText() == "Custom")
        self.figsize.currentTextChanged.connect(_sync)
        _sync()
        self.title = QLineEdit(s.get("title", ""))
        self.xlabel = QLineEdit(s.get("xlabel", ""))
        self.ylabel = QLineEdit(s.get("ylabel", ""))
        for e in (self.title, self.xlabel, self.ylabel):
            e.setPlaceholderText("(default)")
            e.setMinimumWidth(220)
        f.addRow("Title:", self.title)
        f.addRow("X-axis label:", self.xlabel)
        f.addRow("Y-axis label:", self.ylabel)
        self.title_fs = _spin(6.0, 36.0, s.get("title_fontsize", 13.0), 0.5, True, 80)
        self.sub_fs = _spin(6.0, 36.0, s.get("subplot_title_fontsize", 10.0), 0.5, True, 80)
        self.axis_fs = _spin(6.0, 36.0, s.get("axis_label_fontsize", 10.0), 0.5, True, 80)
        f.addRow("Title font size:", self.title_fs)
        f.addRow("Panel title font size:", self.sub_fs)
        f.addRow("Axis label font size:", self.axis_fs)
        f.addRow(_hint("Blank titles and labels keep each plot's own. Error analysis figures size themselves."))

    def _surface(self):
        s = self.style
        f = self._page("Surface")
        self.cmap = _combo(COLORMAPS, s.get("colormap", "RdBu_r"), 120)
        f.addRow("Colour map:", self.cmap)
        self.levels = _spin(5, 200, int(s.get("levels", 50)), width=80)
        f.addRow("Contour levels:", self.levels)
        self.res = _combo(["80", "100", "150", "200", "300"], str(s.get("resolution", 100)), 80)
        f.addRow("Grid resolution:", self.res)
        self.auto_range = QCheckBox("Automatic")
        self.auto_range.setChecked(bool(s.get("auto_range", True)))
        self.vmin = _spin(-1e6, 1e6, s.get("vmin", -1.0), None, True, 90)
        self.vmax = _spin(-1e6, 1e6, s.get("vmax", 1.0), None, True, 90)
        cr = QWidget()
        crl = QHBoxLayout(cr)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(self.auto_range)
        crl.addWidget(self.vmin)
        crl.addWidget(QLabel("to"))
        crl.addWidget(self.vmax)

        def _sync(_s=None):
            on = not self.auto_range.isChecked()
            self.vmin.setEnabled(on)
            self.vmax.setEnabled(on)
        self.auto_range.toggled.connect(_sync)
        _sync()
        f.addRow("Colour range:", cr)
        self.colorbar = QCheckBox("Show colour bar")
        self.colorbar.setChecked(bool(s.get("colorbar", True)))
        f.addRow("", self.colorbar)
        self.snaps = _spin(1, 10, int(s.get("n_2d_snapshots", 2)), width=80)
        if self.dim in ("2D", "3D") and not self.steady:
            f.addRow("Time snapshots:", self.snaps)
        self.swap = QCheckBox("Time on the x-axis")
        self.swap.setChecked(bool(s.get("swap_xt", True)))
        if self.dim == "1D":
            f.addRow("1D surface:", self.swap)

    def _steps(self, anim):
        """Time steps (Line) or frames (Animation): Solve's own setting, or
        Restore's."""
        w = self.win
        if self.context == "restore":
            return int(self.view.get("n_steps_anim" if anim else "n_steps_line", 20 if anim else 5)), 100
        spin = w.timesteps_spin_anim if anim else w.timesteps_spin_line
        return spin.value(), spin.maximum()

    def _line(self):
        s = self.style
        f = self._page("Line")
        v, mx = self._steps(False)
        self.steps_line = _spin(2, mx, v, width=80)
        f.addRow("Time steps to show:", self.steps_line)
        self.lw_line = _combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(float(s.get("linewidth", 2.0))), 80)
        f.addRow("Line width:", self.lw_line)
        self.slices = {}
        if self.dim in ("2D", "3D"):
            f.addRow(_hint("A line plot shows u along x; the rest of the domain is cut at:"))
            for ax, rng in (("y", self.y_rng), ("z", self.z_rng)):
                if ax == "z" and self.dim != "3D":
                    continue
                if self.context == "restore":
                    auto = bool(self.view.get(f"line_slice_{ax}_auto", True))
                    val = self.view.get(f"line_slice_{ax}", (rng[0] + rng[1]) / 2.0)
                else:
                    auto = getattr(self.win, f"line_slice_{ax}_auto_cb").isChecked()
                    val = getattr(self.win, f"line_slice_{ax}_spin").value()
                cb = QCheckBox("middle of the domain")
                cb.setChecked(auto)
                sp = _spin(-1e6, 1e6, (rng[0] + rng[1]) / 2.0 if auto else val, None, True, 90)
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(cb)
                rl.addWidget(sp)
                cb.toggled.connect(lambda on, sp=sp: sp.setEnabled(not on))
                sp.setEnabled(not auto)
                f.addRow(f"{ax} =", row)
                self.slices[ax] = (cb, sp)

    def _animation(self):
        s = self.style
        f = self._page("Animation")
        v, mx = self._steps(True)
        self.frames = _spin(2, mx, v, width=80)
        f.addRow("Frames:", self.frames)
        self.fps = _combo(["5", "8", "10", "15", "20", "24", "30"], str(int(s.get("fps", 10))), 80)
        f.addRow("Frame rate (fps):", self.fps)
        self.lw_anim = _combo(["1.0", "1.5", "2.0", "2.5", "3.0", "4.0"], str(float(s.get("linewidth_anim", 3.0))), 80)
        f.addRow("Line width:", self.lw_anim)
        f.addRow(_hint("Surface animations use the Surface tab's colour map, levels and colour range."))

    def _loss_tab(self):
        s = self.loss
        f = self._page("Loss")
        self.loss_mode = _combo([("Train + Test totals", "train_test"), ("Each loss term", "individual"),
                                 ("Totals and each term", "all")], s.get("mode", "train_test"), 190)
        f.addRow("Show:", self.loss_mode)
        self.loss_lw = _combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(float(s.get("linewidth", 2.0))), 80)
        f.addRow("Line width:", self.loss_lw)
        self.loss_log = QCheckBox("Logarithmic")
        self.loss_log.setChecked(bool(s.get("log_y", True)))
        f.addRow("Loss axis:", self.loss_log)
        self.loss_every = _spin(1, 100000, int(s.get("display_every", 1000)), width=90)
        f.addRow("Record every:", self.loss_every)
        f.addRow(_hint("Record every: how often (in iterations) the loss is recorded and printed during training. "
                       "DeepXDE keeps one test loss, so each term is shown for the training loss only."))

    def _param(self):
        s = self.style
        f = self._page("Parameters")
        self.p_color = _combo(PARAM_COLORS, s.get("param_color", "#69db7c"), 110)
        f.addRow("Line colour:", self.p_color)
        self.p_lw = _combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(float(s.get("param_linewidth", 1.5))), 80)
        f.addRow("Line width:", self.p_lw)
        self.p_markers = QCheckBox("Mark each saved value")
        self.p_markers.setChecked(bool(s.get("param_markers", False)))
        f.addRow("Markers:", self.p_markers)
        self.p_true = QCheckBox("True value (when it is known)")
        self.p_true.setChecked(bool(s.get("param_show_true", True)))
        self.p_final = QCheckBox("Final value")
        self.p_final.setChecked(bool(s.get("param_show_final", True)))
        f.addRow("Dashed lines:", self.p_true)
        f.addRow("", self.p_final)
        self.p_log = QCheckBox("Logarithmic")
        self.p_log.setChecked(self.win.inv_param_log_scale.isChecked())
        f.addRow("Value axis:", self.p_log)
        f.addRow(_hint("The animation uses the Animation tab's frame rate. Size, resolution and fonts are on "
                       "the General tab."))
        if not self.is_inverse:          # (only inverse problems have parameters to plot)
            self.tabs.setTabVisible(self.tabs.count() - 1, False)

    # ── buttons ──
    def _reset_tab(self):
        from pinnstudio.core.config import PINNConfig
        d = PINNConfig()
        name = self.tabs.tabText(self.tabs.currentIndex())
        if name == "General":
            self.dpi.setCurrentText(str(d.plot_dpi))
            self.figsize.setCurrentText(d.plot_figsize_mode)
            self.fig_w.setValue(d.plot_figsize_w)
            self.fig_h.setValue(d.plot_figsize_h)
            for e in (self.title, self.xlabel, self.ylabel):
                e.clear()
            self.title_fs.setValue(d.plot_title_fontsize)
            self.sub_fs.setValue(d.plot_subplot_title_fontsize)
            self.axis_fs.setValue(d.plot_axis_label_fontsize)
        elif name == "Surface":
            self.cmap.setCurrentText(d.plot_colormap)
            self.levels.setValue(d.plot_levels)
            self.res.setCurrentText(str(d.plot_resolution))
            self.auto_range.setChecked(d.plot_auto_range)
            self.colorbar.setChecked(d.plot_colorbar)
            self.snaps.setValue(d.plot_n_2d_snapshots)
            self.swap.setChecked(d.plot_swap_xt)
        elif name == "Line":
            self.lw_line.setCurrentText(str(float(d.plot_linewidth)))
            for cb, _sp in self.slices.values():
                cb.setChecked(True)
        elif name == "Animation":
            self.fps.setCurrentText(str(d.plot_fps))
            self.lw_anim.setCurrentText(str(float(d.plot_linewidth_anim)))
        elif name == "Loss":
            self.loss_mode.setCurrentIndex(self.loss_mode.findData(d.loss_plot_mode))
            self.loss_lw.setCurrentText(str(float(d.loss_plot_linewidth)))
            self.loss_log.setChecked(d.loss_plot_log_y)
            self.loss_every.setValue(d.loss_display_every)
        else:
            self.p_color.setCurrentIndex(max(0, self.p_color.findData(d.param_plot_color)))
            self.p_lw.setCurrentText(str(float(d.param_plot_linewidth)))
            self.p_markers.setChecked(d.param_plot_markers)
            self.p_true.setChecked(d.param_plot_show_true)
            self.p_final.setChecked(d.param_plot_show_final)
            self.p_log.setChecked(d.inv_param_log_scale)

    def values(self):
        s = dict(self.style)
        s.update({
            "dpi": int(self.dpi.currentText()),
            "figsize_mode": self.figsize.currentText(),
            "figsize_w": self.fig_w.value(), "figsize_h": self.fig_h.value(),
            "title": self.title.text().strip(), "xlabel": self.xlabel.text().strip(),
            "ylabel": self.ylabel.text().strip(),
            "title_fontsize": self.title_fs.value(), "subplot_title_fontsize": self.sub_fs.value(),
            "axis_label_fontsize": self.axis_fs.value(),
            "colormap": self.cmap.currentText(), "levels": self.levels.value(),
            "resolution": int(self.res.currentText()), "auto_range": self.auto_range.isChecked(),
            "vmin": self.vmin.value(), "vmax": self.vmax.value(), "colorbar": self.colorbar.isChecked(),
            "n_2d_snapshots": self.snaps.value(), "swap_xt": self.swap.isChecked(),
            "linewidth": float(self.lw_line.currentText()), "linewidth_anim": float(self.lw_anim.currentText()),
            "fps": int(self.fps.currentText()),
            "param_color": self.p_color.currentData(), "param_linewidth": float(self.p_lw.currentText()),
            "param_markers": self.p_markers.isChecked(), "param_show_true": self.p_true.isChecked(),
            "param_show_final": self.p_final.isChecked(),
        })
        loss = {"mode": self.loss_mode.currentData(), "display_every": self.loss_every.value(),
                "linewidth": float(self.loss_lw.currentText()), "log_y": self.loss_log.isChecked()}
        return s, loss

    def _ok(self):
        w = self.win
        s, loss = self.values()
        w._plot_viz_settings = s
        w._loss_plot_settings = loss
        w.inv_param_log_scale.setChecked(self.p_log.isChecked())
        if self.context == "restore":
            view = dict(self.view)
            view["n_steps_line"] = self.steps_line.value()
            view["n_steps_anim"] = self.frames.value()
            for ax, (cb, sp) in self.slices.items():
                view[f"line_slice_{ax}_auto"] = cb.isChecked()
                view[f"line_slice_{ax}"] = sp.value()
            w._restore_viz_settings = view
            if hasattr(w, "restore_tsteps_spin"):
                anim = "Animation" in w.restore_viz_combo.currentText()
                w.restore_tsteps_spin.setValue(self.frames.value() if anim else self.steps_line.value())
        else:
            w.timesteps_spin_line.setValue(self.steps_line.value())
            w.timesteps_spin_anim.setValue(self.frames.value())
            for ax, (cb, sp) in self.slices.items():
                getattr(w, f"line_slice_{ax}_auto_cb").setChecked(cb.isChecked())
                getattr(w, f"line_slice_{ax}_spin").setValue(sp.value())
        try:
            w.log_box.append("✅ Figure style saved")
        except Exception:
            pass
        self.accept()


def open_figure_style(win, viz_type=None, context="solve", tab=None):
    dlg = FigureStyleDialog(win, context, tab or tab_for(viz_type))
    return dlg.exec()
