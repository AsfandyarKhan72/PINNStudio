"""Plot settings: one dialog for how every saved figure looks.

Solve, the exported script and Restore & visualize draw with the same
settings (colours, sizes, fonts, line widths, time steps ...), kept in the
window's _plot_viz_settings, _loss_plot_settings and its steps / slice
widgets, and saved with the problem. The dialog has a tab per kind of plot;
the Results page's Plot settings button opens it on the tab of the plot
type, Settings -> Plot Settings on General. An inverse problem's parameter
convergence plot is set on the Line tab, its animation on Animation.
"""
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

TABS = ("General", "Surface", "Line", "Animation", "Loss")

COLORMAPS = ["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"]
PARAM_COLORS = [("Red", "#e03131"), ("Blue", "#1c7ed6"), ("Black", "#212529"), ("Grey", "#868e96"),
                ("Green", "#2f9e44"), ("Orange", "#f08c00"), ("Purple", "#7048e8")]
LINE_COLORS = [("Light red", "#ff6b6b"), ("Light blue", "#4dabf7")] + PARAM_COLORS
LINE_STYLES = [("Dashed", "--"), ("Dotted", ":"), ("Dash-dot", "-."), ("Solid", "-")]
DRAW_AS = [("Line", "line"), ("Line and markers", "line_markers"), ("Markers only", "markers")]
MARKERS = [("Circle", "o"), ("Square", "s"), ("Triangle", "^"), ("Cross", "x")]


def tab_for(viz_type):
    """The tab a plot type's Style... button opens."""
    v = viz_type or ""
    if "Animation" in v or "(GIF)" in v:
        return "Animation"
    if "Parameter" in v:
        return "Line"
    if v.startswith("Line"):
        return "Line"
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
    if i < 0 and isinstance(value, str) and value.startswith("#"):
        c.addItem(value, value)          # a colour of your own, kept as it is
        i = c.count() - 1
    if i < 0 and value not in (None, "") and not any(isinstance(it, tuple) for it in items):
        # a value that is not in the list (a DeepXDE example's grid of 500,
        # a colormap of your own) is kept, not replaced by the first item
        c.addItem(str(value), str(value))
        i = c.count() - 1
    if i >= 0:
        c.setCurrentIndex(i)
    if width:
        c.setMinimumWidth(width)
    return c


def _edit_combo(items, value, width=None):
    """_combo whose value can also be typed: any number (DPI, grid, fps,
    line width) or any matplotlib colormap name."""
    c = _combo(items, value, width)
    c.setEditable(True)
    c.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    return c


def _num(combo, kind, fallback):
    """A typed number from an editable list, or `fallback` if it isn't one."""
    try:
        v = kind(float(combo.currentText().strip()))
        return v if v > 0 else fallback
    except (TypeError, ValueError):
        return fallback


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
        self.setWindowTitle("Plot Settings")
        self.setMinimumWidth(540)
        self.style = dict(win._plot_viz_settings)
        self.loss = dict(win._loss_plot_settings)
        self.view = dict(getattr(win, "_restore_viz_settings", {}) or {})
        self._problem()

        lay = QVBoxLayout(self)
        lay.addWidget(_hint("Used by every figure PINNStudio saves -- after training, in the exported script and "
                            "in Restore & visualize -- and saved with the problem."))
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self._general()
        self._surface()
        self._line()
        self._animation()
        self._loss_tab()
        self._hide_what_does_not_apply()
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
        self.dpi = _edit_combo(["100", "150", "200", "300", "600"], str(s.get("dpi", 100)), 90)
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
        self.legend_fs = _spin(0.0, 36.0, float(s.get("legend_fontsize", 0.0) or 0.0), 0.5, True, 80)
        self.legend_fs.setSpecialValueText("Automatic")
        self.legend_fs.setToolTip("Font size of every legend. Automatic keeps each figure's own (7 to 10).")
        f.addRow("Legend font size:", self.legend_fs)
        self.grid_cb = QCheckBox("Show on line-type figures")
        self.grid_cb.setToolTip("Line plots and animations, the line comparisons of Error Analysis,\n"
                                "the loss plot and the parameter convergence plot and animation.")
        self.grid_cb.setChecked(bool(s.get("grid", True)))
        f.addRow("Grid lines:", self.grid_cb)
        f.addRow(_hint("Blank titles and labels keep each plot's own. Error analysis figures scale their panels with the figure size."))
        # Numbers written in figures: times in titles, errors, parameter
        # values (two digits more), and the notation of axis/colorbar ticks.
        self._section(f, "Numbers")
        self.sig_digits = _spin(1, 10, int(s.get("sig_digits", 4)), 1, False, 70)
        self.sig_digits.setToolTip("Significant digits of the numbers written in figures: times in titles,\n"
                                   "error values, and (two digits more) inferred parameter values.\n"
                                   "Axis and colorbar tick labels pick their own decimals.")
        f.addRow("Significant digits:", self.sig_digits)
        self.sci_notation = _combo([("Automatic (below 0.01 or from 100,000)", "auto"), ("Always", "always"),
                                    ("Never", "never")], s.get("sci_notation", "auto"), 260)
        self.sci_notation.setToolTip("When numbers are written with a power of ten -- in titles, legends\n"
                                     "and the axis and colorbar tick labels.")
        f.addRow("Scientific notation:", self.sci_notation)
        self.sci_style = _combo([("1.23 × 10⁻⁴", "times"), ("1.23e-04", "e")], s.get("sci_style", "times"), 120)
        f.addRow("Written as:", self.sci_style)

    def _surface(self):
        s = self.style
        f = self._page("Surface")
        self.cmap = _edit_combo(COLORMAPS, s.get("colormap", "jet"), 120)
        f.addRow("Colour map:", self.cmap)
        _ec = s.get("err_colormap", "") or ""
        self.err_cmap = _edit_combo([("Same as the solution", "")] + [(c, c) for c in COLORMAPS]
                                    + ([(_ec, _ec)] if _ec and _ec not in COLORMAPS else []), _ec, 150)
        self.err_cmap.setToolTip("Colour map of the |Error| panels of Error Analysis. The error is never\n"
                                 "negative, so a one-sided map (viridis, magma, ...) often reads better.")
        f.addRow("Error colour map:", self.err_cmap)
        self.levels = _spin(5, 200, int(s.get("levels", 50)), width=80)
        f.addRow("Contour levels:", self.levels)
        self.res = _edit_combo(["80", "100", "150", "200", "300"], str(s.get("resolution", 100)), 80)
        f.addRow("Grid resolution:", self.res)
        self.res3d = _spin(10, 200, int(s.get("resolution_3d", 40) or 40), 5, False, 80)
        self.res3d.setToolTip("Points per side on each face of a 3D surface (plots, animations, Error Analysis).\n"
                              "Above about 60 the 3D figures get slow to draw.")
        f.addRow("3D grid resolution:", self.res3d)
        self.auto_range = QCheckBox("Automatic")
        self.auto_range.setChecked(bool(s.get("auto_range", True)))
        from pinnstudio.ui.sci_spinbox import SciSpinBox
        self.vmin, self.vmax = SciSpinBox(), SciSpinBox()
        for _b, _v in ((self.vmin, s.get("vmin", -1.0)), (self.vmax, s.get("vmax", 1.0))):
            _b.setRange(-1e12, 1e12)
            _b.setValue(float(_v))
            _b.setFixedWidth(90)
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
        self.lw_line = _edit_combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(float(s.get("linewidth", 2.0))), 80)
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
                sp = _spin(-1e6, 1e6, 0.0, None, True, 90)
                sp.setDecimals(4)
                sp.setValue((rng[0] + rng[1]) / 2.0 if auto else val)
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(cb)
                rl.addWidget(sp)
                cb.toggled.connect(lambda on, sp=sp: sp.setEnabled(not on))
                sp.setEnabled(not auto)
                f.addRow(f"{ax} =", row)
                self.slices[ax] = (cb, sp)
        self.ref_style = _combo([("Line", "line"), ("Dots", "dots")], s.get("ref_style", "line"), 90)
        self.ref_style.setToolTip("How the reference is drawn in the line comparisons of Error Analysis:\n"
                                  "a line, or dots (the PINN stays a dashed line).")
        f.addRow("Reference in line comparisons:", self.ref_style)
        self.pinn_color = _combo(LINE_COLORS, s.get("pinn_color", "#ff6b6b"), 120)
        self.ref_color = _combo(LINE_COLORS, s.get("ref_color", "#4dabf7"), 120)
        f.addRow("PINN line colour:", self.pinn_color)
        f.addRow("Reference line colour:", self.ref_color)
        self.ea_cols = _spin(1, 12, int(s.get("ea_cols", 4)), width=70)
        self.ea_cols.setToolTip("Panels per row in Error Analysis figures with one panel per reference time")
        f.addRow("Error Analysis panels per row:", self.ea_cols)
        self._param_plot_section(f)

    def _animation(self):
        s = self.style
        f = self._page("Animation")
        v, mx = self._steps(True)
        self.frames = _spin(2, mx, v, width=80)
        f.addRow("Frames:", self.frames)
        self.fps = _edit_combo(["5", "8", "10", "15", "20", "24", "30"], str(int(s.get("fps", 10))), 80)
        f.addRow("Frame rate (fps):", self.fps)
        self.dpi_anim = _edit_combo(["72", "100", "150", "200", "300"], str(int(s.get("dpi_anim", 150))), 80)
        self.dpi_anim.setToolTip("Resolution of every saved animation (solution and parameter convergence)")
        f.addRow("Resolution (DPI):", self.dpi_anim)
        self.lw_anim = _edit_combo(["1.0", "1.5", "2.0", "2.5", "3.0", "4.0"], str(float(s.get("linewidth_anim", 3.0))), 80)
        f.addRow("Line width:", self.lw_anim)
        if self.dim in ("2D", "3D") and not self.steady:
            f.addRow(_hint("Surface animations use the Surface tab's colour map, levels and colour range."))
        self._param_anim_section(f)

    def _loss_tab(self):
        s = self.loss
        f = self._page("Loss")
        self.loss_mode = _combo([("Train + Test totals", "train_test"), ("Each loss term", "individual"),
                                 ("Totals and each term", "all")], s.get("mode", "train_test"), 190)
        f.addRow("Show:", self.loss_mode)
        self.loss_lw = _edit_combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(float(s.get("linewidth", 2.0))), 80)
        f.addRow("Line width:", self.loss_lw)
        self.loss_log = QCheckBox("Logarithmic")
        self.loss_log.setChecked(bool(s.get("log_y", True)))
        f.addRow("Loss axis:", self.loss_log)
        self.loss_every = _spin(1, 100000, int(s.get("display_every", 1000)), width=90)
        f.addRow("Record every:", self.loss_every)
        f.addRow(_hint("Record every: how often (in iterations) the loss is recorded and printed during training. "
                       "DeepXDE keeps one test loss, so each term is shown for the training loss only."))

    def _keep_hidden(self, *widgets):
        """Settings not shown (a forward problem): kept, so OK saves them as they are."""
        for wdg in widgets:
            wdg.setParent(self)
            wdg.hide()

    @staticmethod
    def _section(f, title):
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 600; padding-top: 10px;")
        f.addRow(lbl)

    def _ref_line(self, show_key, color_key, style_key, label, defaults):
        """A reference line's row: on/off, colour, line style."""
        s = self.style
        cb = QCheckBox(label)
        cb.setChecked(bool(s.get(show_key, True)))
        col = _combo(PARAM_COLORS, s.get(color_key, defaults[0]), 90)
        sty = _combo(LINE_STYLES, s.get(style_key, defaults[1]), 90)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(cb)
        rl.addWidget(col)
        rl.addWidget(sty)
        rl.addStretch()
        cb.toggled.connect(lambda on: (col.setEnabled(on), sty.setEnabled(on)))
        col.setEnabled(cb.isChecked())
        sty.setEnabled(cb.isChecked())
        return row, cb, col, sty

    def _param_plot_section(self, f):
        """Inverse problems: the parameter convergence plot (a parameter's
        value against the iteration), on the Line tab."""
        s = self.style
        self.p_color = _combo(PARAM_COLORS, s.get("param_color", "#e03131"), 100)
        self.p_lw = _edit_combo(["1.0", "1.5", "2.0", "2.5", "3.0", "4.0"], str(float(s.get("param_linewidth", 3.0))), 80)
        # "Mark each saved value" (older setting) shows as "Line and markers"
        _draw = s.get("param_draw", "line")
        if _draw == "line" and s.get("param_markers", False):
            _draw = "line_markers"
        self.p_draw = _combo(DRAW_AS, _draw, 150)
        self.p_marker = _combo(MARKERS, s.get("param_marker", "o"), 100)
        self.p_msize = _spin(1.0, 20.0, float(s.get("param_marker_size", 4.0)), 0.5, True, 70)
        self.p_figs = _combo([("One for all parameters", "combined"), ("One per parameter", "separate")], s.get("param_plot_figures", "combined"), 190)
        self.p_figs.setToolTip("One figure with a panel per parameter, or one figure per parameter as well\n"
                               "(<name>_convergence_plot.png)")
        self.p_log = QCheckBox("Logarithmic")
        self.p_log.setChecked(self.win.inv_param_log_scale.isChecked())
        true_row, self.p_true, self.p_true_color, self.p_true_style = self._ref_line(
            "param_show_true", "param_true_color", "param_true_style", "True value", ("#212529", "--"))
        final_row, self.p_final, self.p_final_color, self.p_final_style = self._ref_line(
            "param_show_final", "param_final_color", "param_final_style", "Final value", ("#868e96", ":"))
        # Always shown: the problem open here may be a forward one while the
        # plots are for an inverse model (Restore uses these settings too).
        self._section(f, "Parameter convergence plot (inverse problems)")
        f.addRow("Figures:", self.p_figs)
        f.addRow("Line colour:", self.p_color)
        f.addRow("Line width:", self.p_lw)
        f.addRow("Draw as:", self.p_draw)
        mrow = QWidget()
        ml = QHBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.p_marker)
        ml.addWidget(QLabel("size"))
        ml.addWidget(self.p_msize)
        ml.addStretch()
        f.addRow("Marker:", mrow)

        def _sync_marker(_i=None):
            mrow.setEnabled(self.p_draw.currentData() != "line" or self.pa_draw.currentData() != "line"
                            if hasattr(self, "pa_draw") else self.p_draw.currentData() != "line")
        self.p_draw.currentIndexChanged.connect(_sync_marker)
        self._sync_marker = _sync_marker
        _sync_marker()
        f.addRow("Value axis:", self.p_log)
        f.addRow("Reference lines:", true_row)
        f.addRow("", final_row)

    def _param_anim_section(self, f):
        """Inverse problems: the parameter convergence animation, on the
        Animation tab (frame rate as above; reference lines and log axis as
        the plot's, on the Line tab)."""
        s = self.style
        self.pa_color = _combo(PARAM_COLORS, s.get("param_anim_color", "#e03131"), 100)
        self.pa_lw = _edit_combo(["1.0", "1.5", "2.0", "2.5", "3.0", "4.0"],
                            str(float(s.get("param_anim_linewidth", 3.0))), 80)
        self.pa_frames = _spin(10, 300, int(s.get("param_anim_frames", 60)), width=80)
        self.pa_figs = _combo([("One for all parameters", "combined"), ("One per parameter", "separate")], s.get("param_anim_figures", "combined"), 190)
        self.pa_figs.setToolTip("One animation with a panel per parameter, or one per parameter as well\n"
                                "(<name>_convergence_animation.gif)")
        self.pa_draw = _combo(DRAW_AS, s.get("param_anim_draw", "line"), 150)
        self.pa_head = QCheckBox("Mark it with a dot")
        self.pa_head.setChecked(bool(s.get("param_anim_head", True)))
        self._section(f, "Parameter convergence animation (inverse problems)")
        f.addRow("Figures:", self.pa_figs)
        f.addRow("Draw as:", self.pa_draw)
        f.addRow("Line colour:", self.pa_color)
        f.addRow("Line width:", self.pa_lw)
        f.addRow("Current value:", self.pa_head)
        f.addRow("Frames (at most):", self.pa_frames)
        f.addRow(_hint("Markers, reference lines and the value axis are as for the plot (Line tab)."))
        self.pa_draw.currentIndexChanged.connect(lambda _i: self._sync_marker())
        self._sync_marker()

    def _hide_what_does_not_apply(self):
        """An ODE or a steady 1D problem is one curve: no surface, no time
        steps, no line animation (an inverse one keeps its parameter
        animation and frame rate)."""
        if not (self.dim == "ODE" or (self.dim == "1D" and self.steady)):
            return

        def hide(w):
            f = w.parentWidget().layout() if w.parentWidget() is not None else None
            if isinstance(f, QFormLayout) and hasattr(f, "setRowVisible"):
                f.setRowVisible(w, False)
        names = [self.tabs.tabText(i) for i in range(self.tabs.count())]
        self.tabs.setTabVisible(names.index("Surface"), False)
        hide(self.steps_line)
        hide(self.frames)
        hide(self.lw_anim)

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
            self.legend_fs.setValue(d.plot_legend_fontsize)
            self.grid_cb.setChecked(d.plot_grid)
            self.sig_digits.setValue(d.plot_sig_digits)
            self.sci_notation.setCurrentIndex(max(0, self.sci_notation.findData(d.plot_sci_notation)))
            self.sci_style.setCurrentIndex(max(0, self.sci_style.findData(d.plot_sci_style)))
        elif name == "Surface":
            self.cmap.setCurrentText(d.plot_colormap)
            self.levels.setValue(d.plot_levels)
            self.err_cmap.setCurrentIndex(0)
            self.res.setCurrentText(str(d.plot_resolution))
            self.res3d.setValue(d.plot_resolution_3d)
            self.auto_range.setChecked(d.plot_auto_range)
            self.colorbar.setChecked(d.plot_colorbar)
            self.snaps.setValue(d.plot_n_2d_snapshots)
            self.swap.setChecked(d.plot_swap_xt)
        elif name == "Line":
            self.lw_line.setCurrentText(str(float(d.plot_linewidth)))
            for cb, _sp in self.slices.values():
                cb.setChecked(True)
            self.p_color.setCurrentIndex(max(0, self.p_color.findData(d.param_plot_color)))
            self.p_lw.setCurrentText(str(float(d.param_plot_linewidth)))
            self.p_draw.setCurrentIndex(max(0, self.p_draw.findData(d.param_plot_draw)))
            self.p_marker.setCurrentIndex(max(0, self.p_marker.findData(d.param_plot_marker)))
            self.p_msize.setValue(d.param_plot_marker_size)
            self.ref_style.setCurrentIndex(max(0, self.ref_style.findData(d.plot_ref_style)))
            self.pinn_color.setCurrentIndex(max(0, self.pinn_color.findData(d.plot_pinn_color)))
            self.ref_color.setCurrentIndex(max(0, self.ref_color.findData(d.plot_ref_color)))
            self.ea_cols.setValue(d.plot_ea_cols)
            self.p_log.setChecked(d.inv_param_log_scale)
            self.p_figs.setCurrentIndex(max(0, self.p_figs.findData(d.param_plot_figures)))
            for cb, col, sty, show, c, st in ((self.p_true, self.p_true_color, self.p_true_style,
                                              d.param_plot_show_true, d.param_true_color, d.param_true_style),
                                             (self.p_final, self.p_final_color, self.p_final_style,
                                              d.param_plot_show_final, d.param_final_color, d.param_final_style)):
                cb.setChecked(show)
                col.setCurrentIndex(max(0, col.findData(c)))
                sty.setCurrentIndex(max(0, sty.findData(st)))
        elif name == "Animation":
            self.fps.setCurrentText(str(d.plot_fps))
            self.dpi_anim.setCurrentText(str(d.plot_dpi_anim))
            self.lw_anim.setCurrentText(str(float(d.plot_linewidth_anim)))
            self.pa_color.setCurrentIndex(max(0, self.pa_color.findData(d.param_anim_color)))
            self.pa_lw.setCurrentText(str(float(d.param_anim_linewidth)))
            self.pa_frames.setValue(d.param_anim_frames)
            self.pa_draw.setCurrentIndex(max(0, self.pa_draw.findData(d.param_anim_draw)))
            self.pa_head.setChecked(d.param_anim_head)
            self.pa_figs.setCurrentIndex(max(0, self.pa_figs.findData(d.param_anim_figures)))
        elif name == "Loss":
            self.loss_mode.setCurrentIndex(self.loss_mode.findData(d.loss_plot_mode))
            self.loss_lw.setCurrentText(str(float(d.loss_plot_linewidth)))
            self.loss_log.setChecked(d.loss_plot_log_y)
            self.loss_every.setValue(d.loss_display_every)

    def _colormap(self):
        """The colormap typed or picked; a name matplotlib does not know
        keeps the one set before."""
        name = self.cmap.currentText().strip()
        try:
            import matplotlib
            matplotlib.colormaps[name]
            return name
        except Exception:
            return self.style.get("colormap", "jet")

    def _err_colormap(self):
        """"" (the solution's) or a colormap matplotlib knows."""
        data = self.err_cmap.currentData()
        text = self.err_cmap.currentText().strip()
        if data == "" and text == "Same as the solution":
            return ""
        try:
            import matplotlib
            matplotlib.colormaps[text]
            return text
        except Exception:
            return self.style.get("err_colormap", "")

    def values(self):
        s = dict(self.style)
        s.update({
            "dpi": _num(self.dpi, int, int(self.style.get("dpi", 300))),
            "figsize_mode": self.figsize.currentText(),
            "figsize_w": self.fig_w.value(), "figsize_h": self.fig_h.value(),
            "title": self.title.text().strip(), "xlabel": self.xlabel.text().strip(),
            "ylabel": self.ylabel.text().strip(),
            "title_fontsize": self.title_fs.value(), "subplot_title_fontsize": self.sub_fs.value(),
            "axis_label_fontsize": self.axis_fs.value(), "legend_fontsize": self.legend_fs.value(),
            "resolution_3d": self.res3d.value(),
            "colormap": self._colormap(), "levels": self.levels.value(),
            "resolution": _num(self.res, int, int(self.style.get("resolution", 200))),
            "auto_range": self.auto_range.isChecked(),
            "vmin": self.vmin.value(), "vmax": self.vmax.value(), "colorbar": self.colorbar.isChecked(),
            "n_2d_snapshots": self.snaps.value(), "swap_xt": self.swap.isChecked(),
            "linewidth": _num(self.lw_line, float, float(self.style.get("linewidth", 2.0))),
            "linewidth_anim": _num(self.lw_anim, float, float(self.style.get("linewidth_anim", 3.0))),
            "fps": _num(self.fps, int, int(self.style.get("fps", 10))),
            "param_color": self.p_color.currentData(), "param_linewidth": _num(self.p_lw, float, float(self.style.get("param_linewidth", 3.0))),
            "param_markers": self.p_draw.currentData() != "line",
            "param_draw": self.p_draw.currentData(), "param_marker": self.p_marker.currentData(),
            "param_marker_size": self.p_msize.value(),
            "param_anim_draw": self.pa_draw.currentData(), "param_anim_head": self.pa_head.isChecked(),
            "grid": self.grid_cb.isChecked(), "ref_style": self.ref_style.currentData(),
            "sig_digits": self.sig_digits.value(), "sci_notation": self.sci_notation.currentData(),
            "pinn_color": self.pinn_color.currentData(), "ref_color": self.ref_color.currentData(),
            "err_colormap": self._err_colormap(), "ea_cols": self.ea_cols.value(),
            "dpi_anim": _num(self.dpi_anim, int, int(self.style.get("dpi_anim", 150))),
            "sci_style": self.sci_style.currentData(),
            "param_show_true": self.p_true.isChecked(), "param_true_color": self.p_true_color.currentData(),
            "param_true_style": self.p_true_style.currentData(),
            "param_show_final": self.p_final.isChecked(), "param_final_color": self.p_final_color.currentData(),
            "param_final_style": self.p_final_style.currentData(),
            "param_anim_color": self.pa_color.currentData(), "param_anim_linewidth": _num(self.pa_lw, float, float(self.style.get("param_anim_linewidth", 3.0))),
            "param_anim_frames": self.pa_frames.value(),
            "param_plot_figures": self.p_figs.currentData(), "param_anim_figures": self.pa_figs.currentData(),
        })
        loss = {"mode": self.loss_mode.currentData(), "display_every": self.loss_every.value(),
                "linewidth": _num(self.loss_lw, float, 2.0), "log_y": self.loss_log.isChecked()}
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
                if not cb.isChecked():   # the middle is worked out at plot time
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
                if not cb.isChecked():   # with "middle" ticked the value isn't used;
                    # writing the shown middle in marked every 2D/3D problem changed
                    getattr(w, f"line_slice_{ax}_spin").setValue(sp.value())
        try:
            w.log_box.append("✅ Plot settings saved")
        except Exception:
            pass
        self.accept()


def open_figure_style(win, viz_type=None, context="solve", tab=None):
    dlg = FigureStyleDialog(win, context, tab or tab_for(viz_type))
    return dlg.exec()
