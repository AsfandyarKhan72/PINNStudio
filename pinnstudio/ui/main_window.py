import sys
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QTextEdit, QGroupBox, QComboBox, QSplitter, QLineEdit,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QDialog, QMenuBar, QMenu, QFrame, QApplication, QColorDialog,
    QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont, QAction, QColor
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.runner import run_pinn
REFERENCE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reference_data",
)


class SciLineEdit(QLineEdit):
    """QLineEdit that accepts scientific notation like 1e-5, 1.5e3."""
    def __init__(self, value=1.0, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self.setText(self._format(value))
        self.editingFinished.connect(self._on_edited)
        self.setFixedHeight(28)

    def _format(self, v):
        v = float(v)
        if v == 0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 1e6:
            return f"{v:.2e}"
        return f"{v:g}"

    def _on_edited(self):
        try:
            self._value = float(self.text().strip())
            self.setText(self._format(self._value))
            self.setStyleSheet("")
        except ValueError:
            self.setStyleSheet("border: 1px solid red;")

    def value(self):
        try:
            return float(self.text().strip())
        except ValueError:
            return self._value

    def setValue(self, v):
        self._value = float(v)
        self.setText(self._format(v))
        self.setStyleSheet("")


class InitGuessLineEdit(SciLineEdit):
    """SciLineEdit variant for Inverse trainable-variable initial guesses.
    SciLineEdit's own _format() (via Python's %g) shows a whole number
    like 1.0 as a bare "1" -- fine for loss-weight boxes, but for an
    initial-guess box "1.0" reads more clearly as a float. This just
    appends ".0" whenever the formatted text has no decimal point and
    isn't in scientific notation; the underlying value is still a plain
    float with no precision or range restriction, so any small or large
    number can still be typed exactly as before."""
    def _format(self, v):
        s = super()._format(v)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s


# ── Background worker thread ─────────────────────────────────
class SolverThread(QThread):
    output_signal = pyqtSignal(str)
    done_signal   = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process = None

    def run(self):
        result = run_pinn(
            self.config,
            on_output=lambda line: self.output_signal.emit(line),
            set_process=self._set_process
        )
        self.done_signal.emit(result)

    def _set_process(self, proc):
        self.process = proc

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()


# --- Update-check background thread ---
class UpdateCheckThread(QThread):
    """Fetches the latest published version from PyPI without blocking the GUI."""
    result_signal = pyqtSignal(str)

    def run(self):
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                "https://pypi.org/pypi/pinnstudio/json",
                headers={"User-Agent": "PINNStudio-update-check"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", "")
            self.result_signal.emit(latest)
        except Exception:
            self.result_signal.emit("")


# ── Main Window ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINNStudio — No-Code GUI for Physics-Informed Neural Networks (PINNs)")
        self.setMinimumSize(1100, 750)
        self._font_size = 18
        self._log_font_size = 18
        self._theme = "Solarized Dark"
        self._accent = "Green (#69db7c)"
        self._float_type = "float32"
        # ── Per-category display settings (font family/size/weight/color for
        # groups of related text: the app title, section headers, field
        # labels, small hint/note text, buttons, and the log console). Each
        # category's 'color' is '' by default, meaning "keep that widget's
        # own original color" -- setting one explicitly overrides it
        # uniformly across the category. self._styled_widgets is the
        # registry of (widget, category, style_fn) used to re-apply these
        # live whenever settings change, without needing to rebuild the UI.
        self._DISP_CATEGORY_DEFAULTS = {
            "title":          {"family": "Arial",      "size": 26, "bold": True,  "color": "#ffffff"},
            "section_header": {"family": "Arial",      "size": 18, "bold": True,  "color": "#ffffff"},
            "field_label":    {"family": "Segoe UI",    "size": 16, "bold": False, "color": "#ffffff"},
            "hint":           {"family": "Segoe UI",    "size": 16, "bold": False, "color": "#ffffff"},
            "button":         {"family": "Segoe UI",    "size": 16, "bold": False, "color": "#ffffff"},
            "log_console":    {"family": "Arial",       "size": 16, "bold": False, "color": "#ffffff"},
        }
        self._disp = {k: dict(v) for k, v in self._DISP_CATEGORY_DEFAULTS.items()}
        self._styled_widgets = []
        self._display_settings_path = os.path.join(
            os.path.expanduser("~"), ".pinnstudio", "display_settings.json")
        self._load_display_settings_from_disk()

        # ── Optimizer / training-callback settings ─────────────
        # Same pattern as self._plot_viz_settings: a plain dict edited via a
        # menu-launched dialog (Settings > Optimizer Settings / Training
        # Callbacks), read directly by _build_config(). weight_decay applies
        # to whichever network is trained (see codegen.py) -- it is 0.0 (off)
        # by default, matching all pre-existing behavior exactly.
        self._optimizer_settings = {
            "weight_decay": 0.0,
        }
        # Every callback is off by default -- matches all pre-existing
        # behavior exactly until the user opts in via the dialog.
        self._callback_settings = {
            "early_stopping": False,
            "early_stopping_min_delta": 0.0,
            "early_stopping_patience": 2000,
            "early_stopping_baseline": "",
            "early_stopping_monitor": "loss_train",
            "early_stopping_start_from": 0,
            "point_resampler": False,
            "point_resampler_period": 100,
            "point_resampler_pde_points": True,
            "point_resampler_bc_points": False,
            "model_checkpoint": False,
            "checkpoint_period": 1000,
            "checkpoint_save_better_only": True,
            "checkpoint_monitor": "train loss",
            "timer": False,
            "timer_minutes": 60.0,
        }
        self._apply_theme()
        self._build_ui()
        self._apply_display_settings()
        self._check_for_updates()

    # ── Display-settings persistence ──────────────────────────
    def _load_display_settings_from_disk(self):
        """Best-effort load of ~/.pinnstudio/display_settings.json, written
        by _save_display_settings_to_disk(). Missing/corrupt file, or a
        file from an older version missing some keys, just falls back to
        the built-in defaults for whatever isn't present -- never blocks
        startup."""
        try:
            import json
            if not os.path.isfile(self._display_settings_path):
                return
            with open(self._display_settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        try:
            self._font_size = int(data.get("font_size", self._font_size))
            self._log_font_size = int(data.get("log_font_size", self._log_font_size))
            self._theme = data.get("theme", self._theme)
            self._accent = data.get("accent", self._accent)
            cats = data.get("categories", {})
            for key, defaults in self._DISP_CATEGORY_DEFAULTS.items():
                saved = cats.get(key, {})
                if not isinstance(saved, dict):
                    continue
                entry = self._disp.setdefault(key, dict(defaults))
                entry["family"] = str(saved.get("family", entry["family"]))
                entry["size"] = int(saved.get("size", entry["size"]))
                entry["bold"] = bool(saved.get("bold", entry["bold"]))
                entry["color"] = str(saved.get("color", entry["color"]))
        except Exception:
            pass  # any malformed field -- keep whatever defaults survived

    def _save_display_settings_to_disk(self):
        """Best-effort save -- persistence is a convenience, never load-
        bearing for the app to function, so failures (e.g. read-only home
        directory) are silently ignored rather than shown as an error."""
        try:
            import json
            os.makedirs(os.path.dirname(self._display_settings_path), exist_ok=True)
            payload = {
                "font_size": self._font_size,
                "log_font_size": self._log_font_size,
                "theme": self._theme,
                "accent": self._accent,
                "categories": self._disp,
            }
            with open(self._display_settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    # ── Per-category style registry ───────────────────────────
    def _cat_css(self, category):
        """Returns a 'font-family: ...; font-size: ...px; font-weight: ...;'
        CSS fragment for the given category, plus 'color: ...;' if that
        category has an explicit color override set (blank/unset means
        "leave each widget's own default color alone")."""
        defaults = self._DISP_CATEGORY_DEFAULTS.get(category, {})
        cat = self._disp.get(category, defaults)
        family = cat.get("family") or defaults.get("family", "Segoe UI")
        size = cat.get("size") or defaults.get("size", 12)
        weight = "bold" if cat.get("bold") else "normal"
        parts = [f"font-family: '{family}';", f"font-size: {size}px;", f"font-weight: {weight};"]
        color = (cat.get("color") or "").strip()
        if color:
            parts.append(f"color: {color};")
        return " ".join(parts)

    def _register_style(self, widget, category, style_fn):
        """Registers `widget` as styled by `category` and applies it
        immediately. `style_fn(css_fragment)` must return the full
        stylesheet string to apply to `widget`, where css_fragment is the
        current _cat_css(category) output -- used at every call site that
        used to hardcode a font-size (and often color) directly, so those
        labels/buttons respond to Display Settings instead of being pinned.
        Returns widget, so this can be chained inline where a plain
        .setStyleSheet(...) call used to sit."""
        self._styled_widgets.append((widget, category, style_fn))
        self._restyle_one(widget, category, style_fn)
        return widget

    def _restyle_one(self, widget, category, style_fn):
        try:
            widget.setStyleSheet(style_fn(self._cat_css(category)))
        except RuntimeError:
            pass  # underlying Qt widget already deleted

    def _restyle_all(self):
        """Re-applies every registered widget's stylesheet from the current
        category settings. Called whenever Display Settings changes
        (Preview/OK) and once at startup. Silently drops any widget whose
        underlying C++ object has since been deleted (e.g. a removed row)."""
        dead = []
        for widget, category, style_fn in self._styled_widgets:
            try:
                widget.setStyleSheet(style_fn(self._cat_css(category)))
            except RuntimeError:
                dead.append((widget, category, style_fn))
        for entry in dead:
            self._styled_widgets.remove(entry)

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #002b36; }
            QWidget { background: #002b36; color: #e0e0e0; font-family: 'Segoe UI', Arial; font-size: 16px; }
            QGroupBox {
                border: 1px solid #c8d2d8;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: #a0c4ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 2px 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #a0c4ff;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background: #586e75;
                border: none;
                width: 16px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QComboBox::drop-down {
                background: #586e75;
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ffffff;
                width: 0px; height: 0px;
            }
            QCheckBox { color: #c0c0c0; spacing: 6px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background: #252526;
            }
            QCheckBox::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QRadioButton { color: #c0c0c0; spacing: 6px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background: #252526;
            }
            QRadioButton::indicator:checked { background: #a0c4ff; border-color: #a0c4ff; }
            QLabel { color: #c0c0c0; }
            QScrollArea { border: none; background: #1e1e1e; }
            QScrollBar:vertical {
                background: #252526; width: 14px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #586e75; border-radius: 6px; min-height: 30px;
            }

            QTextEdit {
                background: #0f0f23;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                color: #a0ffb0;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
            QSplitter::handle { background: #3e3e42; width: 2px; }
            QMenuBar { background: #252526; color: #c0c0c0; border-bottom: 1px solid #3e3e42; }
            QMenuBar::item:selected { background: #3e3e42; }
            QMenu { background: #252526; border: 1px solid #3e3e42; }
            QMenu::item:selected { background: #3e3e42; }
        """)

    def _check_for_updates(self):
        """Kick off a background check against PyPI for a newer PINNStudio release."""
        self._current_pkg_version = self._get_installed_version()
        if not self._current_pkg_version:
            return
        self._update_thread = UpdateCheckThread()
        self._update_thread.result_signal.connect(self._on_update_check_result)
        self._update_thread.start()

    def _get_installed_version(self):
        try:
            from importlib.metadata import version as _pkg_version
            return _pkg_version("pinnstudio")
        except Exception:
            return None

    def _version_tuple(self, v):
        parts = []
        for p in v.split("."):
            digits = ""
            for ch in p:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    def _on_update_check_result(self, latest_version):
        if not latest_version or not getattr(self, "_current_pkg_version", None):
            return
        try:
            is_newer = self._version_tuple(latest_version) > self._version_tuple(self._current_pkg_version)
        except Exception:
            is_newer = False
        if is_newer:
            self.update_banner_label.setText(
                f"🔔  A newer version of PINNStudio is available: {latest_version}  (you have {self._current_pkg_version}).   Update with:  pip install --upgrade pinnstudio"
            )
            self.update_banner.setVisible(True)

    def _build_ui(self):
        from PyQt6.QtWidgets import QScrollArea

        # ── Menu bar ─────────────────────────────────────────
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        problem_menu = menubar.addMenu("📁 Problem")
        open_problem_action = QAction("Open Saved Problem...", self)
        open_problem_action.triggered.connect(self._open_problem)
        problem_menu.addAction(open_problem_action)
        save_problem_action = QAction("Save Problem As...", self)
        save_problem_action.triggered.connect(self._save_problem)
        problem_menu.addAction(save_problem_action)
        settings_menu = menubar.addMenu("⚙ Settings")
        lbfgs_action = QAction("L-BFGS Options", self)
        lbfgs_action.triggered.connect(self._on_lbfgs_settings)
        settings_menu.addAction(lbfgs_action)

        float_action = QAction("Float Precision...", self)
        float_action.triggered.connect(self._on_float_settings)
        settings_menu.addAction(float_action)

        optimizer_settings_action = QAction("Optimizer Settings...", self)
        optimizer_settings_action.triggered.connect(self._on_optimizer_settings)
        settings_menu.addAction(optimizer_settings_action)

        callbacks_action = QAction("Training Callbacks...", self)
        callbacks_action.triggered.connect(self._on_callback_settings)
        settings_menu.addAction(callbacks_action)

        view_menu = menubar.addMenu("🎨 Display")
        display_action = QAction("Display Settings...", self)
        display_action.triggered.connect(self._on_display_settings)
        view_menu.addAction(display_action)

        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.update_banner = QWidget()
        self.update_banner.setStyleSheet("background: #2aa198;")
        self.update_banner.setVisible(False)
        banner_layout = QHBoxLayout(self.update_banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        self.update_banner_label = QLabel("")
        self.update_banner_label.setStyleSheet("color: #002b36; font-weight: bold; background: transparent;")
        banner_layout.addWidget(self.update_banner_label)
        banner_layout.addStretch()
        update_dismiss_btn = QPushButton("✕")
        update_dismiss_btn.setFixedSize(24, 24)
        update_dismiss_btn.setStyleSheet("background: transparent; color: #002b36; border: none; font-weight: bold;")
        update_dismiss_btn.clicked.connect(lambda: self.update_banner.setVisible(False))
        banner_layout.addWidget(update_dismiss_btn)
        outer_layout.addWidget(self.update_banner)
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        outer_layout.addLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ───────────────────────────────────────
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 8, 10, 8)

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_inner)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(300)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Title -- styled from the "title" Display Settings category (family/
        # size/weight/color) instead of a hardcoded QFont(), which is what
        # previously made it (and the subtitle below) immune to the Display
        # Settings font-size control entirely.
        title = QLabel("PINNStudio")
        self._register_style(title, "title", lambda css: f"color: #a0c4ff; margin-bottom: 2px; {css}")
        left_layout.addWidget(title)

        def _subtitle_style_fn(_css):
            t = self._disp.get("title", self._DISP_CATEGORY_DEFAULTS["title"])
            fam = t.get("family") or "Arial"
            size = max(9, round((t.get("size") or 16) * 0.69))
            color = (t.get("color") or "").strip() or "#7070a0"
            return (f"color: {color}; margin-bottom: 6px; "
                    f"font-family: '{fam}'; font-size: {size}px; font-weight: normal;")
        subtitle = QLabel("Physics-Informed Neural Network Solver")
        self._register_style(subtitle, "title", _subtitle_style_fn)
        left_layout.addWidget(subtitle)

        # ── Dimension selector ────────────────────────────────
        dim_group = QGroupBox("Problem Dimension")
        dim_layout = QHBoxLayout(dim_group)
        self.radio_1d = QRadioButton("1D  (x, t)")
        self.radio_2d = QRadioButton("2D  (x, y, t)")
        self.radio_3d = QRadioButton("3D  (x, y, z, t)")
        self.radio_1d.setChecked(True)
        self.dim_group_btn = QButtonGroup()
        self.dim_group_btn.addButton(self.radio_1d)
        self.dim_group_btn.addButton(self.radio_2d)
        self.dim_group_btn.addButton(self.radio_3d)
        dim_layout.addWidget(self.radio_1d)
        dim_layout.addWidget(self.radio_2d)
        dim_layout.addWidget(self.radio_3d)
        left_layout.addWidget(dim_group)
        self.radio_1d.toggled.connect(self._on_dim_changed)
        self.radio_2d.toggled.connect(self._on_dim_changed)
        self.radio_3d.toggled.connect(self._on_dim_changed)

        # ── Quick Examples (shown right after dimension) ──────
        examples_group = QGroupBox("📋 Quick Examples")
        examples_layout = QHBoxLayout(examples_group)
        examples_layout.addWidget(QLabel("Load example:"))
        self.quick_examples_combo = QComboBox()
        self.quick_examples_combo.addItems(["None", "1D Heat", "1D Allen-Cahn"])
        self.quick_examples_combo.setFixedHeight(28)
        self.quick_examples_combo.currentTextChanged.connect(self._on_quick_example_selected)
        examples_layout.addWidget(self.quick_examples_combo)
        left_layout.addWidget(examples_group)

        # ── Problem type ──────────────────────────────────────
        type_group = QGroupBox("Problem Type")
        type_layout = QHBoxLayout(type_group)
        self.radio_forward = QRadioButton("Forward")
        self.radio_inverse = QRadioButton("Inverse")
        self.radio_forward.setChecked(True)
        self.problem_type_group = QButtonGroup()
        self.problem_type_group.addButton(self.radio_forward)
        self.problem_type_group.addButton(self.radio_inverse)
        type_layout.addWidget(self.radio_forward)
        type_layout.addWidget(self.radio_inverse)
        left_layout.addWidget(type_group)
        self.radio_forward.toggled.connect(self._on_problem_type_changed)

        # ── Number of PDEs ────────────────────────────────────
        nout_group = QGroupBox("Number of PDEs / Outputs")
        nout_layout = QHBoxLayout(nout_group)
        nout_layout.addWidget(QLabel("How many PDEs:"))
        self.num_outputs_spin = QSpinBox()
        self.num_outputs_spin.setRange(1, 4)
        self.num_outputs_spin.setValue(1)
        self.num_outputs_spin.setFixedHeight(30)
        self.num_outputs_spin.setFixedWidth(60)
        self.num_outputs_spin.valueChanged.connect(self._on_num_outputs_changed)
        nout_layout.addWidget(self.num_outputs_spin)
        nout_layout.addStretch()
        left_layout.addWidget(nout_group)

        # ── PDE input ─────────────────────────────────────────
        self.pde_group = QGroupBox("PDE Definition")
        self.pde_main_layout = QVBoxLayout(self.pde_group)
        self.pde_main_layout.setSpacing(6)
        self.pde_inputs = []
        self.output_name_inputs = []
        self._build_pde_inputs(1)
        left_layout.addWidget(self.pde_group)

        # ── Domain ────────────────────────────────────────────
        # ── Geometry type (2D / 3D only) -- shown above Domain so the ──
        # ── shape choice governs what Domain asks for below it. ────────
        self.geom_type_group = QGroupBox("Geometry Type")
        geom_type_layout = QVBoxLayout(self.geom_type_group)
        geom_type_layout.setSpacing(6)

        geom_combo_row = QHBoxLayout()
        geom_combo_row.addWidget(QLabel("Shape:"))
        self.geometry_type_combo = QComboBox()
        self.geometry_type_combo.setFixedHeight(28)
        geom_combo_row.addWidget(self.geometry_type_combo)
        geom_type_layout.addLayout(geom_combo_row)
        self.geometry_type_combo.currentTextChanged.connect(self._on_geometry_type_changed)

        self.geom_type_group.setVisible(False)
        left_layout.addWidget(self.geom_type_group)

        domain_group = QGroupBox("Domain")
        domain_layout = QVBoxLayout(domain_group)
        domain_layout.setSpacing(6)

        # x/y/z rows: only meaningful for the box shapes (Interval/Rectangle/
        # Cuboid), where the domain really is an axis-aligned box. For every
        # other shape these are replaced by that shape's own parameters
        # (center + radius, vertices, ...) below -- see
        # _update_domain_fields_visibility().
        self.x_row_widget = QWidget()
        x_row = QHBoxLayout(self.x_row_widget)
        x_row.setContentsMargins(0, 0, 0, 0)
        x_row.addWidget(QLabel("x:"))
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1e6, 1e6); self.x_min.setValue(0.0); self.x_min.setSingleStep(0.5)
        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1e6, 1e6); self.x_max.setValue(1.0); self.x_max.setSingleStep(0.5)
        x_row.addWidget(self.x_min); x_row.addWidget(QLabel("to")); x_row.addWidget(self.x_max)
        domain_layout.addWidget(self.x_row_widget)

        self.y_row_widget = QWidget()
        y_row = QHBoxLayout(self.y_row_widget)
        y_row.setContentsMargins(0, 0, 0, 0)
        y_row.addWidget(QLabel("y:"))
        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(-1e6, 1e6); self.y_min.setValue(0.0); self.y_min.setSingleStep(0.5)
        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(-1e6, 1e6); self.y_max.setValue(1.0); self.y_max.setSingleStep(0.5)
        y_row.addWidget(self.y_min); y_row.addWidget(QLabel("to")); y_row.addWidget(self.y_max)
        self.y_row_widget.setVisible(False)
        domain_layout.addWidget(self.y_row_widget)

        self.z_row_widget = QWidget()
        z_row = QHBoxLayout(self.z_row_widget)
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.addWidget(QLabel("z:"))
        self.z_min = QDoubleSpinBox()
        self.z_min.setRange(-1e6, 1e6); self.z_min.setValue(0.0); self.z_min.setSingleStep(0.5)
        self.z_max = QDoubleSpinBox()
        self.z_max.setRange(-1e6, 1e6); self.z_max.setValue(1.0); self.z_max.setSingleStep(0.5)
        z_row.addWidget(self.z_min); z_row.addWidget(QLabel("to")); z_row.addWidget(self.z_max)
        self.z_row_widget.setVisible(False)
        domain_layout.addWidget(self.z_row_widget)

        # -- Disk panel: center (x, y) + radius --
        self.geom_panel_disk = QWidget()
        _p = QHBoxLayout(self.geom_panel_disk); _p.setContentsMargins(0, 0, 0, 0)
        _p.addWidget(QLabel("center x,y:"))
        self.geom_disk_cx = QDoubleSpinBox(); self.geom_disk_cx.setRange(-1e6, 1e6); self.geom_disk_cx.setValue(0.5); self.geom_disk_cx.setSingleStep(0.1)
        self.geom_disk_cy = QDoubleSpinBox(); self.geom_disk_cy.setRange(-1e6, 1e6); self.geom_disk_cy.setValue(0.5); self.geom_disk_cy.setSingleStep(0.1)
        _p.addWidget(self.geom_disk_cx); _p.addWidget(self.geom_disk_cy)
        _p.addWidget(QLabel("radius:"))
        self.geom_disk_r = QDoubleSpinBox(); self.geom_disk_r.setRange(1e-6, 1e6); self.geom_disk_r.setValue(0.5); self.geom_disk_r.setSingleStep(0.1)
        _p.addWidget(self.geom_disk_r)
        self.geom_panel_disk.setVisible(False)
        domain_layout.addWidget(self.geom_panel_disk)

        # -- Ellipse panel: center (x, y) + semi-major/minor + angle --
        self.geom_panel_ellipse = QWidget()
        _p = QVBoxLayout(self.geom_panel_ellipse); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(4)
        _row_a = QHBoxLayout()
        _row_a.addWidget(QLabel("center x,y:"))
        self.geom_ellipse_cx = QDoubleSpinBox(); self.geom_ellipse_cx.setRange(-1e6, 1e6); self.geom_ellipse_cx.setValue(0.5); self.geom_ellipse_cx.setSingleStep(0.1)
        self.geom_ellipse_cy = QDoubleSpinBox(); self.geom_ellipse_cy.setRange(-1e6, 1e6); self.geom_ellipse_cy.setValue(0.5); self.geom_ellipse_cy.setSingleStep(0.1)
        _row_a.addWidget(self.geom_ellipse_cx); _row_a.addWidget(self.geom_ellipse_cy)
        _p.addLayout(_row_a)
        _row_b = QHBoxLayout()
        _row_b.addWidget(QLabel("semi-major, semi-minor:"))
        self.geom_ellipse_a = QDoubleSpinBox(); self.geom_ellipse_a.setRange(1e-6, 1e6); self.geom_ellipse_a.setValue(0.5); self.geom_ellipse_a.setSingleStep(0.1)
        self.geom_ellipse_b = QDoubleSpinBox(); self.geom_ellipse_b.setRange(1e-6, 1e6); self.geom_ellipse_b.setValue(0.3); self.geom_ellipse_b.setSingleStep(0.1)
        _row_b.addWidget(self.geom_ellipse_a); _row_b.addWidget(self.geom_ellipse_b)
        _p.addLayout(_row_b)
        _row_c = QHBoxLayout()
        _row_c.addWidget(QLabel("angle (rad):"))
        self.geom_ellipse_angle = QDoubleSpinBox(); self.geom_ellipse_angle.setRange(-100, 100); self.geom_ellipse_angle.setValue(0.0); self.geom_ellipse_angle.setSingleStep(0.1)
        _row_c.addWidget(self.geom_ellipse_angle)
        _p.addLayout(_row_c)
        self.geom_panel_ellipse.setVisible(False)
        domain_layout.addWidget(self.geom_panel_ellipse)

        # -- Triangle panel: vertices as "x1,y1;x2,y2;x3,y3" --
        self.geom_panel_triangle = QWidget()
        _p = QVBoxLayout(self.geom_panel_triangle); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(2)
        _p.addWidget(QLabel("vertices (x1,y1;x2,y2;x3,y3):"))
        self.geom_triangle_verts_input = QLineEdit("0,0;1,0;0,1")
        self.geom_triangle_verts_input.setFixedHeight(26)
        _p.addWidget(self.geom_triangle_verts_input)
        self.geom_panel_triangle.setVisible(False)
        domain_layout.addWidget(self.geom_panel_triangle)

        # -- Polygon panel: vertices as "x1,y1;x2,y2;...;xn,yn" --
        self.geom_panel_polygon = QWidget()
        _p = QVBoxLayout(self.geom_panel_polygon); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(2)
        _p.addWidget(QLabel("vertices (x1,y1;x2,y2;...), any count ≥ 3:"))
        self.geom_polygon_verts_input = QLineEdit("0,0;1,0;1,1;0,1")
        self.geom_polygon_verts_input.setFixedHeight(26)
        _p.addWidget(self.geom_polygon_verts_input)
        self.geom_panel_polygon.setVisible(False)
        domain_layout.addWidget(self.geom_panel_polygon)

        # -- Sphere panel: center (x, y, z) + radius --
        self.geom_panel_sphere = QWidget()
        _p = QHBoxLayout(self.geom_panel_sphere); _p.setContentsMargins(0, 0, 0, 0)
        _p.addWidget(QLabel("center x,y,z:"))
        self.geom_sphere_cx = QDoubleSpinBox(); self.geom_sphere_cx.setRange(-1e6, 1e6); self.geom_sphere_cx.setValue(0.5); self.geom_sphere_cx.setSingleStep(0.1)
        self.geom_sphere_cy = QDoubleSpinBox(); self.geom_sphere_cy.setRange(-1e6, 1e6); self.geom_sphere_cy.setValue(0.5); self.geom_sphere_cy.setSingleStep(0.1)
        self.geom_sphere_cz = QDoubleSpinBox(); self.geom_sphere_cz.setRange(-1e6, 1e6); self.geom_sphere_cz.setValue(0.5); self.geom_sphere_cz.setSingleStep(0.1)
        _p.addWidget(self.geom_sphere_cx); _p.addWidget(self.geom_sphere_cy); _p.addWidget(self.geom_sphere_cz)
        _p.addWidget(QLabel("radius:"))
        self.geom_sphere_r = QDoubleSpinBox(); self.geom_sphere_r.setRange(1e-6, 1e6); self.geom_sphere_r.setValue(0.5); self.geom_sphere_r.setSingleStep(0.1)
        _p.addWidget(self.geom_sphere_r)
        self.geom_panel_sphere.setVisible(False)
        domain_layout.addWidget(self.geom_panel_sphere)

        self._geom_shape_panels = [
            self.geom_panel_disk, self.geom_panel_ellipse, self.geom_panel_triangle,
            self.geom_panel_polygon, self.geom_panel_sphere,
        ]
        self.geom_triangle_verts_input.textChanged.connect(self._on_geom_vertices_changed)
        self.geom_polygon_verts_input.textChanged.connect(self._on_geom_vertices_changed)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("t:"))
        self.t_min = QDoubleSpinBox()
        self.t_min.setRange(0.0, 1e6); self.t_min.setValue(0.0); self.t_min.setSingleStep(0.5)
        self.t_max = QDoubleSpinBox()
        self.t_max.setRange(0.0, 1e6); self.t_max.setValue(1.0); self.t_max.setSingleStep(0.5)
        row2.addWidget(self.t_min); row2.addWidget(QLabel("to")); row2.addWidget(self.t_max)
        domain_layout.addLayout(row2)
        left_layout.addWidget(domain_group)

        # ── Collocation Points ────────────────────────────────
        points_group = QGroupBox("Collocation Points")
        points_layout = QVBoxLayout(points_group)
        points_layout.setSpacing(4)

        def _pts_row(label, default, min_v, max_v, step):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox()
            sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(default)
            sb.setFixedHeight(28); sb.setFixedWidth(100)
            row.addStretch(); row.addWidget(sb)
            points_layout.addLayout(row)
            return sb

        self.num_domain   = _pts_row("Domain points:",   2000, 100, 100000, 500)
        self.num_boundary = _pts_row("Boundary points:", 200,  10,  10000,  100)
        self.num_initial  = _pts_row("Initial points:",  200,  0,  10000,  100)
        self.num_test     = _pts_row("Test points:",     1000, 100, 50000,  500)

        pts_dist_row = QHBoxLayout()
        pts_dist_row.addWidget(QLabel("Point distribution:"))
        self.pts_dist_combo = QComboBox()
        self.pts_dist_combo.addItems(["Hammersley", "uniform", "Halton", "LHS", "Sobol", "pseudorandom"])
        self.pts_dist_combo.setFixedHeight(28)
        pts_dist_row.addStretch()
        pts_dist_row.addWidget(self.pts_dist_combo)
        points_layout.addLayout(pts_dist_row)

        self.view_domain_check = QCheckBox("View domain & point distribution")
        self.view_domain_check.setChecked(False)
        self.view_domain_check.setVisible(False)
        self.view_domain_check.stateChanged.connect(self._on_view_domain_changed)
        points_layout.addWidget(self.view_domain_check)
        left_layout.addWidget(points_group)

        # ── Boundary & Initial conditions ─────────────────────
        self.bc_group = QGroupBox("Initial Condition")
        self.bc_main_layout = QVBoxLayout(self.bc_group)
        self.bc_main_layout.setSpacing(4)
        self.bc_left_types = [];  self.bc_left_vals = [];   self.bc_left_active = [];  self.bc_left_deriv = []
        self.bc_right_types = []; self.bc_right_vals = [];  self.bc_right_active = []; self.bc_right_deriv = []
        self.bc_bottom_types = []; self.bc_bottom_vals = []; self.bc_bottom_active = []; self.bc_bottom_deriv = []
        self.bc_top_types = [];   self.bc_top_vals = [];    self.bc_top_active = [];   self.bc_top_deriv = []
        # Cuboid-only sides (3D)
        self.bc_front_types = []; self.bc_front_vals = [];  self.bc_front_active = []
        self.bc_back_types = [];  self.bc_back_vals = [];   self.bc_back_active = []
        # Single unified boundary group, per output (Disk/Ellipse/Sphere)
        self.bc_boundary_types = []; self.bc_boundary_vals = []; self.bc_boundary_active = []
        # Per-edge BC groups, per output (Triangle/Polygon) -- nested: [output][edge]
        self.bc_edge_types = []; self.bc_edge_vals = []; self.bc_edge_active = []
        self.ic_inputs = [];      self.ic_active = []
        self._2d_bc_widgets = []
        # Wrapper widgets (one per output) around the hardcoded per-shape BC
        # rows above -- hidden whenever a custom (non-template) problem is
        # active, so the flexible builder below takes over instead. Rebuilt
        # every _build_bc_inputs() call; see _update_bc_mode_visibility().
        self._shape_bc_wrappers = []
        self._build_bc_inputs(1)
        left_layout.addWidget(self.bc_group)

        # ── Boundary Conditions (unified panel) ────────────────
        # ONE list-of-entries panel, used whether or not a template is
        # selected. Each entry is any DeepXDE BC class, on any output, at
        # any location the user describes. When a Quick Example template is
        # selected, this list is auto-filled (in this exact same visual
        # style) with that template's BCs, shown read-only -- the legacy
        # per-side widgets that still actually drive codegen for templates
        # (self.bc_left_types etc.) are kept alive internally but never
        # shown; see _populate_locked_bc_entries_from_legacy(). When no
        # template is selected ("None"), the list starts empty and every
        # entry is fully editable/removable -- this is the only path that
        # affects the generated training script today (codegen for these
        # entries is still pending; see _build_custom_bc_json()'s docstring).
        self.custom_bc_group = QGroupBox("Boundary Conditions")
        self.custom_bc_main_layout = QVBoxLayout(self.custom_bc_group)
        self.custom_bc_main_layout.setSpacing(4)
        self.custom_bc_note = QLabel()
        self._register_style(self.custom_bc_note, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        self.custom_bc_note.setWordWrap(True)
        self.custom_bc_main_layout.addWidget(self.custom_bc_note)
        self.custom_bc_list_widget = QWidget()
        self.custom_bc_list_layout = QVBoxLayout(self.custom_bc_list_widget)
        self.custom_bc_list_layout.setSpacing(4)
        self.custom_bc_list_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_bc_main_layout.addWidget(self.custom_bc_list_widget)
        self.custom_bc_list = []
        self.add_custom_bc_btn = QPushButton("➕ Add Boundary Condition")
        self.add_custom_bc_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }"
            "QPushButton:disabled { color: #4a5a6a; border-color: #2a3a4a; }")
        def _on_add_custom_bc_clicked():
            self._add_custom_bc_entry()
            self._update_bc_mode_visibility()
        self.add_custom_bc_btn.clicked.connect(_on_add_custom_bc_clicked)
        self.custom_bc_main_layout.addWidget(self.add_custom_bc_btn)
        left_layout.addWidget(self.custom_bc_group)
        QTimer.singleShot(0, self._update_bc_mode_visibility)

        # ── Neural Network ────────────────────────────────────
        nn_group = QGroupBox("Neural Network")
        nn_layout = QVBoxLayout(nn_group)
        nn_layout.setSpacing(6)

        def _nn_row(label, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addStretch()
            row.addWidget(widget)
            nn_layout.addLayout(row)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 8); self.layers_spin.setValue(3)
        self.layers_spin.setFixedWidth(100); self.layers_spin.setFixedHeight(28)
        _nn_row("Hidden layers:", self.layers_spin)

        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(8, 512); self.neurons_spin.setSingleStep(8); self.neurons_spin.setValue(64)
        self.neurons_spin.setFixedWidth(100); self.neurons_spin.setFixedHeight(28)
        _nn_row("Neurons per layer:", self.neurons_spin)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["tanh", "relu", "sigmoid", "swish"])
        self.activation_combo.setFixedWidth(100); self.activation_combo.setFixedHeight(28)
        _nn_row("Activation:", self.activation_combo)

        self.kernel_init_combo = QComboBox()
        self.kernel_init_combo.addItems(
            ["Glorot uniform", "Glorot normal", "He uniform", "He normal", "zeros"])
        self.kernel_init_combo.setFixedWidth(130); self.kernel_init_combo.setFixedHeight(28)
        _nn_row("Kernel initializer:", self.kernel_init_combo)

        # ── Input / output transform (optional) ───────────────
        # x_transformed = x_raw * scale + shift  (one row per input dim)
        # y_transformed = y_raw * scale + shift  (one row per output)
        # Off by default; scale=1/shift=0 is the identity, so turning a
        # transform on with untouched defaults changes nothing until a
        # value is edited. Rows are rebuilt back to identity defaults
        # whenever the dimension or output count changes (see
        # _rebuild_input_transform_rows()/_rebuild_output_transform_rows(),
        # called from _on_dim_changed()/_on_num_outputs_changed()) -- a row
        # left over from a different shape doesn't mean anything once the
        # number of columns it applies to changes, same reasoning as the
        # Boundary Conditions panel clearing on dimension switch.
        def _transform_row(label_text):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.addWidget(QLabel(label_text))
            row_l.addStretch()
            row_l.addWidget(QLabel("×"))
            scale_spin = QDoubleSpinBox()
            scale_spin.setRange(-1e6, 1e6); scale_spin.setDecimals(4)
            scale_spin.setSingleStep(0.1); scale_spin.setValue(1.0)
            scale_spin.setFixedWidth(85); scale_spin.setFixedHeight(26)
            row_l.addWidget(scale_spin)
            row_l.addWidget(QLabel("+"))
            shift_spin = QDoubleSpinBox()
            shift_spin.setRange(-1e6, 1e6); shift_spin.setDecimals(4)
            shift_spin.setSingleStep(0.1); shift_spin.setValue(0.0)
            shift_spin.setFixedWidth(85); shift_spin.setFixedHeight(26)
            row_l.addWidget(shift_spin)
            return row_w, scale_spin, shift_spin
        self._transform_row_factory = _transform_row

        self.input_transform_cb = QCheckBox("Enable input transform")
        self.input_transform_cb.setChecked(False)
        nn_layout.addWidget(self.input_transform_cb)

        self.input_transform_widget = QWidget()
        it_layout = QVBoxLayout(self.input_transform_widget)
        it_layout.setSpacing(3); it_layout.setContentsMargins(12, 0, 0, 4)
        it_hint = QLabel("x_transformed = x_raw × scale + shift")
        self._register_style(it_hint, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        it_layout.addWidget(it_hint)
        self.input_transform_rows_layout = QVBoxLayout()
        self.input_transform_rows_layout.setSpacing(3)
        it_layout.addLayout(self.input_transform_rows_layout)
        self.input_transform_rows = []
        nn_layout.addWidget(self.input_transform_widget)
        self.input_transform_widget.setVisible(False)
        self.input_transform_cb.stateChanged.connect(
            lambda: self.input_transform_widget.setVisible(self.input_transform_cb.isChecked()))

        self.output_transform_cb = QCheckBox("Enable output transform")
        self.output_transform_cb.setChecked(False)
        nn_layout.addWidget(self.output_transform_cb)

        self.output_transform_widget = QWidget()
        ot_layout = QVBoxLayout(self.output_transform_widget)
        ot_layout.setSpacing(3); ot_layout.setContentsMargins(12, 0, 0, 4)
        ot_hint = QLabel("y_transformed = y_raw × scale + shift")
        self._register_style(ot_hint, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        ot_layout.addWidget(ot_hint)
        self.output_transform_rows_layout = QVBoxLayout()
        self.output_transform_rows_layout.setSpacing(3)
        ot_layout.addLayout(self.output_transform_rows_layout)
        self.output_transform_rows = []
        nn_layout.addWidget(self.output_transform_widget)
        self.output_transform_widget.setVisible(False)
        self.output_transform_cb.stateChanged.connect(
            lambda: self.output_transform_widget.setVisible(self.output_transform_cb.isChecked()))

        self._rebuild_input_transform_rows()
        self._rebuild_output_transform_rows(1)

        left_layout.addWidget(nn_group)

        # ── Mini-batch ────────────────────────────────────────
        batch_group = QGroupBox("Mini-batch Training")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.setSpacing(5)

        self.batch_check = QCheckBox("Enable mini-batch training")
        self.batch_check.setChecked(True)
        self.batch_check.stateChanged.connect(self._on_batch_changed)
        batch_layout.addWidget(self.batch_check)

        self.batch_widget = QWidget()
        bw_layout = QHBoxLayout(self.batch_widget)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.addWidget(QLabel("Batch size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 10000)
        self.batch_spin.setSingleStep(16)
        self.batch_spin.setValue(32)
        self.batch_spin.setFixedWidth(100)
        self.batch_spin.setFixedHeight(28)
        bw_layout.addStretch()
        bw_layout.addWidget(self.batch_spin)
        self.batch_widget.setVisible(True)
        batch_layout.addWidget(self.batch_widget)

        left_layout.addWidget(batch_group)

        # ── Training ──────────────────────────────────────────
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(5)
        train_layout.setContentsMargins(10, 10, 10, 10)

        def _train_row(label, widget):
            train_layout.addWidget(QLabel(label))
            train_layout.addWidget(widget)

        div0 = QLabel("─── Initial Condition (IC) Pre-Training (optional) ───")
        self._register_style(div0, "hint", lambda css, _c='#505080', _e='': f"color: {_c}; {_e}{css}")
        train_layout.addWidget(div0)

        self.ic_pretrain_cb = QCheckBox("Enable IC-guided pre-training")
        self.ic_pretrain_cb.setChecked(False)
        self._register_style(self.ic_pretrain_cb, "hint", lambda css, _c='#69db7c', _e='': f"color: {_c}; {_e}{css}")
        self.ic_pretrain_cb.stateChanged.connect(self._on_ic_pretrain_changed)
        train_layout.addWidget(self.ic_pretrain_cb)

        self.ic_pretrain_widget = QWidget()
        ic_pt_layout = QVBoxLayout(self.ic_pretrain_widget)
        ic_pt_layout.setSpacing(4); ic_pt_layout.setContentsMargins(0, 0, 0, 0)

        # Train-from-start vs Restore-from-checkpoint -- a single either/or
        # choice, not a checkbox layered on top of the train fields. Only
        # one of these ever actually applies at run time (see codegen.py:
        # restoring loads the saved weights and skips training entirely,
        # so the optimizer/iterations/test/initial-points fields below are
        # meaningless while restoring), so only the fields for whichever
        # is chosen are ever shown -- picking "Restore" hides the training
        # fields instead of just adding the restore path underneath them.
        ic_mode_row = QHBoxLayout()
        self.ic_pretrain_mode_train = QRadioButton("▶ Train from start")
        self.ic_pretrain_mode_restore = QRadioButton("🔄 Restore from checkpoint")
        self.ic_pretrain_mode_train.setChecked(True)
        self._ic_pretrain_mode_group = QButtonGroup(self.ic_pretrain_widget)
        self._ic_pretrain_mode_group.addButton(self.ic_pretrain_mode_train)
        self._ic_pretrain_mode_group.addButton(self.ic_pretrain_mode_restore)
        ic_mode_row.addWidget(self.ic_pretrain_mode_train)
        ic_mode_row.addWidget(self.ic_pretrain_mode_restore)
        ic_pt_layout.addLayout(ic_mode_row)
        self.ic_pretrain_mode_train.toggled.connect(self._update_ic_pretrain_visibility)

        # Train-from-start fields
        self.ic_pretrain_train_fields_widget = QWidget()
        ic_train_fields_layout = QVBoxLayout(self.ic_pretrain_train_fields_widget)
        ic_train_fields_layout.setSpacing(4); ic_train_fields_layout.setContentsMargins(0, 0, 0, 0)
        ic_train_fields_layout.addWidget(QLabel("IC pre-train optimizer:"))
        self.ic_pretrain_opt = QComboBox()
        self.ic_pretrain_opt.addItem("Adam", "adam")
        self.ic_pretrain_opt.setFixedHeight(28)
        ic_train_fields_layout.addWidget(self.ic_pretrain_opt)
        ic_train_fields_layout.addWidget(QLabel("IC pre-train iterations:"))
        self.ic_pretrain_iters = QSpinBox()
        self.ic_pretrain_iters.setRange(100, 500000)
        self.ic_pretrain_iters.setSingleStep(1000)
        self.ic_pretrain_iters.setValue(20000)
        self.ic_pretrain_iters.setFixedHeight(28)
        ic_train_fields_layout.addWidget(self.ic_pretrain_iters)

        # Test points
        ic_test_row = QHBoxLayout()
        ic_test_row.addWidget(QLabel("Test points:"))
        self.ic_pretrain_test = QSpinBox()
        self.ic_pretrain_test.setRange(100, 100000)
        self.ic_pretrain_test.setSingleStep(1000)
        self.ic_pretrain_test.setValue(10000)
        self.ic_pretrain_test.setFixedHeight(28)
        self.ic_pretrain_test.setFixedWidth(100)
        ic_test_row.addStretch(); ic_test_row.addWidget(self.ic_pretrain_test)
        ic_train_fields_layout.addLayout(ic_test_row)

        # Initial points (only shown when IC is from expression, not file --
        # see _update_ic_pretrain_visibility(), which also accounts for that)
        self.ic_pretrain_init_widget = QWidget()
        ic_init_row = QHBoxLayout(self.ic_pretrain_init_widget)
        ic_init_row.setContentsMargins(0, 0, 0, 0)
        ic_init_row.addWidget(QLabel("Initial points:"))
        self.ic_pretrain_init = QSpinBox()
        # Minimum of 1, not 0: this is the count of points sampled at the
        # initial-time slice for IC pre-training, and with 0 of them no
        # training points exist at all -- DeepXDE doesn't fail cleanly on
        # that (an empty-array indexing bug deep inside its IC filtering),
        # it's a confusing crash instead. See _validate_optimizer_settings-
        # adjacent codegen.py guard for the belt-and-suspenders clamp.
        self.ic_pretrain_init.setRange(1, 10000)
        self.ic_pretrain_init.setSingleStep(100)
        self.ic_pretrain_init.setValue(1000)
        self.ic_pretrain_init.setFixedHeight(28)
        self.ic_pretrain_init.setFixedWidth(100)
        ic_init_row.addStretch(); ic_init_row.addWidget(self.ic_pretrain_init)
        ic_train_fields_layout.addWidget(self.ic_pretrain_init_widget)
        ic_pt_layout.addWidget(self.ic_pretrain_train_fields_widget)

        # Restore-from-checkpoint field (path only -- see mode toggle above)
        self.ic_pretrain_restore_widget = QWidget()
        ic_restore_layout = QHBoxLayout(self.ic_pretrain_restore_widget)
        ic_restore_layout.setContentsMargins(0, 0, 0, 0)
        self.ic_pretrain_restore_path = QLineEdit()
        self.ic_pretrain_restore_path.setPlaceholderText("Browse for IC pre-train .pt file...")
        self.ic_pretrain_restore_path.setFixedHeight(26)
        ic_restore_layout.addWidget(self.ic_pretrain_restore_path)
        ic_restore_browse = QPushButton("Browse")
        ic_restore_browse.setFixedHeight(26); ic_restore_browse.setFixedWidth(65)
        ic_restore_browse.clicked.connect(lambda: self.ic_pretrain_restore_path.setText(
            QFileDialog.getOpenFileName(None, "Select IC pre-train model", "", "Model (*.pt)")[0]))
        ic_restore_layout.addWidget(ic_restore_browse)
        ic_pt_layout.addWidget(self.ic_pretrain_restore_widget)

        self._update_ic_pretrain_visibility()
        self.ic_pretrain_widget.setVisible(False)
        train_layout.addWidget(self.ic_pretrain_widget)

        # ── Optimizer Scheduler ───────────────────────────────
        div_sched = QLabel("─── Training Phases ───")
        self._register_style(div_sched, "hint", lambda css, _c='#505080', _e='': f"color: {_c}; {_e}{css}")
        train_layout.addWidget(div_sched)
        self.sched_cb = QCheckBox("Enable Optimizer Scheduler")
        self.sched_cb.setChecked(True)
        self.sched_cb.setVisible(False)  # always on, hidden
        train_layout.addWidget(self.sched_cb)
        self.sched_widget = QWidget()

        # Hidden legacy widgets — kept for _build_config compatibility
        self.opt1_combo = QComboBox(); self.opt1_combo.addItems(["adam", "sgd", "rmsprop"])
        self.opt1_combo.setVisible(False)
        self.iter1_spin = QSpinBox(); self.iter1_spin.setRange(0, 1000000)
        self.iter1_spin.setValue(0); self.iter1_spin.setVisible(False)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6); self.lr_spin.setValue(0.001); self.lr_spin.setVisible(False)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "MAE", "mean l2 relative error",
                                   "mean absolute percentage error", "softplus"])
        self.loss_combo.setVisible(False)
        self.opt2_combo = QComboBox(); self.opt2_combo.addItems(["none", "lbfgs"])
        self.opt2_combo.setCurrentText("none"); self.opt2_combo.setVisible(False)
        self.iter2_spin = QSpinBox(); self.iter2_spin.setRange(0, 100000)
        self.iter2_spin.setValue(0); self.iter2_spin.setVisible(False)

        self.sched_widget = QWidget()
        sched_layout = QVBoxLayout(self.sched_widget)
        sched_layout.setSpacing(4)
        sched_layout.setContentsMargins(0, 0, 0, 0)

        # Same weights checkbox
        self.sched_same_weights_cb = QCheckBox("Use same weights for all phases")
        self.sched_same_weights_cb.setChecked(True)
        self._register_style(self.sched_same_weights_cb, "hint", lambda css, _c='#69db7c', _e='': f"color: {_c}; {_e}{css}")
        self.sched_same_weights_cb.stateChanged.connect(
            lambda s: self._build_weight_inputs(self.num_outputs_spin.value()))
        sched_layout.addWidget(self.sched_same_weights_cb)

        # Phase list container
        self.sched_phases_widget = QWidget()
        self.sched_phases_layout = QVBoxLayout(self.sched_phases_widget)
        self.sched_phases_layout.setSpacing(4)
        self.sched_phases_layout.setContentsMargins(0, 0, 0, 0)
        sched_layout.addWidget(self.sched_phases_widget)
        self.sched_phase_list = []  # list of dicts with widgets
        # Add default phases after UI is built
        QTimer.singleShot(0, lambda: self._setup_default_scheduler_phases(''))

        # Add phase button
        add_phase_btn = QPushButton("➕ Add Phase")
        add_phase_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_phase_btn.clicked.connect(lambda: self._add_scheduler_phase())
        sched_layout.addWidget(add_phase_btn)

        self.sched_widget.setVisible(True)
        train_layout.addWidget(self.sched_widget)

        # L-BFGS settings
        self.lbfgs_widget = QWidget()
        lbfgs_layout = QVBoxLayout(self.lbfgs_widget)
        lbfgs_layout.setSpacing(4); lbfgs_layout.setContentsMargins(0, 0, 0, 0)

        # Row: recommended checkbox + float precision
        _lbfgs_top_row = QHBoxLayout()
        self.lbfgs_use_default_cb = QCheckBox("Use L-BFGS recommended settings")
        self.lbfgs_use_default_cb.setChecked(True)
        self.lbfgs_use_default_cb.stateChanged.connect(self._on_lbfgs_default_changed)
        _lbfgs_top_row.addWidget(self.lbfgs_use_default_cb)
        _lbfgs_top_row.addStretch()
        _lbfgs_top_row.addWidget(QLabel("Float:"))
        self.lbfgs_float_combo = QComboBox()
        self.lbfgs_float_combo.addItems(["float64", "float32"])
        self.lbfgs_float_combo.setCurrentText("float32")
        self.lbfgs_float_combo.setFixedWidth(90)
        self.lbfgs_float_combo.setToolTip("Float precision for L-BFGS (float64 recommended for accuracy)")
        _lbfgs_top_row.addWidget(self.lbfgs_float_combo)
        lbfgs_layout.addLayout(_lbfgs_top_row)

        self.lbfgs_manual_widget = QWidget()
        lbfgs_manual_layout = QVBoxLayout(self.lbfgs_manual_widget)
        lbfgs_manual_layout.setSpacing(4); lbfgs_manual_layout.setContentsMargins(0, 0, 0, 0)

        def _lbfgs_row(label, val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = SciLineEdit(val); sb.setFixedWidth(120)
            row.addStretch(); row.addWidget(sb)
            lbfgs_manual_layout.addLayout(row)
            return sb

        self.lbfgs_maxcor  = _lbfgs_row("maxcor:",  200)
        self.lbfgs_ftol    = _lbfgs_row("ftol:",     1e-16)
        self.lbfgs_gtol    = _lbfgs_row("gtol:",     1e-16)
        self.lbfgs_maxiter = _lbfgs_row("maxiter:",  50000)
        self.lbfgs_maxfun  = _lbfgs_row("maxfun:",   62500)
        self.lbfgs_maxls   = _lbfgs_row("maxls:",    50)

        self.lbfgs_manual_widget.setVisible(False)
        lbfgs_layout.addWidget(self.lbfgs_manual_widget)
        self.lbfgs_widget.setVisible(False)
        train_layout.addWidget(self.lbfgs_widget)
        self.opt2_combo.currentTextChanged.connect(self._on_opt2_changed)
        left_layout.addWidget(train_group)

        # ── Loss Weights ──────────────────────────────────────
        self.weights_group = QGroupBox("Loss Weights")
        self.weights_main_layout = QVBoxLayout(self.weights_group)
        self.weights_main_layout.setSpacing(4)
        self.weight_widgets = {}
        self._build_weight_inputs(1)
        left_layout.addWidget(self.weights_group)

        # ── Adaptive Training ─────────────────────────────────
        adapt_group = QGroupBox("Adaptive Training")
        adapt_layout = QVBoxLayout(adapt_group)
        adapt_layout.setSpacing(5)

        row_a1 = QHBoxLayout()
        row_a1.addWidget(QLabel("Method:"))
        self.adapt_combo = QComboBox()
        self.adapt_combo.addItem("None", "None")
        self.adapt_combo.addItem("Residual-based Adaptive Refinement (RAR)", "RAR")
        self.adapt_combo.addItem("Time Adaptive", "Time Adaptive")
        self.adapt_combo.setFixedHeight(28)
        self.adapt_combo.currentTextChanged.connect(self._on_adapt_changed)
        row_a1.addWidget(self.adapt_combo)
        adapt_layout.addLayout(row_a1)

        # RAR widget
        self.rar_widget = QWidget()
        rar_layout = QVBoxLayout(self.rar_widget)
        rar_layout.setSpacing(4); rar_layout.setContentsMargins(0, 0, 0, 0)

        def _rar_row(label, default, min_v, max_v, step):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox(); sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(default)
            sb.setFixedHeight(28); sb.setFixedWidth(100)
            row.addStretch(); row.addWidget(sb)
            rar_layout.addLayout(row)
            return sb

        self.rar_cycles     = _rar_row("RAR cycles:",         3,     1,    20,     1)
        self.rar_candidates = _rar_row("Candidate points:",   50000, 1000, 200000, 5000)
        self.rar_add_points = _rar_row("Points per cycle:",   500,   10,   10000,  100)
        self.rar_adam_iters = _rar_row("Adam iters/cycle:",   5000,  100,  50000,  1000)
        self.rar_lbfgs_iters= _rar_row("L-BFGS iters/cycle:", 0,    0,    50000,  1000)
        self.rar_widget.setVisible(False)
        adapt_layout.addWidget(self.rar_widget)

        # Time Adaptive widget
        self.ta_widget = QWidget()
        ta_layout = QVBoxLayout(self.ta_widget)
        ta_layout.setSpacing(4); ta_layout.setContentsMargins(0, 0, 0, 0)

        # ── Step groups ───────────────────────────────────────
        ta_groups_label = QLabel("Time step groups:")
        self._register_style(ta_groups_label, "hint", lambda css, _c='#a0c4ff', _e='': f"color: {_c}; {_e}{css}")
        ta_layout.addWidget(ta_groups_label)

        self.ta_groups_widget = QWidget()
        self.ta_groups_layout = QVBoxLayout(self.ta_groups_widget)
        self.ta_groups_layout.setSpacing(3)
        self.ta_groups_layout.setContentsMargins(0, 0, 0, 0)
        ta_layout.addWidget(self.ta_groups_widget)

        self.ta_group_rows = []  # list of dicts with widgets

        add_group_btn = QPushButton("➕ Add step group")
        add_group_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_group_btn.clicked.connect(lambda: self._add_ta_step_group())
        ta_layout.addWidget(add_group_btn)

        # Add default group
        self._add_ta_step_group(0.0, 1.0, 10)

        # Keep ta_steps for backward compat — hidden
        self.ta_steps = QSpinBox(); self.ta_steps.setRange(2, 500)
        self.ta_steps.setValue(10); self.ta_steps.setVisible(False)
        ta_layout.addWidget(self.ta_steps)

        row_ta2 = QHBoxLayout()
        row_ta2.addWidget(QLabel("IC grid resolution:"))
        self.ta_grid = QComboBox(); self.ta_grid.addItems(["101", "51", "21", "11"])
        self.ta_grid.setFixedHeight(28); self.ta_grid.setFixedWidth(100)
        row_ta2.addStretch(); row_ta2.addWidget(self.ta_grid)
        ta_layout.addLayout(row_ta2)

        # Transfer learning
        self.ta_transfer_cb = QCheckBox("Enable transfer learning (warm start from previous step)")
        self.ta_transfer_cb.setChecked(False)
        self._register_style(self.ta_transfer_cb, "hint", lambda css, _c='#69db7c', _e='': f"color: {_c}; {_e}{css}")
        ta_layout.addWidget(self.ta_transfer_cb)

        self.ta_transfer_opt_widget = QWidget()
        tl_row = QHBoxLayout(self.ta_transfer_opt_widget)
        tl_row.setContentsMargins(0, 0, 0, 0)
        tl_row.addWidget(QLabel("Transfer optimizer:"))
        self.ta_transfer_opt = QComboBox()
        self.ta_transfer_opt.addItem("Adam", "adam")
        self.ta_transfer_opt.addItem("L-BFGS", "lbfgs")
        self.ta_transfer_opt.setFixedHeight(26); self.ta_transfer_opt.setFixedWidth(80)
        tl_row.addStretch(); tl_row.addWidget(self.ta_transfer_opt)
        self.ta_transfer_opt_widget.setVisible(False)
        ta_layout.addWidget(self.ta_transfer_opt_widget)
        self.ta_transfer_cb.stateChanged.connect(
            lambda s: self.ta_transfer_opt_widget.setVisible(s == 2))

        self.ta_widget.setVisible(False)
        adapt_layout.addWidget(self.ta_widget)
        left_layout.addWidget(adapt_group)

        # ── Inverse PINN panel ────────────────────────────────
        self.inverse_group = QGroupBox("Inverse PINN Settings")
        inv_layout = QVBoxLayout(self.inverse_group)
        inv_layout.setSpacing(5)

        inv_layout.addWidget(QLabel("Trainable (unknown) variables:"))
        self.inv_vars_widget = QWidget()
        self.inv_vars_layout = QVBoxLayout(self.inv_vars_widget)
        self.inv_vars_layout.setSpacing(3)
        self.inv_vars_layout.setContentsMargins(0, 0, 0, 0)
        inv_layout.addWidget(self.inv_vars_widget)

        self.inv_var_rows = []  # list of dicts with widgets

        add_inv_var_btn = QPushButton("➕ Add trainable variable")
        add_inv_var_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_inv_var_btn.clicked.connect(lambda: self._add_inverse_var_row())
        inv_layout.addWidget(add_inv_var_btn)

        self.inv_multi_var_hint = QLabel(
            "⚠ Add each new variable's name to your PDE expression(s) too\n"
            "(PDE Builder tab), wherever that unknown quantity belongs.")
        self._register_style(self.inv_multi_var_hint, "hint", lambda css, _c='#ffd43b', _e='': f"color: {_c}; {_e}{css}")
        self.inv_multi_var_hint.setWordWrap(True)
        self.inv_multi_var_hint.setVisible(False)
        inv_layout.addWidget(self.inv_multi_var_hint)

        # Add the first (primary) trainable variable row -- always present;
        # renaming it keeps a loaded Quick Example's PDE box in sync
        # (INVERSE_AUTO_CONST / _sync_inverse_pde_substitution), same as the
        # single-variable behavior this replaces.
        self._add_inverse_var_row("trainable_variable_1", 1.0)

        inv_ref_row = QHBoxLayout()
        inv_ref_toggle = QCheckBox("\U0001F4D6 Show inverse reference")
        inv_ref_toggle.setChecked(False)
        self._register_style(inv_ref_toggle, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        inv_ref_row.addWidget(inv_ref_toggle)
        inv_ref_row.addStretch()
        inv_ref_row_widget = QWidget()
        inv_ref_row_widget.setLayout(inv_ref_row)
        inv_layout.addWidget(inv_ref_row_widget)
        inv_ref_text = ("Each trainable (unknown) variable is a value DeepXDE infers (optimizes)\n"
                        "during training, e.g. a diffusion coefficient or reaction rate.\n"
                        "The first is named 'trainable_variable_1' by default -- rename any of\n"
                        "them to anything you like (any valid Python identifier); renaming the\n"
                        "first one keeps a loaded Quick Example's PDE box in sync automatically.\n"
                        "For a custom PDE (no example), or for any variable beyond the first,\n"
                        "make sure the same name also appears in your PDE expression(s) (PDE\n"
                        "Builder tab) wherever that quantity belongs, e.g. rename to D and write:\n"
                        "du_t - D*du_xx\n"
                        "Initial guess sets each variable's starting value before optimization.\n"
                        "All trainable variables are fit against the same shared measured data\n"
                        "file(s) below (x, t, u, ...) -- add more than one if you have several\n"
                        "observation datasets; each gets its own \"which output\" selector and its\n"
                        "own loss weight, since each becomes a separate loss term.")
        inv_ref_hint = QLabel(inv_ref_text)
        self._register_style(inv_ref_hint, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        inv_ref_hint.setWordWrap(True)
        inv_ref_hint.setVisible(False)
        inv_layout.addWidget(inv_ref_hint)
        inv_ref_toggle.stateChanged.connect(lambda state, h=inv_ref_hint: h.setVisible(state == 2))

        inv_layout.addWidget(QLabel("Measured data file(s) (x, t, u):"))
        self.inv_data_files_widget = QWidget()
        self.inv_data_files_layout = QVBoxLayout(self.inv_data_files_widget)
        self.inv_data_files_layout.setSpacing(6)
        self.inv_data_files_layout.setContentsMargins(0, 0, 0, 0)
        inv_layout.addWidget(self.inv_data_files_widget)

        self.inv_data_rows = []  # list of dicts with widgets

        add_inv_data_btn = QPushButton("➕ Add measured data file")
        add_inv_data_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_inv_data_btn.clicked.connect(lambda: self._add_inverse_data_row())
        inv_layout.addWidget(add_inv_data_btn)

        self.inv_multi_data_hint = QLabel(
            "⚠ Each additional measured data file adds its own\n"
            "observation loss term, weighted by its own box below.")
        self._register_style(self.inv_multi_data_hint, "hint", lambda css, _c='#ffd43b', _e='': f"color: {_c}; {_e}{css}")
        self.inv_multi_data_hint.setWordWrap(True)
        self.inv_multi_data_hint.setVisible(False)
        inv_layout.addWidget(self.inv_multi_data_hint)

        # Add the first (primary) measured-data-file row -- always present,
        # matching the single-file behavior this generalizes by default.
        self._add_inverse_data_row()

        # v19: the separate "IC type" (expression/file) selector that used
        # to live here was removed -- it was confusing to have two places
        # to set the IC when the Initial Condition panel above already
        # covers both expression and file-based ICs, for every problem
        # type (including Inverse) and dimension. Inverse problems now
        # just use that one panel, same as Forward problems.
        self.inv_param_log_scale = QCheckBox("Log scale for parameter convergence plot")
        self.inv_param_log_scale.setChecked(False)
        inv_layout.addWidget(self.inv_param_log_scale)

        self.inverse_group.setVisible(False)
        left_layout.insertWidget(7, self.inverse_group)  # right after PDE Definition

        # Parametric Study removed (untested, not exposed in the GUI).

        # ── Solve / Stop buttons ──────────────────────────────
        self.solve_btn = QPushButton("▶  Solve")
        self.solve_btn.setMinimumHeight(44)
        self._register_style(self.solve_btn, "button", lambda css: f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0078d4,stop:1 #005a9e);
                          color: white; {css} border-radius: 6px; border: none; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a8ae8,stop:1 #0070c0); }}
            QPushButton:disabled {{ background: #333355; color: #666; }}
        """)
        self.solve_btn.clicked.connect(self._on_solve)
        left_layout.addWidget(self.solve_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self._register_style(self.stop_btn, "button", lambda css: f"""
            QPushButton {{ background: #6b1f1f; color: white; {css} border-radius: 6px; border: none; }}
            QPushButton:hover {{ background: #8b2f2f; }}
            QPushButton:disabled {{ background: #333355; color: #666; }}
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        left_layout.addWidget(self.stop_btn)

        # ── Model Restore & Visualization ─────────────────────
        restore_group = QGroupBox("Model Restore && Visualization")
        restore_layout = QVBoxLayout(restore_group)
        restore_layout.setSpacing(5)

        restore_toggle = QCheckBox("🔄 Enable Model Restore && Visualization")
        restore_toggle.setChecked(False)
        self._register_style(restore_toggle, "hint", lambda css, _c='#a0c4ff', _e='': f"color: {_c}; {_e}{css}")
        restore_layout.addWidget(restore_toggle)

        restore_content = QWidget()
        restore_content_layout = QVBoxLayout(restore_content)
        restore_content_layout.setContentsMargins(0, 0, 0, 0)
        restore_content_layout.setSpacing(5)
        restore_content.setVisible(False)
        restore_toggle.stateChanged.connect(lambda s: restore_content.setVisible(s == 2))
        restore_layout.addWidget(restore_content)

        # Restore mode -- Forward vs Inverse. Forward keeps the panel
        # exactly as it always was (model + config + optimizer + one of
        # the four field-plotting viz types). Inverse gets everything
        # Forward has -- an Inverse-trained model's network weights
        # restore and plot the solution field the same way -- plus two
        # more viz types, Parameter Convergence Plot/Animation, which
        # read the *_convergence.txt file(s) saved alongside the model
        # instead of the model itself (the trained parameter's value was
        # never part of the .pt checkpoint to begin with -- see
        # _build_restore_script's comment for why).
        restore_mode_row = QHBoxLayout()
        restore_mode_row.addWidget(QLabel("Restore:"))
        self.restore_mode_combo = QComboBox()
        self.restore_mode_combo.addItems(["Forward Model", "Inverse Model"])
        self.restore_mode_combo.setFixedHeight(28)
        self.restore_mode_combo.currentTextChanged.connect(self._on_restore_mode_changed)
        restore_mode_row.addWidget(self.restore_mode_combo)
        restore_content_layout.addLayout(restore_mode_row)

        self._RESTORE_FORWARD_VIZ = ["Surface", "Line (time steps)", "Animation Line (GIF)", "Animation Surface (GIF)"]
        self._RESTORE_PARAM_VIZ = ["Parameter Convergence Plot (PNG)", "Parameter Convergence Animation (GIF)"]

        self.restore_model_fields_widget = QWidget()
        _rmf_layout = QVBoxLayout(self.restore_model_fields_widget)
        _rmf_layout.setContentsMargins(0, 0, 0, 0)
        _rmf_layout.setSpacing(5)

        _rmf_layout.addWidget(QLabel("Model file (.pt):"))
        restore_path_row = QHBoxLayout()
        self.restore_model_path = QLineEdit()
        self.restore_model_path.setPlaceholderText("Browse for model .pt file...")
        self.restore_model_path.setFixedHeight(28)
        restore_path_row.addWidget(self.restore_model_path)
        self.restore_browse_btn = QPushButton("Browse")
        self.restore_browse_btn.setFixedHeight(28); self.restore_browse_btn.setFixedWidth(65)
        self.restore_browse_btn.clicked.connect(self._on_browse_restore_model)
        restore_path_row.addWidget(self.restore_browse_btn)
        _rmf_layout.addLayout(restore_path_row)

        _rmf_layout.addWidget(QLabel("Config file (model_config.json):"))
        config_path_row = QHBoxLayout()
        self.restore_config_path = QLineEdit()
        self.restore_config_path.setPlaceholderText("Auto-detected or browse...")
        self.restore_config_path.setFixedHeight(28)
        config_path_row.addWidget(self.restore_config_path)
        self.restore_config_browse_btn = QPushButton("Browse")
        self.restore_config_browse_btn.setFixedHeight(28); self.restore_config_browse_btn.setFixedWidth(65)
        self.restore_config_browse_btn.clicked.connect(self._on_browse_restore_config)
        config_path_row.addWidget(self.restore_config_browse_btn)
        _rmf_layout.addLayout(config_path_row)
        restore_content_layout.addWidget(self.restore_model_fields_widget)

        self.restore_optimizer_widget = QWidget()
        _ro_layout = QVBoxLayout(self.restore_optimizer_widget)
        _ro_layout.setContentsMargins(0, 0, 0, 0)
        _ro_layout.setSpacing(5)
        _ro_layout.addWidget(QLabel("Optimizer used for this model:"))
        self.restore_optimizer_combo = QComboBox()
        self.restore_optimizer_combo.addItem("Adam", "adam")
        self.restore_optimizer_combo.addItem("L-BFGS", "lbfgs")
        self.restore_optimizer_combo.setFixedHeight(28)
        _ro_layout.addWidget(self.restore_optimizer_combo)
        restore_content_layout.addWidget(self.restore_optimizer_widget)

        # Inverse-only: lets an older Inverse-trained model (saved before
        # model_config.json recorded this) still restore. model.restore()
        # needs the compiled model to have the SAME number of
        # external_trainable_variables the optimizer had at training time,
        # or it crashes with a parameter-group-size mismatch -- newer
        # models auto-fill this from model_config.json's "inverse_variables"
        # field; this box lets you type it in for older ones that don't
        # have that field. The variable's actual trained VALUE is never
        # recoverable from the checkpoint either way, so any placeholder
        # init value here is fine -- only the count/names matter.
        self.restore_inv_vars_widget = QWidget()
        _riv_layout = QVBoxLayout(self.restore_inv_vars_widget)
        _riv_layout.setContentsMargins(0, 0, 0, 0)
        _riv_layout.setSpacing(5)
        _riv_layout.addWidget(QLabel("Trainable variable names (comma-separated -- blank = auto-detect):"))
        self.restore_inv_var_names = QLineEdit()
        self.restore_inv_var_names.setPlaceholderText("e.g. D  or  D,k  (leave blank to auto-detect from config)")
        self.restore_inv_var_names.setFixedHeight(28)
        _riv_layout.addWidget(self.restore_inv_var_names)
        restore_content_layout.addWidget(self.restore_inv_vars_widget)
        self.restore_inv_vars_widget.setVisible(False)

        restore_content_layout.addWidget(QLabel("Visualization type:"))
        self.restore_viz_combo = QComboBox()
        self.restore_viz_combo.addItems(self._RESTORE_FORWARD_VIZ)
        self.restore_viz_combo.setFixedHeight(28)
        self.restore_viz_combo.currentTextChanged.connect(self._on_restore_viz_changed)
        restore_content_layout.addWidget(self.restore_viz_combo)

        self.restore_tsteps_spin = QSpinBox()
        self.restore_tsteps_spin.setRange(2, 50); self.restore_tsteps_spin.setValue(10)
        self.restore_tsteps_spin.setVisible(False)

        self._restore_viz_settings = {
            'colormap': 'jet',
            'surface_time': 1.0,
            'n_steps': 10,
            'colorbar': True,
            'levels': 100,
            'resolution': 200,
            'dpi': 300,
            'auto_range': True,
            'vmin': -1.0,
            'vmax': 1.0,
            'linewidth': 2.0,
            'fps': 10,
            'title': '',
            'xlabel': '',
            'ylabel': '',
        }

        self.restore_output_widget = QWidget()
        _rout_layout = QVBoxLayout(self.restore_output_widget)
        _rout_layout.setContentsMargins(0, 0, 0, 0)
        _rout_layout.setSpacing(5)
        _rout_layout.addWidget(QLabel("Output to plot:"))
        self.restore_output_combo = QComboBox()
        self.restore_output_combo.addItems(["Output 1 (u)"])
        self.restore_output_combo.setFixedHeight(28)
        _rout_layout.addWidget(self.restore_output_combo)
        restore_content_layout.addWidget(self.restore_output_widget)

        # Inverse-only: Parameter Convergence Plot/Animation settings.
        # One or more *_convergence.txt files, one per trainable variable
        # -- same add/remove row-list pattern used everywhere else in this
        # app (Boundary Conditions rows, trainable-variable rows,
        # measured-data-file rows). Unlike those, every row here is
        # removable (including the first) since there's no single legacy
        # field to keep aliased to a "primary" row.
        self.restore_param_widget = QWidget()
        _rp_layout = QVBoxLayout(self.restore_param_widget)
        _rp_layout.setContentsMargins(0, 0, 0, 0)
        _rp_layout.setSpacing(5)
        _rp_layout.addWidget(QLabel("Parameter convergence text file(s) (<name>_convergence.txt):"))
        self.restore_param_files_widget = QWidget()
        self.restore_param_files_layout = QVBoxLayout(self.restore_param_files_widget)
        self.restore_param_files_layout.setSpacing(6)
        self.restore_param_files_layout.setContentsMargins(0, 0, 0, 0)
        _rp_layout.addWidget(self.restore_param_files_widget)
        self.restore_param_rows = []
        add_param_file_btn = QPushButton("➕ Add parameter file")
        add_param_file_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_param_file_btn.clicked.connect(lambda: self._add_restore_param_row())
        _rp_layout.addWidget(add_param_file_btn)
        self._add_restore_param_row()

        _rp_combine_row = QHBoxLayout()
        _rp_combine_row.addWidget(QLabel("Output:"))
        self.restore_param_combine_combo = QComboBox()
        self.restore_param_combine_combo.addItems([
            "Combine all variables into one file",
            "Separate file per variable",
        ])
        self.restore_param_combine_combo.setFixedHeight(28)
        _rp_combine_row.addWidget(self.restore_param_combine_combo)
        _rp_layout.addLayout(_rp_combine_row)

        self.restore_param_log_cb = QCheckBox("Log-scale y-axis")
        self.restore_param_log_cb.setStyleSheet("color: #a0c4ff;")
        _rp_layout.addWidget(self.restore_param_log_cb)

        restore_content_layout.addWidget(self.restore_param_widget)
        self.restore_param_widget.setVisible(False)

        restore_content_layout.addWidget(QLabel("Save visualization to:"))
        restore_save_row = QHBoxLayout()
        self.restore_save_path = QLineEdit()
        self.restore_save_path.setPlaceholderText("Directory to save output...")
        self.restore_save_path.setFixedHeight(28)
        restore_save_row.addWidget(self.restore_save_path)
        self.restore_save_browse_btn = QPushButton("Browse")
        self.restore_save_browse_btn.setFixedHeight(28); self.restore_save_browse_btn.setFixedWidth(65)
        self.restore_save_browse_btn.clicked.connect(self._on_browse_restore_save)
        restore_save_row.addWidget(self.restore_save_browse_btn)
        restore_content_layout.addLayout(restore_save_row)

        self.restore_btn = QPushButton("🔄  Restore & Visualize")

        self.restore_btn.setMinimumHeight(38)
        self._register_style(self.restore_btn, "button", lambda css: f"""
            QPushButton {{ background: #1a5c3a; color: white;
                          {css} border-radius: 6px; border: none; }}
            QPushButton:hover {{ background: #2a7c4a; }}
            QPushButton:disabled {{ background: #333355; color: #666; }}
        """)
        self.restore_btn.clicked.connect(self._on_restore)
        restore_content_layout.addWidget(self.restore_btn)
        left_layout.addWidget(restore_group)

        left_layout.addStretch()
        splitter.addWidget(left_scroll)

        # ── Right panel ───────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 4, 8, 4)
        right_layout.setSpacing(3)

        # Vertical splitter for log + plots
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(right_splitter)

        # Top part — log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)

        log_label = QLabel("📋 Training Log")
        self._register_style(log_label, "hint", lambda css, _c='#a0c4ff', _e='margin-top: 2px; ': f"color: {_c}; {_e}{css}")
        log_label.setFixedHeight(22)
        log_layout.addWidget(log_label)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Training log will appear here...")
        log_layout.addWidget(self.log_box)
        right_splitter.addWidget(log_widget)

        # Bottom part — controls + plots
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(3)
        right_splitter.addWidget(bottom_widget)
        right_splitter.setSizes([150, 650])

        # Controls row
        ctrl_row = QHBoxLayout()

        # Output selector
        ctrl_row.addWidget(QLabel("Plot output:"))
        self.plot_output_combo = QComboBox()
        self.plot_output_combo.addItems(["Output 1 (u)"])
        self.plot_output_combo.setFixedHeight(28)
        self.plot_output_combo.setFixedWidth(130)
        ctrl_row.addWidget(self.plot_output_combo)

        ctrl_row.addWidget(QLabel("  Plot type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Surface", "Line (time steps)"])
        self.plot_type_combo.setFixedHeight(28)
        self.plot_type_combo.setFixedWidth(140)
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        ctrl_row.addWidget(self.plot_type_combo)

        self.plot_settings_btn = QPushButton("⚙")
        self.plot_settings_btn.setFixedHeight(28)
        self.plot_settings_btn.setFixedWidth(28)
        self.plot_settings_btn.setToolTip("Plot settings")
        self._register_style(self.plot_settings_btn, "button", lambda css: f"""
            QPushButton {{ background: #3e3e42; color: #a0c4ff; {css}
                          border-radius: 4px; border: 1px solid #586e75; }}
            QPushButton:hover {{ background: #586e75; }}
        """)
        self.plot_settings_btn.clicked.connect(self._on_plot_settings)
        ctrl_row.addWidget(self.plot_settings_btn)

        self.ea_btn = QPushButton("📊 Error Analysis")
        self.ea_btn.setFixedHeight(28)
        self._register_style(self.ea_btn, "button", lambda css: f"""
            QPushButton {{ background: #1a3a5a; color: #74c0fc; {css}
                          border-radius: 4px; border: 1px solid #2a5a8a; padding: 0 8px; }}
            QPushButton:hover {{ background: #2a5a8a; }}
        """)
        self.ea_btn.clicked.connect(self._on_error_analysis_btn)
        ctrl_row.addWidget(self.ea_btn)

        self._plot_viz_settings = {
            'colormap': 'jet',
            'surface_time': 1.0,
            'n_steps': 4,
            'n_2d_snapshots': 2,
            'colorbar': True,
            'levels': 100,
            'resolution': 200,
            'dpi': 300,
            'auto_range': True,
            'vmin': -1.0,
            'vmax': 1.0,
            'linewidth': 2.0,
            'fps': 10,
        }

        self.export_btn = QPushButton("💾 Export Solution")
        self.export_btn.setFixedHeight(28)
        self._register_style(self.export_btn, "button", lambda css: f"""
            QPushButton {{ background: #1a4a3a; color: #69db7c; {css}
                          border-radius: 4px; border: 1px solid #2a6a4a; padding: 0 8px; }}
            QPushButton:hover {{ background: #2a6a4a; }}
        """)
        self.export_btn.clicked.connect(self._on_export_settings)
        ctrl_row.addWidget(self.export_btn)

        self.param_save_label = QLabel("  Save parameter:")
        self.param_save_label.setVisible(False)
        ctrl_row.addWidget(self.param_save_label)
        self.param_save_combo = QComboBox()
        self.param_save_combo.addItems(["No", "Every 100 iters", "Every 1000 iters"])
        # Default to saving every 100 iterations for every Inverse problem
        # -- without this, the per-variable iteration-vs-value convergence
        # text files never get written unless the user remembers to change
        # this dropdown first, which is easy to miss since it's a small
        # control that's only visible once Inverse is selected.
        self.param_save_combo.setCurrentText("Every 100 iters")
        self.param_save_combo.setFixedHeight(28)
        self.param_save_combo.setFixedWidth(140)
        self.param_save_combo.setVisible(False)
        ctrl_row.addWidget(self.param_save_combo)

        self.timesteps_spin = QSpinBox()
        self.timesteps_spin.setRange(2, 20); self.timesteps_spin.setValue(4)
        self.timesteps_spin.setVisible(False)

        ctrl_row.addStretch()
        bottom_layout.addLayout(ctrl_row)

        # Save / export row
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save to:"))
        self.save_dir_input = QLineEdit()
        self.save_dir_input.setPlaceholderText("Save directory — saves plots, logs, models & data")
        self.save_dir_input.setText(os.path.join(os.path.expanduser("~"), "PINNStudio_Results"))
        self.save_dir_input.setFixedHeight(28)
        save_row.addWidget(self.save_dir_input)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setFixedHeight(28); self.browse_btn.setFixedWidth(65)
        self.browse_btn.clicked.connect(self._on_browse)
        save_row.addWidget(self.browse_btn)

        self.export_grid_combo = QComboBox()
        self.export_grid_combo.addItems(["101", "51", "21", "11"])
        self.export_grid_combo.setVisible(False)

        self.export_tsteps_spin = QSpinBox()
        self.export_tsteps_spin.setRange(2, 50); self.export_tsteps_spin.setValue(11)
        self.export_tsteps_spin.setVisible(False)
        bottom_layout.addLayout(save_row)

        # Plot area
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(6)

        self.loss_label = QLabel()
        self.loss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loss_label.setText("📉 Loss plot")
        self.loss_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.loss_label.setMinimumSize(500, 450)

        self.solution_label = QLabel()
        self.solution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solution_label.setText("🗺 Solution plot")
        self.solution_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.solution_label.setMinimumSize(500, 450)

        plots_layout.addWidget(self.loss_label)
        plots_layout.addWidget(self.solution_label)
        bottom_layout.addLayout(plots_layout)

        splitter.addWidget(right)
        splitter.setSizes([390, 720])

    # ── Dimension change ──────────────────────────────────────
    GEOM_TYPES_2D = ["Rectangle", "Disk", "Ellipse", "Triangle", "Polygon"]
    GEOM_TYPES_3D = ["Cuboid", "Sphere"]

    def _current_input_dim_labels(self):
        if self.radio_3d.isChecked():
            return ["x", "y", "z", "t"]
        elif self.radio_2d.isChecked():
            return ["x", "y", "t"]
        else:
            return ["x", "t"]

    def _rebuild_input_transform_rows(self):
        while self.input_transform_rows_layout.count():
            item = self.input_transform_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.input_transform_rows = []
        for label in self._current_input_dim_labels():
            row_w, scale_spin, shift_spin = self._transform_row_factory(f"{label}:")
            self.input_transform_rows_layout.addWidget(row_w)
            self.input_transform_rows.append(
                {"label": label, "scale": scale_spin, "shift": shift_spin})

    def _rebuild_output_transform_rows(self, n_outputs):
        while self.output_transform_rows_layout.count():
            item = self.output_transform_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.output_transform_rows = []
        for i in range(max(1, n_outputs)):
            row_w, scale_spin, shift_spin = self._transform_row_factory(f"Output {i + 1}:")
            self.output_transform_rows_layout.addWidget(row_w)
            self.output_transform_rows.append({"scale": scale_spin, "shift": shift_spin})

    def _on_dim_changed(self):
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked()
        # Clear the Boundary Conditions panel -- its rows (template-seeded
        # or hand-built) are location EXPRESSIONS in the previous
        # dimension's variables (e.g. "z <= 0" from a 3D problem), which
        # are either meaningless or, worse, reference a variable that
        # no longer exists at all once the dimension changes (x only for
        # 1D; x, y for 2D; x, y, z for 3D). Left in place, a stale row
        # crashes the generated script with a NameError the moment
        # training starts, since the location function's namespace only
        # defines the variables the *current* dimension actually has. A
        # template selected after this still repopulates the panel with
        # that template's own (dimension-correct) rows, exactly as before;
        # this only clears what dimension-switching itself would otherwise
        # leave stale.
        if hasattr(self, 'custom_bc_list') and self.custom_bc_list:
            for e in list(self.custom_bc_list):
                e['widget'].deleteLater()
            self.custom_bc_list.clear()
        # Update quick examples list to match dimension
        self.quick_examples_combo.blockSignals(True)
        self.quick_examples_combo.clear()
        if is_2d:
            self.quick_examples_combo.addItems([
                "None",
                "2D Heat",
                "2D Allen-Cahn (Mattey & Ghosh)",
                "2D Allen-Cahn (Wight & Zhao)"
            ])
        elif is_3d:
            self.quick_examples_combo.addItems(["None", "3D Heat"])
        else:
            self.quick_examples_combo.addItems([
                "None",
                "1D Heat",
                "1D Allen-Cahn"
            ])
        self.quick_examples_combo.blockSignals(False)
        self.view_domain_check.setVisible(is_2d or is_3d)
        for w in self._2d_bc_widgets:
            w.setVisible(is_2d)

        # Repopulate the geometry-type selector for the new dimension.
        self.geom_type_group.setVisible(is_2d or is_3d)
        self.geometry_type_combo.blockSignals(True)
        self.geometry_type_combo.clear()
        if is_2d:
            self.geometry_type_combo.addItems(self.GEOM_TYPES_2D)
        elif is_3d:
            self.geometry_type_combo.addItems(self.GEOM_TYPES_3D)
        self.geometry_type_combo.blockSignals(False)
        self._on_geometry_type_changed(self.geometry_type_combo.currentText())

        self._build_pde_inputs(self.num_outputs_spin.value())
        self._build_bc_inputs(self.num_outputs_spin.value())
        self._build_weight_inputs(self.num_outputs_spin.value())
        self._rebuild_input_transform_rows()

        # Force an immediate layout/repaint pass so the Geometry Type box
        # (and any other widgets just toggled above) always show up right
        # away, instead of waiting on the next natural repaint cycle.
        self.geom_type_group.updateGeometry()
        self.geom_type_group.repaint()
        QApplication.processEvents()

    def _current_geometry_type(self):
        """The active geometry type, or the implicit box type for 1D/unselected."""
        if self.radio_3d.isChecked():
            return self.geometry_type_combo.currentText() or "Cuboid"
        if self.radio_2d.isChecked():
            return self.geometry_type_combo.currentText() or "Rectangle"
        return "Interval"

    def _on_geometry_type_changed(self, text):
        self._update_domain_fields_visibility()
        # Rebuild BCs: the boundary-condition layout (sides / unified /
        # per-edge) depends on which geometry type is now selected.
        if hasattr(self, 'bc_main_layout'):
            self._build_bc_inputs(self.num_outputs_spin.value())
            self._build_weight_inputs(self.num_outputs_spin.value())

    def _update_domain_fields_visibility(self):
        """Show the plain x/y/z domain-bounds rows only for the box shapes
        (Interval/Rectangle/Cuboid). For every other shape, those rows don't
        mean anything -- e.g. an Ellipse needs a center + semi-axes, not an
        x/y range -- so hide them and show that shape's own parameter panel
        instead. The "t:" row always stays, after whichever of the two is
        showing."""
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked()
        geom_type = self._current_geometry_type()
        box_shape = geom_type in ("Interval", "Rectangle", "Cuboid")
        self.x_row_widget.setVisible(box_shape)
        self.y_row_widget.setVisible(box_shape and (is_2d or is_3d))
        self.z_row_widget.setVisible(box_shape and is_3d)
        for panel in self._geom_shape_panels:
            panel.setVisible(False)
        panel_map = {
            "Disk": self.geom_panel_disk,
            "Ellipse": self.geom_panel_ellipse,
            "Triangle": self.geom_panel_triangle,
            "Polygon": self.geom_panel_polygon,
            "Sphere": self.geom_panel_sphere,
        }
        panel = panel_map.get(geom_type)
        if panel is not None:
            panel.setVisible(True)

    def _on_geom_vertices_changed(self, text):
        # Triangle/Polygon edge count changed -- rebuild the per-edge BC rows.
        geom_type = self._current_geometry_type()
        if geom_type in ("Triangle", "Polygon") and hasattr(self, 'bc_main_layout'):
            self._build_bc_inputs(self.num_outputs_spin.value())
            self._build_weight_inputs(self.num_outputs_spin.value())

    @staticmethod
    def _parse_vertices(text):
        """'x1,y1;x2,y2;...' -> [(x1,y1), (x2,y2), ...]. Skips malformed pairs."""
        verts = []
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(",")
            if len(parts) != 2:
                continue
            try:
                verts.append((float(parts[0].strip()), float(parts[1].strip())))
            except ValueError:
                continue
        return verts

    @staticmethod
    def _is_axis_aligned_rectangle(verts):
        """True if these 4 vertices form an axis-aligned rectangle -- mirrors
        DeepXDE's own Rectangle.is_valid() check (every edge is purely
        horizontal or purely vertical), so callers can warn before DeepXDE's
        Polygon constructor rejects it with "The polygon is a rectangle.
        Use Rectangle instead."."""
        if len(verts) != 4:
            return False
        for i in range(4):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % 4]
            if abs((x1 - x0) * (y1 - y0)) > 1e-8:
                return False
        return True

    def _on_quick_example_selected(self, text):
        if text == "None":
            # Leave the Boundary Conditions panel exactly as it is -- rows
            # are no longer tied to template identity, so whatever's showing
            # (template-seeded or hand-built, edited or not) just becomes
            # the starting point for custom mode. Switching TO a template
            # (below) still always resets the list to that template's
            # defaults via _populate_locked_bc_entries_from_legacy.
            self._update_bc_mode_visibility()
            return
        # _on_template_selected() already calls _update_bc_mode_visibility()
        # itself once the panel and weight rows are in their final state
        # (including any template-specific weight default, e.g. 2D Heat's
        # right-Dirichlet BC) -- an extra call here would rebuild the
        # weight rows a second time and silently wipe such a default back
        # to 1.0, so it's intentionally not repeated.
        self._on_template_selected(text)

    def _auto_configure_ea(self, ref_dir):
        """Auto-configure error analysis when a template with ground truth files is loaded."""
        import glob, re, numpy as np
        if not ref_dir or not os.path.isdir(ref_dir):
            return
        # Ground-truth snapshots for error analysis must match "t_<number>.txt"
        # exactly -- a sibling file like "t_1.0_LessData.txt" (a thinned-down
        # version of the same snapshot meant only as an easier Inverse
        # observation set, see below) is intentionally NOT another ground
        # truth snapshot and must not be swept in just because it also starts
        # with "t_" and ends with ".txt".
        _gt_name_re = re.compile(r'^t_[0-9]+(\.[0-9]+)?\.txt$')
        txt_files = sorted(
            fp for fp in glob.glob(os.path.join(ref_dir, 't_*.txt'))
            if _gt_name_re.match(os.path.basename(fp))
        )
        if not txt_files:
            return
        valid_files = []
        for fp in txt_files:
            try:
                d = np.loadtxt(fp)
                if d.ndim == 1: d = d.reshape(1, -1)
                is_2d = self.radio_2d.isChecked()
                is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
                t_val = float(d[0, 3]) if is_3d else (float(d[0, 2]) if is_2d else float(d[0, 1]))
                valid_files.append((t_val, fp))
            except Exception:
                continue
        if not valid_files:
            return
        valid_files.sort(key=lambda x: x[0])
        # Only offer/select reference snapshots within the currently
        # configured time domain -- e.g. 2D Allen-Cahn (Wight & Zhao) in
        # Inverse mode trains over a shorter t range than Forward, so it
        # should not compare against or auto-load a snapshot beyond that.
        if hasattr(self, 't_max'):
            _tmax_cur = self.t_max.value()
            _filtered = [vf for vf in valid_files if vf[0] <= _tmax_cur + 1e-6]
            if _filtered:
                valid_files = _filtered
        self._ea_settings = {
            'files': valid_files,
            'do_line': True,
            'do_surface': True,
            'do_l2': True,
            'do_mse': True,
            'do_max': True,
        }
        self.log_box.append(f"✅ Error analysis auto-configured — {len(valid_files)} ground truth files from template")
        # Auto-select the end-time (largest t) reference file as the Inverse
        # observed-data file, so the user doesn't have to browse for it. If a
        # thinned "<name>_LessData.txt" sibling of that snapshot exists in
        # the same folder, prefer it for the Inverse default instead: fewer
        # observation points make the inverse fit numerically easier, while
        # the full snapshot above stays untouched as the error-analysis
        # ground truth.
        if hasattr(self, 'inv_data_path'):
            _end_time_file = valid_files[-1][1]
            _less_data_variant = _end_time_file[:-4] + "_LessData.txt"
            _inv_default_file = _less_data_variant if os.path.isfile(_less_data_variant) else _end_time_file
            self.inv_data_path.setText(_inv_default_file)
            self.log_box.append(
                f"📂 Inverse observed data auto-loaded: {os.path.basename(_inv_default_file)} "
                f"(t={valid_files[-1][0]:.4g})")

    # ── Plot type change ──────────────────────────────────────
    def _on_plot_type_changed(self, text):
        if text == "Line (time steps)":
            self._on_line_plot_settings()
    

    def _on_line_plot_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Line Plot Settings")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        info = QLabel("Select number of time steps to plot.")
        self._register_style(info, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        layout.addWidget(info)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Time steps to show:"))
        steps_spin = QSpinBox()
        steps_spin.setRange(2, 20); steps_spin.setValue(self.timesteps_spin.value())
        steps_spin.setFixedWidth(80)
        steps_row.addStretch(); steps_row.addWidget(steps_spin)
        layout.addLayout(steps_row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _on_cancel():
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()

        cancel_btn.clicked.connect(_on_cancel)

        def _on_ok():
            self.timesteps_spin.setValue(steps_spin.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_error_analysis_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Error Analysis Settings")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)

        # Source selection
        src_group = QGroupBox("Reference Data Source")
        src_layout = QVBoxLayout(src_group)
        self._ea_radio_expr = QRadioButton("Analytical Expression")
        self._ea_radio_csv  = QRadioButton("CSV File")
        self._ea_radio_expr.setChecked(True)
        src_layout.addWidget(self._ea_radio_expr)
        src_layout.addWidget(self._ea_radio_csv)
        layout.addWidget(src_group)

        # Expression section
        self._ea_expr_widget = QWidget()
        expr_layout = QVBoxLayout(self._ea_expr_widget)
        expr_layout.setContentsMargins(0, 0, 0, 0)
        expr_layout.addWidget(QLabel("Analytical expression (simplified syntax):"))
        self._ea_expr_input = QLineEdit()
        self._ea_expr_input.setPlaceholderText("e.g. sin(pi*x)*exp(-0.4*pi**2*t)")
        self._ea_expr_input.setFixedHeight(28)
        expr_layout.addWidget(self._ea_expr_input)
        expr_layout.addWidget(QLabel("Evaluation times (comma separated):"))
        self._ea_times_input = QLineEdit()
        self._ea_times_input.setText("0.25, 0.5, 0.75, 1.0")
        self._ea_times_input.setFixedHeight(28)
        expr_layout.addWidget(self._ea_times_input)
        layout.addWidget(self._ea_expr_widget)

        # CSV section
        self._ea_csv_widget = QWidget()
        csv_layout = QVBoxLayout(self._ea_csv_widget)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_layout.addWidget(QLabel("CSV file:"))
        csv_row = QHBoxLayout()
        self._ea_csv_path = QLineEdit()
        self._ea_csv_path.setPlaceholderText("Browse for CSV file...")
        self._ea_csv_path.setFixedHeight(28)
        csv_row.addWidget(self._ea_csv_path)
        csv_browse = QPushButton("Browse")
        csv_browse.setFixedHeight(28); csv_browse.setFixedWidth(65)
        csv_browse.clicked.connect(lambda: self._ea_csv_path.setText(
            QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV (*.csv *.txt)")[0]))
        csv_row.addWidget(csv_browse)
        csv_layout.addLayout(csv_row)
        info = QLabel("Expected format — 1D: x, t, u  |  2D: x, y, t, u")
        self._register_style(info, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        csv_layout.addWidget(info)
        self._ea_csv_widget.setVisible(False)
        layout.addWidget(self._ea_csv_widget)

        # Toggle visibility
        self._ea_radio_expr.toggled.connect(lambda c: (
            self._ea_expr_widget.setVisible(c),
            self._ea_csv_widget.setVisible(not c)
        ))

        # Time snapshot for 2D
        self._ea_2d_widget = QWidget()
        td_layout = QHBoxLayout(self._ea_2d_widget)
        td_layout.setContentsMargins(0, 0, 0, 0)
        td_layout.addWidget(QLabel("2D snapshot time:"))
        self._ea_2d_time = QDoubleSpinBox()
        self._ea_2d_time.setRange(0.0, 1e6); self._ea_2d_time.setValue(0.5)
        self._ea_2d_time.setFixedHeight(28); self._ea_2d_time.setFixedWidth(100)
        td_layout.addStretch(); td_layout.addWidget(self._ea_2d_time)
        self._ea_2d_widget.setVisible(self.radio_2d.isChecked())
        layout.addWidget(self._ea_2d_widget)

        # Error metrics
        metrics_group = QGroupBox("Error Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self._ea_abs  = QCheckBox("Absolute Error plot"); self._ea_abs.setChecked(True)
        self._ea_l2   = QCheckBox("L2 Relative Error");   self._ea_l2.setChecked(True)
        self._ea_mse  = QCheckBox("MSE");                  self._ea_mse.setChecked(True)
        self._ea_max  = QCheckBox("Max Error");            self._ea_max.setChecked(True)
        for w in [self._ea_abs, self._ea_l2, self._ea_mse, self._ea_max]:
            metrics_layout.addWidget(w)
        layout.addWidget(metrics_group)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Run Analysis")
        ok_btn.setStyleSheet("QPushButton { background: #1a4a6a; color: #74c0fc; font-weight: bold; border-radius: 4px; padding: 4px 12px; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _on_cancel():
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()

        cancel_btn.clicked.connect(_on_cancel)
        ok_btn.clicked.connect(lambda: self._run_error_analysis(dialog))
        dialog.exec()

    def _run_error_analysis(self, dialog):
        save_dir = self.save_dir_input.text().strip()
        if not save_dir:
            self.log_box.append("❌ Please set a save directory first.")
            self.plot_type_combo.setCurrentText("Surface")
            dialog.reject()
            return

        use_expr = self._ea_radio_expr.isChecked()
        expr_raw = self._ea_expr_input.text().strip()
        csv_path = self._ea_csv_path.text().strip()

        if use_expr and not expr_raw:
            self.log_box.append("❌ Please enter an analytical expression."); return
        if not use_expr and not csv_path:
            self.log_box.append("❌ Please select a CSV file."); return

        self._ea_settings = {
            'use_expr': use_expr,
            'expr': expr_raw,
            'csv_path': csv_path,
            'times': self._ea_times_input.text().strip(),
            'snap_time': self._ea_2d_time.value(),
            'do_abs': self._ea_abs.isChecked(),
            'do_l2': self._ea_l2.isChecked(),
            'do_mse': self._ea_mse.isChecked(),
            'do_max': self._ea_max.isChecked(),
        }
        self.log_box.append("✅ Error analysis configured — will run after training completes.")
        dialog.accept()

    def _on_ea_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete!")
            plot_path = os.path.join(save_dir, "comparison_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.log_box.append("❌ Error analysis failed — check log.")

    def _add_ta_step_group(self, t_start=0.0, t_end=1.0, steps=10):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        t_start_sb = QDoubleSpinBox()
        t_start_sb.setRange(0.0, 1e6); t_start_sb.setValue(t_start)
        t_start_sb.setFixedHeight(26); t_start_sb.setFixedWidth(85)
        t_start_sb.setDecimals(2)
        row_layout.addWidget(t_start_sb)

        row_layout.addWidget(QLabel("→"))

        t_end_sb = QDoubleSpinBox()
        t_end_sb.setRange(0.0, 1e6); t_end_sb.setValue(t_end)
        t_end_sb.setFixedHeight(26); t_end_sb.setFixedWidth(85)
        t_end_sb.setDecimals(2)
        row_layout.addWidget(t_end_sb)

        row_layout.addWidget(QLabel("n="))

        steps_sb = QSpinBox()
        steps_sb.setRange(1, 500); steps_sb.setValue(steps)
        steps_sb.setFixedHeight(26); steps_sb.setFixedWidth(70)
        row_layout.addWidget(steps_sb)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        row_layout.addWidget(remove_btn)

        self.ta_groups_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget,
            't_start': t_start_sb,
            't_end': t_end_sb,
            'steps': steps_sb
        }
        self.ta_group_rows.append(row_data)

        def _remove():
            row_widget.deleteLater()
            if row_data in self.ta_group_rows:
                self.ta_group_rows.remove(row_data)
        remove_btn.clicked.connect(_remove)

    def _add_inverse_var_row(self, name=None, init=1.0, is_primary=None):
        """Add one trainable-variable row to the Inverse panel. The first
        (primary) row is always present and cannot be removed -- it is the
        one that participates in INVERSE_AUTO_CONST's automatic PDE-box
        substitution when a Quick Example is loaded, exactly like the
        single trainable_variable this generalizes. Rows beyond the first
        are optional, freely added/removed, and share the panel's single
        measured-data file / output selector / loss weight (all trainable
        variables are fit against the same shared observation dataset)."""
        if is_primary is None:
            is_primary = (len(self.inv_var_rows) == 0)
        if not name:
            name = f"trainable_variable_{len(self.inv_var_rows) + 1}"

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        name_edit = QLineEdit()
        name_edit.setText(name)
        name_edit.setFixedHeight(26)
        row_layout.addWidget(name_edit)

        row_layout.addWidget(QLabel("initial guess:"))
        init_spin = InitGuessLineEdit(init)
        init_spin.setFixedHeight(26); init_spin.setFixedWidth(85)
        row_layout.addWidget(init_spin)

        remove_btn = None
        if is_primary:
            name_edit.editingFinished.connect(self._on_inv_param_name_changed)
        else:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
            remove_btn.setStyleSheet(
                "QPushButton { color: #ff8787; background: transparent; border: none; }")
            row_layout.addWidget(remove_btn)

        self.inv_vars_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget,
            'name': name_edit,
            'init': init_spin,
            'is_primary': is_primary,
        }
        self.inv_var_rows.append(row_data)
        if is_primary:
            # Keep the legacy single-variable attribute names as aliases to
            # the primary row's widgets -- every existing call site that
            # reads/writes self.inv_param_name / self.inv_param_init (PDE
            # auto-substitution, config build/restore) keeps working
            # unchanged, including after a config-restore rebuild replaces
            # the row widgets these point to.
            self.inv_param_name = name_edit
            self.inv_param_init = init_spin

        if remove_btn is not None:
            def _remove():
                row_widget.deleteLater()
                if row_data in self.inv_var_rows:
                    self.inv_var_rows.remove(row_data)
                self._update_inv_multi_var_hint()
            remove_btn.clicked.connect(_remove)

        self._update_inv_multi_var_hint()
        return row_data

    def _update_inv_multi_var_hint(self):
        if hasattr(self, 'inv_multi_var_hint'):
            self.inv_multi_var_hint.setVisible(len(self.inv_var_rows) > 1)

    def _build_inverse_variables_json(self):
        import json
        variables = []
        for r in self.inv_var_rows:
            nm = r['name'].text().strip() or f"trainable_variable_{len(variables) + 1}"
            variables.append({'name': nm, 'init': r['init'].value()})
        if not variables:
            variables.append({'name': 'trainable_variable_1', 'init': 1.0})
        return json.dumps(variables)

    def _add_inverse_data_row(self, path="", output_idx=0, weight=100.0, is_primary=None):
        """Add one measured-data-file row to the Inverse panel. The first
        (primary) row is always present -- it's the one the legacy single-
        file config fields (inverse_data_file / inverse_obs_output_idx /
        loss_weight_obs) mirror, so every existing call site and every
        previously-saved config keeps working unchanged. Rows beyond the
        first are optional, freely added/removed; unlike trainable
        variables (which all share one measured-data setup), each
        additional file here is its OWN observation dataset with its own
        "which output" selector and its own loss weight, since each
        becomes its own separate loss term."""
        if is_primary is None:
            is_primary = (len(self.inv_data_rows) == 0)

        row_widget = QWidget()
        row_v = QVBoxLayout(row_widget)
        row_v.setContentsMargins(0, 0, 0, 0)
        row_v.setSpacing(2)

        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_edit = QLineEdit()
        path_edit.setText(path)
        path_edit.setPlaceholderText("Browse...")
        path_edit.setFixedHeight(28)
        path_row.addWidget(path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedHeight(28); browse_btn.setFixedWidth(65)
        browse_btn.clicked.connect(lambda _checked=False, e=path_edit: self._on_browse_inv_data(e))
        path_row.addWidget(browse_btn)

        remove_btn = None
        if not is_primary:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedHeight(28); remove_btn.setFixedWidth(26)
            remove_btn.setStyleSheet(
                "QPushButton { color: #ff8787; background: transparent; border: none; }")
            path_row.addWidget(remove_btn)
        path_row_widget = QWidget()
        path_row_widget.setLayout(path_row)
        row_v.addWidget(path_row_widget)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.addWidget(QLabel("output:"))
        output_combo = QComboBox()
        output_combo.setFixedHeight(26)
        _n_out = self.num_outputs_spin.value() if hasattr(self, 'num_outputs_spin') else 1
        for _i in range(_n_out):
            _name = (self.output_name_inputs[_i].text()
                      if hasattr(self, 'output_name_inputs') and _i < len(self.output_name_inputs)
                      else f"u{_i+1}")
            output_combo.addItem(f"Output {_i+1} ({_name})")
        if 0 <= output_idx < output_combo.count():
            output_combo.setCurrentIndex(output_idx)
        meta_row.addWidget(output_combo)
        meta_row.addWidget(QLabel("Data loss weight:"))
        weight_edit = SciLineEdit(weight)
        weight_edit.setFixedHeight(26); weight_edit.setFixedWidth(85)
        meta_row.addWidget(weight_edit)
        meta_row.addStretch()
        meta_row_widget = QWidget()
        meta_row_widget.setLayout(meta_row)
        row_v.addWidget(meta_row_widget)

        self.inv_data_files_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget,
            'path': path_edit,
            'browse': browse_btn,
            'output_combo': output_combo,
            'weight': weight_edit,
            'is_primary': is_primary,
        }
        self.inv_data_rows.append(row_data)
        if is_primary:
            # Keep the legacy single-file attribute names as aliases to the
            # primary row's widgets -- every existing call site that reads
            # or writes self.inv_data_path / self.inv_obs_output_combo /
            # self.inv_obs_weight (config build/restore, the template
            # auto-load-observed-data helper, the scheduler weight-string
            # builder) keeps working unchanged, including after a config-
            # restore rebuild replaces the row widgets these point to.
            self.inv_data_path = path_edit
            self.inv_data_browse = browse_btn
            self.inv_obs_output_combo = output_combo
            self.inv_obs_weight = weight_edit

        if remove_btn is not None:
            def _remove():
                row_widget.deleteLater()
                if row_data in self.inv_data_rows:
                    self.inv_data_rows.remove(row_data)
                self._update_inv_multi_data_hint()
            remove_btn.clicked.connect(_remove)

        self._update_inv_multi_data_hint()
        return row_data

    def _update_inv_multi_data_hint(self):
        if hasattr(self, 'inv_multi_data_hint'):
            self.inv_multi_data_hint.setVisible(len(self.inv_data_rows) > 1)

    def _build_inverse_obs_files_json(self):
        import json
        files = []
        for r in self.inv_data_rows:
            _oi = r['output_combo'].currentIndex()
            files.append({
                'path': r['path'].text().strip(),
                'output_idx': _oi if _oi >= 0 else 0,
                'weight': r['weight'].value(),
            })
        if not files:
            files.append({'path': '', 'output_idx': 0, 'weight': 100.0})
        return json.dumps(files)

    def _set_combo_data(self, combo, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _on_adapt_changed(self, text):
        self.rar_widget.setVisible(text == "Residual-based Adaptive Refinement (RAR)")
        self.ta_widget.setVisible(text == "Time Adaptive")
    
    def _build_ta_step_groups_json(self):
        import json
        groups = []
        for r in self.ta_group_rows:
            groups.append({
                't_start': r['t_start'].value(),
                't_end':   r['t_end'].value(),
                'steps':   r['steps'].value()
            })
        if not groups:
            # fallback to single group from domain
            groups.append({
                't_start': self.t_min.value(),
                't_end':   self.t_max.value(),
                'steps':   self.ta_steps.value()
            })
        return json.dumps(groups)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if folder:
            self.save_dir_input.setText(folder)

    def _on_opt2_changed(self, text):
        self.lbfgs_widget.setVisible(text == "lbfgs")

    def _on_lbfgs_default_changed(self, state):
        self.lbfgs_manual_widget.setVisible(state != 2)

    def _on_bc_left_changed(self, text):
        pass

    def _on_bc_right_changed(self, text):
        pass

    # ── Build PDE inputs ──────────────────────────────────────
    def _build_pde_inputs(self, n):
        for i in reversed(range(self.pde_main_layout.count())):
            w = self.pde_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.pde_inputs.clear()
        self.output_name_inputs.clear()

        is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False

        for i in range(n):
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel(f"Output {i+1} name:"))
            name_input = QLineEdit()
            name_input.setText(["u", "v", "w", "p"][i] if i < 4 else f"u{i+1}")
            name_input.setFixedHeight(26); name_input.setFixedWidth(55)
            name_row.addWidget(name_input); name_row.addStretch()
            self.output_name_inputs.append(name_input)
            nw = QWidget(); nw.setLayout(name_row)
            self.pde_main_layout.addWidget(nw)

            self.pde_main_layout.addWidget(QLabel(f"PDE {i+1} (residual = 0):"))
            pde_inp = QLineEdit()
            if is_3d:
                pde_inp.setText("du_t - 0.4 * (du_xx + du_yy + du_zz)" if i == 0 else "dv_t - 0.1 * (dv_xx + dv_yy + dv_zz)")
            elif is_2d:
                pde_inp.setText("du_t - 0.0001 * (du_xx + du_yy) + 1 * (u**3 - u)" if i == 0 else "dv_t - 0.0001 * (dv_xx + dv_yy) + 1 * (v**3 - v)")
            else:
                pde_inp.setText("du_t - 0.4 * du_xx" if i == 0 else "dv_t - 0.1 * dv_xx")
            pde_inp.setFixedHeight(28)
            self.pde_inputs.append(pde_inp)
            self.pde_main_layout.addWidget(pde_inp)

        names_ex = ", ".join([["u","v","w","p"][i] if i < 4 else f"u{i+1}" for i in range(n)])
        if is_3d:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x  du_y→∂u/∂y  du_z→∂u/∂z  du_t→∂u/∂t\n"
                         f"du_xx→∂²u/∂x²  du_yy→∂²u/∂y²  du_zz→∂²u/∂z²  du_tt→∂²u/∂t²\n"
                         f"du_xy→∂²u/∂x∂y  du_xz→∂²u/∂x∂z  du_yz→∂²u/∂y∂z\n"
                         f"du_xt→∂²u/∂x∂t  du_yt→∂²u/∂y∂t  du_zt→∂²u/∂z∂t\n"
                         f"du_xxxx→∂⁴u/∂x⁴  du_yyyy→∂⁴u/∂y⁴  du_zzzz→∂⁴u/∂z⁴\n"
                         f"du_xxyy→∂⁴u/∂x²∂y²  du_xxzz→∂⁴u/∂x²∂z²  du_yyzz→∂⁴u/∂y²∂z²\n"
                         f"du_xxtt→∂⁴u/∂x²∂t²  du_yytt→∂⁴u/∂y²∂t²  du_zztt→∂⁴u/∂z²∂t²\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi\n"
                         f"e.g. 3D Heat:       du_t - 0.4*(du_xx+du_yy+du_zz)\n"
                         f"e.g. 3D Reaction:   du_t - 0.001*(du_xx+du_yy+du_zz) + u**3 - u")
        elif is_2d:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x  du_y→∂u/∂y  du_t→∂u/∂t\n"
                         f"du_xx→∂²u/∂x²  du_yy→∂²u/∂y²  du_xy→∂²u/∂x∂y\n"
                         f"du_tt→∂²u/∂t²  du_xt→∂²u/∂x∂t  du_yt→∂²u/∂y∂t\n"
                         f"du_xxxx→∂⁴u/∂x⁴  du_yyyy→∂⁴u/∂y⁴\n"
                         f"du_xxyy→∂⁴u/∂x²∂y²  du_xxtt→∂⁴u/∂x²∂t²\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi\n"
                         f"e.g. Allen-Cahn 2D: du_t - 0.001*(du_xx+du_yy) + u**3 - u\n"
                         f"e.g. sin(pi*u)*du_xx + cos(u)*du_yy")
        else:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x  du_t→∂u/∂t\n"
                         f"du_xx→∂²u/∂x²  du_tt→∂²u/∂t²  du_xt→∂²u/∂x∂t\n"
                         f"du_xxxx→∂⁴u/∂x⁴  du_tttt→∂⁴u/∂t⁴  du_xxtt→∂⁴u/∂x²∂t²\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi\n"
                         f"e.g. Diffusion:      du_t - 0.4*du_xx\n"
                         f"e.g. Burgers:        du_t + u*du_x - 0.01*du_xx\n"
                         f"e.g. 4th-order:      du_t - (du_xx - du_xxxx)\n"
                         f"e.g. Nonlinear:      du_t - sin(u)*du_xx")
            
        # Templates + derivative reference row
        tmpl_ref_row = QHBoxLayout()

        hint_toggle = QCheckBox("📖 Show derivative reference")
        hint_toggle.setChecked(False)
        self._register_style(hint_toggle, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        tmpl_ref_row.addWidget(hint_toggle)

        tmpl_ref_row.addStretch()

        tmpl_ref_row_widget = QWidget()
        tmpl_ref_row_widget.setLayout(tmpl_ref_row)
        self.pde_main_layout.addWidget(tmpl_ref_row_widget)

        hint = QLabel(hint_text)
        self._register_style(hint, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        hint.setWordWrap(True)
        hint.setVisible(False)
        self.pde_main_layout.addWidget(hint)

        hint_toggle.stateChanged.connect(lambda state, h=hint: h.setVisible(state == 2))

    # ── Build BC inputs ───────────────────────────────────────
    def _build_bc_inputs(self, n):
        for i in reversed(range(self.bc_main_layout.count())):
            w = self.bc_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.bc_left_types.clear();   self.bc_left_vals.clear();   self.bc_left_active.clear();   self.bc_left_deriv.clear()
        self.bc_right_types.clear();  self.bc_right_vals.clear();  self.bc_right_active.clear();  self.bc_right_deriv.clear()
        self.bc_bottom_types.clear(); self.bc_bottom_vals.clear(); self.bc_bottom_active.clear(); self.bc_bottom_deriv.clear()
        self.bc_top_types.clear();    self.bc_top_vals.clear();    self.bc_top_active.clear();    self.bc_top_deriv.clear()
        self.bc_front_types.clear();  self.bc_front_vals.clear();  self.bc_front_active.clear()
        self.bc_back_types.clear();   self.bc_back_vals.clear();   self.bc_back_active.clear()
        self.bc_boundary_types.clear(); self.bc_boundary_vals.clear(); self.bc_boundary_active.clear()
        self.bc_edge_types.clear();   self.bc_edge_vals.clear();   self.bc_edge_active.clear()
        self.ic_inputs.clear();       self.ic_active.clear()
        if not hasattr(self, 'ic_from_file'): self.ic_from_file = []
        if not hasattr(self, 'ic_file_paths'): self.ic_file_paths = []
        self.ic_from_file.clear()
        self.ic_file_paths.clear()
        self._2d_bc_widgets.clear()
        if not hasattr(self, '_3d_bc_widgets'): self._3d_bc_widgets = []
        self._3d_bc_widgets.clear()
        if not hasattr(self, '_shape_bc_wrappers'): self._shape_bc_wrappers = []
        self._shape_bc_wrappers.clear()

        is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
        geom_type = self._current_geometry_type()
        box_shape = geom_type in ("Interval", "Rectangle", "Cuboid")
        unified_shape = geom_type in ("Disk", "Ellipse", "Sphere")
        edge_shape = geom_type in ("Triangle", "Polygon")

        def _make_bc_block(label_txt, types_list, vals_list, active_list, deriv_list, x_pos):
            """Create a BC row: checkbox + type combo + value spinbox + deriv checkbox"""
            active = QCheckBox(label_txt); active.setChecked(True)
            active_list.append(active)
            self.bc_main_layout.addWidget(active)

            bc_type = QComboBox(); bc_type.addItems(["Dirichlet", "Neumann", "Periodic"]); bc_type.setFixedHeight(26)
            types_list.append(bc_type)
            self.bc_main_layout.addWidget(bc_type)

            bc_val = QDoubleSpinBox(); bc_val.setRange(-1000, 1000); bc_val.setValue(0.0); bc_val.setFixedHeight(26)
            vals_list.append(bc_val)
            self.bc_main_layout.addWidget(bc_val)

            bc_deriv = QCheckBox("  + derivative periodic BC"); bc_deriv.setChecked(False); bc_deriv.setVisible(False)
            deriv_list.append(bc_deriv)
            self.bc_main_layout.addWidget(bc_deriv)

            bc_type.currentTextChanged.connect(
                lambda t, vw=bc_val, dw=bc_deriv: (
                    vw.setVisible(t != "Periodic"),
                    dw.setVisible(t == "Periodic"),
                    self._build_weight_inputs(self.num_outputs_spin.value())
                )
            )
            bc_deriv.stateChanged.connect(
                lambda s: self._build_weight_inputs(self.num_outputs_spin.value())
            )

        def _make_bc_block_simple(label_txt, types_list, vals_list, active_list):
            """Create a BC row with no Periodic option and no derivative toggle --
            used for single-boundary shapes (Disk/Ellipse/Sphere) and polygon/
            triangle edges, where there's no natural 'paired opposite side'."""
            active = QCheckBox(label_txt); active.setChecked(True)
            active_list.append(active)
            self.bc_main_layout.addWidget(active)

            bc_type = QComboBox(); bc_type.addItems(["Dirichlet", "Neumann"]); bc_type.setFixedHeight(26)
            types_list.append(bc_type)
            self.bc_main_layout.addWidget(bc_type)

            bc_val = QDoubleSpinBox(); bc_val.setRange(-1000, 1000); bc_val.setValue(0.0); bc_val.setFixedHeight(26)
            vals_list.append(bc_val)
            self.bc_main_layout.addWidget(bc_val)

            bc_type.currentTextChanged.connect(
                lambda t: self._build_weight_inputs(self.num_outputs_spin.value())
            )

        for i in range(n):
            name = ["u", "v", "w", "p"][i] if i < 4 else f"u{i+1}"
            sep = QLabel(f"── Output {i+1} ({name}) ──")
            self._register_style(sep, "hint", lambda css, _c='#505080', _e='margin-top: 4px; ': f"color: {_c}; {_e}{css}")
            self.bc_main_layout.addWidget(sep)

            # Master toggle for outputs > 0
            if i > 0:
                _bc_enable_cb = QCheckBox(f"Enable boundary conditions for {name}")
                _bc_enable_cb.setChecked(True)
                self._register_style(_bc_enable_cb, "hint", lambda css, _c='#ffa94d', _e='': f"color: {_c}; {_e}{css}")
                self.bc_main_layout.addWidget(_bc_enable_cb)
            else:
                _bc_enable_cb = None

            # Container widget for all BC+IC of this output
            _bc_container = QWidget()
            _bc_cont_layout = QVBoxLayout(_bc_container)
            _bc_cont_layout.setSpacing(3)
            _bc_cont_layout.setContentsMargins(0, 0, 0, 0)
            _main_layout_save = self.bc_main_layout
            self.bc_main_layout = _bc_cont_layout

            # Hardcoded per-shape BC rows (left/right/top/bottom/etc, or one
            # unified/per-edge row) -- wrapped so the whole block can be
            # hidden in one shot when a custom (non-template) problem is
            # active and the flexible Custom Boundary Conditions builder
            # takes over instead. See _update_bc_mode_visibility().
            _shape_bc_wrap = QWidget()
            _shape_bc_wrap_layout = QVBoxLayout(_shape_bc_wrap)
            _shape_bc_wrap_layout.setContentsMargins(0, 0, 0, 0)
            _shape_bc_wrap_layout.setSpacing(3)
            _shape_bc_save = self.bc_main_layout
            self.bc_main_layout = _shape_bc_wrap_layout

            if box_shape:
                _make_bc_block(f"BC left (x=xmin) for {name}", self.bc_left_types, self.bc_left_vals, self.bc_left_active, self.bc_left_deriv, "x_min")
                _idx = len(self.bc_left_types) - 1
                _make_bc_block(f"BC right (x=xmax) for {name}", self.bc_right_types, self.bc_right_vals, self.bc_right_active, self.bc_right_deriv, "x_max")
                # Hide right BC widgets when left is Periodic
                _right_active = self.bc_right_active[-1]
                _right_type   = self.bc_right_types[-1]
                _right_val    = self.bc_right_vals[-1]
                _right_deriv  = self.bc_right_deriv[-1]
                def _sync_right(t, ra=_right_active, rt=_right_type, rv=_right_val, rd=_right_deriv):
                    ra.setVisible(t != "Periodic")
                    rt.setVisible(t != "Periodic")
                    rv.setVisible(t != "Periodic")
                    rd.setVisible(False)
                self.bc_left_types[-1].currentTextChanged.connect(_sync_right)

                # BCs — bottom and top (2D Rectangle, or 3D Cuboid's y-axis)
                _2d_w = QWidget()
                _2d_l = QVBoxLayout(_2d_w); _2d_l.setContentsMargins(0,0,0,0); _2d_l.setSpacing(3)

                # Temporarily redirect bc_main_layout to _2d_l
                _2d_save = self.bc_main_layout
                self.bc_main_layout = _2d_l
                _make_bc_block(f"BC bottom (y=ymin) for {name}", self.bc_bottom_types, self.bc_bottom_vals, self.bc_bottom_active, self.bc_bottom_deriv, "y_min")
                _make_bc_block(f"BC top (y=ymax) for {name}", self.bc_top_types, self.bc_top_vals, self.bc_top_active, self.bc_top_deriv, "y_max")
                self.bc_main_layout = _2d_save
                # Hide top BC widgets when bottom is Periodic
                _top_active = self.bc_top_active[-1]
                _top_type   = self.bc_top_types[-1]
                _top_val    = self.bc_top_vals[-1]
                _top_deriv  = self.bc_top_deriv[-1]
                def _sync_top(t, ta=_top_active, tt=_top_type, tv=_top_val, td=_top_deriv):
                    ta.setVisible(t != "Periodic")
                    tt.setVisible(t != "Periodic")
                    tv.setVisible(t != "Periodic")
                    td.setVisible(False)
                self.bc_bottom_types[-1].currentTextChanged.connect(_sync_top)

                _2d_w.setVisible(is_2d or is_3d)
                self._2d_bc_widgets.append(_2d_w)
                self.bc_main_layout.addWidget(_2d_w)

                # Cuboid only — front and back (z-axis)
                if is_3d:
                    _3d_w = QWidget()
                    _3d_l = QVBoxLayout(_3d_w); _3d_l.setContentsMargins(0,0,0,0); _3d_l.setSpacing(3)
                    _3d_save = self.bc_main_layout
                    self.bc_main_layout = _3d_l
                    _make_bc_block_simple(f"BC front (z=zmin) for {name}", self.bc_front_types, self.bc_front_vals, self.bc_front_active)
                    _make_bc_block_simple(f"BC back (z=zmax) for {name}", self.bc_back_types, self.bc_back_vals, self.bc_back_active)
                    self.bc_main_layout = _3d_save
                    self._3d_bc_widgets.append(_3d_w)
                    self.bc_main_layout.addWidget(_3d_w)
            elif unified_shape:
                _make_bc_block_simple(f"Boundary for {name}", self.bc_boundary_types, self.bc_boundary_vals, self.bc_boundary_active)
            elif edge_shape:
                if geom_type == "Triangle":
                    _verts = self._parse_vertices(self.geom_triangle_verts_input.text())
                else:
                    _verts = self._parse_vertices(self.geom_polygon_verts_input.text())
                _n_edges = max(len(_verts), 3)
                self.bc_edge_types.append([]); self.bc_edge_vals.append([]); self.bc_edge_active.append([])
                for _e in range(_n_edges):
                    _make_bc_block_simple(
                        f"Edge {_e + 1} for {name}",
                        self.bc_edge_types[-1], self.bc_edge_vals[-1], self.bc_edge_active[-1]
                    )

            self.bc_main_layout = _shape_bc_save
            self._shape_bc_wrappers.append(_shape_bc_wrap)
            self.bc_main_layout.addWidget(_shape_bc_wrap)

            # IC
            ic_act = QCheckBox(f"IC for {name}"); ic_act.setChecked(True)
            self.ic_active.append(ic_act)
            self.bc_main_layout.addWidget(ic_act)

            ic_inp = QLineEdit()
            if is_2d:
                ic_inp.setText("sin(4*pi*x)*cos(4*pi*y)" if i == 0 else "cos(4*pi*x)*sin(4*pi*y)")
            else:
                ic_inp.setText("sin(pi*x)" if i == 0 else "cos(pi*x)")
            ic_inp.setFixedHeight(26)
            self.ic_inputs.append(ic_inp)
            self.bc_main_layout.addWidget(ic_inp)

            # IC reference toggle
            ic_hint_toggle = QCheckBox("📖 Show IC reference")
            ic_hint_toggle.setChecked(False)
            self._register_style(ic_hint_toggle, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
            self.bc_main_layout.addWidget(ic_hint_toggle)

            if is_2d:
                ic_hint_text = (
                    "Use x, y as spatial variables, t for time.\n"
                    "sin(pi*x)*cos(pi*y)  →  sin wave 2D\n"
                    "exp(-(x**2+y**2))    →  Gaussian\n"
                    "sin(4*pi*x)*cos(4*pi*y)  →  higher freq\n"
                    "0                    →  zero IC\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            else:
                ic_hint_text = (
                    "Use x as spatial variable, t for time.\n"
                    "sin(pi*x)       →  sine wave\n"
                    "exp(-x**2)      →  Gaussian\n"
                    "sin(4*pi*x)     →  higher frequency\n"
                    "x*(1-x)         →  parabola\n"
                    "0               →  zero IC\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            ic_hint = QLabel(ic_hint_text)
            self._register_style(ic_hint, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
            ic_hint.setWordWrap(True)
            ic_hint.setVisible(False)
            self.bc_main_layout.addWidget(ic_hint)
            ic_hint_toggle.stateChanged.connect(lambda state, h=ic_hint: h.setVisible(state == 2))

            # IC from file option — available for 1D, 2D, and 3D; the
            # expected no-header column format changes with dimension:
            # 1D: x,t,c   2D: x,y,t,c   3D: x,y,z,t,c. Expression stays the
            # default; this is opt-in per output, and only ever applies to
            # output 0's IC (single global IC file per problem).
            _ic_file_fmt = "x,y,z,t,c" if is_3d else ("x,y,t,c" if is_2d else "x,t,c")
            ic_file_cb = QCheckBox(f"📂 Load IC from file ({_ic_file_fmt} format)")
            ic_file_cb.setChecked(False)
            self._register_style(ic_file_cb, "hint", lambda css, _c='#ffa94d', _e='': f"color: {_c}; {_e}{css}")
            self.bc_main_layout.addWidget(ic_file_cb)
            self.ic_from_file.append(ic_file_cb)

            ic_file_widget = QWidget()
            ic_file_layout = QHBoxLayout(ic_file_widget)
            ic_file_layout.setContentsMargins(0, 0, 0, 0)
            ic_file_path = QLineEdit()
            ic_file_path.setPlaceholderText(f"Browse for IC file ({_ic_file_fmt})...")
            ic_file_path.setFixedHeight(26)
            ic_file_layout.addWidget(ic_file_path)
            ic_file_browse = QPushButton("Browse")
            ic_file_browse.setFixedHeight(26); ic_file_browse.setFixedWidth(65)
            def _make_browse(path_edit):
                def _browse():
                    f, _ = QFileDialog.getOpenFileName(None, "Select IC file", "", "Data files (*.txt *.csv *.dat)")
                    if f: path_edit.setText(f)
                return _browse
            ic_file_browse.clicked.connect(_make_browse(ic_file_path))
            ic_file_layout.addWidget(ic_file_browse)
            ic_file_widget.setVisible(False)
            self.bc_main_layout.addWidget(ic_file_widget)
            self.ic_file_paths.append(ic_file_path)

            ic_file_cb.stateChanged.connect(
                lambda state, w=ic_file_widget, inp=ic_inp, act=ic_act:
                (w.setVisible(state == 2), inp.setVisible(state != 2),
                 act.setChecked(state != 2),
                 self._update_ic_pretrain_visibility())
            )

            # Restore layout and add container
            self.bc_main_layout = _main_layout_save
            self.bc_main_layout.addWidget(_bc_container)

            # Wire master toggle
            if _bc_enable_cb is not None:
                def _make_toggle(cont, la, ra, ica):
                    def _tog(state):
                        cont.setVisible(state == 2)
                        if la: la.setChecked(state == 2)
                        if ra: ra.setChecked(state == 2)
                        if ica: ica.setChecked(state == 2)
                        self._build_weight_inputs(self.num_outputs_spin.value())
                    return _tog
                _i_capture = i
                _bc_enable_cb.stateChanged.connect(_make_toggle(
                    _bc_container,
                    self.bc_left_active[_i_capture] if _i_capture < len(self.bc_left_active) else None,
                    self.bc_right_active[_i_capture] if _i_capture < len(self.bc_right_active) else None,
                    self.ic_active[_i_capture] if _i_capture < len(self.ic_active) else None
                ))

        self._update_bc_mode_visibility()

    def _is_custom_problem(self):
        """True when no Quick Example template is selected -- i.e. the user
        is building their own problem from scratch, and every entry in the
        Boundary Conditions panel is theirs to add/edit/remove. False means
        a template governs BCs; the panel still shows them (in the same
        style), just read-only -- see _populate_locked_bc_entries_from_legacy()."""
        if not hasattr(self, 'quick_examples_combo'):
            return True
        return self.quick_examples_combo.currentText() == "None"

    def _update_bc_mode_visibility(self):
        """The hardcoded per-side widgets (self.bc_left_types etc.) never
        appear in the UI any more -- they're kept alive only as a
        display-source for templates (see _populate_locked_bc_entries_from_legacy)
        -- so just keep them hidden. The single Boundary Conditions panel is
        always shown, always editable, and is itself what drives training
        (see codegen.py's custom_bc_json path) -- a template just pre-fills
        it as a starting point. This only updates the note text and keeps
        the per-BC weight rows in the Loss Weights panel matching whatever
        is currently in the list. Called after every _build_bc_inputs()
        rebuild, whenever the Quick Example selection changes, and whenever
        a BC row is added or removed."""
        for w in getattr(self, '_shape_bc_wrappers', []):
            w.setVisible(False)
        if not hasattr(self, 'custom_bc_group'):
            return
        is_custom = self._is_custom_problem()
        if is_custom:
            self.custom_bc_note.setText(
                "No template selected — build your own boundary conditions.")
        else:
            template_name = self.quick_examples_combo.currentText() if hasattr(self, 'quick_examples_combo') else ''
            self.custom_bc_note.setText(
                f"Pre-filled from the \"{template_name}\" template.")
        self.add_custom_bc_btn.setEnabled(True)
        self.add_custom_bc_btn.setToolTip("")
        if hasattr(self, 'weights_main_layout'):
            self._build_weight_inputs(self.num_outputs_spin.value())

    # ── Boundary Conditions builder (all problems) ─────────────
    # DeepXDE BC classes offered in the "Type" dropdown, and which optional
    # rows each one needs. Location/value are plain Python expressions using
    # bare x/y/z (and, for Robin, u for the current solution value) -- same
    # convention as the PDE/IC expression fields -- combined by the eventual
    # codegen as `lambda x, on_boundary: on_boundary and (<location>)`, so
    # the user only has to say *which* boundary they mean, not re-derive
    # on_boundary from scratch.
    # Covers the full deepxde.icbc surface: every class listed at
    # https://deepxde.readthedocs.io/en/latest/modules/deepxde.icbc.html
    # except the abstract base BC and initial_conditions.IC (ICs keep their
    # own separate, already-flexible expression field above).
    CUSTOM_BC_TYPES = [
        ("Dirichlet BC", "dirichlet"),
        ("Neumann BC", "neumann"),
        ("Robin BC", "robin"),
        ("Periodic BC", "periodic"),
        ("Point Set BC (from file)", "pointset"),
        ("Point Set Operator BC (advanced, from file)", "pointset_operator"),
        ("Operator BC (advanced)", "operator"),
        ("Interface 2D BC (advanced, Rectangle/Polygon only)", "interface2d"),
    ]
    # Which optional rows each type needs.
    _BC_NEEDS_COMPONENT = {"dirichlet", "neumann", "robin", "periodic", "pointset"}
    _BC_NEEDS_LOCATION = {"dirichlet", "neumann", "robin", "periodic", "operator", "interface2d"}
    _BC_NEEDS_VALUE = {"dirichlet", "neumann", "robin", "operator", "interface2d", "pointset_operator"}
    _BC_NEEDS_PERIODIC_ROW = {"periodic"}
    _BC_NEEDS_POINTS_FILE = {"pointset", "pointset_operator"}
    _BC_NEEDS_ADVANCED_NOTE = {"operator", "pointset_operator"}
    _BC_NEEDS_INTERFACE_ROW = {"interface2d"}

    def _add_custom_bc_entry(self, bc_type="dirichlet", component=0, location="",
                              value="0", axis="x", deriv_order=0, points_file="",
                              location2="", direction="normal", locked=False):
        entry_widget = QWidget()
        entry_layout = QVBoxLayout(entry_widget)
        entry_layout.setSpacing(3)
        entry_layout.setContentsMargins(0, 0, 0, 0)

        entry_num = len(self.custom_bc_list) + 1
        header_row = QHBoxLayout()
        header_txt = f"── BC {entry_num} ──" + (" \U0001F512" if locked else "")
        header_lbl = QLabel(header_txt)
        _header_color = "#8a8a8a" if locked else "#a0c4ff"
        self._register_style(header_lbl, "hint", lambda css, _c=_header_color: f"color: {_c}; {css}")
        header_row.addWidget(header_lbl)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        remove_btn.setVisible(not locked)
        header_row.addStretch(); header_row.addWidget(remove_btn)
        entry_layout.addLayout(header_row)

        # Type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        type_combo = QComboBox()
        for label_txt, key in self.CUSTOM_BC_TYPES:
            type_combo.addItem(label_txt, key)
        self._set_combo_data(type_combo, bc_type)
        type_combo.setFixedHeight(26); type_combo.setFixedWidth(210)
        type_row.addStretch(); type_row.addWidget(type_combo)
        entry_layout.addLayout(type_row)

        # Output (component)
        comp_widget = QWidget()
        comp_row = QHBoxLayout(comp_widget)
        comp_row.setContentsMargins(0, 0, 0, 0)
        comp_row.addWidget(QLabel("Output #:"))
        comp_spin = QSpinBox(); comp_spin.setRange(0, 7); comp_spin.setValue(component)
        comp_spin.setFixedHeight(26); comp_spin.setFixedWidth(60)
        comp_spin.setToolTip("0-indexed output this BC applies to (0 = first output, 1 = second, ...)")
        comp_row.addStretch(); comp_row.addWidget(comp_spin)
        entry_layout.addWidget(comp_widget)

        # Location (hidden for PointSet/PointSetOperator, which take points
        # from a file instead)
        loc_widget = QWidget()
        loc_layout = QVBoxLayout(loc_widget)
        loc_layout.setContentsMargins(0, 0, 0, 0); loc_layout.setSpacing(2)
        loc_row = QHBoxLayout()
        loc_row.addWidget(QLabel("Where:"))
        loc_edit = QLineEdit(location)
        loc_edit.setPlaceholderText("e.g. x <= 1e-8   (boolean expression in x, y, z)")
        loc_edit.setFixedHeight(26)
        loc_row.addWidget(loc_edit)
        loc_layout.addLayout(loc_row)
        loc_hint_toggle = QCheckBox("📖 Show location examples")
        loc_hint_toggle.setChecked(False)
        self._register_style(loc_hint_toggle, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        loc_layout.addWidget(loc_hint_toggle)
        loc_hint = QLabel(
            "True/False expression in x, y, z picking out the boundary you\n"
            "mean (only y if 2D/3D, only z if 3D) -- points not actually on\n"
            "the geometry's boundary are already excluded automatically.\n"
            "x <= 1e-8            → left edge (x = xmin)\n"
            "x >= 0.999           → right edge (x close to xmax = 1.0)\n"
            "y <= 1e-8            → bottom edge\n"
            "np.isclose(x**2 + y**2, 0.25)  → circle boundary, radius 0.5"
        )
        self._register_style(loc_hint, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        loc_hint.setWordWrap(True)
        loc_hint.setVisible(False)
        loc_layout.addWidget(loc_hint)
        loc_hint_toggle.stateChanged.connect(lambda s, h=loc_hint: h.setVisible(s == 2))
        entry_layout.addWidget(loc_widget)

        # Value/function (hidden for Periodic, which has no func; hidden for
        # PointSet, whose values come from its file)
        val_widget = QWidget()
        val_layout = QVBoxLayout(val_widget)
        val_layout.setContentsMargins(0, 0, 0, 0); val_layout.setSpacing(2)
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("Value:"))
        val_edit = QLineEdit(value)
        val_edit.setPlaceholderText("e.g. 0, sin(pi*y), x**2  (Robin: may also use u)")
        val_edit.setFixedHeight(26)
        val_row.addWidget(val_edit)
        val_layout.addLayout(val_row)
        entry_layout.addWidget(val_widget)

        # Periodic-only: which coordinate is periodic + derivative order
        periodic_widget = QWidget()
        periodic_layout = QHBoxLayout(periodic_widget)
        periodic_layout.setContentsMargins(0, 0, 0, 0)
        periodic_layout.addWidget(QLabel("Periodic in:"))
        axis_combo = QComboBox(); axis_combo.addItems(["x", "y", "z"])
        axis_combo.setCurrentText(axis); axis_combo.setFixedHeight(26); axis_combo.setFixedWidth(60)
        periodic_layout.addWidget(axis_combo)
        periodic_layout.addWidget(QLabel("Derivative order:"))
        deriv_spin = QSpinBox(); deriv_spin.setRange(0, 1); deriv_spin.setValue(deriv_order)
        deriv_spin.setFixedHeight(26); deriv_spin.setFixedWidth(50)
        periodic_layout.addWidget(deriv_spin)
        periodic_layout.addStretch()
        entry_layout.addWidget(periodic_widget)

        # PointSet / PointSetOperator: a data file of points (+ values,
        # unless PointSetOperator's Value row supplies the func instead) --
        # same convention as the existing "Load IC from file" option above.
        pointset_widget = QWidget()
        pointset_layout = QHBoxLayout(pointset_widget)
        pointset_layout.setContentsMargins(0, 0, 0, 0)
        pointset_path = QLineEdit(points_file)
        pointset_path.setPlaceholderText("Browse for points file (x[,y[,z]],value)...")
        pointset_path.setFixedHeight(26)
        pointset_layout.addWidget(pointset_path)
        pointset_browse = QPushButton("Browse")
        pointset_browse.setFixedHeight(26); pointset_browse.setFixedWidth(65)
        pointset_browse.clicked.connect(lambda: pointset_path.setText(
            QFileDialog.getOpenFileName(None, "Select points file", "", "Data files (*.txt *.csv *.dat)")[0]
            or pointset_path.text()))
        pointset_layout.addWidget(pointset_browse)
        entry_layout.addWidget(pointset_widget)

        # Interface2DBC-only: a second location (its geometry needs two
        # matching-length boundary pieces) + normal/tangent direction.
        interface_widget = QWidget()
        interface_layout = QVBoxLayout(interface_widget)
        interface_layout.setContentsMargins(0, 0, 0, 0); interface_layout.setSpacing(2)
        loc2_row = QHBoxLayout()
        loc2_row.addWidget(QLabel("Where (side 2):"))
        loc2_edit = QLineEdit(location2)
        loc2_edit.setPlaceholderText("boolean expression in x, y -- the matching second edge")
        loc2_edit.setFixedHeight(26)
        loc2_row.addWidget(loc2_edit)
        interface_layout.addLayout(loc2_row)
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Direction:"))
        direction_combo = QComboBox(); direction_combo.addItems(["normal", "tangent"])
        direction_combo.setCurrentText(direction)
        direction_combo.setFixedHeight(26); direction_combo.setFixedWidth(110)
        dir_row.addWidget(direction_combo); dir_row.addStretch()
        interface_layout.addLayout(dir_row)
        entry_layout.addWidget(interface_widget)

        advanced_note = QLabel(
            "Advanced: Value is a Python expression using inputs, outputs, X\n"
            "directly, i.e. DeepXDE's raw func(inputs, outputs, X) -- it should\n"
            "evaluate to 0 on the boundary/points selected above."
        )
        self._register_style(advanced_note, "hint", lambda css, _c='#ffa94d', _e='': f"color: {_c}; {_e}{css}")
        advanced_note.setWordWrap(True)
        advanced_note.setVisible(False)
        entry_layout.addWidget(advanced_note)

        val_placeholders = {
            "robin": "e.g. 0, sin(pi*y), u   (Robin: x, y, z, and u for the solution)",
            "interface2d": "Python expression in x, y, z for the interface func(x)",
            "pointset_operator": "Python expression using inputs, outputs, X (advanced)",
        }
        default_val_placeholder = "e.g. 0, sin(pi*y), x**2  (Robin: may also use u)"

        def _sync_type(_t=None):
            key = type_combo.currentData()
            comp_widget.setVisible(key in self._BC_NEEDS_COMPONENT)
            loc_widget.setVisible(key in self._BC_NEEDS_LOCATION)
            val_widget.setVisible(key in self._BC_NEEDS_VALUE)
            periodic_widget.setVisible(key in self._BC_NEEDS_PERIODIC_ROW)
            pointset_widget.setVisible(key in self._BC_NEEDS_POINTS_FILE)
            interface_widget.setVisible(key in self._BC_NEEDS_INTERFACE_ROW)
            advanced_note.setVisible(key in self._BC_NEEDS_ADVANCED_NOTE)
            val_edit.setPlaceholderText(val_placeholders.get(key, default_val_placeholder))

        def _sync_type_and_refresh_weights(_t=None):
            _sync_type(_t)
            # Keep the Loss Weights panel's "BC n (type, out k):" label current
            # -- the row itself already exists, this just re-labels it.
            if hasattr(self, 'weights_main_layout') and hasattr(self, 'num_outputs_spin'):
                self._build_weight_inputs(self.num_outputs_spin.value())
        type_combo.currentIndexChanged.connect(_sync_type_and_refresh_weights)
        comp_spin.valueChanged.connect(_sync_type_and_refresh_weights)
        _sync_type()

        if locked:
            for w in (type_combo, comp_spin, loc_edit, loc_hint_toggle, val_edit,
                      axis_combo, deriv_spin, pointset_path, pointset_browse,
                      loc2_edit, direction_combo):
                w.setEnabled(False)

        self.custom_bc_list_layout.addWidget(entry_widget)
        entry_data = {
            'widget': entry_widget,
            'type': type_combo,
            'component': comp_spin,
            'location': loc_edit,
            'value': val_edit,
            'axis': axis_combo,
            'deriv_order': deriv_spin,
            'points_file': pointset_path,
            'location2': loc2_edit,
            'direction': direction_combo,
            'locked': locked,
            # Containers, exposed mainly so tests can check per-type
            # visibility directly rather than guessing from the leaf inputs.
            'location_widget': loc_widget,
            'value_widget': val_widget,
            'periodic_widget': periodic_widget,
            'pointset_widget': pointset_widget,
            'interface_widget': interface_widget,
            'component_widget': comp_widget,
        }
        self.custom_bc_list.append(entry_data)

        def _remove():
            entry_widget.deleteLater()
            if entry_data in self.custom_bc_list:
                self.custom_bc_list.remove(entry_data)
            self._update_bc_mode_visibility()

        remove_btn.clicked.connect(_remove)
        return entry_data

    def _build_custom_bc_json(self):
        """Serialize the Boundary Conditions list for PINNConfig
        (custom_bc_json) -- one dict per BC row, independent of
        geometry_type. Template-derived (locked) rows are included too, but
        only for display continuity within a session; the "locked" flag
        itself isn't persisted (nothing here is), since Quick Example
        selection also isn't persisted across Save/Load Problem today --
        see _apply_config(). Only the un-locked (custom-mode) rows actually
        drive training today; codegen doesn't read this list for template
        problems (it still reads the legacy bc_left_types/etc fields, kept
        in sync internally) or for any row a template didn't produce --
        wiring that up is the natural next step."""
        import json
        entries = []
        for e in self.custom_bc_list:
            entries.append({
                'type': e['type'].currentData(),
                'component': e['component'].value(),
                'location': e['location'].text(),
                'value': e['value'].text(),
                'axis': e['axis'].currentText(),
                'deriv_order': e['deriv_order'].value(),
                'points_file': e['points_file'].text(),
                'location2': e['location2'].text(),
                'direction': e['direction'].currentText(),
            })
        return json.dumps(entries)

    def _apply_custom_bc_json(self, custom_bc_json):
        """Rebuild the Boundary Conditions list from a saved custom_bc_json
        string (config load). Restored rows are always unlocked/editable --
        see _build_custom_bc_json()'s docstring for why "locked" isn't
        persisted."""
        import json
        for e in list(self.custom_bc_list):
            e['widget'].deleteLater()
        self.custom_bc_list.clear()
        if not custom_bc_json:
            return
        try:
            entries = json.loads(custom_bc_json)
        except (ValueError, TypeError):
            return
        for e in entries:
            self._add_custom_bc_entry(
                bc_type=e.get('type', 'dirichlet'),
                component=e.get('component', 0),
                location=e.get('location', ''),
                value=e.get('value', '0'),
                axis=e.get('axis', 'x'),
                deriv_order=e.get('deriv_order', 0),
                points_file=e.get('points_file', ''),
                location2=e.get('location2', ''),
                direction=e.get('direction', 'normal'),
                locked=False,
            )

    def _populate_locked_bc_entries_from_legacy(self, n_out, is_2d):
        """Seed the Boundary Conditions panel with whatever was just written
        to the legacy per-side widgets (self.bc_left_types etc., set by
        _on_template_selected just before this is called) -- so a
        template's BCs show up pre-filled in the same panel used for a
        hand-built list, editable from the moment they appear. This is
        purely a starting point: once seeded, the legacy widgets are no
        longer consulted for this problem -- _build_config() serializes
        this panel's rows (whether edited or left as the template set
        them) into custom_bc_json, which is what codegen.py actually reads."""
        for e in list(self.custom_bc_list):
            e['widget'].deleteLater()
        self.custom_bc_list.clear()
        x_min, x_max = self.x_min.value(), self.x_max.value()
        y_min = self.y_min.value() if is_2d else None
        y_max = self.y_max.value() if is_2d else None
        for i in range(n_out):
            if i < len(self.bc_left_types):
                self._add_locked_side_entry("left", i, x_min, x_max, y_min, y_max, is_2d)
            if i < len(self.bc_right_types) and self.bc_left_types[i].currentText() != "Periodic":
                self._add_locked_side_entry("right", i, x_min, x_max, y_min, y_max, is_2d)
            if is_2d:
                if i < len(self.bc_bottom_types):
                    self._add_locked_side_entry("bottom", i, x_min, x_max, y_min, y_max, is_2d)
                if i < len(self.bc_top_types) and self.bc_bottom_types[i].currentText() != "Periodic":
                    self._add_locked_side_entry("top", i, x_min, x_max, y_min, y_max, is_2d)

    def _add_locked_side_entry(self, side, i, x_min, x_max, y_min, y_max, is_2d):
        types_map = {"left": self.bc_left_types, "right": self.bc_right_types,
                     "bottom": getattr(self, "bc_bottom_types", []),
                     "top": getattr(self, "bc_top_types", [])}
        vals_map = {"left": self.bc_left_vals, "right": self.bc_right_vals,
                    "bottom": getattr(self, "bc_bottom_vals", []),
                    "top": getattr(self, "bc_top_vals", [])}
        loc_map = {
            "left": f"x <= {x_min:g}",
            "right": f"x >= {x_max:g}",
            "bottom": f"y <= {y_min:g}" if is_2d else "",
            "top": f"y >= {y_max:g}" if is_2d else "",
        }
        axis_map = {"left": "x", "right": "x", "bottom": "y", "top": "y"}
        bc_type_txt = types_map[side][i].currentText()
        if bc_type_txt == "Periodic":
            self._add_custom_bc_entry(bc_type="periodic", component=i, location=loc_map[side],
                                       axis=axis_map[side], deriv_order=0, locked=False)
        else:
            val = vals_map[side][i].value()
            self._add_custom_bc_entry(bc_type=bc_type_txt.lower(), component=i,
                                       location=loc_map[side], value=str(val), locked=False)

    def _populate_3d_heat_bc_entries(self, n_out):
        """Seed the Boundary Conditions panel with the 6 faces of the 3D
        Heat template's box domain (x/y/z min/max). There's no legacy
        per-side widget layer for a third dimension (only left/right/
        bottom/top ever existed) -- and since the panel is the real source
        of truth for training now anyway (see codegen.py's custom_bc_json
        path), 3D templates populate it directly instead of going through
        _populate_locked_bc_entries_from_legacy, which only knows 1D/2D."""
        for e in list(self.custom_bc_list):
            e['widget'].deleteLater()
        self.custom_bc_list.clear()
        x_min, x_max = self.x_min.value(), self.x_max.value()
        y_min, y_max = self.y_min.value(), self.y_max.value()
        z_min, z_max = self.z_min.value(), self.z_max.value()
        # Same convention as heat2d: one Dirichlet face (right/x-max), the
        # rest Neumann=0 (insulated).
        faces = [
            (f"x <= {x_min:g}", "neumann", "0"),
            (f"x >= {x_max:g}", "dirichlet", "1"),
            (f"y <= {y_min:g}", "neumann", "0"),
            (f"y >= {y_max:g}", "neumann", "0"),
            (f"z <= {z_min:g}", "neumann", "0"),
            (f"z >= {z_max:g}", "neumann", "0"),
        ]
        for i in range(n_out):
            for loc, btype, val in faces:
                self._add_custom_bc_entry(bc_type=btype, component=i, location=loc,
                                           value=val, locked=False)

    # ── Build weight inputs ───────────────────────────────────
    def _build_weight_inputs(self, n):
        for i in reversed(range(self.weights_main_layout.count())):
            w = self.weights_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.weight_widgets.clear()

        # Check if per-phase weights needed
        _sched_enabled = True  # scheduler always on
        _same_weights = hasattr(self, 'sched_same_weights_cb') and self.sched_same_weights_cb.isChecked()
        _skip_main_weights = not _same_weights  # skip main weights when per-phase

        def _w_row(label, key):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            _default_w = 100.0 if key.startswith("ic_") else 1.0
            w = SciLineEdit(_default_w); w.setFixedWidth(100)
            row.addStretch(); row.addWidget(w)
            self.weight_widgets[key] = w
            ww = QWidget(); ww.setLayout(row)
            self.weights_main_layout.addWidget(ww)

        if _same_weights:
            _sep_all = QLabel("── Shared weights (all phases) ──")
            self._register_style(_sep_all, "hint", lambda css, _c='#505080', _e='': f"color: {_c}; {_e}{css}")
            _sw_all = QWidget(); _sl_all = QHBoxLayout(_sw_all)
            _sl_all.setContentsMargins(0,0,0,0); _sl_all.addWidget(_sep_all)
            self.weights_main_layout.addWidget(_sw_all)

            for i in range(n):
                name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
                _w_row(f"PDE {i+1} ({name}):", f"pde_{i}")
            # One weight row per row in the Boundary Conditions panel --
            # matches exactly what will train (see codegen.py's
            # custom_bc_json path), whether it's a hand-built row or one
            # pre-filled from a template.
            for _bj, _be in enumerate(getattr(self, 'custom_bc_list', [])):
                _btype_label = _be['type'].currentText()
                _bcomp = _be['component'].value()
                _w_row(f"BC {_bj + 1} ({_btype_label}, Output {_bcomp}):", f"bc_{_bj}")
            for i in range(n):
                name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
                _ic_from_file_checked = (
                    hasattr(self, 'ic_from_file') and
                    i < len(self.ic_from_file) and
                    self.ic_from_file[i] is not None and
                    self.ic_from_file[i].isChecked()
                )
                if (i < len(self.ic_active) and self.ic_active[i].isChecked()) or _ic_from_file_checked:
                    _w_row(f"IC {i+1} ({name}):", f"ic_{i}")

        # Per-phase weight rows if scheduler enabled and different weights
        if _sched_enabled and not _same_weights:
            for _pi, _ph in enumerate(self.sched_phase_list):
                _phase_num = _ph['phase_num']
                sep = QLabel(f"── Phase {_phase_num} weights ──")
                self._register_style(sep, "hint", lambda css, _c='#505080', _e='': f"color: {_c}; {_e}{css}")
                _sw = QWidget(); _sl = QHBoxLayout(_sw)
                _sl.setContentsMargins(0,0,0,0); _sl.addWidget(sep)
                self.weights_main_layout.addWidget(_sw)
                for _i in range(n):
                    _name = self.output_name_inputs[_i].text() if _i < len(self.output_name_inputs) else f"u{_i+1}"
                    _w_row(f"PDE {_i+1} ({_name}) P{_phase_num}:", f"pde_{_i}_p{_phase_num}")
                for _bj, _be in enumerate(getattr(self, 'custom_bc_list', [])):
                    _btype_label = _be['type'].currentText()
                    _bcomp = _be['component'].value()
                    _w_row(f"BC {_bj + 1} ({_btype_label}, Output {_bcomp}) P{_phase_num}:", f"bc_{_bj}_p{_phase_num}")
                for _i in range(n):
                    _name = self.output_name_inputs[_i].text() if _i < len(self.output_name_inputs) else f"u{_i+1}"
                    _ic_ff = (hasattr(self, 'ic_from_file') and _i < len(self.ic_from_file)
                              and self.ic_from_file[_i] is not None and self.ic_from_file[_i].isChecked())
                    if (_i < len(self.ic_active) and self.ic_active[_i].isChecked()) or _ic_ff:
                        _w_row(f"IC {_i+1} ({_name}) P{_phase_num}:", f"ic_{_i}_p{_phase_num}")

    def _flat_loss_weights_string(self, n_out):
        """Build the comma-separated loss_weights_multi string in the exact
        order codegen.py expects: PDE (one per output), then one weight per
        Boundary Conditions panel row (in that panel's order), then IC (one
        per active output) -- mirroring _build_weight_inputs()'s row order.
        When per-phase weights are in use, this uses phase 1's weight_widgets
        as the flat/representative set (matching the pre-existing convention
        this replaces)."""
        use_phase1 = (hasattr(self, 'sched_phase_list') and self.sched_phase_list
                      and hasattr(self, 'sched_same_weights_cb')
                      and not self.sched_same_weights_cb.isChecked())
        suffix = f"_p{self.sched_phase_list[0]['phase_num']}" if use_phase1 else ""
        parts = []
        for i in range(n_out):
            key = f"pde_{i}{suffix}"
            if key in self.weight_widgets:
                parts.append(str(self.weight_widgets[key].value()))
        for j in range(len(getattr(self, 'custom_bc_list', []))):
            key = f"bc_{j}{suffix}"
            if key in self.weight_widgets:
                parts.append(str(self.weight_widgets[key].value()))
        for i in range(n_out):
            key = f"ic_{i}{suffix}"
            if key in self.weight_widgets:
                parts.append(str(self.weight_widgets[key].value()))
        return ",".join(parts)

    # ── Build config ──────────────────────────────────────────
    def _build_config(self):
        n = self.layers_spin.value()
        w = self.neurons_spin.value()
        n_out = self.num_outputs_spin.value()
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
        input_size = 4 if is_3d else (3 if is_2d else 2)
        layers = [input_size] + [w] * n + [n_out]

        def _safe_val(lst, i, default=0.0):
            try:
                return lst[i].value()
            except Exception:
                return default

        def _safe_text(lst, i, default="Dirichlet"):
            try:
                return lst[i].currentText()
            except Exception:
                return default

        def _safe_checked(lst, i, default=True):
            try:
                return lst[i].isChecked()
            except Exception:
                return default

        box_shape = self._current_geometry_type() in ("Interval", "Rectangle", "Cuboid")

        def _bc_group_json(types_list, vals_list, active_list, n_out):
            """Serialize a flat per-output BC group (Boundary for Disk/Ellipse/
            Sphere) to a JSON string: one {active,type,value} dict per output."""
            import json as _json
            out = []
            for _i in range(n_out):
                out.append({
                    "active": _i < len(active_list) and active_list[_i].isChecked(),
                    "type": _safe_text(types_list, _i),
                    "value": _safe_val(vals_list, _i),
                })
            return _json.dumps(out)

        def _bc_edge_json(n_out):
            """Serialize the per-output, per-edge BC groups (Triangle/Polygon)
            to a JSON string: a list (per output) of lists (per edge) of
            {active,type,value} dicts."""
            import json as _json
            out = []
            for _i in range(n_out):
                _edges = []
                if _i < len(self.bc_edge_types):
                    for _e in range(len(self.bc_edge_types[_i])):
                        _edges.append({
                            "active": self.bc_edge_active[_i][_e].isChecked(),
                            "type": self.bc_edge_types[_i][_e].currentText(),
                            "value": self.bc_edge_vals[_i][_e].value(),
                        })
                out.append(_edges)
            return _json.dumps(out)

        return PINNConfig(
            problem_dim="3D" if is_3d else ("2D" if is_2d else "1D"),
            geometry_type=self._current_geometry_type(),
            z_min=self.z_min.value(), z_max=self.z_max.value(),
            geom_center_x=(self.geom_sphere_cx.value() if is_3d else
                           self.geom_ellipse_cx.value() if self.geometry_type_combo.currentText() == "Ellipse" else
                           self.geom_disk_cx.value()),
            geom_center_y=(self.geom_sphere_cy.value() if is_3d else
                           self.geom_ellipse_cy.value() if self.geometry_type_combo.currentText() == "Ellipse" else
                           self.geom_disk_cy.value()),
            geom_center_z=self.geom_sphere_cz.value(),
            geom_radius=self.geom_sphere_r.value() if is_3d else self.geom_disk_r.value(),
            geom_semi_major=self.geom_ellipse_a.value(),
            geom_semi_minor=self.geom_ellipse_b.value(),
            geom_angle=self.geom_ellipse_angle.value(),
            geom_triangle_vertices=self.geom_triangle_verts_input.text(),
            geom_polygon_vertices=self.geom_polygon_verts_input.text(),
            bc_boundary_json=_bc_group_json(self.bc_boundary_types, self.bc_boundary_vals, self.bc_boundary_active, n_out),
            bc_edge_json=_bc_edge_json(n_out),
            custom_bc_json=self._build_custom_bc_json(),
            num_outputs=n_out,
            output_names=",".join([self.output_name_inputs[i].text() for i in range(n_out)]),
            pde_expressions="|".join([self.pde_inputs[i].text() for i in range(n_out)]),

            bc_left_types=",".join([_safe_text(self.bc_left_types, i) for i in range(n_out)]),
            bc_right_types=",".join([_safe_text(self.bc_right_types, i) for i in range(n_out)]),
            bc_left_values=",".join([str(_safe_val(self.bc_left_vals, i)) for i in range(n_out)]),
            bc_right_values=",".join([str(_safe_val(self.bc_right_vals, i)) for i in range(n_out)]),
            bc_left_active=",".join([str(_safe_checked(self.bc_left_active, i)) for i in range(n_out)]),
            bc_right_active=",".join([
                "False" if (i < len(self.bc_left_types) and self.bc_left_types[i].currentText() == "Periodic")
                else str(_safe_checked(self.bc_right_active, i))
                for i in range(n_out)
            ]),
            bc_left_deriv=",".join([str(_safe_checked(self.bc_left_deriv, i, False)) for i in range(n_out)]),
            bc_right_deriv=",".join([str(_safe_checked(self.bc_right_deriv, i, False)) for i in range(n_out)]),

            bc_bottom_types=",".join([_safe_text(self.bc_bottom_types, i) for i in range(n_out)]) if box_shape else "Dirichlet",
            bc_top_types=",".join([_safe_text(self.bc_top_types, i) for i in range(n_out)]) if box_shape else "Dirichlet",
            bc_bottom_values=",".join([str(_safe_val(self.bc_bottom_vals, i)) for i in range(n_out)]) if box_shape else "0.0",
            bc_top_values=",".join([str(_safe_val(self.bc_top_vals, i)) for i in range(n_out)]) if box_shape else "0.0",
            bc_bottom_active=",".join([str(_safe_checked(self.bc_bottom_active, i)) for i in range(n_out)]) if box_shape else "True",
            bc_top_active=",".join([
                "False" if (i < len(self.bc_bottom_types) and self.bc_bottom_types[i].currentText() == "Periodic")
                else str(_safe_checked(self.bc_top_active, i))
                for i in range(n_out)
            ]) if box_shape else "True",
            bc_bottom_deriv=",".join([str(_safe_checked(self.bc_bottom_deriv, i, False)) for i in range(n_out)]) if box_shape else "False",
            bc_top_deriv=",".join([str(_safe_checked(self.bc_top_deriv, i, False)) for i in range(n_out)]) if box_shape else "False",

            ic_expressions="|".join([self.ic_inputs[i].text() for i in range(n_out)]),
            ic_active=",".join([str(self.ic_active[i].isChecked()) for i in range(n_out)]),
            forward_ic_from_file=any(
                i < len(self.ic_from_file) and self.ic_from_file[i] is not None
                and self.ic_from_file[i].isChecked()
                for i in range(n_out)
            ),
            forward_ic_file=next(
                (self.ic_file_paths[i].text().strip()
                 for i in range(n_out)
                 if i < len(self.ic_from_file)
                 and self.ic_from_file[i] is not None
                 and self.ic_from_file[i].isChecked()
                 and self.ic_file_paths[i] is not None),
                ""
            ),

            loss_weights_multi=self._flat_loss_weights_string(n_out),
            plot_output_idx=self.plot_output_combo.currentIndex(),

            pde_expression=self.pde_inputs[0].text() if self.pde_inputs else "du_t - 0.4 * du_xx",
            bc_left=_safe_val(self.bc_left_vals, 0),
            bc_right=_safe_val(self.bc_right_vals, 0),
            bc_left_type=_safe_text(self.bc_left_types, 0),
            bc_right_type=_safe_text(self.bc_right_types, 0),
            ic_expression=self.ic_inputs[0].text() if self.ic_inputs else "np.sin(np.pi * x[:, 0])",
            loss_weights=[
                self.weight_widgets.get("pde_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("bc_left_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("bc_right_0", SciLineEdit(1.0)).value(),
                self.weight_widgets.get("ic_0", SciLineEdit(1.0)).value(),
            ],
            loss_weight_obs=self.inv_obs_weight.value(),
            inv_param_log_scale=self.inv_param_log_scale.isChecked(),
            inv_param_save=self.param_save_combo.currentText(),

            x_min=self.x_min.value(), x_max=self.x_max.value(),
            y_min=self.y_min.value(), y_max=self.y_max.value(),
            t_min=self.t_min.value(), t_max=self.t_max.value(),
            layers=layers,
            activation=self.activation_combo.currentText(),
            kernel_initializer=self.kernel_init_combo.currentText(),
            input_transform_enabled=self.input_transform_cb.isChecked(),
            input_transform_scale=[r["scale"].value() for r in self.input_transform_rows],
            input_transform_shift=[r["shift"].value() for r in self.input_transform_rows],
            output_transform_enabled=self.output_transform_cb.isChecked(),
            output_transform_scale=[r["scale"].value() for r in self.output_transform_rows],
            output_transform_shift=[r["shift"].value() for r in self.output_transform_rows],
            iterations=self.iter1_spin.value(),
            optimizer=self.opt1_combo.currentText(),
            optimizer2=self.opt2_combo.currentText(),
            iterations2=self.iter2_spin.value(),
            num_domain=self.num_domain.value(),
            num_boundary=self.num_boundary.value(),
            num_initial=self.num_initial.value(),
            num_test=self.num_test.value(),
            point_distribution=self.pts_dist_combo.currentText(),
            plot_type=self.plot_type_combo.currentText(),
            num_timesteps=self.timesteps_spin.value(),
            save_dir=self.save_dir_input.text(),
            adapt_method=self.adapt_combo.currentData(),
            rar_cycles=self.rar_cycles.value(),
            rar_candidates=self.rar_candidates.value(),
            rar_add_points=self.rar_add_points.value(),
            rar_adam_iters=self.rar_adam_iters.value(),
            rar_lbfgs_iters=self.rar_lbfgs_iters.value(),
            time_adaptive=self.adapt_combo.currentData() == "Time Adaptive",
            ta_num_steps=sum(r['steps'].value() for r in self.ta_group_rows) if self.ta_group_rows else self.ta_steps.value(),
            ta_grid_size=int(self.ta_grid.currentText()),
            ta_step_groups=self._build_ta_step_groups_json(),
            ta_transfer_learning=self.ta_transfer_cb.isChecked(),
            ta_transfer_optimizer=self.ta_transfer_opt.currentData(),
            learning_rate=self.lr_spin.value(),
            loss_type=self.loss_combo.currentText(),
            parametric_study=False,
            parametric_param="none",
            parametric_values="",
            problem_type="Inverse" if self.radio_inverse.isChecked() else "Forward",
            inverse_param_name=self.inv_param_name.text().strip(),
            inverse_param_init=self.inv_param_init.value(),
            inverse_variables_json=self._build_inverse_variables_json(),
            inverse_obs_output_idx=self.inv_obs_output_combo.currentIndex() if self.inv_obs_output_combo.currentIndex() >= 0 else 0,
            inverse_data_file=self.inv_data_path.text().strip(),
            inverse_obs_files_json=self._build_inverse_obs_files_json(),
            # v19: the Inverse panel's own IC-type selector is gone (see the
            # comment where it used to be built) -- always "expression" now,
            # so codegen always takes the Initial Condition panel's path.
            inverse_ic_type="expression",
            inverse_ic_file="",
            export_grid_size=int(self.export_grid_combo.currentText()),
            export_t_steps=self.export_tsteps_spin.value(),
            template_type=getattr(self, '_current_template_type', ''),
            optimizer_scheduler=self.sched_cb.isChecked() if hasattr(self, 'sched_cb') else False,
            scheduler_same_weights=self.sched_same_weights_cb.isChecked() if hasattr(self, 'sched_same_weights_cb') else True,
            scheduler_phases=self._build_scheduler_phases_json(),
            lbfgs_maxcor=int(self.lbfgs_maxcor.value()),
            lbfgs_ftol=self.lbfgs_ftol.value(),
            lbfgs_gtol=self.lbfgs_gtol.value(),
            lbfgs_maxiter=int(self.lbfgs_maxiter.value()),
            lbfgs_maxfun=int(self.lbfgs_maxfun.value()),
            lbfgs_maxls=int(self.lbfgs_maxls.value()),
            lbfgs_float_type=self.lbfgs_float_combo.currentText() if hasattr(self, 'lbfgs_float_combo') else 'float64',
            float_type=getattr(self, '_float_type', 'float64'),
            batch_size=self.batch_spin.value() if self.batch_check.isChecked() else 0,
            ic_pretrain=self.ic_pretrain_cb.isChecked(),
            ic_pretrain_optimizer=self.ic_pretrain_opt.currentData(),
            ic_pretrain_iterations=self.ic_pretrain_iters.value(),
            ic_pretrain_num_test=self.ic_pretrain_test.value(),
            ic_pretrain_num_initial=self.ic_pretrain_init.value(),
            ic_pretrain_restore=self.ic_pretrain_mode_restore.isChecked(),
            ic_pretrain_restore_path=self.ic_pretrain_restore_path.text().strip(),
            weight_decay=self._optimizer_settings.get("weight_decay", 0.0),
            cb_early_stopping=self._callback_settings.get("early_stopping", False),
            cb_early_stopping_min_delta=self._callback_settings.get("early_stopping_min_delta", 0.0),
            cb_early_stopping_patience=self._callback_settings.get("early_stopping_patience", 2000),
            cb_early_stopping_baseline=self._callback_settings.get("early_stopping_baseline", ""),
            cb_early_stopping_monitor=self._callback_settings.get("early_stopping_monitor", "loss_train"),
            cb_early_stopping_start_from=self._callback_settings.get("early_stopping_start_from", 0),
            cb_point_resampler=self._callback_settings.get("point_resampler", False),
            cb_point_resampler_period=self._callback_settings.get("point_resampler_period", 100),
            cb_point_resampler_pde_points=self._callback_settings.get("point_resampler_pde_points", True),
            cb_point_resampler_bc_points=self._callback_settings.get("point_resampler_bc_points", False),
            cb_model_checkpoint=self._callback_settings.get("model_checkpoint", False),
            cb_checkpoint_period=self._callback_settings.get("checkpoint_period", 1000),
            cb_checkpoint_save_better_only=self._callback_settings.get("checkpoint_save_better_only", True),
            cb_checkpoint_monitor=self._callback_settings.get("checkpoint_monitor", "train loss"),
            cb_timer=self._callback_settings.get("timer", False),
            cb_timer_minutes=self._callback_settings.get("timer_minutes", 60.0),
            plot_colormap=self._plot_viz_settings.get('colormap', 'RdBu_r'),
            plot_levels=self._plot_viz_settings.get('levels', 50),
            plot_resolution=self._plot_viz_settings.get('resolution', 100),
            plot_dpi=self._plot_viz_settings.get('dpi', 100),
            plot_colorbar=self._plot_viz_settings.get('colorbar', True),
            plot_auto_range=self._plot_viz_settings.get('auto_range', True),
            plot_vmin=self._plot_viz_settings.get('vmin', -1.0),
            plot_vmax=self._plot_viz_settings.get('vmax', 1.0),
            plot_linewidth=self._plot_viz_settings.get('linewidth', 2.0),
            plot_n_2d_snapshots=self._plot_viz_settings.get('n_2d_snapshots', 2),
            ea_files=repr(self._ea_settings.get('files', [])) if getattr(self, '_ea_settings', None) else "[]",
            ea_do_line=self._ea_settings.get('do_line', True) if getattr(self, '_ea_settings', None) else True,
            ea_do_surface=self._ea_settings.get('do_surface', True) if getattr(self, '_ea_settings', None) else True,
        )

    # ── Save / Open a problem definition ─────────────────────
    # A saved problem is just a PINNConfig -- the same object _build_config()
    # already assembles for Solve -- serialized to JSON. Opening one reverses
    # that: parse the JSON back into a PINNConfig, then _apply_config() writes
    # every field back onto its widget. Together these make _build_config()
    # and _apply_config() exact mirrors of each other; the split is deliberate
    # so a bug in one direction can't silently paper over a bug in the other.
    def _save_problem(self):
        import json
        import dataclasses
        from datetime import datetime
        default_name = getattr(self, "_current_problem_name", "") or "problem"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Problem As", f"{default_name}.pinn.json",
            "PINNStudio Problem (*.pinn.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".pinn.json"
        config = self._build_config()
        try:
            from importlib.metadata import version as _pkg_version
            installed_version = _pkg_version("pinnstudio")
        except Exception:
            installed_version = ""
        problem_name = os.path.splitext(os.path.basename(path))[0]
        if problem_name.endswith(".pinn"):
            problem_name = problem_name[:-5]
        payload = {
            "pinnstudio_problem_format": 1,
            "problem_name": problem_name,
            "saved_with_pinnstudio_version": installed_version,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config": dataclasses.asdict(config),
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self._current_problem_name = problem_name
            self.log_box.append(f"💾 Problem saved: {os.path.basename(path)}")
        except Exception as e:
            self.log_box.append(f"❌ Failed to save problem: {e}")

    def _open_problem(self):
        import json
        import dataclasses
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Saved Problem", "", "PINNStudio Problem (*.pinn.json *.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Failed to read problem file: {e}")
            return
        raw_config = data.get("config", data) if isinstance(data, dict) else None
        if not isinstance(raw_config, dict):
            self.log_box.append("❌ This doesn't look like a valid PINNStudio problem file.")
            return
        known_fields = {f.name for f in dataclasses.fields(PINNConfig)}
        filtered = {k: v for k, v in raw_config.items() if k in known_fields}
        try:
            config = PINNConfig(**filtered)
        except Exception as e:
            self.log_box.append(f"❌ Failed to parse problem file: {e}")
            return
        try:
            self._apply_config(config)
        except Exception as e:
            self.log_box.append(f"❌ Failed to load problem into the interface: {e}")
            return
        problem_name = (
            data.get("problem_name", os.path.splitext(os.path.basename(path))[0])
            if isinstance(data, dict) else os.path.splitext(os.path.basename(path))[0]
        )
        self._current_problem_name = problem_name
        self.log_box.append(f"✅ Problem loaded: {problem_name}")

    def _apply_weight_string(self, weights_str, keys):
        """Assign a comma-separated weight string back onto self.weight_widgets,
        walking `keys` in order and consuming one value per key that actually
        exists in weight_widgets -- the exact mirror of how each weight string
        is built (see _build_config's loss_weights_multi and
        _build_scheduler_phases_json)."""
        parts = (weights_str or "").split(",")
        vi = 0
        for k in keys:
            if k not in self.weight_widgets:
                continue
            if vi < len(parts):
                try:
                    self.weight_widgets[k].setValue(float(parts[vi]))
                except ValueError:
                    pass
                vi += 1

    def _apply_flat_weights(self, weights_str, n_out, is_2d):
        # Order matches _flat_loss_weights_string() / _build_weight_inputs():
        # PDE per output, then one weight per Boundary Conditions panel row,
        # then IC per output. is_2d is accepted for call-site compatibility
        # but no longer changes which keys are read.
        keys = [f"pde_{i}" for i in range(n_out)]
        keys.extend(f"bc_{j}" for j in range(len(getattr(self, 'custom_bc_list', []))))
        keys.extend(f"ic_{i}" for i in range(n_out))
        self._apply_weight_string(weights_str, keys)

    def _apply_scheduler_phases_build_only(self, phases_json):
        """Rebuild self.sched_phase_list's optimizer/iterations/lr rows from
        saved JSON -- weight VALUES are applied separately afterward, once
        _build_weight_inputs() has (re)created the matching weight_widgets."""
        import json
        try:
            phases = json.loads(phases_json) if phases_json else []
        except (json.JSONDecodeError, TypeError):
            phases = []
        for ph in list(self.sched_phase_list):
            ph["widget"].deleteLater()
        self.sched_phase_list.clear()
        for ph in phases:
            self._add_scheduler_phase(
                ph.get("optimizer", "adam"), ph.get("iterations", 10000), ph.get("lr", 0.001)
            )
            new_row = self.sched_phase_list[-1]
            self._set_combo_data(new_row['decay_type'], ph.get('decay_type', 'none'))
            new_row['decay_p1'].setValue(ph.get('decay_p1', 0) or 0)
            new_row['decay_p2'].setValue(ph.get('decay_p2', 0) or 0)
        return phases

    def _apply_phase_weights(self, phases, n_out, is_2d):
        """Only meaningful when scheduler_same_weights is False -- that's the
        only case where per-phase (_p{n}-suffixed) weight_widgets exist.
        The saved JSON has no phase_num key (_build_scheduler_phases_json
        never wrote one) -- phase_num is 1-based position in the list, the
        same convention _add_scheduler_phase itself uses. Key order/set below
        matches _build_scheduler_phases_json exactly: PDE for every output
        first, then one weight per Boundary Conditions panel row, then IC
        for every output."""
        for pn, ph in enumerate(phases, start=1):
            keys = [f"pde_{i}_p{pn}" for i in range(n_out)]
            keys.extend(f"bc_{j}_p{pn}" for j in range(len(getattr(self, 'custom_bc_list', []))))
            keys.extend(f"ic_{i}_p{pn}" for i in range(n_out))
            self._apply_weight_string(ph.get("weights", ""), keys)

    def _apply_config(self, config):
        """Populate every widget in the interface from a PINNConfig -- the
        exact mirror of _build_config(). Used by _open_problem() to restore a
        previously saved problem. Order matters: dimension, problem type, and
        output count are set first, since changing any of them rebuilds the
        per-output PDE/BC/IC widget rows from scratch -- only once those rows
        exist can their individual values be populated."""
        import json
        import ast as _ast

        def _bools(s, n, default=True):
            parts = (s or "").split(",")
            return [(parts[i].strip() == "True" if i < len(parts) else default) for i in range(n)]

        def _floats(s, n, default=0.0):
            parts = (s or "").split(",")
            out = []
            for i in range(n):
                if i < len(parts):
                    try:
                        out.append(float(parts[i]))
                    except ValueError:
                        out.append(default)
                else:
                    out.append(default)
            return out

        def _texts(s, n, sep, default=""):
            parts = (s or "").split(sep)
            return [(parts[i] if i < len(parts) else default) for i in range(n)]

        n_out = max(1, config.num_outputs)
        is_2d = config.problem_dim == "2D"
        is_3d = config.problem_dim == "3D"

        # Clear per-session template bookkeeping so leftover state from a
        # previously loaded built-in template (or a previously loaded saved
        # problem) can't leak into this one via _on_problem_type_changed /
        # _sync_inverse_pde_substitution.
        self._current_template = ""
        self._current_template_type = config.template_type or ""
        self._template_ref_dir = ""
        self._current_template_forward_tmax = None
        self._current_template_inverse_tmax = None
        self._current_ta_cfg = None
        self._ta_suspended_for_inverse = False
        self._inverse_sub_active = None

        # 1) Dimension -- rebuilds PDE/BC/IC rows for the CURRENT output count.
        if is_3d:
            self.radio_3d.setChecked(True)
        elif is_2d:
            self.radio_2d.setChecked(True)
        else:
            self.radio_1d.setChecked(True)

        # 2) Problem type -- must be set before output count / adapt_method,
        # since Inverse mode removes "Time Adaptive" from adapt_combo.
        if config.problem_type == "Inverse":
            self.radio_inverse.setChecked(True)
        else:
            self.radio_forward.setChecked(True)

        # 3) Output count -- rebuilds PDE/BC/IC rows again, this time for the
        # real target count. Let Qt actually create the new widgets first.
        self.num_outputs_spin.setValue(n_out)
        QApplication.processEvents()

        # Output names
        names = _texts(config.output_names, n_out, ",", "")
        for i in range(n_out):
            if i < len(self.output_name_inputs) and names[i]:
                self.output_name_inputs[i].setText(names[i])

        # Refresh the plot/restore output dropdowns now that names are real
        # (they were populated with placeholder names when output count
        # changed, before we had the saved names to give them).
        self.plot_output_combo.clear()
        self.restore_output_combo.clear()
        for _r in getattr(self, 'inv_data_rows', []):
            _r['output_combo'].clear()
        for i in range(n_out):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            self.plot_output_combo.addItem(f"Output {i+1} ({name})")
            self.restore_output_combo.addItem(f"Output {i+1} ({name})")
            for _r in getattr(self, 'inv_data_rows', []):
                _r['output_combo'].addItem(f"Output {i+1} ({name})")

        # PDE expressions
        pdes = _texts(config.pde_expressions, n_out, "|", config.pde_expression)
        for i in range(n_out):
            if i < len(self.pde_inputs):
                self.pde_inputs[i].setText(pdes[i])

        # 3.5) Domain z bounds + per-shape geometry parameters, and vertex
        # text (set before the geometry-type combo below, so that when it
        # rebuilds the BC rows for Triangle/Polygon, the edge count already
        # reflects the loaded vertices).
        self.z_min.setValue(config.z_min); self.z_max.setValue(config.z_max)
        self.geom_disk_cx.setValue(config.geom_center_x); self.geom_disk_cy.setValue(config.geom_center_y)
        self.geom_disk_r.setValue(config.geom_radius)
        self.geom_ellipse_cx.setValue(config.geom_center_x); self.geom_ellipse_cy.setValue(config.geom_center_y)
        self.geom_ellipse_a.setValue(config.geom_semi_major); self.geom_ellipse_b.setValue(config.geom_semi_minor)
        self.geom_ellipse_angle.setValue(config.geom_angle)
        self.geom_sphere_cx.setValue(config.geom_center_x); self.geom_sphere_cy.setValue(config.geom_center_y)
        self.geom_sphere_cz.setValue(config.geom_center_z); self.geom_sphere_r.setValue(config.geom_radius)
        self.geom_triangle_verts_input.setText(config.geom_triangle_vertices or "0,0;1,0;0,1")
        self.geom_polygon_verts_input.setText(config.geom_polygon_vertices or "0,0;1,0;1,1;0,1")

        # 3.6) Geometry type -- rebuilds the BC rows one more time, this
        # time in the shape (box / unified boundary / per-edge) that
        # matches the saved geometry.
        _geom_items = [self.geometry_type_combo.itemText(k) for k in range(self.geometry_type_combo.count())]
        if config.geometry_type in _geom_items:
            self.geometry_type_combo.setCurrentText(config.geometry_type)
        QApplication.processEvents()
        box_shape = self._current_geometry_type() in ("Interval", "Rectangle", "Cuboid")
        unified_shape = self._current_geometry_type() in ("Disk", "Ellipse", "Sphere")
        edge_shape = self._current_geometry_type() in ("Triangle", "Polygon")

        # Boundary conditions -- left/right (1D and 2D), bottom/top (2D only)
        bl_types = _texts(config.bc_left_types, n_out, ",", "Dirichlet")
        br_types = _texts(config.bc_right_types, n_out, ",", "Dirichlet")
        bl_vals = _floats(config.bc_left_values, n_out, 0.0)
        br_vals = _floats(config.bc_right_values, n_out, 0.0)
        bl_active = _bools(config.bc_left_active, n_out, True)
        br_active = _bools(config.bc_right_active, n_out, True)
        bl_deriv = _bools(config.bc_left_deriv, n_out, False)
        br_deriv = _bools(config.bc_right_deriv, n_out, False)
        for i in range(n_out):
            if i < len(self.bc_left_types):
                self.bc_left_active[i].setChecked(bl_active[i])
                self.bc_left_types[i].setCurrentText(bl_types[i])
                self.bc_left_vals[i].setValue(bl_vals[i])
                self.bc_left_deriv[i].setChecked(bl_deriv[i])
            if i < len(self.bc_right_types):
                self.bc_right_active[i].setChecked(br_active[i])
                self.bc_right_types[i].setCurrentText(br_types[i])
                self.bc_right_vals[i].setValue(br_vals[i])
                self.bc_right_deriv[i].setChecked(br_deriv[i])

        if box_shape:
            bb_types = _texts(config.bc_bottom_types, n_out, ",", "Dirichlet")
            bt_types = _texts(config.bc_top_types, n_out, ",", "Dirichlet")
            bb_vals = _floats(config.bc_bottom_values, n_out, 0.0)
            bt_vals = _floats(config.bc_top_values, n_out, 0.0)
            bb_active = _bools(config.bc_bottom_active, n_out, True)
            bt_active = _bools(config.bc_top_active, n_out, True)
            bb_deriv = _bools(config.bc_bottom_deriv, n_out, False)
            bt_deriv = _bools(config.bc_top_deriv, n_out, False)
            for i in range(n_out):
                if i < len(self.bc_bottom_types):
                    self.bc_bottom_active[i].setChecked(bb_active[i])
                    self.bc_bottom_types[i].setCurrentText(bb_types[i])
                    self.bc_bottom_vals[i].setValue(bb_vals[i])
                    self.bc_bottom_deriv[i].setChecked(bb_deriv[i])
                if i < len(self.bc_top_types):
                    self.bc_top_active[i].setChecked(bt_active[i])
                    self.bc_top_types[i].setCurrentText(bt_types[i])
                    self.bc_top_vals[i].setValue(bt_vals[i])
                    self.bc_top_deriv[i].setChecked(bt_deriv[i])

        # Unified boundary (Disk/Ellipse/Sphere)
        if unified_shape and config.bc_boundary_json:
            try:
                _bgroups = json.loads(config.bc_boundary_json)
            except Exception:
                _bgroups = []
            for i in range(min(n_out, len(_bgroups), len(self.bc_boundary_types))):
                _g = _bgroups[i] or {}
                self.bc_boundary_active[i].setChecked(bool(_g.get("active", True)))
                self.bc_boundary_types[i].setCurrentText(_g.get("type", "Dirichlet"))
                self.bc_boundary_vals[i].setValue(float(_g.get("value", 0.0)))

        # Per-edge boundary (Triangle/Polygon)
        if edge_shape and config.bc_edge_json:
            try:
                _egroups = json.loads(config.bc_edge_json)
            except Exception:
                _egroups = []
            for i in range(min(n_out, len(_egroups), len(self.bc_edge_types))):
                _edges = _egroups[i] or []
                for e in range(min(len(_edges), len(self.bc_edge_types[i]))):
                    _ed = _edges[e] or {}
                    self.bc_edge_active[i][e].setChecked(bool(_ed.get("active", True)))
                    self.bc_edge_types[i][e].setCurrentText(_ed.get("type", "Dirichlet"))
                    self.bc_edge_vals[i][e].setValue(float(_ed.get("value", 0.0)))

        # Custom Boundary Conditions (non-template problems)
        if hasattr(self, 'custom_bc_list'):
            self._apply_custom_bc_json(getattr(config, 'custom_bc_json', ''))
        self._update_bc_mode_visibility()

        # Initial conditions
        ics = _texts(config.ic_expressions, n_out, "|", config.ic_expression)
        ic_active = _bools(config.ic_active, n_out, True)
        for i in range(n_out):
            if i < len(self.ic_inputs):
                self.ic_inputs[i].setText(ics[i])
            if i < len(self.ic_active):
                self.ic_active[i].setChecked(ic_active[i])

        # IC-from-file (any dimension; the builder only supports one output at a time)
        if config.forward_ic_from_file and hasattr(self, "ic_from_file"):
            for i in range(n_out):
                if i < len(self.ic_from_file) and self.ic_from_file[i] is not None:
                    self.ic_from_file[i].setChecked(i == 0)
                    if i == 0 and i < len(self.ic_file_paths) and self.ic_file_paths[i] is not None:
                        self.ic_file_paths[i].setText(config.forward_ic_file)

        # Domain
        self.x_min.setValue(config.x_min); self.x_max.setValue(config.x_max)
        if is_2d or is_3d:
            self.y_min.setValue(config.y_min); self.y_max.setValue(config.y_max)
        self.t_min.setValue(config.t_min); self.t_max.setValue(config.t_max)

        # Collocation points
        self.num_domain.setValue(config.num_domain)
        self.num_boundary.setValue(config.num_boundary)
        self.num_initial.setValue(config.num_initial)
        self.num_test.setValue(config.num_test)
        self.pts_dist_combo.setCurrentText(config.point_distribution)
        self.plot_type_combo.setCurrentText(config.plot_type)
        self.timesteps_spin.setValue(config.num_timesteps)

        # Network
        n_hidden = max(0, len(config.layers) - 2)
        neurons = config.layers[1] if len(config.layers) > 2 else self.neurons_spin.value()
        self.layers_spin.setValue(n_hidden)
        self.neurons_spin.setValue(neurons)
        self.activation_combo.setCurrentText(config.activation)
        self.kernel_init_combo.setCurrentText(config.kernel_initializer)

        # Input/output transform rows were already rebuilt to match this
        # config's dimension (step 1 above) and output count (step 3
        # above) via _on_dim_changed()/_on_num_outputs_changed() -- just
        # fill in the saved values now, defaulting to identity (scale=1,
        # shift=0) for any row a shorter/older saved list doesn't cover.
        self.input_transform_cb.setChecked(bool(config.input_transform_enabled))
        for i, row in enumerate(self.input_transform_rows):
            scale = config.input_transform_scale[i] if i < len(config.input_transform_scale) else 1.0
            shift = config.input_transform_shift[i] if i < len(config.input_transform_shift) else 0.0
            row["scale"].setValue(scale)
            row["shift"].setValue(shift)
        self.output_transform_cb.setChecked(bool(config.output_transform_enabled))
        for i, row in enumerate(self.output_transform_rows):
            scale = config.output_transform_scale[i] if i < len(config.output_transform_scale) else 1.0
            shift = config.output_transform_shift[i] if i < len(config.output_transform_shift) else 0.0
            row["scale"].setValue(scale)
            row["shift"].setValue(shift)

        # Training / optimizers
        self.iter1_spin.setValue(config.iterations)
        self.opt1_combo.setCurrentText(config.optimizer)
        self.opt2_combo.setCurrentText(config.optimizer2)
        self.iter2_spin.setValue(config.iterations2)
        self.save_dir_input.setText(config.save_dir)
        self.lr_spin.setValue(config.learning_rate)
        self.loss_combo.setCurrentText(config.loss_type)

        # Inverse-problem settings -- trainable-variable rows. Rebuild from
        # inverse_variables_json when present (configs saved with the
        # multi-variable feature); otherwise fall back to the single
        # legacy inverse_param_name/inverse_param_init fields (configs
        # saved before this feature existed) so old saved configs still
        # load correctly with exactly one variable.
        for row in list(self.inv_var_rows):
            row['widget'].deleteLater()
        self.inv_var_rows.clear()
        try:
            _inv_vars = json.loads(config.inverse_variables_json) if config.inverse_variables_json else []
        except (json.JSONDecodeError, TypeError):
            _inv_vars = []
        if _inv_vars:
            for _v in _inv_vars:
                self._add_inverse_var_row(
                    _v.get('name', f"trainable_variable_{len(self.inv_var_rows) + 1}"),
                    _v.get('init', 1.0))
        else:
            self._add_inverse_var_row(
                config.inverse_param_name or "trainable_variable_1",
                config.inverse_param_init)
        # Inverse-problem settings -- measured-data-file rows. Rebuild from
        # inverse_obs_files_json when present; otherwise fall back to a
        # single row built from the legacy inverse_data_file/
        # inverse_obs_output_idx/loss_weight_obs fields, so old saved
        # configs still load correctly with exactly one measured-data
        # file.
        for row in list(self.inv_data_rows):
            row['widget'].deleteLater()
        self.inv_data_rows.clear()
        try:
            _obs_files = json.loads(config.inverse_obs_files_json) if config.inverse_obs_files_json else []
        except (json.JSONDecodeError, TypeError):
            _obs_files = []
        if _obs_files:
            for _f in _obs_files:
                self._add_inverse_data_row(
                    _f.get('path', ''), _f.get('output_idx', 0), _f.get('weight', 100.0))
        else:
            self._add_inverse_data_row(
                config.inverse_data_file, config.inverse_obs_output_idx, config.loss_weight_obs)
        # v19: config.inverse_ic_type/inverse_ic_file are no longer surfaced
        # in the UI (see _build_config()) -- a config saved by an older
        # version that still has "File (x, t, u)" in there just loads with
        # that field silently ignored; the Initial Condition panel above
        # (ic_active/ic_inputs/ic_from_file) is what actually restores now.
        self.inv_param_log_scale.setChecked(config.inv_param_log_scale)
        self.param_save_combo.setCurrentText(config.inv_param_save)

        # Adaptive training method (Forward-only options depend on problem
        # type already being set above)
        self._set_combo_data(self.adapt_combo, config.adapt_method)
        self.rar_cycles.setValue(config.rar_cycles)
        self.rar_candidates.setValue(config.rar_candidates)
        self.rar_add_points.setValue(config.rar_add_points)
        self.rar_adam_iters.setValue(config.rar_adam_iters)
        self.rar_lbfgs_iters.setValue(config.rar_lbfgs_iters)

        # Time-adaptive step groups
        for row in list(self.ta_group_rows):
            row["widget"].deleteLater()
        self.ta_group_rows.clear()
        try:
            groups = json.loads(config.ta_step_groups) if config.ta_step_groups else []
        except (json.JSONDecodeError, TypeError):
            groups = []
        if groups:
            for g in groups:
                self._add_ta_step_group(g.get("t_start", 0.0), g.get("t_end", 1.0), g.get("steps", 10))
        else:
            self._add_ta_step_group(config.t_min, config.t_max, config.ta_num_steps)
        self.ta_grid.setCurrentText(str(config.ta_grid_size))
        self.ta_transfer_cb.setChecked(config.ta_transfer_learning)
        self._set_combo_data(self.ta_transfer_opt, config.ta_transfer_optimizer)

        # Export / plot-output settings
        self.export_grid_combo.setCurrentText(str(config.export_grid_size))
        self.export_tsteps_spin.setValue(config.export_t_steps)
        if 0 <= config.plot_output_idx < self.plot_output_combo.count():
            self.plot_output_combo.setCurrentIndex(config.plot_output_idx)

        # L-BFGS
        self.lbfgs_use_default_cb.setChecked(config.lbfgs_use_default)
        self.lbfgs_maxcor.setValue(config.lbfgs_maxcor)
        self.lbfgs_ftol.setValue(config.lbfgs_ftol)
        self.lbfgs_gtol.setValue(config.lbfgs_gtol)
        self.lbfgs_maxiter.setValue(config.lbfgs_maxiter)
        self.lbfgs_maxfun.setValue(config.lbfgs_maxfun)
        self.lbfgs_maxls.setValue(config.lbfgs_maxls)
        if hasattr(self, "lbfgs_float_combo"):
            self.lbfgs_float_combo.setCurrentText(config.lbfgs_float_type)
        self._float_type = config.float_type

        # Optimizer / training-callback settings
        self._optimizer_settings = {
            "weight_decay": config.weight_decay,
        }
        self._callback_settings = {
            "early_stopping": config.cb_early_stopping,
            "early_stopping_min_delta": config.cb_early_stopping_min_delta,
            "early_stopping_patience": config.cb_early_stopping_patience,
            "early_stopping_baseline": config.cb_early_stopping_baseline,
            "early_stopping_monitor": config.cb_early_stopping_monitor,
            "early_stopping_start_from": config.cb_early_stopping_start_from,
            "point_resampler": config.cb_point_resampler,
            "point_resampler_period": config.cb_point_resampler_period,
            "point_resampler_pde_points": config.cb_point_resampler_pde_points,
            "point_resampler_bc_points": config.cb_point_resampler_bc_points,
            "model_checkpoint": config.cb_model_checkpoint,
            "checkpoint_period": config.cb_checkpoint_period,
            "checkpoint_save_better_only": config.cb_checkpoint_save_better_only,
            "checkpoint_monitor": config.cb_checkpoint_monitor,
            "timer": config.cb_timer,
            "timer_minutes": config.cb_timer_minutes,
        }

        # Mini-batch training
        if config.batch_size and config.batch_size > 0:
            self.batch_check.setChecked(True)
            self.batch_spin.setValue(config.batch_size)
        else:
            self.batch_check.setChecked(False)

        # IC pre-training
        self.ic_pretrain_cb.setChecked(config.ic_pretrain)
        self._set_combo_data(self.ic_pretrain_opt, config.ic_pretrain_optimizer)
        self.ic_pretrain_iters.setValue(config.ic_pretrain_iterations)
        self.ic_pretrain_test.setValue(config.ic_pretrain_num_test)
        self.ic_pretrain_init.setValue(config.ic_pretrain_num_initial)
        self.ic_pretrain_mode_restore.setChecked(bool(config.ic_pretrain_restore))
        self.ic_pretrain_mode_train.setChecked(not config.ic_pretrain_restore)
        self.ic_pretrain_restore_path.setText(config.ic_pretrain_restore_path)
        self._update_ic_pretrain_visibility()

        # Optimizer scheduler + weights. Phase rows (optimizer/iterations/lr
        # per phase) are only present in the saved file when the scheduler was
        # actually enabled at save time; rebuild them first so weight-widget
        # generation below (which depends on phase count when weights differ
        # per phase) sees the right shape.
        self.sched_cb.setChecked(config.optimizer_scheduler)
        self.sched_same_weights_cb.setChecked(config.scheduler_same_weights)
        phases = []
        if config.scheduler_phases:
            phases = self._apply_scheduler_phases_build_only(config.scheduler_phases)
        self._build_weight_inputs(n_out)
        # loss_weights_multi is the authoritative flat/shared weight set --
        # it's a no-op when scheduler_same_weights is False, since in that
        # case weight_widgets only contains the _p{n}-suffixed keys.
        self._apply_flat_weights(config.loss_weights_multi, n_out, is_2d)
        if phases and not config.scheduler_same_weights:
            self._apply_phase_weights(phases, n_out, is_2d)

        # Plot / visualization settings
        self._plot_viz_settings = {
            "colormap": config.plot_colormap,
            "levels": config.plot_levels,
            "resolution": config.plot_resolution,
            "dpi": config.plot_dpi,
            "colorbar": config.plot_colorbar,
            "auto_range": config.plot_auto_range,
            "vmin": config.plot_vmin,
            "vmax": config.plot_vmax,
            "linewidth": config.plot_linewidth,
            "n_2d_snapshots": config.plot_n_2d_snapshots,
        }

        # Error-analysis settings
        try:
            ea_files = _ast.literal_eval(config.ea_files) if config.ea_files else []
        except (ValueError, SyntaxError):
            ea_files = []
        self._ea_settings = {
            "files": ea_files,
            "do_line": config.ea_do_line,
            "do_surface": config.ea_do_surface,
        } if ea_files else None

        self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())

    def _on_num_outputs_changed(self, n):
        self._build_pde_inputs(n)
        self._build_bc_inputs(n)
        self._build_weight_inputs(n)
        self._rebuild_output_transform_rows(n)
        self.plot_output_combo.clear()
        self.restore_output_combo.clear()
        for _r in getattr(self, 'inv_data_rows', []):
            _r['output_combo'].clear()
        for i in range(n):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            self.plot_output_combo.addItem(f"Output {i+1} ({name})")
            self.restore_output_combo.addItem(f"Output {i+1} ({name})")
            for _r in getattr(self, 'inv_data_rows', []):
                _r['output_combo'].addItem(f"Output {i+1} ({name})")

    # Per built-in template: (PDE row index, original constant substring)
    # of the "diffusion coefficient"-style constant that gets swapped for
    # the inverse trainable variable.
    INVERSE_AUTO_CONST = {
        "1D Heat": (0, "0.4"),
        "1D Allen-Cahn": (0, "0.0001"),
        "2D Heat": (0, "0.4"),
        "2D Allen-Cahn (Mattey & Ghosh)": (0, "0.0001"),
        "2D Allen-Cahn (Wight & Zhao)": (0, "0.00625"),
        "3D Heat": (0, "0.4"),
    }

    def _sync_inverse_pde_substitution(self, is_inv):
        """Inverse ON: replace the current template's known constant with
        the trainable-variable name in its PDE box. Inverse OFF: restore
        the original numeric constant so Forward mode stays valid."""
        template = getattr(self, '_current_template', '')
        entry = self.INVERSE_AUTO_CONST.get(template)
        if is_inv:
            if not entry:
                return
            idx, const_str = entry
            if idx >= len(self.pde_inputs):
                return
            var_name = self.inv_param_name.text().strip() or "trainable_variable_1"
            text = self.pde_inputs[idx].text()
            if const_str in text:
                self.pde_inputs[idx].setText(text.replace(const_str, var_name, 1))
                self._inverse_sub_active = (template, idx, const_str, var_name)
        else:
            state = getattr(self, '_inverse_sub_active', None)
            if state:
                s_template, s_idx, s_const, s_var = state
                if s_idx < len(self.pde_inputs):
                    cur = self.pde_inputs[s_idx].text()
                    if s_var in cur:
                        self.pde_inputs[s_idx].setText(cur.replace(s_var, s_const, 1))
                self._inverse_sub_active = None

    def _on_inv_param_name_changed(self):
        """Keep an already-substituted PDE in sync if the trainable
        variable name is renamed while Inverse mode is active."""
        state = getattr(self, '_inverse_sub_active', None)
        if not state:
            return
        template, idx, const_str, old_var = state
        new_var = self.inv_param_name.text().strip()
        if not new_var or new_var == old_var or idx >= len(self.pde_inputs):
            return
        text = self.pde_inputs[idx].text()
        if old_var in text:
            self.pde_inputs[idx].setText(text.replace(old_var, new_var, 1))
            self._inverse_sub_active = (template, idx, const_str, new_var)

    def _on_problem_type_changed(self, checked):
        is_inv = self.radio_inverse.isChecked()
        self._sync_inverse_pde_substitution(is_inv)
        self.inverse_group.setVisible(is_inv)
        self.param_save_label.setVisible(is_inv)
        self.param_save_combo.setVisible(is_inv)
        # Time Adaptive training does not yet wire the inferred parameter
        # into its per-step training loop (no external_trainable_variables
        # there), so it silently would not converge for Inverse problems --
        # remove it as a selectable option in Inverse mode and restore any
        # template default that was suspended when returning to Forward.
        _ta_idx = self.adapt_combo.findText("Time Adaptive")
        if is_inv:
            if self.adapt_combo.currentText() == "Time Adaptive":
                self._ta_suspended_for_inverse = True
                self.adapt_combo.setCurrentText("None")
            if _ta_idx != -1:
                self.adapt_combo.removeItem(_ta_idx)
        else:
            if _ta_idx == -1:
                self.adapt_combo.addItem("Time Adaptive", "Time Adaptive")
            if getattr(self, '_ta_suspended_for_inverse', False):
                self._ta_suspended_for_inverse = False
                _ta_cfg = getattr(self, '_current_ta_cfg', None)
                if _ta_cfg:
                    for _row in list(self.ta_group_rows):
                        _row['widget'].deleteLater()
                    self.ta_group_rows.clear()
                    self.adapt_combo.setCurrentText("Time Adaptive")
                    for _g_start, _g_end, _g_steps in _ta_cfg['step_groups']:
                        self._add_ta_step_group(_g_start, _g_end, _g_steps)
                    self.ta_transfer_cb.setChecked(_ta_cfg.get('transfer_learning', False))
                    self.ta_grid.setCurrentText(str(_ta_cfg.get('ic_grid', 101)))
                    self._set_combo_data(self.ta_transfer_opt, _ta_cfg.get('transfer_optimizer', 'adam'))
        # 2D Allen-Cahn (Wight & Zhao): Inverse trains over a shorter t
        # range than Forward (see the template's inverse_t_max) since
        # fitting the unknown parameter over the full t in [0,10] window
        # used for Forward is much harder -- restore whichever t_max
        # belongs to the problem type we just switched to.
        _fwd_tmax = getattr(self, '_current_template_forward_tmax', None)
        _inv_tmax = getattr(self, '_current_template_inverse_tmax', None)
        if _inv_tmax is not None and _fwd_tmax is not None:
            self.t_max.setValue(_inv_tmax if is_inv else _fwd_tmax)
            self._auto_configure_ea(getattr(self, '_template_ref_dir', ''))

    def _on_browse_inv_data(self, target=None):
        if target is None:
            target = self.inv_data_path
        f, _ = QFileDialog.getOpenFileName(self, "Select measured data file", "", "Data files (*.txt *.csv *.dat)")
        if f:
            target.setText(f)


    def _geometry_supported_for_training(self):
        """Phase 2 of the geometry-type feature: codegen.py's _build_geom()
        now constructs every shape in the Geometry Type selector (Interval;
        Rectangle/Disk/Ellipse/Triangle/Polygon in 2D; Cuboid/Sphere in
        3D), so Solve is no longer blocked for any of them. Kept as a real
        gate (rather than deleted outright) so a future shape added to the
        selector without matching codegen support fails the same clean,
        explicit way Phase 1 did, instead of silently training on the
        wrong geometry."""
        geom_type = self._current_geometry_type()
        _supported = ("Interval", "Rectangle", "Disk", "Ellipse", "Triangle",
                      "Polygon", "Cuboid", "Sphere")
        if geom_type not in _supported:
            return False, (
                f"⚠️ Training for '{geom_type}' geometry isn't wired up yet -- "
                f"switch Geometry Type to one of {', '.join(_supported)} to train."
            )
        return True, ""

    def _validate_optimizer_settings(self, config):
        """Catches three real DeepXDE incompatibilities before a run starts,
        rather than letting them surface as a mid-training crash or a
        cryptic traceback: (1) weight_decay > 0 combined with L-BFGS or
        NNCG anywhere in the run -- both raise ValueError for any nonzero
        weight_decay; (2) AdamW selected with weight_decay == 0 -- DeepXDE
        raises ValueError since AdamW requires non-zero weight decay; (3)
        NNCG selected without a new-enough deepxde installed -- NNCG was
        added in deepxde 1.13.0, and this app's minimum pinned version is
        1.10.0, so older installs would hit a NotImplementedError deep
        inside training instead of a clear upgrade message."""
        import json as _json_val
        try:
            phases = _json_val.loads(config.scheduler_phases) if config.scheduler_phases else []
        except (ValueError, TypeError):
            phases = []
        # Include the legacy single phase-2 optimizer field too -- it's
        # hidden in the current UI (the Training Phases scheduler is always
        # on), but still drives training for a config saved before the
        # scheduler existed and then re-solved without ever touching the
        # (now-hidden) phase-2 widgets.
        phase_opts = [p.get("optimizer") for p in phases] + [config.optimizer2]
        uses_lbfgs_or_nncg = (
            "lbfgs" in phase_opts or "nncg" in phase_opts
            or (config.adapt_method == "RAR" and config.rar_lbfgs_iters > 0)
        )
        if config.weight_decay > 0 and uses_lbfgs_or_nncg:
            return False, (
                "⚠️ Weight decay is set > 0 (Settings > Optimizer Settings), but "
                "this run uses L-BFGS and/or NNCG somewhere (Training Phases, or "
                "RAR's optional L-BFGS polish) — both reject any nonzero weight "
                "decay. Set weight decay to 0, or remove L-BFGS/NNCG from the "
                "run, before solving.")
        if "adamw" in phase_opts and config.weight_decay == 0:
            return False, (
                "⚠️ A Training Phase is set to use AdamW, which requires a "
                "nonzero weight decay. Set one in Settings > Optimizer "
                "Settings, or switch that phase to Adam instead.")
        if "nncg" in phase_opts:
            try:
                import deepxde as _dde_check
                _ver = tuple(int(x) for x in _dde_check.__version__.split(".")[:3])
            except Exception:
                _ver = None
            if _ver is not None and _ver < (1, 13, 0):
                return False, (
                    f"⚠️ A Training Phase is set to use NNCG, which requires "
                    f"deepxde>=1.13.0 (you have {_dde_check.__version__} installed). "
                    f"Upgrade with: pip install --upgrade deepxde — or switch that "
                    f"phase to Adam or L-BFGS instead.")
        return True, ""

    def _on_solve(self):
        _ok, _msg = self._geometry_supported_for_training()
        if not _ok:
            self.log_box.append(_msg)
            return
        _config_preview = self._build_config()
        _ok2, _msg2 = self._validate_optimizer_settings(_config_preview)
        if not _ok2:
            self.log_box.append(_msg2)
            return
        self.solve_btn.setEnabled(False)
        self.solve_btn.setText("⏳  Solving...")
        self.stop_btn.setEnabled(True)
        self.log_box.clear()
        self.loss_label.setText("⏳ Training...")
        self.solution_label.setText("⏳ Training...")
        for _p in ["/tmp/loss_plot.png", "/tmp/solution_plot.png", "/tmp/param_plot.png"]:
            if os.path.exists(_p):
                os.remove(_p)
        config = _config_preview
        self.thread = SolverThread(config)
        self.thread.output_signal.connect(self._on_output)
        self.thread.done_signal.connect(self._on_done)
        self.thread.start()

    def _on_stop(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.log_box.append("\n⏹ Stopped by user.")
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText("▶  Solve")
            self.stop_btn.setEnabled(False)

    def _on_output(self, line):
        self.log_box.append(line)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _on_done(self, result):
        self.solve_btn.setEnabled(True)
        self.solve_btn.setText("▶  Solve")
        self.stop_btn.setEnabled(False)

        if result == "DONE":
            self.log_box.append("\n✅ Training complete!")
            self._last_config = self._build_config()

            save_dir = self.save_dir_input.text().strip()
            sol_dir = os.path.join(save_dir, "solution_results") if save_dir else "/tmp"
            loss_path = os.path.join(sol_dir, "loss_plot.png")
            solution_path = os.path.join(sol_dir, "solution_plot.png")
            # Fallback to root save dir for older runs
            if not os.path.exists(loss_path):
                loss_path = os.path.join(save_dir, "loss_plot.png") if save_dir else "/tmp/loss_plot.png"
            if not os.path.exists(solution_path):
                solution_path = os.path.join(save_dir, "solution_plot.png") if save_dir else "/tmp/solution_plot.png"

            if os.path.exists(loss_path):
                self.loss_label.setPixmap(QPixmap(loss_path).scaled(
                    self.loss_label.width(), self.loss_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            if os.path.exists(solution_path):
                self.solution_label.setPixmap(QPixmap(solution_path).scaled(
                    self.solution_label.width(), self.solution_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            
            if save_dir:
                self.log_box.append(f"💾 Results saved to: {save_dir}")

            if hasattr(self, '_ea_settings') and self._ea_settings:
                self.log_box.append("✅ Error analysis ran inline — check error_analysis/ folder.")
                self._ea_settings = None
            else:
                self.log_box.append("ℹ️ No error analysis configured — click '📊 Error Analysis' before training.")

        else:
            self.log_box.append("\n❌ Error during training. Check log above.")
            self.log_box.append("💡 Tip: Check PDE/IC syntax — use * for multiplication (e.g. 5*u not 5u)")

    def _on_float_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Float Precision")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)

        info = QLabel("Float64 recommended for L-BFGS convergence.\nFloat32 is faster but L-BFGS may stop early.")
        self._register_style(info, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        info.setWordWrap(True)
        layout.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Float type:"))
        self._float_combo = QComboBox()
        self._float_combo.addItems(["float64", "float32"])
        self._float_combo.setCurrentText(getattr(self, '_float_type', 'float32'))
        self._float_combo.setFixedWidth(100)
        row.addStretch(); row.addWidget(self._float_combo)
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._float_type = self._float_combo.currentText()
            self.log_box.append(f"✅ Float precision set to: {self._float_type}")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_lbfgs_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("L-BFGS Settings")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        use_default = QCheckBox("Use DeepXDE defaults")
        use_default.setChecked(self.lbfgs_use_default_cb.isChecked())
        layout.addWidget(use_default)

        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)
        manual_layout.setSpacing(6)

        def _make_row(label, val):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = SciLineEdit(val); sb.setFixedWidth(130)
            row.addStretch(); row.addWidget(sb)
            manual_layout.addLayout(row)
            return sb

        sb_maxcor  = _make_row("maxcor:",  self.lbfgs_maxcor.value())
        sb_ftol    = _make_row("ftol:",    self.lbfgs_ftol.value())
        sb_gtol    = _make_row("gtol:",    self.lbfgs_gtol.value())
        sb_maxiter = _make_row("maxiter:", self.lbfgs_maxiter.value())
        sb_maxfun  = _make_row("maxfun:",  self.lbfgs_maxfun.value())
        sb_maxls   = _make_row("maxls:",   self.lbfgs_maxls.value())

        manual_widget.setVisible(not use_default.isChecked())
        layout.addWidget(manual_widget)
        use_default.stateChanged.connect(lambda s: manual_widget.setVisible(s != 2))

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self.lbfgs_use_default_cb.setChecked(use_default.isChecked())
            self.lbfgs_maxcor.setValue(sb_maxcor.value())
            self.lbfgs_ftol.setValue(sb_ftol.value())
            self.lbfgs_gtol.setValue(sb_gtol.value())
            self.lbfgs_maxiter.setValue(sb_maxiter.value())
            self.lbfgs_maxfun.setValue(sb_maxfun.value())
            self.lbfgs_maxls.setValue(sb_maxls.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_optimizer_settings(self):
        """Weight decay (L2 regularization) -- applies to the whole network
        being trained (main model, every scheduler phase, IC pre-training,
        RAR refinement), since DeepXDE sets this at network-construction
        time rather than per optimizer call. NOT compatible with L-BFGS or
        NNCG (DeepXDE raises an error if weight_decay > 0 for either) --
        checked at Solve time in _validate_optimizer_settings(), not just
        left to crash mid-run."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Optimizer Settings")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)

        note = QLabel(
            "Weight decay (L2 regularization) applies to Adam, AdamW, SGD\n"
            "and RMSprop phases. It is NOT compatible with L-BFGS or NNCG --\n"
            "leave it at 0 if any phase in Training Phases uses either of\n"
            "those. AdamW requires a non-zero weight decay to be selected.")
        note.setWordWrap(True)
        self._register_style(note, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        layout.addWidget(note)

        wd_row = QHBoxLayout()
        wd_row.addWidget(QLabel("Weight decay (L2):"))
        wd_spin = SciLineEdit(self._optimizer_settings.get("weight_decay", 0.0))
        wd_spin.setFixedWidth(130)
        wd_row.addStretch(); wd_row.addWidget(wd_spin)
        layout.addLayout(wd_row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._optimizer_settings["weight_decay"] = wd_spin.value()
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_callback_settings(self):
        """Optional DeepXDE training callbacks -- all off by default. Applied
        to the live Training Phases scheduler (and the legacy single/dual-
        phase fallback), NOT to IC pre-training or the RAR refinement
        sub-loop, which are short, purpose-built inner loops of their own
        (see codegen.py's _train_cbs construction for the full scoping
        note)."""
        from PyQt6.QtWidgets import QScrollArea
        dialog = QDialog(self)
        dialog.setWindowTitle("Training Callbacks")
        dialog.setMinimumWidth(420)
        dialog.setMinimumHeight(520)
        outer = QVBoxLayout(dialog)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        cs = self._callback_settings

        # ── Early Stopping ──────────────────────────────────
        es_group = QGroupBox("Early Stopping")
        es_layout = QVBoxLayout(es_group)
        es_cb = QCheckBox("Stop training when loss stops improving")
        es_cb.setChecked(cs.get("early_stopping", False))
        es_layout.addWidget(es_cb)
        es_fields = QWidget()
        es_fl = QVBoxLayout(es_fields); es_fl.setContentsMargins(0, 0, 0, 0)

        def _es_row(label, widget):
            row = QHBoxLayout(); row.addWidget(QLabel(label))
            row.addStretch(); row.addWidget(widget)
            es_fl.addLayout(row)

        es_min_delta = SciLineEdit(cs.get("early_stopping_min_delta", 0.0)); es_min_delta.setFixedWidth(110)
        _es_row("Min. delta:", es_min_delta)
        es_patience = QSpinBox(); es_patience.setRange(1, 1000000); es_patience.setSingleStep(100)
        es_patience.setValue(int(cs.get("early_stopping_patience", 2000))); es_patience.setFixedWidth(110)
        _es_row("Patience (iters):", es_patience)
        es_baseline = QLineEdit(str(cs.get("early_stopping_baseline", "") or ""))
        es_baseline.setPlaceholderText("(none)"); es_baseline.setFixedWidth(110)
        _es_row("Baseline loss:", es_baseline)
        es_monitor = QComboBox()
        es_monitor.addItem("Training loss", "loss_train")
        es_monitor.addItem("Testing loss", "loss_test")
        self._set_combo_data(es_monitor, cs.get("early_stopping_monitor", "loss_train"))
        es_monitor.setFixedWidth(110)
        _es_row("Monitor:", es_monitor)
        es_start = QSpinBox(); es_start.setRange(0, 1000000); es_start.setSingleStep(100)
        es_start.setValue(int(cs.get("early_stopping_start_from", 0))); es_start.setFixedWidth(110)
        _es_row("Start after (iters):", es_start)
        es_layout.addWidget(es_fields)
        es_fields.setVisible(es_cb.isChecked())
        es_cb.stateChanged.connect(lambda s: es_fields.setVisible(s == 2))
        layout.addWidget(es_group)

        # ── Point Resampling ────────────────────────────────
        pr_group = QGroupBox("Point Resampling")
        pr_layout = QVBoxLayout(pr_group)
        pr_cb = QCheckBox("Periodically resample collocation points")
        pr_cb.setChecked(cs.get("point_resampler", False))
        pr_layout.addWidget(pr_cb)
        pr_fields = QWidget()
        pr_fl = QVBoxLayout(pr_fields); pr_fl.setContentsMargins(0, 0, 0, 0)
        pr_period = QSpinBox(); pr_period.setRange(1, 1000000); pr_period.setSingleStep(10)
        pr_period.setValue(int(cs.get("point_resampler_period", 100))); pr_period.setFixedWidth(110)
        _pr_row = QHBoxLayout(); _pr_row.addWidget(QLabel("Resample every (iters):"))
        _pr_row.addStretch(); _pr_row.addWidget(pr_period)
        pr_fl.addLayout(_pr_row)
        pr_pde = QCheckBox("Resample PDE (domain) points")
        pr_pde.setChecked(cs.get("point_resampler_pde_points", True))
        pr_fl.addWidget(pr_pde)
        pr_bc = QCheckBox("Also resample boundary-condition points")
        pr_bc.setChecked(cs.get("point_resampler_bc_points", False))
        pr_fl.addWidget(pr_bc)
        pr_layout.addWidget(pr_fields)
        pr_fields.setVisible(pr_cb.isChecked())
        pr_cb.stateChanged.connect(lambda s: pr_fields.setVisible(s == 2))
        layout.addWidget(pr_group)

        # ── Model Checkpoint ─────────────────────────────────
        ck_group = QGroupBox("Model Checkpoint")
        ck_layout = QVBoxLayout(ck_group)
        ck_cb = QCheckBox("Periodically save the model during training")
        ck_cb.setChecked(cs.get("model_checkpoint", False))
        ck_layout.addWidget(ck_cb)
        ck_fields = QWidget()
        ck_fl = QVBoxLayout(ck_fields); ck_fl.setContentsMargins(0, 0, 0, 0)
        ck_period = QSpinBox(); ck_period.setRange(1, 1000000); ck_period.setSingleStep(100)
        ck_period.setValue(int(cs.get("checkpoint_period", 1000))); ck_period.setFixedWidth(110)
        _ck_row = QHBoxLayout(); _ck_row.addWidget(QLabel("Check every (iters):"))
        _ck_row.addStretch(); _ck_row.addWidget(ck_period)
        ck_fl.addLayout(_ck_row)
        ck_better = QCheckBox("Only save when the monitored loss improves")
        ck_better.setChecked(cs.get("checkpoint_save_better_only", True))
        ck_fl.addWidget(ck_better)
        ck_monitor = QComboBox()
        ck_monitor.addItem("Training loss", "train loss")
        ck_monitor.addItem("Testing loss", "test loss")
        self._set_combo_data(ck_monitor, cs.get("checkpoint_monitor", "train loss"))
        ck_monitor.setFixedWidth(110)
        _ck_mrow = QHBoxLayout(); _ck_mrow.addWidget(QLabel("Monitor:"))
        _ck_mrow.addStretch(); _ck_mrow.addWidget(ck_monitor)
        ck_fl.addLayout(_ck_mrow)
        ck_layout.addWidget(ck_fields)
        ck_fields.setVisible(ck_cb.isChecked())
        ck_cb.stateChanged.connect(lambda s: ck_fields.setVisible(s == 2))
        layout.addWidget(ck_group)

        # ── Timer ────────────────────────────────────────────
        tm_group = QGroupBox("Training Timer")
        tm_layout = QVBoxLayout(tm_group)
        tm_cb = QCheckBox("Stop training after a time budget")
        tm_cb.setChecked(cs.get("timer", False))
        tm_layout.addWidget(tm_cb)
        tm_fields = QWidget()
        tm_fl = QVBoxLayout(tm_fields); tm_fl.setContentsMargins(0, 0, 0, 0)
        tm_minutes = QDoubleSpinBox(); tm_minutes.setRange(0.5, 100000); tm_minutes.setDecimals(1)
        tm_minutes.setValue(float(cs.get("timer_minutes", 60.0))); tm_minutes.setFixedWidth(110)
        _tm_row = QHBoxLayout(); _tm_row.addWidget(QLabel("Available time (minutes):"))
        _tm_row.addStretch(); _tm_row.addWidget(tm_minutes)
        tm_fl.addLayout(_tm_row)
        tm_layout.addWidget(tm_fields)
        tm_fields.setVisible(tm_cb.isChecked())
        tm_cb.stateChanged.connect(lambda s: tm_fields.setVisible(s == 2))
        layout.addWidget(tm_group)

        layout.addStretch()

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        outer.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._callback_settings = {
                "early_stopping": es_cb.isChecked(),
                "early_stopping_min_delta": es_min_delta.value(),
                "early_stopping_patience": es_patience.value(),
                "early_stopping_baseline": es_baseline.text().strip(),
                "early_stopping_monitor": es_monitor.currentData(),
                "early_stopping_start_from": es_start.value(),
                "point_resampler": pr_cb.isChecked(),
                "point_resampler_period": pr_period.value(),
                "point_resampler_pde_points": pr_pde.isChecked(),
                "point_resampler_bc_points": pr_bc.isChecked(),
                "model_checkpoint": ck_cb.isChecked(),
                "checkpoint_period": ck_period.value(),
                "checkpoint_save_better_only": ck_better.isChecked(),
                "checkpoint_monitor": ck_monitor.currentData(),
                "timer": tm_cb.isChecked(),
                "timer_minutes": tm_minutes.value(),
            }
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_view_domain_changed(self, state):
        if state == 2:
            self._preview_domain()
        else:
            self.loss_label.setText("📉 Loss plot")
            self.solution_label.setText("🗺 Solution plot")

    def _preview_domain(self):
        import tempfile, subprocess, sys, math
        geom_type = self._current_geometry_type()
        if geom_type not in ("Rectangle", "Disk", "Ellipse", "Triangle", "Polygon", "Cuboid", "Sphere"):
            self.log_box.append(
                f"⚠️ Domain preview for '{geom_type}' geometry isn't supported yet."
            )
            self.loss_label.setText("📉 Loss plot")
            self.solution_label.setText("🗺 Solution plot")
            return
        t_min = self.t_min.value(); t_max = self.t_max.value()
        n_domain   = self.num_domain.value()
        n_boundary = self.num_boundary.value()
        n_initial  = self.num_initial.value()
        dist       = self.pts_dist_combo.currentText()

        try:
            self.pts_dist_combo.currentTextChanged.disconnect(self._on_dist_changed)
        except Exception:
            pass
        self.pts_dist_combo.currentTextChanged.connect(self._on_dist_changed)
        for sb in [self.num_domain, self.num_boundary, self.num_initial]:
            try:
                sb.valueChanged.disconnect(self._on_pts_changed)
            except Exception:
                pass
            sb.valueChanged.connect(self._on_pts_changed)

        is_3d_shape = geom_type in ("Cuboid", "Sphere")
        if is_3d_shape:
            script = self._build_3d_preview_script(
                geom_type, t_min, t_max, n_domain, n_boundary, n_initial, dist)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
                tf.write(script)
                tmp = tf.name
            self._launch_preview_thread(tmp)
            return

        # Build the DeepXDE geometry constructor call, a matching matplotlib
        # outline patch, and the bounding box for the plot axes -- one branch
        # per shape, reading that shape's own parameters. x_min/x_max/y_min/
        # y_max only mean something for Rectangle; the others (center +
        # radius, semi-axes, vertex list) live on their own widgets.
        if geom_type == "Rectangle":
            x_min = self.x_min.value(); x_max = self.x_max.value()
            y_min = self.y_min.value(); y_max = self.y_max.value()
            geom_code = f"dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])"
            patch_code = (
                f"plt.Rectangle(({x_min},{y_min}), {x_max}-{x_min}, {y_max}-{y_min}, "
                "linewidth=2, edgecolor='#a0c4ff', facecolor='none')"
            )
            bbox = (x_min, x_max, y_min, y_max)
        elif geom_type == "Disk":
            cx = self.geom_disk_cx.value(); cy = self.geom_disk_cy.value(); r = self.geom_disk_r.value()
            geom_code = f"dde.geometry.Disk([{cx}, {cy}], {r})"
            patch_code = f"plt.Circle(({cx},{cy}), {r}, linewidth=2, edgecolor='#a0c4ff', facecolor='none')"
            bbox = (cx - r, cx + r, cy - r, cy + r)
        elif geom_type == "Ellipse":
            cx = self.geom_ellipse_cx.value(); cy = self.geom_ellipse_cy.value()
            a = self.geom_ellipse_a.value(); b = self.geom_ellipse_b.value()
            angle = self.geom_ellipse_angle.value()
            geom_code = f"dde.geometry.Ellipse([{cx}, {cy}], {a}, {b}, {angle})"
            patch_code = (
                f"matplotlib.patches.Ellipse(({cx},{cy}), {2*a}, {2*b}, angle={math.degrees(angle)}, "
                "linewidth=2, edgecolor='#a0c4ff', facecolor='none')"
            )
            dx = math.sqrt((a * math.cos(angle)) ** 2 + (b * math.sin(angle)) ** 2)
            dy = math.sqrt((a * math.sin(angle)) ** 2 + (b * math.cos(angle)) ** 2)
            bbox = (cx - dx, cx + dx, cy - dy, cy + dy)
        else:  # Triangle / Polygon
            verts_text = (self.geom_triangle_verts_input.text() if geom_type == "Triangle"
                          else self.geom_polygon_verts_input.text())
            verts = self._parse_vertices(verts_text)
            if len(verts) < 3 or (geom_type == "Triangle" and len(verts) != 3):
                self.log_box.append(
                    f"⚠️ Enter {'exactly 3' if geom_type == 'Triangle' else 'at least 3'} "
                    f"valid vertices for a {geom_type} domain preview."
                )
                self.loss_label.setText("📉 Loss plot")
                self.solution_label.setText("🗺 Solution plot")
                return
            if geom_type == "Polygon" and len(verts) == 3:
                self.log_box.append(
                    "⚠️ 3 vertices is a Triangle, not a Polygon -- switch Geometry "
                    "Type to Triangle, or add a 4th vertex."
                )
                self.loss_label.setText("📉 Loss plot")
                self.solution_label.setText("🗺 Solution plot")
                return
            if geom_type == "Polygon" and self._is_axis_aligned_rectangle(verts):
                self.log_box.append(
                    "⚠️ Those 4 vertices form an axis-aligned rectangle -- DeepXDE "
                    "requires the Rectangle shape for that. Switch Geometry Type to "
                    "Rectangle, or edit a vertex so it's not a rectangle."
                )
                self.loss_label.setText("📉 Loss plot")
                self.solution_label.setText("🗺 Solution plot")
                return
            vlist = [list(v) for v in verts]
            if geom_type == "Triangle":
                geom_code = f"dde.geometry.Triangle({vlist[0]}, {vlist[1]}, {vlist[2]})"
            else:
                geom_code = f"dde.geometry.Polygon({vlist})"
            patch_code = f"plt.Polygon({vlist}, closed=True, linewidth=2, edgecolor='#a0c4ff', facecolor='none')"
            xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
            bbox = (min(xs), max(xs), min(ys), max(ys))

        bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox
        pad_x = max((bbox_x_max - bbox_x_min) * 0.05, 1e-6)
        pad_y = max((bbox_y_max - bbox_y_min) * 0.05, 1e-6)
        xlim_lo, xlim_hi = bbox_x_min - pad_x, bbox_x_max + pad_x
        ylim_lo, ylim_hi = bbox_y_min - pad_y, bbox_y_max + pad_y

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

geom  = {geom_code}
t_dom = dde.geometry.TimeDomain({t_min}, {t_max})
gt    = dde.geometry.GeometryXTime(geom, t_dom)

def pde(x, y): return y[:, 0:1] * 0
data = dde.data.TimePDE(gt, pde, [],
    num_domain={n_domain}, num_boundary={n_boundary},
    num_initial={n_initial}, num_test=10,
    train_distribution="{dist}")

pts = data.train_points()
t_range = {t_max} - {t_min}
tol = max(t_range * 0.05, 1e-6)

plt.rcParams['figure.dpi'] = 120
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#1e1e1e')

for ax in axes:
    ax.set_facecolor('#252526')
    ax.tick_params(colors='#c0c0c0')
    for sp in ax.spines.values(): sp.set_color('#3e3e42')

# The shape outline is a spatial (x,y) patch -- it only makes sense on the
# left, spatial panel. The right panel's y-axis is time, not y, so drawing
# the same patch there would show the shape's y-extent as if it were a time
# range, which is meaningless.
axes[0].set_xlim({xlim_lo}, {xlim_hi})
axes[0].set_ylim({ylim_lo}, {ylim_hi})
shape_patch = {patch_code}
axes[0].add_patch(shape_patch)

# Classify points using the geometry's own on_boundary() test -- this works
# for any 2D shape (circle, ellipse, triangle, polygon, ...), not just an
# axis-aligned box.
ic_mask  = pts[:, 2] <= {t_min} + tol
bnd_mask = gt.geometry.on_boundary(pts[:, :2]) & ~ic_mask
dom_mask = ~ic_mask & ~bnd_mask
dom_pts = pts[dom_mask]
bnd_pts = pts[bnd_mask]
ic_pts  = pts[ic_mask]

# Left: spatial distribution (x,y) -- domain + boundary only. IC points
# are t=t_min points that live inside the domain, so overlaying them here
# just adds clutter on top of the domain cloud; they're already shown
# clearly in the time panel on the right.
ax = axes[0]
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,1], s=8, c='#74c0fc', alpha=0.7, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,1], s=20, c='#f03e3e', alpha=1.0, label=f'Boundary ({{len(bnd_pts)}})')
ax.set_xlabel('x', color='#e0e0e0', fontsize=11)
ax.set_ylabel('y', color='#e0e0e0', fontsize=11)
ax.set_title('{geom_type}: Spatial (x,y) | {dist} | D={n_domain} B={n_boundary} IC={n_initial}', color='#74c0fc', fontsize=10, fontweight='bold')
ax.legend(fontsize=10, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5)

# Right: time distribution (x vs t)
ax = axes[1]
ax.set_xlim({xlim_lo}, {xlim_hi}); ax.set_ylim({t_min}, {t_max})
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,2], s=6, c='#74c0fc', alpha=0.5, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,2], s=14, c='#f03e3e', alpha=0.9, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax.scatter(ic_pts[:,0],  ic_pts[:,2],  s=14, c='#2f9e44', alpha=1.0, label=f'IC ({{len(ic_pts)}})')
ax.set_xlabel('x', color='#e0e0e0', fontsize=11)
ax.set_ylabel('t', color='#e0e0e0', fontsize=11)
ax.set_title(f'Time Distribution (x vs t)\\ntotal={{len(pts)}} points', color='#74c0fc', fontsize=10, fontweight='bold')
ax.legend(fontsize=10, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5)
plt.tight_layout()
plt.savefig('/tmp/domain_preview.png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
plt.close()
print("DOMAIN_PREVIEW_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script)
            tmp = tf.name
        self._launch_preview_thread(tmp)

    def _launch_preview_thread(self, tmp):
        """Run a generated domain-preview script (2D or 3D) in a background
        thread and stream its output into the log box, same as before this
        was factored out -- shared by both the 2D and 3D preview paths."""
        self.loss_label.setText("⏳ Generating preview...")

        from PyQt6.QtCore import QThread, pyqtSignal as _sig

        class _PreviewThread(QThread):
            done_sig = _sig(bool)
            log_sig  = _sig(str)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.log_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._preview_thread = _PreviewThread(tmp)
        self._preview_thread.done_sig.connect(self._on_preview_done)
        self._preview_thread.log_sig.connect(self.log_box.append)
        self._preview_thread.start()

    def _build_3d_preview_script(self, geom_type, t_min, t_max, n_domain, n_boundary, n_initial, dist):
        """Build the domain-preview script for a 3D shape (Cuboid or
        Sphere). Points here are (x, y, z, t) -- 4 columns instead of the
        2D preview's 3 -- so this uses a real 3D scatter (mpl_toolkits
        Axes3D) for the spatial panel instead of the 2D preview's flat
        (x,y) panel, plus the same x-vs-t time panel as before. Boundary
        classification still goes through the geometry's own on_boundary()
        test, same as the 2D preview."""
        if geom_type == "Cuboid":
            x_min = self.x_min.value(); x_max = self.x_max.value()
            y_min = self.y_min.value(); y_max = self.y_max.value()
            z_min = self.z_min.value(); z_max = self.z_max.value()
            geom_code = f"dde.geometry.Cuboid([{x_min}, {y_min}, {z_min}], [{x_max}, {y_max}, {z_max}])"
            bbox3 = (x_min, x_max, y_min, y_max, z_min, z_max)
            outline_code = (
                f"_x0, _x1, _y0, _y1, _z0, _z1 = {x_min}, {x_max}, {y_min}, {y_max}, {z_min}, {z_max}\n"
                "_corners = [(_x0,_y0,_z0),(_x1,_y0,_z0),(_x1,_y1,_z0),(_x0,_y1,_z0),\n"
                "            (_x0,_y0,_z1),(_x1,_y0,_z1),(_x1,_y1,_z1),(_x0,_y1,_z1)]\n"
                "_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]\n"
                "for _i, _j in _edges:\n"
                "    _p0, _p1 = _corners[_i], _corners[_j]\n"
                "    ax.plot([_p0[0],_p1[0]], [_p0[1],_p1[1]], [_p0[2],_p1[2]], color='#a0c4ff', linewidth=1.5)\n"
            )
        else:  # Sphere
            cx = self.geom_sphere_cx.value(); cy = self.geom_sphere_cy.value()
            cz = self.geom_sphere_cz.value(); r = self.geom_sphere_r.value()
            geom_code = f"dde.geometry.Sphere([{cx}, {cy}, {cz}], {r})"
            bbox3 = (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)
            outline_code = (
                "_u = np.linspace(0, 2*np.pi, 30)\n"
                "_v = np.linspace(0, np.pi, 20)\n"
                f"_sx = {cx} + {r}*np.outer(np.cos(_u), np.sin(_v))\n"
                f"_sy = {cy} + {r}*np.outer(np.sin(_u), np.sin(_v))\n"
                f"_sz = {cz} + {r}*np.outer(np.ones_like(_u), np.cos(_v))\n"
                "ax.plot_wireframe(_sx, _sy, _sz, color='#a0c4ff', linewidth=0.5, alpha=0.4, rstride=2, cstride=2)\n"
            )

        bx0, bx1, by0, by1, bz0, bz1 = bbox3
        pad_x = max((bx1 - bx0) * 0.08, 1e-6)
        pad_y = max((by1 - by0) * 0.08, 1e-6)
        pad_z = max((bz1 - bz0) * 0.08, 1e-6)
        xlim_lo, xlim_hi = bx0 - pad_x, bx1 + pad_x
        ylim_lo, ylim_hi = by0 - pad_y, by1 + pad_y
        zlim_lo, zlim_hi = bz0 - pad_z, bz1 + pad_z

        return f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

geom  = {geom_code}
t_dom = dde.geometry.TimeDomain({t_min}, {t_max})
gt    = dde.geometry.GeometryXTime(geom, t_dom)

def pde(x, y): return y[:, 0:1] * 0
data = dde.data.TimePDE(gt, pde, [],
    num_domain={n_domain}, num_boundary={n_boundary},
    num_initial={n_initial}, num_test=10,
    train_distribution="{dist}")

pts = data.train_points()
t_range = {t_max} - {t_min}
tol = max(t_range * 0.05, 1e-6)

plt.rcParams['figure.dpi'] = 120
fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#1e1e1e')
ax  = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

for _a in (ax, ax2):
    _a.set_facecolor('#252526')
    _a.tick_params(colors='#c0c0c0')
    for sp in _a.spines.values(): sp.set_color('#3e3e42')
_pane = (0.145, 0.145, 0.149, 1.0)
ax.xaxis.set_pane_color(_pane); ax.yaxis.set_pane_color(_pane); ax.zaxis.set_pane_color(_pane)

ax.set_xlim({xlim_lo}, {xlim_hi})
ax.set_ylim({ylim_lo}, {ylim_hi})
ax.set_zlim({zlim_lo}, {zlim_hi})

# Shape outline -- box edges for Cuboid, a wireframe sphere for Sphere.
{outline_code}
# Classify points using the geometry's own on_boundary() test -- same
# generalization as the 2D preview, just over (x,y,z) instead of (x,y).
ic_mask  = pts[:, 3] <= {t_min} + tol
bnd_mask = gt.geometry.on_boundary(pts[:, :3]) & ~ic_mask
dom_mask = ~ic_mask & ~bnd_mask
dom_pts = pts[dom_mask]
bnd_pts = pts[bnd_mask]
ic_pts  = pts[ic_mask]

# Spatial (3D) panel -- domain + boundary only, same reasoning as the 2D
# preview: IC points are already shown clearly in the time panel below.
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,1], dom_pts[:,2], s=6, c='#74c0fc', alpha=0.5, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,1], bnd_pts[:,2], s=16, c='#f03e3e', alpha=1.0, label=f'Boundary ({{len(bnd_pts)}})')
ax.set_xlabel('x', color='#e0e0e0', fontsize=10)
ax.set_ylabel('y', color='#e0e0e0', fontsize=10)
ax.set_zlabel('z', color='#e0e0e0', fontsize=10)
ax.set_title('{geom_type}: Spatial (x,y,z) | {dist} | D={n_domain} B={n_boundary} IC={n_initial}', color='#74c0fc', fontsize=10, fontweight='bold')
ax.legend(fontsize=9, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5, loc='upper left')

# Right: time distribution (x vs t) -- same layout as the 2D preview's
# right panel, just reading column 3 (t) instead of column 2.
ax2.set_xlim({xlim_lo}, {xlim_hi}); ax2.set_ylim({t_min}, {t_max})
if len(dom_pts): ax2.scatter(dom_pts[:,0], dom_pts[:,3], s=6, c='#74c0fc', alpha=0.5, label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax2.scatter(bnd_pts[:,0], bnd_pts[:,3], s=14, c='#f03e3e', alpha=0.9, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax2.scatter(ic_pts[:,0],  ic_pts[:,3],  s=14, c='#2f9e44', alpha=1.0, label=f'IC ({{len(ic_pts)}})')
ax2.set_xlabel('x', color='#e0e0e0', fontsize=11)
ax2.set_ylabel('t', color='#e0e0e0', fontsize=11)
ax2.set_title(f'Time Distribution (x vs t)\\ntotal={{len(pts)}} points', color='#74c0fc', fontsize=10, fontweight='bold')
ax2.legend(fontsize=10, facecolor='#2a2a2a', labelcolor='#e0e0e0', edgecolor='#555', markerscale=1.5)
plt.tight_layout()
plt.savefig('/tmp/domain_preview.png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
plt.close()
print("DOMAIN_PREVIEW_DONE")
"""

    def _on_dist_changed(self, text):
        if self.view_domain_check.isChecked():
            self._preview_domain()

    def _on_pts_changed(self, val):
        if self.view_domain_check.isChecked():
            self._preview_domain()
    
    def _on_preview_done(self, success):
        if success and os.path.exists('/tmp/domain_preview.png'):
            # Split the wide image across both panels
            from PyQt6.QtGui import QPixmap as _QPix
            full = _QPix('/tmp/domain_preview.png')
            w = full.width(); h = full.height()
            left_half  = full.copy(0,       0, w//2, h)
            right_half = full.copy(w//2, 0, w//2, h)
            self.loss_label.setPixmap(left_half.scaled(
                500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.solution_label.setPixmap(right_half.scaled(
                500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            self.loss_label.setText("❌ Preview failed")
    
    _DISP_FONT_CHOICES = [
        "Segoe UI", "Arial", "Helvetica", "Verdana", "Tahoma",
        "Trebuchet MS", "Georgia", "Times New Roman",
        "Courier New", "Consolas",
    ]
    _DISP_CATEGORY_LABELS = [
        ("title", "App Title / Subtitle"),
        ("section_header", 'Section Headers (e.g. "Problem Definition", "Quick Examples")'),
        ("field_label", "Field Labels (regular text next to inputs)"),
        ("hint", "Hints / Notes (small tips && warnings)"),
        ("button", "Buttons"),
        ("log_console", "Log Console"),
    ]

    def _on_display_settings(self):
        # Snapshot everything so Cancel can revert live-previewed changes,
        # not just close the dialog leaving a half-applied preview behind.
        snapshot = {
            "font_size": self._font_size,
            "log_font_size": self._log_font_size,
            "theme": self._theme,
            "accent": self._accent,
            "disp": {k: dict(v) for k, v in self._disp.items()},
        }

        dialog = QDialog(self)
        dialog.setWindowTitle("Display Settings")
        dialog.setMinimumWidth(620)
        outer = QVBoxLayout(dialog)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        # ── Tab 1: Theme & Base ───────────────────────────────
        theme_tab = QWidget()
        theme_layout = QVBoxLayout(theme_tab)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Base UI font size (spinboxes, combos, checkboxes):"))
        font_spin = QSpinBox(); font_spin.setRange(9, 22); font_spin.setValue(self._font_size)
        font_spin.setFixedWidth(80)
        font_row.addStretch(); font_row.addWidget(font_spin)
        theme_layout.addLayout(font_row)

        log_font_row = QHBoxLayout()
        log_font_row.addWidget(QLabel("Log console font size:"))
        log_font_spin = QSpinBox(); log_font_spin.setRange(8, 24); log_font_spin.setValue(self._log_font_size)
        log_font_spin.setFixedWidth(80)
        log_font_row.addStretch(); log_font_row.addWidget(log_font_spin)
        theme_layout.addLayout(log_font_row)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Color theme:"))
        theme_combo = QComboBox()
        theme_combo.addItems(["Dark Grey", "GitHub Dark", "Monokai", "Solarized Dark", "Navy Blue", "White"])
        theme_combo.setCurrentText(self._theme)
        theme_combo.setFixedWidth(160)
        theme_row.addStretch(); theme_row.addWidget(theme_combo)
        theme_layout.addLayout(theme_row)

        accent_row = QHBoxLayout()
        accent_row.addWidget(QLabel("Accent color (default Section Header color):"))
        accent_combo = QComboBox()
        accent_combo.addItems(["Blue (#a0c4ff)", "Green (#69db7c)", "Orange (#ffa94d)", "Purple (#cc5de8)", "Teal (#38d9a9)", "Black (#000000)"])
        accent_combo.setCurrentText(self._accent)
        accent_combo.setFixedWidth(160)
        accent_row.addStretch(); accent_row.addWidget(accent_combo)
        theme_layout.addLayout(accent_row)
        theme_layout.addStretch()
        tabs.addTab(theme_tab, "Theme")

        # ── Tab 2: Text Styles (per-category) ─────────────────
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        intro = QLabel(
            "Set family/size/bold/color independently for each kind of text. "
            "Color left on \"Auto\" keeps that text's own default color."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #74c0fc; font-size: 12px;")
        text_layout.addWidget(intro)

        cat_widgets = {}  # category -> dict of controls, for reading back on preview/OK

        for cat_key, cat_label in self._DISP_CATEGORY_LABELS:
            cur = self._disp.get(cat_key, self._DISP_CATEGORY_DEFAULTS[cat_key])
            group = QGroupBox(cat_label)
            row = QHBoxLayout(group)

            family_combo = QComboBox()
            family_combo.addItems(self._DISP_FONT_CHOICES)
            if cur["family"] not in self._DISP_FONT_CHOICES:
                family_combo.addItem(cur["family"])
            family_combo.setCurrentText(cur["family"])
            family_combo.setFixedWidth(140)
            row.addWidget(family_combo)

            size_spin = QSpinBox()
            size_spin.setRange(7, 40)
            size_spin.setValue(cur["size"])
            size_spin.setFixedWidth(60)
            size_spin.setSuffix(" px")
            row.addWidget(size_spin)

            bold_cb = QCheckBox("Bold")
            bold_cb.setChecked(bool(cur["bold"]))
            row.addWidget(bold_cb)

            color_btn = QPushButton()
            color_btn.setFixedWidth(90)
            color_btn.setToolTip("Click to pick a color for this category")
            color_state = {"color": (cur.get("color") or "").strip()}

            def _refresh_color_btn(btn=color_btn, state=color_state):
                c = state["color"]
                if c:
                    btn.setText(c)
                    btn.setStyleSheet(f"background: {c}; color: {'#000' if QColor(c).lightnessF() > 0.5 else '#fff'};")
                else:
                    btn.setText("Color: Auto")
                    btn.setStyleSheet("")
            _refresh_color_btn()

            def _pick_color(_checked=False, state=color_state, btn=color_btn):
                start = QColor(state["color"]) if state["color"] else QColor("#a0c4ff")
                picked = QColorDialog.getColor(start, dialog, "Choose color")
                if picked.isValid():
                    state["color"] = picked.name()
                    _refresh_color_btn(btn, state)
                    _apply_preview()
            color_btn.clicked.connect(_pick_color)
            row.addWidget(color_btn)

            reset_btn = QPushButton("✕ Reset")
            reset_btn.setFixedWidth(60)
            reset_btn.setToolTip("Clear the color override, back to this text's own default color")

            def _reset_color(_checked=False, state=color_state, btn=color_btn):
                state["color"] = ""
                _refresh_color_btn(btn, state)
                _apply_preview()
            reset_btn.clicked.connect(_reset_color)
            row.addWidget(reset_btn)

            text_layout.addWidget(group)
            cat_widgets[cat_key] = {
                "family": family_combo, "size": size_spin, "bold": bold_cb, "color_state": color_state,
            }

        text_layout.addStretch()
        tabs.addTab(text_tab, "Text Styles")

        def _apply_preview():
            self._font_size = font_spin.value()
            self._log_font_size = log_font_spin.value()
            self._theme = theme_combo.currentText()
            self._accent = accent_combo.currentText()
            for cat_key, widgets in cat_widgets.items():
                self._disp[cat_key] = {
                    "family": widgets["family"].currentText(),
                    "size": widgets["size"].value(),
                    "bold": widgets["bold"].isChecked(),
                    "color": widgets["color_state"]["color"],
                }
            self._apply_display_settings()

        # Live preview: every control applies immediately, so changes are
        # visible right away while comparing options -- no separate
        # "Preview" click needed.
        font_spin.valueChanged.connect(_apply_preview)
        log_font_spin.valueChanged.connect(_apply_preview)
        theme_combo.currentTextChanged.connect(_apply_preview)
        accent_combo.currentTextChanged.connect(_apply_preview)
        for widgets in cat_widgets.values():
            widgets["family"].currentTextChanged.connect(_apply_preview)
            widgets["size"].valueChanged.connect(_apply_preview)
            widgets["bold"].stateChanged.connect(_apply_preview)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK (save)"); cancel_btn = QPushButton("Cancel (revert)")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        outer.addLayout(btn_row)

        def _revert_settings():
            # Runs on Cancel AND on closing the dialog via the window's own
            # close button, so a live-previewed-but-not-saved change never
            # silently sticks either way.
            self._font_size = snapshot["font_size"]
            self._log_font_size = snapshot["log_font_size"]
            self._theme = snapshot["theme"]
            self._accent = snapshot["accent"]
            self._disp = snapshot["disp"]
            self._apply_display_settings()

        def _on_ok():
            _apply_preview()
            self._save_display_settings_to_disk()
            dialog.accept()

        cancel_btn.clicked.connect(dialog.reject)
        dialog.rejected.connect(_revert_settings)
        ok_btn.clicked.connect(_on_ok)
        dialog.exec()


    def _apply_display_settings(self):
        themes = {
            "Dark Grey":     ("#1e1e1e", "#252526", "#3e3e42"),
            "GitHub Dark":   ("#0d1117", "#161b22", "#30363d"),
            "Monokai":       ("#272822", "#1e1f1c", "#49483e"),
            "Solarized Dark":("#002b36", "#073642", "#586e75"),
            "Navy Blue":     ("#1a1a2e", "#16213e", "#3a3a5c"),
            "White":         ("#ffffff", "#f5f5f5", "#d0d0d0"),
        }
        accents = {
            "Blue (#a0c4ff)":   "#a0c4ff",
            "Green (#69db7c)":  "#69db7c",
            "Orange (#ffa94d)": "#ffa94d",
            "Purple (#cc5de8)": "#cc5de8",
            "Teal (#38d9a9)":   "#38d9a9",
            "Black (#000000)":  "#000000",
        }
        bg, widget_bg, border = themes.get(self._theme, themes["Dark Grey"])
        accent = accents.get(self._accent, "#a0c4ff")
        fs = self._font_size
        lfs = self._log_font_size
        is_white = self._theme == "White"
        text_color = "#1e1e1e" if is_white else "#e0e0e0"
        label_color = "#333333" if is_white else "#c0c0c0"
        arrow_color = "#333333" if is_white else "#ffffff"
        # Panel (QGroupBox) borders specifically, brightened relative to
        # the theme's general-purpose {border} color (also used for input
        # fields/scrollbars/menus, which should keep their own subtler
        # value) so separate panels read clearly against a dark
        # background. The White theme's own border is already light
        # against its white background, so it's left as-is.
        panel_border = border if is_white else "#c8d2d8"

        # Section Headers (QGroupBox titles, e.g. "Problem Definition",
        # "Quick Examples") and Field Labels (plain QLabel text) each get
        # their own independently-adjustable family/size/weight, on top of
        # the base widget font-size above -- these are two of the Display
        # Settings categories. An unset category color falls back to the
        # theme's existing accent/label color exactly as before.
        sh = self._disp.get("section_header", self._DISP_CATEGORY_DEFAULTS["section_header"])
        sh_color = (sh.get("color") or "").strip() or accent
        sh_weight = "bold" if sh.get("bold") else "normal"
        fl = self._disp.get("field_label", self._DISP_CATEGORY_DEFAULTS["field_label"])
        fl_color = (fl.get("color") or "").strip() or label_color
        fl_weight = "bold" if fl.get("bold") else "normal"
        lc = self._disp.get("log_console", self._DISP_CATEGORY_DEFAULTS["log_console"])
        log_color = (lc.get("color") or "").strip() or ('#1e1e1e' if is_white else '#a0ffb0')

        self.setStyleSheet(f"""
            QMainWindow {{ background: {bg}; }}
            QWidget {{ background: {bg}; color: {text_color}; font-family: 'Segoe UI', Arial; font-size: {fs}px; }}
            QGroupBox {{
                border: 1px solid {panel_border};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: {accent};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
                font-family: '{sh['family']}'; font-size: {sh['size']}px;
                font-weight: {sh_weight}; color: {sh_color};
            }}
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                background: {widget_bg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px 6px;
                color: {text_color};
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
                border: 1px solid {accent};
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background: {border};
                border: none;
                width: 16px;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QComboBox::drop-down {{ background: {border}; border: none; width: 20px; }}
            QComboBox::down-arrow {{
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid {arrow_color};
                width: 0px; height: 0px;
            }}
            QCheckBox {{ color: {label_color}; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {border};
                border-radius: 3px;
                background: {widget_bg};
            }}
            QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
            QRadioButton {{ color: {label_color}; spacing: 6px; }}
            QRadioButton::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {border};
                border-radius: 7px;
                background: {widget_bg};
            }}
            QRadioButton::indicator:checked {{ background: {accent}; border-color: {accent}; }}
            QLabel {{
                color: {fl_color};
                font-family: '{fl['family']}'; font-size: {fl['size']}px; font-weight: {fl_weight};
            }}
            QScrollArea {{ border: none; background: {bg}; }}
            QScrollBar:vertical {{ background: {widget_bg}; width: 14px; border-radius: 6px; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 6px; min-height: 30px; }}
            QTextEdit {{
                background: {'#f8f8f8' if is_white else '#0f0f23'};
                border: 1px solid {border};
                border-radius: 6px;
                color: {log_color};
                font-family: '{lc['family']}';
                font-size: {lfs}px;
            }}
            QSplitter::handle {{ background: {border}; width: 2px; }}
            QMenuBar {{ background: {widget_bg}; color: {label_color}; border-bottom: 1px solid {border}; }}
            QMenuBar::item:selected {{ background: {border}; }}
            QMenu {{ background: {widget_bg}; border: 1px solid {border}; }}
            QMenu::item:selected {{ background: {border}; }}
        """)
        self.loss_label.setStyleSheet(f"border: 1px solid {border}; border-radius: 6px; color: #505080; background: {widget_bg};")
        self.solution_label.setStyleSheet(f"border: 1px solid {border}; border-radius: 6px; color: #505080; background: {widget_bg};")
        # Re-apply every registered per-category widget (App Title,
        # Hint/Note labels, Buttons) so they pick up the latest settings too
        # -- the QSS block above only covers Section Headers/Field Labels/
        # Log Console, which don't need per-widget registration.
        if hasattr(self, "_styled_widgets"):
            self._restyle_all()

    def _on_export_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Solution Data")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        info = QLabel("Saves solution as CSV files (one per time step)\nfor forward problems after training.")
        self._register_style(info, "hint", lambda css, _c='#74c0fc', _e='': f"color: {_c}; {_e}{css}")
        info.setWordWrap(True)
        layout.addWidget(info)

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid size (points per axis):"))
        grid_combo = QComboBox()
        grid_combo.addItems(["101", "51", "21", "11"])
        grid_combo.setCurrentText(self.export_grid_combo.currentText())
        grid_combo.setFixedWidth(80)
        grid_row.addStretch(); grid_row.addWidget(grid_combo)
        layout.addLayout(grid_row)

        tsteps_row = QHBoxLayout()
        tsteps_row.addWidget(QLabel("Number of time snapshots:"))
        tsteps_spin = QSpinBox()
        tsteps_spin.setRange(2, 50); tsteps_spin.setValue(self.export_tsteps_spin.value())
        tsteps_spin.setFixedWidth(80)
        tsteps_row.addStretch(); tsteps_row.addWidget(tsteps_spin)
        layout.addLayout(tsteps_row)

        note = QLabel("Output: solution_data/solution_t{time}.txt\nColumns: x, t, u (1D) or x, y, t, u (2D)")
        self._register_style(note, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        note.setWordWrap(True)
        layout.addWidget(note)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self.export_grid_combo.setCurrentText(grid_combo.currentText())
            self.export_tsteps_spin.setValue(tsteps_spin.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _execute_error_analysis(self, ea, config):
        import tempfile, glob, json
        save_dir = self.save_dir_input.text().strip()
        is_2d = config.problem_dim == "2D"

        # Find most recently modified model
        model_path = ""
        all_models = glob.glob(os.path.join(save_dir, "model_lbfgs-*.pt")) + \
                     glob.glob(os.path.join(save_dir, "model_adam-*.pt"))
        if all_models:
            model_path = max(all_models, key=os.path.getmtime)
            self.log_box.append(f"📂 Using model: {os.path.basename(model_path)}")

        if not model_path:
            self.log_box.append("❌ No saved model found — set save directory before training."); return

        config_path = model_path.replace(".pt", ".json")
        if not os.path.exists(config_path):
            config_path = os.path.join(save_dir, "model_config.json")

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read model config: {e}"); return

        layers = cfg["layers"]; activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        loss_type = cfg.get("loss_type", "MSE")
        compile_opt = "lbfgs" if "lbfgs" in model_path else "adam"

        use_expr = ea['use_expr']
        expr_raw = ea['expr']
        csv_path = ea['csv_path']
        times_str = ea['times']
        snap_time = ea['snap_time']

        from pinnstudio.core.codegen import _simplify_expr
        expr_converted = _simplify_expr(expr_raw, is_2d) if use_expr else ""

        try:
            times = [float(t.strip()) for t in times_str.split(",") if t.strip()]
        except Exception:
            times = [0.5]

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if {str(is_2d)}:
    geom = dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])
else:
    geom = dde.geometry.Interval({x_min}, {x_max})
td = dde.geometry.TimeDomain({t_min}, {t_max})
gt = dde.geometry.GeometryXTime(geom, td)
def pde(x, y): return y[:, 0:1] * 0
data  = dde.data.TimePDE(gt, pde, [], num_domain=100, num_test=100)
net   = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)
if "{compile_opt}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}")
else:
    model.compile("adam", lr=0.001, loss="{loss_type}")
model.restore(r"{model_path}", verbose=0)
print("Model restored for error analysis.")
os.makedirs(r"{save_dir}", exist_ok=True)
metrics_lines = []
"""
        if not is_2d:
            script += f"""
x_vals = np.linspace({x_min}, {x_max}, 200)
times  = {times}
n_t = len(times)
fig, axes = plt.subplots(n_t, 3, figsize=(15, 4*n_t))
if n_t == 1: axes = [axes]
fig.suptitle("PINN vs Reference — Error Analysis", fontsize=13, fontweight='bold')
for row, tv in enumerate(times):
    xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    u_pred = model.predict(xt)[:, 0].flatten()
"""
            if use_expr:
                script += f"""
    x = x_vals; t = tv
    u_ref = {expr_converted}
    u_ref = np.atleast_1d(u_ref)
    if len(u_ref) != len(x_vals):
        u_ref = np.full_like(x_vals, float(u_ref[0]))
"""
            else:
                script += f"""
    data_csv = np.loadtxt(r"{csv_path}", delimiter=",", skiprows=1)
    t_col = data_csv[:, 1]
    t_unique = np.unique(t_col)
    t_closest = t_unique[np.argmin(np.abs(t_unique - tv))]
    mask = np.abs(t_col - t_closest) < 1e-10
    u_ref = np.interp(x_vals, data_csv[mask, 0], data_csv[mask, 2])
"""
            script += f"""
    abs_err = np.abs(u_pred - u_ref)
    l2   = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10)
    mse  = np.mean((u_pred - u_ref)**2)
    maxe = np.max(abs_err)
    metrics_lines.append(f"t={{tv:.4f}}: L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
    print(f"  t={{tv:.3f}} — L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
    ax0, ax1, ax2 = axes[row]
    ax0.plot(x_vals, u_pred, color='#4dabf7', lw=2, label='PINN')
    ax0.plot(x_vals, u_ref,  color='#ff8787', lw=2, ls='--', label='Reference')
    ax0.set_title(f"t={{tv:.3f}} — PINN vs Reference"); ax0.legend(); ax0.grid(True, alpha=0.3)
    ax1.plot(x_vals, u_ref, color='#ff8787', lw=2)
    ax1.set_title(f"t={{tv:.3f}} — Reference"); ax1.grid(True, alpha=0.3)
    ax2.plot(x_vals, abs_err, color='#69db7c', lw=2)
    ax2.fill_between(x_vals, abs_err, alpha=0.3, color='#69db7c')
    ax2.set_title(f"t={{tv:.3f}} — |Error| L2={{l2:.2e}}"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
"""
        else:
            script += f"""
tv = {snap_time}
x_vals = np.linspace({x_min}, {x_max}, 80)
y_vals = np.linspace({y_min}, {y_max}, 80)
Xg, Yg = np.meshgrid(x_vals, y_vals)
XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
u_pred = model.predict(XYT)[:, 0].reshape(80, 80)
"""
            if use_expr:
                script += f"""
x = Xg; y = Yg; t = tv
u_ref = {expr_converted}
if not hasattr(u_ref, 'shape') or u_ref.shape != (80, 80):
    u_ref = np.full((80, 80), float(u_ref))
"""
            else:
                script += f"""
data_csv = np.loadtxt(r"{csv_path}", delimiter=",", skiprows=1)
t_col = data_csv[:, 2]
t_unique = np.unique(t_col)
t_closest = t_unique[np.argmin(np.abs(t_unique - tv))]
mask = np.abs(t_col - t_closest) < 1e-10
from scipy.interpolate import griddata
u_ref = griddata(data_csv[mask, :2], data_csv[mask, 3], (Xg, Yg), method='linear', fill_value=0.0)
"""
            script += f"""
abs_err = np.abs(u_pred - u_ref)
l2   = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-10)
mse  = np.mean((u_pred - u_ref)**2)
maxe = np.max(abs_err)
metrics_lines.append(f"t={snap_time:.4f}: L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
print(f"  t={snap_time:.3f} — L2={{l2:.4e}}, MSE={{mse:.4e}}, Max={{maxe:.4e}}")
vmin = min(u_pred.min(), u_ref.min()); vmax = max(u_pred.max(), u_ref.max())
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].contourf(Xg, Yg, u_pred, levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[0].set_title(f"PINN at t={snap_time}"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].contourf(Xg, Yg, u_ref, levels=40, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].set_title(f"Reference at t={snap_time}"); axes[1].set_xlabel("x")
fig.colorbar(im1, ax=axes[1])
im2 = axes[2].contourf(Xg, Yg, abs_err, levels=40, cmap='YlOrRd')
axes[2].set_title(f"|Error| L2={{l2:.2e}}"); axes[2].set_xlabel("x")
fig.colorbar(im2, ax=axes[2])
fig.suptitle("PINN vs Reference — Error Analysis", fontsize=13, fontweight='bold')
plt.tight_layout()
"""
        script += f"""
out_path = os.path.join(r"{save_dir}", "comparison_plot.png")
plt.savefig(out_path, dpi=100); plt.close()
print(f"Comparison plot saved: {{out_path}}")
metrics_path = os.path.join(r"{save_dir}", "error_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Error Analysis Results\\n" + "="*40 + "\\n")
    for line in metrics_lines:
        f.write(line + "\\n")
print(f"Metrics saved: {{metrics_path}}")
print("ERROR_ANALYSIS_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script); tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig
        class _EAThread(QThread):
            line_sig = _sig(str)
            done_sig = _sig(bool)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.line_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._ea_thread = _EAThread(tmp)
        self._ea_thread.line_sig.connect(self.log_box.append)
        self._ea_thread.done_sig.connect(self._on_ea_done)
        self._ea_thread.start()

    def _on_ea_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete!")
            plot_path = os.path.join(save_dir, "comparison_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        else:
            self.log_box.append("❌ Error analysis failed — check log.")
    
    def _on_restore_mode_changed(self, text):
        is_inverse = (text == "Inverse Model")
        self.restore_inv_vars_widget.setVisible(is_inverse)
        current_viz = self.restore_viz_combo.currentText()
        items = list(self._RESTORE_FORWARD_VIZ) + (list(self._RESTORE_PARAM_VIZ) if is_inverse else [])
        self.restore_viz_combo.blockSignals(True)
        self.restore_viz_combo.clear()
        self.restore_viz_combo.addItems(items)
        if current_viz in items:
            self.restore_viz_combo.setCurrentText(current_viz)
        self.restore_viz_combo.blockSignals(False)
        self._on_restore_viz_changed(self.restore_viz_combo.currentText())

    def _on_restore_viz_changed(self, text):
        # Parameter Convergence Plot/Animation read text files, not the
        # model -- hide the model/config/optimizer/output fields (not
        # needed) and show the parameter-file row list instead. Every
        # other viz type keeps the panel exactly as it always was.
        is_param = text in getattr(self, '_RESTORE_PARAM_VIZ', [])
        self.restore_model_fields_widget.setVisible(not is_param)
        self.restore_optimizer_widget.setVisible(not is_param)
        self.restore_output_widget.setVisible(not is_param)
        self.restore_param_widget.setVisible(is_param)
        self._on_restore_viz_settings(text)

    def _on_restore_viz_settings(self, viz_type=None):
        if viz_type is None:
            viz_type = self.restore_viz_combo.currentText()

        is_param = viz_type in getattr(self, '_RESTORE_PARAM_VIZ', [])

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._restore_viz_settings

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"])
        cmap_combo.setCurrentText(current.get('colormap', 'RdBu_r'))
        cmap_combo.setFixedWidth(120)
        cmap_row.addStretch(); cmap_row.addWidget(cmap_combo)
        if not is_param:
            layout.addLayout(cmap_row)

        # Contour levels
        levels_row = QHBoxLayout()
        levels_row.addWidget(QLabel("Contour levels:"))
        levels_spin = QSpinBox()
        levels_spin.setRange(5, 200); levels_spin.setValue(current.get('levels', 40))
        levels_spin.setFixedWidth(80)
        levels_row.addStretch(); levels_row.addWidget(levels_spin)
        if not is_param:
            layout.addLayout(levels_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Grid resolution:"))
        res_combo = QComboBox()
        res_combo.addItems(["80", "100", "150", "200"])
        res_combo.setCurrentText(str(current.get('resolution', 100)))
        res_combo.setFixedWidth(80)
        res_row.addStretch(); res_row.addWidget(res_combo)
        if not is_param:
            layout.addLayout(res_row)

        # DPI
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Output DPI:"))
        dpi_combo = QComboBox()
        dpi_combo.addItems(["100", "150", "200", "300"])
        dpi_combo.setCurrentText(str(current.get('dpi', 100)))
        dpi_combo.setFixedWidth(80)
        dpi_row.addStretch(); dpi_row.addWidget(dpi_combo)
        if not is_param:
            layout.addLayout(dpi_row)

        # Surface time — only for Surface
        surface_time_widget = QWidget()
        st_layout = QHBoxLayout(surface_time_widget)
        st_layout.setContentsMargins(0, 0, 0, 0)
        st_layout.addWidget(QLabel("Plot at time t ="))
        surface_time_spin = QDoubleSpinBox()
        surface_time_spin.setRange(0.0, 1e6)
        surface_time_spin.setValue(current.get('surface_time', self.t_max.value()))
        surface_time_spin.setFixedWidth(100)
        st_layout.addStretch(); st_layout.addWidget(surface_time_spin)
        surface_time_widget.setVisible(viz_type == "Surface")
        if not is_param:
            layout.addWidget(surface_time_widget)

        # Color range — for Surface and Animation Surface
        color_range_widget = QWidget()
        cr_layout = QVBoxLayout(color_range_widget)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_auto_cb = QCheckBox("Auto color range")
        cr_auto_cb.setChecked(current.get('auto_range', True))
        cr_layout.addWidget(cr_auto_cb)
        cr_manual_widget = QWidget()
        cr_manual_layout = QHBoxLayout(cr_manual_widget)
        cr_manual_layout.setContentsMargins(0, 0, 0, 0)
        cr_manual_layout.addWidget(QLabel("vmin:"))
        vmin_spin = QDoubleSpinBox(); vmin_spin.setRange(-1e6, 1e6); vmin_spin.setValue(current.get('vmin', -1.0)); vmin_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmin_spin)
        cr_manual_layout.addWidget(QLabel("vmax:"))
        vmax_spin = QDoubleSpinBox(); vmax_spin.setRange(-1e6, 1e6); vmax_spin.setValue(current.get('vmax', 1.0)); vmax_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmax_spin)
        cr_manual_widget.setVisible(not cr_auto_cb.isChecked())
        cr_layout.addWidget(cr_manual_widget)
        cr_auto_cb.stateChanged.connect(lambda s: cr_manual_widget.setVisible(s != 2))
        color_range_widget.setVisible("Surface" in viz_type)
        if not is_param:
            layout.addWidget(color_range_widget)

        # Steps/frames
        steps_widget = QWidget()
        steps_layout = QHBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        label_text = "Time steps:" if "Line" in viz_type else "Animation frames:"
        steps_layout.addWidget(QLabel(label_text))
        steps_spin = QSpinBox()
        steps_spin.setRange(2, 100); steps_spin.setValue(current.get('n_steps', 10))
        steps_spin.setFixedWidth(80)
        steps_layout.addStretch(); steps_layout.addWidget(steps_spin)
        steps_widget.setVisible(viz_type != "Surface")
        if not is_param:
            layout.addWidget(steps_widget)

        # Line width — only for Line plots
        lw_widget = QWidget()
        lw_layout = QHBoxLayout(lw_widget)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width:"))
        lw_combo = QComboBox()
        lw_combo.addItems(["1.0", "1.5", "2.0", "2.5", "3.0"])
        lw_combo.setCurrentText(str(current.get('linewidth', 2.0)))
        lw_combo.setFixedWidth(80)
        lw_layout.addStretch(); lw_layout.addWidget(lw_combo)
        lw_widget.setVisible("Line" in viz_type)
        if not is_param:
            layout.addWidget(lw_widget)

        # FPS — only for animations
        fps_widget = QWidget()
        fps_layout = QHBoxLayout(fps_widget)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        fps_layout.addWidget(QLabel("Animation FPS:"))
        fps_combo = QComboBox()
        fps_combo.addItems(["5", "8", "10", "15", "20"])
        fps_combo.setCurrentText(str(current.get('fps', 10)))
        fps_combo.setFixedWidth(80)
        fps_layout.addStretch(); fps_layout.addWidget(fps_combo)
        fps_widget.setVisible("Animation" in viz_type)
        if not is_param:
            layout.addWidget(fps_widget)

        # Colorbar
        colorbar_cb = QCheckBox("Show colorbar")
        colorbar_cb.setChecked(current.get('colorbar', True))
        colorbar_cb.setVisible("Surface" in viz_type)
        if not is_param:
            layout.addWidget(colorbar_cb)

        # Title / axis labels — every viz type gets these; blank keeps the
        # existing default text exactly as before.
        labels_line = QLabel("Leave blank to keep the default title/axis labels.")
        self._register_style(labels_line, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        labels_line.setWordWrap(True)
        layout.addWidget(labels_line)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        title_edit = QLineEdit()
        title_edit.setText(current.get('title', ''))
        title_edit.setPlaceholderText("(default)")
        title_row.addWidget(title_edit)
        layout.addLayout(title_row)

        xlabel_row = QHBoxLayout()
        xlabel_row.addWidget(QLabel("X-axis label:"))
        xlabel_edit = QLineEdit()
        xlabel_edit.setText(current.get('xlabel', ''))
        xlabel_edit.setPlaceholderText("(default)")
        xlabel_row.addWidget(xlabel_edit)
        layout.addLayout(xlabel_row)

        ylabel_row = QHBoxLayout()
        ylabel_row.addWidget(QLabel("Y-axis label:"))
        ylabel_edit = QLineEdit()
        ylabel_edit.setText(current.get('ylabel', ''))
        ylabel_edit.setPlaceholderText("(default)")
        ylabel_row.addWidget(ylabel_edit)
        layout.addLayout(ylabel_row)

        info_texts = {
            "Surface": "Single heatmap/contour at specified time.",
            "Line (time steps)": "Solution lines at evenly spaced time steps.",
            "Animation Line (GIF)": "Animated GIF of line plots over time.",
            "Animation Surface (GIF)": "Animated GIF of surface plots with colorbar.",
            "Parameter Convergence Plot (PNG)": "Static plot of each trainable variable's value vs. iteration.",
            "Parameter Convergence Animation (GIF)": "Animated GIF of each trainable variable's convergence, growing curve up to each logged iteration.",
        }
        info = QLabel(info_texts.get(viz_type, ""))
        self._register_style(info, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            new_settings = dict(current)
            if not is_param:
                new_settings.update({
                    'colormap': cmap_combo.currentText(),
                    'surface_time': surface_time_spin.value(),
                    'n_steps': steps_spin.value(),
                    'colorbar': colorbar_cb.isChecked(),
                    'levels': levels_spin.value(),
                    'resolution': int(res_combo.currentText()),
                    'dpi': int(dpi_combo.currentText()),
                    'auto_range': cr_auto_cb.isChecked(),
                    'vmin': vmin_spin.value(),
                    'vmax': vmax_spin.value(),
                    'linewidth': float(lw_combo.currentText()),
                    'fps': int(fps_combo.currentText()),
                })
                self.restore_tsteps_spin.setValue(steps_spin.value())
            new_settings['title'] = title_edit.text().strip()
            new_settings['xlabel'] = xlabel_edit.text().strip()
            new_settings['ylabel'] = ylabel_edit.text().strip()
            self._restore_viz_settings = new_settings
            self.log_box.append(f"✅ Viz settings saved — {viz_type}")
            dialog.accept()
        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _on_template_selected(self, text):
        if text == "📋 Examples":
            return

        templates_2d = {
            "2D Heat": {
                'pde': ["du_t - 0.4*(du_xx + du_yy)"],
                'ic': ["0.0"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 400,
                'layers': 4,
                'neurons': 64,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'periodic_bc': False,
                'bc_config': 'heat2d',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "heat"),
            },
            "2D Allen-Cahn (Mattey & Ghosh)": {
                'pde': ["du_t - 0.0001*(du_xx + du_yy) + (u**3 - u)"],
                'ic': ["sin(4*pi*x)*cos(4*pi*y)"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'periodic_bc': True,
                'bc_config': 'periodic_all',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "allen_cahn_mattey"),
                'ta_default': {'step_groups': [(0.0, 1.0, 4)], 'transfer_learning': True, 'ic_grid': 51, 'transfer_optimizer': 'lbfgs'},
            },
            "2D Allen-Cahn (Wight & Zhao)": {
                'pde': ["du_t - 0.00625*(du_xx + du_yy) + 10*(u**3 - u)"],
                'ic': ["tanh((0.35 - sqrt((x-0.5)**2 + (y-0.5)**2)) / (2*0.025))"],
                'num_domain': 10000,
                'num_boundary': 400,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                't_max': 10.0,
                'periodic_bc': True,
                'bc_config': 'periodic_all',
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "allen_cahn_wight"),
                'ta_default': {'step_groups': [(0.0, 10.0, 10)], 'transfer_learning': True, 'ic_grid': 51, 'transfer_optimizer': 'lbfgs'},
                'inverse_t_max': 2.5,
            },
        }
        if text in templates_2d:
            t = templates_2d[text]
            # Set number of outputs first
            n_out = t.get('num_outputs', 1)
            if n_out != self.num_outputs_spin.value():
                self.num_outputs_spin.setValue(n_out)
            # Set output names
            for i, name in enumerate(t.get('output_names', ['u'])):
                if i < len(self.output_name_inputs):
                    self.output_name_inputs[i].setText(name)
            # Set PDEs
            for i, pde_text in enumerate(t['pde']):
                if i < len(self.pde_inputs):
                    self.pde_inputs[i].setText(pde_text)
            # Set ICs
            for i, ic_text in enumerate(t['ic']):
                if i < len(self.ic_inputs):
                    self.ic_inputs[i].setText(ic_text)
            # Set domain
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            if 't_max' in t: self.t_max.setValue(t['t_max'])
            self._current_template_forward_tmax = t.get('t_max')
            self._current_template_inverse_tmax = t.get('inverse_t_max')
            if self._current_template_inverse_tmax is not None and self.radio_inverse.isChecked():
                self.t_max.setValue(self._current_template_inverse_tmax)
            # Set collocation points
            self.num_domain.setValue(t['num_domain'])
            self.num_boundary.setValue(t['num_boundary'])
            self.num_initial.setValue(t['num_initial'])
            if 'num_test' in t:
                self.num_test.setValue(t['num_test'])
            # Set network
            self.layers_spin.setValue(t['layers'])
            self.neurons_spin.setValue(t['neurons'])
            # Set training
            self.iter1_spin.setValue(t['iterations'])
            self.opt2_combo.setCurrentText(t['optimizer2'])
            self.iter2_spin.setValue(t['iterations2'])
            # Set IC weight
            for i in range(n_out):
                key = f"ic_{i}"
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(t.get('ic_weight', 100.0))
            # Set BCs
            bc_config = t.get('bc_config', 'periodic_all')
            if bc_config == 'periodic_all':
                # All boundaries periodic for all outputs
                for i in range(n_out):
                    if i < len(self.bc_left_types):
                        self.bc_left_types[i].setCurrentText("Periodic")
                    if i < len(self.bc_bottom_types):
                        self.bc_bottom_types[i].setCurrentText("Periodic")
            elif bc_config == 'heat2d':
                # Right: Dirichlet=1, Left/Bottom/Top: Neumann=0
                for i in range(n_out):
                    if i < len(self.bc_left_types):
                        self.bc_left_types[i].setCurrentText("Neumann")
                        self.bc_left_vals[i].setValue(0.0)
                    if i < len(self.bc_right_types):
                        self.bc_right_types[i].setCurrentText("Dirichlet")
                        self.bc_right_vals[i].setValue(1.0)
                    if i < len(self.bc_bottom_types):
                        self.bc_bottom_types[i].setCurrentText("Neumann")
                        self.bc_bottom_vals[i].setValue(0.0)
                    if i < len(self.bc_top_types):
                        self.bc_top_types[i].setCurrentText("Neumann")
                        self.bc_top_vals[i].setValue(0.0)
            self._populate_locked_bc_entries_from_legacy(n_out, is_2d=True)
            self._update_bc_mode_visibility()
            for row in list(self.ta_group_rows):
                row['widget'].deleteLater()
            self.ta_group_rows.clear()
            ta_cfg = t.get('ta_default')
            self._current_ta_cfg = ta_cfg
            self._ta_suspended_for_inverse = False
            if ta_cfg and not self.radio_inverse.isChecked():
                self.adapt_combo.setCurrentText("Time Adaptive")
                for g_start, g_end, g_steps in ta_cfg['step_groups']:
                    self._add_ta_step_group(g_start, g_end, g_steps)
                self.ta_transfer_cb.setChecked(ta_cfg.get('transfer_learning', False))
                self.ta_grid.setCurrentText(str(ta_cfg.get('ic_grid', 101)))
                self._set_combo_data(self.ta_transfer_opt, ta_cfg.get('transfer_optimizer', 'adam'))
            else:
                self.adapt_combo.setCurrentText("None")
                self._add_ta_step_group(0.0, 1.0, 10)
                self.ta_transfer_cb.setChecked(False)
                self.ta_grid.setCurrentText("101")
                self._set_combo_data(self.ta_transfer_opt, "adam")
                if ta_cfg and self.radio_inverse.isChecked():
                    self._ta_suspended_for_inverse = True
            self._template_ref_dir = t.get('ref_dir', '')
            self._current_template = text
            self._current_template_type = t.get('template_type', '')
            self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
            # Setup default scheduler phases
            if hasattr(self, 'sched_cb'):
                self.sched_cb.setChecked(True)
                self._setup_default_scheduler_phases(t.get('template_type', ''), t['iterations'], t.get('iterations2', 10000))
            if bc_config == 'heat2d':
                # The right-edge Dirichlet BC trains more accurately with a
                # heavier weight than the default 1 -- 100 matches the other
                # loss terms' scale here and was confirmed empirically. Set
                # after _setup_default_scheduler_phases(), since that call
                # rebuilds the weight_widgets (and would otherwise wipe this
                # back to the default of 1.0).
                for _bj, _be in enumerate(self.custom_bc_list):
                    if _be['type'].currentData() == 'dirichlet':
                        _bk = f"bc_{_bj}"
                        if _bk in self.weight_widgets:
                            self.weight_widgets[_bk].setValue(100.0)
            self._auto_configure_ea(self._template_ref_dir)
            self.log_box.append(f"✅ Template loaded: {text}")
            return

        templates_3d = {
            "3D Heat": {
                # Same domain and physics as 2D Heat, extended with a z
                # axis -- diffusion in a unit cube, insulated on 5 faces,
                # held at u=1 on the x=x_max face.
                'pde': ["du_t - 0.4*(du_xx + du_yy + du_zz)"],
                'ic': ["0.0"],
                'num_domain': 15000,
                'num_boundary': 1000,
                'num_initial': 1000,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 20000,
                'num_test': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'z_min': 0.0, 'z_max': 1.0,
                'num_outputs': 1,
                'output_names': ['u'],
                'ic_weight': 100.0,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "3D", "heat"),
            },
        }
        if text in templates_3d:
            t = templates_3d[text]
            n_out = t.get('num_outputs', 1)
            if n_out != self.num_outputs_spin.value():
                self.num_outputs_spin.setValue(n_out)
            for i, name in enumerate(t.get('output_names', ['u'])):
                if i < len(self.output_name_inputs):
                    self.output_name_inputs[i].setText(name)
            for i, pde_text in enumerate(t['pde']):
                if i < len(self.pde_inputs):
                    self.pde_inputs[i].setText(pde_text)
            for i, ic_text in enumerate(t['ic']):
                if i < len(self.ic_inputs):
                    self.ic_inputs[i].setText(ic_text)
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            self.z_min.setValue(t['z_min']); self.z_max.setValue(t['z_max'])
            if 't_max' in t: self.t_max.setValue(t['t_max'])
            self._current_template_forward_tmax = t.get('t_max')
            self._current_template_inverse_tmax = t.get('inverse_t_max')
            if self._current_template_inverse_tmax is not None and self.radio_inverse.isChecked():
                self.t_max.setValue(self._current_template_inverse_tmax)
            self.num_domain.setValue(t['num_domain'])
            self.num_boundary.setValue(t['num_boundary'])
            self.num_initial.setValue(t['num_initial'])
            if 'num_test' in t:
                self.num_test.setValue(t['num_test'])
            self.layers_spin.setValue(t['layers'])
            self.neurons_spin.setValue(t['neurons'])
            self.iter1_spin.setValue(t['iterations'])
            self.opt2_combo.setCurrentText(t['optimizer2'])
            self.iter2_spin.setValue(t['iterations2'])
            for i in range(n_out):
                key = f"ic_{i}"
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(t.get('ic_weight', 100.0))
            self._populate_3d_heat_bc_entries(n_out)
            self._update_bc_mode_visibility()
            for row in list(self.ta_group_rows):
                row['widget'].deleteLater()
            self.ta_group_rows.clear()
            # 3D Heat doesn't use Time-Adaptive mode (same as 2D Heat).
            self._current_ta_cfg = None
            self._ta_suspended_for_inverse = False
            self.adapt_combo.setCurrentText("None")
            self._add_ta_step_group(0.0, 1.0, 10)
            self.ta_transfer_cb.setChecked(False)
            self.ta_grid.setCurrentText("101")
            self._set_combo_data(self.ta_transfer_opt, "adam")
            self._template_ref_dir = t.get('ref_dir', '')
            self._current_template = text
            self._current_template_type = t.get('template_type', '')
            self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
            if hasattr(self, 'sched_cb'):
                self.sched_cb.setChecked(True)
                self._setup_default_scheduler_phases(t.get('template_type', ''), t['iterations'], t.get('iterations2', 10000))
            # The x-max Dirichlet face defaults to weight 100, same
            # empirically-confirmed convention as 2D Heat's right edge. Set
            # after _setup_default_scheduler_phases(), since that call
            # rebuilds weight_widgets (and would otherwise wipe this back
            # to the default of 1.0).
            for _bj, _be in enumerate(self.custom_bc_list):
                if _be['type'].currentData() == 'dirichlet':
                    _bk = f"bc_{_bj}"
                    if _bk in self.weight_widgets:
                        self.weight_widgets[_bk].setValue(100.0)
            self._auto_configure_ea(self._template_ref_dir)
            self.log_box.append(f"✅ Template loaded: {text}")
            return

        templates = {
            "1D Heat": {
                'pde': ["du_t - 0.4 * du_xx"],
                'ic': ["sin(pi*x)"],
                'num_domain': 5000,
                'num_boundary': 200,
                'num_initial': 200,
                'layers': 4,
                'neurons': 64,
                'iterations': 10000,
                'optimizer2': 'none',
                'iterations2': 5000,
                'x_min': 0.0, 'x_max': 1.0,
                'periodic_bc': False,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "1D", "heat"),
            },
            "1D Allen-Cahn": {
                'pde': ["du_t - 0.0001*du_xx + 5*u**3 - 5*u"],
                'ic': ["x**2*cos(pi*x)"],
                'num_domain': 10000,
                'num_boundary': 200,
                'num_initial': 512,
                'layers': 4,
                'neurons': 128,
                'iterations': 20000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': -1.0, 'x_max': 1.0,
                'periodic_bc': True,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "1D", "allen_cahn"),
                'ta_default': {'step_groups': [(0.0, 1.0, 4)], 'transfer_learning': True, 'ic_grid': 101, 'transfer_optimizer': 'lbfgs'},
            },
        }

        t = templates.get(text)
        if not t:
            return
        
        # Set num outputs and names if specified
        if 'num_outputs' in t:
            self.num_outputs_spin.setValue(t['num_outputs'])
            QApplication.processEvents()
        if 'output_names' in t:
            names = [n.strip() for n in t['output_names'].split(',')]
            for i, name in enumerate(names):
                if i < len(self.output_name_inputs):
                    self.output_name_inputs[i].setText(name)

        # Set PDE
        for i, pde_text in enumerate(t['pde']):
            if i < len(self.pde_inputs):
                self.pde_inputs[i].setText(pde_text)

        # Set IC
        for i, ic_text in enumerate(t['ic']):
            if i < len(self.ic_inputs):
                self.ic_inputs[i].setText(ic_text)

        # Set collocation points
        self.num_domain.setValue(t['num_domain'])
        self.num_boundary.setValue(t['num_boundary'])
        self.num_initial.setValue(t['num_initial'])

        # Set network
        self.layers_spin.setValue(t['layers'])
        self.neurons_spin.setValue(t['neurons'])

        # Set training
        self.iter1_spin.setValue(t['iterations'])
        self.opt2_combo.setCurrentText(t['optimizer2'])
        self.iter2_spin.setValue(t['iterations2'])

        # Set IC weight to 100 for Allen-Cahn
        if text in ["1D Allen-Cahn"]:
            for i in range(self.num_outputs_spin.value()):
                key = f"ic_{i}"
                if key in self.weight_widgets:
                    self.weight_widgets[key].setValue(100.0)

        # Set domain x range
        if 'x_min' in t:
            self.x_min.setValue(t['x_min'])
            self.x_max.setValue(t['x_max'])

        # Set periodic BC
        if t.get('periodic_bc', False):
            for i in range(self.num_outputs_spin.value()):
                if i < len(self.bc_left_types):
                    self.bc_left_types[i].setCurrentText("Periodic")
        else:
            for i in range(self.num_outputs_spin.value()):
                if i < len(self.bc_left_types):
                    self.bc_left_types[i].setCurrentText("Dirichlet")
                if i < len(self.bc_right_types):
                    self.bc_right_types[i].setCurrentText("Dirichlet")
        self._populate_locked_bc_entries_from_legacy(self.num_outputs_spin.value(), is_2d=False)
        self._update_bc_mode_visibility()

        # Set Time-Adaptive default (e.g. 1D Allen-Cahn), same pattern as
        # the 2D/3D template blocks above.
        for row in list(self.ta_group_rows):
            row['widget'].deleteLater()
        self.ta_group_rows.clear()
        ta_cfg = t.get('ta_default')
        self._current_ta_cfg = ta_cfg
        self._ta_suspended_for_inverse = False
        if ta_cfg and not self.radio_inverse.isChecked():
            self.adapt_combo.setCurrentText("Time Adaptive")
            for g_start, g_end, g_steps in ta_cfg['step_groups']:
                self._add_ta_step_group(g_start, g_end, g_steps)
            self.ta_transfer_cb.setChecked(ta_cfg.get('transfer_learning', False))
            self.ta_grid.setCurrentText(str(ta_cfg.get('ic_grid', 101)))
            self._set_combo_data(self.ta_transfer_opt, ta_cfg.get('transfer_optimizer', 'adam'))
        else:
            self.adapt_combo.setCurrentText("None")
            self._add_ta_step_group(0.0, 1.0, 10)
            self.ta_transfer_cb.setChecked(False)
            self.ta_grid.setCurrentText("101")
            self._set_combo_data(self.ta_transfer_opt, "adam")
            if ta_cfg and self.radio_inverse.isChecked():
                self._ta_suspended_for_inverse = True

        # Store ref_dir for error analysis auto-population
        self._template_ref_dir = t.get('ref_dir', '')
        self._current_template = text
        self._current_template_type = t.get('template_type', '')
        self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
        if hasattr(self, 'sched_cb'):
            self.sched_cb.setChecked(True)
            self._setup_default_scheduler_phases(t.get('template_type', ''), t['iterations'], t.get('iterations2', 10000))
        self._auto_configure_ea(self._template_ref_dir)
        self.log_box.append(f"✅ Template loaded: {text}")
    def _on_plot_settings(self):
        viz_type = self.plot_type_combo.currentText()
        if viz_type == "📊 Error Analysis":
            self._on_error_analysis_settings()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._plot_viz_settings

        # Colormap
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"])
        cmap_combo.setCurrentText(current.get('colormap', 'RdBu_r'))
        cmap_combo.setFixedWidth(120)
        cmap_row.addStretch(); cmap_row.addWidget(cmap_combo)
        layout.addLayout(cmap_row)

        # Contour levels
        levels_row = QHBoxLayout()
        levels_row.addWidget(QLabel("Contour levels:"))
        levels_spin = QSpinBox()
        levels_spin.setRange(5, 200); levels_spin.setValue(current.get('levels', 50))
        levels_spin.setFixedWidth(80)
        levels_row.addStretch(); levels_row.addWidget(levels_spin)
        layout.addLayout(levels_row)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Grid resolution:"))
        res_combo = QComboBox()
        res_combo.addItems(["80", "100", "150", "200"])
        res_combo.setCurrentText(str(current.get('resolution', 100)))
        res_combo.setFixedWidth(80)
        res_row.addStretch(); res_row.addWidget(res_combo)
        layout.addLayout(res_row)

        # DPI
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Output DPI:"))
        dpi_combo = QComboBox()
        dpi_combo.addItems(["100", "150", "200", "300"])
        dpi_combo.setCurrentText(str(current.get('dpi', 100)))
        dpi_combo.setFixedWidth(80)
        dpi_row.addStretch(); dpi_row.addWidget(dpi_combo)
        layout.addLayout(dpi_row)

        # Color range — Surface only
        color_range_widget = QWidget()
        cr_layout = QVBoxLayout(color_range_widget)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        cr_auto_cb = QCheckBox("Auto color range")
        cr_auto_cb.setChecked(current.get('auto_range', True))
        cr_layout.addWidget(cr_auto_cb)
        cr_manual_widget = QWidget()
        cr_manual_layout = QHBoxLayout(cr_manual_widget)
        cr_manual_layout.setContentsMargins(0, 0, 0, 0)
        cr_manual_layout.addWidget(QLabel("vmin:"))
        vmin_spin = QDoubleSpinBox(); vmin_spin.setRange(-1e6, 1e6)
        vmin_spin.setValue(current.get('vmin', -1.0)); vmin_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmin_spin)
        cr_manual_layout.addWidget(QLabel("vmax:"))
        vmax_spin = QDoubleSpinBox(); vmax_spin.setRange(-1e6, 1e6)
        vmax_spin.setValue(current.get('vmax', 1.0)); vmax_spin.setFixedWidth(80)
        cr_manual_layout.addWidget(vmax_spin)
        cr_manual_widget.setVisible(not cr_auto_cb.isChecked())
        cr_layout.addWidget(cr_manual_widget)
        cr_auto_cb.stateChanged.connect(lambda s: cr_manual_widget.setVisible(s != 2))
        color_range_widget.setVisible(viz_type == "Surface")
        layout.addWidget(color_range_widget)

        # 2D snapshots — Surface only, 2D mode
        snap_widget = QWidget()
        snap_layout = QHBoxLayout(snap_widget)
        snap_layout.setContentsMargins(0, 0, 0, 0)
        snap_layout.addWidget(QLabel("2D time snapshots:"))
        snap_spin = QSpinBox()
        snap_spin.setRange(1, 10); snap_spin.setValue(current.get('n_2d_snapshots', 2))
        snap_spin.setFixedWidth(80)
        snap_layout.addStretch(); snap_layout.addWidget(snap_spin)
        snap_widget.setVisible(viz_type == "Surface" and self.radio_2d.isChecked())
        layout.addWidget(snap_widget)

        # Colorbar — Surface only
        colorbar_cb = QCheckBox("Show colorbar")
        colorbar_cb.setChecked(current.get('colorbar', True))
        colorbar_cb.setVisible(viz_type == "Surface")
        layout.addWidget(colorbar_cb)

        # Line width — Line only
        lw_widget = QWidget()
        lw_layout = QHBoxLayout(lw_widget)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        lw_layout.addWidget(QLabel("Line width:"))
        lw_combo = QComboBox()
        lw_combo.addItems(["1.0", "1.5", "2.0", "2.5", "3.0"])
        lw_combo.setCurrentText(str(current.get('linewidth', 2.0)))
        lw_combo.setFixedWidth(80)
        lw_layout.addStretch(); lw_layout.addWidget(lw_combo)
        lw_widget.setVisible(viz_type == "Line (time steps)")
        layout.addWidget(lw_widget)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def _on_ok():
            self._plot_viz_settings = {
                'colormap': cmap_combo.currentText(),
                'levels': levels_spin.value(),
                'resolution': int(res_combo.currentText()),
                'dpi': int(dpi_combo.currentText()),
                'auto_range': cr_auto_cb.isChecked(),
                'vmin': vmin_spin.value(),
                'vmax': vmax_spin.value(),
                'colorbar': colorbar_cb.isChecked(),
                'linewidth': float(lw_combo.currentText()),
                'n_steps': current.get('n_steps', 4),
                'n_2d_snapshots': snap_spin.value(),
                'surface_time': current.get('surface_time', 1.0),
                'fps': current.get('fps', 10),
            }
            self.log_box.append(f"✅ Plot settings saved — {viz_type}, cmap={cmap_combo.currentText()}, levels={levels_spin.value()}, dpi={dpi_combo.currentText()}")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _on_error_analysis_btn(self):
        """Standalone error analysis button — opens dialog."""
        self._ea_ref_files = getattr(self, '_ea_ref_files', [])
        self._show_ea_dialog()

    def _show_ea_dialog(self):
        import os, glob

        dialog = QDialog(self)
        dialog.setWindowTitle("📊 Error Analysis")
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)

        # ── Plot type selection ────────────────────────────────
        plot_group = QGroupBox("Plot Types")
        plot_layout = QVBoxLayout(plot_group)
        self._ea_line_cb    = QCheckBox("Line comparison (PINN vs FEM at each time step)")
        self._ea_surface_cb = QCheckBox("Surface comparison (PINN | FEM | Error)")
        self._ea_line_cb.setChecked(True)
        self._ea_surface_cb.setChecked(True)
        plot_layout.addWidget(self._ea_line_cb)
        plot_layout.addWidget(self._ea_surface_cb)
        layout.addWidget(plot_group)

        # ── Error metrics ──────────────────────────────────────
        metrics_group = QGroupBox("Error Metrics to Compute")
        metrics_layout = QHBoxLayout(metrics_group)
        self._ea_l2_cb  = QCheckBox("L2 Relative"); self._ea_l2_cb.setChecked(True)
        self._ea_mse_cb = QCheckBox("MSE");          self._ea_mse_cb.setChecked(True)
        self._ea_max_cb = QCheckBox("Max Error");    self._ea_max_cb.setChecked(True)
        for w in [self._ea_l2_cb, self._ea_mse_cb, self._ea_max_cb]:
            metrics_layout.addWidget(w)
        layout.addWidget(metrics_group)

        # ── Reference files ────────────────────────────────────
        files_group = QGroupBox("Reference Files (FEM/Exact)")
        files_layout = QVBoxLayout(files_group)

        hint = QLabel("Format: space-separated, 3 columns: x  t  u  (no header)\nEach file = one time snapshot.")
        self._register_style(hint, "hint", lambda css, _c='#586e75', _e='': f"color: {_c}; {_e}{css}")
        files_layout.addWidget(hint)

        # File list widget
        self._ea_file_list_widget = QWidget()
        self._ea_file_list_layout = QVBoxLayout(self._ea_file_list_widget)
        self._ea_file_list_layout.setSpacing(4)
        self._ea_file_list_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(self._ea_file_list_widget)

        self._ea_file_rows = []  # list of (path_label, t_label, remove_btn)

        def _add_file_row(path='', t_val=None):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            path_edit = QLineEdit()
            path_edit.setPlaceholderText("Browse for reference file...")
            path_edit.setFixedHeight(26)
            if path:
                path_edit.setText(path)
            row_layout.addWidget(path_edit)

            browse_btn = QPushButton("Browse")
            browse_btn.setFixedHeight(26); browse_btn.setFixedWidth(60)
            row_layout.addWidget(browse_btn)

            t_label = QLabel("")
            self._register_style(t_label, "hint", lambda css, _c='#69db7c', _e='min-width: 80px; ': f"color: {_c}; {_e}{css}")
            row_layout.addWidget(t_label)

            remove_btn = QPushButton("✕")
            remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
            remove_btn.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
            row_layout.addWidget(remove_btn)

            self._ea_file_list_layout.addWidget(row_widget)
            row_data = {'widget': row_widget, 'path': path_edit, 't_label': t_label}
            self._ea_file_rows.append(row_data)

            def _on_browse():
                f, _ = QFileDialog.getOpenFileName(dialog, "Select reference file", "", "Text files (*.txt *.csv *.dat)")
                if f:
                    path_edit.setText(f)
                    _detect_time(f, t_label)

            def _detect_time(fpath, lbl):
                try:
                    import numpy as np
                    data = np.loadtxt(fpath)
                    if data.ndim == 1: data = data.reshape(1, -1)
                    # Auto-detect: 2D files have 4 cols (x,y,t,u), 1D have 3 cols (x,t,u)
                    t_col = 2 if data.shape[1] >= 4 else 1
                    t_detected = float(data[0, t_col])
                    lbl.setText(f"✅ t = {t_detected:.4f} ({len(data)} pts)")
                    row_data['t_val'] = t_detected
                except Exception as e:
                    lbl.setText(f"❌ {str(e)[:30]}")

            browse_btn.clicked.connect(_on_browse)

            def _on_remove():
                row_widget.deleteLater()
                if row_data in self._ea_file_rows:
                    self._ea_file_rows.remove(row_data)

            remove_btn.clicked.connect(_on_remove)

            if path and t_val is not None:
                t_label.setText(f"✅ t = {t_val:.4f}")
                row_data['t_val'] = t_val
            elif path:
                _detect_time(path, t_label)

            return row_data

        # Auto-populate from template if available
        ref_dir = getattr(self, '_template_ref_dir', '')
        if ref_dir and os.path.isdir(ref_dir):
            txt_files = sorted(glob.glob(os.path.join(ref_dir, 't_*.txt')))
            if txt_files:
                hint2 = QLabel(f"📂 Auto-loaded from template: {os.path.basename(ref_dir)}")
                self._register_style(hint2, "hint", lambda css, _c='#a0c4ff', _e='': f"color: {_c}; {_e}{css}")
                files_layout.addWidget(hint2)
                for f in txt_files:
                    _add_file_row(f)
        else:
            _add_file_row()  # start with one empty row

        add_btn = QPushButton("➕ Add another file")
        add_btn.setStyleSheet("QPushButton { color: #69db7c; background: transparent; border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        add_btn.clicked.connect(lambda: _add_file_row())
        files_layout.addWidget(add_btn)
        layout.addWidget(files_group)

        # ── Buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()
        run_btn = QPushButton("▶ Run Error Analysis")
        run_btn.setStyleSheet("QPushButton { background: #1a4a6a; color: #74c0fc; font-weight: bold; border-radius: 4px; padding: 4px 14px; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(run_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        cancel_btn.clicked.connect(dialog.reject)

        def _on_run():
            save_dir = self.save_dir_input.text().strip()
            if not save_dir:
                self.log_box.append("❌ Error Analysis: please set a save directory first.")
                dialog.reject(); return

            # Collect valid file rows
            valid_files = []
            for row in self._ea_file_rows:
                p = row['path'].text().strip()
                t = row.get('t_val', None)
                if p and os.path.exists(p) and t is not None:
                    valid_files.append((t, p))

            if not valid_files:
                self.log_box.append("❌ Error Analysis: no valid reference files loaded.")
                return

            valid_files.sort(key=lambda x: x[0])

            self._ea_settings = {
                'files': valid_files,
                'do_line': self._ea_line_cb.isChecked(),
                'do_surface': self._ea_surface_cb.isChecked(),
                'do_l2': self._ea_l2_cb.isChecked(),
                'do_mse': self._ea_mse_cb.isChecked(),
                'do_max': self._ea_max_cb.isChecked(),
            }
            self.log_box.append(f"✅ Error Analysis configured — {len(valid_files)} reference files, will run after training.")
            dialog.accept()

        run_btn.clicked.connect(_on_run)
        dialog.exec()

    def _execute_error_analysis_v2(self, ea, config):
        import tempfile, glob, json
        save_dir = self.save_dir_input.text().strip()
        is_2d = config.problem_dim == "2D"

        # Find most recently modified model
        model_path = ""
        all_models = glob.glob(os.path.join(save_dir, "model_lbfgs-*.pt")) + \
                     glob.glob(os.path.join(save_dir, "model_adam-*.pt"))
        if all_models:
            model_path = max(all_models, key=os.path.getmtime)
            self.log_box.append(f"📂 Using model: {os.path.basename(model_path)}")

        if not model_path:
            self.log_box.append("❌ No saved model found."); return

        config_path = model_path.replace(".pt", ".json")
        if not os.path.exists(config_path):
            config_path = os.path.join(save_dir, "model_config.json")

        try:
            with open(config_path) as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read model config: {e}"); return

        layers     = cfg["layers"]
        activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        loss_type  = cfg.get("loss_type", "MSE")
        compile_opt = "lbfgs" if "lbfgs" in model_path else "adam"

        files_list = ea['files']   # list of (t_val, filepath)
        do_line    = ea['do_line']
        do_surface = ea['do_surface']
        do_l2      = ea['do_l2']
        do_mse     = ea['do_mse']
        do_max     = ea['do_max']

        # Serialise file list for script
        files_repr = repr(files_list)

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Restore model ─────────────────────────────────────────────
if {is_2d}:
    geom = dde.geometry.Rectangle([{x_min}, {cfg.get('y_min', 0.0)}], [{x_max}, {cfg.get('y_max', 1.0)}])
else:
    geom  = dde.geometry.Interval({x_min}, {x_max})
td    = dde.geometry.TimeDomain({t_min}, {t_max})
gt    = dde.geometry.GeometryXTime(geom, td)
def pde(x, y): return y[:, 0:1] * 0
data  = dde.data.TimePDE(gt, pde, [], num_domain=100, num_test=100)
net   = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)
if "{compile_opt}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}")
else:
    model.compile("adam", lr=0.001, loss="{loss_type}")

model.restore(r"{model_path}", verbose=0)
print("✅ Model restored for error analysis.")
print(f"   Model path: {model_path}")
print(f"   x range: [{x_min}, {x_max}]")
print(f"   t range: [{t_min}, {t_max}]")

# ── Output directory ──────────────────────────────────────────
_ea_dir = os.path.join(r"{save_dir}", "error_analysis")
os.makedirs(_ea_dir, exist_ok=True)

# ── Load reference files ──────────────────────────────────────
_files = {files_repr}
_times = []; _x_refs = []; _u_refs = []; _d_refs = []; _d_shape = 0
for _tv, _fp in _files:
    _d = np.loadtxt(_fp)
    if _d.ndim == 1: _d = _d.reshape(1, -1)
    _d_shape = _d.shape[1]
    _idx = np.argsort(_d[:, 0])
    _x_refs.append(_d[_idx, 0])
    _u_col = 3 if _d_shape >= 4 else 2
    _u_refs.append(_d[_idx, _u_col])
    _d_refs.append(_d[_idx])
    _times.append(float(_tv))
    print(f"  Loaded t={{_tv:.4f}}: {{len(_d)}} points from {{os.path.basename(_fp)}}")

_n_t = len(_times)

# ── Predict PINN at EXACT FEM x values for each time ─────────
_u_pinns = []
for _i, _tv in enumerate(_times):
    _x_fem = _x_refs[_i]
    if _d_shape >= 4:  # 2D file: has x,y,t,c columns
        _y_fem = _d_refs[_i][:, 1]
        _xt = np.column_stack([_x_fem, _y_fem, np.full_like(_x_fem, _tv)])
    else:
        _xt = np.column_stack([_x_fem, np.full_like(_x_fem, _tv)])
    _u_pinns.append(model.predict(_xt)[:, 0].flatten())
    print(f"  PINN predicted at t={{_tv:.4f}}: {{len(_x_fem)}} points")

# ── Error metrics at exact FEM points ────────────────────────
_metrics = []
for _i, _tv in enumerate(_times):
    _up = _u_pinns[_i]; _uf = _u_refs[_i]
    _abs_err = np.abs(_up - _uf)
    _l2      = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
    _mse     = np.mean((_up - _uf)**2)
    _mx      = np.max(_abs_err)
    _ma      = np.mean(_abs_err)
    _metrics.append((_tv, _l2, _mse, _mx, _ma))
    print(f"  t={{_tv:.4f}} — L2={{_l2:.4e}}, MSE={{_mse:.4e}}, Max={{_mx:.4e}}, MeanAbs={{_ma:.4e}}")

# Save metrics
with open(os.path.join(_ea_dir, "error_metrics.txt"), "w") as _mf:
    _mf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
    for _tv, _l2, _mse, _mx, _ma in _metrics:
        _mf.write(f"{{_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
print(f"Metrics saved: {{os.path.join(_ea_dir, 'error_metrics.txt')}}")

# ── Line comparison — PINN vs FEM at exact x values ──────────
if {do_line}:
    _ncols = min(4, _n_t)
    _nrows = (_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    fig.suptitle("PINN vs FEM — Line Comparison", fontsize=13, fontweight='bold')
    _ax_flat = axes.flatten()
    for _i in range(_n_t):
        ax = _ax_flat[_i]
        _xv = _x_refs[_i]
        _tv, _l2, _mse, _mx, _ma = _metrics[_i]
        ax.plot(_xv, _u_refs[_i],  color='#4dabf7', linewidth=2.0, label='FEM (Exact)')
        ax.plot(_xv, _u_pinns[_i], color='#ff6b6b', linewidth=1.8, linestyle='--', label='PINN')
        ax.set_title(f"t = {{_tv:.3f}}  |  L2 = {{_l2:.2e}}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
        ax.grid(True, alpha=0.3)
    for _j in range(_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    handles, labels = _ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _lp = os.path.join(_ea_dir, "line_comparison.png")
    plt.savefig(_lp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Line comparison saved: {{_lp}}")

# ── Absolute error lines ──────────────────────────────────────
if {do_line}:
    _ncols = min(4, _n_t)
    _nrows = (_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    fig.suptitle("Absolute Error  |PINN - FEM|", fontsize=13, fontweight='bold')
    _ax_flat = axes.flatten()
    for _i in range(_n_t):
        ax = _ax_flat[_i]
        _xv = _x_refs[_i]
        _abs_err = np.abs(_u_pinns[_i] - _u_refs[_i])
        _tv, _l2, _mse, _mx, _ma = _metrics[_i]
        ax.plot(_xv, _abs_err, color='#69db7c', linewidth=2.0)
        ax.fill_between(_xv, _abs_err, alpha=0.25, color='#69db7c')
        ax.set_title(f"t = {{_tv:.3f}}  |  Max = {{_mx:.2e}}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("|error|")
        ax.grid(True, alpha=0.3)
    for _j in range(_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    plt.tight_layout()
    _ep = os.path.join(_ea_dir, "absolute_error_lines.png")
    plt.savefig(_ep, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Absolute error saved: {{_ep}}")

# ── Surface comparison — interpolate FEM to common grid ──────
if {do_surface}:
    _x_common = np.linspace({x_min}, {x_max}, 300)
    _t_arr = np.array(_times)
    _U_pinn_surf = np.zeros((len(_t_arr), len(_x_common)))
    _U_fem_surf  = np.zeros((len(_t_arr), len(_x_common)))

    for _i, _tv in enumerate(_times):
        if _d_shape >= 4:
            _y_common = _d_refs[_i][:, 1].mean() * np.ones_like(_x_common)
            _xt_c = np.column_stack([_x_common, _y_common, np.full_like(_x_common, _tv)])
        else:
            _xt_c = np.column_stack([_x_common, np.full_like(_x_common, _tv)])
        _U_pinn_surf[_i, :] = model.predict(_xt_c)[:, 0].flatten()
        _fi = interp1d(_x_refs[_i], _u_refs[_i], kind='linear', fill_value='extrapolate')
        _U_fem_surf[_i, :] = _fi(_x_common)

    _Xg, _Tg = np.meshgrid(_x_common, _t_arr)
    _U_err_surf = np.abs(_U_pinn_surf - _U_fem_surf)
    _vmin = min(_U_pinn_surf.min(), _U_fem_surf.min())
    _vmax = max(_U_pinn_surf.max(), _U_fem_surf.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("PINN vs FEM — Surface Comparison", fontsize=13, fontweight='bold')

    im0 = axes[0].contourf(_Tg, _Xg, _U_pinn_surf, levels=50, cmap='viridis', vmin=_vmin, vmax=_vmax)
    axes[0].set_title("PINN  u(x,t)"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(_Tg, _Xg, _U_fem_surf, levels=50, cmap='viridis', vmin=_vmin, vmax=_vmax)
    axes[1].set_title("FEM  u(x,t)"); axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(_Tg, _Xg, _U_err_surf, levels=50, cmap='YlOrRd')
    axes[2].set_title("Error  |PINN - FEM|"); axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    _sp = os.path.join(_ea_dir, "surface_comparison.png")
    plt.savefig(_sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Surface comparison saved: {{_sp}}")

print("ERROR_ANALYSIS_V2_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script); tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig
        class _EAThread2(QThread):
            line_sig = _sig(str)
            done_sig = _sig(bool)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.line_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)

        self._ea_thread2 = _EAThread2(tmp)
        self._ea_thread2.line_sig.connect(self.log_box.append)
        self._ea_thread2.done_sig.connect(self._on_ea_v2_done)
        self._ea_thread2.start()

    def _on_ea_v2_done(self, success):
        save_dir = self.save_dir_input.text().strip()
        if success:
            self.log_box.append("✅ Error analysis complete! Results in error_analysis/ folder.")
            # Show surface comparison if it exists
            for name in ["surface_comparison.png", "line_comparison.png"]:
                p = os.path.join(save_dir, "error_analysis", name)
                if os.path.exists(p):
                    self.solution_label.setPixmap(QPixmap(p).scaled(
                        500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    break
        else:
            self.log_box.append("❌ Error analysis failed — check log.")
    
    def _build_scheduler_phases_json(self):
        import json
        if not hasattr(self, 'sched_cb') or not self.sched_cb.isChecked():
            return ""
        phases = []
        same_w = self.sched_same_weights_cb.isChecked()
        n_out_sc = self.num_outputs_spin.value()
        n_bc_sc = len(getattr(self, 'custom_bc_list', []))
        for ph in self.sched_phase_list:
            pn = ph['phase_num']
            # Block order matches _flat_loss_weights_string()/codegen.py:
            # PDE per output, then one weight per Boundary Conditions panel
            # row, then IC per output -- never interleaved per-output.
            _suffix = "" if same_w else f"_p{pn}"
            w_str = ",".join(
                [str(self.weight_widgets.get(f"pde_{i}{_suffix}", SciLineEdit(1.0)).value())
                 for i in range(n_out_sc)]
                + [str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                   for j in range(n_bc_sc)
                   for k in [f"bc_{j}{_suffix}"]
                   if k in self.weight_widgets]
                + [str(self.weight_widgets.get(k, SciLineEdit(1.0)).value())
                   for i in range(n_out_sc)
                   for k in [f"ic_{i}{_suffix}"]
                   if k in self.weight_widgets]
            )
            if hasattr(self, 'radio_inverse') and self.radio_inverse.isChecked() and getattr(self, 'inv_data_rows', None):
                # Match codegen's _multi_weights: one extra observation-loss
                # weight appended at the end per measured-data file (one for
                # the legacy single file, or one per row when the panel has
                # more than one measured-data file configured).
                w_str = w_str + "," + ",".join(str(r['weight'].value()) for r in self.inv_data_rows)
            phases.append({
                'optimizer': ph['opt'].currentData(),
                'iterations': ph['iters'].value(),
                'lr': ph['lr'].value(),
                'loss': ph.get('loss', self.loss_combo).currentText() if 'loss' in ph else 'MSE',
                'weights': w_str,
                'decay_type': ph['decay_type'].currentData() if 'decay_type' in ph else 'none',
                'decay_p1': ph['decay_p1'].value() if 'decay_p1' in ph else 0,
                'decay_p2': ph['decay_p2'].value() if 'decay_p2' in ph else 0,
            })
        import json
        return json.dumps(phases)
    
    def _on_scheduler_changed(self, state):
        self.sched_widget.setVisible(state == 2)
        self._build_weight_inputs(self.num_outputs_spin.value())

    def _setup_default_scheduler_phases(self, template_type='', adam_iters=10000, lbfgs_iters=10000):
        """Clear existing phases and add defaults based on template."""
        # Clear existing phases
        for ph in list(self.sched_phase_list):
            ph['widget'].deleteLater()
        self.sched_phase_list.clear()

        # Default: Adam warm-up phase, then L-BFGS refinement, using same weights
        self._add_scheduler_phase('adam', adam_iters, 0.001)
        self._add_scheduler_phase('lbfgs', lbfgs_iters, 0.001)
        self.sched_same_weights_cb.setChecked(True)
        self._build_weight_inputs(self.num_outputs_spin.value())

    def _add_scheduler_phase(self, optimizer='adam', iterations=50000, lr=0.001):
        phase_num = len(self.sched_phase_list) + 1  # phase 1, 2, 3...
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        phase_layout.setSpacing(3)
        phase_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_row = QHBoxLayout()
        header_lbl = QLabel(f"── Phase {phase_num} ──")
        self._register_style(header_lbl, "hint", lambda css, _c='#a0c4ff', _e='': f"color: {_c}; {_e}{css}")
        header_row.addWidget(header_lbl)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        header_row.addStretch(); header_row.addWidget(remove_btn)
        phase_layout.addLayout(header_row)

        # Optimizer
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Optimizer:"))
        opt_combo = QComboBox()
        opt_combo.addItem("Adam", "adam")
        opt_combo.addItem("AdamW", "adamw")
        opt_combo.addItem("L-BFGS", "lbfgs")
        opt_combo.addItem("NNCG", "nncg")
        self._set_combo_data(opt_combo, optimizer)
        opt_combo.setFixedHeight(26); opt_combo.setFixedWidth(80)
        opt_row.addStretch(); opt_row.addWidget(opt_combo)
        phase_layout.addLayout(opt_row)

        # Iterations
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterations:"))
        iter_spin = QSpinBox()
        iter_spin.setRange(0, 500000); iter_spin.setSingleStep(1000)
        iter_spin.setValue(iterations); iter_spin.setFixedHeight(26)
        iter_row.addStretch(); iter_row.addWidget(iter_spin)
        phase_layout.addLayout(iter_row)

        # Learning rate (not used by L-BFGS/NNCG — hidden for those phases;
        # NNCG has its own internal learning rate via DeepXDE's NNCG
        # hyperparameters, left at DeepXDE's defaults for now)
        lr_widget = QWidget()
        lr_row = QHBoxLayout(lr_widget)
        lr_row.setContentsMargins(0, 0, 0, 0)
        lr_row.addWidget(QLabel("Learning rate:"))
        lr_spin = QDoubleSpinBox()
        lr_spin.setRange(1e-6, 1.0); lr_spin.setDecimals(6)
        lr_spin.setSingleStep(0.0001); lr_spin.setValue(lr)
        lr_spin.setFixedHeight(26)
        lr_row.addStretch(); lr_row.addWidget(lr_spin)
        phase_layout.addWidget(lr_widget)
        _lr_capable = optimizer not in ("lbfgs", "nncg")
        lr_widget.setVisible(_lr_capable)

        # Learning-rate decay (Adam/AdamW/SGD/RMSprop only — ignored by
        # L-BFGS/NNCG, so hidden together with the learning-rate field)
        decay_widget = QWidget()
        decay_layout = QVBoxLayout(decay_widget)
        decay_layout.setContentsMargins(0, 0, 0, 0)
        decay_layout.setSpacing(2)
        decay_type_row = QHBoxLayout()
        decay_type_row.addWidget(QLabel("Learning rate decay:"))
        decay_type_combo = QComboBox()
        decay_type_combo.addItem("None", "none")
        decay_type_combo.addItem("Step", "step")
        decay_type_combo.addItem("Cosine", "cosine")
        decay_type_combo.addItem("Exponential", "exponential")
        decay_type_combo.setFixedHeight(24); decay_type_combo.setFixedWidth(100)
        decay_type_row.addStretch(); decay_type_row.addWidget(decay_type_combo)
        decay_layout.addLayout(decay_type_row)

        decay_p1_row = QHBoxLayout()
        decay_p1_lbl = QLabel("Step size:")
        decay_p1_row.addWidget(decay_p1_lbl)
        decay_p1_spin = QDoubleSpinBox()
        decay_p1_spin.setRange(1, 500000); decay_p1_spin.setDecimals(0)
        decay_p1_spin.setValue(5000); decay_p1_spin.setFixedHeight(24)
        decay_p1_row.addStretch(); decay_p1_row.addWidget(decay_p1_spin)
        decay_layout.addLayout(decay_p1_row)

        decay_p2_row = QHBoxLayout()
        decay_p2_lbl = QLabel("Gamma:")
        decay_p2_row.addWidget(decay_p2_lbl)
        decay_p2_spin = QDoubleSpinBox()
        decay_p2_spin.setRange(0.0, 1.0); decay_p2_spin.setDecimals(4)
        decay_p2_spin.setSingleStep(0.01); decay_p2_spin.setValue(0.9)
        decay_p2_spin.setFixedHeight(24)
        decay_p2_row.addStretch(); decay_p2_row.addWidget(decay_p2_spin)
        decay_layout.addLayout(decay_p2_row)
        phase_layout.addWidget(decay_widget)
        decay_widget.setVisible(_lr_capable)

        def _update_decay_fields(_idx=None, c=decay_type_combo, l1=decay_p1_lbl,
                                  s1=decay_p1_spin, r2=decay_p2_row, l2=decay_p2_lbl,
                                  s2=decay_p2_spin):
            kind = c.currentData()
            if kind == "step":
                l1.setText("Step size:"); s1.setRange(1, 500000); s1.setDecimals(0)
                l2.setText("Gamma:"); r2widget_visible = True
            elif kind == "cosine":
                l1.setText("T_max (iters):"); s1.setRange(1, 500000); s1.setDecimals(0)
                l2.setText("Eta min:"); r2widget_visible = True
            elif kind == "exponential":
                l1.setText("Gamma:"); s1.setRange(0.0, 1.0); s1.setDecimals(4)
                r2widget_visible = False
            else:
                r2widget_visible = False
            s1.setVisible(kind != "none"); l1.setVisible(kind != "none")
            for i in range(r2.count()):
                item = r2.itemAt(i).widget()
                if item is not None:
                    item.setVisible(r2widget_visible)
        decay_type_combo.currentIndexChanged.connect(_update_decay_fields)
        _update_decay_fields()

        def _update_lr_capable(_idx=None, w1=lr_widget, w2=decay_widget, c=opt_combo):
            capable = c.currentData() not in ("lbfgs", "nncg")
            w1.setVisible(capable); w2.setVisible(capable)
        opt_combo.currentIndexChanged.connect(_update_lr_capable)

        # Loss function
        loss_row = QHBoxLayout()
        loss_row.addWidget(QLabel("Loss function:"))
        loss_combo_ph = QComboBox()
        loss_combo_ph.addItems(["MSE", "MAE", "mean l2 relative error",
                                 "mean absolute percentage error", "softplus"])
        loss_combo_ph.setFixedHeight(26); loss_combo_ph.setFixedWidth(160)
        loss_row.addStretch(); loss_row.addWidget(loss_combo_ph)
        phase_layout.addLayout(loss_row)

        self.sched_phases_layout.addWidget(phase_widget)
        phase_data = {
            'widget': phase_widget,
            'opt': opt_combo,
            'iters': iter_spin,
            'lr': lr_spin,
            'loss': loss_combo_ph,
            'phase_num': phase_num,
            'decay_type': decay_type_combo,
            'decay_p1': decay_p1_spin,
            'decay_p2': decay_p2_spin,
        }
        self.sched_phase_list.append(phase_data)

        def _remove():
            phase_widget.deleteLater()
            if phase_data in self.sched_phase_list:
                self.sched_phase_list.remove(phase_data)
            self._build_weight_inputs(self.num_outputs_spin.value())

        remove_btn.clicked.connect(_remove)
        self._build_weight_inputs(self.num_outputs_spin.value())
    
    def _on_ic_pretrain_changed(self, state):
        self.ic_pretrain_widget.setVisible(state == 2)

    def _update_ic_pretrain_visibility(self):
        """Show exactly the IC Pre-Training fields that matter for the
        currently-chosen mode: Train-from-start shows the optimizer/
        iterations/test-points/initial-points fields (initial-points also
        hidden if the IC itself is coming from a file, unrelated to this
        toggle); Restore-from-checkpoint shows only the checkpoint path,
        since restoring loads the saved weights and skips training
        entirely -- see codegen.py's ic_pretrain_restore branch."""
        if not hasattr(self, 'ic_pretrain_mode_restore'):
            return
        restoring = self.ic_pretrain_mode_restore.isChecked()
        self.ic_pretrain_train_fields_widget.setVisible(not restoring)
        self.ic_pretrain_restore_widget.setVisible(restoring)
        if hasattr(self, 'ic_from_file'):
            _any_ic_from_file = any(
                cb is not None and cb.isChecked() for cb in self.ic_from_file)
        else:
            _any_ic_from_file = False
        self.ic_pretrain_init_widget.setVisible(not restoring and not _any_ic_from_file)

    def _on_batch_changed(self, state):
        self.batch_widget.setVisible(state == 2)

    def _on_browse_restore_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select model file", "", "PyTorch model (*.pt)")
        if f:
            self.restore_model_path.setText(f)
            base = os.path.dirname(f)
            fname = os.path.basename(f).lower()
            # First try: same name as .pt but .json
            specific = f.replace(".pt", ".json")
            if os.path.exists(specific):
                self.restore_config_path.setText(specific)
                self.log_box.append(f"✅ Auto-detected config: {specific}")
            elif os.path.exists(os.path.join(base, "model_config.json")):
                self.restore_config_path.setText(os.path.join(base, "model_config.json"))
                self.log_box.append(f"✅ Auto-detected config (generic): {os.path.join(base, 'model_config.json')}")
            else:
                self.log_box.append("⚠️ No config found — please browse manually.")

            # Inverse-only convenience: auto-detect the *_convergence.txt
            # file(s) saved alongside this run (one level up from
            # solution_results/, same folder the model_config.json above
            # was auto-detected in) and the trainable-variable names read
            # from each file's own header row -- the same auto-detect
            # convenience as the config.json above, so an older model
            # missing "inverse_variables" in its config still gets this
            # filled in without the user typing anything.
            if getattr(self, 'restore_mode_combo', None) is not None and self.restore_mode_combo.currentText() == "Inverse Model":
                import glob as _glob_brm
                run_dir = os.path.dirname(base) if os.path.basename(base) == "solution_results" else base
                conv_files = sorted(_glob_brm.glob(os.path.join(run_dir, "*_convergence.txt")))
                if conv_files:
                    self._set_restore_param_rows(conv_files)
                    self.log_box.append(f"✅ Auto-detected {len(conv_files)} parameter convergence file(s) in {run_dir}")
                    names = []
                    for cf in conv_files:
                        try:
                            with open(cf) as _cf:
                                header = _cf.readline().strip().split(",")
                            if len(header) >= 2 and header[1].strip():
                                names.append(header[1].strip())
                        except Exception:
                            pass
                    if names and not self.restore_inv_var_names.text().strip():
                        self.restore_inv_var_names.setText(",".join(names))

    def _on_browse_restore_config(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select config file", "", "JSON (*.json)")
        if f:
            self.restore_config_path.setText(f)

    def _on_browse_restore_save(self):
        folder = QFileDialog.getExistingDirectory(self, "Select save directory")
        if folder:
            self.restore_save_path.setText(folder)

    def _add_restore_param_row(self, path=""):
        """Add one parameter-convergence-file row (Inverse restore's
        Parameter Convergence Plot/Animation viz types). Every row is
        freely removable, including the first -- unlike the trainable-
        variable / measured-data-file row lists elsewhere, there's no
        single legacy field a "primary" row needs to stay aliased to
        here, so at-least-one-path is simply checked at Restore time."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        path_edit = QLineEdit()
        path_edit.setText(path)
        path_edit.setPlaceholderText("Browse for <name>_convergence.txt...")
        path_edit.setFixedHeight(26)
        row_layout.addWidget(path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedHeight(26); browse_btn.setFixedWidth(65)
        browse_btn.clicked.connect(lambda _checked=False, e=path_edit: self._on_browse_restore_param(e))
        row_layout.addWidget(browse_btn)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
        remove_btn.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
        row_layout.addWidget(remove_btn)
        self.restore_param_files_layout.addWidget(row_widget)
        row_data = {'widget': row_widget, 'path': path_edit, 'browse': browse_btn}
        self.restore_param_rows.append(row_data)

        def _remove():
            row_widget.deleteLater()
            if row_data in self.restore_param_rows:
                self.restore_param_rows.remove(row_data)
        remove_btn.clicked.connect(_remove)
        return row_data

    def _set_restore_param_rows(self, paths):
        for row in list(self.restore_param_rows):
            row['widget'].deleteLater()
        self.restore_param_rows.clear()
        if not paths:
            self._add_restore_param_row()
        else:
            for p in paths:
                self._add_restore_param_row(p)

    def _on_browse_restore_param(self, target):
        f, _ = QFileDialog.getOpenFileName(self, "Select parameter convergence file", "", "Text files (*.txt)")
        if f:
            target.setText(f)

    def _on_restore(self):
        import json, tempfile, subprocess, sys
        viz_type    = self.restore_viz_combo.currentText()
        save_dir    = self.restore_save_path.text().strip()

        # Parameter Convergence Plot/Animation: entirely independent of
        # the model checkpoint (see _build_restore_param_script) -- reads
        # the *_convergence.txt file(s) directly, no model/config/
        # optimizer/output selection needed.
        if viz_type in getattr(self, '_RESTORE_PARAM_VIZ', []):
            if not save_dir:
                self.log_box.append("❌ Please select a save directory."); return
            paths = [r['path'].text().strip() for r in self.restore_param_rows if r['path'].text().strip()]
            if not paths:
                self.log_box.append("❌ Please add at least one parameter convergence .txt file."); return
            combine = self.restore_param_combine_combo.currentText().startswith("Combine")
            log_scale = self.restore_param_log_cb.isChecked()
            animate = (viz_type == "Parameter Convergence Animation (GIF)")

            self.restore_btn.setEnabled(False)
            self.restore_btn.setText("⏳ Restoring...")
            self.log_box.append(
                f"🔄 Building parameter convergence {'animation' if animate else 'plot'} "
                f"from {len(paths)} file(s)...")
            script = self._build_restore_param_script(paths, save_dir, combine, log_scale, animate)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
                tf.write(script)
                tmp = tf.name

            from PyQt6.QtCore import QThread, pyqtSignal as _sig

            class _ParamRestoreThread(QThread):
                line_signal = _sig(str)
                done_signal = _sig(bool)
                def __init__(self, tmp):
                    super().__init__()
                    self._tmp = tmp
                def run(self):
                    proc = subprocess.Popen(
                        [sys.executable, self._tmp],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                    )
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line: self.line_signal.emit(line)
                    proc.wait()
                    os.unlink(self._tmp)
                    self.done_signal.emit(proc.returncode == 0)

            self._restore_thread = _ParamRestoreThread(tmp)
            self._restore_thread.line_signal.connect(self.log_box.append)
            self._restore_thread.done_signal.connect(self._on_restore_done)
            self._restore_thread.start()
            return

        model_path  = self.restore_model_path.text().strip()
        config_path = self.restore_config_path.text().strip()
        optimizer   = self.restore_optimizer_combo.currentData()
        output_idx  = self.restore_output_combo.currentIndex()
        t_steps     = self.restore_tsteps_spin.value()

        if not model_path:
            self.log_box.append("❌ Please select a model file."); return
        if not config_path:
            self.log_box.append("❌ Please select a model_config.json file."); return
        if not save_dir:
            self.log_box.append("❌ Please select a save directory."); return

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read config: {e}"); return

        # Inverse restore: let the user override/supply the trainable-
        # variable names for an older model saved before model_config.json
        # recorded them ("inverse_variables"). model.restore() needs the
        # freshly-compiled model to have the SAME number of
        # external_trainable_variables the optimizer had at training time
        # or it crashes with a parameter-group-size mismatch -- see
        # _build_restore_script's comment. Only used when the box actually
        # has something typed in it; otherwise cfg is left exactly as
        # loaded (new-format configs already have this, old ones without
        # it and without an override still get the original crash, same
        # as before this box existed).
        if self.restore_mode_combo.currentText() == "Inverse Model":
            override_names = [n.strip() for n in self.restore_inv_var_names.text().split(",") if n.strip()]
            if override_names:
                cfg["problem_type"] = "Inverse"
                cfg["inverse_variables"] = [{"name": n, "init": 1.0} for n in override_names]

        self.restore_btn.setEnabled(False)
        self.restore_btn.setText("⏳ Restoring...")
        self.log_box.append(f"🔄 Restoring model from: {model_path}")
        viz_settings = getattr(self, '_restore_viz_settings', {})
        script = self._build_restore_script(model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir)

        # ── Append error analysis if files configured ─────────
        ea = getattr(self, '_ea_settings', None)
        if ea and ea.get('files'):
            # Read t range from step_config.json in same folder as model
            import json as _json_ea
            step_dir = os.path.dirname(model_path)
            step_cfg_path = os.path.join(step_dir, 'step_config.json')
            t_min_restore = cfg.get('t_min', 0.0)
            t_max_restore = cfg.get('t_max', 1.0)
            try:
                with open(step_cfg_path) as _sf:
                    _sc = _json_ea.load(_sf)
                t_min_restore = _sc.get('t_min', t_min_restore)
                t_max_restore = _sc.get('t_max', t_max_restore)
            except Exception:
                pass
            # Filter files within t range
            matching_files = [
                (t, f) for t, f in ea['files']
                if t_min_restore - 1e-10 <= t <= t_max_restore + 1e-10
            ]
            if matching_files:
                self.log_box.append(f"📊 Error analysis: {len(matching_files)} reference files match t=[{t_min_restore:.4f}, {t_max_restore:.4f}]")
                is_2d = cfg.get('problem_dim', '1D') == '2D'
                script += self._build_restore_ea_script(
                    matching_files, save_dir, is_2d,
                    ea.get('do_line', True), ea.get('do_surface', True),
                    cfg.get('x_min', 0.0), cfg.get('x_max', 1.0),
                    cfg.get('y_min', 0.0), cfg.get('y_max', 1.0),
                    cfg.get('output_names', 'u').split(',')[output_idx].strip(),
                    viz_settings
                )
            else:
                self.log_box.append(f"ℹ️ No reference files match t=[{t_min_restore:.4f}, {t_max_restore:.4f}] — skipping error analysis")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script)
            tmp = tf.name

        from PyQt6.QtCore import QThread, pyqtSignal as _sig

        class _RestoreThread(QThread):
            line_signal = _sig(str)
            done_signal = _sig(bool)
            def __init__(self, tmp):
                super().__init__()
                self._tmp = tmp
            def run(self):
                proc = subprocess.Popen(
                    [sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    if line: self.line_signal.emit(line)
                proc.wait()
                os.unlink(self._tmp)
                self.done_signal.emit(proc.returncode == 0)

        self._restore_thread = _RestoreThread(tmp)
        self._restore_thread.line_signal.connect(self.log_box.append)
        self._restore_thread.done_signal.connect(self._on_restore_done)
        self._restore_thread.start()

    def _on_restore_done(self, success):
        self.restore_btn.setEnabled(True)
        self.restore_btn.setText("🔄  Restore && Visualize")
        save_dir = self.restore_save_path.text().strip()
        if success:
            self.log_box.append("✅ Restore complete!")
            plot_path = os.path.join(save_dir, "restored_plot.png")
            if os.path.exists(plot_path):
                self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            gif_path = os.path.join(save_dir, "restored_animation.gif")
            if os.path.exists(gif_path):
                self.log_box.append(f"🎬 Animation saved: {gif_path}")
            # Parameter Convergence Plot/Animation outputs -- combined
            # (one file) or per-variable (several), PNG or GIF. Preview
            # whichever comes first alphabetically; every file produced is
            # already named in the log above from the script's own prints.
            import glob as _glob_done
            param_pngs = sorted(_glob_done.glob(os.path.join(save_dir, "*convergence_plot.png")))
            if param_pngs:
                self.solution_label.setPixmap(QPixmap(param_pngs[0]).scaled(
                    500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            param_gifs = sorted(_glob_done.glob(os.path.join(save_dir, "*convergence_animation.gif")))
            for _pg in param_gifs:
                self.log_box.append(f"🎬 Parameter convergence animation saved: {_pg}")
            # Show error analysis plot if available
            for _ea_plot in ["surface_comparison_restore.png", "line_comparison_restore.png"]:
                _ea_path = os.path.join(save_dir, "error_analysis", _ea_plot)
                if os.path.exists(_ea_path):
                    self.loss_label.setPixmap(QPixmap(_ea_path).scaled(
                        500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    self.log_box.append(f"📊 Error analysis plot: {_ea_path}")
                    break
        else:
            self.log_box.append("❌ Restore failed — check architecture matches saved model.")

    def _build_restore_param_script(self, paths, save_dir, combine, log_scale, animate):
        """Parameter Convergence Plot/Animation (Inverse restore). Reads
        one or more <name>_convergence.txt files (iteration,<name> header
        then iter,value rows -- exactly what _SaveParamCallback writes
        during training) and either draws them as-is (static PNG) or as a
        growing-curve animation (GIF), combined into one figure (one
        subplot per variable, same layout the training-time param_plot.png
        already uses) or as separate files per variable. Deliberately
        doesn't touch the model checkpoint at all -- neither
        net.state_dict() nor optimizer.state_dict() ever stores a trainable
        variable's actual value (only the network weights and optimizer
        momentum buffers), so there's nothing for a model restore to add
        here; these text files, written throughout training, are the only
        place the value's full history exists."""
        paths_literal = repr(list(paths))
        combine_literal = repr(bool(combine))
        log_literal = repr(bool(log_scale))
        animate_literal = repr(bool(animate))
        viz_settings = getattr(self, '_restore_viz_settings', {})
        title_override = (viz_settings.get('title') or '').strip()
        xlabel_override = (viz_settings.get('xlabel') or '').strip()
        ylabel_override = (viz_settings.get('ylabel') or '').strip()
        script = f"""
import os
os.makedirs(r"{save_dir}", exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as _anim
import numpy as np

_paths = {paths_literal}
_combine = {combine_literal}
_log_scale = {log_literal}
_animate = {animate_literal}
_title_override = {title_override!r}
_xlabel_override = {xlabel_override!r}
_ylabel_override = {ylabel_override!r}

def _load_conv(path):
    name = None
    iters, vals = [], []
    with open(path) as f:
        header = f.readline().strip().split(",")
        if len(header) >= 2 and header[0].strip().lower() == "iteration":
            name = header[1].strip()
        for line in f:
            line = line.strip()
            if not line:
                continue
            bits = line.split(",")
            if len(bits) < 2:
                continue
            try:
                it = int(float(bits[0])); val = float(bits[1])
            except ValueError:
                continue
            iters.append(it); vals.append(val)
    if not name:
        name = os.path.splitext(os.path.basename(path))[0].replace("_convergence", "")
    return name, np.array(iters), np.array(vals)

series = []
for _p in _paths:
    _p = _p.strip()
    if not _p:
        continue
    try:
        _s = _load_conv(_p)
        if len(_s[1]) < 1:
            print(f"⚠️ {{_p}}: no data rows found, skipping")
            continue
        series.append(_s)
    except Exception as e:
        print(f"⚠️ Could not read {{_p}}: {{e}}")

if not series:
    print("❌ No valid parameter-convergence files loaded.")
    raise SystemExit(1)

def _use_log(vals):
    ok = _log_scale and np.all(vals > 0)
    if _log_scale and not ok:
        print("⚠️ Log-scale requested but values aren't all positive -- using linear scale instead.")
    return ok

def _draw_static_ax(ax, name, iters, vals):
    final_val = vals[-1]
    if _use_log(vals):
        ax.semilogy(iters, vals, color="#69db7c", linewidth=1.5)
        ax.set_ylabel(_ylabel_override or f"log({{name}})")
    else:
        ax.plot(iters, vals, color="#69db7c", linewidth=1.5)
        ax.set_ylabel(_ylabel_override or name)
    ax.axhline(y=final_val, color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{final_val:.6f}}")
    ax.set_xlabel(_xlabel_override or "Iteration")
    ax.set_title(_title_override or f"Inferred Parameter: {{name}}")
    ax.legend(); ax.grid(True, alpha=0.3)

def _setup_anim_ax(ax, name, iters, vals):
    final_val = vals[-1]
    use_log = _use_log(vals)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel(_ylabel_override or f"log({{name}})")
    else:
        ax.set_ylabel(_ylabel_override or name)
    x_hi = iters.max() if iters.max() > iters.min() else iters.min() + 1
    ax.set_xlim(iters.min(), x_hi)
    vmin, vmax = vals.min(), vals.max()
    pad = 0.05 * (abs(vmax - vmin) if vmax != vmin else (abs(vmax) + 1))
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.axhline(y=final_val, color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{final_val:.6f}}")
    ax.set_xlabel(_xlabel_override or "Iteration")
    ax.set_title(_title_override or f"Inferred Parameter: {{name}}")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], color="#69db7c", linewidth=1.5)
    return line

if not _animate:
    if _combine:
        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3.2 * n), squeeze=False)
        for i, (name, iters, vals) in enumerate(series):
            _draw_static_ax(axes[i][0], name, iters, vals)
        plt.tight_layout()
        out_path = os.path.join(r"{save_dir}", "param_convergence_plot.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Parameter convergence plot saved: {{out_path}}")
    else:
        for name, iters, vals in series:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            _draw_static_ax(ax, name, iters, vals)
            plt.tight_layout()
            out_path = os.path.join(r"{save_dir}", f"{{name}}_convergence_plot.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"✅ Parameter convergence plot saved: {{out_path}}")
else:
    # The reveal-frame count is capped for GIF size/render time -- the
    # final frame always shows every logged point regardless, this only
    # affects how many intermediate steps the growth is drawn across. A
    # series with fewer points than the frame budget finishes revealing
    # early and holds its completed curve for the rest of the animation.
    if _combine:
        n = len(series)
        n_frames = min(max(len(s[1]) for s in series), 120)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3.2 * n), squeeze=False)
        lines = [_setup_anim_ax(axes[i][0], name, iters, vals) for i, (name, iters, vals) in enumerate(series)]
        plt.tight_layout()
        def update(frame):
            for line, (name, iters, vals) in zip(lines, series):
                frac = (frame + 1) / n_frames
                idx = max(1, min(len(iters), int(round(frac * len(iters)))))
                line.set_data(iters[:idx], vals[:idx])
            return lines
        ani = _anim.FuncAnimation(fig, update, frames=n_frames, interval=80)
        out_path = os.path.join(r"{save_dir}", "param_convergence_animation.gif")
        ani.save(out_path, writer='pillow', fps=15)
        plt.close(fig)
        print(f"✅ Parameter convergence animation saved: {{out_path}}")
    else:
        for name, iters, vals in series:
            n_frames = min(len(iters), 120)
            fig, ax = plt.subplots(figsize=(6, 3.2))
            line = _setup_anim_ax(ax, name, iters, vals)
            plt.tight_layout()
            def update(frame, iters=iters, vals=vals, line=line, n_frames=n_frames):
                frac = (frame + 1) / n_frames
                idx = max(1, min(len(iters), int(round(frac * len(iters)))))
                line.set_data(iters[:idx], vals[:idx])
                return [line]
            ani = _anim.FuncAnimation(fig, update, frames=n_frames, interval=80)
            out_path = os.path.join(r"{save_dir}", f"{{name}}_convergence_animation.gif")
            ani.save(out_path, writer='pillow', fps=15)
            plt.close(fig)
            print(f"✅ Parameter convergence animation saved: {{out_path}}")

print("RESTORE_DONE")
"""
        return script

    def _build_restore_script(self, model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir):
        viz_settings = getattr(self, '_restore_viz_settings', {})
        colormap = viz_settings.get('colormap', 'RdBu_r')
        surface_time = viz_settings.get('surface_time', cfg.get('t_max', 1.0))
        show_colorbar = viz_settings.get('colorbar', True)
        n_steps = viz_settings.get('n_steps', t_steps)
        levels = viz_settings.get('levels', 40)
        resolution = viz_settings.get('resolution', 100)
        dpi = viz_settings.get('dpi', 100)
        auto_range = viz_settings.get('auto_range', True)
        vmin_val = viz_settings.get('vmin', -1.0)
        vmax_val = viz_settings.get('vmax', 1.0)
        linewidth = viz_settings.get('linewidth', 2.0)
        fps = viz_settings.get('fps', 10)
        title_override = (viz_settings.get('title') or '').strip()
        xlabel_override = (viz_settings.get('xlabel') or '').strip()
        ylabel_override = (viz_settings.get('ylabel') or '').strip()
        layers     = cfg["layers"]
        activation = cfg["activation"]
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        z_min = cfg.get("z_min", 0.0); z_max = cfg.get("z_max", 1.0)
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        is_2d = cfg.get("problem_dim", "1D") == "2D"
        is_3d = cfg.get("problem_dim", "1D") == "3D"
        loss_type = cfg.get("loss_type", "MSE")
        out_names = cfg.get("output_names", "u").split(",")
        out_name = out_names[output_idx].strip() if output_idx < len(out_names) else "u"

        # Inverse models were compiled with one or more external_trainable_
        # variables (D, k, ...) added to the optimizer as an extra parameter
        # group -- model.restore() checks the saved optimizer state against
        # whatever the freshly-compiled model's optimizer looks like, so
        # restoring without recreating that same extra group fails with a
        # "different number of parameter groups" / "doesn't match the size
        # of optimizer's group" error (the exact crash this fixes). Their
        # actual VALUE doesn't matter here and isn't recoverable from the
        # checkpoint either way (neither net.state_dict() nor
        # opt.state_dict() ever stores it) -- only the same COUNT is needed
        # so the optimizer's structure matches what was saved.
        _restore_inv_vars = cfg.get("inverse_variables") or []
        is_inverse_restore = cfg.get("problem_type") == "Inverse" and bool(_restore_inv_vars)
        _inv_var_def_lines = []
        _inv_var_names_restore = []
        for _riv_i, _riv in enumerate(_restore_inv_vars):
            _riv_name = str(_riv.get("name") or f"trainable_variable_{_riv_i + 1}").strip() or f"trainable_variable_{_riv_i + 1}"
            try:
                _riv_init = float(_riv.get("init", 1.0))
            except (TypeError, ValueError):
                _riv_init = 1.0
            _inv_var_def_lines.append(f"{_riv_name} = dde.Variable({_riv_init})")
            _inv_var_names_restore.append(_riv_name)
        inv_var_defs_restore = "\n".join(_inv_var_def_lines)
        inv_var_list_restore = "[" + ", ".join(_inv_var_names_restore) + "]" if is_inverse_restore else "None"

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Build minimal geometry for model restore
if {str(is_3d)}:
    geom = dde.geometry.Cuboid([{x_min}, {y_min}, {z_min}], [{x_max}, {y_max}, {z_max}])
elif {str(is_2d)}:
    geom = dde.geometry.Rectangle([{x_min}, {y_min}], [{x_max}, {y_max}])
else:
    geom = dde.geometry.Interval({x_min}, {x_max})
timedomain = dde.geometry.TimeDomain({t_min}, {t_max})
geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y): return y[:, 0:1] * 0

{inv_var_defs_restore}

data = dde.data.TimePDE(geomtime, pde, [], num_domain=100, num_test=100)
net  = dde.nn.FNN({layers}, "{activation}", "Glorot uniform")
model = dde.Model(data, net)

if "{optimizer}" == "lbfgs":
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS", loss="{loss_type}", external_trainable_variables={inv_var_list_restore})
else:
    model.compile("{optimizer}", lr=0.001, loss="{loss_type}", external_trainable_variables={inv_var_list_restore})

if {is_inverse_restore}:
    print("ℹ️ Inverse model: network weights restored for solution plotting. The "
          "trained parameter value itself isn't stored in the checkpoint -- see "
          "the *_convergence.txt file(s) saved alongside the model for that.")

try:
    model.restore(r"{model_path}", verbose=1)
    print("✅ Model restored successfully.")
except Exception as e:
    print(f"❌ Restore error: {{e}}")
    print("Architecture mismatch — make sure network matches saved model.")
    exit(1)

os.makedirs(r"{save_dir}", exist_ok=True)
x_vals = np.linspace({x_min}, {x_max}, 100)
y_vals = np.linspace({y_min}, {y_max}, 100)
z_vals = np.linspace({z_min}, {z_max}, 100)
is_2d  = {str(is_2d)}
is_3d  = {str(is_3d)}
"""

        if viz_type == "Surface":
            vrange = f"vmin={vmin_val}, vmax={vmax_val}" if not auto_range else ""
            _xlabel_3d = xlabel_override or "x"
            _ylabel_3d = ylabel_override or "y"
            _title_3d = title_override or f"Restored Model — {out_name}(x,y,z) at t={surface_time}"
            _xlabel_2d = xlabel_override or "x"
            _ylabel_2d = ylabel_override or "y"
            _title_2d = title_override or f"Restored Model — {out_name}(x,y) at t={surface_time}"
            _xlabel_1d = xlabel_override or "x"
            _ylabel_1d = ylabel_override or "t"
            _title_1d = title_override or f"Restored Model — {out_name}(x,t) Surface"
            script += f"""
res = {resolution}
x_vals = np.linspace({x_min}, {x_max}, res)
y_vals = np.linspace({y_min}, {y_max}, res)
if is_3d:
    # Genuine smooth 3D surface: the restored model is always a Cuboid
    # in this restore/visualize flow, so each of its 6 flat faces (from
    # geom.bbox) is predicted directly on a fine regular grid -- no
    # slicing or interpolation needed, it's an exact prediction at every
    # grid point -- and drawn with plot_surface's per-quad facecolors.
    # Unlike a scatter of discrete points, adjacent same-ish-colored grid
    # quads blend into a continuous-looking colored surface.
    _res3 = max(24, res // 2)
    _bbox3 = np.asarray(geom.bbox)
    _cx0, _cy0, _cz0 = _bbox3[0]; _cx1, _cy1, _cz1 = _bbox3[1]
    _xg3 = np.linspace(_cx0, _cx1, _res3)
    _yg3 = np.linspace(_cy0, _cy1, _res3)
    _zg3 = np.linspace(_cz0, _cz1, _res3)
    _Xxy3, _Yxy3 = np.meshgrid(_xg3, _yg3)   # z-faces (free: x,y)
    _Xxz3, _Zxz3 = np.meshgrid(_xg3, _zg3)   # y-faces (free: x,z)
    _Yyz3, _Zyz3 = np.meshgrid(_yg3, _zg3)   # x-faces (free: y,z)
    _faces3 = [
        (_Xxy3, _Yxy3, np.full_like(_Xxy3, _cz0)),
        (_Xxy3, _Yxy3, np.full_like(_Xxy3, _cz1)),
        (_Xxz3, np.full_like(_Xxz3, _cy0), _Zxz3),
        (_Xxz3, np.full_like(_Xxz3, _cy1), _Zxz3),
        (np.full_like(_Yyz3, _cx0), _Yyz3, _Zyz3),
        (np.full_like(_Yyz3, _cx1), _Yyz3, _Zyz3),
    ]
    _face_preds3 = []
    for _fX3, _fY3, _fZ3 in _faces3:
        _fpts3 = np.column_stack([_fX3.ravel(), _fY3.ravel(), _fZ3.ravel(), np.full(_fX3.size, {surface_time})])
        _face_preds3.append(model.predict(_fpts3)[:, {output_idx}].reshape(_fX3.shape))
    if {auto_range}:
        _pv_min3 = min(_f.min() for _f in _face_preds3)
        _pv_max3 = max(_f.max() for _f in _face_preds3)
    else:
        _pv_min3, _pv_max3 = {vmin_val}, {vmax_val}
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    _norm3 = plt.Normalize(vmin=_pv_min3, vmax=_pv_max3)
    _cmap_obj3 = plt.get_cmap("{colormap}")
    for _fi3, (_fX3, _fY3, _fZ3) in enumerate(_faces3):
        ax.plot_surface(_fX3, _fY3, _fZ3, facecolors=_cmap_obj3(_norm3(_face_preds3[_fi3])),
                         rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    if {show_colorbar}:
        _sm3 = plt.cm.ScalarMappable(cmap=_cmap_obj3, norm=_norm3)
        fig.colorbar(_sm3, ax=ax, shrink=0.6, pad=0.12)
    ax.set_xlabel({_xlabel_3d!r}); ax.set_ylabel({_ylabel_3d!r}); ax.set_zlabel("z")
    ax.set_title({_title_3d!r})
    try:
        ax.set_box_aspect((_cx1 - _cx0, _cy1 - _cy0, _cz1 - _cz0))
    except Exception:
        pass  # older matplotlib without set_box_aspect -- cosmetic only
elif is_2d:
    Xg, Yg = np.meshgrid(x_vals, y_vals)
    XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, {surface_time})])
    pred = model.predict(XYT)[:, {output_idx}].reshape(res, res)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(Xg, Yg, pred, levels={levels}, cmap="{colormap}", {vrange})
    if {show_colorbar}: fig.colorbar(im, ax=ax)
    ax.set_xlabel({_xlabel_2d!r}); ax.set_ylabel({_ylabel_2d!r})
    ax.set_title({_title_2d!r})
else:
    t_vals = np.linspace({t_min}, {t_max}, res)
    X, T = np.meshgrid(x_vals, t_vals)
    XT   = np.vstack([X.ravel(), T.ravel()]).T
    pred = model.predict(XT)[:, {output_idx}].reshape(res, res)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.contourf(X, T, pred, levels={levels}, cmap="{colormap}", {vrange})
    if {show_colorbar}: fig.colorbar(im, ax=ax)
    ax.set_xlabel({_xlabel_1d!r}); ax.set_ylabel({_ylabel_1d!r})
    ax.set_title({_title_1d!r})
plt.tight_layout()
out_path = os.path.join(r"{save_dir}", "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Surface plot saved to: {{out_path}}")
"""
            
        elif viz_type == "Line (time steps)":
            _xlabel_line = xlabel_override or "x"
            _ylabel_line = ylabel_override or out_name
            _title_line = title_override or f"Restored Model — {out_name}(x,t) Line Plot"
            script += f"""
x_vals = np.linspace({x_min}, {x_max}, {resolution})
t_steps_vals = np.linspace({t_min}, {t_max}, {n_steps})
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.get_cmap("{colormap}")(np.linspace(0, 1, {n_steps}))
y_mid = ({y_min} + {y_max}) / 2.0
z_mid = ({z_min} + {z_max}) / 2.0
for i, tv in enumerate(t_steps_vals):
    if is_3d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])
    elif is_2d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])
    else:
        xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    u_line = model.predict(xt)[:, {output_idx}].flatten()
    ax.plot(x_vals, u_line, color=colors[i], linewidth={linewidth}, label=f"t={{tv:.3f}}")
ax.set_xlabel({_xlabel_line!r}); ax.set_ylabel({_ylabel_line!r})
ax.set_title({_title_line!r})
ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
out_path = os.path.join(r"{save_dir}", "restored_plot.png")
plt.savefig(out_path, dpi={dpi}, bbox_inches='tight'); plt.close()
print(f"Line plot saved to: {{out_path}}")
"""
        elif viz_type == "Animation Line (GIF)":
            _xlabel_animline = xlabel_override or "x"
            _ylabel_animline = ylabel_override or out_name
            _title_line_stmt = f"ax.set_title({title_override!r})" if title_override else ""
            script += f"""
import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
y_mid = ({y_min} + {y_max}) / 2.0
z_mid = ({z_min} + {z_max}) / 2.0
all_u = []
for tv in t_frames:
    if is_3d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, z_mid), np.full_like(x_vals, tv)])
    elif is_2d:
        xt = np.column_stack([x_vals, np.full_like(x_vals, y_mid), np.full_like(x_vals, tv)])
    else:
        xt = np.column_stack([x_vals, np.full_like(x_vals, tv)])
    all_u.append(model.predict(xt)[:, {output_idx}].flatten())
u_min = min(u.min() for u in all_u) 
u_max = max(u.max() for u in all_u)
fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim({x_min}, {x_max})
ax.set_ylim(u_min - 0.05*abs(u_min), u_max + 0.05*abs(u_max))
ax.set_xlabel({_xlabel_animline!r}); ax.set_ylabel({_ylabel_animline!r})
{_title_line_stmt}
line, = ax.plot([], [], color="#4dabf7", linewidth=2)
time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='#ff8787')
ax.grid(True, alpha=0.2)
def init():
    line.set_data([], []); time_txt.set_text(''); return line, time_txt
def update(i):
    line.set_data(x_vals, all_u[i])
    time_txt.set_text(f"t = {{t_frames[i]:.3f}}")
    return line, time_txt
ani = _anim.FuncAnimation(fig, update, init_func=init, frames={n_steps}, interval=100, blit=True)
out_path = os.path.join(r"{save_dir}", "restored_animation.gif")
ani.save(out_path, writer='pillow', fps={fps})
plt.close()
print(f"Animation saved to: {{out_path}}")
"""
        
        elif viz_type == "Animation Surface (GIF)":
            _xlabel_animsurf3d = xlabel_override or "x"
            _ylabel_animsurf3d = ylabel_override or "y"
            _animsurf_title_line_3d = (
                f"ax.set_title({title_override!r})" if title_override
                else 'ax.set_title(f"t = {t_frames[i]:.3f}")'
            )
            _xlabel_animsurf_else = xlabel_override or "x"
            _ylabel_animsurf_else = ylabel_override or ("y" if is_2d else "t")
            _animsurf_title_line_else = (
                f"ax.set_title({title_override!r})" if title_override
                else 'ax.set_title(f"t = {t_frames[i]:.3f}")'
            )
            script += f"""
import matplotlib.animation as _anim
t_frames = np.linspace({t_min}, {t_max}, {n_steps})
x_anim = np.linspace({x_min}, {x_max}, 80)
all_frames = []
if is_3d:
    # Genuine smooth 3D surface animated over time: same 6-flat-face
    # exact-prediction approach as the static 3D Surface option above,
    # repeated once per frame (the face grids are fixed across frames,
    # only the predicted color values change per t) and rendered with
    # plot_surface's per-quad facecolors for a continuous-looking colored
    # surface instead of a scatter of discrete points.
    _res3a = 28
    _bbox3a = np.asarray(geom.bbox)
    _cx0a, _cy0a, _cz0a = _bbox3a[0]; _cx1a, _cy1a, _cz1a = _bbox3a[1]
    _xg3a = np.linspace(_cx0a, _cx1a, _res3a)
    _yg3a = np.linspace(_cy0a, _cy1a, _res3a)
    _zg3a = np.linspace(_cz0a, _cz1a, _res3a)
    _Xxy3a, _Yxy3a = np.meshgrid(_xg3a, _yg3a)
    _Xxz3a, _Zxz3a = np.meshgrid(_xg3a, _zg3a)
    _Yyz3a, _Zyz3a = np.meshgrid(_yg3a, _zg3a)
    _faces3a = [
        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz0a)),
        (_Xxy3a, _Yxy3a, np.full_like(_Xxy3a, _cz1a)),
        (_Xxz3a, np.full_like(_Xxz3a, _cy0a), _Zxz3a),
        (_Xxz3a, np.full_like(_Xxz3a, _cy1a), _Zxz3a),
        (np.full_like(_Yyz3a, _cx0a), _Yyz3a, _Zyz3a),
        (np.full_like(_Yyz3a, _cx1a), _Yyz3a, _Zyz3a),
    ]
    for tv in t_frames:
        _frame_faces = []
        for _fX3a, _fY3a, _fZ3a in _faces3a:
            _fpts3a = np.column_stack([_fX3a.ravel(), _fY3a.ravel(), _fZ3a.ravel(), np.full(_fX3a.size, tv)])
            _frame_faces.append(model.predict(_fpts3a)[:, {output_idx}].reshape(_fX3a.shape))
        all_frames.append(_frame_faces)
    v_min = min(_f.min() for _frame in all_frames for _f in _frame)
    v_max = max(_f.max() for _frame in all_frames for _f in _frame)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    _norm3a = plt.Normalize(vmin=v_min, vmax=v_max)
    _cmap_obj3a = plt.get_cmap("{colormap}")
    _sm3a = plt.cm.ScalarMappable(cmap=_cmap_obj3a, norm=_norm3a)
    if {show_colorbar}: fig.colorbar(_sm3a, ax=ax, shrink=0.6, pad=0.12)
    def update(i):
        ax.cla()
        for _fi3a, (_fX3a, _fY3a, _fZ3a) in enumerate(_faces3a):
            ax.plot_surface(_fX3a, _fY3a, _fZ3a, facecolors=_cmap_obj3a(_norm3a(all_frames[i][_fi3a])),
                             rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel({_xlabel_animsurf3d!r}); ax.set_ylabel({_ylabel_animsurf3d!r}); ax.set_zlabel("z")
        {_animsurf_title_line_3d}
        try:
            ax.set_box_aspect((_cx1a - _cx0a, _cy1a - _cy0a, _cz1a - _cz0a))
        except Exception:
            pass
    ani = _anim.FuncAnimation(fig, update, frames={n_steps}, interval=150)
    out_path = os.path.join(r"{save_dir}", "restored_animation.gif")
    ani.save(out_path, writer='pillow', fps={fps})
    plt.close()
    print(f"3D surface animation saved to: {{out_path}}")
else:
    if is_2d:
        y_anim = np.linspace({y_min}, {y_max}, 80)
        Xg, Yg = np.meshgrid(x_anim, y_anim)
        for tv in t_frames:
            XYT = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, tv)])
            pred = model.predict(XYT)[:, {output_idx}].reshape(80, 80)
            all_frames.append((Xg, Yg, pred))
    else:
        t_anim = np.linspace({t_min}, {t_max}, 80)
        X_anim, T_anim = np.meshgrid(x_anim, t_anim)
        for tv in t_frames:
            XT = np.vstack([X_anim.ravel(), np.full(X_anim.size, tv)]).T
            pred = model.predict(XT)[:, {output_idx}].reshape(80, 80)
            all_frames.append((X_anim, T_anim, pred))
    v_min = min(f[2].min() for f in all_frames)
    v_max = max(f[2].max() for f in all_frames)
    fig, ax = plt.subplots(figsize=(8, 6))
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    _div = make_axes_locatable(ax)
    _cax = _div.append_axes("right", size="5%", pad=0.1)
    _sm = plt.cm.ScalarMappable(cmap="{colormap}", norm=plt.Normalize(vmin=v_min, vmax=v_max))
    if {show_colorbar}: fig.colorbar(_sm, cax=_cax)
    def update(i):
        ax.cla()
        Xp, Yp, Zp = all_frames[i]
        ax.contourf(Xp, Yp, Zp, levels={levels}, cmap="{colormap}", vmin=v_min, vmax=v_max)
        ax.set_xlabel({_xlabel_animsurf_else!r})
        ax.set_ylabel({_ylabel_animsurf_else!r})
        {_animsurf_title_line_else}
    ani = _anim.FuncAnimation(fig, update, frames={n_steps}, interval=150)
    out_path = os.path.join(r"{save_dir}", "restored_animation.gif")
    ani.save(out_path, writer='pillow', fps={fps})
    plt.close()
    print(f"Surface animation saved to: {{out_path}}")
"""
        script += '\nprint("RESTORE_DONE")\n'
        return script
    
    def _build_restore_ea_script(self, files, save_dir, is_2d, do_line, do_surface,
                                  x_min, x_max, y_min, y_max, out_name, viz_settings=None):
        if viz_settings is None:
            viz_settings = {}
        _cmap     = viz_settings.get('colormap', 'viridis')
        _levels   = viz_settings.get('levels', 40)
        _dpi      = viz_settings.get('dpi', 150)
        _colorbar = viz_settings.get('colorbar', True)
        _auto     = viz_settings.get('auto_range', True)
        _vmin     = viz_settings.get('vmin', -1.0)
        _vmax     = viz_settings.get('vmax', 1.0)
        files_repr = repr(files)
        return f"""

# ── Restore Error Analysis ────────────────────────────────────
import numpy as np
from scipy.interpolate import interp1d as _interp1d
_ea_dir = os.path.join(r"{save_dir}", "error_analysis")
os.makedirs(_ea_dir, exist_ok=True)
print("\\n=== Running Restore Error Analysis ===")

_ea_files = {files_repr}
_ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_u_refs = []
for _tv, _fp in _ea_files:
    _d = np.loadtxt(_fp)
    if _d.ndim == 1: _d = _d.reshape(1, -1)
    _d_shape = _d.shape[1]
    if {is_2d}:
        _idx = np.lexsort((_d[:, 1], _d[:, 0]))
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(_d[_idx, 1])
        _ea_u_refs.append(_d[_idx, 3])
        _ea_times.append(float(_d[0, 2]))
    else:
        _idx = np.argsort(_d[:, 0])
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(np.zeros_like(_d[_idx, 0]))
        _ea_u_refs.append(_d[_idx, 2])
        _ea_times.append(float(_tv))
    print(f"  Loaded t={{_ea_times[-1]:.4f}}: {{len(_d)}} pts from {{os.path.basename(_fp)}}")

_ea_n_t = len(_ea_times)
_ea_u_pinns = []
for _i, _tv in enumerate(_ea_times):
    _xf = _ea_x_refs[_i]
    if {is_2d}:
        _yf = _ea_y_refs[_i]
        _xt = np.column_stack([_xf, _yf, np.full_like(_xf, _tv)])
    else:
        _xt = np.column_stack([_xf, np.full_like(_xf, _tv)])
    _ea_u_pinns.append(model.predict(_xt)[:, 0].flatten())
    print(f"  Predicted at t={{_tv:.4f}}: {{len(_xf)}} points")

# Metrics
_ea_metrics = []
for _i, _tv in enumerate(_ea_times):
    _up = _ea_u_pinns[_i]; _uf = _ea_u_refs[_i]
    _l2  = np.linalg.norm(_up - _uf) / (np.linalg.norm(_uf) + 1e-10)
    _mse = np.mean((_up - _uf)**2)
    _mx  = np.max(np.abs(_up - _uf))
    _ma  = np.mean(np.abs(_up - _uf))
    _ea_metrics.append((_tv, _l2, _mse, _mx, _ma))
    print(f"  t={{_tv:.4f}} — L2={{_l2:.4e}}, MSE={{_mse:.4e}}, Max={{_mx:.4e}}")

with open(os.path.join(_ea_dir, "error_metrics_restore.txt"), "w") as _mf:
    _mf.write("t,L2_relative,MSE,Max_error,Mean_abs_error\\n")
    for _tv, _l2, _mse, _mx, _ma in _ea_metrics:
        _mf.write(f"{{_tv:.6f}},{{_l2:.6e}},{{_mse:.6e}},{{_mx:.6e}},{{_ma:.6e}}\\n")
print(f"  Metrics saved: {{os.path.join(_ea_dir, 'error_metrics_restore.txt')}}")

# Line comparison
if {do_line}:
    _ncols = min(4, _ea_n_t)
    _nrows = (_ea_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    fig.suptitle("Restored Model vs Ground Truth — Line Comparison", fontsize=13, fontweight='bold')
    _ax_flat = axes.flatten()
    for _i in range(_ea_n_t):
        ax = _ax_flat[_i]
        _xv = _ea_x_refs[_i]
        _tv, _l2, _mse, _mx, _ma = _ea_metrics[_i]
        ax.plot(_xv, _ea_u_refs[_i],  color='#4dabf7', linewidth=2.0, label='Ground Truth')
        ax.plot(_xv, _ea_u_pinns[_i], color='#ff6b6b', linewidth=2.0, linestyle='--', label='PINN')
        ax.set_title(f"t={{_tv:.3f}}  |  L2={{_l2:.2e}}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("{out_name}"); ax.grid(True, alpha=0.3)
    for _j in range(_ea_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    handles, labels = _ax_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _lp = os.path.join(_ea_dir, "line_comparison_restore.png")
    plt.savefig(_lp, dpi={_dpi}, bbox_inches='tight'); plt.close()
    print(f"  Line comparison saved: {{_lp}}")

# Surface comparison
if {do_surface}:
    if {is_2d}:
        from scipy.interpolate import griddata as _gd
        _res_ea = 60
        _xg_ea = np.linspace({x_min}, {x_max}, _res_ea)
        _yg_ea = np.linspace({y_min}, {y_max}, _res_ea)
        _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
        fig, axes = plt.subplots(_ea_n_t, 3, figsize=(15, 4*_ea_n_t), squeeze=False)
        fig.suptitle("Restored Model vs Ground Truth — 2D Heatmaps", fontsize=13, fontweight='bold')
        for _i, _tv in enumerate(_ea_times):
            _tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_i]
            _xyt_g = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _tv)])
            _u_pinn_g = model.predict(_xyt_g)[:, 0].reshape(_res_ea, _res_ea)
            _u_fem_g  = _gd(np.column_stack([_ea_x_refs[_i], _ea_y_refs[_i]]),
                            _ea_u_refs[_i], (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
            _u_err_g  = np.abs(_u_pinn_g - _u_fem_g)
            _vmin_data = min(_u_pinn_g.min(), _u_fem_g.min()) if {_auto} else {_vmin}
            _vmax_data = max(_u_pinn_g.max(), _u_fem_g.max()) if {_auto} else {_vmax}
            im0 = axes[_i][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_g, levels={_levels}, cmap='{_cmap}', vmin=_vmin_data, vmax=_vmax_data)
            axes[_i][0].set_title(f"PINN t={{_tv:.3f}} L2={{_l2:.2e}}"); axes[_i][0].set_xlabel("x"); axes[_i][0].set_ylabel("y")
            if {_colorbar}: fig.colorbar(im0, ax=axes[_i][0])
            im1 = axes[_i][1].contourf(_Xg_ea, _Yg_ea, _u_fem_g, levels={_levels}, cmap='{_cmap}', vmin=_vmin_data, vmax=_vmax_data)
            axes[_i][1].set_title(f"Ground Truth t={{_tv:.3f}}"); axes[_i][1].set_xlabel("x"); axes[_i][1].set_ylabel("y")
            if {_colorbar}: fig.colorbar(im1, ax=axes[_i][1])
            im2 = axes[_i][2].contourf(_Xg_ea, _Yg_ea, _u_err_g, levels={_levels}, cmap='{_cmap}')
            axes[_i][2].set_title(f"|Error| Max={{_mx:.2e}}"); axes[_i][2].set_xlabel("x"); axes[_i][2].set_ylabel("y")
            if {_colorbar}: fig.colorbar(im2, ax=axes[_i][2])
        plt.tight_layout()
    else:
        _x_common = np.linspace({x_min}, {x_max}, 300)
        _t_arr = np.array(_ea_times)
        _U_pinn = np.zeros((len(_t_arr), len(_x_common)))
        _U_fem  = np.zeros((len(_t_arr), len(_x_common)))
        for _i, _tv in enumerate(_ea_times):
            _xt_c = np.column_stack([_x_common, np.full_like(_x_common, _tv)])
            _U_pinn[_i] = model.predict(_xt_c)[:, 0].flatten()
            _fi = _interp1d(_ea_x_refs[_i], _ea_u_refs[_i], kind='linear', fill_value='extrapolate')
            _U_fem[_i]  = _fi(_x_common)
        _Xg, _Tg = np.meshgrid(_x_common, _t_arr)
        _U_err = np.abs(_U_pinn - _U_fem)
        _vmin_data = min(_U_pinn.min(), _U_fem.min()) if {_auto} else {_vmin}
        _vmax_data = max(_U_pinn.max(), _U_fem.max()) if {_auto} else {_vmax}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Restored Model vs Ground Truth — Surface", fontsize=13, fontweight='bold')
        im0 = axes[0].contourf(_Tg, _Xg, _U_pinn, levels={_levels}, cmap='{_cmap}', vmin=_vmin_data, vmax=_vmax_data)
        axes[0].set_title("PINN"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
        if {_colorbar}: fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].contourf(_Tg, _Xg, _U_fem, levels={_levels}, cmap='{_cmap}', vmin=_vmin_data, vmax=_vmax_data)
        axes[1].set_title("Ground Truth"); axes[1].set_xlabel("t")
        if {_colorbar}: fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].contourf(_Tg, _Xg, _U_err, levels={_levels}, cmap='{_cmap}')
        axes[2].set_title("|Error|"); axes[2].set_xlabel("t")
        if {_colorbar}: fig.colorbar(im2, ax=axes[2])
        plt.tight_layout()
    _sp = os.path.join(_ea_dir, "surface_comparison_restore.png")
    plt.savefig(_sp, dpi={_dpi}, bbox_inches='tight'); plt.close()
    print(f"  Surface comparison saved: {{_sp}}")

print("=== Restore Error Analysis Complete ===")
"""
