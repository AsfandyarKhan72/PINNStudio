import os
import math
import re as _re_log
_re_log_l2 = _re_log.compile(r"(?<![A-Za-z0-9_])L2(?![A-Za-z0-9_])")
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QTextEdit, QGroupBox, QComboBox, QSplitter, QLineEdit,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QDialog, QApplication, QColorDialog,
    QTabWidget, QMessageBox, QScrollArea, QGridLayout, QPlainTextEdit, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QAction, QColor, QMovie, QIntValidator, QKeySequence
from pinnstudio.core.config import PINNConfig
from pinnstudio.core.runner import run_pinn
from pinnstudio.core import deepxde_examples as _dx_examples
from pinnstudio.core import example_catalog as _ex_catalog
from pinnstudio.core.config import LBFGS_RECOMMENDED
from pinnstudio.ui.sketch_canvas import SketchPanel
from pinnstudio.ui import theme as _theme
REFERENCE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reference_data",
)


class ExampleCombo(QComboBox):
    """Quick Examples list: shows each example's name
    ("Burgers 1D – RAR") under dimension headers, while
    currentText()/setCurrentText() and keyChanged work with the example's
    internal key ("Burgers equation with RAR", "1D Burgers", or "None") --
    the name the rest of the GUI, saved projects and tests use."""
    keyChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentIndexChanged.connect(self._emit_key)

    def set_entries(self, entries):
        """Entries with `hidden` set (templates a DeepXDE example replaces)
        go in as hidden, disabled rows at the end: not shown or reachable
        from the list, but setCurrentText() still loads them by name, so
        saved projects and scripts that name them keep working."""
        self.blockSignals(True)
        self.clear()
        self.addItem("None", "None")
        last = None
        shown = [e for e in entries if not getattr(e, "hidden", False)]
        for e in shown:
            if e.dim != last:
                self.addItem(f"── {e.dim} ──", None)
                item = self.model().item(self.count() - 1)
                item.setEnabled(False)
                last = e.dim
            self.addItem(e.label, e.key)
        self._hidden_keys = []
        for e in entries:
            if getattr(e, "hidden", False):
                self.addItem(e.label, e.key)
                row = self.count() - 1
                self.model().item(row).setEnabled(False)
                self.view().setRowHidden(row, True)
                self._hidden_keys.append(e.key)
        self.setCurrentIndex(0)
        self.blockSignals(False)

    def keys(self):
        """The examples listed (hidden ones left out)."""
        hidden = getattr(self, "_hidden_keys", [])
        return [self.itemData(i) for i in range(self.count()) if self.itemData(i) and self.itemData(i) not in hidden]

    def currentText(self):
        k = self.currentData()
        return k if isinstance(k, str) else super().currentText()

    def setCurrentText(self, text):
        i = self.findData(text)
        if i < 0:
            i = self.findText(text)
        if i >= 0:
            self.setCurrentIndex(i)

    def _emit_key(self, _i):
        k = self.currentData()
        if isinstance(k, str):
            self.keyChanged.emit(k)


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


class DomainSpinBox(QDoubleSpinBox):
    """Domain bound spinbox that keeps 10 decimals (so pi, pi/2, 2 pi are
    stored to ~1e-10, not rounded to 3.14) but shows only the digits a
    value actually has: 1 shows as "1", 0.5 as "0.5". Simple multiples of
    pi are shown as such ("π/2", "π", "2π", "-3π/4") and can be typed that
    way too ("pi/2", "2*pi", "π/2", or a plain number)."""
    DECIMALS = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDecimals(self.DECIMALS)

    @staticmethod
    def _pi_text(v):
        import math
        if v == 0 or abs(v - round(v)) < 1e-12:
            return None                    # whole numbers stay numbers
        r = v / math.pi
        for den in (1, 2, 3, 4, 6, 8):
            num = round(r * den)
            if num and abs(r * den - num) < 1e-8 and math.gcd(num, den) == 1:
                sign = "-" if num < 0 else ""
                n = abs(num)
                return f"{sign}{'' if n == 1 else n}π" + ("" if den == 1 else f"/{den}")
        return None

    def textFromValue(self, v):
        t = self._pi_text(v)
        if t:
            return t
        s = f"{v:.{self.DECIMALS}f}".rstrip("0").rstrip(".")
        return "0" if s in ("", "-0") else s

    @staticmethod
    def _parse(text):
        import math
        import re as _re
        t = (text or "").strip().replace("π", "pi").replace(" ", "")
        if not t:
            return None
        if not _re.fullmatch(r"[0-9.eE+\-*/()pi]+", t):
            return None
        t = _re.sub(r"(\d)(pi)", r"\1*\2", t)      # "2pi" -> "2*pi"
        try:
            return float(eval(t, {"__builtins__": {}}, {"pi": math.pi}))
        except Exception:
            return None

    def valueFromText(self, text):
        v = self._parse(text)
        return self.value() if v is None else v

    def validate(self, text, pos):
        from PyQt6.QtGui import QValidator
        if self._parse(text) is not None:
            return (QValidator.State.Acceptable, text, pos)
        import re as _re
        if _re.fullmatch(r"[0-9.eE+\-*/() πpi]*", text or ""):
            return (QValidator.State.Intermediate, text, pos)
        return (QValidator.State.Invalid, text, pos)


class OutputSpinBox(QSpinBox):
    """An output index shown to the user counting from 1 ("Output 1" is the
    first output, as in the PDE Definition panel), while value() stays the
    0-based index the configs and generated scripts use."""

    def textFromValue(self, v):
        return str(v + 1)

    def valueFromText(self, text):
        try:
            return int(text) - 1
        except ValueError:
            return self.value()


def _components_display(text):
    """A PointSet BC's 0-based "Components" field (e.g. "0,1"), shown
    counting from 1 ("1,2"). Text that isn't a plain list of integers is
    left as it is."""
    try:
        parts = [p.strip() for p in (text or "").split(",") if p.strip()]
        return ",".join(str(int(p) + 1) for p in parts)
    except ValueError:
        return text


def _components_stored(text):
    """Inverse of _components_display: the field's 1-based numbers back to
    the 0-based indices stored in the config."""
    try:
        parts = [p.strip() for p in (text or "").split(",") if p.strip()]
        return ",".join(str(int(p) - 1) for p in parts)
    except ValueError:
        return text


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


class TrueValueLineEdit(InitGuessLineEdit):
    """InitGuessLineEdit variant for the optional known ground-truth value
    of an Inverse trainable variable. Unlike the initial guess (always a
    number), this box may be left BLANK to mean "no known true value" --
    e.g. for a manually added/custom variable with no known answer, in
    which case codegen.py's parameter-convergence plot simply skips
    drawing a dashed True= reference line for it. Blank is a valid,
    first-class state here (not an error): only non-blank text that fails
    to parse as a float gets the red-border treatment SciLineEdit uses
    for invalid input."""
    def __init__(self, value=None, parent=None):
        QLineEdit.__init__(self, parent)
        self._value = None if value is None else float(value)
        self.setText("" if value is None else self._format(value))
        self.editingFinished.connect(self._on_edited)
        self.setFixedHeight(28)

    def _on_edited(self):
        text = self.text().strip()
        if not text:
            self._value = None
            self.setStyleSheet("")
            return
        try:
            self._value = float(text)
            self.setText(self._format(self._value))
            self.setStyleSheet("")
        except ValueError:
            self.setStyleSheet("border: 1px solid red;")

    def value(self):
        text = self.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return self._value

    def setValue(self, v):
        if v is None:
            self._value = None
            self.setText("")
            self.setStyleSheet("")
            return
        self._value = float(v)
        self.setText(self._format(v))
        self.setStyleSheet("")


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
        # terminate() (SIGTERM) alone can leave Stop/close hanging
        # indefinitely if the child is blocked inside a CUDA/PyTorch call
        # and doesn't respond promptly -- runner.py's `for line in
        # process.stdout:` loop keeps blocking until the pipe actually
        # closes, which only happens once the child exits. Waiting a short,
        # bounded amount of time and then kill()ing (SIGKILL, which the
        # child cannot ignore or delay) caps that hang instead of leaving
        # it unbounded; a brief block here (up to a few seconds, only when
        # the child doesn't exit promptly on its own) is the deliberate
        # trade-off for a Cancel/close action that must not hang forever.
        if self.process and self.process.poll() is None:
            import subprocess
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class SweepThread(QThread):
    """Runs a whole Parameter Sweep (pinnstudio.core.sweep_runner.run_sweep)
    on a background thread, the same way SolverThread runs a single Solve
    -- so the sweep's sequence of real training subprocesses never blocks
    the GUI event loop. Stop/Cancel now hard-stops exactly the way
    SolverThread.stop() already does for a normal Solve: the CURRENTLY
    RUNNING training subprocess is terminated (and killed if it doesn't
    exit promptly) immediately, not allowed to keep training to
    completion -- an earlier version of this class only set a flag
    checked between runs, which looked like Stop wasn't doing anything
    (the in-flight run kept printing iterations for a while longer)."""
    sweep_root_signal = pyqtSignal(str)
    run_start_signal = pyqtSignal(int, int, str)
    output_signal = pyqtSignal(int, int, str)
    run_done_signal = pyqtSignal(int, int, str, dict)
    finished_signal = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop_requested = False
        self.process = None

    def run(self):
        from pinnstudio.core.sweep_runner import run_sweep
        run_sweep(
            self.config,
            on_sweep_root=lambda root: self.sweep_root_signal.emit(root),
            on_run_start=lambda i, total, label: self.run_start_signal.emit(i, total, label),
            on_output=lambda i, total, line: self.output_signal.emit(i, total, line),
            on_run_done=lambda i, total, label, result: self.run_done_signal.emit(i, total, label, result),
            should_stop=lambda: self._stop_requested,
            set_process=self._set_process,
        )
        self.finished_signal.emit()

    def _set_process(self, proc):
        self.process = proc

    def stop(self):
        # Mark should_stop() True first so run_sweep()'s loop won't start
        # another run once the current one is forced to exit below, then
        # hard-kill whichever subprocess is in flight right now -- same
        # terminate()-then-kill() approach, and the same deliberate
        # bounded-wait trade-off, as SolverThread.stop() above.
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            import subprocess
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


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
# Every font-family CSS declaration below names a single font (default
# "Segoe UI", or whatever the user picks in Display Settings -> _DISP_FONT_
# CHOICES, itself a Windows-centric list). "Segoe UI" only actually exists
# on Windows; on macOS/Linux, Qt has to substitute *some* font for it, and
# with no fallback names given in these per-widget stylesheets (unlike the
# base QWidget stylesheet a few lines below, which did already list
# 'Arial' as a second choice), that substitution picks whatever generic
# sans-serif Qt's font matcher lands on -- which can render noticeably
# wider/larger than "Segoe UI" does on Windows. Since the left control
# panel packs many fixed pixel-width fields/labels into single rows, wider
# substituted text pushes up that panel's actual minimum content width,
# which in turn is what a QSplitter honors as how far its handle can be
# dragged -- so a font substitution difference on macOS/Linux can plausibly
# show up as "the app looks bigger" and "the divider won't drag as far
# right" on those platforms, without either being a deliberate size limit
# anywhere in this file (there is no setMaximumWidth/setMaximumSize on any
# of these widgets or splitters). Appending this fallback chain after
# whatever single family name is configured keeps Windows' exact behavior
# unchanged (that name still resolves first there) while giving macOS/
# Linux native-appropriate fonts to fall through to instead of an
# unpredictable Qt-chosen substitute, ending in the generic 'sans-serif'
# keyword as a final catch-all.
_CROSS_PLATFORM_FONT_FALLBACK = (
    "'Helvetica Neue', 'Helvetica', 'Ubuntu', 'Noto Sans', 'DejaVu Sans', "
    "'Arial', sans-serif"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINNStudio — A No-Code Scientific Computing Environment for Forward and Inverse Physics-Informed Neural Networks")
        self.setMinimumSize(1100, 750)
        try:
            from pinnstudio.ui import icons as _icons_app
            self.setWindowIcon(_icons_app.app_icon())
        except Exception:
            pass
        # Default launch size: previously there was no explicit resize()
        # call anywhere (main.py just does MainWindow().show()), so Qt fell
        # back to the layout's own computed sizeHint(), which lands very
        # close to the 1100x750 minimum above -- the user had to manually
        # stretch the window every time just to get a usable amount of
        # room for the left configuration panel and the plot area. This is
        # only an initial default: the user can still freely resize the
        # window and drag both splitters (see their own setSizes() calls
        # below) afterward exactly as before.
        self.resize(1650, 950)
        # Every stylesheet a widget sets is translated to the current theme's
        # colours (see pinnstudio/ui/theme.py).
        _theme.StyleRemapper.install()
        self._font_size = 14          # base UI text, px (before the UI scale)
        self._log_font_size = 13
        self._ui_scale = 100          # percent
        self._theme = _theme.DEFAULT_THEME
        self._accent = "Theme default"
        # Panel border (the outline drawn around each Setup-tab section and
        # the loss/solution figure panels). '' color = keep the existing
        # theme-derived default exactly as before; width 1 matches the
        # hardcoded "1px solid" every panel always used before this setting
        # existed, so the default look is unchanged.
        self._panel_border_color = ""
        self._panel_border_width = 1
        self._float_type = "float32"
        self._gpu_device_index = 0
        self._gpu_memory_fraction = 0.95
        self._use_random_seed = True
        self._random_seed = 2026
        # ── Per-category display settings (font family/size/weight/color for
        # groups of related text: the app title, section headers, field
        # labels, small hint/note text, buttons, and the log console). Each
        # category's 'color' is '' by default, meaning "keep that widget's
        # own original color" -- setting one explicitly overrides it
        # uniformly across the category. self._styled_widgets is the
        # registry of (widget, category, style_fn) used to re-apply these
        # live whenever settings change, without needing to rebuild the UI.
        # family "" = the system's own UI font (fixed-width for the log);
        # color "" = the theme's colour for that kind of text.
        self._DISP_CATEGORY_DEFAULTS = {
            "title":          {"family": "", "size": 20, "bold": True,  "color": ""},
            "section_header": {"family": "", "size": 14, "bold": True,  "color": ""},
            "field_label":    {"family": "", "size": 14, "bold": False, "color": ""},
            "hint":           {"family": "", "size": 13, "bold": False, "color": ""},
            "button":         {"family": "", "size": 14, "bold": False, "color": ""},
            "log_console":    {"family": "", "size": 13, "bold": False, "color": ""},
        }
        self._disp = {k: dict(v) for k, v in self._DISP_CATEGORY_DEFAULTS.items()}
        self._styled_widgets = []
        self._display_settings_path = os.path.join(
            os.path.expanduser("~"), ".pinnstudio", "display_settings.json")
        self._load_display_settings_from_disk()

        # ── Optimizer settings ──────────────────────────────────
        # Same pattern as self._plot_viz_settings: a plain dict edited via a
        # menu-launched dialog (Settings > Optimizer Settings), read
        # directly by _build_config(). weight_decay applies to whichever
        # network is trained (see codegen.py) -- it is 0.0 (off) by
        # default, matching all pre-existing behavior exactly.
        self._optimizer_settings = {
            "weight_decay": 0.0,
        }
        # Training Callbacks (Early Stopping/Point Resampling/Model
        # Checkpoint/Timer) used to be a similar dialog-backed dict here;
        # they're now inline widgets in the Training panel itself (built in
        # _build_ui, read directly by _build_config() -- see the "Training
        # Callbacks" section of train_group's construction), all unchecked
        # by default exactly as before.
        self._apply_theme()
        self._build_ui()
        self._apply_display_settings()
        self._check_for_updates()
        # The new top-level QTabWidget (central_tabs, added for Parameter
        # Sweep) has been reported built correctly but with its tab bar
        # not actually painted on first show on some real displays --
        # the exact same class of "built but not painted until something
        # nudges a repaint" Qt quirk pinnstudio/main.py's own comment
        # already documents for PyTorch's import ordering (unrelated
        # widgets, same underlying cause). A deferred hide/show cycle on
        # just the tab bar is a cheap, reliable nudge; harmless if the
        # platform never needed it.
        QTimer.singleShot(0, self._nudge_central_tabs_repaint)
        # Open on the Start tab; the problem as it is now is "unchanged"
        self.right_tabs.setCurrentIndex(self._start_tab_index)
        self._current_path = None
        self._refresh_recent()
        self._fresh_config = None
        self._clean_key = None
        # Some defaults (training phases, sweep rows) are filled in on the
        # first passes of the event loop: the starting problem -- what New
        # gives and what counts as "unchanged" -- is taken after them.
        QTimer.singleShot(0, lambda: QTimer.singleShot(0, self._capture_fresh_problem))

    def _nudge_central_tabs_repaint(self):
        bar = self.central_tabs.tabBar()
        bar.hide()
        # Only re-show it if there's more than one tab -- with a single
        # "Setup" tab (the normal case now that Parameter Sweep lives
        # inline instead of as its own tab), the bar is meant to stay
        # hidden permanently (see its setVisible(False) where the tab
        # is added), and this nudge shouldn't undo that.
        if self.central_tabs.count() > 1:
            bar.show()

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
        if int(data.get("style_version", 1) or 1) < 2:
            # Settings from before the 2026 restyle (18 px fonts, the old
            # theme list): start from the new defaults instead.
            return
        try:
            self._ui_scale = int(data.get("ui_scale", self._ui_scale))
            self._font_size = int(data.get("font_size", self._font_size))
            self._log_font_size = int(data.get("log_font_size", self._log_font_size))
            self._theme = data.get("theme", self._theme)
            self._accent = data.get("accent", self._accent)
            self._panel_border_color = str(data.get("panel_border_color", self._panel_border_color))
            self._panel_border_width = int(data.get("panel_border_width", self._panel_border_width))
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
                "style_version": 2,
                "ui_scale": self._ui_scale,
                "font_size": self._font_size,
                "log_font_size": self._log_font_size,
                "theme": self._theme,
                "accent": self._accent,
                "panel_border_color": self._panel_border_color,
                "panel_border_width": self._panel_border_width,
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
        family = cat.get("family") or (_theme.mono_font_family() if category == "log_console"
                                       else _theme.ui_font_family())
        size = self._scaled(cat.get("size") or defaults.get("size", 14))
        weight = "600" if cat.get("bold") else "normal"
        parts = [f"font-family: '{family}', {_CROSS_PLATFORM_FONT_FALLBACK};",
                 f"font-size: {size}px;", f"font-weight: {weight};"]
        color = (cat.get("color") or "").strip()
        if color:
            parts.append(f"color: {color};")
        return " ".join(parts)

    def _scaled(self, px):
        return max(8, int(round(px * getattr(self, '_ui_scale', 100) / 100.0)))

    @staticmethod
    def _add_ok_cancel_row(layout, dialog=None):
        """cleanup Round 51 (audit §4): the right-aligned OK/Cancel footer
        hand-typed in nine settings dialogs. Adds it to `layout` and returns
        (ok_btn, cancel_btn); when `dialog` is given, Cancel rejects it
        (pass None for a dialog with its own cancel handler)."""
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK"); cancel_btn = QPushButton("Cancel")
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        if dialog is not None:
            cancel_btn.clicked.connect(dialog.reject)
        return ok_btn, cancel_btn

    def _register_hint(self, widget, color, extra=""):
        """cleanup Round 51 (audit §4): the "hint" text style, registered
        with one fixed color (and optional extra CSS) -- this exact
        _register_style(widget, "hint", lambda css, _c=..., _e=...: ...)
        call used to be hand-typed at 58 call sites. Same resulting
        stylesheet ("color: <color>; <extra><Display Settings css>")."""
        # Informational hints read as quiet text; warning / error / success
        # colours keep their meaning (theme.remap_css turns them into the
        # theme's own).
        if str(color).lower() in ("#a0c4ff", "#74c0fc", "#4dabf7", "#8a8a8a", "#868e96", "#999999", "#adb5bd"):
            color = "#868e96"
        return self._register_style(widget, "hint", lambda css, _c=color, _e=extra: f"color: {_c}; {_e}{css}")

    def _register_section_divider(self, widget, extra=""):
        """A "── Title ──" divider that starts a part of a panel (Training
        Phases, Training Callbacks, Output 1 (u), Shared weights, ...): bold,
        one style everywhere. Item headers inside a part ("── Phase 1 ──",
        "── BC 1 ──", "── Parameter 1 ──") stay regular weight."""
        # Bold is set after the Display Settings fragment (which carries its
        # own font-weight), so it holds whatever hint style is chosen there.
        return self._register_style(
            widget, "hint",
            lambda css, _e=extra: f"color: #8d9fd0; {_e}{css} font-weight: bold;")

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
        self._apply_display_settings()

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

        # ── Menu bar ─────────────────────────────────────────
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        problem_menu = menubar.addMenu("Problem")
        new_menu = problem_menu.addMenu("New")
        for _dim, _name, _meta in (("1D", "1D problem", ""), ("2D", "2D problem", ""), ("3D", "3D problem", ""),
                                   ("ODE", "ODE system", "")):
            _a = QAction(_name, self)
            _a.triggered.connect(lambda _c=False, d=_dim: self._new_problem(d))
            new_menu.addAction(_a)
        open_problem_action = QAction("Open Saved Problem...", self)
        open_problem_action.setShortcut(QKeySequence.StandardKey.Open)
        open_problem_action.triggered.connect(lambda: self._open_problem())
        problem_menu.addAction(open_problem_action)
        self.recent_menu = problem_menu.addMenu("Open Recent")
        problem_menu.addSeparator()
        save_action = QAction("Save Problem", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(lambda: self._save_problem())
        problem_menu.addAction(save_action)
        save_problem_action = QAction("Save Problem As...", self)
        save_problem_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_problem_action.triggered.connect(lambda: self._save_problem(save_as=True))
        problem_menu.addAction(save_problem_action)
        export_script_action = QAction("Export as DeepXDE Script...", self)
        export_script_action.setShortcut(QKeySequence("Ctrl+E"))
        export_script_action.triggered.connect(self._export_deepxde_script)
        problem_menu.addAction(export_script_action)
        problem_menu.addSeparator()
        solve_action = QAction("Solve", self)
        solve_action.setShortcuts([QKeySequence("Ctrl+Return"), QKeySequence("F5")])
        solve_action.triggered.connect(self._shortcut_solve)
        problem_menu.addAction(solve_action)
        stop_action = QAction("Stop", self)
        stop_action.setShortcut(QKeySequence("Shift+F5"))
        stop_action.triggered.connect(self._shortcut_stop)
        problem_menu.addAction(stop_action)
        self.solve_action, self.stop_action = solve_action, stop_action
        settings_menu = menubar.addMenu("Settings")
        lbfgs_action = QAction("L-BFGS Options", self)
        lbfgs_action.triggered.connect(self._on_lbfgs_settings)
        settings_menu.addAction(lbfgs_action)

        float_action = QAction("Float Precision...", self)
        float_action.triggered.connect(self._on_float_settings)
        settings_menu.addAction(float_action)

        optimizer_settings_action = QAction("Optimizer Settings...", self)
        optimizer_settings_action.triggered.connect(self._on_optimizer_settings)
        settings_menu.addAction(optimizer_settings_action)

        gpu_settings_action = QAction("GPU / Hardware...", self)
        gpu_settings_action.triggered.connect(self._on_gpu_settings)
        settings_menu.addAction(gpu_settings_action)

        seed_settings_action = QAction("Random Seed...", self)
        seed_settings_action.triggered.connect(self._on_seed_settings)
        settings_menu.addAction(seed_settings_action)

        # Training Callbacks used to live here as its own dialog; it's now
        # an inline section of the Training panel itself (see train_group's
        # construction in _build_ui), alongside IC Pre-Training/Training
        # Phases/L-BFGS, so it's visible and editable without a popup.

        settings_menu.addSeparator()
        display_action = QAction("🎨 Display Settings...", self)
        display_action.triggered.connect(self._on_display_settings)
        settings_menu.addAction(display_action)

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
        # Toolbar (filled in _build_workspace, once its buttons exist)
        self.app_toolbar = QWidget()
        self.app_toolbar.setObjectName("appToolbar")
        outer_layout.addWidget(self.app_toolbar)

        # Single top-level tab ("Setup") -- Parameter Sweep used to be a
        # second tab here, but now lives inline inside the left Training
        # panel instead (an Enable toggle plus the parameter list itself,
        # right after Adaptive Training -- see _add_sweep_row() and the
        # "Parameter Sweep" group built alongside it), reusing the main
        # Solve/Stop buttons and Training Log rather than any dedicated
        # tab or results section, so everything about training, sweep
        # included, is on one screen. QTabWidget is kept (rather than
        # ripping it out) since central_tabs is referenced elsewhere and
        # this way nothing else has to change -- its tab bar is just
        # hidden since a single-tab bar has nothing useful to show.
        self.central_tabs = QTabWidget()
        outer_layout.addWidget(self.central_tabs)

        setup_tab = QWidget()
        root = QHBoxLayout(setup_tab)
        root.setContentsMargins(0, 0, 0, 0)
        self.central_tabs.addTab(setup_tab, "Setup")
        self.central_tabs.tabBar().setVisible(False)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Qt's default for a QSplitter is childrenCollapsible=True: once a
        # drag would shrink either pane below its minimum size, Qt snaps
        # that pane fully closed (0 width) instead of just stopping the
        # drag at the minimum -- so a normal, gentle resize attempt that
        # happens to cross that boundary looks like the panel "jumping" to
        # one extreme, with no way to land on an in-between width. Disabling
        # it makes the handle behave like an ordinary resize: it stops
        # smoothly at each side's minimum size instead of collapsing past
        # it, all the way down to a fully intermediate position.
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        # ── Panels ───────────────────────────────────────────
        # Each panel is built by its own method below, into a holding
        # column; _build_workspace then moves them onto the pages of the
        # model tree (tree | settings | graphics, after COMSOL).
        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(10, 8, 10, 8)

        # Title -- styled from the "title" Display Settings category (family/
        # size/weight/color); it sits at the left of the toolbar.
        title = QLabel("PINNStudio")
        self._register_style(title, "title", lambda css: f"{css}")
        self.app_title_label = title
        left_layout.addWidget(title)

        self._build_setup_panels(left_layout)
        self._build_domain_panels(left_layout)
        self._build_network_panel(left_layout)
        self._build_training_panel(left_layout)
        self._build_adaptive_and_sweep_panels(left_layout)
        self._build_inverse_panel(left_layout)
        self._build_solve_buttons(left_layout)
        self._build_restore_panel(left_layout)

        self._build_workspace(splitter, left_layout)
        self._build_right_panel(splitter)
        self._finish_workspace(splitter)
        self._build_status_bar()

    # ── Workspace: model tree | settings | graphics ──────────
    # (key, label, icon glyph, icon colour, parent, panel titles, description)
    _WORKSPACE_NODES = (
        ("problem", "Problem", "P", "#4dabf7", None, ("Problem Dimension", "Problem Type"),
         "What kind of problem this is: dimension, time dependence, forward or inverse."),
        ("equations", "Equations", "∂", "#b197fc", None, ("PDE Definition",),
         "The outputs, the equations (residual = 0) and named constants."),
        ("inverse", "Inverse problem", "?", "#ffa94d", None, ("Inverse PINN Settings",),
         "The unknown parameters to learn and the measured data to fit."),
        ("geometry", "Geometry", "G", "#51cf66", None, ("Domain",),
         "The domain: a basic shape, a combination of shapes, or a sketch -- and the time interval."),
        ("points", "Collocation points", "⁙", "#38d9a9", None, ("Collocation Points",),
         "How many training points are sampled in the domain, on the boundary and at the initial time."),
        ("ic", "Initial conditions", "t", "#ffd43b", None, ("Initial Condition",),
         "The outputs' values at the initial time."),
        ("bcs", "Boundary conditions", "B", "#ff8787", None, ("Boundary Conditions",),
         "Conditions on the boundary. Where: picks the part of the boundary each one applies to."),
        ("network", "Neural network", "N", "#63e6be", None, ("Neural Network",),
         "Network size and type, activation, and input/output transforms."),
        ("training", "Training", "T", "#74c0fc", None, ("Training",),
         "Optimizers and training phases, initial-condition pre-training, callbacks."),
        ("weights", "Loss weights", "W", "#91a7ff", "training", ("Loss Weights",),
         "How much each loss term counts."),
        ("adaptive", "Adaptive training", "A", "#91a7ff", "training", ("Adaptive Training",),
         "Residual-based refinement, point resampling and time-adaptive training."),
        ("sweep", "Parameter sweep", "S", "#e599f7", None, ("Parameter Sweep",),
         "Run the problem over a range of settings and compare the results."),
        ("results", "Results", "R", "#fcc419", None, (),
         "What to plot and save, error analysis, and solution export."),
        ("restore", "Restore & visualize", "↺", "#ffe066", "results", ("Model Restore && Visualization",),
         "Load a trained model and plot it again, without retraining."),
    )

    def _build_workspace(self, splitter, left_layout):
        """Move the panels built into `left_layout` onto the model tree's
        pages, and the title, Quick Examples and Solve/Stop into the toolbar.
        Adds the tree and settings panels to `splitter`."""
        from pinnstudio.ui.workspace import ModelWorkspace
        groups = {}
        while left_layout.count():
            it = left_layout.takeAt(0)
            w = it.widget()
            if isinstance(w, QGroupBox):
                groups[w.title()] = w
        self.workspace = ws = ModelWorkspace(self)
        for key, label, glyph, color, parent, titles, desc in self._WORKSPACE_NODES:
            missing = [t for t in titles if t not in groups]
            assert not missing, f"workspace: panels not found: {missing}"
            ws.add_node(key, label, glyph, color, desc, parent, [groups[t] for t in titles])
        self._quick_examples_group = groups.get("📋 Quick Examples")
        self._sync_bc_nodes()
        splitter.addWidget(ws.tree_panel)
        splitter.addWidget(ws.settings_panel)

    def _finish_workspace(self, splitter):
        """After the graphics panel exists: the Results page, the toolbar,
        initial sizes and the saved layout."""
        ws = self.workspace
        ws.add_widget("results", self.results_controls)
        self._split_geometry_page()
        self._split_training_page()
        # The sketch's "selected edges -> BC" controls sit on the Boundary
        # conditions page, shown while the sketch is picking edges.
        from pinnstudio.ui.workspace import VisibilityFollower
        self.sketch_selection_box = QGroupBox("Edges picked in the sketch")
        _sel_lay = QVBoxLayout(self.sketch_selection_box)
        _sel_lay.addWidget(self.sketch_panel.edge_bar)
        _sel_lay.addWidget(self.domain_view.edge_bar)
        ws.add_widget("bcs", self.sketch_selection_box)
        bcs_lay = ws.pages["bcs"]["layout"]
        bcs_lay.insertWidget(0, bcs_lay.takeAt(bcs_lay.indexOf(self.sketch_selection_box)).widget())
        self._sketch_sel_follower = VisibilityFollower(self.sketch_selection_box,
                                                       [self.sketch_panel.edge_bar, self.domain_view.edge_bar])
        self.domain_view.edge_bar.setVisible(False)

        # ── toolbar ──
        bar = QHBoxLayout(self.app_toolbar)
        bar.setContentsMargins(12, 6, 12, 6)
        bar.setSpacing(8)
        bar.addWidget(self.app_title_label)
        bar.addSpacing(18)
        ex_label = QLabel("Examples:")
        ex_label.setObjectName("toolbarLabel")
        bar.addWidget(ex_label)
        self.quick_examples_category_combo.setFixedWidth(120)
        bar.addWidget(self.quick_examples_category_combo)
        self.quick_examples_combo.setMinimumWidth(260)
        self.quick_examples_combo.setMaximumWidth(380)
        bar.addWidget(self.quick_examples_combo, 1)
        if self._quick_examples_group is not None:
            self._quick_examples_group.hide()
        bar.addStretch(1)

        def tool_btn(text, tip, slot):
            b = QPushButton(text)
            b.setObjectName("toolButton")
            b.setToolTip(tip)
            b.setFixedHeight(32)
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        self.problems_btn = QPushButton("")
        self.problems_btn.setObjectName("problemsButton")
        self.problems_btn.setFixedHeight(32)
        self.problems_btn.setVisible(False)
        self.problems_btn.setToolTip("Settings that would stop Solve -- click to go to the first one")
        self.problems_btn.clicked.connect(self._goto_first_problem)
        bar.addWidget(self.problems_btn)
        bar.addSpacing(6)
        self.tb_open_btn = tool_btn("Open", "Open a saved problem (Ctrl+O)", lambda: self._open_problem())
        self.tb_save_btn = tool_btn("Save", "Save this problem (Ctrl+S)", lambda: self._save_problem())
        self.tb_export_btn = tool_btn("Export script", "Export as a DeepXDE script (Ctrl+E)",
                                      self._export_deepxde_script)
        from pinnstudio.ui import icons as _icons
        for _b, _n in ((self.tb_open_btn, "open"), (self.tb_save_btn, "save"), (self.tb_export_btn, "export")):
            _icons.apply(_b, _n, "text")
        _icons.apply(self.solve_btn, "play", "on_accent", size=14)
        _icons.apply(self.stop_btn, "stop", "on_danger", size=14)
        _icons.apply(self.problems_btn, "warning", "text", size=16)
        _icons.apply((self.right_tabs, self._results_tab_index), "results", "muted", size=16)
        _icons.apply((self.right_tabs, self._sketch_tab_index), "sketch", "muted", size=16)
        _icons.apply((self.right_tabs, self._domain_tab_index), "geometry", "muted", size=16)
        _icons.apply((self.bottom_tabs, 0), "log", "muted", size=16)
        _icons.apply((self.bottom_tabs, self._script_tab_index), "code", "muted", size=16)
        _icons.apply((self.right_tabs, self._start_tab_index), "home", "muted", size=16)
        _icons.apply(self.script_export_btn, "export", "text", size=16)
        bar.addSpacing(14)
        # Solve is the one filled button; Stop turns red while a run can be stopped
        self._styled_widgets = [e for e in self._styled_widgets if e[0] not in (self.solve_btn, self.stop_btn)]
        self.solve_btn.setObjectName("solveButton")
        self.stop_btn.setObjectName("stopButton")
        for b, w in ((self.solve_btn, 120), (self.stop_btn, 92)):
            b.setStyleSheet("")
            b.setMinimumHeight(34)
            b.setMaximumHeight(34)
            b.setFixedWidth(w)
            bar.addWidget(b)

        # ── sizes ──
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([265, 660, 920])
        self.main_splitter = splitter
        ws.select("problem")
        ws.node_selected.connect(self._on_workspace_node)
        ws.menu_provider = self._tree_menu
        ws.help_urls = {k: self.DOCS_URL + anchor for k, anchor in self._HELP_ANCHORS.items()}

        # Settings that would stop Solve show up on their tree nodes as you go
        self._problems_timer = QTimer(self)
        self._problems_timer.setInterval(1500)
        self._problems_timer.timeout.connect(self._check_problems)
        self._problems_timer.start()

        view_menu = self.menuBar().addMenu("View")
        reset = QAction("Reset Layout", self)
        reset.triggered.connect(self._reset_layout)
        view_menu.addAction(reset)
        self._restore_layout()

    def _split_geometry_page(self):
        """Geometry page: the shape picker, the shape's own numbers, and
        the time interval each in their own box."""
        from pinnstudio.ui.workspace import VisibilityFollower
        page = self.workspace.pages["geometry"]
        lay = page["layout"]
        self.geom_shape_box = QGroupBox("Shape")
        QVBoxLayout(self.geom_shape_box).addWidget(self.geom_type_row_widget)
        self.geom_time_box = QGroupBox("Time interval")
        QVBoxLayout(self.geom_time_box).addWidget(self.t_row_widget)
        i = lay.indexOf(self.domain_group)
        lay.insertWidget(i, self.geom_shape_box)
        lay.insertWidget(i + 2, self.geom_time_box)
        page["widgets"] += [self.geom_shape_box, self.geom_time_box]
        self._geom_followers = [VisibilityFollower(self.geom_shape_box, [self.geom_type_row_widget]),
                                VisibilityFollower(self.geom_time_box, [self.t_row_widget])]
        self._update_domain_box_title()

    def _split_training_page(self):
        """Training page: phases, initial-condition pre-training and
        callbacks each in their own box (they were parts of one panel)."""
        page = self.workspace.pages["training"]
        train = page["widgets"][0]
        lay = train.layout()
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if isinstance(w, QLabel) and w.text().startswith("──"):
                w.hide()   # the boxes' titles say it now

        def box(title, widgets):
            g = QGroupBox(title)
            gl = QVBoxLayout(g)
            gl.setSpacing(6)
            for w in widgets:
                lay.removeWidget(w)
                gl.addWidget(w)
            return g

        ic_box = box("Initial-condition pre-training", (self.ic_pretrain_cb, self.ic_pretrain_widget))
        cb_box = box("Callbacks", (self.train_callbacks_show_cb, self.train_callbacks_container))
        train.setTitle("Training phases")
        self.train_ic_box, self.train_callbacks_box = ic_box, cb_box
        L = page["layout"]
        i = L.indexOf(train)
        L.insertWidget(i + 1, self.train_ic_box)
        L.insertWidget(i + 2, self.train_callbacks_box)
        page["widgets"] += [self.train_ic_box, self.train_callbacks_box]

    def _update_domain_box_title(self):
        """The middle Geometry box is named after what it holds."""
        if not hasattr(self, 'geom_shape_box'):
            return
        t = self._current_geometry_type()
        if t == "Custom":
            title = "Shapes"
        elif t == "Interval":
            title = "Interval"
        else:
            title = f"{t} dimensions"
        self.domain_group.setTitle(title)

    # (keywords in a validation message -> the node that fixes it; first match wins)
    _PROBLEM_KEYWORDS = (
        ("sweep", "sweep"),
        ("inverse", ("inverse", "observ", "unknown variable", "true value")),
        ("adaptive", ("rar", "time-adaptive", "time adaptive", "resampl", "adaptive")),
        ("weights", ("scale by initial losses", "loss weight", "weights")),
        ("network", ("network", "layer", "msffn", "neuron", "fourier", "activation", "initializer")),
        ("points", ("domain points", "boundary points", "initial points", "test points", "anchor",
                    "num_domain", "num_boundary")),
        ("training", ("phase", "iteration", "learning rate", "weight decay", "optimizer", "l-bfgs", "lbfgs",
                      "pre-train", "pretrain", "early stopping", "checkpoint", "callback")),
        ("bcs", ("boundary condition", " bc", "bc ", "where", "periodic", "robin", "dirichlet", "neumann",
                 "point set", "interface")),
        ("ic", ("initial condition", " ic ", "ic expression")),
        ("geometry", ("x_min", "y_min", "z_min", "t_min", "geometry", "shape", "vert", "polygon", "disk",
                      "ellipse", "rectangle", "cuboid", "sphere", "triangle", "domain")),
        ("results", ("error analysis", "reference", "plot", "export", "formula")),
        ("equations", ("integral", "frac", "alpha", "output", "equation", "pde", "constant", "residual")),
    )

    def _problem_node(self, msg):
        m = " " + msg.lower() + " "
        for key, words in self._PROBLEM_KEYWORDS:
            words = (words,) if isinstance(words, str) else words
            if any(wd in m for wd in words):
                return key
        return "problem"

    def _check_problems(self):
        """Mark the tree nodes whose settings would stop Solve."""
        if not hasattr(self, 'workspace'):
            return
        try:
            config = self._build_config()
            errors = config.validate()
        except Exception:
            return
        self._update_dirty(config)
        self._refresh_script_tab(config)
        problems = {}
        for msg in errors:
            key = self._problem_node(str(msg))
            if key not in self.workspace.pages or self.workspace.pages[key]["item"].isHidden():
                key = "problem"
            problems.setdefault(key, []).append(str(msg))
        if problems == getattr(self.workspace, "problems", None):
            return
        self.workspace.set_problems(problems)
        n = sum(len(v) for v in problems.values())
        self.problems_btn.setVisible(n > 0)
        self.problems_btn.setText(f" {n} setting{'s' if n != 1 else ''} to fix")

    def _goto_first_problem(self):
        for key in self.workspace.order:
            if (self.workspace.problems or {}).get(key):
                self.workspace.select(key)
                return

    def _gpanel(self):
        """The graphics panel showing the domain: the Sketch when the domain
        is sketched, else the Geometry view."""
        return self.sketch_panel if self.geometry_type_combo.currentText() == "Sketch" else self.domain_view

    def _has_domain_graphics(self):
        return self.radio_2d.isChecked() or self.radio_3d.isChecked()

    def _on_workspace_node(self, key):
        """Selecting a node brings up its graphics: Geometry, Collocation
        points, Initial and Boundary conditions show the domain (the Sketch
        when it is sketched), Results the plots. Boundary conditions
        switches the domain to picking edges, and a BC node lets you click
        that BC's edges (faces in 3D) in and out."""
        sketched = self.geometry_type_combo.currentText() == "Sketch"
        gp = self._gpanel()
        other = self.domain_view if gp is self.sketch_panel else self.sketch_panel
        active = None
        domain_keys = ("geometry", "points", "ic", "bcs")
        if self._has_domain_graphics() and (key in domain_keys or key.startswith("bc:")):
            self.right_tabs.setCurrentIndex(self._sketch_tab_index if sketched else self._domain_tab_index)
            if gp is self.domain_view:
                self._refresh_domain_view()
            if key == "bcs" or key.startswith("bc:"):
                if gp.canvas.tool != "Edges":
                    gp.set_tool("Edges")
                if key.startswith("bc:"):
                    entry = next((e for e in self.custom_bc_list if e.get('_node') == key), None)
                    if entry is not None:
                        i = self.custom_bc_list.index(entry)
                        active = (i, f"BC {i + 1}")
            elif key == "geometry" and sketched and gp.canvas.tool == "Edges":
                gp.set_tool("Select")
        elif key in ("results", "restore"):
            self.right_tabs.setCurrentWidget(self.results_tabs)
        if other.active_bc is not None:
            other.set_active_bc(None)
        other.edge_bar.setVisible(False)
        gp.edge_bar.setVisible(gp.canvas.tool == "Edges" and self._has_domain_graphics())
        self.sketch_selection_box.setTitle("Faces picked in the sketch" if gp.canvas.dim == 3
                                           else "Edges picked in the sketch")
        if active is None:
            if gp.active_bc is not None:
                gp.set_active_bc(None)
        else:
            gp.set_active_bc(*active)
        self._refresh_bc_sketch_boxes()

    def _domain_view_shapes(self):
        """The domain set in the Domain panel as a shape list (2D/3D)."""
        import json
        t = self._current_geometry_type()
        sp = lambda w: float(w.value())
        if t == "Custom":
            return self._custom_geom_shapes_list()
        if t == "Rectangle":
            return [{"type": "Rectangle", "params": {"x_min": sp(self.x_min), "x_max": sp(self.x_max),
                                                     "y_min": sp(self.y_min), "y_max": sp(self.y_max)}}]
        if t == "Disk":
            return [{"type": "Disk", "params": {"cx": sp(self.geom_disk_cx), "cy": sp(self.geom_disk_cy),
                                                "r": sp(self.geom_disk_r)}}]
        if t == "Ellipse":
            return [{"type": "Ellipse", "params": {"cx": sp(self.geom_ellipse_cx), "cy": sp(self.geom_ellipse_cy),
                                                   "a": sp(self.geom_ellipse_a), "b": sp(self.geom_ellipse_b),
                                                   "angle": sp(self.geom_ellipse_angle)}}]
        if t == "Triangle":
            return [{"type": "Triangle", "params": {"vertices_text": self.geom_triangle_verts_input.text()}}]
        if t == "Polygon":
            return [{"type": "Polygon", "params": {"vertices_text": self.geom_polygon_verts_input.text()}}]
        if t == "Cuboid":
            return [{"type": "Cuboid", "params": {"x_min": sp(self.x_min), "x_max": sp(self.x_max),
                                                  "y_min": sp(self.y_min), "y_max": sp(self.y_max),
                                                  "z_min": sp(self.z_min), "z_max": sp(self.z_max)}}]
        if t == "Sphere":
            return [{"type": "Sphere", "params": {"cx": sp(self.geom_sphere_cx), "cy": sp(self.geom_sphere_cy),
                                                  "cz": sp(self.geom_sphere_cz), "r": sp(self.geom_sphere_r)}}]
        return []

    def _refresh_domain_view(self):
        """Keep the Geometry tab in step with the Domain panel (cheap check
        on a timer; redraws only when the domain changed)."""
        import json
        if not hasattr(self, 'domain_view'):
            return
        on = self._has_domain_graphics() and self.geometry_type_combo.currentText() != "Sketch"
        if self.right_tabs.isTabVisible(self._domain_tab_index) != on:
            self.right_tabs.setTabVisible(self._domain_tab_index, on)
        if not on:
            return
        dim = 3 if self.radio_3d.isChecked() else 2
        shapes = self._domain_view_shapes()
        key = json.dumps([dim, shapes], sort_keys=True)
        if key == self._domain_view_key:
            return
        first = self._domain_view_key is None or json.loads(self._domain_view_key)[0] != dim
        self._domain_view_key = key
        self.domain_view.set_dim(dim)
        self.domain_view.set_shapes(shapes, fit=True)
        self.domain_view.refresh_bcs()
        self._refresh_bc_sketch_boxes()

    def _refresh_bc_sketch_boxes(self):
        """Each BC page's "In the sketch" box: which edges (faces) the BC's
        Where: picks. Shown only for a sketched domain."""
        sketched = self._has_domain_graphics()
        gp = self._gpanel()
        bc_of = gp.canvas.edge_bcs() if sketched else []
        pre = "E" if gp.canvas.dim == 2 else "F"
        what = "edges" if pre == "E" else "faces"
        for i, e in enumerate(self.custom_bc_list):
            box = e.get('sketch_box')
            if box is None:
                continue
            try:
                box.setVisible(sketched and e['type'].currentData() not in self._SKETCH_NO_WHERE_TYPES)
                if sketched:
                    picked = [f"{pre}{k + 1}" for k, lst in enumerate(bc_of) if i in lst]
                    e['sketch_lbl'].setText(
                        (f"Picks {', '.join(picked)}." if picked else f"Picks no {what} of the domain yet.") +
                        f"  Click {what} in the graphics to add them to this BC or take them out.")
            except RuntimeError:
                pass

    def _on_sketch_toggle_edge(self, index, k, add):
        """A click on an edge while a BC node is selected: put the edge in
        that BC's Where: or take it out."""
        from pinnstudio.ui.sketch_edges import add_to_where, join_where, remove_from_where
        if not (0 <= index < len(self.custom_bc_list)):
            return
        gp = self._gpanel()
        edges = gp.canvas.edges()
        if not (0 <= k < len(edges)):
            return
        entry = self.custom_bc_list[index]
        cur = entry['location'].text().strip()
        expr = edges[k]['expr']
        if add:
            entry['location'].setText(add_to_where(cur, expr))
        else:
            if cur in ("", "True"):
                rest = [e['expr'] for j, e in enumerate(edges) if j != k]
                new = join_where(rest)
            else:
                new = remove_from_where(cur, [expr])
            if new is None:
                gp.hint.setText(
                    f"BC {index + 1}'s Where: was written by hand, so it can't be edited by clicking -- "
                    "change it in the panel, or clear it and pick the edges again.")
                return
            if not new:
                gp.hint.setText(
                    f"That's the last edge of BC {index + 1} -- a BC needs at least one; remove the BC instead.")
                return
            entry['location'].setText(new)
        gp.refresh_bcs()
        self._refresh_bc_sketch_boxes()

    # ── New / recent files / unsaved changes / shortcuts ────────────
    _recent_mem = []   # recent files when settings aren't kept (tests)
    RECENT_MAX = 8

    def _recent_files(self):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return list(MainWindow._recent_mem)
        try:
            v = self._layout_settings().value("recent_files") or []
            if isinstance(v, str):
                v = [v]
            return [str(x) for x in v]
        except Exception:
            return []

    def _set_recent_files(self, paths):
        paths = paths[:self.RECENT_MAX]
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            MainWindow._recent_mem = list(paths)
        else:
            try:
                self._layout_settings().setValue("recent_files", paths)
            except Exception:
                pass
        self._refresh_recent()

    def _add_recent(self, path):
        path = os.path.abspath(path)
        self._set_recent_files([path] + [p for p in self._recent_files() if p != path])

    def _remove_recent(self, path):
        self._set_recent_files([p for p in self._recent_files() if p != os.path.abspath(path)])

    def _refresh_recent(self):
        paths = self._recent_files()
        if hasattr(self, "start_page"):
            self.start_page.set_recent(paths)
        if hasattr(self, "recent_menu"):
            self.recent_menu.clear()
            for i, p in enumerate(paths):
                a = QAction(f"&{i + 1}  {os.path.basename(p)}", self)
                a.setToolTip(p)
                a.triggered.connect(lambda _c=False, q=p: self._open_problem(q))
                self.recent_menu.addAction(a)
            if paths:
                self.recent_menu.addSeparator()
                clear = QAction("Clear list", self)
                clear.triggered.connect(lambda: self._set_recent_files([]))
                self.recent_menu.addAction(clear)
            self.recent_menu.setEnabled(bool(paths))

    @staticmethod
    def _config_key(config):
        """What "changed" compares: every setting except the per-run folder."""
        import dataclasses
        import json
        try:
            d = dataclasses.asdict(config)
        except Exception:
            return None
        d.pop("save_dir", None)
        return json.dumps(d, sort_keys=True, default=str)

    def _mark_clean(self, config=None):
        try:
            self._clean_key = self._config_key(config or self._build_config())
        except Exception:
            self._clean_key = None
        self._set_dirty(False)

    def _capture_fresh_problem(self):
        try:
            self._fresh_config = self._build_config()
        except Exception:
            self._fresh_config = None
        if getattr(self, "_current_path", None) is None and not getattr(self, "_current_problem_name", ""):
            self._mark_clean(self._fresh_config)

    def _set_dirty(self, dirty):
        self._dirty = bool(dirty)
        name = getattr(self, "_current_problem_name", "") or "Untitled"
        self.setWindowTitle(f"{_ex_catalog.display_name(name)}[*] — PINNStudio")
        self.setWindowModified(self._dirty)

    def _update_dirty(self, config):
        if getattr(self, "_clean_key", None) is None:
            return
        dirty = self._config_key(config) != self._clean_key
        if dirty != getattr(self, "_dirty", False):
            self._set_dirty(dirty)

    def _maybe_save_changes(self):
        """Before the problem is replaced: offer to save unsaved changes.
        False = cancel."""
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return True
        try:
            self._update_dirty(self._build_config())
        except Exception:
            pass
        if not getattr(self, "_dirty", False):
            return True
        name = _ex_catalog.display_name(getattr(self, "_current_problem_name", "") or "Untitled")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"Save the changes to \"{name}\"?")
        box.setInformativeText("Your changes will be lost if you don't save them.")
        box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                               QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        r = box.exec()
        if r == QMessageBox.StandardButton.Save:
            return bool(self._save_problem())
        return r == QMessageBox.StandardButton.Discard

    def _new_problem(self, dim="1D"):
        """A fresh problem: the settings PINNStudio starts with (keeping the
        save folder, GPU and plot preferences), in `dim` (1D/2D/3D/ODE)."""
        if not self._maybe_save_changes():
            return
        fresh = getattr(self, "_fresh_config", None)
        if fresh is not None:
            import dataclasses
            try:
                current = self._build_config()
                keep = {k: getattr(current, k) for k in self._DEEPXDE_KEEP_FIELDS if hasattr(current, k)}
            except Exception:
                keep = {}
            keep.pop("save_dir", None)
            save_dir_text = self.save_dir_input.text()
            cfg = dataclasses.replace(fresh, **keep)
            self._applying_template = True
            try:
                self._apply_config(cfg)
            finally:
                self._applying_template = False
            self.save_dir_input.setText(save_dir_text)
        self.quick_examples_combo.blockSignals(True)
        self.quick_examples_combo.setCurrentText("None")
        self.quick_examples_combo.blockSignals(False)
        self._current_template = None
        self._example_snapshot = None
        radio = {"1D": self.radio_1d, "2D": self.radio_2d, "3D": self.radio_3d, "ODE": self.radio_ode}.get(dim)
        if radio is not None and not radio.isChecked():
            radio.setChecked(True)
        self._current_path = None
        self._current_problem_name = ""
        self._mark_clean()
        self.workspace.select("equations")
        self.log_box.append(f"✅ New {dim} problem")

    def _load_example_from_start(self, key):
        if not self._maybe_save_changes():
            return
        cat = _ex_catalog.FORWARD
        if self.quick_examples_category_combo.currentText() != cat:
            self.quick_examples_category_combo.setCurrentText(cat)
        self._from_start_gallery = True   # already asked above
        try:
            if self.quick_examples_combo.currentText() == key:
                self._on_quick_example_selected(key)
            else:
                self.quick_examples_combo.setCurrentText(key)
        finally:
            self._from_start_gallery = False
        self._current_problem_name = key
        self._current_path = None
        self._mark_clean()
        self.workspace.select("problem")

    # Each tree node's "?" opens its section of the README.
    DOCS_URL = "https://github.com/AsfandyarKhan72/pinnstudio"
    _HELP_ANCHORS = {
        "problem": "#problem-setup", "equations": "#problem-setup", "inverse": "#problem-setup",
        "geometry": "#problem-setup", "points": "#problem-setup", "ic": "#problem-setup",
        "bcs": "#problem-setup", "network": "#problem-setup",
        "training": "#training", "weights": "#training", "adaptive": "#training",
        "sweep": "#parameter-sweep", "results": "#analysis--output", "restore": "#restore--visualize",
    }

    def _tree_menu(self, key):
        """Right-click actions on a model-tree node."""
        items = []
        if key == "bcs" or key.startswith("bc:"):
            if self.add_custom_bc_btn.isEnabled():
                items.append(("Add boundary condition", self._menu_add_bc, "add"))
            entry = next((e for e in self.custom_bc_list if e.get('_node') == key), None)
            if entry is not None:
                items.append(("Duplicate", lambda e=entry: self._duplicate_bc(e), None))
                if entry.get('remove_btn') is not None and not entry['remove_btn'].isHidden():
                    items.append(("Delete", lambda e=entry: e['remove_btn'].click(), "delete"))
        elif key == "geometry":
            if self._has_domain_graphics():
                items.append(("Sketch the domain", lambda: self.geometry_type_combo.setCurrentText("Sketch"),
                              "sketch"))
        elif key == "training":
            items.append(("Add training phase", lambda: self._add_scheduler_phase(), "add"))
        elif key == "inverse":
            items.append(("Add trainable variable", lambda: self._add_inverse_var_row(), "add"))
        elif key == "equations":
            items.append(("Add constant", lambda: self._add_constant_row(), "add"))
        elif key in ("results", "restore"):
            items.append(("Open a saved problem…", lambda: self._open_problem(), "open"))
        if key == "problem":
            items += [("New 1D problem", lambda: self._new_problem("1D"), "add"),
                      ("New 2D problem", lambda: self._new_problem("2D"), None),
                      ("New 3D problem", lambda: self._new_problem("3D"), None),
                      ("New ODE system", lambda: self._new_problem("ODE"), None)]
        return items

    def _menu_add_bc(self):
        self.add_custom_bc_btn.click()
        self._sync_bc_nodes()
        if self.custom_bc_list:
            self.workspace.select(self.custom_bc_list[-1].get('_node', "bcs"))

    def _duplicate_bc(self, entry):
        """A copy of a BC row, added at the end."""
        import json
        if entry not in self.custom_bc_list:
            return
        i = self.custom_bc_list.index(entry)
        try:
            d = json.loads(self._build_custom_bc_json())[i]
        except Exception:
            return
        d['bc_type'] = d.pop('type', 'dirichlet')
        import inspect
        allowed = set(inspect.signature(self._add_custom_bc_entry).parameters)
        new = self._add_custom_bc_entry(**{k: v for k, v in d.items() if k in allowed})
        self._update_bc_mode_visibility()
        self._sync_bc_nodes()
        if new.get('_node'):
            self.workspace.select(new['_node'])

    def _show_saved_run(self, run):
        """A run picked in Results -> History: its figures in Overview, its
        errors in Error analysis."""
        from types import SimpleNamespace
        folder = run["folder"]
        sol = os.path.join(folder, "solution_results")
        ptype = run.get("plot_type") or ""
        if not ptype and os.path.exists(os.path.join(sol, "solution_plot.gif")) and \
                not os.path.exists(os.path.join(sol, "solution_plot.png")):
            ptype = "Line Animation (GIF)"
        self.loss_label.setText("No loss plot saved for this run")
        self.solution_label.setText("No solution plot saved for this run")
        self._display_run_result_plots(SimpleNamespace(plot_type=ptype), folder)
        self.ea_panel.show_run(run.get("metrics") or [], os.path.join(folder, "error_analysis"))
        self.results_tabs.setCurrentIndex(0)
        if hasattr(self, "status_text"):
            bits = [f"Showing {run.get('label', os.path.basename(folder))} ({run.get('when', '')})"]
            if run.get("loss") is not None:
                bits.append(f"final loss {run['loss']:.3e}")
            if run.get("l2") is not None:
                bits.append(f"mean relative L₂ {run['l2']:.3e}")
            self.status_text.setText("  ·  ".join(bits))

    def _shortcut_solve(self):
        if self.solve_btn.isVisible() and self.solve_btn.isEnabled():
            self.solve_btn.click()

    def _shortcut_stop(self):
        if self.stop_btn.isEnabled():
            self.stop_btn.click()

    def _refresh_script_tab(self, config=None):
        """Regenerate the Script tab when it's showing and the problem changed."""
        if not hasattr(self, "script_view") or self.bottom_tabs.currentWidget() is not self.script_tab:
            return
        try:
            config = config or self._build_config()
        except Exception as e:
            self.script_status.setText(f"Can't build the problem yet: {e}")
            return
        key = self._config_key(config)
        if key == self._script_key:
            return
        self._script_key = key
        try:
            from pinnstudio.core.codegen import generate_clean_script
            script = generate_clean_script(config)
        except Exception as e:
            self.script_status.setText(f"The script can't be written for these settings yet: {e}")
            return
        bar = self.script_view.verticalScrollBar()
        pos = bar.value()
        self.script_view.setPlainText(script)
        bar.setValue(pos)
        n = script.count("\n") + 1
        self.script_status.setText(f"The DeepXDE script for this problem ({n} lines), as Export script would write it.")

    def _copy_script(self):
        self._script_key = None
        self._refresh_script_tab()
        QApplication.clipboard().setText(self.script_view.toPlainText())
        if hasattr(self, "status_text"):
            self.status_text.setText("Script copied to the clipboard")

    def _layout_settings(self):
        from PyQt6.QtCore import QSettings
        return QSettings("PINNStudio", "PINNStudio")

    def _restore_layout(self):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return   # tests: always the default layout
        try:
            st = self._layout_settings()
            main, right = st.value("layout/main"), st.value("layout/right")
            if main is not None:
                self._layout_restored = bool(self.main_splitter.restoreState(main))
            if right is not None:
                self.right_splitter.restoreState(right)
        except Exception:
            pass

    def _save_layout(self):
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        try:
            st = self._layout_settings()
            st.setValue("layout/main", self.main_splitter.saveState())
            st.setValue("layout/right", self.right_splitter.saveState())
        except Exception:
            pass

    def _reset_layout(self):
        self._apply_default_sizes()

    def _apply_default_sizes(self):
        """Panel sizes for the window's current size: the tree narrow, the
        settings about a third, the graphics the rest; graphics over log 7:3."""
        W = max(self.main_splitter.width(), 900)
        tree = max(200, min(270, int(0.14 * W)))
        settings = max(420, min(680, int(0.34 * W)))
        self.main_splitter.setSizes([tree, settings, max(300, W - tree - settings)])
        H = max(self.right_splitter.height(), 400)
        self.right_splitter.setSizes([int(0.7 * H), H - int(0.7 * H)])

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, '_sized_once', False) and hasattr(self, 'main_splitter'):
            self._sized_once = True
            if not getattr(self, '_layout_restored', False):
                QTimer.singleShot(0, self._apply_default_sizes)

    def _build_setup_panels(self, left_layout):
        """Quick Examples, dimension, Forward/Inverse, and the PDE input panel."""
        # ── Quick Examples (shown first: picking one drives everything ──
        # ── else below it -- dimension, Forward/Inverse, Time ───────────
        # ── Dependent/Stationary -- instead of the user having to set ───
        # ── those by hand first before the right examples even appear) ──
        examples_group = QGroupBox("📋 Quick Examples")
        examples_layout = QGridLayout(examples_group)
        # Category: PINNStudio's own templates, or DeepXDE's documented
        # forward / inverse examples (pinnstudio/core/deepxde_examples.py).
        examples_layout.addWidget(QLabel("Category:"), 0, 0)
        self.quick_examples_category_combo = QComboBox()
        self.quick_examples_category_combo.addItems(list(_ex_catalog.CATEGORIES))
        self.quick_examples_category_combo.setToolTip(
            "Forward: every example, solving for the field.\n"
            "Inverse: examples that learn unknown parameters or fields from data -- DeepXDE's inverse\n"
            "examples, and PINNStudio's templates loaded in Inverse mode.")
        self.quick_examples_category_combo.setFixedHeight(28)
        self.quick_examples_category_combo.currentTextChanged.connect(self._on_quick_example_category_changed)
        examples_layout.addWidget(self.quick_examples_category_combo, 0, 1)
        examples_layout.addWidget(QLabel("Load example:"), 1, 0)
        self.quick_examples_combo = ExampleCombo()
        # Every template across all three dimensions, up front -- not
        # filtered by whatever dimension happens to be selected right
        # now (there's nothing to filter BY yet, since this sits above
        # the Dimension selector). _on_template_selected sets the
        # matching dimension radio itself once something here is picked;
        # _on_dim_changed resets this back to "None" when the user
        # changes dimension by hand instead (see both for the full
        # two-way sync).
        self.quick_examples_combo.set_entries(_ex_catalog.for_category_all(_ex_catalog.FORWARD))
        self.quick_examples_combo.setFixedHeight(28)
        # Long DeepXDE names: keep the combo's width set by the panel, and
        # let the drop-down list itself be as wide as its longest entry.
        self.quick_examples_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.quick_examples_combo.setMinimumContentsLength(18)
        self.quick_examples_combo.view().setMinimumWidth(320)
        self.quick_examples_combo.keyChanged.connect(self._on_quick_example_selected)
        examples_layout.addWidget(self.quick_examples_combo, 1, 1)
        examples_layout.setColumnStretch(1, 1)
        left_layout.addWidget(examples_group)

        # ── Dimension selector ────────────────────────────────
        dim_group = QGroupBox("Problem Dimension")
        dim_layout = QHBoxLayout(dim_group)
        self.radio_1d = QRadioButton("1D  (x)")
        self.radio_2d = QRadioButton("2D  (x, y)")
        self.radio_3d = QRadioButton("3D  (x, y, z)")
        # ODE (Round 74): only time, no space -- e.g. y1' = y2, y2' = -y1.
        self.radio_ode = QRadioButton("ODE  (t)")
        self.radio_ode.setToolTip("Ordinary differential equations in t only (no space): write derivatives\n"
                                  "as dy_t, dy_tt; initial conditions are values at t = t₀.")
        self.radio_1d.setChecked(True)
        self.dim_group_btn = QButtonGroup()
        self.dim_group_btn.addButton(self.radio_ode)
        self.dim_group_btn.addButton(self.radio_1d)
        self.dim_group_btn.addButton(self.radio_2d)
        self.dim_group_btn.addButton(self.radio_3d)
        dim_layout.addWidget(self.radio_ode)
        dim_layout.addWidget(self.radio_1d)
        dim_layout.addWidget(self.radio_2d)
        dim_layout.addWidget(self.radio_3d)
        left_layout.addWidget(dim_group)
        self.radio_1d.toggled.connect(self._on_dim_changed)
        self.radio_2d.toggled.connect(self._on_dim_changed)
        self.radio_3d.toggled.connect(self._on_dim_changed)
        self.radio_ode.toggled.connect(self._on_ode_toggled)

        # ── Problem type ──────────────────────────────────────
        # Time Dependent/Stationary now lives here too, right above
        # Forward/Inverse -- grouping "what kind of problem is this" in
        # one place instead of splitting stationarity off into the
        # Domain section below, where it read more like a domain-bounds
        # detail than the fairly fundamental modeling choice it actually
        # is. The underlying state is still self.steady_state_check (a
        # QCheckBox, never added to any visible layout -- every one of
        # its many existing isChecked()/setChecked()/toggled call sites
        # elsewhere in this file is unaffected): these two radios are a
        # two-way-synced VIEW onto it, not a replacement, so a template
        # that calls steady_state_check.setChecked(...) (e.g. a Poisson
        # template turning Stationary on) updates these radios too, and
        # picking a radio here updates steady_state_check (and so
        # _on_steady_state_changed and everything it drives) the same
        # way checking the old checkbox always did.
        type_group = QGroupBox("Problem Type")
        type_outer_layout = QVBoxLayout(type_group)
        stationarity_layout = QHBoxLayout()
        self.radio_time_dependent = QRadioButton("Time Dependent")
        self.radio_stationary = QRadioButton("Stationary")
        self.radio_time_dependent.setChecked(True)
        self.stationarity_group_btn = QButtonGroup()
        self.stationarity_group_btn.addButton(self.radio_time_dependent)
        self.stationarity_group_btn.addButton(self.radio_stationary)
        stationarity_layout.addWidget(self.radio_time_dependent)
        stationarity_layout.addWidget(self.radio_stationary)
        type_outer_layout.addLayout(stationarity_layout)

        type_layout = QHBoxLayout()
        self.radio_forward = QRadioButton("Forward")
        self.radio_inverse = QRadioButton("Inverse")
        self.radio_forward.setChecked(True)
        self.problem_type_group = QButtonGroup()
        self.problem_type_group.addButton(self.radio_forward)
        self.problem_type_group.addButton(self.radio_inverse)
        type_layout.addWidget(self.radio_forward)
        type_layout.addWidget(self.radio_inverse)
        type_outer_layout.addLayout(type_layout)
        left_layout.addWidget(type_group)
        self.radio_forward.toggled.connect(self._on_problem_type_changed)
        self.radio_forward.toggled.connect(lambda _c: self._refresh_equation_previews())
        # The two-way sync to self.steady_state_check is wired once that
        # checkbox itself is created, further down (see its own comment).

        # ── PDE input (Number of PDEs/Outputs folded in at the top, ────
        # ── since they belong together and don't need their own panel) ─
        self.pde_group = QGroupBox("PDE Definition")
        pde_outer_layout = QVBoxLayout(self.pde_group)
        pde_outer_layout.setSpacing(6)

        nout_row = QHBoxLayout()
        nout_row.addWidget(QLabel("Outputs:"))
        self.num_outputs_spin = QSpinBox()
        self.num_outputs_spin.setRange(1, 10)
        self.num_outputs_spin.setValue(1)
        self.num_outputs_spin.setFixedHeight(30)
        self.num_outputs_spin.setFixedWidth(60)
        self.num_outputs_spin.valueChanged.connect(self._on_num_outputs_changed)
        nout_row.addWidget(self.num_outputs_spin)
        # Equations (Round 63): how many PDE residuals -- follows Outputs
        # unless changed (DeepXDE allows any number, e.g. three for one
        # output, or one for two outputs when the second is an unknown field).
        nout_row.addSpacing(12)
        nout_row.addWidget(QLabel("Equations:"))
        self.num_equations_spin = QSpinBox()
        self.num_equations_spin.setRange(1, 10)
        self.num_equations_spin.setValue(1)
        self.num_equations_spin.setFixedHeight(30)
        self.num_equations_spin.setFixedWidth(60)
        self.num_equations_spin.setToolTip("Number of PDE residuals; resets to the number of outputs when that changes.")
        self.num_equations_spin.valueChanged.connect(self._on_num_equations_changed)
        nout_row.addWidget(self.num_equations_spin)
        # Round 76: quadrature points for integral(...) terms (ODE mode).
        self.integral_points_label = QLabel("∫ points:")
        self.integral_points_spin = QSpinBox()
        self.integral_points_spin.setRange(2, 200)
        self.integral_points_spin.setValue(20)
        self.integral_points_spin.setFixedHeight(30)
        self.integral_points_spin.setFixedWidth(60)
        _int_tip = ("Gauss-Legendre points for integral(...) terms in the equations.\n"
                    "integral(f) is the integral of f from the start t₀ to the current t, where f uses\n"
                    "s (the integration variable), t, and the outputs at s -- e.g.\n"
                    "du_t + u - integral(exp(s - t)*u)   (a Volterra integro-differential equation)")
        self.integral_points_label.setToolTip(_int_tip)
        self.integral_points_spin.setToolTip(_int_tip)
        nout_row.addSpacing(12)
        nout_row.addWidget(self.integral_points_label)
        nout_row.addWidget(self.integral_points_spin)
        self.integral_points_label.setVisible(False)
        self.integral_points_spin.setVisible(False)
        # "Show derivative reference" sits on this row, after Equations; its
        # hint text (self.pde_hint) opens below the Constants row.
        # (it sits on the Constants row below, where there's room)
        self.pde_hint_toggle = QCheckBox("📖 Show derivative reference")
        self.pde_hint_toggle.setChecked(False)
        self._register_hint(self.pde_hint_toggle, '#74c0fc')
        nout_row.addStretch()
        pde_outer_layout.addLayout(nout_row)

        # Named constants (Round 70): "Re = 20", "nu = 1/Re", ... usable by
        # name in the equations, BCs, ICs, transforms and the exact-solution
        # reference. Empty by default; rows are added with the button.
        self.constants_widget = QWidget()
        _cw = QVBoxLayout(self.constants_widget)
        _cw.setContentsMargins(0, 0, 0, 0); _cw.setSpacing(2)
        _crow = QHBoxLayout()
        self.constants_label = QLabel("Constants:")
        self.constants_label.setToolTip(
            "Named values used in your expressions, e.g. Re = 20, nu = 1/Re.\n"
            "A value can be a number or an expression in math functions (sqrt, exp, pi, ...)\n"
            "and the constants above it. Usable in the equations, boundary and initial\n"
            "conditions, transforms and the exact-solution reference.")
        _crow.addWidget(self.constants_label)
        self.add_constant_btn = QPushButton("+ Add constant")
        self.add_constant_btn.setFixedHeight(24)
        self.add_constant_btn.clicked.connect(lambda: self._add_constant_row())
        _crow.addWidget(self.add_constant_btn)
        _crow.addSpacing(12)
        _crow.addWidget(self.pde_hint_toggle)
        _crow.addStretch()
        _cw.addLayout(_crow)
        self.constants_rows_layout = QVBoxLayout()
        self.constants_rows_layout.setSpacing(2)
        _cw.addLayout(self.constants_rows_layout)
        self.constant_rows = []
        pde_outer_layout.addWidget(self.constants_widget)

        # One shared "📖 Show derivative reference" toggle + hint, at the
        # TOP of this panel (same persistent-outside-the-rebuild pattern
        # the IC panel's "Show IC reference" toggle already uses) rather
        # than once per output-count rebuild at the bottom -- shown once
        # here, kept up to date (text only, dimension/output-count can
        # change what it says) by _build_pde_inputs() below, never itself
        # torn down or recreated.

        self.pde_hint = QLabel("")
        self._register_hint(self.pde_hint, '#74c0fc')
        self.pde_hint.setWordWrap(True)
        self.pde_hint.setVisible(False)
        pde_outer_layout.addWidget(self.pde_hint)
        self.pde_hint_toggle.stateChanged.connect(lambda state: self.pde_hint.setVisible(state == 2))

        # self.pde_main_layout stays a nested layout (not the group's own
        # top-level layout) so _build_pde_inputs() can freely clear/rebuild
        # it on every output-count change without disturbing the "How many
        # PDEs" row above it.
        self.pde_main_layout = QVBoxLayout()
        self.pde_main_layout.setSpacing(6)
        pde_outer_layout.addLayout(self.pde_main_layout)
        self.pde_inputs = []
        self.output_name_inputs = []

        # Round 77: fractional operator settings -- shown while an equation uses frac(...).
        self.frac_widget = QWidget()
        _fl = QHBoxLayout(self.frac_widget)
        _fl.setContentsMargins(0, 0, 0, 0); _fl.setSpacing(4)
        _frac_tip = ("frac(u): the fractional operator of order α applied to output u, discretized as DeepXDE's FPDE\n"
                     "(Grunwald-Letnikov). 1D: the sum of the left and right Riemann-Liouville derivatives;\n"
                     "2D/3D: the integral over directions of the directional derivative -- times\n"
                     "C(α, D) = Γ((1-α)/2) Γ((D+α)/2) / (2 π^((D+1)/2)) it is minus the fractional Laplacian.\n"
                     "α: a number, a constant, or an inverse variable's name (to learn α).\n"
                     "Resolution: 1D the number of mesh points (static) or points per unit length (dynamic);\n"
                     "2D '8, 100' = directions, points per unit length; 3D '8, 8, 100'.\n"
                     "Mesh: static (1D interval only, a fixed uniform mesh) or dynamic.\n"
                     "The test loss uses the training points (DeepXDE's fractional data), not Test points.")
        _flab = QLabel("Fractional  α:")
        _flab.setToolTip(_frac_tip)
        _fl.addWidget(_flab)
        self.frac_alpha_edit = QLineEdit("1.5"); self.frac_alpha_edit.setFixedWidth(60)
        self.frac_alpha_edit.setToolTip(_frac_tip)
        _fl.addWidget(self.frac_alpha_edit)
        _fl.addWidget(QLabel("Resolution:"))
        self.frac_res_edit = QLineEdit("100"); self.frac_res_edit.setFixedWidth(80)
        self.frac_res_edit.setToolTip(_frac_tip)
        _fl.addWidget(self.frac_res_edit)
        _fl.addWidget(QLabel("Mesh:"))
        self.frac_mesh_combo = QComboBox()
        self.frac_mesh_combo.addItem("dynamic", "dynamic"); self.frac_mesh_combo.addItem("static", "static")
        self.frac_mesh_combo.setToolTip(_frac_tip)
        _fl.addWidget(self.frac_mesh_combo)
        _fl.addStretch()
        pde_outer_layout.addWidget(self.frac_widget)
        self.frac_widget.setVisible(False)
        self._build_pde_inputs(1)
        left_layout.addWidget(self.pde_group)

    def _refresh_equation_previews(self):
        for w in getattr(self, "pde_inputs", []):
            prev = getattr(w, "_preview", None)
            if prev is not None:
                try:
                    prev._timer.start()
                except RuntimeError:
                    pass

    def _equation_known_names(self):
        """Names an equation may use besides functions and constants: the
        outputs, the named constants and the inverse problem's variables."""
        outs = {w.text().strip() for w in getattr(self, "output_name_inputs", []) if w.text().strip()}
        other = set()
        for r in getattr(self, "constant_rows", []):
            try:
                if r["name"].text().strip():
                    other.add(r["name"].text().strip())
            except RuntimeError:
                pass
        inverse = getattr(self, "radio_inverse", None) is not None and self.radio_inverse.isChecked()
        for r in (getattr(self, "inv_var_rows", []) if inverse else []):
            try:
                if r['name'].text().strip():
                    other.add(r['name'].text().strip())
            except (RuntimeError, KeyError):
                pass
        if self._is_ode():
            coords = "t"
        elif getattr(self, "radio_3d", None) is not None and self.radio_3d.isChecked():
            coords = "xyzt"
        elif getattr(self, "radio_2d", None) is not None and self.radio_2d.isChecked():
            coords = "xyt"
        else:
            coords = "xt"
        return {"outputs": outs, "other": other, "coords": coords}

    # ── Named constants (Round 70) ────────────────────────────
    def _add_constant_row(self, name="", value=""):
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
        name_edit = QLineEdit(name); name_edit.setPlaceholderText("name, e.g. Re"); name_edit.setFixedWidth(80)
        value_edit = QLineEdit(value); value_edit.setPlaceholderText("value, e.g. 20 or 1/Re")
        result = QLabel(""); result.setMinimumWidth(70)
        remove = QPushButton("✕"); remove.setFixedSize(24, 24); remove.setToolTip("Remove this constant")
        lay.addWidget(name_edit); lay.addWidget(QLabel("=")); lay.addWidget(value_edit, 1); lay.addWidget(result)
        lay.addWidget(remove)
        entry = {"widget": row, "name": name_edit, "value": value_edit, "result": result}
        self.constant_rows.append(entry)
        self.constants_rows_layout.addWidget(row)
        name_edit.textChanged.connect(lambda _t: self._refresh_constant_values())
        name_edit.textChanged.connect(lambda _t: self._refresh_equation_previews())
        value_edit.textChanged.connect(lambda _t: self._refresh_constant_values())

        def _remove():
            row.deleteLater()
            if entry in self.constant_rows:
                self.constant_rows.remove(entry)
            self._refresh_constant_values()
            self._refresh_equation_previews()
        remove.clicked.connect(_remove)
        self._refresh_constant_values()
        self._refresh_equation_previews()
        return entry

    def _constants_json(self):
        import json as _json_c
        rows = [{"name": e["name"].text().strip(), "value": e["value"].text().strip()}
                for e in getattr(self, "constant_rows", [])]
        rows = [r for r in rows if r["name"] or r["value"]]
        return _json_c.dumps(rows) if rows else ""

    def _set_constants(self, text):
        from pinnstudio.core import constants as _cmod
        for e in list(getattr(self, "constant_rows", [])):
            e["widget"].deleteLater()
        self.constant_rows = []
        try:
            rows = _cmod.parse(text)
        except ValueError:
            rows = []
        for name, value in rows:
            self._add_constant_row(name, value)

    def _refresh_constant_values(self):
        """Show each constant's value next to it, or the problem in red."""
        from pinnstudio.core import constants as _cmod
        rows = getattr(self, "constant_rows", [])
        for e in rows:
            e["result"].setText(""); e["result"].setStyleSheet(""); e["result"].setToolTip("")
        if not rows:
            return
        names = [w.text() for w in getattr(self, "output_name_inputs", [])]
        try:
            vals = {n: v for n, _x, v in _cmod.evaluate(self._constants_json(), names)}
            for e in rows:
                n = e["name"].text().strip()
                if n in vals:
                    e["result"].setText(f"= {vals[n]:.6g}")
        except ValueError as exc:
            for e in rows:
                if e["name"].text().strip() and e["name"].text().strip() in str(exc):
                    e["result"].setText("⚠")
                    e["result"].setStyleSheet("color: #e03131;")
                    e["result"].setToolTip(str(exc))
                    break
            else:
                rows[-1]["result"].setText("⚠"); rows[-1]["result"].setStyleSheet("color: #e03131;")
                rows[-1]["result"].setToolTip(str(exc))

    def _build_domain_panels(self, left_layout):
        """Domain/geometry, collocation points, and the initial/boundary condition panels."""
        # ── Domain (Geometry Type folded in as the first field, since ──
        # ── the shape choice governs what the rest of this panel asks ──
        # ── for below it -- they belong in one panel, not two). ────────
        domain_group = QGroupBox("Domain")
        self.domain_group = domain_group
        domain_layout = QVBoxLayout(domain_group)
        domain_layout.setSpacing(6)

        self.geometry_type_combo = QComboBox()
        self.geometry_type_combo.setFixedHeight(28)
        self.geom_type_row_widget = self._domain_row("Shape:", self.geometry_type_combo)
        self.geom_type_row_widget.setVisible(False)
        domain_layout.addWidget(self.geom_type_row_widget)
        self.geometry_type_combo.currentTextChanged.connect(self._on_geometry_type_changed)

        # x/y/z rows: only meaningful for the box shapes (Interval/Rectangle/
        # Cuboid), where the domain really is an axis-aligned box. For every
        # other shape these are replaced by that shape's own parameters
        # (center + radius, vertices, ...) below -- see
        # _update_domain_fields_visibility().
        #
        # Shared fixed width for every domain spinbox (x/y/z/t min & max) so
        # they all render the same size whatever value they show. All are
        # DomainSpinBox: 10 decimals stored, trailing zeros not shown, so
        # domains like [-pi, pi] or t in [0, pi/2] are kept exactly.
        _DOMAIN_SPIN_WIDTH = 85
        self.x_row_widget = QWidget()
        x_row = QHBoxLayout(self.x_row_widget)
        x_row.setContentsMargins(0, 0, 0, 0)
        x_row.addWidget(self._domain_label("x:")); x_row.setSpacing(6)
        self.x_min = DomainSpinBox()
        self.x_min.setRange(-1e6, 1e6); self.x_min.setValue(0.0); self.x_min.setSingleStep(0.5)
        self.x_max = DomainSpinBox()
        self.x_max.setRange(-1e6, 1e6); self.x_max.setValue(1.0); self.x_max.setSingleStep(0.5)
        # Fixed width so this box is the same size as every other domain
        # spinbox (x/y/z/t) regardless of how many decimals it shows --
        # see the matching comment on self.t_min/self.t_max below, which
        # need 4 decimals (not 2) for precision and would otherwise render
        # visibly wider than these, uneven row/box sizes across the panel.
        self.x_min.setFixedWidth(_DOMAIN_SPIN_WIDTH); self.x_max.setFixedWidth(_DOMAIN_SPIN_WIDTH)
        x_row.addWidget(self.x_min); x_row.addWidget(QLabel("to")); x_row.addWidget(self.x_max); x_row.addStretch()
        domain_layout.addWidget(self.x_row_widget)

        self.y_row_widget = QWidget()
        y_row = QHBoxLayout(self.y_row_widget)
        y_row.setContentsMargins(0, 0, 0, 0)
        y_row.addWidget(self._domain_label("y:")); y_row.setSpacing(6)
        self.y_min = DomainSpinBox()
        self.y_min.setRange(-1e6, 1e6); self.y_min.setValue(0.0); self.y_min.setSingleStep(0.5)
        self.y_max = DomainSpinBox()
        self.y_max.setRange(-1e6, 1e6); self.y_max.setValue(1.0); self.y_max.setSingleStep(0.5)
        self.y_min.setFixedWidth(_DOMAIN_SPIN_WIDTH); self.y_max.setFixedWidth(_DOMAIN_SPIN_WIDTH)
        y_row.addWidget(self.y_min); y_row.addWidget(QLabel("to")); y_row.addWidget(self.y_max); y_row.addStretch()
        self.y_row_widget.setVisible(False)
        domain_layout.addWidget(self.y_row_widget)

        self.z_row_widget = QWidget()
        z_row = QHBoxLayout(self.z_row_widget)
        z_row.setContentsMargins(0, 0, 0, 0)
        z_row.addWidget(self._domain_label("z:")); z_row.setSpacing(6)
        self.z_min = DomainSpinBox()
        self.z_min.setRange(-1e6, 1e6); self.z_min.setValue(0.0); self.z_min.setSingleStep(0.5)
        self.z_max = DomainSpinBox()
        self.z_max.setRange(-1e6, 1e6); self.z_max.setValue(1.0); self.z_max.setSingleStep(0.5)
        self.z_min.setFixedWidth(_DOMAIN_SPIN_WIDTH); self.z_max.setFixedWidth(_DOMAIN_SPIN_WIDTH)
        z_row.addWidget(self.z_min); z_row.addWidget(QLabel("to")); z_row.addWidget(self.z_max); z_row.addStretch()
        self.z_row_widget.setVisible(False)
        domain_layout.addWidget(self.z_row_widget)

        # -- Disk panel: center (x, y) + radius --
        self.geom_disk_cx = self._spin(-1e6, 1e6, 0.5)
        self.geom_disk_cy = self._spin(-1e6, 1e6, 0.5)
        self.geom_disk_r = self._spin(1e-6, 1e6, 0.5)
        self.geom_panel_disk = QWidget()
        _p = QVBoxLayout(self.geom_panel_disk); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(4)
        _p.addWidget(self._domain_row("Center:", self.geom_disk_cx, self.geom_disk_cy, tip="Center (x, y)"))
        _p.addWidget(self._domain_row("Radius:", self.geom_disk_r))
        self.geom_panel_disk.setVisible(False)
        domain_layout.addWidget(self.geom_panel_disk)

        # -- Ellipse panel: center (x, y) + semi-axes + rotation angle --
        self.geom_ellipse_cx = self._spin(-1e6, 1e6, 0.5)
        self.geom_ellipse_cy = self._spin(-1e6, 1e6, 0.5)
        self.geom_ellipse_a = self._spin(1e-6, 1e6, 0.5)
        self.geom_ellipse_b = self._spin(1e-6, 1e6, 0.3)
        self.geom_ellipse_angle = self._spin(-100, 100, 0.0)
        self.geom_panel_ellipse = QWidget()
        _p = QVBoxLayout(self.geom_panel_ellipse); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(4)
        _p.addWidget(self._domain_row("Center:", self.geom_ellipse_cx, self.geom_ellipse_cy, tip="Center (x, y)"))
        _p.addWidget(self._domain_row("Semi-axes:", self.geom_ellipse_a, self.geom_ellipse_b,
                                      tip="Semi-major and semi-minor axis"))
        _p.addWidget(self._domain_row("Angle (rad):", self.geom_ellipse_angle,
                                      tip="Rotation of the major axis from the x axis, in radians;\n"
                                                    "a positive angle turns it clockwise (DeepXDE's convention)"))
        self.geom_panel_ellipse.setVisible(False)
        domain_layout.addWidget(self.geom_panel_ellipse)

        # -- Triangle / Polygon panels: vertices as "x1,y1;x2,y2;..." --
        self.geom_triangle_verts_input = QLineEdit("0,0;1,0;0,1")
        self.geom_triangle_verts_input.setFixedHeight(26)
        self.geom_panel_triangle = self._domain_row(
            "Vertices:", self.geom_triangle_verts_input,
            tip="Three vertices as x,y pairs separated by semicolons, e.g. 0,0; 1,0; 0,1")
        self.geom_panel_triangle.setVisible(False)
        domain_layout.addWidget(self.geom_panel_triangle)

        self.geom_polygon_verts_input = QLineEdit("0,0;1,0;1,0.6;0.5,1;0,0.6")
        self.geom_polygon_verts_input.setFixedHeight(26)
        self.geom_panel_polygon = self._domain_row(
            "Vertices:", self.geom_polygon_verts_input,
            tip="Three or more vertices as x,y pairs separated by semicolons, in order around the boundary, "
                "e.g. 0,0; 1,0; 1,0.6; 0.5,1; 0,0.6")
        self.geom_panel_polygon.setVisible(False)
        domain_layout.addWidget(self.geom_panel_polygon)

        # -- Sphere panel: center (x, y, z) + radius --
        self.geom_sphere_cx = self._spin(-1e6, 1e6, 0.5)
        self.geom_sphere_cy = self._spin(-1e6, 1e6, 0.5)
        self.geom_sphere_cz = self._spin(-1e6, 1e6, 0.5)
        self.geom_sphere_r = self._spin(1e-6, 1e6, 0.5)
        self.geom_panel_sphere = QWidget()
        _p = QVBoxLayout(self.geom_panel_sphere); _p.setContentsMargins(0, 0, 0, 0); _p.setSpacing(4)
        _p.addWidget(self._domain_row("Center:", self.geom_sphere_cx, self.geom_sphere_cy, self.geom_sphere_cz,
                                      tip="Center (x, y, z)"))
        _p.addWidget(self._domain_row("Radius:", self.geom_sphere_r))
        self.geom_panel_sphere.setVisible(False)
        domain_layout.addWidget(self.geom_panel_sphere)

        # -- Custom panel (2D only): a list of primitive shapes, each
        # after the first combined with the running result via a boolean
        # op (Union/Subtract/Intersect) -- see _add_custom_geom_shape().
        # This is how a shape outside the fixed list above (e.g. the
        # triangle-with-a-circular-hole in the reference image) gets
        # built: as a small CSG chain of the primitives already offered
        # elsewhere in this panel, rather than as a wholly new shape kind.
        self.geom_panel_custom = QWidget()
        _cp = QVBoxLayout(self.geom_panel_custom)
        _cp.setContentsMargins(0, 0, 0, 0); _cp.setSpacing(4)
        _cp.addWidget(QLabel("Shapes (combined top to bottom):"))
        self.custom_geom_shapes_container = QWidget()
        self.custom_geom_shapes_layout = QVBoxLayout(self.custom_geom_shapes_container)
        self.custom_geom_shapes_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_geom_shapes_layout.setSpacing(2)
        _cp.addWidget(self.custom_geom_shapes_container)
        custom_geom_add_btn = QPushButton("+ Add Shape")
        custom_geom_add_btn.setFixedHeight(26)
        custom_geom_add_btn.clicked.connect(lambda: self._add_custom_geom_shape())
        _cp.addWidget(custom_geom_add_btn)
        self.geom_panel_custom.setVisible(False)
        domain_layout.addWidget(self.geom_panel_custom)
        self.custom_geom_shape_rows = []

        self._geom_shape_panels = [
            self.geom_panel_disk, self.geom_panel_ellipse, self.geom_panel_triangle,
            self.geom_panel_polygon, self.geom_panel_sphere, self.geom_panel_custom,
        ]
        self.geom_triangle_verts_input.textChanged.connect(self._on_geom_vertices_changed)
        self.geom_polygon_verts_input.textChanged.connect(self._on_geom_vertices_changed)

        self.t_row_widget = QWidget()
        row2 = QHBoxLayout(self.t_row_widget)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(self._domain_label("t:")); row2.setSpacing(6)
        self.t_min = DomainSpinBox()
        self.t_min.setRange(0.0, 1e6); self.t_min.setValue(0.0); self.t_min.setSingleStep(0.5)
        self.t_max = DomainSpinBox()
        self.t_max.setRange(0.0, 1e6); self.t_max.setValue(1.0); self.t_max.setSingleStep(0.5)
        # DomainSpinBox keeps 10 decimals, so a time domain that isn't a
        # round number -- e.g. 1D Schrodinger's t in [0, pi/2] -- is stored
        # to full precision rather than rounded to 1.57.
        self.t_min.setFixedWidth(_DOMAIN_SPIN_WIDTH); self.t_max.setFixedWidth(_DOMAIN_SPIN_WIDTH)
        row2.addWidget(self.t_min); row2.addWidget(QLabel("to")); row2.addWidget(self.t_max); row2.addStretch()
        domain_layout.addWidget(self.t_row_widget)

        # -- Steady-state (time-independent) toggle, e.g. a Poisson
        # equation -- no time axis at all: no t domain, no Initial
        # Condition, no Time Adaptive Training (inherently time-stepped).
        # RAR has no such dependency on a time axis -- it just resamples
        # collocation points toward wherever the PDE residual is currently
        # largest, which works the same with or without a time dimension
        # -- so it stays available for a steady-state problem too. See
        # _on_steady_state_changed() for what else this hides/resets.
        # No longer shown here (or added to any layout) -- the visible
        # control for this is now the Time Dependent/Stationary radio
        # pair in the Problem Type section above (self.radio_stationary/
        # radio_time_dependent), which this stays two-way synced with so
        # every existing isChecked()/setChecked()/toggled call site for
        # this checkbox elsewhere in this file keeps working unchanged.
        self.steady_state_check = QCheckBox("Steady-state (no time axis, e.g. Poisson equation)")
        self.steady_state_check.setVisible(False)
        self.steady_state_check.toggled.connect(self._on_steady_state_changed)
        self.steady_state_check.toggled.connect(
            lambda checked: self.radio_stationary.setChecked(True) if checked
            else self.radio_time_dependent.setChecked(True))
        self.radio_stationary.toggled.connect(self.steady_state_check.setChecked)
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

        # Minimums low enough for DeepXDE's own small examples (e.g. 16
        # domain / 2 boundary points for its 1D Poisson problems).
        self.num_domain   = _pts_row("Domain points:",   2000, 1,   200000, 500)
        self.num_boundary = _pts_row("Boundary points:", 200,  0,   50000,  100)
        self.num_initial  = _pts_row("Initial points:",  200,  0,   50000,  100)
        self.num_test     = _pts_row("Test points:",     1000, 10,  200000, 500)

        pts_dist_row = QHBoxLayout()
        pts_dist_row.addWidget(QLabel("Point distribution:"))
        self.pts_dist_combo = QComboBox()
        # "pseudo" is DeepXDE's own accepted name for this (see
        # deepxde/geometry/sampler.py) -- it used to be listed here as
        # "pseudorandom", which DeepXDE actually rejects with
        # ValueError: pseudorandom sampling is not available.
        self.pts_dist_combo.addItems(["Hammersley", "uniform", "Halton", "LHS", "Sobol", "pseudo"])
        self.pts_dist_combo.setFixedHeight(28)
        pts_dist_row.addStretch()
        pts_dist_row.addWidget(self.pts_dist_combo)
        points_layout.addLayout(pts_dist_row)

        # Extra training points ("anchors", Round 65): always included in
        # training on top of the sampled ones -- from a points file, or
        # random points concentrated in a box (e.g. where the solution
        # changes fastest).
        anchors_row = QHBoxLayout()
        anchors_row.addWidget(QLabel("Extra training points:"))
        self.anchors_mode_combo = QComboBox()
        self.anchors_mode_combo.addItem("None", "none")
        self.anchors_mode_combo.addItem("From a file", "file")
        self.anchors_mode_combo.addItem("Random in a box", "box")
        self.anchors_mode_combo.setFixedHeight(28)
        anchors_row.addStretch(); anchors_row.addWidget(self.anchors_mode_combo)
        points_layout.addLayout(anchors_row)
        self.anchors_file_widget = QWidget()
        af_row = QHBoxLayout(self.anchors_file_widget)
        af_row.setContentsMargins(12, 0, 0, 0)
        self.anchors_file_edit = QLineEdit()
        self.anchors_file_edit.setPlaceholderText("points file: one column per input (x[, y[, z]][, t])")
        self.anchors_file_edit.setFixedHeight(26)
        af_browse = QPushButton("Browse")
        af_browse.setFixedHeight(26)
        af_browse.clicked.connect(lambda: self._browse_into(self.anchors_file_edit, "Extra training points file",
                                                            "Data (*.txt *.dat *.csv);;All files (*)"))
        af_row.addWidget(self.anchors_file_edit); af_row.addWidget(af_browse)
        points_layout.addWidget(self.anchors_file_widget)
        self.anchors_box_widget = QWidget()
        ab_layout = QHBoxLayout(self.anchors_box_widget)
        ab_layout.setContentsMargins(12, 0, 0, 0)
        ab_layout.addWidget(QLabel("Count:"))
        self.anchors_count_spin = QSpinBox()
        self.anchors_count_spin.setRange(1, 1000000); self.anchors_count_spin.setValue(1000)
        self.anchors_count_spin.setFixedHeight(26); self.anchors_count_spin.setFixedWidth(90)
        ab_layout.addWidget(self.anchors_count_spin)
        ab_layout.addWidget(QLabel("Ranges:"))
        self.anchors_box_edit = QLineEdit()
        self.anchors_box_edit.setPlaceholderText("lo,hi per input, ';'-separated; blank = whole domain")
        self.anchors_box_edit.setFixedHeight(26)
        ab_layout.addWidget(self.anchors_box_edit)
        points_layout.addWidget(self.anchors_box_widget)

        def _sync_anchors(_i=None):
            m = self.anchors_mode_combo.currentData()
            self.anchors_file_widget.setVisible(m == "file")
            self.anchors_box_widget.setVisible(m == "box")
        self.anchors_mode_combo.currentIndexChanged.connect(_sync_anchors)
        _sync_anchors()

        self.view_domain_check = QCheckBox("View domain && point distribution")
        self.view_domain_check.setChecked(False)
        self.view_domain_check.setVisible(False)
        self.view_domain_check.stateChanged.connect(self._on_view_domain_changed)
        points_layout.addWidget(self.view_domain_check)
        left_layout.addWidget(points_group)

        # ── Boundary & Initial conditions ─────────────────────
        self.bc_group = QGroupBox("Initial Condition")
        _ic_group_top_layout = QVBoxLayout(self.bc_group)
        _ic_group_top_layout.setSpacing(4)

        # One shared "Show IC reference" toggle for the whole panel -- shown
        # once here rather than once per output (which it used to be,
        # duplicated under every output's own IC box). The reference text
        # only depends on the problem's dimension (1D/2D/3D), not which
        # output it's under, so one copy covers all of them; its text is
        # kept up to date by _build_bc_inputs() whenever dimension changes.
        self.ic_hint_toggle = QCheckBox("📖 Show IC reference")
        self.ic_hint_toggle.setChecked(False)
        self._register_hint(self.ic_hint_toggle, '#74c0fc')
        _ic_group_top_layout.addWidget(self.ic_hint_toggle)
        self.ic_hint = QLabel()
        self._register_hint(self.ic_hint, '#74c0fc')
        self.ic_hint.setWordWrap(True)
        self.ic_hint.setVisible(False)
        _ic_group_top_layout.addWidget(self.ic_hint)
        self.ic_hint_toggle.stateChanged.connect(lambda s: self.ic_hint.setVisible(s == 2))

        # Nested container whose own layout (bc_main_layout) is what
        # _build_bc_inputs() freely clears and rebuilds every time the
        # number of outputs, dimension, or template changes -- the two
        # widgets above live in the group box's own top-level layout
        # instead, so they're never touched by that rebuild.
        _ic_rows_widget = QWidget()
        self.bc_main_layout = QVBoxLayout(_ic_rows_widget)
        self.bc_main_layout.setSpacing(4)
        _ic_group_top_layout.addWidget(_ic_rows_widget)

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
        self._register_hint(self.custom_bc_note, '#74c0fc')
        self.custom_bc_note.setWordWrap(True)
        self.custom_bc_main_layout.addWidget(self.custom_bc_note)

        # One shared "Where" hint for the whole panel -- shown once here
        # rather than repeated on every single BC row (which it used to be;
        # each row had its own collapsed toggle). The hint text itself
        # doesn't depend on which row you're looking at, so one copy is
        # enough; each row's own "Where:" field still has a short inline
        # placeholder example of its own.
        self.bc_loc_hint_toggle = QCheckBox("📖 Show location examples")
        self.bc_loc_hint_toggle.setChecked(False)
        self._register_hint(self.bc_loc_hint_toggle, '#74c0fc')
        self.custom_bc_main_layout.addWidget(self.bc_loc_hint_toggle)
        self.bc_loc_hint = QLabel(
            "\"Where\" picks WHICH part of the boundary a condition applies to.\n"
            "It's only checked at points already on the boundary, so you never\n"
            "need to detect \"on the boundary\" yourself -- only say which part.\n"
            "Use the picker, or type any True/False expression:\n"
            "True                  → the whole boundary\n"
            "left / right          → x = x_min / x = x_max (your domain's own limits)\n"
            "bottom / top          → y = y_min / y = y_max (2D/3D)\n"
            "front / back          → z = z_min / z = z_max (3D)\n"
            "left or right         → both ends\n"
            "x <= x_min            → same as left; x_min, x_max, y_min, ... are\n"
            "                        defined in every BC expression, Value included\n"
            "np.isclose(x**2 + y**2, 0.25) → a circle of radius 0.5 (for shapes\n"
            "                        without named sides, e.g. a hole)\n"
            "Neumann/Robin values are du/dn, the derivative along the OUTWARD\n"
            "normal: at a right/top side that is +du/dx or +du/dy, at a\n"
            "left/bottom side it is -du/dx or -du/dy."
        )
        self._register_hint(self.bc_loc_hint, '#74c0fc')
        self.bc_loc_hint.setWordWrap(True)
        self.bc_loc_hint.setVisible(False)
        self.custom_bc_main_layout.addWidget(self.bc_loc_hint)
        self.bc_loc_hint_toggle.stateChanged.connect(lambda s: self.bc_loc_hint.setVisible(s == 2))

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

    def _build_network_panel(self, left_layout):
        """Neural Network panel: size, network type, input/output transforms."""
        # ── Neural Network ────────────────────────────────────
        nn_group = QGroupBox("Neural Network")
        nn_layout = QVBoxLayout(nn_group)
        nn_layout.setSpacing(6)

        def _nn_row(label, widget):
            # In its own container widget, so a row can be shown/hidden.
            _w = QWidget()
            row = QHBoxLayout(_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(label))
            row.addStretch()
            row.addWidget(widget)
            nn_layout.addWidget(_w)
            return _w

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 8); self.layers_spin.setValue(3)
        self.layers_spin.setFixedWidth(100); self.layers_spin.setFixedHeight(28)
        _nn_row("Hidden layers:", self.layers_spin)

        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(8, 512); self.neurons_spin.setSingleStep(8); self.neurons_spin.setValue(64)
        self.neurons_spin.setFixedWidth(100); self.neurons_spin.setFixedHeight(28)
        _nn_row("Neurons per layer:", self.neurons_spin)

        self.activation_combo = QComboBox()
        # Every identifier DeepXDE's own deepxde.nn.activations.get() dict
        # supports (elu/gelu/relu/selu/sigmoid/silu/sin/swish/tanh -- see
        # that module's source), spelled exactly the way DeepXDE's own
        # get() docstring capitalizes them ("ELU, GELU, ReLU, SELU,
        # Sigmoid, SiLU, sin, Swish, tanh") so the GUI's labels match
        # DeepXDE's own naming convention instead of an ad hoc lowercase
        # list. get() itself lowercases whatever string it's handed before
        # looking it up, so this is a pure labeling/completeness fix --
        # nothing about how the chosen activation reaches DeepXDE changes.
        # "swish" and "SiLU" are DeepXDE's own two names for the identical
        # function (bkd.silu) -- both kept as separate choices since
        # DeepXDE documents both spellings itself. "sin" is included
        # un-capitalized, matching DeepXDE's own docstring, and is
        # specifically relevant for PINNs (SIREN-style sinusoidal
        # activations are a known good fit for problems with smooth/
        # periodic solutions). See _ACTIVATION_CHOICES/
        # _canonical_activation_label() for the shared source of truth.
        self.activation_combo.addItems(self._ACTIVATION_CHOICES)
        self.activation_combo.setFixedHeight(28)
        self._fit_combo_width(self.activation_combo, min_width=100)
        _nn_row("Activation:", self.activation_combo)

        self.kernel_init_combo = QComboBox()
        self.kernel_init_combo.addItems(
            ["Glorot uniform", "Glorot normal", "He uniform", "He normal", "zeros"])
        self.kernel_init_combo.setFixedHeight(28)
        self._fit_combo_width(self.kernel_init_combo, min_width=130)
        _nn_row("Kernel initializer:", self.kernel_init_combo)

        # ── Network type (FNN / PFNN) ──────────────────────────
        # PFNN (DeepXDE's own built-in dde.nn.PFNN class) gives each
        # output its own parallel sub-network, merging only at the final
        # layer, instead of FNN's single shared trunk -- helpful for
        # multi-output problems where outputs have very different scales/
        # behavior. PFNN has no `regularization` kwarg in DeepXDE, so
        # Weight Decay is ignored (and codegen suppresses the arg
        # entirely) when this is PFNN.
        self.network_type_combo = QComboBox()
        self.network_type_combo.addItems(["FNN", "PFNN", "MsFFN", "STMsFFN"])
        self.network_type_combo.setToolTip(
            "FNN: one fully-connected network.\n"
            "PFNN: a separate sub-network for each output.\n"
            "MsFFN: multi-scale Fourier feature network -- random Fourier features of the inputs at\n"
            "  several scales (sigma), for solutions with both slow and fast variations.\n"
            "STMsFFN: the same with separate space and time features (time-dependent problems).")
        self.network_type_combo.setFixedHeight(28)
        self._fit_combo_width(self.network_type_combo, min_width=100)
        _nn_row("Network type:", self.network_type_combo)
        # Fourier-feature scales (Round 73), shown for MsFFN/STMsFFN.
        self.msffn_sigmas_edit = QLineEdit("1, 10")
        self.msffn_sigmas_edit.setFixedHeight(28)
        self.msffn_sigmas_edit.setToolTip("Scales sigma of the random Fourier features, comma-separated, e.g. 1, 10.\n"
                                          "Large sigma lets the network represent fast oscillations.")
        self.msffn_sigmas_row = _nn_row("Fourier scales (σ):", self.msffn_sigmas_edit)
        self.msffn_sigmas_t_edit = QLineEdit("1, 10")
        self.msffn_sigmas_t_edit.setFixedHeight(28)
        self.msffn_sigmas_t_edit.setToolTip("Scales sigma of the time features (STMsFFN), comma-separated.")
        self.msffn_sigmas_t_row = _nn_row("Time scales (σₜ):", self.msffn_sigmas_t_edit)
        self.network_type_combo.currentTextChanged.connect(self._on_network_type_changed)
        self._on_network_type_changed(self.network_type_combo.currentText())

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
        # Mode (Round 62): "Scale/shift" rows, or a formula giving the
        # network's input features (DeepXDE's apply_feature_transform).
        self.input_transform_mode_combo, in_scale_w, in_formula_w = self._transform_mode_widgets(it_layout)
        it_scale_layout = QVBoxLayout(in_scale_w)
        it_scale_layout.setSpacing(3); it_scale_layout.setContentsMargins(0, 0, 0, 0)
        it_hint = QLabel("x_transformed = x_raw × scale + shift")
        self._register_hint(it_hint, '#586e75')
        it_scale_layout.addWidget(it_hint)
        self.input_transform_rows_layout = QVBoxLayout()
        self.input_transform_rows_layout.setSpacing(3)
        it_scale_layout.addLayout(self.input_transform_rows_layout)
        self.input_transform_rows = []
        it_formula_layout = QVBoxLayout(in_formula_w)
        it_formula_layout.setSpacing(3); it_formula_layout.setContentsMargins(0, 0, 0, 0)
        it_f_hint = QLabel("Network input features, separated by ';', written in the coordinates "
                           "x, y, z, t -- e.g. x*sin(y); x*cos(y). The first layer takes one input per feature.")
        it_f_hint.setWordWrap(True)
        self._register_hint(it_f_hint, '#586e75')
        it_formula_layout.addWidget(it_f_hint)
        self.input_feature_formula_edit = QLineEdit()
        self.input_feature_formula_edit.setPlaceholderText("e.g. x*sin(y); x*cos(y)")
        self.input_feature_formula_edit.setFixedHeight(26)
        it_formula_layout.addWidget(self.input_feature_formula_edit)
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
        # Mode (Round 62): "Scale/shift" rows, or one formula per output --
        # DeepXDE's apply_output_transform, typically a hard constraint.
        self.output_transform_mode_combo, out_scale_w, out_formula_w = self._transform_mode_widgets(ot_layout)
        ot_scale_layout = QVBoxLayout(out_scale_w)
        ot_scale_layout.setSpacing(3); ot_scale_layout.setContentsMargins(0, 0, 0, 0)
        ot_hint = QLabel("y_transformed = y_raw × scale + shift")
        self._register_hint(ot_hint, '#586e75')
        ot_scale_layout.addWidget(ot_hint)
        self.output_transform_rows_layout = QVBoxLayout()
        self.output_transform_rows_layout.setSpacing(3)
        ot_scale_layout.addLayout(self.output_transform_rows_layout)
        self.output_transform_rows = []
        ot_formula_layout = QVBoxLayout(out_formula_w)
        ot_formula_layout.setSpacing(3); ot_formula_layout.setContentsMargins(0, 0, 0, 0)
        ot_f_hint = QLabel("One formula per output, in the coordinates x, y, z, t and the raw network "
                           "output by name -- e.g. a hard constraint u = 0 on x = 0 and x = 1:  x*(1-x)*u. "
                           "Blank leaves that output unchanged.")
        ot_f_hint.setWordWrap(True)
        self._register_hint(ot_f_hint, '#586e75')
        ot_formula_layout.addWidget(ot_f_hint)
        self.output_formula_rows_layout = QVBoxLayout()
        self.output_formula_rows_layout.setSpacing(3)
        ot_formula_layout.addLayout(self.output_formula_rows_layout)
        self.output_formula_edits = []
        nn_layout.addWidget(self.output_transform_widget)
        self.output_transform_widget.setVisible(False)
        self.output_transform_cb.stateChanged.connect(
            lambda: self.output_transform_widget.setVisible(self.output_transform_cb.isChecked()))

        self._rebuild_input_transform_rows()
        self._rebuild_output_transform_rows(1)

        left_layout.addWidget(nn_group)

    def _build_training_panel(self, left_layout):
        """Training panel: optimizers, scheduler, callbacks, loss weights."""
        # ── Training ─────────────────────────────────────────────────
        # (This group previously also had a "Mini-batch training" checkbox
        # here -- removed. DeepXDE's dde.data.PDE/TimePDE.train_next_batch
        # ignores the batch_size argument entirely for this data type
        # (confirmed against the installed DeepXDE source), and DeepXDE's
        # own docs say explicitly not to use batch_size with PDE/TimePDE --
        # use dde.callbacks.PDEPointResampler instead, which this app
        # already implements as its "RAR" adaptive-refinement feature. The
        # checkbox and its `data.batch_size = N` codegen were a complete
        # no-op: every run trained full-batch regardless of the setting.)
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(5)
        train_layout.setContentsMargins(10, 10, 10, 10)

        def _train_row(label, widget):
            train_layout.addWidget(QLabel(label))
            train_layout.addWidget(widget)


        self.ic_pretrain_cb = QCheckBox("Enable IC-guided pre-training")
        self.ic_pretrain_cb.setChecked(False)
        self._register_hint(self.ic_pretrain_cb, '#69db7c')
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
        # Same rows, labels and order as a Training Phase: Optimizer,
        # Iterations, Learning rate, Learning rate decay, Loss function --
        # then the two IC-only point counts.
        ic_opt_row = QHBoxLayout()
        ic_opt_row.addWidget(QLabel("Optimizer:"))
        self.ic_pretrain_opt = QComboBox()
        self.ic_pretrain_opt.addItem("Adam", "adam")
        self.ic_pretrain_opt.setFixedHeight(26)
        self._fit_combo_width(self.ic_pretrain_opt, min_width=80)
        ic_opt_row.addStretch(); ic_opt_row.addWidget(self.ic_pretrain_opt)
        ic_train_fields_layout.addLayout(ic_opt_row)

        ic_iter_row = QHBoxLayout()
        ic_iter_row.addWidget(QLabel("Iterations:"))
        self.ic_pretrain_iters = QSpinBox()
        self.ic_pretrain_iters.setRange(100, 500000)
        self.ic_pretrain_iters.setSingleStep(1000)
        self.ic_pretrain_iters.setValue(20000)
        self.ic_pretrain_iters.setFixedHeight(26)
        ic_iter_row.addStretch(); ic_iter_row.addWidget(self.ic_pretrain_iters)
        ic_train_fields_layout.addLayout(ic_iter_row)

        # IC pre-training has its own learning rate (it used to silently
        # reuse the main training's), so it can be tuned independently.
        ic_lr_row = QHBoxLayout()
        ic_lr_row.addWidget(QLabel("Learning rate:"))
        self.ic_pretrain_lr = QDoubleSpinBox()
        self.ic_pretrain_lr.setRange(1e-6, 1.0); self.ic_pretrain_lr.setDecimals(6)
        self.ic_pretrain_lr.setSingleStep(0.0001); self.ic_pretrain_lr.setValue(0.001)
        self.ic_pretrain_lr.setFixedHeight(26)
        ic_lr_row.addStretch(); ic_lr_row.addWidget(self.ic_pretrain_lr)
        ic_train_fields_layout.addLayout(ic_lr_row)

        (self.ic_pretrain_decay_widget, self.ic_pretrain_decay_type,
         self.ic_pretrain_decay_p1, self.ic_pretrain_decay_p2) = self._make_lr_decay_widget()
        ic_train_fields_layout.addWidget(self.ic_pretrain_decay_widget)

        # Loss function -- same option list as a Training Phase's.
        ic_loss_row = QHBoxLayout()
        ic_loss_row.addWidget(QLabel("Loss function:"))
        self.ic_pretrain_loss_combo = QComboBox()
        self.ic_pretrain_loss_combo.addItems(
            ["MSE", "MAE", "mean l2 relative error",
             "mean absolute percentage error", "softplus"])
        self.ic_pretrain_loss_combo.setFixedHeight(26)
        self._fit_combo_width(self.ic_pretrain_loss_combo, min_width=160)
        ic_loss_row.addStretch(); ic_loss_row.addWidget(self.ic_pretrain_loss_combo)
        ic_train_fields_layout.addLayout(ic_loss_row)

        # Test points
        ic_test_row = QHBoxLayout()
        ic_test_row.addWidget(QLabel("Test points:"))
        self.ic_pretrain_test = QSpinBox()
        self.ic_pretrain_test.setRange(100, 100000)
        self.ic_pretrain_test.setSingleStep(1000)
        self.ic_pretrain_test.setValue(10000)
        self.ic_pretrain_test.setFixedHeight(26)
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
        self.ic_pretrain_init.setFixedHeight(26)
        ic_init_row.addStretch(); ic_init_row.addWidget(self.ic_pretrain_init)
        ic_train_fields_layout.addWidget(self.ic_pretrain_init_widget)
        ic_pt_layout.addWidget(self.ic_pretrain_train_fields_widget)

        # Restore-from-checkpoint field (path only -- see mode toggle above)
        self.ic_pretrain_restore_widget = QWidget()
        ic_restore_layout = QHBoxLayout(self.ic_pretrain_restore_widget)
        ic_restore_layout.setContentsMargins(0, 0, 0, 0)
        self.ic_pretrain_restore_path = QLineEdit()
        # Only an Adam-trained checkpoint can be restored here (the "IC
        # pre-train optimizer" dropdown above only ever offers Adam, and
        # restoring just loads net.state_dict() -- no optimizer state --
        # so the file itself must genuinely have come from an Adam IC
        # pre-training run to have valid, matching weights).
        self.ic_pretrain_restore_path.setPlaceholderText(
            "Browse for Adam-trained IC pre-train model (.pt)...")
        self.ic_pretrain_restore_path.setFixedHeight(26)
        ic_restore_layout.addWidget(self.ic_pretrain_restore_path)
        ic_restore_browse = QPushButton("Browse")
        ic_restore_browse.setFixedHeight(26); ic_restore_browse.setFixedWidth(65)
        ic_restore_browse.clicked.connect(lambda: self.ic_pretrain_restore_path.setText(
            QFileDialog.getOpenFileName(None, "Select Adam-trained IC pre-train model", "", "Model (*.pt)")[0]))
        ic_restore_layout.addWidget(ic_restore_browse)
        ic_pt_layout.addWidget(self.ic_pretrain_restore_widget)

        self._update_ic_pretrain_visibility()
        self.ic_pretrain_widget.setVisible(False)
        train_layout.addWidget(self.ic_pretrain_widget)

        # ── Optimizer Scheduler ───────────────────────────────
        div_sched = QLabel("── Training Phases ──")
        self._register_section_divider(div_sched)
        train_layout.addWidget(div_sched)
        self.sched_cb = QCheckBox("Enable Optimizer Scheduler")
        self.sched_cb.setChecked(True)
        self.sched_cb.setVisible(False)  # always on, hidden
        train_layout.addWidget(self.sched_cb)

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
        self._register_hint(self.sched_same_weights_cb, '#69db7c')
        self.sched_same_weights_cb.stateChanged.connect(
            lambda s: self._build_weight_inputs(self.num_outputs_spin.value()))
        # (placed at the top of the Loss Weights panel, which it controls --
        # see _build_training_panel's Loss Weights section)

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
        self.lbfgs_use_default_cb = QCheckBox("Use recommended L-BFGS settings")
        self.lbfgs_use_default_cb.setToolTip(
            "DeepXDE's default L-BFGS options: maxcor = 100, ftol = 0, gtol = 1e-8, maxls = 50;\n"
            "each L-BFGS phase runs for its own Iterations (maxfun = 1.25 x Iterations).\n"
            "Untick to set them yourself.")
        self.lbfgs_use_default_cb.setChecked(True)
        self.lbfgs_use_default_cb.stateChanged.connect(self._on_lbfgs_default_changed)
        _lbfgs_top_row.addWidget(self.lbfgs_use_default_cb)
        _lbfgs_top_row.addStretch()
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

        _rec = LBFGS_RECOMMENDED
        self.lbfgs_maxcor  = _lbfgs_row("maxcor:",  _rec["maxcor"])
        self.lbfgs_ftol    = _lbfgs_row("ftol:",    _rec["ftol"])
        self.lbfgs_gtol    = _lbfgs_row("gtol:",    _rec["gtol"])
        self.lbfgs_maxiter = _lbfgs_row("maxiter:", _rec["maxiter"])
        self.lbfgs_maxfun  = _lbfgs_row("maxfun:",  _rec["maxfun"])
        self.lbfgs_maxls   = _lbfgs_row("maxls:",   _rec["maxls"])

        self.lbfgs_manual_widget.setVisible(False)
        lbfgs_layout.addWidget(self.lbfgs_manual_widget)
        self.lbfgs_widget.setVisible(False)
        train_layout.addWidget(self.lbfgs_widget)

        # Precision -- its own row under the phases, since float64 here
        # switches the whole run to float64 (codegen's _effective_float),
        # not only the L-BFGS phases.
        _prec_row = QHBoxLayout()
        _prec_lbl = QLabel("Precision:")
        _prec_row.addWidget(_prec_lbl)
        self.lbfgs_float_combo = QComboBox()
        self.lbfgs_float_combo.addItems(["float64", "float32"])
        self.lbfgs_float_combo.setCurrentText("float32")
        self.lbfgs_float_combo.setFixedHeight(26)
        self._fit_combo_width(self.lbfgs_float_combo, min_width=80)
        _prec_tip = ("Floating-point precision for training. float64 runs the whole training in\n"
                     "double precision: slower, but L-BFGS converges further and is less likely\n"
                     "to stop early. float32 is faster.")
        _prec_lbl.setToolTip(_prec_tip); self.lbfgs_float_combo.setToolTip(_prec_tip)
        _prec_row.addStretch(); _prec_row.addWidget(self.lbfgs_float_combo)
        train_layout.addLayout(_prec_row)
        self.opt2_combo.currentTextChanged.connect(self._on_opt2_changed)

        # ── Training Callbacks (optional) ───────────────────────
        # Optional DeepXDE training callbacks -- all off by default. Used
        # to live in their own Settings-menu dialog; now inline here so
        # they're visible alongside every other Training setting instead of
        # behind a popup. Applied to the live Training Phases scheduler
        # (and the legacy single/dual-phase fallback), NOT to IC
        # pre-training or the RAR refinement sub-loop, which are short,
        # purpose-built inner loops of their own (see codegen.py's
        # _train_cbs construction for the full scoping note). Read directly
        # by _build_config() and set directly by _apply_config() -- no
        # intermediate dict, same as every other Training-panel setting.
        div_cb = QLabel("── Training Callbacks ──")
        self._register_section_divider(div_cb)
        train_layout.addWidget(div_cb)

        # The four callback groups below (Early Stopping/Point Resampling/
        # Model Checkpoint/Training Timer) used to always render, even
        # fully unchecked and unused -- which meant every config paid for
        # 4 group boxes' worth of vertical space in the left panel no
        # matter what. They're now tucked inside train_callbacks_container,
        # hidden by default, and only shown once this switch is ticked --
        # same "off by default, nothing to look at until you ask for it"
        # convention as Adaptive Training/Parameter Sweep already use for
        # their own optional configuration. _apply_config() below also
        # ticks this automatically when restoring a saved config that
        # already has any one of the four enabled, so a previously-
        # configured callback is never hidden from view on load.
        self.train_callbacks_show_cb = QCheckBox("Show Training Callbacks")
        self.train_callbacks_show_cb.setChecked(False)
        self.train_callbacks_show_cb.setToolTip(
            "Early Stopping / Point Resampling / Model Checkpoint / "
            "Training Timer -- optional extras most runs don't need, "
            "hidden by default so they don't take up space when unused.")
        train_layout.addWidget(self.train_callbacks_show_cb)

        self.train_callbacks_container = QWidget()
        train_callbacks_layout = QVBoxLayout(self.train_callbacks_container)
        train_callbacks_layout.setContentsMargins(0, 0, 0, 0)
        train_callbacks_layout.setSpacing(5)
        train_layout.addWidget(self.train_callbacks_container)
        self.train_callbacks_container.setVisible(False)
        self.train_callbacks_show_cb.stateChanged.connect(
            lambda s: self.train_callbacks_container.setVisible(s == 2))

        def _cb_row(target_layout, label, widget):
            row = QHBoxLayout(); row.addWidget(QLabel(label))
            row.addStretch(); row.addWidget(widget)
            target_layout.addLayout(row)

        # Early Stopping
        es_group = QGroupBox("Early Stopping")
        es_layout = QVBoxLayout(es_group)
        self.cb_early_stopping_cb = QCheckBox("Stop training when loss stops improving")
        self.cb_early_stopping_cb.setChecked(False)
        es_layout.addWidget(self.cb_early_stopping_cb)
        es_fields = QWidget()
        es_fl = QVBoxLayout(es_fields); es_fl.setContentsMargins(0, 0, 0, 0)
        self.cb_es_min_delta = SciLineEdit(0.0); self.cb_es_min_delta.setFixedWidth(110)
        _cb_row(es_fl, "Min. delta:", self.cb_es_min_delta)
        self.cb_es_patience = QSpinBox(); self.cb_es_patience.setRange(1, 1000000)
        self.cb_es_patience.setSingleStep(100); self.cb_es_patience.setValue(2000)
        self.cb_es_patience.setFixedWidth(110)
        _cb_row(es_fl, "Patience (iters):", self.cb_es_patience)
        self.cb_es_baseline = QLineEdit()
        self.cb_es_baseline.setPlaceholderText("(none)"); self.cb_es_baseline.setFixedWidth(110)
        _cb_row(es_fl, "Baseline loss:", self.cb_es_baseline)
        self.cb_es_monitor = QComboBox()
        self.cb_es_monitor.addItem("Training loss", "loss_train")
        self.cb_es_monitor.addItem("Testing loss", "loss_test")
        self.cb_es_monitor.setFixedWidth(110)
        _cb_row(es_fl, "Monitor:", self.cb_es_monitor)
        self.cb_es_start = QSpinBox(); self.cb_es_start.setRange(0, 1000000)
        self.cb_es_start.setSingleStep(100); self.cb_es_start.setValue(0)
        self.cb_es_start.setFixedWidth(110)
        _cb_row(es_fl, "Start after (iters):", self.cb_es_start)
        es_layout.addWidget(es_fields)
        es_fields.setVisible(False)
        self.cb_early_stopping_cb.stateChanged.connect(lambda s: es_fields.setVisible(s == 2))
        train_callbacks_layout.addWidget(es_group)

        # Point Resampling
        pr_group = QGroupBox("Point Resampling")
        pr_layout = QVBoxLayout(pr_group)
        self.cb_point_resampler_cb = QCheckBox("Periodically resample collocation points")
        self.cb_point_resampler_cb.setChecked(False)
        pr_layout.addWidget(self.cb_point_resampler_cb)
        pr_fields = QWidget()
        pr_fl = QVBoxLayout(pr_fields); pr_fl.setContentsMargins(0, 0, 0, 0)
        self.cb_pr_period = QSpinBox(); self.cb_pr_period.setRange(1, 1000000)
        self.cb_pr_period.setSingleStep(10); self.cb_pr_period.setValue(100)
        self.cb_pr_period.setFixedWidth(110)
        _cb_row(pr_fl, "Resample every (iters):", self.cb_pr_period)
        self.cb_pr_pde = QCheckBox("Resample PDE (domain) points")
        self.cb_pr_pde.setChecked(True)
        pr_fl.addWidget(self.cb_pr_pde)
        self.cb_pr_bc = QCheckBox("Also resample boundary-condition points")
        self.cb_pr_bc.setChecked(False)
        pr_fl.addWidget(self.cb_pr_bc)
        pr_layout.addWidget(pr_fields)
        pr_fields.setVisible(False)
        self.cb_point_resampler_cb.stateChanged.connect(lambda s: pr_fields.setVisible(s == 2))
        train_callbacks_layout.addWidget(pr_group)

        # Model Checkpoint
        ck_group = QGroupBox("Model Checkpoint")
        ck_layout = QVBoxLayout(ck_group)
        self.cb_model_checkpoint_cb = QCheckBox("Periodically save the model during training")
        self.cb_model_checkpoint_cb.setChecked(False)
        ck_layout.addWidget(self.cb_model_checkpoint_cb)
        ck_fields = QWidget()
        ck_fl = QVBoxLayout(ck_fields); ck_fl.setContentsMargins(0, 0, 0, 0)
        self.cb_ck_period = QSpinBox(); self.cb_ck_period.setRange(1, 1000000)
        self.cb_ck_period.setSingleStep(100); self.cb_ck_period.setValue(1000)
        self.cb_ck_period.setFixedWidth(110)
        _cb_row(ck_fl, "Check every (iters):", self.cb_ck_period)
        self.cb_ck_better = QCheckBox("Only save when the monitored loss improves")
        self.cb_ck_better.setChecked(True)
        ck_fl.addWidget(self.cb_ck_better)
        self.cb_ck_monitor = QComboBox()
        self.cb_ck_monitor.addItem("Training loss", "train loss")
        self.cb_ck_monitor.addItem("Testing loss", "test loss")
        self.cb_ck_monitor.setFixedWidth(110)
        _cb_row(ck_fl, "Monitor:", self.cb_ck_monitor)
        ck_layout.addWidget(ck_fields)
        ck_fields.setVisible(False)
        self.cb_model_checkpoint_cb.stateChanged.connect(lambda s: ck_fields.setVisible(s == 2))
        train_callbacks_layout.addWidget(ck_group)

        # Training Timer
        tm_group = QGroupBox("Training Timer")
        tm_layout = QVBoxLayout(tm_group)
        self.cb_timer_cb = QCheckBox("Stop training after a time budget")
        self.cb_timer_cb.setChecked(False)
        tm_layout.addWidget(self.cb_timer_cb)
        tm_fields = QWidget()
        tm_fl = QVBoxLayout(tm_fields); tm_fl.setContentsMargins(0, 0, 0, 0)
        self.cb_tm_minutes = QDoubleSpinBox(); self.cb_tm_minutes.setRange(0.5, 100000)
        self.cb_tm_minutes.setDecimals(1); self.cb_tm_minutes.setValue(60.0)
        self.cb_tm_minutes.setFixedWidth(110)
        _cb_row(tm_fl, "Available time (minutes):", self.cb_tm_minutes)
        tm_layout.addWidget(tm_fields)
        tm_fields.setVisible(False)
        self.cb_timer_cb.stateChanged.connect(lambda s: tm_fields.setVisible(s == 2))
        train_callbacks_layout.addWidget(tm_group)

        # Training Monitors -- pure-observability dde.callbacks.
        # OperatorPredictor instances, watching an expression (plain output
        # names and/or derivative names, same d{name}_x / d{name}_xx syntax
        # the Custom PDE expression box already uses) at a fixed set of
        # points, logged every N iterations to its own file. Has zero
        # effect on training itself -- see config.py's
        # training_monitors_enabled/training_monitors docstring for the
        # full design. self.training_monitor_list mirrors
        # self.sched_phase_list's pattern: a plain list of dicts holding
        # each row's widgets, read by _build_training_monitors_json() /
        # rebuilt by _apply_training_monitors_json().
        tmon_group = QGroupBox("Training Monitors")
        tmon_layout = QVBoxLayout(tmon_group)
        self.training_monitors_cb = QCheckBox("Watch expressions during training (diagnostic only)")
        self.training_monitors_cb.setChecked(False)
        self.training_monitors_cb.setToolTip(
            "Logs the value of one or more expressions -- an output name "
            "(e.g. 'u') and/or a derivative using the same d{name}_x / "
            "d{name}_xx / d{name}_xy syntax the Custom PDE box uses -- at "
            "a fixed set of points, every N iterations. Read-only: never "
            "affects training. Useful for watching whether a quantity "
            "that should settle instead drifts while loss keeps dropping "
            "(e.g. structural non-identifiability).")
        tmon_layout.addWidget(self.training_monitors_cb)

        tmon_fields = QWidget()
        tmon_fields_layout = QVBoxLayout(tmon_fields)
        tmon_fields_layout.setContentsMargins(0, 0, 0, 0)
        tmon_hint = QLabel(
            "Points: one per parenthesized group, semicolon-separated,\n"
            "e.g. (0.3, 0.0); (0.5, 0.0) -- matching this problem's own\n"
            "input order (x[, y][, z][, t]). Expressions: comma-separated,\n"
            "logged together, e.g. u, du_x, du_xx")
        self._register_hint(tmon_hint, '#808090')
        tmon_fields_layout.addWidget(tmon_hint)

        self.tmon_rows_widget = QWidget()
        self.tmon_rows_layout = QVBoxLayout(self.tmon_rows_widget)
        self.tmon_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.tmon_rows_layout.setSpacing(4)
        tmon_fields_layout.addWidget(self.tmon_rows_widget)
        self.training_monitor_list = []

        tmon_add_btn = QPushButton("+ Add Monitor")
        tmon_add_btn.clicked.connect(lambda: self._add_training_monitor_row())
        tmon_fields_layout.addWidget(tmon_add_btn)

        tmon_layout.addWidget(tmon_fields)
        tmon_fields.setVisible(False)
        self.training_monitors_cb.stateChanged.connect(lambda s: tmon_fields.setVisible(s == 2))
        train_callbacks_layout.addWidget(tmon_group)

        left_layout.addWidget(train_group)

        # ── Loss Weights ──────────────────────────────────────
        self.weights_group = QGroupBox("Loss Weights")
        _wg_outer = QVBoxLayout(self.weights_group)
        _wg_outer.setSpacing(4)
        # Round 75: weights from the initial losses (wᵢ = c / Lᵢ(0)).
        _wi_row = QHBoxLayout()
        self.loss_w_init_cb = QCheckBox("Scale by initial losses:  wᵢ = c / Lᵢ(0),  c =")
        self.loss_w_init_cb.setToolTip(
            "Before training, evaluate every loss term once and set its weight to c divided by\n"
            "its initial value, so all terms start at the same size (DeepXDE's wave example uses\n"
            "c = 5). The weights below are then ignored, in every training phase.")
        self.loss_w_init_c = SciLineEdit(5.0); self.loss_w_init_c.setFixedWidth(70)
        self.loss_w_init_c.setEnabled(False)
        self.loss_w_init_cb.toggled.connect(self.loss_w_init_c.setEnabled)
        _wi_row.addWidget(self.loss_w_init_cb); _wi_row.addWidget(self.loss_w_init_c); _wi_row.addStretch()
        _wg_outer.addLayout(_wi_row)
        _wg_outer.addWidget(self.sched_same_weights_cb)
        _wg_inner = QWidget()
        self.weights_main_layout = QVBoxLayout(_wg_inner)
        self.weights_main_layout.setContentsMargins(0, 0, 0, 0)
        self.weights_main_layout.setSpacing(4)
        _wg_outer.addWidget(_wg_inner)
        self.weight_widgets = {}
        self._build_weight_inputs(1)
        left_layout.addWidget(self.weights_group)

    def _build_adaptive_and_sweep_panels(self, left_layout):
        """Adaptive Training (RAR / Time-Adaptive step groups) and Parameter Sweep panels."""
        # ── Adaptive Training ─────────────────────────────────
        adapt_group = QGroupBox("Adaptive Training")
        self.adapt_group = adapt_group
        adapt_layout = QVBoxLayout(adapt_group)
        adapt_layout.setSpacing(5)

        row_a1 = QHBoxLayout()
        row_a1.addWidget(QLabel("Method:"))
        self.adapt_combo = QComboBox()
        self.adapt_combo.addItem("None", "None")
        self.adapt_combo.addItem("Time Adaptive Training", "Time Adaptive")
        self.adapt_combo.addItem("Residual-based Adaptive Refinement (RAR)", "RAR")
        self.adapt_combo.setFixedHeight(28)
        self.adapt_combo.currentTextChanged.connect(self._on_adapt_changed)
        row_a1.addWidget(self.adapt_combo)
        adapt_layout.addLayout(row_a1)

        # Time Adaptive widget
        self.ta_widget = QWidget()
        ta_layout = QVBoxLayout(self.ta_widget)
        ta_layout.setSpacing(4); ta_layout.setContentsMargins(0, 0, 0, 0)
        self.ta_ref_toggle, self.ta_ref_hint = self._add_reference_toggle(
            ta_layout, "Show Time-Adaptive reference", self._TA_REFERENCE_TEXT)

        # ── Step groups ───────────────────────────────────────
        ta_groups_label = QLabel("Time step groups:")
        self._register_hint(ta_groups_label, '#a0c4ff')
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
        # Editable so a user can type their own per-axis resolution (e.g.
        # 5000 for a fine 1D grid) instead of being limited to the 4
        # presets -- same mechanism either way: whatever integer ends up
        # in ta_grid.currentText() is read straight into config.ta_grid_size
        # (see _build_config()) and used as-is for np.linspace's point
        # count (1D), squared (2D), or cubed (3D) -- see codegen.py's
        # Time-Adaptive grid-building block. A QIntValidator keeps typed
        # input to positive integers only; _ta_grid_size() below is the
        # single place that parses it back out safely (falls back to 101
        # on anything unparseable, e.g. while the field is mid-edit/empty).
        self.ta_grid.setEditable(True)
        self.ta_grid.setValidator(QIntValidator(1, 1_000_000, self.ta_grid))
        self.ta_grid.setToolTip(
            "Per-axis resolution for the grid handed between time steps.\n"
            "1D: this many points. 2D: squared. 3D: cubed -- e.g. 101 in "
            "3D is over a million points. Pick a smaller value (11 or 21) "
            "for 3D Time-Adaptive runs.\n"
            "Pick a preset or type your own value -- e.g. typing 5000 for "
            "a 1D problem uses 5000 evenly-spaced points (25,000,000 for "
            "2D since it's squared, so keep custom values modest in 2D/3D)."
        )
        row_ta2.addStretch(); row_ta2.addWidget(self.ta_grid)
        ta_layout.addLayout(row_ta2)

        # Visible in-panel warning (not just the tooltip above, which is
        # easy to miss) for the 3D + large-grid combination that's known
        # to exhaust GPU memory -- confirmed by a real run: an RTX 4090
        # (24GB) hit a CUDA out-of-memory error training 3D Heat with
        # Time-Adaptive at grid=101 (~1.03M points/step), on step 2 of 5.
        # 1D/2D never cube the grid, so they never trigger this.
        self.ta_grid_warning = QLabel(
            "⚠ 3D cubes this (grid³ points/step) -- 101 is over 1 million "
            "points and can exhaust GPU memory during training. Use 11 or "
            "21 for 3D Time-Adaptive runs."
        )
        self.ta_grid_warning.setWordWrap(True)
        self.ta_grid_warning.setStyleSheet(
            "QLabel { color: #ffa94d; background: rgba(255, 169, 77, 0.08); "
            "border: 1px solid #ffa94d; border-radius: 4px; padding: 4px 6px; }"
        )
        self.ta_grid_warning.setVisible(False)
        ta_layout.addWidget(self.ta_grid_warning)
        self.ta_grid.currentTextChanged.connect(self._update_ta_grid_warning)

        # Transfer learning
        self.ta_transfer_cb = QCheckBox("Enable transfer learning (warm start from previous step)")
        self.ta_transfer_cb.setChecked(False)
        self._register_hint(self.ta_transfer_cb, '#69db7c')
        ta_layout.addWidget(self.ta_transfer_cb)

        self.ta_transfer_opt_widget = QWidget()
        tl_row = QHBoxLayout(self.ta_transfer_opt_widget)
        tl_row.setContentsMargins(0, 0, 0, 0)
        tl_row.addWidget(QLabel("Transfer optimizer:"))
        self.ta_transfer_opt = QComboBox()
        self.ta_transfer_opt.addItem("Adam", "adam")
        self.ta_transfer_opt.addItem("L-BFGS", "lbfgs")
        self.ta_transfer_opt.setFixedHeight(26)
        self._fit_combo_width(self.ta_transfer_opt, min_width=80)
        tl_row.addStretch(); tl_row.addWidget(self.ta_transfer_opt)
        self.ta_transfer_opt_widget.setVisible(False)
        ta_layout.addWidget(self.ta_transfer_opt_widget)
        self.ta_transfer_cb.stateChanged.connect(
            lambda s: self.ta_transfer_opt_widget.setVisible(s == 2))

        self.ta_widget.setVisible(False)
        adapt_layout.addWidget(self.ta_widget)

        # RAR widget
        self.rar_widget = QWidget()
        rar_layout = QVBoxLayout(self.rar_widget)
        rar_layout.setSpacing(4); rar_layout.setContentsMargins(0, 0, 0, 0)
        self.rar_ref_toggle, self.rar_ref_hint = self._add_reference_toggle(
            rar_layout, "Show RAR reference", self._RAR_REFERENCE_TEXT)

        def _rar_row(label, default, min_v, max_v, step):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sb = QSpinBox(); sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(default)
            sb.setFixedHeight(28); sb.setFixedWidth(100)
            row.addStretch(); row.addWidget(sb)
            rar_layout.addLayout(row)
            return sb

        # RAR ("Residual-based Adaptive Refinement") periodically samples a
        # large pool of random points, evaluates the current model's PDE
        # residual at each one, and adds the worst-residual ones as new
        # training points -- then retrains for a fixed number of iterations.
        # That's one "round"; it repeats for the configured number of rounds.
        # Limits wide enough for DeepXDE's own Burgers-RAR recipe (one point
        # per round, as many rounds as it takes to reach a tolerance).
        self.rar_cycles      = _rar_row("RAR training rounds:",          3,     1,    1000,   1)
        self.rar_candidates  = _rar_row("Residual sampling points:",     50000, 1000, 500000, 5000)
        self.rar_add_points  = _rar_row("Data points to add per cycle:", 500,   1,    10000,  100)
        self.rar_adam_iters  = _rar_row("Adam iterations:",              20000, 0,    500000, 1000)
        self.rar_lbfgs_iters = _rar_row("L-BFGS iterations:",            10000, 0,    500000, 1000)
        # Early stop (Round 66): end refinement once the mean residual over
        # the sampled candidates is below this; 0 = always run every round.
        rar_tol_row = QHBoxLayout()
        rar_tol_row.addWidget(QLabel("Stop when mean residual <"))
        self.rar_tolerance = QDoubleSpinBox()
        self.rar_tolerance.setRange(0.0, 1e6); self.rar_tolerance.setDecimals(6)
        self.rar_tolerance.setSingleStep(0.001); self.rar_tolerance.setValue(0.0)
        self.rar_tolerance.setSpecialValueText("off")
        self.rar_tolerance.setFixedHeight(28); self.rar_tolerance.setFixedWidth(100)
        rar_tol_row.addStretch(); rar_tol_row.addWidget(self.rar_tolerance)
        rar_layout.addLayout(rar_tol_row)

        # Which equation's residual to rank candidate points by, for a
        # multi-output problem (model.predict(x, operator=pde) returns one
        # residual per governing equation -- "Output N" here means the Nth
        # equation entered in the PDE editor, which is equation order, not
        # a separately tracked output mapping; every current template has
        # exactly one equation per output, in the same order, so this lines
        # up correctly in practice). Refreshed by _on_num_outputs_changed()/
        # _apply_config() alongside plot_output_combo/restore_output_combo
        # -- see those for the shared refresh pattern. "All outputs
        # (combined)" (index 0, config.rar_output_selector = -1) is the
        # default and reproduces the original, unchanged behavior (sum of
        # absolute residuals across every equation).
        rar_out_row = QHBoxLayout()
        rar_out_row.addWidget(QLabel("Points from:"))
        self.rar_output_combo = QComboBox()
        # Matches plot_output_combo's own initial seed (addItems(["Output 1
        # (u)", "Custom..."]) at its creation site) -- num_outputs_spin is
        # set to 1 before _on_num_outputs_changed is even connected, so
        # nothing else repopulates this for the single-output default case.
        self.rar_output_combo.addItems(["All outputs (combined)", "Output 1 (u)"])
        self.rar_output_combo.setFixedHeight(28)
        rar_out_row.addStretch(); rar_out_row.addWidget(self.rar_output_combo)
        rar_layout.addLayout(rar_out_row)

        self.rar_widget.setVisible(False)
        adapt_layout.addWidget(self.rar_widget)

        left_layout.addWidget(adapt_group)

        # ── Parameter Sweep ──────────────────────────────────────
        # Off by default. Everything about a sweep -- enable switch,
        # mode, and the parameter list itself -- lives in this one
        # panel, right after Adaptive Training, the same as every other
        # training setting: no separate tab, no separate results
        # section taking over the right panel. A sweep reuses the main
        # Solve/Stop buttons (see _on_solve/_on_stop) and the main
        # Training Log for progress -- it doesn't need its own Run/
        # Cancel/results UI. It also reuses the main "Save to:" folder
        # and the main plot/export settings below (each run just gets
        # its own subfolder under that same save location, same as
        # before) -- there's nothing sweep-specific left to configure
        # beyond which parameters to vary and how.
        sweep_toggle_group = QGroupBox("Parameter Sweep")
        sweep_toggle_layout = QVBoxLayout(sweep_toggle_group)
        sweep_toggle_layout.setSpacing(5)

        # Top row: Enable, the "📖 Show sweep reference" toggle and
        # Refresh (shown only while the sweep is enabled) side by side.
        sweep_top_row = QHBoxLayout()
        sweep_top_row.setSpacing(10)
        self.sweep_enable_cb = QCheckBox("Enable Parameter Sweep")
        self.sweep_enable_cb.toggled.connect(self._on_sweep_enabled_toggled)
        sweep_top_row.addWidget(self.sweep_enable_cb)
        sweep_ref_toggle = QCheckBox("\U0001F4D6 Show sweep reference")
        sweep_ref_toggle.setChecked(False)
        self._register_hint(sweep_ref_toggle, '#74c0fc')
        sweep_top_row.addWidget(sweep_ref_toggle)
        sweep_top_row.addStretch()
        self.sweep_refresh_btn = QPushButton("Refresh")
        self.sweep_refresh_btn.setToolTip(
            "Re-reads the rest of this panel's current settings to update which parameters can be swept")
        self.sweep_refresh_btn.clicked.connect(lambda: self._refresh_sweep_param_choices())
        from pinnstudio.ui import icons as _icons_s
        _icons_s.apply(self.sweep_refresh_btn, "refresh", "text", size=16)
        self.sweep_refresh_btn.setVisible(False)
        sweep_top_row.addWidget(self.sweep_refresh_btn)
        sweep_toggle_layout.addLayout(sweep_top_row)
        sweep_ref_text = ("Runs the SAME problem multiple times while varying one or more\n"
                        "settings (learning rate, a PDE coefficient, number of iterations, ...),\n"
                        "so you can compare results side by side instead of changing a value\n"
                        "and re-running by hand each time. Each run gets its own numbered\n"
                        "subfolder under the main \"Save to:\" location, with the full normal set\n"
                        "of plots/checkpoints for that one run.\n"
                        "One-at-a-time: one run per value of each swept parameter in turn,\n"
                        "everything else held at its current value.\n"
                        "All combinations: every value of every swept parameter, crossed --\n"
                        "N parameters with k values each means k^N total runs.\n"
                        "Specified combinations: parameter 1's 1st value runs together with\n"
                        "parameter 2's 1st value, and so on -- every parameter needs the same\n"
                        "number of values listed.\n"
                        "\"Points from:\" (where shown) picks which equation's values are read\n"
                        "when a parameter lives inside a PDE coefficient.\n"
                        "Baseline: the box below Mode adds one run with the current settings\n"
                        "unchanged, before the swept values. When the sweep finishes, its folder\n"
                        "also holds sweep_error_vs_parameter.png (error and final loss vs the\n"
                        "swept values) and sweep_summary.csv.")
        sweep_ref_hint = QLabel(sweep_ref_text)
        self._register_hint(sweep_ref_hint, '#74c0fc')
        sweep_ref_hint.setWordWrap(True)
        sweep_ref_hint.setVisible(False)
        sweep_toggle_layout.addWidget(sweep_ref_hint)
        sweep_ref_toggle.stateChanged.connect(lambda state, h=sweep_ref_hint: h.setVisible(state == 2))

        # Everything below (mode, sweep-parameter rows, add/refresh
        # buttons) is only relevant once the sweep is actually enabled --
        # wrapped in its own container so the whole cluster can be
        # shown/hidden with one setVisible() call from
        # _on_sweep_enabled_toggled, instead of taking up panel space
        # (several rows' worth) whenever the sweep is off, which is the
        # default state.
        self.sweep_options_widget = QWidget()
        sweep_options_layout = QVBoxLayout(self.sweep_options_widget)
        sweep_options_layout.setContentsMargins(0, 0, 0, 0)
        sweep_options_layout.setSpacing(5)

        sweep_mode_row = QHBoxLayout()
        sweep_mode_row.addWidget(QLabel("Mode:"))
        self.sweep_mode_combo = QComboBox()
        # long item texts shouldn't set the panel's minimum width
        self.sweep_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sweep_mode_combo.setMinimumContentsLength(18)
        self.sweep_mode_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # "All combinations" / "Specified combinations" is COMSOL's own
        # terminology for its Parametric Sweep feature -- named the
        # same way here (grid/zip underneath) since it's a familiar
        # framing for anyone who's used that kind of tool before.
        self.sweep_mode_combo.addItem("One-at-a-time (vary each parameter separately)", "oat")
        self.sweep_mode_combo.addItem("All combinations (every value of every parameter)", "grid")
        self.sweep_mode_combo.addItem("Specified combinations (parameters' Nth values run together)", "zip")
        self.sweep_mode_combo.addItem("Bayesian search (each run chosen from the results so far)", "bayes")
        self.sweep_mode_combo.setFixedHeight(26)
        self.sweep_mode_combo.setToolTip(
            "One-at-a-time: vary ONE parameter per run, the others held at their current values.\n"
            "All combinations: every value of every parameter, crossed (a Cartesian product/Grid).\n"
            "Specified combinations: parameter 1's 1st value with parameter 2's 1st value, etc. -- "
            "every parameter needs the same number of values.\n"
            "Bayesian search: each run picked by a Gaussian-process model of the\n"
            "results so far (DeepXDE's hyperparameter-optimization example) -- each parameter's\n"
            "Min/Max (linear or log) is its search range, a list its choices; Steps is not used.")
        sweep_mode_row.addWidget(self.sweep_mode_combo)
        sweep_options_layout.addLayout(sweep_mode_row)

        # Baseline run (the current settings, unchanged) before the swept
        # values. Follows the mode and the number of parameters (on for
        # One-at-a-time with 2+ parameters and Bayesian search, where the
        # other runs are compared against it) until the user clicks it;
        # then their choice is kept (config "on"/"off").
        self.sweep_baseline_cb = QCheckBox("Also run the current settings (baseline)")
        self.sweep_baseline_cb.setToolTip(
            "Runs the panel's settings unchanged first, as a reference point.\n"
            "On by default for One-at-a-time with 2+ parameters and for Bayesian search\n"
            "(its first point); off otherwise, so only the swept values run.")
        self._sweep_baseline_user = None   # None = automatic, else True/False
        self.sweep_baseline_cb.clicked.connect(self._on_sweep_baseline_clicked)
        sweep_options_layout.addWidget(self.sweep_baseline_cb)
        self.sweep_mode_combo.currentIndexChanged.connect(lambda _i: self._update_sweep_baseline_default())
        self._update_sweep_baseline_default()

        # Round 75: Bayesian search settings.
        self.sweep_bayes_widget = QWidget()
        _bl = QHBoxLayout(self.sweep_bayes_widget)
        _bl.setContentsMargins(0, 0, 0, 0); _bl.setSpacing(4)
        _bl.addWidget(QLabel("Runs:"))
        self.sweep_bayes_calls_spin = QSpinBox(); self.sweep_bayes_calls_spin.setRange(2, 1000)
        self.sweep_bayes_calls_spin.setValue(20); self.sweep_bayes_calls_spin.setFixedWidth(60)
        self.sweep_bayes_calls_spin.setToolTip("Total training runs (the baseline included, when it runs)")
        _bl.addWidget(self.sweep_bayes_calls_spin)
        _bl.addWidget(QLabel("Random first:"))
        self.sweep_bayes_init_spin = QSpinBox(); self.sweep_bayes_init_spin.setRange(1, 100)
        self.sweep_bayes_init_spin.setValue(5); self.sweep_bayes_init_spin.setFixedWidth(50)
        self.sweep_bayes_init_spin.setToolTip("Runs with random settings (after the baseline, if any) before the model takes over")
        _bl.addWidget(self.sweep_bayes_init_spin)
        _bl.addWidget(QLabel("Minimize:"))
        self.sweep_bayes_obj_combo = QComboBox()
        self.sweep_bayes_obj_combo.addItem("Final loss", "loss")
        self.sweep_bayes_obj_combo.addItem("L2 error", "l2")
        self.sweep_bayes_obj_combo.setToolTip("L2 error needs Error Analysis reference data (files or a formula)")
        _bl.addWidget(self.sweep_bayes_obj_combo)
        _bl.addStretch()
        sweep_options_layout.addWidget(self.sweep_bayes_widget)
        self.sweep_bayes_widget.setVisible(False)
        self.sweep_mode_combo.currentIndexChanged.connect(
            lambda _i: self.sweep_bayes_widget.setVisible(self.sweep_mode_combo.currentData() == "bayes"))

        # ── Sweep Parameters -- same repeatable add/remove-row pattern
        # as the Training Phases list above (_add_scheduler_phase), and
        # the same stacked-field-per-row shape (a narrow column has no
        # room for one wide line per parameter) rather than v69's
        # single-line-per-row table layout, which needed ~700px of
        # width this column never has.
        div_sweep_params = QLabel("── Sweep Parameters ──")
        self._register_section_divider(div_sweep_params)
        sweep_options_layout.addWidget(div_sweep_params)


        self.sweep_rows_widget = QWidget()
        self.sweep_rows_layout = QVBoxLayout(self.sweep_rows_widget)
        self.sweep_rows_layout.setSpacing(0)
        self.sweep_rows_layout.setContentsMargins(0, 2, 0, 2)
        sweep_options_layout.addWidget(self.sweep_rows_widget)
        self.sweep_row_list = []  # list of dicts with widgets, mirrors sched_phase_list

        sweep_add_row_btn = QPushButton("➕ Add Parameter")
        sweep_add_row_btn.setStyleSheet(
            "QPushButton { color: #69db7c; background: transparent; "
            "border: 1px solid #2a6a4a; border-radius: 4px; padding: 2px 8px; }")
        sweep_add_row_btn.clicked.connect(lambda: self._add_sweep_row())
        sweep_options_layout.addWidget(sweep_add_row_btn)

        sweep_toggle_layout.addWidget(self.sweep_options_widget)
        # Hidden by default -- self.sweep_enable_cb starts unchecked, so
        # these options would otherwise take up panel space for a
        # feature that's off until the user opts in.
        self.sweep_options_widget.setVisible(False)

        left_layout.addWidget(sweep_toggle_group)

        # One default row so the panel isn't empty on first open. Its
        # parameter list is populated again once the Optimizer
        # Scheduler's own default phases exist (those are themselves
        # deferred via QTimer.singleShot(0, ...) in _build_ui, so
        # phase-scoped sweep parameters -- learning rate/iterations/
        # optimizer -- aren't available yet at this exact point in
        # __init__).
        self._add_sweep_row()
        QTimer.singleShot(0, lambda: self._refresh_sweep_param_choices())

    def _build_inverse_panel(self, left_layout):
        """Inverse PINN panel."""
        # ── Inverse PINN panel ────────────────────────────────
        self.inverse_group = QGroupBox("Inverse PINN Settings")
        inv_layout = QVBoxLayout(self.inverse_group)
        inv_layout.setSpacing(5)

        # "📖 Show inverse reference" toggle + hint, at the TOP of this
        # panel (same positioning as the IC/BC/PDE panels' own reference
        # toggles) rather than mid-panel, after the trainable-variable
        # rows -- shown once here, applies to the whole panel.
        inv_ref_row = QHBoxLayout()
        inv_ref_toggle = QCheckBox("\U0001F4D6 Show inverse reference")
        inv_ref_toggle.setChecked(False)
        self._register_hint(inv_ref_toggle, '#74c0fc')
        inv_ref_row.addWidget(inv_ref_toggle)
        inv_ref_row.addStretch()
        inv_ref_row_widget = QWidget()
        inv_ref_row_widget.setLayout(inv_ref_row)
        inv_layout.addWidget(inv_ref_row_widget)
        inv_ref_text = ("Each trainable (unknown) variable is a value the solver infers (fits)\n"
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
        self._register_hint(inv_ref_hint, '#74c0fc')
        inv_ref_hint.setWordWrap(True)
        inv_ref_hint.setVisible(False)
        inv_layout.addWidget(inv_ref_hint)
        inv_ref_toggle.stateChanged.connect(lambda state, h=inv_ref_hint: h.setVisible(state == 2))

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
        self._register_hint(self.inv_multi_var_hint, '#ffd43b')
        self.inv_multi_var_hint.setWordWrap(True)
        self.inv_multi_var_hint.setVisible(False)
        inv_layout.addWidget(self.inv_multi_var_hint)

        # Add the first (primary) trainable variable row -- always present;
        # renaming it keeps a loaded Quick Example's PDE box in sync
        # (INVERSE_AUTO_CONST / _sync_inverse_pde_substitution), same as the
        # single-variable behavior this replaces.
        self._add_inverse_var_row("trainable_variable_1", 1.0)

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
        self._register_hint(self.inv_multi_data_hint, '#ffd43b')
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
        # Index 5 = right after PDE Definition (0 title, 1 examples, 2 dim,
        # 3 type, 4 PDE Definition), before Domain -- trainable variables
        # are conceptually tied to the PDE, so they're shown immediately
        # after it rather than after Domain/Collocation Points.
        left_layout.insertWidget(5, self.inverse_group)  # right after PDE Definition

        # Parametric Study removed (untested, not exposed in the GUI).

    def _build_solve_buttons(self, left_layout):
        """Solve / Stop buttons."""
        # ── Solve / Stop buttons ──────────────────────────────
        # These are the two most important buttons in the app, so they're
        # deliberately sized/weighted up from the shared "Buttons" Display
        # Settings category rather than just inheriting it plain -- but
        # still scaled relative to that category's own font size/family
        # (like the subtitle label does with the "title" category) so they
        # stay in sync if the user changes Display Settings, instead of a
        # flatly hardcoded font-size that would silently stop responding
        # to it (the exact bug class fixed in v14/v21).
        def _cta_css_fn(_css):
            cat = self._disp.get("button", self._DISP_CATEGORY_DEFAULTS.get("button", {}))
            defaults = self._DISP_CATEGORY_DEFAULTS.get("button", {})
            family = cat.get("family") or defaults.get("family", "Segoe UI")
            size = round((cat.get("size") or defaults.get("size", 12)) * 1.25)
            return (f"font-family: '{family}', {_CROSS_PLATFORM_FONT_FALLBACK}; "
                    f"font-size: {size}px; font-weight: bold;")

        self.solve_btn = QPushButton("Solve")
        self.solve_btn.setMinimumHeight(52)
        self._register_style(self.solve_btn, "button", lambda css: f"""
            QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0078d4,stop:1 #005a9e);
                          color: white; {_cta_css_fn(css)} border-radius: 6px; border: none; }}
            QPushButton:hover {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a8ae8,stop:1 #0070c0); }}
            QPushButton:disabled {{ background: #333355; color: #666; }}
        """)
        self.solve_btn.clicked.connect(self._on_solve)
        left_layout.addWidget(self.solve_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setEnabled(False)
        self._register_style(self.stop_btn, "button", lambda css: f"""
            QPushButton {{ background: #6b1f1f; color: white; {_cta_css_fn(css)} border-radius: 6px; border: none; }}
            QPushButton:hover {{ background: #8b2f2f; }}
            QPushButton:disabled {{ background: #333355; color: #666; }}
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        left_layout.addWidget(self.stop_btn)

    def _build_restore_panel(self, left_layout):
        """Model Restore & Visualization panel."""
        # ── Model Restore & Visualization ─────────────────────
        restore_group = QGroupBox("Model Restore && Visualization")
        restore_layout = QVBoxLayout(restore_group)
        restore_layout.setSpacing(5)

        restore_toggle = QCheckBox("🔄 Enable Model Restore && Visualization")
        restore_toggle.setChecked(False)
        self._register_hint(restore_toggle, '#a0c4ff')
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
        # Populated by _detect_restore_ta_steps() when the browsed model
        # file lives inside a Time-Adaptive run's "time_adaptive_steps/
        # step_*/" folder -- see that method and _build_restore_script_ta.
        self._restore_ta_steps = []

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

        # Time-Adaptive step auto-detection (see _detect_restore_ta_steps)
        # -- shown only when the browsed model file lives inside a
        # "time_adaptive_steps/step_*/" folder and at least one sibling
        # step is found alongside it. For a static viz type (Surface /
        # Line (time steps)), the checkbox below has no effect -- restore
        # always auto-routes each requested time to whichever step's
        # model actually covers it, since a single step's model was never
        # valid outside its own training window anyway (see
        # _build_restore_script_ta). It only changes behavior for the two
        # Animation types: checked (default when detected) produces one
        # continuous animation spanning the full time domain by restoring
        # each step's own model in turn; unchecked keeps today's original
        # behavior of restoring only the one step file you browsed to,
        # confined to its own narrow window.
        self.restore_ta_info_label = QLabel("")
        self._register_hint(self.restore_ta_info_label, '#69db7c')
        self.restore_ta_info_label.setWordWrap(True)
        self.restore_ta_info_label.setVisible(False)
        _rmf_layout.addWidget(self.restore_ta_info_label)
        self.restore_ta_combine_cb = QCheckBox("Combine all detected steps into one full-domain animation")
        self.restore_ta_combine_cb.setChecked(True)
        self.restore_ta_combine_cb.setVisible(False)
        _rmf_layout.addWidget(self.restore_ta_combine_cb)

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
        self._fit_combo_width(self.restore_optimizer_combo, min_width=80)
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
            'n_2d_snapshots': 2,
            # Split from one shared "n_steps" into two: a static multi-
            # snapshot line plot ("Line (time steps)") wants few enough
            # lines to stay readable, while an animation wants more
            # frames for a smoother/slower playback -- see
            # _viz_steps_key()/_viz_linewidth_key() for which one a given
            # viz_type actually uses.
            'n_steps_line': 5,
            'n_steps_anim': 20,
            'colorbar': True,
            'levels': 100,
            'resolution': 200,
            'dpi': 300,
            'auto_range': True,
            'vmin': -1.0,
            'vmax': 1.0,
            'linewidth': 2.0,
            'linewidth_anim': 3.0,
            'fps': 10,
            'title': '',
            'xlabel': '',
            'ylabel': '',
            'figsize_mode': 'Default',
            'figsize_w': 7.0,
            'figsize_h': 5.0,
            # y/z slice for the two Line-type viz types ("Line (time
            # steps)"/"Animation Line (GIF)") on a 2D/3D restore -- a
            # genuinely live, independently-overridable control (see
            # _on_restore_viz_settings' slice rows and the matching
            # comment in _build_restore_script), not a readback of
            # whatever the run was originally solved/plotted with.
            # _auto=True (the default) means "domain midpoint of whatever
            # model gets restored", recomputed fresh each time since
            # different restored configs can have different domains.
            'line_slice_y_auto': True,
            'line_slice_y': 0.0,
            'line_slice_z_auto': True,
            'line_slice_z': 0.0,
            # Round 31: same 3 font-size settings as the Setup tab's own
            # Plot Settings dialog (self._plot_viz_settings) -- independent
            # of it, same defaults, same keys (see _on_restore_viz_settings).
            'title_fontsize': 13.0,
            'subplot_title_fontsize': 10.0,
            'axis_label_fontsize': 10.0,
        }

        self.restore_output_widget = QWidget()
        _rout_layout = QVBoxLayout(self.restore_output_widget)
        _rout_layout.setContentsMargins(0, 0, 0, 0)
        _rout_layout.setSpacing(5)
        _rout_layout.addWidget(QLabel("Output to plot:"))
        self.restore_output_combo = QComboBox()
        # "Custom..." mirrors the live Results panel's own Plot output
        # selector (see plot_output_combo/_on_plot_output_combo_changed) --
        # picking it reveals the two fields below instead of a raw output
        # column, so a restored multi-output model can plot any derived
        # field of its own outputs (e.g. 1D Schrodinger's |h| =
        # sqrt(u**2+v**2)) the same way the live Results panel already can,
        # rather than being limited to one raw output at a time.
        self.restore_output_combo.addItems(["Output 1 (u)", "Custom..."])
        self.restore_output_combo.setFixedHeight(28)
        self.restore_output_combo.currentTextChanged.connect(self._on_restore_output_combo_changed)
        _rout_layout.addWidget(self.restore_output_combo)
        _rout_custom_row = QHBoxLayout()
        self.restore_custom_expr_input = QLineEdit()
        self.restore_custom_expr_input.setPlaceholderText("expression, e.g. sqrt(u**2+v**2) or du_x")
        self.restore_custom_expr_input.setToolTip(
            "A combination of this model's outputs (e.g. 'sqrt(u**2+v**2)') "
            "and/or their derivatives, using the same d{name}_x / d{name}_xx "
            "/ d{name}_xy syntax the Custom PDE box and Training Monitors "
            "use (e.g. 'du_x', 'u + du_xx'). Evaluated via DeepXDE's own "
            "dde.Model.predict(x, operator=...), the same mechanism "
            "OperatorPredictor uses during training.")
        self.restore_custom_expr_input.setFixedHeight(26)
        self.restore_custom_expr_input.setVisible(False)
        _rout_custom_row.addWidget(self.restore_custom_expr_input)
        self.restore_custom_label_input = QLineEdit()
        self.restore_custom_label_input.setPlaceholderText("label, e.g. |h|")
        self.restore_custom_label_input.setFixedHeight(26)
        self.restore_custom_label_input.setFixedWidth(90)
        self.restore_custom_label_input.setVisible(False)
        _rout_custom_row.addWidget(self.restore_custom_label_input)
        _rout_layout.addLayout(_rout_custom_row)
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
        # Same default as the main Solve panel's "Save to:" field
        # (self.save_dir_input) -- the main results directory, so Restore
        # doesn't open to an empty path the user has to fill in by hand
        # every time. Still freely editable/browsable afterward.
        self.restore_save_path.setText(os.path.join(os.path.expanduser("~"), "PINNStudio_Results"))
        self.restore_save_path.setFixedHeight(28)
        restore_save_row.addWidget(self.restore_save_path)
        self.restore_save_browse_btn = QPushButton("Browse")
        self.restore_save_browse_btn.setFixedHeight(28); self.restore_save_browse_btn.setFixedWidth(65)
        self.restore_save_browse_btn.clicked.connect(self._on_browse_restore_save)
        restore_save_row.addWidget(self.restore_save_browse_btn)
        restore_content_layout.addLayout(restore_save_row)

        self.restore_btn = QPushButton("Restore && Visualize")
        # the page's main action: filled like Solve, in every theme
        self.restore_btn.setObjectName("primaryButton")
        from pinnstudio.ui import icons as _icons_r
        _icons_r.apply(self.restore_btn, "history", "on_accent", size=16)
        self.restore_btn.clicked.connect(self._on_restore)
        restore_content_layout.addWidget(self.restore_btn)
        left_layout.addWidget(restore_group)

    def _build_right_panel(self, splitter):
        """Right side (added to `splitter`): the graphics area -- result
        plots and the Sketch, as tabs -- above the Training Log. The plot
        and output controls are built here too, into self.results_controls,
        which the workspace puts on the Results page of the settings panel."""
        # ── Right panel ───────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 6, 4)
        right_layout.setSpacing(3)

        # Vertical splitter: graphics above, log below
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        # See the matching comment on the main horizontal `splitter` above --
        # same fix, same reason (Qt's default childrenCollapsible=True was
        # making this handle snap the log panel fully open/closed instead
        # of resizing smoothly).
        right_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(right_splitter)

        # Graphics: the result plots, and the Sketch (Domain -> Shape ->
        # Sketch; its tab is hidden for every other shape).
        self.right_tabs = QTabWidget()
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.setObjectName("graphicsTabs")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self._log_highlighter = _theme.LogHighlighter(self.log_box.document())
        self.log_box.setPlaceholderText("Training log will appear here...")
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.bottom_tabs.setObjectName("bottomTabs")
        self.bottom_tabs.addTab(self.log_box, "Training Log")
        self.sketch_panel = SketchPanel()
        self.sketch_panel.shapes_edited.connect(self._on_sketch_edited)
        self.sketch_panel.set_bc_provider(self._sketch_bc_rows, self._sketch_domain)
        self.sketch_panel.new_bc_requested.connect(self._on_sketch_new_bc)
        self.sketch_panel.add_to_bc_requested.connect(self._on_sketch_add_to_bc)
        self.sketch_panel.remove_from_bcs_requested.connect(self._on_sketch_remove_from_bcs)
        self.sketch_panel.toggle_edge_bc.connect(self._on_sketch_toggle_edge)
        plots_widget = QWidget()
        bottom_layout = QVBoxLayout(plots_widget)   # (plots and monitors go here)
        bottom_layout.setContentsMargins(4, 6, 4, 4)
        bottom_layout.setSpacing(6)
        # Results: the saved figures (Overview), the loss while training
        # (Live loss) and the error analysis of the last run.
        from pinnstudio.ui.results import ErrorAnalysisPanel, LiveLossPlot
        self.results_tabs = QTabWidget()
        self.results_tabs.setDocumentMode(True)
        self.results_tabs.setObjectName("resultsTabs")
        self.results_tabs.addTab(plots_widget, "Overview")
        self.live_loss = LiveLossPlot()
        self.results_tabs.addTab(self.live_loss, "Live loss")
        self.ea_panel = ErrorAnalysisPanel()
        self.results_tabs.addTab(self.ea_panel, "Error analysis")
        from pinnstudio.ui.results import RunHistory
        self.run_history = RunHistory(lambda: self.save_dir_input.text())
        self.run_history.run_activated.connect(self._show_saved_run)
        self.results_tabs.addTab(self.run_history, "History")
        # Start (the first tab): new / open / recent / example gallery
        from pinnstudio.ui.start_page import StartPage
        self.start_page = StartPage()
        self.start_page.new_requested.connect(self._new_problem)
        self.start_page.open_requested.connect(self._open_problem)
        self.start_page.open_path.connect(self._open_problem)
        self.start_page.example_requested.connect(self._load_example_from_start)
        self._start_tab_index = self.right_tabs.addTab(self.start_page, "Start")
        self._results_tab_index = self.right_tabs.addTab(self.results_tabs, "Results")
        self._sketch_tab_index = self.right_tabs.addTab(self.sketch_panel, "Sketch")
        self.right_tabs.setTabVisible(self._sketch_tab_index, False)
        # Geometry: the domain set in the Domain panel (any 2D/3D shape),
        # its edges / faces numbered and coloured by BC, like the Sketch.
        self.domain_view = SketchPanel()
        self.domain_view.make_viewer()
        self.domain_view.set_bc_provider(self._sketch_bc_rows, self._sketch_domain)
        self.domain_view.new_bc_requested.connect(self._on_sketch_new_bc)
        self.domain_view.add_to_bc_requested.connect(self._on_sketch_add_to_bc)
        self.domain_view.remove_from_bcs_requested.connect(self._on_sketch_remove_from_bcs)
        self.domain_view.toggle_edge_bc.connect(self._on_sketch_toggle_edge)
        self._domain_tab_index = self.right_tabs.addTab(self.domain_view, "Geometry")
        self.right_tabs.setTabVisible(self._domain_tab_index, False)
        self._domain_view_key = None
        self._domain_view_timer = QTimer(self)
        self._domain_view_timer.setInterval(400)
        self._domain_view_timer.timeout.connect(self._refresh_domain_view)
        self._domain_view_timer.start()
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)
        right_splitter.addWidget(self.right_tabs)
        # Script: the DeepXDE script Export would write, kept up to date
        self.script_tab = QWidget()
        _sl = QVBoxLayout(self.script_tab)
        _sl.setContentsMargins(0, 2, 0, 0)
        _sl.setSpacing(2)
        _sbar = QHBoxLayout()
        _sbar.setContentsMargins(6, 0, 6, 0)
        self.script_status = QLabel("The DeepXDE script for this problem, as Export script would write it.")
        self.script_status.setObjectName("pageDesc")
        _sbar.addWidget(self.script_status, 1)
        self.script_copy_btn = QPushButton("Copy")
        self.script_copy_btn.clicked.connect(self._copy_script)
        self.script_export_btn = QPushButton("Export…")
        self.script_export_btn.clicked.connect(self._export_deepxde_script)
        _sbar.addWidget(self.script_copy_btn)
        _sbar.addWidget(self.script_export_btn)
        _sl.addLayout(_sbar)
        self.script_view = QPlainTextEdit()
        self.script_view.setObjectName("scriptView")
        self.script_view.setReadOnly(True)
        self.script_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._script_highlighter = _theme.PythonHighlighter(self.script_view.document())
        _sl.addWidget(self.script_view, 1)
        self._script_key = None
        self._script_tab_index = self.bottom_tabs.addTab(self.script_tab, "Script")
        self.bottom_tabs.currentChanged.connect(lambda _i: self._refresh_script_tab())
        right_splitter.addWidget(self.bottom_tabs)
        self.right_splitter = right_splitter
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([640, 260])

        # Plot / output controls -> the Results page (see _build_workspace)
        self.results_controls = QGroupBox("Plots && Output")
        controls_layout = QVBoxLayout(self.results_controls)
        controls_layout.setSpacing(6)

        # Controls row
        ctrl_row = QHBoxLayout()

        # Output selector
        ctrl_row.addWidget(QLabel("Plot output:"))
        self.plot_output_combo = QComboBox()
        # "Custom..." is always the last item (see _on_num_outputs_changed
        # and _apply_config, where the combo is rebuilt) -- picking it
        # reveals the two fields below instead of a raw output column, so
        # a multi-output problem can plot any derived field of its own
        # outputs (e.g. 1D Schrodinger's |h| = sqrt(u**2+v**2) from its
        # u,v outputs) rather than only ever one of them individually.
        self.plot_output_combo.addItems(["Output 1 (u)", "Custom..."])
        self.plot_output_combo.setFixedHeight(28)
        self._fit_combo_width(self.plot_output_combo, min_width=130)
        self.plot_output_combo.currentTextChanged.connect(self._on_plot_output_combo_changed)
        ctrl_row.addWidget(self.plot_output_combo)
        # Multi-output problems: also save each output's own plot, named
        # after it (solution_plot_<name>.png) -- Round 69.
        self.plot_all_outputs_cb = QCheckBox("All outputs")
        self.plot_all_outputs_cb.setChecked(True)
        self.plot_all_outputs_cb.setToolTip("With more than one output, also save every output's own plot\n"
                                            "(solution_plot_<name>.png), not only the one selected here.")
        ctrl_row.addWidget(self.plot_all_outputs_cb)

        # Custom field expression/label -- only visible when "Custom..." is
        # selected above. Expression variables are this problem's own
        # output_names (see codegen.py's _extract_plot_field); Label is
        # just the colorbar/axis text (falls back to the expression itself
        # when left blank).
        self.plot_custom_expr_input = QLineEdit()
        self.plot_custom_expr_input.setPlaceholderText("expression, e.g. sqrt(u**2+v**2) or du_x")
        self.plot_custom_expr_input.setToolTip(
            "A combination of this model's outputs (e.g. 'sqrt(u**2+v**2)') "
            "and/or their derivatives, using the same d{name}_x / d{name}_xx "
            "/ d{name}_xy syntax the Custom PDE box and Training Monitors "
            "use (e.g. 'du_x', 'u + du_xx'). Evaluated via DeepXDE's own "
            "dde.Model.predict(x, operator=...), the same mechanism "
            "OperatorPredictor uses during training.")
        self.plot_custom_expr_input.setFixedHeight(28)
        self.plot_custom_expr_input.setFixedWidth(190)
        self.plot_custom_expr_input.setVisible(False)
        ctrl_row.addWidget(self.plot_custom_expr_input)
        self.plot_custom_label_input = QLineEdit()
        self.plot_custom_label_input.setPlaceholderText("label, e.g. |h|")
        self.plot_custom_label_input.setFixedHeight(28)
        self.plot_custom_label_input.setFixedWidth(80)
        self.plot_custom_label_input.setVisible(False)
        ctrl_row.addWidget(self.plot_custom_label_input)

        ctrl_row.addStretch()
        controls_layout.addLayout(ctrl_row)
        ctrl_row = QHBoxLayout()   # plot type on its own line
        ctrl_row.addWidget(QLabel("Plot type:"))
        self.plot_type_combo = QComboBox()
        # "Surface Animation (GIF)" starts OUT of this list -- the app
        # opens in 1D (self.radio_1d.setChecked(True) above, before this
        # widget even exists), and that option only means anything in
        # 2D/3D (see the matching comment in _on_dim_changed, which adds
        # it back the moment the dimension switches to 2D/3D, and removes
        # it again switching back to 1D). _on_dim_changed never runs at
        # startup itself (it's wired to the radio buttons' toggled signal,
        # which fires on change, not on the initial setChecked(True)
        # before any connection exists) -- same reason
        # self.quick_examples_combo a few lines below is seeded with 1D's
        # own template list directly here rather than relying on
        # _on_dim_changed to populate it.
        self.plot_type_combo.addItems([
            "Surface", "Line (time steps)",
            "Line Animation (GIF)",
        ])
        self.plot_type_combo.setFixedHeight(28)
        self._fit_combo_width(self.plot_type_combo, min_width=160)
        # currentTextChanged fires _on_plot_type_changed, which now always
        # pops the unified plot-settings dialog itself (see that method) --
        # there's no separate "⚙" button any more (removed: a standalone
        # settings button beside this combo was confusing alongside a combo
        # that already auto-pops its own settings on selection, exactly the
        # mismatch Restore & Visualize's own plot-type combo never had).
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        ctrl_row.addWidget(self.plot_type_combo)
        ctrl_row.addStretch()
        controls_layout.addLayout(ctrl_row)

        # Save-parameter row -- visually grouped with Plot output/Plot
        # type (directly below them) since that's conceptually where it
        # belongs, but deliberately its OWN QHBoxLayout rather than
        # appended onto ctrl_row itself. ctrl_row already carries the
        # Custom expression/label fields (~270px, shown only when a
        # template defines its own derived plot field, e.g. 1D
        # Schrodinger's |h|), and this row's own Save Parameter label+
        # combo (~200-300px, shown only in Inverse mode) can be visible
        # at the same time (1D Schrodinger + Inverse). Summed into one
        # row, those two optional blocks previously forced that single
        # row's minimum width up by ~600px -- more than the window has
        # spare room for, which forced the horizontal splitter to clamp
        # the left configuration panel down to its own hard minimum width
        # with no way to drag it back out while both stayed visible. Each
        # row here keeps its OWN minimum width bounded independently, so
        # revealing both at once no longer sums them into one oversized
        # row -- same reasoning as ctrl_row2 (Error Analysis/Export)
        # already being split out below for the same reason.
        ctrl_row_save = QHBoxLayout()
        self.param_save_label = QLabel("Save parameter:")
        self.param_save_label.setVisible(False)
        ctrl_row_save.addWidget(self.param_save_label)
        self.param_save_combo = QComboBox()
        self.param_save_combo.addItems(["No", "Every 100 iters", "Every 1000 iters"])
        # Default to saving every 100 iterations for every Inverse problem
        # -- without this, the per-variable iteration-vs-value convergence
        # text files never get written unless the user remembers to change
        # this dropdown first, which is easy to miss since it's a small
        # control that's only visible once Inverse is selected.
        self.param_save_combo.setCurrentText("Every 100 iters")
        self.param_save_combo.setFixedHeight(28)
        self._fit_combo_width(self.param_save_combo, min_width=150)
        self.param_save_combo.setVisible(False)
        ctrl_row_save.addWidget(self.param_save_combo)
        ctrl_row_save.addStretch()
        controls_layout.addLayout(ctrl_row_save)

        # Second row for Error Analysis/Export -- its own row for the same
        # independent-minimum-width reason as ctrl_row_save above.
        ctrl_row2 = QHBoxLayout()
        self.ea_btn = QPushButton("📊 Error Analysis")
        self.ea_btn.setFixedHeight(28)
        self._register_style(self.ea_btn, "button", lambda css: f"""
            QPushButton {{ background: #1a3a5a; color: #74c0fc; {css}
                          border-radius: 4px; border: 1px solid #2a5a8a; padding: 0 8px; }}
            QPushButton:hover {{ background: #2a5a8a; }}
        """)
        self.ea_btn.clicked.connect(self._on_error_analysis_btn)
        ctrl_row2.addWidget(self.ea_btn)

        # Loss Plot Settings (Round 28) -- which losses to show (Train+Test
        # totals / individual components / all), the shared display_every
        # sampling cadence, line thickness, and log/linear y-axis -- see
        # _on_loss_plot_settings() and codegen.py's _loss_plot_runtime_code/
        # _clean_loss_series_lines, which both read config.loss_plot_* at
        # generation time. Backed by this dict (not dedicated widgets),
        # same pattern as self._plot_viz_settings just below, and
        # round-tripped into/out of PINNConfig by _build_config()/
        # _apply_config().
        self.loss_plot_btn = QPushButton("📉 Loss Plot Settings")
        self.loss_plot_btn.setFixedHeight(28)
        self._register_style(self.loss_plot_btn, "button", lambda css: f"""
            QPushButton {{ background: #3a2a1a; color: #ffa94d; {css}
                          border-radius: 4px; border: 1px solid #6a4a2a; padding: 0 8px; }}
            QPushButton:hover {{ background: #5a3a2a; }}
        """)
        self.loss_plot_btn.clicked.connect(self._on_loss_plot_settings)
        ctrl_row2.addWidget(self.loss_plot_btn)

        self._loss_plot_settings = {
            'mode': 'train_test',       # "train_test" | "individual" | "all"
            'display_every': 1000,
            'linewidth': 2.0,
            'log_y': True,
        }

        self._plot_viz_settings = {
            'colormap': 'jet',
            # NOTE: steps (n_steps_line/n_steps_anim) and the y/z slice
            # live in the persistent hidden widgets (self.timesteps_spin_line/
            # _anim, self.line_slice_y_auto_cb/_spin, self.line_slice_z_auto_cb/
            # _spin) instead of this dict -- see _build_config()/_apply_config(),
            # which round-trip those widgets directly into PINNConfig fields.
            # This dict only backs the settings that don't already have a
            # dedicated widget of their own.
            'n_2d_snapshots': 2,
            'colorbar': True,
            'levels': 100,
            'resolution': 200,
            'dpi': 300,
            'auto_range': True,
            'vmin': -1.0,
            'vmax': 1.0,
            'linewidth': 2.0,
            'linewidth_anim': 3.0,
            'fps': 10,
            'figsize_mode': 'Default',
            'figsize_w': 7.0,
            'figsize_h': 5.0,
            'title': '',
            'xlabel': '',
            'ylabel': '',
            # Round 31: figure-suptitle / per-panel-subplot-title / axis-
            # label font sizes -- same 3 settings, same defaults, as
            # PINNConfig.plot_title_fontsize/plot_subplot_title_fontsize/
            # plot_axis_label_fontsize (see config.py for why these 3
            # particular values were chosen as the app-wide defaults).
            'title_fontsize': 13.0,
            'subplot_title_fontsize': 10.0,
            'axis_label_fontsize': 10.0,
        }

        self.export_btn = QPushButton("💾 Export Solution")
        self.export_btn.setFixedHeight(28)
        self._register_style(self.export_btn, "button", lambda css: f"""
            QPushButton {{ background: #1a4a3a; color: #69db7c; {css}
                          border-radius: 4px; border: 1px solid #2a6a4a; padding: 0 8px; }}
            QPushButton:hover {{ background: #2a6a4a; }}
        """)
        self.export_btn.clicked.connect(self._on_export_settings)
        ctrl_row2.addWidget(self.export_btn)

        # Split into two hidden value-holders (never added to any layout --
        # same as the single one this replaces) so "Line (time steps)" and
        # the two GIF animation types can default differently and keep
        # separate values instead of fighting over one shared spinbox: 5
        # steps keeps the static overlay readable, while 20 frames makes
        # the animation play back slower/smoother. _on_line_plot_settings
        # picks the right one based on which plot type is active.
        self.timesteps_spin_line = QSpinBox()
        self.timesteps_spin_line.setRange(2, 20); self.timesteps_spin_line.setValue(5)
        self.timesteps_spin_line.setVisible(False)
        self.timesteps_spin_anim = QSpinBox()
        self.timesteps_spin_anim.setRange(2, 20); self.timesteps_spin_anim.setValue(20)
        self.timesteps_spin_anim.setVisible(False)

        # Hidden value-holders for the 2D/3D line-type plots' y/z slice
        # (see PINNConfig.line_slice_y_auto/line_slice_y and the matching
        # z fields) -- same pattern as the two spinboxes just above.
        # Checked (default) = "use the domain midpoint", matching every
        # line-type plot's previous hardcoded behavior exactly; unchecked
        # lets the paired spinbox's value override it.
        self.line_slice_y_auto_cb = QCheckBox("Auto (domain midpoint)")
        self.line_slice_y_auto_cb.setChecked(True)
        self.line_slice_y_auto_cb.setVisible(False)
        self.line_slice_y_spin = QDoubleSpinBox()
        self.line_slice_y_spin.setRange(-1e6, 1e6); self.line_slice_y_spin.setValue(0.0)
        self.line_slice_y_spin.setVisible(False)
        self.line_slice_z_auto_cb = QCheckBox("Auto (domain midpoint)")
        self.line_slice_z_auto_cb.setChecked(True)
        self.line_slice_z_auto_cb.setVisible(False)
        self.line_slice_z_spin = QDoubleSpinBox()
        self.line_slice_z_spin.setRange(-1e6, 1e6); self.line_slice_z_spin.setValue(0.0)
        self.line_slice_z_spin.setVisible(False)

        ctrl_row2.addStretch()
        controls_layout.addLayout(ctrl_row2)

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
        # Editable, same mechanism as Time-Adaptive's ta_grid (see
        # _ta_grid_size()): lets the user type a custom per-axis
        # resolution instead of being limited to the 4 presets.
        # _export_grid_size() below is the single place that parses it
        # back out safely.
        self.export_grid_combo.setEditable(True)
        self.export_grid_combo.setValidator(QIntValidator(1, 1_000_000, self.export_grid_combo))

        self.export_tsteps_spin = QSpinBox()
        self.export_tsteps_spin.setRange(2, 50); self.export_tsteps_spin.setValue(11)
        self.export_tsteps_spin.setVisible(False)

        # Off by default: solution data used to be written unconditionally
        # on every non-Inverse run, whether or not the user ever opened
        # this panel. Now it's opt-in -- see _on_export_settings() for the
        # one visible checkbox that drives this.
        self.export_enabled_cb = QCheckBox("Enable solution data export")
        self.export_enabled_cb.setChecked(False)
        self.export_enabled_cb.setVisible(False)
        controls_layout.addLayout(save_row)

        # Plot area
        plots_layout = QHBoxLayout()
        plots_layout.setSpacing(6)

        self.loss_label = QLabel()
        self.loss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loss_label.setText("📉 Loss plot")
        self.loss_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        # Was 500x450 each (1000px combined, just for these two boxes) --
        # on a laptop-class window (e.g. a 13" MacBook's ~1440-logical-point
        # width), that alone left the whole right side needing >1000px
        # minimum, which in turn left almost no slack for the control-panel
        # <-> plots splitter to actually move: confirmed by measuring this
        # app's own minimumSizeHint() at 1022px wide for the right side as
        # a whole. The saved plot image is scaled to fit this box either
        # way (see the .scaled(...) calls where loss_path/solution_path get
        # loaded), so shrinking the minimum doesn't crop or distort
        # anything -- it just lets the box (and so the window, and so the
        # splitter) go smaller when there isn't 1000+px of laptop screen to
        # spare, while it's still free to grow as large as the window
        # allows once there is room.
        self.loss_label.setMinimumSize(240, 180)
        self.loss_label._source_path = None

        self.solution_label = QLabel()
        self.solution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.solution_label.setText("🗺 Solution plot")
        self.solution_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.solution_label.setMinimumSize(240, 180)
        self.solution_label._source_path = None
        from pinnstudio.ui.export_tools import attach_image_menu
        for _lbl in (self.loss_label, self.solution_label):
            _lbl.setToolTip("Double-click to open the figure at full size (zoom and pan); "
                            "right-click to copy or save it")
            _lbl.mouseDoubleClickEvent = (lambda _ev, _l=_lbl: self._open_figure(_l))
            attach_image_menu(_lbl, lambda _l=_lbl: getattr(_l, "_source_path", None),
                              lambda _l=_lbl: self._open_figure(_l))
            from pinnstudio.ui.results import FitImage
            _lbl._fit = FitImage(_lbl)

        # Each figure gets its own small header (title + a "save this
        # figure" button) above the image itself, both wrapped in one
        # container so plots_layout still just sees two side-by-side items.
        loss_container = QWidget()
        loss_vlayout = QVBoxLayout(loss_container)
        loss_vlayout.setContentsMargins(0, 0, 0, 0)
        loss_vlayout.setSpacing(2)
        loss_header = QHBoxLayout()
        self.loss_header_label = QLabel("Loss")
        loss_header.addWidget(self.loss_header_label)
        loss_header.addStretch()
        # a quiet icon: right-clicking the figure also copies or saves it
        self.loss_save_btn = self._figure_save_button("Save the loss plot shown (or right-click the figure to copy it)")
        self.loss_save_btn.clicked.connect(lambda: self._save_figure(self.loss_label, "loss_plot"))
        loss_header.addWidget(self.loss_save_btn)
        loss_vlayout.addLayout(loss_header)
        loss_vlayout.addWidget(self.loss_label)

        solution_container = QWidget()
        solution_vlayout = QVBoxLayout(solution_container)
        solution_vlayout.setContentsMargins(0, 0, 0, 0)
        solution_vlayout.setSpacing(2)
        solution_header = QHBoxLayout()
        self.solution_header_label = QLabel("Solution")
        solution_header.addWidget(self.solution_header_label)
        solution_header.addStretch()
        # a quiet icon: right-clicking the figure also copies or saves it
        self.solution_save_btn = self._figure_save_button("Save the solution plot shown (or right-click the figure to copy it)")
        self.solution_save_btn.clicked.connect(lambda: self._save_figure(self.solution_label, "solution_plot"))
        solution_header.addWidget(self.solution_save_btn)
        solution_vlayout.addLayout(solution_header)
        solution_vlayout.addWidget(self.solution_label)

        plots_layout.addWidget(loss_container)
        plots_layout.addWidget(solution_container)
        bottom_layout.addLayout(plots_layout)

        # Training Monitors -- hidden unless the last run actually had at
        # least one monitor configured (see _display_monitor_plots(),
        # called from _display_run_result_plots()/_on_done()). A run can
        # have any number of monitors, unlike the fixed loss/solution
        # pair above, so instead of a variable-width row this shows one
        # monitor's plot at a time with Prev/Next, the same bounded-size
        # box the other two plots already use.
        self.monitor_container = QWidget()
        monitor_vlayout = QVBoxLayout(self.monitor_container)
        monitor_vlayout.setContentsMargins(0, 0, 0, 0)
        monitor_vlayout.setSpacing(2)
        monitor_header = QHBoxLayout()
        self.monitor_header_label = QLabel("Training Monitors")
        monitor_header.addWidget(self.monitor_header_label)
        monitor_header.addStretch()
        self.monitor_prev_btn = QPushButton("◀")
        self.monitor_prev_btn.setFixedHeight(24); self.monitor_prev_btn.setFixedWidth(28)
        self.monitor_prev_btn.clicked.connect(lambda: self._show_monitor_plot(self.monitor_plot_idx - 1))
        monitor_header.addWidget(self.monitor_prev_btn)
        self.monitor_idx_label = QLabel("")
        monitor_header.addWidget(self.monitor_idx_label)
        self.monitor_next_btn = QPushButton("▶")
        self.monitor_next_btn.setFixedHeight(24); self.monitor_next_btn.setFixedWidth(28)
        self.monitor_next_btn.clicked.connect(lambda: self._show_monitor_plot(self.monitor_plot_idx + 1))
        monitor_header.addWidget(self.monitor_next_btn)
        # a quiet icon: right-clicking the figure also copies or saves it
        self.monitor_save_btn = self._figure_save_button("Save the training monitor plot shown (or right-click the figure to copy it)")
        self.monitor_save_btn.clicked.connect(lambda: self._save_figure(self.monitor_label, "monitor_plot"))
        monitor_header.addWidget(self.monitor_save_btn)
        monitor_vlayout.addLayout(monitor_header)

        self.monitor_label = QLabel()
        self.monitor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.monitor_label.setText("📈 Training monitor plot")
        self.monitor_label.setStyleSheet("border: 1px solid #3e3e42; border-radius: 6px; color: #505080; background: #252526;")
        self.monitor_label.setMinimumSize(240, 180)
        self.monitor_label._source_path = None
        from pinnstudio.ui.export_tools import attach_image_menu as _aim
        _aim(self.monitor_label, lambda: getattr(self.monitor_label, "_source_path", None),
             lambda: self._open_figure(self.monitor_label))
        from pinnstudio.ui.results import FitImage as _FI
        self.monitor_label._fit = _FI(self.monitor_label)
        monitor_vlayout.addWidget(self.monitor_label)

        self.monitor_plot_files = []  # [(display_name, png_path), ...] for the last run
        self.monitor_plot_idx = 0
        bottom_layout.addWidget(self.monitor_container)
        self.monitor_container.setVisible(False)

        splitter.addWidget(right)


    # ── Parameter Sweep panel (inline in the Setup tab's right side) ──
    def _on_sweep_enabled_toggled(self, checked):
        """Wired to self.sweep_enable_cb (left panel, right after
        Adaptive Training). The sweep's own configuration (mode,
        parameters) lives inline in that same left-panel group -- this
        doesn't change anything about the right panel (Training Log and
        plots stay exactly where they are for a normal Solve). All it
        does is relabel the shared Solve button so it's clear a click
        will run the sweep instead (see _on_solve, which checks
        config.sweep_enabled itself to decide which one actually
        happens, and _on_stop, which does the matching thing for
        Cancel), show/hide the sweep's own options (mode, parameter
        rows, add/refresh buttons) so they don't take up panel space
        while the sweep is off, and refresh which parameters are
        available to sweep over against the panel's current settings."""
        if hasattr(self, 'sweep_options_widget'):
            self.sweep_options_widget.setVisible(checked)
        if hasattr(self, 'sweep_refresh_btn'):
            self.sweep_refresh_btn.setVisible(checked)
        if hasattr(self, 'solve_btn'):
            self.solve_btn.setText(" Run Sweep" if checked else " Solve")
            self.solve_btn.setToolTip(
                "Runs every configured sweep combination, one training run "
                "at a time, saving each to its own folder." if checked else "")
        if checked:
            self._refresh_sweep_param_choices()

    def _on_sweep_baseline_clicked(self, checked):
        self._sweep_baseline_user = bool(checked)

    def _sweep_baseline_setting(self):
        """config.sweep_include_baseline from the checkbox: "auto" until
        the user clicks it, then "on"/"off"."""
        u = getattr(self, '_sweep_baseline_user', None)
        return "auto" if u is None else ("on" if u else "off")

    def _update_sweep_baseline_default(self):
        """Shows the automatic baseline choice for the current mode and
        number of parameters (sweep_runner.includes_baseline), unless the
        user has set it themselves."""
        if not hasattr(self, 'sweep_baseline_cb'):
            return
        if self._sweep_baseline_user is not None:
            self.sweep_baseline_cb.setChecked(self._sweep_baseline_user)
            return
        mode = self.sweep_mode_combo.currentData()
        n = len(getattr(self, 'sweep_row_list', []) or [])
        self.sweep_baseline_cb.setChecked(mode == "bayes" or (mode == "oat" and n >= 2))

    def _add_sweep_row(self):
        """One parameter block, stacked top-to-bottom (header row with a
        remove button, then Parameter, then Sweep-as, then Values/range
        fields) -- the same layout convention the left panel's own
        Optimizer Scheduler phases already use (_add_scheduler_phase),
        rather than v69's single-line-per-row table layout, which needed
        roughly 700px of width to read well. That traded-off design
        made sense back when the sweep parameters lived in their own
        wide tab/panel; now that the whole sweep configuration lives
        inline in this narrow left column instead (alongside every
        other training setting), a stacked block fits the same way the
        rest of this column's multi-field settings already do."""
        row_widget = QWidget()
        row_widget.setObjectName("sweepParamRow")
        row_widget.setStyleSheet(
            "QWidget#sweepParamRow { border-bottom: 1px solid #333338; }")
        row_layout = QVBoxLayout(row_widget)
        row_layout.setSpacing(3)
        row_layout.setContentsMargins(0, 4, 0, 6)

        # Header: "── Parameter N ──" + remove button, same shape as
        # each Training Phase's own header row.
        header_row = QHBoxLayout()
        row_num = len(self.sweep_row_list) + 1
        header_lbl = QLabel(f"── Parameter {row_num} ──")
        self._register_hint(header_lbl, '#a0c4ff')
        header_row.addWidget(header_lbl)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        header_row.addStretch(); header_row.addWidget(remove_btn)
        row_layout.addLayout(header_row)

        # Parameter
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Parameter:"))
        param_combo = QComboBox()
        param_combo.setFixedHeight(26)
        param_row.addStretch(); param_row.addWidget(param_combo)
        row_layout.addLayout(param_row)

        # Sweep as (hidden for categorical parameters -- those are
        # always a plain list of choices, there's no range to pick)
        mode_row_widget = QWidget()
        mode_row = QHBoxLayout(mode_row_widget)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.addWidget(QLabel("Sweep as:"))
        mode_combo = QComboBox()
        mode_combo.addItem("List of values", "list")
        mode_combo.addItem("Linear range", "linear")
        mode_combo.addItem("Log range", "log")
        mode_combo.setFixedHeight(24)
        mode_row.addStretch(); mode_row.addWidget(mode_combo)
        row_layout.addWidget(mode_row_widget)

        # Values (list mode) -- label above the field on its own line
        # (like "PDE 1 (residual = 0):" elsewhere in this panel) rather
        # than label-then-field on one line, so a longer comma-separated
        # list has real room to be typed/read in this narrow column.
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(2)
        list_layout.addWidget(QLabel("Values:"))
        list_edit = QLineEdit()
        list_edit.setPlaceholderText("comma-separated, e.g. 0.001, 0.005, 0.01")
        list_edit.setFixedHeight(26)
        list_layout.addWidget(list_edit)
        row_layout.addWidget(list_widget)

        # Range (linear/log mode)
        range_widget = QWidget()
        range_row = QHBoxLayout(range_widget)
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(4)
        range_row.addWidget(QLabel("Min:"))
        min_spin = QDoubleSpinBox()
        min_spin.setRange(-1e9, 1e9); min_spin.setDecimals(4)
        min_spin.setFixedWidth(62); min_spin.setFixedHeight(24)
        range_row.addWidget(min_spin)
        range_row.addWidget(QLabel("Max:"))
        max_spin = QDoubleSpinBox()
        max_spin.setRange(-1e9, 1e9); max_spin.setDecimals(4); max_spin.setValue(1.0)
        max_spin.setFixedWidth(62); max_spin.setFixedHeight(24)
        range_row.addWidget(max_spin)
        range_row.addWidget(QLabel("Steps:"))
        n_spin = QSpinBox()
        n_spin.setRange(2, 100); n_spin.setValue(5)
        n_spin.setFixedWidth(48); n_spin.setFixedHeight(24)
        range_row.addWidget(n_spin)
        range_row.addStretch()
        row_layout.addWidget(range_widget)

        def _update_value_widgets():
            meta = param_combo.itemData(param_combo.currentIndex(), Qt.ItemDataRole.UserRole + 1) or {}
            is_categorical = meta.get("value_type") == "categorical"
            if is_categorical:
                mode_row_widget.setVisible(False)
                range_widget.setVisible(False)
                list_widget.setVisible(True)
                choices = meta.get("choices") or []
                list_edit.setPlaceholderText(f"comma-separated, choices: {', '.join(choices)}")
            else:
                mode_row_widget.setVisible(True)
                mode = mode_combo.currentData()
                list_widget.setVisible(mode == "list")
                range_widget.setVisible(mode in ("linear", "log"))
                list_edit.setPlaceholderText("comma-separated, e.g. 0.001, 0.005, 0.01")

        param_combo.currentIndexChanged.connect(lambda _i: _update_value_widgets())
        mode_combo.currentIndexChanged.connect(lambda _i: _update_value_widgets())

        self.sweep_rows_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget,
            'param_combo': param_combo,
            'mode_combo': mode_combo,
            'list_edit': list_edit,
            'min_spin': min_spin,
            'max_spin': max_spin,
            'n_spin': n_spin,
        }
        self.sweep_row_list.append(row_data)
        self._update_sweep_baseline_default()

        def _remove():
            row_widget.deleteLater()
            if row_data in self.sweep_row_list:
                self.sweep_row_list.remove(row_data)
            self._update_sweep_baseline_default()

        remove_btn.clicked.connect(_remove)
        self._refresh_sweep_param_choices()
        # _refresh_sweep_param_choices() populates param_combo with
        # blockSignals(True) (so repopulating it elsewhere doesn't spam
        # currentIndexChanged on every other row), which means
        # _update_value_widgets() never actually ran for THIS brand new
        # row -- without this, a freshly added row shows both the Values
        # field AND the Min/Max/Steps range fields at once until the
        # user happens to touch one of the combos.
        _update_value_widgets()

    def _refresh_sweep_param_choices(self):
        """Re-populates every sweep row's Parameter dropdown from
        sweep_registry.available_params() against whatever _build_config()
        produces right now, preserving each row's current selection when
        that parameter is still available. Guarded with a try/except: the
        Setup tab's widgets can be in a transient, not-yet-valid state
        (e.g. mid-edit) when this fires, and a stale parameter list is far
        less disruptive than a crash."""
        try:
            config = self._build_config()
        except Exception:
            return
        from pinnstudio.core import sweep_registry as reg
        params = reg.available_params(config)
        for row in getattr(self, 'sweep_row_list', []):
            combo = row['param_combo']
            previous_id = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for p in params:
                # Most registry entries have a bare label ("Hidden
                # layers") that wants its category prefixed for display
                # ("Network: Hidden layers"). The per-phase entries
                # (phase{i}_lr/_iterations/_optimizer) already embed
                # their own "Phase N: " prefix in p.label itself -- that
                # text is also what ends up in run folder names/labels
                # (sweep_runner.py), so it's kept as-is there rather
                # than changed -- only skip re-prepending it here so the
                # dropdown doesn't read "Phase 1: Phase 1: Optimizer".
                display = p.label if p.label.startswith(f"{p.category}:") else f"{p.category}: {p.label}"
                combo.addItem(display, p.id)
                combo.setItemData(
                    combo.count() - 1,
                    {"value_type": p.value_type, "choices": p.choices},
                    Qt.ItemDataRole.UserRole + 1,
                )
            if previous_id:
                idx = combo.findData(previous_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
            # Same convention as the Training Phases' own Optimizer
            # combo (opt_combo, min_width=80) a few sections up --
            # this narrow left column clips long combo text the same
            # way other long field values here already do (e.g. a
            # Boundary Conditions entry naming its side/type/output),
            # which is a pre-existing, accepted tradeoff of this
            # column's width, not something unique to this combo.
            self._fit_combo_width(combo, min_width=80)

    def _apply_sweep_config(self, config):
        """Round 75: a loaded config's Parameter Sweep (enabled, mode,
        Bayesian settings, parameter rows) back into the panel."""
        import json
        if not hasattr(self, 'sweep_enable_cb'):
            return
        i = self.sweep_mode_combo.findData(getattr(config, "sweep_mode", "oat") or "oat")
        self.sweep_mode_combo.setCurrentIndex(max(0, i))
        self.sweep_bayes_calls_spin.setValue(int(getattr(config, "sweep_bayes_calls", 20) or 20))
        _b = str(getattr(config, "sweep_include_baseline", "auto"))
        self._sweep_baseline_user = True if _b in ("on", "True") else (False if _b in ("off", "False") else None)
        self.sweep_bayes_init_spin.setValue(int(getattr(config, "sweep_bayes_initial", 5) or 5))
        j = self.sweep_bayes_obj_combo.findData(getattr(config, "sweep_bayes_objective", "loss"))
        self.sweep_bayes_obj_combo.setCurrentIndex(max(0, j))
        try:
            entries = json.loads(getattr(config, "sweep_parameters", "[]") or "[]")
        except ValueError:
            entries = []
        if entries:
            for row in list(self.sweep_row_list):
                row['widget'].setParent(None)
                row['widget'].deleteLater()
            self.sweep_row_list.clear()
            for e in entries:
                self._add_sweep_row()
                row = self.sweep_row_list[-1]
                k = row['param_combo'].findData(e.get("id"))
                if k >= 0:
                    row['param_combo'].setCurrentIndex(k)
                m = row['mode_combo'].findData(e.get("mode", "list"))
                row['mode_combo'].setCurrentIndex(max(0, m))
                if e.get("mode") == "list":
                    row['list_edit'].setText(", ".join(str(v) for v in e.get("values", [])))
                else:
                    for key, spin in (("min", 'min_spin'), ("max", 'max_spin')):
                        v = float(e.get(key, 0.0))
                        if v:
                            # enough decimals for small bounds such as a learning rate of 1e-5
                            row[spin].setDecimals(max(4, 2 - int(math.floor(math.log10(abs(v))))))
                        row[spin].setValue(v)
                    row['n_spin'].setValue(int(e.get("n", 5)))
        self.sweep_enable_cb.setChecked(bool(getattr(config, "sweep_enabled", False)))
        self.sweep_bayes_widget.setVisible(self.sweep_mode_combo.currentData() == "bayes")
        self._update_sweep_baseline_default()

    def _build_sweep_parameters_json(self):
        import json
        entries = []
        for row in getattr(self, 'sweep_row_list', []):
            pid = row['param_combo'].currentData()
            if not pid:
                continue
            meta = row['param_combo'].itemData(row['param_combo'].currentIndex(), Qt.ItemDataRole.UserRole + 1) or {}
            is_categorical = meta.get('value_type') == 'categorical'
            mode = row['mode_combo'].currentData() if not is_categorical else "list"
            if mode == "list":
                raw = row['list_edit'].text().strip()
                vals = [v.strip() for v in raw.split(',') if v.strip()]
                if not is_categorical:
                    parsed = []
                    for v in vals:
                        try:
                            parsed.append(float(v))
                        except ValueError:
                            parsed.append(v)
                    vals = parsed
                entries.append({"id": pid, "mode": "list", "values": vals})
            else:
                entries.append({
                    "id": pid, "mode": mode,
                    "min": row['min_spin'].value(),
                    "max": row['max_spin'].value(),
                    "n": row['n_spin'].value(),
                })
        return json.dumps(entries)

    # ── Dimension change ──────────────────────────────────────
    GEOM_TYPES_2D = ["Rectangle", "Disk", "Ellipse", "Triangle", "Polygon", "Custom", "Sketch"]
    GEOM_TYPES_3D = ["Cuboid", "Sphere", "Custom", "Sketch"]
    CUSTOM_GEOM_SHAPE_TYPES_2D = ["Rectangle", "Disk", "Ellipse", "Triangle", "Polygon"]
    CUSTOM_GEOM_SHAPE_TYPES_3D = ["Cuboid", "Sphere"]
    CUSTOM_GEOM_SHAPE_TYPES_ALL = CUSTOM_GEOM_SHAPE_TYPES_2D + CUSTOM_GEOM_SHAPE_TYPES_3D
    CUSTOM_GEOM_OPS = [("Union (add)", "union"), ("Subtract (cut hole)", "subtract"), ("Intersect", "intersect")]

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

    def _browse_into(self, line_edit, title, file_filter="All files (*)"):
        """Open-file dialog that writes the chosen path into `line_edit`."""
        path, _ = QFileDialog.getOpenFileName(self, title, line_edit.text().strip() or "", file_filter)
        if path:
            line_edit.setText(path)

    def _transform_mode_widgets(self, parent_layout):
        """A "Mode: Scale/shift | Formula" row plus the two containers it
        switches between, added to `parent_layout`. Returns
        (combo, scale_widget, formula_widget); the caller fills both."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        combo = QComboBox()
        combo.addItem("Scale/shift", "scale")
        combo.addItem("Formula", "formula")
        combo.setFixedHeight(26)
        row.addWidget(combo)
        row.addStretch()
        parent_layout.addLayout(row)
        scale_w, formula_w = QWidget(), QWidget()
        parent_layout.addWidget(scale_w)
        parent_layout.addWidget(formula_w)
        formula_w.setVisible(False)

        def _on_mode(_i):
            is_formula = combo.currentData() == "formula"
            scale_w.setVisible(not is_formula)
            formula_w.setVisible(is_formula)
        combo.currentIndexChanged.connect(_on_mode)
        return combo, scale_w, formula_w

    def _rebuild_output_formula_rows(self, n_outputs):
        while self.output_formula_rows_layout.count():
            item = self.output_formula_rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.output_formula_edits = []
        for i in range(max(1, n_outputs)):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.addWidget(QLabel(f"Output {i + 1} ="))
            edit = QLineEdit()
            edit.setFixedHeight(26)
            edit.setPlaceholderText("unchanged")
            row_l.addWidget(edit)
            self.output_formula_rows_layout.addWidget(row_w)
            self.output_formula_edits.append(edit)

    def _rebuild_output_transform_rows(self, n_outputs):
        if hasattr(self, "output_formula_rows_layout"):
            self._rebuild_output_formula_rows(n_outputs)
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

    def _ta_grid_size(self):
        """Safely parse ta_grid's current text (one of the 4 presets, or
        a custom value the user typed directly into the now-editable
        combo -- see its construction) back into the int config.ta_grid_size
        actually needs. The QIntValidator on the widget keeps keystrokes
        restricted to digits, but currentText() can still be empty or
        transiently invalid while the user is mid-edit (e.g. selected-all-
        then-about-to-type), so this falls back to the ordinary 101
        default rather than letting int() raise. The single place both
        _update_ta_grid_warning and _build_config should read this from,
        so the two can't disagree on what an unparseable value means."""
        try:
            grid = int(self.ta_grid.currentText())
        except (ValueError, TypeError):
            grid = 101
        return max(grid, 1)

    def _export_grid_size(self):
        """Safely parse export_grid_combo's current text (one of the 4
        presets, or a custom value typed into the now-editable combo) back
        into the int config.export_grid_size needs -- exact same pattern
        as _ta_grid_size() above, for the exact same reason (a QIntValidator
        restricts keystrokes but not every transient mid-edit state)."""
        try:
            grid = int(self.export_grid_combo.currentText())
        except (ValueError, TypeError):
            grid = 101
        return max(grid, 1)

    def _update_ta_grid_warning(self, *_args):
        """Time-Adaptive's "IC grid resolution" is a per-axis count -- 1D
        uses it directly, 2D squares it, 3D cubes it. A value that's fine
        in 1D/2D (101 -> 10201 points in 2D) becomes over a million points
        per step in 3D, which can exhaust GPU memory: confirmed by a real
        run (RTX 4090, 24GB) that hit a CUDA out-of-memory error training
        3D Heat with Time-Adaptive at grid=101, on step 2 of 5. The combo's
        hover tooltip already warns about this but is easy to miss, so
        show a visible in-panel warning too, specifically when 3D is
        selected and the grid is at a size known to risk it (51 or 101,
        or now, since the field accepts custom typed values too, anything
        >= 51). 1D/2D never cube the grid, so the warning never shows for
        them."""
        if not hasattr(self, 'ta_grid_warning'):
            return
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
        grid = self._ta_grid_size()
        self.ta_grid_warning.setVisible(is_3d and grid >= 51)

    def _on_dim_changed(self):
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked()
        # Time-Adaptive's per-axis grid resolution is cubed in 3D (see
        # _update_ta_grid_warning) -- switching into 3D while it's still
        # at a size that's only safe because the previous dimension didn't
        # cube it (101 or 51) silently inherits a risky default. Step it
        # down automatically; 1D/2D never cube it, so nothing needs to
        # change when switching away from 3D.
        if is_3d and hasattr(self, 'ta_grid') and self.ta_grid.currentText() in ("101", "51"):
            self.ta_grid.setCurrentText("21")
        self._update_ta_grid_warning()
        self._refresh_bc_where_pickers()
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
            self.custom_bc_list.clear(); self._schedule_bc_nodes_sync()
        # Same reasoning for the Custom-geometry shape list: a 2D shape
        # (Rectangle/Disk/Ellipse/Triangle/Polygon) isn't a valid 3D
        # primitive and vice versa (Cuboid/Sphere aren't valid in 2D), so
        # carrying shapes across a dimension switch would leave Custom
        # geometry pointing at constructors that don't exist for the new
        # dimension. _on_geometry_type_changed() below re-seeds one
        # dimension-appropriate starting shape if Custom ends up selected
        # again after this clears the list.
        if hasattr(self, 'custom_geom_shape_rows') and self.custom_geom_shape_rows:
            for row in list(self.custom_geom_shape_rows):
                row['widget'].deleteLater()
            self.custom_geom_shape_rows.clear()
        # Quick Examples lists every template across all three dimensions
        # at once now (see _build_ui) -- it no longer needs rebuilding
        # per dimension here. What it DOES still need: when the user
        # changes dimension BY HAND (clicking a dimension radio directly,
        # not via picking a template -- see _on_template_selected's
        # _applying_template guard), whatever template was active no
        # longer necessarily matches, so the selection resets to "None",
        # the same way picking "None" itself leaves everything as a
        # custom/hand-built problem. A template-driven dimension change
        # skips this so the combo still shows the template that was just
        # loaded, instead of immediately reverting to "None".
        if not getattr(self, '_applying_template', False):
            self.quick_examples_combo.blockSignals(True)
            self.quick_examples_combo.setCurrentText("None")
            self.quick_examples_combo.blockSignals(False)

        self.view_domain_check.setVisible(is_2d or is_3d)
        for w in self._2d_bc_widgets:
            w.setVisible(is_2d)

        # "Surface Animation (GIF)" only means anything in 2D/3D, where a
        # frame genuinely has two real spatial axes that vary (an x-y grid,
        # or 3D faces). In 1D there's only one spatial axis (x); the GIF
        # code fakes a second axis by drawing a cosmetic t-range alongside
        # it, but every point along that width within a single frame is
        # evaluated at the SAME fixed t -- it's u(x) stretched sideways for
        # padding, not real data varying on that axis. Line Animation
        # already covers 1D's actual time-animated case correctly, and
        # Surface (static) still shows the full, real x-t field in one
        # image -- so remove this one option for 1D specifically rather
        # than leave a plot type that looks like data but isn't. Same
        # add/remove-by-index idiom as "Parameter Convergence" above
        # (_on_radio_inverse_toggled) -- removed (not just hidden) so a
        # stale selection can't silently linger into a script that no
        # longer offers it.
        _sa_idx = self.plot_type_combo.findText("Surface Animation (GIF)")
        if is_2d or is_3d:
            if _sa_idx == -1:
                self.plot_type_combo.addItem("Surface Animation (GIF)")
                self._fit_combo_width(self.plot_type_combo, min_width=160)
        elif _sa_idx != -1:
            if self.plot_type_combo.currentText() == "Surface Animation (GIF)":
                # blockSignals: this is a programmatic fallback caused by
                # the dimension switch itself, not a user picking a new
                # plot type from the combo -- without this,
                # setCurrentText below fires currentTextChanged ->
                # _on_plot_type_changed -> _on_line_plot_settings, which
                # pops the modal "Line Plot Settings" dialog out of
                # nowhere every time someone switches back to 1D with
                # this option selected.
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText("Line Animation (GIF)")
                self.plot_type_combo.blockSignals(False)
            self.plot_type_combo.removeItem(_sa_idx)

        # Repopulate the geometry-type selector for the new dimension.
        self.geom_type_row_widget.setVisible(is_2d or is_3d)
        self.geometry_type_combo.blockSignals(True)
        self.geometry_type_combo.clear()
        if is_2d:
            self.geometry_type_combo.addItems(self.GEOM_TYPES_2D)
        elif is_3d:
            self.geometry_type_combo.addItems(self.GEOM_TYPES_3D)
        self._fit_combo_width(self.geometry_type_combo, min_width=90)
        self.geometry_type_combo.blockSignals(False)
        self._on_geometry_type_changed(self.geometry_type_combo.currentText())

        self._build_pde_inputs(self.num_outputs_spin.value())
        self._build_bc_inputs(self.num_outputs_spin.value())
        self._build_weight_inputs(self.num_outputs_spin.value())
        self._rebuild_input_transform_rows()

        # Force an immediate layout/repaint pass so the Geometry Type row
        # (and any other widgets just toggled above) always show up right
        # away, instead of waiting on the next natural repaint cycle.
        self.domain_group.updateGeometry()
        self.domain_group.repaint()
        QApplication.processEvents()

    def _current_geometry_type(self):
        """The active geometry type, or the implicit box type for 1D/unselected."""
        if self.radio_3d.isChecked():
            t = self.geometry_type_combo.currentText() or "Cuboid"
            return "Custom" if t == "Sketch" else t
        if self.radio_2d.isChecked():
            t = self.geometry_type_combo.currentText() or "Rectangle"
            # Sketch draws a Custom geometry (the same shape list, trained by
            # the same code) -- see _on_geometry_type_changed.
            return "Custom" if t == "Sketch" else t
        return "Interval"

    def _on_geometry_type_changed(self, text):
        # Seed the Custom-geometry builder with one starting shape the
        # first time it's selected, rather than showing an empty list
        # with nothing but an "Add Shape" button -- mirrors how the
        # Parameter Sweep panel always starts with one row already
        # present. Only seeds once per MainWindow instance (once the
        # list is non-empty, switching away and back leaves it as the
        # user left it).
        if text == "Custom" and hasattr(self, 'custom_geom_shape_rows') and not self.custom_geom_shape_rows:
            self._add_custom_geom_shape()
        self._update_domain_fields_visibility()
        self._update_domain_box_title()
        self._show_sketch_tab(text == "Sketch")
        # Rebuild BCs: the boundary-condition layout (sides / unified /
        # per-edge) depends on which geometry type is now selected.
        if hasattr(self, 'bc_main_layout'):
            self._build_bc_inputs(self.num_outputs_spin.value())
            self._build_weight_inputs(self.num_outputs_spin.value())

    def _sync_frac_widget(self):
        """Show the fractional settings while an equation uses frac(...)."""
        if hasattr(self, "frac_widget"):
            import re as _re_fw
            self.frac_widget.setVisible(any(_re_fw.search(r"(?<![\w.])frac\(", w.text())
                                            for w in getattr(self, "pde_inputs", [])))

    def _is_ode(self):
        return hasattr(self, "radio_ode") and self.radio_ode.isChecked()

    def _on_ode_toggled(self, checked):
        """ODE mode (Round 74) is 1D's machinery with t as the only variable:
        the x row hides, the problem is always time-dependent (initial
        conditions at t₀), and Time-Adaptive training is unavailable."""
        if checked and hasattr(self, "radio_time_dependent"):
            self.radio_time_dependent.setChecked(True)
            if self.steady_state_check.isChecked():
                self.steady_state_check.setChecked(False)
        if hasattr(self, "radio_stationary"):
            self.radio_stationary.setEnabled(not checked)
        if hasattr(self, "integral_points_spin"):
            self.integral_points_label.setVisible(checked)
            self.integral_points_spin.setVisible(checked)
        self._on_dim_changed()

    def _update_domain_fields_visibility(self):
        """Show the plain x/y/z domain-bounds rows only for the box shapes
        (Interval/Rectangle/Cuboid). For every other shape, those rows don't
        mean anything -- e.g. an Ellipse needs a center + semi-axes, not an
        x/y range -- so hide them and show that shape's own parameter panel
        instead. The "t:" row follows whichever of the two is showing,
        unless Steady-state is on (see _on_steady_state_changed), in which
        case it stays hidden regardless."""
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked()
        geom_type = self._current_geometry_type()
        box_shape = geom_type in ("Interval", "Rectangle", "Cuboid")
        self.x_row_widget.setVisible(box_shape and not self._is_ode())
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
            "Custom": self.geom_panel_custom,
        }
        panel = panel_map.get(geom_type)
        if panel is not None:
            panel.setVisible(True)
        if hasattr(self, 'steady_state_check'):
            self.t_row_widget.setVisible(not self.steady_state_check.isChecked())

    def _on_steady_state_changed(self, checked):
        """Steady-state (time-independent) problems, e.g. a Poisson
        equation, have no time axis at all -- hide everything that only
        means something with one: the "t:" domain row, the Initial
        Condition panel, and IC Pre-Training. Time Adaptive Training is
        inherently time-stepped (it trains one model per time step) and
        stays unavailable for a steady problem -- its "Adaptive Training"
        combo item is disabled below (not removed -- that would mean
        rebuilding the same suspend/restore bookkeeping Inverse mode
        already uses to disable this same item for a different reason,
        see _current_ta_cfg/_ta_suspended_for_inverse further down this
        file), and the selection is forced back to "None" if it was the
        one active when this toggles on. Residual-based Adaptive
        Refinement (RAR) has no such dependency on a time axis -- it just
        resamples collocation points toward wherever the PDE residual is
        currently largest, which works identically whether or not the
        domain has a time dimension -- so the "Adaptive Training" group
        itself now stays visible, and RAR stays selectable and usable,
        for steady problems too (previously the whole group was folded
        away here, hiding RAR along with Time Adaptive even though only
        the latter actually needs a time axis). Turning Steady-state back
        off restores Time Adaptive Training as a selectable option again;
        nothing else is destroyed by any of this, just hidden/disabled,
        except for the two settings (adapt method when it was Time
        Adaptive, IC pre-training) explicitly reset below so a stale
        time-based choice doesn't silently linger under the hood while
        its panel is hidden.
        """
        self.t_row_widget.setVisible(not checked)
        self.bc_group.setVisible(not checked)
        self.ic_pretrain_cb.setVisible(not checked)
        self.ic_pretrain_widget.setVisible(not checked and self.ic_pretrain_cb.isChecked())
        if hasattr(self, 'adapt_combo'):
            _ta_item_idx = self.adapt_combo.findText("Time Adaptive Training")
            if _ta_item_idx != -1:
                _ta_item = self.adapt_combo.model().item(_ta_item_idx)
                if _ta_item is not None:
                    _ta_item.setEnabled(not checked)
                    _ta_item.setToolTip(
                        "Time Adaptive Training needs a time axis -- not available for a "
                        "steady-state problem. Residual-based Adaptive Refinement (RAR) "
                        "still works here." if checked else "")
        if checked:
            if self.adapt_combo.currentText() == "Time Adaptive Training":
                self.adapt_combo.setCurrentText("None")
            if self.ic_pretrain_cb.isChecked():
                self.ic_pretrain_cb.setChecked(False)
        # The Loss Weights panel's own "IC {n} (...)" rows are built by
        # _build_weight_inputs(), gated on this same checkbox's state --
        # without refreshing it here too, toggling Steady-state on its own
        # (with no other change that happens to rebuild the panel, e.g. a
        # BC row or output-count edit) would leave stale IC weight rows
        # visible until something else triggered a rebuild. Guarded by
        # hasattr since weights_main_layout/num_outputs_spin are built
        # after this checkbox during __init__, and toggled can in
        # principle fire before that (it won't with today's init order,
        # but there's no reason to rely on that staying true).
        if hasattr(self, 'weights_main_layout') and hasattr(self, 'num_outputs_spin'):
            self._build_weight_inputs(self.num_outputs_spin.value())

    def _on_geom_vertices_changed(self, text):
        # Triangle/Polygon edge count changed -- rebuild the per-edge BC rows.
        geom_type = self._current_geometry_type()
        if geom_type in ("Triangle", "Polygon") and hasattr(self, 'bc_main_layout'):
            self._build_bc_inputs(self.num_outputs_spin.value())
            self._build_weight_inputs(self.num_outputs_spin.value())

    @staticmethod
    def _parse_vertices(text):
        """'x1,y1;x2,y2;...' -> [(x1,y1), (x2,y2), ...]. Skips malformed pairs.
        cleanup Round 48: delegates to codegen.py's _parse_vertex_list (the
        parser training uses), which was already a line-for-line mirror of
        this one -- now there's only one copy to keep in sync."""
        from pinnstudio.core.codegen import _parse_vertex_list
        return [tuple(v) for v in _parse_vertex_list(text)]

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

    def _add_custom_geom_shape(self, shape_type="Rectangle", op="union", params=None):
        """One shape block in the Custom-geometry builder, stacked top-to-
        bottom the same way _add_sweep_row's parameter blocks are. The
        first shape in the list is the starting geometry and has no
        combine-op; every shape after it is combined with the running
        result so far via the chosen boolean op (Union/Subtract/
        Intersect), in list order -- the same semantics as DeepXDE's
        CSGUnion/CSGDifference/CSGIntersection applied left to right, so
        e.g. a triangle with a circular hole is just [Triangle, (Disk,
        subtract)]."""
        params = params or {}
        row_widget = QWidget()
        row_widget.setObjectName("customGeomShapeRow")
        row_widget.setStyleSheet(
            "QWidget#customGeomShapeRow { border-bottom: 1px solid #333338; }")
        row_layout = QVBoxLayout(row_widget)
        row_layout.setSpacing(3)
        row_layout.setContentsMargins(0, 4, 0, 6)

        header_row = QHBoxLayout()
        row_num = len(self.custom_geom_shape_rows) + 1
        header_lbl = QLabel(f"── Shape {row_num} ──")
        self._register_hint(header_lbl, '#a0c4ff')
        header_row.addWidget(header_lbl)
        header_row.addStretch()
        up_btn = QPushButton("▲")
        down_btn = QPushButton("▼")
        for _btn in (up_btn, down_btn):
            _btn.setFixedHeight(22); _btn.setFixedWidth(22)
            _btn.setStyleSheet(
                "QPushButton { color: #a0c4ff; background: transparent; border: none; }")
            header_row.addWidget(_btn)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        header_row.addWidget(remove_btn)
        row_layout.addLayout(header_row)

        # Combine-with-previous op -- hidden for the first shape, which
        # has nothing before it in the list to combine with.
        op_combo = QComboBox()
        for _label, _val in self.CUSTOM_GEOM_OPS:
            op_combo.addItem(_label, _val)
        _op_idx = op_combo.findData(op)
        if _op_idx >= 0:
            op_combo.setCurrentIndex(_op_idx)
        op_combo.setFixedHeight(26)
        op_row_widget = self._domain_row("Combine:", op_combo)
        row_layout.addWidget(op_row_widget)

        # Shape type -- which primitives are offered depends on the
        # current problem dimension (2D: Rectangle/Disk/Ellipse/Triangle/
        # Polygon; 3D: Cuboid/Sphere). The shape list is cleared whenever
        # the dimension changes (see _on_dim_changed), so a row is never
        # built while the "wrong" dimension's primitives would apply.
        type_combo = QComboBox()
        _is_3d_row = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
        type_combo.addItems(self.CUSTOM_GEOM_SHAPE_TYPES_3D if _is_3d_row else self.CUSTOM_GEOM_SHAPE_TYPES_2D)
        type_combo.setFixedHeight(26)
        self._fit_combo_width(type_combo, min_width=110)
        row_layout.addWidget(self._domain_row("Shape:", type_combo))

        # Per-type parameter panels -- same rows as the top-level shape
        # panels above, scoped to this one shape (a Custom geometry can
        # hold several shapes of the same type, e.g. two Disks).
        P = params.get

        def _panel(*rows):
            w = QWidget()
            lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)
            for r in rows:
                lay.addWidget(r)
            return w

        rect_xmin, rect_xmax = self._spin(-1e6, 1e6, P("x_min", 0.0)), self._spin(-1e6, 1e6, P("x_max", 1.0))
        rect_ymin, rect_ymax = self._spin(-1e6, 1e6, P("y_min", 0.0)), self._spin(-1e6, 1e6, P("y_max", 1.0))
        rect_panel = _panel(self._domain_row("x:", rect_xmin, "to", rect_xmax),
                            self._domain_row("y:", rect_ymin, "to", rect_ymax))

        disk_cx, disk_cy = self._spin(-1e6, 1e6, P("cx", 0.5)), self._spin(-1e6, 1e6, P("cy", 0.5))
        disk_r = self._spin(1e-6, 1e6, P("r", 0.2))
        disk_panel = _panel(self._domain_row("Center:", disk_cx, disk_cy, tip="Center (x, y)"),
                            self._domain_row("Radius:", disk_r))

        ell_cx, ell_cy = self._spin(-1e6, 1e6, P("cx", 0.5)), self._spin(-1e6, 1e6, P("cy", 0.5))
        ell_a, ell_b = self._spin(1e-6, 1e6, P("a", 0.5)), self._spin(1e-6, 1e6, P("b", 0.3))
        ell_angle = self._spin(-100, 100, P("angle", 0.0))
        ellipse_panel = _panel(self._domain_row("Center:", ell_cx, ell_cy, tip="Center (x, y)"),
                               self._domain_row("Semi-axes:", ell_a, ell_b, tip="Semi-major and semi-minor axis"),
                               self._domain_row("Angle (rad):", ell_angle,
                                                tip="Rotation of the major axis from the x axis, in radians;\n"
                                                    "a positive angle turns it clockwise (DeepXDE's convention)"))

        tri_verts = QLineEdit(P("vertices_text", "0,0;1,0;0,1"))
        tri_verts.setFixedHeight(26)
        tri_panel = self._domain_row("Vertices:", tri_verts,
                                     tip="Three vertices as x,y pairs separated by semicolons, e.g. 0,0; 1,0; 0,1")

        poly_verts = QLineEdit(P("vertices_text", "0,0;1,0;1,0.6;0.5,1;0,0.6"))
        poly_verts.setFixedHeight(26)
        poly_panel = self._domain_row("Vertices:", poly_verts,
                                      tip="Three or more vertices as x,y pairs separated by semicolons, "
                                          "in order around the boundary, e.g. 0,0; 1,0; 1,0.6; 0.5,1; 0,0.6")

        cuboid_xmin, cuboid_xmax = self._spin(-1e6, 1e6, P("x_min", 0.0)), self._spin(-1e6, 1e6, P("x_max", 1.0))
        cuboid_ymin, cuboid_ymax = self._spin(-1e6, 1e6, P("y_min", 0.0)), self._spin(-1e6, 1e6, P("y_max", 1.0))
        cuboid_zmin, cuboid_zmax = self._spin(-1e6, 1e6, P("z_min", 0.0)), self._spin(-1e6, 1e6, P("z_max", 1.0))
        cuboid_panel = _panel(self._domain_row("x:", cuboid_xmin, "to", cuboid_xmax),
                              self._domain_row("y:", cuboid_ymin, "to", cuboid_ymax),
                              self._domain_row("z:", cuboid_zmin, "to", cuboid_zmax))

        sphere_cx, sphere_cy = self._spin(-1e6, 1e6, P("cx", 0.5)), self._spin(-1e6, 1e6, P("cy", 0.5))
        sphere_cz, sphere_r = self._spin(-1e6, 1e6, P("cz", 0.5)), self._spin(1e-6, 1e6, P("r", 0.2))
        sphere_panel = _panel(self._domain_row("Center:", sphere_cx, sphere_cy, sphere_cz, tip="Center (x, y, z)"),
                              self._domain_row("Radius:", sphere_r))
        for _w in (rect_panel, disk_panel, ellipse_panel, tri_panel, poly_panel, cuboid_panel, sphere_panel):
            row_layout.addWidget(_w)

        type_panel_map = {
            "Rectangle": rect_panel, "Disk": disk_panel, "Ellipse": ellipse_panel,
            "Triangle": tri_panel, "Polygon": poly_panel,
            "Cuboid": cuboid_panel, "Sphere": sphere_panel,
        }

        def _update_type_panels():
            cur = type_combo.currentText()
            for _t, _panel in type_panel_map.items():
                _panel.setVisible(_t == cur)

        type_combo.currentTextChanged.connect(lambda _t: _update_type_panels())
        type_combo.setCurrentText(shape_type)
        _update_type_panels()

        self.custom_geom_shapes_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget,
            'op_row_widget': op_row_widget,
            'op_combo': op_combo,
            'type_combo': type_combo,
            'rect_xmin': rect_xmin, 'rect_xmax': rect_xmax,
            'rect_ymin': rect_ymin, 'rect_ymax': rect_ymax,
            'disk_cx': disk_cx, 'disk_cy': disk_cy, 'disk_r': disk_r,
            'ell_cx': ell_cx, 'ell_cy': ell_cy, 'ell_a': ell_a, 'ell_b': ell_b, 'ell_angle': ell_angle,
            'tri_verts': tri_verts,
            'poly_verts': poly_verts,
            'cuboid_xmin': cuboid_xmin, 'cuboid_xmax': cuboid_xmax,
            'cuboid_ymin': cuboid_ymin, 'cuboid_ymax': cuboid_ymax,
            'cuboid_zmin': cuboid_zmin, 'cuboid_zmax': cuboid_zmax,
            'sphere_cx': sphere_cx, 'sphere_cy': sphere_cy, 'sphere_cz': sphere_cz, 'sphere_r': sphere_r,
        }
        self.custom_geom_shape_rows.append(row_data)
        op_row_widget.setVisible(len(self.custom_geom_shape_rows) > 1)

        def _remove():
            row_widget.deleteLater()
            if row_data in self.custom_geom_shape_rows:
                self.custom_geom_shape_rows.remove(row_data)
            # The shape now first in the list (if any) has nothing before
            # it any more -- hide its combine-op row the same way it's
            # hidden for whichever shape is added first.
            if self.custom_geom_shape_rows:
                self.custom_geom_shape_rows[0]['op_row_widget'].setVisible(False)

        remove_btn.clicked.connect(_remove)
        remove_btn.clicked.connect(self._schedule_sketch_from_rows)
        up_btn.clicked.connect(lambda: self._move_custom_geom_shape(row_data, -1))
        down_btn.clicked.connect(lambda: self._move_custom_geom_shape(row_data, 1))
        up_btn.clicked.connect(self._schedule_sketch_from_rows)
        down_btn.clicked.connect(self._schedule_sketch_from_rows)
        # Any edit in this shape's fields redraws the sketch (Shape -> Sketch).
        for _k, _w in row_data.items():
            if isinstance(_w, QDoubleSpinBox):
                _w.valueChanged.connect(self._schedule_sketch_from_rows)
            elif isinstance(_w, QLineEdit):
                _w.textChanged.connect(self._schedule_sketch_from_rows)
            elif isinstance(_w, QComboBox):
                _w.currentIndexChanged.connect(self._schedule_sketch_from_rows)
        self._schedule_sketch_from_rows()
        return row_data

    # ── Sketch (Domain -> Shape -> Sketch) ─────────────────────
    def _show_sketch_tab(self, on):
        """Show the Sketch tab beside the Training Log while Shape is Sketch
        (switching to it, with the log's area enlarged so there is room to
        draw); hide it otherwise."""
        if not hasattr(self, 'right_tabs'):
            return
        self.right_tabs.setTabVisible(self._sketch_tab_index, on)
        if on:
            self.sketch_panel.set_dim(3 if self.radio_3d.isChecked() else 2)
            self.right_tabs.setCurrentIndex(self._sketch_tab_index)
            self._on_right_tab_changed(self._sketch_tab_index)
            QApplication.processEvents()
            self.sketch_panel.set_shapes(self._custom_geom_shapes_list(), fit=True)
        elif self.right_tabs.currentIndex() == self._sketch_tab_index:
            self.right_tabs.setCurrentWidget(self.results_tabs)

    def _on_right_tab_changed(self, idx):
        """(The graphics area has room for the Sketch now; nothing to resize.)"""
        return

    def _custom_geom_shapes_list(self):
        import json
        try:
            return json.loads(self._build_custom_geom_shapes_json())
        except Exception:
            return []

    def _schedule_sketch_from_rows(self, *_a):
        """Shape rows edited -> redraw the sketch (once, after the edit
        settles; not while the sketch itself is rewriting the rows)."""
        if getattr(self, '_sketch_writing_rows', False) or not hasattr(self, 'sketch_panel'):
            return
        if not getattr(self, '_sketch_sync_pending', False):
            self._sketch_sync_pending = True
            QTimer.singleShot(120, self._sketch_from_rows)

    def _sketch_from_rows(self):
        self._sketch_sync_pending = False
        if hasattr(self, 'sketch_panel') and not getattr(self, '_sketch_writing_rows', False):
            self.sketch_panel.set_shapes(self._custom_geom_shapes_list())

    def _on_sketch_edited(self, shapes):
        """The sketch changed -> rewrite the Custom-geometry shape rows (the
        single source the config and the generated script are built from)."""
        import json
        self._sketch_writing_rows = True
        try:
            self._apply_custom_geom_shapes_json(json.dumps(shapes))
        finally:
            self._sketch_writing_rows = False
        if hasattr(self, 'workspace'):
            self._refresh_bc_sketch_boxes()

    # ── Boundary conditions as model-tree nodes ────────────────
    def _schedule_bc_nodes_sync(self, *_a):
        if not getattr(self, '_bc_nodes_pending', False):
            self._bc_nodes_pending = True
            QTimer.singleShot(0, self._sync_bc_nodes)

    def _sync_bc_nodes(self):
        """One child node of "Boundary conditions" per BC row, named and
        coloured like the sketch draws its edges; each row's panel lives on
        its node's page."""
        self._bc_nodes_pending = False
        ws = getattr(self, 'workspace', None)
        if ws is None or "bcs" not in ws.pages:
            return
        from pinnstudio.ui.sketch_canvas import BC_COLORS
        live = set()
        for i, e in enumerate(self.custom_bc_list):
            key = e.setdefault('_node', f"bc:{id(e)}")
            live.add(key)
            try:
                desc = self._bc_weight_desc(e)
            except RuntimeError:
                continue
            label = f"BC {i + 1}  ·  {desc}"
            color = BC_COLORS[i % len(BC_COLORS)]
            if key not in ws.pages:
                card = QGroupBox("Condition")
                QVBoxLayout(card).addWidget(e['widget'])
                e['card'] = card
                ws.add_node(key, label, str(i + 1), color,
                            "One boundary condition: its type, the output it applies to, where on the "
                            "boundary (Where:) and its value.", parent="bcs", widgets=[card])
                box = QGroupBox("On the domain")
                lbl = QLabel("")
                lbl.setWordWrap(True)
                QVBoxLayout(box).addWidget(lbl)
                box.setVisible(False)
                e['sketch_box'], e['sketch_lbl'] = box, lbl
                ws.add_widget(key, box)
            ws.set_label(key, label, str(i + 1), color)
            hdr = e.get('header')
            if hdr is not None:
                try:
                    hdr.setText(f"── BC {i + 1} ──" + (" \U0001F512" if e.get('locked') else ""))
                    hdr.setVisible(False)   # the page title names it
                except RuntimeError:
                    pass
        for key in [k for k in list(ws.pages) if k.startswith("bc:") and k not in live]:
            ws.remove_node(key)

    # ── Sketch: edges -> Boundary Conditions rows ──────────────
    _SKETCH_NO_WHERE_TYPES = ("pointset", "pointset_operator")

    def _sketch_bc_rows(self):
        """The BC rows the sketch can colour edges by (those with a Where:)."""
        rows = []
        for i, e in enumerate(getattr(self, 'custom_bc_list', [])):
            if e['type'].currentData() in self._SKETCH_NO_WHERE_TYPES:
                continue
            rows.append({"index": i, "label": f"BC {i + 1}: {self._bc_weight_desc(e)}",
                         "expr": e['location'].text()})
        return rows

    def _sketch_domain(self):
        names = {"x_min": self.x_min, "x_max": self.x_max, "y_min": self.y_min, "y_max": self.y_max,
                 "z_min": self.z_min, "z_max": self.z_max, "t_min": self.t_min}
        return {n: w.value() for n, w in names.items()}

    def _sketch_bcs_changed(self, *_a):
        for _p in (getattr(self, 'sketch_panel', None), getattr(self, 'domain_view', None)):
            if _p is not None and _p.isVisible():
                _p.refresh_bcs()
        if hasattr(self, 'workspace'):
            self._refresh_bc_sketch_boxes()

    def _reveal_bc_row(self, entry):
        """Scroll the left panel to a BC row and briefly highlight it."""
        def _go():
            if hasattr(self, 'workspace'):
                self.workspace.reveal(entry['widget'])
            loc = entry['location']
            old = loc.styleSheet()
            loc.setStyleSheet("QLineEdit { border: 2px solid #ffd43b; }")
            QTimer.singleShot(1600, lambda: loc.setStyleSheet(old))
        QTimer.singleShot(50, _go)

    def _on_sketch_new_bc(self, where):
        entry = self._add_custom_bc_entry(bc_type="dirichlet", location=where, value="0")
        self._update_bc_mode_visibility()
        if hasattr(self, 'weights_main_layout'):
            self._build_weight_inputs(self.num_outputs_spin.value())
        self.log_box.append(f"✏ Sketch: added BC {len(self.custom_bc_list)} (Dirichlet, value 0) on the selected edges.")
        self._reveal_bc_row(entry)
        self._gpanel().refresh_bcs()

    def _on_sketch_add_to_bc(self, index, where):
        from pinnstudio.ui.sketch_edges import add_to_where
        if not (0 <= index < len(self.custom_bc_list)):
            return
        entry = self.custom_bc_list[index]
        cur = entry['location'].text().strip()
        if cur in ("", "True"):
            QMessageBox.information(self, "Add to BC",
                                    f"BC {index + 1} already applies to the whole boundary, so it covers these "
                                    "edges already. Use “New BC on selected” to give them their own condition.")
            return
        entry['location'].setText(add_to_where(cur, where))
        self._reveal_bc_row(entry)
        self._gpanel().refresh_bcs()

    def _on_sketch_remove_from_bcs(self, exprs):
        from pinnstudio.ui.sketch_edges import remove_from_where
        changed, kept, emptied = [], [], []
        for i, e in enumerate(self.custom_bc_list):
            if e['type'].currentData() in self._SKETCH_NO_WHERE_TYPES:
                continue
            new = remove_from_where(e['location'].text(), exprs)
            if new is None:
                continue
            if new:
                e['location'].setText(new)
                changed.append(i + 1)
            else:
                emptied.append(i + 1)
        # an edge that is still coloured is picked by a Where: written another way
        gp = self._gpanel()
        gp.refresh_bcs()
        if emptied:
            QMessageBox.information(
                self, "Remove from BC",
                "No edges would be left in " + ", ".join(f"BC {n}" for n in emptied) +
                ", so it was left as it is -- remove the row itself with its ✕ if it's no longer needed.")
        bc_of = gp.canvas.edge_bcs()
        if any(bc_of[k] for k in gp.canvas.sel_edges if k < len(bc_of)) and not emptied:
            QMessageBox.information(
                self, "Remove from BC",
                "Some selected edges are still picked by a BC whose Where: was typed differently "
                "(for example 'True' or 'left'); edit that row's Where: in Boundary Conditions.")

    def _move_custom_geom_shape(self, row_data, delta):
        """Move one shape up/down in the combine order -- CSG ops are not
        commutative (Triangle-subtract-Disk and Disk-subtract-Triangle are
        different shapes), so letting the user fix the order without
        deleting and re-adding every shape matters here more than it does
        for e.g. the Parameter Sweep rows, which have no such ordering
        dependency."""
        if row_data not in self.custom_geom_shape_rows:
            return
        i = self.custom_geom_shape_rows.index(row_data)
        j = i + delta
        if j < 0 or j >= len(self.custom_geom_shape_rows):
            return
        rows = self.custom_geom_shape_rows
        rows[i], rows[j] = rows[j], rows[i]
        # Re-lay the row widgets out in the new order (no widgets are
        # destroyed here -- just taken out of the layout and re-added).
        while self.custom_geom_shapes_layout.count():
            self.custom_geom_shapes_layout.takeAt(0)
        for row in rows:
            self.custom_geom_shapes_layout.addWidget(row['widget'])
        self._schedule_sketch_from_rows()
        # Only the first shape in the list has no combine-op; every other
        # position (including one that just became/stopped being first)
        # needs its op row's visibility re-checked.
        for idx, row in enumerate(rows):
            row['op_row_widget'].setVisible(idx > 0)

    def _custom_geom_row_to_dict(self, row, is_first):
        """One shape row's widget values -> the plain dict codegen.py
        reads (see config.py's geom_custom_shapes_json docs). 'op' is
        omitted for the first shape -- it's the starting geometry, not
        combined with anything."""
        shape_type = row['type_combo'].currentText()
        if shape_type == "Rectangle":
            shape_params = {
                "x_min": row['rect_xmin'].value(), "x_max": row['rect_xmax'].value(),
                "y_min": row['rect_ymin'].value(), "y_max": row['rect_ymax'].value(),
            }
        elif shape_type == "Disk":
            shape_params = {"cx": row['disk_cx'].value(), "cy": row['disk_cy'].value(), "r": row['disk_r'].value()}
        elif shape_type == "Ellipse":
            shape_params = {
                "cx": row['ell_cx'].value(), "cy": row['ell_cy'].value(),
                "a": row['ell_a'].value(), "b": row['ell_b'].value(), "angle": row['ell_angle'].value(),
            }
        elif shape_type == "Triangle":
            shape_params = {"vertices_text": row['tri_verts'].text()}
        elif shape_type == "Polygon":
            shape_params = {"vertices_text": row['poly_verts'].text()}
        elif shape_type == "Cuboid":
            shape_params = {
                "x_min": row['cuboid_xmin'].value(), "x_max": row['cuboid_xmax'].value(),
                "y_min": row['cuboid_ymin'].value(), "y_max": row['cuboid_ymax'].value(),
                "z_min": row['cuboid_zmin'].value(), "z_max": row['cuboid_zmax'].value(),
            }
        else:  # Sphere
            shape_params = {
                "cx": row['sphere_cx'].value(), "cy": row['sphere_cy'].value(), "cz": row['sphere_cz'].value(),
                "r": row['sphere_r'].value(),
            }
        entry = {"type": shape_type, "params": shape_params}
        if not is_first:
            entry["op"] = row['op_combo'].currentData()
        return entry

    def _build_custom_geom_shapes_json(self):
        """The Custom-geometry shape list, in builder order -> JSON, for
        config.geom_custom_shapes_json. Mirrors _build_custom_bc_json's
        role for the Boundary Conditions panel: this is the single
        source of truth codegen.py reads to build the CSG chain."""
        import json
        entries = [
            self._custom_geom_row_to_dict(row, is_first=(i == 0))
            for i, row in enumerate(self.custom_geom_shape_rows)
        ]
        return json.dumps(entries)

    def _apply_custom_geom_shapes_json(self, geom_custom_shapes_json):
        """Rebuild the Custom-geometry shape list from a saved
        geom_custom_shapes_json (loading a saved config/template)."""
        import json
        for row in list(self.custom_geom_shape_rows):
            row['widget'].deleteLater()
        self.custom_geom_shape_rows.clear()
        if not geom_custom_shapes_json:
            return
        try:
            entries = json.loads(geom_custom_shapes_json)
        except (ValueError, TypeError):
            entries = []
        if not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            shape_type = entry.get("type")
            if shape_type not in self.CUSTOM_GEOM_SHAPE_TYPES_ALL:
                continue
            params = entry.get("params") or {}
            op = entry.get("op") or "union"
            self._add_custom_geom_shape(shape_type=shape_type, op=op, params=params)

    _PINNSTUDIO_EXAMPLES = (
        "1D Heat", "1D Allen-Cahn", "1D Burgers", "1D Schrödinger",
        "2D Heat", "2D Allen-Cahn (Mattey & Ghosh)", "2D Allen-Cahn (Wight & Zhao)",
        "2D Burgers (Mathias)", "2D Poisson (L-Shape)", "2D Poisson (Disk)",
        "3D Heat", "3D Poisson (Sphere)",
    )

    # Settings that are the user's own preferences rather than part of a
    # problem: loading a DeepXDE example keeps them as they are (unless the
    # example itself sets one, e.g. a coarser plot grid for a 3D problem).
    _DEEPXDE_KEEP_FIELDS = (
        "save_dir", "sweep_save_dir", "sweep_label", "gpu_device_index", "gpu_memory_fraction",
        "use_random_seed", "random_seed",
        "plot_colormap", "plot_levels", "plot_dpi", "plot_colorbar", "plot_figsize_mode",
        "plot_figsize_w", "plot_figsize_h", "plot_linewidth", "plot_linewidth_anim", "plot_fps",
        "plot_title_fontsize", "plot_subplot_title_fontsize", "plot_axis_label_fontsize",
        "loss_plot_mode", "loss_display_every", "loss_plot_linewidth", "loss_plot_log_y",
    )

    def _on_quick_example_category_changed(self, category):
        """Refill "Load example:" with the chosen category's examples. The
        problem currently set up is left as it is (the list shows "None")."""
        self.quick_examples_combo.set_entries(_ex_catalog.for_category_all(category))
        self._update_bc_mode_visibility()

    def _apply_deepxde_example(self, name, inverse=False):
        """Load one of DeepXDE's examples: every setting comes from
        deepxde_examples.py (DeepXDE's own script), applied through
        _apply_config() exactly like opening a saved problem. `inverse`
        loads the example's Inverse-mode version (Example.inverse)."""
        ex = _dx_examples.BY_NAME[name]
        try:
            current = self._build_config()
            keep = {k: getattr(current, k) for k in self._DEEPXDE_KEEP_FIELDS if hasattr(current, k)}
        except Exception:
            keep = {}
        # The Save-to folder as typed (config.save_dir is the per-run,
        # timestamped folder inside it).
        keep["save_dir"] = self.save_dir_input.text()
        cfg = _dx_examples.build_config(ex, REFERENCE_DATA_DIR, keep=keep, inverse=inverse)
        self._applying_template = True
        try:
            self._apply_config(cfg)
        finally:
            self._applying_template = False
        # Show it in the list: switch to its category first if needed.
        want_cat = _ex_catalog.INVERSE if (ex.category == _dx_examples.INVERSE or inverse) else _ex_catalog.FORWARD
        if name not in self.quick_examples_combo.keys() or (
                inverse and self.quick_examples_category_combo.currentText() != _ex_catalog.INVERSE):
            self.quick_examples_category_combo.blockSignals(True)
            self.quick_examples_category_combo.setCurrentText(want_cat)
            self.quick_examples_category_combo.blockSignals(False)
            self.quick_examples_combo.set_entries(_ex_catalog.for_category_all(want_cat))
        self.quick_examples_combo.blockSignals(True)
        self.quick_examples_combo.setCurrentText(name)
        self.quick_examples_combo.blockSignals(False)
        self._current_template = name
        # Only the note: _update_bc_mode_visibility() would also rebuild the
        # loss-weight rows, resetting the example's weights to defaults.
        if hasattr(self, 'custom_bc_note'):
            self.custom_bc_note.setText(self._bc_note_with_transform(f"Pre-filled from the \"{_ex_catalog.display_name(name)}\" example."))
        # What this example looks like in the panels right after loading --
        # Solve compares against it to log any settings the user changed.
        try:
            self._example_snapshot = (name, self._build_config())
        except Exception:
            self._example_snapshot = None
        self._current_path = None
        self._current_problem_name = name
        self._mark_clean()
        self.log_box.append(f"✅ DeepXDE example loaded: {_ex_catalog.display_name(name)}"
                            + (" (Inverse mode)" if inverse else ""))
        self.log_box.append(f"   Source: {ex.source}")
        if ex.note:
            self.log_box.append(f"   Note: {ex.note}")
        if inverse and ex.inverse_note:
            self.log_box.append(f"   Inverse: {ex.inverse_note}")

    def _on_quick_example_selected(self, text):
        # loading an example replaces the problem: offer to save it first
        if text != "None" and not getattr(self, "_from_start_gallery", False) \
                and not self._maybe_save_changes():
            self.quick_examples_combo.blockSignals(True)
            self.quick_examples_combo.setCurrentText(getattr(self, "_shown_example", None) or "None")
            self.quick_examples_combo.blockSignals(False)
            return
        self._shown_example = text
        if text in _dx_examples.BY_NAME:
            # A forward DeepXDE example with an Inverse mode, picked from the
            # Inverse list, loads its Inverse version (like the templates).
            self._apply_deepxde_example(
                text, inverse=(self.quick_examples_category_combo.currentText() == _ex_catalog.INVERSE
                               and _ex_catalog.has_inverse_mode(text)))
            return
        self._example_snapshot = None
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
        # The Inverse category loads a template in Inverse mode (which brings
        # in the template's own measured-data file). The Forward list leaves
        # the Forward/Inverse choice as it is, as before.
        if (self.quick_examples_category_combo.currentText() == _ex_catalog.INVERSE
                and not self.radio_inverse.isChecked()):
            self.radio_inverse.setChecked(True)
        self._current_problem_name = text
        self._current_path = None
        self._mark_clean()

    def _auto_configure_ea(self, ref_dir, is_steady=False):
        """Auto-configure error analysis when a template with ground truth files is loaded."""
        import glob, re, numpy as np
        # Always start from a clean slate: a template with no matching
        # reference data (or none at all) must not silently inherit the
        # previous template's Error Analysis config -- e.g. switching from
        # 2D Poisson (Disk) to 2D Burgers before 2D Burgers has its own
        # COMSOL reference file dropped in must not keep comparing 2D
        # Burgers' training run against Disk's leftover ground truth.
        self._ea_settings = None
        if not ref_dir or not os.path.isdir(ref_dir):
            return
        # This problem's real output names -- used to recognize a reference
        # file that names its own output explicitly (e.g. "t_0_u.txt" /
        # "t_0_v.txt" for a 2-output template like 2D Burgers (Mathias)),
        # matched case-insensitively. Falls back to a single "u" if the
        # output-name boxes aren't built yet.
        try:
            _out_names = [(w.text().strip() or f"output{_oi}")
                          for _oi, w in enumerate(self.output_name_inputs)]
        except Exception:
            _out_names = []
        if not _out_names:
            _out_names = ["u"]
        _name_to_idx = {n.lower(): i for i, n in enumerate(_out_names)}
        if is_steady:
            # Steady-state (time-independent) templates have a single
            # reference file with no time column at all -- "solution.txt",
            # columns x,y,u for 2D / x,y,z,u for 3D -- matching exactly what
            # PINNStudio's own steady-state "Export solution data" writes.
            # A multi-output problem can additionally provide one
            # "solution_<output_name>.txt" per output, the steady-state
            # counterpart of the time-dependent "t_<time>_<output_name>.txt"
            # naming below. Points outside a non-rectangular geometry
            # (L-Shape's missing quadrant, Disk/Sphere corners of the
            # bounding box) are NaN in the file; codegen's error-analysis
            # loader drops those rows before computing metrics, so nothing
            # needs filtering here.
            _found = []
            _sol_path = os.path.join(ref_dir, "solution.txt")
            if os.path.isfile(_sol_path):
                _found.append((0.0, _sol_path, None))
            for _on in _out_names:
                _named_path = os.path.join(ref_dir, f"solution_{_on}.txt")
                if os.path.isfile(_named_path):
                    _found.append((0.0, _named_path, _name_to_idx[_on.lower()]))
            if not _found:
                return
            self._ea_settings = {
                'files': _found,
                'do_line': True,
                'do_surface': True,
                # Round 31: default changed to L2-only -- these three
                # previously had no effect at all (see PINNConfig.
                # ea_metric_l2/mse/max's own comment), so their old
                # all-checked default silently did nothing. Now that they
                # actually gate the Restore/Solve Error Analysis metrics
                # print line and error_metrics.txt columns, the default
                # matches what the user asked for.
                'do_l2': True,
                'do_mse': False,
                'do_max': False,
            }
            _n_groups = len({sel for _, _, sel in _found})
            _suffix = f" across {_n_groups} outputs" if _n_groups > 1 else ""
            self.log_box.append(f"✅ Error analysis auto-configured — {len(_found)} ground truth file(s) (steady-state){_suffix} from template")
            return
        # Ground-truth snapshots for error analysis must match either
        # "t_<number>.txt" (the implicit-default-output case, unchanged
        # from before) or "t_<number>_<output_name>.txt" (an explicit
        # per-output snapshot -- e.g. 2D Burgers (Mathias)'s own
        # "t_0_u.txt"/"t_0_v.txt", auto-assigned to whichever of this
        # problem's outputs is named "u"/"v"). A sibling file like
        # "t_1.0_LessData.txt" (a thinned-down version of the same snapshot
        # meant only as an easier Inverse observation set, see below) is
        # intentionally NOT another ground truth snapshot and must not be
        # swept in just because it also starts with "t_" and ends with
        # ".txt" -- nor is a "t_<n>_<word>.txt" file whose <word> doesn't
        # match any real output name of this problem (dropped rather than
        # silently mis-assigned).
        _plain_re = re.compile(r'^t_[0-9]+(\.[0-9]+)?\.txt$')
        _named_re = re.compile(r'^t_([0-9]+(?:\.[0-9]+)?)_([A-Za-z0-9]+)\.txt$')
        txt_files = sorted(glob.glob(os.path.join(ref_dir, 't_*.txt')))
        if not txt_files:
            return
        valid_files = []  # (t_val, fp, output_selector)
        for fp in txt_files:
            base = os.path.basename(fp)
            m_named = _named_re.match(base)
            if m_named and m_named.group(2).lower() in _name_to_idx:
                _sel = _name_to_idx[m_named.group(2).lower()]
            elif _plain_re.match(base):
                _sel = None
            else:
                continue
            try:
                d = np.loadtxt(fp)
                if d.ndim == 1: d = d.reshape(1, -1)
                is_2d = self.radio_2d.isChecked()
                is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
                t_val = float(d[0, 3]) if is_3d else (float(d[0, 2]) if is_2d else float(d[0, 1]))
                valid_files.append((t_val, fp, _sel))
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
            # Round 31: see the matching comment on the steady-state
            # branch above -- default changed to L2-only now that these
            # flags actually do something.
            'do_l2': True,
            'do_mse': False,
            'do_max': False,
        }
        _n_groups = len({sel for _, _, sel in valid_files})
        _suffix = f" across {_n_groups} outputs" if _n_groups > 1 else ""
        self.log_box.append(f"✅ Error analysis auto-configured — {len(valid_files)} ground truth files{_suffix} from template")
        # Auto-select the end-time (largest t) reference file as the Inverse
        # observed-data file, so the user doesn't have to browse for it. If a
        # thinned "<name>_LessData.txt" sibling of that snapshot exists in
        # the same folder, prefer it for the Inverse default instead: fewer
        # observation points make the inverse fit numerically easier, while
        # the full snapshot above stays untouched as the error-analysis
        # ground truth. Unchanged by multi-output grouping -- still just the
        # single latest-time file overall (Inverse's own observed-data field
        # is a separate, single-file mechanism from Error Analysis' now
        # possibly-multiple output groups).
        # A template with its own observed-data files (INVERSE_AUTO_OBS,
        # e.g. 2D Burgers (Mathias)'s U and V snapshots, one row each) keeps
        # them -- this single-file default would overwrite its first row.
        _tpl_obs = self.INVERSE_AUTO_OBS.get(getattr(self, '_current_template', ''))
        if hasattr(self, 'inv_data_path') and not (_tpl_obs and _tpl_obs[0][0]):
            _end_time_file = valid_files[-1][1]
            _less_data_variant = _end_time_file[:-4] + "_LessData.txt"
            _inv_default_file = _less_data_variant if os.path.isfile(_less_data_variant) else _end_time_file
            self.inv_data_path.setText(_inv_default_file)
            # A per-output snapshot ("t_1.0_v.txt") is data for THAT output,
            # not output 1 -- point the row's output selector at it.
            _end_sel = valid_files[-1][2]
            if (_end_sel is not None and hasattr(self, 'inv_obs_output_combo')
                    and _end_sel < self.inv_obs_output_combo.count()):
                self.inv_obs_output_combo.setCurrentIndex(_end_sel)
            # Remember which template this path was just auto-loaded for --
            # _sync_inverse_multi_setup reads this to tell "a real file we
            # just found for the CURRENTLY selected template" apart from
            # "stale leftovers from whatever template was selected before",
            # so it can preserve the former and still safely discard the
            # latter (see its own comment for the bug this fixes: 1D
            # Schrodinger's own INVERSE_AUTO_OBS entry has an intentionally
            # empty default path, and switching Problem Type to Inverse
            # right after this ran was unconditionally wiping this
            # just-auto-loaded path back to empty).
            self._ea_auto_inv_path = _inv_default_file
            self._ea_auto_inv_template = getattr(self, '_current_template', '')
            self.log_box.append(
                f"📂 Inverse observed data auto-loaded: {os.path.basename(_inv_default_file)} "
                f"(t={valid_files[-1][0]:.4g})")

    # ── Plot type change ──────────────────────────────────────
    def _on_plot_type_changed(self, text):
        # Every plot type auto-pops the unified settings dialog on
        # selection now -- exactly matching Restore & Visualize's own
        # plot-type combo (_on_restore_viz_changed -> always
        # _on_restore_viz_settings()), and removing the need for a
        # separate "⚙" settings button beside this combo (removed).
        # _plot_type_prev is deliberately NOT updated here -- it still
        # holds whatever was selected before this change, which is
        # exactly what _on_plot_settings()'s Cancel button needs to
        # restore. It's updated inside _on_plot_settings itself instead,
        # once a selection actually sticks (either Cancel's own revert, or
        # this dialog's OK).
        self._on_plot_settings(text)

    def _on_plot_output_combo_changed(self, text):
        """Show the custom expression/label fields only while "Custom..."
        is selected in the Plot output dropdown -- see _build_config()
        (which only serializes them in that case) and codegen.py's
        _extract_plot_field (which only evaluates them in that case)."""
        _is_custom = (text == "Custom...")
        self.plot_custom_expr_input.setVisible(_is_custom)
        self.plot_custom_label_input.setVisible(_is_custom)

    def _on_restore_output_combo_changed(self, text):
        """Mirrors _on_plot_output_combo_changed for the Restore panel's
        own output selector -- see _on_restore (which only reads these two
        fields while "Custom..." is selected) and _build_restore_script/
        _build_restore_script_ta (whose own _extract_plot_field helper only
        evaluates the expression in that case)."""
        _is_custom = (text == "Custom...")
        self.restore_custom_expr_input.setVisible(_is_custom)
        self.restore_custom_label_input.setVisible(_is_custom)


    # _on_error_analysis_settings / _run_error_analysis / this file's
    # first _on_ea_done were dead code -- only reachable from a
    # viz_type == "\U0001F4CA Error Analysis" branch in _on_plot_settings()
    # that could never actually fire (plot_type_combo never contains that
    # string; the real Error Analysis entry point is the separate ea_btn
    # -> _on_error_analysis_btn()). This round found that the "real,
    # reachable" _on_ea_done this comment used to point to (connected to
    # _ea_thread.done_sig inside _execute_error_analysis) was itself dead
    # for the same reason -- _execute_error_analysis/_execute_error_analysis_v2
    # (and their own _on_ea_done/_on_ea_v2_done done-handlers) were never
    # called from anywhere either; the actual, live Error Analysis path
    # runs entirely through codegen.py's generated scripts (ea_btn's
    # dialog just populates self._ea_settings, which feeds
    # generate_script()'s config fields -- see _on_error_analysis_btn/
    # _show_ea_dialog and generate_script()'s own EA config handling).
    # All four were removed together.

    def _add_ta_step_group(self, t_start=0.0, t_end=1.0, steps=10):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        row_layout.addWidget(QLabel("Start time:"))

        t_start_sb = DomainSpinBox()
        t_start_sb.setRange(0.0, 1e6)
        # DomainSpinBox (10 decimals), like self.t_min/t_max: a time
        # domain that isn't a round number -- e.g. 1D Schrodinger's
        # [0, pi/2] -- has to be enterable here too. The range is set
        # before the value, since QDoubleSpinBox clamps/rounds on setValue.
        t_start_sb.setValue(t_start)
        t_start_sb.setFixedHeight(26); t_start_sb.setFixedWidth(85)
        row_layout.addWidget(t_start_sb)

        row_layout.addWidget(QLabel("→"))

        row_layout.addWidget(QLabel("End time:"))

        t_end_sb = DomainSpinBox()
        t_end_sb.setRange(0.0, 1e6)
        t_end_sb.setValue(t_end)
        t_end_sb.setFixedHeight(26); t_end_sb.setFixedWidth(85)
        row_layout.addWidget(t_end_sb)

        row_layout.addWidget(QLabel("Steps (n):"))

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

    def _add_inverse_var_row(self, name=None, init=1.0, is_primary=None, true_value=None):
        """Add one trainable-variable row to the Inverse panel. The first
        (primary) row is always present and cannot be removed -- it is the
        one that participates in INVERSE_AUTO_CONST's automatic PDE-box
        substitution when a Quick Example is loaded, exactly like the
        single trainable_variable this generalizes. Rows beyond the first
        are optional, freely added/removed, and share the panel's single
        measured-data file / output selector / loss weight (all trainable
        variables are fit against the same shared observation dataset).
        true_value is the known ground-truth value for a built-in
        template's auto-substituted PDE constant (see INVERSE_AUTO_VARS
        below), shown in its own editable "true value:" box so the user
        can see, correct, or (for a manually added/custom variable) fill
        in the known answer themselves -- _build_inverse_variables_json
        reads the box's current value (not this argument) when it
        serializes the variable list for codegen.py's parameter-
        convergence plot to draw a dashed reference line at. None (the
        default) leaves the box blank -- for a manually added variable or
        a legacy saved config with no known true value -- in which case
        that line is simply not drawn."""
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

        row_layout.addWidget(QLabel("true value:"))
        true_edit = TrueValueLineEdit(true_value)
        true_edit.setFixedHeight(26); true_edit.setFixedWidth(85)
        true_edit.setToolTip(
            "Known ground-truth value for this variable, if you have one.\n"
            "Pre-filled for built-in templated examples; leave blank for a\n"
            "custom PDE, or fill it in yourself if you know the answer.\n"
            "Only used to draw a dashed reference line on the parameter-\n"
            "convergence plot -- it never affects training itself.")
        row_layout.addWidget(true_edit)

        remove_btn = None
        if is_primary:
            name_edit.editingFinished.connect(self._on_inv_param_name_changed)
        else:
            # Secondary rows can also be the target of an active
            # substitution (e.g. the diffusion-reaction template's "kf"
            # row) -- _on_inv_param_name_changed re-derives every active
            # entry's current name regardless of which row's edit fired
            # this, so the same slot covers both cases.
            name_edit.editingFinished.connect(self._on_inv_param_name_changed)
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
            'true': true_edit,
        }
        self.inv_var_rows.append(row_data)
        name_edit.textChanged.connect(lambda _t: self._refresh_equation_previews())
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
            entry = {'name': nm, 'init': r['init'].value()}
            _true = r['true'].value()
            if _true is not None:
                entry['true'] = _true
            variables.append(entry)
        if not variables:
            variables.append({'name': 'trainable_variable_1', 'init': 1.0})
        return json.dumps(variables)

    def _add_inverse_data_row(self, path="", output_idx=0, weight=100.0, is_primary=None, custom_expr=""):
        """Add one measured-data-file row to the Inverse panel. The first
        (primary) row is always present -- it's the one the legacy single-
        file config fields (inverse_data_file / inverse_obs_output_idx /
        loss_weight_obs) mirror, so every existing call site and every
        previously-saved config keeps working unchanged. Rows beyond the
        first are optional, freely added/removed; unlike trainable
        variables (which all share one measured-data setup), each
        additional file here is its OWN observation dataset with its own
        "which output" selector and its own loss weight, since each
        becomes its own separate loss term.

        The "output:" dropdown also offers "Custom..." (last item, same
        convention as the Results panel's Plot-output dropdown), for a
        measured-data file whose value is a DERIVED field the network's
        raw outputs don't have as a single column -- e.g. 1D Schrodinger's
        data is the magnitude |h| = sqrt(u**2+v**2), not output u or v
        alone. Selecting it reveals a expression box (custom_expr);
        codegen.py's _parse_inverse_obs_files/_make_obs_func evaluate it
        against ALL of this problem's outputs by name, via
        dde.icbc.PointSetOperatorBC, instead of matching a single raw
        output column."""
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
        output_combo.addItem("Custom...")
        custom_edit = QLineEdit()
        custom_edit.setText(custom_expr)
        custom_edit.setPlaceholderText("expression, e.g. sqrt(u**2+v**2)")
        custom_edit.setFixedHeight(26)
        custom_edit.setVisible(bool(custom_expr))
        if custom_expr:
            output_combo.setCurrentIndex(output_combo.count() - 1)
        elif 0 <= output_idx < output_combo.count() - 1:
            output_combo.setCurrentIndex(output_idx)

        def _on_obs_output_changed(_idx, _combo=output_combo, _edit=custom_edit):
            _edit.setVisible(_combo.currentText() == "Custom...")
        output_combo.currentIndexChanged.connect(_on_obs_output_changed)

        meta_row.addWidget(output_combo)
        meta_row.addWidget(custom_edit)
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
            'custom_expr': custom_edit,
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
            _is_custom = (r['output_combo'].currentText() == "Custom...")
            files.append({
                'path': r['path'].text().strip(),
                'output_idx': _oi if (_oi >= 0 and not _is_custom) else 0,
                'weight': r['weight'].value(),
                'custom_expr': r['custom_expr'].text().strip() if _is_custom else '',
            })
        if not files:
            files.append({'path': '', 'output_idx': 0, 'weight': 100.0, 'custom_expr': ''})
        return json.dumps(files)

    def _set_combo_data(self, combo, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _fit_combo_width(self, combo, extra=56, min_width=0):
        """Widen a QComboBox so its box is at least as wide as its widest
        item's text, instead of a hand-picked fixed width that can cut off
        a longer option (e.g. "Glorot uniform" or "L-BFGS" getting clipped
        in a box sized for a shorter default). `extra` accounts for the
        dropdown arrow and internal padding; `min_width` is an optional
        floor on top of that. Call this after every combo.addItem(s) call
        -- it reads the combo's current font, so it stays correct even if
        Display Settings later changes the app-wide font size/family."""
        fm = combo.fontMetrics()
        widest = 0
        for i in range(combo.count()):
            w = fm.horizontalAdvance(combo.itemText(i))
            if w > widest:
                widest = w
        combo.setMinimumWidth(max(min_width, widest + extra))

    def _on_adapt_changed(self, text):
        self.rar_widget.setVisible(text == "Residual-based Adaptive Refinement (RAR)")
        self.ta_widget.setVisible(text == "Time Adaptive Training")

    # Adaptive Training reference texts -- one toggle per method, inside
    # that method's own section, so each only shows while its method is
    # selected (it used to be one shared toggle whose text followed the
    # Method dropdown).
    _TA_REFERENCE_TEXT = ("Splits the time domain into consecutive steps (see the step groups\n"
        "below) and trains one model per step in sequence, each only over its\n"
        "own narrow time window instead of the whole domain at once --\n"
        "usually tracks fast-changing solutions better than a single model\n"
        "trained over the full time range.\n"
        "\"Transfer learning\" warm-starts each new step's model from the\n"
        "previous step's trained weights instead of starting fresh every time.\n"
        "\"IC grid resolution\" sets how finely the previous step's solution is\n"
        "sampled to build the next step's initial condition.")
    _RAR_REFERENCE_TEXT = ("Periodically samples a large pool of random points, checks how badly\n"
        "each one currently satisfies the PDE (its \"residual\"), and adds the\n"
        "worst-residual points to training -- concentrating effort where the\n"
        "model is currently least accurate, instead of resampling uniformly.\n"
        "That's one \"round\"; it repeats for the configured number of rounds,\n"
        "retraining for a fixed number of iterations after each one.\n"
        "\"Points from:\" picks which equation's residual to rank candidate\n"
        "points by, for a problem with more than one output/equation.")

    def _add_reference_toggle(self, layout, label, text):
        """A "📖 <label>" checkbox plus the hint text it shows/hides, added
        to `layout` -- the same look as every panel's reference toggle.
        Returns (toggle, hint)."""
        toggle = QCheckBox(f"\U0001F4D6 {label}")
        toggle.setChecked(False)
        self._register_hint(toggle, '#74c0fc')
        hint = QLabel(text)
        self._register_hint(hint, '#74c0fc')
        hint.setWordWrap(True)
        hint.setVisible(False)
        toggle.stateChanged.connect(lambda state, h=hint: h.setVisible(state == 2))
        layout.addWidget(toggle)
        layout.addWidget(hint)
        return toggle, hint

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

    def _lbfgs_config_values(self):
        """lbfgs_maxcor ... lbfgs_maxls for the config: the recommended set
        whenever "Use recommended L-BFGS settings" is ticked (so a hidden
        field left over from a loaded DeepXDE example can't apply), the
        fields' own values otherwise."""
        if self.lbfgs_use_default_cb.isChecked():
            v = dict(LBFGS_RECOMMENDED)
        else:
            v = dict(maxcor=self.lbfgs_maxcor.value(), ftol=self.lbfgs_ftol.value(),
                     gtol=self.lbfgs_gtol.value(), maxiter=self.lbfgs_maxiter.value(),
                     maxfun=self.lbfgs_maxfun.value(), maxls=self.lbfgs_maxls.value())
        ints = ("maxcor", "maxiter", "maxfun", "maxls")
        return {f"lbfgs_{k}": (int(x) if k in ints else float(x)) for k, x in v.items()}

    def _reset_lbfgs_to_recommended(self):
        """Tick "Use recommended L-BFGS settings" and put the recommended
        values in the fields, so unticking starts from them."""
        self.lbfgs_use_default_cb.setChecked(True)
        for k, x in LBFGS_RECOMMENDED.items():
            getattr(self, f"lbfgs_{k}").setValue(x)

    def _on_lbfgs_default_changed(self, state):
        if state == 2 and hasattr(self, 'lbfgs_maxls'):
            for k, x in LBFGS_RECOMMENDED.items():
                getattr(self, f"lbfgs_{k}").setValue(x)
        self.lbfgs_manual_widget.setVisible(state != 2)

    # ── Build PDE inputs ──────────────────────────────────────
    def _n_equations(self):
        """Number of PDE residuals currently set (Round 63)."""
        return self.num_equations_spin.value() if hasattr(self, "num_equations_spin") else self.num_outputs_spin.value()

    def _on_num_equations_changed(self, _m):
        """Rebuild just the PDE rows (keeping what's typed) and the weights."""
        self._build_pde_inputs(self.num_outputs_spin.value(), keep_text=True)
        self._build_weight_inputs(self.num_outputs_spin.value())
        self._refresh_rar_points_combo()

    def _refresh_rar_points_combo(self):
        """RAR's "Points from:" -- one entry per equation, since it picks
        which residual ranks candidate points."""
        if not hasattr(self, "rar_output_combo"):
            return
        idx = self.rar_output_combo.currentIndex()
        n_out, n_eq = self.num_outputs_spin.value(), self._n_equations()
        self.rar_output_combo.clear()
        self.rar_output_combo.addItem("All outputs (combined)" if n_eq == n_out else "All equations (combined)")
        for i in range(n_eq):
            if n_eq == n_out:
                name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
                self.rar_output_combo.addItem(f"Output {i+1} ({name})")
            else:
                self.rar_output_combo.addItem(f"Equation {i+1}")
        if 0 <= idx < self.rar_output_combo.count():
            self.rar_output_combo.setCurrentIndex(idx)

    def _build_pde_inputs(self, n, keep_text=False):
        kept_names = [w.text() for w in self.output_name_inputs] if keep_text else []
        kept_pdes = [w.text() for w in self.pde_inputs] if keep_text else []
        n_eq = self._n_equations()
        for i in reversed(range(self.pde_main_layout.count())):
            w = self.pde_main_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self.pde_inputs.clear()
        self.output_name_inputs.clear()

        is_2d = self.radio_2d.isChecked() if hasattr(self, 'radio_2d') else False
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False

        # Output name i then PDE i, for as many rows as the larger count --
        # identical to the old one-PDE-per-output layout when they match.
        for i in range(max(n, n_eq)):
            if i < n:
                name_row = QHBoxLayout()
                name_row.addWidget(QLabel(f"Output {i+1} name:"))
                name_input = QLineEdit()
                name_input.setText(kept_names[i] if i < len(kept_names) else (["u", "v", "w", "p"][i] if i < 4 else f"u{i+1}"))
                name_input.setFixedHeight(26); name_input.setFixedWidth(55)
                name_row.addWidget(name_input); name_row.addStretch()
                self.output_name_inputs.append(name_input)
                name_input.textChanged.connect(lambda _t: self._refresh_equation_previews())
                nw = QWidget(); nw.setLayout(name_row)
                self.pde_main_layout.addWidget(nw)

            if i < n_eq:
                self.pde_main_layout.addWidget(QLabel(f"PDE {i+1} (residual = 0):"))
                pde_inp = QLineEdit()
                if is_3d:
                    pde_inp.setText("du_t - 0.4 * (du_xx + du_yy + du_zz)" if i == 0 else "dv_t - 0.1 * (dv_xx + dv_yy + dv_zz)")
                elif is_2d:
                    pde_inp.setText("du_t - 0.0001 * (du_xx + du_yy) + 1 * (u**3 - u)" if i == 0 else "dv_t - 0.0001 * (dv_xx + dv_yy) + 1 * (v**3 - v)")
                else:
                    pde_inp.setText("du_t - 0.4 * du_xx" if i == 0 else "dv_t - 0.1 * dv_xx")
                if i < len(kept_pdes):
                    pde_inp.setText(kept_pdes[i])
                pde_inp.setFixedHeight(28)
                pde_inp.textChanged.connect(lambda _t: self._sync_frac_widget())
                self.pde_inputs.append(pde_inp)
                self.pde_main_layout.addWidget(pde_inp)
                # the equation typeset under the box, and names it doesn't know
                from pinnstudio.ui.equation import EquationPreview
                prev = EquationPreview(pde_inp, self._equation_known_names)
                pde_inp._preview = prev
                self.pde_main_layout.addWidget(prev)

        names_ex = ", ".join([["u","v","w","p"][i] if i < 4 else f"u{i+1}" for i in range(n)])
        if is_3d:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x;  du_y→∂u/∂y;  du_z→∂u/∂z;  du_t→∂u/∂t;\n"
                         f"du_xx→∂²u/∂x²;  du_yy→∂²u/∂y²;  du_zz→∂²u/∂z²;  du_tt→∂²u/∂t²;\n"
                         f"du_xy→∂²u/∂x∂y;  du_xz→∂²u/∂x∂z;  du_yz→∂²u/∂y∂z;\n"
                         f"du_xt→∂²u/∂x∂t;  du_yt→∂²u/∂y∂t;  du_zt→∂²u/∂z∂t;\n"
                         f"du_xxxx→∂⁴u/∂x⁴;  du_yyyy→∂⁴u/∂y⁴;  du_zzzz→∂⁴u/∂z⁴;\n"
                         f"du_xxyy→∂⁴u/∂x²∂y²;  du_xxzz→∂⁴u/∂x²∂z²;  du_yyzz→∂⁴u/∂y²∂z²;\n"
                         f"du_xxtt→∂⁴u/∂x²∂t²;  du_yytt→∂⁴u/∂y²∂t²;  du_zztt→∂⁴u/∂z²∂t².\n"
                         f"── ── ──\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi;  coordinates: x, y, z, t\n"
                         f"(e.g. a source term: - exp(-t)*sin(pi*x))\n\n"
                         f"e.g. 3D Heat:       du_t - 0.4*(du_xx+du_yy+du_zz)\n"
                         f"e.g. 3D Reaction:   du_t - 0.001*(du_xx+du_yy+du_zz) + u**3 - u")
        elif is_2d:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x;  du_y→∂u/∂y;  du_t→∂u/∂t;\n"
                         f"du_xx→∂²u/∂x²;  du_yy→∂²u/∂y²;  du_xy→∂²u/∂x∂y;\n"
                         f"du_tt→∂²u/∂t²;  du_xt→∂²u/∂x∂t;  du_yt→∂²u/∂y∂t;\n"
                         f"du_xxxx→∂⁴u/∂x⁴;  du_yyyy→∂⁴u/∂y⁴;\n"
                         f"du_xxyy→∂⁴u/∂x²∂y²;  du_xxtt→∂⁴u/∂x²∂t².\n"
                         f"── ── ──\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi;  coordinates: x, y, t\n"
                         f"(e.g. a source term: - exp(-t)*sin(pi*x))\n\n"
                         f"e.g. 2D Heat:        du_t - 0.4*(du_xx+du_yy)\n"
                         f"e.g. Allen-Cahn 2D:  du_t - 0.001*(du_xx+du_yy) + u**3 - u\n"
                         f"e.g. sin(pi*u)*du_xx + cos(u)*du_yy")
        else:
            hint_text = (f"Outputs: {names_ex}\n"
                         f"du_x→∂u/∂x;  du_t→∂u/∂t;\n"
                         f"du_xx→∂²u/∂x²;  du_tt→∂²u/∂t²;  du_xt→∂²u/∂x∂t;\n"
                         f"du_xxxx→∂⁴u/∂x⁴;  du_tttt→∂⁴u/∂t⁴;  du_xxtt→∂⁴u/∂x²∂t².\n"
                         f"── ── ──\n"
                         f"Functions: sin, cos, exp, log, sqrt, tanh, pi;  coordinates: x, t\n"
                         f"(e.g. a source term: - exp(-t)*sin(pi*x))\n\n"
                         f"e.g. Diffusion:      du_t - 0.4*du_xx\n"
                         f"e.g. Burgers:        du_t + u*du_x - 0.01*du_xx\n"
                         f"e.g. 4th-order:      du_t - (du_xx - du_xxxx)\n"
                         f"e.g. Nonlinear:      du_t - sin(u)*du_xx")

        # The toggle+hint widgets themselves now live once at the top of
        # this panel (see pde_outer_layout, above, where they're built) --
        # only the TEXT needs refreshing here each time dimension/output
        # count changes what it should say; its current shown/hidden state
        # and the toggle's checked state are both left exactly as the user
        # last set them.
        if self._is_ode():
            hint_text = ("ODE mode: the only variable is t.\n"
                         "Write derivatives as du_t, du_tt (for an output named u), e.g.\n"
                         "  dy1_t - y2      (one equation per output: y1' = y2)\n"
                         "  du_tt - 10*du_t + 9*u - 5*t\n"
                         "Initial values go in the Initial Conditions panel; a derivative\n"
                         "initial condition (u'(0) = 2) is an Operator BC 'du_t - 2' applied\n"
                         "at the initial time. Name outputs anything but x, t and s.\n"
                         "Integro-differential: integral(f) is the integral of f from t₀ to t,\n"
                         "f in s (integration variable), t and the outputs at s, e.g.\n"
                         "  du_t + u - integral(exp(s - t)*u)")
        self.pde_hint.setText(hint_text)

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

        # Keep the panel's one shared IC-reference hint text (see its
        # construction above, outside this rebuilt region) up to date with
        # the current dimension -- this used to be rebuilt fresh under every
        # single output's own IC box; the text never actually varied with
        # the output, only with dimension, so one update here covers all of
        # them.
        if hasattr(self, 'ic_hint'):
            if is_3d:
                self.ic_hint.setText(
                    "The IC expression describes u at t = t_min, as a function of x, y, z.\n"
                    "Use x, y, z as spatial variables (t is fixed at t_min here, so it\n"
                    "should not appear in this expression).\n"
                    "── ── ──\n"
                    "sin(pi*x)*sin(pi*y)*sin(pi*z)  →  3D sine wave\n"
                    "exp(-(x**2+y**2+z**2))         →  3D Gaussian\n"
                    "sin(4*pi*x)*cos(4*pi*y)*sin(4*pi*z)  →  higher freq\n"
                    "0                              →  zero IC\n"
                    "── ── ──\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            elif is_2d:
                self.ic_hint.setText(
                    "The IC expression describes u at t = t_min, as a function of x, y.\n"
                    "Use x, y as spatial variables (t is fixed at t_min here, so it\n"
                    "should not appear in this expression).\n"
                    "── ── ──\n"
                    "sin(pi*x)*cos(pi*y)      →  2D sine wave\n"
                    "exp(-(x**2+y**2))        →  2D Gaussian\n"
                    "sin(4*pi*x)*cos(4*pi*y)  →  higher freq\n"
                    "0                        →  zero IC\n"
                    "── ── ──\n"
                    "No need for np. or x[:,0] — handled automatically."
                )
            else:
                self.ic_hint.setText(
                    "The IC expression describes u at t = t_min, as a function of x.\n"
                    "Use x as the spatial variable (t is fixed at t_min here, so it\n"
                    "should not appear in this expression).\n"
                    "── ── ──\n"
                    "sin(pi*x)       →  sine wave\n"
                    "exp(-x**2)      →  Gaussian\n"
                    "sin(4*pi*x)     →  higher frequency\n"
                    "x*(1-x)         →  parabola\n"
                    "0               →  zero IC\n"
                    "── ── ──\n"
                    "No need for np. or x[:,0] — handled automatically."
                )

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
            self._register_section_divider(sep, 'margin-top: 4px; ')
            self.bc_main_layout.addWidget(sep)

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

            # IC from file option — available for 1D, 2D, and 3D; the
            # expected no-header column format changes with dimension:
            # 1D: x,t,c   2D: x,y,t,c   3D: x,y,z,t,c. Expression stays the
            # default; this is opt-in per output, and only ever applies to
            # output 0's IC (single global IC file per problem).
            _ic_file_fmt = "x,y,z,t,c" if is_3d else ("x,y,t,c" if is_2d else "x,t,c")
            ic_file_cb = QCheckBox(f"📂 Load IC from file ({_ic_file_fmt} format)")
            ic_file_cb.setChecked(False)
            self._register_hint(ic_file_cb, '#ffa94d')
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

    def _on_network_type_changed(self, kind):
        """Fourier scales only for MsFFN / STMsFFN (Round 73)."""
        if hasattr(self, "msffn_sigmas_row"):
            self.msffn_sigmas_row.setVisible(kind in ("MsFFN", "STMsFFN"))
            self.msffn_sigmas_t_row.setVisible(kind == "STMsFFN")

    def _bc_note_with_transform(self, text):
        """With no BC rows and an output-transform formula, say where the
        boundary conditions are (hard constraints) -- Round 72."""
        try:
            hard = (not self.custom_bc_list and self.output_transform_cb.isChecked()
                    and (self.output_transform_mode_combo.currentData() == "formula"))
        except AttributeError:
            hard = False
        if hard:
            return (text + "\nNo rows: the boundary conditions are imposed exactly by the output "
                    "transform formula (Network panel → Output transform).")
        return text

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
            self.custom_bc_note.setText(self._bc_note_with_transform(
                "No template selected — build your own boundary conditions."))
        else:
            template_name = self.quick_examples_combo.currentText() if hasattr(self, 'quick_examples_combo') else ''
            self.custom_bc_note.setText(self._bc_note_with_transform(
                f"Pre-filled from the \"{_ex_catalog.display_name(template_name)}\" example."))
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
    # PointSetBC's component can match several output columns at once
    # (DeepXDE's own list-of-ints support, PyTorch backend) -- not offered
    # for Point Set Operator BC, whose DeepXDE constructor has no
    # `component` parameter at all (its func already returns exactly what
    # gets compared against `values`).
    _BC_NEEDS_MULTI_COMPONENT = {"pointset"}
    # PointSetBC/PointSetOperatorBC minibatching (DeepXDE's own batch_size/
    # shuffle constructor kwargs, requiring dde.callbacks.PDEPointResampler
    # (bc_points=True) in the training callbacks to actually resample each
    # batch -- wired in automatically by codegen whenever this is used).
    _BC_NEEDS_BATCH_ROW = {"pointset", "pointset_operator"}

    def _interface2d_eligible(self):
        """Interface 2D BC's real DeepXDE preconditions that this GUI can
        check up front, before Solve/Export even runs. dde.icbc.Interface2DBC
        requires the network output to have exactly 2 elements -- its
        "tangent" direction indexes outputs[:, 0:1] / outputs[:, 1:2] by
        position, and its "normal" direction treats the whole output as a
        2-column normal vector -- and it only exists for 2D geometries at
        all (its matching-length-edge requirement is checked against
        Rectangle/Polygon, both 2D-only DeepXDE classes). The geometry
        check itself (Rectangle/Polygon specifically, within 2D) still
        happens as an error+hint at codegen time in
        _bc_check_interface2d_preconditions, since there's no sensible
        geometry to auto-pick the way Point Distribution was auto-corrected
        in Round 33b -- but Dimension and Output Size can be checked here,
        so this BC type simply isn't offered until both are right."""
        return self.radio_2d.isChecked() and self.num_outputs_spin.value() == 2

    def _refresh_bc_interface2d_availability(self):
        """Add/remove "Interface 2D BC" from every existing row's Type
        dropdown as Output Size changes (Dimension changes don't need this:
        _on_dim_changed() already wipes the whole Boundary Conditions list,
        so any row built afterward already goes through the eligibility
        filter in _add_custom_bc_entry below). A row currently set to
        Interface 2D BC that becomes ineligible is reset to Dirichlet --
        the same "demoted" convention _apply_custom_bc_json already uses
        for a type this version of PINNStudio doesn't recognize at all --
        and the user is warned once for all affected rows."""
        if not hasattr(self, 'custom_bc_list'):
            return
        eligible = self._interface2d_eligible()
        _label = next(lbl for lbl, key in self.CUSTOM_BC_TYPES if key == "interface2d")
        _demoted_rows = []
        for _i, e in enumerate(self.custom_bc_list):
            combo = e['type']
            idx = combo.findData("interface2d")
            if eligible:
                if idx == -1:
                    combo.addItem(_label, "interface2d")
                continue
            if idx == -1:
                continue
            if combo.currentData() == "interface2d":
                _demoted_rows.append(_i + 1)
                # Not blocked: this fires _sync_type_and_refresh_weights,
                # which is what actually updates this row's visible
                # widgets and the Loss Weights panel label for it.
                combo.setCurrentIndex(0)
                idx = combo.findData("interface2d")
            if idx != -1:
                combo.removeItem(idx)
        if _demoted_rows:
            _rows_txt = ", ".join(f"BC {n}" for n in _demoted_rows)
            QMessageBox.warning(
                self, "Interface 2D BC no longer available",
                f"{_rows_txt} (was Interface 2D BC) reset to Dirichlet: "
                "Interface 2D BC requires Dimension = 2D and Output Size = 2 "
                "(DeepXDE's own requirements for this boundary condition), "
                "and that's no longer the case -- please review and fix "
                "before solving."
            )

    _BC_CUSTOM_WHERE = "Custom expression…"

    def _bc_where_choices(self):
        """(label, expression) pairs for the "Where:" picker, for the
        current dimension (Round 68). The side names are defined in the
        generated scripts against this problem's own domain limits."""
        is_2d = getattr(self, 'radio_2d', None) is not None and self.radio_2d.isChecked()
        is_3d = getattr(self, 'radio_3d', None) is not None and self.radio_3d.isChecked()
        if self._is_ode():
            return [("Both ends (t₀ and t₁)", "True"), ("Start (t = t₀)", "start"), ("End (t = t₁)", "end")]
        ch = [("Whole boundary", "True"),
              ("Left (x = x_min)", "left"), ("Right (x = x_max)", "right"), ("Left and right", "left or right")]
        if is_2d or is_3d:
            ch += [("Bottom (y = y_min)", "bottom"), ("Top (y = y_max)", "top"), ("Bottom and top", "bottom or top")]
        if is_3d:
            ch += [("Front (z = z_min)", "front"), ("Back (z = z_max)", "back"), ("Front and back", "front or back")]
        return ch

    def _fill_bc_where_picker(self, pick, edit):
        pick.blockSignals(True)
        pick.clear()
        for label, expr in self._bc_where_choices():
            pick.addItem(label, expr)
        pick.addItem(self._BC_CUSTOM_WHERE, None)
        pick.blockSignals(False)
        self._sync_bc_where_picker(pick, edit)

    def _sync_bc_where_picker(self, pick, edit):
        text = " ".join(edit.text().split())
        if text == "":
            text = "True"  # an empty "Where" means the whole boundary
        idx = pick.findData(text)
        pick.blockSignals(True)
        pick.setCurrentIndex(idx if idx >= 0 else pick.count() - 1)
        pick.blockSignals(False)

    def _on_bc_where_picked(self, pick, edit):
        expr = pick.currentData()
        if expr is None:
            edit.setFocus()
            return
        edit.setText(expr)

    def _refresh_bc_where_pickers(self):
        """Dimension changed: offer that dimension's sides."""
        for e in getattr(self, 'custom_bc_list', []):
            if e.get('location_picker') is not None:
                self._fill_bc_where_picker(e['location_picker'], e['location'])

    def _add_custom_bc_entry(self, bc_type="dirichlet", component=0, location="",
                              value="0", axis="x", deriv_order=0, points_file="",
                              location2="", direction="normal", locked=False,
                              components="", batch_size=0, shuffle=True, apply_at="boundary",
                              points_mode="file", points_list="", points_count=10, points_where="boundary",
                              values_mode="file", values_formula=""):
        entry_widget = QWidget()
        entry_layout = QVBoxLayout(entry_widget)
        entry_layout.setSpacing(3)
        entry_layout.setContentsMargins(0, 0, 0, 0)

        entry_num = len(self.custom_bc_list) + 1
        header_row = QHBoxLayout()
        header_txt = f"── BC {entry_num} ──" + (" \U0001F512" if locked else "")
        header_lbl = QLabel(header_txt)
        _header_color = "#8a8a8a" if locked else "#a0c4ff"
        self._register_hint(header_lbl, _header_color)
        header_row.addWidget(header_lbl)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        remove_btn.setVisible(not locked)
        header_row.addStretch(); header_row.addWidget(remove_btn)
        entry_layout.addLayout(header_row)

        # Type + Output, on one row -- was two separate rows, each with its
        # label on the left and its control pushed out to the far right by
        # an addStretch() before it, leaving a wide empty gap between the
        # label and the dropdown/spinbox. Putting both on one line, with
        # the controls right next to their own labels, is more compact and
        # reads better; a single addStretch() at the end still keeps both
        # left-aligned instead of spreading across the whole row.
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        type_combo = QComboBox()
        for label_txt, key in self.CUSTOM_BC_TYPES:
            # Interface 2D BC only ever makes sense for Dimension=2D,
            # Output Size=2 (see _interface2d_eligible) -- left out of the
            # dropdown entirely rather than offered and then erroring at
            # Solve/Export time. _refresh_bc_interface2d_availability keeps
            # already-built rows in sync as Output Size changes later.
            if key == "interface2d" and not self._interface2d_eligible():
                continue
            type_combo.addItem(label_txt, key)
        self._set_combo_data(type_combo, bc_type)
        type_combo.setFixedHeight(26); type_combo.setFixedWidth(210)
        type_row.addWidget(type_combo)
        type_row.addSpacing(14)

        # Output (component)
        comp_widget = QWidget()
        comp_row = QHBoxLayout(comp_widget)
        comp_row.setContentsMargins(0, 0, 0, 0)
        comp_row.addWidget(QLabel("Output:"))
        comp_spin = OutputSpinBox(); comp_spin.setRange(0, 7); comp_spin.setValue(component)
        comp_spin.setFixedHeight(26); comp_spin.setFixedWidth(60)
        comp_spin.setToolTip("The output this BC applies to (1 = first output, 2 = second, ...), "
                             "numbered as in the PDE Definition panel")
        comp_row.addWidget(comp_spin)
        type_row.addWidget(comp_widget)
        type_row.addStretch()
        entry_layout.addLayout(type_row)

        # Location (hidden for PointSet/PointSetOperator, which take points
        # from a file instead). The detailed multi-line example text used to
        # live here too, behind a per-row toggle -- now shown once for the
        # whole panel instead (see self.bc_loc_hint_toggle above
        # custom_bc_list_widget), so this row stays just the input itself
        # plus its short inline placeholder example.
        loc_widget = QWidget()
        loc_layout = QVBoxLayout(loc_widget)
        loc_layout.setContentsMargins(0, 0, 0, 0); loc_layout.setSpacing(2)
        loc_row = QHBoxLayout()
        loc_row.addWidget(QLabel("Where:"))
        # Quick picker for the common choices (Round 68): it writes the
        # expression into the field below, and follows whatever is typed
        # there ("Custom expression" when it's not one of its choices).
        loc_pick = QComboBox()
        loc_pick.setFixedHeight(26)
        loc_pick.setToolTip("Which part of the boundary this condition applies to.\n"
                            "Sides are the domain's own limits: left = x_min, right = x_max,\n"
                            "bottom/top = y_min/y_max, front/back = z_min/z_max.")
        loc_row.addWidget(loc_pick)
        loc_edit = QLineEdit(location)
        # a long Where: set by the sketch or a template shows from its start
        loc_edit.setCursorPosition(0)
        loc_edit.textChanged.connect(lambda _t, _le=loc_edit: None if _le.hasFocus() else _le.setCursorPosition(0))
        loc_edit.setPlaceholderText("e.g. left, right, x <= x_min, np.isclose(x**2 + y**2, 0.25)")
        loc_edit.setToolTip(loc_pick.toolTip() + "\nOr any True/False expression in x, y, z (and t), "
                            "e.g. left or right, x <= x_min, np.isclose(x**2 + y**2, 0.25).")
        loc_edit.setFixedHeight(26)
        loc_row.addWidget(loc_edit, 1)
        loc_layout.addLayout(loc_row)
        self._fill_bc_where_picker(loc_pick, loc_edit)
        loc_pick.activated.connect(lambda _i: self._on_bc_where_picked(loc_pick, loc_edit))
        loc_edit.textChanged.connect(lambda _t: self._sync_bc_where_picker(loc_pick, loc_edit))
        entry_layout.addWidget(loc_widget)

        # Value/function (hidden for Periodic, which has no func; hidden for
        # PointSet, whose values come from its file)
        val_widget = QWidget()
        val_layout = QVBoxLayout(val_widget)
        val_layout.setContentsMargins(0, 0, 0, 0); val_layout.setSpacing(2)
        val_row = QHBoxLayout()
        val_label = QLabel("Value:")
        val_row.addWidget(val_label)
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
        # PeriodicBC's geom.periodic_point(x, component_x) only ever
        # indexes the coordinate columns the CURRENT dimension actually
        # has (see _bc_check_periodic_preconditions in codegen.py) -- "y"
        # in 1D or "z" outside 3D would reference a column that doesn't
        # exist, so only offer the axes this dimension really has. No
        # dimension-change refresh hook is needed: _on_dim_changed() already
        # wipes the whole Boundary Conditions list, so a row built after a
        # dimension switch is always constructed fresh, right here, under
        # the new dimension.
        _axis_choices = ["x"]
        if self.radio_2d.isChecked() or self.radio_3d.isChecked():
            _axis_choices.append("y")
        if self.radio_3d.isChecked():
            _axis_choices.append("z")
        axis_combo = QComboBox(); axis_combo.addItems(_axis_choices)
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
        # Round 71: the points can also be typed in or generated, and the
        # target values given as a formula -- no file needed for a few points.
        pointset_widget = QWidget()
        _ps_outer = QVBoxLayout(pointset_widget)
        _ps_outer.setContentsMargins(0, 0, 0, 0); _ps_outer.setSpacing(2)
        pointset_layout = QHBoxLayout()
        pointset_layout.setContentsMargins(0, 0, 0, 0)
        pointset_layout.addWidget(QLabel("Points:"))
        points_mode_combo = QComboBox(); points_mode_combo.setFixedHeight(26)
        for _lbl, _key in (("From file", "file"), ("Typed list", "list"), ("Generated", "generate")):
            points_mode_combo.addItem(_lbl, _key)
        points_mode_combo.setCurrentIndex(max(0, points_mode_combo.findData(points_mode)))
        pointset_layout.addWidget(points_mode_combo)
        pointset_path = QLineEdit(points_file)
        pointset_path.setPlaceholderText("Browse for points file (x[,y[,z]][,t],value)...")
        pointset_path.setFixedHeight(26)
        pointset_layout.addWidget(pointset_path, 1)
        pointset_browse = QPushButton("Browse")
        pointset_browse.setFixedHeight(26); pointset_browse.setFixedWidth(65)
        pointset_browse.clicked.connect(lambda: pointset_path.setText(
            QFileDialog.getOpenFileName(None, "Select points file", "", "Data files (*.txt *.csv *.dat)")[0]
            or pointset_path.text()))
        pointset_layout.addWidget(pointset_browse)
        points_list_edit = QLineEdit(points_list)
        points_list_edit.setPlaceholderText("e.g. 1   or   0.2, 0.5   (1D);   0,0; 1,0.5   (2D: ';' between points)")
        points_list_edit.setToolTip("Coordinates of each point, in x, y, z order, with t last when time-dependent.\n"
                                    "Separate points with ';' (in 1D steady problems ',' also works).")
        points_list_edit.setFixedHeight(26)
        pointset_layout.addWidget(points_list_edit, 1)
        points_count_spin = QSpinBox(); points_count_spin.setRange(1, 1000000); points_count_spin.setValue(int(points_count or 10))
        points_count_spin.setFixedHeight(26); points_count_spin.setToolTip("Number of random points")
        pointset_layout.addWidget(points_count_spin)
        points_where_combo = QComboBox(); points_where_combo.setFixedHeight(26)
        points_where_combo.addItem("on the boundary (Where)", "boundary"); points_where_combo.addItem("in the domain", "domain")
        points_where_combo.setCurrentIndex(max(0, points_where_combo.findData(points_where)))
        pointset_layout.addWidget(points_where_combo)
        _ps_outer.addLayout(pointset_layout)
        values_row = QHBoxLayout(); values_row.setContentsMargins(0, 0, 0, 0)
        values_row.addWidget(QLabel("Target values:"))
        values_mode_combo = QComboBox(); values_mode_combo.setFixedHeight(26)
        values_mode_combo.addItem("From the file's value column(s)", "file"); values_mode_combo.addItem("Formula", "formula")
        values_mode_combo.setCurrentIndex(max(0, values_mode_combo.findData(values_mode)))
        values_row.addWidget(values_mode_combo)
        values_formula_edit = QLineEdit(values_formula)
        values_formula_edit.setPlaceholderText("e.g. 4  or  2*(x + 1)  ('|' between components)")
        values_formula_edit.setFixedHeight(26)
        values_row.addWidget(values_formula_edit, 1)
        _ps_outer.addLayout(values_row)
        entry_layout.addWidget(pointset_widget)

        def _sync_points_mode(_i=None):
            m = points_mode_combo.currentData()
            pointset_path.setVisible(m == "file"); pointset_browse.setVisible(m == "file")
            points_list_edit.setVisible(m == "list")
            points_count_spin.setVisible(m == "generate"); points_where_combo.setVisible(m == "generate")
            if m != "file" and values_mode_combo.currentData() == "file":
                values_mode_combo.setCurrentIndex(values_mode_combo.findData("formula"))
            values_mode_combo.model().item(0).setEnabled(m == "file")
            values_formula_edit.setVisible(values_mode_combo.currentData() == "formula")
            # Generated on the boundary: the Where field picks which part.
            loc_widget.setVisible(type_combo.currentData() in self._BC_NEEDS_LOCATION
                                  or (type_combo.currentData() in self._BC_NEEDS_POINTS_FILE
                                      and m == "generate" and points_where_combo.currentData() == "boundary"))
        points_mode_combo.currentIndexChanged.connect(_sync_points_mode)
        points_where_combo.currentIndexChanged.connect(_sync_points_mode)
        values_mode_combo.currentIndexChanged.connect(_sync_points_mode)

        # PointSetBC-only: match several output components against the
        # same points file at once (DeepXDE's own component=[i, j, ...]
        # support) instead of just the single "Output #:" above -- the
        # points file then needs one extra value column per entry here,
        # in the same order. Left blank (the default), "Output #:" above
        # is used exactly as before -- purely additive, no behavior change
        # for any existing single-component row.
        multi_comp_widget = QWidget()
        multi_comp_layout = QHBoxLayout(multi_comp_widget)
        multi_comp_layout.setContentsMargins(0, 0, 0, 0)
        multi_comp_layout.addWidget(QLabel("Components (comma-separated):"))
        multi_comp_edit = QLineEdit(_components_display(components))
        multi_comp_edit.setPlaceholderText("e.g. 1,2 -- overrides Output above, needs matching extra file columns")
        multi_comp_edit.setFixedHeight(26)
        multi_comp_layout.addWidget(multi_comp_edit)
        entry_layout.addWidget(multi_comp_widget)

        # PointSet / PointSetOperator minibatching: DeepXDE resamples a new
        # random batch of this size from the points file each training
        # step (via the PDEPointResampler callback, added automatically by
        # codegen whenever any BC row here uses batching) instead of using
        # every point every step. 0 (the default) means no batching at
        # all -- unchanged from before this existed.
        batch_widget = QWidget()
        batch_layout = QHBoxLayout(batch_widget)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.addWidget(QLabel("Batch size (0 = off):"))
        batch_spin = QSpinBox(); batch_spin.setRange(0, 1_000_000); batch_spin.setValue(int(batch_size or 0))
        batch_spin.setFixedHeight(26); batch_spin.setFixedWidth(90)
        batch_layout.addWidget(batch_spin)
        shuffle_check = QCheckBox("Shuffle")
        shuffle_check.setChecked(bool(shuffle))
        batch_layout.addWidget(shuffle_check)
        batch_layout.addStretch()
        entry_layout.addWidget(batch_widget)

        # Operator BC only (Round 64): where it applies -- the spatial
        # boundary (DeepXDE's OperatorBC default) or the initial time t = t0,
        # e.g. a wave equation's second initial condition du_t = 0.
        apply_widget = QWidget()
        apply_row = QHBoxLayout(apply_widget)
        apply_row.setContentsMargins(0, 0, 0, 0)
        apply_row.addWidget(QLabel("Applies on:"))
        apply_combo = QComboBox()
        apply_combo.addItem("the boundary", "boundary")
        apply_combo.addItem("the initial time (t = t₀)", "initial")
        apply_combo.setFixedHeight(26)
        _ai = apply_combo.findData(apply_at or "boundary")
        apply_combo.setCurrentIndex(_ai if _ai >= 0 else 0)
        apply_row.addWidget(apply_combo); apply_row.addStretch()
        entry_layout.addWidget(apply_widget)

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

        # Interface2DBC-only: DeepXDE's own real requirement that
        # train_distribution="uniform" for this BC type to pair points
        # correctly across its two edges (see codegen.py's
        # _bc_entries_use_interface2d docstring) -- surfaced here, in the
        # panel itself, rather than only at Solve/Export time, so the user
        # sees it the moment they pick this type. _sync_type() below also
        # switches the Point Distribution combo to "uniform" automatically
        # right then, so this note explains what just happened rather than
        # asking the user to go make the change themselves.
        interface2d_note = QLabel(
            "Interface 2D BC requires Point Distribution = Uniform (DeepXDE's\n"
            "own requirement, so matching points can be paired correctly\n"
            "across the two edges). PINNStudio switches Point Distribution to\n"
            "Uniform automatically while this BC type is selected."
        )
        self._register_hint(interface2d_note, '#ffa94d')
        interface2d_note.setWordWrap(True)
        interface2d_note.setVisible(False)
        entry_layout.addWidget(interface2d_note)

        # Operator BC / Point Set Operator BC: DeepXDE's own real signature
        # for this func (see dde.icbc.OperatorBC/PointSetOperatorBC's own
        # docstrings) is func(inputs, outputs, X) -> tensor, where inputs/
        # outputs are the network's raw input/output tensors and X is the
        # plain NumPy array of inputs -- all three still usable directly,
        # unchanged. New: the same derivative shorthand already available
        # in the PDE box, Training Monitors, and Custom plot expressions
        # (the output name itself, e.g. u, plus du_x/du_xx/du_xy/...) can
        # be used here too, instead of writing out raw
        # dde.grad.jacobian/hessian calls by hand.
        _op_note_txt = (
            "Operator BC: the function must equal 0 on the part of the boundary\n"
            "picked by Where (or at the initial time) -- write the target into it,\n"
            "e.g. du_x - 4. Use it for a condition along a whole side.\n"
            "Derivative shorthand works (u, du_x, du_xx, ... as in the PDE box);\n"
            "DeepXDE's raw inputs/outputs tensors and X are available too.")
        _pso_note_txt = (
            "Point Set Operator BC: the function (e.g. du_x) must equal the target\n"
            "values at the listed points, which can be anywhere -- inside the domain\n"
            "too. Use it for measured derivatives/fluxes at given locations; for a\n"
            "condition along a whole side an Operator BC (du_x - 4) or a Neumann BC\n"
            "does the same. Derivative shorthand works (u, du_x, du_xx, ...).")
        advanced_note = QLabel(_op_note_txt)
        self._register_hint(advanced_note, '#ffa94d')
        advanced_note.setWordWrap(True)
        advanced_note.setVisible(False)
        entry_layout.addWidget(advanced_note)

        val_placeholders = {
            "neumann": "du/dn, outward normal: e.g. 4  (at a left/bottom side du/dn = -du/dx or -du/dy)",
            "robin": "du/dn = this, e.g. u, 2*u - 1  (x, y, z, and u for the solution)",
            "interface2d": "Python expression in x, y, z for the interface func(x)",
            "operator": "e.g. du_x - u  (func(inputs, outputs, X); u/du_x/... shorthand OK)",
            "pointset_operator": "e.g. du_x - u  (func(inputs, outputs, X); u/du_x/... shorthand OK)",
        }
        default_val_placeholder = "e.g. 0, sin(pi*y), x**2  (Robin: may also use u)"
        val_labels = {
            "operator": "Function (func):",
            "pointset_operator": "Function (func):",
        }
        default_val_label = "Value:"

        def _sync_type(_t=None):
            key = type_combo.currentData()
            comp_widget.setVisible(key in self._BC_NEEDS_COMPONENT)
            loc_widget.setVisible(key in self._BC_NEEDS_LOCATION)
            val_widget.setVisible(key in self._BC_NEEDS_VALUE)
            periodic_widget.setVisible(key in self._BC_NEEDS_PERIODIC_ROW)
            pointset_widget.setVisible(key in self._BC_NEEDS_POINTS_FILE)
            multi_comp_widget.setVisible(key in self._BC_NEEDS_MULTI_COMPONENT)
            batch_widget.setVisible(key in self._BC_NEEDS_BATCH_ROW)
            interface_widget.setVisible(key in self._BC_NEEDS_INTERFACE_ROW)
            apply_widget.setVisible(key == "operator")
            advanced_note.setVisible(key in self._BC_NEEDS_ADVANCED_NOTE)
            advanced_note.setText(_pso_note_txt if key == "pointset_operator" else _op_note_txt)
            _sync_points_mode()
            val_edit.setPlaceholderText(val_placeholders.get(key, default_val_placeholder))
            val_label.setText(val_labels.get(key, default_val_label))
            # Interface 2D BC's real DeepXDE "uniform" requirement (see
            # interface2d_note above) -- shown and auto-applied the moment
            # this row's type is Interface 2D, not just at Solve/Export
            # time. Deliberately one-directional: switching this row away
            # from Interface 2D (or removing it) never reverts Point
            # Distribution back on its own -- whatever the user had it set
            # to before stays, same as every other config value in the app.
            _is_interface2d = key == "interface2d"
            interface2d_note.setVisible(_is_interface2d)
            if _is_interface2d and hasattr(self, "pts_dist_combo"):
                if self.pts_dist_combo.currentText() != "uniform":
                    self.pts_dist_combo.setCurrentText("uniform")

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
            for w in (type_combo, comp_spin, loc_edit, val_edit,
                      axis_combo, deriv_spin, pointset_path, pointset_browse,
                      loc2_edit, direction_combo, multi_comp_edit, batch_spin,
                      shuffle_check, apply_combo):
                w.setEnabled(False)

        self.custom_bc_list_layout.addWidget(entry_widget)
        entry_data = {
            'widget': entry_widget,
            'apply_at': apply_combo,
            'type': type_combo,
            'component': comp_spin,
            'location': loc_edit,
            'value': val_edit,
            'axis': axis_combo,
            'deriv_order': deriv_spin,
            'points_file': pointset_path,
            'location2': loc2_edit,
            'location_picker': loc_pick,
            'points_mode': points_mode_combo,
            'points_list': points_list_edit,
            'points_count': points_count_spin,
            'points_where': points_where_combo,
            'values_mode': values_mode_combo,
            'values_formula': values_formula_edit,
            'direction': direction_combo,
            'components': multi_comp_edit,
            'batch_size': batch_spin,
            'shuffle': shuffle_check,
            'locked': locked,
            # Containers, exposed mainly so tests can check per-type
            # visibility directly rather than guessing from the leaf inputs.
            'location_widget': loc_widget,
            'value_widget': val_widget,
            'periodic_widget': periodic_widget,
            'pointset_widget': pointset_widget,
            'interface_widget': interface_widget,
            'component_widget': comp_widget,
            'multi_component_widget': multi_comp_widget,
            'batch_widget': batch_widget,
        }
        self.custom_bc_list.append(entry_data)

        def _remove():
            entry_widget.deleteLater()
            if entry_data in self.custom_bc_list:
                self.custom_bc_list.remove(entry_data)
            self._update_bc_mode_visibility()
            self._sync_bc_nodes()
            self._sketch_bcs_changed()

        remove_btn.clicked.connect(_remove)
        entry_data['header'] = header_lbl
        entry_data['remove_btn'] = remove_btn
        self._sync_bc_nodes()
        # the sketch colours its edges by these rows' Where:
        loc_edit.textChanged.connect(self._sketch_bcs_changed)
        type_combo.currentIndexChanged.connect(self._sketch_bcs_changed)
        comp_spin.valueChanged.connect(self._sketch_bcs_changed)
        type_combo.currentIndexChanged.connect(self._schedule_bc_nodes_sync)
        comp_spin.valueChanged.connect(self._schedule_bc_nodes_sync)
        self._sketch_bcs_changed()
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
                'components': _components_stored(e['components'].text()),
                'batch_size': e['batch_size'].value(),
                'shuffle': e['shuffle'].isChecked(),
                'apply_at': e['apply_at'].currentData() if 'apply_at' in e else 'boundary',
                'points_mode': e['points_mode'].currentData(),
                'points_list': e['points_list'].text(),
                'points_count': e['points_count'].value(),
                'points_where': e['points_where'].currentData(),
                'values_mode': e['values_mode'].currentData(),
                'values_formula': e['values_formula'].text(),
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
        self.custom_bc_list.clear(); self._schedule_bc_nodes_sync()
        if not custom_bc_json:
            return
        try:
            entries = json.loads(custom_bc_json)
        except (ValueError, TypeError):
            return
        _known_bc_types = {key for _, key in self.CUSTOM_BC_TYPES}
        _unknown_bc_rows = []
        # Dimension/Output Size are already restored by the time this is
        # called (_apply_config sets radio_2d/radio_3d and num_outputs_spin
        # before calling this) -- so this reflects the real, current state
        # a saved "interface2d" row is being loaded into.
        _interface2d_ineligible_rows = []
        _interface2d_now_eligible = self._interface2d_eligible()
        for _ei, e in enumerate(entries):
            _bc_type = e.get('type', 'dirichlet')
            if _bc_type not in _known_bc_types:
                # The combo box itself already falls back to its first item
                # (Dirichlet) silently when the saved value isn't one of its
                # entries (QComboBox.findData returns -1, setCurrentIndex is
                # just skipped) -- that's a real, permanent change to this
                # BC's type the moment the project is saved again, so it's
                # tracked here and surfaced to the user instead of passing
                # unnoticed.
                _unknown_bc_rows.append((_ei + 1, _bc_type))
            elif _bc_type == "interface2d" and not _interface2d_now_eligible:
                # Same silent-fallback mechanism as above (the combo simply
                # won't have an "interface2d" item to select -- see
                # _add_custom_bc_entry's own eligibility filter), but the
                # type itself is perfectly recognized; it's just not valid
                # for THIS project's current Dimension/Output Size. Worth
                # its own, more specific message.
                _interface2d_ineligible_rows.append(_ei + 1)
            self._add_custom_bc_entry(
                bc_type=_bc_type,
                component=e.get('component', 0),
                location=e.get('location', ''),
                value=e.get('value', '0'),
                axis=e.get('axis', 'x'),
                deriv_order=e.get('deriv_order', 0),
                points_file=e.get('points_file', ''),
                location2=e.get('location2', ''),
                direction=e.get('direction', 'normal'),
                components=e.get('components', ''),
                batch_size=e.get('batch_size', 0),
                shuffle=e.get('shuffle', True),
                apply_at=e.get('apply_at', 'boundary'),
                points_mode=e.get('points_mode', 'file'),
                points_list=e.get('points_list', ''),
                points_count=e.get('points_count', 10),
                points_where=e.get('points_where', 'boundary'),
                values_mode=e.get('values_mode', 'file'),
                values_formula=e.get('values_formula', ''),
                locked=False,
            )
        if _unknown_bc_rows:
            _rows_txt = ", ".join(f"BC {n} (was {t!r})" for n, t in _unknown_bc_rows)
            QMessageBox.warning(
                self, "Unrecognized boundary condition type",
                "This project has a boundary condition with a type this version "
                f"of PINNStudio doesn't recognize: {_rows_txt}. It has been reset "
                "to Dirichlet in the Boundary Conditions panel -- please review "
                "and fix it before solving, since saving the project again will "
                "make this change permanent."
            )
        if _interface2d_ineligible_rows:
            _rows_txt = ", ".join(f"BC {n}" for n in _interface2d_ineligible_rows)
            QMessageBox.warning(
                self, "Interface 2D BC not available",
                "This project has a boundary condition set to Interface 2D BC, "
                "but the current Dimension/Output Size settings don't support "
                f"it (Interface 2D BC requires Dimension = 2D and Output Size "
                f"= 2): {_rows_txt}. It has been reset to Dirichlet in the "
                "Boundary Conditions panel -- please review and fix it before "
                "solving, since saving the project again will make this "
                "change permanent."
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
        self.custom_bc_list.clear(); self._schedule_bc_nodes_sync()
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
        # Named sides (Round 68): defined against the domain's own limits in
        # the generated scripts, so they stay right if the domain changes.
        loc_map = {"left": "left", "right": "right",
                   "bottom": "bottom" if is_2d else "", "top": "top" if is_2d else ""}
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
        self.custom_bc_list.clear(); self._schedule_bc_nodes_sync()
        # Same convention as heat2d: one Dirichlet face (right/x-max), the
        # rest Neumann=0 (insulated).
        faces = [
            ("left", "neumann", "0"),
            ("right", "dirichlet", "1"),
            ("bottom", "neumann", "0"),
            ("top", "neumann", "0"),
            ("front", "neumann", "0"),
            ("back", "neumann", "0"),
        ]
        for i in range(n_out):
            for loc, btype, val in faces:
                self._add_custom_bc_entry(bc_type=btype, component=i, location=loc,
                                           value=val, locked=False)

    def _populate_schrodinger_bc_entries(self, n_out):
        """Add the first-derivative periodic rows the 1D Schrodinger
        template's h_x(t,-5)=h_x(t,5) condition needs (on both the real
        output u and the imaginary output v), on top of whatever
        _populate_locked_bc_entries_from_legacy already seeded for this
        template (value-only periodicity, derivative_order=0 -- the
        legacy per-side widgets have no concept of derivative order at
        all). Same "template populates the panel directly" pattern as
        _populate_3d_heat_bc_entries above."""
        for i in range(n_out):
            self._add_custom_bc_entry(bc_type="periodic", component=i,
                                       location="left", axis="x",
                                       deriv_order=1, locked=False)

    # Every row of the Domain panel (and of each Custom-geometry shape) is
    # "label | fields": one label width and one field width throughout, so
    # the fields line up in a column whatever the shape.
    DOMAIN_LABEL_WIDTH = 96
    DOMAIN_FIELD_WIDTH = 85

    def _domain_label(self, text):
        lbl = QLabel(text)
        lbl.setFixedWidth(self.DOMAIN_LABEL_WIDTH)
        return lbl

    def _domain_row(self, label, *items, tip=""):
        """A Domain-panel row: a fixed-width label, then `items` left to
        right -- a str becomes a small QLabel (e.g. "to"), a DomainSpinBox
        gets the shared field width, a QLineEdit stretches to the panel
        edge. Returns the row's QWidget."""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lbl = self._domain_label(label)
        if tip:
            lbl.setToolTip(tip)
        lay.addWidget(lbl)
        stretch = True
        for it in items:
            if isinstance(it, str):
                lay.addWidget(QLabel(it))
                continue
            if isinstance(it, QDoubleSpinBox):
                it.setFixedWidth(self.DOMAIN_FIELD_WIDTH)
            if isinstance(it, QLineEdit):
                stretch = False
            if tip:
                it.setToolTip(tip)
            lay.addWidget(it)
        if stretch:
            lay.addStretch()
        return row

    @staticmethod
    def _spin(lo, hi, value, step=0.1):
        sb = DomainSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(value)
        sb.setSingleStep(step)
        return sb

    def _bc_weight_desc(self, entry):
        """"Dirichlet, u" for a Boundary Conditions row's loss-weight label:
        its type without the trailing "BC", and its output by name."""
        btype = entry['type'].currentText()
        if btype.endswith(" BC"):
            btype = btype[:-3]
        comp = entry['component'].value()
        names = getattr(self, 'output_name_inputs', [])
        name = names[comp].text().strip() if comp < len(names) else ""
        return f"{btype}, {name or f'output {comp + 1}'}"

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
        # Steady-state problems have no time axis, so there's no Initial
        # Condition to weight at all -- codegen.py already builds an empty
        # _ic_w list in this case (confirmed at training time by the
        # "IC weights: []" log line), independently of these widgets. This
        # mirrors that at the GUI level: skip the "IC {n} (...)" weight
        # rows below whenever Steady-state is checked, the same way
        # _on_steady_state_changed() already hides the separate Initial
        # Condition panel itself. Previously these rows stayed visible
        # (driven only by ic_active[i], with no steady_state check at
        # all), showing weight fields that were silently never used.
        _is_steady = self.steady_state_check.isChecked() if hasattr(self, 'steady_state_check') else False

        def _w_row(label, key):
            row = QHBoxLayout()
            # No default layout margins: rows sit as close together as the
            # other panels' label + field rows.
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(QLabel(label))
            _default_w = 100.0 if key.startswith("ic_") else 1.0
            w = SciLineEdit(_default_w); w.setFixedWidth(100)
            row.addStretch(); row.addWidget(w)
            self.weight_widgets[key] = w
            ww = QWidget(); ww.setLayout(row)
            self.weights_main_layout.addWidget(ww)

        if _same_weights:
            _sep_all = QLabel("── Shared weights (all phases) ──")
            self._register_section_divider(_sep_all)
            _sw_all = QWidget(); _sl_all = QHBoxLayout(_sw_all)
            _sl_all.setContentsMargins(0,0,0,0); _sl_all.addWidget(_sep_all)
            self.weights_main_layout.addWidget(_sw_all)

            for i in range(self._n_equations()):
                name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
                _w_row(f"PDE {i+1} ({name}):" if self._n_equations() == n else f"PDE {i+1}:", f"pde_{i}")
            # One weight row per row in the Boundary Conditions panel --
            # matches exactly what will train (see codegen.py's
            # custom_bc_json path), whether it's a hand-built row or one
            # pre-filled from a template.
            for _bj, _be in enumerate(getattr(self, 'custom_bc_list', [])):
                _w_row(f"BC {_bj + 1} ({self._bc_weight_desc(_be)}):", f"bc_{_bj}")
            if not _is_steady:
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
                self._register_section_divider(sep)
                _sw = QWidget(); _sl = QHBoxLayout(_sw)
                _sl.setContentsMargins(0,0,0,0); _sl.addWidget(sep)
                self.weights_main_layout.addWidget(_sw)
                for _i in range(self._n_equations()):
                    _name = self.output_name_inputs[_i].text() if _i < len(self.output_name_inputs) else f"u{_i+1}"
                    _w_row(f"PDE {_i+1} ({_name}) P{_phase_num}:" if self._n_equations() == n else f"PDE {_i+1} P{_phase_num}:",
                           f"pde_{_i}_p{_phase_num}")
                for _bj, _be in enumerate(getattr(self, 'custom_bc_list', [])):
                    _w_row(f"BC {_bj + 1} ({self._bc_weight_desc(_be)}) P{_phase_num}:", f"bc_{_bj}_p{_phase_num}")
                if not _is_steady:
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
        for i in range(self._n_equations()):
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

    # ── Per-run results folder naming ──────────────────────────
    # Standard naming so results from different runs are never silently
    # overwritten and each one's save time is obvious at a glance --
    # useful for anyone (e.g. for a paper) who wants every run's
    # model/plots to stick around instead of only the latest surviving.
    # Chosen format: "<template-or-fallback-label>__<timestamp>", e.g.
    # "2D_Heat__2026-10-05_14-32-07" for a Quick Example, or
    # "2D_TimeDependent_Forward__2026-10-05_14-32-07" for a custom
    # (non-template) problem. The "Save to:" base path itself (e.g.
    # /Users/asykhan/PINNStudio_Results) is left exactly as configured --
    # only this one subfolder is added underneath it.
    @staticmethod
    def _sanitize_run_label(text):
        import re
        text = re.sub(r"\s*[–—]\s*", "-", text.strip())   # "Poisson 1D – Dirichlet" -> "Poisson 1D-Dirichlet"
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_\-]", "", text)
        return text

    # Every activation identifier deepxde.nn.activations.get() supports,
    # spelled exactly as DeepXDE's own get() docstring capitalizes them --
    # the single source of truth for both the Activation combo's item list
    # (see its own addItems() call) and _canonical_activation_label()
    # below, which maps an arbitrary-case string (e.g. a pre-existing
    # saved .json problem's "relu", written back when the combo only ever
    # offered lowercase names) back onto its canonical display form.
    _ACTIVATION_CHOICES = ["tanh", "sin", "Sigmoid", "ReLU", "SiLU", "Swish", "ELU", "GELU", "SELU"]

    @staticmethod
    def _canonical_activation_label(value):
        """DeepXDE's own activations.get() lowercases whatever string it's
        handed, so any case reaches DeepXDE identically -- but the combo
        box matches its current item by exact text (QComboBox.setCurrentText
        does a case-sensitive lookup), so a saved problem written before
        the combo's items were renamed to DeepXDE's own capitalization
        (e.g. a stored "relu"/"sigmoid"/"swish") would otherwise silently
        fail to select anything, leaving whatever activation happened to
        already be selected -- changing the restored problem's actual
        trained-with activation out from under the user. Falls back to the
        value unchanged if it isn't one of DeepXDE's known identifiers, so
        a combo that's somehow missing an item still shows the raw text
        rather than silently losing it."""
        value = str(value or "").strip()
        for choice in MainWindow._ACTIVATION_CHOICES:
            if choice.lower() == value.lower():
                return choice
        return value

    def _run_folder_label(self, is_2d, is_3d, is_steady):
        """The non-timestamp half of a run's folder name: the active
        Quick Example's own name when one is selected (read live from
        the combo, not a cached attribute -- those aren't reliably kept
        in sync with "None" yet, see the Quick Examples/dimension
        reorg this is a smaller, independent piece of), otherwise a
        short label built from the problem's own shape so a custom
        (non-template) run still gets something more informative than
        a bare timestamp."""
        label = ""
        if hasattr(self, 'quick_examples_combo'):
            label = self.quick_examples_combo.currentText().strip()
        if not label or label == "None":
            dim = "3D" if is_3d else ("2D" if is_2d else "1D")
            stationarity = "Stationary" if is_steady else "TimeDependent"
            problem_kind = "Inverse" if self.radio_inverse.isChecked() else "Forward"
            label = f"{dim}_{stationarity}_{problem_kind}"
        return self._sanitize_run_label(label) or "Run"

    @staticmethod
    def _timestamped_save_dir(base_dir, label):
        """base_dir (the user's unmodified "Save to:"/restore save-path
        text) plus a "<label>__<timestamp>" subfolder -- or base_dir
        unchanged when it's blank (save-to-disk is off entirely, same as
        before this feature existed). Shared by both the Setup tab's
        _run_results_dir (below) and the Restore tab's _on_restore. The
        timestamp is computed here, once per Solve/Export/Restore click,
        not inside the generated script itself -- Parameter Sweep already
        points each of its own runs at its own folder right after this
        (see sweep_runner._apply_run_output_settings), unconditionally
        overwriting whatever a Solve click computed, so a sweep run never
        ends up double-nested under an extra timestamped folder of its
        own."""
        import datetime
        base_dir = (base_dir or "").strip()
        if not base_dir:
            return base_dir
        label = MainWindow._sanitize_run_label(label) or "Run"
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(base_dir, f"{label}__{stamp}")

    @staticmethod
    def _save_root(save_dir):
        """The Save-to folder a run folder was made in: a config's save_dir is
        the run's own "<label>__<timestamp>" folder inside it, and putting
        that back in Save-to would nest every later run one level deeper."""
        import re
        d = (save_dir or "").strip()
        if re.search(r"__\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", d.rstrip("/\\")):
            return os.path.dirname(d.rstrip("/\\"))
        return d

    def _run_results_dir(self, base_dir, is_2d, is_3d, is_steady):
        """The actual directory a Setup-tab Solve/Export run's results
        should be written to -- see _timestamped_save_dir for the shared
        "<label>__<timestamp>" mechanics and _run_folder_label for how
        the label itself is picked."""
        label = self._run_folder_label(is_2d, is_3d, is_steady)
        return self._timestamped_save_dir(base_dir, label)

    # ── Build config ──────────────────────────────────────────
    def _build_config(self):
        n = self.layers_spin.value()
        w = self.neurons_spin.value()
        n_out = self.num_outputs_spin.value()
        is_2d = self.radio_2d.isChecked()
        is_3d = self.radio_3d.isChecked() if hasattr(self, 'radio_3d') else False
        is_steady = self.steady_state_check.isChecked() if hasattr(self, 'steady_state_check') else False
        # Steady-state problems have no time axis, so the network takes one
        # fewer input column than the time-dependent case (x[,y[,z]] only).
        input_size = (3 if is_3d else (2 if is_2d else 1)) if is_steady else (4 if is_3d else (3 if is_2d else 2))
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
            problem_dim="ODE" if self._is_ode() else ("3D" if is_3d else ("2D" if is_2d else "1D")),
            steady_state=is_steady,
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
            geom_custom_shapes_json=self._build_custom_geom_shapes_json(),
            geom_sketch=((self.radio_2d.isChecked() or self.radio_3d.isChecked())
                         and self.geometry_type_combo.currentText() == "Sketch"),
            bc_boundary_json=_bc_group_json(self.bc_boundary_types, self.bc_boundary_vals, self.bc_boundary_active, n_out),
            bc_edge_json=_bc_edge_json(n_out),
            custom_bc_json=self._build_custom_bc_json(),
            num_outputs=n_out,
            output_names=",".join([self.output_name_inputs[i].text() for i in range(n_out)]),
            pde_expressions="|".join([self.pde_inputs[i].text() for i in range(self._n_equations())]),
            num_equations=(self._n_equations() if self._n_equations() != n_out else 0),

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
            loss_weights_from_initial=(self.loss_w_init_c.value() if self.loss_w_init_cb.isChecked() else 0.0),
            integral_points=self.integral_points_spin.value(),
            fractional_alpha=self.frac_alpha_edit.text().strip(),
            fractional_resolution=self.frac_res_edit.text().strip(),
            fractional_meshtype=self.frac_mesh_combo.currentData(),
            plot_output_idx=self.plot_output_combo.currentIndex(),
            plot_all_outputs=self.plot_all_outputs_cb.isChecked(),
            constants=self._constants_json(),
            plot_custom_expr=(self.plot_custom_expr_input.text().strip()
                               if self.plot_output_combo.currentText() == "Custom..." else ""),
            plot_custom_label=(self.plot_custom_label_input.text().strip()
                                if self.plot_output_combo.currentText() == "Custom..." else ""),

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
            network_type=self.network_type_combo.currentText() if hasattr(self, 'network_type_combo') else "FNN",
            msffn_sigmas=self.msffn_sigmas_edit.text().strip() or "1, 10",
            msffn_sigmas_t=self.msffn_sigmas_t_edit.text().strip() or "1, 10",
            input_transform_enabled=self.input_transform_cb.isChecked(),
            input_transform_scale=[r["scale"].value() for r in self.input_transform_rows],
            input_transform_shift=[r["shift"].value() for r in self.input_transform_rows],
            output_transform_enabled=self.output_transform_cb.isChecked(),
            output_transform_scale=[r["scale"].value() for r in self.output_transform_rows],
            output_transform_shift=[r["shift"].value() for r in self.output_transform_rows],
            anchors_mode=self.anchors_mode_combo.currentData() or "none",
            anchors_file=self.anchors_file_edit.text().strip(),
            anchors_count=self.anchors_count_spin.value(),
            anchors_box=self.anchors_box_edit.text().strip(),
            input_transform_mode=self.input_transform_mode_combo.currentData() or "scale",
            input_feature_formula=self.input_feature_formula_edit.text().strip(),
            output_transform_mode=self.output_transform_mode_combo.currentData() or "scale",
            output_transform_formula="|".join(e.text().strip() for e in self.output_formula_edits),
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
            num_timesteps_line=self.timesteps_spin_line.value(),
            num_timesteps_anim=self.timesteps_spin_anim.value(),
            line_slice_y_auto=self.line_slice_y_auto_cb.isChecked(),
            line_slice_y=self.line_slice_y_spin.value(),
            line_slice_z_auto=self.line_slice_z_auto_cb.isChecked(),
            line_slice_z=self.line_slice_z_spin.value(),
            save_dir=self._run_results_dir(self.save_dir_input.text(), is_2d, is_3d, is_steady),
            adapt_method=self.adapt_combo.currentData(),
            rar_cycles=self.rar_cycles.value(),
            rar_candidates=self.rar_candidates.value(),
            rar_add_points=self.rar_add_points.value(),
            rar_adam_iters=self.rar_adam_iters.value(),
            rar_lbfgs_iters=self.rar_lbfgs_iters.value(),
            rar_tolerance=self.rar_tolerance.value(),
            # Index 0 ("All outputs (combined)") -> -1 (original, unchanged
            # behavior); index k (k>=1) -> equation/output (k-1).
            rar_output_selector=(self.rar_output_combo.currentIndex() - 1),
            time_adaptive=self.adapt_combo.currentData() == "Time Adaptive",
            ta_num_steps=sum(r['steps'].value() for r in self.ta_group_rows) if self.ta_group_rows else self.ta_steps.value(),
            ta_grid_size=self._ta_grid_size(),
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
            export_enabled=self.export_enabled_cb.isChecked(),
            export_grid_size=self._export_grid_size(),
            export_t_steps=self.export_tsteps_spin.value(),
            template_type=getattr(self, '_current_template_type', ''),
            optimizer_scheduler=self.sched_cb.isChecked() if hasattr(self, 'sched_cb') else False,
            scheduler_same_weights=self.sched_same_weights_cb.isChecked() if hasattr(self, 'sched_same_weights_cb') else True,
            scheduler_phases=self._build_scheduler_phases_json(),
            lbfgs_use_default=self.lbfgs_use_default_cb.isChecked(),
            **self._lbfgs_config_values(),
            lbfgs_float_type=self.lbfgs_float_combo.currentText() if hasattr(self, 'lbfgs_float_combo') else 'float64',
            float_type=getattr(self, '_float_type', 'float64'),
            gpu_device_index=getattr(self, '_gpu_device_index', 0),
            gpu_memory_fraction=getattr(self, '_gpu_memory_fraction', 0.95),
            use_random_seed=getattr(self, '_use_random_seed', True),
            random_seed=getattr(self, '_random_seed', 2026),
            ic_pretrain=self.ic_pretrain_cb.isChecked(),
            ic_pretrain_optimizer=self.ic_pretrain_opt.currentData(),
            ic_pretrain_iterations=self.ic_pretrain_iters.value(),
            ic_pretrain_num_test=self.ic_pretrain_test.value(),
            ic_pretrain_num_initial=self.ic_pretrain_init.value(),
            ic_pretrain_lr=self.ic_pretrain_lr.value(),
            ic_pretrain_decay_type=self.ic_pretrain_decay_type.currentData(),
            ic_pretrain_decay_p1=self.ic_pretrain_decay_p1.value(),
            ic_pretrain_decay_p2=self.ic_pretrain_decay_p2.value(),
            ic_pretrain_loss=self.ic_pretrain_loss_combo.currentText(),
            ic_pretrain_restore=self.ic_pretrain_mode_restore.isChecked(),
            ic_pretrain_restore_path=self.ic_pretrain_restore_path.text().strip(),
            weight_decay=self._optimizer_settings.get("weight_decay", 0.0),
            cb_early_stopping=self.cb_early_stopping_cb.isChecked(),
            cb_early_stopping_min_delta=self.cb_es_min_delta.value(),
            cb_early_stopping_patience=self.cb_es_patience.value(),
            cb_early_stopping_baseline=self.cb_es_baseline.text().strip(),
            cb_early_stopping_monitor=self.cb_es_monitor.currentData(),
            cb_early_stopping_start_from=self.cb_es_start.value(),
            cb_point_resampler=self.cb_point_resampler_cb.isChecked(),
            cb_point_resampler_period=self.cb_pr_period.value(),
            cb_point_resampler_pde_points=self.cb_pr_pde.isChecked(),
            cb_point_resampler_bc_points=self.cb_pr_bc.isChecked(),
            cb_model_checkpoint=self.cb_model_checkpoint_cb.isChecked(),
            cb_checkpoint_period=self.cb_ck_period.value(),
            cb_checkpoint_save_better_only=self.cb_ck_better.isChecked(),
            cb_checkpoint_monitor=self.cb_ck_monitor.currentData(),
            cb_timer=self.cb_timer_cb.isChecked(),
            cb_timer_minutes=self.cb_tm_minutes.value(),
            training_monitors_enabled=self.training_monitors_cb.isChecked(),
            training_monitors=self._build_training_monitors_json(),
            plot_colormap=self._plot_viz_settings.get('colormap', 'RdBu_r'),
            plot_levels=self._plot_viz_settings.get('levels', 50),
            plot_resolution=self._plot_viz_settings.get('resolution', 100),
            plot_dpi=self._plot_viz_settings.get('dpi', 100),
            plot_colorbar=self._plot_viz_settings.get('colorbar', True),
            plot_auto_range=self._plot_viz_settings.get('auto_range', True),
            plot_vmin=self._plot_viz_settings.get('vmin', -1.0),
            plot_vmax=self._plot_viz_settings.get('vmax', 1.0),
            plot_linewidth=self._plot_viz_settings.get('linewidth', 2.0),
            plot_linewidth_anim=self._plot_viz_settings.get('linewidth_anim', 3.0),
            plot_fps=self._plot_viz_settings.get('fps', 10),
            plot_n_2d_snapshots=self._plot_viz_settings.get('n_2d_snapshots', 2),
            plot_swap_xt=self._plot_viz_settings.get('swap_xt', True),
            plot_figsize_mode=self._plot_viz_settings.get('figsize_mode', 'Default'),
            plot_figsize_w=self._plot_viz_settings.get('figsize_w', 7.0),
            plot_figsize_h=self._plot_viz_settings.get('figsize_h', 5.0),
            plot_title_override=self._plot_viz_settings.get('title', ''),
            plot_xlabel_override=self._plot_viz_settings.get('xlabel', ''),
            plot_ylabel_override=self._plot_viz_settings.get('ylabel', ''),
            plot_title_fontsize=self._plot_viz_settings.get('title_fontsize', 13.0),
            plot_subplot_title_fontsize=self._plot_viz_settings.get('subplot_title_fontsize', 10.0),
            plot_axis_label_fontsize=self._plot_viz_settings.get('axis_label_fontsize', 10.0),
            loss_plot_mode=self._loss_plot_settings.get('mode', 'train_test') if getattr(self, '_loss_plot_settings', None) else 'train_test',
            loss_display_every=self._loss_plot_settings.get('display_every', 1000) if getattr(self, '_loss_plot_settings', None) else 1000,
            loss_plot_linewidth=self._loss_plot_settings.get('linewidth', 2.0) if getattr(self, '_loss_plot_settings', None) else 2.0,
            loss_plot_log_y=self._loss_plot_settings.get('log_y', True) if getattr(self, '_loss_plot_settings', None) else True,
            ea_files=repr(self._ea_settings.get('files', [])) if getattr(self, '_ea_settings', None) else "[]",
            ea_ref_mode=(self._ea_settings or {}).get('ref_mode', 'files') if getattr(self, '_ea_settings', None) else "files",
            ea_ref_formula=(self._ea_settings or {}).get('ref_formula', '') if getattr(self, '_ea_settings', None) else "",
            ea_ref_times=(self._ea_settings or {}).get('ref_times', '') if getattr(self, '_ea_settings', None) else "",
            ea_ref_grid=int((self._ea_settings or {}).get('ref_grid', 0) or 0) if getattr(self, '_ea_settings', None) else 0,
            ea_do_line=self._ea_settings.get('do_line', True) if getattr(self, '_ea_settings', None) else True,
            ea_do_surface=self._ea_settings.get('do_surface', True) if getattr(self, '_ea_settings', None) else True,
            # Round 31: forward the "Error Metrics to Compute" checkboxes
            # into config too -- previously ea['do_l2']/['do_mse']/['do_max']
            # were stored in self._ea_settings but never read past this
            # point, so the live Solve-time Error Analysis always computed
            # and showed every metric regardless of what was checked.
            ea_metric_l2=self._ea_settings.get('do_l2', True) if getattr(self, '_ea_settings', None) else True,
            ea_metric_mse=self._ea_settings.get('do_mse', False) if getattr(self, '_ea_settings', None) else False,
            ea_metric_max=self._ea_settings.get('do_max', False) if getattr(self, '_ea_settings', None) else False,
            sweep_enabled=self.sweep_enable_cb.isChecked() if hasattr(self, 'sweep_enable_cb') else False,
            sweep_mode=self.sweep_mode_combo.currentData() if hasattr(self, 'sweep_mode_combo') else "oat",
            sweep_include_baseline=self._sweep_baseline_setting() if hasattr(self, 'sweep_baseline_cb') else "auto",
            sweep_parameters=self._build_sweep_parameters_json() if hasattr(self, '_build_sweep_parameters_json') else "[]",
            sweep_bayes_calls=self.sweep_bayes_calls_spin.value() if hasattr(self, 'sweep_bayes_calls_spin') else 20,
            sweep_bayes_initial=self.sweep_bayes_init_spin.value() if hasattr(self, 'sweep_bayes_init_spin') else 5,
            sweep_bayes_objective=self.sweep_bayes_obj_combo.currentData() if hasattr(self, 'sweep_bayes_obj_combo') else "loss",
            # A sweep no longer has its own separate save-location/export
            # fields -- it reuses the Setup tab's own "Save to:" folder
            # (each run still gets its own subfolder under a timestamped
            # sweep_YYYYmmdd_HHMMSS/ there, so nothing collides with a
            # normal single-run Solve's own output) and its own plot/
            # export settings, unconditionally -- "custom" per-sweep
            # overrides (sweep_plot_type/_plot_output_idx/_plot_custom_
            # expr/_plot_custom_label/_export_t_steps below) no longer
            # have a GUI to set them from, so sweep_runner.py's own
            # _apply_run_output_settings() always takes its "same_as_
            # setup" branch now and leaves them at these unused defaults.
            sweep_save_dir=self.save_dir_input.text().strip() if hasattr(self, 'save_dir_input') else "",
            sweep_export_mode="same_as_setup",
            sweep_plot_type="Surface",
            sweep_plot_output_idx=0,
            sweep_plot_custom_expr="",
            sweep_plot_custom_label="",
            sweep_export_t_steps=11,
        )

    # ── Save / Open a problem definition ─────────────────────
    # A saved problem is just a PINNConfig -- the same object _build_config()
    # already assembles for Solve -- serialized to JSON. Opening one reverses
    # that: parse the JSON back into a PINNConfig, then _apply_config() writes
    # every field back onto its widget. Together these make _build_config()
    # and _apply_config() exact mirrors of each other; the split is deliberate
    # so a bug in one direction can't silently paper over a bug in the other.
    def _save_problem(self, path=None, save_as=False):
        """Save to `path`; else to the file this problem came from (Ctrl+S),
        asking for a name when there is none or `save_as`. True when saved."""
        import json
        import dataclasses
        from datetime import datetime
        if not path and not save_as:
            path = getattr(self, "_current_path", None)
        if not path:
            default_name = getattr(self, "_current_problem_name", "") or "problem"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Problem As", f"{default_name}.pinn.json",
                "PINNStudio Problem (*.pinn.json)"
            )
        if not path:
            return False
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
            self._current_path = path
            self._add_recent(path)
            self._mark_clean(config)
            self.log_box.append(f"💾 Problem saved: {os.path.basename(path)}")
            return True
        except Exception as e:
            self.log_box.append(f"❌ Failed to save problem: {e}")
            return False

    def _open_problem(self, path=None):
        import json
        import dataclasses
        if isinstance(path, str) and path and not os.path.isfile(path):
            self.log_box.append(f"❌ Problem file not found: {path}")
            self._remove_recent(path)
            return
        if not self._maybe_save_changes():
            return
        if not isinstance(path, str) or not path:
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
        # A hand-edited or corrupted save file passes the dataclass
        # construction above fine (any type-compatible value is accepted)
        # but can still contain a negative iteration count, an empty
        # layers list, an inverted domain range, invalid JSON in one of
        # the JSON-encoded fields, etc. -- previously this either silently
        # produced a degenerate run or crashed deep inside DeepXDE/PyTorch
        # with a traceback giving no hint it was a config problem. Catch it
        # here, before it's ever applied to the interface.
        _val_errors = config.validate()
        if _val_errors:
            self.log_box.append("❌ This problem file has invalid settings and was not loaded:")
            for _ve in _val_errors:
                self.log_box.append(f"   • {_ve}")
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
        self._current_path = path
        self._add_recent(path)
        self._mark_clean()
        self.log_box.append(f"✅ Problem loaded: {problem_name}")

    def _export_deepxde_script(self):
        """Export the problem exactly as currently configured in the GUI as
        a standalone, runnable DeepXDE/PyTorch + matplotlib script -- built
        by generate_clean_script() in codegen.py, a short, tutorial-style
        generator that's independent of what Solve itself actually runs
        (generate_script(), the app's own internal generator, untouched).
        Only the features actually configured for this problem are written
        in at all (an unticked one -- RAR, IC Pre-Training, Training
        Callbacks, weight decay, Inverse, Time-Adaptive, Error Analysis,
        input/output transforms -- is left out entirely, not hidden behind
        a runtime "if"), in plain DeepXDE calls instead of the internal
        generator's runtime dispatch over every GUI option. Lets a user see
        precisely what their GUI settings translate to in real, readable
        DeepXDE code, and run/edit it themselves outside the app."""
        default_name = getattr(self, "_current_problem_name", "") or "problem"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as DeepXDE Script", f"{default_name}.py",
            "Python Script (*.py)"
        )
        if not path:
            return
        if not path.lower().endswith(".py"):
            path += ".py"
        config = self._build_config()
        try:
            from pinnstudio.core.codegen import generate_clean_script
            script = generate_clean_script(config)
        except Exception as e:
            self.log_box.append(f"❌ Failed to generate DeepXDE script: {e}")
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(script)
            self.log_box.append(f"🐍 DeepXDE script exported: {os.path.basename(path)}")
        except Exception as e:
            self.log_box.append(f"❌ Failed to write script file: {e}")

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
        keys = [f"pde_{i}" for i in range(self._n_equations())]
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
            keys = [f"pde_{i}_p{pn}" for i in range(self._n_equations())]
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
        self._example_snapshot = None   # set again by _apply_deepxde_example() when an example is loaded
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
        elif config.problem_dim == "ODE":
            self.radio_ode.setChecked(True)
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
        # The measured-data-file rows rebuilt here get replaced wholesale by
        # the inverse_obs_files_json restore pass further below anyway (each
        # row re-added via _add_inverse_data_row, which adds its own
        # "Custom..." item), so this pass doesn't need to add one for them.
        self.plot_output_combo.addItem("Custom...")
        self.restore_output_combo.addItem("Custom...")
        self._fit_combo_width(self.plot_output_combo, min_width=130)

        # Equations (Round 63; 0 = one per output) -- rebuilds the PDE rows
        # and weights, keeping the output names just set.
        n_eq = int(getattr(config, "num_equations", 0) or 0) or n_out
        self.num_equations_spin.setValue(n_eq)
        self._refresh_rar_points_combo()

        # PDE expressions
        pdes = _texts(config.pde_expressions, n_eq, "|", config.pde_expression)
        for i in range(n_eq):
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
        self.geom_polygon_verts_input.setText(config.geom_polygon_vertices or "0,0;1,0;1,0.6;0.5,1;0,0.6")
        self._apply_custom_geom_shapes_json(getattr(config, 'geom_custom_shapes_json', '') or '')

        # 3.6) Geometry type -- rebuilds the BC rows one more time, this
        # time in the shape (box / unified boundary / per-edge) that
        # matches the saved geometry.
        _geom_items = [self.geometry_type_combo.itemText(k) for k in range(self.geometry_type_combo.count())]
        if config.geometry_type in _geom_items:
            self.geometry_type_combo.setCurrentText(
                "Sketch" if (config.geometry_type == "Custom" and getattr(config, "geom_sketch", False)
                             and "Sketch" in _geom_items) else config.geometry_type)
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
        if hasattr(self, 'steady_state_check'):
            self.steady_state_check.setChecked(bool(getattr(config, 'steady_state', False)))
            self._on_steady_state_changed(self.steady_state_check.isChecked())

        # Collocation points
        self.num_domain.setValue(config.num_domain)
        self.num_boundary.setValue(config.num_boundary)
        self.num_initial.setValue(config.num_initial)
        self.num_test.setValue(config.num_test)
        # Backward compat: a config saved before the dropdown's item was
        # renamed from "pseudorandom" to "pseudo" (see its construction)
        # still restores to the right item instead of silently matching
        # nothing and leaving the combo on its previous value.
        _pd = "pseudo" if config.point_distribution == "pseudorandom" else config.point_distribution
        self.pts_dist_combo.setCurrentText(_pd)
        self._set_combo_data(self.anchors_mode_combo, getattr(config, "anchors_mode", "none") or "none")
        self.anchors_file_edit.setText(getattr(config, "anchors_file", "") or "")
        self.anchors_count_spin.setValue(int(getattr(config, "anchors_count", 1000) or 1000))
        self.anchors_box_edit.setText(getattr(config, "anchors_box", "") or "")
        # Blocked like the dimension-switch/Inverse-toggle combos just above:
        # _on_plot_type_changed() now auto-pops the unified settings dialog on
        # every change, and it must not fire here -- the hidden holder widgets
        # it reads (timesteps_spin_line/_anim, line_slice_*) are restored to
        # this config's real values on the lines right below, so a dialog
        # popped mid-restore would show stale defaults rather than them.
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.setCurrentText(config.plot_type)
        self.plot_type_combo.blockSignals(False)
        self._plot_type_prev = config.plot_type
        self.timesteps_spin_line.setValue(getattr(config, 'num_timesteps_line', 5))
        self.timesteps_spin_anim.setValue(getattr(config, 'num_timesteps_anim', 20))
        self.line_slice_y_auto_cb.setChecked(getattr(config, 'line_slice_y_auto', True))
        self.line_slice_y_spin.setValue(getattr(config, 'line_slice_y', 0.0))
        self.line_slice_z_auto_cb.setChecked(getattr(config, 'line_slice_z_auto', True))
        self.line_slice_z_spin.setValue(getattr(config, 'line_slice_z', 0.0))

        # Network
        n_hidden = max(0, len(config.layers) - 2)
        neurons = config.layers[1] if len(config.layers) > 2 else self.neurons_spin.value()
        self.layers_spin.setValue(n_hidden)
        self.neurons_spin.setValue(neurons)
        self.activation_combo.setCurrentText(self._canonical_activation_label(config.activation))
        self.kernel_init_combo.setCurrentText(config.kernel_initializer)
        self.network_type_combo.setCurrentText(getattr(config, 'network_type', 'FNN') or 'FNN')
        self.msffn_sigmas_edit.setText(getattr(config, 'msffn_sigmas', '1, 10') or '1, 10')
        self.msffn_sigmas_t_edit.setText(getattr(config, 'msffn_sigmas_t', '1, 10') or '1, 10')

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
        self._set_combo_data(self.input_transform_mode_combo, getattr(config, "input_transform_mode", "scale") or "scale")
        self.input_feature_formula_edit.setText(getattr(config, "input_feature_formula", "") or "")
        self._set_combo_data(self.output_transform_mode_combo, getattr(config, "output_transform_mode", "scale") or "scale")
        _out_formulas = (getattr(config, "output_transform_formula", "") or "").split("|")
        for i, edit in enumerate(self.output_formula_edits):
            edit.setText(_out_formulas[i] if i < len(_out_formulas) else "")

        # Training / optimizers
        self.iter1_spin.setValue(config.iterations)
        self.opt1_combo.setCurrentText(config.optimizer)
        self.opt2_combo.setCurrentText(config.optimizer2)
        self.iter2_spin.setValue(config.iterations2)
        self.save_dir_input.setText(self._save_root(config.save_dir))
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
                    _v.get('init', 1.0), true_value=_v.get('true'))
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
                    _f.get('path', ''), _f.get('output_idx', 0), _f.get('weight', 100.0),
                    custom_expr=_f.get('custom_expr', ''))
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
        self.rar_tolerance.setValue(float(getattr(config, "rar_tolerance", 0.0) or 0.0))
        _rar_sel_idx = getattr(config, "rar_output_selector", -1) + 1
        if 0 <= _rar_sel_idx < self.rar_output_combo.count():
            self.rar_output_combo.setCurrentIndex(_rar_sel_idx)
        else:
            self.rar_output_combo.setCurrentIndex(0)

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
        self.export_enabled_cb.setChecked(getattr(config, "export_enabled", False))
        self.export_grid_combo.setCurrentText(str(config.export_grid_size))
        self.export_tsteps_spin.setValue(config.export_t_steps)
        if 0 <= config.plot_output_idx < self.plot_output_combo.count():
            self.plot_output_combo.setCurrentIndex(config.plot_output_idx)
        self.plot_all_outputs_cb.setChecked(bool(getattr(config, "plot_all_outputs", True)))
        self._set_constants(getattr(config, "constants", "") or "")
        self.plot_custom_expr_input.setText(getattr(config, 'plot_custom_expr', '') or '')
        self.plot_custom_label_input.setText(getattr(config, 'plot_custom_label', '') or '')
        self._on_plot_output_combo_changed(self.plot_output_combo.currentText())

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
        self._gpu_device_index = getattr(config, 'gpu_device_index', 0)
        self._gpu_memory_fraction = getattr(config, 'gpu_memory_fraction', 0.95)
        self._use_random_seed = getattr(config, 'use_random_seed', True)
        self._random_seed = getattr(config, 'random_seed', 2026)

        # Optimizer settings
        self._optimizer_settings = {
            "weight_decay": config.weight_decay,
        }

        # Training Callbacks -- inline Training-panel widgets, set directly
        # (no intermediate dict; see their construction in _build_ui for
        # the full scoping note).
        self.cb_early_stopping_cb.setChecked(config.cb_early_stopping)
        self.cb_es_min_delta.setValue(config.cb_early_stopping_min_delta)
        self.cb_es_patience.setValue(int(config.cb_early_stopping_patience))
        self.cb_es_baseline.setText(str(config.cb_early_stopping_baseline or ""))
        self._set_combo_data(self.cb_es_monitor, config.cb_early_stopping_monitor)
        self.cb_es_start.setValue(int(config.cb_early_stopping_start_from))
        self.cb_point_resampler_cb.setChecked(config.cb_point_resampler)
        self.cb_pr_period.setValue(int(config.cb_point_resampler_period))
        self.cb_pr_pde.setChecked(config.cb_point_resampler_pde_points)
        self.cb_pr_bc.setChecked(config.cb_point_resampler_bc_points)
        self.cb_model_checkpoint_cb.setChecked(config.cb_model_checkpoint)
        self.cb_ck_period.setValue(int(config.cb_checkpoint_period))
        self.cb_ck_better.setChecked(config.cb_checkpoint_save_better_only)
        self._set_combo_data(self.cb_ck_monitor, config.cb_checkpoint_monitor)
        self.cb_timer_cb.setChecked(config.cb_timer)
        self.cb_tm_minutes.setValue(float(config.cb_timer_minutes))
        self.training_monitors_cb.setChecked(getattr(config, 'training_monitors_enabled', False))
        self._apply_training_monitors_json(getattr(config, 'training_monitors', '') or '[]')
        # Auto-reveal the Training Callbacks group on load if this saved
        # config already has any one of the five enabled -- a previously-
        # configured callback should never come back hidden just because
        # the panel defaults to collapsed for a brand-new config.
        self.train_callbacks_show_cb.setChecked(
            bool(config.cb_early_stopping or config.cb_point_resampler or
                 config.cb_model_checkpoint or config.cb_timer or
                 getattr(config, 'training_monitors_enabled', False)))

        # IC pre-training
        self.ic_pretrain_cb.setChecked(config.ic_pretrain)
        self._set_combo_data(self.ic_pretrain_opt, config.ic_pretrain_optimizer)
        self.ic_pretrain_iters.setValue(config.ic_pretrain_iterations)
        self.ic_pretrain_test.setValue(config.ic_pretrain_num_test)
        self.ic_pretrain_init.setValue(config.ic_pretrain_num_initial)
        self.ic_pretrain_lr.setValue(config.ic_pretrain_lr)
        self._set_combo_data(self.ic_pretrain_decay_type,
                             getattr(config, "ic_pretrain_decay_type", "none") or "none")
        self.ic_pretrain_decay_p1.setValue(getattr(config, "ic_pretrain_decay_p1", 5000) or 0)
        self.ic_pretrain_decay_p2.setValue(getattr(config, "ic_pretrain_decay_p2", 0.9) or 0)
        self.ic_pretrain_loss_combo.setCurrentText(config.ic_pretrain_loss)
        self.ic_pretrain_mode_restore.setChecked(bool(config.ic_pretrain_restore))
        self.ic_pretrain_mode_train.setChecked(not config.ic_pretrain_restore)
        self.ic_pretrain_restore_path.setText(config.ic_pretrain_restore_path)
        self._update_ic_pretrain_visibility()

        # Optimizer scheduler + weights. Phase rows (optimizer/iterations/lr
        # per phase) are only present in the saved file when the scheduler was
        # actually enabled at save time; rebuild them first so weight-widget
        # generation below (which depends on phase count when weights differ
        # per phase) sees the right shape.
        # The scheduler is always on in the GUI: a config that still uses the
        # older single-optimizer fields gets the equivalent Training Phases,
        # so the panel shows -- and Solve uses -- what will actually run.
        if not config.optimizer_scheduler:
            import copy as _copy_ph
            from pinnstudio.core.config import use_training_phases
            config = use_training_phases(_copy_ph.copy(config))
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
        self.integral_points_spin.setValue(int(getattr(config, "integral_points", 20) or 20))
        self.frac_alpha_edit.setText(str(getattr(config, "fractional_alpha", "1.5")))
        self.frac_res_edit.setText(str(getattr(config, "fractional_resolution", "100")))
        self.frac_mesh_combo.setCurrentIndex(max(0, self.frac_mesh_combo.findData(getattr(config, "fractional_meshtype", "dynamic"))))
        self._sync_frac_widget()
        _wi = float(getattr(config, "loss_weights_from_initial", 0.0) or 0.0)
        self.loss_w_init_cb.setChecked(_wi > 0)
        self.loss_w_init_c.setValue(_wi if _wi > 0 else 5.0)
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
            "linewidth_anim": getattr(config, "plot_linewidth_anim", 3.0),
            "n_2d_snapshots": config.plot_n_2d_snapshots,
            "fps": getattr(config, "plot_fps", 10),
            "swap_xt": getattr(config, "plot_swap_xt", True),
            "figsize_mode": getattr(config, "plot_figsize_mode", "Default"),
            "figsize_w": getattr(config, "plot_figsize_w", 7.0),
            "figsize_h": getattr(config, "plot_figsize_h", 5.0),
            "title": getattr(config, "plot_title_override", ""),
            "xlabel": getattr(config, "plot_xlabel_override", ""),
            "ylabel": getattr(config, "plot_ylabel_override", ""),
            "title_fontsize": getattr(config, "plot_title_fontsize", 13.0),
            "subplot_title_fontsize": getattr(config, "plot_subplot_title_fontsize", 10.0),
            "axis_label_fontsize": getattr(config, "plot_axis_label_fontsize", 10.0),
        }

        # Loss Plot Settings (Round 28) -- getattr defaults so a config
        # saved before this feature existed restores cleanly.
        self._loss_plot_settings = {
            "mode": getattr(config, "loss_plot_mode", "train_test"),
            "display_every": getattr(config, "loss_display_every", 1000),
            "linewidth": getattr(config, "loss_plot_linewidth", 2.0),
            "log_y": getattr(config, "loss_plot_log_y", True),
        }

        # Error-analysis settings
        try:
            ea_files = _ast.literal_eval(config.ea_files) if config.ea_files else []
        except (ValueError, SyntaxError):
            ea_files = []
        _ref_formula_mode = (getattr(config, "ea_ref_mode", "files") or "files") == "formula"
        self._ea_settings = {
            "files": ea_files,
            "do_line": config.ea_do_line,
            "do_surface": config.ea_do_surface,
            "do_l2": config.ea_metric_l2,
            "do_mse": config.ea_metric_mse,
            "do_max": config.ea_metric_max,
            "ref_mode": "formula" if _ref_formula_mode else "files",
            "ref_formula": getattr(config, "ea_ref_formula", "") or "",
            "ref_times": getattr(config, "ea_ref_times", "") or "",
            "ref_grid": int(getattr(config, "ea_ref_grid", 0) or 0),
        } if (ea_files or _ref_formula_mode) else None

        self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
        self._apply_sweep_config(config)

    def _on_num_outputs_changed(self, n):
        # Interface 2D BC requires Output Size = 2; keep the Boundary
        # Conditions panel's Type dropdowns (and any row currently set to
        # it) in sync with the new value. Dimension changes don't need the
        # equivalent call here -- _on_dim_changed() already wipes the whole
        # Boundary Conditions list.
        self._refresh_bc_interface2d_availability()
        # Equations follow the output count when it changes (Round 63);
        # signals blocked so the PDE rows are built once, just below.
        if hasattr(self, "num_equations_spin"):
            self.num_equations_spin.blockSignals(True)
            self.num_equations_spin.setValue(n)
            self.num_equations_spin.blockSignals(False)
        self._build_pde_inputs(n)
        self._build_bc_inputs(n)
        self._build_weight_inputs(n)
        self._rebuild_output_transform_rows(n)
        self.plot_output_combo.clear()
        self.restore_output_combo.clear()
        self._refresh_rar_points_combo()
        for _r in getattr(self, 'inv_data_rows', []):
            _r['output_combo'].clear()
        for i in range(n):
            name = self.output_name_inputs[i].text() if i < len(self.output_name_inputs) else f"u{i+1}"
            self.plot_output_combo.addItem(f"Output {i+1} ({name})")
            self.restore_output_combo.addItem(f"Output {i+1} ({name})")
            for _r in getattr(self, 'inv_data_rows', []):
                _r['output_combo'].addItem(f"Output {i+1} ({name})")
        self.plot_output_combo.addItem("Custom...")
        self.restore_output_combo.addItem("Custom...")
        self._fit_combo_width(self.plot_output_combo, min_width=130)
        for _r in getattr(self, 'inv_data_rows', []):
            _r['output_combo'].addItem("Custom...")
            # Re-select "Custom..." (and keep the expression box visible)
            # for any row that already had a custom expression typed in --
            # clear()/re-add above reset the combo to its first item, which
            # would otherwise silently drop back to a raw output column.
            if _r['custom_expr'].text().strip():
                _r['output_combo'].setCurrentIndex(_r['output_combo'].count() - 1)
                _r['custom_expr'].setVisible(True)

    # Per built-in template: a list of (PDE row index, original constant
    # substring, trainable-variable row index) triples -- one per constant
    # that gets swapped for a trainable variable's live name when Inverse
    # mode is on. Most templates have exactly one entry (their single
    # unknown, always row 0 = the primary/first trainable variable); a
    # template with more than one unknown (e.g. the diffusion-reaction
    # system's D and kf, each appearing in both of its two PDE rows) lists
    # one triple per (PDE row, constant) pair it needs substituted.
    INVERSE_AUTO_CONST = {
        "1D Heat": [(0, "0.4", 0)],
        # gamma_1 (diffusion, 0.0001) and gamma_2 (reaction, the two
        # literal "5"s -- one on u**3, one on u), matching Wight & Zhao's
        # own eq. 3.2/3.5 naming. Two separate (idx, "5", 1) entries are
        # needed (not one) because the substitution is a single-occurrence
        # text replace and "5" appears twice in this PDE row -- each entry
        # replaces the next remaining occurrence, left to right, exactly
        # like the multi-row substitutions below (D/kf-style) but within
        # one row instead of across rows.
        "1D Allen-Cahn": [(0, "0.0001", 0), (0, "5", 1), (0, "5", 1)],
        "2D Heat": [(0, "0.4", 0)],
        # c_1^2 (interfacial-thickness-squared, 0.0001) and c_2 (reaction,
        # 1) -- Mattey & Ghosh's own eq. 13 coefficients. c_2's "1" isn't
        # written explicitly in the original PDE box text (an implicit
        # coefficient on (u**3 - u)), so the forward 'pde' string below was
        # given an explicit "1.0*" for exactly this purpose -- see that
        # template's own comment.
        "2D Allen-Cahn (Mattey & Ghosh)": [(0, "0.0001", 0), (0, "1.0", 1)],
        # lambda_ (the paper's own lambda=10, the reaction-term
        # coefficient) and diff_coeff (lambda*epsilon^2=0.00625, the
        # diffusion-term coefficient) -- Wight & Zhao's own eq. 3.11
        # parameters. Unlike the two templates above, no PDE-string rewrite
        # was needed here: both "10" and "0.00625" already appear as their
        # own literal substrings in the existing forward PDE box text.
        # NOTE: diff_coeff must NOT be spelled "lambda_eps_sq" (or anything
        # else starting with "lambda_") -- the substitution/restore logic
        # matches on substrings, and "lambda_" would then be a literal
        # prefix of the other variable's own name, corrupting it on
        # restore (str.replace finds "lambda_" wherever it appears first,
        # including inside "lambda_eps_sq" itself). Verified with a direct
        # round-trip simulation before choosing this name.
        "2D Allen-Cahn (Wight & Zhao)": [(0, "10", 0), (0, "0.00625", 1)],
        "3D Heat": [(0, "0.4", 0)],
        # lambda_1 (convection, the paper's true value 1.0 -- written
        # explicitly as "1.0*" in the forward PDE below so there's a
        # literal substring to substitute, matching the diffusion-reaction-
        # style pattern used above) and lambda_2 (viscosity numerator,
        # 0.01/pi) -- Raissi et al.'s own eq. B.1 naming (Appendix B, the
        # data-driven-discovery form of Burgers' equation).
        "1D Burgers": [(0, "1.0", 0), (0, "0.01", 1)],
        # The viscosity nu = 0.01/pi itself (the whole "0.01/pi", so the
        # unknown IS nu and its true value is 0.01/pi -- substituting only
        # the "0.01" would make the unknown pi*nu while INVERSE_AUTO_VARS
        # reported 0.01/pi as its true value), identical in both PDE rows
        # (U's and V's).
        "2D Burgers (Mathias)": [(0, "0.01/pi", 0), (1, "0.01/pi", 0)],
        # The nonlinear Schrodinger equation's shared 0.5 coefficient
        # (on dv_xx in u's PDE, du_xx in v's) -- one unknown, substituted
        # into both rows exactly like D/kf above.
        "1D Schrödinger": [(0, "0.5", 0), (1, "0.5", 0)],
        # The Poisson equation's RHS forcing constant (-Δu = 1) -- steady-
        # state problems have no time axis, but nothing about Inverse mode
        # (external_trainable_variables, the observed-data PointSetBC/
        # PointSetOperatorBC constraints, dde.data.PDE's anchors=) is
        # actually time-axis-specific, so this works the same way as every
        # other template here.
        "2D Poisson (L-Shape)": [(0, "1", 0)],
        "2D Poisson (Disk)": [(0, "1", 0)],
        # Same -Δu = 1 forcing constant as its 2D siblings above -- Inverse
        # mode substitution/observation-fitting is not itself time-axis- or
        # dimension-specific, so this template was previously missing all
        # three of INVERSE_AUTO_CONST/VARS/OBS purely because they were
        # never added, not because 3D needs anything different. Without
        # this entry, selecting Inverse mode for this template left the PDE
        # box's "1" unsubstituted and the Inverse Variables/observed-data
        # panels empty (or stale from whatever template was selected
        # before) -- training "succeeded" while fitting a variable that
        # played no role in the actual equation.
        "3D Poisson (Sphere)": [(0, "1", 0)],
    }

    def _inv_var_name(self, row_index):
        if 0 <= row_index < len(self.inv_var_rows):
            return self.inv_var_rows[row_index]['name'].text().strip() or f"trainable_variable_{row_index + 1}"
        return "trainable_variable_1" if row_index == 0 else f"trainable_variable_{row_index + 1}"

    # Per built-in template that has a known ground-truth value for at
    # least one unknown (every template with an INVERSE_AUTO_CONST entry
    # now has one here too -- previously six of them -- 1D Heat, 1D
    # Allen-Cahn, 2D Heat, 2D Allen-Cahn (Mattey & Ghosh), 2D Allen-Cahn
    # (Wight & Zhao), 3D Heat -- had an INVERSE_AUTO_CONST entry (so their
    # PDE box WAS correctly substituted) but no entry here, so the
    # always-present default variable row simply kept whatever init/true
    # values were left over from whichever template had been selected
    # before it, e.g. selecting 1D Allen-Cahn after 1D Burgers showed
    # Burgers' own init=0.02/true=0.01 instead of Allen-Cahn's real
    # init=1.0/true=0.0001): the trainable-variable rows and
    # observed-data-file rows to seed the Inverse panel with
    # automatically, so the example works out of the box instead of
    # requiring the user to hand-add a row and name it to match
    # INVERSE_AUTO_CONST exactly. (name, init, true) per variable row,
    # primary first -- init is deliberately NOT the true value (an inverse
    # problem that starts already at the answer proves nothing about
    # whether the fit actually recovers it -- every template here starts
    # at init=1.0 for exactly this reason, except the Poisson family,
    # whose true value already IS 1.0, so it starts at 0.1 instead), true
    # is the real PDE constant this variable replaced, used only by
    # codegen.py's parameter-convergence plot to draw a dashed reference
    # line at. (path, output_idx, weight) or (path, output_idx, weight,
    # custom_expr) per observation file row, primary first -- the optional
    # 4th element pre-selects the "Custom..." derived-field option (e.g.
    # 1D Schrodinger's data is |h| = sqrt(u**2+v**2), not output u or v
    # alone) even when, as for Schrodinger today, no bundled file path is
    # supplied and the user still has to browse to one. Paths that ARE
    # supplied point at the template's own reference-data folder, generated
    # by a patch as a subsample of each template's own bundled solution.txt
    # (see u_obs.txt in the same folder). For 2D Poisson (L-Shape/Disk)
    # that solution.txt is a real numerical solution, never fabricated
    # from a closed form, since the L-Shape domain's reentrant corner has
    # no simple one (that's the point of the example) -- so the Disk
    # template (which DOES have one, u=(1-r^2)/4) isn't treated any
    # differently from it here. 3D Poisson (Sphere) is the one exception:
    # its solution.txt IS generated directly from its known closed form
    # u=(R^2-r^2)/6 (see templates_3d_steady above) since no
    # reentrant-corner-style complication exists for a sphere to make a
    # closed form untrustworthy. The 6 templates below with no bundled
    # observed-data file (1D/2D/3D Heat, 1D/2D Allen-Cahn) have no
    # INVERSE_AUTO_OBS entry either -- same as 1D Schrodinger, the user
    # just has to Browse to a file of their own.
    # Every name below is a valid Python identifier (checked against
    # keyword.iskeyword() and non-identifier characters) since it gets
    # spliced directly into generated code as "{name} = dde.Variable(...)"
    # -- codegen.py's _sanitize_python_identifier is a safety net for
    # anything a user later types into the name box by hand, not a
    # substitute for picking valid names here. Two names deliberately
    # aren't the paper's own bare symbol: "lambda_" (2D Allen-Cahn,
    # Wight & Zhao) has a trailing underscore because "lambda" alone is a
    # Python keyword; "source_term" (the Poisson family) and "alpha" (the
    # Heat family) aren't named by their own papers/have no cited paper at
    # all, so these are just descriptive choices, not paper notation.
    INVERSE_AUTO_VARS = {
        "1D Heat": [("alpha", 1.0, 0.4)],
        # gamma_1 (diffusion) and gamma_2 (reaction) -- Wight & Zhao's own
        # eq. 3.2/3.5 names. Both start at init=1.0 (neither's true value
        # is 1, so this doesn't start the fit already at the answer).
        "1D Allen-Cahn": [("gamma_1", 1.0, 0.0001), ("gamma_2", 1.0, 5.0)],
        "2D Heat": [("alpha", 1.0, 0.4)],
        # c_1^2 (interfacial-thickness-squared) and c_2 (reaction) --
        # Mattey & Ghosh's own eq. 13 names. c_2's true value is already 1
        # (same situation as the Poisson family below), so its guess also
        # starts an order of magnitude away instead of right at the answer.
        "2D Allen-Cahn (Mattey & Ghosh)": [("c1_sq", 1.0, 0.0001), ("c2", 0.1, 1.0)],
        # lambda_ (reaction-term coefficient) and diff_coeff (the lumped
        # lambda*epsilon^2 diffusion-term coefficient, written in words
        # since there's no clean single symbol for a product, and
        # deliberately NOT "lambda_eps_sq" -- see INVERSE_AUTO_CONST's
        # comment on this template for why) -- Wight & Zhao's own eq. 3.11
        # parameters, recovered independently exactly as they appear in
        # the PDE box (not epsilon on its own).
        "2D Allen-Cahn (Wight & Zhao)": [("lambda_", 1.0, 10.0), ("diff_coeff", 1.0, 0.00625)],
        "3D Heat": [("alpha", 1.0, 0.4)],
        # lambda_1 (convection) and lambda_2 (viscosity numerator) --
        # Raissi et al.'s own eq. B.1 names (Appendix B). lambda_1's true
        # value is already 1, so -- same reasoning as the Poisson family --
        # its guess starts at 0.1 rather than right at the answer;
        # lambda_2 keeps the existing init=1.0 (true 0.01/pi is nowhere
        # near it).
        "1D Burgers": [("lambda_1", 0.1, 1.0), ("lambda_2", 1.0, 0.01)],
        "2D Burgers (Mathias)": [("nu", 1.0, 0.01 / math.pi)],
        "1D Schrödinger": [("trainable_variable_1", 1.0, 0.5)],
        # True value is already 1 -- starting the initial guess there
        # would make the "inference" trivial (zero iterations needed),
        # so it starts an order of magnitude away instead.
        "2D Poisson (L-Shape)": [("source_term", 0.1, 1.0)],
        "2D Poisson (Disk)": [("source_term", 0.1, 1.0)],
        "3D Poisson (Sphere)": [("source_term", 0.1, 1.0)],
    }
    INVERSE_AUTO_OBS = {
        # No bundled file -- there's no default "the" |h| measurement file
        # for this template (unlike the Poisson family's u_obs.txt below),
        # but the measured quantity for this template is always |h|, never
        # a raw output, so the output-selector default is still worth seeding:
        # the user only has to Browse to a file, not also discover and
        # turn on "Custom...". The "+1e-12" inside the sqrt isn't cosmetic:
        # d/du sqrt(u**2+v**2) = u/sqrt(u**2+v**2) is singular at u=v=0, and
        # a freshly-initialized 2-output network genuinely can (and, in
        # testing, reliably did on small networks) land exactly on that
        # point at some collocation points early in training, blowing the
        # gradient up to inf/nan and poisoning every other loss term for
        # the rest of the run. The epsilon is far below any physically
        # meaningful |h| for this problem (h ranges roughly 0-2), so it
        # doesn't change what's being fit -- it only removes the
        # zero-crossing singularity. A user who edits this expression by
        # hand takes that risk back on knowingly.
        "1D Schrödinger": [("", 0, 100.0, "sqrt(u**2+v**2+1e-12)")],
        "2D Poisson (L-Shape)": [
            (os.path.join(REFERENCE_DATA_DIR, "2D", "poisson_lshape", "u_obs.txt"), 0, 100.0),
        ],
        "2D Poisson (Disk)": [
            (os.path.join(REFERENCE_DATA_DIR, "2D", "poisson_disk", "u_obs.txt"), 0, 100.0),
        ],
        # Unlike the two 2D Poisson templates (whose ground truth is a real
        # numerical solution and has no simple closed form for L-Shape),
        # the Sphere's exact closed-form solution u=(R^2-r^2)/6 IS known
        # (see templates_3d_steady above), so both solution.txt (a
        # closed-form grid over the bounding cuboid, NaN outside the
        # sphere) and this subsample were generated directly from it --
        # not fabricated data, just the same closed form already used
        # elsewhere in this codebase for this exact template.
        "3D Poisson (Sphere)": [
            (os.path.join(REFERENCE_DATA_DIR, "3D", "poisson_sphere", "u_obs.txt"), 0, 100.0),
        ],
        # Both velocity components measured at t = 1, each against its own
        # output (a single file would leave the other component unmeasured,
        # and the end-time default picked V's file for output U).
        "2D Burgers (Mathias)": [
            (os.path.join(REFERENCE_DATA_DIR, "2D", "burgers_mathias", "t_1.0_u.txt"), 0, 100.0),
            (os.path.join(REFERENCE_DATA_DIR, "2D", "burgers_mathias", "t_1.0_v.txt"), 1, 100.0),
        ],
    }

    def _sync_inverse_multi_setup(self, is_inv):
        """Inverse ON, for a template in INVERSE_AUTO_VARS/INVERSE_AUTO_OBS:
        rebuild the Inverse panel's variable and observed-data-file rows
        from that template's list, replacing whatever was there (mirrors
        _populate_locked_bc_entries_from_legacy's "template output
        replaces prior panel state" convention). Only runs when the
        current template actually has an entry -- a template with no
        known true value / default observed-data setup keeps using
        whatever the user already has in the (always-present) primary
        row, exactly as before this method existed. Inverse OFF is a
        no-op: the rows are left as they are (same as every other
        Inverse-only panel), since nothing here needs undoing the way a
        PDE-box text substitution does."""
        if not is_inv:
            return
        template = getattr(self, '_current_template', '')
        var_list = self.INVERSE_AUTO_VARS.get(template)
        if var_list:
            for r in list(self.inv_var_rows):
                r['widget'].deleteLater()
            self.inv_var_rows.clear()
            for i, (name, init, true) in enumerate(var_list):
                self._add_inverse_var_row(name=name, init=init, true_value=true, is_primary=(i == 0))
        obs_list = self.INVERSE_AUTO_OBS.get(template)
        if obs_list:
            # If _auto_configure_ea already found and auto-loaded a real
            # observed-data file for the CURRENTLY selected template (e.g.
            # 1D Schrodinger, whose own default path below is intentionally
            # ""  --  no bundled |h| file ships with it), preserve that
            # path instead of unconditionally wiping it back to the
            # template's static default: without this, switching Problem
            # Type to Inverse right after a template that just auto-loaded
            # a file produced exactly the contradiction reported --
            # "auto-loaded: t_1.0.txt" in the log, immediately followed by
            # "no observation row has a file selected" at Solve time,
            # because this rebuild ran after the auto-load and silently
            # discarded it. Guarded on the auto-load being for THIS SAME
            # template (not a stale path pointing at a file that belongs
            # to whatever template was selected before this one) -- when
            # _auto_configure_ea instead runs AFTER this rebuild (the
            # normal order when a template is freshly selected while
            # already in Inverse mode), it sets the row's path directly,
            # so there's nothing to preserve here and this stays empty.
            _preserve_path = ""
            if (not obs_list[0][0]
                    and getattr(self, '_ea_auto_inv_template', None) == template):
                _preserve_path = getattr(self, '_ea_auto_inv_path', '') or ""
            for r in list(self.inv_data_rows):
                r['widget'].deleteLater()
            self.inv_data_rows.clear()
            for i, entry in enumerate(obs_list):
                path, output_idx, weight = entry[0], entry[1], entry[2]
                custom_expr = entry[3] if len(entry) > 3 else ""
                if i == 0 and not path and _preserve_path:
                    path = _preserve_path
                self._add_inverse_data_row(path=path, output_idx=output_idx, weight=weight,
                                            custom_expr=custom_expr, is_primary=(i == 0))

    def _sync_inverse_pde_substitution(self, is_inv):
        """Inverse ON: replace the current template's known constants with
        their trainable-variable names in its PDE box(es). Inverse OFF:
        restore the original numeric constants so Forward mode stays
        valid. Also seeds the Inverse panel's variable/observed-data rows
        for templates that need more than the single always-present
        default row (see _sync_inverse_multi_setup) -- called from here,
        not duplicated at each of this method's call sites, since the two
        always need to happen together."""
        self._sync_inverse_multi_setup(is_inv)
        template = getattr(self, '_current_template', '')
        entries = self.INVERSE_AUTO_CONST.get(template)
        if is_inv:
            if not entries:
                return
            active = []
            for idx, const_str, var_row in entries:
                if idx >= len(self.pde_inputs):
                    continue
                var_name = self._inv_var_name(var_row)
                text = self.pde_inputs[idx].text()
                if const_str in text:
                    self.pde_inputs[idx].setText(text.replace(const_str, var_name, 1))
                    active.append((template, idx, const_str, var_name, var_row))
            self._inverse_sub_active = active
        else:
            for s_template, s_idx, s_const, s_var, s_var_row in getattr(self, '_inverse_sub_active', None) or []:
                if s_idx < len(self.pde_inputs):
                    cur = self.pde_inputs[s_idx].text()
                    if s_var in cur:
                        self.pde_inputs[s_idx].setText(cur.replace(s_var, s_const, 1))
            self._inverse_sub_active = []

    def _on_inv_param_name_changed(self):
        """Keep any already-substituted PDE(s) in sync if a trainable
        variable is renamed while Inverse mode is active -- covers every
        active substitution tied to the row that changed, not just the
        first."""
        active = getattr(self, '_inverse_sub_active', None) or []
        if not active:
            return
        new_active = []
        for template, idx, const_str, old_var, var_row in active:
            new_var = self._inv_var_name(var_row)
            if new_var and new_var != old_var and idx < len(self.pde_inputs):
                text = self.pde_inputs[idx].text()
                if old_var in text:
                    self.pde_inputs[idx].setText(text.replace(old_var, new_var, 1))
                    old_var = new_var
            new_active.append((template, idx, const_str, old_var, var_row))
        self._inverse_sub_active = new_active

    def _on_problem_type_changed(self, checked):
        is_inv = self.radio_inverse.isChecked()
        self._sync_inverse_pde_substitution(is_inv)
        self.inverse_group.setVisible(is_inv)
        self.param_save_label.setVisible(is_inv)
        self.param_save_combo.setVisible(is_inv)
        # "Parameter Convergence" (iteration vs. inferred-value chart) only
        # means anything for Inverse problems -- add it as a selectable
        # Plot Type only while Inverse is active, and default to it (the
        # most useful view for an Inverse run) the moment Inverse is
        # switched on. Removing it when leaving Inverse (rather than just
        # hiding it) keeps a stray leftover selection from silently
        # producing no solution plot for a Forward problem.
        _pc_idx = self.plot_type_combo.findText("Parameter Convergence")
        # blockSignals: like the dimension-switch fallback above, these two
        # setCurrentText calls are a programmatic side effect of toggling
        # Inverse/Forward, not a user picking a new plot type -- now that
        # _on_plot_type_changed() auto-pops the settings dialog for every
        # selection (not just line/animation types as before), leaving
        # these unguarded would pop it out of nowhere every time Inverse
        # mode is switched on or off.
        if is_inv:
            if _pc_idx == -1:
                self.plot_type_combo.addItem("Parameter Convergence")
                # "Parameter Convergence" is the widest item this combo
                # ever holds -- re-fit now so it isn't clipped the moment
                # Inverse mode adds it (see _fit_combo_width's own note
                # that it must be re-called after every addItem).
                self._fit_combo_width(self.plot_type_combo, min_width=160)
            self.plot_type_combo.blockSignals(True)
            self.plot_type_combo.setCurrentText("Parameter Convergence")
            self.plot_type_combo.blockSignals(False)
            self._plot_type_prev = "Parameter Convergence"
        elif _pc_idx != -1:
            if self.plot_type_combo.currentText() == "Parameter Convergence":
                self.plot_type_combo.blockSignals(True)
                self.plot_type_combo.setCurrentText("Surface")
                self.plot_type_combo.blockSignals(False)
                self._plot_type_prev = "Surface"
            self.plot_type_combo.removeItem(_pc_idx)
        # Time Adaptive training does not yet wire the inferred parameter
        # into its per-step training loop (no external_trainable_variables
        # there), so it silently would not converge for Inverse problems --
        # remove it as a selectable option in Inverse mode and restore any
        # template default that was suspended when returning to Forward.
        _ta_idx = self.adapt_combo.findText("Time Adaptive Training")
        if is_inv:
            if self.adapt_combo.currentText() == "Time Adaptive Training":
                self._ta_suspended_for_inverse = True
                self.adapt_combo.setCurrentText("None")
            if _ta_idx != -1:
                self.adapt_combo.removeItem(_ta_idx)
        else:
            if _ta_idx == -1:
                self.adapt_combo.addItem("Time Adaptive Training", "Time Adaptive")
            if getattr(self, '_ta_suspended_for_inverse', False):
                self._ta_suspended_for_inverse = False
                _ta_cfg = getattr(self, '_current_ta_cfg', None)
                if _ta_cfg:
                    for _row in list(self.ta_group_rows):
                        _row['widget'].deleteLater()
                    self.ta_group_rows.clear()
                    self.adapt_combo.setCurrentText("Time Adaptive Training")
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
        #
        # Previously this required BOTH _fwd_tmax and _inv_tmax to be
        # non-None before restoring either, which only worked by accident:
        # every template dispatch block now always records a real forward
        # t_max (defaulting to 1.0, the same fallback already used to set
        # the widget itself at template-load time -- see the t.get('t_max',
        # 1.0) calls above), so _fwd_tmax is never actually None for a
        # non-steady template. But a future template that set only
        # inverse_t_max (relying on the generic 1.0 forward default without
        # ever recording it) would have made this whole block a no-op --
        # switching to Inverse would apply inverse_t_max, and switching
        # back to Forward would then silently leave t_max stuck at the
        # shortened Inverse value forever, with no restore happening at
        # all. Only _inv_tmax needs to be set for this template to need any
        # restoring in the first place -- a template with no shortened
        # Inverse range needs no override either way.
        _fwd_tmax = getattr(self, '_current_template_forward_tmax', None)
        _inv_tmax = getattr(self, '_current_template_inverse_tmax', None)
        if _inv_tmax is not None:
            self.t_max.setValue(_inv_tmax if is_inv else (_fwd_tmax if _fwd_tmax is not None else 1.0))
            self._auto_configure_ea(getattr(self, '_template_ref_dir', ''))

    def _on_browse_inv_data(self, target=None):
        if target is None:
            target = self.inv_data_path
        f, _ = QFileDialog.getOpenFileName(self, "Select measured data file", "", "Data files (*.txt *.csv *.dat)")
        if f:
            target.setText(f)


    def _geometry_supported_for_training(self):
        """codegen.py's _build_geom() constructs every shape in the
        Geometry Type selector (Interval; Rectangle/Disk/Ellipse/Triangle/
        Polygon in 2D; Cuboid/Sphere in 3D; Custom in both, via
        _build_custom_geom_code()'s CSG chain), so Solve is no longer
        blocked for any of them. Kept as a real gate (rather than deleted
        outright) so a future shape added to the selector without
        matching codegen support fails the same clean, explicit way this
        always has, instead of silently training on the wrong geometry.

        Custom's own emptiness/malformed-shape-list problems are not this
        gate's job -- those already surface through config.validate()'s
        own specific error messages (called right after this gate, in
        _on_solve), so this only needs to know that the *type* "Custom"
        itself has codegen support, which it has had since Phase 1 (2D)
        and the 3D extension (Cuboid/Sphere) -- this list simply hadn't
        been updated to say so until now."""
        geom_type = self._current_geometry_type()
        _supported = ("Interval", "Rectangle", "Disk", "Ellipse", "Triangle",
                      "Polygon", "Cuboid", "Sphere", "Custom")
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

    def _log_example_changes(self, config):
        """After the log is cleared for a run: if a DeepXDE example is
        loaded and some of its settings were changed, say which."""
        snap = getattr(self, "_example_snapshot", None)
        if not snap:
            return
        try:
            changes = _dx_examples.describe_changes(snap[1], config)
        except Exception:
            return
        if changes:
            self.log_box.append(f"✏️ Changed from the \"{_ex_catalog.display_name(snap[0])}\" example:")
            for c in changes:
                self.log_box.append(f"   • {c}")
        else:
            self.log_box.append(f"ℹ️ \"{_ex_catalog.display_name(snap[0])}\" example with its own settings (unchanged).")

    def _on_solve(self):
        if hasattr(self, 'right_tabs'):
            self.right_tabs.setCurrentWidget(self.results_tabs)   # the run streams into the log
        _ok, _msg = self._geometry_supported_for_training()
        if not _ok:
            self.log_box.append(_msg)
            return
        _config_preview = self._build_config()
        _ok2, _msg2 = self._validate_optimizer_settings(_config_preview)
        if not _ok2:
            self.log_box.append(_msg2)
            return
        # Belt-and-suspenders alongside the JSON-load-time check in
        # _open_problem(): the live UI's own widgets should never be able
        # to produce an invalid config, but this also catches a config
        # loaded earlier in the same session (before this check existed, or
        # from an older PINNStudio version's save file) and never
        # re-validated since, without duplicating the optimizer-specific
        # checks above.
        _val_errors = _config_preview.validate()
        if _val_errors:
            self._check_problems()
            self._goto_first_problem()
            self.log_box.append("❌ Cannot start -- this configuration has invalid settings:")
            for _ve in _val_errors:
                self.log_box.append(f"   • {_ve}")
            return
        config = _config_preview
        # Enable Parameter Sweep (left panel, right after Adaptive
        # Training) is what decides which of the two Solve actually
        # does -- there's no separate Run Sweep button anymore, and
        # Stop/Cancel is the matching single branch in _on_stop below.
        if config.sweep_enabled:
            self._start_sweep(config)
            return
        self.solve_btn.setEnabled(False)
        self.solve_btn.setText(" Solving…")
        self.stop_btn.setEnabled(True)
        self.log_box.clear()
        self._clear_solution_movie()
        self._reset_plot_headers()
        self.loss_label.setText("⏳ Training...")
        self.solution_label.setText("⏳ Training...")
        self.loss_label._source_path = None
        self.solution_label._source_path = None
        for _p in ["/tmp/loss_plot.png", "/tmp/solution_plot.png", "/tmp/solution_plot.gif", "/tmp/param_plot.png"]:
            if os.path.exists(_p):
                os.remove(_p)
        self._log_example_changes(config)
        self._start_progress(config)
        self.thread = SolverThread(config)
        self.thread.output_signal.connect(self._on_output)
        self.thread.done_signal.connect(self._on_done)
        self.thread.start()

    def _start_sweep(self, config):
        """Runs a Parameter Sweep through the exact same Solve/Stop
        buttons a normal single run uses (see _on_solve, which routes
        here once config.sweep_enabled is true, and _on_stop below) --
        there's no separate Run Sweep/Cancel UI anymore. Progress and
        each run's result stream into the same Training Log a normal
        Solve already writes to (see the signal handlers below), rather
        than a dedicated results table: every run still gets its own
        folder under the sweep's root (logged once up front, as soon as
        SweepThread knows it), and sweep_runner.run_sweep() itself
        already writes a sweep_manifest.json/sweep_summary.csv there
        once everything finishes, exactly as before -- only where that
        progress is shown changed, not what gets saved to disk."""
        self.solve_btn.setEnabled(False)
        self.solve_btn.setText(" Running sweep…")
        self.stop_btn.setEnabled(True)
        self.log_box.clear()
        self._clear_solution_movie()
        self._reset_plot_headers()
        self.loss_label.setText("⏳ Training...")
        self.solution_label.setText("⏳ Training...")
        self.loss_label._source_path = None
        self.solution_label._source_path = None
        self._log_example_changes(config)
        # The sweep's folder is named like a normal run's
        # ("<label>__<timestamp>"): config.save_dir already ends in that
        # run folder, so its label part names the sweep too.
        config.sweep_label = os.path.basename(os.path.normpath(config.save_dir or "")).rsplit("__", 1)[0] \
            if "__" in os.path.basename(os.path.normpath(config.save_dir or "")) else ""
        self._sweep_run_count = 0
        self._sweep_run_root = None
        self._sweep_base_config = config
        self._sweep_last_run_dir = None
        self._sweep_last_run_label = None
        self.sweep_thread = SweepThread(config)
        self.sweep_thread.sweep_root_signal.connect(self._on_sweep_root)
        self.sweep_thread.run_start_signal.connect(self._on_sweep_run_start)
        self.sweep_thread.output_signal.connect(self._on_sweep_output)
        self.sweep_thread.run_done_signal.connect(self._on_sweep_run_done)
        self.sweep_thread.finished_signal.connect(self._on_sweep_finished)
        self.sweep_thread.start()

    def _on_sweep_root(self, root):
        """Fires once, right as the sweep starts -- see
        sweep_runner.run_sweep()'s on_sweep_root param. Logged
        immediately so the user knows where results are landing without
        waiting for the whole sweep to finish."""
        self._sweep_run_root = root
        self.log_box.append(f"📁 Sweep results will be saved under: {root}")

    def _on_sweep_run_start(self, i, total, label):
        self.log_box.append(f"▶ Run {i + 1}/{total}: {label}")
        self._start_progress(getattr(self, '_sweep_base_config', None), f"Sweep run {i + 1}/{total}")

    @staticmethod
    def _log_text(line):
        """How a script line is shown in the Training Log: the relative L2
        error written L₂ (the script's own output and files keep "L2")."""
        return _re_log_l2.sub("L₂", line) if "L2" in line else line

    def _on_sweep_output(self, i, total, line):
        self._progress_feed(line)
        if line.strip().startswith(self._LOG_NOISE):
            return
        self.log_box.append(f"[Sweep {i + 1}/{total}] {self._log_text(line)}")

    def _on_sweep_run_done(self, i, total, label, result):
        self._sweep_run_count += 1
        status_icon = "✅" if result.get("status") == "done" else "❌"
        fl = result.get("final_loss")
        msg = (f"[Sweep {i + 1}/{total}] {status_icon} {label} -- final loss: "
               f"{fl:.4e}" if fl is not None else
               f"[Sweep {i + 1}/{total}] {status_icon} {label} -- final loss: —")
        l2 = result.get("l2_relative")
        if l2 is not None:
            n_t = int((result.get("metrics") or {}).get("n_times", 1))
            msg += (f", L₂ relative error: {l2:.4e}" if n_t <= 1
                    else f", L₂ relative error (mean over {n_t} reference times): {l2:.4e}")
        self.log_box.append(msg)
        # Remember the last run that actually finished successfully (not
        # necessarily the last one in the sequence -- if the sweep is
        # stopped partway or a later run errors, the right panel should
        # still end up showing a real figure from whichever run last
        # produced one, not nothing). See _on_sweep_finished, which
        # displays this once the whole sweep is done.
        if result.get("status") == "done":
            self._sweep_last_run_dir = result.get("save_dir")
            self._sweep_last_run_label = label

    def _on_sweep_finished(self):
        if not getattr(self, "_stopped", False):
            self._finish_progress("Sweep finished", folder=getattr(self, '_sweep_last_run_dir', None) or "")
        self.solve_btn.setEnabled(True)
        self.solve_btn.setText(" Run Sweep" if self.sweep_enable_cb.isChecked() else " Solve")
        self.stop_btn.setEnabled(False)
        self.log_box.append(f"✅ Sweep finished -- {self._sweep_run_count} run(s) completed.")
        # sweep_runner.run_sweep() already wrote sweep_manifest.json and
        # sweep_summary.csv into the sweep's own root folder -- this is
        # just telling the user that's there, not writing it itself.
        if getattr(self, '_sweep_run_root', None):
            self.log_box.append(
                f"📁 Per-run folders, manifest and summary CSV saved under: {self._sweep_run_root}")
            _summary = os.path.join(self._sweep_run_root, "sweep_error_vs_parameter.png")
            if os.path.exists(_summary):
                self.log_box.append(f"📈 Error vs swept parameter: {_summary}")
        # Show the last successfully-completed run's own loss/solution
        # figure in the right panel -- same as a normal single Solve
        # already does, using whichever plot type the Setup tab itself
        # has chosen (Surface by default), since every run in the sweep
        # reuses that same setting (see sweep_runner._apply_run_output_
        # settings -- "same_as_setup" leaves plot_type untouched). There
        # is no separate sweep-vs-parameter comparison chart -- just
        # this one run's own figure, exactly like a normal Solve leaves
        # behind.
        if getattr(self, '_sweep_last_run_dir', None):
            self._display_run_result_plots(self._sweep_base_config, self._sweep_last_run_dir)
            self.log_box.append(
                f"🖼️ Showing the loss/solution figure for the last completed run: {self._sweep_last_run_label}")

    # ── run progress: status bar, live loss, error analysis ───
    def _build_status_bar(self):
        from PyQt6.QtWidgets import QProgressBar, QStatusBar
        sb = QStatusBar()
        sb.setObjectName("appStatus")
        sb.setSizeGripEnabled(True)
        self.setStatusBar(sb)
        self.status_text = QLabel("Ready")
        self.status_text.setObjectName("statusText")
        sb.addWidget(self.status_text)
        self.status_progress = QProgressBar()
        self.status_progress.setFixedWidth(220)
        self.status_progress.setFixedHeight(14)
        self.status_progress.setTextVisible(False)
        self.status_progress.setVisible(False)
        sb.addWidget(self.status_progress)
        self.status_detail = QLabel("")
        self.status_detail.setObjectName("statusDetail")
        sb.addWidget(self.status_detail, 1)
        self.status_env = QLabel("")
        self.status_env.setObjectName("statusEnv")
        sb.addPermanentWidget(self.status_env)
        self._progress = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_status)
        self._refresh_status_env()
        self.save_dir_input.textChanged.connect(self._refresh_status_env)

    def _refresh_status_env(self, *_a):
        if not hasattr(self, 'status_env'):
            return
        folder = (self.save_dir_input.text() or "").strip()
        short = folder if len(folder) <= 42 else "…" + folder[-41:]
        self.status_env.setText(f"{getattr(self, '_float_type', 'float32')}   ·   Save to: {short}")
        self.status_env.setToolTip(folder)

    def _start_progress(self, config, label="Training"):
        from pinnstudio.ui.results import RunProgress
        if not hasattr(self, 'status_text'):
            return
        self._progress = RunProgress(config)
        self._progress_label = label
        self._stopped = False
        self.status_progress.setVisible(True)
        self.status_progress.setRange(0, 1000)
        self.status_progress.setValue(0)
        self.live_loss.start(self._progress)
        self.ea_panel.clear()
        self.results_tabs.setCurrentWidget(self.live_loss)
        self._progress_timer.start()
        self._refresh_status()

    def _progress_feed(self, line):
        p = getattr(self, '_progress', None)
        if p is None:
            return
        n = len(p.history)
        p.feed(line)
        if len(p.history) != n:
            self.live_loss.update_from(p)

    def _refresh_status(self):
        from pinnstudio.ui.results import fmt_seconds
        p = getattr(self, '_progress', None)
        if p is None:
            return
        parts = [self._progress_label]
        if p.n_time_steps:
            parts.append(f"time step {p.time_step}/{p.n_time_steps}")
        if p.phase:
            name = {"adam": "Adam", "lbfgs": "L-BFGS", "l-bfgs": "L-BFGS"}.get(p.phase_name.lower(), p.phase_name)
            parts.append(f"phase {p.phase}/{max(len(p.phases), p.phase)} ({name})")
        self.status_text.setText("  ·  ".join(parts))
        f = p.fraction()
        if f is None:
            self.status_progress.setRange(0, 0)   # busy
        else:
            self.status_progress.setRange(0, 1000)
            self.status_progress.setValue(int(1000 * f))
        det = []
        if p.total:
            det.append(f"iteration {p.step:,} / {p.total:,}")
        tl = p.train_loss()
        if tl is not None:
            det.append(f"loss {tl:.3e}")
        det.append(f"elapsed {fmt_seconds(p.elapsed())}")
        eta = p.eta()
        if eta is not None:
            det.append(f"about {fmt_seconds(eta)} left")
        self.status_detail.setText("   ·   ".join(det))

    def _finish_progress(self, message, ok=True, folder=None):
        """The run ended: sum it up in the status bar and, when it finished,
        show its error analysis (from `folder`, the run's own folder)."""
        from pinnstudio.ui.results import fmt_seconds
        p = getattr(self, '_progress', None)
        self._progress = None    # later output (after Stop) is not this run's
        if hasattr(self, "run_history"):
            self.run_history.stale = True
        if not hasattr(self, 'status_text'):
            return
        self._progress_timer.stop()
        self.status_progress.setVisible(False)
        if p is None:
            self.status_text.setText(message)
            return
        self.live_loss.update_from(p)
        self.live_loss.stop()
        tl = p.final_loss if p.final_loss is not None else p.train_loss()
        bits = [message, f"in {fmt_seconds(p.elapsed())}"]
        if tl is not None:
            bits.append(f"final loss {tl:.3e}")
        l2s = [m[1] for m in p.metrics if m[1] is not None]
        if l2s:
            bits.append(f"mean relative L₂ {sum(l2s) / len(l2s):.3e}")
        self.status_text.setText("  ·  ".join(bits))
        self.status_detail.setText("")
        if ok:
            ea = os.path.join(folder, "error_analysis") if folder else ""
            self.ea_panel.show_run(p.metrics, ea, p.figures)
            self.results_tabs.setCurrentIndex(0)

    def _figure_save_button(self, tip):
        from PyQt6.QtWidgets import QToolButton
        from pinnstudio.ui import icons as _icons_f
        b = QToolButton()
        b.setObjectName("figureSaveButton")
        b.setAutoRaise(True)
        b.setFixedSize(26, 24)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        _icons_f.apply(b, "download", "muted", size=16)
        return b

    def _open_figure(self, label):
        from pinnstudio.ui.results import ImageViewer
        path = getattr(label, "_source_path", None)
        if path and os.path.exists(path) and not path.lower().endswith(".gif"):
            ImageViewer(path, self).exec()

    def _on_stop(self):
        if hasattr(self, 'sweep_thread') and self.sweep_thread.isRunning():
            # Same hard-stop semantics as the normal-Solve branch below:
            # SweepThread.stop() kills the currently-running training
            # subprocess right away (see its own docstring) rather than
            # letting it keep training to completion first.
            self.sweep_thread.stop()
            self.sweep_thread.wait(3000)
            self.log_box.append("\n⏹ Stopped by user.")
            self._stopped = True
            self._finish_progress("Stopped", ok=False)
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText(
                " Run Sweep" if getattr(self, 'sweep_enable_cb', None) and self.sweep_enable_cb.isChecked()
                else " Solve")
            self.stop_btn.setEnabled(False)
            return
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(3000)
            self.log_box.append("\n⏹ Stopped by user.")
            self._stopped = True
            self._finish_progress("Stopped", ok=False)
            self.solve_btn.setEnabled(True)
            self.solve_btn.setText(
                " Run Sweep" if getattr(self, 'sweep_enable_cb', None) and self.sweep_enable_cb.isChecked()
                else " Solve")
            self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        """Make sure no training/restore subprocess is left running in the
        background when the window closes. Previously there was no
        closeEvent override at all, so closing PINNStudio mid-run never
        stopped the active SolverThread (or _RestoreThread) -- the child
        Python/PyTorch process, and any GPU memory it holds, kept running
        detached from the GUI indefinitely. If a run is active, ask before
        quitting rather than silently killing it."""
        self._save_layout()
        # isinstance(..., QThread) rather than a plain hasattr check: every
        # QObject (QMainWindow included) already has its own built-in
        # .thread() method from Qt itself, so hasattr(self, 'thread') is
        # True even on a freshly-opened window that never clicked Solve --
        # self.thread only becomes a real SolverThread once _on_solve()
        # first assigns it. Without this check, closing a window that never
        # trained anything would crash here instead of just closing.
        active = []
        _th = getattr(self, 'thread', None)
        if isinstance(_th, QThread) and _th.isRunning():
            active.append(("training", _th))
        _swth = getattr(self, 'sweep_thread', None)
        if isinstance(_swth, QThread) and _swth.isRunning():
            active.append(("sweep", _swth))
        _rth = getattr(self, '_restore_thread', None)
        if isinstance(_rth, QThread) and _rth.isRunning():
            active.append(("restore", _rth))
        _pth = getattr(self, '_preview_thread', None)
        if isinstance(_pth, QThread) and _pth.isRunning():
            active.append(("preview", _pth))
        if active:
            names = " and ".join(name for name, _ in active)
            reply = QMessageBox.question(
                self, "Run in progress",
                f"A {names} run is still in progress. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if not self._maybe_save_changes():   # after the run question: cancel there asks nothing more
            event.ignore()
            return
        if active:
            for _name, th in active:
                if hasattr(th, 'stop'):
                    th.stop()
                th.wait(3000)
                if th.isRunning():
                    # terminate() didn't get a response in time -- force it
                    # rather than let the window hang on close.
                    th.terminate()
                    th.wait(1000)
        event.accept()

    # DeepXDE's import-time advertisement, printed by every script it runs
    _LOG_NOISE = ("Other supported backends:", "paddle supports more examples now and is recommended")

    def _on_output(self, line):
        self._progress_feed(line)
        if line.strip().startswith(self._LOG_NOISE):
            return
        self.log_box.append(self._log_text(line))
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _clear_solution_movie(self):
        """Stop and drop any QMovie currently animating in solution_label.
        QLabel.setPixmap()/.setText() replace what's *displayed* but don't
        stop a previously-set QMovie from continuing to run/decode frames
        in the background, so this has to be called explicitly before
        showing anything else there -- otherwise the old animation keeps
        running invisibly."""
        movie = getattr(self, '_solution_movie', None)
        if movie is not None:
            movie.stop()
            self._solution_movie = None

    def _set_solution_gif(self, path):
        """Animate a solution GIF (Line/Surface Animation plot types)
        directly in solution_label, scaled to fit the same box a static
        solution plot would use."""
        self._clear_solution_movie()
        movie = QMovie(path)
        movie.setScaledSize(self.solution_label.size())
        self.solution_label.setMovie(movie)
        self._solution_movie = movie
        movie.start()

    def _reset_plot_headers(self):
        """Restore both output panels' headers/tooltips to their normal
        "Loss"/"Solution" captions and save-button hints. The Domain
        Preview feature (_preview_domain) temporarily relabels these same
        two panels ("Domain: Spatial (x,y)" / "Domain: Time (x,t)") since
        it reuses loss_label/solution_label rather than adding two more
        panels -- this is called wherever those panels go back to showing
        an actual Solve or Restore result, so a stale "Domain: ..." header
        doesn't linger over an unrelated loss/solution plot."""
        self.loss_header_label.setText("Loss")
        self.loss_save_btn.setToolTip("Save the loss plot currently shown")
        self.solution_header_label.setText("Solution")
        self.solution_save_btn.setToolTip("Save the solution plot currently shown")

    def _save_figure(self, label, default_basename):
        """Save whichever figure is currently shown in `label` (loss_label
        or solution_label after a Solve) to a location the user picks --
        just copies the already-generated source file, so a saved PNG/GIF
        is byte-identical to what's on screen."""
        src = getattr(label, '_source_path', None)
        if not src or not os.path.exists(src):
            self.log_box.append("⚠️ No figure to save yet — run Solve first.")
            return
        ext = os.path.splitext(src)[1] or ".png"
        default_name = f"{default_basename}{ext}"
        file_filter = "GIF Animation (*.gif)" if ext.lower() == ".gif" else "PNG Image (*.png)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Figure", default_name, file_filter)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ext
        try:
            import shutil as _fig_shutil
            _fig_shutil.copy(src, path)
            self.log_box.append(f"💾 Figure saved: {os.path.basename(path)}")
        except Exception as e:
            self.log_box.append(f"❌ Failed to save figure: {e}")

    def _display_run_result_plots(self, config, save_dir):
        """Loads and shows whichever loss/solution plot files a finished
        run actually wrote into `save_dir` (or /tmp, when blank), honoring
        config.plot_type to know whether to expect a static PNG or an
        animated GIF (see codegen.py's _sol_ext) -- exactly the plot-
        loading logic a normal single Solve already used inline in
        _on_done(), pulled out here so a finished Parameter Sweep's LAST
        completed run (see _on_sweep_finished) can show its own figure
        the same way, without duplicating this. `config` only needs to be
        whichever PINNConfig that run actually used (for plot_type);
        `save_dir` is wherever that run's own config.save_dir pointed --
        the Setup tab's own save location for a normal run, or that run's
        own per-run folder under the sweep's root for a sweep."""
        self._clear_solution_movie()

        # The two GIF animation plot types write "solution_plot.gif"
        # instead of "solution_plot.png" (see codegen.py's _sol_ext).
        _gif_types = ("Line Animation (GIF)", "Surface Animation (GIF)")
        _is_gif = config.plot_type in _gif_types
        _sol_ext = "gif" if _is_gif else "png"

        save_dir = (save_dir or "").strip()
        sol_dir = os.path.join(save_dir, "solution_results") if save_dir else "/tmp"
        loss_path = os.path.join(sol_dir, "loss_plot.png")
        solution_path = os.path.join(sol_dir, f"solution_plot.{_sol_ext}")
        # Fallback to root save dir for older runs
        if not os.path.exists(loss_path):
            loss_path = os.path.join(save_dir, "loss_plot.png") if save_dir else "/tmp/loss_plot.png"
        if not os.path.exists(solution_path):
            solution_path = os.path.join(save_dir, f"solution_plot.{_sol_ext}") if save_dir else f"/tmp/solution_plot.{_sol_ext}"

        if os.path.exists(loss_path):
            self.loss_label.setPixmap(QPixmap(loss_path).scaled(
                self.loss_label.width(), self.loss_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.loss_label._source_path = loss_path
        if os.path.exists(solution_path):
            if _is_gif:
                self._set_solution_gif(solution_path)
            else:
                self.solution_label.setPixmap(QPixmap(solution_path).scaled(
                    self.solution_label.width(), self.solution_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            self.solution_label._source_path = solution_path

        self._display_monitor_plots(save_dir)
        for _lbl in (self.loss_label, self.solution_label):
            if getattr(_lbl, "_fit", None) is not None:
                QTimer.singleShot(0, _lbl._fit.refit)

    def _display_monitor_plots(self, save_dir):
        """Finds every monitor_*_plot.png a finished run's
        _build_training_monitors_plot_code()-generated code wrote (see
        codegen.py) into `save_dir`'s solution_results/ (same directory
        convention as loss_plot.png/solution_plot.png above, including
        the same older-run root-folder fallback), and shows the first one
        -- or hides the whole Training Monitors box if none were
        configured/none wrote a plot (e.g. a run that was stopped before
        any monitor's first checkpoint)."""
        save_dir = (save_dir or "").strip()
        sol_dir = os.path.join(save_dir, "solution_results") if save_dir else "/tmp"
        search_dirs = [sol_dir]
        if save_dir and save_dir not in search_dirs:
            search_dirs.append(save_dir)

        found = []
        seen_names = set()
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if fname.startswith("monitor_") and fname.endswith("_plot.png"):
                    display_name = fname[len("monitor_"):-len("_plot.png")]
                    if display_name in seen_names:
                        continue
                    seen_names.add(display_name)
                    found.append((display_name, os.path.join(d, fname)))

        self.monitor_plot_files = found
        self.monitor_plot_idx = 0
        self.monitor_container.setVisible(bool(found))
        if found:
            self._show_monitor_plot(0)

    def _show_monitor_plot(self, idx):
        if not self.monitor_plot_files:
            return
        idx = idx % len(self.monitor_plot_files)
        self.monitor_plot_idx = idx
        name, path = self.monitor_plot_files[idx]
        if os.path.exists(path):
            self.monitor_label.setPixmap(QPixmap(path).scaled(
                self.monitor_label.width(), self.monitor_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.monitor_label._source_path = path
        multi = len(self.monitor_plot_files) > 1
        self.monitor_idx_label.setText(f"{idx + 1}/{len(self.monitor_plot_files)}: {name}" if multi else name)
        self.monitor_prev_btn.setVisible(multi)
        self.monitor_next_btn.setVisible(multi)

    def _on_done(self, result):
        self.solve_btn.setEnabled(True)
        self.solve_btn.setText(" Solve")
        self.stop_btn.setEnabled(False)
        if not getattr(self, "_stopped", False):   # Stop already summed it up
            _cfg = getattr(getattr(self, 'thread', None), 'config', None)
            self._finish_progress("Finished" if result == "DONE" else "Failed -- see the Training Log",
                                  ok=result == "DONE", folder=getattr(_cfg, "save_dir", "") or "")

        if result == "DONE":
            self.log_box.append("\n✅ Training complete!")
            # Reuse the EXACT config object SolverThread actually trained
            # with (self.thread.config), not a fresh self._build_config()
            # call -- _run_results_dir()/_timestamped_save_dir() stamp a
            # brand-new "<label>__<timestamp>" folder name every time
            # they're called, so rebuilding the config here would compute
            # a *different* timestamp than the one the subprocess actually
            # wrote its results into, pointing save_dir at a folder that
            # was never created. Likewise, self.save_dir_input.text() is
            # just the Setup tab's parent "Save to:" location (e.g.
            # PINNStudio_Results/), not this run's own unique subfolder --
            # reading from it (plus solution_results/'s "older runs"
            # fallback straight into that parent folder) is how a stale
            # plot left over from a previous, different run could get
            # displayed here instead of this run's own figure. The sweep
            # path (_on_sweep_finished, below) already gets this right by
            # tracking each run's own save_dir the same way.
            self._last_config = self.thread.config
            save_dir = self._last_config.save_dir
            self._display_run_result_plots(self._last_config, save_dir)

            if save_dir:
                self.log_box.append(f"💾 Results saved to: {save_dir}")

            if hasattr(self, '_ea_settings') and self._ea_settings:
                self.log_box.append("✅ Error analysis ran inline — check error_analysis/ folder.")
                # Deliberately NOT cleared here (unlike before): leaving it
                # in place means a later "Restore & Visualize" pass on a
                # checkpoint from this same run (e.g. one saved mid-
                # training via the Model Checkpoint callback) can still
                # compare against the same reference data, instead of
                # silently skipping Error Analysis there with no
                # indication why. _auto_configure_ea() already resets this
                # to a clean slate the moment a different template is
                # selected (see its own comment), so nothing goes stale
                # across templates by leaving it set here.
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
        self._register_hint(info, '#74c0fc')
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

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self._float_type = self._float_combo.currentText()
            self.log_box.append(f"✅ Float precision set to: {self._float_type}")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_gpu_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("GPU / Hardware")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        # Best-effort GPU detection for a helpful hint only -- never blocks
        # the dialog from opening if torch/CUDA isn't importable here.
        _hint_text = "Applies to both the in-app Solve run and any exported script."
        try:
            import torch as _torch_probe
            if _torch_probe.cuda.is_available():
                _n = _torch_probe.cuda.device_count()
                _names = ", ".join(
                    f"{i}: {_torch_probe.cuda.get_device_name(i)}" for i in range(_n)
                )
                _hint_text = f"Detected {_n} GPU(s) -- {_names}"
            else:
                _hint_text = "No CUDA GPU detected -- training will run on CPU regardless of these settings."
        except Exception:
            pass

        info = QLabel(_hint_text)
        self._register_hint(info, '#74c0fc')
        info.setWordWrap(True)
        layout.addWidget(info)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("GPU device index:"))
        dev_spin = QSpinBox()
        dev_spin.setRange(0, 31)
        dev_spin.setValue(getattr(self, '_gpu_device_index', 0))
        dev_spin.setFixedWidth(100)
        dev_row.addStretch(); dev_row.addWidget(dev_spin)
        layout.addLayout(dev_row)

        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Max GPU memory to reserve (%):"))
        mem_spin = QSpinBox()
        mem_spin.setRange(1, 100)
        mem_spin.setValue(round(getattr(self, '_gpu_memory_fraction', 0.95) * 100))
        mem_spin.setFixedWidth(100)
        mem_row.addStretch(); mem_row.addWidget(mem_spin)
        layout.addLayout(mem_row)

        note = QLabel("Ignored on machines with no GPU (CPU training is unaffected). "
                       "Device index only matters if you have more than one GPU.")
        self._register_hint(note, '#868e96')
        note.setWordWrap(True)
        layout.addWidget(note)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self._gpu_device_index = dev_spin.value()
            self._gpu_memory_fraction = mem_spin.value() / 100.0
            self.log_box.append(
                f"✅ GPU settings: device {self._gpu_device_index}, "
                f"{mem_spin.value()}% memory reserved"
            )
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_seed_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Seed")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        info = QLabel(
            "Seeds NumPy, PyTorch, and point sampling together so a run's "
            "collocation points and network initialization are the same "
            "every time. Doesn't force GPU runs to be bit-for-bit identical "
            "(that costs training speed) -- results will be consistent, "
            "not necessarily byte-identical, on GPU."
        )
        self._register_hint(info, '#74c0fc')
        info.setWordWrap(True)
        layout.addWidget(info)

        use_cb = QCheckBox("Fix random seed for reproducible runs")
        use_cb.setChecked(getattr(self, '_use_random_seed', True))
        layout.addWidget(use_cb)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        seed_spin = QSpinBox()
        seed_spin.setRange(0, 2_147_483_647)
        seed_spin.setValue(getattr(self, '_random_seed', 2026))
        seed_spin.setFixedWidth(120)
        seed_row.addStretch(); seed_row.addWidget(seed_spin)
        layout.addLayout(seed_row)

        def _sync_seed_enabled(checked):
            seed_spin.setEnabled(checked)
        seed_spin.setEnabled(use_cb.isChecked())
        use_cb.toggled.connect(_sync_seed_enabled)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self._use_random_seed = use_cb.isChecked()
            self._random_seed = seed_spin.value()
            if self._use_random_seed:
                self.log_box.append(f"✅ Random seed fixed at {self._random_seed}")
            else:
                self.log_box.append("✅ Random seed: off (runs will vary each time)")
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _on_lbfgs_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("L-BFGS Settings")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        use_default = QCheckBox("Use recommended L-BFGS settings")
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

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

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
        self._register_hint(note, '#586e75')
        layout.addWidget(note)

        wd_row = QHBoxLayout()
        wd_row.addWidget(QLabel("Weight decay (L2):"))
        wd_spin = SciLineEdit(self._optimizer_settings.get("weight_decay", 0.0))
        wd_spin.setFixedWidth(130)
        wd_row.addStretch(); wd_row.addWidget(wd_spin)
        layout.addLayout(wd_row)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self._optimizer_settings["weight_decay"] = wd_spin.value()
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()


    def _on_view_domain_changed(self, state):
        if state == 2:
            self._preview_domain()
        else:
            self._reset_plot_headers()
            self.loss_label.setText("📉 Loss plot")
            self.solution_label.setText("🗺 Solution plot")
            # Un-checking the box means "nothing shown here" -- without
            # this, Save Figure on either panel would silently re-save the
            # stale domain-preview PNG from before the checkbox was
            # unchecked, even though the panel now just shows placeholder
            # text (setText() clears the pixmap but not this attribute).
            self.loss_label._source_path = None
            self.solution_label._source_path = None

    # Domain Preview ("View domain & point distribution") shared styling.
    # Previously each panel used a dark theme (#1e1e1e figure / #252526
    # axes, ~10-11pt fonts) that was visually inconsistent with every
    # other plot the app generates (loss/solution plots are plain
    # matplotlib-default white, no special styling) and, per user
    # feedback, just didn't look good -- too dark, legend/axis text too
    # small to read comfortably. This is a clean white/light theme with
    # noticeably larger fonts, a light grid, and higher-DPI output,
    # matching the professional look of the rest of the app's plots. Kept
    # as ONE shared constant (rather than duplicated inline in both the
    # 2D and 3D preview builders below) so both panel types always look
    # identical and any future tweak only has to be made once.
    _DOMAIN_PREVIEW_DPI = 150
    # Point marker sizes -- noticeably bigger than the old dark-theme
    # defaults (which were tuned for small, dim dots against a busy dark
    # background) since these now sit on a clean white background with a
    # visible light grid, so slightly larger, crisper points read better.
    _DOMAIN_PT_SIZE = 14        # spatial panel: domain points
    _BND_PT_SIZE = 40           # spatial panel: boundary points
    _TIME_DOM_PT_SIZE = 10      # time panel: domain points
    _TIME_BND_PT_SIZE = 26      # time panel: boundary/IC points
    # Fixed paths -- overwritten by each new preview (2D or 3D, whichever
    # geometry is currently selected; the two are mutually exclusive) and
    # read back by _on_preview_done, which also points loss_label's/
    # solution_label's _source_path at them so the existing "💾 Save
    # Figure" buttons work for these too, saving exactly what's on screen.
    _DOMAIN_PREVIEW_SPATIAL_PATH = "/tmp/domain_preview_spatial.png"
    _DOMAIN_PREVIEW_TIME_PATH = "/tmp/domain_preview_time.png"
    # Plain string (NOT an f-string) -- spliced verbatim into each
    # generated preview script via an outer f-string, so its own `{`/`}`
    # characters must stay literal, not get double-evaluated.
    _DOMAIN_PREVIEW_STYLE_SETUP = """\
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 100

DOM_COLOR = '#4dabf7'   # same blue used for Train loss elsewhere in the app
BND_COLOR = '#e03131'   # deeper red -- reads clearly against white
IC_COLOR  = '#2f9e44'
TITLE_FS  = 15
LABEL_FS  = 13
TICK_FS   = 11
LEGEND_FS = 11.5

def _style_axes(ax):
    ax.set_facecolor('#fbfbfc')
    ax.tick_params(labelsize=TICK_FS, colors='#333333')
    for _sp in ax.spines.values():
        _sp.set_color('#999999')
        _sp.set_linewidth(0.8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, color='#adb5bd')
    ax.set_axisbelow(True)

def _style_legend(ax):
    _leg = ax.legend(fontsize=LEGEND_FS, loc='best', framealpha=0.95,
                      facecolor='white', edgecolor='#ced4da', markerscale=1.8)
    if _leg is not None:
        _leg.get_frame().set_linewidth(0.8)
"""

    @staticmethod
    def _preview_single_geom_code(geom_type, **vals):
        """cleanup Round 48: the dde.geometry constructor for a single-shape
        Domain Preview, built by the same codegen.py function the exported
        script trains on (_clean_geom_inner, behind _clean_geom_line) from
        the preview's own widget values -- previously each preview branch
        hand-typed its own copy of that constructor."""
        import types as _types
        from pinnstudio.core.codegen import _clean_geom_inner
        ns = _types.SimpleNamespace(
            geometry_type=geom_type,
            x_min=vals.get("x_min", 0.0), x_max=vals.get("x_max", 1.0),
            y_min=vals.get("y_min", 0.0), y_max=vals.get("y_max", 1.0),
            z_min=vals.get("z_min", 0.0), z_max=vals.get("z_max", 1.0),
            geom_center_x=vals.get("cx", 0.5), geom_center_y=vals.get("cy", 0.5),
            geom_center_z=vals.get("cz", 0.5), geom_radius=vals.get("r", 0.5),
            geom_semi_major=vals.get("a", 0.5), geom_semi_minor=vals.get("b", 0.3),
            geom_angle=vals.get("angle", 0.0),
        )
        is_3d = geom_type in ("Cuboid", "Sphere")
        verts = vals.get("verts") or []
        inner, _wrap = _clean_geom_inner(ns, not is_3d, is_3d, verts, verts)
        return inner

    def _custom_geom_leaf_code(self, shape_type, params):
        """One leaf shape's own dde.geometry constructor call + matplotlib
        outline patch + bounding box, from a params dict in the schema
        _custom_geom_row_to_dict() produces. Used only by the Custom-
        geometry branch of _preview_domain() below -- the single-shape-
        type branches there read straight from their own top-level
        widgets and are left as they are, so this doesn't touch any
        already-working preview path. Raises ValueError (message meant
        to be shown to the user) for a Triangle/Polygon whose vertices
        DeepXDE's own constructors would reject, mirroring the same
        checks the Triangle/Polygon branch below already makes for the
        single-shape case."""
        import math as _math
        if shape_type == "Rectangle":
            x_min, x_max = params["x_min"], params["x_max"]
            y_min, y_max = params["y_min"], params["y_max"]
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("needs x_min < x_max and y_min < y_max.")
            patch_code = (
                f"plt.Rectangle(({x_min},{y_min}), {x_max}-{x_min}, {y_max}-{y_min}, "
                "linewidth=2, edgecolor='#1971c2', facecolor='none')"
            )
            bbox = (x_min, x_max, y_min, y_max)
        elif shape_type == "Disk":
            cx, cy, r = params["cx"], params["cy"], params["r"]
            patch_code = f"plt.Circle(({cx},{cy}), {r}, linewidth=2, edgecolor='#1971c2', facecolor='none')"
            bbox = (cx - r, cx + r, cy - r, cy + r)
        elif shape_type == "Ellipse":
            cx, cy = params["cx"], params["cy"]
            a, b, angle = params["a"], params["b"], params["angle"]
            patch_code = (
                f"matplotlib.patches.Ellipse(({cx},{cy}), {2*a}, {2*b}, angle={_math.degrees(angle)}, "
                "linewidth=2, edgecolor='#1971c2', facecolor='none')"
            )
            dx = _math.sqrt((a * _math.cos(angle)) ** 2 + (b * _math.sin(angle)) ** 2)
            dy = _math.sqrt((a * _math.sin(angle)) ** 2 + (b * _math.cos(angle)) ** 2)
            bbox = (cx - dx, cx + dx, cy - dy, cy + dy)
        else:  # Triangle / Polygon
            verts = self._parse_vertices(params.get("vertices_text", ""))
            if len(verts) < 3 or (shape_type == "Triangle" and len(verts) != 3):
                raise ValueError(
                    f"needs {'exactly 3' if shape_type == 'Triangle' else 'at least 3'} valid vertices."
                )
            if shape_type == "Polygon" and len(verts) == 3:
                raise ValueError(
                    "has only 3 vertices -- that's a Triangle, not a Polygon; "
                    "change its shape type, or add a 4th vertex."
                )
            if shape_type == "Polygon" and self._is_axis_aligned_rectangle(verts):
                raise ValueError(
                    "those 4 vertices form an axis-aligned rectangle -- use the "
                    "Rectangle shape type instead."
                )
            vlist = [list(v) for v in verts]
            patch_code = f"plt.Polygon({vlist}, closed=True, linewidth=2, edgecolor='#1971c2', facecolor='none')"
            xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
            bbox = (min(xs), max(xs), min(ys), max(ys))
        # cleanup Round 48: the constructor itself comes from the same
        # codegen.py function training uses for a Custom leaf, so the preview
        # can never sample a different shape than the one that gets trained.
        # The branches above only validate and build the outline/bbox.
        from pinnstudio.core.codegen import _custom_geom_leaf_literal
        geom_code = _custom_geom_leaf_literal(shape_type, params)
        return geom_code, patch_code, bbox

    def _custom_geom_3d_leaf_code(self, idx, shape_type, params):
        """3D counterpart of _custom_geom_leaf_code() -- one leaf shape's
        dde.geometry constructor call + a 3D outline-drawing code block
        (box edges for Cuboid, a wireframe sphere for Sphere, same style
        as _build_3d_preview_script's own single-shape outlines) + its
        bounding box. idx makes every local variable name in the outline
        code unique (_corners_3, _sx_3, ...) so several leaves' outline
        blocks can be concatenated into one script without colliding,
        since -- unlike the 2D preview, where each leaf is a self-
        contained matplotlib Patch object -- a 3D outline here is a few
        lines of plotting code run directly against the shared `ax`."""
        params = params or {}
        if shape_type == "Cuboid":
            x_min, x_max = params.get("x_min", 0.0), params.get("x_max", 1.0)
            y_min, y_max = params.get("y_min", 0.0), params.get("y_max", 1.0)
            z_min, z_max = params.get("z_min", 0.0), params.get("z_max", 1.0)
            if x_min >= x_max or y_min >= y_max or z_min >= z_max:
                raise ValueError("needs x_min<x_max, y_min<y_max, and z_min<z_max.")
            outline_code = (
                f"_x0_{idx}, _x1_{idx}, _y0_{idx}, _y1_{idx}, _z0_{idx}, _z1_{idx} = "
                f"{x_min}, {x_max}, {y_min}, {y_max}, {z_min}, {z_max}\n"
                f"_corners_{idx} = [(_x0_{idx},_y0_{idx},_z0_{idx}),(_x1_{idx},_y0_{idx},_z0_{idx}),"
                f"(_x1_{idx},_y1_{idx},_z0_{idx}),(_x0_{idx},_y1_{idx},_z0_{idx}),\n"
                f"               (_x0_{idx},_y0_{idx},_z1_{idx}),(_x1_{idx},_y0_{idx},_z1_{idx}),"
                f"(_x1_{idx},_y1_{idx},_z1_{idx}),(_x0_{idx},_y1_{idx},_z1_{idx})]\n"
                f"_edges_{idx} = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]\n"
                f"for _i, _j in _edges_{idx}:\n"
                f"    _p0, _p1 = _corners_{idx}[_i], _corners_{idx}[_j]\n"
                "    ax.plot([_p0[0],_p1[0]], [_p0[1],_p1[1]], [_p0[2],_p1[2]], color='#1971c2', linewidth=2.0)\n"
            )
            bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
        elif shape_type == "Sphere":
            cx, cy, cz = params.get("cx", 0.5), params.get("cy", 0.5), params.get("cz", 0.5)
            r = params.get("r", 0.5)
            outline_code = (
                f"_u_{idx} = np.linspace(0, 2*np.pi, 30)\n"
                f"_v_{idx} = np.linspace(0, np.pi, 20)\n"
                f"_sx_{idx} = {cx} + {r}*np.outer(np.cos(_u_{idx}), np.sin(_v_{idx}))\n"
                f"_sy_{idx} = {cy} + {r}*np.outer(np.sin(_u_{idx}), np.sin(_v_{idx}))\n"
                f"_sz_{idx} = {cz} + {r}*np.outer(np.ones_like(_u_{idx}), np.cos(_v_{idx}))\n"
                f"ax.plot_wireframe(_sx_{idx}, _sy_{idx}, _sz_{idx}, color='#1971c2', "
                "linewidth=0.7, alpha=0.5, rstride=2, cstride=2)\n"
            )
            bbox = (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)
        else:
            raise ValueError(f"unsupported 3D shape type: {shape_type!r}")
        # cleanup Round 48: same single source for the constructor as the 2D
        # leaf above -- codegen.py's _custom_geom_leaf_literal.
        from pinnstudio.core.codegen import _custom_geom_leaf_literal
        geom_code = _custom_geom_leaf_literal(shape_type, params)
        return geom_code, outline_code, bbox

    def _preview_domain(self):
        import tempfile, math
        spatial_path = self._DOMAIN_PREVIEW_SPATIAL_PATH
        time_path = self._DOMAIN_PREVIEW_TIME_PATH
        _DOMAIN_PREVIEW_STYLE_SETUP = self._DOMAIN_PREVIEW_STYLE_SETUP
        _DOMAIN_PREVIEW_DPI = self._DOMAIN_PREVIEW_DPI
        _DOMAIN_PT_SIZE = self._DOMAIN_PT_SIZE
        _BND_PT_SIZE = self._BND_PT_SIZE
        _TIME_DOM_PT_SIZE = self._TIME_DOM_PT_SIZE
        _TIME_BND_PT_SIZE = self._TIME_BND_PT_SIZE
        geom_type = self._current_geometry_type()
        if geom_type not in ("Rectangle", "Disk", "Ellipse", "Triangle", "Polygon", "Cuboid", "Sphere", "Custom"):
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

        # Custom geometry is 2D or 3D depending on which primitives it's
        # built from -- since the shape list is cleared on every dimension
        # switch (see _on_dim_changed) and each row's type combo only ever
        # offers the current dimension's own primitives, the problem's own
        # radio_3d state is a reliable proxy for which Custom this is.
        is_3d_shape = geom_type in ("Cuboid", "Sphere") or (geom_type == "Custom" and self.radio_3d.isChecked())
        # Read by _on_preview_done to label the spatial panel's header
        # correctly ("(x,y,z)" vs "(x,y)") once the background script
        # finishes -- it has no other way to know which builder ran.
        self._last_preview_is_3d = is_3d_shape
        if is_3d_shape:
            if geom_type == "Custom":
                if not self.custom_geom_shape_rows:
                    self.log_box.append("⚠️ Add at least one shape to preview a Custom domain.")
                    self.loss_label.setText("📉 Loss plot")
                    self.solution_label.setText("🗺 Solution plot")
                    return
                entries = [
                    self._custom_geom_row_to_dict(row, is_first=(i == 0))
                    for i, row in enumerate(self.custom_geom_shape_rows)
                ]
                from pinnstudio.core.codegen import _CUSTOM_GEOM_OP_CTORS as _CSG_OP_CTORS
                geom_code = None
                outline_blocks = []
                bboxes = []
                for i, entry in enumerate(entries, start=1):
                    try:
                        leaf_code, leaf_outline, leaf_bbox = self._custom_geom_3d_leaf_code(i, entry["type"], entry["params"])
                    except ValueError as e:
                        self.log_box.append(f"⚠️ Shape {i} ({entry['type']}) {e}")
                        self.loss_label.setText("📉 Loss plot")
                        self.solution_label.setText("🗺 Solution plot")
                        return
                    outline_blocks.append(leaf_outline)
                    bboxes.append(leaf_bbox)
                    if geom_code is None:
                        geom_code = leaf_code
                    else:
                        ctor = _CSG_OP_CTORS.get(entry.get("op") or "union", "CSGUnion")
                        geom_code = f"dde.geometry.{ctor}({geom_code}, {leaf_code})"
                bbox3 = (
                    min(b[0] for b in bboxes), max(b[1] for b in bboxes),
                    min(b[2] for b in bboxes), max(b[3] for b in bboxes),
                    min(b[4] for b in bboxes), max(b[5] for b in bboxes),
                )
                outline_code = "\n".join(outline_blocks)
                script = self._render_3d_preview_script(
                    "Custom", geom_code, outline_code, bbox3,
                    t_min, t_max, n_domain, n_boundary, n_initial, dist)
            else:
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
        if geom_type == "Custom":
            if not self.custom_geom_shape_rows:
                self.log_box.append("⚠️ Add at least one shape to preview a Custom domain.")
                self.loss_label.setText("📉 Loss plot")
                self.solution_label.setText("🗺 Solution plot")
                return
            entries = [
                self._custom_geom_row_to_dict(row, is_first=(i == 0))
                for i, row in enumerate(self.custom_geom_shape_rows)
            ]
            # No single matplotlib Patch can represent an arbitrary CSG
            # result, so each leaf shape's own reference outline is drawn
            # overlaid instead -- correctness of the point cloud itself
            # comes entirely from on_boundary() below (shape-agnostic,
            # works for any CSG combination with zero changes), so these
            # patches are a visual aid only, not what's actually sampled.
            patch_codes = []
            bboxes = []
            from pinnstudio.core.codegen import _CUSTOM_GEOM_OP_CTORS as _CSG_OP_CTORS
            geom_code = None
            for i, entry in enumerate(entries, start=1):
                try:
                    leaf_code, leaf_patch, leaf_bbox = self._custom_geom_leaf_code(entry["type"], entry["params"])
                except ValueError as e:
                    self.log_box.append(f"⚠️ Shape {i} ({entry['type']}) {e}")
                    self.loss_label.setText("📉 Loss plot")
                    self.solution_label.setText("🗺 Solution plot")
                    return
                patch_codes.append(leaf_patch)
                bboxes.append(leaf_bbox)
                if geom_code is None:
                    geom_code = leaf_code
                else:
                    ctor = _CSG_OP_CTORS.get(entry.get("op") or "union", "CSGUnion")
                    geom_code = f"dde.geometry.{ctor}({geom_code}, {leaf_code})"
            bbox = (
                min(b[0] for b in bboxes), max(b[1] for b in bboxes),
                min(b[2] for b in bboxes), max(b[3] for b in bboxes),
            )
        elif geom_type == "Rectangle":
            x_min = self.x_min.value(); x_max = self.x_max.value()
            y_min = self.y_min.value(); y_max = self.y_max.value()
            geom_code = self._preview_single_geom_code(
                "Rectangle", x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            patch_code = (
                f"plt.Rectangle(({x_min},{y_min}), {x_max}-{x_min}, {y_max}-{y_min}, "
                "linewidth=2, edgecolor='#1971c2', facecolor='none')"
            )
            bbox = (x_min, x_max, y_min, y_max)
        elif geom_type == "Disk":
            cx = self.geom_disk_cx.value(); cy = self.geom_disk_cy.value(); r = self.geom_disk_r.value()
            geom_code = self._preview_single_geom_code("Disk", cx=cx, cy=cy, r=r)
            patch_code = f"plt.Circle(({cx},{cy}), {r}, linewidth=2, edgecolor='#1971c2', facecolor='none')"
            bbox = (cx - r, cx + r, cy - r, cy + r)
        elif geom_type == "Ellipse":
            cx = self.geom_ellipse_cx.value(); cy = self.geom_ellipse_cy.value()
            a = self.geom_ellipse_a.value(); b = self.geom_ellipse_b.value()
            angle = self.geom_ellipse_angle.value()
            geom_code = self._preview_single_geom_code("Ellipse", cx=cx, cy=cy, a=a, b=b, angle=angle)
            patch_code = (
                f"matplotlib.patches.Ellipse(({cx},{cy}), {2*a}, {2*b}, angle={math.degrees(angle)}, "
                "linewidth=2, edgecolor='#1971c2', facecolor='none')"
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
            geom_code = self._preview_single_geom_code(geom_type, verts=vlist)
            patch_code = f"plt.Polygon({vlist}, closed=True, linewidth=2, edgecolor='#1971c2', facecolor='none')"
            xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
            bbox = (min(xs), max(xs), min(ys), max(ys))

        if geom_type != "Custom":
            patch_codes = [patch_code]

        bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox
        pad_x = max((bbox_x_max - bbox_x_min) * 0.05, 1e-6)
        pad_y = max((bbox_y_max - bbox_y_min) * 0.05, 1e-6)
        xlim_lo, xlim_hi = bbox_x_min - pad_x, bbox_x_max + pad_x
        ylim_lo, ylim_hi = bbox_y_min - pad_y, bbox_y_max + pad_y
        _patch_codes_src = ",\n    ".join(patch_codes)

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

{_DOMAIN_PREVIEW_STYLE_SETUP}

# Classify points using the geometry's own on_boundary() test -- this works
# for any 2D shape (circle, ellipse, triangle, polygon, ...), not just an
# axis-aligned box.
ic_mask  = pts[:, 2] <= {t_min} + tol
bnd_mask = gt.geometry.on_boundary(pts[:, :2]) & ~ic_mask
dom_mask = ~ic_mask & ~bnd_mask
dom_pts = pts[dom_mask]
bnd_pts = pts[bnd_mask]
ic_pts  = pts[ic_mask]

# ── Spatial (x,y) panel -- its own standalone figure/file ──────
# Domain + boundary only. IC points are t=t_min points that live inside
# the domain, so overlaying them here just adds clutter on top of the
# domain cloud; they're already shown clearly in the time panel below.
fig, ax = plt.subplots(figsize=(7, 6.2))
_style_axes(ax)
ax.set_xlim({xlim_lo}, {xlim_hi})
ax.set_ylim({ylim_lo}, {ylim_hi})
shape_patches = [
    {_patch_codes_src}
]
for _sp in shape_patches:
    ax.add_patch(_sp)
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,1], s={_DOMAIN_PT_SIZE}, c=DOM_COLOR, alpha=0.75, edgecolors='none', label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,1], s={_BND_PT_SIZE}, c=BND_COLOR, alpha=1.0, edgecolors='white', linewidths=0.4, label=f'Boundary ({{len(bnd_pts)}})')
ax.set_xlabel('x', fontsize=LABEL_FS)
ax.set_ylabel('y', fontsize=LABEL_FS)
ax.set_aspect('equal', adjustable='box')
ax.set_title('{geom_type}  |  {dist}  |  D={n_domain}  B={n_boundary}  IC={n_initial}', fontsize=TITLE_FS, fontweight='bold', pad=12)
_style_legend(ax)
plt.tight_layout()
plt.savefig({spatial_path!r}, dpi={_DOMAIN_PREVIEW_DPI}, bbox_inches='tight')
plt.close(fig)

# ── Time distribution (x vs t) panel -- its own standalone figure ──
fig, ax = plt.subplots(figsize=(7, 6.2))
_style_axes(ax)
ax.set_xlim({xlim_lo}, {xlim_hi}); ax.set_ylim({t_min}, {t_max})
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,2], s={_TIME_DOM_PT_SIZE}, c=DOM_COLOR, alpha=0.6, edgecolors='none', label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,2], s={_TIME_BND_PT_SIZE}, c=BND_COLOR, alpha=0.9, edgecolors='white', linewidths=0.3, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax.scatter(ic_pts[:,0],  ic_pts[:,2],  s={_TIME_BND_PT_SIZE}, c=IC_COLOR, alpha=1.0, edgecolors='white', linewidths=0.3, label=f'IC ({{len(ic_pts)}})')
ax.set_xlabel('x', fontsize=LABEL_FS)
ax.set_ylabel('t', fontsize=LABEL_FS)
ax.set_title(f'Time Distribution (x vs t) — total={{len(pts)}} points', fontsize=TITLE_FS, fontweight='bold', pad=12)
_style_legend(ax)
plt.tight_layout()
plt.savefig({time_path!r}, dpi={_DOMAIN_PREVIEW_DPI}, bbox_inches='tight')
plt.close(fig)
print("DOMAIN_PREVIEW_DONE")
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script)
            tmp = tf.name
        self._launch_preview_thread(tmp)

    def _launch_preview_thread(self, tmp):
        """Run a generated domain-preview script (2D or 3D) in a background
        thread and stream its output into the log box, same as before this
        was factored out -- shared by both the 2D and 3D preview paths.

        Point Distribution / num_domain / num_boundary / num_initial changes
        (_on_dist_changed / _on_pts_changed) call this every time their
        value changes -- which can fire many times in a row while a spinbox
        is being dragged or scrolled. Each call spawns a new background
        subprocess that writes to the SAME two fixed-path PNG files
        (_DOMAIN_PREVIEW_SPATIAL_PATH / _DOMAIN_PREVIEW_TIME_PATH). Without
        cancelling a still-running previous preview first, two of these
        subprocesses' writes to those files can overlap, producing a
        truncated/corrupted PNG -- which is exactly the "libpng error:
        IDAT: CRC error" / "QPixmap::scaled: Pixmap is a null pixmap"
        failure this guards against. Only one preview subprocess is ever
        allowed to be writing to those files at a time."""
        old_thread = getattr(self, '_preview_thread', None)
        if old_thread is not None and old_thread.isRunning():
            try:
                old_thread.done_sig.disconnect()
            except Exception:
                pass
            old_thread.stop()
            old_thread.wait(3000)

        self.loss_label.setText("⏳ Generating preview...")

        from PyQt6.QtCore import QThread, pyqtSignal as _sig

        class _PreviewThread(QThread):
            done_sig = _sig(bool)
            log_sig  = _sig(str)
            def __init__(self, tmp):
                super().__init__(); self._tmp = tmp
                self.process = None
            def run(self):
                import subprocess, sys
                proc = subprocess.Popen([sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.process = proc
                for line in proc.stdout:
                    self.log_sig.emit(line.rstrip())
                proc.wait()
                os.unlink(self._tmp)
                self.done_sig.emit(proc.returncode == 0)
            def stop(self):
                # Same fix as SolverThread/_RestoreThread/_ParamRestoreThread
                # -- see closeEvent. terminate()-then-kill() with a bounded
                # wait (see SolverThread.stop() for the full rationale) --
                # not just terminate() alone, which can hang indefinitely.
                if self.process and self.process.poll() is None:
                    import subprocess
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()

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
            geom_code = self._preview_single_geom_code(
                "Cuboid", x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
            bbox3 = (x_min, x_max, y_min, y_max, z_min, z_max)
            outline_code = (
                f"_x0, _x1, _y0, _y1, _z0, _z1 = {x_min}, {x_max}, {y_min}, {y_max}, {z_min}, {z_max}\n"
                "_corners = [(_x0,_y0,_z0),(_x1,_y0,_z0),(_x1,_y1,_z0),(_x0,_y1,_z0),\n"
                "            (_x0,_y0,_z1),(_x1,_y0,_z1),(_x1,_y1,_z1),(_x0,_y1,_z1)]\n"
                "_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]\n"
                "for _i, _j in _edges:\n"
                "    _p0, _p1 = _corners[_i], _corners[_j]\n"
                "    ax.plot([_p0[0],_p1[0]], [_p0[1],_p1[1]], [_p0[2],_p1[2]], color='#1971c2', linewidth=2.0)\n"
            )
        else:  # Sphere
            cx = self.geom_sphere_cx.value(); cy = self.geom_sphere_cy.value()
            cz = self.geom_sphere_cz.value(); r = self.geom_sphere_r.value()
            geom_code = self._preview_single_geom_code("Sphere", cx=cx, cy=cy, cz=cz, r=r)
            bbox3 = (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)
            outline_code = (
                "_u = np.linspace(0, 2*np.pi, 30)\n"
                "_v = np.linspace(0, np.pi, 20)\n"
                f"_sx = {cx} + {r}*np.outer(np.cos(_u), np.sin(_v))\n"
                f"_sy = {cy} + {r}*np.outer(np.sin(_u), np.sin(_v))\n"
                f"_sz = {cz} + {r}*np.outer(np.ones_like(_u), np.cos(_v))\n"
                "ax.plot_wireframe(_sx, _sy, _sz, color='#1971c2', linewidth=0.7, alpha=0.5, rstride=2, cstride=2)\n"
            )

        return self._render_3d_preview_script(
            geom_type, geom_code, outline_code, bbox3,
            t_min, t_max, n_domain, n_boundary, n_initial, dist)

    def _render_3d_preview_script(self, geom_type_label, geom_code, outline_code, bbox3,
                                   t_min, t_max, n_domain, n_boundary, n_initial, dist):
        """The actual 3D preview script template -- factored out of
        _build_3d_preview_script() so the Custom-geometry branch of
        _preview_domain() below can reuse it directly with a CSG-chained
        geom_code and a concatenation of each leaf shape's own outline
        code, instead of duplicating this ~60-line template. geom_code/
        outline_code/bbox3 are pre-built by the caller; geom_type_label is
        only used for the plot title."""
        bx0, bx1, by0, by1, bz0, bz1 = bbox3
        pad_x = max((bx1 - bx0) * 0.08, 1e-6)
        pad_y = max((by1 - by0) * 0.08, 1e-6)
        pad_z = max((bz1 - bz0) * 0.08, 1e-6)
        xlim_lo, xlim_hi = bx0 - pad_x, bx1 + pad_x
        ylim_lo, ylim_hi = by0 - pad_y, by1 + pad_y
        zlim_lo, zlim_hi = bz0 - pad_z, bz1 + pad_z

        spatial_path = self._DOMAIN_PREVIEW_SPATIAL_PATH
        time_path = self._DOMAIN_PREVIEW_TIME_PATH
        _DOMAIN_PREVIEW_STYLE_SETUP = self._DOMAIN_PREVIEW_STYLE_SETUP
        _DOMAIN_PREVIEW_DPI = self._DOMAIN_PREVIEW_DPI
        _TIME_DOM_PT_SIZE = self._TIME_DOM_PT_SIZE
        _TIME_BND_PT_SIZE = self._TIME_BND_PT_SIZE

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

{_DOMAIN_PREVIEW_STYLE_SETUP}

# Classify points using the geometry's own on_boundary() test -- same
# generalization as the 2D preview, just over (x,y,z) instead of (x,y).
ic_mask  = pts[:, 3] <= {t_min} + tol
bnd_mask = gt.geometry.on_boundary(pts[:, :3]) & ~ic_mask
dom_mask = ~ic_mask & ~bnd_mask
dom_pts = pts[dom_mask]
bnd_pts = pts[bnd_mask]
ic_pts  = pts[ic_mask]

# ── Spatial (3D) panel -- its own standalone figure/file ───────
# Domain + boundary only, same reasoning as the 2D preview: IC points are
# already shown clearly in the time panel below.
fig = plt.figure(figsize=(7.2, 6.6))
ax  = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_facecolor('#fbfbfc')
ax.tick_params(labelsize=TICK_FS, colors='#333333')
_pane = (0.98, 0.98, 0.99, 1.0)
ax.xaxis.set_pane_color(_pane); ax.yaxis.set_pane_color(_pane); ax.zaxis.set_pane_color(_pane)
ax.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)
ax.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)
ax.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)

ax.set_xlim({xlim_lo}, {xlim_hi})
ax.set_ylim({ylim_lo}, {ylim_hi})
ax.set_zlim({zlim_lo}, {zlim_hi})

# Shape outline -- box edges for Cuboid, a wireframe sphere for Sphere.
{outline_code}
if len(dom_pts): ax.scatter(dom_pts[:,0], dom_pts[:,1], dom_pts[:,2], s=10, c=DOM_COLOR, alpha=0.55, edgecolors='none', label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax.scatter(bnd_pts[:,0], bnd_pts[:,1], bnd_pts[:,2], s=26, c=BND_COLOR, alpha=1.0, edgecolors='white', linewidths=0.4, label=f'Boundary ({{len(bnd_pts)}})')
ax.set_xlabel('x', fontsize=LABEL_FS, labelpad=10)
ax.set_ylabel('y', fontsize=LABEL_FS, labelpad=10)
ax.set_zlabel('z', fontsize=LABEL_FS, labelpad=6)
ax.set_title('{geom_type_label}  |  {dist}  |  D={n_domain}  B={n_boundary}  IC={n_initial}', fontsize=TITLE_FS, fontweight='bold', pad=16)
_leg3d = ax.legend(fontsize=LEGEND_FS, loc='upper left', framealpha=0.95, facecolor='white', edgecolor='#ced4da', markerscale=1.8)
if _leg3d is not None: _leg3d.get_frame().set_linewidth(0.8)
plt.tight_layout()
# pad_inches=0.4 (vs. the default/other panels' plain bbox_inches='tight')
# -- mplot3d's set_zlabel() sits further outside the axes than Matplotlib's
# tight-bbox calculation accounts for, so without extra padding here the
# 'z' axis label gets silently cropped off the right edge of the saved
# PNG. Confirmed by rendering this panel and inspecting the actual output
# file before landing this fix -- the label really was being cut off, not
# just theoretically at risk of it.
plt.savefig({spatial_path!r}, dpi={_DOMAIN_PREVIEW_DPI}, bbox_inches='tight', pad_inches=0.4)
plt.close(fig)

# ── Time distribution (x vs t) panel -- its own standalone figure ──
# Same layout as the 2D preview's time panel, just reading column 3 (t)
# instead of column 2.
fig, ax2 = plt.subplots(figsize=(7, 6.2))
_style_axes(ax2)
ax2.set_xlim({xlim_lo}, {xlim_hi}); ax2.set_ylim({t_min}, {t_max})
if len(dom_pts): ax2.scatter(dom_pts[:,0], dom_pts[:,3], s={_TIME_DOM_PT_SIZE}, c=DOM_COLOR, alpha=0.6, edgecolors='none', label=f'Domain ({{len(dom_pts)}})')
if len(bnd_pts): ax2.scatter(bnd_pts[:,0], bnd_pts[:,3], s={_TIME_BND_PT_SIZE}, c=BND_COLOR, alpha=0.9, edgecolors='white', linewidths=0.3, label=f'Boundary ({{len(bnd_pts)}})')
if len(ic_pts):  ax2.scatter(ic_pts[:,0],  ic_pts[:,3],  s={_TIME_BND_PT_SIZE}, c=IC_COLOR, alpha=1.0, edgecolors='white', linewidths=0.3, label=f'IC ({{len(ic_pts)}})')
ax2.set_xlabel('x', fontsize=LABEL_FS)
ax2.set_ylabel('t', fontsize=LABEL_FS)
ax2.set_title(f'Time Distribution (x vs t) — total={{len(pts)}} points', fontsize=TITLE_FS, fontweight='bold', pad=12)
_style_legend(ax2)
plt.tight_layout()
plt.savefig({time_path!r}, dpi={_DOMAIN_PREVIEW_DPI}, bbox_inches='tight')
plt.close(fig)
print("DOMAIN_PREVIEW_DONE")
"""

    def _on_dist_changed(self, text):
        if self.view_domain_check.isChecked():
            self._preview_domain()

    def _on_pts_changed(self, val):
        if self.view_domain_check.isChecked():
            self._preview_domain()
    
    def _on_preview_done(self, success):
        spatial_path = self._DOMAIN_PREVIEW_SPATIAL_PATH
        time_path = self._DOMAIN_PREVIEW_TIME_PATH
        if success and os.path.exists(spatial_path) and os.path.exists(time_path):
            # Two independently-rendered files now (see _preview_domain /
            # _build_3d_preview_script) instead of one wide combined image
            # pixel-cropped in half -- each panel gets its own clean,
            # correctly-bbox'd figure, and each has a real file on disk to
            # point _source_path at, so the existing "💾 Save Figure"
            # buttons on these same two panels work for Domain Preview
            # too (previously they'd either show the misleading "No
            # figure to save yet" message or, worse, silently re-save
            # whatever unrelated Solve/Restore plot was shown last).
            self.loss_label.setPixmap(QPixmap(spatial_path).scaled(
                self.loss_label.width(), self.loss_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.solution_label.setPixmap(QPixmap(time_path).scaled(
                self.solution_label.width(), self.solution_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            self.loss_label._source_path = spatial_path
            self.solution_label._source_path = time_path
            _spatial_dims = "(x,y,z)" if getattr(self, '_last_preview_is_3d', False) else "(x,y)"
            self.loss_header_label.setText(f"Domain: Spatial {_spatial_dims}")
            self.loss_save_btn.setToolTip("Save the domain preview's spatial panel currently shown")
            self.solution_header_label.setText("Domain: Time (x,t)")
            self.solution_save_btn.setToolTip("Save the domain preview's time-distribution panel currently shown")
        else:
            self.loss_label.setText("❌ Preview failed")
            self.loss_label._source_path = None
            self.solution_label._source_path = None
    
    _DISP_FONT_CHOICES = [
        "Segoe UI", "Arial", "Helvetica", "Verdana", "Tahoma",
        "Trebuchet MS", "Georgia", "Times New Roman",
        "Courier New", "Consolas",
    ]
    _DISP_CATEGORY_LABELS = [
        ("title", "App Title / Subtitle"),
        ("section_header", 'Section Headers (e.g. "Problem Definition", "Quick Examples")'),
        ("field_label", "Field Labels (regular text, checkboxes && radio buttons)"),
        ("hint", "Hints / Notes (small tips && warnings)"),
        ("button", "Buttons"),
        ("log_console", "Log Console"),
    ]

    def _on_display_settings(self):
        # Snapshot everything so Cancel can revert live-previewed changes,
        # not just close the dialog leaving a half-applied preview behind.
        snapshot = {
            "ui_scale": self._ui_scale,
            "font_size": self._font_size,
            "log_font_size": self._log_font_size,
            "theme": self._theme,
            "accent": self._accent,
            "panel_border_color": self._panel_border_color,
            "panel_border_width": self._panel_border_width,
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

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("UI scale (all text and controls):"))
        scale_combo = QComboBox()
        for pct in (90, 100, 110, 125, 150):
            scale_combo.addItem(f"{pct} %", pct)
        scale_combo.setCurrentIndex(max(0, scale_combo.findData(self._ui_scale)))
        scale_combo.setFixedWidth(160)
        scale_row.addStretch(); scale_row.addWidget(scale_combo)
        theme_layout.addLayout(scale_row)

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Base text size (px, at 100 %):"))
        font_spin = QSpinBox(); font_spin.setRange(10, 22); font_spin.setValue(self._font_size)
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
        theme_combo.addItems(list(_theme.THEMES))
        theme_combo.setCurrentText(self._theme)
        theme_combo.setFixedWidth(160)
        theme_row.addStretch(); theme_row.addWidget(theme_combo)
        theme_layout.addLayout(theme_row)

        accent_row = QHBoxLayout()
        accent_row.addWidget(QLabel("Accent color (Solve, selection, focus):"))
        accent_combo = QComboBox()
        accent_combo.addItems(list(_theme.ACCENTS))
        accent_combo.setCurrentText(self._accent)
        accent_combo.setFixedWidth(160)
        accent_row.addStretch(); accent_row.addWidget(accent_combo)
        theme_layout.addLayout(accent_row)

        # Panel border: the outline drawn around every Setup-tab section
        # (QGroupBox) and the loss/solution figure panels. Color left on
        # "Auto" keeps the existing theme-derived border color exactly as
        # before; thickness defaults to 1px, also matching every panel's
        # previous hardcoded border.
        border_row = QHBoxLayout()
        border_row.addWidget(QLabel("Panel border color:"))
        border_color_btn = QPushButton()
        border_color_btn.setFixedWidth(90)
        border_color_btn.setToolTip("Click to pick a border color for every panel section; Reset keeps the theme's own default")
        border_color_state = {"color": (self._panel_border_color or "").strip()}

        def _refresh_border_color_btn(btn=border_color_btn, state=border_color_state):
            c = state["color"]
            if c:
                btn.setText(c)
                btn.setStyleSheet(f"background: {c}; color: {'#000' if QColor(c).lightnessF() > 0.5 else '#fff'};")
            else:
                btn.setText("Color: Auto")
                btn.setStyleSheet("")
        _refresh_border_color_btn()

        def _pick_border_color(_checked=False, state=border_color_state, btn=border_color_btn):
            start = QColor(state["color"]) if state["color"] else QColor("#c8d2d8")
            picked = QColorDialog.getColor(start, dialog, "Choose panel border color")
            if picked.isValid():
                state["color"] = picked.name()
                _refresh_border_color_btn(btn, state)
                _apply_preview()
        border_color_btn.clicked.connect(_pick_border_color)
        border_row.addStretch(); border_row.addWidget(border_color_btn)

        border_reset_btn = QPushButton("✕ Reset")
        border_reset_btn.setFixedWidth(60)
        border_reset_btn.setToolTip("Clear the border color override, back to the theme's own default")

        def _reset_border_color(_checked=False, state=border_color_state, btn=border_color_btn):
            state["color"] = ""
            _refresh_border_color_btn(btn, state)
            _apply_preview()
        border_reset_btn.clicked.connect(_reset_border_color)
        border_row.addWidget(border_reset_btn)
        theme_layout.addLayout(border_row)

        border_width_row = QHBoxLayout()
        border_width_row.addWidget(QLabel("Panel border thickness:"))
        border_width_spin = QSpinBox()
        border_width_spin.setRange(0, 6)
        border_width_spin.setValue(self._panel_border_width)
        border_width_spin.setSuffix(" px")
        border_width_spin.setFixedWidth(80)
        border_width_row.addStretch(); border_width_row.addWidget(border_width_spin)
        theme_layout.addLayout(border_width_row)

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
            family_combo.addItem("System default")
            family_combo.addItems(self._DISP_FONT_CHOICES)
            if cur["family"] and cur["family"] not in self._DISP_FONT_CHOICES:
                family_combo.addItem(cur["family"])
            family_combo.setCurrentText(cur["family"] or "System default")
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
            self._ui_scale = int(scale_combo.currentData() or 100)
            self._font_size = font_spin.value()
            self._log_font_size = log_font_spin.value()
            self._theme = theme_combo.currentText()
            self._accent = accent_combo.currentText()
            self._panel_border_color = border_color_state["color"]
            self._panel_border_width = border_width_spin.value()
            for cat_key, widgets in cat_widgets.items():
                self._disp[cat_key] = {
                    "family": "" if widgets["family"].currentText() == "System default"
                              else widgets["family"].currentText(),
                    "size": widgets["size"].value(),
                    "bold": widgets["bold"].isChecked(),
                    "color": widgets["color_state"]["color"],
                }
            self._apply_display_settings()

        # Live preview: every control applies immediately, so changes are
        # visible right away while comparing options -- no separate
        # "Preview" click needed.
        scale_combo.currentIndexChanged.connect(_apply_preview)
        font_spin.valueChanged.connect(_apply_preview)
        log_font_spin.valueChanged.connect(_apply_preview)
        theme_combo.currentTextChanged.connect(_apply_preview)
        accent_combo.currentTextChanged.connect(_apply_preview)
        border_width_spin.valueChanged.connect(_apply_preview)
        for widgets in cat_widgets.values():
            widgets["family"].currentTextChanged.connect(_apply_preview)
            widgets["size"].valueChanged.connect(_apply_preview)
            widgets["bold"].stateChanged.connect(_apply_preview)

        btn_row = QHBoxLayout()
        defaults_btn = QPushButton("Reset to defaults")
        defaults_btn.setToolTip("Studio Light, 100 %, the system font -- then OK to keep it")
        ok_btn = QPushButton("OK (save)"); cancel_btn = QPushButton("Cancel (revert)")
        btn_row.addWidget(defaults_btn)
        btn_row.addStretch(); btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)

        def _defaults():
            scale_combo.setCurrentIndex(max(0, scale_combo.findData(100)))
            font_spin.setValue(14)
            log_font_spin.setValue(13)
            theme_combo.setCurrentText(_theme.DEFAULT_THEME)
            accent_combo.setCurrentText("Theme default")
            border_color_state["color"] = ""
            border_width_spin.setValue(1)
            for cat_key, widgets in cat_widgets.items():
                d = self._DISP_CATEGORY_DEFAULTS[cat_key]
                widgets["family"].setCurrentText("System default")
                widgets["size"].setValue(d["size"])
                widgets["bold"].setChecked(d["bold"])
                widgets["color_state"]["color"] = ""
            _apply_preview()
        defaults_btn.clicked.connect(_defaults)
        outer.addLayout(btn_row)

        def _revert_settings():
            # Runs on Cancel AND on closing the dialog via the window's own
            # close button, so a live-previewed-but-not-saved change never
            # silently sticks either way.
            self._ui_scale = snapshot["ui_scale"]
            self._font_size = snapshot["font_size"]
            self._log_font_size = snapshot["log_font_size"]
            self._theme = snapshot["theme"]
            self._accent = snapshot["accent"]
            self._panel_border_color = snapshot["panel_border_color"]
            self._panel_border_width = snapshot["panel_border_width"]
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
        """Theme tokens + fonts -> the window's stylesheet (pinnstudio/ui/theme.py);
        then every registered widget, and every widget's own stylesheet
        translated to the theme."""
        t = _theme.tokens(self._theme, self._accent)
        _theme.set_current(t)
        family = _theme.ui_font_family()
        size = self._scaled(self._font_size)
        fl = self._disp.get("field_label", self._DISP_CATEGORY_DEFAULTS["field_label"])
        sh = self._disp.get("section_header", self._DISP_CATEGORY_DEFAULTS["section_header"])
        lc = self._disp.get("log_console", self._DISP_CATEGORY_DEFAULTS["log_console"])

        def _css(cat, default_color):
            fam = cat.get("family") or family
            color = (cat.get("color") or "").strip() or default_color
            return (f"font-family: '{fam}'; font-size: {self._scaled(cat.get('size') or 14)}px; "
                    f"font-weight: {'600' if cat.get('bold') else 'normal'}; color: {color};")

        qss = _theme.build_qss(
            t, family, size, self._scaled(18),
            label_css=_css(fl, t["text"]), header_css=_css(sh, t["text"]),
            log_family=lc.get("family") or _theme.mono_font_family(), log_size=self._scaled(self._log_font_size),
            panel_border=(self._panel_border_color or "").strip() or None,
            panel_border_width=self._panel_border_width)
        self.setStyleSheet(qss)
        pb = (self._panel_border_color or "").strip() or t["border"]
        for name in ("loss_label", "solution_label", "monitor_label"):
            lbl = getattr(self, name, None)
            if lbl is not None:
                lbl.setStyleSheet(f"border: {self._panel_border_width}px solid {pb}; border-radius: 6px; "
                                  f"color: {t['text_faint']}; background: {t['card']};")
        if getattr(self, "_log_highlighter", None) is not None:
            self._log_highlighter.rehighlight()
        if hasattr(self, "_styled_widgets"):
            self._restyle_all()
        _theme.StyleRemapper.rethemeall(self)
        from pinnstudio.ui import icons as _icons
        _icons.retint_all()
        # the typeset equations are drawn in the theme's text colour
        for w in getattr(self, "pde_inputs", []):
            prev = getattr(w, "_preview", None)
            if prev is not None:
                try:
                    prev.refresh()
                except RuntimeError:
                    pass
        if getattr(self, "_script_highlighter", None) is not None:
            self._script_highlighter.rehighlight()
        if getattr(self, "start_page", None) is not None:
            self.start_page.retheme()
        if getattr(self, "run_history", None) is not None and self.run_history.compare.isVisible():
            self.run_history.compare.redraw()
        if hasattr(self, "workspace"):
            self.workspace.tree.viewport().update()

    def _on_export_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Solution Data")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        info = QLabel("Saves solution as CSV files (one per time step)\nfor forward problems after training.")
        self._register_hint(info, '#74c0fc')
        info.setWordWrap(True)
        layout.addWidget(info)

        # Off by default -- solution data used to be written on every run
        # whether or not this dialog was ever opened; now it only happens
        # when the user turns this on. Everything below only matters once
        # it's checked, so it's disabled (not hidden) otherwise, to make
        # that dependency obvious.
        enable_cb = QCheckBox("Enable solution data export")
        enable_cb.setChecked(self.export_enabled_cb.isChecked())
        layout.addWidget(enable_cb)

        grid_row = QHBoxLayout()
        grid_row_label = QLabel("Grid size (points per axis):")
        grid_row.addWidget(grid_row_label)
        grid_combo = QComboBox()
        grid_combo.addItems(["101", "51", "21", "11"])
        grid_combo.setCurrentText(self.export_grid_combo.currentText())
        grid_combo.setFixedWidth(80)
        # Editable, same as Time-Adaptive's ta_grid -- lets the user type a
        # custom per-axis resolution for 1D/2D/3D instead of only picking
        # one of the 4 presets.
        grid_combo.setEditable(True)
        grid_combo.setValidator(QIntValidator(1, 1_000_000, grid_combo))
        grid_row.addStretch(); grid_row.addWidget(grid_combo)
        layout.addLayout(grid_row)

        tsteps_row = QHBoxLayout()
        tsteps_row_label = QLabel("Number of time snapshots:")
        tsteps_row.addWidget(tsteps_row_label)
        tsteps_spin = QSpinBox()
        tsteps_spin.setRange(2, 50); tsteps_spin.setValue(self.export_tsteps_spin.value())
        tsteps_spin.setFixedWidth(80)
        tsteps_row.addStretch(); tsteps_row.addWidget(tsteps_spin)
        layout.addLayout(tsteps_row)

        note = QLabel("Output: solution_data/solution_t{time}.txt\nColumns: x, t, u (1D) or x, y, t, u (2D)")
        self._register_hint(note, '#586e75')
        note.setWordWrap(True)
        layout.addWidget(note)

        def _sync_enabled_state(checked):
            for w in (grid_row_label, grid_combo, tsteps_row_label, tsteps_spin, note):
                w.setEnabled(checked)
        enable_cb.stateChanged.connect(lambda s: _sync_enabled_state(s == 2))
        _sync_enabled_state(enable_cb.isChecked())

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self.export_enabled_cb.setChecked(enable_cb.isChecked())
            self.export_grid_combo.setCurrentText(grid_combo.currentText())
            self.export_tsteps_spin.setValue(tsteps_spin.value())
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()
    
    def _on_restore_mode_changed(self, text):
        is_inverse = (text == "Inverse Model")
        self.restore_inv_vars_widget.setVisible(is_inverse)
        current_viz = self.restore_viz_combo.currentText()
        items = self._restore_forward_viz_items() + (list(self._RESTORE_PARAM_VIZ) if is_inverse else [])
        self.restore_viz_combo.blockSignals(True)
        self.restore_viz_combo.clear()
        self.restore_viz_combo.addItems(items)
        if current_viz in items:
            self.restore_viz_combo.setCurrentText(current_viz)
        self.restore_viz_combo.blockSignals(False)
        self._on_restore_viz_changed(self.restore_viz_combo.currentText())

    def _restore_forward_viz_items(self):
        """Forward viz-type options for the Restore panel's dropdown --
        all four (see self._RESTORE_FORWARD_VIZ) normally, but just
        ["Surface"] when the config about to be restored is steady-state:
        a steady-state problem has no time axis at all (same reason
        _on_steady_state_changed already hides Time Adaptive Training and
        the Initial Condition panel on the main Setup tab), so "Line (time
        steps)" and the two Animation types -- all inherently about
        stepping through or animating over time -- don't apply. Read
        directly from the config file about to be restored (same safe-no-
        op-on-unreadable-config pattern as _on_restore_viz_settings's own
        _restore_dim) rather than any cached state, so this always reflects
        whatever's actually currently browsed to; defaults to showing all
        four when that file can't be read yet (e.g. nothing picked yet).

        "Animation Surface (GIF)" is further excluded for a 1D config --
        same reasoning as the main Setup tab's plot_type_combo (see
        _on_dim_changed): a 1D animation frame has only one real spatial
        axis, so there's no second axis for a surface to vary across
        within a single frame. "Animation Line (GIF)" stays -- that's 1D's
        actual correct time-animated option.

        Only excluded when "problem_dim" is explicitly present and equal
        to "1D" -- a config saved before this field existed (or written
        directly in a test/script without it) has it missing entirely,
        and its real dimension is simply unknown here, not necessarily
        1D. Defaulting a missing field to "1D" would silently hide this
        option for an old 2D/3D config too, which is worse than just
        showing all four the way every config was treated before this
        exclusion existed."""
        import json
        try:
            with open(self.restore_config_path.text().strip()) as f:
                _restore_cfg_probe = json.load(f)
            is_steady = bool(_restore_cfg_probe.get("steady_state", False))
            is_1d = _restore_cfg_probe.get("problem_dim") == "1D"
        except Exception:
            is_steady = False
            is_1d = False
        if is_steady:
            return ["Surface"]
        items = list(self._RESTORE_FORWARD_VIZ)
        if is_1d:
            items = [v for v in items if v != "Animation Surface (GIF)"]
        return items

    def _refresh_restore_viz_options(self):
        """Re-apply the steady-state filtering above to the dropdown's
        current contents -- called alongside _refresh_restore_output_combo
        (same two call sites: browsing a model file auto-detects its
        config, browsing a config file directly) so switching to or from a
        steady-state model's config immediately hides/restores the three
        time-based options, same as _on_restore_mode_changed already does
        when switching Forward/Inverse."""
        is_inverse = (getattr(self, 'restore_mode_combo', None) is not None
                      and self.restore_mode_combo.currentText() == "Inverse Model")
        items = self._restore_forward_viz_items() + (list(self._RESTORE_PARAM_VIZ) if is_inverse else [])
        current_viz = self.restore_viz_combo.currentText()
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
        self._update_restore_ta_combine_visibility()
        self._on_restore_viz_settings(text)

    def _update_restore_ta_combine_visibility(self):
        """The 'Combine steps' checkbox only makes sense for the two
        Animation viz types with >=2 Time-Adaptive steps detected -- a
        static Surface/Line restore always auto-routes to the right step
        regardless of this checkbox (see _on_restore), so showing it
        there would just be a confusing no-op control."""
        has_steps = len(getattr(self, '_restore_ta_steps', []) or []) >= 2
        is_anim = self.restore_viz_combo.currentText() in ("Animation Line (GIF)", "Animation Surface (GIF)")
        self.restore_ta_combine_cb.setVisible(has_steps and is_anim)

    @staticmethod
    def _viz_steps_key(viz_type):
        """Which n_steps_* key a given plot-type name reads/writes --
        an animation ("Line Animation (GIF)" on the Setup tab,
        "Animation Line (GIF)"/"Animation Surface (GIF)" on Restore --
        the two tabs order the words differently) wants its own frame
        count, separate from the static "Line (time steps)" plot's step
        count, so the two can have different defaults (more frames for a
        smoother/slower animation; fewer lines for a readable static
        overlay) instead of fighting over one shared number."""
        return 'n_steps_anim' if 'Animation' in viz_type else 'n_steps_line'

    @staticmethod
    def _viz_linewidth_key(viz_type):
        """Which linewidth_* key a given plot-type name reads/writes --
        same split as _viz_steps_key, for the same reason, but only
        matters for the "Line"-named types (a Surface plot has no line
        width control at all, see lw_widget's own visibility below)."""
        return 'linewidth_anim' if 'Animation' in viz_type else 'linewidth'

    def _on_restore_viz_settings(self, viz_type=None):
        if viz_type is None:
            viz_type = self.restore_viz_combo.currentText()

        is_param = viz_type in getattr(self, '_RESTORE_PARAM_VIZ', [])

        # The dimension that matters for this dialog (e.g. the "Swap axes"
        # option below, 1D-only) is whatever's being restored, not whatever
        # the main Setup tab's dimension radios currently show -- those two
        # can easily differ (e.g. Setup tab left on 3D while restoring an
        # older 1D checkpoint). Read it from the config about to be
        # restored; fall back to "1D" (same default _build_restore_script
        # itself uses) if it can't be read yet, e.g. no config picked yet.
        _restore_dim = "1D"
        _restore_y_min, _restore_y_max = 0.0, 1.0
        _restore_z_min, _restore_z_max = 0.0, 1.0
        _restore_is_steady = False
        try:
            import json as _json_dim
            with open(self.restore_config_path.text().strip()) as _df:
                _restore_cfg_dim = _json_dim.load(_df)
            _restore_dim = _restore_cfg_dim.get("problem_dim", "1D")
            _restore_y_min = _restore_cfg_dim.get("y_min", 0.0); _restore_y_max = _restore_cfg_dim.get("y_max", 1.0)
            _restore_z_min = _restore_cfg_dim.get("z_min", 0.0); _restore_z_max = _restore_cfg_dim.get("z_max", 1.0)
            _restore_is_steady = bool(_restore_cfg_dim.get("steady_state", False))
        except Exception:
            pass
        _restore_is_1d = _restore_dim == "1D"
        _restore_is_2d = _restore_dim == "2D"
        _restore_is_3d = _restore_dim == "3D"
        _restore_is_line_type = viz_type in ("Line (time steps)", "Animation Line (GIF)")

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._restore_viz_settings
        # Slice defaults: an override saved earlier for this restore, else
        # the midpoint of the domain about to be restored (recomputed from
        # whatever config was just picked, not an earlier model's domain).
        _slice = None
        if _restore_is_2d or _restore_is_3d:
            _y_auto = current.get('line_slice_y_auto', True)
            _slice = {"y": (_y_auto, (_restore_y_min + _restore_y_max) / 2.0 if _y_auto else current.get('line_slice_y', 0.0))}
            if _restore_is_3d:
                _z_auto = current.get('line_slice_z_auto', True)
                _slice["z"] = (_z_auto, (_restore_z_min + _restore_z_max) / 2.0 if _z_auto else current.get('line_slice_z', 0.0))
        # Shared with the Setup tab's dialog -- see _build_plot_settings_fields.
        w = self._build_plot_settings_fields(
            layout, current, viz_type, is_param=is_param, is_1d=_restore_is_1d, is_2d=_restore_is_2d,
            is_3d=_restore_is_3d, is_steady=_restore_is_steady, is_line_type=_restore_is_line_type,
            is_anim="Animation" in viz_type, surface_types=("Surface", "Animation Surface (GIF)"),
            levels_default=40, steps_value=current.get(self._viz_steps_key(viz_type), 10), steps_max=100,
            slice_defaults=_slice)

        info_texts = {
            "Surface": "Single heatmap/contour at specified time.",
            "Line (time steps)": "Solution lines at evenly spaced time steps.",
            "Animation Line (GIF)": "Animated GIF of line plots over time.",
            "Animation Surface (GIF)": "Animated GIF of surface plots with colorbar.",
            "Parameter Convergence Plot (PNG)": "Static plot of each trainable variable's value vs. iteration.",
            "Parameter Convergence Animation (GIF)": "Animated GIF of each trainable variable's convergence, growing curve up to each logged iteration.",
        }
        info = QLabel(info_texts.get(viz_type, ""))
        self._register_hint(info, '#586e75')
        info.setWordWrap(True)
        layout.addWidget(info)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            new_settings = self._plot_settings_values(w, current, viz_type, is_param)
            if not is_param:
                # Written to whichever n_steps_* key this viz_type actually
                # uses (see _viz_steps_key), so Animation Line's frame count
                # doesn't clobber Line (time steps)'s step count or vice versa.
                new_settings[self._viz_steps_key(viz_type)] = w.steps_spin.value()
                self.restore_tsteps_spin.setValue(w.steps_spin.value())
                if w.y_auto_cb is not None:
                    new_settings['line_slice_y_auto'] = w.y_auto_cb.isChecked()
                    new_settings['line_slice_y'] = w.y_spin.value()
                if w.z_auto_cb is not None:
                    new_settings['line_slice_z_auto'] = w.z_auto_cb.isChecked()
                    new_settings['line_slice_z'] = w.z_spin.value()
            self._restore_viz_settings = new_settings
            self.log_box.append(f"✅ Viz settings saved — {viz_type}")
            dialog.accept()
        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    # ── Quick Examples: shared template-apply steps (cleanup Round 59) ──
    # _on_template_selected applies five template tables (2D, 2D steady,
    # 3D, 3D steady, 1D); each used to repeat these steps by hand, and the
    # copies had drifted (only the 1D branch reset the L-BFGS float type,
    # only the 1D branch ended the fallback Time-Adaptive row at the
    # template's own t_max). Every branch now goes through these.

    def _tpl_set_equations(self, n_out, names, pdes, ics):
        """Output count (None leaves it alone), output names, PDE and IC text."""
        if n_out is not None and n_out != self.num_outputs_spin.value():
            self.num_outputs_spin.setValue(n_out)
        for widgets, values in ((self.output_name_inputs, names), (self.pde_inputs, pdes), (self.ic_inputs, ics)):
            for i, value in enumerate(values):
                if i < len(widgets):
                    widgets[i].setText(value)

    def _tpl_set_time_range(self, t):
        """t_max plus the Forward/Inverse t_max pair _on_problem_type_changed
        switches between; t=None for a steady template (no time axis).
        Always reset, so a template never inherits another's inverse_t_max."""
        if t is None:
            self._current_template_forward_tmax = None
            self._current_template_inverse_tmax = None
            return
        self.t_max.setValue(t.get('t_max', 1.0))
        self._current_template_forward_tmax = t.get('t_max', 1.0)
        self._current_template_inverse_tmax = t.get('inverse_t_max')
        if self._current_template_inverse_tmax is not None and self.radio_inverse.isChecked():
            self.t_max.setValue(self._current_template_inverse_tmax)

    def _tpl_set_sampling_and_training(self, t, *, num_initial, optimizer2):
        """Collocation points, network size, both optimizer stages. num_test
        and the L-BFGS float type fall back to 1000 / float32 when the
        template doesn't set them, so a previous template's value (3D
        Heat's num_test=10000, Schrodinger's float64) never lingers."""
        self.num_domain.setValue(t['num_domain'])
        self.num_boundary.setValue(t['num_boundary'])
        self.num_initial.setValue(num_initial)
        self.num_test.setValue(t.get('num_test', 1000))
        self.layers_spin.setValue(t['layers'])
        self.neurons_spin.setValue(t['neurons'])
        self.iter1_spin.setValue(t['iterations'])
        self.opt2_combo.setCurrentText(optimizer2)
        self.iter2_spin.setValue(t['iterations2'])
        if hasattr(self, 'lbfgs_float_combo'):
            self.lbfgs_float_combo.setCurrentText(t.get('lbfgs_float_type', 'float32'))
        # A template always trains with the recommended L-BFGS options, even
        # straight after a DeepXDE example set its own.
        self._reset_lbfgs_to_recommended()

    def _tpl_set_ic_weights(self, n_out, weight):
        for i in range(n_out):
            key = f"ic_{i}"
            if key in self.weight_widgets:
                self.weight_widgets[key].setValue(weight)

    def _tpl_reset_time_adaptive(self, ta_cfg, t_end, grid_default):
        """Rebuild the Time-Adaptive panel: the template's own ta_default
        when it has one (suspended, not applied, in Inverse mode), else
        Adaptation=None with one pre-filled 0..t_end row ready in case the
        user turns it on manually. grid_default is "21" for 3D (see
        _update_ta_grid_warning), "101" otherwise."""
        for row in list(self.ta_group_rows):
            row['widget'].deleteLater()
        self.ta_group_rows.clear()
        self._current_ta_cfg = ta_cfg
        self._ta_suspended_for_inverse = False
        if ta_cfg and not self.radio_inverse.isChecked():
            self.adapt_combo.setCurrentText("Time Adaptive Training")
            for g_start, g_end, g_steps in ta_cfg['step_groups']:
                self._add_ta_step_group(g_start, g_end, g_steps)
            self.ta_transfer_cb.setChecked(ta_cfg.get('transfer_learning', False))
            self.ta_grid.setCurrentText(str(ta_cfg.get('ic_grid', 101)))
            self._set_combo_data(self.ta_transfer_opt, ta_cfg.get('transfer_optimizer', 'adam'))
        else:
            self.adapt_combo.setCurrentText("None")
            self._add_ta_step_group(0.0, t_end, 10)
            self.ta_transfer_cb.setChecked(False)
            self.ta_grid.setCurrentText(grid_default)
            self._set_combo_data(self.ta_transfer_opt, "adam")
            if ta_cfg and self.radio_inverse.isChecked():
                self._ta_suspended_for_inverse = True

    def _tpl_set_steady_dirichlet_bc(self):
        """The steady Poisson templates' BC: Dirichlet u = 0 on the whole
        boundary -- "True" as the location, the convention for a shape with
        no natural "side" (DeepXDE's on_boundary restricts it)."""
        for e in list(self.custom_bc_list):
            e['widget'].deleteLater()
        self.custom_bc_list.clear(); self._schedule_bc_nodes_sync()
        self._add_custom_bc_entry(bc_type="dirichlet", component=0,
                                   location="True", value="0", locked=False)
        self._update_bc_mode_visibility()

    def _tpl_finish(self, text, t, *, steady=False, dirichlet_bc_weight=None):
        """Common tail: remember the template, sync the Inverse PDE
        substitution, default scheduler phases, optional Dirichlet BC
        weight, Error Analysis reference files, log line. The BC weight is
        set after _setup_default_scheduler_phases(), which rebuilds
        weight_widgets (and would otherwise reset it to 1.0)."""
        self._template_ref_dir = t.get('ref_dir', '')
        self._current_template = text
        self._current_template_type = t.get('template_type', '')
        self._sync_inverse_pde_substitution(self.radio_inverse.isChecked())
        if hasattr(self, 'sched_cb'):
            self.sched_cb.setChecked(True)
            self._setup_default_scheduler_phases(t.get('template_type', ''), t['iterations'], t.get('iterations2', 10000))
        if dirichlet_bc_weight is not None:
            for _bj, _be in enumerate(self.custom_bc_list):
                if _be['type'].currentData() == 'dirichlet':
                    _bk = f"bc_{_bj}"
                    if _bk in self.weight_widgets:
                        self.weight_widgets[_bk].setValue(dirichlet_bc_weight)
        if steady:
            self._auto_configure_ea(self._template_ref_dir, is_steady=True)
        else:
            self._auto_configure_ea(self._template_ref_dir)
        self.log_box.append(f"✅ Template loaded: {_ex_catalog.display_name(text)}")

    def _on_template_selected(self, text):
        if text == "📋 Examples":
            return

        # Quick Examples now lists every template across all three
        # dimensions at once (see _build_ui's Quick Examples combo and
        # _on_dim_changed, which no longer rebuilds this combo's items per
        # dimension) -- so picking one has to set the matching dimension
        # radio itself, rather than relying on the user having already
        # picked the right dimension before a dimension-filtered list
        # ever offered this template. Every template name is prefixed
        # with its own dimension ("1D Heat", "2D Heat", "3D Heat", ...),
        # so that prefix is the single source of truth here -- no
        # separate name->dimension table to keep in sync. _applying_
        # template guards _on_dim_changed's own "a template no longer
        # matches -- reset Quick Examples to None" logic, so setting the
        # radio here doesn't immediately wipe the very selection this
        # function is in the middle of applying. The actual template
        # dispatch further down (the `if text in templates_2d:` / etc.
        # chain) doesn't depend on this radio at all -- it matches purely
        # on which dict `text` is a key of -- but the REST of the GUI
        # (domain field visibility, z_min/z_max, the 2D/3D geometry-type
        # selector, ...) does, via _on_dim_changed, so this still has to
        # happen for the loaded template to actually display correctly.
        _target_dim_radio = (
            self.radio_2d if text.startswith("2D") else
            self.radio_3d if text.startswith("3D") else
            self.radio_1d if text.startswith("1D") else None
        )
        if _target_dim_radio is not None and not _target_dim_radio.isChecked():
            self._applying_template = True
            try:
                _target_dim_radio.setChecked(True)
            finally:
                self._applying_template = False

        # Every template starts from a clean Steady-state toggle -- only the
        # Poisson (L-Shape/Disk/Sphere) branches below turn it back on for
        # themselves. Without this reset, picking a steady template and then
        # a time-dependent one (e.g. Disk -> 2D Heat) would silently leave
        # Heat training with no time axis at all, since nothing else ever
        # turns this checkbox back off.
        if hasattr(self, 'steady_state_check') and self.steady_state_check.isChecked():
            self.steady_state_check.setChecked(False)

        # Every template also starts from a clean Results-panel plot field --
        # only the 1D dispatch block below (the one template dict that
        # currently uses this at all, for 1D Schrödinger's |h| field) ever
        # re-selects "Custom..." for itself, further down in this same
        # function. Without this reset here, switching dimension and/or
        # template away from 1D Schrödinger left the plot combo on
        # "Custom..." with its stale expression (e.g. "sqrt(u**2+v**2)")
        # for every 2D/3D template dispatch block, which never reference
        # this field at all -- harmless when the newly selected template
        # happens to share the same output names (silently plots a
        # meaningless derived field instead of the requested output), and
        # an uncaught NameError from evaluating a leftover expression
        # against a different template's output names otherwise. Same
        # staleness class as the steady-state reset just above.
        if hasattr(self, 'plot_output_combo') and self.plot_output_combo.count():
            self.plot_output_combo.setCurrentIndex(0)
        if hasattr(self, 'plot_custom_expr_input'):
            self.plot_custom_expr_input.clear()
        if hasattr(self, 'plot_custom_label_input'):
            self.plot_custom_label_input.clear()

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
                # The "1.0*" on the reaction term is written explicitly
                # (rather than the mathematically-equivalent bare
                # "(u**3 - u)") so there's a literal substring for Inverse
                # mode to substitute c_2 into -- see INVERSE_AUTO_CONST's
                # comment above. Forward-mode behavior is identical either
                # way (1.0 times anything is a no-op).
                'pde': ["du_t - 0.0001*(du_xx + du_yy) + 1.0*(u**3 - u)"],
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
            # Physics only, from Mathias, de Almeida, de Barros, Coelho et
            # al. 2022 ("Augmenting a Physics-Informed Neural Network for
            # the 2D Burgers Equation by Addition of Solution Data Points",
            # BRACIS 2022; arXiv:2301.07824), eq. (1)-(4):
            #   dt U + U dx U + V dy U = nu (dxx U + dyy U),
            #   dt V + U dx V + V dy V = nu (dxx V + dyy V),
            # x,y in [0,1], t in [0,1], nu = 0.01/pi, with
            #   U(0,x,y) = sin(2 pi x) sin(2 pi y),
            #   V(0,x,y) = sin(pi x) sin(pi y),
            # and Dirichlet U=V=0 on all four edges for all time.
            # Deliberately NOT replicated here (out of scope for this
            # template, per the user's own choice): the paper's actual
            # point -- augmenting training with sparse ground-truth data
            # points -- and its hard-constrained boundary/IC-encoding
            # output layer plus residual-block network; this uses the same
            # soft Dirichlet/IC loss and plain MLP every other template
            # already trains with. No reference data file existed when this
            # template was first added; the user has since generated and
            # supplied real reference data (t_0_u.txt/t_0_v.txt below), and
            # confirmed the best results come from a wider network (4
            # hidden layers x 128 neurons, up from the app-wide 3x64
            # default -- a coupled two-output velocity field benefits from
            # the extra capacity) and real Time-Adaptive stepping (4 steps
            # of 0.25 each, not one full-domain pass -- an earlier version
            # of this template used a single full-domain TA step, which
            # the user has since confirmed is not what they actually run;
            # the 4-step schedule below, with L-BFGS transfer between
            # steps, is the corrected out-of-the-box default). This
            # template previously existed alongside an earlier "2D
            # Burgers" template (Lu/Meng/Mao/Karniadakis's Re=5000 DeepXDE
            # example); that one was removed at the user's request once
            # this template's own results proved out, since its own BC/IC
            # never reproduced the paper's results and it's no longer
            # needed.
            "2D Burgers (Mathias)": {
                'pde': ["du_t + u*du_x + v*du_y - (0.01/pi)*(du_xx + du_yy)",
                        "dv_t + u*dv_x + v*dv_y - (0.01/pi)*(dv_xx + dv_yy)"],
                'ic': ["sin(2*pi*x)*sin(2*pi*y)",
                       "sin(pi*x)*sin(pi*y)"],
                'num_domain': 8000,
                'num_boundary': 2000,
                'num_initial': 2000,
                'layers': 4,
                'neurons': 128,
                'iterations': 15000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': 0.0, 'x_max': 1.0,
                'y_min': 0.0, 'y_max': 1.0,
                'periodic_bc': False,
                'bc_config': 'dirichlet_zero_all',
                'num_outputs': 2,
                'output_names': ['u', 'v'],
                # Ground truth for both outputs -- t_0_u.txt / t_0_v.txt,
                # auto-assigned to output "u" / "v" respectively by
                # _auto_configure_ea's per-output filename matching, exactly
                # like every other template's single-output reference data
                # (for ease of use). Users can still reassign or add their
                # own via the "📊 Error Analysis" dialog.
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "burgers_mathias"),
                # 4 real Time-Adaptive steps (0->0.25->0.5->0.75->1.0),
                # L-BFGS transfer learning between steps, 51-point IC
                # continuity grid -- matches the 2D Allen-Cahn (Mattey &
                # Ghosh) template's own step count/grid convention.
                'ta_default': {'step_groups': [(0.0, 1.0, 4)], 'transfer_learning': True, 'ic_grid': 51, 'transfer_optimizer': 'lbfgs'},
            },
        }
        if text in templates_2d:
            t = templates_2d[text]
            n_out = t.get('num_outputs', 1)
            self._tpl_set_equations(n_out, t.get('output_names', ['u']), t['pde'], t['ic'])
            # Every template in this dict is a plain rectangle -- reset the
            # Geometry selector explicitly, or switching here from a Polygon/
            # Disk Poisson template left it (and its shape panel) stuck.
            self.geometry_type_combo.setCurrentText('Rectangle')
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            self._tpl_set_time_range(t)
            self._tpl_set_sampling_and_training(t, num_initial=t['num_initial'], optimizer2=t['optimizer2'])
            self._tpl_set_ic_weights(n_out, t.get('ic_weight', 100.0))
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
            elif bc_config == 'dirichlet_zero_all':
                # All four edges Dirichlet=0, every output -- Mathias et
                # al.'s 2D Burgers velocities are both zero at the domain
                # boundary for all time, and unlike the DeepXDE 2D Burgers
                # template's BC this is a plain constant, not a
                # time-dependent expression, so the ordinary per-side
                # legacy widgets (which only ever hold a constant) are
                # enough -- no custom expression rows needed.
                for i in range(n_out):
                    if i < len(self.bc_left_types):
                        self.bc_left_types[i].setCurrentText("Dirichlet")
                        self.bc_left_vals[i].setValue(0.0)
                    if i < len(self.bc_right_types):
                        self.bc_right_types[i].setCurrentText("Dirichlet")
                        self.bc_right_vals[i].setValue(0.0)
                    if i < len(self.bc_bottom_types):
                        self.bc_bottom_types[i].setCurrentText("Dirichlet")
                        self.bc_bottom_vals[i].setValue(0.0)
                    if i < len(self.bc_top_types):
                        self.bc_top_types[i].setCurrentText("Dirichlet")
                        self.bc_top_vals[i].setValue(0.0)
            self._populate_locked_bc_entries_from_legacy(n_out, is_2d=True)
            self._update_bc_mode_visibility()
            self._tpl_reset_time_adaptive(t.get('ta_default'), t.get('t_max', 1.0), "101")
            # heat2d: the right-edge Dirichlet BC trains more accurately at
            # weight 100 (empirically confirmed).
            self._tpl_finish(text, t, dirichlet_bc_weight=100.0 if bc_config == 'heat2d' else None)
            return

        # ── Steady-state (time-independent) 2D Quick Examples ──────────
        # Both templates are the same Poisson family (-Δu = 1, u = 0 on
        # the whole boundary) on a non-rectangular domain -- L-Shape is
        # DeepXDE's own official example (poisson.Lshape.py: Polygon
        # geometry, [2]+[50]*4+[1] tanh/Glorot uniform network, Adam
        # lr=0.001 for 50000 iterations then L-BFGS to convergence,
        # num_domain=1200/num_boundary=120/num_test=1500); Disk reuses the
        # same recipe on a unit disk, whose closed-form solution
        # u = (R^2 - r^2)/4 gives an independent, cheap correctness check
        # (not wired into the app -- confirmed by hand against the
        # exported solution data during verification). Both need
        # Steady-state support (config.steady_state / codegen.py's
        # _is_steady) since neither problem has a time variable at all.
        templates_2d_steady = {
            "2D Poisson (L-Shape)": {
                'pde': "-du_xx - du_yy - 1",
                'geometry_type': 'Polygon',
                'geom_polygon_vertices': "0,0;1,0;1,-1;-1,-1;-1,1;0,1",
                'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0,
                'num_domain': 1200, 'num_boundary': 120, 'num_test': 1500,
                'layers': 4, 'neurons': 50,
                'iterations': 50000, 'iterations2': 50000,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "poisson_lshape"),
            },
            "2D Poisson (Disk)": {
                'pde': "-du_xx - du_yy - 1",
                'geometry_type': 'Disk',
                'geom_center_x': 0.0, 'geom_center_y': 0.0, 'geom_radius': 1.0,
                'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0,
                'num_domain': 1200, 'num_boundary': 120, 'num_test': 1500,
                'layers': 4, 'neurons': 50,
                'iterations': 50000, 'iterations2': 50000,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "2D", "poisson_disk"),
            },
        }
        if text in templates_2d_steady:
            t = templates_2d_steady[text]
            self._tpl_set_equations(1, ['u'], [t['pde']], [])
            # Domain bounds (display/bbox-fallback only -- the Polygon/Disk
            # geometry panel below is what actually defines the shape).
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            self._tpl_set_time_range(None)
            self._tpl_set_sampling_and_training(t, num_initial=0, optimizer2='lbfgs')
            # Geometry: switch shape and set that shape's own parameters --
            # setCurrentText fires _on_geometry_type_changed, which shows
            # the right shape panel and hides the plain x/y rows.
            self.geometry_type_combo.setCurrentText(t['geometry_type'])
            if t['geometry_type'] == 'Polygon':
                self.geom_polygon_verts_input.setText(t['geom_polygon_vertices'])
            elif t['geometry_type'] == 'Disk':
                self.geom_disk_cx.setValue(t['geom_center_x'])
                self.geom_disk_cy.setValue(t['geom_center_y'])
                self.geom_disk_r.setValue(t['geom_radius'])
            # Steady-state: no time axis, no IC, no Time-Adaptive Training
            # (see _on_steady_state_changed). The explicit Time-Adaptive reset
            # below still matters: setChecked(True) is a no-op when already
            # steady, and a previous TA template (especially one suspended
            # for Inverse) could otherwise leave config.time_adaptive True.
            self.steady_state_check.setChecked(True)
            self._tpl_reset_time_adaptive(None, 1.0, "101")
            self._tpl_set_steady_dirichlet_bc()
            self._tpl_finish(text, t, steady=True)
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
            self._tpl_set_equations(n_out, t.get('output_names', ['u']), t['pde'], t['ic'])
            # Same Geometry reset as the 2D dict above (3D Heat is always a
            # Cuboid; without this, coming from the Sphere template stuck).
            self.geometry_type_combo.setCurrentText('Cuboid')
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            self.z_min.setValue(t['z_min']); self.z_max.setValue(t['z_max'])
            self._tpl_set_time_range(t)
            self._tpl_set_sampling_and_training(t, num_initial=t['num_initial'], optimizer2=t['optimizer2'])
            self._tpl_set_ic_weights(n_out, t.get('ic_weight', 100.0))
            self._populate_3d_heat_bc_entries(n_out)
            self._update_bc_mode_visibility()
            # 3D Heat doesn't use Time-Adaptive, but if turned on manually
            # the grid should already be at a 3D-safe size ("21", see
            # _update_ta_grid_warning) rather than the 1D/2D "101".
            self._tpl_reset_time_adaptive(t.get('ta_default'), t.get('t_max', 1.0), "21")
            # The x-max Dirichlet face defaults to weight 100, same
            # convention as 2D Heat's right edge.
            self._tpl_finish(text, t, dirichlet_bc_weight=100.0)
            return

        # ── Steady-state (time-independent) 3D Quick Example ────────────
        # Same -Δu = 1, u = 0 Poisson family as the 2D steady templates
        # above, on a unit Sphere -- closed-form solution u = (R^2 - r^2)/6,
        # again used only as a hand-checked sanity check during
        # verification, not an app feature. See templates_2d_steady above
        # for the shared rationale (Steady-state support, "True" as the
        # unified-boundary BC location).
        templates_3d_steady = {
            "3D Poisson (Sphere)": {
                'pde': "-du_xx - du_yy - du_zz - 1",
                'geom_center_x': 0.0, 'geom_center_y': 0.0, 'geom_center_z': 0.0, 'geom_radius': 1.0,
                'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0, 'z_min': -1.0, 'z_max': 1.0,
                'num_domain': 2000, 'num_boundary': 300, 'num_test': 2000,
                'layers': 4, 'neurons': 50,
                'iterations': 20000, 'iterations2': 20000,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "3D", "poisson_sphere"),
            },
        }
        if text in templates_3d_steady:
            t = templates_3d_steady[text]
            self._tpl_set_equations(1, ['u'], [t['pde']], [])
            self.x_min.setValue(t['x_min']); self.x_max.setValue(t['x_max'])
            self.y_min.setValue(t['y_min']); self.y_max.setValue(t['y_max'])
            self.z_min.setValue(t['z_min']); self.z_max.setValue(t['z_max'])
            self._tpl_set_time_range(None)
            self._tpl_set_sampling_and_training(t, num_initial=0, optimizer2='lbfgs')
            self.geometry_type_combo.setCurrentText('Sphere')
            self.geom_sphere_cx.setValue(t['geom_center_x'])
            self.geom_sphere_cy.setValue(t['geom_center_y'])
            self.geom_sphere_cz.setValue(t['geom_center_z'])
            self.geom_sphere_r.setValue(t['geom_radius'])
            # Same steady-state / Time-Adaptive reset as templates_2d_steady,
            # with the 3D-safe grid default.
            self.steady_state_check.setChecked(True)
            self._tpl_reset_time_adaptive(None, 1.0, "21")
            self._tpl_set_steady_dirichlet_bc()
            self._tpl_finish(text, t, steady=True)
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
            # Exact equation, IC and BC as in Lu, Meng, Mao & Karniadakis
            # 2019 (DeepXDE), section 4.2 / Raissi, Perdikaris & Karniadakis
            # 2019 (JCP), eq. (A.1): u_t + u u_x - (0.01/pi) u_xx = 0,
            # x in [-1,1], t in [0,1], u(x,0) = -sin(pi x), u(-1,t)=u(1,t)=0.
            # No RAR (residual-based adaptive refinement) this round -- a
            # larger fixed collocation count is used instead to help resolve
            # the shock that forms near x=0 as t -> 1. Network/training
            # recipe follows DeepXDE Table 3, Example 2 (depth 3, width 20,
            # Adam then L-BFGS, lr 0.001, 15000 Adam iterations).
            "1D Burgers": {
                # The "1.0*" on the convection term is written explicitly
                # (rather than the mathematically-equivalent bare
                # "u*du_x") so there's a literal substring for Inverse mode
                # to substitute lambda_1 into, matching Raissi et al.'s own
                # eq. B.1 two-coefficient form -- see INVERSE_AUTO_CONST's
                # comment above. Forward-mode behavior is unchanged (1.0
                # times anything is a no-op).
                'pde': ["du_t + 1.0*u*du_x - (0.01/pi)*du_xx"],
                'ic': ["-sin(pi*x)"],
                'num_domain': 8000,
                'num_boundary': 200,
                'num_initial': 200,
                'layers': 3,
                'neurons': 20,
                'iterations': 15000,
                'optimizer2': 'lbfgs',
                'iterations2': 10000,
                'x_min': -1.0, 'x_max': 1.0,
                'periodic_bc': False,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "1D", "burgers"),
            },
            # Exact equation, IC and periodic BC as in Raissi, Perdikaris &
            # Karniadakis 2019 (JCP), section 3.1.1: the 1D nonlinear
            # Schrodinger equation i h_t + 0.5 h_xx + |h|^2 h = 0,
            # x in [-5,5], t in [0, pi/2], h(0,x) = 2 sech(x),
            # h(t,-5)=h(t,5), h_x(t,-5)=h_x(t,5). h is complex-valued and
            # represented as two real outputs h = u + iv (2-output PDE);
            # splitting i h_t + 0.5 h_xx + |h|^2 h = 0 into real/imaginary
            # parts gives the two real PDEs below (the standard PINN
            # Schrodinger residual split). Periodic BC is enforced on both
            # u and v AND on their first x-derivatives (matching the
            # paper's h_x(t,-5)=h_x(t,5) condition) -- the derivative-order-1
            # rows are added on top of the standard value-periodic rows by
            # _populate_schrodinger_bc_entries below, since the generic
            # legacy-BC population only ever adds derivative_order=0.
            # Network follows the paper's own recipe: 5-layer, 100 neurons
            # per layer, trained mainly via L-BFGS (a short Adam warm-up is
            # used first since PINNStudio's phase-1 optimizer is always Adam).
            "1D Schrödinger": {
                'num_outputs': 2,
                'output_names': 'u,v',
                'pde': ["du_t + 0.5*dv_xx + (u**2+v**2)*v",
                        "dv_t - 0.5*du_xx - (u**2+v**2)*u"],
                'ic': ["2/cosh(x)", "0"],
                'num_domain': 20000,
                'num_boundary': 200,
                'num_initial': 200,
                'layers': 5,
                'neurons': 100,
                'iterations': 1000,
                'optimizer2': 'lbfgs',
                'iterations2': 20000,
                'x_min': -5.0, 'x_max': 5.0,
                't_max': 1.5707963267948966,
                'periodic_bc': True,
                'ref_dir': os.path.join(REFERENCE_DATA_DIR, "1D", "schrodinger"),
                # h = u + iv is complex-valued -- what's physically meaningful
                # to look at is its magnitude |h| = sqrt(u^2+v^2) (this is
                # what Raissi et al.'s own Figure 1 plots), not u or v
                # individually. Pre-fills the Plot output panel's "Custom..."
                # option -- see _populate_locked_bc_entries_from_legacy's
                # sibling hook below and codegen.py's _extract_plot_field.
                # The same expression is also what Error Analysis compares
                # against by default (its own _ea_sel is None -> delegates
                # to _extract_plot_field, see codegen.py's _ea_extract),
                # matching this template's own reference_data/1D/schrodinger
                # files, which are themselves |h| snapshots, not raw u/v.
                'plot_custom_expr': 'sqrt(u**2+v**2)',
                'plot_custom_label': '|h|',
                # L-BFGS in float32 (this GUI's general default) hits a
                # numerical noise floor around ~1e-7 well before this
                # problem's loss has actually bottomed out -- a previously
                # documented lesson from this same user's other PINN work.
                # float64 lets L-BFGS keep making real progress for its
                # full iteration budget instead of plateauing early;
                # confirmed via a real side-by-side run on this exact
                # template (same PDE/IC/BC, same iteration budget): float64
                # reached a materially lower final loss (1.75e-04) than
                # float32 (4.64e-04). Only this template defaults it to
                # float64 -- every other template's own default (float32,
                # the Settings tab's own combo default) is unchanged.
                'lbfgs_float_type': 'float64',
            },
        }

        t = templates.get(text)
        if not t:
            return

        if 'num_outputs' in t:
            self.num_outputs_spin.setValue(t['num_outputs'])
            QApplication.processEvents()
        names = [n.strip() for n in t['output_names'].split(',')] if 'output_names' in t else []
        self._tpl_set_equations(None, names, t['pde'], t['ic'])
        self._tpl_set_sampling_and_training(t, num_initial=t['num_initial'], optimizer2=t['optimizer2'])
        if text == "1D Allen-Cahn":
            self._tpl_set_ic_weights(self.num_outputs_spin.value(), 100.0)
        if 'x_min' in t:
            self.x_min.setValue(t['x_min'])
            self.x_max.setValue(t['x_max'])
        # t range too, for templates whose t domain isn't the default 1.0
        # (e.g. Schrodinger's t in [0, pi/2]).
        self._tpl_set_time_range(t)

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
        # Non-zero/non-default Dirichlet BC values -- the generic branch
        # above only ever sets BC *type*, leaving whatever value was
        # already in the spinbox (a prior template's leftovers) in place,
        # so a template whose Dirichlet condition isn't 0 on both sides
        # has to set its values explicitly. 1D Burgers' u(-1,t)=u(1,t)=0
        # matches the spinbox default already, but is set explicitly too
        # so it's correct regardless of prior UI state.
        if text == "1D Burgers":
            for i in range(self.num_outputs_spin.value()):
                if i < len(self.bc_left_vals): self.bc_left_vals[i].setValue(0.0)
                if i < len(self.bc_right_vals): self.bc_right_vals[i].setValue(0.0)
        self._populate_locked_bc_entries_from_legacy(self.num_outputs_spin.value(), is_2d=False)
        if text == "1D Schrödinger":
            self._populate_schrodinger_bc_entries(self.num_outputs_spin.value())
        self._update_bc_mode_visibility()

        # Plot output: switch to "Custom..." and pre-fill the expression/
        # label for a template that defines one (currently just 1D
        # Schrodinger's |h|); reset to plain "Output 1" for every other
        # template, so a previous template's custom field never lingers
        # (same staleness class as the Error Analysis reset in
        # _auto_configure_ea -- see its docstring).
        if 'plot_custom_expr' in t:
            self.plot_output_combo.setCurrentIndex(self.plot_output_combo.count() - 1)  # "Custom..." is always last
            self.plot_custom_expr_input.setText(t['plot_custom_expr'])
            self.plot_custom_label_input.setText(t.get('plot_custom_label', ''))
        else:
            self.plot_output_combo.setCurrentIndex(0)
            self.plot_custom_expr_input.clear()
            self.plot_custom_label_input.clear()

        # Time-Adaptive default (e.g. 1D Allen-Cahn). The fallback row ends
        # at this template's own t_max (Schrodinger's ~1.5708, not 1.0).
        self._tpl_reset_time_adaptive(t.get('ta_default'), t.get('t_max', 1.0), "101")
        self._tpl_finish(text, t)

    def _on_loss_plot_settings(self):
        """Loss Plot Settings dialog (Round 28) -- the chat-agreed design:
        which losses to show (Train+Test totals / Individual components /
        All), a single display_every sampling cadence applied everywhere
        (per the user's "should be a single value that applies everywhere"
        -- replaces the old scattered 1000/200/500 constants across every
        model.train() call site, see codegen.py), line thickness, and a
        log/linear y-axis toggle (log matches every loss plot's previous,
        always-semilogy behavior). Backed by self._loss_plot_settings,
        read into PINNConfig by _build_config() and restored by
        _apply_config() -- same shape as self._plot_viz_settings and
        _on_plot_settings() just below, but a separate, self-contained
        dialog since Loss Plot Settings apply globally (not per plot-type
        like Surface/Line/GIF's own settings)."""
        current = self._loss_plot_settings

        dialog = QDialog(self)
        dialog.setWindowTitle("Loss Plot Settings")
        dialog.setMinimumWidth(340)
        layout = QVBoxLayout(dialog)

        info = QLabel(
            "Controls every loss plot in the app (live Solve, RAR, "
            "Time-Adaptive, and \"Export as DeepXDE Script\").")
        info.setWordWrap(True)
        self._register_hint(info, '#586e75')
        layout.addWidget(info)

        # Which losses to show
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Show:"))
        mode_combo = QComboBox()
        mode_combo.addItem("Train + Test totals", "train_test")
        mode_combo.addItem("Individual components", "individual")
        mode_combo.addItem("All (totals + individual)", "all")
        _mode_idx = {"train_test": 0, "individual": 1, "all": 2}.get(current.get('mode', 'train_test'), 0)
        mode_combo.setCurrentIndex(_mode_idx)
        mode_combo.setFixedWidth(190)
        mode_row.addStretch(); mode_row.addWidget(mode_combo)
        layout.addLayout(mode_row)

        mode_hint = QLabel(
            "\"Individual\" and \"All\" plot each PDE/BC/IC loss term "
            "separately (PDE terms named by output; others as "
            "\"Constraint N\") -- DeepXDE only tracks a single combined "
            "Test loss, so individual Test-term lines aren't available.")
        mode_hint.setWordWrap(True)
        self._register_hint(mode_hint, '#74c0fc')
        layout.addWidget(mode_hint)

        # display_every -- single shared value
        de_row = QHBoxLayout()
        de_row.addWidget(QLabel("Display every (iterations):"))
        de_spin = QSpinBox()
        de_spin.setRange(1, 100000)
        de_spin.setValue(int(current.get('display_every', 1000)))
        de_spin.setFixedWidth(90)
        de_row.addStretch(); de_row.addWidget(de_spin)
        layout.addLayout(de_row)

        de_hint = QLabel(
            "How often loss is recorded during training -- one shared "
            "value used everywhere (Adam, L-BFGS, RAR rounds, every "
            "Time-Adaptive step). Smaller = more detail, slower logging.")
        de_hint.setWordWrap(True)
        self._register_hint(de_hint, '#586e75')
        layout.addWidget(de_hint)

        # Line thickness
        lw_row = QHBoxLayout()
        lw_row.addWidget(QLabel("Line width:"))
        lw_combo = QComboBox()
        lw_combo.addItems(["1.0", "1.5", "2.0", "2.5", "3.0"])
        lw_combo.setCurrentText(str(current.get('linewidth', 2.0)))
        lw_combo.setFixedWidth(90)
        lw_row.addStretch(); lw_row.addWidget(lw_combo)
        layout.addLayout(lw_row)

        # Log / linear y-axis
        log_cb = QCheckBox("Logarithmic y-axis (semilogy)")
        log_cb.setChecked(bool(current.get('log_y', True)))
        layout.addWidget(log_cb)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout, dialog)

        def _on_ok():
            self._loss_plot_settings = {
                'mode': mode_combo.currentData(),
                'display_every': de_spin.value(),
                'linewidth': float(lw_combo.currentText()),
                'log_y': log_cb.isChecked(),
            }
            dialog.accept()

        ok_btn.clicked.connect(_on_ok)
        dialog.exec()

    def _build_plot_settings_fields(self, layout, current, viz_type, *, is_param, is_1d, is_2d, is_3d,
                                    is_steady, is_line_type, is_anim, surface_types, levels_default,
                                    steps_value, steps_max, slice_defaults):
        """Builds every field the two Plot Settings dialogs share --
        cleanup Round 57. The Setup tab's dialog (_on_plot_settings) and
        Restore & Visualize's (_on_restore_viz_settings) used to build these
        ~300 lines independently and had drifted: same fields in a different
        order, different labels ("Animation FPS:" vs "Frame rate (fps):"),
        different frame-rate choices. Now both show the same fields in the
        same order with the same labels; what genuinely differs between them
        comes in as arguments:
          surface_types -- the two Surface-type names each panel uses;
          levels_default -- the contour-level default when unset;
          steps_value/steps_max -- the steps/frames spinbox's start value and
              upper limit (the Setup tab writes it back to its own hidden
              spinboxes, which top out lower);
          slice_defaults -- {"y": (auto, value), "z": (auto, value)} for a
              2D/3D problem (z only in 3D), or None.
        Parameter-convergence types (is_param) only get figure size and the
        title/label/font fields, as before. Returns a namespace of the input
        widgets; _plot_settings_values() reads the shared ones back."""
        import types as _types
        w = _types.SimpleNamespace(y_auto_cb=None, y_spin=None, z_auto_cb=None, z_spin=None)

        def _row(label, widget, width):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            widget.setFixedWidth(width)
            row.addStretch(); row.addWidget(widget)
            return row

        def _combo(items, value):
            c = QComboBox()
            c.addItems(items)
            c.setCurrentText(value)
            return c

        def _add(item, always=False):
            if always or not is_param:
                if isinstance(item, QHBoxLayout):
                    layout.addLayout(item)
                else:
                    layout.addWidget(item)

        def _wrap(row):
            box = QWidget()
            row.setContentsMargins(0, 0, 0, 0)
            box.setLayout(row)
            return box

        # Colormap, contour levels, grid resolution, output DPI
        w.cmap_combo = _combo(["RdBu_r", "viridis", "plasma", "jet", "coolwarm", "inferno", "turbo", "seismic", "bwr"],
                              current.get('colormap', 'RdBu_r'))
        _add(_row("Colormap:", w.cmap_combo, 120))
        w.levels_spin = QSpinBox()
        w.levels_spin.setRange(5, 200); w.levels_spin.setValue(current.get('levels', levels_default))
        _add(_row("Contour levels:", w.levels_spin, 80))
        w.res_combo = _combo(["80", "100", "150", "200"], str(current.get('resolution', 100)))
        _add(_row("Grid resolution:", w.res_combo, 80))
        w.dpi_combo = _combo(["100", "150", "200", "300"], str(current.get('dpi', 100)))
        _add(_row("Output DPI:", w.dpi_combo, 80))

        # Figure size -- every viz type, Parameter Convergence included;
        # Error Analysis comparison grids size themselves and ignore it.
        w.figsize_combo = _combo(["Default", "Square", "Wide", "Custom"], current.get('figsize_mode', 'Default'))
        _add(_row("Figure size:", w.figsize_combo, 100), always=True)
        figsize_custom_widget = QWidget()
        figsize_custom_layout = QHBoxLayout(figsize_custom_widget)
        figsize_custom_layout.setContentsMargins(0, 0, 0, 0)
        w.figsize_w_spin = QDoubleSpinBox(); w.figsize_h_spin = QDoubleSpinBox()
        for _lbl, _sp, _key, _dflt in (("Width:", w.figsize_w_spin, 'figsize_w', 7.0),
                                       ("Height:", w.figsize_h_spin, 'figsize_h', 5.0)):
            figsize_custom_layout.addWidget(QLabel(_lbl))
            _sp.setRange(2.0, 30.0); _sp.setSingleStep(0.5)
            _sp.setValue(current.get(_key, _dflt)); _sp.setFixedWidth(70)
            figsize_custom_layout.addWidget(_sp)
        figsize_custom_layout.addStretch()
        figsize_custom_widget.setVisible(w.figsize_combo.currentText() == "Custom")
        w.figsize_combo.currentTextChanged.connect(lambda t: figsize_custom_widget.setVisible(t == "Custom"))
        _add(figsize_custom_widget, always=True)

        # Color range -- Surface types only
        color_range_widget = QWidget()
        cr_layout = QVBoxLayout(color_range_widget)
        cr_layout.setContentsMargins(0, 0, 0, 0)
        w.cr_auto_cb = QCheckBox("Auto color range")
        w.cr_auto_cb.setChecked(current.get('auto_range', True))
        cr_layout.addWidget(w.cr_auto_cb)
        cr_manual_widget = QWidget()
        cr_manual_layout = QHBoxLayout(cr_manual_widget)
        cr_manual_layout.setContentsMargins(0, 0, 0, 0)
        w.vmin_spin = QDoubleSpinBox(); w.vmax_spin = QDoubleSpinBox()
        for _lbl, _sp, _key, _dflt in (("vmin:", w.vmin_spin, 'vmin', -1.0), ("vmax:", w.vmax_spin, 'vmax', 1.0)):
            cr_manual_layout.addWidget(QLabel(_lbl))
            _sp.setRange(-1e6, 1e6); _sp.setValue(current.get(_key, _dflt)); _sp.setFixedWidth(80)
            cr_manual_layout.addWidget(_sp)
        cr_manual_widget.setVisible(not w.cr_auto_cb.isChecked())
        cr_layout.addWidget(cr_manual_widget)
        w.cr_auto_cb.stateChanged.connect(lambda st: cr_manual_widget.setVisible(st != 2))
        color_range_widget.setVisible(viz_type in surface_types)
        _add(color_range_widget)

        # 2D/3D time snapshots -- Surface only, and only for a time-dependent
        # 2D/3D problem (the multi-snapshot branch: several evenly spaced
        # heatmaps side by side). A steady problem or any 1D Surface (the
        # x-t heatmap already shows the whole time range) is one plot.
        w.snap_spin = QSpinBox()
        w.snap_spin.setRange(1, 10); w.snap_spin.setValue(current.get('n_2d_snapshots', 2))
        snap_widget = _wrap(_row("2D/3D time snapshots:", w.snap_spin, 80))
        snap_widget.setVisible(viz_type == "Surface" and (is_2d or is_3d) and not is_steady)
        _add(snap_widget)

        # Swap x/t axes -- 1D Surface types only (2D/3D Surface plots are
        # spatial snapshots with no x/t axis to swap); t on the x-axis by
        # default, same as the main Results panel's 1D Surface plot.
        w.swap_xt_cb = QCheckBox("Swap axes (x-axis = t, y-axis = x)")
        w.swap_xt_cb.setChecked(current.get('swap_xt', True))
        w.swap_xt_cb.setVisible(viz_type in surface_types and is_1d)
        _add(w.swap_xt_cb)

        # Colorbar -- Surface types only
        w.colorbar_cb = QCheckBox("Show colorbar")
        w.colorbar_cb.setChecked(current.get('colorbar', True))
        w.colorbar_cb.setVisible(viz_type in surface_types)
        _add(w.colorbar_cb)

        # Steps/frames -- every type except Surface (whose analogous
        # control is the snapshot count above)
        w.steps_spin = QSpinBox()
        w.steps_spin.setRange(2, steps_max); w.steps_spin.setValue(steps_value)
        steps_widget = _wrap(_row("Frames to show:" if is_anim else "Time steps to show:", w.steps_spin, 80))
        steps_widget.setVisible(viz_type != "Surface")
        _add(steps_widget)

        # y/z slice -- the two Line-type plots on a 2D/3D problem only:
        # they plot u vs x, so the rest of the domain is fixed at a slice.
        show_slice = is_line_type and (is_2d or is_3d)
        slice_info = QLabel("This plot type only shows u vs x -- pick where to slice the rest of the domain.")
        slice_info.setWordWrap(True)
        self._register_hint(slice_info, '#74c0fc')
        slice_info.setVisible(show_slice)
        _add(slice_info)

        def _slice_row(label, auto_default, val_default):
            row = QHBoxLayout()
            auto_cb = QCheckBox("Auto (domain midpoint)")
            auto_cb.setChecked(auto_default)
            row.addWidget(QLabel(f"{label} value:"))
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6); spin.setValue(val_default)
            spin.setFixedWidth(90)
            spin.setVisible(show_slice and not auto_cb.isChecked())
            row.addStretch(); row.addWidget(spin)
            auto_cb.stateChanged.connect(lambda st: spin.setVisible(show_slice and st != 2))
            auto_cb.setVisible(show_slice)
            _add(auto_cb)
            _add(row)
            return auto_cb, spin

        if slice_defaults and (is_2d or is_3d):
            w.y_auto_cb, w.y_spin = _slice_row("y", *slice_defaults["y"])
            if is_3d and "z" in slice_defaults:
                w.z_auto_cb, w.z_spin = _slice_row("z", *slice_defaults["z"])

        # Line width -- Line types only
        w.lw_combo = _combo(["1.0", "1.5", "2.0", "2.5", "3.0"], str(current.get(self._viz_linewidth_key(viz_type), 2.0)))
        lw_widget = _wrap(_row("Line width:", w.lw_combo, 80))
        lw_widget.setVisible(is_line_type)
        _add(lw_widget)

        # Frame rate -- the GIF animation types only
        w.fps_combo = _combo(["5", "8", "10", "15", "20", "24", "30"], str(current.get('fps', 10)))
        fps_widget = _wrap(_row("Frame rate (fps):", w.fps_combo, 80))
        fps_widget.setVisible(is_anim)
        _add(fps_widget)

        # Title / axis labels -- every viz type; blank keeps the default text
        labels_line = QLabel("Leave blank to keep the default title/axis labels.")
        self._register_hint(labels_line, '#586e75')
        labels_line.setWordWrap(True)
        _add(labels_line, always=True)
        w.title_edit = QLineEdit(); w.xlabel_edit = QLineEdit(); w.ylabel_edit = QLineEdit()
        for _lbl, _ed, _key in (("Title:", w.title_edit, 'title'), ("X-axis label:", w.xlabel_edit, 'xlabel'),
                                ("Y-axis label:", w.ylabel_edit, 'ylabel')):
            row = QHBoxLayout()
            row.addWidget(QLabel(_lbl))
            _ed.setText(current.get(_key, ''))
            _ed.setPlaceholderText("(default)")
            row.addWidget(_ed)
            _add(row, always=True)

        # Font sizes (Round 31) -- figure title, per-panel subplot title,
        # axis labels; every viz type
        w.title_fs_spin = QDoubleSpinBox(); w.subplot_title_fs_spin = QDoubleSpinBox(); w.axis_label_fs_spin = QDoubleSpinBox()
        for _lbl, _sp, _key, _dflt in (("Title font size:", w.title_fs_spin, 'title_fontsize', 13.0),
                                       ("Subplot title font size:", w.subplot_title_fs_spin, 'subplot_title_fontsize', 10.0),
                                       ("Axis label font size:", w.axis_label_fs_spin, 'axis_label_fontsize', 10.0)):
            _sp.setRange(6.0, 36.0); _sp.setSingleStep(0.5); _sp.setValue(current.get(_key, _dflt))
            _add(_row(_lbl, _sp, 70), always=True)
        return w

    def _plot_settings_values(self, w, current, viz_type, is_param):
        """The settings dict both Plot Settings dialogs save on OK, from the
        widgets _build_plot_settings_fields() made -- starting from a copy
        of the existing settings so keys this viz_type doesn't surface
        (e.g. the other plot type's linewidth) survive. Each dialog then
        adds what only it handles (steps/slice write-back)."""
        new_settings = dict(current)
        if not is_param:
            new_settings.update({
                'colormap': w.cmap_combo.currentText(),
                'levels': w.levels_spin.value(),
                'resolution': int(w.res_combo.currentText()),
                'dpi': int(w.dpi_combo.currentText()),
                'auto_range': w.cr_auto_cb.isChecked(),
                'vmin': w.vmin_spin.value(),
                'vmax': w.vmax_spin.value(),
                'colorbar': w.colorbar_cb.isChecked(),
                'n_2d_snapshots': w.snap_spin.value(),
                'fps': int(w.fps_combo.currentText()),
                'swap_xt': w.swap_xt_cb.isChecked(),
            })
            new_settings[self._viz_linewidth_key(viz_type)] = float(w.lw_combo.currentText())
        new_settings.update({
            'title': w.title_edit.text().strip(),
            'xlabel': w.xlabel_edit.text().strip(),
            'ylabel': w.ylabel_edit.text().strip(),
            'figsize_mode': w.figsize_combo.currentText(),
            'figsize_w': w.figsize_w_spin.value(),
            'figsize_h': w.figsize_h_spin.value(),
            'title_fontsize': w.title_fs_spin.value(),
            'subplot_title_fontsize': w.subplot_title_fs_spin.value(),
            'axis_label_fontsize': w.axis_label_fs_spin.value(),
        })
        return new_settings


    def _on_plot_settings(self, viz_type=None):
        """Unified Plot Settings dialog for the Setup tab's "Plot output"
        panel -- merges what used to be two separate dialogs (this one,
        opened via a standalone "⚙" button that's now removed, plus
        _on_line_plot_settings's own steps/slice-only popup opened
        automatically for just the line/animation types) into one dialog
        that auto-pops for every plot-type selection, gating each field's
        visibility by viz_type/dimension -- the same single-dialog shape
        _on_restore_viz_settings already uses for the Restore & Visualize
        panel, so both panels show the exact same settings for a given
        plot type. (plot_type_combo never actually contains "📊 Error
        Analysis" -- see the dead-code removal note above
        _add_ta_step_group -- the real Error Analysis entry point is the
        separate ea_btn -> _on_error_analysis_btn().)
        """
        if viz_type is None:
            viz_type = self.plot_type_combo.currentText()

        # "Parameter Convergence" isn't a spatial plot at all -- it's the
        # iteration-vs-inferred-value chart for an Inverse problem (see
        # codegen.py's own Parameter Convergence branch) -- so, exactly
        # like Restore & Visualize's own is_param gating for its
        # "Parameter Convergence Plot/Animation" entries, almost every
        # field below (colormap, contour levels, resolution, DPI, color
        # range, 2D snapshots, swap axes, colorbar, line width, steps,
        # fps) means nothing for it and stays hidden; only figure size and
        # the title/axis-label overrides still apply.
        is_param = (viz_type == "Parameter Convergence")
        _is_1d = self.radio_1d.isChecked()
        _is_2d = self.radio_2d.isChecked()
        _is_3d = self.radio_3d.isChecked()
        _is_steady = bool(self.steady_state_check.isChecked()) if hasattr(self, 'steady_state_check') else False
        _is_anim = "Animation" in viz_type
        _is_line_type = viz_type in ("Line (time steps)", "Line Animation (GIF)")

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Plot Settings — {viz_type}")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        current = self._plot_viz_settings
        # The steps/frames value and the y/z slice live in the Setup tab's
        # own (hidden) widgets -- timesteps_spin_line/_anim and
        # line_slice_*_auto_cb/_spin -- which the generators read via
        # PINNConfig; this dialog edits those.
        _target_spin = self.timesteps_spin_anim if _is_anim else self.timesteps_spin_line
        _slice = None
        if _is_2d or _is_3d:
            _slice = {"y": (self.line_slice_y_auto_cb.isChecked(), self.line_slice_y_spin.value())}
            if _is_3d:
                _slice["z"] = (self.line_slice_z_auto_cb.isChecked(), self.line_slice_z_spin.value())
        # Shared with Restore & Visualize's dialog -- see _build_plot_settings_fields.
        w = self._build_plot_settings_fields(
            layout, current, viz_type, is_param=is_param, is_1d=_is_1d, is_2d=_is_2d, is_3d=_is_3d,
            is_steady=_is_steady, is_line_type=_is_line_type, is_anim=_is_anim,
            surface_types=("Surface", "Surface Animation (GIF)"), levels_default=50,
            steps_value=_target_spin.value(), steps_max=_target_spin.maximum(), slice_defaults=_slice)

        ok_btn, cancel_btn = self._add_ok_cancel_row(layout)

        def _on_cancel():
            # Restore whichever plot type was actually selected before
            # this dialog opened -- not a hardcoded "Surface" -- so
            # reconsidering a different plot type and then Cancelling
            # lands back on the real previous selection, the same Cancel
            # behavior every other Settings dialog in this app already
            # has. blockSignals avoids re-triggering _on_plot_type_changed
            # (and re-opening this same dialog) for the revert itself.
            self.plot_type_combo.blockSignals(True)
            self.plot_type_combo.setCurrentText(getattr(self, '_plot_type_prev', 'Surface'))
            self.plot_type_combo.blockSignals(False)
            dialog.reject()

        cancel_btn.clicked.connect(_on_cancel)

        def _on_ok():
            new_settings = self._plot_settings_values(w, current, viz_type, is_param)
            if not is_param:
                _target_spin.setValue(w.steps_spin.value())
                if w.y_auto_cb is not None:
                    self.line_slice_y_auto_cb.setChecked(w.y_auto_cb.isChecked())
                    self.line_slice_y_spin.setValue(w.y_spin.value())
                if w.z_auto_cb is not None:
                    self.line_slice_z_auto_cb.setChecked(w.z_auto_cb.isChecked())
                    self.line_slice_z_spin.setValue(w.z_spin.value())
            self._plot_viz_settings = new_settings
            self._plot_type_prev = self.plot_type_combo.currentText()
            self.log_box.append(f"✅ Plot settings saved — {viz_type}")
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
        # Round 31: default changed to L2-only (previously all three
        # were checked by default even though none of them had any
        # effect -- see PINNConfig.ea_metric_l2/mse/max's own comment).
        self._ea_l2_cb  = QCheckBox("L2 Relative"); self._ea_l2_cb.setChecked(True)
        self._ea_mse_cb = QCheckBox("MSE");          self._ea_mse_cb.setChecked(False)
        self._ea_max_cb = QCheckBox("Max Error");    self._ea_max_cb.setChecked(False)
        for w in [self._ea_l2_cb, self._ea_mse_cb, self._ea_max_cb]:
            metrics_layout.addWidget(w)
        layout.addWidget(metrics_group)

        # ── Reference source (Round 67): files, or an exact formula ──
        _prev_ea = getattr(self, '_ea_settings', None) or {}
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Reference:"))
        self._ea_ref_mode_combo = QComboBox()
        self._ea_ref_mode_combo.addItem("Reference files", "files")
        self._ea_ref_mode_combo.addItem("Exact solution formula", "formula")
        self._ea_ref_mode_combo.setFixedHeight(26)
        src_row.addWidget(self._ea_ref_mode_combo); src_row.addStretch()
        layout.addLayout(src_row)

        formula_group = QGroupBox("Exact Solution Formula")
        fg_layout = QVBoxLayout(formula_group)
        fg_hint = QLabel("One formula per output in x, y, z, t -- e.g. exp(-t)*sin(pi*x). Functions: sin, cos, exp, "
                         "log, sqrt, tanh, erf, norm_cdf, pi. Leave an output blank to skip it. The reference values "
                         "are computed on a grid over the domain, so no files are needed.")
        fg_hint.setWordWrap(True)
        self._register_hint(fg_hint, '#586e75')
        fg_layout.addWidget(fg_hint)
        _n_out_f = self.num_outputs_spin.value() if hasattr(self, 'num_outputs_spin') else 1
        _prev_formulas = (_prev_ea.get('ref_formula', '') or '').split('|')
        self._ea_formula_edits = []
        for _i in range(_n_out_f):
            _row = QHBoxLayout()
            _nm = self.output_name_inputs[_i].text() if _i < len(self.output_name_inputs) else f"u{_i+1}"
            _row.addWidget(QLabel(f"{_nm} ="))
            _ed = QLineEdit(_prev_formulas[_i] if _i < len(_prev_formulas) else "")
            _ed.setFixedHeight(26)
            _row.addWidget(_ed)
            fg_layout.addLayout(_row)
            self._ea_formula_edits.append(_ed)
        _tr = QHBoxLayout()
        _tr.addWidget(QLabel("Compare at times:"))
        self._ea_ref_times_edit = QLineEdit(_prev_ea.get('ref_times', '') or '')
        self._ea_ref_times_edit.setPlaceholderText("e.g. 0.25, 0.5, 1  (blank = 5 evenly spaced)")
        self._ea_ref_times_edit.setFixedHeight(26)
        _tr.addWidget(self._ea_ref_times_edit)
        _tr.addWidget(QLabel("Grid:"))
        self._ea_ref_grid_spin = QSpinBox()
        self._ea_ref_grid_spin.setRange(0, 2001)
        self._ea_ref_grid_spin.setSpecialValueText("auto")
        self._ea_ref_grid_spin.setValue(int(_prev_ea.get('ref_grid', 0) or 0))
        self._ea_ref_grid_spin.setFixedHeight(26); self._ea_ref_grid_spin.setFixedWidth(70)
        _tr.addWidget(self._ea_ref_grid_spin)
        fg_layout.addLayout(_tr)
        layout.addWidget(formula_group)

        # ── Reference files ────────────────────────────────────
        files_group = QGroupBox("Reference Files (FEM/Exact)")
        files_layout = QVBoxLayout(files_group)

        def _sync_ref_mode(_i=None):
            _is_f = self._ea_ref_mode_combo.currentData() == "formula"
            formula_group.setVisible(_is_f)
            files_group.setVisible(not _is_f)
        self._ea_ref_mode_combo.currentIndexChanged.connect(_sync_ref_mode)
        self._ea_ref_mode_combo.setCurrentIndex(1 if _prev_ea.get('ref_mode') == "formula" else 0)
        _sync_ref_mode()

        hint = QLabel("Format: space-separated, 3 columns: x  t  u  (no header)\nEach file = one time snapshot.")
        self._register_hint(hint, '#586e75')
        files_layout.addWidget(hint)

        # File list widget
        self._ea_file_list_widget = QWidget()
        self._ea_file_list_layout = QVBoxLayout(self._ea_file_list_widget)
        self._ea_file_list_layout.setSpacing(4)
        self._ea_file_list_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(self._ea_file_list_widget)

        self._ea_file_rows = []  # list of (path_label, t_label, remove_btn)

        # A model with more than one output is ambiguous about which
        # output a reference file belongs to -- show an "Output:" picker
        # per row in that case (real output names, or a custom derived
        # expression reusing the same mechanism as the Results panel's
        # "Custom..." plot field). A single-output model has nothing to
        # disambiguate, so the dialog stays exactly as before for it.
        n_out = self.num_outputs_spin.value() if hasattr(self, 'num_outputs_spin') else 1
        try:
            out_names = [(w.text().strip() or f"output{_oi}")
                         for _oi, w in enumerate(self.output_name_inputs)]
        except Exception:
            out_names = []
        if not out_names:
            out_names = ["u"]

        def _add_file_row(path='', t_val=None, sel=None):
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
            self._register_hint(t_label, '#69db7c', 'min-width: 80px; ')
            row_layout.addWidget(t_label)

            out_combo = None
            expr_edit = None
            label_edit = None
            if n_out > 1:
                out_combo = QComboBox()
                out_combo.setFixedHeight(26)
                for _on in out_names:
                    out_combo.addItem(_on)
                out_combo.addItem("Custom expression...")
                row_layout.addWidget(out_combo)

                expr_edit = QLineEdit()
                expr_edit.setPlaceholderText("expression, e.g. sqrt(u**2+v**2)")
                expr_edit.setFixedHeight(26)
                expr_edit.setVisible(False)
                row_layout.addWidget(expr_edit)

                label_edit = QLineEdit()
                label_edit.setPlaceholderText("label (optional)")
                label_edit.setFixedHeight(26)
                label_edit.setFixedWidth(90)
                label_edit.setVisible(False)
                row_layout.addWidget(label_edit)

                def _on_out_combo_changed(_txt, _ee=expr_edit, _le=label_edit):
                    _is_custom = (_txt == "Custom expression...")
                    _ee.setVisible(_is_custom)
                    _le.setVisible(_is_custom)
                out_combo.currentTextChanged.connect(_on_out_combo_changed)

                # Pre-select from an existing selector -- auto-configured
                # from the template's own ref_dir, or a previously-run
                # dialog's settings being reopened.
                if isinstance(sel, int) and 0 <= sel < len(out_names):
                    out_combo.setCurrentIndex(sel)
                elif isinstance(sel, (list, tuple)) and len(sel) >= 1:
                    out_combo.setCurrentText("Custom expression...")
                    expr_edit.setText(str(sel[0]))
                    expr_edit.setVisible(True)
                    if len(sel) > 1 and sel[1]:
                        label_edit.setText(str(sel[1]))
                        label_edit.setVisible(True)
                elif sel is None:
                    # sel=None means "delegate to the Results panel's own
                    # plot field" (see codegen.py's _ea_extract: _ea_sel is
                    # None -> _extract_plot_field). But this combo has no
                    # "delegate" item of its own, so leaving it untouched
                    # just silently sits at index 0 (whatever out_names[0]
                    # happens to be -- "u" for Schrodinger) by accident of
                    # Qt's own default, not by any deliberate choice. That's
                    # not just cosmetic: clicking "Run Error Analysis"
                    # always freezes the combo's *current* text into a
                    # concrete selector via _row_selector() below (reading
                    # back "u" produces sel=0, a raw-output-only comparison)
                    # -- so an auto-populated row the user never touches
                    # would silently replace the correct delegate-to-
                    # Custom-expression comparison (e.g. Schrodinger's
                    # sqrt(u**2+v**2) / "|h|") with a bare "u" comparison
                    # the instant Run is clicked. Mirror the Results panel's
                    # CURRENT plot-field choice here instead, so the combo
                    # shows -- and, if Run is clicked without changes,
                    # produces -- the exact same comparison target that
                    # sel=None would have delegated to anyway.
                    _po_combo = getattr(self, 'plot_output_combo', None)
                    _po_is_custom = (_po_combo is not None
                                      and _po_combo.currentText() == "Custom...")
                    _po_expr = (self.plot_custom_expr_input.text().strip()
                                if hasattr(self, 'plot_custom_expr_input') else "")
                    if _po_is_custom and _po_expr:
                        out_combo.setCurrentText("Custom expression...")
                        expr_edit.setText(_po_expr)
                        expr_edit.setVisible(True)
                        _po_label = (self.plot_custom_label_input.text().strip()
                                     if hasattr(self, 'plot_custom_label_input') else "")
                        if _po_label:
                            label_edit.setText(_po_label)
                            label_edit.setVisible(True)
                    elif (_po_combo is not None
                          and 0 <= _po_combo.currentIndex() < len(out_names)):
                        out_combo.setCurrentIndex(_po_combo.currentIndex())
                    # else: no Results-panel reference available (shouldn't
                    # normally happen) -- leave the combo at its implicit
                    # default, same as before this fix.

            remove_btn = QPushButton("✕")
            remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
            remove_btn.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
            row_layout.addWidget(remove_btn)

            self._ea_file_list_layout.addWidget(row_widget)
            row_data = {'widget': row_widget, 'path': path_edit, 't_label': t_label,
                        'out_combo': out_combo, 'expr_edit': expr_edit, 'label_edit': label_edit}
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

        def _row_selector(row):
            # Reads one file row's UI back into an output_selector value
            # (None / int / [expr, label]) -- the mirror of the pre-select
            # logic in _add_file_row above.
            _oc = row.get('out_combo')
            if _oc is None:
                return None
            _txt = _oc.currentText()
            if _txt == "Custom expression...":
                _expr = (row['expr_edit'].text() if row.get('expr_edit') else "").strip()
                if not _expr:
                    return None
                _lbl = (row['label_edit'].text() if row.get('label_edit') else "").strip()
                return [_expr, _lbl]
            return out_names.index(_txt) if _txt in out_names else None

        # Auto-populate from the template's own auto-configured Error
        # Analysis settings when available (already carries the right
        # output selector per file, e.g. 2D Burgers (Mathias)'s u/v split)
        # -- falling back to a plain glob of the template's ref_dir, then
        # to one empty row, for a problem with no auto-config at all.
        _existing_ea = getattr(self, '_ea_settings', None)
        ref_dir = getattr(self, '_template_ref_dir', '')
        if _existing_ea and _existing_ea.get('files'):
            hint2 = QLabel("📂 Auto-loaded from template" + (f": {os.path.basename(ref_dir)}" if ref_dir else ""))
            self._register_hint(hint2, '#a0c4ff')
            files_layout.addWidget(hint2)
            for _ea_entry in _existing_ea['files']:
                _tv0, _fp0 = _ea_entry[0], _ea_entry[1]
                _sel0 = _ea_entry[2] if len(_ea_entry) > 2 else None
                _add_file_row(_fp0, _tv0, _sel0)
        elif ref_dir and os.path.isdir(ref_dir):
            txt_files = sorted(glob.glob(os.path.join(ref_dir, 't_*.txt')))
            if txt_files:
                hint2 = QLabel(f"📂 Auto-loaded from template: {os.path.basename(ref_dir)}")
                self._register_hint(hint2, '#a0c4ff')
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

            if self._ea_ref_mode_combo.currentData() == "formula":
                _formulas = [e.text().strip() for e in self._ea_formula_edits]
                if not any(_formulas):
                    self.log_box.append("❌ Error Analysis: enter an exact-solution formula for at least one output.")
                    return
                self._ea_settings = {
                    'files': [],
                    'ref_mode': 'formula',
                    'ref_formula': "|".join(_formulas),
                    'ref_times': self._ea_ref_times_edit.text().strip(),
                    'ref_grid': self._ea_ref_grid_spin.value(),
                    'do_line': self._ea_line_cb.isChecked(),
                    'do_surface': self._ea_surface_cb.isChecked(),
                    'do_l2': self._ea_l2_cb.isChecked(),
                    'do_mse': self._ea_mse_cb.isChecked(),
                    'do_max': self._ea_max_cb.isChecked(),
                }
                self.log_box.append("✅ Error Analysis configured — exact-solution formula, will run after training.")
                dialog.accept()
                return

            # Collect valid file rows
            valid_files = []
            for row in self._ea_file_rows:
                p = row['path'].text().strip()
                t = row.get('t_val', None)
                if p and os.path.exists(p) and t is not None:
                    if n_out > 1:
                        valid_files.append((t, p, _row_selector(row)))
                    else:
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
                 for i in range(self._n_equations())]
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
        return json.dumps(phases)

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

    def _parse_monitor_points_text(self, text):
        """Parses a Training Monitor row's points field: a semicolon-
        separated list of parenthesized coordinate tuples, e.g.
        "(0.3, 0.0); (0.5, 0.0)". Uses ast.literal_eval (never bare eval)
        wrapped in a list, mirroring the same safe-literal-only parsing
        convention _parse_vertex_list()/ea_files already use elsewhere in
        this codebase. Returns a list of plain lists of floats (JSON-
        serializable), or [] if the text is empty or fails to parse."""
        import ast
        text = (text or "").strip()
        if not text:
            return []
        try:
            raw = ast.literal_eval("[" + text.replace(";", ",") + "]")
        except (ValueError, SyntaxError, TypeError):
            return []
        if not isinstance(raw, (list, tuple)):
            return []
        points = []
        for p in raw:
            if isinstance(p, (list, tuple)):
                points.append([float(c) for c in p])
            else:
                points.append([float(p)])
        return points

    def _format_monitor_points_text(self, points):
        """Inverse of _parse_monitor_points_text() -- rebuilds the row's
        display text from a parsed points list (loading a saved config)."""
        if not points:
            return ""
        return "; ".join(
            "(" + ", ".join(str(c) for c in p) + ")"
            for p in points if isinstance(p, (list, tuple))
        )

    def _add_training_monitor_row(self, name="", points_text="", expr="", period=1000):
        """Adds one Training Monitor row -- mirrors _add_scheduler_phase()'s
        pattern exactly (a removable QWidget row appended to a container
        layout, its field widgets kept in a dict appended to
        self.training_monitor_list, read later by
        _build_training_monitors_json())."""
        row_widget = QWidget()
        row_layout = QVBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(3)

        header_row = QHBoxLayout()
        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("label, e.g. u_check")
        header_row.addWidget(name_edit)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(22); remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #ff8787; background: transparent; border: none; }")
        header_row.addWidget(remove_btn)
        row_layout.addLayout(header_row)

        points_edit = QLineEdit(points_text)
        points_edit.setPlaceholderText("points, e.g. (0.3, 0.0); (0.5, 0.0)")
        row_layout.addWidget(points_edit)

        expr_edit = QLineEdit(expr)
        expr_edit.setPlaceholderText("expression(s), e.g. u, du_x, du_xx")
        row_layout.addWidget(expr_edit)

        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("Every (iterations):"))
        period_spin = QSpinBox()
        period_spin.setRange(1, 1000000); period_spin.setSingleStep(100)
        period_spin.setValue(int(period) if period else 1000)
        period_spin.setFixedHeight(24)
        period_row.addStretch(); period_row.addWidget(period_spin)
        row_layout.addLayout(period_row)

        sep_line = QLabel()
        sep_line.setFixedHeight(1)
        sep_line.setStyleSheet("background-color: #3a3a4a;")
        row_layout.addWidget(sep_line)

        self.tmon_rows_layout.addWidget(row_widget)
        row_data = {
            'widget': row_widget, 'name': name_edit,
            'points': points_edit, 'expr': expr_edit, 'period': period_spin,
        }
        self.training_monitor_list.append(row_data)

        def _remove():
            if row_data in self.training_monitor_list:
                self.training_monitor_list.remove(row_data)
            row_widget.deleteLater()
        remove_btn.clicked.connect(_remove)
        return row_data

    def _build_training_monitors_json(self):
        """Builds config.py's training_monitors JSON string from
        self.training_monitor_list -- a blank row (no name/points/expr
        typed at all) is skipped rather than written out as a half-empty
        entry, same leniency _build_custom_geom_shapes_json() etc. use."""
        import json
        monitors = []
        for row in self.training_monitor_list:
            name = row['name'].text().strip()
            points_text = row['points'].text().strip()
            expr = row['expr'].text().strip()
            if not (name or points_text or expr):
                continue
            monitors.append({
                "name": name,
                "points": self._parse_monitor_points_text(points_text),
                "expr": expr,
                "period": int(row['period'].value()),
            })
        return json.dumps(monitors)

    def _apply_training_monitors_json(self, training_monitors_json):
        """Rebuilds self.training_monitor_list's rows from saved JSON --
        mirrors _apply_scheduler_phases_build_only()'s clear-then-rebuild
        pattern."""
        import json
        for row in list(self.training_monitor_list):
            row['widget'].deleteLater()
        self.training_monitor_list.clear()
        try:
            monitors = json.loads(training_monitors_json) if training_monitors_json else []
        except (json.JSONDecodeError, TypeError):
            monitors = []
        if not isinstance(monitors, list):
            return
        for m in monitors:
            if not isinstance(m, dict):
                continue
            self._add_training_monitor_row(
                name=m.get('name', '') or '',
                points_text=self._format_monitor_points_text(m.get('points') or []),
                expr=m.get('expr', '') or '',
                period=m.get('period', 1000) or 1000,
            )

    def _make_lr_decay_widget(self):
        """The "Learning rate decay" type combo and its one or two
        parameters (step size / gamma, T_max / eta min, ...), shared by
        every Training Phase and by IC-guided pre-training. Returns
        (widget, type_combo, p1_spin, p2_spin)."""
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
        # DeepXDE's "inverse time": lr / (1 + rate * step / steps) (Round 64).
        decay_type_combo.addItem("Inverse time", "inverse time")
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
            elif kind == "inverse time":
                l1.setText("Decay steps:"); s1.setRange(1, 500000); s1.setDecimals(0)
                l2.setText("Decay rate:"); r2widget_visible = True
            else:
                r2widget_visible = False
            s1.setVisible(kind != "none"); l1.setVisible(kind != "none")
            for i in range(r2.count()):
                item = r2.itemAt(i).widget()
                if item is not None:
                    item.setVisible(r2widget_visible)
        decay_type_combo.currentIndexChanged.connect(_update_decay_fields)
        _update_decay_fields()
        return decay_widget, decay_type_combo, decay_p1_spin, decay_p2_spin

    def _add_scheduler_phase(self, optimizer='adam', iterations=50000, lr=0.001):
        phase_num = len(self.sched_phase_list) + 1  # phase 1, 2, 3...
        phase_widget = QWidget()
        phase_layout = QVBoxLayout(phase_widget)
        phase_layout.setSpacing(3)
        phase_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_row = QHBoxLayout()
        header_lbl = QLabel(f"── Phase {phase_num} ──")
        self._register_hint(header_lbl, '#a0c4ff')
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
        opt_combo.setFixedHeight(26)
        self._fit_combo_width(opt_combo, min_width=80)
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
        decay_widget, decay_type_combo, decay_p1_spin, decay_p2_spin = self._make_lr_decay_widget()
        phase_layout.addWidget(decay_widget)
        decay_widget.setVisible(_lr_capable)

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
        loss_combo_ph.setFixedHeight(26)
        self._fit_combo_width(loss_combo_ph, min_width=160)
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

    def _on_browse_restore_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select model file", "", "PyTorch model (*.pt)")
        if f:
            self.restore_model_path.setText(f)
            base = os.path.dirname(f)
            # First try: same name as .pt but .json
            specific = f.replace(".pt", ".json")
            if os.path.exists(specific):
                self.restore_config_path.setText(specific)
                self.log_box.append(f"✅ Auto-detected config: {specific}")
            elif os.path.exists(os.path.join(base, "model_config.json")):
                self.restore_config_path.setText(os.path.join(base, "model_config.json"))
                self.log_box.append(f"✅ Auto-detected config (generic): {os.path.join(base, 'model_config.json')}")
            else:
                # A Time-Adaptive step folder has no model_config.json of
                # its own (that only lives in the run's top-level
                # solution_results/ folder) -- but it does have its own
                # step_config.json, one level up from neither of the two
                # checks above, which already has THIS step's correct
                # t_min/t_max/x_min/x_max/... (the run's top-level
                # model_config.json has the FULL domain's bounds instead,
                # which would silently let a restore evaluate this step's
                # model outside the only window it was ever trained on).
                step_cfg_fallback = os.path.join(base, "step_config.json")
                if os.path.exists(step_cfg_fallback):
                    self.restore_config_path.setText(step_cfg_fallback)
                    self.log_box.append(f"✅ Auto-detected Time-Adaptive step config: {step_cfg_fallback}")
                else:
                    self.log_box.append("⚠️ No config found — please browse manually.")

            self._restore_ta_steps = self._detect_restore_ta_steps(f)
            if self._restore_ta_steps:
                n_found = len(self._restore_ta_steps)
                t0_full = self._restore_ta_steps[0]['t0']
                t1_full = self._restore_ta_steps[-1]['t1']
                self.restore_ta_info_label.setText(
                    f"⏱ Detected {n_found} Time-Adaptive steps (t={t0_full:.4g} → {t1_full:.4g}) "
                    "in this run folder. Surface/Line snapshots now auto-route to whichever "
                    "step actually covers the requested time; check \"Combine steps\" below "
                    "for a full-domain animation.")
                self.restore_ta_info_label.setVisible(True)
                self.restore_ta_combine_cb.setChecked(True)
                self.log_box.append(f"✅ Detected {n_found} Time-Adaptive steps (t={t0_full:.4g} → {t1_full:.4g})")
            else:
                self.restore_ta_info_label.setVisible(False)
            self._update_restore_ta_combine_visibility()

            self._refresh_restore_output_combo()
            self._refresh_restore_viz_options()
            self._auto_detect_restore_optimizer()

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
                import json as _json_brm
                run_dir = os.path.dirname(base) if os.path.basename(base) == "solution_results" else base
                conv_files = sorted(_glob_brm.glob(os.path.join(run_dir, "*_convergence.txt")))
                if conv_files:
                    # names[i] stays aligned with conv_files[i] (an empty
                    # string for a file with no usable header), unlike the
                    # restore_inv_var_names text below which -- to keep its
                    # own prior behavior -- only ever joins the names that
                    # were actually found. The aligned version is what lets
                    # true_by_name below match each row to the right file
                    # even when some file in the middle lacks a header.
                    names = []
                    for cf in conv_files:
                        name = ""
                        try:
                            with open(cf) as _cf:
                                header = _cf.readline().strip().split(",")
                            if len(header) >= 2 and header[1].strip():
                                name = header[1].strip()
                        except Exception:
                            pass
                        names.append(name)

                    # Auto-fill each row's True-value field from this run's
                    # own model_config.json ("inverse_variables": [{"name",
                    # "init", "true"}, ...] -- "true" added specifically for
                    # this) when available, so the plot/GIF can draw the
                    # same dashed true-value line the training-time plot
                    # already does. An older run saved before this field
                    # existed, or a manually-added variable with no known
                    # ground truth, simply leaves it blank here too --
                    # exactly as if the user had left it blank themselves.
                    true_by_name = {}
                    try:
                        with open(self.restore_config_path.text().strip()) as _mcf:
                            _mc = _json_brm.load(_mcf)
                        for _iv in _mc.get("inverse_variables", []) or []:
                            if isinstance(_iv, dict) and _iv.get("name") and _iv.get("true") is not None:
                                true_by_name[_iv["name"]] = _iv["true"]
                    except Exception:
                        pass
                    true_vals = [true_by_name.get(n) if n else None for n in names]

                    self._set_restore_param_rows(conv_files, true_vals)
                    self.log_box.append(f"✅ Auto-detected {len(conv_files)} parameter convergence file(s) in {run_dir}")
                    if any(v is not None for v in true_vals):
                        self.log_box.append("✅ Auto-filled true parameter value(s) from model_config.json")
                    if any(names) and not self.restore_inv_var_names.text().strip():
                        self.restore_inv_var_names.setText(",".join(n for n in names if n))

    def _on_browse_restore_config(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select config file", "", "JSON (*.json)")
        if f:
            self.restore_config_path.setText(f)
            self._refresh_restore_output_combo()
            self._refresh_restore_viz_options()
            self._auto_detect_restore_optimizer()

    def _auto_detect_restore_optimizer(self):
        """Pre-select restore_optimizer_combo from the model_config.json
        about to be restored, instead of leaving it at whatever it already
        was (Adam by default) and relying on the user to notice and change
        it themselves -- picking the wrong one here is a real footgun for
        anyone new to this panel, since model.restore() needs the model
        compiled with a matching optimizer or it errors.

        Reads "optimizer2" first (a 2-phase legacy scheduler's SECOND
        phase -- e.g. Adam then L-BFGS -- is the one a saved checkpoint's
        weights actually match, the same "last phase wins" reasoning
        generate_script() itself already applies when deciding what to
        compile with right before saving), falling back to "optimizer"
        when there's no second phase ("none", or the key is missing
        entirely on an older save). Only acts when the detected value is
        one restore_optimizer_combo actually offers (currently "adam" /
        "lbfgs" -- an NNCG-trained run has no matching item yet, so it's
        safely left alone rather than guessing). Still fully overridable
        by hand afterward -- this only changes the starting selection.
        Safe no-op if the config can't be read yet, same pattern as
        _refresh_restore_output_combo."""
        import json
        try:
            with open(self.restore_config_path.text().strip()) as f:
                cfg = json.load(f)
        except Exception:
            return
        opt2 = str(cfg.get("optimizer2", "none") or "none").strip().lower()
        opt1 = str(cfg.get("optimizer", "") or "").strip().lower()
        detected = opt2 if opt2 not in ("", "none") else opt1
        if not hasattr(self, 'restore_optimizer_combo'):
            return
        idx = self.restore_optimizer_combo.findData(detected)
        if idx < 0:
            return
        if self.restore_optimizer_combo.currentIndex() != idx:
            self.restore_optimizer_combo.setCurrentIndex(idx)
            self.log_box.append(
                f"✅ Auto-detected optimizer from model_config.json: "
                f"{self.restore_optimizer_combo.currentText()}")

    def _refresh_restore_output_combo(self):
        """Repopulate restore_output_combo from the config file about to
        be restored (its own num_outputs/output_names), not the main
        Setup tab's current output count -- the two can easily disagree
        (e.g. the Setup tab left at 1 output while restoring an older
        2-output checkpoint, which is exactly why this only ever showed
        "Output 1 (u)" before), same class of fix as _restore_dim already
        reading the model being restored's own problem_dim instead of
        trusting the Setup tab's dimension radios. Safe no-op (leaves the
        combo as whatever it already was) if the config can't be read
        yet -- e.g. no file picked, or Browse cancelled."""
        import json
        try:
            with open(self.restore_config_path.text().strip()) as f:
                cfg = json.load(f)
        except Exception:
            return
        n_out = cfg.get("num_outputs", 1)
        out_names = [n.strip() for n in str(cfg.get("output_names", "u")).split(",")]
        self.restore_output_combo.clear()
        for i in range(n_out):
            name = out_names[i] if i < len(out_names) and out_names[i] else f"u{i + 1}"
            self.restore_output_combo.addItem(f"Output {i + 1} ({name})")
        self.restore_output_combo.addItem("Custom...")

    def _detect_restore_ta_steps(self, model_path):
        """If `model_path` lives inside a '.../time_adaptive_steps/
        step_NNN_t{t0}_to_t{t1}/' folder (the layout generate_script()
        saves one checkpoint per Time-Adaptive step into -- see the
        matching _step_cfg comment in codegen.py), return every sibling
        step's own info, sorted by t0. A Time-Adaptive run's model is
        really N separate models, each only valid on its own narrow
        [t0, t1] window, never the training's full [t_min, t_max].
        Returns [] if model_path isn't inside such a folder, or if fewer
        than 2 usable sibling steps are found (nothing to combine, and a
        single "step" folder isn't meaningfully different from an
        ordinary restore).

        Each returned dict: {'dir', 't0', 't1', 'model_path', 'cfg'} --
        'cfg' is that step's own step_config.json content when it has
        one (t_min/t_max already equal to t0/t1 there); a step missing
        its own file (shouldn't normally happen -- generate_script()
        writes one for every step -- but an older or hand-edited run
        might not have it) falls back to parsing t0/t1 from the folder
        name itself, with every other field copied from whatever's
        currently in self.restore_config_path (best-effort only)."""
        import json
        step_dir = os.path.dirname(model_path)
        step_base = os.path.basename(step_dir)
        parent_dir = os.path.dirname(step_dir)
        if os.path.basename(parent_dir) != "time_adaptive_steps" or not step_base.startswith("step_"):
            return []

        import glob as _glob_ta
        candidate_dirs = sorted(
            d for d in _glob_ta.glob(os.path.join(parent_dir, "step_*")) if os.path.isdir(d))
        if len(candidate_dirs) < 2:
            return []

        fallback_cfg = {}
        try:
            with open(self.restore_config_path.text().strip()) as f:
                fallback_cfg = json.load(f)
        except Exception:
            pass

        steps = []
        for d in candidate_dirs:
            cfg_path = os.path.join(d, "step_config.json")
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except Exception:
                parts = os.path.basename(d).split("_")
                try:
                    t0_parsed = float(parts[2].replace("t", ""))
                    t1_parsed = float(parts[4].replace("t", ""))
                except Exception:
                    continue
                cfg = dict(fallback_cfg)
                cfg["t_min"] = t0_parsed
                cfg["t_max"] = t1_parsed

            t0 = cfg.get("t_min", cfg.get("t0", 0.0))
            t1 = cfg.get("t_max", cfg.get("t1", 1.0))

            pt_path = ""
            for pat in ["model_lbfgs-*.pt", "model_lbfgs.pt", "model_adam-*.pt", "model_adam.pt"]:
                pts = sorted(_glob_ta.glob(os.path.join(d, pat)))
                if pts:
                    pt_path = max(pts, key=os.path.getmtime)
                    break
            if not pt_path:
                continue

            steps.append({'dir': d, 't0': t0, 't1': t1, 'model_path': pt_path, 'cfg': cfg})

        steps.sort(key=lambda s: s['t0'])
        return steps

    def _on_browse_restore_save(self):
        folder = QFileDialog.getExistingDirectory(self, "Select save directory")
        if folder:
            self.restore_save_path.setText(folder)

    def _add_restore_param_row(self, path="", true_val=None):
        """Add one parameter-convergence-file row (Inverse restore's
        Parameter Convergence Plot/Animation viz types). Every row is
        freely removable, including the first -- unlike the trainable-
        variable / measured-data-file row lists elsewhere, there's no
        single legacy field a "primary" row needs to stay aliased to
        here, so at-least-one-path is simply checked at Restore time.

        The optional True-value field lets the plot/GIF draw the same
        dashed true-value reference line the training-time parameter-
        convergence plot already draws -- these two are otherwise
        disconnected (a *_convergence.txt file only ever has iteration,
        value rows, no ground truth), so this has to be supplied here,
        either typed in directly or auto-filled from the run's own
        model_config.json when known (see _on_browse_restore_model).
        Left blank -- the default -- simply skips that reference line,
        same as an unknown/manually-added variable."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        path_edit = QLineEdit()
        path_edit.setText(path)
        path_edit.setPlaceholderText("Browse for <name>_convergence.txt...")
        path_edit.setFixedHeight(26)
        row_layout.addWidget(path_edit)
        true_edit = QLineEdit()
        true_edit.setText("" if true_val is None else str(true_val))
        true_edit.setPlaceholderText("True value (optional)")
        true_edit.setFixedHeight(26)
        true_edit.setFixedWidth(130)
        true_edit.setToolTip(
            "Known ground-truth value for this parameter, if any -- draws a "
            "dashed reference line on the plot/GIF, same as the training-time "
            "convergence plot. Auto-filled when the restored run's own "
            "model_config.json recorded it; leave blank to skip the line.")
        row_layout.addWidget(true_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedHeight(26); browse_btn.setFixedWidth(65)
        browse_btn.clicked.connect(lambda _checked=False, e=path_edit: self._on_browse_restore_param(e))
        row_layout.addWidget(browse_btn)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedHeight(26); remove_btn.setFixedWidth(26)
        remove_btn.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
        row_layout.addWidget(remove_btn)
        self.restore_param_files_layout.addWidget(row_widget)
        row_data = {'widget': row_widget, 'path': path_edit, 'true': true_edit, 'browse': browse_btn}
        self.restore_param_rows.append(row_data)

        def _remove():
            row_widget.deleteLater()
            if row_data in self.restore_param_rows:
                self.restore_param_rows.remove(row_data)
        remove_btn.clicked.connect(_remove)
        return row_data

    def _set_restore_param_rows(self, paths, true_vals=None):
        for row in list(self.restore_param_rows):
            row['widget'].deleteLater()
        self.restore_param_rows.clear()
        if not paths:
            self._add_restore_param_row()
        else:
            true_vals = true_vals if true_vals is not None else [None] * len(paths)
            for p, tv in zip(paths, true_vals):
                self._add_restore_param_row(p, tv)

    def _on_browse_restore_param(self, target):
        f, _ = QFileDialog.getOpenFileName(self, "Select parameter convergence file", "", "Text files (*.txt)")
        if f:
            target.setText(f)

    def _on_restore(self):
        import json, tempfile, subprocess, sys
        viz_type    = self.restore_viz_combo.currentText()
        # Every restore run gets its own timestamped subfolder under the
        # configured save path too (same convention as a normal Solve --
        # see _run_results_dir/_timestamped_save_dir), not just plain
        # Solve runs. _on_restore_done() (which looks for this run's
        # output files once the subprocess finishes) reads the SAME
        # resolved path back from self._last_restore_save_dir rather than
        # recomputing it from the raw widget text, so it always looks in
        # the right folder even though the label below can differ branch
        # to branch (a restored checkpoint doesn't carry which Quick
        # Example it came from, so the label is generic here and refined
        # once a model_config.json is actually loaded, below).
        _base_restore_dir = self.restore_save_path.text().strip()
        save_dir    = self._timestamped_save_dir(_base_restore_dir, "Restore")
        self._last_restore_save_dir = save_dir

        # Parameter Convergence Plot/Animation: entirely independent of
        # the model checkpoint (see _build_restore_param_script) -- reads
        # the *_convergence.txt file(s) directly, no model/config/
        # optimizer/output selection needed.
        if viz_type in getattr(self, '_RESTORE_PARAM_VIZ', []):
            if not _base_restore_dir:
                self.log_box.append("❌ Please select a save directory."); return
            save_dir = self._timestamped_save_dir(_base_restore_dir, "ParameterConvergence")
            self._last_restore_save_dir = save_dir
            _param_rows_used = [r for r in self.restore_param_rows if r['path'].text().strip()]
            paths = [r['path'].text().strip() for r in _param_rows_used]
            if not paths:
                self.log_box.append("❌ Please add at least one parameter convergence .txt file."); return
            # True-value field is optional and free-typed -- an empty or
            # unparseable entry just means "no known true value" (same as
            # leaving it blank), never an error that blocks the restore.
            true_vals = []
            for r in _param_rows_used:
                tv_text = r['true'].text().strip()
                try:
                    true_vals.append(float(tv_text) if tv_text else None)
                except ValueError:
                    true_vals.append(None)
            combine = self.restore_param_combine_combo.currentText().startswith("Combine")
            log_scale = self.restore_param_log_cb.isChecked()
            animate = (viz_type == "Parameter Convergence Animation (GIF)")

            self.restore_btn.setEnabled(False)
            self.restore_btn.setText(" Restoring…")
            self.log_box.append(
                f"🔄 Building parameter convergence {'animation' if animate else 'plot'} "
                f"from {len(paths)} file(s)...")
            # Tracked so _on_restore_done knows which branch just ran --
            # without this it glob-searched save_dir for *convergence_*
            # files after EVERY restore regardless of viz_type, picking up
            # and misreporting stale files left over from an unrelated
            # earlier restore/solve in the same folder as if freshly
            # created just now. Also reset both display panels (and stop
            # any GIF still animating) so nothing left over from an
            # earlier action can be mistaken for this restore's own
            # output while it's in flight.
            self._last_restore_is_param = True
            # No single original model/training folder exists for this
            # branch (the user can list several unrelated *_convergence.txt
            # files) -- clear any directory list a PREVIOUS model-based
            # restore left behind so _on_restore_done doesn't mistakenly
            # show that earlier run's loss plot here.
            self._last_restore_model_dirs = []
            self._clear_solution_movie()
            self._reset_plot_headers()
            self.loss_label.setText("⏳ Restoring...")
            self.solution_label.setText("⏳ Restoring...")
            self.loss_label._source_path = None
            self.solution_label._source_path = None
            script = self._build_restore_param_script(paths, save_dir, combine, log_scale, animate, true_vals)

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
                    self.process = None

                def run(self):
                    proc = subprocess.Popen(
                        [sys.executable, self._tmp],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                    )
                    self.process = proc
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line: self.line_signal.emit(line)
                    proc.wait()
                    os.unlink(self._tmp)
                    self.done_signal.emit(proc.returncode == 0)

                def stop(self):
                    # Same fix as _RestoreThread/SolverThread -- see
                    # closeEvent, which needs this to stop this thread's
                    # actual subprocess rather than just abandoning it.
                    # terminate()-then-kill() with a bounded wait (see
                    # SolverThread.stop() for the full rationale) -- not
                    # just terminate() alone, which can hang indefinitely.
                    if self.process and self.process.poll() is None:
                        import subprocess
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait()

            self._restore_thread = _ParamRestoreThread(tmp)
            self._restore_thread.line_signal.connect(self.log_box.append)
            self._restore_thread.done_signal.connect(self._on_restore_done)
            self._restore_thread.start()
            return

        model_path  = self.restore_model_path.text().strip()
        config_path = self.restore_config_path.text().strip()
        optimizer   = self.restore_optimizer_combo.currentData()
        # "Custom..." mirrors the live Results panel's own Plot output
        # selector (see plot_output_combo) -- output_idx falls back to 0 in
        # that case (never actually used for indexing: _build_restore_script/
        # _build_restore_script_ta's _extract_plot_field only reads it when
        # custom_expr is empty), and the expression/label are read from the
        # two fields _on_restore_output_combo_changed reveals.
        _is_custom_restore = (self.restore_output_combo.currentText() == "Custom...")
        custom_expr  = self.restore_custom_expr_input.text().strip() if _is_custom_restore else ""
        custom_label = self.restore_custom_label_input.text().strip() if _is_custom_restore else ""
        output_idx  = 0 if _is_custom_restore else self.restore_output_combo.currentIndex()
        t_steps     = self.restore_tsteps_spin.value()

        if not model_path:
            self.log_box.append("❌ Please select a model file."); return
        if not config_path:
            self.log_box.append("❌ Please select a model_config.json file."); return
        if not save_dir:
            self.log_box.append("❌ Please select a save directory."); return

        # Loss plot: lives alongside the ORIGINAL model being restored,
        # never in this restore run's own brand-new save_dir (a restore
        # run doesn't retrain, so it never writes its own loss_plot.png --
        # looking in save_dir, as this used to, could never find one).
        # For the Standard/RAR path, model_path already points straight
        # into the run's "solution_results" folder, sibling to
        # loss_plot.png. For a Time-Adaptive combined restore, model_path
        # instead points into a "time_adaptive_steps/step_*/" folder (see
        # _detect_restore_ta_steps) -- TA's own loss plot is one shared
        # file for the whole run, saved under that run's own
        # "solution_results" folder, two levels up from there.
        _model_dir = os.path.dirname(model_path)
        _step_base = os.path.basename(_model_dir)
        _step_parent = os.path.dirname(_model_dir)
        if os.path.basename(_step_parent) == "time_adaptive_steps" and _step_base.startswith("step_"):
            _orig_run_dir = os.path.dirname(_step_parent)
            self._last_restore_model_dirs = [os.path.join(_orig_run_dir, "solution_results"), _model_dir]
        else:
            self._last_restore_model_dirs = [_model_dir]

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            self.log_box.append(f"❌ Could not read config: {e}"); return

        # Now that the restored model's own config is loaded, refine the
        # generic "Restore__<timestamp>" folder picked above into one that
        # says what's actually being restored -- same dimension/
        # stationarity/problem-type label a fallback (non-template) Solve
        # run gets (see _run_folder_label), since a restored checkpoint
        # has no Quick-Example name of its own to use instead.
        if _base_restore_dir:
            _restore_label = (
                f"Restore_{cfg.get('problem_dim', '1D')}_"
                f"{'Stationary' if cfg.get('steady_state', False) else 'TimeDependent'}_"
                f"{'Inverse' if cfg.get('problem_type', 'Forward') == 'Inverse' else 'Forward'}"
            )
            save_dir = self._timestamped_save_dir(_base_restore_dir, _restore_label)
            self._last_restore_save_dir = save_dir

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
        self.restore_btn.setText(" Restoring…")
        self.log_box.append(f"🔄 Restoring model from: {model_path}")
        # See the matching comment in the Parameter Convergence branch
        # above -- same reasons, same fix.
        self._last_restore_is_param = False
        self._clear_solution_movie()
        self._reset_plot_headers()
        self.loss_label.setText("⏳ Restoring...")
        self.solution_label.setText("⏳ Restoring...")
        self.loss_label._source_path = None
        self.solution_label._source_path = None
        viz_settings = getattr(self, '_restore_viz_settings', {})
        # Script-building below is pure computation (no subprocess started
        # yet), but it reads several cfg[...] keys directly rather than
        # cfg.get(...) -- a malformed or older-format model_config.json
        # that parsed as valid JSON but is missing one of them raises a
        # KeyError here, which (before this try/except existed) propagated
        # straight out of this Qt slot, leaving restore_btn stuck disabled
        # and reading "⏳ Restoring..." forever, since only the background
        # thread's done_signal (never started in that case) normally
        # re-enables it.
        # Time-Adaptive combined restore: _restore_ta_steps is only
        # populated (see _detect_restore_ta_steps, called from
        # _on_browse_restore_model) when the browsed model file lives in a
        # "time_adaptive_steps/step_*/" folder alongside >=2 sibling steps.
        # The two static viz types always route through the TA-aware
        # builder in that case -- auto-routing to whichever step actually
        # covers the requested time is strictly more correct than
        # _build_restore_script's single-model-across-the-whole-domain
        # behavior, never a regression -- while the two Animation types
        # only do when the "Combine steps" checkbox (default: checked,
        # see _update_restore_ta_combine_visibility) is also checked, so
        # an unchecked box keeps producing exactly the single-step
        # animation _build_restore_script always has.
        _ta_steps_for_restore = getattr(self, '_restore_ta_steps', []) or []
        _is_anim_viz = viz_type in ("Animation Line (GIF)", "Animation Surface (GIF)")
        _use_ta_restore = len(_ta_steps_for_restore) >= 2 and (not _is_anim_viz or self.restore_ta_combine_cb.isChecked())
        try:
            if _use_ta_restore:
                script = self._build_restore_script_ta(_ta_steps_for_restore, cfg, optimizer, viz_type, output_idx, t_steps, save_dir, custom_expr, custom_label)
            else:
                script = self._build_restore_script(model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir, custom_expr, custom_label)
        except Exception as e:
            self.log_box.append(f"❌ Could not build the restore script from this config: {e}")
            self.restore_btn.setEnabled(True)
            self.restore_btn.setText(" Restore && Visualize")
            return

        # ── Append error analysis if files configured ─────────
        # Wrapped in its own try/except, separate from the one around
        # _build_restore_script above: Error Analysis here is a
        # supplementary extra on top of the restore/visualize script, not
        # the thing the user actually clicked the button for, so a problem
        # building it (e.g. output_idx no longer matching this cfg's own
        # output_names, from an older/mismatched model_config.json) logs a
        # warning and continues with the restore script as already built,
        # rather than aborting the whole restore over the EA add-on.
        try:
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
                # Filter files within t range. ea['files'] entries are always
                # (time, path, output_selector) 3-tuples since the v40 multi-
                # output Error Analysis feature (output_selector is None for a
                # single-output template's file, an int raw output index for a
                # named per-output file like "t_0_u.txt", or a [expr, label]
                # pair for a custom derived field) -- unpacking as a 2-tuple
                # here (as before v40) raises "too many values to unpack" the
                # moment any reference file is configured, which is effectively
                # always, since every write site (auto-config and the manual
                # Error Analysis dialog) has stored 3-tuples since that patch.
                # This restore-and-visualize path only ever predicts a single
                # chosen field (output_idx, or -- since the Restore panel's own
                # "Custom..." option -- a derived custom_expr), so rather than
                # building out the full per-group multi-output report the main
                # Error Analysis paths have, a file is included here only if it
                # belongs to that same field: when Custom is selected, only a
                # [expr, label] selector whose expr matches custom_expr exactly
                # (comparing a derived field's reference data against a
                # DIFFERENT prediction, raw or derived, would silently compare
                # the wrong physical quantity); otherwise, same as before this
                # option existed, selector is None (the implicit-default/
                # single-output case) or an int matching output_idx exactly --
                # a [expr, label] selector is excluded in that branch too, same
                # reasoning in reverse.
                if custom_expr:
                    matching_files = [
                        (t, f) for t, f, sel in ea['files']
                        if isinstance(sel, (list, tuple)) and len(sel) >= 1
                        and str(sel[0]).strip() == custom_expr
                        and t_min_restore - 1e-10 <= t <= t_max_restore + 1e-10
                    ]
                else:
                    matching_files = [
                        (t, f) for t, f, sel in ea['files']
                        if (sel is None or sel == output_idx)
                        and t_min_restore - 1e-10 <= t <= t_max_restore + 1e-10
                    ]
                if matching_files:
                    self.log_box.append(f"📊 Error analysis: {len(matching_files)} reference files match t=[{t_min_restore:.4f}, {t_max_restore:.4f}]")
                    is_2d = cfg.get('problem_dim', '1D') == '2D'
                    is_3d = cfg.get('problem_dim', '1D') == '3D'
                    _ea_out_name = (
                        (custom_label or custom_expr) if custom_expr
                        else cfg.get('output_names', 'u').split(',')[output_idx].strip()
                    )
                    # Same live, independently-overridable y/z slice
                    # (_on_restore_viz_settings' slice rows, in
                    # viz_settings -- not a passive readback of
                    # model_config.json) that the restore script's own
                    # Line plot/animation use -- so the Error-Analysis
                    # line comparison below extracts reference points near
                    # the SAME y/z slice it is being compared against,
                    # instead of plotting every reference point regardless
                    # of its y/z coordinate.
                    _ea_y_min = cfg.get('y_min', 0.0); _ea_y_max = cfg.get('y_max', 1.0)
                    _ea_z_min = cfg.get('z_min', 0.0); _ea_z_max = cfg.get('z_max', 1.0)
                    _ea_line_slice_y = (_ea_y_min + _ea_y_max) / 2.0 if viz_settings.get('line_slice_y_auto', True) else viz_settings.get('line_slice_y', 0.0)
                    _ea_line_slice_z = (_ea_z_min + _ea_z_max) / 2.0 if viz_settings.get('line_slice_z_auto', True) else viz_settings.get('line_slice_z', 0.0)
                    script += self._build_restore_ea_script(
                        matching_files, save_dir, is_2d,
                        ea.get('do_line', True), ea.get('do_surface', True),
                        cfg.get('x_min', 0.0), cfg.get('x_max', 1.0),
                        _ea_y_min, _ea_y_max,
                        _ea_out_name,
                        viz_settings,
                        is_3d=is_3d, output_idx=output_idx,
                        custom_expr=custom_expr,
                        output_names=cfg.get('output_names', 'u'),
                        is_steady=cfg.get('steady_state', False),
                        is_ta=_use_ta_restore,
                        z_min=_ea_z_min, z_max=_ea_z_max,
                        line_slice_y=_ea_line_slice_y, line_slice_z=_ea_line_slice_z,
                        # Round 31: the same "Error Metrics to Compute"
                        # checkboxes (ea['do_l2']/['do_mse']/['do_max'],
                        # already stored by both the auto-configure path
                        # and the manual Error Analysis dialog) now also
                        # gate the Restore & Visualize path's own metrics
                        # print line and error_metrics_restore.txt columns
                        # -- previously always computed/shown all three
                        # here regardless of the dialog's checkboxes.
                        ea_metric_l2=ea.get('do_l2', True),
                        ea_metric_mse=ea.get('do_mse', False),
                        ea_metric_max=ea.get('do_max', False),
                    )
                else:
                    self.log_box.append(f"ℹ️ No reference files match t=[{t_min_restore:.4f}, {t_max_restore:.4f}] — skipping error analysis")
        except Exception as e:
            self.log_box.append(f"⚠️ Skipping error analysis for this restore -- couldn't build it from this config: {e}")

        # Clear any restored_plot.png/restored_animation.gif (and the EA
        # comparison PNGs) already sitting in save_dir from a PREVIOUS,
        # unrelated restore -- e.g. an earlier Animation-type restore of a
        # completely different (transient) problem run against this same
        # folder. _on_restore_done below picks which file to show/log purely
        # by os.path.exists(...), so a stale leftover with the right
        # filename was indistinguishable from this run's own output: a
        # steady-state Surface restore that writes only restored_plot.png
        # would still show/report a leftover restored_animation.gif (and/or
        # a leftover surface_comparison_restore.png) from whatever ran in
        # this folder before, instead of what was actually just produced.
        # Deleting them here means every os.path.exists(...) check after
        # the subprocess finishes can only ever see THIS run's own output.
        for _stale_name in ("restored_plot.png", "restored_animation.gif"):
            _stale_path = os.path.join(save_dir, _stale_name)
            if os.path.exists(_stale_path):
                try:
                    os.remove(_stale_path)
                except OSError:
                    pass
        for _stale_ea_name in ("surface_comparison_restore.png", "line_comparison_restore.png",
                                "error_metrics_restore.txt"):
            _stale_ea_path = os.path.join(save_dir, "error_analysis", _stale_ea_name)
            if os.path.exists(_stale_ea_path):
                try:
                    os.remove(_stale_ea_path)
                except OSError:
                    pass

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
                self.process = None

            def run(self):
                proc = subprocess.Popen(
                    [sys.executable, self._tmp],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                self.process = proc
                for line in proc.stdout:
                    line = line.rstrip()
                    if line: self.line_signal.emit(line)
                proc.wait()
                os.unlink(self._tmp)
                self.done_signal.emit(proc.returncode == 0)

            def stop(self):
                # Mirrors SolverThread.stop() -- previously this thread had
                # no way at all to be asked to stop (its subprocess handle
                # wasn't even kept as an attribute), so it could only ever
                # end on its own or be abandoned as an orphaned process if
                # the GUI closed mid-restore. terminate()-then-kill() with a
                # bounded wait (see SolverThread.stop() for the full
                # rationale) -- not just terminate() alone, which can hang
                # indefinitely.
                if self.process and self.process.poll() is None:
                    import subprocess
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()

        self._restore_thread = _RestoreThread(tmp)
        self._restore_thread.line_signal.connect(self.log_box.append)
        self._restore_thread.done_signal.connect(self._on_restore_done)
        self._restore_thread.start()

    def _on_restore_done(self, success):
        self.restore_btn.setEnabled(True)
        self.restore_btn.setText(" Restore && Visualize")
        # The run that just finished actually wrote into its own
        # timestamped subfolder (see _on_restore), not the bare widget
        # text -- read that resolved path back rather than recomputing a
        # fresh (and wrong -- a different, nonexistent timestamp) one
        # from self.restore_save_path here.
        save_dir = getattr(self, '_last_restore_save_dir', '') or self.restore_save_path.text().strip()
        is_param = getattr(self, '_last_restore_is_param', False)
        if success:
            self.log_box.append("✅ Restore complete!")

            # LEFT panel: the ORIGINAL run's training loss curve -- always,
            # regardless of dimension or which viz type was just restored
            # (matches the main Solve tab's own loss_label/solution_label
            # split in _on_done: loss always on the left, the requested
            # visualization always on the right). This has to look inside
            # the model's own original training folder(s) (see
            # _last_restore_model_dirs, set in _on_restore) -- NOT this
            # restore run's own brand-new save_dir, which never gets its
            # own loss_plot.png written into it (a restore doesn't retrain
            # anything). Looking in save_dir, as this previously did, could
            # never succeed, leaving this panel stuck on "Restoring..."
            # forever for every restore, forward or inverse alike.
            loss_path = None
            for _mdir in getattr(self, '_last_restore_model_dirs', []) or []:
                for _candidate in (os.path.join(_mdir, "loss_plot.png"),
                                    os.path.join(_mdir, "solution_results", "loss_plot.png")):
                    if os.path.exists(_candidate):
                        loss_path = _candidate
                        break
                if loss_path:
                    break
            if loss_path:
                self.loss_label.setPixmap(QPixmap(loss_path).scaled(
                    self.loss_label.width(), self.loss_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                self.loss_label._source_path = loss_path

            if is_param:
                # Parameter Convergence Plot/Animation outputs -- combined
                # (one file) or per-variable (several), PNG or GIF. Preview
                # whichever comes first alphabetically; every file produced
                # is already named in the log above from the script's own
                # prints. Only reached for an actual Parameter Convergence
                # restore (see _last_restore_is_param) -- previously this
                # glob ran after every restore, regardless of viz_type, and
                # could pick up files left over from an unrelated earlier
                # restore/solve in the same save_dir, misreporting them as
                # freshly created by a run that never touched them.
                import glob as _glob_done
                param_pngs = sorted(_glob_done.glob(os.path.join(save_dir, "*convergence_plot.png")))
                if param_pngs:
                    self.solution_label.setPixmap(QPixmap(param_pngs[0]).scaled(
                        500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    self.solution_label._source_path = param_pngs[0]
                param_gifs = sorted(_glob_done.glob(os.path.join(save_dir, "*convergence_animation.gif")))
                if param_gifs:
                    self._set_solution_gif(param_gifs[0])
                    self.solution_label._source_path = param_gifs[0]
                for _pg in param_gifs:
                    self.log_box.append(f"🎬 Parameter convergence animation saved: {_pg}")
            else:
                # RIGHT panel: whatever this restore actually produced --
                # a static image, or an animated GIF played in place (same
                # _set_solution_gif the main Solve tab uses for its own
                # Line/Surface Animation plot types), never both.
                # Previously a GIF result only ever got a log line here --
                # the file was genuinely saved, but the panel itself never
                # showed it, static or animated.
                plot_path = os.path.join(save_dir, "restored_plot.png")
                gif_path = os.path.join(save_dir, "restored_animation.gif")
                if os.path.exists(gif_path):
                    self._set_solution_gif(gif_path)
                    self.solution_label._source_path = gif_path
                    self.log_box.append(f"🎬 Animation saved: {gif_path}")
                elif os.path.exists(plot_path):
                    self.solution_label.setPixmap(QPixmap(plot_path).scaled(
                        500, 420, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    self.solution_label._source_path = plot_path

            # Error Analysis is a supplementary extra on top of whichever
            # restore actually ran (see _on_restore) -- log where its
            # output landed, same as the main Solve tab's own EA handling
            # in _on_done, rather than displacing either display panel
            # from what was actually requested above.
            for _ea_plot in ["surface_comparison_restore.png", "line_comparison_restore.png"]:
                _ea_path = os.path.join(save_dir, "error_analysis", _ea_plot)
                if os.path.exists(_ea_path):
                    self.log_box.append(f"📊 Error analysis plot: {_ea_path}")
        else:
            self.log_box.append("❌ Restore failed — check architecture matches saved model.")

    def _build_restore_param_script(self, paths, save_dir, combine, log_scale, animate, true_vals=None):
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
        place the value's full history exists.

        `true_vals` -- a list of known ground-truth values parallel to
        `paths` (None entries where unknown) -- lets both the static plot
        and the GIF draw the same dashed true-value reference line the
        training-time param_convergence.png already draws (see codegen.py).
        These files alone never carry that information (only iteration,
        value rows), so it has to be supplied from outside: typed in by the
        user, or auto-filled from the restored run's own model_config.json
        when it was recorded there (see _on_browse_restore_model)."""
        paths_literal = repr(list(paths))
        true_vals_literal = repr(list(true_vals) if true_vals is not None else [None] * len(paths))
        combine_literal = repr(bool(combine))
        log_literal = repr(bool(log_scale))
        animate_literal = repr(bool(animate))
        viz_settings = getattr(self, '_restore_viz_settings', {})
        title_override = (viz_settings.get('title') or '').strip()
        xlabel_override = (viz_settings.get('xlabel') or '').strip()
        ylabel_override = (viz_settings.get('ylabel') or '').strip()
        figsize_mode = viz_settings.get('figsize_mode', 'Default')
        figsize_w = viz_settings.get('figsize_w', 7.0)
        figsize_h = viz_settings.get('figsize_h', 5.0)
        subplot_title_fontsize = viz_settings.get('subplot_title_fontsize', 10.0)
        axis_label_fontsize = viz_settings.get('axis_label_fontsize', 10.0)
        from pinnstudio.core.codegen import _figsize_helper_code, _param_axis_helper_code, _contour_levels_helper_code
        script = f"""
import os
os.makedirs({save_dir!r}, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as _anim
import numpy as np

_paths = {paths_literal}
_true_vals = {true_vals_literal}
_combine = {combine_literal}
_log_scale = {log_literal}
_animate = {animate_literal}
_title_override = {title_override!r}
_xlabel_override = {xlabel_override!r}
_ylabel_override = {ylabel_override!r}

# ── Plot Settings: figure-size standardization (see codegen.py's
# _figsize_helper_code for the full rationale) -- "Default" mode renders
# every figsize=_plot_figsize(w, h) call below byte-identical to before
# this existed.
{_figsize_helper_code(figsize_mode, figsize_w, figsize_h)}

# Shared-color-scale contourf levels (see codegen.py's _contour_levels_helper_code).
{_contour_levels_helper_code()}

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

# _true_vals stays index-aligned with _paths (both built together in
# main_window.py from the same row list), so zip before filtering blanks --
# each series carries its own true value (or None) straight through.
series = []
for _p, _tv in zip(_paths, _true_vals):
    _p = _p.strip()
    if not _p:
        continue
    try:
        _name, _iters, _vals = _load_conv(_p)
        if len(_iters) < 1:
            print(f"⚠️ {{_p}}: no data rows found, skipping")
            continue
        series.append((_name, _iters, _vals, _tv))
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

# cleanup Round 49: the static plot draws through _draw_param_ax, the same
# function the live Solve and exported scripts use (codegen.py's
# _param_axis_helper_code), instead of its own hand-typed copy.
{_param_axis_helper_code(axis_label_fontsize, subplot_title_fontsize)}

def _draw_static_ax(ax, name, iters, vals, true_val=None):
    _draw_param_ax(ax, name, iters, vals, true_val, _log_scale,
                   _title_override, _xlabel_override, _ylabel_override)

def _setup_anim_ax(ax, name, iters, vals, true_val=None):
    final_val = vals[-1]
    use_log = _use_log(vals)
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel(_ylabel_override or f"log({{name}})", fontsize={axis_label_fontsize})
    else:
        ax.set_ylabel(_ylabel_override or name, fontsize={axis_label_fontsize})
    x_hi = iters.max() if iters.max() > iters.min() else iters.min() + 1
    ax.set_xlim(iters.min(), x_hi)
    # Include the true value in the y-range too (not just the logged
    # trajectory) -- otherwise a run that hasn't fully converged yet can
    # clip the true-value line right off the animated axes, defeating the
    # point of drawing it.
    _range_vals = vals if true_val is None else np.append(vals, true_val)
    vmin, vmax = _range_vals.min(), _range_vals.max()
    pad = 0.05 * (abs(vmax - vmin) if vmax != vmin else (abs(vmax) + 1))
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.axhline(y=final_val, color="#ff8787", linestyle="--", alpha=0.5, label=f"Final = {{_num_txt(final_val)}}")
    if true_val is not None:
        ax.axhline(y=true_val, color="#ffd43b", linestyle="--", alpha=0.8, label=f"True = {{_num_txt(true_val)}}")
    ax.set_xlabel(_xlabel_override or "Iteration", fontsize={axis_label_fontsize})
    ax.set_title(_title_override or f"Inferred Parameter: {{name}}", fontsize={subplot_title_fontsize})
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], color="#69db7c", linewidth=1.5)
    return line

if not _animate:
    if _combine:
        n = len(series)
        fig, axes = plt.subplots(n, 1, figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * n), squeeze=False)
        for i, (name, iters, vals, true_val) in enumerate(series):
            _draw_static_ax(axes[i][0], name, iters, vals, true_val)
        plt.tight_layout()
        out_path = os.path.join({save_dir!r}, "param_convergence_plot.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ Parameter convergence plot saved: {{out_path}}")
    else:
        for name, iters, vals, true_val in series:
            fig, ax = plt.subplots(figsize=_plot_figsize(6, 3.2))
            _draw_static_ax(ax, name, iters, vals, true_val)
            plt.tight_layout()
            out_path = os.path.join({save_dir!r}, f"{{name}}_convergence_plot.png")
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
        fig, axes = plt.subplots(n, 1, figsize=(_plot_figsize(6, 3.2)[0], _plot_figsize(6, 3.2)[1] * n), squeeze=False)
        lines = [_setup_anim_ax(axes[i][0], name, iters, vals, true_val) for i, (name, iters, vals, true_val) in enumerate(series)]
        plt.tight_layout()
        def update(frame):
            for line, (name, iters, vals, true_val) in zip(lines, series):
                frac = (frame + 1) / n_frames
                idx = max(1, min(len(iters), int(round(frac * len(iters)))))
                line.set_data(iters[:idx], vals[:idx])
            return lines
        ani = _anim.FuncAnimation(fig, update, frames=n_frames, interval=80)
        out_path = os.path.join({save_dir!r}, "param_convergence_animation.gif")
        ani.save(out_path, writer='pillow', fps=15)
        plt.close(fig)
        print(f"✅ Parameter convergence animation saved: {{out_path}}")
    else:
        for name, iters, vals, true_val in series:
            n_frames = min(len(iters), 120)
            fig, ax = plt.subplots(figsize=_plot_figsize(6, 3.2))
            line = _setup_anim_ax(ax, name, iters, vals, true_val)
            plt.tight_layout()
            def update(frame, iters=iters, vals=vals, line=line, n_frames=n_frames):
                frac = (frame + 1) / n_frames
                idx = max(1, min(len(iters), int(round(frac * len(iters)))))
                line.set_data(iters[:idx], vals[:idx])
                return [line]
            ani = _anim.FuncAnimation(fig, update, frames=n_frames, interval=80)
            out_path = os.path.join({save_dir!r}, f"{{name}}_convergence_animation.gif")
            ani.save(out_path, writer='pillow', fps=15)
            plt.close(fig)
            print(f"✅ Parameter convergence animation saved: {{out_path}}")

print("RESTORE_DONE")
"""
        return script

    def _build_restore_script(self, model_path, cfg, optimizer, viz_type, output_idx, t_steps, save_dir, custom_expr="", custom_label=""):
        viz_settings = getattr(self, '_restore_viz_settings', {})
        colormap = viz_settings.get('colormap', 'RdBu_r')
        # A genuinely live, multiple-evenly-spaced-snapshots Surface plot
        # (one subplot per snapshot, same convention as the Setup tab's own
        # "Plot output" -> Surface -> 2D/3D non-steady case, see
        # codegen.py's own "elif _is_2d:"/"elif _is_3d:" branches) -- NOT a
        # single user-chosen "Plot at time t=" snapshot the way this used
        # to work (the "surface_time" key/dialog row, now removed). Only
        # meaningful for a non-steady 2D/3D restore; a steady restore (no
        # time axis) or a 1D restore (the x-t heatmap already shows the
        # whole time range in one plot) both ignore this and always render
        # exactly one panel.
        n_2d_snapshots = viz_settings.get('n_2d_snapshots', 2)
        show_colorbar = viz_settings.get('colorbar', True)
        # Type-aware keys (see _viz_steps_key/_viz_linewidth_key): the
        # static "Line (time steps)" plot and the GIF animations keep
        # separate step counts and line widths instead of sharing one.
        n_steps = viz_settings.get(self._viz_steps_key(viz_type), t_steps)
        levels = viz_settings.get('levels', 40)
        resolution = viz_settings.get('resolution', 100)
        dpi = viz_settings.get('dpi', 100)
        auto_range = viz_settings.get('auto_range', True)
        vmin_val = viz_settings.get('vmin', -1.0)
        vmax_val = viz_settings.get('vmax', 1.0)
        linewidth = viz_settings.get(self._viz_linewidth_key(viz_type), 2.0)
        fps = viz_settings.get('fps', 10)
        swap_xt = viz_settings.get('swap_xt', True)
        title_override = (viz_settings.get('title') or '').strip()
        xlabel_override = (viz_settings.get('xlabel') or '').strip()
        ylabel_override = (viz_settings.get('ylabel') or '').strip()
        figsize_mode = viz_settings.get('figsize_mode', 'Default')
        figsize_w = viz_settings.get('figsize_w', 7.0)
        figsize_h = viz_settings.get('figsize_h', 5.0)
        title_fontsize = viz_settings.get('title_fontsize', 13.0)
        subplot_title_fontsize = viz_settings.get('subplot_title_fontsize', 10.0)
        axis_label_fontsize = viz_settings.get('axis_label_fontsize', 10.0)
        layers     = cfg["layers"]
        activation = cfg["activation"]
        from pinnstudio.core.codegen import (_net_construction_helper_code,
            _training_monitor_runtime_code, _figsize_helper_code, _contour_levels_helper_code,
            _math_namespace_code)
        net_helper_code = _net_construction_helper_code(cfg.get("network_type", "FNN"), cfg.get("msffn_sigmas", "1, 10"),
                                                        cfg.get("msffn_sigmas_t", "1, 10"))
        # Re-attach the model's input/output transforms (Round 62) -- see
        # codegen._net_transforms_code; a model_config.json from before
        # this round has no transform keys and gets none, as before.
        from pinnstudio.core.codegen import _net_transforms_code
        net_transforms_code = _net_transforms_code(cfg)
        # Reuses Training Monitors' own derivative-building helpers
        # (_tm_hess/_tm_build_dvars) unchanged, so a restored model's
        # "Custom..." expression can reference derivatives too (du_x,
        # du_xx, du_xy, ...) via DeepXDE's own dde.Model.predict(x,
        # operator=...) built-in -- see _extract_plot_field below.
        _tm_runtime_code_restore = _training_monitor_runtime_code()
        x_min = cfg["x_min"]; x_max = cfg["x_max"]
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        z_min = cfg.get("z_min", 0.0); z_max = cfg.get("z_max", 1.0)
        t_min = cfg["t_min"]; t_max = cfg["t_max"]
        is_2d = cfg.get("problem_dim", "1D") == "2D"
        is_3d = cfg.get("problem_dim", "1D") == "3D"
        _restore_dim_str = "3D" if is_3d else ("2D" if is_2d else "1D")
        # y/z slice value -- a genuinely live, independently-overridable
        # Restore-time control (see _on_restore_viz_settings' slice rows),
        # NOT a passive readback of whatever this run was originally
        # solved/plotted with (model_config.json's own line_slice_y/_z,
        # which this used to read before this round). Always defaults to
        # the domain midpoint the first time the dialog opens for a given
        # restore (viz_settings won't have these keys yet); overriding it
        # there re-plots/re-animates at the new slice without re-training.
        line_slice_y = (y_min + y_max) / 2.0 if viz_settings.get("line_slice_y_auto", True) else viz_settings.get("line_slice_y", 0.0)
        line_slice_z = (z_min + z_max) / 2.0 if viz_settings.get("line_slice_z_auto", True) else viz_settings.get("line_slice_z", 0.0)
        # Steady-state problems have no time axis at all (see
        # _on_steady_state_changed and generate_script()'s own _is_steady
        # branch in codegen.py): the restored network's first layer has one
        # fewer input than a transient model's (cfg["layers"] is already
        # correctly sized for this -- it's read straight from the saved
        # config), so every model.predict(...) input array built below must
        # likewise drop the time column for a steady-state restore, or it
        # crashes with a shape mismatch like "mat1 and mat2 shapes cannot be
        # multiplied (Nx3 and 2x64)" -- exactly the bug this fixes.
        is_steady = bool(cfg.get("steady_state", False))
        loss_type = cfg.get("loss_type", "MSE")
        out_names = cfg.get("output_names", "u").split(",")
        # Optional derived scalar field (e.g. "sqrt(u**2+v**2)" for 1D
        # Schrodinger's |h|), same "Custom..." mechanism as the live Results
        # panel's own plot_custom_expr -- see _extract_plot_field, spliced
        # into the generated script below, which is what actually evaluates
        # it at predict time instead of indexing a single raw output column.
        _custom_expr_val = (custom_expr or "").strip()
        _custom_label_val = (custom_label or "").strip()
        if _custom_expr_val:
            out_name = _custom_label_val or _custom_expr_val
        else:
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

        # Reconstruct the ACTUAL problem geometry (Disk/Ellipse/Triangle/
        # Polygon/Sphere/Custom CSG combo, not just its rectangular/cuboid
        # bounding box) from the saved config, instead of always building a
        # plain Rectangle/Cuboid/Interval here. Without this, a restored
        # model for e.g. a triangular-cavity Custom geometry would predict
        # and plot over the full bounding-box rectangle it happens to sit
        # inside, not the real (possibly non-convex / CSG-combined) domain
        # -- reusing generate_clean_script()'s own geometry-building helpers
        # (_clean_geom_line/_build_custom_geom_code/_parse_vertex_list) so
        # this doesn't reimplement per-shape dispatch a third time, and so
        # any new geometry type added there is automatically picked up here
        # too. geometry_type and the geom_* fields are only present in
        # model_config.json for models saved after this fix -- cfg.get(...)
        # below falls back to the same defaults PINNConfig itself uses, so
        # an older saved config just restores as a plain Rectangle/Cuboid/
        # Interval, exactly as before.
        import types as _restore_types
        from pinnstudio.core.codegen import _clean_geom_line, _parse_vertex_list, _dtype_safe_geom_class_code
        _tri_verts_restore = _parse_vertex_list(cfg.get("geom_triangle_vertices", "0,0;1,0;0,1"))
        if len(_tri_verts_restore) != 3:
            _tri_verts_restore = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        _poly_verts_restore = _parse_vertex_list(cfg.get("geom_polygon_vertices", "0,0;1,0;1,1;0,1"))
        if len(_poly_verts_restore) < 3:
            _poly_verts_restore = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        _geom_cfg_restore = _restore_types.SimpleNamespace(
            geometry_type=cfg.get("geometry_type", "Rectangle"),
            geom_center_x=cfg.get("geom_center_x", 0.5), geom_center_y=cfg.get("geom_center_y", 0.5),
            geom_center_z=cfg.get("geom_center_z", 0.5), geom_radius=cfg.get("geom_radius", 0.5),
            geom_semi_major=cfg.get("geom_semi_major", 0.5), geom_semi_minor=cfg.get("geom_semi_minor", 0.3),
            geom_angle=cfg.get("geom_angle", 0.0),
            geom_custom_shapes_json=cfg.get("geom_custom_shapes_json", "[]"),
            problem_dim=cfg.get("problem_dim", "2D"),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        )
        _geom_line_restore, _geom_needs_dtype_wrap = _clean_geom_line(
            _geom_cfg_restore, is_2d, is_3d, _tri_verts_restore, _poly_verts_restore)
        # Same wrapper class generate_clean_script() emits for the same
        # reason (see codegen.py's _DTypeSafeGeom) -- only spliced in when
        # the reconstructed geometry actually needs it.
        _dtype_safe_geom_class_restore = (_dtype_safe_geom_class_code() + "\n") if _geom_needs_dtype_wrap else ''

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Plot Settings: figure-size standardization (see codegen.py's
# _figsize_helper_code for the full rationale) -- "Default" mode renders
# every figsize=_plot_figsize(w, h) call below byte-identical to before
# this existed.
{_figsize_helper_code(figsize_mode, figsize_w, figsize_h)}

# Shared-color-scale contourf levels (see codegen.py's _contour_levels_helper_code).
{_contour_levels_helper_code()}

# Build the actual problem geometry for model restore (not just its
# bounding box) so the restored prediction can be masked back down to the
# real domain -- see the comment above this script's construction.
{_dtype_safe_geom_class_restore}
{_geom_line_restore}
if {str(is_steady)}:
    # Steady-state: no time axis at all -- geomtime is just geom itself,
    # same convention generate_script() uses for a steady config (see
    # codegen.py's own _is_steady branch). The real architecture match
    # comes from cfg["layers"]/net below either way; this is only a
    # placeholder so model.restore() has something to load weights into.
    geomtime = geom
else:
    timedomain = dde.geometry.TimeDomain({t_min}, {t_max})
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y): return y[:, 0:1] * 0

{inv_var_defs_restore}

if {str(is_steady)}:
    data = dde.data.PDE(geomtime, pde, [], num_domain=100, num_test=100)
else:
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=100, num_test=100)
{net_helper_code}
{net_transforms_code}
net  = _apply_net_transforms(_make_net({layers}, "{activation}", "Glorot uniform"))
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
    model.restore({model_path!r}, verbose=1)
    print("✅ Model restored successfully.")
except Exception as e:
    print(f"❌ Restore error: {{e}}")
    print("Architecture mismatch — make sure network matches saved model.")
    exit(1)

os.makedirs({save_dir!r}, exist_ok=True)
x_vals = np.linspace({x_min}, {x_max}, 100)
y_vals = np.linspace({y_min}, {y_max}, 100)
z_vals = np.linspace({z_min}, {z_max}, 100)
is_2d  = {str(is_2d)}
is_3d  = {str(is_3d)}
is_steady = {str(is_steady)}

{_tm_runtime_code_restore}

# Predicted field: a raw output column (output_idx, matching the Restore
# panel's own "Output to plot" selector) or -- when "Custom..." is picked
# there instead -- an expression over all of this model's outputs AND
# their derivatives (du_x, du_xx, du_xy, ... -- same syntax the Custom PDE
# box and Training Monitors already use), evaluated through DeepXDE's own
# dde.Model.predict(x, operator=...) built-in so autograd is available at
# predict time -- the same built-in Training Monitors' own callback uses
# during training, just invoked once here instead of periodically.
# Same math namespace/mechanism as the live Results panel's own
# _extract_plot_field (codegen.py). Every model.predict(...) call below
# goes through this instead of indexing a fixed column directly.
_restore_custom_expr = {_custom_expr_val!r}
_restore_output_names = {[n.strip() for n in out_names]!r}
_restore_n_out = len(_restore_output_names)
_restore_is_steady = {str(is_steady)}
_restore_dim = {_restore_dim_str!r}
{_math_namespace_code('_RESTORE_MATH_NS', 'torch', '')}
def _restore_custom_op(inputs, outputs):
    _dvars = _tm_build_dvars(inputs, outputs, _restore_n_out, _restore_output_names,
                              _restore_is_steady, _restore_dim, _restore_custom_expr)
    _ns = dict(_dvars)
    _ns.update(_RESTORE_MATH_NS)
    _ns["torch"] = torch
    return eval(_restore_custom_expr, _ns)

def _extract_plot_field(x_grid):
    if _restore_custom_expr:
        return model.predict(x_grid, operator=_restore_custom_op)[:, 0]
    return model.predict(x_grid)[:, {output_idx}]
"""

        if viz_type == "Surface":
            _xlabel_3d = xlabel_override or "x"
            _ylabel_3d = ylabel_override or "y"
            _title_3d = title_override or (
                f"Restored Model — {out_name}(x,y,z)" if is_steady
                else f"Restored Model — {out_name}(x,y,z,t)")
            _xlabel_2d = xlabel_override or "x"
            _ylabel_2d = ylabel_override or "y"
            _title_2d = title_override or (
                f"Restored Model — {out_name}(x,y)" if is_steady
                else f"Restored Model — {out_name}(x,y,t)")
            # Same "Swap axes" convention as the main Results panel's 1D
            # Surface plot (Plot Settings dialog) -- t on the x-axis by
            # default, with the setting above to go back to x on the
            # x-axis; an explicit override in this dialog's own X/Y-axis
            # label fields still wins over either default. Steady-state has
            # no t axis at all, so it gets its own plain x/u(x) line labels
            # instead (the steady-1D line plot in _restore_surface_code).
            _xlabel_1d = xlabel_override or ("t" if swap_xt else "x")
            _ylabel_1d = ylabel_override or ("x" if swap_xt else "t")
            _title_1d = title_override or f"Restored Model — {out_name}(x,t) Surface"
            _xlabel_1d_steady = xlabel_override or "x"
            _ylabel_1d_steady = ylabel_override or out_name
            _title_1d_steady = title_override or f"Restored Model — {out_name}(x)"
            # cleanup Round 45 (audit §2e, sub-round 3): this branch and its
            # _build_restore_script_ta counterpart were two hand-typed copies
            # -- see _restore_surface_code's own docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_surface_code
            script += "\n" + _restore_surface_code(
                resolution, x_min, x_max, y_min, y_max, t_min, t_max, n_2d_snapshots,
                colormap, show_colorbar, levels, swap_xt, auto_range, vmin_val, vmax_val,
                linewidth, title_fontsize, subplot_title_fontsize, axis_label_fontsize,
                save_dir, dpi, _xlabel_3d, _ylabel_3d, _title_3d, _xlabel_2d, _ylabel_2d,
                _title_2d, _xlabel_1d, _ylabel_1d, _title_1d, _xlabel_1d_steady,
                _ylabel_1d_steady, _title_1d_steady, is_time_adaptive=False) + "\n"
            
        elif viz_type == "Line (time steps)":
            _xlabel_line = xlabel_override or "x"
            _ylabel_line = ylabel_override or out_name
            _slice_suffix_line = (f", y={line_slice_y:.3g}, z={line_slice_z:.3g}" if is_3d else (f", y={line_slice_y:.3g}" if is_2d else ""))
            _title_line = title_override or f"Restored Model — {out_name}(x,t{_slice_suffix_line}) Line Plot"
            # cleanup Round 42 (audit §2e): this branch and its
            # _build_restore_script_ta counterpart were two independently
            # hand-typed, ~90%-duplicated copies -- see
            # _restore_line_plot_code's own docstring in codegen.py for the
            # full rationale and the fuzz-verified byte-for-byte parity
            # check this extraction was held to.
            from pinnstudio.core.codegen import _restore_line_plot_code
            script += "\n" + _restore_line_plot_code(
                x_min, x_max, resolution, t_min, t_max, n_steps, colormap,
                line_slice_y, line_slice_z, linewidth, _xlabel_line, _ylabel_line,
                _title_line, subplot_title_fontsize, axis_label_fontsize, save_dir, dpi,
                is_time_adaptive=False) + "\n"
        elif viz_type == "Animation Line (GIF)":
            _slice_suffix_animline = (f", y={line_slice_y:.3g}, z={line_slice_z:.3g}" if is_3d
                                       else (f", y={line_slice_y:.3g}" if is_2d else ""))
            _xlabel_animline = xlabel_override or "x"
            _ylabel_animline = ylabel_override or (f"{out_name}(x,t{_slice_suffix_animline})" if (is_2d or is_3d) else out_name)
            _title_line_stmt = f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override else ""
            # The per-frame "t = ..." indicator below is a blit-compatible
            # Text artist (not a real title -- see its own comment in
            # generate_script()'s matching Line Animation (GIF) branch)
            # normally positioned just above the axes, exactly where a
            # real title (set once, above) would also render -- so when
            # title_override is set, move it inside the axes instead, to
            # avoid the two overlapping into illegible text.
            _tt_x, _tt_y, _tt_ha = (0.02, 0.95, "left") if title_override else (0.5, 1.02, "center")
            # cleanup Round 42 (audit §2e): same extraction as the
            # "Line (time steps)" branch above -- see
            # _restore_line_animation_code's own docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_line_animation_code
            script += "\n" + _restore_line_animation_code(
                x_min, x_max, t_min, t_max, n_steps, line_slice_y, line_slice_z,
                linewidth, _xlabel_animline, _ylabel_animline, _title_line_stmt,
                _tt_x, _tt_y, _tt_ha, axis_label_fontsize, fps, save_dir, dpi,
                is_time_adaptive=False) + "\n"

        elif viz_type == "Animation Surface (GIF)":
            _xlabel_animsurf3d = xlabel_override or "x"
            _ylabel_animsurf3d = ylabel_override or "y"
            _animsurf_title_line_3d = (
                f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override
                else f'ax.set_title(f"t = {{t_frames[i]:.3f}}", fontsize={subplot_title_fontsize})'
            )
            _xlabel_animsurf_else = xlabel_override or ("t" if (swap_xt and not is_2d) else "x")
            _ylabel_animsurf_else = ylabel_override or ("y" if is_2d else ("x" if swap_xt else "t"))
            _animsurf_title_line_else = (
                f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override
                else f'ax.set_title(f"t = {{t_frames[i]:.3f}}", fontsize={subplot_title_fontsize})'
            )
            # cleanup Round 43 (audit §2e, sub-round 2): this branch and its
            # Time-Adaptive counterpart below were ~82% textually identical --
            # see _restore_anim_surface_code's own docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_anim_surface_code
            script += "\n" + _restore_anim_surface_code(
                x_min, x_max, y_min, y_max, t_min, t_max, n_steps, colormap,
                show_colorbar, levels, swap_xt, axis_label_fontsize, fps, save_dir,
                _xlabel_animsurf3d, _ylabel_animsurf3d, _animsurf_title_line_3d,
                _xlabel_animsurf_else, _ylabel_animsurf_else, _animsurf_title_line_else,
                is_time_adaptive=False) + "\n"
        script += '\nprint("RESTORE_DONE")\n'
        return script

    def _build_restore_script_ta(self, ta_steps, cfg, optimizer, viz_type, output_idx, t_steps, save_dir, custom_expr="", custom_label=""):
        """Time-Adaptive-aware counterpart to _build_restore_script.

        Restoring a SINGLE Time-Adaptive step's .pt file and evaluating it
        across the model_config.json's full [t_min, t_max] (what
        _build_restore_script always does) silently extrapolates outside
        the narrow time window that step's model was ever trained on --
        e.g. step 3 of 4's model asked to predict at t=0.05. This instead
        restores EACH detected step's own model lazily (only when a
        requested time actually falls in its window) and picks the right
        one per time value, mirroring the proven "restore step models,
        route by t-range" pattern codegen.py's own Time-Adaptive Error
        Analysis code already uses (see generate_script's two Error
        Analysis blocks) -- just driven from the Restore & Visualize panel
        instead of an Error Analysis run.

        Used instead of _build_restore_script when >=2 Time-Adaptive steps
        were auto-detected next to the browsed model file (see
        _detect_restore_ta_steps) -- always for the two static viz types
        (Surface, Line (time steps)), since auto-routing to the correct
        step is strictly more correct there than the single-model path's
        behavior, and for the two Animation types only when the "Combine
        steps" checkbox is checked (see _on_restore for the branch logic
        and _update_restore_ta_combine_visibility for when the box shows).

        Unlike _build_restore_script this doesn't support Inverse-model
        restore (Time-Adaptive + Inverse isn't offered together in the
        Setup tab) or the Error Analysis add-on's per-step t-range
        narrowing for a *combined* animation (it still narrows to whatever
        single step's step_config.json happened to load in _on_restore,
        same as before this feature) -- both fall back to the single-model
        path's existing (unchanged) behavior in those cases.
        """
        viz_settings = getattr(self, '_restore_viz_settings', {})
        colormap = viz_settings.get('colormap', 'RdBu_r')
        show_colorbar = viz_settings.get('colorbar', True)
        # Type-aware keys (see _viz_steps_key/_viz_linewidth_key): the
        # static "Line (time steps)" plot and the GIF animations keep
        # separate step counts and line widths instead of sharing one.
        n_steps = viz_settings.get(self._viz_steps_key(viz_type), t_steps)
        levels = viz_settings.get('levels', 40)
        resolution = viz_settings.get('resolution', 100)
        dpi = viz_settings.get('dpi', 100)
        auto_range = viz_settings.get('auto_range', True)
        vmin_val = viz_settings.get('vmin', -1.0)
        vmax_val = viz_settings.get('vmax', 1.0)
        linewidth = viz_settings.get(self._viz_linewidth_key(viz_type), 2.0)
        fps = viz_settings.get('fps', 10)
        swap_xt = viz_settings.get('swap_xt', True)
        title_override = (viz_settings.get('title') or '').strip()
        xlabel_override = (viz_settings.get('xlabel') or '').strip()
        ylabel_override = (viz_settings.get('ylabel') or '').strip()
        figsize_mode = viz_settings.get('figsize_mode', 'Default')
        figsize_w = viz_settings.get('figsize_w', 7.0)
        figsize_h = viz_settings.get('figsize_h', 5.0)
        title_fontsize = viz_settings.get('title_fontsize', 13.0)
        subplot_title_fontsize = viz_settings.get('subplot_title_fontsize', 10.0)
        axis_label_fontsize = viz_settings.get('axis_label_fontsize', 10.0)

        # Spatial bounds / problem dimension / output names come from
        # whichever config got loaded (the run's top-level
        # model_config.json, or -- when none was found next to a step
        # folder -- that step's own step_config.json via the fallback in
        # _on_browse_restore_model); these don't vary between steps, only
        # t_min/t_max do. t_min/t_max are overridden below with the FULL
        # combined range across every detected step rather than trusting
        # whichever single config happened to load, so a static
        # Surface/Line restore can request a time anywhere across the
        # combined run and an Animation can span the whole thing.
        x_min = cfg.get("x_min", 0.0); x_max = cfg.get("x_max", 1.0)
        y_min = cfg.get("y_min", 0.0); y_max = cfg.get("y_max", 1.0)
        z_min = cfg.get("z_min", 0.0); z_max = cfg.get("z_max", 1.0)
        is_2d = cfg.get("problem_dim", "1D") == "2D"
        is_3d = cfg.get("problem_dim", "1D") == "3D"
        # Same live, independently-overridable slice-value convention as
        # _build_restore_script -- see the matching comment there.
        line_slice_y = (y_min + y_max) / 2.0 if viz_settings.get("line_slice_y_auto", True) else viz_settings.get("line_slice_y", 0.0)
        line_slice_z = (z_min + z_max) / 2.0 if viz_settings.get("line_slice_z_auto", True) else viz_settings.get("line_slice_z", 0.0)
        out_names = cfg.get("output_names", "u").split(",")
        _custom_expr_val = (custom_expr or "").strip()
        _custom_label_val = (custom_label or "").strip()
        if _custom_expr_val:
            out_name = _custom_label_val or _custom_expr_val
        else:
            out_name = out_names[output_idx].strip() if output_idx < len(out_names) else "u"

        t_min = ta_steps[0]['t0']
        t_max = ta_steps[-1]['t1']
        # Same multi-snapshot convention as _build_restore_script -- see
        # its own comment. A Time-Adaptive restore is never steady-state
        # (the whole point of Time-Adaptive training is splitting a real
        # time axis into phases), so this always applies for 2D/3D here,
        # with no is_steady gate needed.
        n_2d_snapshots = viz_settings.get('n_2d_snapshots', 2)

        # Per-step literals baked directly into the generated script as
        # plain data (no step_config.json re-parsing at script-runtime) --
        # these are exactly the dicts/floats _detect_restore_ta_steps
        # already validated when populating self._restore_ta_steps.
        _step_literals = []
        for _s in ta_steps:
            _scfg = _s.get('cfg') or {}
            _step_literals.append({
                "t0": _s['t0'], "t1": _s['t1'], "model_path": _s['model_path'],
                "layers": _scfg.get("layers", cfg.get("layers")),
                "activation": _scfg.get("activation", cfg.get("activation", "tanh")),
                "loss_type": _scfg.get("loss_type", cfg.get("loss_type", "MSE")),
                "is_lbfgs": "lbfgs" in os.path.basename(_s['model_path']).lower(),
            })

        # Network architecture is a whole-run choice (not something that
        # varies step-to-step), so this reads the top-level model_config.json
        # (cfg), not each step's own step_config.json -- same reasoning as
        # this function's docstring already gives for x_min/x_max/etc.
        from pinnstudio.core.codegen import (_net_construction_helper_code,
            _training_monitor_runtime_code, _figsize_helper_code, _contour_levels_helper_code,
            _math_namespace_code)
        net_helper_code = _net_construction_helper_code(cfg.get("network_type", "FNN"), cfg.get("msffn_sigmas", "1, 10"),
                                                        cfg.get("msffn_sigmas_t", "1, 10"))
        # Re-attach the model's input/output transforms (Round 62) -- see
        # codegen._net_transforms_code; a model_config.json from before
        # this round has no transform keys and gets none, as before.
        from pinnstudio.core.codegen import _net_transforms_code
        net_transforms_code = _net_transforms_code(cfg)
        # Same reuse of Training Monitors' derivative-building helpers as
        # _build_restore_script -- see that function's own comment.
        _tm_runtime_code_restore = _training_monitor_runtime_code()
        _restore_dim_str = "3D" if is_3d else ("2D" if is_2d else "1D")

        # Reconstruct the ACTUAL problem geometry (Disk/Ellipse/Triangle/
        # Polygon/Sphere/Custom CSG combo, not just its rectangular/cuboid
        # bounding box) -- same reasoning and the same reused
        # generate_clean_script() helpers as _build_restore_script's own
        # geometry-reconstruction comment (see there for the full
        # rationale). Previously _ta_build_geomtime() below always built a
        # plain Cuboid/Rectangle/Interval, so a Time-Adaptive restore of
        # e.g. an L-Shape/Disk/Triangle problem would evaluate and plot
        # the model across the full bounding box, past the true (possibly
        # non-convex) boundary.
        import types as _restore_types_ta
        from pinnstudio.core.codegen import _clean_geom_line, _parse_vertex_list, _dtype_safe_geom_class_code
        _tri_verts_restore_ta = _parse_vertex_list(cfg.get("geom_triangle_vertices", "0,0;1,0;0,1"))
        if len(_tri_verts_restore_ta) != 3:
            _tri_verts_restore_ta = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        _poly_verts_restore_ta = _parse_vertex_list(cfg.get("geom_polygon_vertices", "0,0;1,0;1,1;0,1"))
        if len(_poly_verts_restore_ta) < 3:
            _poly_verts_restore_ta = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        _geom_cfg_restore_ta = _restore_types_ta.SimpleNamespace(
            geometry_type=cfg.get("geometry_type", "Rectangle"),
            geom_center_x=cfg.get("geom_center_x", 0.5), geom_center_y=cfg.get("geom_center_y", 0.5),
            geom_center_z=cfg.get("geom_center_z", 0.5), geom_radius=cfg.get("geom_radius", 0.5),
            geom_semi_major=cfg.get("geom_semi_major", 0.5), geom_semi_minor=cfg.get("geom_semi_minor", 0.3),
            geom_angle=cfg.get("geom_angle", 0.0),
            geom_custom_shapes_json=cfg.get("geom_custom_shapes_json", "[]"),
            problem_dim=cfg.get("problem_dim", "2D"),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max,
        )
        _geom_line_restore_ta, _geom_needs_dtype_wrap_ta = _clean_geom_line(
            _geom_cfg_restore_ta, is_2d, is_3d, _tri_verts_restore_ta, _poly_verts_restore_ta)
        # Same wrapper class _build_restore_script splices in for the same
        # reason (see codegen.py's _DTypeSafeGeom) -- only included when
        # the reconstructed geometry actually needs it.
        _dtype_safe_geom_class_restore_ta = (_dtype_safe_geom_class_code() + "\n") if _geom_needs_dtype_wrap_ta else ''

        script = f"""
import os
os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Plot Settings: figure-size standardization (see codegen.py's
# _figsize_helper_code for the full rationale) -- "Default" mode renders
# every figsize=_plot_figsize(w, h) call below byte-identical to before
# this existed.
{_figsize_helper_code(figsize_mode, figsize_w, figsize_h)}

# Shared-color-scale contourf levels (see codegen.py's _contour_levels_helper_code).
{_contour_levels_helper_code()}

is_2d = {str(is_2d)}
is_3d = {str(is_3d)}

_ta_steps = {_step_literals!r}
print(f"ℹ️ Time-Adaptive combined restore: {{len(_ta_steps)}} step(s), "
      f"t=[{{_ta_steps[0]['t0']}}, {{_ta_steps[-1]['t1']}}]")

{_dtype_safe_geom_class_restore_ta}
# Built once at module scope (geometry is time-independent, same across
# every step) -- reused both by _ta_build_geomtime() below and directly by
# the Line (time steps)/Animation Line (GIF) branches further down to
# mask a line slice down to the real domain with geom.inside(...).
{_geom_line_restore_ta}

def _ta_build_geomtime(t0, t1):
    timedomain = dde.geometry.TimeDomain(t0, t1)
    return dde.geometry.GeometryXTime(geom, timedomain)

def _ta_pde(x, y): return y[:, 0:1] * 0

{net_helper_code}
{net_transforms_code}

_ta_models = {{}}
def _ta_get_model(i):
    if i in _ta_models:
        return _ta_models[i]
    _s = _ta_steps[i]
    _geomtime = _ta_build_geomtime(_s["t0"], _s["t1"])
    _data = dde.data.TimePDE(_geomtime, _ta_pde, [], num_domain=100, num_test=100)
    _net = _apply_net_transforms(_make_net(_s["layers"], _s["activation"], "Glorot uniform"))
    _model = dde.Model(_data, _net)
    if _s["is_lbfgs"]:
        dde.optimizers.set_LBFGS_options(maxiter=1)
        _model.compile("L-BFGS", loss=_s["loss_type"])
    else:
        _model.compile("adam", lr=0.001, loss=_s["loss_type"])
    try:
        _model.restore(_s["model_path"], verbose=0)
    except Exception as e:
        print(f"❌ Restore error on step {{i+1}}/{{len(_ta_steps)}} ({{_s['model_path']}}): {{e}}")
        raise
    print(f"✅ Restored step {{i+1}}/{{len(_ta_steps)}}: t=[{{_s['t0']:.4g}}, {{_s['t1']:.4g}}] ({{os.path.basename(_s['model_path'])}})")
    _ta_models[i] = _model
    return _model

def _ta_pick_step(tv):
    n = len(_ta_steps)
    for i, _s in enumerate(_ta_steps):
        is_last = (i == n - 1)
        if is_last:
            if _s["t0"] - 1e-9 <= tv <= _s["t1"] + 1e-9:
                return i
        else:
            if (_s["t0"] - 1e-9 <= tv < _s["t1"] - 1e-9) or abs(tv - _s["t1"]) < 1e-9:
                return i
    return 0 if tv <= _ta_steps[0]["t0"] else n - 1

def _ta_model_for_t(tv):
    return _ta_get_model(_ta_pick_step(tv))

os.makedirs({save_dir!r}, exist_ok=True)
x_vals = np.linspace({x_min}, {x_max}, 100)
y_vals = np.linspace({y_min}, {y_max}, 100)
z_vals = np.linspace({z_min}, {z_max}, 100)

{_tm_runtime_code_restore}

# See the matching comment in _build_restore_script -- same mechanism, same
# math namespace, just spliced into this Time-Adaptive-aware script instead.
# One difference from the single-model script: which model.predict(...)
# to call isn't fixed here -- a different step's restored model may apply
# per time value (see _ta_model_for_t/_ta_pick_step above) -- so the model
# object itself is passed in alongside the grid instead of being baked in.
_restore_custom_expr = {_custom_expr_val!r}
_restore_output_names = {[n.strip() for n in out_names]!r}
_restore_n_out = len(_restore_output_names)
_restore_is_steady = False
_restore_dim = {_restore_dim_str!r}
{_math_namespace_code('_RESTORE_MATH_NS', 'torch', '')}
def _restore_custom_op(inputs, outputs):
    _dvars = _tm_build_dvars(inputs, outputs, _restore_n_out, _restore_output_names,
                              _restore_is_steady, _restore_dim, _restore_custom_expr)
    _ns = dict(_dvars)
    _ns.update(_RESTORE_MATH_NS)
    _ns["torch"] = torch
    return eval(_restore_custom_expr, _ns)

def _extract_plot_field(x_grid, _tm_model):
    if _restore_custom_expr:
        return _tm_model.predict(x_grid, operator=_restore_custom_op)[:, 0]
    return _tm_model.predict(x_grid)[:, {output_idx}]
"""

        if viz_type == "Surface":
            _xlabel_3d = xlabel_override or "x"
            _ylabel_3d = ylabel_override or "y"
            _title_3d = title_override or f"Restored Model — {out_name}(x,y,z,t)"
            _xlabel_2d = xlabel_override or "x"
            _ylabel_2d = ylabel_override or "y"
            _title_2d = title_override or f"Restored Model — {out_name}(x,y,t)"
            _xlabel_1d = xlabel_override or ("t" if swap_xt else "x")
            _ylabel_1d = ylabel_override or ("x" if swap_xt else "t")
            _title_1d = title_override or f"Restored Model — {out_name}(x,t) Surface"
            # cleanup Round 45 (audit §2e, sub-round 3): shared with
            # _build_restore_script's own "Surface" branch -- see
            # _restore_surface_code's docstring in codegen.py. A
            # Time-Adaptive restore is never steady-state, so the steady-1D
            # labels and line width are unused here (None).
            from pinnstudio.core.codegen import _restore_surface_code
            script += "\n" + _restore_surface_code(
                resolution, x_min, x_max, y_min, y_max, t_min, t_max, n_2d_snapshots,
                colormap, show_colorbar, levels, swap_xt, auto_range, vmin_val, vmax_val,
                None, title_fontsize, subplot_title_fontsize, axis_label_fontsize,
                save_dir, dpi, _xlabel_3d, _ylabel_3d, _title_3d, _xlabel_2d, _ylabel_2d,
                _title_2d, _xlabel_1d, _ylabel_1d, _title_1d, None, None, None,
                is_time_adaptive=True) + "\n"

        elif viz_type == "Line (time steps)":
            _slice_suffix_line = (f", y={line_slice_y:.3g}, z={line_slice_z:.3g}" if is_3d
                                   else (f", y={line_slice_y:.3g}" if is_2d else ""))
            _xlabel_line = xlabel_override or "x"
            _ylabel_line = ylabel_override or out_name
            _title_line = title_override or f"Restored Model — {out_name}(x,t{_slice_suffix_line}) Line Plot"
            # cleanup Round 42 (audit §2e): shared with _build_restore_script's
            # own "Line (time steps)" branch -- see _restore_line_plot_code's
            # docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_line_plot_code
            script += "\n" + _restore_line_plot_code(
                x_min, x_max, resolution, t_min, t_max, n_steps, colormap,
                line_slice_y, line_slice_z, linewidth, _xlabel_line, _ylabel_line,
                _title_line, subplot_title_fontsize, axis_label_fontsize, save_dir, dpi,
                is_time_adaptive=True) + "\n"
        elif viz_type == "Animation Line (GIF)":
            _slice_suffix_animline = (f", y={line_slice_y:.3g}, z={line_slice_z:.3g}" if is_3d
                                       else (f", y={line_slice_y:.3g}" if is_2d else ""))
            _xlabel_animline = xlabel_override or "x"
            _ylabel_animline = ylabel_override or (f"{out_name}(x,t{_slice_suffix_animline})" if (is_2d or is_3d) else out_name)
            _title_line_stmt = f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override else ""
            # The per-frame "t = ..." indicator below is a blit-compatible
            # Text artist (not a real title -- see its own comment in
            # generate_script()'s matching Line Animation (GIF) branch)
            # normally positioned just above the axes, exactly where a
            # real title (set once, above) would also render -- so when
            # title_override is set, move it inside the axes instead, to
            # avoid the two overlapping into illegible text.
            _tt_x, _tt_y, _tt_ha = (0.02, 0.95, "left") if title_override else (0.5, 1.02, "center")
            # cleanup Round 42 (audit §2e): shared with _build_restore_script's
            # own "Animation Line (GIF)" branch -- see
            # _restore_line_animation_code's docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_line_animation_code
            script += "\n" + _restore_line_animation_code(
                x_min, x_max, t_min, t_max, n_steps, line_slice_y, line_slice_z,
                linewidth, _xlabel_animline, _ylabel_animline, _title_line_stmt,
                _tt_x, _tt_y, _tt_ha, axis_label_fontsize, fps, save_dir, dpi,
                is_time_adaptive=True) + "\n"

        elif viz_type == "Animation Surface (GIF)":
            _xlabel_animsurf3d = xlabel_override or "x"
            _ylabel_animsurf3d = ylabel_override or "y"
            _animsurf_title_line_3d = (
                f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override
                else f'ax.set_title(f"t = {{t_frames[i]:.3f}}", fontsize={subplot_title_fontsize})'
            )
            _xlabel_animsurf_else = xlabel_override or ("t" if (swap_xt and not is_2d) else "x")
            _ylabel_animsurf_else = ylabel_override or ("y" if is_2d else ("x" if swap_xt else "t"))
            _animsurf_title_line_else = (
                f"ax.set_title({title_override!r}, fontsize={subplot_title_fontsize})" if title_override
                else f'ax.set_title(f"t = {{t_frames[i]:.3f}}", fontsize={subplot_title_fontsize})'
            )
            # cleanup Round 43 (audit §2e, sub-round 2): see
            # _restore_anim_surface_code's docstring in codegen.py.
            from pinnstudio.core.codegen import _restore_anim_surface_code
            script += "\n" + _restore_anim_surface_code(
                x_min, x_max, y_min, y_max, t_min, t_max, n_steps, colormap,
                show_colorbar, levels, swap_xt, axis_label_fontsize, fps, save_dir,
                _xlabel_animsurf3d, _ylabel_animsurf3d, _animsurf_title_line_3d,
                _xlabel_animsurf_else, _ylabel_animsurf_else, _animsurf_title_line_else,
                is_time_adaptive=True) + "\n"
        script += '\nprint("RESTORE_DONE")\n'
        return script

    def _build_restore_ea_script(self, files, save_dir, is_2d, do_line, do_surface,
                                  x_min, x_max, y_min, y_max, out_name, viz_settings=None,
                                  is_3d=False, output_idx=0, custom_expr="", output_names="u",
                                  is_steady=False, is_ta=False,
                                  z_min=0.0, z_max=1.0, line_slice_y=None, line_slice_z=None,
                                  ea_metric_l2=True, ea_metric_mse=False, ea_metric_max=False):
        # line_slice_y/_z: the same configured 2D/3D line-plot slice value
        # (PINNConfig.line_slice_y/_z) the restore script's own Line plot
        # uses. None means "caller didn't pass one" (e.g. an older call
        # site) -- fall back to the domain midpoint, matching this
        # feature's behavior everywhere else when left on Auto.
        from pinnstudio.core.codegen import _math_namespace_code
        if line_slice_y is None:
            line_slice_y = (y_min + y_max) / 2.0
        if line_slice_z is None:
            line_slice_z = (z_min + z_max) / 2.0
        if viz_settings is None:
            viz_settings = {}
        _cmap     = viz_settings.get('colormap', 'viridis')
        _levels   = viz_settings.get('levels', 40)
        _dpi      = viz_settings.get('dpi', 150)
        _colorbar = viz_settings.get('colorbar', True)
        _auto     = viz_settings.get('auto_range', True)
        _vmin     = viz_settings.get('vmin', -1.0)
        _vmax     = viz_settings.get('vmax', 1.0)
        # Round 31: same 3 font-size settings as the restore surface/line
        # builders above (see _build_restore_script's matching comment).
        _title_fs      = viz_settings.get('title_fontsize', 13.0)
        _subtitle_fs   = viz_settings.get('subplot_title_fontsize', 10.0)
        _axislabel_fs  = viz_settings.get('axis_label_fontsize', 10.0)
        # Same resolution the restore script's own Line plot uses for its
        # true-line x scan -- see the Line comparison section below.
        _ea_line_res = viz_settings.get('resolution', 100)
        files_repr = repr(files)
        # Which model to call .predict() on, per reference time _tv: a
        # Time-Adaptive combined restore (_build_restore_script_ta) never
        # defines a plain `model` -- it restores each detected step's own
        # model lazily and routes by time range via _ta_model_for_t(tv)
        # (defined unconditionally near the top of that generated script,
        # before any viz-type branching -- see that method). Calling this
        # per _tv instead of once is required here anyway: the reference
        # times being compared can span more than one TA step's own
        # window, so no single restored model is even valid for all of
        # them. A plain single-model restore (_build_restore_script) still
        # just uses its own `model` exactly as before.
        _predict_call = "_ta_model_for_t(_tv).predict" if is_ta else "model.predict"
        # Predicted field: a raw output column (output_idx, matching the
        # Restore panel's own "output:" selector) or -- when a custom
        # expression is set, e.g. |h| = sqrt(u**2+v**2) -- a NumPy
        # expression over all of this model's outputs. Previously this
        # always compared against column 0 regardless of which output was
        # actually selected above, silently analyzing the wrong field for
        # any multi-output model where "Output 1" wasn't the one chosen.
        _custom_expr_val = (custom_expr or "").strip()
        # Round 31: build the metrics print/CSV fragments once here,
        # honoring ea_metric_l2/mse/max (the "Error Metrics to Compute"
        # checkboxes) -- t and Mean_abs_error have no corresponding
        # checkbox in that dialog, so both always show, same as the
        # matching codegen.py helpers this mirrors
        # (_ea_metrics_csv_header/_ea_metrics_row_code/
        # _ea_metrics_print_fragment).
        _ea_csv_header = ",".join(
            ["t"]
            + (["L2_relative"] if ea_metric_l2 else [])
            + (["MSE"] if ea_metric_mse else [])
            + (["Max_error"] if ea_metric_max else [])
            + ["Mean_abs_error"]
        )
        _ea_csv_row = ",".join(
            ["{_tv:.6f}"]
            + (["{_l2:.6e}"] if ea_metric_l2 else [])
            + (["{_mse:.6e}"] if ea_metric_mse else [])
            + (["{_mx:.6e}"] if ea_metric_max else [])
            + ["{_ma:.6e}"]
        )
        _ea_print_frag = ", ".join(
            (["L2={_l2:.4e}"] if ea_metric_l2 else [])
            + (["MSE={_mse:.4e}"] if ea_metric_mse else [])
            + (["Max={_mx:.4e}"] if ea_metric_max else [])
            + ["MeanAbs={_ma:.4e}"]
        )
        _extract_field_code = f'''_ea_custom_expr = {_custom_expr_val!r}
_ea_output_names = {[n.strip() for n in output_names.split(",")]!r}
{_math_namespace_code("_EA_MATH_NS", "np", "")}
def _extract_restore_field(pred):
    if _ea_custom_expr:
        ns = {{**_EA_MATH_NS, "np": np}}
        for _i, _n in enumerate(_ea_output_names):
            ns[_n] = pred[:, _i]
        return np.asarray(eval(_ea_custom_expr, ns))
    return pred[:, {output_idx}]
'''
        return f"""

# ── Restore Error Analysis ────────────────────────────────────
import numpy as np
from scipy.interpolate import interp1d as _interp1d
_ea_dir = os.path.join({save_dir!r}, "error_analysis")
os.makedirs(_ea_dir, exist_ok=True)
print("\\n=== Running Restore Error Analysis ===")

{_extract_field_code}
_ea_files = {files_repr}
_ea_times = []; _ea_x_refs = []; _ea_y_refs = []; _ea_z_refs = []; _ea_u_refs = []
for _tv, _fp in _ea_files:
    _d = np.loadtxt(_fp)
    if _d.ndim == 1: _d = _d.reshape(1, -1)
    _d_shape = _d.shape[1]
    if {is_steady} and {is_3d}:
        # Steady 3D reference format: x, y, z, u — no time column at all.
        _idx = np.lexsort((_d[:, 2], _d[:, 1], _d[:, 0]))
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(_d[_idx, 1])
        _ea_z_refs.append(_d[_idx, 2])
        _ea_u_refs.append(_d[_idx, 3])
        _ea_times.append(0.0)
    elif {is_steady} and {is_2d}:
        # Steady 2D reference format: x, y, u — no time column at all.
        _idx = np.lexsort((_d[:, 1], _d[:, 0]))
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(_d[_idx, 1])
        _ea_z_refs.append(np.zeros_like(_d[_idx, 0]))
        _ea_u_refs.append(_d[_idx, 2])
        _ea_times.append(0.0)
    elif {is_3d}:
        # 3D time-dependent reference format: x, y, z, t, u
        _idx = np.lexsort((_d[:, 2], _d[:, 1], _d[:, 0]))
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(_d[_idx, 1])
        _ea_z_refs.append(_d[_idx, 2])
        _ea_u_refs.append(_d[_idx, 4])
        _ea_times.append(float(_d[0, 3]))
    elif {is_2d}:
        _idx = np.lexsort((_d[:, 1], _d[:, 0]))
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(_d[_idx, 1])
        _ea_z_refs.append(np.zeros_like(_d[_idx, 0]))
        _ea_u_refs.append(_d[_idx, 3])
        _ea_times.append(float(_d[0, 2]))
    else:
        _idx = np.argsort(_d[:, 0])
        _ea_x_refs.append(_d[_idx, 0])
        _ea_y_refs.append(np.zeros_like(_d[_idx, 0]))
        _ea_z_refs.append(np.zeros_like(_d[_idx, 0]))
        _ea_u_refs.append(_d[_idx, 2])
        _ea_times.append(float(_tv))
    if {is_steady}:
        print(f"  Loaded ground truth (steady-state): {{len(_d)}} pts from {{os.path.basename(_fp)}}")
    else:
        print(f"  Loaded t={{_ea_times[-1]:.4f}}: {{len(_d)}} pts from {{os.path.basename(_fp)}}")

_ea_n_t = len(_ea_times)
_ea_u_pinns = []
for _i, _tv in enumerate(_ea_times):
    _xf = _ea_x_refs[_i]
    if {is_steady} and {is_3d}:
        _xt = np.column_stack([_xf, _ea_y_refs[_i], _ea_z_refs[_i]])
    elif {is_steady} and {is_2d}:
        _xt = np.column_stack([_xf, _ea_y_refs[_i]])
    elif {is_3d}:
        _yf = _ea_y_refs[_i]; _zf = _ea_z_refs[_i]
        _xt = np.column_stack([_xf, _yf, _zf, np.full_like(_xf, _tv)])
    elif {is_2d}:
        _yf = _ea_y_refs[_i]
        _xt = np.column_stack([_xf, _yf, np.full_like(_xf, _tv)])
    else:
        _xt = np.column_stack([_xf, np.full_like(_xf, _tv)])
    _ea_u_pinns.append(_extract_restore_field({_predict_call}(_xt)).flatten())
    if {is_steady}:
        print(f"  PINN predicted (steady-state): {{len(_xf)}} points")
    else:
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
    print(f"  t={{_tv:.4f}} — {_ea_print_frag}")

with open(os.path.join(_ea_dir, "error_metrics_restore.txt"), "w") as _mf:
    _mf.write("{_ea_csv_header}\\n")
    for _tv, _l2, _mse, _mx, _ma in _ea_metrics:
        _mf.write(f"{_ea_csv_row}\\n")
print(f"  Metrics saved: {{os.path.join(_ea_dir, 'error_metrics_restore.txt')}}")

# Line comparison
if {do_line}:
    _ncols = min(4, _ea_n_t)
    _nrows = (_ea_n_t + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(4*_ncols, 3.5*_nrows), squeeze=False)
    _ea_line_suptitle = "Restored Model vs Reference — Line Comparison"
    if {is_3d}:
        _ea_line_suptitle += f" (y={line_slice_y:.3g}, z={line_slice_z:.3g})"
    elif {is_2d}:
        _ea_line_suptitle += f" (y={line_slice_y:.3g})"
    fig.suptitle(_ea_line_suptitle.replace(" — Line Comparison", ""), fontsize={_subtitle_fs} + 2)
    _ax_flat = axes.flatten()
    if {is_2d} or {is_3d}:
        # A true line at the exact slice (the restore panel's own
        # line_slice_y/_z), masked to the real geometry with
        # geom.inside(...) -- not a scatter of whichever reference-file
        # points happened to fall within a tolerance BAND around it. The
        # PINN curve is evaluated directly on that line, and the
        # reference is interpolated onto it from its own scattered points
        # via griddata (nearest-neighbor fallback for any point outside
        # its convex hull). Same fix Rounds 36/37 already gave
        # generate_script()'s own Error Analysis Line Comparison (see its
        # comment for the full "why" -- the previous tolerance-band
        # approach smeared multiple y/z values onto the same x, especially
        # badly on a non-box geometry). geom is the real reconstructed
        # geometry built above, for both the single-model and
        # Time-Adaptive restore paths.
        from scipy.interpolate import griddata as _gd_line
        _xv_line = np.linspace({x_min}, {x_max}, {_ea_line_res})
        _y_mid = {line_slice_y!r}
        _z_mid = {line_slice_z!r}
    for _i in range(_ea_n_t):
        ax = _ax_flat[_i]
        _tv, _l2, _mse, _mx, _ma = _ea_metrics[_i]
        if {is_3d}:
            _line_pts3 = np.column_stack([_xv_line, np.full_like(_xv_line, _y_mid), np.full_like(_xv_line, _z_mid)])
            _line_in3 = geom.inside(_line_pts3)
            _xv_s = _xv_line[_line_in3]
            _line_pts3_in = _line_pts3[_line_in3]
            if len(_xv_s) < 2:
                ax.text(0.5, 0.5, "Line outside the domain at this (y, z) slice", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            _line_eval3 = _line_pts3_in if {is_steady} else np.column_stack([_line_pts3_in, np.full(len(_xv_s), _tv)])
            _pinn_s = _extract_restore_field({_predict_call}(_line_eval3)).flatten()
            _ref_pts3 = np.column_stack([_ea_x_refs[_i], _ea_y_refs[_i], _ea_z_refs[_i]])
            _gt_s = _gd_line(_ref_pts3, _ea_u_refs[_i], _line_pts3_in, method='linear')
            _gt_nan3 = np.isnan(_gt_s)
            if _gt_nan3.any():
                _gt_s[_gt_nan3] = _gd_line(_ref_pts3, _ea_u_refs[_i], _line_pts3_in[_gt_nan3], method='nearest')
        elif {is_2d}:
            _line_pts2 = np.column_stack([_xv_line, np.full_like(_xv_line, _y_mid)])
            _line_in2 = geom.inside(_line_pts2)
            _xv_s = _xv_line[_line_in2]
            _line_pts2_in = _line_pts2[_line_in2]
            if len(_xv_s) < 2:
                ax.text(0.5, 0.5, "Line outside the domain at this y slice", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            _line_eval2 = _line_pts2_in if {is_steady} else np.column_stack([_line_pts2_in, np.full(len(_xv_s), _tv)])
            _pinn_s = _extract_restore_field({_predict_call}(_line_eval2)).flatten()
            _ref_pts2 = np.column_stack([_ea_x_refs[_i], _ea_y_refs[_i]])
            _gt_s = _gd_line(_ref_pts2, _ea_u_refs[_i], _line_pts2_in, method='linear')
            _gt_nan2 = np.isnan(_gt_s)
            if _gt_nan2.any():
                _gt_s[_gt_nan2] = _gd_line(_ref_pts2, _ea_u_refs[_i], _line_pts2_in[_gt_nan2], method='nearest')
        else:
            _xv = _ea_x_refs[_i]
            _ea_sort = np.argsort(_xv)
            _xv_s   = _xv[_ea_sort]
            _gt_s   = _ea_u_refs[_i][_ea_sort]
            _pinn_s = _ea_u_pinns[_i][_ea_sort]
        ax.plot(_xv_s, _gt_s,   color='#4dabf7', linewidth=2.0, label='Reference')
        ax.plot(_xv_s, _pinn_s, color='#ff6b6b', linewidth=2.0, linestyle='--', label='PINN')
        ax.set_title(f"t = {{_tv:.3g}}  ({{_l2_txt(_l2)}})", fontsize={_subtitle_fs})
        _ea_line_ylabel = (f"{out_name}(x,y={line_slice_y:.3g},z={line_slice_z:.3g})" if {is_3d}
                            else (f"{out_name}(x,y={line_slice_y:.3g})" if {is_2d} else "{out_name}"))
        ax.set_xlabel("x", fontsize={_axislabel_fs}); ax.set_ylabel(_ea_line_ylabel, fontsize={_axislabel_fs}); ax.grid(True, alpha=0.3)
    for _j in range(_ea_n_t, len(_ax_flat)):
        _ax_flat[_j].set_visible(False)
    _ax_flat[0].legend(loc='best', fontsize=10, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _lp = os.path.join(_ea_dir, "line_comparison_restore.png")
    plt.savefig(_lp, dpi={_dpi}, bbox_inches='tight'); plt.close()
    print(f"  Line comparison saved: {{_lp}}")

# Surface comparison
if {do_surface} and {is_3d}:
    print("  ⚠️  Surface comparison isn't available yet for a restored 3D "
          "model -- skipping (metrics above and the line comparison below, "
          "if enabled, are unaffected).")
elif {do_surface}:
    if {is_2d}:
        from scipy.interpolate import griddata as _gd
        _res_ea = 60
        _xg_ea = np.linspace({x_min}, {x_max}, _res_ea)
        _yg_ea = np.linspace({y_min}, {y_max}, _res_ea)
        _Xg_ea, _Yg_ea = np.meshgrid(_xg_ea, _yg_ea)
        fig, axes = plt.subplots(_ea_n_t, 3, figsize=(15, 4*_ea_n_t), squeeze=False)
        fig.suptitle("Restored Model vs Reference — 2D Heatmaps", fontsize={_title_fs}, fontweight='bold')
        for _i, _tv in enumerate(_ea_times):
            _tv_r, _l2, _mse, _mx, _ma = _ea_metrics[_i]
            if {is_steady}:
                _xyt_g = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel()])
            else:
                _xyt_g = np.column_stack([_Xg_ea.ravel(), _Yg_ea.ravel(), np.full(_Xg_ea.size, _tv)])
            _u_pinn_g = _extract_restore_field({_predict_call}(_xyt_g)).reshape(_res_ea, _res_ea)
            _u_fem_g  = _gd(np.column_stack([_ea_x_refs[_i], _ea_y_refs[_i]]),
                            _ea_u_refs[_i], (_Xg_ea, _Yg_ea), method='linear', fill_value=0.0)
            _u_err_g  = np.abs(_u_pinn_g - _u_fem_g)
            _vmin_data = min(_u_pinn_g.min(), _u_fem_g.min()) if {_auto} else {_vmin}
            _vmax_data = max(_u_pinn_g.max(), _u_fem_g.max()) if {_auto} else {_vmax}
            _ea_pinn_title = f"PINN (steady-state) {{_l2_txt(_l2)}}" if {is_steady} else f"PINN t={{_tv:.3f}} {{_l2_txt(_l2)}}"
            _ea_gt_title = "Reference" if {is_steady} else f"Reference t={{_tv:.3f}}"
            im0 = axes[_i][0].contourf(_Xg_ea, _Yg_ea, _u_pinn_g, cmap='{_cmap}', **_cf_levels({_levels}, _vmin_data, _vmax_data, _u_pinn_g))
            axes[_i][0].set_title(_ea_pinn_title, fontsize={_subtitle_fs}); axes[_i][0].set_xlabel("x", fontsize={_axislabel_fs}); axes[_i][0].set_ylabel("y", fontsize={_axislabel_fs})
            if {_colorbar}: fig.colorbar(im0, ax=axes[_i][0])
            im1 = axes[_i][1].contourf(_Xg_ea, _Yg_ea, _u_fem_g, cmap='{_cmap}', **_cf_levels({_levels}, _vmin_data, _vmax_data, _u_fem_g))
            axes[_i][1].set_title(_ea_gt_title, fontsize={_subtitle_fs}); axes[_i][1].set_xlabel("x", fontsize={_axislabel_fs}); axes[_i][1].set_ylabel("y", fontsize={_axislabel_fs})
            if {_colorbar}: fig.colorbar(im1, ax=axes[_i][1])
            im2 = axes[_i][2].contourf(_Xg_ea, _Yg_ea, _u_err_g, levels={_levels}, cmap='{_cmap}')
            axes[_i][2].set_title(f"|Error| max = {{_sci_txt(_mx)}}", fontsize={_subtitle_fs}); axes[_i][2].set_xlabel("x", fontsize={_axislabel_fs}); axes[_i][2].set_ylabel("y", fontsize={_axislabel_fs})
            if {_colorbar}: fig.colorbar(im2, ax=axes[_i][2])
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        _x_common = np.linspace({x_min}, {x_max}, 300)
        _t_arr = np.array(_ea_times)
        _U_pinn = np.zeros((len(_t_arr), len(_x_common)))
        _U_fem  = np.zeros((len(_t_arr), len(_x_common)))
        for _i, _tv in enumerate(_ea_times):
            _xt_c = np.column_stack([_x_common, np.full_like(_x_common, _tv)])
            _U_pinn[_i] = _extract_restore_field({_predict_call}(_xt_c)).flatten()
            _fi = _interp1d(_ea_x_refs[_i], _ea_u_refs[_i], kind='linear', fill_value='extrapolate')
            _U_fem[_i]  = _fi(_x_common)
        _Xg, _Tg = np.meshgrid(_x_common, _t_arr)
        _U_err = np.abs(_U_pinn - _U_fem)
        _vmin_data = min(_U_pinn.min(), _U_fem.min()) if {_auto} else {_vmin}
        _vmax_data = max(_U_pinn.max(), _U_fem.max()) if {_auto} else {_vmax}
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Restored Model vs Reference — Surface", fontsize={_title_fs}, fontweight='bold')
        im0 = axes[0].contourf(_Tg, _Xg, _U_pinn, cmap='{_cmap}', **_cf_levels({_levels}, _vmin_data, _vmax_data, _U_pinn))
        axes[0].set_title("PINN", fontsize={_subtitle_fs}); axes[0].set_xlabel("t", fontsize={_axislabel_fs}); axes[0].set_ylabel("x", fontsize={_axislabel_fs})
        if {_colorbar}: fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].contourf(_Tg, _Xg, _U_fem, cmap='{_cmap}', **_cf_levels({_levels}, _vmin_data, _vmax_data, _U_fem))
        axes[1].set_title("Reference", fontsize={_subtitle_fs}); axes[1].set_xlabel("t", fontsize={_axislabel_fs})
        if {_colorbar}: fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].contourf(_Tg, _Xg, _U_err, levels={_levels}, cmap='{_cmap}')
        axes[2].set_title("|Error|", fontsize={_subtitle_fs}); axes[2].set_xlabel("t", fontsize={_axislabel_fs})
        if {_colorbar}: fig.colorbar(im2, ax=axes[2])
        plt.tight_layout(rect=[0, 0, 1, 0.93])
    _sp = os.path.join(_ea_dir, "surface_comparison_restore.png")
    plt.savefig(_sp, dpi={_dpi}, bbox_inches='tight'); plt.close()
    print(f"  Surface comparison saved: {{_sp}}")

print("=== Restore Error Analysis Complete ===")
"""
