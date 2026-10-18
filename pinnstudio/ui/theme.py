"""PINNStudio's look: colour themes as tokens, and the stylesheet built from them.

A theme is a small set of named colours ("tokens") -- background, panels,
text, muted text, one accent, and status colours for success, warning and
error. Every stylesheet in the app is built from them, so switching theme
changes everything consistently.

Many widgets still set their own small stylesheets with colours written for
the original dark look (a light-blue hint, a green "Add" button...). Rather
than editing each of them, `remap_css` translates those colours to the
current theme's tokens by their role -- text colour, background or border,
normal or hover -- and `StyleRemapper` applies it to every widget whenever
its stylesheet is set. New colours in old stylesheets are therefore themed
automatically; a colour it doesn't know is left as written.

Fonts: the operating system's own UI and fixed-width fonts, at desktop
sizes (14 px text), multiplied by a UI scale (Display Settings).
"""
import re

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtGui import QColor, QFontDatabase, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

# ── themes ───────────────────────────────────────────────────────────────
THEMES = {
    "Studio Light": dict(
        dark=False,
        bg="#f4f5f7", surface="#e9ecf0", card="#ffffff", input="#ffffff",
        border="#d3d8de", border_strong="#b6bec8",
        text="#1f2328", text_muted="#5c6672", text_faint="#8b949e",
        accent="#2f6fde", accent_hover="#255bbd", accent_text="#2459b8", on_accent="#ffffff",
        sel="rgba(47, 111, 222, 0.14)", hover="rgba(47, 111, 222, 0.07)",
        btn="#ffffff", btn_hover="#eef3fb", btn_border="#c9d0d8",
        disabled_bg="#eceef1", disabled_text="#a3abb5",
        success="#1a7f37", warning="#9a5b00", danger="#cf222e", danger_hover="#a40e26", info="#0969da",
        warning_bg="rgba(212, 130, 0, 0.10)", warning_border="rgba(212, 130, 0, 0.45)",
        log_bg="#ffffff", log_text="#24292f", scroll="#c3c9d0",
    ),
    "Studio Dark": dict(
        dark=True,
        bg="#1e1f22", surface="#2b2d30", card="#232427", input="#2b2d30",
        border="#3c3f44", border_strong="#4e5157",
        text="#dfe1e5", text_muted="#9da0a8", text_faint="#6f737a",
        accent="#3574f0", accent_hover="#4682f5", accent_text="#78a6f8", on_accent="#ffffff",
        sel="rgba(53, 116, 240, 0.30)", hover="rgba(53, 116, 240, 0.12)",
        btn="#2b2d30", btn_hover="#35383c", btn_border="#4e5157",
        disabled_bg="#26282b", disabled_text="#62666d",
        success="#5fb865", warning="#e0a240", danger="#f0625a", danger_hover="#d64a43", info="#5aa1f2",
        warning_bg="rgba(224, 162, 64, 0.12)", warning_border="rgba(224, 162, 64, 0.50)",
        log_bg="#18191b", log_text="#c9ccd1", scroll="#4e5157",
    ),
    "Solarized Dark": dict(
        dark=True,
        bg="#002b36", surface="#073642", card="#00303c", input="#073642",
        border="#24525e", border_strong="#3b6874",
        text="#e3e8e8", text_muted="#93a1a1", text_faint="#657b83",
        accent="#268bd2", accent_hover="#3497dc", accent_text="#6cb6e8", on_accent="#ffffff",
        sel="rgba(38, 139, 210, 0.32)", hover="rgba(38, 139, 210, 0.13)",
        btn="#073642", btn_hover="#0d4452", btn_border="#3b6874",
        disabled_bg="#05303b", disabled_text="#4f6b73",
        success="#93b300", warning="#d4a017", danger="#ef5350", danger_hover="#dc322f", info="#2aa198",
        warning_bg="rgba(212, 160, 23, 0.12)", warning_border="rgba(212, 160, 23, 0.50)",
        log_bg="#001e26", log_text="#c9d4d6", scroll="#3b6874",
    ),
}
DEFAULT_THEME = "Studio Light"
ACCENTS = {   # optional override of the theme's accent: (accent, hover, as text on light, as text on dark)
    "Theme default": None,
    "Blue": ("#2f6fde", "#255bbd", "#2459b8", "#78a6f8"),
    "Teal": ("#0f8b8d", "#0c7274", "#0b6f71", "#4cc3c5"),
    "Indigo": ("#5b5bd6", "#4848b8", "#4a4abf", "#9b9bf0"),
    "Green": ("#2e8540", "#256b34", "#22692f", "#6cc77d"),
    "Purple": ("#8e4ec6", "#7639ad", "#7a3cb0", "#c39af0"),
}

# Boundary-condition colours (sketch edges, BC nodes): Okabe-Ito, which
# people with red-green colour blindness can tell apart.
BC_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#882255", "#44AA99",
             "#332288", "#999933")

_current = dict(THEMES[DEFAULT_THEME])


def tokens(name=None, accent="Theme default"):
    t = dict(THEMES.get(name or DEFAULT_THEME, THEMES[DEFAULT_THEME]))
    a = ACCENTS.get(accent)
    if a:
        t["accent"], t["accent_hover"] = a[0], a[1]
        t["accent_text"] = a[3] if t["dark"] else a[2]
        c = QColor(a[0])
        t["sel"] = f"rgba({c.red()}, {c.green()}, {c.blue()}, {0.30 if t['dark'] else 0.14})"
        t["hover"] = f"rgba({c.red()}, {c.green()}, {c.blue()}, {0.12 if t['dark'] else 0.07})"
    return t


def set_current(t):
    _current.clear()
    _current.update(t)


def current():
    return _current


# ── fonts ────────────────────────────────────────────────────────────────
def ui_font_family():
    try:
        return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    except Exception:
        return "Segoe UI"


def mono_font_family():
    try:
        return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
    except Exception:
        return "Consolas"


# ── the stylesheet ───────────────────────────────────────────────────────
def build_qss(t, family, size, heading_size, label_css, header_css, log_family, log_size,
              panel_border=None, panel_border_width=1):
    pb = panel_border or t["border"]
    return f"""
QWidget {{ background: transparent; color: {t['text']}; font-family: '{family}'; font-size: {size}px;
           font-weight: normal; }}
QMainWindow, QDialog, QMessageBox, QInputDialog, QFileDialog, QColorDialog {{ background: {t['bg']}; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QAbstractScrollArea {{ background: {t['bg']}; }}
QToolTip {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border_strong']}; padding: 4px 6px; }}

QGroupBox {{
    background: {t['card']}; border: {panel_border_width}px solid {pb}; border-radius: 6px;
    margin-top: 16px; padding: 10px 8px 8px 8px; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px;
    background: transparent; {header_css}
}}

QLabel {{ background: transparent; {label_css} }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 7px; {label_css} }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {t['border_strong']}; background: {t['input']};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t['accent']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
QRadioButton::indicator:checked {{ background: {t['input']}; border: 1px solid {t['accent']}; image: url("{dot(t['accent'])}"); }}
QCheckBox:disabled, QRadioButton:disabled, QLabel:disabled {{ color: {t['disabled_text']}; }}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit {{
    background: {t['input']}; border: 1px solid {t['border']}; border-radius: 4px;
    padding: 3px 6px; color: {t['text']}; min-height: 20px;
    selection-background-color: {t['accent']}; selection-color: {t['on_accent']};
}}
QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{ border-color: {t['border_strong']}; }}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {t['accent']};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background: {t['disabled_bg']}; color: {t['disabled_text']};
}}
QLineEdit:read-only {{ background: {t['disabled_bg']}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent; border: none; width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{ background: {t['hover']}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{arrow(t['text_muted'], 'up')}"); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{arrow(t['text_muted'], 'down')}"); width: 9px; height: 9px; }}
QComboBox::drop-down {{ background: transparent; border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{arrow(t['text_muted'], 'down')}"); width: 10px; height: 10px; }}
QCheckBox::indicator:checked {{ image: url("{check(t['on_accent'])}"); }}
QComboBox QAbstractItemView {{
    background: {t['card']}; color: {t['text']}; border: 1px solid {t['border_strong']};
    selection-background-color: {t['sel']}; selection-color: {t['text']}; outline: 0;
}}

QPushButton {{
    background: {t['btn']}; color: {t['text']}; border: 1px solid {t['btn_border']};
    border-radius: 5px; padding: 4px 12px; min-height: 20px;
}}
QPushButton:hover {{ background: {t['btn_hover']}; border-color: {t['accent']}; }}
QPushButton:pressed {{ background: {t['sel']}; }}
QPushButton:checked {{ background: {t['sel']}; border-color: {t['accent']}; color: {t['text']}; }}
QPushButton:disabled {{ background: {t['disabled_bg']}; color: {t['disabled_text']}; border-color: {t['border']}; }}
QPushButton:focus {{ outline: none; }}

QScrollArea {{ border: none; background: {t['bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t['scroll']}; border-radius: 5px; min-height: 30px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {t['scroll']}; border-radius: 5px; min-width: 30px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTextEdit {{
    background: {t['log_bg']}; color: {t['log_text']}; border: 1px solid {t['border']}; border-radius: 6px;
    font-family: '{log_family}'; font-size: {log_size}px; padding: 4px;
    selection-background-color: {t['accent']}; selection-color: {t['on_accent']};
}}
QSplitter::handle {{ background: {t['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {t['accent']}; }}
QMenuBar {{ background: {t['surface']}; color: {t['text']}; border-bottom: 1px solid {t['border']}; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background: {t['sel']}; }}
QMenu {{ background: {t['card']}; color: {t['text']}; border: 1px solid {t['border_strong']}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t['sel']}; }}
QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 6px; background: {t['card']}; }}
QTabBar::tab {{
    background: transparent; color: {t['text_muted']}; padding: 6px 14px; border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QListWidget, QTreeWidget, QTableWidget {{
    background: {t['card']}; color: {t['text']}; border: 1px solid {t['border']}; border-radius: 4px; outline: 0;
}}
QListWidget::item:selected, QTableWidget::item:selected {{ background: {t['sel']}; color: {t['text']}; }}
QHeaderView::section {{ background: {t['surface']}; color: {t['text_muted']}; border: none; padding: 4px; }}

QToolButton#collapseButton {{
    background: {t['card']}; color: {t['text_muted']}; border: none; font-size: {size}px; font-weight: 600;
}}
QToolButton#collapseButton:hover {{ color: {t['accent']}; }}

/* ── workspace frame ── */
QWidget#appToolbar {{ background: {t['surface']}; border-bottom: 1px solid {t['border']}; }}
QWidget#appToolbar QLabel {{ background: transparent; }}
QWidget#appToolbar QLabel#toolbarLabel {{ color: {t['text_muted']}; }}
QPushButton#toolButton {{ background: transparent; border: 1px solid transparent; padding: 0 10px; }}
QPushButton#toolButton:hover {{ background: {t['hover']}; border-color: {t['border_strong']}; }}
QPushButton#problemsButton {{
    background: {t['warning_bg']}; color: {t['warning']}; border: 1px solid {t['warning_border']};
    border-radius: 5px; padding: 0 10px; font-weight: 600;
}}
QWidget#treePanel {{ background: {t['surface']}; }}
QTreeWidget#modelTree {{ background: {t['surface']}; border: none; color: {t['text']}; }}
QTreeWidget#modelTree::item {{ padding: 4px 4px; border-radius: 4px; }}
QTreeWidget#modelTree::item:hover {{ background: {t['hover']}; }}
QTreeWidget#modelTree::item:selected {{ background: {t['sel']}; color: {t['text']}; }}
QTreeWidget#modelTree {{ show-decoration-selected: 0; selection-background-color: transparent; }}
QTreeWidget#modelTree::branch, QTreeWidget#modelTree::branch:selected {{ background: transparent; }}
QTreeWidget#modelTree::branch:has-children:!has-siblings:closed, QTreeWidget#modelTree::branch:closed:has-children:has-siblings {{
    image: url("{arrow(t['text_muted'], 'right')}");
}}
QTreeWidget#modelTree::branch:open:has-children:!has-siblings, QTreeWidget#modelTree::branch:open:has-children:has-siblings {{
    image: url("{arrow(t['text_muted'], 'down')}");
}}
QWidget#settingsPanel, QScrollArea#settingsScroll {{ background: {t['bg']}; }}
QWidget#rightPanel {{ background: {t['bg']}; }}
QLabel#panelHeader {{
    color: {t['text_faint']}; font-size: {max(10, size - 3)}px; font-weight: 600;
    letter-spacing: 1px; padding: 2px 4px 4px 4px;
}}
QLabel#pageTitle {{ color: {t['text']}; font-size: {heading_size}px; font-weight: 600; padding: 2px 4px 0 4px; }}
QLabel#pageDesc {{ color: {t['text_muted']}; padding: 0 4px 6px 4px; }}
QFrame#pageRule {{ background: {t['border']}; border: none; }}
QLabel#problemBanner {{
    color: {t['warning']}; background: {t['warning_bg']}; border: 1px solid {t['warning_border']};
    border-radius: 5px; padding: 6px 8px; margin: 0 4px 6px 4px;
}}
QTabWidget#graphicsTabs::pane, QTabWidget#bottomTabs::pane {{ border: none; background: {t['bg']}; }}
QPushButton#solveButton, QPushButton#primaryButton {{
    background: {t['accent']}; color: {t['on_accent']}; border: 1px solid {t['accent']};
    border-radius: 5px; font-weight: 600; padding: 0 14px;
}}
QPushButton#primaryButton {{ min-height: 24px; padding: 5px 14px; }}
QPushButton#solveButton:hover, QPushButton#primaryButton:hover {{ background: {t['accent_hover']}; border-color: {t['accent_hover']}; }}
QPushButton#solveButton:disabled, QPushButton#primaryButton:disabled {{ background: {t['disabled_bg']}; color: {t['disabled_text']}; border-color: {t['border']}; }}
QPushButton#stopButton {{
    background: {t['danger']}; color: #ffffff; border: 1px solid {t['danger']};
    border-radius: 5px; font-weight: 600; padding: 0 14px;
}}
QPushButton#stopButton:hover {{ background: {t['danger_hover']}; }}
QPushButton#stopButton:disabled {{ background: transparent; color: {t['disabled_text']}; border-color: {t['border']}; }}
QWidget#startPage, QScrollArea#startScroll {{ background: {t['bg']}; }}
QLabel#startTitle {{ color: {t['text']}; font-size: {heading_size + 10}px; font-weight: 600; }}
QLabel#startSubtitle {{ color: {t['text_muted']}; }}
QLabel#startSection {{
    color: {t['text_faint']}; font-size: {max(10, size - 3)}px; font-weight: 600; letter-spacing: 1px;
    padding-top: 10px;
}}
QPushButton#startCard {{
    background: {t['card']}; border: 1px solid {t['border']}; border-radius: 8px;
    padding: 10px 12px; text-align: left; color: {t['text']};
}}
QPushButton#startCard:hover {{ border-color: {t['accent']}; background: {t['hover']}; }}
QLabel#startCardTitle {{ color: {t['text']}; font-weight: 600; background: transparent; }}
QLabel#startCardMeta {{ color: {t['text_muted']}; background: transparent; }}
QPushButton#recentLink {{
    background: transparent; border: none; text-align: left; padding: 3px 4px; color: {t['accent_text']};
}}
QPushButton#recentLink:hover {{ text-decoration: underline; }}
QLabel#recentPath {{ color: {t['text_faint']}; }}
QPlainTextEdit#scriptView {{
    background: {t['log_bg']}; color: {t['log_text']}; border: none;
    font-family: "{log_family}"; font-size: {log_size}px;
}}
QLabel#equationNote {{ background: transparent; }}
QToolButton#helpButton {{
    color: {t['text_muted']}; border: 1px solid {t['border_strong']}; border-radius: 11px;
    font-weight: 600; background: transparent; padding: 0;
}}
QToolButton#helpButton:hover {{ color: {t['accent_text']}; border-color: {t['accent']}; }}
"""


# ── small icons for the stylesheet (Qt needs them as files) ──────────────
def _icon_dir():
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "pinnstudio_icons")
    os.makedirs(d, exist_ok=True)
    return d


def _svg_file(name, svg):
    import os
    path = os.path.join(_icon_dir(), name)
    if not os.path.isfile(path):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
        except OSError:
            return ""
    return path.replace("\\", "/")


def arrow(color, direction):
    pts = {"up": "2,7 6,3 10,7", "down": "2,4 6,8 10,4", "right": "4,2 8,6 4,10"}[direction]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
           f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.7" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return _svg_file(f"arrow_{direction}_{color.strip('#')}.svg", svg)


def dot(color):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
           f'<circle cx="7" cy="7" r="4" fill="{color}"/></svg>')
    return _svg_file(f"dot_{color.strip('#')}.svg", svg)


def check(color):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
           f'<polyline points="3,7.5 6,10.5 11,4" fill="none" stroke="{color}" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return _svg_file(f"check_{color.strip('#')}.svg", svg)


# ── translating old dark-theme colours ───────────────────────────────────
def _hexset(*cs):
    return {c.lower() for c in cs}


_TEXT = (
    (_hexset("#a0c4ff", "#74c0fc", "#4dabf7", "#91a7ff", "#1971c2", "#cc5de8", "#b197fc", "#e599f7",
             "#2a5a8a"), "accent_text"),
    (_hexset("#69db7c", "#51cf66", "#2f9e44", "#38d9a9", "#63e6be", "#a0ffb0", "#2a7c4a"), "success"),
    (_hexset("#ffa94d", "#ff922b", "#ffd43b", "#fcc419", "#ffe066"), "warning"),
    (_hexset("#ff8787", "#ff6b6b", "#e03131"), "danger"),
    (_hexset("#ffffff", "#fff", "#e0e0e0", "#c0c0c0"), "text"),
    (_hexset("#8a8a8a", "#868e96", "#808090", "#999999", "#999", "#adb5bd", "#6c7a86", "#8a9aa6",
             "#505080", "#586e75", "#8d9fd0", "#4a5a6a"), "text_muted"),
    (_hexset("#666", "#666666"), "disabled_text"),
)
_BG = (
    (_hexset("#0078d4", "#1a8ae8"), "accent"),
    (_hexset("#005a9e", "#0070c0"), "accent_hover"),
    (_hexset("#6b1f1f"), "danger"),
    (_hexset("#8b2f2f"), "danger_hover"),
    (_hexset("#333355"), "disabled_bg"),
    (_hexset("#252526", "#1e1e1e", "#333338", "#2a3a4a", "#3e3e42", "#586e75", "#073642", "#1a1a2e",
             "#1a3a5a", "#3a2a1a", "#1a4a3a", "#1a5c3a", "#1a4a6a", "#5a3a2a", "#2a5a8a", "#2a6a4a",
             "#3a3a4a", "#3a3a5c", "#0f0f23"), "btn"),
)
_BORDER = (
    (_hexset("#a0c4ff", "#74c0fc", "#1971c2", "#4dabf7"), "accent"),
    (_hexset("#ffa94d", "#ff922b", "#ffd43b"), "warning"),
    (_hexset("#ff8787", "#ff6b6b", "#e03131"), "danger"),
    (_hexset("#69db7c", "#51cf66"), "success"),
    (_hexset("#2a6a4a", "#2a5a8a", "#6a4a2a", "#3e3e42", "#586e75", "#c8d2d8", "#4a5a6a", "#2a3a4a",
             "#3a3a5c", "#333355", "#1a4a3a", "#1a3a5a"), "btn_border"),
)
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_DECL = re.compile(r"(?P<prop>[a-zA-Z-]+)\s*:\s*(?P<val>[^;{}]+)")


def _lookup(table, c):
    c = c.lower()
    for keys, tok in table:
        if c in keys:
            return tok
    return None


def remap_css(css, t=None):
    """A stylesheet written for the original dark look -> the same with its
    colours replaced by the theme's, by role (text / background / border,
    and :hover / :disabled states)."""
    t = t or _current
    if not css or "#" not in css:
        return css
    out, pos = [], 0
    # walk "selector { decls }" blocks, or a bare list of declarations
    for m in re.finditer(r"([^{}]*)\{([^{}]*)\}", css):
        out.append(css[pos:m.start()])
        out.append(m.group(1) + "{" + _remap_decls(m.group(2), t, m.group(1)) + "}")
        pos = m.end()
    tail = css[pos:]
    out.append(_remap_decls(tail, t, "") if pos == 0 else tail)
    return "".join(out)


def _remap_decls(decls, t, selector):
    hover = ":hover" in selector or ":pressed" in selector
    disabled = ":disabled" in selector
    # a background that becomes an ordinary (light) button colour can't keep
    # white text written for the old dark-green / dark-blue fills
    bg_m = re.search(r"background(?:-color)?\s*:\s*[^;]*?(#[0-9a-fA-F]{3,8})", decls)
    plain_bg = bool(bg_m) and _lookup(_BG, bg_m.group(1)) == "btn"

    def repl(m):
        prop, val = m.group("prop").lower(), m.group("val")
        if prop == "color" and plain_bg and val.strip().lower() in ("white", "#fff", "#ffffff"):
            return f"{m.group('prop')}: {t['disabled_text'] if disabled else t['text']}"
        if prop == "color" or prop == "selection-color":
            table, role = _TEXT, "text"
        elif prop.startswith("background") or prop == "selection-background-color":
            table, role = _BG, "bg"
        elif prop.startswith("border"):
            table, role = _BORDER, "border"
        else:
            return m.group(0)

        def sub(h):
            tok = _lookup(table, h.group(0))
            if tok is None:
                return h.group(0)
            if role == "bg" and tok == "btn":
                tok = "disabled_bg" if disabled else ("btn_hover" if hover else "btn")
            if role == "border" and hover and tok == "btn_border":
                tok = "accent"
            if role == "text" and disabled:
                tok = "disabled_text"
            return t.get(tok, h.group(0))
        return f"{m.group('prop')}: {_HEX.sub(sub, val)}"
    return _DECL.sub(repl, decls)


class StyleRemapper(QObject):
    """Application-wide: every time a widget's stylesheet is set, translate
    its colours to the current theme (keeping the original, so a theme
    change can translate it again)."""
    _instance = None

    @classmethod
    def install(cls):
        app = QApplication.instance()
        if app is None or cls._instance is not None:
            return cls._instance
        cls._instance = cls(app)
        app.installEventFilter(cls._instance)
        # Stop filtering before Python shuts down: otherwise Qt, tearing the
        # widgets down after the interpreter has started to finalize, calls
        # back into this Python filter and crashes (seen as a segfault at
        # the end of scripts that never call app.exec()).
        import atexit
        atexit.register(cls._uninstall)
        return cls._instance

    @classmethod
    def _uninstall(cls):
        inst, cls._instance = cls._instance, None
        app = QApplication.instance()
        if inst is not None and app is not None:
            try:
                app.removeEventFilter(inst)
            except RuntimeError:
                pass

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Show and isinstance(obj, QDialog):
            self._fit_dialog(obj)
        if ev.type() == QEvent.Type.FontChange and isinstance(obj, QWidget):
            self._fit_text_button(obj)       # the UI scale changed the text size
            self._fit_combo(obj)
        if ev.type() in (QEvent.Type.StyleChange, QEvent.Type.Polish) and isinstance(obj, QWidget):
            self._fit_text_button(obj)
            self._fit_combo(obj)
            if ev.type() == QEvent.Type.Polish:
                self._fit_small_button(obj)
                from pinnstudio.ui import icons as _icons
                _icons.iconize(obj)
            self.apply(obj)
            self._fit_height(obj)
            if ev.type() == QEvent.Type.Polish:
                self._fit_height_later(obj)
        return False

    @staticmethod
    def _fit_dialog(d):
        """A dialog given a minimum width in code (setMinimumWidth(520)) no
        longer grows to its contents by itself, so with larger text its
        widest rows were cut off. It opens at least as large as they need."""
        lay = d.layout()
        if lay is None:
            return
        need = lay.totalMinimumSize()
        if need.width() > d.minimumWidth():
            d.setMinimumWidth(need.width())
        if need.height() > d.minimumHeight():
            d.setMinimumHeight(need.height())
        if d.width() < need.width() or d.height() < need.height():
            d.resize(max(d.width(), need.width()), max(d.height(), need.height()))

    @staticmethod
    def _fit_combo(w):
        """A drop-down given a fixed width is widened when the option it shows
        does not fit (with a larger UI scale); at most 1.8 times wider."""
        from PyQt6.QtWidgets import QComboBox
        if not isinstance(w, QComboBox) or w.minimumWidth() != w.maximumWidth() or w.maximumWidth() <= 48:
            return
        if not w.property("_pinn_fit_combo"):
            w.setProperty("_pinn_fit_combo", True)
            w.setProperty("_pinn_combo_w", w.maximumWidth())
            w.currentTextChanged.connect(lambda _t, c=w: StyleRemapper._fit_combo(c))
        base = int(w.property("_pinn_combo_w") or w.maximumWidth())
        need = w.fontMetrics().horizontalAdvance(w.currentText()) + 40
        width = min(max(base, need), int(base * 1.8))
        if width != w.maximumWidth():
            w.setFixedWidth(width)

    @staticmethod
    def _fit_text_button(w):
        """A button given a fixed width for an older, smaller font ("Browse"
        at 65 px) is widened until its text fits with the theme's padding."""
        from PyQt6.QtWidgets import QPushButton
        if not isinstance(w, QPushButton) or w.minimumWidth() != w.maximumWidth() or w.maximumWidth() <= 48:
            return
        text = w.text().replace("&&", "&")
        if not text.strip():
            return
        need = w.fontMetrics().horizontalAdvance(text) + 2 * 12 + 6
        if not w.icon().isNull():
            need += w.iconSize().width() + 4
        need = max(need, w.sizeHint().width())     # the style sheet's own padding and border
        if w.maximumWidth() < need:
            w.setFixedWidth(need)

    @classmethod
    def _fit_height_later(cls, w):
        """Again once the widget has been polished: the style sheet sets the
        larger minimum only then."""
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QPushButton
        if isinstance(w, (QLineEdit, QAbstractSpinBox, QComboBox, QPushButton)):
            from PyQt6 import sip
            QTimer.singleShot(0, lambda: None if sip.isdeleted(w) else cls._fit_height(w))

    @staticmethod
    def _fit_height(w):
        """A box given a fixed height for an older, smaller style (24-26 px)
        keeps that as its maximum, while the theme's padding and border make
        its minimum larger (28-30 px). The theme then draws the box at the
        larger height, but the row around it is laid out for the smaller
        one, so the box's bottom edge was cut off (Add constant, the BC rows,
        the training phases ...). The maximum is raised to the minimum."""
        from PyQt6.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit, QPushButton
        if not isinstance(w, (QLineEdit, QAbstractSpinBox, QComboBox, QPushButton)):
            return
        if w.maximumHeight() < w.minimumHeight() <= 60:
            w.setMaximumHeight(w.minimumHeight())

    @staticmethod
    def _fit_small_button(w):
        """Icon-sized buttons (✕, ▲, ◀ ...) get no padding, so the theme's
        button padding doesn't push their one character out of view."""
        from PyQt6.QtWidgets import QPushButton
        if not isinstance(w, QPushButton) or w.maximumWidth() > 48 or w.property("_pinn_small"):
            return
        w.setProperty("_pinn_small", True)
        ss = w.styleSheet() or ""
        extra = "QPushButton { padding: 0px; min-height: 0px; }" if "{" in ss else "padding: 0px; min-height: 0px;"
        w.setStyleSheet((ss + "\n" + extra).strip())

    @staticmethod
    def apply(w, force=False):
        try:
            ss = w.styleSheet()
        except RuntimeError:
            return
        from PyQt6.QtWidgets import QCheckBox, QRadioButton
        if not ss:
            return
        themed = w.property("_pinn_themed")
        if ss == themed and not force:
            return
        orig = ss if ss != themed else (w.property("_pinn_orig") or ss)
        new = remap_css(orig)
        if isinstance(w, (QCheckBox, QRadioButton)):
            # an option's label is ordinary text, whatever colour it was given
            new = re.sub(r"(?<![-\w])color\s*:\s*[^;}]+;?", "", new)
        w.setProperty("_pinn_orig", orig)
        w.setProperty("_pinn_themed", new)
        if new != ss:
            w.setStyleSheet(new)

    @staticmethod
    def rethemeall(root):
        for w in [root] + root.findChildren(QWidget):
            if w.property("_pinn_orig"):
                StyleRemapper.apply(w, force=True)


class LogHighlighter(QSyntaxHighlighter):
    """Training log: lines with ❌ / ⚠ / ✅ (and error/warning words) tinted."""

    def highlightBlock(self, text):
        t = _current
        low = text.lower()
        color = None
        if "❌" in text or "traceback" in low or low.startswith("error") or " error:" in low:
            color = t["danger"]
        elif "⚠" in text or "warning" in low:
            color = t["warning"]
        elif "✅" in text:
            color = t["success"]
        if color:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            self.setFormat(0, len(text), f)


class PythonHighlighter(QSyntaxHighlighter):
    """The Script tab: Python keywords, strings, numbers and comments in the
    theme's colours."""
    _KW = r"\b(?:and|as|assert|break|class|continue|def|del|elif|else|except|False|finally|for|from|global|if|" \
          r"import|in|is|lambda|None|nonlocal|not|or|pass|raise|return|True|try|while|with|yield)\b"

    def __init__(self, doc):
        super().__init__(doc)
        import re
        self._rules = [(re.compile(self._KW), "kw"), (re.compile(r"\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b"), "num"),
                       (re.compile(r"\b(?:def|class)\s+(\w+)"), "name")]
        self._str = re.compile(r"('''|\"\"\"|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")

    def _fmt(self, role):
        t = _current
        f = QTextCharFormat()
        f.setForeground(QColor({"kw": t["accent_text"], "num": t["warning"], "str": t["success"],
                                "com": t["text_faint"], "name": t["info"]}[role]))
        if role == "kw":
            f.setFontWeight(600)
        if role == "com":
            f.setFontItalic(True)
        return f

    def highlightBlock(self, text):
        for rx, role in self._rules:
            for m in rx.finditer(text):
                g = 1 if role == "name" else 0
                self.setFormat(m.start(g), m.end(g) - m.start(g), self._fmt(role))
        # triple-quoted strings span lines: state 1 = inside one
        start = 0
        in_doc = self.previousBlockState() == 1
        strf, comf = self._fmt("str"), self._fmt("com")
        if in_doc:
            end = text.find('"""')
            if end < 0:
                end = text.find("'''")
            if end < 0:
                self.setFormat(0, len(text), strf)
                self.setCurrentBlockState(1)
                return
            self.setFormat(0, end + 3, strf)
            start = end + 3
        self.setCurrentBlockState(0)
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "#":
                self.setFormat(i, len(text) - i, comf)
                return
            if text.startswith('"""', i) or text.startswith("'''", i):
                q = text[i:i + 3]
                end = text.find(q, i + 3)
                if end < 0:
                    self.setFormat(i, len(text) - i, strf)
                    self.setCurrentBlockState(1)
                    return
                self.setFormat(i, end + 3 - i, strf)
                i = end + 3
                continue
            if ch in "'\"":
                m = self._str.match(text, i)
                if m:
                    self.setFormat(i, m.end() - i, strf)
                    i = m.end()
                    continue
            i += 1
