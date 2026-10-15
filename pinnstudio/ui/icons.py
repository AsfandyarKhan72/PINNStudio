"""PINNStudio's icons: one line-icon set (24-unit grid, 2-unit round
strokes, in the style of Lucide/Feather), drawn in the theme's colours.

Emoji render differently on every platform and look informal in a
toolbar; these are crisp at any size and follow the theme. `icon(name)`
gives a QIcon; `apply(widget, name)` sets it on a button/tab and keeps it
coloured right when the theme changes (`retint_all()`).
"""
from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from pinnstudio.ui import theme as _theme

# inner SVG of each icon (viewBox 0 0 24 24, stroke = currentColor)
_ICONS = {
    "open": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "save": '<path d="M5 3h11l3 3v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
            '<path d="M7 3v5h8V3"/><rect x="7" y="13" width="10" height="6" rx="1"/>',
    "export": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'
              '<path d="m10 13-2 2 2 2"/><path d="m14 13 2 2-2 2"/>',
    "play": '<path d="M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5z" fill="currentColor"/>',
    "stop": '<rect x="5.5" y="5.5" width="13" height="13" rx="2" fill="currentColor"/>',
    "results": '<path d="M3 3v18h18"/><path d="m7 15 4-5 3 3 5-7"/>',
    "sketch": '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    "geometry": '<path d="M12 2 2 7l10 5 10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "log": '<path d="m4 17 6-6-6-6"/><path d="M12 19h8"/>',
    "select": '<path d="m4 4 7 17 2.5-7.5L21 11z"/>',
    "rectangle": '<rect x="3" y="6" width="18" height="12" rx="1"/>',
    "circle": '<circle cx="12" cy="12" r="8.5"/>',
    "ellipse": '<ellipse cx="12" cy="12" rx="9.5" ry="6"/>',
    "polygon": '<path d="m12 3 9 6.5-3.5 10.5h-11L3 9.5z"/>',
    "edges": '<path d="M4 20 4 8 12 4 20 8 20 20"/><circle cx="4" cy="20" r="1.6" fill="currentColor"/>'
             '<circle cx="20" cy="20" r="1.6" fill="currentColor"/><path d="M4 20h16" stroke-dasharray="2 3"/>',
    "box": '<path d="M21 8 12 3 3 8v8l9 5 9-5z"/><path d="m3 8 9 5 9-5"/><path d="M12 13v8"/>',
    "sphere": '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="9" ry="3.5"/>',
    "import": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/>'
              '<path d="M12 18v-6"/><path d="m9 15 3 3 3-3"/>',
    "fit": '<path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M21 8V5a2 2 0 0 0-2-2h-3"/>'
           '<path d="M3 16v3a2 2 0 0 0 2 2h3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "delete": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "clear": '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/>',
    "add": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "analysis": '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/>'
                '<rect x="17" y="5" width="3" height="13"/>',
    "loss": '<path d="M3 3v18h18"/><path d="m6 8 4 4 3-2 7 7"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/>'
             '<path d="m21 15-5-5L5 21"/>',
    "download": '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    "warning": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
               '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "code": '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
    "home": '<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    "up": '<path d="m6 15 6-6 6 6"/>',
    "undo": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "redo": '<path d="m15 14 5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "down": '<path d="m6 9 6 6 6-6"/>',
}


def svg(name, color, fill_color=None):
    inner = _ICONS[name].replace("currentColor", fill_color or color)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


def pixmap(name, color, size=18, scale=2.0):
    r = QSvgRenderer(QByteArray(svg(name, color).encode()))
    pm = QPixmap(int(size * scale), int(size * scale))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    r.render(p, QRectF(0, 0, size * scale, size * scale))
    p.end()
    pm.setDevicePixelRatio(scale)
    return pm


def icon(name, color=None, disabled_color=None, size=18):
    t = _theme.current()
    ic = QIcon()
    ic.addPixmap(pixmap(name, color or t["text"], size), QIcon.Mode.Normal)
    ic.addPixmap(pixmap(name, disabled_color or t["disabled_text"], size), QIcon.Mode.Disabled)
    return ic


_registry = []   # (widget or (tabwidget, index), name, role)

_ROLE_COLOR = {"text": "text", "muted": "text_muted", "accent": "accent_text", "on_accent": "on_accent",
               "danger": "danger", "on_danger": None}


def _color(role):
    t = _theme.current()
    if role == "on_danger":
        return "#ffffff"
    return t.get(_ROLE_COLOR.get(role, "text"), t["text"])


def apply(target, name, role="text", size=18):
    """Set icon `name` on a button (or a (QTabWidget, index) tab), tinted by
    `role`, and keep it tinted when the theme changes."""
    _registry.append((target, name, role, size))
    if len(_registry) % 200 == 0:
        _prune()
    _set(target, name, role, size)


def _prune():
    """Forget buttons that have been deleted (removed BC rows, closed dialogs)."""
    try:
        from PyQt6 import sip
    except ImportError:
        return
    def alive(t):
        w = t[0] if isinstance(t, tuple) else t
        try:
            return not sip.isdeleted(w)
        except TypeError:
            return True
    _registry[:] = [e for e in _registry if alive(e[0])]


def _set(target, name, role, size):
    ic = icon(name, _color(role), size=size)
    if isinstance(target, tuple):
        tabs, idx = target
        tabs.setTabIcon(idx, ic)
        tabs.setIconSize(QSize(size, size))
    else:
        target.setIcon(ic)
        target.setIconSize(QSize(size, size))
        txt = target.text() if hasattr(target, "text") else ""
        if txt and not txt.startswith(" "):
            target.setText(" " + txt)   # a little room between the icon and the label


def retint_all():
    alive = []
    for target, name, role, size in _registry:
        try:
            _set(target, name, role, size)
            alive.append((target, name, role, size))
        except RuntimeError:
            pass
    _registry[:] = alive


# Buttons written with a symbol in front of their text ("➕ Add Phase",
# "💾 Save Figure") get the matching icon instead, wherever they are built.
_PREFIXES = (("➕", "add", "accent"), ("+ ", "add", "accent"), ("📊", "analysis", "text"),
             ("📉", "loss", "text"), ("💾", "download", "text"), ("▶ Run", "play", "accent"))


def iconize(btn):
    from PyQt6.QtWidgets import QPushButton
    if not isinstance(btn, QPushButton) or btn.property("_pinn_icon"):
        return
    txt = btn.text()
    for pre, name, role in _PREFIXES:
        if txt.startswith(pre):
            btn.setProperty("_pinn_icon", name)
            keep = txt[len(pre):] if pre != "▶ Run" else txt[1:]
            btn.setText(keep.strip())
            apply(btn, name, role, size=16)
            return


_APP = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        '<rect x="2" y="2" width="60" height="60" rx="14" fill="#2f6fde"/>'
        '<path d="M10 44 C 20 44, 22 20, 32 20 S 44 40, 54 40" fill="none" stroke="#ffffff" stroke-width="4.5" '
        'stroke-linecap="round"/>'
        '<circle cx="21" cy="32" r="3.6" fill="#ffd166"/><circle cx="32" cy="20" r="3.6" fill="#ffd166"/>'
        '<circle cx="43" cy="30" r="3.6" fill="#ffd166"/></svg>')


def app_icon():
    """PINNStudio's window icon: a smooth solution curve through a few
    training points."""
    ic = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        r = QSvgRenderer(QByteArray(_APP.encode()))
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        r.render(p, QRectF(0, 0, size, size))
        p.end()
        ic.addPixmap(pm)
    return ic
