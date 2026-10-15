#!/usr/bin/env python3
"""Themes: every stylesheet is built from a theme's colour tokens, and the
colours older widgets set for the original dark look are translated by
role (text / background / border). Checks the translation, that a theme
switch re-translates widgets' own stylesheets, the default look (Studio
Light, 14 px system font), the UI scale, and that pre-restyle display
settings on disk are not carried over.

Run directly:
    QT_QPA_PLATFORM=offscreen python3 tests/test_theme.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

app = QApplication.instance() or QApplication(sys.argv)

from pinnstudio.ui import theme
from pinnstudio.ui.main_window import MainWindow


def test_remap_by_role():
    t = theme.tokens("Studio Light")
    css = ("QPushButton { background: #1a3a5a; color: #74c0fc; border: 1px solid #2a5a8a; }"
           "QPushButton:hover { background: #2a5a8a; }")
    out = theme.remap_css(css, t)
    assert t["btn"] in out and t["accent_text"] in out and t["btn_border"] in out, out
    assert f"background: {t['btn_hover']}" in out, out
    assert theme.remap_css("color: #ff8787;", t) == f"color: {t['danger']};"
    assert theme.remap_css("color: #123456;", t) == "color: #123456;", "unknown colours stay"
    assert theme.remap_css("", t) == ""


def test_window_theme_and_switch():
    home = tempfile.mkdtemp()
    old = os.environ.get("HOME")
    os.environ["HOME"] = home
    try:
        os.makedirs(os.path.join(home, ".pinnstudio"))
        with open(os.path.join(home, ".pinnstudio", "display_settings.json"), "w") as fh:
            json.dump({"font_size": 18, "theme": "Monokai", "accent": "Green (#69db7c)"}, fh)
        w = MainWindow()
        assert w._theme == theme.DEFAULT_THEME == "Studio Light", "old display settings are not carried over"
        assert w._font_size == 14 and w._ui_scale == 100
        app.processEvents()
        lbl = QLabel("hint", w)
        lbl.setStyleSheet("color: #74c0fc;")
        app.processEvents()
        light = theme.tokens("Studio Light")["accent_text"]
        assert light in lbl.styleSheet(), lbl.styleSheet()
        w._theme = "Studio Dark"
        w._apply_display_settings()
        dark = theme.tokens("Studio Dark")["accent_text"]
        assert dark in lbl.styleSheet() and light not in lbl.styleSheet(), lbl.styleSheet()
        assert theme.tokens("Studio Dark")["bg"] in w.styleSheet()
        w._ui_scale = 125
        w._apply_display_settings()
        assert "font-size: 18px" in w.styleSheet(), "14 px at 125 %"
        assert w.solve_btn.objectName() == "solveButton" and w.solve_btn.styleSheet() == ""
        small = QPushButton("✕", w)
        small.setFixedWidth(24)
        small.setStyleSheet("QPushButton { color: #ff8787; background: transparent; border: none; }")
        small.show()
        app.processEvents()
        assert "padding: 0px" in small.styleSheet(), small.styleSheet()
    finally:
        if old is not None:
            os.environ["HOME"] = old


if __name__ == "__main__":
    test_remap_by_role()
    test_window_theme_and_switch()
    print("PASSED")
