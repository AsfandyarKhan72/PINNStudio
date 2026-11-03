"""The Help menu: the documentation, About PINNStudio (version, MIT License, the LGPL Qt for Python it is built on)
and About Qt. The About text names the installed version and the licenses stated in LICENSE and
THIRD_PARTY_LICENSES.md."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication([])

from pinnstudio.ui.main_window import MainWindow  # noqa: E402


def test_help_menu_and_about_text():
    w = MainWindow()
    try:
        menus = {a.text(): a.menu() for a in w.menuBar().actions() if a.menu() is not None}
        assert list(menus)[-1] == "Help", list(menus)
        items = [a.text() for a in menus["Help"].actions() if a.text()]
        assert items == ["Documentation", "About PINNStudio", "About Qt"], items
        text = w._about_text()
        import pinnstudio
        ver = w._get_installed_version()
        assert ver == pinnstudio.__version__ and f"PINNStudio {ver}" in text      # the running code's version
        setup_src = open(os.path.join(ROOT, "setup.py"), encoding="utf-8").read()
        assert "version=VERSION" in setup_src and "pinnstudio\" / \"__init__.py" in setup_src.replace("'", '"')
        for want in ("MIT License", "Asfandyar Khan", "PySide6", "GNU Lesser General Public License version 3",
                     "THIRD_PARTY_LICENSES.md", w.DOCS_URL):
            assert want in text, want
        assert "GPL-3.0-only" not in text and "PyQt" not in text

        # the dialog opens and closes (closed by a timer, as a user would)
        seen = []

        def _close():
            for top in QApplication.topLevelWidgets():
                if isinstance(top, QMessageBox) and top.isVisible():
                    seen.append(top.objectName())       # not the title: macOS shows none on message boxes
                    top.accept()
        QTimer.singleShot(300, _close)
        w._show_about()
        assert seen == ["aboutPINNStudio"], seen
    finally:
        w.close()
        w.deleteLater()


if __name__ == "__main__":
    test_help_menu_and_about_text()
    print("HELP MENU TEST PASSED")
