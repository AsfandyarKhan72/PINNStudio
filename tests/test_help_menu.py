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
        assert items == ["Help for This Page", "Quick Reference…", "Documentation", "Tutorials", "Getting Started",
                         "User Guide", "Choosing a Strategy", "Troubleshooting", "FAQ", "Report a Problem…",
                         "About PINNStudio", "About Qt"], items
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


def test_tutorial_list_matches_the_website():
    """Help › Tutorials and the start page list the tutorials of website/tutorials/index.md, each with its page."""
    import re
    from pinnstudio.ui import help as _help
    index = open(os.path.join(ROOT, "website", "tutorials", "index.md"), encoding="utf-8").read()
    rows = re.findall(r"^\| (\d+) \| \[([^\]]+)\]\(([\w-]+)\.md\)", index, flags=re.M)
    assert [(int(n), t, p) for n, t, p in rows] == list(_help.TUTORIALS)
    for _n, _t, page in _help.TUTORIALS:
        assert os.path.isfile(os.path.join(ROOT, "website", "tutorials", page + ".md")), page
    nav = open(os.path.join(ROOT, "mkdocs.yml"), encoding="utf-8").read()
    for _n, _t, page in _help.TUTORIALS:
        assert f"tutorials/{page}.md" in nav, page
    for _text, page in _help.DOC_PAGES:
        assert os.path.isfile(os.path.join(ROOT, "website", page.rstrip("/") + ".md")) or \
            os.path.isfile(os.path.join(ROOT, "website", page, "index.md")), page


def test_page_help_and_quick_reference():
    w = MainWindow()
    opened = []
    w._open_url = opened.append
    try:
        w._open_page_help()
        w.workspace.select("bcs")
        w._open_page_help()
        assert opened == [w.DOCS_URL + "guide/problem/", w.DOCS_URL + "guide/boundary-conditions/"], opened
        tut = next(a.menu() for a in menus_of(w)["Help"].actions() if a.text() == "Tutorials")
        acts = [a for a in tut.actions() if a.text() and not a.isSeparator()]
        acts[1].trigger()
        assert opened[-1] == w.DOCS_URL + "tutorials/01-ode/", opened
        w.start_page.learn_cards["tutorials"].click()
        assert opened[-1] == w.DOCS_URL + "tutorials/", opened

        w._show_quick_reference()
        d = w._quick_ref
        assert d.isVisible() and not d.isModal()
        names = [d.tabs.tabText(i) for i in range(d.tabs.count())]
        assert names == ["Expressions", "Conditions", "Workflow", "Shortcuts"], names
        shortcuts = d.browsers[-1].toPlainText()
        for want in ("Solve", "F5", "Help for This Page", "F1"):
            assert want in shortcuts, want
        d.filter.setText("Periodic")
        assert d.tabs.currentIndex() == 1 and d.status.text() == ""
        d.filter.setText("no such words anywhere")
        assert d.status.text() == "Not found"
        w._show_quick_reference()
        assert w._quick_ref is d, "one window, raised again"
        d.close()
    finally:
        w.close()
        w.deleteLater()


def menus_of(w):
    return {a.text(): a.menu() for a in w.menuBar().actions() if a.menu() is not None}


if __name__ == "__main__":
    test_help_menu_and_about_text()
    test_tutorial_list_matches_the_website()
    test_page_help_and_quick_reference()
    print("HELP MENU TEST PASSED")
