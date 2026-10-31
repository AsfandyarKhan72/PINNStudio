#!/usr/bin/env python3
"""Screenshots of the PINNStudio interface for the documentation site.

    QT_QPA_PLATFORM=offscreen python tools/docs/make_screenshots.py website/assets/screenshots [--solve]

Opens the real MainWindow (offscreen), loads shipped examples the way a user picks them from the
Examples list, selects each page of the model tree and saves the window. With --solve it also runs one
short Solve of the 1D Burgers example (CPU, a few minutes) and captures the Results tabs. Dialogs
are captured while open (a timer grabs and closes them). Everything shown is the application's own
state; nothing is drawn or edited afterwards.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DDE_BACKEND", "pytorch")
# a neutral home folder, so the status bar shows ~/PINNStudio_Results-style paths, not the build machine's
os.environ["HOME"] = os.environ.get("PINNSTUDIO_DOCS_HOME", "/tmp/home/user")
os.makedirs(os.environ["HOME"], exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from PyQt6.QtCore import QCoreApplication, QEvent, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

W, H = 1600, 1000


def settle(s=0.3):
    t = time.time()
    while time.time() - t < s:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        app.processEvents()


def pick(w, category, key):
    w.quick_examples_category_combo.setCurrentText(category)
    settle(0.2)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    if i < 0:
        raise SystemExit(f"example not found: {category} / {key}")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    settle(1.0)
    w.log_box.clear()          # only this example's own messages in the log
    settle(0.1)


def page(w, key):
    w.workspace.select(key)
    settle(0.4)


def save(w, out, name):
    p = os.path.join(out, name)
    w.grab().save(p)
    print("saved", p, flush=True)


def grab_dialog(out, name, delay_ms=800):
    """Grab the next modal dialog that opens, then close it."""
    def _do():
        d = QApplication.activeModalWidget()
        if d is not None:
            d.grab().save(os.path.join(out, name))
            print("saved", name, flush=True)
            d.reject()
    QTimer.singleShot(delay_ms, _do)


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "website/assets/screenshots")
    os.makedirs(out, exist_ok=True)
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(W, H)
    w.show()
    settle(1.0)
    save(w, out, "start.png")

    # A forward problem: every page of the model tree
    pick(w, "Forward", "1D Burgers")
    for key in ("problem", "equations", "geometry", "points", "ic", "bcs", "network", "training",
                "weights", "adaptive", "sweep", "results", "restore"):
        page(w, key)
        save(w, out, f"burgers_{key}.png")
    # the first boundary-condition row
    bc_keys = [k for k in w.workspace.pages if str(k).startswith("bc:")]
    if bc_keys:
        page(w, bc_keys[0])
        save(w, out, "burgers_bc_row.png")

    # 2D geometry and domain view
    pick(w, "Forward", "2D Heat")
    page(w, "geometry")
    w.right_tabs.setCurrentIndex(w._domain_tab_index)
    settle(1.0)
    save(w, out, "heat2d_geometry.png")
    page(w, "adaptive")
    save(w, out, "heat2d_adaptive.png")

    # An inverse problem
    pick(w, "Inverse", "1D Heat")
    page(w, "inverse")
    save(w, out, "inverse_heat1d.png")
    page(w, "equations")
    save(w, out, "inverse_heat1d_equations.png")

    # Dialogs from the menus
    for act_text, name in (("Float Precision...", "dialog_precision.png"), ("Random Seed...", "dialog_seed.png")):
        acts = [a for m in w.menuBar().actions() for a in (m.menu().actions() if m.menu() else [])
                if a.text().replace("&", "") == act_text]
        if acts:
            grab_dialog(out, name)
            acts[0].trigger()
            settle(1.2)

    if "--solve" in sys.argv:
        import json
        pick(w, "Forward", "1D Burgers")
        # a short run, so the screenshots can be made on a laptop CPU in minutes
        for ph in w.sched_phase_list:
            ph["iters"].setValue(2000 if ph["opt"].currentData() != "lbfgs" else 500)
        settle(0.3)
        w._on_solve()
        t0 = time.time()
        while time.time() - t0 < 3600:
            settle(1.0)
            th = getattr(w, "solver_thread", None) or getattr(w, "thread", None)
            if th is not None and not th.isRunning() and time.time() - t0 > 10:
                break
        settle(3.0)
        for i in range(w.results_tabs.count()):
            w.results_tabs.setCurrentIndex(i)
            settle(0.8)
            name = "results_" + w.results_tabs.tabText(i).strip().lower().replace(" ", "_").replace("&", "and") + ".png"
            save(w, out, name)
        w.bottom_tabs.setCurrentIndex(0)
        settle(0.3)
        save(w, out, "solve_log.png")
    w.close()
    os._exit(0)


if __name__ == "__main__":
    main()
