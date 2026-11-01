"""Stress the real window: load examples back and forth, visit every page, paint the views.

    python tools/gui_stress.py [--rounds N] [--pair default|all] [--pages] [--tabs]

A native crash (heap corruption, segmentation fault) ends the process with a signal; the exit status
then tells it apart from a normal run (0) or a failed check (1). Memory (RSS) and the number of live
widgets are printed after every round, so a leak shows as steady growth.

Found with this loop: PySide6 6.12.0 aborted ("malloc_consolidate(): unaligned fastbin chunk detected")
or crashed in about 7 of 100 rounds of loading "Heat 3D" and "Allen-Cahn 1D -- Wight & Zhao
(time-adaptive)" one after the other; PyQt6 never did (150 rounds). See PYSIDE6_TRIAL.md.
Runs offscreen unless QT_QPA_PLATFORM is set (xcb for a real X server or Xvfb).
"""
import argparse
import faulthandler
import gc
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

PAIRS = {
    "default": [("Forward", "Heat 3D"), ("Forward", "Allen-Cahn 1D – Wight & Zhao (time-adaptive)")],
    "all": [("Forward", "Lotka-Volterra ODE"), ("Forward", "Burgers 1D – Raissi et al."),
            ("Inverse", "Diffusion 1D – unknown coefficient"), ("Forward", "Poisson 2D – L-shape, Lu et al."),
            ("Forward", "Poisson 2D – disk"), ("Forward", "Poisson 3D – sphere"), ("Forward", "Heat 3D"),
            ("Forward", "Allen-Cahn 1D – Wight & Zhao (time-adaptive)"), ("Inverse", "Poisson 2D – disk")],
}


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--pair", choices=sorted(PAIRS), default="default")
    ap.add_argument("--pages", action="store_true", help="visit and paint every model-tree page")
    ap.add_argument("--tabs", action="store_true", help="show and paint the geometry views")
    a = ap.parse_args()
    faulthandler.enable(all_threads=True)
    app = QApplication.instance() or QApplication(["pinnstudio"])
    # no modal dialog may block an unattended run
    QDialog.exec = lambda self: 1
    for n in ("question", "warning", "information", "critical"):
        setattr(QMessageBox, n, staticmethod(lambda *x, **k: QMessageBox.StandardButton.Yes))
    from pinnstudio.ui.main_window import MainWindow

    def settle(s):
        t = time.time()
        while time.time() - t < s:
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
            app.processEvents()

    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(1600, 1000)
    w.show()
    settle(0.5)
    failures = 0
    for r in range(a.rounds):
        for cat, name in PAIRS[a.pair]:
            w.quick_examples_category_combo.setCurrentText(cat)
            settle(0.1)
            i = w.quick_examples_combo.findText(name)
            if i < 0:
                print(f"FAIL example not found: {name}", flush=True)
                return 1
            if w.quick_examples_combo.currentIndex() == i:
                w._on_quick_example_selected(name)
            else:
                w.quick_examples_combo.setCurrentIndex(i)
            settle(0.8)
            if a.tabs:
                for idx_name in ("_domain_tab_index", "_sketch_tab_index"):
                    idx = getattr(w, idx_name, None)
                    if idx is not None:
                        w.right_tabs.setCurrentIndex(idx)
                        settle(0.3)
                        w.right_tabs.currentWidget().grab()
            if a.pages:
                for key in list(w.workspace.order):
                    w.workspace.select(key)
                    settle(0.05)
                    w.workspace.stack.currentWidget().grab()
            errs = w._build_config().validate()
            if errs:
                failures += 1
                print(f"FAIL round {r} {name}: {errs}", flush=True)
        gc.collect()
        print(f"round {r + 1}/{a.rounds} rss={_rss_mb():.0f}MB widgets={len(app.allWidgets())}", flush=True)
    w.close()
    w.deleteLater()
    settle(0.5)
    print(f"STRESS DONE: {a.rounds} rounds, {failures} failed checks", flush=True)
    sys.stdout.flush()
    os._exit(1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())
