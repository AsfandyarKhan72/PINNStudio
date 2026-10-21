"""End-to-end check of the PINNStudio window on this computer.

Loads a template, trains it briefly through the same code path as Solve,
then uses Restore & Visualize on the trained model for every plot type,
the way a user would. Any error shown by the window (errors.py) or a failed
step makes it exit with code 1. Used by the platform workflow; run by hand:

    python tools/smoke_gui.py            (offscreen; takes a minute or two)
"""
import glob
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _s in (sys.stdout, sys.stderr):   # widget texts (L₂, ...) on a cp1252 console
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(HERE, "pinnstudio")) and "--installed" not in sys.argv:
    sys.path.insert(0, HERE)

import pinnstudio.main  # noqa: F401,E402  (loads PyTorch before PyQt6 on Windows)
from pinnstudio import errors  # noqa: E402

errors.install()
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

SHOWN = []
QMessageBox.exec = lambda self: SHOWN.append(self.text() + "\n" + (self.detailedText() or "")) or 0
# settings dialogs (Restore's plot settings, ...) are confirmed with OK as they are
from PyQt6.QtWidgets import QDialog  # noqa: E402
QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value
# questions (save unsaved changes?, ...) are answered "No" so nothing waits for a click
for _name in ("question", "warning", "information"):
    setattr(QMessageBox, _name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
_orig_report = errors._report


def _report(t, e, tb, where=""):
    SHOWN.append(f"{t.__name__}: {e}")
    _orig_report(t, e, tb, where)


errors._report = _report

app = QApplication(["pinnstudio"])
from pinnstudio.core.runner import run_pinn  # noqa: E402
from pinnstudio.ui.main_window import MainWindow  # noqa: E402

FAILED = []


def step(msg):
    print(f"--- {msg}", flush=True)


GH = os.environ.get("GITHUB_ACTIONS") == "true"
OS = {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux")


def _gh(level, title, text):
    """An annotation on the GitHub Actions run (its log is not readable from here)."""
    if GH:
        esc = lambda t: t.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::{level} title={esc(OS + ' smoke: ' + title)}::{esc(text[-3500:])}", flush=True)


def check(ok, msg, detail=""):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        FAILED.append(msg)
        _gh("error", msg, detail or msg)


def settle(sec=0.3):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()


def _blocking_widget(pane):
    """What stops a divider: from the pane down, the visible child with the
    largest minimum width at each level (the pane's minimum comes from it),
    and at the end the widgets in that child's layout."""
    from PyQt6.QtWidgets import QWidget

    def mw(w):
        return max(w.minimumSizeHint().width(), w.minimumWidth())

    def name(c):
        text = c.text() if hasattr(c, "text") and callable(c.text) else ""
        text = text if isinstance(text, str) else ""
        return f"{mw(c)}px {type(c).__name__}({c.objectName()})" + (f" {text[:30]!r}" if text else "")

    path, w = [], pane
    while True:
        kids = [c for c in w.children() if isinstance(c, QWidget) and c.isVisible() and not c.isWindow()]
        if not kids:
            break
        top = max(kids, key=mw)
        if mw(top) < mw(w) - 40:     # no single child sets it: the layout's row does
            row = sorted(kids, key=mw, reverse=True)[:5]
            path.append("[row: " + ", ".join(name(c) for c in row) + "]")
            break
        path.append(name(top))
        w = top
    return f"pane {mw(pane)}px: " + " > ".join(path[-6:])


def _watchdog(seconds):
    """If the check hangs, report every thread's stack (where it waits) and stop."""
    import faulthandler
    import threading

    def fire():
        path = os.path.join(tempfile.gettempdir(), "pinnstudio_smoke_hang.txt")
        with open(path, "w", encoding="utf-8") as f:
            faulthandler.dump_traceback(file=f, all_threads=True)
        text = open(path, encoding="utf-8").read()
        print("HANG after %d s, all threads:\n%s" % (seconds, text), flush=True)
        _gh("error", "hang", text)
        print("SMOKE FAILED: hang", flush=True)
        os._exit(1)

    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()


def splitter_report(w):
    """Drag every horizontal divider to both ends; print the reachable range."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QSplitter
    for sp in w.findChildren(QSplitter):
        if sp.orientation() != Qt.Orientation.Horizontal or not sp.isVisible() or sp.count() < 2:
            continue
        total = sum(sp.sizes())
        for i in range(1, sp.count()):
            orig = sp.sizes()
            sp.moveSplitter(0, i)
            settle(0.2)
            lo = sp.sizes()
            sp.moveSplitter(total, i)
            settle(0.2)
            hi = sp.sizes()
            sp.setSizes(orig)
            settle(0.2)
            left, right = sp.widget(i - 1), sp.widget(i)
            line = (f"handle {i} (total {total}px): left pane {lo[i - 1]}..{hi[i - 1]}px, "
                    f"right pane {hi[i]}..{lo[i]}px\n  left min: {_blocking_widget(left)}\n"
                    f"  right min: {_blocking_widget(right)}")
            print("SPLITTER " + line, flush=True)
            _gh("notice", f"divider {i}", line)


def main():
    tmp = tempfile.mkdtemp(prefix="pinnstudio smoke ")   # a space in the path, as in many home folders
    w = MainWindow()
    w.show()
    settle()

    step("Heat 1D template, short training through the Solve code path")
    w.quick_examples_combo.setCurrentText("1D Heat")
    settle()
    w.save_dir_input.setText(tmp)
    cfg = w._build_config()
    cfg.save_dir = tmp
    phases = json.loads(cfg.scheduler_phases or "[]")
    for p in phases:
        p["iterations"] = 300 if p.get("optimizer") == "adam" else 50
    cfg.scheduler_phases = json.dumps(phases)
    cfg.iterations, cfg.iterations2 = 300, 50
    log = []
    t0 = time.time()
    result = run_pinn(cfg, on_output=log.append)
    check(result == "DONE", f"training finished ({time.time() - t0:.0f} s): {result}", "\n".join(log[-60:]))
    if result != "DONE":
        print("\n".join(log[-40:]))
        return
    models = sorted(glob.glob(os.path.join(tmp, "**", "*.pt"), recursive=True))
    check(bool(models), f"model saved ({len(models)} .pt files)")
    pngs = glob.glob(os.path.join(tmp, "**", "*.png"), recursive=True)
    check(bool(pngs), f"figures saved ({len(pngs)} .png files)")
    if not models:
        return

    step("Results with Error analysis open: how far each divider can be dragged")
    from pinnstudio.ui.results import read_run
    run_dir = os.path.dirname(os.path.dirname(models[-1])) if "solution_results" in models[-1] else os.path.dirname(models[-1])
    run = read_run(run_dir)
    check(run is not None, f"run readable from History ({run_dir})")
    if run:
        w.resize(1600, 950)
        settle()
        w._show_saved_run(run)
        w.right_tabs.setCurrentWidget(w.results_tabs)
        w.results_tabs.setCurrentWidget(w.ea_panel)
        settle(1.0)
        splitter_report(w)

    step("Restore & Visualize, every plot type")
    _watchdog(600)
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (models[-1], ""))
    w._on_browse_restore_model()
    settle()
    check(os.path.isfile(w.restore_config_path.text()), "restore config found")
    w.restore_save_path.setText(os.path.join(tmp, "restored"))
    types = [w.restore_viz_combo.itemText(i) for i in range(w.restore_viz_combo.count())]
    for viz in types:
        if "param" in viz.lower():         # inverse-only
            continue
        w.restore_viz_combo.setCurrentText(viz)
        settle(0.2)
        before = len(SHOWN)
        n_log = len(w.log_box.toPlainText())
        w._on_restore()
        settle(0.5)
        t0 = time.time()
        th = getattr(w, "_restore_thread", None)
        while th is not None and th.isRunning() and time.time() - t0 < 300:
            settle(0.5)
        settle(1.0)
        new_log = w.log_box.toPlainText()[n_log:]
        ok = "Restore complete" in new_log and len(SHOWN) == before
        check(ok, f"restore: {viz}", new_log[-3000:] + "\n" + "\n".join(SHOWN[before:]))
        if not ok:
            print(new_log[-3000:])
    w.close()
    settle()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:      # a failure of the check itself
        import traceback
        traceback.print_exc()
        FAILED.append(f"smoke script: {e}")
    if SHOWN:
        print("Errors shown by the window:")
        for s in SHOWN:
            print("  " + s[:3000])
    if SHOWN:
        _gh("error", "errors shown by the window", "\n\n".join(SHOWN))
    print("SMOKE " + ("FAILED: " + "; ".join(FAILED) if FAILED or SHOWN else "PASSED"), flush=True)
    _gh("notice" if not (FAILED or SHOWN) else "error", "result", "PASSED" if not (FAILED or SHOWN) else "; ".join(FAILED))
    # os._exit, not sys.exit: the result is already printed, and Qt's own
    # teardown at interpreter exit sometimes crashed (exit code 139 after
    # "SMOKE PASSED" on Ubuntu 24.04 with Python 3.13), failing a passed check.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if FAILED or SHOWN else 0)
