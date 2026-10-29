"""The README's screenshots of the main window, from a real training run in the window.

Burgers 1D – Raissi et al. is loaded from the Examples list (Settings: Default) and solved with the Solve button:
  assets/screenshots/live_loss.png       the Training page and Results → Live loss, near the end of the L-BFGS phase
  assets/screenshots/interface.png       the Equations page and Results → Overview, after the run
  assets/screenshots/error_analysis.png  Results → Error analysis after the run (the results pane only)

    QT_QPA_PLATFORM=offscreen python tools/make_readme_screenshots.py           (full training: use a GPU)
    QT_QPA_PLATFORM=offscreen python tools/make_readme_screenshots.py --quick   (short training, to check this
                                                                                  script; writes to a temp folder)
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pinnstudio.main  # noqa: F401,E402  (loads PyTorch before PyQt6 on Windows)
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

QUICK = "--quick" in sys.argv
SIZE = (1680, 1000)          # as the earlier screenshots
OUT = tempfile.mkdtemp(prefix="pinnstudio_shots_") if QUICK else os.path.join(ROOT, "assets", "screenshots")
for _name in ("question", "warning", "information"):     # nothing waits for a click
    setattr(QMessageBox, _name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value

app = QApplication(["pinnstudio"])


def pump(sec):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.01)


def main():
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(*SIZE)
    w.show()
    pump(1.0)
    save = tempfile.mkdtemp(prefix="PINNStudio_Results_")
    w.save_dir_input.setText(save)
    w.quick_examples_category_combo.setCurrentText("Forward")
    pump(0.3)
    combo = w.quick_examples_combo
    i = combo.findData("1D Burgers")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    pump(2.0)
    if QUICK:                                  # a short run, through the window as a user would set it
        import json
        cfg = w._build_config()
        ph = json.loads(cfg.scheduler_phases or "[]")
        for p in ph:
            p["iterations"] = 600 if p.get("optimizer") == "adam" else 300
        cfg.scheduler_phases = json.dumps(ph)
        w._apply_config(cfg)
        pump(1.0)
    w.log_box.clear()
    w.workspace.select("training")
    pump(0.5)
    w.solve_btn.click()
    pump(1.0)
    w.results_tabs.setCurrentWidget(w.live_loss)
    shot_live, last = None, 0.0
    t0 = time.time()
    while w.thread is not None and w.thread.isRunning():
        pump(0.5)
        p = getattr(w, "_progress", None)
        if p and "bfgs" in (p.phase_name or "").lower() and time.time() - last >= 5:   # L-BFGS can stop early:
            # keep the latest frame
            shot_live, last = (w.grab(), p.step), time.time()
        if time.time() - t0 > 4 * 3600:
            raise SystemExit("training did not finish")
    pump(3.0)
    if shot_live is None:
        print("note: no frame of the L-BFGS phase; no live_loss.png", flush=True)
    else:
        shot_live[0].save(os.path.join(OUT, "live_loss.png"))
        print("live_loss.png at step", shot_live[1], flush=True)
    w.workspace.select("equations")
    w.results_tabs.setCurrentIndex(0)                   # Overview
    pump(2.0)
    w.grab().save(os.path.join(OUT, "interface.png"))
    w.results_tabs.setCurrentWidget(w.ea_panel)
    pump(2.0)
    from PyQt6.QtCore import QRect                     # the results pane, taken from the whole window (its
    r = w.results_tabs                                 # own grab leaves the tab bar's background black)
    p0 = r.mapTo(w, r.rect().topLeft())
    w.grab(QRect(p0.x(), p0.y(), r.width(), r.height())).save(os.path.join(OUT, "error_analysis.png"))
    print("wrote", OUT, sorted(os.listdir(OUT)), flush=True)
    w.close()
    os._exit(0)


if __name__ == "__main__":
    main()
