"""Figure 2 of the PINNStudio paper, from the real window: Burgers 1D - Raissi et al. (Examples, Settings: Default)
is solved with the Solve button, the window is closed, and a new window restores the saved run with
Results -> History (as a user reopening a finished run would). Two panels are cut from the window:
  (a) the model tree and the Equations page;  (b) Results -> Overview: the training loss and the solution.
No file paths are in either panel. The window is drawn at device pixel ratio 2 (QT_SCALE_FACTOR=2).

    QT_SCALE_FACTOR=2 QT_QPA_PLATFORM=offscreen python tools/make_paper_fig2.py RUN_DIR [--quick] [--train]

--train  solve first and save the run in RUN_DIR (otherwise RUN_DIR must hold a saved run)
--quick  short training, to check this script
Writes RUN_DIR/fig2_a_full.png, RUN_DIR/fig2_b_full.png, and the two stacked as
benchmarks/figures/paper_figure2.png ((a) above (b), labels in a left margin).
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pinnstudio.main  # noqa: F401,E402
from PyQt6.QtCore import QRect  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

RUN = os.path.abspath(sys.argv[1])
QUICK, TRAIN = "--quick" in sys.argv, "--train" in sys.argv
SIZE = (1440, 820)
LOSS_EVERY = int(os.environ.get("FIG2_LOSS_EVERY", "100"))
FULL_SIZE = tuple(int(v) for v in os.environ.get("FIG2_FULL_SIZE", "1280x800").split("x"))
B_HEIGHT = int(os.environ.get("FIG2_B_HEIGHT", "400"))   # panel (b): the results pane
A_HEIGHT = int(os.environ.get("FIG2_A_HEIGHT", "455"))   # panel (a): the model tree down to Results
for _name in ("question", "warning", "information"):
    setattr(QMessageBox, _name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
QDialog.exec = lambda self: QDialog.DialogCode.Accepted.value
app = QApplication(["pinnstudio"])


def pump(sec):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.01)


def window():
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(*SIZE)
    w.show()
    pump(1.0)
    w.save_dir_input.setText(RUN)
    w.quick_examples_category_combo.setCurrentText("Forward")
    pump(0.3)
    combo = w.quick_examples_combo
    i = combo.findData("1D Burgers")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    pump(2.0)
    return w


def grab(w, widgets, path, height=None):
    rs = [QRect(x.mapTo(w, x.rect().topLeft()), x.size()) for x in widgets]
    r = rs[0]
    for q in rs[1:]:
        r = r.united(q)
    if height:
        r.setHeight(min(r.height(), height))
    w.grab(r).save(path)
    print("wrote", path, r, flush=True)


def train():
    w = window()
    # Plot settings (Results -> Plot settings), as a user would set them for a figure in a paper: smaller saved
    # figures, so that their text stays readable when Overview shows them in the narrow results pane
    cfg = w._build_config()
    cfg.plot_figsize_mode, cfg.plot_figsize_w, cfg.plot_figsize_h = "Custom", 3.4, 2.8
    cfg.plot_dpi = 300
    cfg.loss_display_every = LOSS_EVERY      # Plot settings -> Loss: record the loss every LOSS_EVERY iterations
    w._apply_config(cfg)
    pump(0.5)
    if QUICK:
        import json
        cfg = w._build_config()
        ph = json.loads(cfg.scheduler_phases or "[]")
        for p in ph:
            p["iterations"] = 600 if p.get("optimizer") == "adam" else 300
        cfg.scheduler_phases = json.dumps(ph)
        w._apply_config(cfg)
        pump(1.0)
    w.workspace.select("training")
    pump(0.5)
    w.solve_btn.click()
    pump(1.0)
    t0 = time.time()
    while w.thread is not None and w.thread.isRunning():
        pump(0.5)
        if time.time() - t0 > 6 * 3600:
            raise SystemExit("training did not finish")
    pump(3.0)
    w.close()


def shots():
    w = window()
    w.results_tabs.setCurrentWidget(w.run_history)
    pump(1.0)
    h = w.run_history
    h.refresh()
    pump(0.5)
    print("history entries:", len(h.runs), flush=True)
    if not h.runs:
        raise SystemExit("no saved run in " + RUN)
    h._activate(0)                     # Results -> History -> Show: the saved run back in Overview
    pump(3.0)
    w.right_tabs.setCurrentIndex(w._results_tab_index)
    w.results_tabs.setCurrentIndex(0)
    pump(2.0)
    return w


def compose(a, b, out):
    """(a) above (b), at the same width; (b) without the Start/Results tab row and the empty bottom of its panes."""
    from PIL import Image, ImageDraw, ImageFont
    A, B = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    B = B.crop((0, 60, B.width, B.height - 36))
    W = max(A.width, B.width)
    A = A.resize((W, round(A.height * W / A.width)), Image.LANCZOS)
    B = B.resize((W, round(B.height * W / B.width)), Image.LANCZOS)
    G, gap = 86, 36
    im = Image.new("RGB", (G + W, A.height + B.height + gap), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
    im.paste(A, (G, 0)); im.paste(B, (G, A.height + gap))
    d.text((2, 4), "(a)", fill="black", font=f); d.text((2, A.height + gap + 4), "(b)", fill="black", font=f)
    for y0, h in ((0, A.height), (A.height + gap, B.height)):
        d.rectangle([G, y0, G + W - 1, y0 + h - 1], outline=(190, 196, 204), width=2)
    im.save(out, dpi=(300, 300))
    print("wrote", out, im.size, flush=True)


def full_window(w):
    """The whole window, as the earlier Figure 2: Equations page, Results -> Overview and the Training Log."""
    from PyQt6.QtWidgets import QSplitter
    W, H = FULL_SIZE
    w.resize(W, H)
    pump(1.0)
    w.workspace.select("equations")
    sp = w.workspace.tree_panel.parent()
    if isinstance(sp, QSplitter):
        sp.setSizes([215, 440, W - 655])
    vs = w.right_tabs.parent()
    while vs is not None and not isinstance(vs, QSplitter):
        vs = vs.parent()
    if vs is not None:
        vs.setSizes([int(H * 0.62), int(H * 0.38)])
    pump(2.0)
    path = os.path.join(ROOT, "benchmarks", "figures", "paper_figure2_full.png")
    w.grab().save(path)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    if "--full" in sys.argv:            # the whole window of a saved run (trained first with --train)
        os.makedirs(RUN, exist_ok=True)
        if TRAIN:
            train()
        full_window(shots())
        os._exit(0)
    if sys.argv[2:3] == ["--compose"]:
        compose(os.path.join(RUN, "fig2_a_full.png"), os.path.join(RUN, "fig2_b_full.png"),
                os.path.join(ROOT, "benchmarks", "figures", "paper_figure2.png"))
        os._exit(0)
    os.makedirs(RUN, exist_ok=True)
    if TRAIN:
        train()
    w = shots()
    w.workspace.select("equations")
    pump(1.0)
    ws = w.workspace
    from PyQt6.QtWidgets import QSplitter
    sp = ws.tree_panel.parent()                 # model tree | settings | graphics
    if isinstance(sp, QSplitter):
        sp.setSizes([250, 470, SIZE[0] - 720])
    vs = w.right_tabs.parent()                   # graphics over the Training Log
    while vs is not None and not isinstance(vs, QSplitter):
        vs = vs.parent()
    if vs is not None:
        n = vs.count()
        vs.setSizes([B_HEIGHT] + [SIZE[1] - B_HEIGHT] * (n - 1))
    pump(2.0)
    grab(w, [ws.tree_panel, ws.settings_panel], os.path.join(RUN, "fig2_a_full.png"), height=A_HEIGHT)
    grab(w, [w.right_tabs], os.path.join(RUN, "fig2_b_full.png"))
    compose(os.path.join(RUN, "fig2_a_full.png"), os.path.join(RUN, "fig2_b_full.png"),
            os.path.join(ROOT, "benchmarks", "figures", "paper_figure2.png"))
    os._exit(0)
