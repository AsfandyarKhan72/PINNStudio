"""Figure 3 of the paper: two configured examples in PINNStudio's Geometry view, with a boundary condition
selected so its edges (2D) or faces (3D) are highlighted and its settings are shown.

(a) Helmholtz 2D – sound-hard scattering: a square with a disk subtracted; BC 3 (Robin, u_re) on the four
    outer edges, picked with the named sides "left or right or bottom or top".
(b) Poisson 3D – cube with spherical cut-out: a cube with a sphere subtracted, in the 3D view; BC 2
    (Neumann, du/dn = 2 e^{-r^2} (x^2 + y^2 + z(z - 1)) / R) on the spherical surface F7, picked with the
    coordinate expression np.isclose(x**2 + y**2 + (z - 1)**2, R**2); BC 1 (Dirichlet, u = e^{-r^2}) on the
    cube's faces.

Only the parts that carry information are kept: the model tree and the empty training log are collapsed with
their splitters, as a user can (the BC types are in the settings and in the legend of the graphics), and the
menu bar, the general help line under the graphics and the status bar are cropped off; nothing in the window
is changed or drawn over. The window has PINNStudio's smallest size, so that the text prints at about 6 pt at
6.5 in (at this width the toolbar leaves out its "PINNStudio" title by itself).

The window is drawn offscreen at twice the screen resolution and the two panels are stacked, (a) above (b):
    QT_QPA_PLATFORM=offscreen python tools/make_geometry_figure.py
writes benchmarks/figures/paper_geometry.png and .pdf.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "benchmarks", "figures")

PANELS = [("Helmholtz sound-hard scattering (absorbing BC)", 2, False),   # (example key, BC row, 3D view)
          ("3D Poisson (Cube with spherical cut-out)", 1, True)]
SIZE = (1100, 750)          # PINNStudio's smallest window, in screen pixels (twice that in the image): its
                            # 14 px text prints at 6 pt when the figure is 6.5 in wide
SETTINGS_W = 640            # wide enough for the whole Where: and Value: expressions of (b)


def _pump(app, s):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()


def shot(app, key, bc_row, view3d):
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w.resize(*SIZE)
    w.show()
    _pump(app, 0.5)
    w.save_dir_input.setText("~/PINNStudio_Results")
    w.quick_examples_category_combo.setCurrentText("Forward")
    _pump(app, 0.3)
    combo = w.quick_examples_combo
    i = combo.findData(key)
    if i < 0:
        raise SystemExit(f"unknown example {key!r}")
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    _pump(app, 2.0)
    w.log_box.clear()
    w.workspace.select(w.custom_bc_list[bc_row]["_node"])  # the BC's settings, its edges/faces highlighted
    _pump(app, 1.5)
    if view3d:
        v = w.domain_view.view_combo
        v.setCurrentIndex(v.findData("3d"))
        _pump(app, 3.0)
    # the model tree collapsed with its splitter, as a user can (the BC types are in the settings and in the
    # legend of the graphics)
    w.main_splitter.setCollapsible(0, True)
    w.main_splitter.setSizes([0, SETTINGS_W, w.main_splitter.width() - SETTINGS_W])
    w.right_splitter.setCollapsible(1, True)
    w.right_splitter.setSizes([w.right_splitter.height(), 0])   # the empty training log closed
    _pump(app, 1.5)
    # leave out the menu bar, the help line under the graphics and the status bar
    top = w.menuBar().height() if w.menuBar().isVisible() else 0
    bottom = w.statusBar().geometry().top() if w.statusBar().isVisible() else w.height()
    h = w.domain_view.hint                                  # the general help line under the graphics
    if h.isVisible():
        bottom = min(bottom, h.mapTo(w, h.rect().topLeft()).y())
    from PyQt6.QtCore import QRect
    img = w.grab(QRect(0, top, w.width(), bottom - top)).toImage()
    w.close()
    return img


def main():
    from PyQt6.QtWidgets import QApplication
    from PIL import Image, ImageDraw, ImageFont
    app = QApplication.instance() or QApplication(["pinnstudio"])
    tmp = []
    for n, (key, row, v3) in enumerate(PANELS):
        path = os.path.join(OUT, f"_geometry_panel_{n}.png")
        shot(app, key, row, v3).save(path)
        tmp.append(path)
    ims = [Image.open(p).convert("RGB") for p in tmp]
    w = max(im.width for im in ims)
    pad, lab = 40, 110
    H = sum(im.height for im in ims) + len(ims) * lab + pad
    fig = Image.new("RGB", (w, H), "white")
    d = ImageDraw.Draw(fig)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for letter, im in zip("ab", ims):
        d.text((10, y + 18), f"({letter})", fill="black", font=font)
        y += lab
        fig.paste(im, (0, y))
        y += im.height + pad // len(ims)
    fig.save(os.path.join(OUT, "paper_geometry.png"), optimize=True)
    fig.save(os.path.join(OUT, "paper_geometry.pdf"), resolution=fig.width / 6.5)   # 6.5 in wide, the print width
    for p in tmp:
        os.remove(p)
    print("wrote", os.path.join(OUT, "paper_geometry.png"), fig.size)
    os._exit(0)                 # skip Qt's teardown


if __name__ == "__main__":
    main()
