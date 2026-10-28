"""Figure 3 of the paper: two configured examples in PINNStudio's Geometry view, with a boundary condition
selected so its edges (2D) or faces (3D) are highlighted and its settings are shown.

(a) Helmholtz 2D – sound-hard scattering: a square with a disk subtracted; BC 3 (Robin, u_re) on the four
    outer edges, picked with the named sides "left or right or bottom or top".
(b) Poisson 3D – cube with spherical cut-out: a cube with a sphere subtracted, in the 3D view; BC 2
    (Dirichlet) on the spherical surface, picked with a coordinate expression.

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
SIZE = (1600, 900)          # window size in screen pixels (twice that in the image)


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
    w.right_splitter.setSizes([SIZE[1] - 230, 70])      # most of the right side for the graphics
    w.workspace.select(w.custom_bc_list[bc_row]["_node"])  # the BC's settings, its edges/faces highlighted
    _pump(app, 1.5)
    if view3d:
        v = w.domain_view.view_combo
        v.setCurrentIndex(v.findData("3d"))
        _pump(app, 3.0)
    img = w.grab().toImage()
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
    fig.save(os.path.join(OUT, "paper_geometry.pdf"), resolution=300)
    for p in tmp:
        os.remove(p)
    print("wrote", os.path.join(OUT, "paper_geometry.png"), fig.size)
    os._exit(0)                 # skip Qt's teardown


if __name__ == "__main__":
    main()
