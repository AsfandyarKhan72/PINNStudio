"""Figure 3 of the paper: two configured examples in PINNStudio's Geometry view, with a boundary condition
selected so its edges (2D) or faces (3D) are highlighted and its settings are shown.

(a) Helmholtz 2D – sound-hard scattering: a square with a disk subtracted; BC 3 (Robin, u_re) on the four
    outer edges E1–E4, picked with the named sides "left or right or bottom or top"; BC 1 and BC 2 (Neumann)
    on the hole E5.
(b) Poisson 3D – cube with spherical cut-out: a cube with a sphere subtracted, in the 3D view; BC 2
    (Neumann, du/dn = 2 e^{-r^2} (x^2 + y^2 + z(z - 1)) / R) on the spherical surface F7, picked with the
    coordinate expression np.isclose(x**2 + y**2 + (z - 1)**2, R**2); BC 1 (Dirichlet, u = e^{-r^2}) on the
    cube's faces.

Each example is loaded from the Examples list as it ships, with PINNStudio's default appearance, and the
window (its smallest size) is drawn offscreen at twice the screen resolution. Each panel is put together from three crops of that one screenshot,
so the text stays readable at a journal's text width (about 5.4 in, 390 pt):
  - the toolbar's example selection (Examples: category, example name, settings set),
  - the settings of the selected condition (its title, the Condition card with Type, Output, Where and Value,
    and the On the domain card with the edges or faces it picks),
  - the graphics, cropped to the geometry and the legend of the boundary conditions.
Nothing in the screenshots is changed or drawn over; a thin grey frame marks each crop. (a) is above (b):
    QT_QPA_PLATFORM=offscreen python tools/make_geometry_figure.py
writes benchmarks/figures/paper_geometry.png and .pdf (the PDF 5.4 in wide).
"""
import io
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
FONT_PX = 14                # Settings -> Appearance -> Base text size: PINNStudio's default. Larger sizes
                            # were tried: they enlarge only the values in the fields, and the wider image
                            # makes the labels and legends smaller in print, so the default reads best
SIZE = (1100, 750)          # PINNStudio's smallest window, in screen pixels (twice that in the image)
SETTINGS_W = 640            # wide enough for the whole Where: and Value: expressions of (b)
PRINT_W_IN = 5.4            # the manuscript's text width (elsarticle preprint, about 390 pt)
GAP, PAD, LAB = 14, 18, 64  # image pixels: between crops, around the geometry, panel-label row


def _pump(app, s):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()


def _rect(w, x):
    p = x.mapTo(w, x.rect().topLeft())
    return p.x(), p.y(), x.width(), x.height()


def shot(app, key, bc_row, view3d):
    """The window as an image (2x) and the three crop boxes, in image pixels."""
    from PIL import Image
    from PyQt6.QtCore import QBuffer, QIODevice
    from PyQt6.QtWidgets import QComboBox, QLabel
    from pinnstudio.ui.main_window import MainWindow
    w = MainWindow()
    w._maybe_save_changes = lambda: True
    w._font_size = FONT_PX                                 # as the Settings dialog sets it
    w._apply_theme()
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
    w.main_splitter.setCollapsible(0, True)                # the model tree collapsed, as a user can
    w.main_splitter.setSizes([0, SETTINGS_W, w.main_splitter.width() - SETTINGS_W])
    w.right_splitter.setCollapsible(1, True)
    w.right_splitter.setSizes([w.right_splitter.height(), 0])   # the empty training log closed
    _pump(app, 1.5)

    def box(x, y, bw, bh):                                 # screen pixels -> image pixels
        return 2 * x, 2 * y, 2 * (x + bw), 2 * (y + bh)
    # 1. the example selection in the toolbar: from "Examples:" to the settings-set combo after the example
    lab = next(c for c in w.findChildren(QLabel) if c.isVisible() and c.text().startswith("Examples"))
    lx, ly, _, lh = _rect(w, lab)
    ex = _rect(w, combo)
    combos = [_rect(w, c) for c in w.findChildren(QComboBox) if c.isVisible()]
    nxt = min((c for c in combos if abs(c[1] - ex[1]) < 4 and c[0] > ex[0]), key=lambda c: c[0])  # "Default"
    tool = box(lx - 6, ly - 6, nxt[0] + nxt[2] + 6 - (lx - 6), lh + 12)
    # 2. the settings: from the column's top to below the "On the domain" card
    col = _rect(w, w.main_splitter.widget(1))
    picks = next(c for c in w.findChildren(QLabel) if c.isVisible() and c.text().startswith("Picks"))
    px, py, _, ph = _rect(w, picks)
    sett = box(col[0], col[1], col[2], py + ph + 26 - col[1])
    # 3. the graphics (cropped to the geometry later)
    cv = w.domain_view.view3d if view3d else w.domain_view.canvas
    gfx = box(*_rect(w, cv))
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    w.grab().save(buf, "PNG")
    img = Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")
    w.close()
    return img, tool, sett, gfx


def _geometry_box(img, gfx, three_d):
    """The part of the graphics with the geometry and the legend: the bounding box of the strongly coloured
    pixels (boundary edges and faces, edge labels, legend swatches), widened to the right edge of the graphics,
    where the legend's text ends."""
    import numpy as np
    x0, y0, x1, y1 = gfx
    a = np.asarray(img.crop(gfx)).astype(float) / 255
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    mask = (sat > 0.45) & (mx > 0.3)
    top = np.where(mask)[0].min()          # the legend's swatches: above them only the view's buttons
    if three_d:                 # the axes' tick labels too (in 2D they are at the edges of the whole canvas)
        mask[top:] |= mx[top:] < 0.55
    rows, cols = np.where(mask)
    return (max(x0, x0 + cols.min() - PAD), max(y0, y0 + top - PAD),
            x1 - 4, min(y1, y0 + rows.max() + PAD))


def _framed(im):
    from PIL import ImageOps
    return ImageOps.expand(im, border=2, fill=(200, 204, 208))


def main():
    from PyQt6.QtWidgets import QApplication
    from PIL import Image, ImageDraw, ImageFont
    app = QApplication.instance() or QApplication(["pinnstudio"])
    panels = []
    for key, row, v3 in PANELS:
        img, tool, sett, gfx = shot(app, key, row, v3)
        panels.append((_framed(img.crop(tool)), _framed(img.crop(sett)), _framed(img.crop(_geometry_box(img, gfx, v3)))))
    W = max(s.width + GAP + g.width for _, s, g in panels)
    H = sum(LAB + t.height + GAP + max(s.height, g.height) for t, s, g in panels) + GAP
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for letter, (t, s, g) in zip("ab", panels):
        d.text((4, y + 4), f"({letter})", fill="black", font=font)
        y += LAB
        fig.paste(t, (0, y))
        y += t.height + GAP
        fig.paste(s, (0, y))
        fig.paste(g, (W - g.width, y))                     # the graphics flush right, the settings left
        y += max(s.height, g.height) + GAP
    fig.save(os.path.join(OUT, "paper_geometry.png"), optimize=True)
    fig.save(os.path.join(OUT, "paper_geometry.pdf"), resolution=fig.width / PRINT_W_IN)
    print("wrote", os.path.join(OUT, "paper_geometry.png"), fig.size,
          f"(at {PRINT_W_IN} in wide: {PRINT_W_IN * fig.height / fig.width:.2f} in tall; the {FONT_PX} px GUI text "
          f"prints at {2 * FONT_PX * PRINT_W_IN * 72 / fig.width:.1f} pt)")
    os._exit(0)                 # skip Qt's teardown


if __name__ == "__main__":
    main()
