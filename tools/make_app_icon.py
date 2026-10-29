"""Render PINNStudio's application icon files from the SVG in pinnstudio/ui/icons.py (_APP):

    pinnstudio/resources/pinnstudio.png    256 x 256 (Linux app menu, README)
    pinnstudio/resources/pinnstudio.ico    16-256 px (Windows shortcuts)
    pinnstudio/resources/pinnstudio.icns   16-1024 px (macOS app bundle)

The icon was drawn for PINNStudio with the assistance of Claude (Oct 2026) and is part of PINNStudio under its
license. Run from the repository root:

    QT_QPA_PLATFORM=offscreen python tools/make_app_icon.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "pinnstudio", "resources")


def render(svg, size):
    """The SVG drawn at size x size pixels, as a PIL image."""
    import io
    from PIL import Image
    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer
    im = QImage(size, size, QImage.Format.Format_ARGB32)
    im.fill(Qt.GlobalColor.transparent)
    p = QPainter(im)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(p, QRectF(0, 0, size, size))
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    im.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGBA")


def main():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(["pinnstudio"])  # noqa: F841
    from pinnstudio.ui.icons import _APP
    render(_APP, 256).save(os.path.join(RES, "pinnstudio.png"), optimize=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    render(_APP, 256).save(os.path.join(RES, "pinnstudio.ico"), sizes=[(s, s) for s in sizes],
                           append_images=[render(_APP, s) for s in sizes])
    render(_APP, 1024).save(os.path.join(RES, "pinnstudio.icns"))
    print("wrote", ", ".join(f"pinnstudio.{e}" for e in ("png", "ico", "icns")), "in", RES)


if __name__ == "__main__":
    main()
