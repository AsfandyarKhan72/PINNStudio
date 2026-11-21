"""The Windows program's splash screen, packaging/windows/splash.png, drawn from PINNStudio's icon
(pinnstudio/resources/pinnstudio.png).

    python packaging/windows/make_splash.py [OUT.png]

The image is kept in the repository: run this only to change it, and look at the result. Qt's offscreen
platform on Windows has no fonts (every letter came out as an empty box), so on Windows it runs with the
normal platform, and it refuses to draw without fonts. PyInstaller's bootloader shows the image while Python,
PyTorch and Qt load, and writes "Starting…" at the bottom left (text_pos in pinnstudio.spec), so that strip is
left plain."""
import os
import sys
from pathlib import Path

if sys.platform != "win32":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
W, H = 440, 250


def main(out):
    app = QGuiApplication(sys.argv[:1])          # noqa: F841 (fonts need it)
    from PySide6.QtGui import QFontDatabase
    if not QFontDatabase.families():
        raise SystemExit("no fonts on this Qt platform: the text would be empty boxes")
    img = QImage(W, H, QImage.Format.Format_ARGB32)
    img.fill(QColor("#f6f8fc"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setPen(QPen(QColor("#d5dbe8"), 1))
    p.drawRect(QRectF(0.5, 0.5, W - 1, H - 1))
    icon = QImage(str(ROOT / "pinnstudio" / "resources" / "pinnstudio.png"))
    if not icon.isNull():
        p.drawImage(QRectF(28, 46, 96, 96), icon)
    title = QFont("Segoe UI")
    title.setPixelSize(34)
    title.setWeight(QFont.Weight.DemiBold)
    p.setFont(title)
    p.setPen(QColor("#1f2a44"))
    p.drawText(QRectF(144, 54, W - 160, 44), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "PINNStudio")
    sub = QFont("Segoe UI")
    sub.setPixelSize(14)
    p.setFont(sub)
    p.setPen(QColor("#4a5873"))
    p.drawText(QRectF(146, 98, W - 162, 44), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
               "Physics-informed neural networks for forward and inverse problems")
    p.end()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not img.save(str(out)):
        raise SystemExit(f"could not write {out}")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("splash.png")))
