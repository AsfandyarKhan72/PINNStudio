"""PINNStudio's cover image: assets/social_preview.png (1280 x 640), for the repository's social preview on GitHub
(Settings -> General -> Social preview; GitHub has no API for it, so it is uploaded by hand) and for the project
page's link preview (docs/assets/social_preview.png). Drawn from the application icon (the SVG in
pinnstudio/ui/icons.py) and the README's interface screenshot. Run from the repository root:

    QT_QPA_PLATFORM=offscreen python tools/make_social_preview.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
W, H = 1280, 640
FONTS = {"bold": ["Inter-Bold.otf", "InterDisplay-Bold.otf", "DejaVuSans-Bold.ttf"],
         "medium": ["Inter-Medium.otf", "DejaVuSans.ttf"],
         "regular": ["Inter-Regular.otf", "DejaVuSans.ttf"]}


def font(kind, size):
    from PIL import ImageFont
    for name in FONTS[kind]:
        for d in ("/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(d, name) if d else name, size)
            except OSError:
                pass
    return ImageFont.load_default()


def main():
    from PIL import Image, ImageDraw, ImageFilter
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(["pinnstudio"])  # noqa: F841
    from make_app_icon import render
    from pinnstudio.ui.icons import _APP

    # background: the icon's two blues, top left to bottom right
    img = Image.new("RGB", (W, H))
    top, bot = (11, 29, 54), (14, 92, 116)
    px = img.load()
    for y in range(H):
        for x in range(W):
            s = 0.65 * x / W + 0.35 * y / H
            px[x, y] = tuple(int(a + (b - a) * s) for a, b in zip(top, bot))

    # the interface screenshot on the right, faded into the background
    shot = Image.open(os.path.join(ROOT, "assets", "screenshots", "interface.png")).convert("RGB")
    sw = 760
    shot = shot.resize((sw, int(shot.height * sw / shot.width)), Image.LANCZOS)
    x0, y0 = W - sw + 230, 120
    mask = Image.new("L", shot.size, 0)
    md = ImageDraw.Draw(mask)
    for i in range(sw):                    # fade in from the left
        md.line([(i, 0), (i, shot.height)], fill=int(235 * min(1.0, i / 260)))
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle([x0 - 4, y0 + 10, x0 + sw + 4, y0 + shot.height + 18], 18, fill=110)
    img = Image.composite(Image.new("RGB", (W, H), (4, 12, 24)), img, shadow.filter(ImageFilter.GaussianBlur(18)))
    rounded = Image.new("L", shot.size, 0)
    ImageDraw.Draw(rounded).rounded_rectangle([0, 0, sw - 1, shot.height - 1], 14, fill=255)
    from PIL import ImageChops
    img.paste(shot, (x0, y0), ImageChops.multiply(mask, rounded))

    # a soft dark band behind the text so it stays readable over the screenshot's edge
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle([0, 0, 780, H], fill=120)
    img = Image.composite(Image.new("RGB", (W, H), top), img, band.filter(ImageFilter.GaussianBlur(90)))

    d = ImageDraw.Draw(img)
    icon = render(_APP, 168)
    img.paste(icon, (80, 96), icon)
    d.text((80, 300), "PINNStudio", font=font("bold", 92), fill=(255, 255, 255))
    d.text((84, 412), "A graphical platform for forward and inverse", font=font("medium", 33), fill=(255, 201, 77))
    d.text((84, 454), "physics-informed neural networks", font=font("medium", 33), fill=(255, 201, 77))
    d.text((84, 530), "ODEs and PDEs in 1D, 2D, 3D  ·  Linux, Windows, macOS",
           font=font("regular", 22), fill=(196, 214, 232))
    for out in (os.path.join(ROOT, "assets", "social_preview.png"),
                os.path.join(ROOT, "docs", "assets", "social_preview.png")):
        img.save(out, optimize=True)
        print("wrote", out, img.size)


if __name__ == "__main__":
    main()
