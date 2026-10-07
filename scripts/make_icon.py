"""Generates the DigiQual app icon in every format the desktop builds need.

Single source of truth for the icon: edit the colours/geometry below and
re-run `uv run python scripts/make_icon.py`. Writes:

- app/icons/digiqual.icns           macOS app (Briefcase)
- app/icons/digiqual.ico            Windows app (Briefcase)
- app/icons/digiqual-<size>.png     Linux / Briefcase fallbacks
- src/digiqual/gui/www/favicon.png  browser tab / Shiny app
- docs/www/digiqual-logo.png        docs navbar logo (512px for high-DPI screens)
- docs/www/favicon.png              docs site browser tab

The design is a probability-of-detection curve: a sigmoid rising across a
rounded tile, with a dashed 90% PoD line and a marker at a90.
"""

from math import exp
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "app" / "icons"
FAVICON = ROOT / "src" / "digiqual" / "gui" / "www" / "favicon.png"
DOCS_DIR = ROOT / "docs" / "www"

BACKGROUND = (31, 78, 121)  # deep blue
GRID = (74, 113, 148)  # background blended 17% towards white; solid so the tile stays opaque
CURVE = (255, 255, 255)
TARGET = (255, 196, 61)  # amber: the 90% line and a90 marker

SIZE = 1024
SUPERSAMPLE = 4  # draw large, downscale for smooth anti-aliased edges


def _sigmoid(x: float) -> float:
    return 1 / (1 + exp(-11 * (x - 0.45)))


def draw_master() -> Image.Image:
    s = SIZE * SUPERSAMPLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # macOS-style tile: ~10% transparent margin, rounded corners
    margin = int(s * 0.10)
    d.rounded_rectangle(
        [margin, margin, s - margin, s - margin], radius=int(s * 0.18), fill=BACKGROUND
    )

    # Plot area inside the tile
    left, right = int(s * 0.24), int(s * 0.78)
    top, bottom = int(s * 0.25), int(s * 0.75)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return left + x * (right - left), bottom - y * (bottom - top)

    # Faint grid
    for frac in (0.25, 0.5, 0.75):
        y = bottom - frac * (bottom - top)
        d.line([(left, y), (right, y)], fill=GRID, width=int(s * 0.006))

    # Axes
    axis_w = int(s * 0.022)
    d.line([(left, top - s * 0.02), (left, bottom), (right + s * 0.02, bottom)],
           fill=CURVE, width=axis_w, joint="curve")

    # 90% PoD target line (dashed) and the a90 crossing point
    y90 = bottom - 0.9 * (bottom - top)
    dash, gap = s * 0.035, s * 0.025
    x = left
    while x < right:
        d.line([(x, y90), (min(x + dash, right), y90)], fill=TARGET, width=int(s * 0.016))
        x += dash + gap

    # PoD curve, stamped as overlapping discs: a thick polyline of many short
    # segments leaves jagged spikes at the joins.
    r = s * 0.0225
    for i in range(2001):
        px, py = to_px(i / 2000, _sigmoid(i / 2000))
        d.ellipse([px - r, py - r, px + r, py + r], fill=CURVE)

    # a90 marker: where the curve crosses 0.9
    a90 = next(i / 200 for i in range(201) if _sigmoid(i / 200) >= 0.9)
    cx, cy = to_px(a90, 0.9)
    r = s * 0.05
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=TARGET, outline=BACKGROUND,
              width=int(s * 0.014))

    return img.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> None:
    master = draw_master()
    ICON_DIR.mkdir(parents=True, exist_ok=True)

    for size in (16, 32, 64, 128, 256, 512, 1024):
        master.resize((size, size), Image.LANCZOS).save(ICON_DIR / f"digiqual-{size}.png")

    master.save(ICON_DIR / "digiqual.icns")
    master.save(
        ICON_DIR / "digiqual.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    master.resize((64, 64), Image.LANCZOS).save(FAVICON)

    # Docs navbar logo; 512px stays crisp on high-DPI screens
    master.resize((512, 512), Image.LANCZOS).save(DOCS_DIR / "digiqual-logo.png")
    master.resize((64, 64), Image.LANCZOS).save(DOCS_DIR / "favicon.png")

    print(f"Wrote icons to {ICON_DIR}, {FAVICON.parent} and {DOCS_DIR}")


if __name__ == "__main__":
    main()
