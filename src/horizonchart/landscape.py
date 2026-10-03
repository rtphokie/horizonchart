"""Silhouetted landscapes along the horizon: trees, houses or city buildings
drawn in near-black over starplot's ground, the way land looks against a
twilight or night sky."""

import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# "hills" is starplot's plain ground, left as drawn
LANDSCAPES = ("hills", "trees", "suburban", "city")

SILHOUETTE = (11, 12, 20)
WINDOW = (236, 192, 108)
SUPERSAMPLE = 3  # shapes are drawn this many times larger, then scaled down to smooth edges
# Space kept between the top of a silhouette and a target above it, as a
# fraction of image height; silhouettes that can't fit under a target are
# shortened or left out so the target stays visible
CLEARANCE = 0.03
MIN_HEIGHT = 0.012  # anything shorter than this (fraction of height) is dropped


def is_ground(rgb: np.ndarray) -> np.ndarray:
    """Pixels in starplot's ground gradient (dark browns, red above blue), or
    the black border the PNG export leaves along its edges. The sky near the
    horizon is much lighter, and constellation lines are blue-green."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    brown = (r + g + b < 330) & (r >= b + 8) & (r >= g)
    return brown | (r + g + b < 40)


def ground_profile(im: Image.Image) -> list[int]:
    """For each column, the y of the top of the ground: the highest row of
    the unbroken run of ground pixels up from the bottom edge."""
    rgb = np.asarray(im.convert("RGB"))
    h = rgb.shape[0]
    # Length of the run of ground pixels from the bottom of each column
    run = np.cumprod(is_ground(rgb)[::-1], axis=0).sum(axis=0)
    top = h - run
    # A running median over a few columns removes the odd stray pixel
    padded = np.pad(top, 3, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 7)
    return [int(y) for y in np.median(windows, axis=1)]


class Painter:
    """Draws silhouettes on a supersampled mask, collecting lit windows to
    paint on afterwards."""

    def __init__(self, size: tuple[int, int]):
        self.mask = Image.new("L", (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE), 0)
        self.draw = ImageDraw.Draw(self.mask)
        self.windows: list[tuple[float, float, float, float]] = []

    def polygon(self, points: Iterable[tuple[float, float]]) -> None:
        self.draw.polygon([(x * SUPERSAMPLE, y * SUPERSAMPLE) for x, y in points], fill=255)

    def rect(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def ellipse(self, cx: float, cy: float, rx: float, ry: float) -> None:
        s = SUPERSAMPLE
        self.draw.ellipse(((cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s), fill=255)

    def alpha(self) -> Image.Image:
        return self.mask.resize(
            (self.mask.width // SUPERSAMPLE, self.mask.height // SUPERSAMPLE),
            Image.Resampling.LANCZOS,
        )


def conifer(p: Painter, x: float, base: float, w: float, h: float, rng: random.Random) -> None:
    """A spruce or fir: tiers of triangles narrowing to a point."""
    trunk = h * 0.12
    p.rect(x - w * 0.06, base - trunk * 1.5, x + w * 0.06, base)
    tiers = rng.choice((3, 4))
    crown = h - trunk
    for i in range(tiers):
        # Each tier's bottom edge, from the lowest up, overlapping the next
        bottom = base - trunk - crown * i / tiers * 0.85
        top = base - h if i == tiers - 1 else bottom - crown * 0.45
        half = w / 2 * (1 - i / (tiers + 1))
        p.polygon([(x - half, bottom), (x + half, bottom), (x, top)])
    p.polygon([(x - w * 0.12, base - h * 0.8), (x + w * 0.12, base - h * 0.8), (x, base - h)])


def broadleaf(p: Painter, x: float, base: float, w: float, h: float, rng: random.Random) -> None:
    """A rounded deciduous tree: a trunk under a cluster of blobs."""
    trunk = h * 0.3
    p.rect(x - w * 0.07, base - trunk - h * 0.1, x + w * 0.07, base)
    cy = base - trunk - (h - trunk) / 2
    rx, ry = w / 2, (h - trunk) / 2
    p.ellipse(x, cy, rx * 0.75, ry)
    for _ in range(4):
        dx = rng.uniform(-0.45, 0.45) * rx
        dy = rng.uniform(-0.3, 0.35) * ry
        p.ellipse(x + dx, cy + dy, rx * rng.uniform(0.45, 0.6), ry * rng.uniform(0.5, 0.65))


def tree(p: Painter, x: float, base: float, h: float, rng: random.Random) -> None:
    if rng.random() < 0.55:
        conifer(p, x, base, h * rng.uniform(0.38, 0.5), h, rng)
    else:
        broadleaf(p, x, base, h * rng.uniform(0.7, 0.95), h, rng)


def house(p: Painter, x: float, base: float, w: float, h: float, rng: random.Random) -> None:
    """A gabled house, sometimes with a chimney, often with a lit window or two."""
    wall_top = base - h * 0.58
    p.rect(x - w / 2, wall_top, x + w / 2, base)
    eave = w * 0.06
    p.polygon([(x - w / 2 - eave, wall_top), (x + w / 2 + eave, wall_top), (x, base - h)])
    if rng.random() < 0.5:
        cx = x + rng.choice((-1, 1)) * w * 0.25
        p.rect(cx - w * 0.05, base - h * 0.92, cx + w * 0.05, wall_top)
    win = w * 0.13
    for wx in (x - w * 0.25, x + w * 0.25):
        if rng.random() < 0.45:
            wy = wall_top + (base - wall_top) * 0.3
            p.windows.append((wx - win / 2, wy, wx + win / 2, wy + win * 1.1))


def building(p: Painter, x0: float, base: float, w: float, h: float, rng: random.Random) -> None:
    """A city block: a tower with an occasional setback or antenna and a
    scatter of lit windows."""
    top = base - h
    p.rect(x0, top, x0 + w, base)
    if rng.random() < 0.3:  # setback
        inset = w * rng.uniform(0.15, 0.3)
        p.rect(x0 + inset, top - h * 0.12, x0 + w - inset, top + 1)
        top -= h * 0.12
    if rng.random() < 0.15:  # antenna
        mid = x0 + w / 2
        p.rect(mid - 1.5, top - h * 0.25, mid + 1.5, top + 1)
    size = max(3.0, w * 0.06)
    rows, cols = int(h * 0.75 / (size * 2.2)), int(w * 0.8 / (size * 2))
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.12:
                wx = x0 + w * 0.1 + c * size * 2
                wy = base - h * 0.9 + r * size * 2.2
                p.windows.append((wx, wy, wx + size, wy + size * 1.2))


def draw_landscape(
    path: Path,
    kind: str,
    seed: str,
    keep_clear: Iterable[tuple[float, float]] = (),
) -> None:
    """Replace starplot's brown ground in the PNG at `path` with a silhouetted
    `kind` of landscape (see LANDSCAPES). `seed` makes the scene repeatable,
    e.g. the same houses for the same place. Silhouettes stay below each
    (x, y) pixel in `keep_clear`, such as the targets' markers."""
    if kind not in LANDSCAPES:
        raise ValueError(f"Unknown landscape {kind!r}; choose from {', '.join(LANDSCAPES)}")
    if kind == "hills":
        return
    rng = random.Random(f"{kind}:{seed}")
    keep_clear = list(keep_clear)

    with Image.open(path) as im:
        im = im.convert("RGB")
    w, h = im.size
    profile = ground_profile(im)
    p = Painter(im.size)
    # The ground itself, slightly overlapping the sky so the edge is clean
    p.polygon([(0, h), *((x, y - 1) for x, y in enumerate(profile)), (w - 1, h)])

    def place(x0: float, width: float, height: float) -> tuple[float, float] | None:
        """Base y and height for something spanning x0..x0+width, sunk to the
        lowest point of the ground under it and shortened to clear targets."""
        lo, hi = max(0, int(x0)), min(w, int(x0 + width) + 1)
        if lo >= hi:
            return None
        base = max(profile[lo:hi]) + 2
        for tx, ty in keep_clear:
            if x0 - h * CLEARANCE < tx < x0 + width + h * CLEARANCE and ty < base:
                height = min(height, base - ty - h * CLEARANCE)
        if height < h * MIN_HEIGHT:
            return None
        return base, height

    x = rng.uniform(-0.02, 0.01) * w
    while x < w:
        if kind == "trees":
            height = h * rng.uniform(0.03, 0.065)
            width = height * 0.6
            if spot := place(x, width, height):
                tree(p, x + width / 2, spot[0], spot[1], rng)
            x += width * rng.uniform(0.35, 0.8)
        elif kind == "suburban":
            if rng.random() < 0.55:
                height = h * rng.uniform(0.03, 0.045)
                width = height * rng.uniform(1.4, 2.1)
                if spot := place(x, width, height):
                    house(p, x + width / 2, spot[0], width, spot[1], rng)
            else:
                height = h * rng.uniform(0.035, 0.07)
                width = height * 0.6
                if spot := place(x, width, height):
                    tree(p, x + width / 2, spot[0], spot[1], rng)
            x += width * rng.uniform(0.9, 1.6)
        else:  # city
            height = h * rng.uniform(0.04, 0.14)
            width = h * rng.uniform(0.03, 0.08)
            if spot := place(x, width, height):
                building(p, x, spot[0], width, spot[1], rng)
            x += width * rng.uniform(0.8, 0.97)  # overlap, so no seams of sky

    alpha = p.alpha()
    im.paste(Image.new("RGB", im.size, SILHOUETTE), (0, 0), alpha)
    draw = ImageDraw.Draw(im)
    for box in p.windows:
        # Only where a building rises above the hills, so the ground below,
        # where the title and timestamp go, stays dark
        x0, _y0, x1, y1 = box
        if 0 <= x0 and x1 < w and y1 < min(profile[int(x0) : int(x1) + 1]) - 4:
            draw.rectangle(box, fill=WINDOW)
    im.save(path)
