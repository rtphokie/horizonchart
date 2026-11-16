"""Horizon-view finder charts for one or more sky targets.

`plot_targets` draws a clean, 16:9 horizon view showing only the targets,
the constellations useful for finding them, highlighted asterisms, and the
brightest stars for orientation.
"""

import math
import re
from dataclasses import dataclass
from functools import cache
from datetime import date, datetime, time, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

# Sets starplot's data path; must come before starplot is imported
from horizonchart import DATA_PATH

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity
from shapely.geometry import Point
from skyfield import almanac
from skyfield.api import Loader, wgs84
from skyfield.api import Star as SkyfieldStar

from starplot import (
    DSO,
    CollisionHandler,
    Constellation,
    HorizonPlot,
    Moon,
    Observer,
    Planet,
    Star,
    _,
    callables,
)
from starplot.styles import LabelStyle, PlotStyle, extensions
from timezonefinder import TimezoneFinder

load = Loader(str(DATA_PATH))


@cache
def ephemeris():
    """The DE421 ephemeris, opened once; skyfield's Loader doesn't cache."""
    return load("de421.bsp")

PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]

# Constellations are shown only if they contain a target, their boundary
# passes within CLOSE_BOUNDARY of one, or they're the home of a highlighted
# asterism. Asterisms are highlighted if within ASTERISM_RADIUS of a target.
CLOSE_BOUNDARY = 3  # degrees

# Constellations that are easy to recognize, shown (and labeled) whenever at
# least LANDMARK_FRACTION of their stick figure is in the view
LANDMARK_CONSTELLATIONS = [
    "ori",  # Orion
    "uma",  # Ursa Major
    "umi",  # Ursa Minor
    "cas",  # Cassiopeia
    "cyg",  # Cygnus
    "sco",  # Scorpius
    "leo",  # Leo
    "cru",  # Crux (Southern Cross)
]
LANDMARK_FRACTION = 0.8
# An asterism covering this much of its constellation's figure is labeled
# with the constellation's name only
SAME_FIGURE_FRACTION = 0.8
ASTERISM_RADIUS = 35  # degrees

# Stars brighter than this are drawn even outside the chosen constellations;
# those brighter than BRIGHT_STAR_LABEL_MAGNITUDE count toward a view's interest
BRIGHT_STAR_MAGNITUDE = 1.5
BRIGHT_STAR_LABEL_MAGNITUDE = 0.0

# At most MAX_STAR_LABELS stars are labeled per view: those in
# STAR_LABEL_PRIORITY first (best known first), then other LABELED_STARS,
# brightest first. Other stars stay unlabeled rather than showing catalog IDs
# or Greek-letter designations.
MAX_STAR_LABELS = 5
STAR_LABEL_PRIORITY = [
    "Polaris", "Sirius", "Betelgeuse", "Vega", "Antares", "Rigel",
    "Rigil Kentaurus",  # Alpha Centauri
    "Aldebaran", "Arcturus", "Spica", "Altair", "Deneb", "Canopus", "Capella",
    "Castor", "Pollux", "Regulus",
]
LABELED_STARS = [
    "Sirius", "Canopus", "Rigil Kentaurus", "Arcturus", "Vega", "Capella",
    "Rigel", "Procyon", "Achernar", "Betelgeuse", "Hadar", "Altair", "Acrux",
    "Aldebaran", "Antares", "Spica", "Pollux", "Fomalhaut", "Deneb", "Mimosa",
    "Regulus", "Castor", "Bellatrix", "Polaris", "Mizar", "Algol", "Denebola",
    "Albireo",
]

ASPECT = 16 / 9
# Degrees of azimuth per degree of altitude for a roughly 16:9 HorizonPlot;
# the result is then cropped to exactly 16:9
AZ_PER_ALT = 1.87
FRAME_MARGIN = 6  # degrees of sky around the content that must be in view

TARGET_FONT_SIZE = 110
# Planets too faint for the naked eye are drawn smaller and semi-transparent
FAINT_PLANETS = {"uranus", "neptune", "pluto"}
FAINT_FONT_SIZE = 64
FAINT_MARKER_SIZE = 26
FAINT_OPACITY = 0.55
DSO_FONT_SIZE = 64
CONSTELLATION_FONT_SIZE = 80
ASTERISM_FONT_SIZE = 64
BRIGHT_STAR_FONT_SIZE = 56
# Target label positions, tried in order (toward the middle of the view first).
# Centered anchors are left out: they start the text on the marker.
RIGHT_ANCHORS = ["right_center", "bottom_right", "top_right"]
LEFT_ANCHORS = ["left_center", "bottom_left", "top_left"]
# For a target with another just below or above it
ABOVE_ANCHORS = ["top_right", "top_left"]
BELOW_ANCHORS = ["bottom_right", "bottom_left"]
# Targets closer than this (display px) are labeled away from each other
NEIGHBOR_DISTANCE = 600
# Gaps (px) between a target's marker and a label beside it: the usual one,
# then farther out if every side is taken; and between the marker and a
# label above or below it
TARGET_LABEL_GAPS = (40, 90)
VERTICAL_LABEL_GAP = 8
# The Moon and planets are drawn in roughly their naked-eye colors, kept light
# enough to read against the dark sky
BODY_COLORS = {
    "MOON": "#c9ced6",  # silver
    "MERCURY": "#cfc4b4",  # grey-tan
    "VENUS": "#fff4d6",  # brilliant white-cream
    "MARS": "#e8734f",  # rusty red
    "JUPITER": "#f2e4c2",  # cream
    "SATURN": "#e9d49a",  # pale butterscotch
    "URANUS": "#b4e3e8",  # pale aqua
    "NEPTUNE": "#8fb0ff",  # blue
    "PLUTO": "#d9c4a6",  # tan
}
TARGET_COLORS = {
    "planet": "#f5d77a",
    "moon": "#e8e8e8",
    "dso": "#9fd8ff",
    "star": "#ffffff",
    "radiant": "#ffb36b",
}
# Drawn as rings so the marker doesn't hide what's there: a deep-sky object, or
# the stars a meteor shower's radiant is among
HOLLOW_KINDS = {"dso", "radiant"}

ASTERISM_COLOR = "#5fc4c9"
# Labels outside an asterism: sides tried after the asterism's preferred one,
# the gap from the outermost star (px), and rough text metrics (per point of
# font size, in px) used to check the label fits in the frame
SIDE_ANCHORS = ["right_center", "left_center", "bottom_center", "top_center"]
LABEL_GAP = 40
CHAR_WIDTH = 0.56
LINE_HEIGHT = 0.8
LABEL_MIN_ALTITUDE = 7  # keep labels above the drawn ground
PLOT_RESOLUTION = 4096  # HorizonPlot's default width in px
# Starplot can't center multi-line text, so each line of an asterism label is
# drawn separately, stacked this many degrees of altitude apart (at the
# reference azimuth span below; scaled for wider or narrower views)
LABEL_LINE_SPACING = 3.2
# The PNG renderer ignores vertical centering and sits text on its baseline,
# so drop labels about half a line to center them on their point
LABEL_BASELINE_SHIFT = -1.2
REFERENCE_AZ_SPAN = 140

# Twilight views: fixed east/west views this many degrees high, showing the
# Moon and planets, and stars and deep-sky objects down to a limiting
# magnitude (default LIMITING_MAGNITUDE, for suburban skies in twilight; dark
# skies reach 4-6). Stars that make up a drawn constellation or asterism are
# always shown so the figures connect.
TWILIGHT_ALTITUDE = 75
TWILIGHT_BODIES = [
    "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"
]
LIMITING_MAGNITUDE = 2.0
DSO_MIN_ALTITUDE = 10  # lower than this, twilight and haze hide them
DSO_STAR_TYPES = ["*", "**"]  # catalog entries that are really single/double stars
EDGE_MARGIN = 3  # degrees; objects this close to the frame edge are left out

# Manual constellation label positions (RA, Dec in degrees) where starplot's
# automatic placement crowds something; Andromeda's default sits on the Square
CONSTELLATION_LABEL_POSITIONS = {
    "and": (359.5, 40.6),
}


@dataclass(frozen=True)
class Asterism:
    name: str  # "\n" splits the label into centered lines
    lines: list[list[int]]  # polylines of Hipparcos IDs
    label_style: dict  # overrides for the label (placed at the asterism's center)
    constellation: str | None  # home constellation, shown along with it


ASTERISMS = [
    Asterism(
        "Great Square\nof Pegasus",
        [[677, 113881, 113963, 1067, 677]],  # Alpheratz, Scheat, Markab, Algenib
        {},
        "peg",
    ),
    Asterism(
        "Circlet of Pisces",
        [[116771, 115830, 114971, 115738, 116928, 116771]],  # ι, θ, γ, κ, λ Psc
        {"anchor_point": "right_center"},
        "psc",
    ),
    Asterism(
        "Big Dipper",
        # Alkaid, Mizar, Alioth, Megrez, Dubhe, Merak, Phecda, Megrez
        [[67301, 65378, 62956, 59774, 54061, 53910, 58001, 59774]],
        {"anchor_point": "bottom_center"},
        "uma",
    ),
    Asterism(
        "Little Dipper",
        # Polaris, Yildun (δ), ε, ζ UMi, then the bowl: Kochab (β), Pherkad (γ), η, ζ
        [[11767, 85822, 82080, 77055, 72607, 75097, 79822, 77055]],
        {"anchor_point": "right_center"},
        "umi",
    ),
    Asterism(
        "W of Cassiopeia",
        [[746, 3179, 4427, 6686, 8886]],  # Caph, Schedar, γ Cas, Ruchbah, Segin
        {"anchor_point": "bottom_center"},
        "cas",
    ),
    Asterism(
        "Summer\nTriangle",
        [[91262, 102098, 97649, 91262]],  # Vega, Deneb, Altair
        {},
        None,
    ),
    Asterism(
        "Northern Cross",
        # Deneb, Sadr, Albireo; Gienah (ε Cyg), Sadr, δ Cyg
        [[102098, 100453, 95947], [102488, 100453, 97165]],
        {"anchor_point": "right_center"},
        "cyg",
    ),
    Asterism(
        "Keystone",
        [[81693, 81833, 84380, 84379, 81693]],  # ζ, η, π, ε Her
        {"anchor_point": "right_center"},
        "her",
    ),
    Asterism(
        "Sickle of Leo",
        [[49669, 49583, 50583, 50335, 48455, 47908]],  # Regulus, η, γ, ζ, μ, ε Leo
        {"anchor_point": "right_center"},
        "leo",
    ),
    Asterism(
        "Orion's Belt",
        [[26727, 26311, 25930]],  # Alnitak, Alnilam, Mintaka
        {"anchor_point": "bottom_center"},
        "ori",
    ),
    Asterism(
        "Winter\nTriangle",
        [[27989, 37279, 32349, 27989]],  # Betelgeuse, Procyon, Sirius
        {},
        None,
    ),
]


# Converting a point exactly on the Sun, Jupiter or Saturn to alt/az applies
# that body's own light deflection, a divide-by-zero that yields NaN (starplot
# then silently drops the marker). Solar system targets are nudged 0.4" in Dec.
DEFLECTION_NUDGE = 1e-4


@dataclass(frozen=True)
class Target:
    label: str
    kind: str  # planet, moon, dso, star or radiant
    ra: float  # degrees
    dec: float  # degrees
    font_size: float = TARGET_FONT_SIZE
    faint: bool = False  # not visible to the naked eye


def resolve_target(name: str, observer: Observer) -> Target:
    """Look up a target by name: the Moon, a planet, a Messier ID (M45), an
    NGC/IC ID (NGC 224), a deep-sky common name (Pleiades) or a star name.
    Planets also accept BSP ephemeris naming ("saturn barycenter", "SATURN
    BARYCENTER", "saturn_barycenter"); at finder-chart scale a barycenter and
    its planet are the same point."""
    key = re.sub(r"[\s_]+", " ", name.strip().lower())
    key = re.sub(r" barycenter$", "", key)
    if key == "moon":
        moon = Moon.get(observer=observer)
        return Target("MOON", "moon", moon.ra, moon.dec + DEFLECTION_NUDGE)
    if key in PLANETS:
        planet = Planet.get(key, observer=observer)
        faint = key in FAINT_PLANETS
        return Target(
            key.upper(),
            "planet",
            planet.ra,
            planet.dec + DEFLECTION_NUDGE,
            FAINT_FONT_SIZE if faint else TARGET_FONT_SIZE,
            faint,
        )

    if match := re.fullmatch(r"m\s*(\d+)", key):
        dso = DSO.get(m=match[1])
    elif match := re.fullmatch(r"(ngc|ic)\s*(\d+)", key):
        dso = DSO.get(**{match[1]: match[2]})
    else:
        found = DSO.find(where=[_.common_names.lower().contains(key)])
        dso = found[0] if found else None
    if dso:
        return Target(name.strip().upper(), "dso", dso.ra, dso.dec)

    stars = Star.find(where=[_.name.lower() == key])
    if stars:
        return Target(stars[0].name.upper(), "star", stars[0].ra, stars[0].dec)

    raise ValueError(f"Unknown target: {name!r}")


def angular_distance(ra1, dec1, ra2, dec2) -> float:
    """Great-circle distance in degrees between two RA/Dec points (degrees)."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cos_d = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(
        dec2
    ) * math.cos(ra1 - ra2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_d))))


class Sky:
    """Alt/az conversions for one observer and moment."""

    def __init__(self, lat: float, lon: float, dt: datetime):
        ts = load.timescale()
        eph = ephemeris()
        self.here = (eph["earth"] + wgs84.latlon(lat, lon)).at(ts.from_datetime(dt))

    def altaz(self, ra: float, dec: float) -> tuple[float, float]:
        target = SkyfieldStar(ra_hours=ra / 15, dec_degrees=dec)
        alt, az, _dist = self.here.observe(target).apparent().altaz()
        return alt.degrees, az.degrees

    def shift_altitude(self, ra: float, dec: float, d_alt: float) -> tuple[float, float]:
        """RA/Dec of the point `d_alt` degrees above `ra`/`dec`."""
        alt, az = self.altaz(ra, dec)
        shifted = self.here.from_altaz(alt_degrees=alt + d_alt, az_degrees=az)
        new_ra, new_dec, _dist = shifted.radec()
        return new_ra._degrees, new_dec.degrees


@cache
def all_constellations() -> list[Constellation]:
    return list(Constellation.all())


def constellation_of(ra: float, dec: float) -> str | None:
    point = Point(ra, dec)
    for constellation in all_constellations():
        if constellation.boundary.contains(point):
            return constellation.iau_id
    return None


def same_as_constellation(asterism: Asterism) -> bool:
    """True when the asterism is most of its home constellation's figure (the
    W of Cassiopeia, the Little Dipper), so the constellation's name alone
    labels it."""
    return _same_as_constellation(asterism.name)


@cache
def _same_as_constellation(name: str) -> bool:
    asterism = next(a for a in ASTERISMS if a.name == name)
    if not asterism.constellation:
        return False
    figure = set(Constellation.get(iau_id=asterism.constellation).star_hip_ids)
    shape = {hip for line in asterism.lines for hip in line}
    return len(figure & shape) / len(figure) >= SAME_FIGURE_FRACTION


def select_asterisms(targets: list[Target]) -> list[tuple[Asterism, list[list[Star]]]]:
    """Asterisms whose center is within ASTERISM_RADIUS of any target."""
    selected = []
    for asterism in ASTERISMS:
        lines = [[Star.get(hip=hip) for hip in line] for line in asterism.lines]
        ra, dec = center_of({s.hip: s for line in lines for s in line}.values())
        if any(angular_distance(ra, dec, t.ra, t.dec) <= ASTERISM_RADIUS for t in targets):
            selected.append((asterism, lines))
    return selected


def boundary_distance(constellation: Constellation, ra: float, dec: float) -> float:
    """Approximate angular distance (degrees) from a point to a constellation's
    boundary; 0 if inside. RA is scaled by cos(Dec) around the point, and the
    point is also tried a full turn either way for boundaries split at 0°/360°."""
    distances = []
    for shift in (-360, 0, 360):
        point = Point(ra + shift, dec)
        boundary = affinity.scale(
            constellation.boundary,
            xfact=math.cos(math.radians(dec)),
            yfact=1,
            origin=point,
        )
        distances.append(boundary.distance(point))
    return min(distances)


def select_constellations(
    targets: list[Target], asterisms: list[Asterism] = ()
) -> list[str]:
    """Constellations containing a target or within CLOSE_BOUNDARY of one,
    plus the home constellations of `asterisms`."""
    selected = {
        c.iau_id
        for c in all_constellations()
        if any(boundary_distance(c, t.ra, t.dec) <= CLOSE_BOUNDARY for t in targets)
    }
    selected |= {a.constellation for a in asterisms if a.constellation}
    return sorted(selected)


def landmarks_in_view(
    sky: Sky, altitude: tuple[float, float], azimuth: tuple[float, float]
) -> list[str]:
    """LANDMARK_CONSTELLATIONS with at least LANDMARK_FRACTION of their stick
    figure stars above the horizon and inside the view."""
    center = (azimuth[0] + azimuth[1]) / 2
    half_span = (azimuth[1] - azimuth[0]) / 2 - EDGE_MARGIN
    found = []
    for c in all_constellations():
        if c.iau_id not in LANDMARK_CONSTELLATIONS:
            continue
        stars = Star.find(where=[_.hip.isin(c.star_hip_ids)])
        inside = 0
        for s in stars:
            alt, az = sky.altaz(s.ra, s.dec)
            offset = ((az - center + 180) % 360) - 180
            inside += 0 < alt < altitude[1] - EDGE_MARGIN and abs(offset) < half_span
        if stars and inside / len(stars) >= LANDMARK_FRACTION:
            found.append(c.iau_id)
    return found


def center_of(stars) -> tuple[float, float]:
    """Mean RA/Dec of `stars`, with RA averaged on the circle so groups
    straddling 0°/360° (or anywhere else) come out right."""
    stars = list(stars)
    ra = math.degrees(
        math.atan2(
            sum(math.sin(math.radians(s.ra)) for s in stars),
            sum(math.cos(math.radians(s.ra)) for s in stars),
        )
    )
    return ra % 360, sum(s.dec for s in stars) / len(stars)


def frame(points_altaz: list[tuple[float, float]]) -> tuple[tuple, tuple]:
    """Altitude and azimuth ranges (degrees) that contain every point plus a
    margin, from the horizon up, sized slightly wider than 16:9."""
    # Azimuths relative to their circular mean, so views across north work
    mean = math.degrees(
        math.atan2(
            sum(math.sin(math.radians(az)) for _alt, az in points_altaz),
            sum(math.cos(math.radians(az)) for _alt, az in points_altaz),
        )
    )
    offsets = [((az - mean + 180) % 360) - 180 for _alt, az in points_altaz]
    az_lo, az_hi = min(offsets) - FRAME_MARGIN, max(offsets) + FRAME_MARGIN
    alt_hi = min(90, max(alt for alt, _az in points_altaz) + FRAME_MARGIN)

    az_span = max(az_hi - az_lo, alt_hi * AZ_PER_ALT)
    if az_span > 180:
        raise ValueError("Targets are too far apart to fit in one view")
    if az_span / AZ_PER_ALT > alt_hi:
        alt_hi = min(90, az_span / AZ_PER_ALT)

    center = (mean + (az_lo + az_hi) / 2) % 360
    az_min = center - az_span / 2
    if az_min < 0:
        az_min += 360
    return (0, alt_hi), (az_min, az_min + az_span)


def crop_to_aspect(path: Path, aspect: float = ASPECT) -> None:
    """Trim a PNG to `aspect`: excess height comes off the top (keeping the
    horizon), excess width comes off both sides equally."""
    with Image.open(path) as im:
        w, h = im.size
        if w / h < aspect:
            box = (0, h - round(w / aspect), w, h)
        else:
            trim = (w - round(h * aspect)) // 2
            box = (trim, 0, trim + round(h * aspect), h)
        im.crop(box).save(path)


# Inter (SIL Open Font License), bundled so titles look the same everywhere
FONT_PATH = files("horizonchart") / "fonts" / "Inter.ttf"
TITLE_SIZE = 0.05  # title text height as a fraction of image height
HEADING_MARGIN = 0.03  # from the top edge, as a fraction of image height
# Sky added above a chart with a heading, as a fraction of its height, so the
# heading never covers what's charted (views can reach the zenith, so the
# room can't come from the view itself)
HEADROOM = 2 * HEADING_MARGIN + TITLE_SIZE
EDGE_SAMPLE = 24  # px of each edge whose median color extends the image
FOOTNOTE_SIZE = 0.018
FOOTNOTE_MARGIN = 0.015  # from the bottom and right edges
FOOTNOTE_COLOR = "#a0a0a8"
# The ground: starplot's brown gradient (gradients.GROUND) darkened to near
# black, the way land looks against a twilight or night sky
GROUND_STOPS = (
    (0.0, "#251A12"),
    (0.15, "#20150F"),
    (0.4, "#19110D"),
    (0.7, "#100A07"),
    (1.0, "#090604"),
)
# Hills between these altitudes (degrees)
GROUND = (3.5, 6)
# Twilight views draw a taller landscape so the title fits on it: hills
# between these altitudes, which is below where twilight haze hides things
TITLED_GROUND = (6, 9)


def title_font(size: int, weight: str = "Bold") -> ImageFont.FreeTypeFont:
    """The bundled Inter at `size` px; `weight` is a named instance such as
    "Bold" or "Medium"."""
    font = ImageFont.truetype(str(FONT_PATH), size)
    font.set_variation_by_name(weight)
    return font


def footnote_text(when: datetime, lat: float, lon: float) -> str:
    """e.g. "Fri Oct 2, 2026 · 10:00 PM EDT · 35.78°N 78.64°W"."""
    lat_text = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
    lon_text = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
    return (
        f"{when:%a %b} {when.day}, {when:%Y} · "
        f"{when.hour % 12 or 12}:{when:%M %p %Z} · {lat_text} {lon_text}"
    )


def add_headroom(im: Image.Image, height: int) -> Image.Image:
    """`im` with `height` px of sky added on top and as much on the sides as
    keeps its aspect ratio. Each added row continues its row's edge color
    (the median over EDGE_SAMPLE px, so a star at the edge doesn't streak),
    which carries on the sky gradient and the hills."""
    pixels = np.asarray(im)
    h, w, _ = pixels.shape
    side = round(height * w / h / 2)
    # Starplot frames the plot with a thin dark border, which would otherwise
    # be left as a line across the sky
    border = 0
    while border < 10 and pixels[border].sum() < 0.8 * pixels[border + 10].sum():
        border += 1
    pixels = pixels[border:]
    height += border
    left = np.median(pixels[:, :EDGE_SAMPLE], axis=1).astype(np.uint8)
    right = np.median(pixels[:, -EDGE_SAMPLE:], axis=1).astype(np.uint8)
    body = np.concatenate(
        [np.repeat(left[:, None], side, axis=1), pixels, np.repeat(right[:, None], side, axis=1)],
        axis=1,
    )
    top_color = np.median(body[:EDGE_SAMPLE].reshape(-1, 3), axis=0).astype(np.uint8)
    top = np.broadcast_to(top_color, (height, body.shape[1], 3))
    return Image.fromarray(np.concatenate([top, body]))


def annotate(
    path: Path,
    footnote: str | None = None,
    title: str | None = None,
    altitude: tuple[float, float] | None = None,
    heading: str | None = None,
) -> None:
    """Draw a small grey footnote in the lower right corner and, if given, a
    white title centered on the landscape (which needs the view's
    `altitude` range to find it) and a white heading centered at the top,
    on sky added above the chart (see add_headroom)."""
    if footnote is None and title is None and heading is None:
        return
    with Image.open(path) as im:
        im = im.convert("RGB")
        if title is not None:
            # From the bottom, which stays put when headroom is added
            ground_height = im.height * TITLED_GROUND[0] / (altitude[1] - altitude[0])
        if heading is not None:
            im = add_headroom(im, round(im.height * HEADROOM))
        w, h = im.size
        draw = ImageDraw.Draw(im)

        if heading is not None:
            # Outlined in the sky's color so it reads over constellation lines
            draw.text(
                (w / 2, h * HEADING_MARGIN),
                heading,
                font=title_font(round(h * TITLE_SIZE)),
                anchor="mt",
                fill="#ffffff",
                stroke_width=round(h * 0.004),
                stroke_fill="#0b1026",
            )

        margin = round(h * FOOTNOTE_MARGIN)
        footnote_left = w - margin
        if footnote is not None:
            footnote_font = title_font(round(h * FOOTNOTE_SIZE), "Medium")
            footnote_left -= draw.textlength(footnote, font=footnote_font)
            draw.text(
                (w - margin, h - margin),
                footnote,
                font=footnote_font,
                anchor="rd",
                fill=FOOTNOTE_COLOR,
            )
        if title is None:
            im.save(path)
            return

        # Center the title in the band below the lowest point of the hills,
        # shrinking it if it would run into the footnote
        ground_top = h - ground_height
        size = round(h * TITLE_SIZE)
        while size > 10:
            font = title_font(size)
            if w / 2 + draw.textlength(title, font=font) / 2 < footnote_left - margin:
                break
            size -= 2
        draw.text(
            (w / 2, (ground_top + h) / 2),
            title,
            font=font,
            anchor="mm",
            fill="#ffffff",
        )
        im.save(path)


def output_filename(targets: list[str], dt: datetime, lat: float, lon: float) -> str:
    """e.g. saturn_20261002T2200_35.78N_78.64W.png (local time, ISO 8601 basic)."""
    names = "_".join(re.sub(r"[^a-z0-9]+", "-", t.strip().lower()).strip("-") for t in targets)
    return f"{names}_{time_and_place(dt, lat, lon)}.png"


def twilight_filename(dt: datetime, lat: float, lon: float, period: str) -> str:
    """e.g. 20261002T0600_35.78N_78.64W_mor.png; period is "morning" or "evening"."""
    return f"{time_and_place(dt, lat, lon)}_{period[:3]}.png"


def time_and_place(dt: datetime, lat: float, lon: float) -> str:
    """Local time (ISO 8601 basic, to the minute) and coordinates with N/S/E/W."""
    lat_part = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
    lon_part = f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
    return f"{dt:%Y%m%dT%H%M}_{lat_part}_{lon_part}"


def register_obstacle_line(p: HorizonPlot, line: list[Star], spacing: float = 12) -> None:
    """Make labels avoid a line drawn with p.line(). Starplot only indexes its
    own constellation lines (which labels may cross anyway), so points along
    the line go into the marker index that labels do avoid. Uses starplot
    internals: _prepare_coords, canvas._to_display and _markers_rtree."""
    display = [p.canvas._to_display(*p._prepare_coords(s.ra, s.dec)) for s in line]
    radius = 6 * p.scale
    for (x1, y1), (x2, y2) in zip(display, display[1:]):
        steps = max(1, int(math.hypot(x2 - x1, y2 - y1) / spacing))
        for i in range(steps + 1):
            x = x1 + (x2 - x1) * i / steps
            y = y1 + (y2 - y1) * i / steps
            p._markers_rtree.insert(0, (x - radius, y - radius, x + radius, y + radius), None)


def place_beside(
    text: str,
    font_size: float,
    preferred: str,
    stars: list[Star],
    sky: Sky,
    altitude: tuple[float, float],
    azimuth: tuple[float, float],
) -> list[tuple[str, Star]]:
    """Sides for a label outside a group of stars where it fits in the frame
    and above the ground, in order: the preferred side, then right, left,
    below, above. Each is the anchor and the outermost star on that side; if
    none fit, just the preferred side. The label's extent is estimated in
    degrees."""
    center_az = (azimuth[0] + azimuth[1]) / 2
    px_per_degree = PLOT_RESOLUTION / (azimuth[1] - azimuth[0])
    lines = text.split("\n")
    width = CHAR_WIDTH * font_size * max(map(len, lines)) / px_per_degree
    height = LINE_HEIGHT * font_size * len(lines) / px_per_degree
    gap = LABEL_GAP / px_per_degree

    # (x, y) = (azimuth offset from the view's center, altitude)
    points = {}
    for s in stars:
        alt, az = sky.altaz(s.ra, s.dec)
        points[s.hip] = (((az - center_az + 180) % 360) - 180, alt)
    half_span = (azimuth[1] - azimuth[0]) / 2

    def fits(x0, x1, y0, y1) -> bool:
        return (
            -half_span + 1 < x0
            and x1 < half_span - 1
            and LABEL_MIN_ALTITUDE < y0
            and y1 < altitude[1] - 1
        )

    order = [preferred] + [a for a in SIDE_ANCHORS if a != preferred]
    sides = []
    for anchor in order:
        if anchor == "right_center":
            star = max(stars, key=lambda s: points[s.hip][0])
            x, y = points[star.hip]
            # Azimuth degrees spread out with altitude in this projection
            w = width / max(0.2, math.cos(math.radians(y)))
            box = (x + gap, x + gap + w, y - height / 2, y + height / 2)
        elif anchor == "left_center":
            star = min(stars, key=lambda s: points[s.hip][0])
            x, y = points[star.hip]
            w = width / max(0.2, math.cos(math.radians(y)))
            box = (x - gap - w, x - gap, y - height / 2, y + height / 2)
        # Starplot's top/bottom "center" anchors start the text at the point
        # rather than centering it
        elif anchor == "bottom_center":
            star = min(stars, key=lambda s: points[s.hip][1])
            x, y = points[star.hip]
            w = width / max(0.2, math.cos(math.radians(y)))
            box = (x, x + w, y - gap - height, y - gap)
        else:  # top_center
            star = max(stars, key=lambda s: points[s.hip][1])
            x, y = points[star.hip]
            w = width / max(0.2, math.cos(math.radians(y)))
            box = (x, x + w, y + gap, y + gap + height)
        if fits(*box):
            sides.append((anchor, star))
    if sides:
        return sides

    # Nothing fits cleanly; fall back to the preferred side
    anchor = preferred
    key = {
        "right_center": lambda s: points[s.hip][0],
        "left_center": lambda s: -points[s.hip][0],
        "bottom_center": lambda s: -points[s.hip][1],
        "top_center": lambda s: points[s.hip][1],
    }[anchor]
    return [(anchor, max(stars, key=key))]


def choose_star_labels(
    sky: Sky,
    altitude: tuple[float, float],
    azimuth: tuple[float, float],
    drawn: Callable[[Star], bool],
    min_altitude: float = 0,
) -> list[int]:
    """Hipparcos IDs of the stars to label: up to MAX_STAR_LABELS well-known
    stars that are drawn and in view, in STAR_LABEL_PRIORITY order, then other
    LABELED_STARS by brightness. Stars below `min_altitude` (behind the drawn
    landscape) are skipped."""
    center = (azimuth[0] + azimuth[1]) / 2
    half_span = (azimuth[1] - azimuth[0]) / 2 - EDGE_MARGIN
    candidates = []
    for star in Star.find(where=[_.name.isin(STAR_LABEL_PRIORITY + LABELED_STARS)]):
        alt, az = sky.altaz(star.ra, star.dec)
        offset = ((az - center + 180) % 360) - 180
        if (
            min_altitude < alt < altitude[1] - EDGE_MARGIN
            and abs(offset) < half_span
            and drawn(star)
        ):
            candidates.append(star)

    def rank(star: Star) -> tuple:
        if star.name in STAR_LABEL_PRIORITY:
            return (0, STAR_LABEL_PRIORITY.index(star.name))
        return (1, star.magnitude)

    return [s.hip for s in sorted(candidates, key=rank)[:MAX_STAR_LABELS]]


def target_anchors(
    target: Target,
    sky: Sky,
    azimuth: tuple[float, float],
    display: dict[Target, tuple[float, float]],
) -> list[str]:
    """Label positions for `target`, tried in order. Normally toward the
    middle of the view, so the label isn't cut off at an edge. With another
    target within NEIGHBOR_DISTANCE px, only away from the nearest one: the
    left one of a close pair on its left and the right one on its right, and
    if that side is taken (by an asterism's label, say), the upper one above
    and the lower one below. The two labels then never compete for the same
    space, whichever is placed first."""
    center_az = (azimuth[0] + azimuth[1]) / 2
    _alt, az = sky.altaz(target.ra, target.dec)
    az_offset = ((az - center_az + 180) % 360) - 180
    anchors = RIGHT_ANCHORS + LEFT_ANCHORS
    if az_offset > (azimuth[1] - azimuth[0]) / 6:  # outer third on the right
        anchors = LEFT_ANCHORS + RIGHT_ANCHORS

    x, y = display[target]
    others = [xy for t, xy in display.items() if t != target]
    if not others:
        return anchors
    ox, oy = min(others, key=lambda o: math.hypot(o[0] - x, o[1] - y))
    dx, dy = ox - x, oy - y  # display y grows downward
    if math.hypot(dx, dy) > NEIGHBOR_DISTANCE:
        return anchors
    sideways = LEFT_ANCHORS if dx > 0 else RIGHT_ANCHORS
    vertical = ABOVE_ANCHORS if dy > 0 else BELOW_ANCHORS
    if abs(dx) >= abs(dy):
        return sideways + [a for a in vertical if a not in sideways]
    toward = "bottom" if dy > 0 else "top"
    return vertical + [a for a in sideways if not a.startswith(toward)]


def label_targets(
    p: HorizonPlot,
    sky: Sky,
    targets: list[Target],
    sizes: dict[Target, float],
    azimuth: tuple[float, float],
) -> None:
    """Label each target beside its (already drawn) marker. Each label tries
    its sides from target_anchors() without landing on another label or
    another target's marker, then the same sides farther out, and only then
    is drawn on its first side regardless. Faint stars and constellation
    lines may be covered: that beats jumping somewhere less obvious. Uses
    starplot internals: _prepare_coords, canvas._to_display, _labels_rtree
    and _offset_from_marker. The markers stay in the label index afterward,
    so the constellation labels drawn later keep off them too."""
    shown = [t for t in targets if p.in_bounds(t.ra, t.dec)]
    display = {t: p.canvas._to_display(*p._prepare_coords(t.ra, t.dec)) for t in shown}
    # The other targets' markers count as labels while each label is placed
    # (the marker index can't be used: labels are allowed over faint stars and
    # asterism lines, which are in it too)
    boxes = {}
    for i, t in enumerate(shown):
        x, y = display[t]
        r = sizes[t] * p.scale / 2
        boxes[t] = (i, (x - r, y - r, x + r, y + r))
        p._labels_rtree.insert(i, boxes[t][1])

    for target in shown:
        color = BODY_COLORS.get(target.label, TARGET_COLORS[target.kind])
        anchors = target_anchors(target, sky, azimuth, display)
        p._labels_rtree.delete(*boxes[target])
        before = p._labels_rtree.get_size()
        # Each side at the usual gap, then each farther out, then the first
        # side at the usual gap, forced
        attempts = [(a, gap) for gap in TARGET_LABEL_GAPS for a in anchors]
        attempts.append((anchors[0], TARGET_LABEL_GAPS[0]))
        for i, (anchor, gap) in enumerate(attempts):
            last = i == len(attempts) - 1
            # Starplot's automatic vertical offset centers the label on the
            # marker's top or bottom edge, which for text much taller than
            # the marker is hardly above or below it at all; above and below
            # instead clear the marker by the gap
            vertical = anchor not in ("left_center", "right_center")
            style = LabelStyle(
                font_size=target.font_size,
                font_weight=700,
                fill=color,
                anchor_point=anchor,
                opacity=FAINT_OPACITY if target.faint else 1.0,
                offset_x=gap,
                offset_y=sizes[target] / 2 + VERTICAL_LABEL_GAP if vertical else "auto",
                zorder=1000,
            )
            p.text(
                target.label,
                target.ra,
                target.dec,
                style=p._offset_from_marker(style, target.label, sizes[target]),
                collision_handler=CollisionHandler(
                    allow_constellation_line_collisions=True,
                    allow_marker_collisions=True,
                    anchor_fallbacks=[anchor],
                    plot_on_fail=last,
                ),
            )
            if p._labels_rtree.get_size() > before:
                break
        p._labels_rtree.insert(*boxes[target])


def render(
    observer: Observer,
    sky: Sky,
    altitude: tuple[float, float],
    azimuth: tuple[float, float],
    targets: list[Target],
    asterisms: list[tuple[Asterism, list[list[Star]]]],
    constellations: list[str],
    star_where: list,
    path: Path,
    ground: tuple[float, float] = GROUND,
    star_magnitude: float | None = None,
    labels: bool = True,
) -> Path:
    """Draw a horizon view and export it to `path` as a 16:9 PNG. `star_where`
    selects the background stars (`star_magnitude` is its magnitude limit, if
    it has one, used to decide which stars can be labeled); stars in the
    drawn constellations and asterisms, and the brightest stars in view, are
    always drawn. `labels=False` leaves off every name: targets, stars,
    asterisms and constellations."""
    az_scale = (azimuth[1] - azimuth[0]) / REFERENCE_AZ_SPAN
    asterism_stars = [s for _asterism, lines in asterisms for line in lines for s in line]

    style = PlotStyle().extend(
        extensions.BLUE_GOLD,
        extensions.HORIZON,
        extensions.GRADIENT_NAUTICAL_TWILIGHT,
    )
    style.figure.padding = 0
    style.constellation_lines.width = 5
    style.constellation_lines.opacity = 0.6
    style.constellation_labels.font_size = CONSTELLATION_FONT_SIZE
    style.star.marker.stroke_width = 0

    p = HorizonPlot(
        altitude=altitude,
        azimuth=azimuth,
        observer=observer,
        style=style,
        scale=1.2,
    )
    p.ground(
        min_altitude=ground[0],
        max_altitude=ground[1],
        style__fill={"stops": GROUND_STOPS, "type": "linear"},
    )

    p.constellations(where=[_.iau_id.isin(constellations)])
    for _asterism, lines in asterisms:
        for line in lines:
            p.line(
                coordinates=[(s.ra, s.dec) for s in line],
                style={
                    "line": {
                        "width": 10,
                        "stroke": ASTERISM_COLOR,
                        "opacity": 0.9,
                        "cap_style": "round",
                    }
                },
            )
            register_obstacle_line(p, line)

    # Labels inside an asterism first, then those beside one, which can then
    # avoid the others
    labeled = [(a, lines) for a, lines in asterisms if labels and not same_as_constellation(a)]
    labeled.sort(key=lambda al: al[0].label_style.get("anchor_point", "center") != "center")
    for asterism, lines in labeled:
        label = {
            "font_size": ASTERISM_FONT_SIZE,
            "fill": ASTERISM_COLOR,
            "anchor_point": "center",
            **asterism.label_style,
        }
        stars = list({s.hip: s for l in lines for s in l}.values())
        if label["anchor_point"] == "center":
            # Shrinks with the asterism when the view is wider than the
            # reference, so it keeps fitting inside the outline
            fit = min(1.0, 1 / az_scale)
            label["font_size"] = ASTERISM_FONT_SIZE * fit
            center_ra, center_dec = center_of(stars)
            label_lines = asterism.name.split("\n")
            for i, text in enumerate(label_lines):
                d_alt = ((len(label_lines) - 1) / 2 - i) * LABEL_LINE_SPACING
                d_alt = (d_alt + LABEL_BASELINE_SHIFT) * az_scale * fit
                p.text(
                    text,
                    *sky.shift_altitude(center_ra, center_dec, d_alt),
                    style=label,
                    collision_handler=CollisionHandler(
                        anchor_fallbacks=["center"], plot_on_fail=True
                    ),
                )
            continue

        # Beside the asterism: the first side that fits in the frame and
        # doesn't land on another label, else the first side that fits
        sides = place_beside(
            asterism.name, label["font_size"], label["anchor_point"],
            stars, sky, altitude, azimuth,
        )
        for attempt, (anchor, edge) in enumerate(sides):
            last = attempt == len(sides) - 1
            if last:
                anchor, edge = sides[0]
            labels_before = p._labels_rtree.get_size()  # starplot internal
            p.text(
                asterism.name,
                edge.ra,
                edge.dec,
                style=label | {"anchor_point": anchor, "offset_x": LABEL_GAP, "offset_y": LABEL_GAP},
                collision_handler=CollisionHandler(
                    allow_marker_collisions=True,
                    allow_constellation_line_collisions=True,
                    anchor_fallbacks=[anchor],
                    plot_on_fail=last,
                ),
            )
            if p._labels_rtree.get_size() > labels_before:
                break

    # Stars that make up the drawn constellations and asterisms are drawn
    # whatever `star_where` says, so the figures connect
    figure_hips = {s.hip for s in asterism_stars}
    for c in all_constellations():
        if c.iau_id in constellations:
            figure_hips.update(c.star_hip_ids)

    def drawn(star: Star) -> bool:
        return (
            star.hip in figure_hips
            or star.magnitude < BRIGHT_STAR_MAGNITUDE
            or (star_magnitude is not None and star.magnitude <= star_magnitude)
        )

    label_hips = (
        choose_star_labels(sky, altitude, azimuth, drawn, min_altitude=ground[1])
        if labels
        else []
    )
    star_labels = [_.hip.isin(label_hips)] if label_hips else False
    p.stars(
        where=star_where,
        where_labels=star_labels,
        style__label__font_size=BRIGHT_STAR_FONT_SIZE,
        color_fn=callables.color_by_bv_gradient,
    )
    if figure_hips:
        p.stars(
            where=[_.hip.isin(list(figure_hips))],
            where_labels=star_labels,
            style__label__font_size=BRIGHT_STAR_FONT_SIZE,
            color_fn=callables.color_by_bv_gradient,
            gid_markers="figure-stars",
            gid_labels="figure-stars-labels",
        )
    # Brightest stars anywhere in view, for orientation
    p.stars(
        where=[_.magnitude < BRIGHT_STAR_MAGNITUDE],
        where_labels=star_labels,
        style__label__font_size=BRIGHT_STAR_FONT_SIZE,
        color_fn=callables.color_by_bv_gradient,
        gid_markers="bright-stars",
        gid_labels="bright-stars-labels",
    )

    # Every target's marker goes down before any target's label, so a label
    # can steer clear of the other targets (a Moon-planet or planet-planet
    # conjunction puts two markers almost on top of each other)
    sizes = {}
    for target in targets:
        color = BODY_COLORS.get(target.label, TARGET_COLORS[target.kind])
        hollow = target.kind in HOLLOW_KINDS
        sizes[target] = 60 if hollow else FAINT_MARKER_SIZE if target.faint else 40
        p.marker(
            ra=target.ra,
            dec=target.dec,
            style={
                "marker": {
                    "symbol": "circle",
                    "size": sizes[target],
                    "fill": None if hollow else color,
                    "stroke": color if hollow else "#ffffff",
                    "stroke_width": 4 if hollow else 2,
                    "opacity": FAINT_OPACITY if target.faint else 1.0,
                    "zorder": 1000,
                },
            },
        )
    if labels:
        label_targets(p, sky, targets, sizes, azimuth)

    # Blank labels are skipped, so the overridden ones get placed by hand
    if labels:
        p.constellation_labels(
            label_fn=lambda c: (
                "" if c.iau_id in CONSTELLATION_LABEL_POSITIONS else Constellation.get_label(c)
            )
        )
    for iau_id in constellations if labels else []:
        if iau_id in CONSTELLATION_LABEL_POSITIONS:
            ra, dec = CONSTELLATION_LABEL_POSITIONS[iau_id]
            p.text(
                Constellation.get_label(Constellation.get(iau_id=iau_id)),
                ra=ra,
                dec=dec,
                style=style.constellation_labels.model_dump(),
                collision_handler=CollisionHandler(
                    anchor_fallbacks=["center"], plot_on_fail=True
                ),
            )

    p.export(str(path))
    crop_to_aspect(path)
    return path


def plot_targets(
    lat: float,
    lon: float,
    targets: list[str | Target],
    when: datetime,
    output_dir: str | Path = ".",
    timestamp: bool = True,
    labels: bool = True,
    heading: str | None = None,
    caption: str | None = None,
    filename: str | None = None,
) -> Path:
    """Render a horizon view of `targets` as seen from `lat`/`lon` at `when`
    (a timezone-aware local datetime). Targets are names (see
    resolve_target) or Target objects, for points no catalog names.
    `timestamp` adds the date, time and place in the lower right corner;
    `labels=False` leaves off every name. `heading` is drawn centered at the
    top and `caption` centered on a taller landscape at the bottom.
    `filename` replaces the default name of the PNG. Returns its path."""
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware")
    if not targets:
        raise ValueError("At least one target is required")

    observer = Observer(lat=lat, lon=lon, dt=when)
    sky = Sky(lat, lon, when)
    resolved = [t if isinstance(t, Target) else resolve_target(t, observer) for t in targets]
    for target in resolved:
        alt, _az = sky.altaz(target.ra, target.dec)
        if alt < 0:
            raise ValueError(f"{target.label} is below the horizon ({alt:.1f}°)")

    # Only highlight asterisms that are entirely above the horizon
    asterisms = [
        (asterism, lines)
        for asterism, lines in select_asterisms(resolved)
        if all(sky.altaz(s.ra, s.dec)[0] > 0 for line in lines for s in line)
    ]
    asterism_stars = [s for _asterism, lines in asterisms for line in lines for s in line]
    constellations = select_constellations(resolved, [a for a, _lines in asterisms])

    # Frame the targets and their asterisms, plus the visible parts of the
    # shown constellations when that still fits in one view
    points = [sky.altaz(t.ra, t.dec) for t in resolved]
    points += [sky.altaz(s.ra, s.dec) for s in asterism_stars]
    constellation_points = [
        pt
        for c in all_constellations()
        if c.iau_id in constellations
        for s in Star.find(where=[_.hip.isin(c.star_hip_ids)])
        if (pt := sky.altaz(s.ra, s.dec))[0] > 0
    ]
    try:
        altitude, azimuth = frame(points + constellation_points)
    except ValueError:
        altitude, azimuth = frame(points)
    constellations = sorted(set(constellations) | set(landmarks_in_view(sky, altitude, azimuth)))

    # Canonical names, so "saturn barycenter" still files as "saturn"
    names = [t.label for t in resolved]
    path = render(
        observer,
        sky,
        altitude,
        azimuth,
        resolved,
        asterisms,
        constellations,
        star_where=[_.magnitude < 4.5, _.constellation_id.isin(constellations)],
        path=Path(output_dir) / (filename or output_filename(names, when, lat, lon)),
        ground=TITLED_GROUND if caption else GROUND,
        labels=labels,
    )
    annotate(
        path,
        footnote_text(when, lat, lon) if timestamp else None,
        caption,
        altitude,
        heading,
    )
    return path


def twilight_times(
    lat: float, lon: float, day: date, tz: str | None = None, hours: int = 1
) -> tuple[datetime, datetime]:
    """Local times `hours` before sunrise and after sunset on `day`, rounded
    to the nearest half hour. The time zone is looked up from the coordinates
    unless `tz` is given."""
    sunrise, sunset = sun_events(lat, lon, day, tz)
    return (
        round_to_half_hour(sunrise - timedelta(hours=hours)),
        round_to_half_hour(sunset + timedelta(hours=hours)),
    )


def sun_events(
    lat: float, lon: float, day: date, tz: str | None = None
) -> tuple[datetime, datetime]:
    """Local sunrise and sunset on `day`."""
    zone = ZoneInfo(tz or TimezoneFinder().timezone_at(lat=lat, lng=lon))
    ts = load.timescale()
    eph = ephemeris()
    start = datetime.combine(day, time(0), tzinfo=zone)
    times, is_up = almanac.find_discrete(
        ts.from_datetime(start),
        ts.from_datetime(start + timedelta(days=1)),
        almanac.sunrise_sunset(eph, wgs84.latlon(lat, lon)),
    )
    events = [(t.astimezone(zone), bool(up)) for t, up in zip(times, is_up)]
    sunrise = next((t for t, up in events if up), None)
    sunset = next((t for t, up in events if not up), None)
    if sunrise is None or sunset is None:
        raise ValueError(f"No sunrise and sunset at {lat}, {lon} on {day}")
    return sunrise, sunset


def round_to_half_hour(dt: datetime) -> datetime:
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes = (dt - midnight).total_seconds() / 60
    # floor(x + 0.5) so exact quarter-hours round up (round() would round
    # half to even)
    return midnight + timedelta(minutes=math.floor(minutes / 30 + 0.5) * 30)


def visible_dsos(limiting_magnitude: float = LIMITING_MAGNITUDE) -> list[Target]:
    """Deep-sky objects (clusters, galaxies, nebulae) at least as bright as
    `limiting_magnitude`."""
    found = DSO.find(
        where=[
            ~_.type.isin(DSO_STAR_TYPES),
            _.magnitude <= limiting_magnitude,
        ]
    )
    return [Target(dso_label(d), "dso", d.ra, d.dec, DSO_FONT_SIZE) for d in found]


def dso_label(dso: DSO) -> str:
    """Common name, else Messier number, else catalog ID ("NGC6231" -> "NGC 6231")."""
    if dso.common_names:
        return dso.common_names[0].upper()
    if dso.m:
        return f"M{dso.m}"
    return re.sub(r"^([A-Za-z]+)0*(\d+)$", r"\1 \2", dso.name).upper()


@dataclass
class Scene:
    """What a fixed-direction view contains."""

    observer: Observer
    sky: Sky
    altitude: tuple[float, float]
    azimuth: tuple[float, float]
    targets: list[Target]
    asterisms: list[tuple[Asterism, list[list[Star]]]]
    constellations: list[str]


def twilight_scene(
    lat: float,
    lon: float,
    when: datetime,
    center_az: float,
    limiting_magnitude: float = LIMITING_MAGNITUDE,
) -> Scene:
    """The Moon, planets, deep-sky objects to `limiting_magnitude`, and
    asterisms (with every constellation their stars belong to) in a view
    centered on `center_az`."""
    observer = Observer(lat=lat, lon=lon, dt=when)
    sky = Sky(lat, lon, when)
    half_span = TWILIGHT_ALTITUDE * AZ_PER_ALT / 2

    def in_view(ra: float, dec: float, min_alt: float = 0, margin: float = 0) -> bool:
        alt, az = sky.altaz(ra, dec)
        offset = ((az - center_az + 180) % 360) - 180
        return min_alt < alt < TWILIGHT_ALTITUDE - margin and abs(offset) < half_span - margin

    targets = [
        t
        for name in TWILIGHT_BODIES
        # Above the drawn landscape, which would otherwise hide them
        if in_view(
            (t := resolve_target(name, observer)).ra,
            t.dec,
            min_alt=TITLED_GROUND[1],
            margin=EDGE_MARGIN,
        )
    ]
    targets += [
        d
        for d in visible_dsos(limiting_magnitude)
        if in_view(d.ra, d.dec, min_alt=DSO_MIN_ALTITUDE, margin=EDGE_MARGIN)
    ]

    asterisms = []
    for asterism in ASTERISMS:
        lines = [[Star.get(hip=hip) for hip in line] for line in asterism.lines]
        if all(in_view(s.ra, s.dec) for line in lines for s in line):
            asterisms.append((asterism, lines))
    # Every constellation an asterism's stars belong to, so the Triangles
    # bring in all three of theirs
    constellations = sorted(
        {
            c
            for _asterism, lines in asterisms
            for line in lines
            for s in line
            if (c := constellation_of(s.ra, s.dec))
        }
    )
    altitude = (0, TWILIGHT_ALTITUDE)
    azimuth = (center_az - half_span, center_az + half_span)
    constellations = sorted(set(constellations) | set(landmarks_in_view(sky, altitude, azimuth)))
    return Scene(
        observer,
        sky,
        altitude,
        azimuth,
        targets,
        asterisms,
        constellations,
    )


# Planets or the Moon lower than this at the usual time (1 hour from the Sun)
# make it worth also trying one hour further from the Sun
LOW_ALTITUDE = 15

# How much each kind of object adds to a view's interest when choosing
# between looking east or west, and between times
INTEREST = {
    "moon": 3,
    "planet": 3,
    "faint planet": 1,
    "dso": 2,
    "asterism": 2,
    "bright star": 1,  # brighter than BRIGHT_STAR_LABEL_MAGNITUDE
}


def interest(scene: Scene) -> float:
    """Score a view by the interesting objects in it. Targets low in the sky,
    where haze and twilight wash them out, count half."""
    score = 0.0
    for t in scene.targets:
        weight = INTEREST["faint planet" if t.faint else t.kind]
        alt, _az = scene.sky.altaz(t.ra, t.dec)
        score += weight / 2 if alt < LOW_ALTITUDE else weight
    score += INTEREST["asterism"] * len(scene.asterisms)
    center = (scene.azimuth[0] + scene.azimuth[1]) / 2
    half_span = (scene.azimuth[1] - scene.azimuth[0]) / 2
    for star in Star.find(where=[_.magnitude < BRIGHT_STAR_LABEL_MAGNITUDE]):
        alt, az = scene.sky.altaz(star.ra, star.dec)
        offset = ((az - center + 180) % 360) - 180
        if 0 < alt < scene.altitude[1] and abs(offset) < half_span:
            score += INTEREST["bright star"]
    return score


def choose_twilight_view(
    lat: float,
    lon: float,
    when: datetime,
    default: str,
    limiting_magnitude: float = LIMITING_MAGNITUDE,
) -> tuple[str, Scene]:
    """Look east or west, whichever has more of interest; ties go to `default`."""
    scenes = {
        direction: twilight_scene(lat, lon, when, center, limiting_magnitude)
        for direction, center in (("east", 90), ("west", 270))
    }
    other = "west" if default == "east" else "east"
    if interest(scenes[other]) > interest(scenes[default]):
        return other, scenes[other]
    return default, scenes[default]


def has_low_bodies(lat: float, lon: float, when: datetime) -> bool:
    """Whether a naked-eye planet or the Moon is near (or just below) the
    horizon, where a different time would show it better."""
    observer = Observer(lat=lat, lon=lon, dt=when)
    sky = Sky(lat, lon, when)
    for name in TWILIGHT_BODIES:
        target = resolve_target(name, observer)
        if not target.faint and -5 < sky.altaz(target.ra, target.dec)[0] < LOW_ALTITUDE:
            return True
    return False


def choose_twilight_moment(
    lat: float,
    lon: float,
    day: date,
    period: str,
    tz: str | None = None,
    limiting_magnitude: float = LIMITING_MAGNITUDE,
) -> tuple[int, datetime, str, Scene]:
    """For `period` "morning" or "evening": the usual time 1 hour from the
    Sun, or 2 hours if planets or the Moon are low then and the later view is
    more interesting; and east or west, whichever is more interesting.
    Returns (hours from the Sun, time, direction, scene)."""
    index, default = (0, "east") if period == "morning" else (1, "west")
    candidates = [1]
    if has_low_bodies(lat, lon, twilight_times(lat, lon, day, tz)[index]):
        candidates.append(2)

    best = None
    for hours in candidates:
        when = twilight_times(lat, lon, day, tz, hours)[index]
        direction, scene = choose_twilight_view(lat, lon, when, default, limiting_magnitude)
        score = interest(scene)
        if best is None or score > best[0]:
            best = (score, hours, when, direction, scene)
    return best[1:]


def plot_twilight_views(
    lat: float,
    lon: float,
    day: date,
    tz: str | None = None,
    output_dir: str | Path = ".",
    limiting_magnitude: float = LIMITING_MAGNITUDE,
    title: bool = True,
    timestamp: bool = True,
    labels: bool = True,
) -> dict[str, Path]:
    """Render the sky before sunrise and after sunset on `day`: usually 1 hour
    from the Sun, or 2 if planets or the Moon are low and that shows more;
    looking east or west, whichever has more of interest (ties go to east in
    the morning and west in the evening). Shows the Moon and planets, stars
    and deep-sky objects at least as bright as `limiting_magnitude`, and
    asterisms with their constellations. `title` names the direction and time
    on the landscape, `timestamp` adds the date, time and place in the lower
    right corner, and `labels=False` leaves off every name. Returns
    {"morning": path, "evening": path}."""
    paths = {}
    for period, event in (("morning", "before sunrise"), ("evening", "after sunset")):
        hours, when, direction, scene = choose_twilight_moment(
            lat, lon, day, period, tz, limiting_magnitude
        )
        paths[period] = render(
            scene.observer,
            scene.sky,
            scene.altitude,
            scene.azimuth,
            scene.targets,
            scene.asterisms,
            scene.constellations,
            star_where=[_.magnitude <= limiting_magnitude],
            path=Path(output_dir) / twilight_filename(when, lat, lon, period),
            ground=TITLED_GROUND,
            star_magnitude=limiting_magnitude,
            labels=labels,
        )
        annotate(
            paths[period],
            footnote_text(when, lat, lon) if timestamp else None,
            f"Looking {direction}, {hours} hour{'s' if hours > 1 else ''} {event}"
            if title
            else None,
            scene.altitude,
        )
    return paths
