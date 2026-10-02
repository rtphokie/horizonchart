"""Command line interface: horizonchart target ... / horizonchart twilight ..."""

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from horizonchart import DATA_PATH, __version__
from horizonchart.skyview import LIMITING_MAGNITUDE, plot_targets, plot_twilight_views

GEOCODE_CACHE = DATA_PATH / "horizonchart-geocode.json"
# Nominatim's usage policy asks applications to identify themselves, ideally
# with a contact address (set HORIZONCHART_CONTACT, e.g. to an email address),
# and to make at most one request per second
GEOCODER_URL = "https://github.com/rtphokie/horizonchart"
GEOCODER_MIN_DELAY = 1.0

# "35.78,-78.64", "35.78 -78.64" or "35.78N 78.64W" (comma optional)
SIGNED_COORDINATES = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$")
HEMISPHERE_COORDINATES = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*°?\s*([NS])\s*[,\s]\s*(\d+(?:\.\d+)?)\s*°?\s*([EW])\s*$",
    re.IGNORECASE,
)


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Latitude and longitude from "35.78,-78.64" or "35.78N 78.64W", else None."""
    if match := SIGNED_COORDINATES.match(text):
        lat, lon = float(match[1]), float(match[2])
    elif match := HEMISPHERE_COORDINATES.match(text):
        lat = float(match[1]) * (-1 if match[2].upper() == "S" else 1)
        lon = float(match[3]) * (-1 if match[4].upper() == "W" else 1)
    else:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Coordinates out of range: {text!r}")
    return lat, lon


def geocode(place: str) -> tuple[float, float]:
    """Coordinates of a place name such as "Raleigh, NC" or "Ottawa, ON", looked
    up with OpenStreetMap's Nominatim service and cached on disk."""
    try:
        cache = json.loads(GEOCODE_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    key = place.strip().lower()
    if key not in cache:
        found = _rate_limited_geocode()(place, timeout=10)
        if found is None:
            raise ValueError(f"Couldn't find a location named {place!r}")
        cache[key] = [found.latitude, found.longitude]
        try:
            GEOCODE_CACHE.write_text(json.dumps(cache, indent=1))
        except OSError:
            pass  # the lookup still worked; it just isn't cached
    lat, lon = cache[key]
    return lat, lon


def geocoder_user_agent() -> str:
    """e.g. "horizonchart/0.1.0 (+https://github.com/rtphokie/horizonchart; you@example.com)"."""
    contact = os.environ.get("HORIZONCHART_CONTACT")
    details = f"+{GEOCODER_URL}" + (f"; {contact}" if contact else "")
    return f"horizonchart/{__version__} ({details})"


_geocoder = None


def _rate_limited_geocode():
    """Nominatim geocoding, at most one request per GEOCODER_MIN_DELAY seconds."""
    global _geocoder
    if _geocoder is None:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim

        nominatim = Nominatim(user_agent=geocoder_user_agent())
        _geocoder = RateLimiter(nominatim.geocode, min_delay_seconds=GEOCODER_MIN_DELAY)
    return _geocoder


def resolve_location(text: str) -> tuple[float, float]:
    """Coordinates from either a coordinate pair or a place name."""
    return parse_coordinates(text) or geocode(text)


def local_zone(lat: float, lon: float, tz: str | None) -> ZoneInfo:
    return ZoneInfo(tz or TimezoneFinder().timezone_at(lat=lat, lng=lon))


def parse_local_datetime(text: str | None, zone: ZoneInfo) -> datetime:
    """ "2026-10-02 22:00" (or "2026-10-02T22:00") as local time in `zone`;
    now if `text` is None."""
    if text is None:
        return datetime.now(zone).replace(second=0, microsecond=0)
    parsed = datetime.fromisoformat(text.strip())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horizonchart",
        description="Clean horizon-view sky charts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    location_help = (
        'where you are: coordinates ("35.78,-78.64" or "35.78N 78.64W") or a '
        'place name ("Raleigh, NC", "Ottawa, ON")'
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-l", "--location", required=True, help=location_help)
    common.add_argument(
        "--tz", help="time zone name (default: looked up from the location)"
    )
    common.add_argument(
        "-o", "--output-dir", default=".", type=Path, help="where to write the PNG"
    )

    target = commands.add_parser(
        "target",
        parents=[common],
        help="finder chart for one or more objects at a given time",
        description="Finder chart for planets, the Moon, stars or deep-sky objects.",
    )
    target.add_argument(
        "targets",
        nargs="+",
        metavar="TARGET",
        help='e.g. Saturn, Moon, "saturn barycenter", Pleiades, M31, "NGC 1976", Vega',
    )
    target.add_argument(
        "-t",
        "--time",
        help='local date and time, e.g. "2026-10-02 22:00" (default: now)',
    )

    twilight = commands.add_parser(
        "twilight",
        parents=[common],
        help="morning and evening twilight views for a date",
        description=(
            "Views before sunrise and after sunset, looking east or west, "
            "whichever shows more."
        ),
    )
    twilight.add_argument(
        "-d", "--date", type=date.fromisoformat, help="e.g. 2026-10-02 (default: today)"
    )
    twilight.add_argument(
        "-m",
        "--limiting-magnitude",
        type=float,
        default=LIMITING_MAGNITUDE,
        help=f"faintest stars and deep-sky objects to show (default: {LIMITING_MAGNITUDE}, "
        "suburban twilight; dark skies 4-6)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lat, lon = resolve_location(args.location)
        zone = local_zone(lat, lon, args.tz)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.command == "target":
            when = parse_local_datetime(args.time, zone)
            paths = [plot_targets(lat, lon, args.targets, when, args.output_dir)]
        else:
            day = args.date or datetime.now(zone).date()
            views = plot_twilight_views(
                lat,
                lon,
                day,
                tz=zone.key,
                output_dir=args.output_dir,
                limiting_magnitude=args.limiting_magnitude,
            )
            paths = list(views.values())
    except ValueError as error:
        print(f"horizonchart: {error}", file=sys.stderr)
        return 1

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
