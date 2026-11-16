"""Meteor shower radiant charts: where to look, and when.

`plot_meteor_shower` draws a 16:9 horizon view with the shower's radiant
marked where it's highest during the night, the shower's name across the top
and advice on where and when to look across the bottom. "Night" runs between
evening and morning civil twilight (the Sun 6° below the horizon): nothing is
charted or recommended in brighter skies.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from skyfield import almanac
from skyfield.api import Star as SkyfieldStar
from skyfield.api import wgs84
from timezonefinder import TimezoneFinder

from horizonchart.skyview import (
    Sky,
    Target,
    ephemeris,
    load,
    output_filename,
    plot_targets,
    round_to_half_hour,
    sun_events,
)


@dataclass(frozen=True)
class MeteorShower:
    name: str  # e.g. "Leonids"
    title: str  # e.g. "Leonid Meteor Shower"
    peak: tuple[int, int]  # (month, day) of the morning of maximum
    ra: float  # radiant at maximum, degrees
    dec: float
    zhr: int  # zenithal hourly rate at maximum


# Major showers best seen before dawn, from the International Meteor
# Organization's working list. Peaks shift by a day or so from year to year.
SHOWERS = [
    MeteorShower("Quadrantids", "Quadrantid Meteor Shower", (1, 4), 230, 49, 80),
    MeteorShower("Lyrids", "Lyrid Meteor Shower", (4, 22), 271, 34, 18),
    MeteorShower("Eta Aquariids", "Eta Aquariid Meteor Shower", (5, 6), 338, -1, 50),
    MeteorShower(
        "Southern Delta Aquariids",
        "Southern Delta Aquariid Meteor Shower",
        (7, 30),
        340,
        -16,
        25,
    ),
    MeteorShower("Perseids", "Perseid Meteor Shower", (8, 13), 48, 58, 100),
    MeteorShower("Orionids", "Orionid Meteor Shower", (10, 21), 95, 16, 20),
    MeteorShower("Leonids", "Leonid Meteor Shower", (11, 17), 152, 22, 15),
    MeteorShower("Geminids", "Geminid Meteor Shower", (12, 14), 112, 33, 150),
    MeteorShower("Ursids", "Ursid Meteor Shower", (12, 22), 217, 76, 10),
]

# Showers with fewer meteors than this need a moonless sky, so they're
# watched from moonset when the Moon sets during the night
WEAK_ZHR = 30
# Usually the time to look: the radiant is high and the sky still dark
HOURS_BEFORE_SUNRISE = 2
# The night is searched for the radiant's highest point at this interval
SEARCH_STEP = timedelta(minutes=5)
# Looking up this high is just looking up
OVERHEAD_ALTITUDE = 70
COMPASS = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]


def find_shower(name: str) -> MeteorShower:
    """A shower by name: "Leonids", "leonid", "Leonid meteor shower", "eta
    aquariids"..."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    key = re.sub(r" meteor shower$| meteors?$", "", key).removesuffix("s")
    for shower in SHOWERS:
        if shower.name.lower().removesuffix("s") == key:
            return shower
    known = ", ".join(s.name for s in SHOWERS)
    raise ValueError(f"Unknown meteor shower: {name!r} (known: {known})")


def peak_morning(shower: MeteorShower, today: date) -> date:
    """The next morning of maximum, counting `today`."""
    month, day = shower.peak
    peak = date(today.year, month, day)
    return peak if peak >= today else date(today.year + 1, month, day)


def clock_text(dt: datetime) -> str:
    """e.g. "12:30 am", "4:00 am" or "Midnight", to the nearest half hour."""
    dt = round_to_half_hour(dt)
    if dt.hour == 0 and dt.minute == 0:
        return "Midnight"
    return f"{dt.hour % 12 or 12}:{dt:%M} {'am' if dt.hour < 12 else 'pm'}"


def compass_direction(az: float) -> str:
    """The nearest of the eight compass points to an azimuth in degrees."""
    return COMPASS[round(az / 45) % 8]


def moonset(lat: float, lon: float, start: datetime, end: datetime) -> datetime | None:
    """The first moonset between `start` and `end`, in their time zone."""
    eph = ephemeris()
    ts = load.timescale()
    times, actually_set = almanac.find_settings(
        eph["earth"] + wgs84.latlon(lat, lon),
        eph["moon"],
        ts.from_datetime(start),
        ts.from_datetime(end),
    )
    for t, did_set in zip(times, actually_set):
        if did_set:
            return t.astimezone(start.tzinfo)
    return None


def night_hours(
    lat: float, lon: float, day: date, tz: str | None = None
) -> tuple[datetime, datetime]:
    """The night before the morning of `day`, from the end of evening civil
    twilight to the start of morning civil twilight. Where it never gets that
    dark (high latitudes in summer), from sunset to sunrise."""
    zone = ZoneInfo(tz or TimezoneFinder().timezone_at(lat=lat, lng=lon))
    ts = load.timescale()
    times, states = almanac.find_discrete(
        ts.from_datetime(datetime.combine(day - timedelta(days=1), time(12), tzinfo=zone)),
        ts.from_datetime(datetime.combine(day, time(12), tzinfo=zone)),
        almanac.dark_twilight_day(ephemeris(), wgs84.latlon(lat, lon)),
    )
    # States count up from 0 (dark) to 4 (day); 3 is civil twilight
    events = [(t.astimezone(zone), int(state)) for t, state in zip(times, states)]
    dusk = next((t for t, state in events if state == 2), None)
    dawn = next((t for t, state in events if dusk and t > dusk and state == 3), None)
    if dusk is None or dawn is None:
        _sunrise, sunset = sun_events(lat, lon, day - timedelta(days=1), zone.key)
        sunrise, _sunset = sun_events(lat, lon, day, zone.key)
        return sunset, sunrise
    return dusk, dawn


def round_within(dt: datetime, start: datetime, end: datetime) -> datetime:
    """`dt` to the nearest half hour, without leaving `start`..`end`."""
    rounded = round_to_half_hour(dt)
    if rounded > end:
        return rounded - timedelta(minutes=30)
    if rounded < start:
        return rounded + timedelta(minutes=30)
    return rounded


def highest_time(
    lat: float, lon: float, shower: MeteorShower, night: tuple[datetime, datetime]
) -> datetime | None:
    """When the radiant is highest during `night` (see night_hours), to the
    nearest half hour, or None if it's still climbing when the night ends
    (its highest point comes after morning twilight begins)."""
    start, end = night
    ts = load.timescale()
    steps = int((end - start) / SEARCH_STEP) + 1
    times = ts.from_datetimes([start + i * SEARCH_STEP for i in range(steps)])
    radiant = SkyfieldStar(ra_hours=shower.ra / 15, dec_degrees=shower.dec)
    here = ephemeris()["earth"] + wgs84.latlon(lat, lon)
    alt, _az, _distance = here.at(times).observe(radiant).apparent().altaz()
    best = int(alt.degrees.argmax())
    if alt.degrees[best] <= 0:
        raise ValueError(f"The {shower.name} radiant doesn't rise during the night")
    if best == steps - 1:
        return None
    return round_within(start + best * SEARCH_STEP, start, end)


def viewing_plan(
    lat: float,
    lon: float,
    shower: MeteorShower,
    day: date,
    tz: str | None = None,
    when: datetime | None = None,
) -> tuple[datetime, str]:
    """The time to chart for the night before the morning of `day` and the
    advice to caption it with. The chart shows the sky when the radiant is
    highest during the night, e.g. "Look overhead around 2:00 am". If it's
    still climbing when morning twilight begins, it shows 2 hours before
    sunrise instead, e.g. "Look southeast 2 hours before sunrise", or the
    start of twilight if that's earlier ("before dawn"). For weak showers,
    when the Moon sets during the night before then with the radiant up, the
    advice is "after moonset at 12:30 am" instead. `when` overrides the
    time charted, not the advice; the compass direction is always the
    radiant's at the time charted. Times are rounded to the half hour."""
    zone = ZoneInfo(tz or TimezoneFinder().timezone_at(lat=lat, lng=lon))
    night = night_hours(lat, lon, day, zone.key)
    look_from = highest_time(lat, lon, shower, night)
    if look_from is not None:
        advice = f"around {clock_text(look_from)}"
    else:
        sunrise, _sunset = sun_events(lat, lon, day, zone.key)
        look_from = sunrise - timedelta(hours=HOURS_BEFORE_SUNRISE)
        if look_from <= night[1]:
            advice = f"{HOURS_BEFORE_SUNRISE} hours before sunrise"
        else:
            look_from = night[1]
            advice = "before dawn"
        look_from = round_within(look_from, *night)

    if shower.zhr < WEAK_ZHR:
        moon_down = moonset(lat, lon, night[0], look_from)
        if moon_down is not None:
            start = round_within(moon_down, *night)
            if Sky(lat, lon, start).altaz(shower.ra, shower.dec)[0] > 0:
                advice = f"after moonset at {clock_text(start)}"

    when = when or look_from
    alt, az = Sky(lat, lon, when).altaz(shower.ra, shower.dec)
    if alt < 0:
        raise ValueError(f"The {shower.name} radiant is below the horizon at {when:%H:%M}")
    where = "overhead" if alt >= OVERHEAD_ALTITUDE else compass_direction(az)
    return when, f"Look {where} {advice}"


def plot_meteor_shower(
    lat: float,
    lon: float,
    shower: str | MeteorShower,
    day: date | None = None,
    tz: str | None = None,
    output_dir: str | Path = ".",
    timestamp: bool = False,
    labels: bool = True,
    when: datetime | None = None,
    asterism_labels: bool = False,
    constellation_labels: bool = False,
) -> Path:
    """Render the sky from `lat`/`lon` with `shower`'s radiant marked, for
    the night before the morning of `day` (default: the next morning of
    maximum), with the shower's name across the top and advice on where and
    when to look across the bottom, at the time viewing_plan() picks or at
    `when` (a timezone-aware local datetime) if given. The radiant is circled, with meteors streaking
    away from it. `timestamp` adds the date, time and place in the lower
    right corner; asterisms and constellations are named only with
    `asterism_labels` and `constellation_labels`; `labels=False` leaves off
    every name except the title and advice. Returns the path of the PNG, e.g.
    leonids_20261117T0500_35.78N_78.64W.png."""
    if isinstance(shower, str):
        shower = find_shower(shower)
    zone = ZoneInfo(tz or TimezoneFinder().timezone_at(lat=lat, lng=lon))
    if day is None:
        day = when.date() if when else peak_morning(shower, datetime.now(zone).date())
    when, advice = viewing_plan(lat, lon, shower, day, zone.key, when)
    radiant = Target("RADIANT", "radiant", shower.ra, shower.dec)
    return plot_targets(
        lat,
        lon,
        [radiant],
        when,
        output_dir,
        timestamp=timestamp,
        labels=labels,
        heading=shower.title,
        caption=advice,
        filename=output_filename([shower.name], when, lat, lon),
        asterism_labels=asterism_labels,
        constellation_labels=constellation_labels,
    )
