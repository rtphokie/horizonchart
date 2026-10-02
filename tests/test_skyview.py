from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from starplot import DSO, Observer, Star
from horizonchart import skyview
from horizonchart.skyview import (
    ASPECT,
    ASTERISMS,
    dso_label,
    output_filename,
    twilight_filename,
    plot_targets,
    choose_star_labels,
    choose_twilight_moment,
    choose_twilight_view,
    interest,
    plot_twilight_views,
    resolve_target,
    round_to_half_hour,
    same_as_constellation,
    select_asterisms,
    select_constellations,
    landmarks_in_view,
    twilight_scene,
    twilight_times,
)

RALEIGH = (35.7796, -78.6382)
OCT_2_2026_10PM = datetime(2026, 10, 2, 22, 0, tzinfo=ZoneInfo("America/New_York"))


def test_saturn_from_raleigh_oct_2_2026(tmp_path):
    """Saturn near opposition, 10 PM EDT: just inside Cetus at the Pisces
    border, found from the Great Square of Pegasus and Circlet of Pisces."""
    path = plot_targets(*RALEIGH, ["Saturn"], OCT_2_2026_10PM, output_dir=tmp_path)

    assert path.name == "saturn_20261002T2200_35.78N_78.64W.png"
    with Image.open(path) as im:
        w, h = im.size
    assert w / h == pytest.approx(ASPECT, abs=0.002)


def test_saturn_finder_context():
    observer = Observer(lat=RALEIGH[0], lon=RALEIGH[1], dt=OCT_2_2026_10PM)
    saturn = resolve_target("Saturn", observer)

    asterisms = select_asterisms([saturn])
    names = {a.name.replace("\n", " ") for a, _lines in asterisms}
    assert names == {"Great Square of Pegasus", "Circlet of Pisces"}

    # The constellation Saturn is in, the one whose border it's on, and the
    # Great Square's home
    assert select_constellations([saturn], [a for a, _lines in asterisms]) == [
        "cet",
        "peg",
        "psc",
    ]


@pytest.mark.parametrize(
    "name, label, kind",
    [
        ("moon", "MOON", "moon"),
        ("Jupiter", "JUPITER", "planet"),
        ("Pleiades", "PLEIADES", "dso"),
        ("M31", "M31", "dso"),
        ("NGC 1976", "NGC 1976", "dso"),
        ("Vega", "VEGA", "star"),
        # BSP ephemeris naming
        ("saturn barycenter", "SATURN", "planet"),
        ("SATURN BARYCENTER", "SATURN", "planet"),
        ("jupiter_barycenter", "JUPITER", "planet"),
    ],
)
def test_resolve_target(name, label, kind):
    observer = Observer(lat=RALEIGH[0], lon=RALEIGH[1], dt=OCT_2_2026_10PM)
    target = resolve_target(name, observer)
    assert (target.label, target.kind) == (label, kind)


def test_barycenter_files_under_planet_name(tmp_path):
    path = plot_targets(
        *RALEIGH, ["saturn barycenter"], OCT_2_2026_10PM, output_dir=tmp_path
    )
    assert path.name == "saturn_20261002T2200_35.78N_78.64W.png"


def test_unknown_target():
    observer = Observer(lat=RALEIGH[0], lon=RALEIGH[1], dt=OCT_2_2026_10PM)
    with pytest.raises(ValueError, match="Unknown target"):
        resolve_target("Krypton", observer)


def test_target_below_horizon(tmp_path):
    # Jupiter is well below the horizon from Raleigh at 10 PM on Oct 2, 2026
    with pytest.raises(ValueError, match="below the horizon"):
        plot_targets(*RALEIGH, ["Jupiter"], OCT_2_2026_10PM, output_dir=tmp_path)


def test_naive_datetime_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        plot_targets(*RALEIGH, ["Saturn"], datetime(2026, 10, 2, 22), tmp_path)


@pytest.mark.parametrize(
    "targets, lat, lon, expected",
    [
        (["Saturn"], 35.7796, -78.6382, "saturn_20261002T2200_35.78N_78.64W.png"),
        (["Moon", "Pleiades"], -33.8688, 151.2093, "moon_pleiades_20261002T2200_33.87S_151.21E.png"),
        (["Andromeda Galaxy"], 0.0, 0.0, "andromeda-galaxy_20261002T2200_0.00N_0.00E.png"),
    ],
)
def test_output_filename(targets, lat, lon, expected):
    assert output_filename(targets, OCT_2_2026_10PM, lat, lon) == expected


# Twilight views: Raleigh, Oct 2, 2026 (sunrise 7:10 AM, sunset 6:56 PM EDT)
OCT_2_2026 = date(2026, 10, 2)


def test_twilight_times_raleigh():
    morning, evening = twilight_times(*RALEIGH, OCT_2_2026)
    # Time zone is inferred from the coordinates
    assert str(morning.tzinfo) == "America/New_York"
    # 6:10 and 7:56 PM, rounded to the nearest half hour
    assert (morning.hour, morning.minute) == (6, 0)
    assert (evening.hour, evening.minute) == (20, 0)


def test_twilight_views_raleigh_oct_2_2026(tmp_path):
    paths = plot_twilight_views(*RALEIGH, OCT_2_2026, output_dir=tmp_path)

    # Mars and Jupiter in the east at dawn. In the evening Saturn rising in
    # the east beats the planet-less west, and since it's low at 8 PM the
    # view moves to 9 PM, 2 hours after sunset
    assert paths["morning"].name == "20261002T0600_35.78N_78.64W_mor.png"
    assert paths["evening"].name == "20261002T2100_35.78N_78.64W_eve.png"
    for path in paths.values():
        with Image.open(path) as im:
            w, h = im.size
        assert w / h == pytest.approx(ASPECT, abs=0.002)


def test_twilight_east_before_sunrise():
    morning, _evening = twilight_times(*RALEIGH, OCT_2_2026)
    scene = twilight_scene(*RALEIGH, morning, center_az=90)

    labels = {t.label for t in scene.targets}
    assert {"MARS", "JUPITER"} <= labels
    # The Beehive (3.1) and Christmas Tree Cluster (3.9) are too faint for
    # the default suburban limit of 2.0
    assert not {"BEEHIVE", "CHRISTMAS TREE CLUSTER"} & labels
    asterisms = {a.name for a, _lines in scene.asterisms}
    assert {"Sickle of Leo", "Big Dipper"} <= asterisms
    # Only the asterisms' constellations
    assert scene.constellations == ["leo", "uma"]


def test_twilight_west_after_sunset():
    _morning, evening = twilight_times(*RALEIGH, OCT_2_2026)
    scene = twilight_scene(*RALEIGH, evening, center_az=270)

    # Mercury and Venus have already set; Saturn is rising in the east
    assert not [t for t in scene.targets if t.kind == "planet"]
    asterisms = {a.name for a, _lines in scene.asterisms}
    assert {"Keystone", "Big Dipper"} <= asterisms
    assert scene.constellations == ["her", "uma"]


def test_twilight_limiting_magnitude_controls_dsos():
    morning, _evening = twilight_times(*RALEIGH, OCT_2_2026)
    scene = twilight_scene(*RALEIGH, morning, center_az=90, limiting_magnitude=4.0)
    labels = {t.label for t in scene.targets}
    assert {"BEEHIVE", "CHRISTMAS TREE CLUSTER"} <= labels


def test_twilight_dsos_need_altitude():
    # NGC 6231 (2.6) is 3° up in the southwest after sunset: too low to see
    # even when the limiting magnitude would include it
    _morning, evening = twilight_times(*RALEIGH, OCT_2_2026)
    scene = twilight_scene(*RALEIGH, evening, center_az=270, limiting_magnitude=4.0)
    assert "NGC 6231" not in {t.label for t in scene.targets}


def test_dso_label():
    assert dso_label(DSO.get(m="45")) == "PLEIADES"
    assert dso_label(DSO.get(ngc="6231")) == "NGC 6231"


def test_landmark_constellations_shown_when_in_view():
    # After sunset on Apr 1, 2027, Orion is low in the west and Cassiopeia in
    # the northwest
    _morning, evening = twilight_times(*RALEIGH, date(2027, 4, 1))
    scene = twilight_scene(*RALEIGH, evening, center_az=270)
    assert {"ori", "cas"} <= set(landmarks_in_view(scene.sky, scene.altitude, scene.azimuth))
    assert {"ori", "cas"} <= set(scene.constellations)


def test_landmark_constellations_need_most_of_the_figure():
    # At dawn on Oct 2, Orion is due south, outside the east-facing view
    morning, _evening = twilight_times(*RALEIGH, OCT_2_2026)
    scene = twilight_scene(*RALEIGH, morning, center_az=90)
    assert "ori" not in landmarks_in_view(scene.sky, scene.altitude, scene.azimuth)


@pytest.mark.parametrize("name", ["Uranus", "Neptune", "Pluto"])
def test_faint_planets_drawn_smaller(name):
    observer = Observer(lat=RALEIGH[0], lon=RALEIGH[1], dt=OCT_2_2026_10PM)
    target = resolve_target(name, observer)
    assert target.faint
    assert target.font_size < resolve_target("Saturn", observer).font_size


def test_naked_eye_planets_not_faint():
    observer = Observer(lat=RALEIGH[0], lon=RALEIGH[1], dt=OCT_2_2026_10PM)
    assert not any(
        resolve_target(n, observer).faint for n in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
    )


@pytest.mark.parametrize(
    "hh, mm, expected",
    [(6, 10, (6, 0)), (6, 15, (6, 30)), (6, 44, (6, 30)), (6, 45, (7, 0)), (23, 50, (0, 0))],
)
def test_round_to_half_hour(hh, mm, expected):
    tz = ZoneInfo("America/New_York")
    rounded = round_to_half_hour(datetime(2026, 10, 2, hh, mm, tzinfo=tz))
    assert (rounded.hour, rounded.minute) == expected


def test_twilight_view_picks_the_more_interesting_direction():
    _morning, evening = twilight_times(*RALEIGH, OCT_2_2026)
    direction, scene = choose_twilight_view(*RALEIGH, evening, default="west")
    assert direction == "east"
    assert "SATURN" in {t.label for t in scene.targets}


def test_twilight_view_tie_keeps_default(monkeypatch):
    morning, _evening = twilight_times(*RALEIGH, OCT_2_2026)
    monkeypatch.setattr(skyview, "interest", lambda scene: 1)
    assert choose_twilight_view(*RALEIGH, morning, default="east")[0] == "east"
    assert choose_twilight_view(*RALEIGH, morning, default="west")[0] == "west"


def test_twilight_moment_moves_away_from_the_sun_for_low_planets():
    # Saturn is ~12° up at 8 PM, so 9 PM is tried, and wins
    hours, when, direction, scene = choose_twilight_moment(*RALEIGH, OCT_2_2026, "evening")
    assert (hours, when.hour, direction) == (2, 21, "east")
    assert "SATURN" in {t.label for t in scene.targets}


def test_twilight_moment_keeps_usual_time_when_planets_are_high():
    # Mars and Jupiter are well up at dawn
    hours, when, _direction, _scene = choose_twilight_moment(*RALEIGH, OCT_2_2026, "morning")
    assert (hours, when.hour) == (1, 6)


def test_asterism_same_as_constellation():
    by_name = {a.name: a for a in ASTERISMS}
    # The W is all of Cassiopeia's figure, the Little Dipper all of Ursa Minor's
    assert same_as_constellation(by_name["W of Cassiopeia"])
    assert same_as_constellation(by_name["Little Dipper"])
    # The Big Dipper is only part of Ursa Major
    assert not same_as_constellation(by_name["Big Dipper"])
    assert not same_as_constellation(by_name["Summer\nTriangle"])



def test_star_labels_limited_and_prioritized():
    # Jan 15, 2027 dawn, looking east: the priority list's stars in view, in
    # its order, then Albireo (a well-known star not on it) as the fifth.
    # Arcturus and Spica are on the list but at the edges of the frame.
    morning, _evening = twilight_times(*RALEIGH, date(2027, 1, 15))
    scene = twilight_scene(*RALEIGH, morning, center_az=90)
    hips = choose_star_labels(scene.sky, scene.altitude, scene.azimuth, lambda s: True)
    names = [Star.get(hip=h).name for h in hips]
    assert names == ["Vega", "Antares", "Altair", "Deneb", "Albireo"]

    # Above the landscape only: Altair is behind the hills, and no other
    # well-known star is in view to take its place
    hips = choose_star_labels(
        scene.sky, scene.altitude, scene.azimuth, lambda s: True, min_altitude=9
    )
    names = [Star.get(hip=h).name for h in hips]
    assert names == ["Vega", "Antares", "Deneb", "Albireo"]


def test_twilight_filename():
    tz = ZoneInfo("Australia/Sydney")
    when = datetime(2026, 10, 2, 19, 30, tzinfo=tz)
    assert twilight_filename(when, -33.8688, 151.2093, "evening") == (
        "20261002T1930_33.87S_151.21E_eve.png"
    )
    assert twilight_filename(when, -33.8688, 151.2093, "morning").endswith("_mor.png")


def test_footnote_text():
    assert (
        skyview.footnote_text(OCT_2_2026_10PM, *RALEIGH)
        == "Fri Oct 2, 2026 · 10:00 PM EDT · 35.78°N 78.64°W"
    )


def lower_right(path):
    """The lower right corner, where the timestamp goes."""
    with Image.open(path) as im:
        w, h = im.size
        return im.convert("RGB").crop((w * 2 // 3, h * 9 // 10, w, h)).tobytes()


def test_target_view_timestamp_and_labels_can_be_left_off(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "bare").mkdir()
    full = plot_targets(*RALEIGH, ["Saturn"], OCT_2_2026_10PM, output_dir=tmp_path / "full")
    bare = plot_targets(
        *RALEIGH,
        ["Saturn"],
        OCT_2_2026_10PM,
        output_dir=tmp_path / "bare",
        timestamp=False,
        labels=False,
    )
    assert lower_right(full) != lower_right(bare)
    # Without labels or timestamp, less is drawn
    assert bare.stat().st_size < full.stat().st_size
