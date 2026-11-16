from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from horizonchart import cli, meteors
from horizonchart.meteors import (
    clock_text,
    compass_direction,
    find_shower,
    night_hours,
    peak_morning,
    plot_meteor_shower,
    round_within,
    viewing_plan,
)
from horizonchart.skyview import ASPECT

RALEIGH = (35.78, -78.64)
EASTERN = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "name", ["Leonids", "leonid", "Leonid meteor shower", "LEONIDS", " leonids "]
)
def test_find_shower(name):
    assert find_shower(name).name == "Leonids"


def test_find_shower_with_a_two_word_name():
    assert find_shower("eta aquariid").name == "Eta Aquariids"


def test_unknown_shower_lists_the_known_ones():
    with pytest.raises(ValueError, match="Unknown meteor shower.*Leonids"):
        find_shower("Krypton")


def test_peak_morning_is_this_year_or_next():
    leonids = find_shower("Leonids")
    assert peak_morning(leonids, date(2026, 11, 16)) == date(2026, 11, 17)
    assert peak_morning(leonids, date(2026, 11, 17)) == date(2026, 11, 17)
    assert peak_morning(leonids, date(2026, 11, 18)) == date(2027, 11, 17)


@pytest.mark.parametrize(
    "hour, minute, expected",
    [
        (0, 0, "Midnight"),
        (23, 50, "Midnight"),
        (0, 10, "Midnight"),
        (0, 20, "12:30 am"),
        (1, 44, "1:30 am"),
        (1, 46, "2:00 am"),
        (4, 30, "4:30 am"),
        (12, 0, "12:00 pm"),
        (22, 59, "11:00 pm"),
    ],
)
def test_clock_text_rounds_to_the_half_hour(hour, minute, expected):
    assert clock_text(datetime(2026, 11, 17, hour, minute, tzinfo=EASTERN)) == expected


@pytest.mark.parametrize(
    "az, expected",
    [(0, "north"), (22, "north"), (23, "northeast"), (90, "east"),
     (116, "southeast"), (270, "west"), (350, "north")],
)
def test_compass_direction(az, expected):
    assert compass_direction(az) == expected


def test_night_runs_between_civil_twilights():
    dusk, dawn = night_hours(*RALEIGH, date(2026, 11, 17))
    assert (dusk.hour, dusk.minute) == (17, 33)
    assert (dawn.hour, dawn.minute) == (6, 25)


def test_round_within_keeps_to_the_night():
    start = datetime(2026, 11, 16, 17, 33, tzinfo=EASTERN)
    end = datetime(2026, 11, 17, 6, 25, tzinfo=EASTERN)
    assert round_within(end, start, end) == datetime(2026, 11, 17, 6, 0, tzinfo=EASTERN)
    assert round_within(start, start, end).time().isoformat() == "18:00:00"
    assert round_within(start + timedelta(hours=1), start, end).time().isoformat() == "18:30:00"


def test_leonids_2026():
    # Charted at the last half hour before civil twilight (the radiant is
    # still climbing then); the Moon sets at 11:21 pm, before the radiant
    # rises, so the usual pre-dawn advice stands
    when, advice = viewing_plan(*RALEIGH, find_shower("Leonids"), date(2026, 11, 17))
    assert when == datetime(2026, 11, 17, 6, 0, tzinfo=EASTERN)
    assert advice == "Look overhead 2 hours before sunrise, around 5:00 am"


def test_chart_time_is_when_the_radiant_is_highest():
    # The Geminid radiant crosses the meridian around 2 am
    when, _advice = viewing_plan(*RALEIGH, find_shower("Geminids"), date(2026, 12, 14))
    assert when == datetime(2026, 12, 14, 2, 0, tzinfo=EASTERN)


def test_direction_is_for_the_chart_time():
    leonids = find_shower("Leonids")
    when = datetime(2026, 11, 17, 1, 0, tzinfo=EASTERN)
    assert viewing_plan(*RALEIGH, leonids, date(2026, 11, 17), when=when) == (
        when,
        "Look east 2 hours before sunrise, around 5:00 am",
    )


def test_weak_shower_starts_at_moonset():
    # The Moon sets at 2:3x am with the Orionid radiant already up
    _when, advice = viewing_plan(*RALEIGH, find_shower("Orionids"), date(2026, 10, 21))
    assert advice == "Look overhead after moonset at 2:30 am"


def test_strong_shower_ignores_the_moon():
    _when, advice = viewing_plan(*RALEIGH, find_shower("Geminids"), date(2026, 12, 14))
    assert advice.endswith("2 hours before sunrise, around 5:30 am")


def test_advice_is_never_after_civil_twilight_begins(monkeypatch):
    # Where twilight is long, 2 hours before sunrise can already be twilight;
    # here, any time before sunrise is
    monkeypatch.setattr(meteors, "HOURS_BEFORE_SUNRISE", 0)
    _when, advice = viewing_plan(*RALEIGH, find_shower("Geminids"), date(2026, 12, 14))
    assert advice.endswith("before dawn, around 6:30 am")


def test_radiant_that_never_rises_at_night():
    with pytest.raises(ValueError, match="doesn't rise during the night"):
        viewing_plan(64.84, -147.72, find_shower("Eta Aquariids"), date(2027, 5, 6))


def test_plot_meteor_shower(tmp_path):
    path = plot_meteor_shower(*RALEIGH, "Leonids", date(2026, 11, 17), output_dir=tmp_path)
    assert path.name == "leonids_20261117T0600_35.78N_78.64W.png"
    with Image.open(path) as im:
        assert im.width / im.height == pytest.approx(ASPECT, abs=0.01)


def test_meteors_command(tmp_path, capsys):
    status = cli.main(
        ["meteors", "Leonids", "-l", "35.78N 78.64W", "-d", "2026-11-17", "-o", str(tmp_path)]
    )
    assert status == 0
    assert capsys.readouterr().out.strip().endswith("leonids_20261117T0600_35.78N_78.64W.png")


def test_meteors_command_reports_unknown_showers(capsys):
    status = cli.main(["meteors", "Krypton", "-l", "35.78N 78.64W", "-d", "2026-11-17"])
    assert status == 1
    assert "Unknown meteor shower" in capsys.readouterr().err
