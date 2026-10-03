from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from horizonchart import cli


@pytest.mark.parametrize(
    "text, expected",
    [
        ("35.7796,-78.6382", (35.7796, -78.6382)),
        ("35.7796, -78.6382", (35.7796, -78.6382)),
        ("35.7796 -78.6382", (35.7796, -78.6382)),
        ("35.78N 78.64W", (35.78, -78.64)),
        ("33.87S, 151.21E", (-33.87, 151.21)),
        ("45.42°N 75.70°W", (45.42, -75.70)),
        ("Raleigh, NC", None),
        ("Ottawa, ON", None),
    ],
)
def test_parse_coordinates(text, expected):
    assert cli.parse_coordinates(text) == expected


def test_parse_coordinates_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        cli.parse_coordinates("95,10")


def test_place_names_are_geocoded_and_cached(tmp_path, monkeypatch):
    calls = []

    class FakeNominatim:
        def __init__(self, user_agent):
            pass

        def geocode(self, place, timeout=None):
            calls.append(place)
            return type("Location", (), {"latitude": 45.42, "longitude": -75.70})()

    monkeypatch.setattr(cli, "GEOCODE_CACHE", tmp_path / "geocode.json")
    monkeypatch.setattr("geopy.geocoders.Nominatim", FakeNominatim)
    monkeypatch.setattr(cli, "_geocoder", None)

    assert cli.resolve_location("Ottawa, ON") == (45.42, -75.70)
    assert cli.resolve_location("ottawa, on") == (45.42, -75.70)
    assert calls == ["Ottawa, ON"]  # second lookup came from the cache


def test_parse_local_datetime():
    zone = ZoneInfo("America/New_York")
    assert cli.parse_local_datetime("2026-10-02 22:00", zone) == datetime(
        2026, 10, 2, 22, 0, tzinfo=zone
    )


def test_local_zone_from_coordinates():
    assert cli.local_zone(45.42, -75.70, None).key == "America/Toronto"


def test_target_command(tmp_path, capsys):
    status = cli.main(
        [
            "target",
            "Saturn",
            "--location",
            "35.7796,-78.6382",
            "--time",
            "2026-10-02 22:00",
            "-o",
            str(tmp_path),
        ]
    )
    assert status == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("saturn_20261002T2200_35.78N_78.64W.png")
    assert (tmp_path / "saturn_20261002T2200_35.78N_78.64W.png").exists()


def test_twilight_command(tmp_path, capsys):
    status = cli.main(
        ["twilight", "-l", "35.78N 78.64W", "-d", "2026-10-02", "-o", str(tmp_path)]
    )
    assert status == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "20261002T0600_35.78N_78.64W_mor.png",
        "20261002T2100_35.78N_78.64W_eve.png",
    ]


def test_errors_are_reported(capsys):
    status = cli.main(
        ["target", "Krypton", "-l", "35.78,-78.64", "-t", "2026-10-02 22:00"]
    )
    assert status == 1
    assert "Unknown target" in capsys.readouterr().err


def test_geocoder_identifies_itself(monkeypatch):
    monkeypatch.delenv("HORIZONCHART_CONTACT", raising=False)
    assert cli.geocoder_user_agent().startswith("horizonchart/")
    monkeypatch.setenv("HORIZONCHART_CONTACT", "me@example.com")
    assert cli.geocoder_user_agent().endswith("; me@example.com)")


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], {"timestamp": True, "labels": True, "landscape": "hills"}),
        (["--no-timestamp"], {"timestamp": False, "labels": True, "landscape": "hills"}),
        (["--no-labels"], {"timestamp": True, "labels": False, "landscape": "hills"}),
        (["--landscape", "city"], {"timestamp": True, "labels": True, "landscape": "city"}),
    ],
)
def test_target_display_flags(tmp_path, monkeypatch, argv, expected):
    seen = {}

    def fake_plot_targets(*args, **kwargs):
        seen.update(kwargs)
        return tmp_path / "chart.png"

    monkeypatch.setattr(cli, "plot_targets", fake_plot_targets)
    cli.main(["target", "Saturn", "-l", "35.78,-78.64", "-t", "2026-10-02 22:00", *argv])
    assert seen == expected


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], {"title": True, "timestamp": True, "labels": True, "landscape": "hills"}),
        (["--no-title"], {"title": False, "timestamp": True, "labels": True, "landscape": "hills"}),
        (
            ["--no-title", "--no-timestamp", "--no-labels", "--landscape", "trees"],
            {"title": False, "timestamp": False, "labels": False, "landscape": "trees"},
        ),
    ],
)
def test_twilight_display_flags(tmp_path, monkeypatch, argv, expected):
    seen = {}

    def fake_plot_twilight_views(*args, **kwargs):
        seen.update({k: kwargs[k] for k in ("title", "timestamp", "labels", "landscape")})
        return {"morning": tmp_path / "m.png", "evening": tmp_path / "e.png"}

    monkeypatch.setattr(cli, "plot_twilight_views", fake_plot_twilight_views)
    cli.main(["twilight", "-l", "35.78,-78.64", "-d", "2026-10-02", *argv])
    assert seen == expected


def test_title_flag_is_twilight_only():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["target", "Saturn", "-l", "0,0", "--no-title"])


def test_default_target_time_is_two_hours_after_sunset(monkeypatch):
    zone = ZoneInfo("America/New_York")

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 2, 9, 30, tzinfo=tz)

    monkeypatch.setattr(cli, "datetime", FixedDatetime)
    # Raleigh sunset on Oct 2, 2026 is 6:56 PM EDT; 8:56 PM rounds to 9:00
    assert cli.default_target_time(35.7796, -78.6382, zone) == datetime(
        2026, 10, 2, 21, 0, tzinfo=zone
    )


def test_unknown_landscape_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["target", "Saturn", "-l", "0,0", "--landscape", "moon"])
