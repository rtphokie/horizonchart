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
