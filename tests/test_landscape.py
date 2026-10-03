import pytest
from PIL import Image

from horizonchart.landscape import LANDSCAPES, SILHOUETTE, draw_landscape, ground_profile

SKY = (120, 110, 200)
GROUND = (71, 48, 36)  # from starplot's ground gradient
W, H = 800, 450


def scene(tmp_path, ground_top=380):
    """Sky over flat brown ground, with starplot's black border along the bottom."""
    im = Image.new("RGB", (W, H), SKY)
    im.paste(GROUND, (0, ground_top, W, H))
    im.paste((0, 0, 0), (0, H - 2, W, H))
    path = tmp_path / "chart.png"
    im.save(path)
    return path


def test_ground_profile_finds_hilltops_past_the_border(tmp_path):
    with Image.open(scene(tmp_path)) as im:
        assert set(ground_profile(im)) == {380}


@pytest.mark.parametrize("kind", [k for k in LANDSCAPES if k != "hills"])
def test_silhouettes_replace_the_ground(tmp_path, kind):
    path = scene(tmp_path)
    draw_landscape(path, kind, seed="35.78,-78.64")
    with Image.open(path) as im:
        assert im.getpixel((W // 2, H - 10)) == SILHOUETTE
        # Something rises above the old ground line
        above = [im.getpixel((x, 370)) for x in range(W)]
        assert SILHOUETTE in above


def test_hills_leaves_the_chart_alone(tmp_path):
    path = scene(tmp_path)
    before = path.read_bytes()
    draw_landscape(path, "hills", seed="x")
    assert path.read_bytes() == before


def test_same_place_same_scene(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa, pb = scene(a), scene(b)
    draw_landscape(pa, "suburban", seed="35.78,-78.64")
    draw_landscape(pb, "suburban", seed="35.78,-78.64")
    assert pa.read_bytes() == pb.read_bytes()


@pytest.mark.parametrize("kind", ["trees", "city"])
def test_targets_stay_visible(tmp_path, kind):
    path = scene(tmp_path)
    target = (400, 360)  # just above the ground, where trees and towers would be
    draw_landscape(path, kind, seed="x", keep_clear=[target])
    with Image.open(path) as im:
        assert im.getpixel(target) == SKY


def test_unknown_landscape(tmp_path):
    with pytest.raises(ValueError, match="Unknown landscape"):
        draw_landscape(scene(tmp_path), "moon", seed="x")
