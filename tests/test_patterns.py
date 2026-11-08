import numpy as np
import pytest

from lifelab.engine import step
from lifelab.patterns import CATEGORY, PATTERNS, get_pattern, parse_rle, place, to_rle, transform

STILL = [n for n, c in CATEGORY.items() if c == "still life"]
PERIODS = {"blinker": 2, "toad": 2, "beacon": 2, "pulsar": 3, "pentadecathlon": 15}
SHIPS = {"glider": 4, "lwss": 4, "mwss": 4, "hwss": 4}


def embed(pattern, margin=12):
    canvas = np.zeros((pattern.shape[0] + 2 * margin, pattern.shape[1] + 2 * margin), np.uint8)
    canvas[margin: margin + pattern.shape[0], margin: margin + pattern.shape[1]] = pattern
    return canvas


@pytest.mark.parametrize("name", STILL)
def test_still_lifes_do_not_change(name):
    canvas = embed(PATTERNS[name])
    assert np.array_equal(step(canvas, boundary="dead"), canvas)


@pytest.mark.parametrize("name,period", PERIODS.items())
def test_oscillator_periods(name, period):
    canvas = embed(PATTERNS[name])
    grid = canvas
    for t in range(1, period + 1):
        grid = step(grid, boundary="dead")
        if t < period:
            assert not np.array_equal(grid, canvas), f"{name} repeated early at {t}"
    assert np.array_equal(grid, canvas)


@pytest.mark.parametrize("name,period", SHIPS.items())
def test_spaceships_translate(name, period):
    canvas = embed(PATTERNS[name])
    grid = canvas
    for _ in range(period):
        grid = step(grid, boundary="dead")
    assert grid.sum() == canvas.sum()
    shifts = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]
    assert any(np.array_equal(np.roll(canvas, s, (0, 1)), grid) for s in shifts)


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_rle_round_trip(name):
    grid, rule = parse_rle(to_rle(PATTERNS[name]))
    assert rule == "B3/S23"
    assert np.array_equal(grid, PATTERNS[name])


def test_parse_rle_with_header_and_comments():
    text = "#N Glider\n#C a comment\nx = 3, y = 3, rule = B3/S23\nbo$2bo$3o!\n"
    grid, rule = parse_rle(text)
    assert np.array_equal(grid, PATTERNS["glider"])
    assert rule == "B3/S23"
    grid, _ = parse_rle("x = 4, y = 3\n2o$$3bo!")
    assert grid.tolist() == [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]


def test_get_pattern_aliases_and_copy():
    assert np.array_equal(get_pattern("Gosper"), PATTERNS["gosper_glider_gun"])
    gun = get_pattern("gosper")
    gun[:] = 0
    assert PATTERNS["gosper_glider_gun"].any()
    with pytest.raises(KeyError):
        get_pattern("not-a-pattern")


def test_place_wraps_on_torus():
    grid = np.zeros((5, 5), np.uint8)
    place(grid, "block", 4, 4)
    assert grid.sum() == 4
    assert grid[0, 0] and grid[4, 4] and grid[0, 4] and grid[4, 0]
    clipped = place(np.zeros((5, 5), np.uint8), "block", 4, 4, wrap=False)
    assert clipped.sum() == 1


def test_transform_keeps_cell_count():
    glider = PATTERNS["glider"]
    for k in range(4):
        for flip in (False, True):
            assert transform(glider, k, flip).sum() == 5
