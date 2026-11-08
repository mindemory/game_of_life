import numpy as np
import pytest

from lifelab.analysis import canonical, census, cluster_labels, extract_objects, find_cycle, identify
from lifelab.engine import step
from lifelab.patterns import CATEGORY, PATTERNS, place, transform


def test_find_cycle_blinker_and_glider():
    blinker = place(np.zeros((8, 8), np.uint8), "blinker", 3, 2)
    cyc = find_cycle(blinker)
    assert (cyc.transient, cyc.period) == (0, 2)
    glider = place(np.zeros((8, 8), np.uint8), "glider", 0, 0)
    cyc = find_cycle(glider)
    assert (cyc.transient, cyc.period) == (0, 32)  # one lap of an 8x8 torus


def test_find_cycle_transient_and_budget():
    diehard = place(np.zeros((60, 60), np.uint8), "diehard", 30, 26)
    cyc = find_cycle(diehard, boundary="dead")
    assert cyc.transient == 130 and cyc.period == 1 and cyc.settled
    gun = place(np.zeros((60, 80), np.uint8), "gosper", 2, 2)
    assert not find_cycle(gun, boundary="dead", max_steps=50).settled


def test_canonical_is_symmetry_invariant():
    loaf = PATTERNS["loaf"]
    keys = {canonical(transform(loaf, k, f)) for k in range(4) for f in (False, True)}
    assert len(keys) == 1
    assert canonical(PATTERNS["boat"]) != canonical(PATTERNS["ship"])


@pytest.mark.parametrize("name", [n for n, c in CATEGORY.items() if c in ("still life", "oscillator", "spaceship")])
def test_identify_every_phase_and_orientation(name):
    pattern = transform(PATTERNS[name], 1, True)
    canvas = np.zeros((pattern.shape[0] + 16, pattern.shape[1] + 16), np.uint8)
    canvas[8: 8 + pattern.shape[0], 8: 8 + pattern.shape[1]] = pattern
    for _ in range(3):  # check a few different phases
        assert identify(canvas).name == name
        canvas = step(canvas, boundary="dead")


def test_identify_unknown_objects():
    assert identify(PATTERNS["r_pentomino"]).name == "unstable"
    info = identify(np.ones((1, 1), np.uint8))
    assert info.period is None


def test_cluster_labels_wraps_and_reach():
    grid = np.zeros((10, 10), np.uint8)
    place(grid, "block", 9, 9)  # split over all four corners
    labels, n = cluster_labels(grid, wrap=True)
    assert n == 1
    _, n_nowrap = cluster_labels(grid, wrap=False)
    assert n_nowrap == 4
    pulsar = np.zeros((20, 20), np.uint8)
    place(pulsar, "pulsar", 3, 3)
    assert cluster_labels(pulsar, reach=1)[1] > 1
    assert cluster_labels(pulsar, reach=2)[1] == 1


def test_census_counts_objects():
    grid = np.zeros((64, 64), np.uint8)
    for r, c in [(2, 2), (2, 20), (40, 50)]:
        place(grid, "block", r, c)
    place(grid, "blinker", 20, 5)
    place(grid, transform(PATTERNS["blinker"], 1), 20, 30)
    place(grid, "pulsar", 40, 10)
    place(grid, "beehive", 62, 30)  # wraps over the bottom edge
    # Two blocks one cell apart are counted separately.
    place(grid, "block", 10, 40)
    place(grid, "block", 10, 43)
    counts = census(grid)
    assert counts == {"block": 5, "blinker": 2, "pulsar": 1, "beehive": 1}


def test_extract_objects_crops():
    grid = np.zeros((16, 16), np.uint8)
    place(grid, "glider", 5, 7)
    (cells, (r, c)), = extract_objects(grid, wrap=False)
    assert np.array_equal(cells, PATTERNS["glider"])
    assert (r, c) == (5, 7)


def test_find_cycles_matches_single_runs():
    from lifelab.analysis import find_cycles
    from lifelab.engine import random_grid
    from lifelab.rules import Rule

    grids = random_grid((6, 24, 24), 0.35, rng=11)
    batch = find_cycles(grids, max_steps=3000)
    for g, tr, pe in zip(grids, batch.transient, batch.period):
        single = find_cycle(g, max_steps=3000)
        assert (single.transient, single.period) == (tr, pe)
    # Per-world rules: an empty world under B/S dies at once; a blinker keeps period 2.
    blinker = place(np.zeros((8, 8), np.uint8), "blinker", 3, 2)
    luts = np.stack([Rule.parse("B/S").lut, Rule.parse("B3/S23").lut])
    out = find_cycles(np.stack([blinker, blinker]), luts=luts, max_steps=50)
    assert out.period.tolist() == [1, 2]
    assert out.density[0] == 0 and out.density[1] == pytest.approx(3 / 64)
    assert out.activity[1] == pytest.approx(4 / 64)


def test_find_cycles_unsettled_window():
    from lifelab.analysis import find_cycles

    gun = place(np.zeros((60, 80), np.uint8), "gosper", 2, 2)
    out = find_cycles(gun[None], boundary="dead", max_steps=20, window=64)
    assert out.period[0] == -1 and out.steps == 20
    assert out.density[0] > 0.0
