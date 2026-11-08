import numpy as np
import pytest
from scipy.signal import convolve2d

from lifelab.engine import Life, neighbor_counts, random_grid, step, step_rules
from lifelab.patterns import get_pattern, place
from lifelab.rules import CONWAY, Rule


def reference_step(grid, rule=CONWAY, boundary="dead"):
    """Slow, obviously-correct update used to check the fast engine."""
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    mode = {"dead": "fill", "wrap": "wrap"}[boundary]
    counts = convolve2d(grid, kernel, mode="same", boundary=mode)
    out = np.zeros_like(grid)
    for (i, j), alive in np.ndenumerate(grid):
        out[i, j] = counts[i, j] in (rule.survival if alive else rule.birth)
    return out


def test_neighbor_counts_single_cell():
    grid = np.zeros((5, 5), np.uint8)
    grid[2, 2] = 1
    counts = neighbor_counts(grid)
    assert counts.sum() == 8
    assert counts[2, 2] == 0
    assert (counts[1:4, 1:4].sum()) == 8


def test_wrap_versus_dead_corner():
    grid = np.zeros((4, 4), np.uint8)
    grid[0, 0] = 1
    assert neighbor_counts(grid, "wrap")[3, 3] == 1
    assert neighbor_counts(grid, "dead")[3, 3] == 0


@pytest.mark.parametrize("boundary", ["dead", "wrap"])
@pytest.mark.parametrize("rule", ["B3/S23", "B36/S23", "B2/S", "B3678/S34678", "B1357/S1357"])
def test_matches_reference(rule, boundary):
    rule = Rule.parse(rule)
    grid = random_grid((17, 23), 0.4, rng=3)
    for _ in range(5):
        expected = reference_step(grid, rule, boundary)
        grid = step(grid, rule, boundary)
        assert np.array_equal(grid, expected)


def test_accepts_bool_and_int_grids():
    grid = random_grid((10, 10), 0.5, rng=1)
    expected = step(grid)
    assert np.array_equal(step(grid.astype(bool)), expected)
    assert np.array_equal(step(grid.astype(np.int64)), expected)


def test_blinker_and_block():
    grid = np.zeros((7, 7), np.uint8)
    place(grid, "blinker", 3, 2)
    once = step(grid)
    assert not np.array_equal(once, grid)
    assert np.array_equal(step(once), grid)
    block = place(np.zeros((6, 6), np.uint8), "block", 2, 2)
    assert np.array_equal(step(block), block)


def test_glider_moves_diagonally_and_wraps():
    grid = place(np.zeros((8, 8), np.uint8), "glider", 0, 0)
    moved = grid
    for _ in range(4):
        moved = step(moved)
    assert np.array_equal(moved, np.roll(grid, (1, 1), (0, 1)))
    for _ in range(4 * 8 - 4):
        moved = step(moved)
    assert np.array_equal(moved, grid)  # one full lap of the torus


def test_diehard_dies_after_130_generations():
    world = Life(place(np.zeros((80, 80), np.uint8), "diehard", 40, 36), boundary="dead")
    world.step(129)
    assert world.population > 0
    world.step(1)
    assert world.population == 0


def test_gosper_gun_emits_a_glider_every_30_generations():
    world = Life(place(np.zeros((100, 100), np.uint8), "gosper", 2, 2), boundary="dead")
    world.step(60)
    pops = []
    for _ in range(3):
        pops.append(world.population)
        world.step(30)
    assert np.diff(pops).tolist() == [5, 5]


def test_synchrony_limits():
    grid = random_grid((32, 32), 0.4, rng=0)
    rng = np.random.default_rng(0)
    assert np.array_equal(step(grid, synchrony=0.0, rng=rng), grid)
    assert np.array_equal(step(grid, synchrony=1.0), step(grid))
    partial = step(grid, synchrony=0.5, rng=rng)
    full = step(grid)
    changed = partial != grid
    # Asynchronous updates only ever apply changes the full update would make.
    assert np.all(partial[changed] == full[changed])
    assert 0 < changed.sum() < (full != grid).sum()


def test_step_rules_matches_individual_rules():
    rules = [Rule.parse(r) for r in ["B3/S23", "B36/S23", "B2/S", "B1357/S1357"]]
    grids = random_grid((len(rules), 20, 20), 0.35, rng=5)
    luts = np.stack([r.lut for r in rules])
    batched = step_rules(grids, luts)
    for i, rule in enumerate(rules):
        assert np.array_equal(batched[i], step(grids[i], rule))


def test_life_object():
    world = Life.random((32, 32), density=0.3, seed=7)
    start = world.grid.copy()
    frames = world.history(10)
    assert frames.shape == (11, 32, 32)
    assert np.array_equal(frames[0], start)
    assert world.generation == 10
    assert 0 <= world.density <= 1
    assert "B3/S23" in repr(world)
    with pytest.raises(ValueError):
        Life(start, boundary="mirror")
