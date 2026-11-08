"""Vectorised NumPy engine for Life-like cellular automata.

Grids are ``uint8`` arrays of 0s and 1s. Every function also accepts a stack
of grids with shape ``(..., H, W)``, so many independent worlds can be run
in one call, which is how the experiments get their speed.

Two boundary conditions are supported:

``"wrap"``
    The grid is a torus: the top edge touches the bottom, left touches right.
``"dead"``
    Everything outside the grid is permanently dead (also called "cold" or
    "open" boundaries in the literature).
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .rules import CONWAY, Rule, RuleLike

__all__ = [
    "BOUNDARIES",
    "neighbor_counts",
    "step",
    "step_rules",
    "random_grid",
    "Life",
]

BOUNDARIES = ("wrap", "dead")


def _check_boundary(boundary: str) -> None:
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")


def neighbor_counts(grid: np.ndarray, boundary: str = "wrap") -> np.ndarray:
    """Number of live cells among the eight neighbours of every cell."""
    _check_boundary(boundary)
    grid = np.asarray(grid, dtype=np.uint8)
    if boundary == "wrap":
        padded = np.concatenate((grid[..., -1:, :], grid, grid[..., :1, :]), axis=-2)
        padded = np.concatenate((padded[..., :, -1:], padded, padded[..., :, :1]), axis=-1)
    else:
        padded = np.pad(grid, [(0, 0)] * (grid.ndim - 2) + [(1, 1), (1, 1)])
    # 3x3 box sum done as two separable 3-sums, then remove the centre cell.
    cols = padded[..., :-2, :] + padded[..., 1:-1, :] + padded[..., 2:, :]
    box = cols[..., :, :-2] + cols[..., :, 1:-1] + cols[..., :, 2:]
    return box - grid


def step(
    grid: np.ndarray,
    rule: RuleLike = CONWAY,
    boundary: str = "wrap",
    synchrony: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Advance ``grid`` by one generation.

    ``synchrony`` is the probability that each cell is updated in a given
    step. ``1.0`` is Conway's original fully synchronous update; smaller
    values give the "alpha-asynchronous" Life studied by Blok & Bergersen
    (1999), where only a random fraction of cells change at each tick.
    """
    lut = Rule.parse(rule).lut
    grid = np.asarray(grid, dtype=np.uint8)
    counts = neighbor_counts(grid, boundary)
    new = lut[(grid << 4) | counts]
    if synchrony < 1.0:
        if rng is None:
            rng = np.random.default_rng()
        # 16-bit random numbers are plenty fine (1/65536 resolution) and fast.
        threshold = int(round(synchrony * 65536))
        update = (rng.integers(0, 65536, grid.shape, dtype=np.uint16) < threshold).view(np.uint8)
        new = grid ^ ((new ^ grid) & update)
    return new


def step_rules(grids: np.ndarray, luts: np.ndarray, boundary: str = "wrap") -> np.ndarray:
    """Advance a stack of grids ``(R, H, W)`` where world ``i`` follows the rule
    whose lookup table is ``luts[i]`` (shape ``(R, 32)``, see :attr:`Rule.lut`)."""
    counts = neighbor_counts(grids, boundary)
    index = (grids << 4) | counts
    offsets = (np.arange(len(grids), dtype=np.intp) * 32)[:, None, None]
    return luts.reshape(-1)[index + offsets]


def random_grid(
    shape: tuple[int, ...],
    density: float = 0.5,
    rng: Optional[np.random.Generator | int] = None,
) -> np.ndarray:
    """A "random soup": each cell alive independently with probability ``density``."""
    rng = np.random.default_rng(rng)
    return (rng.random(shape) < density).astype(np.uint8)


class Life:
    """A single world that remembers its rule, boundary and generation count.

    >>> world = Life.random((64, 64), density=0.35, seed=1)
    >>> world.step(100).generation
    100
    """

    def __init__(
        self,
        grid: np.ndarray,
        rule: RuleLike = CONWAY,
        boundary: str = "wrap",
        synchrony: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        _check_boundary(boundary)
        self.grid = np.ascontiguousarray(grid, dtype=np.uint8)
        self.rule = Rule.parse(rule)
        self.boundary = boundary
        self.synchrony = float(synchrony)
        self.rng = np.random.default_rng(seed)
        self.generation = 0

    @classmethod
    def random(
        cls,
        shape: tuple[int, int],
        density: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "Life":
        rng = np.random.default_rng(seed)
        world = cls(random_grid(shape, density, rng), **kwargs)
        world.rng = rng
        return world

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def population(self) -> int:
        return int(self.grid.sum())

    @property
    def density(self) -> float:
        return self.population / self.grid.size

    def step(self, n: int = 1) -> "Life":
        for _ in range(n):
            self.grid = step(self.grid, self.rule, self.boundary, self.synchrony, self.rng)
            self.generation += 1
        return self

    def run(self, n: int) -> Iterator[np.ndarray]:
        """Yield the grid after each of the next ``n`` generations."""
        for _ in range(n):
            self.step()
            yield self.grid

    def history(self, n: int) -> np.ndarray:
        """Return an array ``(n + 1, H, W)``: the current grid and ``n`` more."""
        frames = np.empty((n + 1, *self.grid.shape), dtype=np.uint8)
        frames[0] = self.grid
        for i, grid in enumerate(self.run(n), start=1):
            frames[i] = grid
        return frames

    def __repr__(self) -> str:
        h, w = self.shape
        return (
            f"Life({h}x{w}, rule={self.rule}, boundary={self.boundary!r}, "
            f"synchrony={self.synchrony}, generation={self.generation}, "
            f"population={self.population})"
        )
