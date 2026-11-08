"""Tools for measuring what a Life world does.

* :func:`find_cycle` runs a world until it repeats a state exactly, which
  gives the transient length (how long it took to settle) and the period of
  the cycle it fell into.
* :func:`census` splits a settled world into separate objects and names
  them (block, blinker, glider, ...), like a field biologist doing a survey.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import ndimage

from .engine import step, step_rules
from .patterns import CATEGORY, PATTERNS
from .rules import CONWAY, Rule, RuleLike

__all__ = [
    "state_key",
    "Cycle",
    "find_cycle",
    "CycleBatch",
    "find_cycles",
    "cluster_labels",
    "extract_objects",
    "canonical",
    "ObjectInfo",
    "identify",
    "census",
]


def state_key(grid: np.ndarray) -> int:
    """A 64-bit fingerprint of a grid, used to spot repeated states."""
    return hash(np.packbits(grid).tobytes())


@dataclass
class Cycle:
    """Result of :func:`find_cycle`.

    ``transient`` is the number of generations before the world first enters
    its final cycle and ``period`` is that cycle's length (1 for a world that
    has frozen). Both are ``None`` if no repeat was found within the budget.
    """

    transient: Optional[int]
    period: Optional[int]
    steps: int
    grid: np.ndarray

    @property
    def settled(self) -> bool:
        return self.period is not None


def find_cycle(
    grid: np.ndarray,
    rule: RuleLike = CONWAY,
    boundary: str = "wrap",
    max_steps: int = 10_000,
) -> Cycle:
    """Run ``grid`` until a state repeats or ``max_steps`` is reached."""
    seen = {state_key(grid): 0}
    for t in range(1, max_steps + 1):
        grid = step(grid, rule, boundary)
        key = state_key(grid)
        first = seen.get(key)
        if first is not None:
            return Cycle(first, t - first, t, grid)
        seen[key] = t
    return Cycle(None, None, max_steps, grid)


@dataclass
class CycleBatch:
    """Result of :func:`find_cycles` for a stack of worlds (``-1`` = not settled).

    ``density`` and ``activity`` (fraction of cells that flip per generation)
    are averaged over the final cycle, or over the last ``window``
    generations for worlds that never settled.
    """

    transient: np.ndarray
    period: np.ndarray
    density: np.ndarray
    activity: np.ndarray
    final: np.ndarray
    steps: int

    @property
    def settled(self) -> np.ndarray:
        return self.period > 0


def find_cycles(
    grids: np.ndarray,
    rule: RuleLike = CONWAY,
    boundary: str = "wrap",
    max_steps: int = 10_000,
    luts: Optional[np.ndarray] = None,
    window: int = 64,
) -> CycleBatch:
    """Batched :func:`find_cycle` for a stack of grids ``(R, H, W)``.

    Pass ``luts`` (shape ``(R, 32)``) to give every world its own rule.
    Worlds that settle are dropped from the batch, so the cost is set by the
    slowest worlds only.
    """
    cur = np.array(grids, dtype=np.uint8, copy=True)
    n = len(cur)
    rule = Rule.parse(rule)
    transient = np.full(n, -1, dtype=np.int64)
    period = np.full(n, -1, dtype=np.int64)
    density = np.zeros(n)
    activity = np.zeros(n)
    final = np.empty_like(cur)
    cells = cur[0].size
    active = np.arange(n)
    cur_luts = None if luts is None else np.asarray(luts, dtype=np.uint8)
    seen: list[dict] = [{} for _ in range(n)]
    dens_buf = np.zeros((n, window))
    flip_buf = np.zeros((n, window))
    packed = np.packbits(cur.reshape(n, -1), axis=1)
    for i, row in enumerate(packed):
        seen[i][hash(row.tobytes())] = 0
    t = 0
    for t in range(1, max_steps + 1):
        nxt = step(cur, rule, boundary) if cur_luts is None else step_rules(cur, cur_luts, boundary)
        slot = t % window
        flip_buf[:, slot] = (nxt != cur).reshape(len(cur), -1).sum(1) / cells
        dens_buf[:, slot] = nxt.reshape(len(cur), -1).sum(1) / cells
        cur = nxt
        packed = np.packbits(cur.reshape(len(cur), -1), axis=1)
        done = []
        for k, (i, row) in enumerate(zip(active, packed)):
            key = hash(row.tobytes())
            first = seen[i].get(key)
            if first is None:
                seen[i][key] = t
                continue
            p = t - first
            transient[i], period[i] = first, p
            idx = [(t - j) % window for j in range(min(p, window))]  # the final cycle
            density[i] = dens_buf[k, idx].mean()
            activity[i] = flip_buf[k, idx].mean()
            final[i] = cur[k]
            seen[i] = {}
            done.append(k)
        if done:
            keep = np.ones(len(active), dtype=bool)
            keep[done] = False
            cur, active = cur[keep], active[keep]
            dens_buf, flip_buf = dens_buf[keep], flip_buf[keep]
            if cur_luts is not None:
                cur_luts = cur_luts[keep]
        if len(active) == 0:
            break
    if len(active):
        idx = [(t - j) % window for j in range(min(t, window))]  # most recent generations
        for k, i in enumerate(active):
            density[i] = dens_buf[k, idx].mean()
            activity[i] = flip_buf[k, idx].mean()
            final[i] = cur[k]
    return CycleBatch(transient, period, density, activity, final, t)


# ------------------------------------------------------------------ objects
def _union_find_merge(labels: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Merge label ids that appear together in ``pairs`` (shape ``(k, 2)``)."""
    parent = np.arange(labels.max() + 1)

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(len(parent))])
    _, compact = np.unique(roots, return_inverse=True)
    return compact[labels]


def cluster_labels(grid: np.ndarray, wrap: bool = True, reach: int = 1) -> tuple[np.ndarray, int]:
    """Label groups of live cells.

    ``reach=1`` groups cells that touch (including diagonally). ``reach=2``
    also groups cells separated by a single dead cell, which keeps objects
    like the pulsar (whose arms do not touch) in one piece.
    """
    grid = np.asarray(grid, dtype=bool)
    if reach == 2:
        # Grow every cell into a 2x2 block; blocks of cells at Chebyshev
        # distance <= 2 then touch, and labelling the blocks groups them.
        if wrap:
            mask = grid | np.roll(grid, 1, 0) | np.roll(grid, 1, 1) | np.roll(grid, (1, 1), (0, 1))
        else:
            mask = grid.copy()
            mask[1:] |= grid[:-1]
            mask[:, 1:] |= grid[:, :-1]
            mask[1:, 1:] |= grid[:-1, :-1]
    elif reach == 1:
        mask = grid
    else:
        raise ValueError("reach must be 1 or 2")
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if wrap and n:
        h, w = labels.shape
        pairs = []
        for shift in (-1, 0, 1):  # bottom row touches top row (and diagonals)
            a, b = labels[-1], np.roll(labels[0], shift)
            sel = (a > 0) & (b > 0)
            pairs.append(np.stack([a[sel], b[sel]], 1))
            a, b = labels[:, -1], np.roll(labels[:, 0], shift)
            sel = (a > 0) & (b > 0)
            pairs.append(np.stack([a[sel], b[sel]], 1))
        pairs = np.concatenate(pairs)
        if len(pairs):
            labels = _union_find_merge(labels, pairs)
            n = int(labels.max())
    labels = np.where(grid, labels, 0)
    present = np.unique(labels[labels > 0])
    remap = np.zeros(labels.max() + 1, dtype=np.int64)
    remap[present] = np.arange(1, len(present) + 1)
    return remap[labels], len(present)


def extract_objects(grid: np.ndarray, wrap: bool = True, reach: int = 2) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Cut a grid into separate objects. Returns ``(cells, (row, col))`` pairs,
    where ``cells`` is the object cropped to its bounding box."""
    grid = np.asarray(grid, dtype=np.uint8)
    labels, n = cluster_labels(grid, wrap=wrap, reach=reach)
    if n == 0:
        return []
    h, w = grid.shape
    objects = []
    rows_all, cols_all = np.nonzero(labels)
    ids = labels[rows_all, cols_all]
    order = np.argsort(ids, kind="stable")
    rows_all, cols_all, ids = rows_all[order], cols_all[order], ids[order]
    splits = np.flatnonzero(np.diff(ids)) + 1
    for rows, cols in zip(np.split(rows_all, splits), np.split(cols_all, splits)):
        if wrap:
            # Re-centre around the first cell so objects crossing an edge stay whole.
            rows = (rows - rows[0] + h // 2) % h
            cols = (cols - cols[0] + w // 2) % w
        r0, c0 = rows.min(), cols.min()
        cells = np.zeros((rows.max() - r0 + 1, cols.max() - c0 + 1), dtype=np.uint8)
        cells[rows - r0, cols - c0] = 1
        objects.append((cells, (int(r0), int(c0))))
    return objects


def _crop(grid: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(grid.any(1))
    cols = np.flatnonzero(grid.any(0))
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return grid[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def canonical(cells: np.ndarray) -> bytes:
    """A key that is identical for a shape and all its rotations/reflections."""
    cells = _crop(np.asarray(cells, dtype=np.uint8))
    variants = []
    for k in range(4):
        r = np.rot90(cells, k)
        for v in (r, r[:, ::-1]):
            variants.append(np.array(v.shape, dtype=np.uint32).tobytes() + np.packbits(v).tobytes())
    return min(variants)


@dataclass(frozen=True)
class ObjectInfo:
    """What an isolated object does: its name, period and whether it moves."""

    name: str
    kind: str
    period: Optional[int]
    moves: bool
    cells: int
    key: bytes


@lru_cache(maxsize=None)
def _known_objects() -> dict[bytes, tuple[str, str]]:
    known = {}
    for name, pattern in PATTERNS.items():
        if CATEGORY[name] in ("still life", "oscillator", "spaceship"):
            info = _signature(pattern.tobytes(), pattern.shape)
            known[info[0]] = (name, CATEGORY[name])
    return known


@lru_cache(maxsize=4096)
def _signature(raw: bytes, shape: tuple[int, int], max_period: int = 60) -> tuple[bytes, Optional[int], bool]:
    """Simulate a cropped object alone and return (phase-independent key, period, moves)."""
    cells = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    margin = max_period // 4 + 4  # room for a spaceship to travel
    canvas = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin), dtype=np.uint8)
    canvas[margin: margin + shape[0], margin: margin + shape[1]] = cells
    first = _crop(canvas)
    first_pos = np.argwhere(canvas)[0]
    phases = [first]
    for t in range(1, max_period + 1):
        canvas = step(canvas, CONWAY, "dead")
        if not canvas.any():
            return canonical(first), None, False
        shape_t = _crop(canvas)
        if shape_t.shape == first.shape and np.array_equal(shape_t, first):
            moved = not np.array_equal(np.argwhere(canvas)[0], first_pos)
            return min(canonical(p) for p in phases), t, moved
        phases.append(shape_t)
    return canonical(first), None, False


def identify(cells: np.ndarray) -> ObjectInfo:
    """Name an object by simulating it in isolation (Conway's rule)."""
    cells = _crop(np.asarray(cells, dtype=np.uint8))
    key, period, moves = _signature(cells.tobytes(), cells.shape)
    known = _known_objects().get(key)
    if known:
        name, kind = known
    elif period is None:
        name, kind = "unstable", "unstable"
    elif moves:
        name, kind = f"spaceship ({int(cells.sum())} cells)", "spaceship"
    elif period == 1:
        name, kind = f"still life ({int(cells.sum())} cells)", "still life"
    else:
        name, kind = f"p{period} oscillator", "oscillator"
    return ObjectInfo(name, kind, period, moves, int(cells.sum()), key)


def census(grid: np.ndarray, wrap: bool = True) -> Counter:
    """Count the named objects in a (settled) grid.

    Cells within two squares of each other are first grouped (so a pulsar is
    one object). A group that is not a known object is split into touching
    pieces when every piece is stable on its own (for example two blocks
    sitting side by side).
    """
    counts: Counter = Counter()
    for cells, _ in extract_objects(grid, wrap=wrap, reach=2):
        info = identify(cells)
        if info.name in _known_names():
            counts[info.name] += 1
            continue
        pieces = [identify(p) for p, _ in extract_objects(cells, wrap=False, reach=1)]
        if len(pieces) > 1 and all(p.period is not None for p in pieces):
            counts.update(p.name for p in pieces)
        else:
            counts[info.name] += 1
    return counts


@lru_cache(maxsize=None)
def _known_names() -> frozenset:
    return frozenset(name for name, _ in _known_objects().values())
