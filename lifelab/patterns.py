"""Named Game of Life patterns plus RLE reading and writing.

Patterns are stored as plaintext pictures (``O`` = alive, ``.`` = dead) so
they can be checked by eye. :func:`parse_rle` and :func:`to_rle` handle the
run-length encoded format used by most Life software and pattern
collections, so any pattern from the wider Life community can be loaded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "PATTERNS",
    "CATEGORY",
    "get_pattern",
    "parse_plaintext",
    "parse_rle",
    "to_rle",
    "load",
    "place",
    "transform",
]


def parse_plaintext(text: str) -> np.ndarray:
    """Turn a picture made of ``O``/``*`` (alive) and ``.`` (dead) into an array."""
    rows = [line.strip() for line in text.strip().splitlines()]
    rows = [r for r in rows if r and not r.startswith("!")]
    width = max(len(r) for r in rows)
    grid = np.zeros((len(rows), width), dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, ch in enumerate(row):
            grid[i, j] = ch in "O*o#"
    return grid


# fmt: off
_PLAINTEXT = {
    # --- still lifes -------------------------------------------------------
    "block":            "OO\nOO",
    "beehive":          ".OO.\nO..O\n.OO.",
    "loaf":             ".OO.\nO..O\n.O.O\n..O.",
    "boat":             "OO.\nO.O\n.O.",
    "ship":             "OO.\nO.O\n.OO",
    "tub":              ".O.\nO.O\n.O.",
    "pond":             ".OO.\nO..O\nO..O\n.OO.",
    "long_boat":        "OO..\nO.O.\n.O.O\n..O.",
    "barge":            ".O..\nO.O.\n.O.O\n..O.",
    "mango":            ".OO..\nO..O.\n.O..O\n..OO.",
    "long_ship":        "OO..\nO.O.\n.O.O\n..OO",
    "snake":            "OO.O\nO.OO",
    "aircraft_carrier": "OO..\nO..O\n..OO",
    "eater_1":          "OO..\nO.O.\n..O.\n..OO",
    "long_barge":       ".O...\nO.O..\n.O.O.\n..O.O\n...O.",
    # --- oscillators -------------------------------------------------------
    "blinker":          "OOO",
    "toad":             ".OOO\nOOO.",
    "beacon":           "OO..\nOO..\n..OO\n..OO",
    "pulsar": """
        ..OOO...OOO..
        .............
        O....O.O....O
        O....O.O....O
        O....O.O....O
        ..OOO...OOO..
        .............
        ..OOO...OOO..
        O....O.O....O
        O....O.O....O
        O....O.O....O
        .............
        ..OOO...OOO..""",
    "pentadecathlon":   "..O....O..\nOO.OOOO.OO\n..O....O..",
    # --- spaceships --------------------------------------------------------
    "glider":           ".O.\n..O\nOOO",
    "lwss":             ".O..O\nO....\nO...O\nOOOO.",
    "mwss":             "...O..\n.O...O\nO.....\nO....O\nOOOOO.",
    "hwss":             "...OO..\n.O....O\nO......\nO.....O\nOOOOOO.",
    # --- methuselahs (small seeds with long lives) -------------------------
    "r_pentomino":      ".OO\nOO.\n.O.",
    "acorn":            ".O.....\n...O...\nOO..OOO",
    "diehard":          "......O.\nOO......\n.O...OOO",
    "pi_heptomino":     "OOO\nO.O\nO.O",
    # --- guns ----------------------------------------------------------------
    "gosper_glider_gun": """
        ........................O...........
        ......................O.O...........
        ............OO......OO............OO
        ...........O...O....OO............OO
        OO........O.....O...OO..............
        OO........O...O.OO....O.O...........
        ..........O.....O.......O...........
        ...........O...O....................
        ............OO......................""",
}
# fmt: on

#: Every built-in pattern as a ``uint8`` array.
PATTERNS: dict[str, np.ndarray] = {name: parse_plaintext(t) for name, t in _PLAINTEXT.items()}

#: What kind of object each built-in pattern is.
CATEGORY: dict[str, str] = {}
for _name in PATTERNS:
    if _name in ("blinker", "toad", "beacon", "pulsar", "pentadecathlon"):
        CATEGORY[_name] = "oscillator"
    elif _name in ("glider", "lwss", "mwss", "hwss"):
        CATEGORY[_name] = "spaceship"
    elif _name in ("r_pentomino", "acorn", "diehard", "pi_heptomino"):
        CATEGORY[_name] = "methuselah"
    elif _name.endswith("_gun"):
        CATEGORY[_name] = "gun"
    else:
        CATEGORY[_name] = "still life"


def get_pattern(name: str) -> np.ndarray:
    """Return a copy of a built-in pattern by name (case/spacing insensitive)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"gosper": "gosper_glider_gun", "gun": "gosper_glider_gun", "r": "r_pentomino",
               "rpentomino": "r_pentomino"}
    key = aliases.get(key, key)
    if key not in PATTERNS:
        raise KeyError(f"unknown pattern {name!r}; choose from {sorted(PATTERNS)}")
    return PATTERNS[key].copy()


# --------------------------------------------------------------------------- RLE
_RLE_TOKEN = re.compile(r"(\d*)([bo$!A-Za-z.])")


def parse_rle(text: str) -> tuple[np.ndarray, Optional[str]]:
    """Parse a run-length encoded pattern. Returns ``(grid, rule_or_None)``.

    Lines starting with ``#`` are comments. The header looks like
    ``x = 3, y = 3, rule = B3/S23``.
    """
    width = height = None
    rule = None
    body = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("x") and "=" in line and width is None:
            fields = dict(
                (k.strip().lower(), v.strip())
                for k, v in (part.split("=", 1) for part in line.split(",") if "=" in part)
            )
            width, height = int(fields["x"]), int(fields["y"])
            rule = fields.get("rule")
            continue
        body.append(line)
    data = "".join(body)
    rows: list[list[int]] = [[]]
    for count, tag in _RLE_TOKEN.findall(data):
        n = int(count) if count else 1
        if tag == "!":
            break
        if tag == "$":
            rows.extend([] for _ in range(n))
        elif tag in "b.":
            rows[-1].extend([0] * n)
        else:  # "o" or any other letter means a live cell
            rows[-1].extend([1] * n)
    if width is None:
        width = max(len(r) for r in rows)
        height = len(rows)
    grid = np.zeros((height, width), dtype=np.uint8)
    for i, row in enumerate(rows[:height]):
        grid[i, : len(row)] = row[:width]
    return grid, rule


def to_rle(grid: np.ndarray, rule: str = "B3/S23") -> str:
    """Encode ``grid`` as RLE text (inverse of :func:`parse_rle`)."""
    grid = np.asarray(grid, dtype=np.uint8)
    height, width = grid.shape
    lines = []
    for row in grid:
        runs = []
        trimmed = row[: (np.flatnonzero(row).max() + 1) if row.any() else 0]
        i = 0
        while i < len(trimmed):
            j = i
            while j < len(trimmed) and trimmed[j] == trimmed[i]:
                j += 1
            n = j - i
            runs.append(f"{n if n > 1 else ''}{'o' if trimmed[i] else 'b'}")
            i = j
        lines.append("".join(runs))
    # Merge empty rows into counted "$" separators.
    out, blank = [], 0
    for k, line in enumerate(lines):
        if k == 0:
            out.append(line)
            continue
        if line:
            out.append(f"{blank + 1 if blank else ''}$" + line)
            blank = 0
        else:
            blank += 1
    body = "".join(out) + "!"
    wrapped = "\n".join(body[i: i + 70] for i in range(0, len(body), 70))
    return f"x = {width}, y = {height}, rule = {rule}\n{wrapped}\n"


def load(path: str | Path) -> np.ndarray:
    """Load a ``.rle`` or plaintext ``.cells`` file."""
    text = Path(path).read_text()
    if Path(path).suffix.lower() == ".rle":
        return parse_rle(text)[0]
    return parse_plaintext(text)


# --------------------------------------------------------------------- placement
def transform(pattern: np.ndarray, rotate: int = 0, flip: bool = False) -> np.ndarray:
    """Rotate by ``rotate`` quarter turns (counter-clockwise) and optionally mirror."""
    out = np.rot90(pattern, rotate)
    return np.ascontiguousarray(out[:, ::-1] if flip else out)


def place(
    grid: np.ndarray,
    pattern: np.ndarray | str,
    top: int,
    left: int,
    wrap: bool = True,
) -> np.ndarray:
    """Stamp ``pattern`` onto ``grid`` (in place) with its corner at ``(top, left)``.

    Live cells of the pattern are OR-ed onto the grid. With ``wrap=True``
    the pattern wraps around the edges of a toroidal grid.
    """
    if isinstance(pattern, str):
        pattern = get_pattern(pattern)
    rows, cols = np.nonzero(pattern)
    rows = rows + top
    cols = cols + left
    if wrap:
        rows %= grid.shape[0]
        cols %= grid.shape[1]
    else:
        keep = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
        rows, cols = rows[keep], cols[keep]
    grid[rows, cols] = 1
    return grid
