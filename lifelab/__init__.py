"""lifelab: a small laboratory for Conway's Game of Life and its relatives."""

from .analysis import census, find_cycle, identify
from .engine import Life, neighbor_counts, random_grid, step, step_rules
from .patterns import PATTERNS, get_pattern, parse_rle, place, to_rle
from .rules import CONWAY, NAMED_RULES, Rule

__version__ = "1.0.0"

__all__ = [
    "CONWAY",
    "Life",
    "NAMED_RULES",
    "PATTERNS",
    "Rule",
    "census",
    "find_cycle",
    "get_pattern",
    "identify",
    "neighbor_counts",
    "parse_rle",
    "place",
    "random_grid",
    "step",
    "step_rules",
    "to_rle",
]
