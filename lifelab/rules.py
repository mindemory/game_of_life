"""Life-like cellular automaton rules written in B/S notation.

A Life-like rule says which neighbour counts make a dead cell come alive
(birth, ``B``) and which let a live cell stay alive (survival, ``S``).
Conway's Game of Life is ``B3/S23``. Counting every possible birth and
survival set gives 2**18 = 262,144 rules, and this module can enumerate them,
parse them, and describe them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from math import comb
from typing import Iterator, Union

import numpy as np

__all__ = ["Rule", "CONWAY", "NAMED_RULES", "RuleLike"]

_BS = re.compile(r"^\s*B(?P<b>[0-8]*)\s*/\s*S(?P<s>[0-8]*)\s*$", re.IGNORECASE)
_SB = re.compile(r"^\s*(?P<s>[0-8]*)\s*/\s*(?P<b>[0-8]*)\s*$")  # classic "23/3" form

#: Well-known Life-like rules (names follow the LifeWiki list of Life-like CA).
NAMED_RULES: dict[str, str] = {
    "life": "B3/S23",
    "highlife": "B36/S23",
    "day_and_night": "B3678/S34678",
    "seeds": "B2/S",
    "maze": "B3/S12345",
    "mazectric": "B3/S1234",
    "coral": "B3/S45678",
    "replicator": "B1357/S1357",
    "life_without_death": "B3/S012345678",
    "diamoeba": "B35678/S5678",
    "2x2": "B36/S125",
    "move": "B368/S245",
    "anneal": "B4678/S35678",
    "34_life": "B34/S34",
    "long_life": "B345/S5",
    "amoeba": "B357/S1358",
    "pseudo_life": "B357/S238",
    "drylife": "B37/S23",
    "stains": "B3678/S235678",
    "coagulations": "B378/S235678",
    "walled_cities": "B45678/S2345",
    "serviettes": "B234/S",
    "gnarl": "B1/S1",
}


def _digits(values) -> str:
    return "".join(str(n) for n in sorted(values))


@dataclass(frozen=True)
class Rule:
    """An outer-totalistic ("Life-like") rule on the Moore neighbourhood."""

    birth: frozenset
    survival: frozenset

    def __post_init__(self) -> None:
        birth = frozenset(int(n) for n in self.birth)
        survival = frozenset(int(n) for n in self.survival)
        if not all(0 <= n <= 8 for n in birth | survival):
            raise ValueError("neighbour counts must be between 0 and 8")
        object.__setattr__(self, "birth", birth)
        object.__setattr__(self, "survival", survival)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def parse(cls, rule: "RuleLike") -> "Rule":
        """Build a rule from ``"B3/S23"``, ``"23/3"``, a name such as
        ``"highlife"``, an 18-bit integer code, or an existing :class:`Rule`."""
        if isinstance(rule, Rule):
            return rule
        if isinstance(rule, (int, np.integer)):
            return cls.from_code(int(rule))
        if not isinstance(rule, str):
            raise TypeError(f"cannot build a Rule from {type(rule).__name__}")
        key = rule.strip().lower().replace(" ", "_").replace("-", "_").replace("&", "and")
        if key in NAMED_RULES:
            return cls.parse(NAMED_RULES[key])
        match = _BS.match(rule) or _SB.match(rule)
        if not match:
            raise ValueError(f"cannot parse rule {rule!r}; expected something like 'B3/S23'")
        return cls(frozenset(map(int, match["b"])), frozenset(map(int, match["s"])))

    @classmethod
    def from_code(cls, code: int) -> "Rule":
        """Inverse of :attr:`code`: bits 0-8 are births, bits 9-17 survivals."""
        if not 0 <= code < 1 << 18:
            raise ValueError("rule codes are 18-bit integers")
        return cls(
            frozenset(n for n in range(9) if code >> n & 1),
            frozenset(n for n in range(9) if code >> (9 + n) & 1),
        )

    @classmethod
    def all_rules(cls, include_b0: bool = False) -> Iterator["Rule"]:
        """Enumerate every Life-like rule (B0 rules make empty space flash and
        are skipped by default)."""
        for code in range(1 << 18):
            if include_b0 or not code & 1:
                yield cls.from_code(code)

    # ------------------------------------------------------------- properties
    @property
    def notation(self) -> str:
        return f"B{_digits(self.birth)}/S{_digits(self.survival)}"

    def __str__(self) -> str:
        return self.notation

    def __repr__(self) -> str:
        return f"Rule('{self.notation}')"

    @property
    def code(self) -> int:
        """Unique 18-bit integer for this rule."""
        return sum(1 << n for n in self.birth) | sum(1 << (9 + n) for n in self.survival)

    @property
    def name(self) -> str | None:
        """Common name of the rule, if it has one."""
        for name, text in NAMED_RULES.items():
            if Rule.parse(text) == self:
                return name
        return None

    @cached_property
    def lut(self) -> np.ndarray:
        """Lookup table of length 32 indexed by ``(state << 4) | neighbours``."""
        table = np.zeros(32, dtype=np.uint8)
        for n in self.birth:
            table[n] = 1
        for n in self.survival:
            table[16 + n] = 1
        return table

    @property
    def masks(self) -> tuple[int, int]:
        """(birth, survival) as 9-bit masks: bit ``n`` set means count ``n`` fires."""
        return sum(1 << n for n in self.birth), sum(1 << n for n in self.survival)

    @property
    def langton_lambda(self) -> float:
        """Langton's lambda: the fraction of the 512 neighbourhood configurations
        that produce a live cell. Conway's Life has lambda = 140/512 = 0.273."""
        live = sum(comb(8, n) for n in self.birth) + sum(comb(8, n) for n in self.survival)
        return live / 512

    # -------------------------------------------------------------- mutations
    def toggled(self, kind: str, count: int) -> "Rule":
        """Return a copy with one birth (``kind='B'``) or survival (``'S'``)
        condition switched on or off."""
        if kind.upper() == "B":
            return Rule(self.birth ^ {count}, self.survival)
        if kind.upper() == "S":
            return Rule(self.birth, self.survival ^ {count})
        raise ValueError("kind must be 'B' or 'S'")

    def mutants(self, include_b0: bool = False) -> list[tuple[str, "Rule"]]:
        """All rules one condition away, labelled like ``'+B6'`` or ``'-S2'``."""
        out = []
        for kind, current in (("B", self.birth), ("S", self.survival)):
            for n in range(9):
                if kind == "B" and n == 0 and not include_b0:
                    continue
                label = f"{'-' if n in current else '+'}{kind}{n}"
                out.append((label, self.toggled(kind, n)))
        return out


RuleLike = Union[Rule, str, int]

#: Conway's Game of Life.
CONWAY = Rule.parse("B3/S23")
