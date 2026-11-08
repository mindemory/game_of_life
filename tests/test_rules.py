import numpy as np
import pytest

from lifelab.rules import CONWAY, NAMED_RULES, Rule


@pytest.mark.parametrize("text", ["B3/S23", "b3/s23", " B3 / S23 ", "23/3", "life", "Life", 6152])
def test_parse_conway_spellings(text):
    assert Rule.parse(text) == CONWAY


def test_notation_and_name():
    assert str(CONWAY) == "B3/S23"
    assert CONWAY.name == "life"
    assert Rule.parse("highlife").notation == "B36/S23"
    assert Rule.parse("seeds").notation == "B2/S"
    assert Rule.parse("B2/S").survival == frozenset()


def test_named_rules_parse():
    for name, text in NAMED_RULES.items():
        rule = Rule.parse(name)
        assert rule == Rule.parse(text)
        assert Rule.parse(rule.notation) == rule


def test_code_round_trip():
    rng = np.random.default_rng(0)
    for code in rng.integers(0, 1 << 18, 200):
        rule = Rule.from_code(int(code))
        assert rule.code == code
        assert Rule.parse(rule.notation) == rule


def test_bad_rules():
    for bad in ["B9/S23", "hello", "B3S23x"]:
        with pytest.raises(ValueError):
            Rule.parse(bad)
    with pytest.raises(ValueError):
        Rule.from_code(1 << 18)


def test_langton_lambda_of_life():
    # B3: C(8,3)=56; S2: C(8,2)=28; S3: 56  ->  140 / 512
    assert CONWAY.langton_lambda == pytest.approx(140 / 512)
    assert Rule.parse("B/S").langton_lambda == 0
    assert Rule.parse("B012345678/S012345678").langton_lambda == 1


def test_lut_matches_sets():
    rule = Rule.parse("B36/S125")
    for n in range(9):
        assert rule.lut[n] == (n in rule.birth)
        assert rule.lut[16 + n] == (n in rule.survival)
    assert rule.masks == (sum(1 << n for n in (3, 6)), sum(1 << n for n in (1, 2, 5)))


def test_mutants():
    mutants = dict(CONWAY.mutants())
    assert len(mutants) == 17  # 18 conditions minus B0
    assert mutants["+B6"] == Rule.parse("highlife")
    assert mutants["-S2"] == Rule.parse("B3/S3")
    assert len(CONWAY.mutants(include_b0=True)) == 18


def test_all_rules_count():
    assert sum(1 for _ in Rule.all_rules()) == 1 << 17
