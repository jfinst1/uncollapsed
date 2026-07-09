"""Tests for the field algebra."""
from uncollapsed.algebra import field_and, field_not, field_or, fuse_consensus, fuse_conservative
from uncollapsed.field import UncollapsedField


def test_not_swaps_channels():
    yes = UncollapsedField.from_bool(True)
    n = field_not(yes)
    assert n.presence == yes.absence
    assert n.absence == yes.presence
    assert n.bias < 0


def test_and_of_yes_and_no_is_no():
    yes = UncollapsedField.from_bool(True)
    no = UncollapsedField.from_bool(False)
    assert field_and([yes, no]).bias < 0


def test_or_of_yes_and_no_is_yes():
    yes = UncollapsedField.from_bool(True)
    no = UncollapsedField.from_bool(False)
    assert field_or([yes, no]).bias > 0


def test_conservative_fusion_of_opposites_is_high_conflict():
    yes = UncollapsedField.from_bool(True)
    no = UncollapsedField.from_bool(False)
    assert fuse_conservative([yes, no]).mass().conflict > 0.6


def test_consensus_fusion_stays_balanced():
    yes = UncollapsedField.from_bool(True)
    no = UncollapsedField.from_bool(False)
    assert abs(fuse_consensus([yes, no]).bias) < 0.2
