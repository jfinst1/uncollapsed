"""Tests for the field model and the two deliberate corrections over the prototype."""

from uncollapsed.field import UncollapsedField, channels_from_mass


def test_masses_sum_to_one():
    for p in (0.0, 0.2, 0.9, 1.6):
        for a in (0.0, 0.2, 0.9, 1.6):
            m = UncollapsedField(p, a).mass()
            assert abs(m.total - 1.0) < 1e-9


def test_void_is_only_both_channels_weak():
    assert UncollapsedField.void().mass().voidness == 1.0
    # a clear-ish lean must NOT be mostly void (the v-prototype bug)
    lean = UncollapsedField(0.85, 0.35).mass()
    assert lean.voidness < 0.35
    assert lean.belief > lean.disbelief


def test_conflict_high_for_loaded_copresence():
    loaded = UncollapsedField(1.5, 1.5).mass()
    assert loaded.conflict > 0.6
    assert abs(loaded.bias) < 1e-6


def test_expectation_of_balanced_conflict_is_base_rate():
    # the negative-zero leak: a fully-loaded contradiction must read ~0.5, not 0.0
    m = UncollapsedField(3.0, 3.0).mass(base_rate=0.5)
    assert m.conflict > 0.9
    assert abs(m.expectation - 0.5) < 0.05


def test_expectation_of_void_is_base_rate():
    m = UncollapsedField.void().mass(base_rate=0.5)
    assert abs(m.expectation - 0.5) < 1e-9


def test_directional_expectation():
    yes = UncollapsedField.from_bool(True).mass()
    no = UncollapsedField.from_bool(False).mass()
    assert yes.expectation > 0.7
    assert no.expectation < 0.3


def test_channels_from_mass_roundtrip_bias_sign():
    f = channels_from_mass(belief=0.6, disbelief=0.05, conflict=0.1, voidness=0.25)
    assert f.bias > 0
