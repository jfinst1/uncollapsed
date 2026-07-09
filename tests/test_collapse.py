"""Tests for edge collapse. The headline guarantee: a balanced state is never
silently defaulted to absence."""
from uncollapsed.field import Collapse, CollapsePolicy, UncollapsedField


def test_clear_presence_collapses_to_presence():
    assert UncollapsedField.from_bool(True).collapse().result is Collapse.PRESENCE


def test_clear_absence_collapses_to_absence():
    assert UncollapsedField.from_bool(False).collapse().result is Collapse.ABSENCE


def test_void_collapses_to_empty():
    assert UncollapsedField.void().collapse().result is Collapse.EMPTY


def test_loaded_contradiction_holds_without_demand():
    assert UncollapsedField(0.9, 0.9).collapse().result is Collapse.HOLD


def test_pressure_never_defaults_a_contradiction_to_absence():
    # A real contradiction under forced pressure must ESCALATE, not become "no".
    loaded = UncollapsedField(1.2, 1.2, pressure=1.0, pressure_bias=-1.0)
    r = loaded.collapse(forced=True).result
    assert r is Collapse.ESCALATE
    assert r is not Collapse.ABSENCE


def test_bad_policy_can_be_asked_for_explicitly():
    loaded = UncollapsedField(1.2, 1.2, pressure=1.0, pressure_bias=-1.0)
    bad = CollapsePolicy(allow_pressure_to_break_ties=True)
    r = loaded.collapse(policy=bad, forced=True).result
    assert r is Collapse.ABSENCE  # only because it was explicitly permitted


def test_active_hold_abstains():
    held = UncollapsedField.hold(0.9)
    assert held.collapse().result in (Collapse.HOLD, Collapse.ABSTAIN)
