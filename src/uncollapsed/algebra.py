"""
uncollapsed.algebra
===================

Logical and evidential operators over :class:`UncollapsedField` objects that
preserve the two-channel structure. Nothing here collapses; results are still
uncollapsed fields.

* ``field_not`` -- swap presence and absence (balanced-ternary negation).
* ``field_and`` / ``field_or`` -- min/max-flavoured Kleene logic on the
  saturated channels (AND needs *all* present; OR needs *any* present).
* ``fuse_conservative`` -- add evidence; genuine disagreement becomes conflict.
* ``fuse_consensus`` -- average opinions; cross-source disagreement is kept as
  conflict rather than cancelling out.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .field import UncollapsedField, _inv_sat, _sat


def field_not(f: UncollapsedField, *, source: str | None = None) -> UncollapsedField:
    return UncollapsedField(
        presence=f.absence,
        absence=f.presence,
        resistance=f.resistance,
        pressure=f.pressure,
        pressure_bias=-f.pressure_bias,
        source=source or f"NOT({f.source})",
    )


def field_and(fields: Sequence[UncollapsedField], *, source: str = "AND") -> UncollapsedField:
    if not fields:
        return UncollapsedField.void(source=source)
    # presence: all present  -> product of saturated presence
    # absence:  any absent    -> 1 - product(1 - saturated absence)
    sp = math.prod(_sat(f.presence) for f in fields)
    sa = 1.0 - math.prod(1.0 - _sat(f.absence) for f in fields)
    return UncollapsedField(_inv_sat(sp), _inv_sat(sa), source=source)


def field_or(fields: Sequence[UncollapsedField], *, source: str = "OR") -> UncollapsedField:
    if not fields:
        return UncollapsedField.void(source=source)
    sp = 1.0 - math.prod(1.0 - _sat(f.presence) for f in fields)
    sa = math.prod(_sat(f.absence) for f in fields)
    return UncollapsedField(_inv_sat(sp), _inv_sat(sa), source=source)


def fuse_conservative(fields: Sequence[UncollapsedField], *, source: str = "fuse_conservative") -> UncollapsedField:
    """Add raw evidence across sources. Two confident opposite sources -> high conflict."""
    if not fields:
        return UncollapsedField.void(source=source)
    p = sum(f.presence for f in fields)
    a = sum(f.absence for f in fields)
    resistance = sum(f.resistance for f in fields) / len(fields)
    return UncollapsedField(p, a, resistance=resistance, source=source)


def fuse_consensus(fields: Sequence[UncollapsedField], *, source: str = "fuse_consensus") -> UncollapsedField:
    """Average opinions; keep cross-source disagreement as conflict, not cancellation."""
    if not fields:
        return UncollapsedField.void(source=source)
    masses = [f.mass() for f in fields]
    n = len(masses)
    belief = sum(m.belief for m in masses) / n
    disbelief = sum(m.disbelief for m in masses) / n
    conflict = sum(m.conflict for m in masses) / n

    cross = 0.0
    for i, mi in enumerate(masses):
        for mj in masses[i + 1:]:
            cross += mi.belief * mj.disbelief + mi.disbelief * mj.belief
    pairs = max(1.0, n * (n - 1) / 2.0)
    conflict = min(1.0, conflict + cross / pairs)

    sp = min(1.0, belief + conflict)
    sa = min(1.0, disbelief + conflict)
    resistance = sum(f.resistance for f in fields) / n
    return UncollapsedField(_inv_sat(sp), _inv_sat(sa), resistance=resistance, source=source)


__all__ = ["field_not", "field_and", "field_or", "fuse_conservative", "fuse_consensus"]
