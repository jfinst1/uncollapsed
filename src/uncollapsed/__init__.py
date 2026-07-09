"""
uncollapsed
===========

Computation that keeps *presence* and *absence* as independent channels and
collapses to a decision only at the edge -- with a genuine, first-class *hold*.

A visible ``0`` is only a dashboard glyph. Underneath it may be a void (nothing
there), a calm centre, a loaded contradiction (both poles strong), or a
directional lean. This library gives those states distinct, testable accounting
and a collapse policy that never silently defaults a balanced state to "no".

Two entry points:

* :mod:`uncollapsed.field` -- the field/collapse *algebra* (for reasoning and
  introspection): :class:`~uncollapsed.field.UncollapsedField`,
  :class:`~uncollapsed.field.Mass`, and the collapse policy.
* :mod:`uncollapsed.net` -- a trainable two-channel *network* that learns, keeps
  its hidden state uncollapsed, and can be taught to abstain.
"""
from .field import (
    Collapse,
    CollapsePolicy,
    CollapseReport,
    Icon,
    Mass,
    UncollapsedField,
    channels_from_mass,
)

__version__ = "0.1.0"

__all__ = [
    "UncollapsedField",
    "Mass",
    "Collapse",
    "Icon",
    "CollapsePolicy",
    "CollapseReport",
    "channels_from_mass",
    "__version__",
]
