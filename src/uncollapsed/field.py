"""
uncollapsed.field
=================

The core state of the library: an :class:`UncollapsedField` that keeps
``presence`` and ``absence`` as *independent* non-negative potentials and never
stores a single collapsed scalar. From it we derive a normalized four-mass
projection and an explicit, edge-only collapse policy.

Four masses
-----------
For saturated channels ``sp = 1 - e^-presence`` and ``sa = 1 - e^-absence``::

    belief    = sp * (1 - sa)     # evidence *only* toward presence
    disbelief = sa * (1 - sp)     # evidence *only* toward absence
    conflict  = sp * sa           # both strongly present (loaded contradiction)
    voidness  = (1 - sp)(1 - sa)  # both weak (nothing there)

These sum to exactly 1 by construction -- it is the joint distribution of two
independent Bernoulli channels (is-present? x is-absent?). This is
subjective-logic / Dempster-Shafer inspired, but ``conflict`` and ``voidness``
are first-class so that four very different states behind a visible ``0`` stay
distinguishable: void, calm, loaded contradiction, and directional lean.

Two deliberate corrections over the naive prototype
---------------------------------------------------
* ``expectation`` (the probability-like readout) projects **both** conflict and
  voidness to the base rate. A fully conflicted state therefore reads ~0.5, not
  0.0 -- a balanced contradiction is *not* secretly a "no".
* ``voidness`` is high only when **both** channels are weak, so a
  confident-but-quiet state is not mislabelled as mostly void.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dfield
from enum import Enum

EPS = 1e-12


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _sat(x: float) -> float:
    """Saturate a non-negative potential into [0, 1)."""
    return 1.0 - math.exp(-max(0.0, x))


def _inv_sat(v: float) -> float:
    """Inverse of :func:`_sat`; map a [0, 1) value back to a potential."""
    return -math.log(max(EPS, 1.0 - _clamp(v, 0.0, 1.0 - 1e-9)))


class Collapse(Enum):
    """The possible results of an edge collapse."""

    PRESENCE = "presence"
    ABSENCE = "absence"
    HOLD = "hold"                 # legitimate co-presence preserved, no forcing
    ESCALATE = "escalate"         # a demand exists but pressure is not evidence
    ABSTAIN = "abstain"           # actively held open
    EMPTY = "empty"               # no potential exists to collapse


class Icon(Enum):
    """The lossy visible glyph. Never the whole story."""

    ABSENCE = -1
    HOLD = 0
    PRESENCE = 1

    def glyph(self) -> str:
        return {Icon.ABSENCE: "\u2212", Icon.HOLD: "0", Icon.PRESENCE: "+"}[self]


@dataclass(frozen=True)
class Mass:
    """Normalized four-mass projection of a field.

    Invariant (up to floating point): ``belief + disbelief + conflict + voidness == 1``.
    """

    belief: float
    disbelief: float
    conflict: float
    voidness: float
    base_rate: float = 0.5

    @property
    def total(self) -> float:
        return self.belief + self.disbelief + self.conflict + self.voidness

    @property
    def expectation(self) -> float:
        """Probability-like readout for presence.

        Both conflict and voidness are "I can't say from evidence" masses, so both
        project to the base rate. A balanced contradiction reads ~base_rate, never 0.
        """
        return _clamp(self.belief + self.base_rate * (self.conflict + self.voidness))

    @property
    def bias(self) -> float:
        return _clamp(self.belief - self.disbelief, -1.0, 1.0)

    @property
    def uncertainty(self) -> float:
        """Lack-of-evidence mass (the subjective-logic sense) == voidness here."""
        return self.voidness

    def as_dict(self) -> dict[str, float]:
        return {
            "belief": round(self.belief, 6),
            "disbelief": round(self.disbelief, 6),
            "conflict": round(self.conflict, 6),
            "voidness": round(self.voidness, 6),
            "expectation": round(self.expectation, 6),
            "bias": round(self.bias, 6),
            "total": round(self.total, 6),
        }


@dataclass(frozen=True)
class CollapsePolicy:
    """Thresholds and switches governing collapse. Defaults are conservative."""

    direction_threshold: float = 0.55
    weak_direction_threshold: float = 0.15
    conflict_threshold: float = 0.45
    pressure_threshold: float = 0.82
    void_threshold: float = 0.94
    active_hold_resistance: float = 0.85
    # Opt-in bad behaviours, present only so they can be demonstrated/tested:
    allow_pressure_to_break_ties: bool = False
    allow_stochastic_forced_collapse: bool = False
    rng_seed: int = 1337


@dataclass
class CollapseReport:
    result: Collapse
    strength: float
    reason: str
    icon: Icon
    mass: Mass

    def as_dict(self) -> dict[str, object]:
        return {
            "result": self.result.value,
            "strength": round(self.strength, 6),
            "reason": self.reason,
            "icon": self.icon.glyph(),
            "mass": self.mass.as_dict(),
        }


@dataclass
class UncollapsedField:
    """Independent presence/absence potentials. Bias is derived; ``0`` is only an icon."""

    presence: float = 0.0
    absence: float = 0.0
    resistance: float = 0.5
    pressure: float = 0.0
    pressure_bias: float = 0.0
    source: str = "field"
    history: list[str] = dfield(default_factory=list)

    def __post_init__(self) -> None:
        self.presence = max(0.0, float(self.presence))
        self.absence = max(0.0, float(self.absence))
        self.resistance = _clamp(self.resistance)
        self.pressure = _clamp(self.pressure)
        self.pressure_bias = _clamp(self.pressure_bias, -1.0, 1.0)

    # ---- constructors ---------------------------------------------------------
    @classmethod
    def from_bool(cls, value: bool, *, source: str = "input") -> UncollapsedField:
        """A boolean becomes a strong two-channel field. ``False`` is active absence, not void."""
        return cls(1.6, 0.02, source=source) if value else cls(0.02, 1.6, source=source)

    @classmethod
    def void(cls, *, source: str = "void") -> UncollapsedField:
        return cls(0.0, 0.0, resistance=1.0, source=source)

    @classmethod
    def hold(cls, energy: float = 0.9, *, source: str = "active hold") -> UncollapsedField:
        """A loaded, actively held co-presence: both channels strong."""
        e = max(0.0, energy)
        return cls(e, e, resistance=0.92, source=source)

    # ---- derived quantities ---------------------------------------------------
    @property
    def total(self) -> float:
        return self.presence + self.absence

    def mass(self, base_rate: float = 0.5) -> Mass:
        sp, sa = _sat(self.presence), _sat(self.absence)
        return Mass(
            belief=sp * (1.0 - sa),
            disbelief=sa * (1.0 - sp),
            conflict=sp * sa,
            voidness=(1.0 - sp) * (1.0 - sa),
            base_rate=base_rate,
        )

    @property
    def bias(self) -> float:
        return self.mass().bias

    @property
    def conflict(self) -> float:
        return self.mass().conflict

    @property
    def voidness(self) -> float:
        return self.mass().voidness

    def icon(self, policy: CollapsePolicy | None = None) -> Icon:
        p = policy or CollapsePolicy()
        b = self.bias
        if b >= p.direction_threshold:
            return Icon.PRESENCE
        if b <= -p.direction_threshold:
            return Icon.ABSENCE
        return Icon.HOLD

    # ---- the edge -------------------------------------------------------------
    def collapse(
        self,
        *,
        policy: CollapsePolicy | None = None,
        forced: bool = False,
        base_rate: float = 0.5,
        label: str = "edge",
    ) -> CollapseReport:
        """Collapse this field to a decision. The only place a scalar answer is produced.

        A genuinely balanced/loaded state is preserved (``HOLD``) or, under a real
        demand, ``ESCALATE``-d. It is never silently defaulted to ``ABSENCE``.
        """
        pol = policy or CollapsePolicy()
        m = self.mass(base_rate=base_rate)
        icon = self.icon(pol)
        demand = forced or self.pressure >= pol.pressure_threshold

        def rep(result: Collapse, strength: float, reason: str) -> CollapseReport:
            return CollapseReport(result, _clamp(strength), f"{label}: {reason}", icon, m)

        if m.voidness >= pol.void_threshold:
            return rep(Collapse.EMPTY, m.voidness, "void field; nothing to collapse")
        if m.belief >= pol.direction_threshold:
            return rep(Collapse.PRESENCE, m.belief, "presence evidence crossed threshold")
        if m.disbelief >= pol.direction_threshold:
            return rep(Collapse.ABSENCE, m.disbelief, "absence evidence crossed threshold")

        if m.conflict >= pol.conflict_threshold:
            if demand:
                if pol.allow_pressure_to_break_ties and abs(self.pressure_bias) >= pol.weak_direction_threshold:
                    r = Collapse.PRESENCE if self.pressure_bias > 0 else Collapse.ABSENCE
                    return rep(r, abs(self.pressure_bias), "BAD POLICY: pressure broke a loaded zero")
                return rep(Collapse.ESCALATE, max(m.conflict, self.pressure),
                           "pressure met a real contradiction; refused the negative-zero default")
            return rep(Collapse.HOLD, m.conflict, "loaded co-presence preserved")

        if demand:
            if abs(m.bias) >= pol.weak_direction_threshold:
                r = Collapse.PRESENCE if m.bias > 0 else Collapse.ABSENCE
                return rep(r, abs(m.bias), "pressured collapse followed a weak lean")
            if pol.allow_stochastic_forced_collapse:
                import random
                prob = m.expectation
                r = Collapse.PRESENCE if random.Random(pol.rng_seed).random() < prob else Collapse.ABSENCE
                return rep(r, max(prob, 1.0 - prob), "stochastic forced collapse (explicit policy)")
            return rep(Collapse.ESCALATE, self.pressure, "a demand exists, but pressure is not evidence")

        if self.resistance >= pol.active_hold_resistance and abs(m.bias) < pol.weak_direction_threshold:
            return rep(Collapse.ABSTAIN, max(m.conflict, m.voidness), "actively held open")

        return rep(Collapse.HOLD, max(m.conflict, m.voidness), "no legitimate collapse force")

    # ---- display --------------------------------------------------------------
    def meter(self, width: int = 18) -> str:
        p = int(round(_clamp(self.presence) * width))
        a = int(round(_clamp(self.absence) * width))
        return (f"P[{'#' * p}{'.' * (width - p)}]{self.presence:5.3f} "
                f"A[{'#' * a}{'.' * (width - a)}]{self.absence:5.3f}")

    def summary(self, policy: CollapsePolicy | None = None) -> str:
        m = self.mass()
        return (f"{self.source}: {self.meter()} bias={m.bias:+.3f} "
                f"conflict={m.conflict:.3f} void={m.voidness:.3f} "
                f"icon={self.icon(policy).glyph()}")

    def clone(self, **overrides: object) -> UncollapsedField:
        data = dict(presence=self.presence, absence=self.absence, resistance=self.resistance,
                    pressure=self.pressure, pressure_bias=self.pressure_bias,
                    source=self.source, history=list(self.history))
        data.update(overrides)
        return UncollapsedField(**data)  # type: ignore[arg-type]


def channels_from_mass(belief: float, disbelief: float, conflict: float, voidness: float,
                        *, source: str = "field") -> UncollapsedField:
    """Build a field whose saturated channels realize the given (non-void) masses."""
    sp = _clamp(belief + conflict)
    sa = _clamp(disbelief + conflict)
    return UncollapsedField(_inv_sat(sp), _inv_sat(sa), source=source)


__all__ = [
    "UncollapsedField", "Mass", "Collapse", "Icon", "CollapsePolicy",
    "CollapseReport", "channels_from_mass",
]
