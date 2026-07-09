# uncollapsed

**Computation that keeps *presence* and *absence* apart, and collapses to a decision only at the edge — with a genuine, first-class _hold_.**

A visible `0` is not one thing. In binary you never have to decide what `0` *means* — it's just "not-1", off, false-by-default. Add a third possibility and `0` turns out to be **plural**: additive identity, false, unknown, null, high-impedance, ground, abstain, origin. Those are different ideas wearing one glyph, and they don't share a truth table.

`uncollapsed` stops storing a single scalar in `[-1, +1]` and instead keeps **two independent non-negative channels — presence and absence** — so four very different states can live behind the same visible `0`, and so a decision is a deliberate act at the boundary rather than a default that quietly happens.

![Learning to hold: the gold band is where the model abstains](assets/training.gif)

## Where to go next

- **[Theory](theory.md)** — why a `0` is plural, the four-mass accounting, and why collapse belongs at the edge.
- **[Usage](usage.md)** — install, the field algebra, and training the two-channel network.
- **[API reference](api.md)** — the full generated API.

## Install

```bash
pip install -e ".[dev]"
```

## The one-paragraph pitch

The whole point of reaching past binary is to get a **presence-zero** — a held, central state that actively means something — instead of the **absence-zero** binary hands you (a pole, defined by negation, that quietly means "no"). Binary is impatient: every bit is a box already opened. `uncollapsed` holds the question open and resolves only when there is real force to resolve it, and never defaults a genuine contradiction to "no".
