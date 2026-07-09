# Theory: the problem of zero

This library exists because of a single observation: **a `0` is not one thing.**

In a two-state (binary) system you never have to decide what `0` *means* -- it is
just "not-1", the other option, off, false-by-default. It is defined
oppositionally, for free. The moment you allow a third possibility, `0` has to
mean something on its own, and it turns out `0` is *plural*: additive identity,
false/off, unknown, null, high-impedance, ground, abstain, coordinate origin.
These are different ideas wearing one glyph, and they do not share a truth table.

So this project does not try to *discover* the true meaning of `0`. It makes you
**declare** it -- and it represents the middle as a first-class state rather than
a tie to be broken.

## Presence and absence, kept apart

The core move is to stop storing a single scalar in `[-1, +1]` and instead keep
two independent non-negative potentials: **presence** and **absence**. Bias is
*derived* from them and never stored. This is the one decision everything else
depends on, because it lets four very different states live behind the same
visible `0`:

| state          | presence | absence | what it is                       |
|----------------|:--------:|:-------:|----------------------------------|
| void           |   low    |   low   | nothing there (a true empty zero)|
| calm centre    |  small   |  small  | low-energy balance               |
| contradiction  |   high   |   high  | both poles strongly present      |
| directional    |  differ  |  differ | a lean toward presence/absence   |

## Four masses

From the two channels we project a normalized four-mass accounting -- this is
subjective-logic / Dempster-Shafer flavoured, with **conflict** and **voidness**
promoted to first-class masses:

```
belief    = sp (1 - sa)      # only presence
disbelief = sa (1 - sp)      # only absence
conflict  = sp sa            # both (loaded contradiction)
voidness  = (1 - sp)(1 - sa) # neither (nothing there)
```

for saturated channels `sp = 1 - e^-presence`, `sa = 1 - e^-absence`. They sum to
exactly 1 by construction (the joint distribution of two independent Bernoulli
channels). Two corrections matter:

* **`expectation` projects both conflict and voidness to the base rate.** A fully
  loaded contradiction reads `~0.5`, never `0.0`. A balanced "yes and no" is not
  secretly a "no".
* **`voidness` is high only when both channels are weak.** A confident-but-quiet
  lean is not mislabelled as mostly void.

## Collapse only at the edge

Nothing collapses until something forces it. Collapse is an explicit operation at
the boundary, with a rich outcome space -- `PRESENCE`, `ABSENCE`, `HOLD`,
`ESCALATE`, `ABSTAIN`, `EMPTY` -- and one non-negotiable rule: a genuinely
balanced or loaded state is preserved (`HOLD`) or escalated, **never silently
defaulted to absence**. Pressure is recorded but is *not* treated as evidence:
you cannot bully a real contradiction into resolving; only aligned evidence moves
it.

## Learning without collapsing

`uncollapsed.net` shows the same idea in a trainable network: hidden units carry
`(presence, absence)` all the way through and only the output edge collapses. A
synapse uses one signed weight whose sign *swaps* the channels (balanced-ternary
negation), made differentiable. Trained on noisy XOR and evaluated on unseen
points, it learns the function (not a lookup), and -- when the ambiguous boundary
is supervised -- it learns to **hold** where the answer is genuinely undecided
rather than guess. Holding, like everything else about the middle, has to be
built on purpose; you do not get it for free.
