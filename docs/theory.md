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

## The head: teaching a readout the two zeros

`uncollapsed.head.FieldHead` carries the same idea to arbitrary features: a small
trainable readout that produces `(presence, absence)` evidence per sample, the
four-mass projection, and an explicit route. Nothing about it is exotic — a tanh
hidden layer and a softplus evidence layer — the substance is entirely in the
**training objective**, which has three parts, each one a philosophical
commitment wearing a loss term:

### 1. The evidential fit term — evidence is earned where data is dense

With evidence `e = (eP, eA)`, pseudo-counts `alpha = e + 1`, strength
`S = alpha_P + alpha_A` and mean `m = alpha_P / S`, each labelled sample
contributes

```
(y - m)^2  +  w · m(1 - m) / (S + 1)
```

The first part fits the label. The second — the **variance term**, weighted by
`fit_var_weight` — is the interesting one: it *rewards accumulating evidence*,
because predictive variance shrinks as total evidence `S` grows. At a point
where the labels genuinely contradict (a 50/50 zone), the fit part pins
`m ≈ 0.5` and the variance part keeps paying for **both channels to grow**. A
contradiction therefore ends up with *high* mass in both channels — `conflict`
— rather than low mass in both — `voidness`. That distinction is the entire
point, and this term is what funds it. It decays as `1/(S+1)²` and vanishes
where `m` is near 0 or 1, so it never disturbs clear regions.

The weight cuts both ways, though: because `m(1 − m)` *also* vanishes at the
poles, an over-large `w` relative to the evidence scale (roughly `w > S + 1`,
always true at initialization) turns the poles into attractors for
weakly-anchored contested points — the cheap escape from the variance penalty
is a fake lean, not more evidence. Training data with both clear poles
represented keeps the channels symmetric enough to prevent this; see
[Usage → the pole attractor](usage.md#the-head-fieldhead-on-your-own-features)
for the failure signature and remedies.

This is deliberately close to evidential deep learning (Sensoy, Kaplan &
Kandemir, NeurIPS 2018); see the README's *Related ideas* for the honest
lineage.

### 2. The misleading-evidence tax — clear regions stay clean

Evidence toward the wrong pole of a labelled sample is taxed (annealed in from
zero so early learning is unconstrained). Kept deliberately small: at a genuine
50/50 point *every* unit of evidence is "misleading" half the time, so an
aggressive tax would strangle exactly the conflict mass the variance term is
trying to build. The tax rate must sit below the variance incentive where data
is dense.

### 3. The background-void tax — void is the ground state

Evidence is taxed on samples drawn from a **background measure**. This is the
presence-zero philosophy as a regularizer: the untrained head says *void
everywhere* (its evidence bias starts strongly negative), and presence must be
earned by data. Off-manifold inputs then read as `voidness` instead of a
confident guess.

The background measure matters more than it looks, and getting it wrong is
instructive:

| `bg_mode` | what it draws | when it works | when it fails |
| --- | --- | --- | --- |
| `"box"` | uniform over the inflated feature box | low dimensions | in high dimensions nearly all its mass is empty corners — it barely taxes the near-manifold region where near-OOD actually lives |
| `"shuffle"` | training columns independently permuted | raw **correlated** features | after PCA or whitening the columns are decorrelated, so the shuffle *reproduces the data distribution itself* and taxes evidence everywhere — including on the data |
| `"shell"` | training points + isotropic Gaussian noise (`bg_sigma`) | standardized real features | taxes the *neighborhood around* the manifold — exactly where near-OOD lives — while the fit term defends the data itself |

The shuffle failure is worth internalizing: a density-contrast trick that is
sound on raw features silently becomes self-sabotage after whitening, because
whitening is precisely the operation that makes marginals sufficient.

### Routing: the two zeros made actionable

The route order is deliberate. Lean labels are assigned first and the two
zeros *override* them, so a contradiction is never read as "0.5-ish, call it a
hold" and an empty region is never guessed on:

```
voidness ≥ 0.5  →  gather     (no evidence exists; collect data)
conflict ≥ 0.5  →  escalate   (the evidence is fighting; a human decides)
otherwise       →  presence / absence / hold from the expectation
```

The two overrides can never compete — the four masses sum to 1, so conflict
and voidness cannot both exceed one half.

## Contradiction you cannot memorize

There is a trap in benchmarking conflict, and it is worth naming because it is
a trap in *deploying* it too. If you mark a region as contradictory by giving
its samples **random** labels, a sufficiently flexible model will simply
memorize them — per-sample noise becomes per-sample fake leans, and your
"conflict" evaporates into overfit confidence. (On the digits benchmark a small
head fit random labels at 0.99.)

The honest construction is **contradictory duplication**: every conflict
sample appears *twice, with both labels*. Two annotators, one disagreement —
the static analogue of multi-annotator datasets like CIFAR-10H. Now the
contradiction is *in the data*, and no amount of capacity can fit it away:
the model's only optimum is `m = 0.5` with the variance term filling both
channels. Under the quadratic fit term, duplication is mathematically
identical to a soft `0.5` target; the duplicated presentation is chosen
because it says what it means.

## The two lieutenants

The two zeros are not only an annotation problem. They are the oldest
distinction in fault-tolerant computing, and distributed systems already
**price** them differently (Lamport, Shostak & Pease, 1982):

| fault | what the observer sees | replicas to survive `f` | the right response |
| --- | --- | --- | --- |
| **crash** | silence | `2f + 1` | wait, re-poll, **gather** — do not infer betrayal from a dead link |
| **Byzantine** | contradiction | `3f + 1` | challenge, attest, **escalate** — the evidence itself is fighting |

Silence is cheaper than lies — the gap between those thresholds is literally
the price of conflict over voidness. And the library's non-negotiable collapse
rule has a distributed-systems reading too: FLP impossibility reduces to the
fact that you cannot distinguish *dead* from *slow* — voidness from
not-yet — and every practical system's answer, the timeout, is a **forced
collapse of a void under pressure**. A system that treats timer expiry as
evidence of death is running `allow_pressure_to_break_ties=True`; the famous
failure mode is split-brain. *Pressure is not evidence* is the same sentence
in both worlds.

`uncollapsed.faultbench` runs this as an experiment on real telemetry — a
learnable crash-vs-Byzantine triage head — and the result (triage AUC 1.000
vs. an entropy baseline at ≈0, *inverted*) is on the
[benchmarks page](benchmarks.md).
