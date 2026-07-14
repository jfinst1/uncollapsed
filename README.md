# uncollapsed

**Computation that keeps *presence* and *absence* apart, and collapses to a decision only at the edge — with a genuine, first-class _hold_.**

[![ci](https://github.com/jfinst1/uncollapsed/actions/workflows/ci.yml/badge.svg)](https://github.com/jfinst1/uncollapsed/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![style: ruff](https://img.shields.io/badge/style-ruff-000000)](https://github.com/astral-sh/ruff)

A visible `0` is not one thing. In binary you never have to decide what `0` *means* — it's just "not‑1", off, false‑by‑default. Add a third possibility and `0` turns out to be **plural**: additive identity, false, unknown, null, high‑impedance, ground, abstain, origin. Those are different ideas wearing one glyph, and they don't share a truth table.

`uncollapsed` takes the middle seriously. It stops storing a single scalar in `[-1, +1]` and instead keeps **two independent non‑negative channels — presence and absence** — so four very different states can live behind the same visible `0`, and so a decision is a deliberate act at the boundary rather than a default that quietly happens.

```
      the same visible 0, four different inner states
      ┌────────────┬──────────┬──────────┬──────────────────────────┐
      │  void      │ presence │ absence  │  nothing there           │
      │  calm      │  small   │  small   │  low-energy balance      │
      │  conflict  │  high    │  high    │  both poles strong  ← !  │
      │  lean      │  differ  │  differ  │  a directional tilt      │
      └────────────┴──────────┴──────────┴──────────────────────────┘
```

<p align="center">
  <img src="assets/training.gif" alt="A two-channel network learning XOR and learning to hold: the gold band is where it abstains" width="440">
  <br>
  <em>A two-channel net learning noisy XOR — and learning <b>where to abstain</b>. The gold band is HOLD.</em>
</p>

📖 **Full documentation:** <https://jfinst1.github.io/uncollapsed/>

---

## Why

The whole point of reaching past binary is to get a **presence‑zero** — a held, positive, central state that actively means something — instead of the **absence‑zero** binary hands you (a pole, defined by negation, that quietly means "no"). Binary is impatient: every bit is a box already opened. `uncollapsed` is built to *hold the question open* and resolve only when there is real force to resolve it — and, crucially, never to default a genuine contradiction to "no".

See [`docs/theory.md`](docs/theory.md) for the full background.

## Install

```bash
pip install -e ".[dev]"      # from a clone
# or, minimal:
pip install -e .             # core (numpy only); add [viz] for the plot
```

## Quickstart

### The field algebra — reasoning about the middle

```python
from uncollapsed import UncollapsedField

void     = UncollapsedField.void()             # nothing there
calm     = UncollapsedField(0.18, 0.18)        # low-energy centre
conflict = UncollapsedField(0.90, 0.90)        # both poles strong
lean     = UncollapsedField(0.85, 0.35)        # a tilt toward presence

for f in (void, calm, conflict, lean):
    m = f.mass()
    print(f.icon().glyph(),                    # all four show "0" or a lean
          f"belief={m.belief:.2f} conflict={m.conflict:.2f} void={m.voidness:.2f}",
          "->", f.collapse().result.value)     # ... but collapse very differently
```

A balanced contradiction **holds** instead of collapsing, and — even under forced pressure — it **escalates** rather than defaulting to absence. Pressure is recorded, but it is not treated as evidence:

```python
from uncollapsed.field import CollapsePolicy

loaded = UncollapsedField(1.2, 1.2, pressure=1.0, pressure_bias=-1.0)
loaded.collapse(forced=True).result            # -> Collapse.ESCALATE  (never ABSENCE)

bad = CollapsePolicy(allow_pressure_to_break_ties=True)   # you have to *ask* for the bad behaviour
loaded.collapse(policy=bad, forced=True).result           # -> Collapse.ABSENCE
```

### The network — learning without collapsing

Hidden units carry `(presence, absence)` all the way through; only the output edge collapses. Synapses use one **signed weight whose sign swaps the channels** (balanced‑ternary negation), made differentiable.

```python
from uncollapsed.net import train, accuracy, grad_check

print(grad_check())                            # ~2e-9: analytical == numerical gradients

net, (_, _, Xte, yte) = train(epochs=3000)     # noisy XOR
accuracy(net, Xte, yte)                         # ~0.99 on UNSEEN noisy points => real learning
```

### CLI

```bash
uncollapsed --demo all
uncollapsed --demo learn --png surface.png
python -m uncollapsed --demo zero
```

## Results

Running `uncollapsed --demo learn`:

- **Gradient check: `max relative error ≈ 2e-9`.** The hand‑derived backprop matches numerical gradients — the learning is correct, not hand‑wavy.
- **Held‑out test accuracy ≈ 0.99** on noisy points the model *never saw during training*. Ninety‑nine percent on unseen data means it learned the XOR **function**, not a lookup table over four corners.
- **Learned abstention.** Trained purely to classify, the net is *overconfident* — it guesses at the ambiguous centre. Supervise the boundary toward `0.5` and it learns to **hold** exactly where XOR is genuinely undecided, while staying accurate on the clear corners (`~0.998`).

The gold band below is the region the trained model collapses to **HOLD** — it abstains precisely along the lines where the answer is undecided. That gold region *is* presence‑zero, learned from data.

![Learned uncollapsed XOR surface: red = presence, blue = absence, gold = HOLD](assets/xor_surface.png)

## The head — "I'm not ready to answer yet" as a drop-in layer

`FieldHead` is a small trainable readout you can bolt onto **any** feature vector (raw features, or the penultimate activations of a model you already have). It produces the four masses per sample and an explicit **route**:

| route      | meaning                                        | the right next action |
| ---------- | ---------------------------------------------- | --------------------- |
| `presence` / `absence` | a clear lean                       | act on it             |
| `hold`     | weak lean, genuinely undecided                 | abstain               |
| `escalate` | **conflict** — strong evidence for *both* poles | a human decides       |
| `gather`   | **voidness** — evidence for *neither*           | collect data          |

```python
from uncollapsed import FieldHead

head = FieldHead(in_dim=features.shape[1]).fit(features, labels)
routes = head.route(new_features)      # "presence" | "absence" | "hold" | "escalate" | "gather"
masses = head.masses(new_features)     # belief / disbelief / conflict / voidness per sample
```

The training objective encodes the library's philosophy directly: an evidential fit term *rewards* accumulating evidence where data is dense (so contradiction zones fill **both** channels instead of neither), while a background-void tax makes evidence cost something everywhere else — **void is the ground state; presence must be earned by data.** Gradients are hand-derived and verified (`grad_check_head()` ≈ 1e-7).

## The two zeros benchmark

Why carry two channels instead of one uncertainty scalar? Because "I can't say" is **two different situations** demanding **opposite actions** — and a single scalar cannot tell them apart:

* **conflict** — strong contradictory evidence → *escalate to a human*
* **voidness** — no evidence at all (off the data manifold) → *go gather data*

`uncollapsed --demo triage` builds a world with two clear clusters, a 50/50-label **conflict** cluster, and an off-manifold **void** ring, then scores `FieldHead` against a capacity-matched vanilla MLP whose only signal is predictive entropy:

| metric (seed 0) | FieldHead | entropy baseline |
| --- | --- | --- |
| clear-region accuracy | **1.000** | 1.000 |
| conflict points flagged (≤5% false-flags on clear) | **1.000** | 1.000 |
| void points flagged | **1.000** | 0.580 |
| **triage AUC** — separating void from conflict | **0.993** | 0.019 |
| routing: void → `gather` | **0.930** | — |
| routing: conflict → `escalate` | **0.755** | — |

Two results worth staring at. First, the baseline is **confidently wrong on 42% of off-manifold points** — its sigmoid saturates far from the data, so entropy reads *low* exactly where the model knows least. Second, the triage AUC of **0.019**: entropy doesn't merely fail to separate the two zeros, it points the *wrong way* (conflict points look "more uncertain" than void points, which look confident). The two-channel head separates them at **0.993** because voidness is literally an axis of its state, not a property it has to fake with one number.

[![Five-panel figure: belief, disbelief, conflict, and voidness mass maps plus the routing map](assets/two_zeros_masses.png)](assets/two_zeros_masses.png)
*Contradiction and ignorance light up different mass maps — and route to different actions. Orange = ESCALATE, grey = GATHER.*

These tests ship in `tests/test_head.py`: the claims above are asserted, not just described.

### It survives real data

The same protocol runs on two real datasets (`uncollapsed --demo real`, add `--dataset fashion`): two real classes as the clear task, a third class in which **every sample appears with both labels** (two annotators, one disagreement — the static analogue of multi-annotator datasets like CIFAR-10H), and the remaining classes held out of training entirely as **near-OOD void** — the hard kind, sharing pixel statistics with the training data.

| metric (seed 0) | digits (3 vs 8, conflict = 5) | fashion (trouser vs boot, conflict = shirt) |
| --- | --- | --- |
| clear accuracy | 0.977 | 0.998 |
| **triage AUC** — head / entropy baseline | **0.908 / 0.231** | **0.704 / 0.281** |
| conflict → `escalate` | 0.736 | 0.976 |
| void → `gather` | 0.657 | 0.006 |

Digits repeats the synthetic story on real data (seeds 1–2: triage 0.915/0.916 vs 0.215/0.188). Fashion maps the method's honest **boundary**: its never-seen classes are semantically entangled with the training classes, and no input-density method can call something "void" when it sits *on* the manifold. But look at *how* it fails — the per-class routing is structured, not random:

```
t-shirt/top  -> mostly escalate  (0.98)   looks like shirt, the contested class
pullover     -> mostly escalate  (0.97)   ditto
coat         -> mostly escalate  (0.95)   ditto
sneaker      -> mostly absence   (0.98)   looks like ankle boot, so reads as one
```

The head routes unfamiliar inputs by what they *resemble*: contested-looking things go to a human, boot-looking things read as boots. What it cannot do — what nothing operating on input density can do — is detect semantic novelty that overlaps the trained manifold. Both boundaries ship as assertions in `tests/test_realbench.py`, not just prose.

Two mechanisms were added for real data, both continuous with the philosophy: a **shell background** (training points plus noise — the void tax lands on the *neighborhood* of the manifold, where near-OOD lives) and an explicit **evidence-accumulation incentive weight** (`fit_var_weight`) so contradiction zones can outbid the void tax exactly where data is dense; the incentive vanishes where the readout is decided, so clear regions are untouched.

### The two lieutenants: crash vs Byzantine on real telemetry

The two zeros are also a **distributed-systems fault-triage problem**, and distributed systems already price them differently: crash faults (a node goes *silent*) are survivable with 2f+1 replicas, Byzantine faults (a node speaks *contradiction*) cost 3f+1 (Lamport, Shostak & Pease, 1982). Silence is cheaper than lies — and they demand opposite responses. Silence → wait, re-poll, **gather**; do not infer betrayal from a dead link. Contradiction → challenge, attest, **escalate**.

`uncollapsed --demo faults` runs this on real telemetry: the Intel Berkeley Lab sensor deployment (Madden et al., 2004) — 54 motes reporting temperature every ~31 s, fetched once from a public mirror. Instances are (mote, hour) windows featurized *against spatial peers*; the split is temporal (train days 0–6, test days 7–9). There is no public corpus of labelled naturally-occurring Byzantine faults, so per the standard BFT/sensor-fusion methodology the signal statistics are real and the fault models are canonical injections: **drift** (±2.5–5 °C calibration bias — the benign, labelled clear-task fault), **Byzantine** (the mote replays its own trace from 12 h earlier: smooth, plausible, wrong thermal regime — labelled *both ways*, because from one observation you cannot tell whether this unit lies or its peers drifted; that is the two lieutenants problem), and **crash** (75–95 % of reports dropped, remnants genuine — never seen in training).

| metric (seeds 0/1/2) | FieldHead | entropy baseline |
| --- | --- | --- |
| clear accuracy (healthy vs drift) | 0.880 / 0.883 / 0.871 | 0.863 / 0.879 / 0.851 |
| **triage AUC** — crash vs Byzantine | **1.000 / 1.000 / 1.000** | 0.064 / 0.030 / 0.000 |
| crash → `gather` | 1.00 / 1.00 / 1.00 | — |
| Byzantine → `escalate` | 0.93 / 0.88 / 0.94 | — |

The head's separation is perfect and the baseline's is perfectly *inverted*: entropy ranks nearly every silent unit as more confident than every lying one, because the MLP saturates on off-manifold sparse inputs. A monitoring system routing on that scalar would page a human for dead batteries and quietly trust replayed telemetry. Clear accuracy sits at the same ~0.87 ceiling for both models — small drifts are genuinely confusable with real thermal gradients, which is the honest residual, not a tuning artifact. Assertions in `tests/test_faultbench.py`.

## Four‑mass accounting

From the two channels (`sp = 1 - e^-presence`, `sa = 1 - e^-absence`):

| mass        | formula            | meaning                              |
|-------------|--------------------|--------------------------------------|
| `belief`    | `sp (1 - sa)`      | evidence *only* toward presence      |
| `disbelief` | `sa (1 - sp)`      | evidence *only* toward absence       |
| `conflict`  | `sp · sa`          | both strong — a loaded contradiction |
| `voidness`  | `(1 - sp)(1 - sa)` | both weak — nothing there            |

They sum to exactly 1 (the joint distribution of two independent Bernoulli channels). This is subjective‑logic / Dempster–Shafer flavoured, with two corrections that matter:

- **`expectation` projects both conflict *and* voidness to the base rate**, so a fully loaded contradiction reads `~0.5`, never `0.0`. A balanced "yes and no" is not secretly a "no".
- **`voidness` is high only when both channels are weak**, so a confident‑but‑quiet lean isn't mislabelled as mostly void.

## Roadmap

- [ ] Subjective‑logic‑exact conjunction/disjunction operators in `algebra.py`.
- [ ] **Unsupervised abstention** — hold driven by the field's own internal conflict, not by labels. (The interesting open problem.)
- [ ] Multi‑class / vector‑valued fields.
- [x] A field‑gated readout layer usable as a drop‑in "I'm not ready to answer yet" head — see `FieldHead` and the two zeros benchmark.

## Related ideas

This is not built in a vacuum. The two‑channel field is closely related to **subjective logic** (Jøsang's belief/disbelief/uncertainty opinions), **Dempster–Shafer evidence theory** (belief vs. plausibility, and conflict `K`), **intuitionistic fuzzy sets** (membership/non‑membership/hesitation), and **three‑valued logics** (Kleene, Łukasiewicz). The `FieldHead` objective is deliberately close to **evidential deep learning** (Sensoy, Kaplan & Kandemir, *Evidential Deep Learning to Quantify Classification Uncertainty*, NeurIPS 2018) and to subjective logic's **vacuity vs. dissonance** split — the two-zeros distinction is not new to this library, and that literature deserves the credit for the mechanism. The distinctive commitments here are keeping `conflict` and `voidness` first‑class throughout the *entire* computation (not only at a Dirichlet output layer), and treating collapse as an explicit edge operation with an actionable routing vocabulary that can legitimately abstain.

## Citing

```bibtex
@software{uncollapsed,
  author  = {Finstad, Jon},
  title   = {uncollapsed: presence/absence fields and edge collapse with a first-class hold},
  year    = {2026},
  url     = {https://github.com/jfinst1/uncollapsed}
}
```

## License

MIT — see [LICENSE](LICENSE).
