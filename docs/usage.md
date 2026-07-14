# Usage

## Install

```bash
pip install -e ".[dev]"   # from a clone; core needs only numpy
```

## The field algebra

Reason about the middle without collapsing it:

```python
from uncollapsed import UncollapsedField

void     = UncollapsedField.void()        # nothing there
calm     = UncollapsedField(0.18, 0.18)   # low-energy centre
conflict = UncollapsedField(0.90, 0.90)   # both poles strong
lean     = UncollapsedField(0.85, 0.35)   # a tilt toward presence

for f in (void, calm, conflict, lean):
    m = f.mass()
    print(f.icon().glyph(), f"conflict={m.conflict:.2f} void={m.voidness:.2f}",
          "->", f.collapse().result.value)
```

A balanced contradiction **holds**, and even under forced pressure it **escalates**
rather than defaulting to absence — pressure is recorded, but it is not evidence:

```python
from uncollapsed.field import CollapsePolicy

loaded = UncollapsedField(1.2, 1.2, pressure=1.0, pressure_bias=-1.0)
loaded.collapse(forced=True).result           # Collapse.ESCALATE  (never ABSENCE)

bad = CollapsePolicy(allow_pressure_to_break_ties=True)  # you have to *ask* for it
loaded.collapse(policy=bad, forced=True).result          # Collapse.ABSENCE
```

## The network

Hidden units stay two-channel; only the output edge collapses. Synapses use one
signed weight whose sign **swaps** the channels (balanced-ternary negation).

```python
from uncollapsed.net import train, accuracy, grad_check

grad_check()                                   # ~2e-9: analytic == numeric gradients
net, (_, _, Xte, yte) = train(epochs=3000)     # noisy XOR
accuracy(net, Xte, yte)                         # ~0.99 on UNSEEN points => real learning
```

### Learned abstention

Trained purely to classify, the net is overconfident at the boundary. Supervise the
ambiguous region toward `0.5` and it learns to **hold** exactly where XOR is undecided:

```python
import numpy as np
net_b, _ = train(epochs=3000, gain=2.0, abstain=True)
labels, prob, P, A = net_b.collapse(np.array([[0.5, 0.5]]))
labels[0]                                      # 'hold' at the ambiguous centre
```

## CLI

```bash
uncollapsed --demo all
uncollapsed --demo learn --png surface.png
python -m uncollapsed --demo zero
```

## The head: `FieldHead` on your own features

`FieldHead` is a drop-in readout for **any** feature vector — raw features, or
the penultimate activations of a model you already have. It returns four masses
per sample and an explicit route.

```python
import numpy as np
from uncollapsed import FieldHead

head = FieldHead(in_dim=features.shape[1], bg_mode="shell").fit(features, labels)

routes = head.route(new_features)
# array(['presence', 'gather', 'escalate', ...], dtype=object)

m = head.masses(new_features)
m["conflict"], m["voidness"]        # the two zeros, as separate numbers
m["expectation"]                    # probability-like readout (conflict/void → base rate)
```

If part of your training data is genuinely contradictory — the same *kind* of
input labelled both ways by different annotators — present it as
**contradictory duplication** (each sample twice, once per label). Do **not**
present it as per-sample random labels: a flexible model will memorize those
into fake leans (see [Theory](theory.md#contradiction-you-cannot-memorize)).

```python
X = np.vstack([X_clear, X_contested, X_contested])
y = np.concatenate([y_clear, np.ones(len(X_contested)), np.zeros(len(X_contested))])
head = FieldHead(in_dim=X.shape[1], bg_mode="shell", fit_var_weight=6.0).fit(X, y)
```

!!! warning "Failure mode worth knowing: the pole attractor"
    If your contested points come out with `conflict ≈ 0` and one of
    `belief`/`disbelief` `≈ 1`, the variance incentive is overpowering the fit
    term early in training: while `m(1 − m)/(S + 1)` rewards growing evidence,
    it *also* vanishes at the poles, and when `fit_var_weight` exceeds the
    evidence scale (roughly `w > S + 1`, which is true at initialization) an
    under-anchored contested region can escape to a pole instead of paying for
    evidence. Two remedies, in order: **make sure both clear poles are
    represented in training** (every benchmark in this repository has a clear
    presence *and* a clear absence class — that symmetry is what shapes the
    evidence channels), and if the data cannot provide that, **lower
    `fit_var_weight`** (a single-pole toy that collapses at `w=6` recovers
    `conflict ≈ 0.94` at `w=2`). Diagnose in one line:
    `head.masses(X_contested)` on a group you know is contested.

### Choosing a background mode

| your features | use |
| --- | --- |
| low-dimensional (≲ 3), any scale | `bg_mode="box"` (default) |
| standardized / PCA / network activations | `bg_mode="shell"` |
| raw, strongly correlated columns | `bg_mode="shuffle"` |

Never use `"shuffle"` after PCA or whitening — decorrelated columns make the
shuffled background reproduce the data itself, taxing evidence everywhere.

### Tuning, honestly

The defaults are sane for the synthetic case; the real-data benchmarks use
`fit_var_weight=6.0, lambda_bg=0.03, lambda_mis=0.002, bg_sigma=2.5` (in
standardized feature units). The knobs interact in one specific way worth
knowing:

| knob | raises | at the cost of |
| --- | --- | --- |
| `fit_var_weight` ↑ | conflict mass at contested points | can leak evidence into the near-manifold surround |
| `lambda_bg` ↑ | voidness off-manifold | suppresses conflict mass (the tax and the incentive fight over the same region) |
| `bg_sigma` ↑ | how far the void tax reaches | too far ≈ box mode; too near taxes the data |
| `lambda_mis` ↑ | clean leans in clear regions | strangles conflict mass (at a 50/50 point *all* evidence is misleading half the time) |

Diagnostics are cheap: `head.masses(X)` per known group, and
`grad_check_head()` (~1e-7) if you modify the loss.

## Benchmarks from the CLI

```bash
uncollapsed --demo triage                     # synthetic two zeros + baseline
uncollapsed --demo triage --png masses.png    # + the five-panel mass-map figure
uncollapsed --demo real                       # digits (needs uncollapsed[bench])
uncollapsed --demo real --dataset fashion     # Fashion-MNIST (downloads once)
uncollapsed --demo faults                     # crash vs Byzantine on real telemetry
```

All results, protocols, and honest boundaries: [Benchmarks](benchmarks.md).

## Data cache

`realbench` and `faultbench` fetch their corpora once (Fashion-MNIST ~30 MB,
Intel Lab telemetry ~33 MB) into `~/.cache/uncollapsed`, override with the
`UNCOLLAPSED_CACHE` environment variable. The digits benchmark ships inside
scikit-learn and needs no download. Tests that need a download auto-skip when
it is unavailable.

## Extras

| install | adds | needed for |
| --- | --- | --- |
| `uncollapsed` | numpy only | field algebra, network, head, faultbench |
| `uncollapsed[viz]` | matplotlib | `--png` figures |
| `uncollapsed[bench]` | scikit-learn | the digits benchmark |
| `uncollapsed[dev]` | pytest, ruff, matplotlib, pillow, scikit-learn | contributing |
| `uncollapsed[docs]` | mkdocs-material, mkdocstrings | building this site |
