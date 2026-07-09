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
net_b.collapse(np.array([[0.5, 0.5]]))         # -> 'hold' at the ambiguous centre
```

## CLI

```bash
uncollapsed --demo all
uncollapsed --demo learn --png surface.png
python -m uncollapsed --demo zero
```
