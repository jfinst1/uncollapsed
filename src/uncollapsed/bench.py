"""
uncollapsed.bench
=================

The "two zeros" benchmark: can a model tell *contradiction* from *ignorance*?

Setup
-----
:func:`uncollapsed.data.make_two_zeros` builds a world with two clear clusters,
a **conflict** cluster (labels drawn 50/50 -- strong contradictory evidence)
and a **void** ring (no training data at all). Both ambiguous zones deserve
"not a normal answer", but they demand *opposite* actions:

* conflict -> **escalate** (the evidence itself is fighting; a human decides)
* void     -> **gather**   (there is no evidence; go collect data)

Contenders
----------
* **Baseline**: an ordinary MLP of matched capacity trained with BCE. Its only
  uncertainty signal is one scalar (predictive entropy / distance from 0.5).
* **FieldHead**: the two-channel evidential head. Its signal is the four-mass
  split, in which conflict and voidness are different numbers.

Scoring
-------
1. **Clear accuracy** -- both models must classify the easy regions well.
2. **Flagging** -- with each model's ambiguity threshold calibrated to flag at
   most 5% of clear validation points, what fraction of conflict / void points
   does it flag as "not a normal answer"?
3. **Triage AUC** (the headline) -- among truly ambiguous points, how well does
   the model's signal separate conflict from void? One entropy scalar has no
   axis to separate them on; two channels do.

Run it: ``uncollapsed --demo triage`` or :func:`run_benchmark`.
"""
from __future__ import annotations

import numpy as np

from .data import make_two_zeros
from .head import FieldHead
from .net import Adam, sigmoid


class BaselineMLP:
    """A perfectly ordinary tanh MLP + sigmoid readout, BCE loss.

    Matched to the FieldHead in hidden width and training budget. Its
    uncertainty signal is entropy of the predictive probability.
    """

    def __init__(self, in_dim: int, hidden: int = 24, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {
            "W1": rng.normal(0, 1.0 / np.sqrt(in_dim), size=(in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, 1)),
            "b2": np.zeros(1),
        }

    def forward(self, z: np.ndarray):
        z = np.atleast_2d(np.asarray(z, dtype=float))
        h = np.tanh(z @ self.p["W1"] + self.p["b1"])
        logit = h @ self.p["W2"] + self.p["b2"]
        prob = sigmoid(logit)[:, 0]
        return prob, dict(z=z, h=h)

    def fit(self, z: np.ndarray, y: np.ndarray, epochs: int = 3000, lr: float = 0.02):
        y = np.asarray(y, dtype=float).ravel()
        opt = Adam(self.p, lr=lr)
        n = y.shape[0]
        for _ in range(epochs):
            prob, c = self.forward(z)
            dlogit = ((prob - y) / n)[:, None]
            grads = {
                "W2": c["h"].T @ dlogit,
                "b2": dlogit.sum(axis=0),
            }
            dh = dlogit @ self.p["W2"].T
            dpre1 = dh * (1.0 - c["h"] ** 2)
            grads["W1"] = c["z"].T @ dpre1
            grads["b1"] = dpre1.sum(axis=0)
            opt.step(self.p, grads)
        return self

    def entropy(self, z: np.ndarray) -> np.ndarray:
        p = np.clip(self.forward(z)[0], 1e-9, 1 - 1e-9)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney), no sklearn needed."""
    s = np.concatenate([scores_pos, scores_neg])
    ranks = np.empty(len(s))
    # midranks for ties
    order = np.argsort(s)
    sorted_s = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks_sorted[i:j + 1] = ranks_sorted[i:j + 1].mean()
        i = j + 1
    ranks[order] = ranks_sorted
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_benchmark(seed: int = 0, epochs: int = 3000, verbose: bool = True) -> dict:
    """Train both contenders and score them. Returns the metrics dict."""
    Xtr, ytr = make_two_zeros(seed=seed, train=True)
    Xte, yte, zone = make_two_zeros(seed=seed + 999, train=False)
    clear, conflict, void = (zone == "clear"), (zone == "conflict"), (zone == "void")

    head = FieldHead(in_dim=2, hidden=24, seed=seed).fit(Xtr, ytr, epochs=epochs)
    base = BaselineMLP(in_dim=2, hidden=24, seed=seed).fit(Xtr, ytr, epochs=epochs)

    # ------------------------------------------------ 1. clear-region accuracy
    m = head.masses(Xte)
    head_pred = (m["expectation"] >= 0.5).astype(float)
    base_prob = base.forward(Xte)[0]
    base_pred = (base_prob >= 0.5).astype(float)
    head_clear_acc = float(np.mean(head_pred[clear] == yte[clear]))
    base_clear_acc = float(np.mean(base_pred[clear] == yte[clear]))

    # ------------------------------------ 2. flagging at 5% clear false-flags
    # Each model's native "this is not a normal answer" score:
    head_amb = m["conflict"] + m["voidness"]           # the can't-say mass
    base_amb = base.entropy(Xte)
    head_thresh = float(np.quantile(head_amb[clear], 0.95))
    base_thresh = float(np.quantile(base_amb[clear], 0.95))
    head_flag = head_amb > head_thresh
    base_flag = base_amb > base_thresh
    flags = {
        "head_conflict_flagged": float(np.mean(head_flag[conflict])),
        "head_void_flagged": float(np.mean(head_flag[void])),
        "base_conflict_flagged": float(np.mean(base_flag[conflict])),
        "base_void_flagged": float(np.mean(base_flag[void])),
    }

    # --------------------------------------------------------- 3. triage AUC
    # Among truly ambiguous points: score should separate void from conflict.
    # Head: voidness is literally the axis. Baseline: entropy is all it has.
    head_triage = auc(m["voidness"][void], m["voidness"][conflict])
    base_triage = auc(base.entropy(Xte)[void], base.entropy(Xte)[conflict])

    # ------------------------------------------------------------ 4. routing
    routes = head.route(Xte)
    routing = {
        "void_to_gather": float(np.mean(routes[void] == "gather")),
        "conflict_to_escalate": float(np.mean(routes[conflict] == "escalate")),
        "clear_to_lean": float(np.mean(np.isin(routes[clear], ["presence", "absence"]))),
    }

    results = dict(
        head_clear_acc=head_clear_acc, base_clear_acc=base_clear_acc,
        head_triage_auc=head_triage, base_triage_auc=base_triage,
        **flags, **routing,
    )

    if verbose:
        print("two zeros benchmark -- can the model tell contradiction from ignorance?")
        print(f"  clear accuracy          head {head_clear_acc:.3f}   baseline {base_clear_acc:.3f}")
        print("  flagged as 'not a normal answer' (<=5% false-flags on clear):")
        print(f"    conflict zone         head {flags['head_conflict_flagged']:.3f}   "
              f"baseline {flags['base_conflict_flagged']:.3f}")
        print(f"    void zone             head {flags['head_void_flagged']:.3f}   "
              f"baseline {flags['base_void_flagged']:.3f}")
        print("  TRIAGE AUC (void vs conflict, higher is better):")
        print(f"    head (voidness axis)  {head_triage:.3f}")
        print(f"    baseline (entropy)    {base_triage:.3f}")
        print("  head routing:")
        print(f"    void     -> gather    {routing['void_to_gather']:.3f}")
        print(f"    conflict -> escalate  {routing['conflict_to_escalate']:.3f}")
        print(f"    clear    -> lean      {routing['clear_to_lean']:.3f}")
    return results


__all__ = ["run_benchmark", "BaselineMLP", "auc"]
