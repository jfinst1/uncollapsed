"""
uncollapsed.head
================

A drop-in, field-gated readout head: the "I'm not ready to answer yet" layer
from the roadmap.

Feed it any feature vector (raw features, or the penultimate activations of a
model you already have) and it produces an :class:`~uncollapsed.field.UncollapsedField`-style
two-channel state per sample -- ``presence`` and ``absence`` evidence -- plus the
four-mass projection and an explicit routing decision. The head never stores a
single collapsed scalar internally; collapse (and abstention) happens only at
the routing edge.

Why a head, and why this loss
-----------------------------
A single uncertainty scalar (entropy, max-softmax) cannot distinguish the two
zeros:

* **conflict** -- strong evidence for *both* poles (contradictory data), and
* **voidness** -- evidence for *neither* (off the data manifold).

Those demand opposite responses (escalate to a human vs. go collect data), so
the head is trained to keep them apart:

1. **Fit term** (evidential, Bayes-risk style): with evidence ``e = (eP, eA)``,
   pseudo-counts ``alpha = e + 1``, strength ``S = sum(alpha)`` and mean
   ``m = alpha_P / S``, each sample contributes::

       (y - m)^2 + w * m(1 - m) / (S + 1)

   where ``w`` is ``fit_var_weight``. The variance term *rewards accumulating
   evidence where data is dense* --
   including 50/50 contradiction zones, which therefore end up with **both**
   channels high (conflict), not both low.

2. **Misleading-evidence penalty** (annealed): evidence toward the wrong pole
   of a labelled sample is taxed, so clear regions stay clean leans.

3. **Background-void penalty**: evidence is taxed on samples drawn from a broad
   background measure covering (and exceeding) the feature range. Void is the
   ground state; presence must be earned by data. This is what makes
   off-manifold inputs read as ``voidness`` instead of a confident guess.

This is deliberately close to evidential deep learning (Sensoy et al., 2018)
and to subjective logic's vacuity/dissonance split -- see README "Related
ideas". The distinctive commitments here are the same as the rest of the
library: two independent non-negative channels, four first-class masses, and
collapse/abstention as an explicit edge operation with a routing vocabulary
(``presence`` / ``absence`` / ``hold`` / ``escalate`` / ``gather``).

All gradients are hand-derived and verified by :func:`grad_check_head`.
"""
from __future__ import annotations

import numpy as np

from .net import sigmoid, softplus

# Routing labels. "escalate" and "gather" are the two zeros made actionable:
#   escalate -> loaded contradiction, a human should look (Collapse.ESCALATE)
#   gather   -> no evidence exists here, collect data     (Collapse.EMPTY)
ROUTES = ("presence", "absence", "hold", "escalate", "gather")


class FieldHead:
    """A small trainable head: features -> (presence, absence) evidence -> route.

    Parameters
    ----------
    in_dim:
        Dimension of the incoming feature vector.
    hidden:
        Width of the single tanh hidden layer.
    seed:
        RNG seed for initialization (and the background sampler).
    lambda_mis:
        Weight of the misleading-evidence penalty (annealed in from 0).
    lambda_bg:
        Weight of the background-void penalty. Larger values demand more data
        before the head will claim evidence anywhere.
    bg_margin:
        For ``bg_mode="box"``: the background measure is a uniform box over
        the training feature range, inflated by this fraction per side.
    bg_mode:
        How background (void-prior) samples are drawn.

        * ``"box"`` -- uniform over the inflated feature box. Good in low
          dimensions; in high dimensions almost all its mass sits in empty
          corners far from the data, so it barely taxes the near-manifold
          region.
        * ``"shuffle"`` -- training features with each column independently
          permuted. Only sensible on *raw correlated* features: after PCA or
          whitening, columns are decorrelated and the shuffle reproduces the
          data distribution itself, taxing evidence everywhere.
        * ``"shell"`` -- training points plus isotropic Gaussian noise of
          scale ``bg_sigma`` (in feature units). Taxes exactly the
          neighborhood *around* the manifold, which is where near-OOD inputs
          (e.g. unseen classes of the same domain) live. The fit term defends
          the data itself. Recommended for standardized real features.
    bg_sigma:
        Noise scale for ``bg_mode="shell"``.
    fit_var_weight:
        Weight on the variance component of the evidential fit term. This is
        the *evidence-accumulation incentive*: it is what pays for filling
        both channels at contradiction points, and it decays as
        ``1/(S+1)^2``, so on real data it must outbid the background tax in
        exactly the regions the tax also covers. It vanishes where ``m`` is
        near 0 or 1, so raising it does not disturb clear regions.
    """

    def __init__(self, in_dim: int, hidden: int = 24, seed: int = 0,
                 lambda_mis: float = 0.005, lambda_bg: float = 0.02,
                 bg_margin: float = 0.5, bg_mode: str = "box", bg_sigma: float = 1.5,
                 fit_var_weight: float = 1.0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.lambda_mis = float(lambda_mis)
        self.lambda_bg = float(lambda_bg)
        self.bg_margin = float(bg_margin)
        assert bg_mode in ("box", "shuffle", "shell"), f"unknown bg_mode: {bg_mode!r}"
        self.bg_mode = bg_mode
        self.bg_sigma = float(bg_sigma)
        self.fit_var_weight = float(fit_var_weight)
        self._bg_lo: np.ndarray | None = None
        self._bg_hi: np.ndarray | None = None
        self._bg_data: np.ndarray | None = None
        self.p: dict[str, np.ndarray] = {
            "W1": rng.normal(0, 1.0 / np.sqrt(in_dim), size=(in_dim, hidden)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, 2)),
            # Start with strongly negative evidence bias: the untrained head
            # says "void" everywhere. Presence must be earned.
            "b2": np.full(2, -2.0),
        }

    # ----------------------------------------------------------------- forward

    def evidence(self, z: np.ndarray) -> tuple[np.ndarray, dict]:
        """Two non-negative evidence channels ``(n, 2)`` = (presence, absence)."""
        z = np.atleast_2d(np.asarray(z, dtype=float))
        pre1 = z @ self.p["W1"] + self.p["b1"]
        h = np.tanh(pre1)
        pre2 = h @ self.p["W2"] + self.p["b2"]
        e = softplus(pre2)
        cache = dict(z=z, h=h, pre2=pre2, e=e)
        return e, cache

    def masses(self, z: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized four-mass projection (same algebra as UncollapsedField)."""
        e, _ = self.evidence(z)
        sp = 1.0 - np.exp(-e[:, 0])
        sa = 1.0 - np.exp(-e[:, 1])
        m = {
            "belief": sp * (1.0 - sa),
            "disbelief": sa * (1.0 - sp),
            "conflict": sp * sa,
            "voidness": (1.0 - sp) * (1.0 - sa),
        }
        total = m["belief"] + m["disbelief"] + m["conflict"] + m["voidness"]
        assert np.allclose(total, 1.0, atol=1e-9), "four masses must sum to 1"
        m["expectation"] = m["belief"] + 0.5 * (m["conflict"] + m["voidness"])
        return m

    def route(self, z: np.ndarray, hold_half: float = 0.15,
              void_thresh: float = 0.5, conflict_thresh: float = 0.5) -> np.ndarray:
        """Explicit edge decision per sample.

        Order matters and is deliberate: lean labels are assigned first and
        the two zeros *override* them, so a contradiction is never silently
        read as "0.5-ish, call it a hold" and an empty region is never
        guessed on. (Conflict and voidness cannot both exceed 0.5 -- the four
        masses sum to 1 -- so the two overrides never compete.)
        """
        m = self.masses(z)
        n = m["belief"].shape[0]
        out = np.empty(n, dtype=object)
        exp = m["expectation"]
        out[:] = "hold"
        out[exp >= 0.5 + hold_half] = "presence"
        out[exp <= 0.5 - hold_half] = "absence"
        out[m["conflict"] >= conflict_thresh] = "escalate"
        out[m["voidness"] >= void_thresh] = "gather"
        return out

    # -------------------------------------------------------------------- loss

    def _bg_sample(self, n: int) -> np.ndarray:
        if self.bg_mode == "shell":
            assert self._bg_data is not None, "fit() must store training features first"
            base = self._bg_data[self.rng.integers(0, self._bg_data.shape[0], size=n)]
            return base + self.rng.normal(0.0, self.bg_sigma, size=base.shape)
        if self.bg_mode == "shuffle":
            assert self._bg_data is not None, "fit() must store training features first"
            idx = self.rng.integers(0, self._bg_data.shape[0],
                                    size=(n, self._bg_data.shape[1]))
            return self._bg_data[idx, np.arange(self._bg_data.shape[1])]
        assert self._bg_lo is not None, "fit() must set the background box first"
        return self.rng.uniform(self._bg_lo, self._bg_hi, size=(n, self._bg_lo.size))

    def loss_and_grads(self, z: np.ndarray, y: np.ndarray, zbg: np.ndarray,
                       lam_mis: float) -> tuple[float, dict[str, np.ndarray]]:
        """Full loss and hand-derived gradients (verified by grad_check_head)."""
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]
        e, c = self.evidence(z)
        alpha = e + 1.0
        S = alpha.sum(axis=1)                      # (n,)
        m = alpha[:, 0] / S                        # mean toward presence
        w = self.fit_var_weight
        fit = (y - m) ** 2 + w * m * (1.0 - m) / (S + 1.0)
        e_mis = (1.0 - y) * e[:, 0] + y * e[:, 1]  # evidence toward the wrong pole
        loss = float(np.mean(fit) + lam_mis * np.mean(e_mis))

        # d(fit)/d(alpha) via m and S. dm/daP = aA/S^2 ; dm/daA = -aP/S^2.
        g_m = -2.0 * (y - m) + w * (1.0 - 2.0 * m) / (S + 1.0)
        g_S = -w * m * (1.0 - m) / (S + 1.0) ** 2
        de = np.empty_like(e)
        de[:, 0] = g_m * alpha[:, 1] / S ** 2 + g_S + lam_mis * (1.0 - y)
        de[:, 1] = -g_m * alpha[:, 0] / S ** 2 + g_S + lam_mis * y
        de /= n
        dpre2 = de * sigmoid(c["pre2"])            # softplus' == sigmoid

        # Background-void branch: flat tax on total evidence off the manifold.
        ebg, cbg = self.evidence(zbg)
        loss += float(self.lambda_bg * np.mean(ebg))
        # mean() above runs over all nb*2 elements, hence the 2 * nb here
        de_bg = np.full_like(ebg, self.lambda_bg / (2.0 * zbg.shape[0]))
        dpre2_bg = de_bg * sigmoid(cbg["pre2"])

        grads: dict[str, np.ndarray] = {}
        grads["W2"] = c["h"].T @ dpre2 + cbg["h"].T @ dpre2_bg
        grads["b2"] = dpre2.sum(axis=0) + dpre2_bg.sum(axis=0)
        dh = dpre2 @ self.p["W2"].T
        dh_bg = dpre2_bg @ self.p["W2"].T
        dpre1 = dh * (1.0 - c["h"] ** 2)
        dpre1_bg = dh_bg * (1.0 - cbg["h"] ** 2)
        grads["W1"] = c["z"].T @ dpre1 + cbg["z"].T @ dpre1_bg
        grads["b1"] = dpre1.sum(axis=0) + dpre1_bg.sum(axis=0)
        return loss, grads

    # --------------------------------------------------------------------- fit

    def fit(self, z: np.ndarray, y: np.ndarray, epochs: int = 3000, lr: float = 0.02,
            anneal_frac: float = 0.3, bg_per_step: int = 256,
            verbose: bool = False) -> FieldHead:
        """Train the head. The misleading-evidence tax anneals in over
        ``anneal_frac`` of training so early learning is unconstrained."""
        from .net import Adam
        z = np.atleast_2d(np.asarray(z, dtype=float))
        span = z.max(axis=0) - z.min(axis=0)
        self._bg_lo = z.min(axis=0) - self.bg_margin * span
        self._bg_hi = z.max(axis=0) + self.bg_margin * span
        self._bg_data = z.copy()
        opt = Adam(self.p, lr=lr)
        t_anneal = max(1, int(epochs * anneal_frac))
        for ep in range(epochs):
            lam = self.lambda_mis * min(1.0, ep / t_anneal)
            loss, grads = self.loss_and_grads(z, y, self._bg_sample(bg_per_step), lam)
            opt.step(self.p, grads)
            if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
                print(f"  epoch {ep:5d}  loss={loss:.4f}")
        return self


def grad_check_head(seed: int = 0, eps: float = 1e-6) -> float:
    """Max relative error between analytical and numerical gradients."""
    rng = np.random.default_rng(seed)
    head = FieldHead(in_dim=3, hidden=5, seed=seed)
    z = rng.normal(0, 1, size=(20, 3))
    y = rng.integers(0, 2, size=20).astype(float)
    zbg = rng.uniform(z.min() - 0.5, z.max() + 0.5, size=(16, 3))  # fixed batch
    _, grads = head.loss_and_grads(z, y, zbg, lam_mis=head.lambda_mis)
    max_rel = 0.0
    for k in head.p:
        flat = head.p[k].ravel()
        for _ in range(min(5, flat.size)):
            i = int(rng.integers(0, flat.size))
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = head.loss_and_grads(z, y, zbg, lam_mis=head.lambda_mis)
            flat[i] = orig - eps
            lm, _ = head.loss_and_grads(z, y, zbg, lam_mis=head.lambda_mis)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[k].ravel()[i]
            max_rel = max(max_rel, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return max_rel


__all__ = ["FieldHead", "grad_check_head", "ROUTES"]
