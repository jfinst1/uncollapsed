"""
uncollapsed.data
================

Small synthetic datasets. Noisy XOR is used throughout because it is the classic
"needs a hidden layer" function, and because a held-out set of *unseen* noisy
points cleanly separates learning from memorization.
"""
from __future__ import annotations

import numpy as np

_CORNERS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
_XOR = np.array([0, 1, 1, 0], dtype=float)


def make_xor(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=n)
    X = _CORNERS[idx] + rng.normal(0, noise, size=(n, 2))
    return X, _XOR[idx]


def make_data(n: int, noise: float, seed: int, abstain: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    abstain=False -> clean noisy-XOR corners with hard {0, 1} labels.
    abstain=True  -> half clear corners (hard labels) + half genuinely ambiguous
                     points with SOFT target 0.5. A point is ambiguous when one
                     input sits near 0.5 (undecided), because XOR(A, B) is
                     undecided exactly when A or B is undecided. This is the
                     supervision that teaches the model to HOLD instead of guess.
    """
    if not abstain:
        return make_xor(n, noise, seed)
    rng = np.random.default_rng(seed)
    n_clear = n // 2
    n_amb = n - n_clear
    idx = rng.integers(0, 4, size=n_clear)
    Xc = _CORNERS[idx] + rng.normal(0, noise, size=(n_clear, 2))
    yc = _XOR[idx]
    Xa = rng.uniform(0.0, 1.0, size=(n_amb, 2))
    which = rng.integers(0, 2, size=n_amb)
    Xa[np.arange(n_amb), which] = rng.uniform(0.35, 0.65, size=n_amb)
    ya = np.full(n_amb, 0.5)
    return np.vstack([Xc, Xa]), np.concatenate([yc, ya])


def make_two_zeros(n_clear: int = 400, n_conflict: int = 200, n_void: int = 200,
                   seed: int = 0, train: bool = True):
    """The "two zeros" benchmark: a world containing both kinds of 0.

    * Two **clear** Gaussian clusters (presence at ``(+2, 0)``, absence at
      ``(-2, 0)``) -- ordinary learnable structure.
    * A **conflict** cluster at ``(0, +2.5)`` where labels are drawn 50/50 --
      strong, genuinely contradictory evidence. The right move is to escalate.
    * A **void** ring at radius ~6 -- far from every training point. There is
      no evidence here at all. The right move is to gather data, not to guess.

    ``train=True`` returns only clear + conflict points (the void region is,
    by definition, unobserved). ``train=False`` returns all three groups plus a
    ``zone`` array in ``{"clear", "conflict", "void"}`` for scoring.
    """
    rng = np.random.default_rng(seed)
    half = n_clear // 2
    Xp = rng.normal([+2.0, 0.0], 0.45, size=(half, 2))
    Xa = rng.normal([-2.0, 0.0], 0.45, size=(n_clear - half, 2))
    yp, ya = np.ones(half), np.zeros(n_clear - half)
    Xc = rng.normal([0.0, +2.5], 0.45, size=(n_conflict, 2))
    yc = rng.integers(0, 2, size=n_conflict).astype(float)
    if train:
        X = np.vstack([Xp, Xa, Xc])
        y = np.concatenate([yp, ya, yc])
        perm = rng.permutation(len(y))
        return X[perm], y[perm]
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_void)
    r = rng.uniform(5.5, 6.5, size=n_void)
    Xv = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    yv = np.full(n_void, np.nan)                     # no true label exists here
    X = np.vstack([Xp, Xa, Xc, Xv])
    y = np.concatenate([yp, ya, yc, yv])
    zone = np.array(["clear"] * n_clear + ["conflict"] * n_conflict + ["void"] * n_void)
    return X, y, zone


__all__ = ["make_xor", "make_data", "make_two_zeros"]
