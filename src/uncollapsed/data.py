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


__all__ = ["make_xor", "make_data"]
