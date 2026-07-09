"""
uncollapsed.net
===============

A genuinely trainable two-channel field network.

Every hidden unit carries ``(presence, absence)`` all the way through and never
collapses; collapse happens only at the output edge. A synapse uses a single
signed weight: ``w > 0`` passes presence->presence and absence->absence;
``w < 0`` **swaps** them (balanced-ternary negation), made differentiable so
gradients flow through the swap.

Layout (all hidden state is two-channel)::

    inputs -> encode -> (P0, A0)
    hidden: preP1 = P0 @ W1+ + A0 @ W1- + b1p     (W+ = relu(W), W- = relu(-W))
            preA1 = A0 @ W1+ + P0 @ W1- + b1a
            P1, A1 = softplus(preP1), softplus(preA1)
    output: same shape -> (P2, A2)
    readout: prob = sigmoid(gain * (P2 - A2))     <- the only collapse used for loss

The backprop is hand-derived and verified by :func:`grad_check`.
"""
from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def encode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A raw feature becomes a two-channel field. Near 1 -> presence; near 0 -> absence."""
    return np.clip(x, 0.0, None), np.clip(1.0 - x, 0.0, None)


class UncollapsedNet:
    def __init__(self, hidden: int = 16, gain: float = 4.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.gain = float(gain)
        self.p: dict[str, np.ndarray] = {
            "W1": rng.normal(0, 1.0 / np.sqrt(2), size=(2, hidden)),
            "b1p": np.zeros(hidden),
            "b1a": np.zeros(hidden),
            "W2": rng.normal(0, 1.0 / np.sqrt(hidden), size=(hidden, 1)),
            "b2p": np.zeros(1),
            "b2a": np.zeros(1),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        p = self.p
        P0, A0 = encode(x)
        W1p, W1n = relu(p["W1"]), relu(-p["W1"])
        preP1 = P0 @ W1p + A0 @ W1n + p["b1p"]
        preA1 = A0 @ W1p + P0 @ W1n + p["b1a"]
        P1, A1 = softplus(preP1), softplus(preA1)
        W2p, W2n = relu(p["W2"]), relu(-p["W2"])
        preP2 = P1 @ W2p + A1 @ W2n + p["b2p"]
        preA2 = A1 @ W2p + P1 @ W2n + p["b2a"]
        P2, A2 = softplus(preP2), softplus(preA2)
        prob = sigmoid(self.gain * (P2 - A2))
        cache = dict(P0=P0, A0=A0, preP1=preP1, preA1=preA1, P1=P1, A1=A1,
                     preP2=preP2, preA2=preA2, P2=P2, A2=A2, prob=prob)
        return prob, cache

    @staticmethod
    def loss(prob: np.ndarray, y: np.ndarray) -> float:
        pr = np.clip(prob[:, 0], 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    def backward(self, cache: dict, y: np.ndarray) -> dict[str, np.ndarray]:
        p = self.p
        n = y.shape[0]
        y = y.reshape(-1, 1)
        P0, A0, P1, A1 = cache["P0"], cache["A0"], cache["P1"], cache["A1"]
        preP1, preA1, preP2, preA2 = cache["preP1"], cache["preA1"], cache["preP2"], cache["preA2"]
        W2p, W2n = relu(p["W2"]), relu(-p["W2"])

        dlogit = (cache["prob"] - y) / n            # BCE + sigmoid combined
        dP2, dA2 = self.gain * dlogit, -self.gain * dlogit
        dpreP2, dpreA2 = dP2 * sigmoid(preP2), dA2 * sigmoid(preA2)

        dW2p = P1.T @ dpreP2 + A1.T @ dpreA2
        dW2n = A1.T @ dpreP2 + P1.T @ dpreA2
        dW2 = dW2p * (p["W2"] > 0) - dW2n * (p["W2"] < 0)
        db2p, db2a = np.sum(dpreP2, axis=0), np.sum(dpreA2, axis=0)

        dP1 = dpreP2 @ W2p.T + dpreA2 @ W2n.T
        dA1 = dpreP2 @ W2n.T + dpreA2 @ W2p.T
        dpreP1, dpreA1 = dP1 * sigmoid(preP1), dA1 * sigmoid(preA1)

        dW1p = P0.T @ dpreP1 + A0.T @ dpreA1
        dW1n = A0.T @ dpreP1 + P0.T @ dpreA1
        dW1 = dW1p * (p["W1"] > 0) - dW1n * (p["W1"] < 0)
        db1p, db1a = np.sum(dpreP1, axis=0), np.sum(dpreA1, axis=0)

        return {"W1": dW1, "b1p": db1p, "b1a": db1a, "W2": dW2, "b2p": db2p, "b2a": db2a}

    def collapse(self, x: np.ndarray, hold_half: float = 0.15):
        """Edge collapse on the calibrated probability. Ambiguous -> HOLD, never a default 'no'."""
        prob, c = self.forward(x)
        prob = prob[:, 0]
        labels = np.where(prob >= 0.5 + hold_half, "presence",
                          np.where(prob <= 0.5 - hold_half, "absence", "hold"))
        return labels, prob, c["P2"][:, 0], c["A2"][:, 0]


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 0.02):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * grads[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)


def accuracy(net: UncollapsedNet, X: np.ndarray, y: np.ndarray) -> float:
    prob, _ = net.forward(X)
    return float(np.mean((prob[:, 0] >= 0.5).astype(float) == y))


def grad_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Return the max relative error between analytical and numerical gradients."""
    from .data import make_data
    net = UncollapsedNet(hidden=6, seed=seed)
    X, y = make_data(24, noise=0.15, seed=99)
    _, cache = net.forward(X)
    grads = net.backward(cache, y)
    rng = np.random.default_rng(1)
    max_rel = 0.0
    for k in net.p:
        flat = net.p[k].ravel()
        for _ in range(min(5, flat.size)):
            i = int(rng.integers(0, flat.size))
            orig = flat[i]
            flat[i] = orig + eps
            lp = net.loss(net.forward(X)[0], y)
            flat[i] = orig - eps
            lm = net.loss(net.forward(X)[0], y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[k].ravel()[i]
            max_rel = max(max_rel, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
    return max_rel


def train(hidden: int = 16, epochs: int = 4000, noise: float = 0.18, lr: float = 0.02,
          gain: float = 4.0, seed: int = 0, abstain: bool = False, verbose: bool = False):
    """Train on noisy XOR; evaluate on unseen CLEAR points. Returns (net, (Xtr, ytr, Xte, yte))."""
    from .data import make_data
    Xtr, ytr = make_data(500, noise=noise, seed=seed + 1, abstain=abstain)
    Xte, yte = make_data(400, noise=noise, seed=seed + 777, abstain=False)
    net = UncollapsedNet(hidden=hidden, gain=gain, seed=seed)
    opt = Adam(net.p, lr=lr)
    for ep in range(epochs):
        prob, cache = net.forward(Xtr)
        opt.step(net.p, net.backward(cache, ytr))
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            print(f"  epoch {ep:5d}  loss={net.loss(prob, ytr):.4f}  "
                  f"test_acc(clear)={accuracy(net, Xte, yte):.3f}")
    return net, (Xtr, ytr, Xte, yte)


__all__ = [
    "UncollapsedNet", "Adam", "encode", "softplus", "sigmoid",
    "accuracy", "grad_check", "train",
]
