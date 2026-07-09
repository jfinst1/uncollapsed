"""Tests for the trainable network: correct gradients, genuine generalization
(not memorization), and learned abstention."""
import numpy as np

from uncollapsed.net import accuracy, grad_check, train


def test_gradients_are_correct():
    assert grad_check() < 1e-4


def test_learns_and_generalizes_to_unseen_points():
    # Train on noisy XOR, evaluate on a DIFFERENT stream of noisy points.
    net, (_, _, Xte, yte) = train(epochs=1500, seed=0)
    assert accuracy(net, Xte, yte) > 0.9  # generalization, not lookup


def test_abstention_is_learnable():
    net, _ = train(epochs=1500, gain=2.0, abstain=True, seed=0)
    probes = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels, _, _, _ = net.collapse(probes, hold_half=0.15)
    # clear corners still collapse correctly to XOR...
    assert list(labels) == ["absence", "presence", "presence", "absence"]
    # ...and the genuinely ambiguous centre is held, not guessed.
    center_label, _, _, _ = net.collapse(np.array([[0.5, 0.5]]), hold_half=0.15)
    assert center_label[0] == "hold"
