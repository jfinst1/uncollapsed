"""Train the two-channel net on noisy XOR and confirm it generalizes to unseen points."""
import numpy as np

from uncollapsed.net import accuracy, grad_check, train

print(f"gradient check: {grad_check():.2e}")
net, (_, _, Xte, yte) = train(epochs=3000, verbose=True)
print(f"held-out test accuracy (unseen noisy points): {accuracy(net, Xte, yte):.3f}")

# teach it to hold on the ambiguous boundary
net_b, _ = train(epochs=3000, gain=2.0, abstain=True)
labels, prob, _, _ = net_b.collapse(np.array([[0.5, 0.5]]))
print(f"at the ambiguous centre (0.5, 0.5): prob={prob[0]:.3f} -> {labels[0]}")
