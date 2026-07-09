"""
Render the HOLD band forming during training as an animated GIF.

Trains the abstention-supervised two-channel net and snapshots the decision
surface at intervals. Early on the surface is undecided; as training proceeds the
four XOR regions sharpen and the gold HOLD band emerges along the boundary --
the network learning *where to abstain*.

Usage:
    python examples/make_training_gif.py            # -> assets/training.gif
    python examples/make_training_gif.py out.gif
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from uncollapsed.data import make_data  # noqa: E402
from uncollapsed.net import Adam, UncollapsedNet, accuracy  # noqa: E402

CMAP = ListedColormap(["#2A4B7C", "#C9A227", "#8A2432"])  # absence / HOLD / presence
CODES = {"absence": 0, "hold": 1, "presence": 2}


def frame(net: UncollapsedNet, epoch: int, acc: float, grid: int = 150,
          hold_half: float = 0.15) -> Image.Image:
    xs = np.linspace(-0.15, 1.15, grid)
    XX, YY = np.meshgrid(xs, xs)
    labels, _, _, _ = net.collapse(np.column_stack([XX.ravel(), YY.ravel()]), hold_half=hold_half)
    code = np.vectorize(CODES.get)(labels).reshape(grid, grid)

    fig, ax = plt.subplots(figsize=(4.6, 4.8))
    ax.imshow(code, origin="lower", extent=(-0.15, 1.15, -0.15, 1.15),
              cmap=CMAP, vmin=0, vmax=2, alpha=0.9, aspect="auto")
    for (cx, cy), lab in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0], strict=True):
        ax.scatter([cx], [cy], s=150, edgecolor="white", linewidth=2,
                   color="#8A2432" if lab else "#2A4B7C", zorder=5)
    ax.set_title(f"learning to hold  ·  epoch {epoch:>4}  ·  clear-acc {acc:.2f}")
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    plt.close(fig)
    return img.quantize(colors=32, method=Image.MEDIANCUT)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "assets/training.gif"
    Xtr, ytr = make_data(500, noise=0.18, seed=1, abstain=True)
    Xte, yte = make_data(400, noise=0.18, seed=777, abstain=False)
    net = UncollapsedNet(hidden=16, gain=2.0, seed=0)
    opt = Adam(net.p, lr=0.02)

    # denser snapshots early, where the surface changes fastest
    snaps = sorted({0, 20, 40, 70, 110, 170, 250, 380, 560, 820, 1200, 1800, 2600, 3200})
    frames: list[Image.Image] = []
    total = snaps[-1] + 1
    for ep in range(total):
        if ep in snaps:
            frames.append(frame(net, ep, accuracy(net, Xte, yte)))
        prob, cache = net.forward(Xtr)
        opt.step(net.p, net.backward(cache, ytr))
    frames.append(frame(net, total, accuracy(net, Xte, yte)))  # final settled frame

    durations = [500] + [420] * (len(frames) - 2) + [2200]  # linger on start and end
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)
    print(f"wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
