"""
uncollapsed.viz
===============

Optional visualization (requires matplotlib). Renders the learned decision
surface, where the gold band is the region the model collapses to HOLD -- it
abstains exactly where XOR is genuinely undecided.
"""
from __future__ import annotations

import numpy as np

from .net import UncollapsedNet


def render_surface(net: UncollapsedNet, path: str,
                   train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
                   grid: int = 220, hold_half: float = 0.15) -> bool:
    """Render presence/absence/HOLD regions to a PNG. Returns True on success."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping PNG: {exc})")
        return False

    xs = np.linspace(-0.15, 1.15, grid)
    XX, YY = np.meshgrid(xs, xs)
    labels, _, _, _ = net.collapse(np.column_stack([XX.ravel(), YY.ravel()]), hold_half=hold_half)
    code = np.vectorize({"absence": 0, "hold": 1, "presence": 2}.get)(labels).reshape(grid, grid)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    cmap = ListedColormap(["#2A4B7C", "#C9A227", "#8A2432"])  # absence / HOLD / presence
    ax.imshow(code, origin="lower", extent=(-0.15, 1.15, -0.15, 1.15),
              cmap=cmap, vmin=0, vmax=2, alpha=0.85, aspect="auto")
    for (cx, cy), lab in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0], strict=True):
        ax.scatter([cx], [cy], s=180, edgecolor="white", linewidth=2,
                   color="#8A2432" if lab else "#2A4B7C", zorder=5)
    if train_data is not None:
        Xtr, ytr, _, _ = train_data
        ax.scatter(Xtr[ytr == 1][:, 0], Xtr[ytr == 1][:, 1], s=6, color="#8A2432", alpha=0.35)
        ax.scatter(Xtr[ytr == 0][:, 0], Xtr[ytr == 0][:, 1], s=6, color="#2A4B7C", alpha=0.35)
    ax.set_title("Learned uncollapsed XOR surface\n(blue=absence, gold=HOLD/abstain, red=presence)")
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def render_two_zeros(head, path: str, grid: int = 240, extent: float = 7.5,
                     train_data: tuple[np.ndarray, np.ndarray] | None = None) -> bool:
    """Render the four mass maps and the routing map for a trained FieldHead.

    The punchline figure: the conflict cluster and the void ring both look like
    "0.5-ish" to a probability readout, but they light up *different* mass maps
    and route to *different* actions.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except Exception as exc:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping PNG: {exc})")
        return False

    xs = np.linspace(-extent, extent, grid)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    m = head.masses(pts)
    routes = head.route(pts)
    route_code = np.vectorize({"absence": 0, "hold": 1, "presence": 2,
                               "escalate": 3, "gather": 4}.get)(routes).reshape(grid, grid)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
    ext = (-extent, extent, -extent, extent)
    for ax, key, cmap in zip(axes[:4],
                             ["belief", "disbelief", "conflict", "voidness"],
                             ["Reds", "Blues", "Oranges", "Greys"], strict=True):
        im = ax.imshow(m[key].reshape(grid, grid), origin="lower", extent=ext,
                       cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(key)
        fig.colorbar(im, ax=ax, fraction=0.046)
    rcmap = ListedColormap(["#2A4B7C", "#C9A227", "#8A2432", "#D2691E", "#BFBFBF"])
    axes[4].imshow(route_code, origin="lower", extent=ext, cmap=rcmap,
                   vmin=0, vmax=4, aspect="auto")
    axes[4].set_title("route\n(blue=absence, gold=hold, red=presence,\n"
                      "orange=ESCALATE, grey=GATHER)")
    if train_data is not None:
        X, y = train_data
        for ax in axes:
            ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=3, color="#8A2432", alpha=0.4)
            ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=3, color="#2A4B7C", alpha=0.4)
    fig.suptitle("Two kinds of zero, kept apart: contradiction (conflict) vs ignorance (voidness)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return True


__all__ = ["render_surface", "render_two_zeros"]
