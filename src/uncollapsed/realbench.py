"""
uncollapsed.realbench
=====================

The two zeros benchmark on **real data**. Same protocol as
:mod:`uncollapsed.bench`, same contenders, no synthetic geometry:

* **clear**    -- two real classes with their true labels (the ordinary task).
* **conflict** -- a third real class in which **every sample appears twice
  with contradictory labels** (two annotators, one disagreement -- the static
  analogue of multi-annotator datasets like CIFAR-10H). The contradiction is
  in the data itself, so no amount of model capacity can fit it away; a model
  that memorizes per-sample random labels would fake leans instead, which is
  exactly the failure this construction rules out. Under the quadratic
  evidential fit term, duplication is mathematically identical to a soft 0.5
  target -- the presentation is chosen for honesty, not convenience.
* **void**     -- classes that never appear in training at all. This is
  held-out-class OOD, deliberately the *hard* kind: void inputs share pixel
  statistics with the training data (near-OOD), unlike a far-away synthetic
  ring.

Both ambiguous groups deserve "not a normal answer"; they demand opposite
actions (escalate vs. gather). The question is whether the model's uncertainty
signal can tell which is which.

Datasets
--------
* ``digits``  -- sklearn's bundled handwritten digits (8x8, 1797 samples).
  Ships inside scikit-learn; no download. Task 3-vs-8, conflict class 5,
  void classes {0, 1, 2, 6, 7, 9}.
* ``fashion`` -- Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017), fetched from
  the official Zalando GitHub repository and cached locally. Task
  trouser-vs-ankle-boot, conflict class shirt, void = the other seven classes.

Features are PCA-projected (plain numpy SVD, fit on training data only) and
standardized -- the head is a *readout*, it sits on features by design.

Run it: ``uncollapsed --demo real`` (digits) or
``uncollapsed --demo real --dataset fashion``.
"""
from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

from .bench import BaselineMLP, auc
from .head import FieldHead

FASHION_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# --------------------------------------------------------------------- loaders

def _cache_dir() -> Path:
    d = Path(os.environ.get("UNCOLLAPSED_CACHE", Path.home() / ".cache" / "uncollapsed"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[2:3], "big")  # data type code lives in byte 2
    assert magic == 0x08, f"expected unsigned-byte idx file, got type {magic:#x}"
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    return data.reshape(dims)


def load_fashion(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (once) and load Fashion-MNIST from the official GitHub repo.

    Returns ``(Xtr, ytr, Xte, yte)`` with X flattened to (n, 784) float in [0, 1].
    """
    cache = _cache_dir()
    arrays = {}
    for key, fname in FASHION_FILES.items():
        path = cache / fname
        if not path.exists():
            if verbose:
                print(f"  downloading {fname} ...")
            urllib.request.urlretrieve(FASHION_BASE + fname, path)  # noqa: S310
        arrays[key] = _read_idx(path)
    Xtr = arrays["train_images"].reshape(-1, 784).astype(float) / 255.0
    Xte = arrays["test_images"].reshape(-1, 784).astype(float) / 255.0
    return Xtr, arrays["train_labels"].astype(int), Xte, arrays["test_labels"].astype(int)


def load_digits_split(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """sklearn's bundled digits, shuffled and split 60/40. No download."""
    from sklearn.datasets import load_digits  # optional dep, [bench] extra
    d = load_digits()
    X = d.data.astype(float) / 16.0
    y = d.target.astype(int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n_tr = int(0.6 * len(y))
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


# ------------------------------------------------------------------------- PCA

class PCA:
    """Minimal numpy PCA with standardized components. Fit on training data only."""

    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray) -> PCA:
        self.mean = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = Vt[: self.k]
        proj = (X - self.mean) @ self.components.T
        self.scale = proj.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) @ self.components.T) / self.scale


# -------------------------------------------------------------------- protocol

# Per-dataset head hyperparameters were tuned once on seed 0 and frozen.
DATASETS = {
    "digits": dict(pos=3, neg=8, conflict=5, loader="digits",
                   pca_k=24, per_class=None, class_names=None,
                   head=dict(hidden=32, bg_mode="shell", bg_sigma=2.5,
                             lambda_bg=0.03, lambda_mis=0.002, fit_var_weight=6.0)),
    "fashion": dict(pos=1, neg=9, conflict=6, loader="fashion",
                    pca_k=32, per_class=800,
                    class_names=["t-shirt/top", "trouser", "pullover", "dress", "coat",
                                 "sandal", "shirt", "sneaker", "bag", "ankle boot"],
                    head=dict(hidden=32, bg_mode="shell", bg_sigma=2.5,
                              lambda_bg=0.03, lambda_mis=0.002, fit_var_weight=6.0)),
}


def _take(rng, X, y, cls, n):
    idx = np.flatnonzero(y == cls)
    if n is not None and len(idx) > n:
        idx = rng.choice(idx, size=n, replace=False)
    return X[idx]


def run_real_benchmark(dataset: str = "digits", seed: int = 0, epochs: int = 3000,
                       verbose: bool = True) -> dict:
    """The two zeros protocol on a real dataset. Returns the metrics dict."""
    assert dataset in DATASETS, f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}"
    cfg = DATASETS[dataset]
    rng = np.random.default_rng(seed)

    if cfg["loader"] == "digits":
        Xtr_all, ytr_all, Xte_all, yte_all = load_digits_split(seed=seed)
    else:
        Xtr_all, ytr_all, Xte_all, yte_all = load_fashion(verbose=verbose)

    pos, neg, conf = cfg["pos"], cfg["neg"], cfg["conflict"]
    void_classes = sorted(set(np.unique(ytr_all).tolist()) - {pos, neg, conf})
    n = cfg["per_class"]

    # ---- training set: two clear classes + the conflict class labeled BOTH
    # ways (contradictory duplication; see module docstring).
    Xp, Xn = _take(rng, Xtr_all, ytr_all, pos, n), _take(rng, Xtr_all, ytr_all, neg, n)
    Xc = _take(rng, Xtr_all, ytr_all, conf, n)
    Xtr = np.vstack([Xp, Xn, Xc, Xc])
    ytr = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn)),
                          np.ones(len(Xc)), np.zeros(len(Xc))])
    perm = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]
    Xtr_unique = np.vstack([Xp, Xn, Xc])  # PCA is fit without the duplicates

    # ---- test set: fresh clear + conflict + never-seen void classes
    Xp_t, Xn_t = _take(rng, Xte_all, yte_all, pos, n), _take(rng, Xte_all, yte_all, neg, n)
    Xc_t = _take(rng, Xte_all, yte_all, conf, n)
    void_blocks = [_take(rng, Xte_all, yte_all, c, (n or 10**9) // max(1, len(void_classes)))
                   for c in void_classes]
    Xv_t = np.vstack(void_blocks)
    void_class_of = np.concatenate([np.full(len(b), c) for b, c in
                                    zip(void_blocks, void_classes, strict=True)])
    Xte = np.vstack([Xp_t, Xn_t, Xc_t, Xv_t])
    yte = np.concatenate([np.ones(len(Xp_t)), np.zeros(len(Xn_t)),
                          np.full(len(Xc_t), np.nan), np.full(len(Xv_t), np.nan)])
    zone = np.array(["clear"] * (len(Xp_t) + len(Xn_t))
                    + ["conflict"] * len(Xc_t) + ["void"] * len(Xv_t))

    # ---- features: PCA fit on training data only, then standardized
    pca = PCA(k=cfg["pca_k"]).fit(Xtr_unique)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    head = FieldHead(in_dim=cfg["pca_k"], seed=seed, **cfg["head"]).fit(Ztr, ytr, epochs=epochs)
    base = BaselineMLP(in_dim=cfg["pca_k"], hidden=cfg["head"]["hidden"],
                       seed=seed).fit(Ztr, ytr, epochs=epochs)

    clear, conflict, void = (zone == "clear"), (zone == "conflict"), (zone == "void")
    m = head.masses(Zte)
    head_pred = (m["expectation"] >= 0.5).astype(float)
    base_pred = (base.forward(Zte)[0] >= 0.5).astype(float)
    head_clear_acc = float(np.mean(head_pred[clear] == yte[clear]))
    base_clear_acc = float(np.mean(base_pred[clear] == yte[clear]))

    head_amb = m["conflict"] + m["voidness"]
    base_amb = base.entropy(Zte)
    head_flag = head_amb > float(np.quantile(head_amb[clear], 0.95))
    base_flag = base_amb > float(np.quantile(base_amb[clear], 0.95))

    head_triage = auc(m["voidness"][void], m["voidness"][conflict])
    base_triage = auc(base_amb[void], base_amb[conflict])

    routes = head.route(Zte)
    results = dict(
        dataset=dataset,
        head_clear_acc=head_clear_acc, base_clear_acc=base_clear_acc,
        head_conflict_flagged=float(np.mean(head_flag[conflict])),
        base_conflict_flagged=float(np.mean(base_flag[conflict])),
        head_void_flagged=float(np.mean(head_flag[void])),
        base_void_flagged=float(np.mean(base_flag[void])),
        head_triage_auc=head_triage, base_triage_auc=base_triage,
        void_to_gather=float(np.mean(routes[void] == "gather")),
        conflict_to_escalate=float(np.mean(routes[conflict] == "escalate")),
        clear_to_lean=float(np.mean(np.isin(routes[clear], ["presence", "absence"]))),
    )

    if verbose:
        names = cfg["class_names"]
        label = (lambda c: names[c]) if names else (lambda c: str(c))
        print(f"two zeros on real data -- {dataset}")
        print(f"  task: {label(pos)} vs {label(neg)} | conflict class: {label(conf)} "
              f"(labeled both ways) | void: {len(void_classes)} never-seen classes")
        print(f"  clear accuracy          head {head_clear_acc:.3f}   baseline {base_clear_acc:.3f}")
        print("  flagged as 'not a normal answer' (<=5% false-flags on clear):")
        print(f"    conflict class        head {results['head_conflict_flagged']:.3f}   "
              f"baseline {results['base_conflict_flagged']:.3f}")
        print(f"    void classes          head {results['head_void_flagged']:.3f}   "
              f"baseline {results['base_void_flagged']:.3f}")
        print("  TRIAGE AUC (void vs conflict, higher is better):")
        print(f"    head (voidness axis)  {results['head_triage_auc']:.3f}")
        print(f"    baseline (entropy)    {results['base_triage_auc']:.3f}")
        print("  head routing:")
        print(f"    void     -> gather    {results['void_to_gather']:.3f}")
        print(f"    conflict -> escalate  {results['conflict_to_escalate']:.3f}")
        print(f"    clear    -> lean      {results['clear_to_lean']:.3f}")
        print("  what the head calls each never-seen class (route distribution):")
        void_routes = routes[void]
        for c in void_classes:
            rr = void_routes[void_class_of == c]
            dist = {k: float(np.mean(rr == k)) for k in
                    ("presence", "absence", "hold", "escalate", "gather")}
            top = max(dist, key=dist.get)
            print(f"    {label(c):<12} -> mostly {top:<9} "
                  + " ".join(f"{k[:4]}={v:.2f}" for k, v in dist.items() if v >= 0.05))
    return results


__all__ = ["run_real_benchmark", "load_fashion", "load_digits_split", "PCA", "DATASETS"]
