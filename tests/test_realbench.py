"""Real-data two zeros: the claims must survive contact with reality.

The digits tests run in CI (sklearn ships the data). The fashion test only
runs if Fashion-MNIST is already cached or downloadable, and is skipped
otherwise -- CI should not depend on an external download succeeding.
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from uncollapsed.realbench import PCA, load_digits_split, run_real_benchmark  # noqa: E402


@pytest.fixture(scope="module")
def digits_results():
    return run_real_benchmark("digits", seed=0, epochs=4000, verbose=False)


def test_digits_clear_accuracy(digits_results):
    assert digits_results["head_clear_acc"] >= 0.95
    assert digits_results["base_clear_acc"] >= 0.95


def test_digits_triage_holds_on_real_data(digits_results):
    """The headline claim survives real data: two channels triage what one
    entropy scalar cannot (and entropy again points the wrong way)."""
    assert digits_results["head_triage_auc"] >= 0.85
    assert digits_results["base_triage_auc"] <= 0.50


def test_digits_routing(digits_results):
    assert digits_results["conflict_to_escalate"] >= 0.60
    assert digits_results["void_to_gather"] >= 0.40   # near-OOD is the hard kind
    assert digits_results["clear_to_lean"] >= 0.80


def test_pca_is_deterministic_and_standardized():
    rng = np.random.default_rng(0)
    X = rng.normal(0, [3.0, 1.0, 0.2], size=(200, 3))
    Z = PCA(k=2).fit(X).transform(X)
    assert Z.shape == (200, 2)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-6)


def test_digits_split_is_disjoint_and_covers_classes():
    Xtr, ytr, Xte, yte = load_digits_split(seed=0)
    assert len(Xtr) + len(Xte) == 1797
    assert set(np.unique(ytr)) == set(range(10)) == set(np.unique(yte))


def _fashion_available() -> bool:
    try:
        from uncollapsed.realbench import load_fashion
        load_fashion(verbose=False)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _fashion_available(), reason="Fashion-MNIST not cached/downloadable")
def test_fashion_triage_beats_entropy():
    r = run_real_benchmark("fashion", seed=0, epochs=3000, verbose=False)
    assert r["head_clear_acc"] >= 0.95
    assert r["head_triage_auc"] >= r["base_triage_auc"] + 0.25
    assert r["conflict_to_escalate"] >= 0.80
