"""Tests for the field-gated readout head and the two-zeros benchmark.

These tests assert the *claims*, not just the plumbing: the gradients are
exact, the four masses stay normalized, and -- the point of the whole module --
the head separates contradiction from ignorance where a single entropy scalar
cannot.
"""
import numpy as np
import pytest

from uncollapsed.bench import BaselineMLP, auc, run_benchmark
from uncollapsed.data import make_two_zeros
from uncollapsed.head import FieldHead, grad_check_head


def test_head_gradients_are_exact():
    assert grad_check_head() < 1e-4


def test_masses_sum_to_one_and_are_nonnegative():
    rng = np.random.default_rng(3)
    head = FieldHead(in_dim=4, hidden=8, seed=3)
    z = rng.normal(0, 2, size=(50, 4))
    m = head.masses(z)
    total = m["belief"] + m["disbelief"] + m["conflict"] + m["voidness"]
    assert np.allclose(total, 1.0)
    for k in ("belief", "disbelief", "conflict", "voidness"):
        assert np.all(m[k] >= 0.0)


def test_untrained_head_defaults_to_void():
    """Presence must be earned: before any data, everything routes to gather."""
    rng = np.random.default_rng(0)
    head = FieldHead(in_dim=2, hidden=8, seed=0)
    z = rng.normal(0, 3, size=(40, 2))
    m = head.masses(z)
    assert float(np.mean(m["voidness"])) > 0.6
    assert np.mean(head.route(z) == "gather") > 0.6


def test_auc_helper():
    assert auc(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0])) == 1.0
    assert auc(np.array([0.0, 1.0]), np.array([2.0, 3.0, 4.0])) == 0.0
    assert abs(auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) - 0.5) < 1e-9


@pytest.fixture(scope="module")
def benchmark_results():
    return run_benchmark(seed=0, epochs=3000, verbose=False)


def test_both_models_learn_the_easy_part(benchmark_results):
    assert benchmark_results["head_clear_acc"] >= 0.95
    assert benchmark_results["base_clear_acc"] >= 0.95


def test_head_separates_the_two_zeros(benchmark_results):
    """The headline claim: two channels triage what one scalar cannot."""
    assert benchmark_results["head_triage_auc"] >= 0.90
    assert benchmark_results["base_triage_auc"] <= 0.75


def test_head_routes_the_two_zeros_to_different_actions(benchmark_results):
    assert benchmark_results["void_to_gather"] >= 0.80
    assert benchmark_results["conflict_to_escalate"] >= 0.50
    assert benchmark_results["clear_to_lean"] >= 0.95


def test_head_flags_offmanifold_points_that_baseline_misses(benchmark_results):
    assert benchmark_results["head_void_flagged"] >= 0.90
    # The ordinary MLP is confidently wrong on a large share of OOD points.
    assert benchmark_results["base_void_flagged"] <= benchmark_results["head_void_flagged"]


def test_two_zeros_dataset_shapes():
    X, y = make_two_zeros(seed=1, train=True)
    assert X.shape[0] == y.shape[0] == 600
    X, y, zone = make_two_zeros(seed=1, train=False)
    assert X.shape[0] == 800
    assert np.all(np.isnan(y[zone == "void"]))  # no true label exists in the void
    assert not np.any(np.isnan(y[zone != "void"]))


def test_baseline_mlp_learns():
    Xtr, ytr = make_two_zeros(seed=2, train=True)
    base = BaselineMLP(in_dim=2, hidden=16, seed=2).fit(Xtr, ytr, epochs=1500)
    clear = np.abs(Xtr[:, 0]) > 1.0  # the two clear clusters
    prob = base.forward(Xtr[clear])[0]
    assert np.mean((prob >= 0.5) == ytr[clear]) >= 0.95
