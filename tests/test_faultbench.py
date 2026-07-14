"""The two lieutenants on real telemetry: assertions, not prose.

These tests require the Intel Lab corpus (downloaded once, ~33 MB, then
cached). They are skipped when it is neither cached nor downloadable, so CI
does not depend on an external fetch succeeding.
"""
import numpy as np
import pytest


def _intel_available() -> bool:
    try:
        from uncollapsed.faultbench import load_intel_traces
        load_intel_traces(verbose=False)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _intel_available(),
                                reason="Intel Lab telemetry not cached/downloadable")


@pytest.fixture(scope="module")
def fault_results():
    from uncollapsed.faultbench import run_fault_benchmark
    return run_fault_benchmark(seed=0, epochs=4000, verbose=False)


def test_clear_task_is_learnable_by_both(fault_results):
    # healthy-vs-drift has irreducible overlap (thermal gradients look like
    # small drifts); both models should sit near the same feature ceiling.
    assert fault_results["head_clear_acc"] >= 0.82
    assert fault_results["base_clear_acc"] >= 0.82


def test_triage_separates_crash_from_byzantine(fault_results):
    """The two lieutenants claim: silence and lies are different numbers to
    the head, and one entropy scalar cannot tell them apart."""
    assert fault_results["head_triage_auc"] >= 0.95
    assert fault_results["base_triage_auc"] <= 0.50


def test_lamport_routing(fault_results):
    # crash is the cheap fault: wait / re-poll. Byzantine is the expensive
    # one: challenge / attest / human.
    assert fault_results["crash_to_gather"] >= 0.90
    assert fault_results["byz_to_escalate"] >= 0.80


def test_temporal_split_and_features():
    from uncollapsed.faultbench import FEATURE_NAMES, N_DAYS, TRAIN_DAYS, _mote_baselines, load_intel_traces
    assert 0 < TRAIN_DAYS < N_DAYS
    data = load_intel_traces(verbose=False)
    assert data["T"].shape[1] == N_DAYS
    base = _mote_baselines(data, day_hi=TRAIN_DAYS)
    assert np.all(np.isfinite(base))
    assert len(FEATURE_NAMES) == 8
