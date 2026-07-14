"""
uncollapsed.faultbench
======================

The **two lieutenants benchmark**: the two zeros as a distributed-systems
fault-triage problem, on real sensor-network telemetry.

Lamport's Byzantine generals result prices the two kinds of failure very
differently: *crash* faults (a node goes silent) are survivable with ``2f+1``
replicas, while *Byzantine* faults (a node speaks contradiction) cost
``3f+1``. Silence is cheaper than lies. Both faults look like "something is
wrong with this unit", but they demand opposite responses:

* **crash / silence** -> ``gather`` -- no evidence exists; wait, re-poll,
  re-establish the link. Do not infer betrayal from silence.
* **Byzantine / contradiction** -> ``escalate`` -- the evidence itself is
  fighting; challenge, attest, put a human in the loop.

A single anomaly scalar (entropy, max-softmax) can flag both but triage
neither. That is the claim under test, on real data.

Data
----
Real telemetry: the Intel Berkeley Research Lab sensor-network deployment
(Madden et al., 2004) -- 54 motes reporting temperature every ~31 s for
weeks, fetched once from a public GitHub mirror and cached. Spatial peers
measuring a shared thermal field make "consistency with peers" physically
meaningful. The first 10 days are used (before widespread battery death) and
readings are filtered to plausible ranges.

Fault injection
---------------
There is no public corpus of labelled naturally-occurring Byzantine faults --
nobody publishes "here are 10,000 traces where node 7 was lying". The
standard methodology in the BFT / sensor-fusion literature is therefore
**real signal statistics, canonical injected fault models**, and that is what
this module does. Instances are (mote, hour) windows summarized as 12
five-minute-bin traces; features are computed *against spatial peers*:

* **healthy** (clear, presence) -- the window as recorded.
* **drift** (clear, absence) -- a +-2.5..5 degC calibration bias: a benign,
  labelled, learnable fault. Deviates in level, still tracks peers.
* **Byzantine** (conflict) -- the mote *replays its own trace from 12 hours
  earlier*: internally smooth, individually plausible values from the wrong
  thermal regime, so peer correlation collapses. Labelled **both ways**
  (contradictory duplication), because this is the two lieutenants problem:
  from one observation you cannot tell whether this unit is lying or its
  peers have drifted -- the contradiction is real, its provenance is
  observer-undecidable.
* **crash** (void, never in training) -- 75-95% of reports dropped; the
  remnant readings are genuine. Silence carries no evidence.

Features per instance (all peer-relative, standardized on train):
report rate, mean and max deviation from the peer-median trace, correlation
with the peer-median trace, roughness, own spread, fraction of bins present,
and *excess deviation* over the mote's own healthy baseline (computed from
training days only -- motes have persistent physical offsets; a warm-corner
mote is not drifting).

The split is **temporal** (train on the first 7 days, test on the last 3),
as is honest for telemetry.

Run it: ``uncollapsed --demo faults``.
"""
from __future__ import annotations

import urllib.request
import zipfile

import numpy as np

from .bench import BaselineMLP, auc
from .head import FieldHead
from .realbench import _cache_dir

INTEL_ZIP_URL = "https://raw.githubusercontent.com/linsea423/Intel_Lab_Data/master/data.zip"
N_BINS = 12          # five-minute bins per hour
EXPECTED = 116       # ~31 s epochs per hour
N_MOTES = 54
N_DAYS = 10          # stable period before widespread battery death
TRAIN_DAYS = 7       # temporal split: first 7 days train, last 3 test

FEATURE_NAMES = ("report_rate", "mean_dev", "max_dev", "peer_corr",
                 "roughness", "own_std", "bins_present", "excess_dev")


# ---------------------------------------------------------------------- data

def load_intel_traces(verbose: bool = True) -> dict[str, np.ndarray]:
    """Download (once), clean, window, and cache the Intel Lab telemetry.

    Returns arrays: ``T`` (mote, day, hour, bin) binned temperature traces
    with NaN gaps, ``counts`` (mote, day, hour) reports per window,
    ``healthy`` boolean mask of fully-reporting windows, and ``peer``
    (day, hour, bin) cross-mote median traces.
    """
    cache = _cache_dir()
    npz = cache / "intel_traces.npz"
    if npz.exists():
        return dict(np.load(npz))
    raw = cache / "intel_lab_data.zip"
    if not raw.exists():
        if verbose:
            print("  downloading Intel Lab telemetry (~33 MB, one time) ...")
        urllib.request.urlretrieve(INTEL_ZIP_URL, raw)  # noqa: S310
    if verbose:
        print("  parsing ~2.3M readings ...")
    with zipfile.ZipFile(raw) as z, z.open("data.txt") as f:
        motes, days, hours, bins, temps = [], [], [], [], []
        for bline in f:
            p = bline.decode("ascii", "ignore").split()
            if len(p) != 8:
                continue
            try:
                t, v, m = float(p[4]), float(p[7]), int(p[3])
            except ValueError:
                continue
            # plausible-temperature and live-battery filter (the raw corpus
            # famously contains battery-death garbage)
            if not (0.0 < t < 45.0) or v < 2.2 or not (1 <= m <= N_MOTES):
                continue
            motes.append(m)
            days.append(p[0])
            hours.append(int(p[1][:2]))
            bins.append(int(p[1][3:5]) // 5)
            temps.append(t)
    motes_a = np.array(motes)
    days_a = np.array(days)
    keep_days = sorted(set(days))[:N_DAYS]
    day_idx = {d: i for i, d in enumerate(keep_days)}
    sel = np.isin(days_a, keep_days)
    di = np.array([day_idx[d] for d in days_a[sel]])
    m_, h_, b_, t_ = (motes_a[sel], np.array(hours)[sel],
                      np.array(bins)[sel], np.array(temps)[sel])

    sums = np.zeros((N_MOTES + 1, N_DAYS, 24, N_BINS))
    cnts = np.zeros_like(sums, dtype=int)
    np.add.at(sums, (m_, di, h_, b_), t_)
    np.add.at(cnts, (m_, di, h_, b_), 1)
    with np.errstate(invalid="ignore"):
        T = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    counts = cnts.sum(axis=3)
    healthy = ((~np.isnan(T)).sum(axis=3) == N_BINS) & (counts >= 0.6 * EXPECTED)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins are expected
        # Median over all 54 motes, not leave-one-out: one member barely moves
        # a median of 54, injections never contaminate it (it is computed from
        # the raw corpus before any fault is injected), and it is consistent
        # across the temporal split.
        peer = np.nanmedian(T[1:], axis=0)
    out = dict(T=T, counts=counts, healthy=healthy, peer=peer)
    np.savez_compressed(npz, **out)
    return out


# ------------------------------------------------------------------ features

def _features(own: np.ndarray, count: float, peer_tr: np.ndarray,
              mote_baseline: float) -> list[float]:
    ok = ~np.isnan(own) & ~np.isnan(peer_tr)
    dev = own[ok] - peer_tr[ok]
    if ok.sum() >= 4 and np.std(own[ok]) > 1e-9 and np.std(peer_tr[ok]) > 1e-9:
        corr = float(np.corrcoef(own[ok], peer_tr[ok])[0, 1])
    else:
        corr = 0.0
    finite = own[~np.isnan(own)]
    diffs = np.diff(finite)
    mean_dev = float(np.mean(dev)) if len(dev) else 0.0
    return [
        min(count / EXPECTED, 1.0),
        mean_dev,
        float(np.max(np.abs(dev))) if len(dev) else 0.0,
        corr,
        float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
        float(np.std(finite)) if len(finite) else 0.0,
        ok.sum() / N_BINS,
        mean_dev - mote_baseline,
    ]


# ------------------------------------------------------------------ protocol

def _build_instances(data: dict, day_lo: int, day_hi: int, rng: np.random.Generator,
                     baselines: np.ndarray) -> dict[str, np.ndarray]:
    """Assemble healthy / drift / Byzantine / crash instances from a day range."""
    T, counts, healthy, peer = data["T"], data["counts"], data["healthy"], data["peer"]
    idx = np.argwhere(healthy)
    idx = idx[(idx[:, 0] >= 1) & (idx[:, 1] >= day_lo) & (idx[:, 1] < day_hi)]
    rng.shuffle(idx)
    groups: dict[str, list] = {"healthy": [], "drift": [], "byz": [], "crash": []}
    for m, dy, hr in idx:
        own, cnt, ptr = T[m, dy, hr].copy(), float(counts[m, dy, hr]), peer[dy, hr]
        kind = rng.choice(["healthy", "drift", "byz", "crash"])
        if kind == "drift":
            own = own + rng.choice([-1.0, 1.0]) * rng.uniform(2.5, 5.0)
        elif kind == "byz":
            dy2, hr2 = (dy, hr - 12) if hr >= 12 else (dy - 1, hr + 12)
            if dy2 < day_lo or not data["healthy"][m, dy2, hr2]:
                continue  # no clean 12h-earlier trace to replay
            own = T[m, dy2, hr2].copy()
        elif kind == "crash":
            keep = rng.choice(N_BINS, size=int(rng.integers(1, 4)), replace=False)
            mask = np.ones(N_BINS, bool)
            mask[keep] = False
            own[mask] = np.nan
            cnt = max(1.0, cnt * rng.uniform(0.05, 0.25))
        groups[kind].append(_features(own, cnt, ptr, baselines[m]))
    return {k: np.array(v) for k, v in groups.items()}


def _mote_baselines(data: dict, day_hi: int) -> np.ndarray:
    """Per-mote median deviation from peers over healthy TRAINING windows only.

    Motes have persistent physical offsets (a warm corner is not a fault);
    computing this from training days avoids temporal leakage.
    """
    T, healthy, peer = data["T"], data["healthy"], data["peer"]
    base = np.zeros(N_MOTES + 1)
    for m in range(1, N_MOTES + 1):
        devs = []
        for dy in range(day_hi):
            for hr in range(24):
                if healthy[m, dy, hr]:
                    d = T[m, dy, hr] - peer[dy, hr]
                    d = d[~np.isnan(d)]
                    if len(d):
                        devs.append(float(np.mean(d)))
        base[m] = float(np.median(devs)) if devs else 0.0
    return base


HEAD_PARAMS = dict(hidden=24, bg_mode="shell", bg_sigma=2.5,
                   lambda_bg=0.03, lambda_mis=0.002, fit_var_weight=6.0)


def run_fault_benchmark(seed: int = 0, epochs: int = 4000, verbose: bool = True) -> dict:
    """The two lieutenants protocol on Intel Lab telemetry. Returns metrics."""
    rng = np.random.default_rng(seed)
    data = load_intel_traces(verbose=verbose)
    baselines = _mote_baselines(data, day_hi=TRAIN_DAYS)
    tr = _build_instances(data, 0, TRAIN_DAYS, rng, baselines)
    te = _build_instances(data, TRAIN_DAYS, N_DAYS, rng, baselines)

    # training: healthy vs drift with true labels; Byzantine labelled both ways
    Xtr = np.vstack([tr["healthy"], tr["drift"], tr["byz"], tr["byz"]])
    ytr = np.concatenate([np.ones(len(tr["healthy"])), np.zeros(len(tr["drift"])),
                          np.ones(len(tr["byz"])), np.zeros(len(tr["byz"]))])
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    z = lambda X: (X - mu) / sd  # noqa: E731

    Xte = np.vstack([te["healthy"], te["drift"], te["byz"], te["crash"]])
    yte = np.concatenate([np.ones(len(te["healthy"])), np.zeros(len(te["drift"])),
                          np.full(len(te["byz"]) + len(te["crash"]), np.nan)])
    zone = np.array(["clear"] * (len(te["healthy"]) + len(te["drift"]))
                    + ["conflict"] * len(te["byz"]) + ["void"] * len(te["crash"]))
    clear, conflict, void = (zone == "clear"), (zone == "conflict"), (zone == "void")

    head = FieldHead(in_dim=len(FEATURE_NAMES), seed=seed,
                     **HEAD_PARAMS).fit(z(Xtr), ytr, epochs=epochs)
    base = BaselineMLP(in_dim=len(FEATURE_NAMES), hidden=HEAD_PARAMS["hidden"],
                       seed=seed).fit(z(Xtr), ytr, epochs=epochs)

    m = head.masses(z(Xte))
    routes = head.route(z(Xte))
    base_amb = base.entropy(z(Xte))
    head_amb = m["conflict"] + m["voidness"]
    head_flag = head_amb > float(np.quantile(head_amb[clear], 0.95))
    base_flag = base_amb > float(np.quantile(base_amb[clear], 0.95))

    results = dict(
        head_clear_acc=float(np.mean((m["expectation"][clear] >= 0.5) == yte[clear])),
        base_clear_acc=float(np.mean((base.forward(z(Xte))[0][clear] >= 0.5) == yte[clear])),
        head_byz_flagged=float(np.mean(head_flag[conflict])),
        base_byz_flagged=float(np.mean(base_flag[conflict])),
        head_crash_flagged=float(np.mean(head_flag[void])),
        base_crash_flagged=float(np.mean(base_flag[void])),
        head_triage_auc=auc(m["voidness"][void], m["voidness"][conflict]),
        base_triage_auc=auc(base_amb[void], base_amb[conflict]),
        crash_to_gather=float(np.mean(routes[void] == "gather")),
        byz_to_escalate=float(np.mean(routes[conflict] == "escalate")),
        clear_to_lean=float(np.mean(np.isin(routes[clear], ["presence", "absence"]))),
        n_train=len(Xtr), n_test=len(Xte),
    )

    if verbose:
        print("two lieutenants on real telemetry -- Intel Lab, 54 motes, temporal split")
        print(f"  train {results['n_train']} instances (days 0-{TRAIN_DAYS - 1}) | "
              f"test {results['n_test']} (days {TRAIN_DAYS}-{N_DAYS - 1})")
        print(f"  clear accuracy (healthy vs drift)   head {results['head_clear_acc']:.3f}   "
              f"baseline {results['base_clear_acc']:.3f}")
        print("  flagged as 'not a normal answer' (<=5% false-flags on clear):")
        print(f"    Byzantine (replayed telemetry)    head {results['head_byz_flagged']:.3f}   "
              f"baseline {results['base_byz_flagged']:.3f}")
        print(f"    crash (dropped reports)           head {results['head_crash_flagged']:.3f}   "
              f"baseline {results['base_crash_flagged']:.3f}")
        print("  TRIAGE AUC (crash vs Byzantine, higher is better):")
        print(f"    head (voidness axis)              {results['head_triage_auc']:.3f}")
        print(f"    baseline (entropy)                {results['base_triage_auc']:.3f}")
        print("  head routing (crash->gather is Lamport's cheap fault; "
              "Byzantine->escalate is the expensive one):")
        print(f"    crash     -> gather   {results['crash_to_gather']:.3f}")
        print(f"    Byzantine -> escalate {results['byz_to_escalate']:.3f}")
        print(f"    clear     -> lean     {results['clear_to_lean']:.3f}")
    return results


__all__ = ["run_fault_benchmark", "load_intel_traces", "FEATURE_NAMES"]
