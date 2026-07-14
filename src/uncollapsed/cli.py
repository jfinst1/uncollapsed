"""
uncollapsed.cli
===============

Command-line entry point. Run ``uncollapsed --help`` after install, or
``python -m uncollapsed``.
"""
from __future__ import annotations

import argparse
import json

from .algebra import field_and, field_not, field_or, fuse_consensus, fuse_conservative
from .field import CollapsePolicy, UncollapsedField
from .net import accuracy, grad_check, train


def _show(f: UncollapsedField) -> None:
    print("\n" + f.summary())
    print("  mass:", json.dumps(f.mass().as_dict()))
    print("  collapse:", json.dumps(f.collapse(label=f.source).as_dict()))


def demo_zero() -> None:
    print("\n=== same visible 0, different inner accounting ===")
    for f in [
        UncollapsedField.void(source="void zero"),
        UncollapsedField(0.18, 0.18, source="calm centre"),
        UncollapsedField(0.9, 0.9, source="loaded contradiction"),
        UncollapsedField(0.85, 0.35, source="presence lean"),
    ]:
        _show(f)


def demo_pressure() -> None:
    print("\n=== pressure is not evidence ===")
    # Potentials of 1.2 saturate to ~0.70 per channel -> conflict mass ~0.49,
    # which crosses the conflict threshold; a weaker field never reaches the
    # branch where the bad policy could act, and both policies would (correctly
    # but unhelpfully for a demo) escalate for the same reason.
    loaded = UncollapsedField(1.2, 1.2, resistance=0.32, pressure=0.9, pressure_bias=-1.0,
                              source="loaded zero under pressure")
    print("\ngood policy:", json.dumps(loaded.collapse(label="good").as_dict()))
    bad = CollapsePolicy(allow_pressure_to_break_ties=True)
    print("bad policy :", json.dumps(loaded.collapse(policy=bad, label="bad").as_dict()))


def demo_algebra() -> None:
    print("\n=== field algebra ===")
    yes = UncollapsedField.from_bool(True, source="YES")
    no = UncollapsedField.from_bool(False, source="NO")
    for f in [
        field_not(yes, source="NOT yes"),
        field_and([yes, no], source="yes AND no"),
        field_or([yes, no], source="yes OR no"),
        fuse_conservative([yes, no], source="conservative fuse yes/no"),
        fuse_consensus([yes, no], source="consensus fuse yes/no"),
    ]:
        _show(f)


def demo_learn(png: str | None = None) -> None:
    print("\n=== learning: gradient check, held-out accuracy, learned abstention ===")
    err = grad_check()
    print(f"  gradient check max relative error: {err:.2e}  ({'PASS' if err < 1e-4 else 'FAIL'})")

    print("\n  classifier-only (proves real learning; test = unseen noisy points):")
    net_a, data_a = train(abstain=False, verbose=True)
    _, _, Xte, yte = data_a
    print(f"  --> held-out test accuracy: {accuracy(net_a, Xte, yte):.3f}  (overconfident on the boundary)")

    print("\n  abstention-trained (holding is taught, not free):")
    net_b, data_b = train(abstain=True, gain=2.0, verbose=True)
    _, _, Xte_b, yte_b = data_b
    print(f"  --> accuracy on clear held-out points: {accuracy(net_b, Xte_b, yte_b):.3f}")
    import numpy as np
    probes = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.5, 0.0], [0.5, 1.0]])
    labels, prob, _, _ = net_b.collapse(probes, hold_half=0.15)
    print("  boundary behaviour:")
    for row, lab, pr in zip(probes, labels, prob, strict=True):
        print(f"    A={row[0]:.1f} B={row[1]:.1f}  prob={pr:.3f}  collapse={lab}")

    if png:
        from .viz import render_surface
        if render_surface(net_b, png, train_data=data_b):
            print(f"  wrote decision-surface PNG: {png}")


def demo_triage(png: str | None = None) -> None:
    print("\n=== triage: telling contradiction from ignorance (the two zeros) ===")
    from .bench import run_benchmark
    run_benchmark(verbose=True)
    if png:
        from .data import make_two_zeros
        from .head import FieldHead
        from .viz import render_two_zeros
        Xtr, ytr = make_two_zeros(train=True)
        head = FieldHead(in_dim=2, hidden=24).fit(Xtr, ytr, epochs=3000)
        if render_two_zeros(head, png, train_data=(Xtr, ytr)):
            print(f"  wrote mass-map PNG: {png}")


def demo_real(dataset: str = "digits") -> None:
    print("\n=== triage on real data (the two zeros, no synthetic geometry) ===")
    from .realbench import run_real_benchmark
    run_real_benchmark(dataset=dataset, epochs=4000, verbose=True)


def demo_faults() -> None:
    print("\n=== the two lieutenants: crash vs Byzantine on real telemetry ===")
    from .faultbench import run_fault_benchmark
    run_fault_benchmark(epochs=4000, verbose=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="uncollapsed", description=__doc__)
    ap.add_argument("--demo", choices=["all", "zero", "pressure", "algebra", "learn", "triage", "real", "faults"],
                    default="all")
    ap.add_argument("--png", type=str, default=None,
                    help="save a PNG (decision surface for learn, mass maps for triage; "
                         "with --demo all it applies to learn)")
    ap.add_argument("--dataset", choices=["digits", "fashion"], default="digits",
                    help="dataset for the real demo (digits needs scikit-learn; "
                         "fashion downloads Fashion-MNIST once)")
    args = ap.parse_args()
    if args.demo in ("all", "zero"):
        demo_zero()
    if args.demo in ("all", "pressure"):
        demo_pressure()
    if args.demo in ("all", "algebra"):
        demo_algebra()
    if args.demo in ("all", "learn"):
        demo_learn(png=args.png if args.demo in ("all", "learn") else None)
    if args.demo in ("all", "triage"):
        demo_triage(png=args.png if args.demo == "triage" else None)
    if args.demo == "real":
        demo_real(dataset=args.dataset)
    if args.demo == "faults":
        demo_faults()


if __name__ == "__main__":
    main()
