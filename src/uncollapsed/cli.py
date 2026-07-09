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
    loaded = UncollapsedField(0.92, 0.92, resistance=0.32, pressure=0.9, pressure_bias=-1.0,
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


def main() -> None:
    ap = argparse.ArgumentParser(prog="uncollapsed", description=__doc__)
    ap.add_argument("--demo", choices=["all", "zero", "pressure", "algebra", "learn"], default="all")
    ap.add_argument("--png", type=str, default=None, help="save the learned decision surface (learn demo)")
    args = ap.parse_args()
    if args.demo in ("all", "zero"):
        demo_zero()
    if args.demo in ("all", "pressure"):
        demo_pressure()
    if args.demo in ("all", "algebra"):
        demo_algebra()
    if args.demo in ("all", "learn"):
        demo_learn(png=args.png)


if __name__ == "__main__":
    main()
