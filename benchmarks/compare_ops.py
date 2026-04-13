#!/usr/bin/env python3
"""
Compare PyTorch vs flame-core op-level benchmark results.

Usage:
    # Run both benchmarks first:
    python benchmarks/op_bench_pytorch.py --csv > /tmp/pt_ops.csv
    ./target/release/op_bench_flame --csv > /tmp/flame_ops.csv

    # Then compare:
    python benchmarks/compare_ops.py /tmp/pt_ops.csv /tmp/flame_ops.csv
"""
from __future__ import annotations

import csv
import sys


def load_csv(path: str) -> dict[str, tuple[float, float]]:
    """Return {op_name: (fwd_us, bwd_us)}."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["op"]] = (float(row["fwd_us"]), float(row["bwd_us"]))
    return data


def ratio_str(pt: float, flame: float) -> str:
    if pt <= 0 or flame <= 0:
        return "—"
    r = flame / pt
    marker = "" if r <= 1.2 else " ⚠" if r <= 2.0 else " 🔴"
    return f"{r:.2f}x{marker}"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pytorch.csv> <flame.csv>")
        sys.exit(1)

    pt = load_csv(sys.argv[1])
    fl = load_csv(sys.argv[2])

    # Merge op names preserving order from PyTorch file
    all_ops = list(pt.keys())
    for op in fl:
        if op not in all_ops:
            all_ops.append(op)

    hdr = (
        f"{'Op':<25} │ {'PT Fwd':>9} │ {'FL Fwd':>9} │ {'Ratio':>8} │ "
        f"{'PT Bwd':>9} │ {'FL Bwd':>9} │ {'Ratio':>8}"
    )
    print(hdr)
    print("─" * len(hdr))

    for op in all_ops:
        pt_fwd, pt_bwd = pt.get(op, (0.0, 0.0))
        fl_fwd, fl_bwd = fl.get(op, (0.0, 0.0))

        fwd_ratio = ratio_str(pt_fwd, fl_fwd)
        bwd_ratio = ratio_str(pt_bwd, fl_bwd) if pt_bwd > 0 and fl_bwd > 0 else "—"

        def fmt(v: float) -> str:
            return f"{v:.1f}" if v > 0 else "—"

        print(
            f"{op:<25} │ {fmt(pt_fwd):>9} │ {fmt(fl_fwd):>9} │ {fwd_ratio:>8} │ "
            f"{fmt(pt_bwd):>9} │ {fmt(fl_bwd):>9} │ {bwd_ratio:>8}"
        )


if __name__ == "__main__":
    main()
