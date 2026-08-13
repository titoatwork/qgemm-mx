#!/usr/bin/env python3
"""Roofline sketch from a bench CSV that has M, pct_of_ideal or achieved_gbps.

Not a substitute for ncu. Uses measured peak as the horizontal roof.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--peak-gbps", type=float, default=124.2)
    ap.add_argument("--ridge-flop-per-byte", type=float, default=113.0)
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv_path)))
    by_m: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if "M" not in r:
            continue
        m = int(float(r["M"]))
        if "pct_of_ideal" in r and r["pct_of_ideal"]:
            by_m[m].append(float(r["pct_of_ideal"]))
        elif "pct_of_ideal_fp16_bytes" in r and r["pct_of_ideal_fp16_bytes"]:
            by_m[m].append(float(r["pct_of_ideal_fp16_bytes"]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; printing table only")
        for m in sorted(by_m):
            v = sum(by_m[m]) / len(by_m[m])
            print(f"M={m} mean_pct_ideal={v:.2f}")
        return

    ms = sorted(by_m)
    ys = [sum(by_m[m]) / len(by_m[m]) for m in ms]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ms, ys, marker="o")
    ax.axhline(100.0, color="gray", ls="--", lw=1, label="100% of measured peak")
    ax.set_xlabel("M")
    ax.set_ylabel("% of ideal bandwidth (CSV)")
    ax.set_title(f"Roofline-adjacent view  peak={args.peak_gbps} GB/s  ridge~{args.ridge_flop_per_byte}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = args.out or str(
        Path("docs/figures") / (Path(args.csv_path).stem + "_roofline_pct.png")
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
