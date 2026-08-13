#!/usr/bin/env python3
"""Summarize a qgemm-mx results CSV.

Loads a harness CSV (cuBLAS baseline, stream ideal, or later fused benches)
and prints a compact table of mean pct_of_ideal by M. Extra columns such as
achieved_gbps are averaged when present.

Usage:
  python/analyze_csv.py results/cublas_fp16_sm86.csv
  python/analyze_csv.py results/raw/stream_ideal_sm86.csv --by shape format
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load(path: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "pandas is required: pip install -r python/requirements.txt"
        ) from e
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="path to a results CSV")
    p.add_argument(
        "--by",
        nargs="+",
        default=None,
        help="extra group-by columns before M (e.g. shape format)",
    )
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"error: not a file: {args.csv}", file=sys.stderr)
        return 1

    df = _load(args.csv)
    if "pct_of_ideal" not in df.columns:
        print("error: CSV missing required column pct_of_ideal", file=sys.stderr)
        return 1
    if "M" not in df.columns:
        print("error: CSV missing required column M", file=sys.stderr)
        return 1

    group_cols = list(args.by or [])
    # Prefer sensible defaults when present and --by not given.
    if not group_cols:
        for c in ("shape", "format"):
            if c in df.columns:
                group_cols.append(c)
    group_cols = [c for c in group_cols if c in df.columns]
    group_cols.append("M")

    agg: dict = {"pct_of_ideal": "mean"}
    if "achieved_gbps" in df.columns:
        agg["achieved_gbps"] = "mean"
    if "graphed_cold_us" in df.columns:
        agg["graphed_cold_us"] = "mean"
    elif "ideal_us" in df.columns:
        agg["ideal_us"] = "mean"

    summary = df.groupby(group_cols, sort=True).agg(agg).reset_index()
    # Stable column order: keys first, then metrics.
    metric_cols = [c for c in summary.columns if c not in group_cols]
    summary = summary[group_cols + metric_cols]

    print(f"# {args.csv}")
    print(f"# rows={len(df)}  groups={len(summary)}  by={group_cols}")
    print()
    # Fixed-width-ish table without requiring tabulate.
    cols = list(summary.columns)
    strs = summary.copy()
    for c in cols:
        if strs[c].dtype.kind == "f":
            strs[c] = strs[c].map(lambda x: f"{x:.2f}")
        else:
            strs[c] = strs[c].astype(str)
    widths = [max(len(c), strs[c].map(len).max()) for c in cols]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("  ".join("-" * w for w in widths))
    for _, row in strs.iterrows():
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)))

    # One-line headline: overall mean pct at M=1 when available.
    m1 = df[df["M"] == 1] if (df["M"] == 1).any() else df
    overall = float(m1["pct_of_ideal"].mean())
    print()
    print(f"mean pct_of_ideal (M=1 if present, else all): {overall:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
