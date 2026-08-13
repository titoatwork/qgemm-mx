#!/usr/bin/env python3
"""Plot latency and %ideal vs M from a qgemm-mx results CSV.

Produces the R0 partial figures:
  1. graphed_cold_us (or first available latency column) vs M  (log-x)
  2. pct_of_ideal vs M

Saves under docs/figures/. Uses the Agg backend when no display is available
(headless CI / SSH).

Usage:
  python/plot_figures.py results/cublas_fp16_sm86.csv
  python/plot_figures.py results/cublas_fp16_sm86.csv --out-dir docs/figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _import_plotting():
    """Import matplotlib with Agg when headless; require pandas."""
    try:
        import matplotlib
    except ImportError as e:
        raise SystemExit(
            "matplotlib required: pip install -r python/requirements.txt"
        ) from e

    # No display, or explicit request: non-interactive backend before pyplot.
    if not os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND") == "Agg":
        matplotlib.use("Agg")
    else:
        # Prefer Agg for batch scripts even when DISPLAY is set (SSH X11 noise).
        # Override with MPLBACKEND=TkAgg etc. if interactive show() is needed.
        if os.environ.get("MPLBACKEND") is None:
            matplotlib.use("Agg")

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "matplotlib and pandas required: pip install -r python/requirements.txt"
        ) from e
    return plt, pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="path to a results CSV")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures"),
        help="directory for PNG outputs (created if missing)",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="filename prefix (default: CSV stem)",
    )
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"error: not a file: {args.csv}", file=sys.stderr)
        return 1

    plt, pd = _import_plotting()
    df = pd.read_csv(args.csv)
    if "M" not in df.columns or "pct_of_ideal" not in df.columns:
        print("error: CSV needs M and pct_of_ideal columns", file=sys.stderr)
        return 1

    latency_col = None
    for c in ("graphed_cold_us", "looped_cold_us", "ideal_us"):
        if c in df.columns:
            latency_col = c
            break

    series_keys = [c for c in ("shape", "format") if c in df.columns]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.csv.stem

    def _groups(frame):
        if not series_keys:
            yield "", frame
            return
        for key, g in frame.groupby(series_keys, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            label = " / ".join(str(k) for k in key)
            yield label, g

    # Figure 1: latency vs M
    if latency_col is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label, g in _groups(df):
            g = g.sort_values("M")
            ax.plot(
                g["M"],
                g[latency_col],
                marker="o",
                linewidth=1.5,
                label=label or latency_col,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("M (batch)")
        ax.set_ylabel(f"{latency_col} (µs)")
        ax.set_title(f"Latency vs M — {args.csv.name}")
        ax.grid(True, which="both", alpha=0.3)
        if series_keys:
            ax.legend(fontsize=8)
        fig.tight_layout()
        out1 = args.out_dir / f"{prefix}_latency_vs_M.png"
        fig.savefig(out1, dpi=150)
        plt.close(fig)
        print(f"wrote {out1}")
    else:
        print("skip latency plot: no graphed_cold_us / looped_cold_us / ideal_us")

    # Figure 2: pct_of_ideal vs M
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, g in _groups(df):
        g = g.sort_values("M")
        ax.plot(
            g["M"],
            g["pct_of_ideal"],
            marker="o",
            linewidth=1.5,
            label=label or "pct_of_ideal",
        )
    ax.axhline(100.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (batch)")
    ax.set_ylabel("% of ideal bandwidth")
    ax.set_title(f"% ideal vs M — {args.csv.name}")
    ax.grid(True, which="both", alpha=0.3)
    if series_keys:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out2 = args.out_dir / f"{prefix}_pct_ideal_vs_M.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
