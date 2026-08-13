#!/usr/bin/env python3
"""Print wave-quantization table (matches occupancy.hpp / test_occupancy)."""
import csv
import sys

LAYERS = [
    ("q_o_proj", 4096),
    ("kv_gqa", 1024),
    ("gate_up", 14336),
    ("down_proj", 4096),
    ("gpt_oss_2880", 2880),
    ("fused_qkv", 6144),
]
TILE = 256
SM86, SM90 = 20, 132


def ctas(n: int, tile: int = TILE) -> int:
    return (n + tile - 1) // tile


def splitk(n: int, sms: int, oversub: float = 2.0) -> int:
    base = ctas(n)
    want = int(-(-oversub * sms // 1))  # ceil
    return max(1, (want + base - 1) // base)


def main() -> None:
    w = csv.writer(sys.stdout)
    w.writerow(
        ["layer", "N", "tile", "ctas", "occ_sm86", "occ_sm90", "starved_h100", "splitk_h100_2x"]
    )
    for name, n in LAYERS:
        c = ctas(n)
        o86 = c / SM86
        o90 = c / SM90
        w.writerow(
            [name, n, TILE, c, f"{o86:.3f}", f"{o90:.3f}", o90 < 0.5, splitk(n, SM90)]
        )


if __name__ == "__main__":
    main()
