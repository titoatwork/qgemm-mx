# Baseline study (R0)

**Status:** Scaffolding — full public-kernel rebench requires deps + ideally sm_90.  
**Protocol:** [`PROTOCOL.md`](PROTOCOL.md) · **Shapes:** [`SHAPES.md`](SHAPES.md)

## Purpose

Re-time **existing** public kernels under our harness and report how published figures move when:

- L2 is flushed/rotated  
- launches are graph-captured  
- peak BW is **measured**, not spec-sheet  
- clocks/env are logged  

This is independently publishable and validates the apparatus before any custom fused kernel.

## Baselines (target list)

| Baseline | Role | Local sm_86 | DGX sm_90 |
|----------|------|-------------|-----------|
| Empirical stream (this repo) | `t_ideal` floor | **yes** `bench_stream_ideal` | yes |
| cuBLAS FP16 | Dense reference | **yes** `bench_cublas` | yes |
| cuBLAS / cuBLASLt FP8 | FP8 dense | if TC path available | yes |
| Marlin / GPTQ-Marlin | INT4 W4A16 SOTA-class | optional (torch/cuda build) | **primary** |
| Machete | Hopper mixed-input | n/a | **primary** |
| vLLM MXFP4/NVFP4 software path | Production gap (~1.0× vs FP8) | if installable | **primary** |
| Naive dequant + cuBLAS | Sanity floor only | planned | planned |

## Already landed (this repo)

| Artifact | Path |
|----------|------|
| FP16 cuBLAS CSV (subset shapes) | `results/cublas_fp16_sm86.csv` |
| Full-shape runner | `scripts/run_r0_sm86.sh` + frozen `shapes.hpp` |
| Stream ideal | `src/bench/bench_stream_ideal.cu` |
| Analysis | `python/analyze_csv.py`, `plot_figures.py` |

## Procedure (when deps ready)

1. `make ARCH=sm_90 run-probe` → record BW in PREREGISTRATION  
2. Run each baseline with **same** shapes, M-sweep, graphs, rotation  
3. Dump CSV under `results/raw/baseline_<name>_<arch>_<ts>.csv`  
4. Fill table: published vs our protocol (delta %)  
5. Update this doc with dated results — never invent  

## Public kernel integration notes

- Prefer **subprocess** / official CLI over vendoring large trees.  
- If vendoring, use `third_party/` + LICENSE.  
- Do not claim Marlin/Machete numbers without our env capture alongside.

## Amendments

| Date | Change |
|------|--------|
| 2026-08-13 | Scaffold created |
