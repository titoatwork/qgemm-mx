# results/ — how to read the CSVs

Every published number in this repository is paired with a CSV here (or under
`raw/`) and an environment capture from `scripts/env_capture.sh`. If either is
missing, treat the number as unreproducible.

## Files currently committed

| Path | What it is |
|------|------------|
| `cublas_fp16_sm86.csv` | FP16 cuBLAS baseline + harness self-check on the sm_86 lab GPU |
| `env_sm86_initial.txt` | Environment snapshot for that validation run |

New runs should prefer `raw/<bench>_<arch>_<date>.csv` and `env/` so the
committed “headline” files stay stable while experiments accumulate.

## Common columns

| Column | Meaning |
|--------|---------|
| `shape` | Layer archetype tag (`q/o_proj`, `gpt-oss`, …) |
| `M,N,K` | Problem dimensions; weights are `[N,K]`, activations `[M,K]` |
| `format` | Weight format name from `formats.cuh` (stream / fused benches) |
| `rotation` | Number of distinct weight buffers used to defeat L2 residency |
| `graphed_cold_us` | **Headline latency**: CUDA-graph, rotated buffers (cold) |
| `graphed_hot_us` | Same graph path, single buffer (hot / cache-resident) |
| `looped_cold_us` | Normal launches + rotation; includes per-launch overhead |
| `ideal_us` | `weight_bytes / measured_gbps` (stream bench); floor time |
| `pct_of_ideal` | **Headline efficiency**: `100 * ideal_us / graphed_cold_us` |
| `achieved_gbps` | Bytes moved ÷ measured time (same byte model as ideal) |
| `l2_inflation` | `cold / hot`. ≫ 1 means the hot loop was measuring L2, not HBM |
| `launch_share` | `%` of looped time attributable to launch vs graphed cold |

## Denominator rule

`pct_of_ideal` always divides by an **empirical** read-only bandwidth from
`./build/bandwidth` (e.g. 124.2 GB/s on the sm_86 lab box). Never by the
spec-sheet peak. That measured figure is passed on the bench CLI:

```bash
./build/bench_cublas 0 124.2 > results/cublas_fp16_sm86.csv
./build/bench_stream_ideal 0 124.2 > results/raw/stream_ideal_sm86.csv
```

If `pct_of_ideal` for pure stream traffic or for cuBLAS FP16 at M=1 sits
meaningfully above 100%, the harness or the denominator is wrong — stop and
fix before trusting fused-kernel numbers.

## How to interpret the sm_86 cuBLAS CSV

- At small M, `pct_of_ideal` near ~100% means the harness, byte accounting, and
  measured BW agree (weight traffic dominates).
- `l2_inflation ≈ 1.0` on this laptop is **expected**: L2 is 1.5 MB and a
  4096×4096 FP16 weight matrix is ~32 MB, so rotation cannot show a hot/cold
  gap. The hot/cold split becomes a finding on sm_90 (50 MB L2).
- A step drop in `%ideal` as M grows (e.g. M=32→64) is a cuBLAS / wave /
  tiling effect, not a harness failure — flag it, do not “fix” it away.
- `launch_share` low under WSL2 still understates native-Linux launch cost;
  never compare absolute launch µs across hosts without noting virtualization.

## Stream-ideal CSV

`bench_stream_ideal` moves only `weight_bytes(format,N,K)` — no GEMM. It
answers: *for this byte volume, how close is pure traffic to the empirical peak?*
Its `ideal_us` column is the floor fused kernels are scored against. M is fixed
at 1 in that CSV (weight traffic is M-independent); the column exists so tools
can join on the same schema.

## Analysis helpers

```bash
python/analyze_csv.py results/cublas_fp16_sm86.csv
python/plot_figures.py results/cublas_fp16_sm86.csv   # -> docs/figures/
```

## Architecture caveat

**sm_90 is authoritative.** sm_86 numbers validate the harness and catch
regressions; they must not be used alone for occupancy or wave-quantization
conclusions (see `docs/PROTOCOL.md`).
