# Measurement protocol

Implementation: `include/qgemm/timing.cuh`, `scripts/env_capture.sh`, probes under `src/probe/`.

## Three hazards

| Hazard | Failure mode | Mitigation in harness |
|--------|--------------|------------------------|
| L2 residency | Measure cache BW, invent super-HBM speedups | Buffer rotation (preferred); `L2Flusher`; report hot vs cold |
| Launch overhead | At M=1, launcher > kernel | `time_graphed` vs `time_looped`; report `launch_share` |
| Spec-sheet peak | Inflated %ideal | Empirical BW from `bandwidth` probe only |

## Required metadata per published run

- `scripts/env_capture.sh` output (driver, clocks, processes, MIG/MPS)  
- Device name, CC, measured RO bandwidth used as denominator  
- ARCH (sm_86 / sm_90)  
- Git commit SHA  
- Rotation count, graphed vs looped  

## Result CSV conventions

- Prefer `results/raw/<bench>_<arch>_<date>.csv`  
- Columns should include: shape, M, N, K, format (if any), graphed_cold_us, graphed_hot_us, looped_cold_us, pct_of_ideal, achieved_gbps, l2_inflation, launch_share  

## Correctness order

1. One-hot / basis probes (exact)  
2. Prime block scales  
3. Monotone ramp weights  
4. Random vs FP32 reference on dequantized weights  
5. Fused vs dequant-then-cuBLAS (when fused exists)  

## Architecture rules

- **Authoritative numbers:** sm_90 only  
- **sm_86:** development, portability chapter, harness bring-up  
- Never claim wave-quantization behavior from sm_86 alone  

## Preregistration

Predictions live in `PREREGISTRATION.md` before measurement. Outcomes append-only.
