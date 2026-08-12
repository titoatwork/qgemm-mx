# Preregistration

**Commit this file before running any GEMM kernel measurement. Do not edit predictions after the fact — append outcomes below them instead.**

The point of this file is to make being wrong a documented finding rather than an
embarrassment. A study that states its expected headline result in advance and
then reports whether it held is doing something categorically different from one
that measures first and explains afterwards.

| Field | Value |
|---|---|
| Created | 2026-08-12 |
| Status | **sm_86 measured · sm_90 UNMEASURED — fill in before R0 begins on the DGX** |
| Git rule | Predictions are append-only. Corrections go in the Outcomes section with a date |

---

## 1. Measured device constants

These are inputs, not predictions. Regenerate with `make run-probe`.

### sm_86 — RTX 3050 6GB Laptop (development box)

| Quantity | Value | Source |
|---|---:|---|
| SMs | 20 | `cudaGetDeviceProperties` |
| L2 | 1.50 MB | same |
| Memory bus | 96-bit | same |
| Theoretical peak BW | 131.7 GB/s | derived from bus × clock |
| **Achieved read-only BW** | **124.2 GB/s** | `build/bandwidth 256`, grid-stride, 256 MiB working set |
| Achieved copy BW | 121.5 GB/s | same |
| BW efficiency | 94.3% of theoretical | — |
| Est. dense FP16 TC peak | 12.1 TFLOP/s @ base clock | 512 FLOP/clk/SM estimate |
| Observed FP16 TC peak | ~13.9 TFLOP/s | cuBLAS at M=1024, `results/cublas_fp16_sm86.csv` |
| **Ridge point** | **~113 FLOP/byte** | 13.9 TFLOP/s ÷ 124.2 GB/s |
| Launch overhead (non-graphed) | **14.0 µs** | `measure_launch_overhead`, WSL2 |

> **WSL2 note.** 14 µs launch overhead is roughly 2–4× native Linux. Absolute
> launch costs are not comparable between this box and the DGX. Ratios of
> graphed-to-looped time are comparable; absolute dispatch cost is not.

### sm_90 — DGX H100 (authoritative)

| Quantity | Value | Source |
|---|---:|---|
| SMs | **TBD** | run `make ARCH=sm_90 run-probe` |
| L2 | **TBD** (expect ~50 MB) | — |
| Achieved read-only BW | **TBD** (expect 2.6–3.0 TB/s) | — |
| Ridge point | **TBD** (expect ~150–190 FLOP/byte) | — |
| Launch overhead | **TBD** (expect 3–8 µs) | — |
| Nsight counter access | **TBD — GATE** | admin policy; resolve before R0 |

---

## 2. Predictions

### P1 · Format byte ratios (arithmetic, not a prediction — stated for reference)

| Format | B/weight | vs FP8 | vs FP16 |
|---|---:|---:|---:|
| FP16 | 2.0000 | 0.50× | 1.00× |
| FP8 E4M3 | 1.0000 | 1.00× | 2.00× |
| INT4 g128 | 0.5156 | 1.94× | 3.88× |
| MXFP4 | 0.5312 | 1.88× | 3.76× |
| NVFP4 | 0.5625 | 1.78× | 3.56× |

### P2 · Compute-bound crossover in M

From `M_crossover ≈ ridge × bytes_per_weight / 2`.

| Format | sm_86 (ridge 113) | sm_90 (ridge TBD, assume 165) |
|---|---:|---:|
| FP16 | 113 | 165 |
| FP8 | 57 | 83 |
| MXFP4 | 30 | 44 |
| INT4 g128 | 29 | 43 |

**Predicted:** MXFP4 percent-of-ideal stays above 85% up to `M`=16 on both
architectures, degrades between `M`=32 and `M`=64, and is clearly compute-bound
by `M`=128.

### P3 · The central claim

**Predicted:** a correctly-written MXFP4 weight-only GEMM on sm_90 will exceed
**1.5×** the throughput of an FP8 GEMM at `M`≤4 on the frozen shape set, against
the ~1.0× that the current production software-fallback path delivers.

**Falsified if:** the best achievable MXFP4 kernel stays below 1.2× FP8 at
`M`=1 after the full ablation. That outcome is a publishable negative result and
R2 should be abandoned in favour of reporting it.

### P4 · Where the missing throughput actually goes

**Predicted:** ranked by contribution to the current gap at `M`=1 on sm_90:

1. Wave quantization / insufficient CTAs (largest)
2. Launch and dispatch overhead
3. Layout-forced shared-memory staging and shuffles
4. Dequantization instruction issue cost (smallest)

**This contradicts the W4A8 literature's framing**, which treats (4) as the
fundamental limit. If (4) turns out to dominate, the prediction is wrong and the
W4A8 argument is vindicated for the decode regime — also a result worth having.

### P5 · MXFP4 versus NVFP4

**Predicted:** MXFP4 is faster to dequantize than NVFP4 — its E8M0 scale is a
power of two, so applying it is an exponent add foldable into the unpack, rather
than a multiply. **Predicted:** MXFP4 is less accurate for the same reason.
**Predicted:** the throughput difference is under 8% (both are dominated by the
same 4-bit element traffic), so the accuracy difference will dominate the
practical choice.

### P6 · CUDA graphs

**Predicted:** on sm_90 at `M`=1, the graphed-to-looped ratio for a 2880×2880
MXFP4 GEMM exceeds **2.0** — i.e. more than half of an un-graphed measurement is
dispatch, not kernel.

### P7 · Triton twin

**Predicted:** a Triton implementation reaches 50–75% of the tuned CUDA kernel,
and the shortfall is attributable primarily to inability to control the `wgmma`
fragment layout rather than to arithmetic codegen quality.

---

## 3. Outcomes

*Append only. Date every entry. Record refutations as prominently as confirmations.*

*(empty — no measurement has been run against these predictions yet)*
