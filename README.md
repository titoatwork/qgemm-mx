# qgemm-mx

**Recovering block-scaled FP4 throughput on GPUs without native FP4 support.**

Block-scaled 4-bit weight formats (MXFP4, NVFP4) are shipping in production
checkpoints. Native tensor-core support for them arrived with Blackwell — one
hardware generation ahead of most deployed capacity. On Hopper there is no native
FP4 multiply, so weights are stored at 4 bits and dequantized in-kernel, and
current production paths therefore **keep the memory saving and forfeit the
speed**.

In the bandwidth-bound decode regime, MXFP4 moves **1.88× fewer bytes per
weight** than FP8. The available speedup is known in advance; roughly none of it
is currently realized. This project measures where it goes and how much of it can
be recovered.

Full brief: [`docs/PROJECT-v2.0-fp4-hopper.md`](docs/PROJECT-v2.0-fp4-hopper.md)
Predictions, committed before measurement: [`PREREGISTRATION.md`](PREREGISTRATION.md)

---

## Status

| Rung | Scope | State |
|---|---|---|
| **R0** | Measurement harness; baseline study of existing kernels | **in progress** — harness built and validated on sm_86 |
| R1 | Decompose the Hopper FP4 gap | not started |
| R2 | `wgmma`/TMA kernel that closes it | not started |
| R3 | MoE / grouped variant, DSL comparison | not started |

No dequantization kernel exists yet, by design. R0 comes first because until the
harness is trustworthy, no number the project produces means anything.

---

## Quick start

```bash
make ARCH=sm_86 run-probe          # dev box; use ARCH=sm_90 on H100
```

That captures the environment, dumps device constants and roofline predictions,
and measures the empirical bandwidth. **The read-only figure it prints is the
project's denominator** — record it in `PREREGISTRATION.md` and pass it to every
benchmark explicitly:

```bash
./build/bench_cublas 0 124.2 > results/cublas_fp16_sm86.csv
```

`bench_cublas` refuses to run without a measured denominator rather than falling
back to the spec-sheet peak, because that fallback silently inflates every
efficiency claim downstream.

Python analysis environment:

```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
```

`torch` is deliberately *not* an R0 dependency — see
`python/requirements-baselines.txt`.

---

## The measurement protocol

Three hazards make naive kernel timing wrong by large factors for memory-bound
kernels. The harness in `include/qgemm/timing.cuh` addresses each.

**1 · L2 residency.** A 4096×4096 MXFP4 weight matrix is 8.5 MB; an H100's L2 is
50 MB. In a hot loop the weights never leave cache and you measure L2 bandwidth,
not HBM. Handled by **buffer rotation** — `rotation_count()` computes how many
distinct weight copies are needed to exceed L2 by 2×, so the timed region stays
pure kernel with no flush inside it. `L2Flusher` is available for cases where
rotation is impossible.

Every result reports both the rotated (cold) and single-buffer (hot) number. The
ratio between them is a finding, not a nuisance.

**2 · Launch overhead.** At `M`=1 an ideal MXFP4 kernel on an H100 finishes in
~1.5 µs while a kernel launch costs 3–8 µs — the launcher is larger than the
kernel. `time_graphed()` captures many iterations into one CUDA graph so the
timed region contains a single graph launch. `time_looped()` runs the same work
through normal launches, and the difference is reported as `launch_share`.

**3 · Spec-sheet denominators.** `pct_of_ideal()` requires an empirically
measured bandwidth. The theoretical peak is printed by the probe purely so the
ratio between them is visible.

Beyond those, `scripts/env_capture.sh` records driver, clocks, throttle reasons,
co-tenant processes, and MIG/MPS state with every run — because on a shared DGX,
a neighbouring job invalidates exclusive-access claims and nothing in the
measurement itself will tell you.

---

## Layout

```
include/qgemm/timing.cuh    measurement primitives (rotation, graphs, stats)
include/qgemm/formats.cuh   byte accounting for FP16/FP8/INT4/MXFP4/NVFP4
src/probe/device_props.cu   device constants, ridge point, wave-quantization
                            and L2-residency checks, incl. H100 projections
src/probe/bandwidth.cu      empirical peak BW -> the project denominator
src/bench/bench_cublas.cu   FP16 baseline; validates the harness against a
                            known-good reference
src/kernels/                (empty — R2)
scripts/env_capture.sh      per-run environment record
results/                    raw CSVs, committed
docs/                       brief, review, positioning, and the original v1.0
```

---

## Conventions

- **Layout:** activations `X` row-major `[M,K]`, weights `W` row-major `[N,K]`,
  output `Y` row-major `[M,N]` — i.e. `Y = X @ Wᵀ`, as a Transformer linear layer
  is actually stored.
- **Accumulation:** FP32. Not a variable.
- **Scales:** symmetric only in v1. `act_order` / `desc_act` out of scope.
- **Authoritative architecture:** sm_90. sm_86 is the development box and appears
  only in the portability chapter — a 20-SM device with 1.5 MB of L2 hides both
  wave quantization and L2 residency, so conclusions drawn there do not transfer.

---

## A note on what this project is

An engineering and measurement contribution, not an algorithmic one. Fused
dequantization-GEMM is well established; the quantization algorithms are someone
else's work. What is new here is the target architecture, the decomposition of
where the throughput goes, and a measurement protocol rigorous enough that the
numbers survive scrutiny.

`docs/REVIEW-fused-w4a16-gemm-v1.0.md` is the external review that redirected
this project away from its original target (dense INT4 on Ampere, a solved
problem), and `docs/v1.0-original-brief.md` is what it looked like before. Both
are kept deliberately: the revision is part of the record.
