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
**Master execution plan (peak standard):** [`docs/MASTER_EXECUTION_PLAN.md`](docs/MASTER_EXECUTION_PLAN.md)  
Living status: [`docs/STATUS.md`](docs/STATUS.md) · Gates: [`docs/GATES.md`](docs/GATES.md) · Shapes: [`docs/SHAPES.md`](docs/SHAPES.md)  
Occupancy table: [`docs/OCCUPANCY.md`](docs/OCCUPANCY.md) · Local workflow: [`docs/LOCAL_WORKFLOW.md`](docs/LOCAL_WORKFLOW.md)  
Predictions: [`PREREGISTRATION.md`](PREREGISTRATION.md)

---

## Status

| Rung | Scope | State |
|---|---|---|
| **R0** | Measurement harness; host pack; occupancy math; fused-ref GPU correctness | **local peak** — harness, pack/probes, occupancy table, full frozen shapes on sm_86 (stream + cuBLAS), naive dequant floor, fused-ref matches host (incl. MX K=2880). **sm_90 + public kernels still open** |
| R1 | Decompose the Hopper FP4 gap | not started |
| R2 | `wgmma`/TMA kernel that closes it | A-fragment table + host permute **done**; kernel not started — fused-ref is **not** this kernel |
| R3 | MoE / grouped variant, DSL comparison | not started |

A **fused-ref** kernel (`include/qgemm/qgemv_ref.cuh`) exists for **correctness
only**: one thread per output, sequential nibble order matching `pack.hpp`, no
Tensor Cores. It proves the encodings on device. It is **not** a performance
claim and must never be cited as SOTA. The R2 `wgmma`/TMA path has not started.

Development happens locally and lands here in curated drops rather than
continuously, so the published history is coarser than the working one. Local
workflow: [`docs/LOCAL_WORKFLOW.md`](docs/LOCAL_WORKFLOW.md).

---

## Quick start

```bash
make ARCH=sm_86 run-probe          # dev box; use ARCH=sm_90 on H100
make test-cpu                      # formats, pack, ref GEMM, occupancy table
make test-gpu                      # fused-ref vs host (correctness, not perf)
```

`run-probe` captures the environment, dumps device constants and roofline
predictions, and measures the empirical bandwidth. **The read-only figure it
prints is the project's denominator** — record it in `PREREGISTRATION.md` and
pass it to every benchmark explicitly:

```bash
./build/bench_cublas 0 124.2 all 256 > results/raw/cublas_fp16_sm86.csv
# optional: [layer|all] [max_M] — layers from docs/SHAPES.md / shapes.hpp
./build/bench_stream_ideal 0 124.2 all > results/raw/stream_ideal_sm86.csv
bash scripts/run_r0_sm86.sh
```

`bench_cublas` refuses to run without a measured denominator rather than falling
back to the spec-sheet peak, because that fallback silently inflates every
efficiency claim downstream.

Wave-quantization / CTA occupancy (paper arithmetic, tiles 256 and 64, no split-K)
is a testable artifact, not a slide claim. Every frozen layer starves H100 at tile
256; at the `m64` atom tile `q_o_proj` is still 0.485 of an H100. The laptop hides
that except on `kv_gqa`. Full tables:
[`docs/OCCUPANCY.md`](docs/OCCUPANCY.md). Reproduce:

```bash
make test-cpu                      # prints the occupancy CSV
python3 python/occupancy_table.py  # same numbers
```

Python analysis environment:

```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/python python/analyze_csv.py results/raw/cublas_fp16_sm86_fullshapes_Mle256.csv
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
include/qgemm/timing.cuh         measurement primitives (rotation, graphs, stats)
include/qgemm/formats.cuh        byte accounting for FP16/FP8/INT4/MXFP4/NVFP4
include/qgemm/pack.hpp           host pack/unpack + reference dequant
include/qgemm/occupancy.hpp      CTA / wave-quant arithmetic (host, testable)
include/qgemm/dequant_device.cuh device decode; keep in sync with pack.hpp
include/qgemm/qgemv_ref.cuh      fused-ref GEMM — correctness, not SOTA
include/qgemm/ref_gemm.hpp       host FP32 Y = X @ Wᵀ
include/qgemm/shapes.hpp         frozen layers + M sweep
include/qgemm/correctness.hpp    structured probes (one-hot, primes, ramp)
src/probe/device_props.cu        device constants, ridge, L2 / wave checks
src/probe/bandwidth.cu           empirical peak BW → the project denominator
src/bench/bench_cublas.cu        FP16 baseline; harness vs a known-good ref
src/bench/bench_stream_ideal.cu  packed-weight stream → t_ideal
src/bench/bench_naive_dequant.cu dequant-then-cuBLAS sanity floor (not headline)
src/bench/bench_qgemv_ref.cu     fused-ref timing (waterfall step, not SOTA)
src/kernels/                     (empty — R2 wgmma/TMA)
tests/                           host + GPU fused-ref (`make test-cpu` / `test-gpu`)
scripts/env_capture.sh           per-run environment record
results/                         raw CSVs, committed
docs/                            thesis, plan, STATUS, OCCUPANCY, LOCAL_WORKFLOW
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

---

## Companion project

**[Slipstream](https://github.com/titoatwork/slipstream)** is the same thesis one
layer up. Both start from the fact that LLM decode is bandwidth-bound, not
compute-bound. qgemm-mx attacks that wall at the **kernel** level — moving fewer
bytes per weight by recovering block-scaled FP4 throughput. Slipstream attacks it
at the **system** level, where paged KV memory and scheduling policy set the batch
size the hardware ever gets to run. A fused FP4 dequant-GEMM is exactly the kind
of kernel that drops into Slipstream's quantized decode path: kernel depth and
system breadth, deliberately paired.
