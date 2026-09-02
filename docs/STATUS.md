# Project status dashboard

**Last updated:** 2026-09-02 (sm_90 build verification + arch-generic R0 runner)  
**Remote policy:** **local-only development** — no routine GitHub push (see [`LOCAL_WORKFLOW.md`](LOCAL_WORKFLOW.md))  
**Authoritative plan:** [`MASTER_EXECUTION_PLAN.md`](MASTER_EXECUTION_PLAN.md)  
**Thesis:** [`PROJECT-v2.0-fp4-hopper.md`](PROJECT-v2.0-fp4-hopper.md)  
**Occupancy table:** [`OCCUPANCY.md`](OCCUPANCY.md)  
**Owner asks:** [`OWNER_CHECKLIST.md`](OWNER_CHECKLIST.md)

---

## Rung status

| Rung | State | Notes |
|------|--------|------|
| **R0 Instrument** | **Local peak + fused-ref GPU path** | Harness, host pack, occupancy math, full frozen shapes on sm_86, stream/cuBLAS/naive benches. **GPU fused-ref matches host** (INT4 one-hot, MX/NV random, MX K=2880). **sm_90 + public kernels still open** |
| R1 Quantify | Not started | Needs sm_90 + public baselines |
| R2 Close | Layout + device permute **done**; `wgmma` not issued | Occupancy (256 + 64). Official A-fragment host+device. SMEM desc bits. Fused-ref is **correctness, not SOTA** |
| R3 Extend | Not started | |

---

## What works locally (sm_86)

| Capability | How to run | Claim |
|------------|------------|--------|
| Empirical BW denominator | `make run-probe` | **124.2 GB/s** RO — pass to every bench |
| Host pack / probes / ref GEMM | `make test-cpu` | INT4-g128, MXFP4, NVFP4 sequential pack |
| Occupancy / CTA table | `make test-cpu` or `python3 python/occupancy_table.py` | Tile 256 and atom-64; H100 starved without split-K either way. [`OCCUPANCY.md`](OCCUPANCY.md) |
| A-fragment permute | `make test-cpu` (`test_a_fragment`) | Official `m64n8k16` table; host pre-shuffle + inverse. [`A_FRAGMENT.md`](A_FRAGMENT.md) |
| A-fragment GPU fused-ref | `make test-gpu` (`test_a_fragment_gpu`) | Device unpack of A-fragment words == sequential. **Correctness only** |
| SMEM desc / MX k16 pair | `make test-cpu` (`test_wgmma_desc`) | Official 64-bit B descriptor bits; two k16 share one MX scale |
| GPU fused-ref vs host | `make test-gpu` | Same decode as `pack.hpp`. **Correctness only** — one thread per output, no Tensor Cores |
| Stream-ideal | `./build/bench_stream_ideal 0 124.2 all` | Pure weight traffic vs measured BW |
| cuBLAS FP16 | `./build/bench_cublas 0 124.2 all 256` | Harness self-check; full frozen shapes, M≤256 |
| Naive dequant + cuBLAS | `./build/bench_naive_dequant 0 124.2 q_o_proj 32` | Sanity floor, **never headline** |
| Fused-ref bench | `./build/bench_qgemv_ref 0 124.2` | Waterfall “fused but untuned” step — **not SOTA** |

`src/kernels/` is still empty. The R2 `wgmma`/TMA kernel has not started.

---

## Hardware

| Device | Probe BW | Harness validated | ncu counters | Exclusive runs |
|--------|----------|-------------------|--------------|----------------|
| sm_86 RTX 3050 laptop | **124.2 GB/s** RO | **Yes** | **FAIL** G2 | N/A |
| sm_90 H100 DGX | **UNMEASURED** | No | UNKNOWN G3 | TBD |

**sm_90 build verified 2026-09-02 without an H100.** CUDA 12.6 on the dev box
emits `compute_90`, so all 17 targets — probes, benches, and every GPU test
including `test_a_fragment_gpu` and `test_wgmma_desc` — cross-compile for
`-arch=sm_90` with zero warnings. This proves only that the code compiles for
Hopper. It says nothing about whether it is correct or fast there; both remain
open until the device runs it.

---

## Key local artifacts

| Path | Content |
|------|---------|
| `results/raw/stream_ideal_sm86_fullshapes.csv` | Pure weight stream, all layers × formats |
| `results/raw/cublas_fp16_sm86_fullshapes_Mle256.csv` | FP16 cuBLAS, all layers, M≤256 |
| `results/raw/naive_dequant_cublas_sm86_qo_Mle32.csv` | Dequant-then-cuBLAS floor, `q_o_proj`, M≤32 |
| `results/raw/qgemv_ref_mx_256_Mle4.csv` | Fused-ref MX 256×256, M≤4 (correctness path timed) |
| `results/raw/occupancy_cta_table.csv` | CTA / occupancy / split-K table |
| `docs/figures/*fullshapes*` | Latency / %ideal / roofline plots |
| `include/qgemm/pack.hpp` + `dequant_device.cuh` | Host pack + matching device decode |
| `include/qgemm/qgemv_ref.cuh` | Fused-ref kernel (correctness, not SOTA) |
| `include/qgemm/occupancy.hpp` | Testable CTA arithmetic (tiles 256 and 64) |
| `include/qgemm/a_fragment.hpp` | Official `m64n8k16` A-fragment + host pre-shuffle |
| `include/qgemm/wgmma_smem_desc.hpp` | Official 64-bit SMEM descriptor for B |

---

## Blockers (need you)

1. **DGX:** `ARCH=sm_90 bash scripts/run_r0.sh` → PREREGISTRATION sm_90 table.
   One command: env capture, build, CPU+GPU correctness, probes, cuBLAS,
   stream-ideal, then a paste-ready block for `PREREGISTRATION.md` §1. It runs
   correctness *before* timing, because numbers from an unvalidated device are
   worse than no numbers.  
2. **ncu laptop:** Windows GPU counter ACL (G2)  
3. **PAT workflow scope** if you want CI YAML on GitHub  
4. **Upstream demand** ping (G6)  

---

## Next (automated when possible)

1. Public baseline scaffolding (Marlin / Machete) when torch + GPU deps available  
2. TMA B = Xᵀ descriptor + emit `wgmma` — **needs sm_90**  
3. sm_90 R0 when the machine is available  

---

## Recent commits (selected)

| SHA | Summary |
|-----|---------|
| `b76508f` | Occupancy math, fused-ref GPU kernel, GPU correctness |
| `2bc11ce` | Naive dequant+cuBLAS sanity-floor baseline |
| `30fe16b` | Full shapes sm_86, ref GEMM, baseline scaffold |
| `da4fd7c` | Host pack / probes / CPU tests |
| `45f6c5e` | Stream ideal + analysis |
| `a6bf3a2` | Peak master plan |
