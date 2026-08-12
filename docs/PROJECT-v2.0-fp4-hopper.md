# Project Description v2.0

## Recovering Block-Scaled FP4 Throughput on GPUs Without Native FP4 Support

**A measurement study and kernel implementation for MXFP4 / NVFP4 weight-only inference on Hopper-class hardware**

| Field | Value |
|-------|-------|
| **Short name** | `qgemm-mx` · working blog/talk title: *"The Missing 1.9×"* |
| **Supersedes** | `fused-w4a16-gemm-project-description.md` v1.0 |
| **Document type** | Project brief for external review |
| **Status** | Proposal / design phase; v1.0 reviewed, scope revised on evidence |
| **Domain** | GPU systems · low-precision LLM inference · CUDA / CUTLASS / Triton kernel engineering |
| **Primary hardware** | DGX H100 (sm_90) — authoritative results. RTX 3050 6GB (sm_86) — development and portability chapter |
| **Audience** | Technical reviewers (faculty, mentors, industry), and the maintainers of the stacks this targets |

---

## 1. Thesis

> **Block-scaled 4-bit formats deliver their memory saving but almost none of their throughput on Hopper-class hardware, because the software dequantization path is not designed for the memory-bound decode regime. This work quantifies that gap, closes as much of it as the architecture permits, and maps the resulting speed/accuracy frontier across MXFP4, NVFP4, and INT4-with-group-scales.**

The thesis is falsifiable three ways, which is the point. The gap may prove structurally unavoidable on Hopper; it may be smaller than the byte arithmetic suggests; the recovered throughput may not survive real model shapes. **Each of those outcomes is a publishable result**, because nobody has measured it carefully. A well-chosen question cannot be lost, only be surprising.

---

## 2. One-paragraph summary

Four-bit block-scaled weight formats — MXFP4 (E2M1 elements with a shared E8M0 power-of-two scale per 32 values) and NVFP4 (E2M1 with an E4M3 scale per 16, plus a tensor-level scale) — are shipping in production checkpoints today. Native tensor-core support for them arrived with Blackwell, one hardware generation ahead of the overwhelming majority of deployed datacenter capacity. On Hopper there is no native FP4 multiply: weights are stored at 4 bits in HBM and dequantized to FP8 or BF16 inside the kernel. Current production paths therefore **retain the memory saving and forfeit the speed** — vLLM's FP4 software fallback on H100/A100 reduces footprint without improving throughput over FP8. Yet in the bandwidth-bound decode regime, where time is weight traffic, MXFP4 moves 1.88× fewer bytes per weight than FP8 and should be correspondingly faster. This project builds the measurement apparatus to locate the missing factor, implements a `wgmma`/TMA-based fused dequantization-GEMM path to recover it, and reports the speed/accuracy trade-off across three competing 4-bit formats. Deliverables are a benchmark harness, kernels, a technical report, and an upstream contribution. The project is explicitly a **kernel-and-measurement study, not an inference engine**.

---

## 3. Motivation

### 3.1 The arithmetic that defines the opportunity

For weight-only quantization at small batch size, per-layer time is dominated by streaming the weight matrix from HBM. Bytes per weight, including scale overhead:

| Format | Element | Scale | Bytes/weight | Ratio vs FP8 | Ratio vs FP16 |
|---|---|---|---:|---:|---:|
| FP16 / BF16 | 16 bits | — | 2.000 | 0.50× | 1.00× |
| FP8 (E4M3) | 8 bits | per-tensor/channel | 1.000 | 1.00× | 2.00× |
| INT4, group 128 | 4 bits | FP16 per 128 | 0.516 | 1.94× | 3.88× |
| **MXFP4** | **E2M1** | **E8M0 per 32** | **0.531** | **1.88×** | **3.76×** |
| **NVFP4** | **E2M1** | **E4M3 per 16 + tensor** | **0.563** | **1.78×** | **3.56×** |

In the memory-bound regime these ratios *are* the available speedup. The current Hopper software path delivers approximately **1.0× against FP8**. The project's central quantity is therefore concrete and known in advance: **~1.9×, of which ~0% is realized.** Very few kernel projects begin with the size of the prize already computed.

### 3.2 Why the gap exists, mechanically

Four causes, which the project separates rather than conflating:

1. **Dequantization instruction cost on the critical path.** Naive unpacking — shift, mask, integer-to-float convert, multiply by scale — costs five or six instructions per weight and bottlenecks on *issue rate* rather than memory at exactly the shapes where 4-bit is supposed to win. The W4A8 literature treats this as the fundamental limit; that claim has never been cleanly isolated from the confounds below.
2. **Wave quantization.** At `M`=1, `N`=4096, with a 256-wide output tile, a kernel launches 16 CTAs. On an H100's 132 SMs that is 12% of the machine. Most reports of "4-bit is barely faster" resolve to this rather than to dequantization cost.
3. **Launch and dispatch overhead.** An ideal MXFP4 kernel for a 2880×2880 layer on an H100 completes in ~1.5 µs. Kernel launch overhead is 3–8 µs. **The launcher is larger than the kernel.** Any un-graphed measurement is measuring the wrong thing.
4. **Layout mismatch with the Hopper datapath.** `wgmma` reads B operands from shared memory under specific swizzle patterns, and TMA imposes its own layout constraints. A weight layout designed for Ampere's register-resident `mma` does not satisfy them, so ports pay shuffles and staging that erase the byte advantage.

### 3.3 Why the problem is real and not academic

- MXFP4 and NVFP4 checkpoints are shipping and being served on Hopper now.
- Production kernels have **open, unclaimed failures** in this exact path: `moe_wna16_marlin_gemm` crashes on MXFP4 GPT-OSS-20B at `K=N=2880` because Marlin requires 128-alignment and 2880/128 = 22.5. The quantization is valid — 2880 is divisible by 32, the MXFP4 block size — only the kernel's thread tiling is incompatible. The issue is closed as *not planned* and stale, and is noted as likely affecting other models with non-128-aligned dimensions.
- Hopper capacity will be deployed and amortized for years. This is not a transitional concern.

### 3.4 Why the answer is not simply "wait for Blackwell"

Recent FP4 work targeting native hardware (B200, RTX 5090) reports 2.2–3.6× and 4–6× end-to-end over FP16 — but it is a *quantization algorithm* contribution and does not address kernel-level dequantization on hardware without native FP4. Two findings from that line of work sharpen this project rather than obsolete it:

- **MXFP4's power-of-two scaling costs accuracy**, and NVFP4's small group size limits outlier mitigation. So the fast dequantization path (a power-of-two scale is an *exponent add*, not a multiply) is speed bought with error. That trade-off is a frontier to measure, not a footnote.
- **"FP4 is not an automatic upgrade over INT4."** This keeps INT4-g128 as a live comparison arm rather than legacy, and preserves the cleanest available roofline baseline.

---

## 4. Objectives

### 4.1 Primary

1. **Build a trustworthy measurement apparatus** for low-precision GEMM: L2-flushed, CUDA-graph-captured, clock-logged, measured against empirical rather than specified peak bandwidth, with structured correctness probes. Release it as a standalone artifact.
2. **Establish the baseline landscape** by benchmarking existing public kernels (Marlin, Machete, vLLM FP4/MXFP4 paths, Triton GPTQ) under that protocol on identical hardware, and reporting how much published figures move under correct measurement.
3. **Decompose the Hopper FP4 gap** into its four mechanical causes (§3.2), with profiler evidence per cause, and test the W4A8 critical-path claim directly.
4. **Implement a fused dequantization-GEMM path for sm_90** using `wgmma` and TMA, with the weight layout *derived from* the datapath's fragment and swizzle requirements rather than adopted from an Ampere design.
5. **Handle non-128-aligned `N`/`K`** correctly and without a performance cliff — the unclaimed bug class from §3.3.
6. **Characterize the speed/accuracy frontier** across MXFP4, NVFP4, and INT4-g128 on a frozen shape set.
7. **Produce the report**: five specified figures (§8.3), preregistered predictions, honest gap analysis, published negative results.
8. **Land a contribution upstream** in vLLM, SGLang, or torchao.

### 4.2 Secondary (time permitting)

1. Triton and CuTe-DSL twins, with a controlled accounting of performance versus lines of code versus time-to-first-correct.
2. Grouped / MoE variant with variable `M` per expert.
3. Routing-skew characterization: how much of the recovered throughput survives real, skewed expert routing distributions.
4. Ampere (sm_86) portability chapter.
5. Autotuning over tile configurations.

### 4.3 Explicit non-objectives

| Out of scope | Rationale |
|---|---|
| Full inference engine (continuous batching, paged KV, serving API) | Saturated product layer; dilutes the kernel contribution |
| Re-implementing vLLM / SGLang / TensorRT-LLM | Not the goal; these are the *integration targets* |
| New quantization *algorithms* | Focus is kernels. MR-GPTQ-class algorithm work is orthogonal and already covered |
| Blackwell native-FP4 kernels | The premise is *absence* of native support. Blackwell is the contrast case, not the target |
| Attention / KV-cache kernels | Different correctness surface, crowded by expert teams. A later project |
| Training kernels, backward pass | Inference-forward only |
| GPTQ `act_order` / `desc_act` | Permutes `K`, destroying the layout contiguity the whole derivation depends on. Explicitly excluded from v1 |
| Asymmetric zero-points | Symmetric only in v1. The activation-sum folding trick is documented as the path forward |
| Multi-GPU tensor/pipeline parallelism as a *subject* | TP appears only as a shape generator (§9), not as a system to build |
| Beating Machete or DeepGEMM on all shapes | Honest gap analysis is the standard, not universal victory |

**Scope note.** With a DGX available, hardware is no longer the constraint that protected this list. That makes the list more important, not less. It is frozen at review sign-off.

---

## 5. Scope

### 5.1 In scope

- **Formats (v1):** MXFP4 (E2M1 + E8M0/32), NVFP4 (E2M1 + E4M3/16 + tensor scale), INT4 symmetric group-128. Activations FP16/BF16; **FP32 accumulation** (decided, not deferred).
- **Operation:** weight-only dense GEMM for Transformer linear layers. Two kernels: `qgemv_mx` for `M`≤8 and `qgemm_mx` for `M`≥16, with the dispatch threshold treated as a measured result.
- **Architectures:** sm_90 authoritative; sm_86 for development and the portability chapter.
- **Languages:** CUDA C++ with inline PTX (primary), CUTLASS/CuTe where it earns its place, Triton (twin), Python (harness).
- **Evaluation:** correctness suite, latency, percent-of-ideal bandwidth, Nsight Compute counters, end-to-end decode latency on one real model.

### 5.2 Hardware

- **DGX H100** — all reported results. 132 SMs, 50 MB L2, HBM3, sm_90.
- **RTX 3050 6GB Laptop** — development. 20 SMs, 1.5 MB L2, 110 GB/s measured (83% of 131.7 GB/s theoretical), sm_86.

Measured on the laptop, and load-bearing for the plan: with 20 SMs, a 16-CTA launch nearly fills the machine, so **wave quantization is invisible locally.** All occupancy and split-K conclusions must be drawn on the DGX. Conversely, the laptop's 1.5 MB L2 makes it an excellent *correctness* rig for bandwidth behaviour, because an 8 MB weight matrix cannot hide in cache there — whereas on the H100's 50 MB L2 every frozen shape fits entirely, making L2 flushing mandatory rather than advisable.

### 5.3 Prerequisite gate

**Nsight Compute counter access must be confirmed on both machines in week one.** `ncu` currently returns `ERR_NVGPUCTRPERM` on the laptop; under WSL2 the fix is Windows-side (NVIDIA Control Panel → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → allow all users, then `wsl --shutdown`). On the DGX this is an administrator policy decision, and shared clusters frequently restrict it because collection serializes the GPU. If counter access is unavailable, the fallback — deriving throughput from timing and known byte counts — is a weaker methodology that must be designed in from the start, not retrofitted. **This is a gate, not a risk.**

---

## 6. Technical approach

### 6.1 The dequantization sequence (this is the core, not the plumbing)

The weight layout is not an input to the kernel design. It is an **output** of the instruction sequence and the datapath's fragment requirements. The derivation runs backwards:

**Step 1 — choose the extraction sequence.** Naive conversion costs 5–6 instructions per weight. The established alternative constructs the target float's bit pattern directly: OR the 4-bit payload into the mantissa of a fixed exponent (`0x6400` yields `1024.0 + v`) with a single `lop3.b32` that produces **two** values as an `f16x2`, then remove the bias and apply the scale with one fused multiply-add. Roughly two instructions per weight.

**Step 2 — for MXFP4, exploit the power-of-two scale.** E8M0 is a pure exponent. Applying it is an **exponent add**, foldable into the same bit manipulation rather than requiring a separate multiply. This is MXFP4's structural speed advantage over both NVFP4's E4M3 scale and INT4's FP16 group scale — and, per §3.4, the source of its accuracy cost. Quantifying both sides of that trade is a primary result.

**Step 3 — derive the nibble order.** `lop3` only yields a usable pair if nibbles within each 32-bit word are pre-permuted (`0,2,4,6,1,3,5,7` rather than sequential). The permutation is dictated by step 1, not chosen.

**Step 4 — derive the global load pattern from the datapath.**
- On **sm_86**: `mma.sync` requires each lane to hold specific B-fragment elements. `ldmatrix` does not operate on 4-bit data, so without an offline permutation you pay a shared-memory round trip plus shuffles. Marlin's answer is to permute at pack time so one 128-bit load per thread delivers exactly that thread's nibbles.
- On **sm_90**: `wgmma` is asynchronous and reads B from **shared memory** under specific swizzle patterns, and TMA imposes its own descriptor-level layout constraints. An Ampere layout does not satisfy them. **This is why Machete exists as a distinct kernel rather than a port of Marlin, and it is the hardest and most valuable derivation in the project.**

**Step 5 — document the 32-bit word diagram and the derivation.** Written from first principles, citing Marlin and Machete as prior art, stating for each element whether it was independently derived or adapted. This is both the integrity answer (§12) and the strongest single artifact for demonstrating depth.

### 6.2 Parallelization

- **Split-K is a first-class design axis**, not an optimization. Put the CTA-count arithmetic for every frozen shape in the design note. Cover reduction strategy (atomic versus two-pass), determinism implications for §8.1's tolerances, and interaction with block-scale boundaries.
- **Persistent / stream-K style partitioning** for awkward tile counts — directly relevant to the non-128-aligned shapes in §9.
- **Thread-block clusters and distributed shared memory** (sm_90) as an explored axis.
- **CUDA graphs are mandatory**, not a refinement. At `M`=1 the ideal kernel is shorter than its own launch overhead.

### 6.3 Correctness

**Structured probes first — random matrices hide layout bugs.** If one nibble in eight lands in the wrong lane, every output is a sum over `K` random terms with a slightly wrong subset; the error averages into something resembling FP16 noise and passes a tolerance check. This is the characteristic failure mode of hand-packed quantized kernels.

1. **One-hot weights, basis-vector activations.** Asserts `W[i,j]` lands where intended. An *index* test — must pass exactly.
2. **Distinct primes as each block's scale.** Any block-index error becomes large, obvious, and localized.
3. **Monotone ramp weights** (`v = (i*j) mod 16`). A nibble-order error shows visible structure.
4. **Then** random matrices against an FP32/FP64 reference computed on the dequantized weights, bounded near `sqrt(K)·eps`, with the error **map** inspected for structure rather than only its maximum.
5. **Cross-check** the fused kernel against a dequantize-then-cuBLAS path. Identical inputs make this the sharpest available test.

Note: bitwise agreement with cuBLAS is **not achievable** and is not claimed — different accumulation order, split-K strategy, and internal precision.

### 6.4 Measurement protocol

Non-negotiable, and released as the harness:

- **L2 flush or buffer rotation** between iterations. Report both flushed and hot figures where they differ; the gap is a finding. On the H100's 50 MB L2, every frozen shape is otherwise fully resident.
- **CUDA graph capture** for all latency measurement, with launch overhead reported as its own line item.
- **Empirical peak bandwidth** from a streaming probe, never the spec sheet.
- **Clock logging** per sample: SM clock, memory clock, power draw, temperature. Locked clocks via `-lgc` where permitted.
- **DGX hygiene:** exclusive allocation where possible with co-tenant processes logged, MIG and MPS off, persistence mode on, one GPU pinned via `CUDA_VISIBLE_DEVICES`, host threads bound with `numactl` to the attached socket.
- **Preregistered predictions** in a timestamped commit before the first measurement (§8.2).
- **Environment reported at node granularity**, not GPU granularity.

### 6.5 Relationship to existing software

| Component | Role |
|---|---|
| **Machete** (CUTLASS 3.x + TMA, Hopper) | Primary performance reference on sm_90 |
| **Marlin / GPTQ-Marlin** | Primary reference on sm_86; the 128-alignment constraint is the bug class in §4.1.5 |
| **vLLM FP4/MXFP4 fallback paths** | The baseline being improved on. The integration target |
| **DeepGEMM** | Reference for grouped/FP8 kernel design, if §4.2.2 is reached |
| **CUTLASS / CuTe** | Design reference; used where it earns its place. "Why not just instantiate CUTLASS mixed-input?" is answered in the report |
| **cuBLAS / cuBLASLt** | FP16 and FP8 dense baselines |
| **Triton** | Second implementation; the DSL-cost measurement |
| **llm-compressor / torchao** | Checkpoint production for the frozen shape set |

---

## 7. The deliverable ladder

Four rungs, each independently publishable, each making the next credible. This structure exists to resolve the tension between contribution and completion: results begin accruing in month two, and the ambitious rung is upside rather than a single point of failure.

| Rung | Months | Content | Risk |
|---|---:|---|---|
| **R0 · Instrument** | 1–3 | Harness and protocol (§6.4). Then benchmark existing public kernels under it and report how much published figures move once L2 is flushed, launches graphed, clocks pinned, and peak measured | **Certain** |
| **R1 · Quantify** | 4–6 | Decompose the Hopper FP4 gap into §3.2's four causes with profiler evidence per cause. Three-way format comparison including accuracy. Test the W4A8 critical-path claim | **High** |
| **R2 · Close** | 7–9 | The `wgmma`/TMA kernel that recovers the gap. Layout derived per §6.1. Non-128-aligned shapes handled — on the path, not a detour | **Moderate** |
| **R3 · Extend** | 10–12 | Grouped/MoE variant; routing-skew characterization; Triton and CuTe-DSL twins with the cost accounting; sm_86 portability chapter | **The swing** |

**R0 produces a result regardless of what happens downstream.** It cannot fail to produce a finding, it is a *tool* others will use rather than merely read about, and it is the reproducibility contribution the field visibly lacks. It is simultaneously the insurance policy, the most likely path to adoption, and what makes every later number trustworthy.

**If R1 shows the gap is structurally unavoidable on Hopper, publish that and stop.** A well-evidenced negative result on a question the field has assumed the answer to is a genuine contribution. Abandoning R2 on evidence is a strength, not a failure.

Cut order if time compresses: R3 first, then R2's optional axes. **Never cut R1 to preserve R3.**

---

## 8. Success criteria

### 8.1 Must pass

| Criterion | Bar |
|---|---:|
| Structured correctness probes (one-hot / ramp / prime-scale) | **exact** |
| Random correctness vs FP32 reference on dequantized weights | **≤ sqrt(K)·eps**, error map free of structure |
| Fused vs dequantize-then-cuBLAS agreement | **≤ 2 ulp** |
| Percent of ideal bandwidth, `M`=1, sm_90 | **≥ 60%** |
| **MXFP4 vs FP8 speedup, `M`=1, sm_90** | **≥ 1.5×** (bytes give 1.88×; current fallback ~1.0×) |
| MXFP4 vs FP16 cuBLAS, `M`=1 | **≥ 2.5×** |
| Gap to Machete (sm_90) / Marlin (sm_86), `M`=1 | **≤ 1.5×** |
| Gap to Machete / Marlin, `M`=16 | **≤ 2.5×** |
| Non-128-aligned shapes (`K=N=2880`) | **correct, benchmarked, no cliff** |
| Crossover `M` identified on both architectures | **required**, with stated mechanism |
| Gap decomposition across §3.2's four causes | **required**, with profiler evidence per cause |
| Ablation waterfall | **≥ 5 steps**, each with measured delta |
| Predictions preregistered before first measurement | **required** |
| Third-party reproduction from instructions | **required** |

### 8.2 Preregistered predictions

Committed before measurement, from the roofline. Reported as held or refuted either way.

```
sm_86 (measured: 110 GB/s achieved, ridge ≈ 92 FLOP/byte)
  W4 compute-bound at    M ≈ 23        FP16 at M ≈ 92
  predicted crossover:   M ≈ 20–40

sm_90 (assume ~3.0 TB/s achieved — MEASURE FIRST; ridge ≈ 165 FLOP/byte)
  W4 compute-bound at    M ≈ 41        FP16 at M ≈ 165
  predicted crossover:   M ≈ 40–80

t_ideal, N=K=4096, sm_90:  MXFP4 2.4 µs · FP8 4.6 µs · FP16 11.2 µs
launch overhead:           3–8 µs   ← larger than the kernel at M=1
```

**Prediction:** the recovered MXFP4-vs-FP8 speedup will exceed 1.5× at `M`≤4, fall below 1.2× by `M`≈32, and invert (FP8 faster) by `M`≈64, because dequantization ALU cost is fixed per weight while the bandwidth advantage decays with arithmetic intensity.

**Prediction:** the dominant term in the current gap will be wave quantization and launch overhead, not dequantization instruction cost — i.e. the W4A8 critical-path argument will prove *partially* wrong for the decode regime specifically.

### 8.3 Required figures

The figure list is the experiment design. If these five exist, the report writes itself.

1. **Latency versus `M`**, log-x, all baselines on one axis, both architectures — the crossover plot.
2. **Percent of ideal bandwidth versus `M`**, per format.
3. **Gap decomposition waterfall** — from the current fallback to the final kernel: +graph capture, +vectorized loads, +`lop3`/exponent-add dequant, +derived layout, +split-K, +async pipeline. Each step with its measured delta. **Publishable at any absolute performance level, which is why it is a must-pass.**
4. **Roofline** with the operational-intensity point per `M` and per format, both architectures.
5. **Speed/accuracy frontier** — MXFP4 vs NVFP4 vs INT4-g128, throughput against a perplexity or task-accuracy proxy.

Plus an Nsight table per variant: memory throughput %, ALU/FMA pipe %, dominant warp stall reason, achieved occupancy.

### 8.4 Non-criteria

- Beating TensorRT-LLM or full-engine end-to-end tokens/second.
- Universal victory over Machete or DeepGEMM.
- GitHub stars, product adoption, or leaderboard positions.
- Novel quantization-algorithm accuracy results.

---

## 9. Frozen shape set

Frozen at review sign-off, before any code.

| Layer archetype | N | K | Why it earns a slot |
|---|---:|---:|---|
| q_proj / o_proj | 4096 | 4096 | Square canonical case |
| k_proj / v_proj (GQA) | 1024 | 4096 | Small `N` — tail tiles, starved occupancy |
| gate_proj / up_proj | 14336 | 4096 | Wide `N` — the friendly case |
| down_proj | 4096 | 14336 | Deep `K` — where split-K matters most |
| **GPT-OSS-class** | **2880** | **2880** | **Not 128-aligned. The live, unclaimed bug class** |
| fused QKV | 6144 | 4096 | For the end-to-end check |

- `M` ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048} — the full sweep, because the crossover *is* the result.
- Block/group sizes: MXFP4 32, NVFP4 16, INT4 128 primary and 64 as stress case.
- `TP` ∈ {1, 2, 4, 8}. With TP=8, per-GPU `N` for a 4096-wide projection falls to 512 — a different tile-count and tail regime, and what production actually runs. Costs a loop; converts "here is the crossover" into "here is how the crossover moves with parallelism strategy," which is the version a serving team can act on.
- Symmetric scales only. `act_order` excluded.

---

## 10. Deliverables

| # | Deliverable | Description |
|---|---|---|
| **D1** | Design note | The layout derivation (§6.1) with the 32-bit word diagram, format definitions, numerical policy, preregistered predictions |
| **D2** | Benchmark harness | Standalone, released independently. L2-flushed, graph-captured, clock-logged, empirical-peak, CSV output, multi-architecture, one command |
| **D3** | Baseline study | Existing public kernels re-benchmarked under D2's protocol, with deltas against published figures |
| **D4** | CUDA sources | `qgemv_mx` / `qgemm_mx` for sm_90 and sm_86, plus build system |
| **D5** | Test suite | Structured probes plus randomized regression, fixed seeds, one-command local run. CI covers compilation and CPU-side pack/unpack round-trips (no free GPU runners) |
| **D6** | Technical report | The five figures of §8.3, the gap decomposition, preregistered predictions with outcomes, published negative results, honest gap analysis |
| **D7** | Reproduction package | Container, one command, raw CSVs committed, plotting scripts |
| **D8** | Upstream contribution | PR to vLLM, SGLang, or torchao. Starting point: the stale 128-alignment bug |
| **D9** | Triton / CuTe twins | With the performance / lines-of-code / time-to-correct accounting |

Optional: preprint, blog post, conference or GPU-MODE talk.

---

## 11. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Profiler counters unavailable** | D6's analysis and §6.4's protocol both blocked | **Gate, resolved in week one on both machines.** Fallback methodology designed in advance if refused |
| **L2 residency invalidates all H100 results** | Silent, total. Physically impossible roofline figures | Flush or rotate from day one, built on the laptop where the effect is invisible so the code is right before it matters |
| **Launch overhead exceeds kernel time** | Primary result measures the launcher | CUDA graphs mandatory from R0; overhead reported separately |
| **Gap proves structurally unavoidable on Hopper** | R2 unreachable | **This is a publishable result.** R0 and R1 stand alone. Stop and report |
| **Laptop conclusions do not transfer** | Occupancy and split-K findings wrong in the optimistic direction | sm_90 authoritative for all reported numbers; sm_86 is development and portability only |
| **`wgmma`/TMA layout derivation stalls** | R2 slips | Timeboxed. Fall back to a CUTLASS-3 mixed-input instantiation and report the comparison — a weaker but real result |
| **Field moves during the project** | Thesis obsoleted mid-flight | Quarterly literature check. The measurement apparatus and the decomposition survive format churn; the kernel may not |
| **Abundant hardware invites scope creep** | Failure to finish | §4.3 frozen at sign-off. This was previously enforced by resource scarcity and must now be enforced deliberately |
| **Silent numerical bugs** | Wrong results that pass tests | Structured probes before random (§6.3); `compute-sanitizer` in CI |
| **Upstream contribution rejected or unwanted** | D8 unreachable | **Validate demand before building** (§13). A named maintainer confirming want, or an early no, either way |

---

## 12. Ethics, integrity, and openness

- The work improves efficiency of general-purpose inference; the same techniques serve beneficial and harmful deployments alike. Efficiency gains are dual-use in the same trivial sense as all systems optimization.
- **No fabricated numbers. Ranges and methodology over single favourable points. Negative and failed experiments published when material.** Preregistration (§8.2) makes these structural rather than aspirational.
- **On studying open kernels:** Marlin and Machete are permissively licensed; reading and being influenced is unambiguously fine. The genuine risk is reimplementation-by-memory of permutation tables — code that cannot be explained, arriving without attribution. Mitigation: derive the layout from first principles and write the derivation down (§6.1 step 5), cite both as prior art, and state per design element whether it was independently derived or adapted. Anything vendored goes in `third_party/` with its license intact.
- Model and checkpoint licenses respected for all shapes and end-to-end validation.

---

## 13. Immediate next steps, in order

**None of the first four involve writing kernel code.**

1. **Resolve profiler counter access** on both machines (§5.3). Gate.
2. **Run the streaming probe on an H100** for real achieved bandwidth. This single number fixes the denominator for every claim in the project and completes §8.2's predictions.
3. **Reproduce the 128-alignment crash** on the DGX with an MXFP4 GPT-OSS-class checkpoint. Confirms the bug is live and unclaimed, and yields the first PR.
4. **Validate demand upstream.** Ask in the vLLM / SGLang / torchao channels whether an MXFP4 Hopper throughput path is wanted and unowned. This is the only step that can tell you the project is *not* worth doing — every other step assumes it is.
5. **Freeze §4.3 and §9**, commit §8.2's predictions, then begin R0.

---

## Appendix · Document control

| Field | Value |
|---|---|
| Version | 2.0 |
| Supersedes | v1.0 (dense W4A16, sm_86-first, naive-dequant baseline) |
| Principal changes from v1.0 | Target changed from dense INT4-on-Ampere to block-scaled FP4-on-Hopper; success criteria replaced with falsifiable numbers; layout derivation promoted from plumbing to core; split-K, CUDA graphs, and L2 flushing added; predictions preregistered; upstream contribution added as a deliverable; deliverables restructured as a risk-stacked ladder |
| Basis for revision | External review dated 2026-08-11; hardware measured directly; state of the art verified by literature and issue-tracker survey |
| Next step after review | Freeze feedback → D1 design note → R0 |
