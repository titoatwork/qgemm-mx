# External Technical Review

## Fused Low-Precision GEMM for Compressed LLM Inference — Project Brief v1.0

| Field | Value |
|-------|-------|
| **Reviewing** | `fused-w4a16-gemm-project-description.md` v1.0 |
| **Review date** | 2026-08-11 |
| **Reviewer role** | External technical review, as invited by §14 |
| **Scope of review** | Technical design, methodology, scope, positioning, and strategic alignment |
| **Hardware verified** | RTX 3050 6GB Laptop (sm_86) measured directly; H100 SXM figures derived |
| **State of the art verified** | Web research conducted 2026-08-11; sources in Appendix C |
| **Recommendation** | **Proceed, with §8.1 rewritten and the primary target changed** |

---

## How to read this document

Sections are ordered so that you can stop early and still have acted on the most important things.

- **§1** is the verdict. Read it first.
- **§2–3** are findings on the brief as written — the things to fix regardless of what project you end up building.
- **§4** is what I measured on your hardware. Two items here are blockers.
- **§5** is what I verified about the current state of the art, including **three corrections to advice I gave you earlier in conversation.**
- **§6–9** are the conversion: a revised thesis, a risk-stacked plan, and replacement success criteria.
- **§10–12** are the scorecard, the structural tensions, and the answers to your §14 checklist.
- **§13** is what to do this week.

Findings are labelled `F1`–`F17` and are referenceable. Severity is **Blocking** (fix before writing code), **Significant** (fix before freezing scope), or **Minor**.

---

## 1. Verdict

**The brief is a strong piece of work aimed at a settled question.**

The scope discipline in §4.3 is genuinely excellent and rare at this stage. The commitments in §15 are the right instincts, stated before anyone asked for them. The writing is clear enough that a reviewer can actually engage with it, which is more than most proposals achieve. These are real strengths and they are not consolation prizes — they are the parts of the project that are hardest to teach.

The problem is threefold, in descending order of importance:

1. **The target is solved.** Dense weight-only INT4 decode GEMM was *the* inference problem of 2023–24 and it got solved — Marlin, Machete, CUTLASS mixed-input, BitBLAS, torchao, exllamav2 — and shipped inside every serving stack. Executing this brief well produces a competent reimplementation of settled work.
2. **The document specifies a process in detail and a kernel hardly at all.** There is one equation, no bit layout, no dequantization instruction sequence, no occupancy arithmetic, no numbers. Roughly a quarter of the document (§§11, 12, 15, 16) carries no technical content. For the industry audience §12 names, the ratio reads as more spec than code.
3. **The one hard performance criterion is guaranteed to pass.** §8.1.2 cannot be failed. See `F1`.

**The conversion is additive, not a restart.** Everything in the brief survives — it stops being the deliverable and becomes the *calibration harness*: the known-good operation, with a known-good reference to check against, that proves your measurement methodology is sound before you point that methodology at something nobody has measured. §6 proposes what to point it at, and the research in §5 changed my answer from what I first suggested.

One thing to internalize before the details: **the artifact being evaluated first is not your kernel, it is your choice of problem.** Kernel ability is scarce but legible — demonstrable in an interview, teachable in a semester, verifiable from a repository. Judgment about which problem deserves eight months is scarcer, harder to teach, and very hard to fake. This brief demonstrates excellent discipline aimed at a settled question, which reads as strong execution paired with weak calibration. That is the specific thing worth fixing, and it is worth more than any amount of additional speedup.

---

## 2. Blocking findings

### F1 · The must-pass baseline is a straw man — **Blocking**

§8.1.2 defines v1 success as "clear, reproducible speedup vs naive dequant + cuBLAS." Work out what that baseline does at `M=1`, `N=K=4096`:

```
read packed INT4 weights          8.4 MB
write dequantized FP16 weights   33.6 MB
cuBLAS reads them back           33.6 MB
                                 ───────
total                           ~75.6 MB
```

An ideal fused kernel moves ~8.65 MB. You win by 5–8× on arithmetic alone, before writing a single good line of CUDA. **A criterion you cannot fail measures nothing**, and a reviewer who performs this division in their head will discount the entire results section — which is a much worse outcome than having no criterion at all.

It also conceals the interesting failure. Your fused path will *lose* to plain FP16 cuBLAS somewhere in the tens of `M`, because you have traded HBM traffic for dequantization work in the inner loop and can no longer feed the Tensor Cores as densely. §6.4 lists FP16 cuBLAS as baseline (2) and Marlin as "optional." **That ordering is inverted.** FP16 cuBLAS is the only baseline that can tell you something you do not already know; Marlin is the only one that can tell you whether your kernel is good.

**Fix.** Four baselines, all mandatory:

| Baseline | Role |
|----------|------|
| Packed-weight stream (`memcpy`-class read of the quantized weights) | Gives `t_ideal`. **"Percent of ideal" becomes your headline metric** — self-normalizing, hardware-independent, impossible to inflate |
| FP16 cuBLAS | The honest comparison. Ideal speedup from bytes alone is 3.9×, not "4× ish" |
| GPTQ-Marlin (and Machete on Hopper) | Calibration against known-good on your own hardware |
| Naive dequant + cuBLAS | Sanity floor only. Report as "×N vs the obvious wrong way," never as the result |

Concrete targets are in §9.

---

### F2 · The actual technical core of the project is missing — **Blocking**

§6.1 draws the pack tool as a box upstream of the kernel. §4.1.1 treats the pack format as something you "specify" and "document." This gets the causality backwards, and it is the deepest gap in the brief.

In a fast low-precision GEMM the weight layout is not an input to the kernel design. It is an **output** of two hard constraints:

**(a) The dequantization instruction sequence.** Naive unpacking — shift, mask, `cvt.rn.f16.s32`, multiply by scale — costs five or six instructions per weight and will bottleneck you on *issue rate*, not memory, at exactly the small-`M` shapes you care about. The established alternative is to never convert at all: OR the 4-bit value directly into the mantissa of a fixed FP16 exponent (`0x6400` yields `1024.0 + v`) using a single `lop3.b32` that produces **two** values as an `f16x2`, then remove the `1024` bias and apply the group scale with one fused multiply-add. Roughly two instructions per weight. This is what TensorRT-LLM's interleaved numeric conversion and Marlin both do.

**(b) The `mma` fragment layout.** `lop3` only yields a usable `f16x2` if the nibbles within each 32-bit word are pre-permuted (order `0,2,4,6,1,3,5,7` rather than sequential). And `mma.sync` requires each lane to already hold specific elements of the B fragment — `ldmatrix` does not operate on 4-bit data, so unless the layout is permuted offline you pay a shared-memory round trip and a pile of shuffles. Marlin's answer is to permute weights at pack time so a single 128-bit global load per thread delivers exactly the nibbles that thread's `mma` instructions need, with no shuffle at all.

So the "weird" layouts in real formats are not arbitrary conventions to be documented. **They are the derivation.** A brief that treats layout as plumbing produces a kernel that is correct and 3–4× off the state of the art, and — more damaging — an author who does not know why.

**Fix.** Make the layout derivation the primary content of D1: start from the chosen `mma` (or `wgmma`) shape, work backwards through the extraction sequence to the required nibble order and the required per-thread global load pattern, and include the 32-bit word diagram. Add an explicit stage between R2 and R3 for it. See also `F16` — on Hopper this derivation is different and harder, and that is where the transferable skill lives.

---

### F3 · No mention of split-K, and split-K *is* the decode bottleneck — **Blocking**

Run the occupancy arithmetic for your flagship shape. At `M=1`, `N=4096`, with a 256-wide output tile you launch **16 CTAs**. On an H100 that is 16 of 132 SMs — 12% of the machine. The overwhelming majority of "my W4 kernel is only 1.5× faster than FP16, why?" reports resolve to exactly this: wave quantization, not dequantization cost.

The fix is partitioning the `K` reduction across CTAs and reducing globally — split-K, or the striped/persistent variants that better handle awkward tile counts. That drags in a design layer the brief never names: reduction buffers, atomics versus a two-pass reduce, determinism of the result, and interaction with group-scale boundaries.

**Fix.** Add split-K to §6.4 as a first-class design axis and to the R3 ablation. Put the CTA-count arithmetic for your frozen shapes *in the brief*. Note in §6.3 that split-K makes accumulation order non-deterministic, which changes what your correctness tolerances can promise.

**Note the hardware trap here:** with 20 SMs on your laptop, a 16-CTA launch nearly fills the machine, so this entire problem is **invisible locally**. See `F13`.

---

### F4 · The benchmark methodology omits the bug that will silently fabricate your results — **Blocking**

§6.4 and §10 treat measurement risk as timing noise, addressed by multi-trial statistics. The real hazard for a memory-bound kernel is **L2 residency.**

A 4096×4096 INT4 weight matrix is 8.4 MB. **An H100's L2 is 50 MB.** Every shape in your frozen set fits — even the 14336×4096 layer is only 29 MB. Benchmark the same buffer in a hot loop and it never leaves L2 after the first iteration: you will measure L2 bandwidth, report a spectacular speedup, produce a roofline plot that is physically impossible, and have no way to notice from inside the harness. This is the standard way projects of this kind generate numbers that do not survive review.

Three more items belong in the same section:

- **Clock stability.** `nvidia-smi -lgc` needs root. On a shared DGX that is somebody else's decision — ask once, and meanwhile log `clocks.sm`, `clocks.mem`, `power.draw`, and `temperature.gpu` alongside every timing sample.
- **Launch overhead.** See `F14` — on an H100 this is not a correction term, it is larger than the kernel.
- **Peak bandwidth.** Measure it (`nvbandwidth` or a STREAM-style kernel), never quote the spec sheet. Using the theoretical figure makes every efficiency claim you publish better than it is.

**Fix.** Flush L2 between iterations, or rotate across enough distinct weight buffers to exceed it — and **report which, plus both numbers if they differ**, because the gap is itself a finding. Then state the external-validity caveat explicitly: in a real decode step the entire model streams through L2, so neither the flushed nor the hot microbenchmark exactly predicts end-to-end behaviour. Owning that limitation is far stronger than having a reviewer find it.

---

## 3. Significant and minor findings

### F5 · Success criteria are unfalsifiable — **Significant**

"Magnitude depends on GPU; direction and analysis required" (§8.1.2) cannot be failed. §8.2's "within a documented factor" likewise — any factor is a factor. You committed in §15 to no inflated claims and then left the criteria loose enough that no claim can be checked against them. Pick numbers *now*, before you know whether you can hit them; that is the only moment at which a target means anything. See §9.

### F6 · You call it a GEMM project; your flagship shape is a GEMV — **Significant**

At `M`=1–8 this is not a GEMM in any useful sense. Tensor Cores are wasted — `mma.m16n8k16` discards 8 to 15 of its 16 `M` slots — `A` fits in registers so there is no `A`-tile to stage, the output is tiny, and time goes to streaming `B` and to the split-K reduction. At `M`≥32 it is a real tiled Tensor Core GEMM with a completely different structure. Production stacks ship two kernels for this reason.

The brief plans one kernel and files "optional Tensor Core path" under R3, which buries the project's most interesting decision inside an optional bullet.

**Fix.** Declare it: v1 is `qgemv_w4a16` for `M`≤8 (no Tensor Cores, split-K, register-resident `A`) and `qgemm_w4a16` for `M`≥16 (Tensor Core, tiled, `M` padded to 16). Then "at which `M` should the dispatcher switch, and why" becomes a headline measured result rather than an unstated assumption. Note that this pattern has prior art worth citing — the MoGE work dispatches between GEMV and GEMM modes by workload intensity.

### F7 · Related work is roughly two years stale — **Significant**

§13 lists GPTQ, AWQ, Marlin, cuBLAS/CUTLASS, Triton, vLLM. That is the 2023 map. §5 below is the current one. Two questions a reviewer will ask immediately, which the brief has no answer to:

- **"Why not just instantiate CUTLASS mixed-input GEMM?"** FP16×INT4 mixed-input is a supported, shipped path in CUTLASS 3.x, and Machete is the CUTLASS-3, TMA-era successor to Marlin. Hand-rolling is now a deliberate choice rather than the only option. *"To understand it, and to have something to compare against"* is a perfectly good answer — but it must be stated.
- **"Isn't W4A16 the wrong target?"** The W4A8 line of work argues that dequantization on the critical path caps quantized throughput, which attacks your v1 premise directly. Engaging with it makes the brief look current; ignoring it makes it look assembled from old blog posts.

**Fix.** Add two paragraphs: *"Why W4A16 and not W4A8 or native FP4"* and *"Why hand-written and not CUTLASS mixed-input."* Replace §13 with the map in §5.

### F8 · §9 is titled "Work plan" and contains no plan — **Significant**

Six phases, zero durations, and a note saying durations assume part-time work — then no durations. Your own reviewer question 1 asks whether scope is realistic for the stated timeline, which the document has made unanswerable.

Estimate at ~12 h/week, assuming you are still acquiring CUDA fluency rather than already having it:

| Phase | Part-time | Note |
|---|---:|---|
| P0 — CUDA GEMM / roofline / format literacy | 4–8 wk | The real cost, routinely underestimated. Tiled matmul plus `mma` plus Nsight is not a weekend |
| P1 — cuBLAS reference + naive W4 + tests | 1–2 wk | Mostly harness. Do the L2-flush work here |
| P2 — correct fused kernel | 2–3 wk | Correct-and-slow is reachable. Layout bugs eat the tail |
| P3 — optimize to within 1.5× of Marlin | 4–10 wk | Unbounded. Timebox it and ship whatever the waterfall shows |
| P4 — Triton twin | 1–2 wk | Cheap — which is why it should not be last, see `F9` |
| P5 — bench, report, repro | 2–3 wk | Chronically underbudgeted. The report *is* the deliverable |

**4–6 months part-time** for a credible v1 as scoped. Narrowing to `M`≤8, symmetric scales only, one architecture, no Tensor Core path reaches **2–3 months** — and a finished narrow project beats a stalled broad one.

### F9 · Triton is sequenced last; it should be near-first — **Significant**

Triton sits at P4/R4, after all CUDA work. But Triton is by far the cheapest route to a *correct and reasonably fast* kernel — days, not weeks. Writing it early buys you a working reference implementation, a real performance number to beat, and a debugged pack format, all before you touch PTX. Sequenced last, it is instead the deliverable that gets cut when P3 overruns — and it was the cheap one.

Two caveats, to keep expectations honest. Triton has no native 4-bit type, so you load `int32`/`uint8` and shift; `tl.dot` wants FP16/BF16 operands; you cannot control the `mma` fragment layout; and the `lop3` trick requires `tl.inline_asm_elementwise` to express at all. Expect 50–75% of good CUDA. **That gap is not a disappointment — it is the finding**, and a quantified account of what the DSL costs you is one of the few genuinely differentiated things this project can publish.

**Fix.** Reorder to P1 → Triton → P2 → P3 → P5. Keep CUDA authoritative per §15; just stop making the cheap artifact depend on the expensive one finishing first.

### F10 · The correctness strategy will pass while the kernel is wrong — **Significant**

Two problems in §6.3.

First, **"bitwise … comparison vs FP16 reference" is not achievable** and should be struck. cuBLAS chooses its own accumulation order, split-K algorithm, and internal precision; you will not reproduce them.

Second, and more serious: **random matrices hide layout bugs.** If one nibble in eight lands in the wrong lane, every output element is a sum over `K`=4096 random terms with a slightly wrong subset — the error averages out to something resembling FP16 noise and sails through a tolerance check. This is *the* characteristic failure mode of hand-packed quantized kernels and it is the bug that costs a week.

**Fix — structured probes before random ones, all cheap:**

1. **One-hot weights and basis-vector activations.** Assert that `W[i,j]` lands exactly where you think. This is an *index* test, not a numerics test, so it must pass exactly.
2. **Distinct primes as each group's scale.** Any group-index error becomes a large, obvious, localized error.
3. **Monotone ramp weights** (`v = (i*j) mod 16`). A nibble-order mistake shows up as visible structure.

Then compare against an FP32/FP64 reference computed on the *dequantized* weights, with a bound around `sqrt(K)·eps`, and inspect the error **map** for structure rather than only its maximum. Separately, assert that your fused kernel and the dequant-then-cuBLAS path agree tightly — they consume identical dequantized values, so that is a much sharper test than either against a float reference.

### F11 · Four decisions are deferred that are not actually open — **Minor**

- **Accumulator.** §5.1 says "FP16/FP32 accumulation as documented." Decide: **FP32**. FP16 accumulation loses too much across `K`=4096 and no serious kernel uses it. Not a variable.
- **Zero points.** §3.2's "optional zeros" is a real fork. Symmetric (scale only) is meaningfully simpler and faster; asymmetric costs an extra operation per weight unless you fold the zero into a per-group correction using activation sums. Pick **symmetric for v1** and note the folding trick as the path to asymmetric.
- **`act_order`.** GPTQ's `desc_act` permutes the `K` dimension, destroying the contiguity your entire layout derivation depends on. Marlin shipped without it initially for exactly this reason. Declare it out of scope in §4.3 **explicitly**, or you will discover it halfway through R3.
- **Group size versus K-tile.** If group size is smaller than your `K`-tile, scales change mid-tile and the `K` loop must be restructured around group boundaries. Fix group size 128 as primary with the K-tile a multiple of it; treat 64 as the stress case.

### F12 · The report is specified by page count instead of by figures — **Minor**

"8–20 pages" (D6) constrains nothing. Specify the **figures**, because the figure list is what forces the experiment design to be complete:

1. **Latency versus `M`**, log-x, all four baselines on one axis — the crossover plot, and the most valuable single output of the project.
2. **Achieved bandwidth as a percentage of empirical peak**, versus `M`.
3. **Ablation waterfall** — naive fused → +vectorized loads → +`lop3` dequant → +permuted layout → +split-K → +double buffering, each step with its measured delta. Very few public repositories publish this. It is your differentiator, and it is publishable *regardless of how fast the kernel ends up* — which is precisely why it belongs in the must-pass set.
4. **Roofline**, with the operational-intensity point plotted for each `M`.
5. **Nsight table** — memory throughput %, FMA/ALU pipe %, dominant warp stall reason, per variant.

Produce those five and the report writes itself.

### F13 · Two small overclaims and one process/content imbalance — **Minor**

- **CI (D4).** "CI-friendly if possible" overpromises; there are no free GPU runners on GitHub Actions. Say instead: tests run locally with one command, CI covers compilation plus CPU-side pack/unpack round-trip tests.
- **Balance.** §§11, 12, 15, 16 are roughly a quarter of the document and contain no technical content. Compress the ceremony and spend the space on the pack-format diagram, the dequantization sequence, and the target table. The dual-use paragraph in particular does very little work — every efficiency improvement is dual-use in that sense.
- **Delete §12 from any public version.** It explains that the project is a portfolio artifact. Keep the reasoning, keep it private. Flagship work does not announce that it exists to be impressive; it makes a claim and defends it.

---

## 4. Hardware reality (measured, not assumed)

I measured your laptop directly rather than reasoning from the SKU, and derived the same arithmetic for an H100 SXM.

### F14 · Your profiler cannot read counters — **Blocking, and fixable today**

```
$ ncu --metrics dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed ./probe
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
          NVIDIA GPU Performance Counters on the target device 0.
```

`sudo` requires a password I do not have, so that path is untested. Every profiler-dependent item — D6's analysis, §6.4's utilization metrics, figure 5 — is blocked until this is resolved.

- **On the laptop (WSL2):** the restriction lives in the *Windows* driver, not Linux. NVIDIA Control Panel → Desktop → Enable Developer Settings, then Developer → Manage GPU Performance Counters → allow access to all users, then `wsl --shutdown` and reopen. Try `sudo ncu` first; it is a ten-second test.
- **On the DGX:** check this *before* depending on it. Counter access there is an admin policy decision, and shared clusters frequently lock it down precisely because collection serializes the GPU. If the answer is no, you need to know in week one — the fallback (deriving memory throughput from timing and known byte counts rather than reading `dram__bytes`) is a weaker methodology you would want to design around from the start rather than retrofit.

### Measured: RTX 3050 6GB Laptop

| Quantity | Value | Consequence |
|---|---:|---|
| Achieved bandwidth | **110 GB/s** | 83% of the 131.7 GB/s theoretical — a healthy copy efficiency. **Use 110 as your denominator everywhere**; the theoretical figure would inflate every efficiency claim you make |
| L2 cache | **1.5 MB** | An 8.4 MB INT4 matrix exceeds it by 5.6×, so weights genuinely stream from HBM and the roofline story is clean *here*. Which is exactly why `F4` will ambush you on the H100 — build the flush in on the machine where you cannot see the difference |
| SM count | **20** | At `M`=1, `N`=4096 with a 256-wide tile you launch 16 CTAs, nearly filling 20 SMs. **The wave-quantization crisis is invisible on this GPU.** The same launch on 132 SMs runs at 12% occupancy. Validate split-K on the DGX or not at all |
| Roofline ridge | **~92 FLOP/byte** | At base clock; ~110–120 at boost |
| VRAM | 6 GB | Ample — your largest frozen shape is 117 MB in FP16. Drop the shape-coverage caveat from §5.3 |

Measurement method: `float4` streaming kernel, 64 MiB in / 64 MiB out, 20 warmup + 50 timed iterations, CUDA events. Source in Appendix B.

### Derived: H100 SXM (assuming 3.0 TB/s achieved — **measure this first**)

```
ridge point                       = 495 TFLOP/s ÷ 3.0 TB/s = 165 FLOP/byte
W4 path compute-bound at      M ≈ 165/4  =  41      (3050: 23)
FP16 path compute-bound at    M ≈ 165/1  = 165      (3050: 92)
predicted crossover              M ≈ 40–80          (3050: 20–40)

t_ideal (W4, 8.65 MB)             = 2.9 µs
t_ideal (FP16 weights, 33.6 MB)   = 11.2 µs
typical kernel launch overhead    = 3–8 µs   ← LARGER THAN THE KERNEL
```

### F15 · At `M`=1 on an H100, an ideal kernel finishes faster than it takes to launch — **Blocking**

Read that last pair again, because it reorders your priorities. Any measurement including per-call launch cost is measuring the launcher; a PyTorch-dispatched benchmark is worse still. **CUDA graphs are not a P5 refinement — they are a precondition for the primary result to mean anything.**

This also reframes the project honestly: at realistic decode batch sizes on modern hardware, kernel time and launch/dispatch overhead are the same order of magnitude, which is why production stacks graph-capture entire decode steps. Saying that in the report with your own numbers behind it marks the difference between someone who benchmarked a kernel and someone who understands where the time goes.

Two further consequences. The crossover `M` is **hardware-dependent and roughly 2× higher on H100 than on your laptop** — so it is not a constant to be discovered but a function to be characterized, which retroactively makes the multi-architecture sweep necessary rather than decorative. And the 50 MB L2 means every shape in your frozen set is fully resident, so `F4` is a first-order threat on your actual target rather than a theoretical one.

### A prediction to commit to before writing any code

Put both predictions — `M`≈20–40 on sm_86, `M`≈40–80 on sm_90 — into the design note, in a **timestamped commit**, before you measure. Then report whether they held.

A study that predicts its own headline result from a roofline and then checks the prediction is doing something categorically different from one that measures and describes, and it costs an afternoon. If a crossover lands well below prediction you have found real dequantization overhead on the critical path — precisely the W4A8 argument — and you will have found it independently, on two architectures, with profiler evidence for why.

### DGX benchmarking hygiene

A shared DGX is a hostile measurement environment, and none of this is in §10:

- **Exclusive allocation.** A co-tenant job on another GPU still contends for host memory bandwidth, PCIe, and power headroom. For microsecond-scale latency work you need the node, or a documented record of what else was running. Log `nvidia-smi` process lists per run.
- **Clocks.** Ask the admin for `-lgc` once. Until then, log SM clock, memory clock, power draw, and temperature with every sample.
- **MIG and MPS off**, persistence mode on, one GPU pinned via `CUDA_VISIBLE_DEVICES` for single-GPU microbenchmarks, host threads bound with `numactl` to the socket attached to that GPU.
- **Report the node, not just the GPU.** "H100 SXM5 80GB, DGX H100, driver X, CUDA Y, exclusive, clocks logged" is what makes §8.1.5 achievable.

---

## 5. State of the art, verified 2026-08-11

§13 needs replacing. Here is the current map, with three corrections to advice I gave earlier in conversation.

### What has already shipped

Quantized MoE is **not** an empty field. vLLM's fused MoE kernel already supports `mxfp4`, `nvfp4`, `int4`, `int8`, and `fp8`, dispatching across Triton and DeepGEMM expert implementations based on type, shape, and quantization parameters. SGLang's MoE system ships Triton, CUTLASS, DeepGEMM, FlashInfer, and AITER backends with FP8, FP4, INT8, and MXFP4 support. DeepSeek's DeepGEMM provides FP8 grouped GEMM kernels with measured throughput gains on H200. Assume coverage exists and look for gaps *within* it.

### The gap that matters, and it is on your hardware

This is the strongest finding of the research, and it changes my primary recommendation:

> **vLLM can load NVFP4-quantized weights on H100 and A100 via a Marlin FP4 software fallback, which reduces memory footprint but does not provide any throughput improvement over FP8.**

On Hopper there is no native MXFP4/NVFP4 multiply. Weights live in HBM in 4-bit and are dequantized to FP8 or BF16 inside the kernel, so you keep the memory saving and lose the speed. Now do the byte arithmetic for the memory-bound decode regime:

| Format | Bytes per weight | Ratio vs FP8 |
|---|---:|---:|
| FP8 (E4M3) | 1.000 | 1.00× |
| MXFP4 (E2M1 + E8M0 per 32) | 0.531 | **1.88×** |
| NVFP4 (E2M1 + E4M3 per 16) | 0.563 | **1.78×** |

At `M`=1, where time is weight traffic, **MXFP4 should be ~1.9× faster than FP8 and currently delivers ~1.0×.** That is a documented production deficit, with a computable prize, on the exact hardware you have, that nobody has closed.

Contrast with what *is* claimed. The MR-GPTQ work ("Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization") targets **B200 and RTX 5090 — native FP4 hardware** — reaching 2.2–3.6× and 4–6× end-to-end over FP16. It is a quantization *algorithm* contribution and it does **not** address kernel-level dequantization on Hopper or Ampere. Separately, FP4 *training* on Hopper without native support has been addressed. **The Hopper inference-kernel gap is open.**

Two nuances from that paper that make your project *better*, not worse:

- **MXFP4's power-of-two scaling causes significant accuracy degradation**, and NVFP4's small group size undermines outlier mitigation. So the exponent-add trick I described is a genuine *speed* advantage bought with an *accuracy* cost. That is a real trade-off to characterize, which makes a three-way MXFP4 / NVFP4 / INT4-g128 comparison substantially more valuable than a single-format benchmark.
- **"FP4 is not an automatic upgrade over INT4."** This directly rescues your original INT4 work as a live baseline rather than legacy — cite it.

### A concrete, unclaimed, named bug

vLLM issue **#38022**: `moe_wna16_marlin_gemm` fails on MXFP4-quantized GPT-OSS-20B at `K=N=2880`. Root cause is that Marlin requires K and N aligned to 128; `2880 / 128 = 22.5`. The quantization itself is valid — 2880 is divisible by 32, the MXFP4 group size — only the kernel's thread tiling is incompatible. The issue is **closed as not planned / stale**, with no maintainer engagement, and the report notes it likely affects other MoE models with non-128-aligned dimensions. A related open issue (**#39000**) reports Gemma-4 MoE MXFP4 crashing during weight loading in the fused MoE layer.

A stale "closed as not planned" bug is arguably the best possible starting point: demonstrable user pain, zero competition, small enough to be a first PR, and it earns maintainer credibility *before* you propose anything larger.

### Correction 1 to my earlier advice

I told you not to burn effort on awkward `N`/`K` because "you own the packing, so pad offline" and Llama dimensions are friendly multiples. **That was wrong in the case that matters.** GPT-OSS's 2880 is a real shipping model dimension that breaks 128-alignment and is a live crash in production. Awkward `N`/`K` alignment is not a synthetic robustness exercise — it is an unclaimed bug class. Keep the awkward-shape work, and put 2880 in the frozen set.

### Correction 2 to my earlier advice

I proposed routing skew as the flagship thesis. **It is more crowded than I implied.** The load-imbalance space is dense at the *system and placement* level: EPLB and redundant expert placement, MoETuner, GRACE-MoE, ReLibra, FEPLB, and LP-based fine-grained balancing. Additionally, MoGE already dispatches dynamically between GEMV and GEMM execution modes by workload intensity — partial prior art on the dispatch-threshold idea from `F6`.

The narrower question — *how does a quantized grouped GEMM behave under skewed per-expert `M`, and is the grouped-launch formulation itself wrong* — appears still open, and the skew itself is well documented as severe. But it is adjacent to a great deal of work, so **survey it properly before committing**, and treat it as an extension rather than the spine.

### Correction 3 to my earlier advice

I framed the FP4 pivot around "my laptop has no FP4." The correct framing is far stronger and is now evidence-backed: **Hopper has no native FP4, Hopper is the dominant deployed datacenter architecture, MXFP4 checkpoints are shipping into it today, and the existing software fallback delivers zero throughput gain.** That is a production gap with named users, not a hardware apology.

### Replacement for §13

| Area | Current references |
|---|---|
| Weight-only quantization | GPTQ, AWQ; MR-GPTQ / block-wise Hadamard for FP4 |
| High-performance W4 kernels | Marlin; Machete (CUTLASS 3.x + TMA, Hopper); BitBLAS; exllamav2; llama.cpp Q4_K; FLUTE (LUT dequant) |
| Block-scaled formats | OCP MX (MXFP4: E2M1 + E8M0/32); NVFP4 (E2M1 + E4M3/16 + tensor scale) |
| The counter-argument | W4A8 / QServe — dequant on the critical path caps throughput |
| MoE kernels | vLLM fused MoE (`moe_wna16_marlin_gemm`, Triton, DeepGEMM); SGLang sgl-kernel (CUTLASS/DeepGEMM/FlashInfer/AITER); DeepGEMM FP8 grouped GEMM; MoGE |
| MoE load balancing (systems) | EPLB, MoETuner, GRACE-MoE, ReLibra, FEPLB, LP-based balancing |
| Dense GEMM libraries | cuBLAS/cuBLASLt, CUTLASS 3.x mixed-input |
| Kernel DSLs | Triton, CuTe DSL, TileLang, Helion, ThunderKittens, Pallas/Mosaic |

---

## 6. The conversion

### F16 · Revised primary thesis

The brief's target is settled; §5 located one that is not. The revision, in priority order:

> **Primary thesis.** *Block-scaled 4-bit formats deliver their memory saving but none of their throughput on Hopper-class hardware, because the software dequantization path is not designed for the memory-bound decode regime. This work quantifies the gap, closes it, and characterizes the resulting speed/accuracy frontier across MXFP4, NVFP4, and INT4-g128.*

Why this and not routing skew:

- **A documented deficit with a computable prize.** ~1.9× is available from bytes; ~0% is currently realized. You know the size of the win before you start, which is rare and valuable.
- **Named users and a named bug.** GPT-OSS and Gemma-4 MoE MXFP4 paths, with open crash reports.
- **It survives Blackwell.** The finding is about a *format-versus-hardware mismatch* and a *measurement method*, both of which outlive the ISA. Hopper capacity will be deployed for years.
- **It rescues your original work.** "FP4 is not an automatic upgrade over INT4" makes your INT4-g128 path a live comparison arm, not legacy.
- **Your DGX is the right instrument** — this is a Hopper problem.

Two consequences for the plan the brief does not currently account for:

**(a) sm_90 is a different kernel, not a recompile.** Hopper means `wgmma` (asynchronous warpgroup MMA), TMA for bulk async copies, thread-block clusters, and B operands read from shared memory rather than registers. A Marlin-style sm_80 design does not port — which is precisely why Machete exists as a separate thing rather than a port of Marlin. The `F2` layout derivation must be redone against `wgmma` fragment requirements and TMA swizzle patterns. **This is where the deep, transferable, currently-scarce skill lives**, and it is the phase that earns the technical-depth score.

**(b) The DGX removes your scope protection.** §4.3 was your best asset when hardware was scarce. With 8×80 GB and no queue for shapes, it is more important, not less.

### F17 · The deliverable ladder

This is the structural answer to the central tension between contribution and completion (see §11). Four rungs, each independently publishable, each making the next credible. You begin collecting results in month two, and the ambitious rung becomes upside rather than a single point of failure.

| Rung | Months | Content | Risk |
|---|---:|---|---|
| **R0 · Instrument** | 1–3 | The harness and protocol from §2, then point it at existing public kernels — Marlin, Machete, the vLLM paths, Triton GPTQ — under correct protocol on identical hardware. Report how much published speedups move once L2 is flushed, launches are graph-captured, clocks pinned, and peak measured | **Certain** |
| **R1 · Quantify** | 4–6 | The MXFP4/NVFP4-on-Hopper gap. Where do the missing 1.9× go? Full decomposition: issue slots, memory stalls, occupancy, launch, reduction, dequant ALU. Three-way format comparison including the accuracy axis | **High** |
| **R2 · Close** | 7–9 | The `wgmma`/TMA kernel that recovers it. Fix the 128-alignment bug class along the way — it is on the path, not a detour | **Moderate** |
| **R3 · Extend** | 10–12 | Grouped/MoE variant; routing-skew characterization; the cross-DSL cost study (CUDA / Triton / CuTe DSL) | **The swing** |

**R0 produces a result no matter what happens.** It cannot fail to produce a finding, it is a *tool* others will use rather than merely read about, and it is the reproducibility contribution the field visibly lacks. It is simultaneously your insurance policy, your most likely path to adoption, and what makes every later number trustworthy.

**If R1 shows the gap is small or structurally unavoidable, publish that and stop.** A well-evidenced negative result on a question people assumed the answer to is a genuine contribution, and abandoning R2 on evidence is a strength.

If you must cut, cut R3. Do not cut R1 to preserve R3.

---

## 7. Answers to your §14 reviewer checklist

**Q1 — Is scope realistic for the stated timeline and hardware?**
There is no stated timeline (`F8`). Hardware is no longer a constraint. Scope is realistic; what is unrealistic is P3 as an untimeboxed phase. Optimization expands to fill all available time — fix the date and ship whatever the ablation waterfall says on that date.

**Q2 — Are success criteria fair and non-inflated?**
They are *deflated*, which is its own failure (`F1`, `F5`). You set a bar you cannot fail and then committed in §15 to not overclaiming. Replace with §9.

**Q3 — Should v1 stay W4A16, or prioritize another format?**
**Change it — to block-scaled FP4 on Hopper, keeping INT4-g128 as a comparison arm** (`F16`). This reverses my initial advice on the strength of §5's evidence. W4A16's virtue was the cleanest roofline story and the strongest calibration baseline; you keep both by retaining it as an arm, and gain a live gap with a computable prize.

**Q4 — Is excluding full engines the right call?**
**Yes, unambiguously.** The best decision in the brief; do not relitigate it. One exception worth taking: swap your kernel into a single real module and measure one model's decode latency end-to-end. Not an engine — an *external-validity check*, which is exactly what a microbenchmark-only study is most vulnerable on (`F4`). With 640 GB this is now days of work rather than a stretch goal.

**Q5 — What minimum shape set and baselines should be frozen?**
See §10. Note the addition of GPT-OSS's 2880 on the strength of Correction 1.

**Q6 — Safety / IP / academic integrity concerns with studying open kernels?**
**No real concern.** Marlin is Apache-2.0; reading it and being influenced is unambiguously fine. The genuine risk is reimplementation-by-memory of permutation tables — code you cannot explain, arriving without attribution. Handle it cleanly: derive your layout from first principles and *write the derivation down* (`F2` — also the strongest thing in your portfolio), cite Marlin and Machete as prior art, and state plainly for each design element whether it was independently derived or adapted. Anything vendored goes in `third_party/` with the license intact.

---

## 8. Replacement for §8.1 — commit to these now

Choose them before you know whether you can hit them. A target selected after the measurement is not a target.

| Criterion | Bar | Rationale |
|---|---:|---|
| Correctness, structured probes | **exact** | One-hot / ramp / prime-scale are *index* tests; no tolerance to negotiate (`F10`) |
| Correctness, random | **≤ sqrt(K)·eps** | Versus FP32 reference on dequantized weights; error *map* checked for structure |
| Fused vs dequant+cuBLAS agreement | **≤ 2 ulp** | Identical inputs — your sharpest test |
| % of ideal bandwidth, `M`=1 | **≥ 60%** | **The headline.** Self-normalizing, hardware-independent, cannot be inflated |
| MXFP4 vs FP8 speedup, `M`=1, H100 | **≥ 1.5×** | Bytes give 1.88×; current fallback gives ~1.0×. This is the project's central claim |
| Speedup vs FP16 cuBLAS, `M`=1 | **≥ 2.5×** | Bytes give 3.9×; 2.5–3.5× is the realistic band |
| Gap to GPTQ-Marlin / Machete, `M`=1 | **≤ 1.5×** | Calibration against known-good on your own hardware |
| Gap to Marlin / Machete, `M`=16 | **≤ 2.5×** | Marlin's real strength is holding ~4× out to batch 16–32; expect to lose ground |
| Crossover `M` identified, both archs | **required** | Where 4-bit loses, with a stated mechanism. A *result*, not a failure |
| Non-128-aligned shapes | **correct + benchmarked** | `K=N=2880` must work. This is the unclaimed bug class |
| Ablation waterfall | **≥ 5 steps** | Each with measured delta. Publishable at any absolute performance |
| Triton twin vs CUDA | **measured** | No bar. Correct, benchmarked, gap explained |
| Predictions preregistered | **required** | Timestamped commit before first measurement |

---

## 9. Frozen shape set

Freeze this before writing code. Llama-3-8B geometry for familiarity, GPT-OSS for the alignment case.

| Layer | N | K | Why it earns a slot |
|---|---:|---:|---|
| q_proj / o_proj | 4096 | 4096 | The square canonical case |
| k_proj / v_proj (GQA) | 1024 | 4096 | Small `N` — exposes tail tiles, starves occupancy |
| gate_proj / up_proj | 14336 | 4096 | Wide `N` — the friendly case |
| down_proj | 4096 | 14336 | Deep `K` — where split-K matters most |
| **GPT-OSS-class** | **2880** | **2880** | **Not 128-aligned. The live bug class (`F16`, Correction 1)** |
| fused QKV (optional) | 6144 | 4096 | Only if you do the end-to-end check |

- `M` ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048} — the full sweep, because the crossover *is* the result.
- Group size 128 primary, 64 as stress case; MXFP4 block 32, NVFP4 block 16.
- Symmetric scales only in v1. `act_order` out of scope, explicitly.
- **Add `TP` ∈ {1, 2, 4, 8}.** With TP=8 on the DGX, per-GPU `N` for a 4096-wide projection drops to 512 — a completely different tile-count and tail-effect regime, and what production actually runs. It costs a loop and turns "here is the crossover" into "here is how the crossover moves with the parallelism strategy," which is the version a serving team can act on.

---

## 10. The five moves that determine whether this is a flagship

None are technical, and they matter more than the kernel. The operative definition: **a project is a flagship when someone with no obligation to care uses it, cites it, or merges it.** Effort is invisible from outside; adoption is not.

1. **Take the question from maintainers, not from a document.** Highest-leverage week in the project. vLLM #38022 and #39000 are real, named, and unclaimed. Ask in the vLLM/SGLang/torchao channels whether an MXFP4 Hopper throughput path is wanted and unowned, and start by fixing the stale alignment bug. You either get a named human saying "yes, nobody is doing that" — converting adoption from a hope into a commitment and handing you a free expert reviewer — or you learn it is being done and save two months. **Everything in this review is my inference; one maintainer conversation outranks all of it,** and the fact that you went and asked is itself the strongest signal available.

2. **Get one credible co-signer.** A systems faculty member, or a maintainer as informal reviewer. Unaffiliated single-author work carries a discount that is unfair but real, and one co-signer removes it. Ask early, while you still have decisions they can influence — people say yes to shaping work far more readily than to endorsing finished work.

3. **Preregister your predictions in a timestamped commit.** The roofline predictions, both crossovers, and what result would falsify the thesis. Commit, then never edit. Essentially unheard of in kernel engineering, and it converts being wrong from an embarrassment into a documented finding — which makes §15's honesty structural rather than aspirational.

4. **Land the code upstream, knowing what it costs.** The only dimension a third party validates for you, and the only one you cannot manufacture. Understand going in that it means supporting cases you did not target and answering strangers' issues for a long time. That obligation is the price.

5. **Present it live, and put the negative results in the talk.** GPU-MODE runs an open speaker pipeline aimed at exactly the audience you want. Lead with what you predicted and got wrong — the most memorable twenty seconds you have, and the part nobody else includes.

---

## 11. Scorecard and structural limits

### Where the brief stands

Left number is the brief as specified; right is the ceiling if executed excellently **without changing scope** — so a low ceiling is a scope problem no amount of effort fixes.

| Dynamic | As specified | Ceiling |
|---|---:|---:|
| Scope discipline | 9 | 9 |
| Written communication | 8 | 8 |
| Intellectual honesty | 9 | 9 |
| Methodological rigor, as specified | 5 | 9 |
| Depth of technical design | 4 | 9 |
| Skill per unit effort | 8 | 9 |
| **Contribution to the field** | **2** | **6** |
| Shelf life of the artifact | 3 | 6 |
| **Differentiation among candidates** | **3** | **8** |
| **Alignment with current bottlenecks** | **3** | **8** |
| **Likelihood anyone external adopts it** | **1** | **7** |
| Feasibility on hardware | 9 | 10 |
| Probability of finishing | 5 | 8 |
| Capital efficiency | 9 | 9 |

**As a learning project: ~8.1/10. As a flagship: ~4.5/10.** The entire deficit sits in the four bolded rows, and all four are *topic* choices — not effort, and not hardware. DGX access moves the flagship figure by about 0.3, because hardware was never the binding constraint.

### Why a flat 10 across all fourteen is not the goal

Score the actual flagships in this field on the same scale. FlashAttention: 10 on contribution, roughly 3 on scope discipline — it sprawls and has no non-goals list. Marlin: perhaps 5 on conceptual novelty (mixed-precision GEMM was not new), 10 on engineering and adoption. PagedAttention: near-zero novelty in the idea (it is operating-system virtual memory), enormous novelty in the application.

None is uniform. What each shares is **one axis pushed to the wall with everything else deliberately sacrificed to fund it.** A project scoring 7 across fourteen dimensions has no axis anyone remembers it for. **Spikiness is the point.**

### The tensions that are real

| Pair | Why it binds | Resolution |
|---|---|---|
| **Contribution ↔ finishing** | Novel work has unknown difficulty by definition. A 10 on contribution means attempting something that might not come out | The ladder (`F17`). Stack deliverables by risk so the guaranteed rung is already publishable. A genuine escape, not a compromise |
| **Shelf life ↔ current alignment** | Anything perfectly aligned with today's bottleneck has short shelf life *by construction* — the bottleneck is current because it is about to be eliminated | Only binds if the *kernel* is the artifact. A protocol, a characterization, and a method outlive the ISA. Make the durable thing primary and the kernel its evidence |
| **Adoption ↔ scope discipline** | Upstream means supporting exactly what §4.3 wisely excludes: alignment edge cases, every group size, CI, and a permanent maintenance obligation | Accept an 8–9 ceiling and buy it with move #1. Do not chase 10 by widening support — that is how projects die at 80% |
| **Depth ↔ learning efficiency** | Learning efficiently means doing the documented thing, where every dead end has a blog post | None needed. Accept 8 and stop looking at it |

### The tensions that are illusions — take these

- **Rigor is free.** Two weeks of protocol work, reused across every experiment forever, and it is what makes every other number believable. Nothing trades against it, and almost nobody picks it up.
- **Honesty versus impressiveness is backwards.** "We predicted the crossover at `M`≈41, measured 28, and here is the profiler evidence for why" reads as *more* competent — that author is running an investigation; the alternative might be running a demo.
- **Differentiation does not require scale.** It requires attempting what nobody bothered to measure carefully, which is usually small, cheap, and in plain sight.

### Projected ceiling under §6's plan plus §10's moves

Nine genuine 10s (rigor, honesty, differentiation, alignment, writing, feasibility, capital efficiency, depth, shelf life), two 9s (contribution, finishing), and three deliberate caps — adoption at 8 because it needs a maintainer's yes, scope discipline and learning efficiency at 8 because those are what you are *spending*.

**The asymmetry worth remembering: adoption is the only score on this table you cannot award yourself.** Everything else is your execution. That is why move #1 belongs in week one rather than month six.

---

## 12. The seven edits to the document, in order

1. **Retitle and reframe** around the block-scaled-FP4-on-Hopper thesis (`F16`). Everything else follows, and it is the edit that makes the project defensible even if the kernel disappoints.
2. **Replace §8.1** with §8's table. Demote naive-dequant to a sanity floor; promote Marlin/Machete from optional to mandatory.
3. **Add the layout derivation** as D1's primary content — `wgmma` shape → extraction sequence → nibble order → per-thread load pattern, with the 32-bit word diagram (`F2`).
4. **Add split-K** to §6.4 and to the ablation, with the CTA-count arithmetic for your frozen shapes written out (`F3`).
5. **Rewrite §6.4's measurement protocol** — L2 flush, CUDA graphs, locked clocks, empirical peak, DGX hygiene (`F4`, `F14`, `F15`).
6. **Split the kernel in two** (`qgemv` / `qgemm`) and make the dispatch threshold a measured result (`F6`).
7. **Move Triton to P2**; add §5's related-work map and the two missing paragraphs; put a concrete duration on every phase; compress §§11/12/15/16 and delete §12 from the public version (`F7`–`F13`).

---

## 13. This week — none of it coding

1. **Settle profiler access on both machines** (`F14`). D6 depends on it, and the DGX answer is somebody else's decision to make.
2. **Run the streaming probe on an H100** (Appendix B) for real achieved bandwidth. That single number fixes the denominator for every claim in the project.
3. **Reproduce vLLM #38022** on your DGX with GPT-OSS-20B MXFP4 — confirm the crash, confirm nobody has fixed it, and you have your first PR plus your entry into the conversation.
4. **Open the upstream question**: is an MXFP4 Hopper throughput path wanted and unowned? Ask before writing anything, while the answer can still change your plan.

Items 3 and 4 are the only ones in this entire review that can tell you the project is worth doing. Everything else assumes it is.

---

## Appendix A · What is genuinely good about the brief

Worth stating explicitly, because the body of this review is corrective by construction and the balance would otherwise mislead.

- **§4.3's non-goals table** is better than most professional project documents manage. Keep it, tighten it, and add `act_order` (`F11`).
- **§15's commitments** — no fabricated numbers, ranges over lucky points, document failed experiments — are exactly right and are the foundation the rigor 10 is built on. Make them structural via preregistration (move #3) rather than aspirational.
- **§8.3's non-criteria** show you already understand the difference between a result and a vanity metric.
- **The writing** is clear enough to argue with, which is rarer than it sounds and is direct evidence you can produce the report that this project's value ultimately rests on.
- **Reviewer question 3** — asking whether the v1 datatype is right — was the correct question to ask, and it is the one whose answer changed the project. Asking it is why this review is useful rather than decorative.

---

## Appendix B · Measurement probe

Bandwidth probe used for the numbers in §4. Compile with `nvcc -O3 -arch=sm_86` (use `-arch=sm_90` on the DGX).

```cuda
#include <cstdio>
__global__ void stream_k(const float4* __restrict__ in, float4* __restrict__ out, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) { float4 v = in[i]; v.x += 1.f; out[i] = v; }
}
int main(){
  int n = 1<<22;                  // 64 MiB in, 64 MiB out
  float4 *a,*b; cudaMalloc(&a,n*16); cudaMalloc(&b,n*16); cudaMemset(a,0,n*16);
  for(int r=0;r<20;r++) stream_k<<<(n+255)/256,256>>>(a,b,n);
  cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s); for(int r=0;r<50;r++) stream_k<<<(n+255)/256,256>>>(a,b,n); cudaEventRecord(e);
  cudaEventSynchronize(e); float ms; cudaEventElapsedTime(&ms,s,e);
  double gbps = (2.0*n*16.0*50)/(ms/1e3)/1e9;
  printf("achieved HBM bandwidth: %.1f GB/s  (%.3f ms/iter)\n", gbps, ms/50);
  return 0;
}
```

Device properties probe (SM count, L2 size, bus width, derived ridge point) is a short companion program using `cudaGetDeviceProperties`; on the 3050 it reported `SMs=20  L2=1.50 MB  bus=96-bit  theoPeakBW=131.7 GB/s  ridge≈92 FLOP/byte`.

**Note:** this probe is deliberately naive — it does not flush L2, which is fine for a 128 MiB working set but is exactly the mistake `F4` warns about at weight-matrix sizes. Do not reuse it as your GEMM harness.

---

## Appendix C · Sources

State-of-the-art claims in §5 were verified on 2026-08-11 against:

- [Fused MoE Kernel Features — vLLM documentation](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
- [vLLM issue #38022 — Marlin MoE kernel fails with MXFP4-quantized GPT-OSS 20B, non-aligned K=N=2880](https://github.com/vllm-project/vllm/issues/38022)
- [vLLM issue #39000 — Gemma 4 MoE runtime MXFP4 quantization crash in fused MoE layer](https://github.com/vllm-project/vllm/issues/39000)
- [Quantization and MoE Optimizations — vllm-project/vllm](https://deepwiki.com/vllm-project/vllm/7-quantization-and-moe-optimizations)
- [Mixture of Experts System — sgl-project/sglang](https://deepwiki.com/sgl-project/sglang/9-high-performance-kernel-library-(sgl-kernel))
- [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization (MR-GPTQ)](https://arxiv.org/abs/2509.23202)
- [Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs](https://arxiv.org/abs/2603.02731)
- [Pretraining LLMs with MXFP4 on native FP4 hardware](https://arxiv.org/pdf/2605.09825)
- [MXFP4: microscaling 4-bit float with shared 8-bit scale](https://zeroentropy.dev/concepts/mxfp4/)
- [TFLOPS Gap: Why FP4 MoE Kernel Engineering Matters on Blackwell](https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison)
- [FP4 Quantization on Blackwell GPUs: Throughput, Cost, and When It's Worth It](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [Deploy DeepEP and DeepGEMM: MoE Inference Kernels Guide](https://www.spheron.network/blog/deploy-deepep-deepgemm-moe-inference-kernels-gpu-cloud/)
- [Pangu Pro MoE: Mixture of Grouped Experts (MoGE)](https://arxiv.org/pdf/2505.21411)
- [MoETuner: Optimized MoE Serving with Balanced Expert Placement and Token Routing](https://arxiv.org/pdf/2502.06643)
- [GRACE-MoE: Grouping and Replication with Locality-Aware Routing](https://arxiv.org/pdf/2509.25041)
- [Fine-grained MoE Load Balancing with Linear Programming](https://arxiv.org/pdf/2511.16947)
- [GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs](https://arxiv.org/pdf/2605.23078)

**Caveat on this appendix.** These were gathered in a single research pass, not a systematic survey. Before committing to §6's thesis, do the literature survey properly — particularly around Hopper FP4 inference kernels and quantized MoE grouped GEMM, where the field is moving fast enough that a three-month-old conclusion can be wrong. Treat §5 as a starting point that tells you *where to look*, not as a settled map.

---

*End of review.*

*Companion artifacts, same content in web form: [Part I — findings](https://claude.ai/code/artifact/90a4f23a-a637-497e-8366-01dd542561c1) · [Part II — scorecard and conversion](https://claude.ai/code/artifact/07c7556b-5c96-48ff-9d34-a90d62ee19e0) · [Part III — the maximum achievable project](https://claude.ai/code/artifact/94d581a0-c7b9-4497-84b3-59db11837668). Note that Parts II and III predate the research in §5 and contain the advice corrected there.*
