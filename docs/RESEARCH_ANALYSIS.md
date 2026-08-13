# Hard research analysis (living)

**Date:** 2026-08-13  
**Purpose:** Evidence-backed map of the field, so we do not optimize the wrong kernel or claim the wrong gap.  
**Method:** Public papers, vendor blogs, vLLM issues, CUTLASS/Machete writeups. **Not** insider hiring intel.  
**Implication for us:** R0 harness + encodings stay; **R2 layout must follow Hopper `wgmma` constraints**, not Ampere-Marlin copy-paste.

---

## 1. Thesis check — is the gap still real?

| Claim | Evidence | Status |
|-------|----------|--------|
| Hopper has **no native FP4 tensor cores** | SemiAnalysis InferenceX / GPU spec tables: H100 dense FP4 = none; FP8 is the Hopper math path | **Holds** |
| MXFP4/NVFP4 **checkpoints ship** into stacks that also run on Hopper | GPT-OSS MXFP4; Red Hat NVFP4 collections; vLLM MXFP4 config | **Holds** |
| Without native FP4, stacks **fall back** (Marlin / software dequant) | vLLM #30135: *“Your GPU does not have native support for FP4. FP4 compression will be used leveraging the Marlin…”* | **Holds** |
| Fallback can **keep bytes, lose speed** vs FP8 | Same class of reports; FP8 is the documented Hopper default (~up to ~1.6× vs FP16 in vLLM FP8 docs when TC is used). Software FP4 is not that path | **Holds as the question we measure** — we do **not** treat “1.0× vs FP8” as re-proven until we rebench under our protocol on sm_90 |
| Awkward **2880** dims break Marlin-class tiling | vLLM #38022: MXFP4 GPT-OSS-20B MoE, `K=N=2880`, invalid thread config (128-align) | **Holds** — we already reject INT4-g128 at K=2880 and accept MX (block 32) |
| Native FP4 **wins are Blackwell / MI355 stories** | HF/SGLang Blackwell NVFP4 TFLOP tables; AMD MI355 MXFP4 vs MI300 FP8 | **Do not cite as Hopper results** |

**Conclusion:** The project thesis is **not obsolete**. The noisy part of the internet is *native* FP4 on new silicon. Our problem is **software FP4 on the installed Hopper base**.

---

## 2. Format physics (OCP + NVIDIA)

Sources: OCP MX spec / Rouhani et al. 2023; NVIDIA NVFP4 blog (Jun 2025); FProx MX notes.

| | **MXFP4** | **NVFP4** | **INT4-g128** |
|--|-----------|-----------|---------------|
| Element | E2M1 (codes → {0, ±0.5…±6}) | Same E2M1 | Signed INT4 |
| Scale | **E8M0 / 32** (power-of-two, bias 127, 0xFF=NaN) | **E4M3 / 16** + **FP32 tensor** | FP16 / 128 |
| Hardware scale (Blackwell) | Yes (MX) | Yes (NV) | Mixed-input / other |
| Hopper native MMA | **No** | **No** | No (dequant to FP16 then MMA/wgmma) |
| Dequant ALU | **Exponent add** (cheap) | FP8-scale **multiply** | FP16 scale mul |
| Accuracy narrative | Coarser; NVIDIA: higher drop risk vs FP8 | Finer block + fractional scale; lower drop risk | Mature GPTQ/AWQ literature |
| Bytes/weight (our `formats.cuh`) | 0.531 | 0.563 (+tiny tensor) | 0.516 |

**Our encodings in `pack.hpp` match this public spec** (E2M1 table, E8M0 bias 127, NV two-level). That is load-bearing: wrong E2M1 table = silent wrong science.

**Implication:** Speed/accuracy frontier (figure 5) is **not** a marketing extra. NVIDIA’s own table says MX is cheaper/coarser and NV is costlier/finer. We should **measure both**, not pick a winner in the README.

---

## 3. Kernel lineage (this is the R2 design law)

### 3.1 Marlin (Ampere)

- Hand-derived **pre-shuffle** because **`ldmatrix` has no 4-bit**.  
- Register-side `mma.sync`.  
- Excellent decode on Ampere; **on Hopper, `mma`-only leaves ~37% of peak FLOPs on the table** (Machete / TK / related citations).  
- Thread tiling historically **128-aligned** → GPT-OSS 2880 crash class.

### 3.2 Machete (Hopper successor)

Source: Wilkinson, Red Hat, Oct 2024, “Introducing Machete”.

| Ingredient | Why it exists |
|------------|----------------|
| **`wgmma`** | Peak Hopper FLOPs; warpgroup async MMA |
| **B from SMEM, A & C from registers** | Hardware rule |
| **Compute `Yᵀ = Wᵀ Xᵀ`** | Upconverted 4-bit lives in **registers** → must be operand **A**, not B |
| **CuTe layout algebra** | Marlin’s hand shuffle does not port to `wgmma` layouts |
| **TMA** | Async GMEM→SMEM, less register pressure |
| **Warp specialization** | Producer/consumer overlap |
| **Pre-shuffle + 128-bit SMEM loads** | Same 4-bit `ldmatrix` hole as Marlin |
| **Interleaved numeric convert** | See `vllm_numeric_conversion.cuh` (prior art for `lop3`/nibble upconvert) |

**Machete’s published strength is especially batch/prefill 128+** (compute-bound). Decode is still a **bandwidth + occupancy** problem (our split-K table).

### 3.3 What this means for *our* kernels

```
R0/R1 fused-ref (qgemv_ref)     = sequential pack, scalar FMA, correctness
R2 sm_86 (optional)             = mma + Ampere-style pre-shuffle (Marlin-class)
R2 sm_90 (authoritative)        = MUST:
                                  1) pre-shuffle for wgmma A-fragment
                                  2) transpose problem so weights are A
                                  3) TMA for tiles
                                  4) split-K or persistent for decode CTA count
                                  5) MX/NV scale groups in the inner loop
```

**Copying Marlin’s permutation tables into a Hopper kernel is a known failure mode.** Cite Marlin/Machete; **derive** with CuTe or a written fragment diagram.

CUTLASS starting point named by Machete authors: **example 55 — Hopper mixed-dtype GEMM**.

---

## 4. Serving-stack reality (2025–2026)

- **Hopper production default for “fast quant” is still largely FP8 W8A8** when hardware supports it (vLLM FP8 docs).  
- **W4A16** (GPTQ/AWQ + Marlin/Machete) remains the Ampere / memory-capacity story.  
- **MXFP4** is how **GPT-OSS-class** weights are published; on non-Blackwell NVIDIA, vLLM **Marlin fallback** is a documented path (`VLLM_MXFP4_USE_MARLIN` / backend env names **change across versions** — do not hardcode one env var as eternal).  
- **Blackwell NVFP4** kernels (FlashInfer, CUTLASS FP4, SGLang) are a **different project**. Using those numbers as our success bar would be scientific fraud.  
- **MoE** grouped GEMM + routing is crowded; our v1 remains **dense GEMM + 2880 alignment**, MoE as R3.

---

## 5. Four-cause gap model (still the R1 experiment)

Unchanged, now better sourced:

| Cause | Who already named it | How we isolate |
|-------|----------------------|----------------|
| Wave quant / CTA count | Our occupancy table; classic GEMM | Occupancy.hpp vs H100 132 SM |
| Launch > kernel | CUDA graphs literature; our sm_86 launch ~14 µs WSL | `time_graphed` vs looped |
| Layout / shuffle / wrong operand | Machete blog (no 4-bit ldmatrix; wgmma B-from-smem) | Ablation: sequential vs pre-shuffle; mma vs wgmma |
| Dequant issue rate | W4A8 / QServe-style “critical path” | `lop3` vs naive; MX exponent-add vs NV mul |

**P4 (preregistration):** we still predict (1)+(2) dominate decode on Hopper, not (4). If wrong, that **supports** W4A8 literature for decode — publish it.

---

## 6. What we will *not* claim

- End-to-end tok/s vs TensorRT-LLM / SGLang Blackwell FP4.  
- “MXFP4 is 1.9× FP8 on H100” until **our** flushed, graphed, empirical-BW protocol says so.  
- That fused-ref `qgemv_ref` (~0.2% of ideal on a 256²) is a performance result.  
- That sequential `pack.hpp` layout is production-ready.

---

## 7. Concrete upgrades this analysis forces in the repo

| Upgrade | Where |
|---------|--------|
| Write the **wgmma A-operand / transpose** constraint into D1 | `D1_LAYOUT_DERIVATION.md` |
| Related-work map current to 2026 | this file + plan §13 replacement |
| Keep **2880** as a first-class shape | already frozen; keep tests |
| Treat **MX vs NV** as a measured frontier | encodings already; figure 5 remains must-pass |
| Cite **Machete + CUTLASS 55 + numeric_conversion.cuh** as prior art | D1 + this file |
| Never use Blackwell native-FP4 papers as Hopper kernel baselines | PROTOCOL / REPORT later |

---

## 8. Source list (selected)

1. Wilkinson, *Introducing Machete*, Red Hat Developer, 2024-10-14.  
2. NVIDIA, *Introducing NVFP4…*, 2025-06-24.  
3. Rouhani et al., *Microscaling Data Formats for Deep Learning*, 2023 (OCP MX).  
4. Frantar et al., Marlin (IST-DASLab / arXiv 2408.11743).  
5. Colfax, *CUTLASS Tutorial: WGMMA on Hopper*.  
6. vLLM issues **#30135** (MXFP4→Marlin fallback), **#38022** (2880 Marlin MoE).  
7. vLLM FP8 quantization docs (Hopper W8A8 role).  
8. SemiAnalysis InferenceX notes: H100 FP4 TC = 0.  
9. CUTLASS example **55** Hopper mixed-dtype GEMM.  
10. vLLM `csrc/cutlass_extensions/vllm_numeric_conversion.cuh` (interleaved upconvert).

---

## 9. Epistemic honesty

Web “NVFP4 is a gamechanger” posts mix **Blackwell native** and **format accuracy**. We separate:

- **Format** (bytes + scales + error) — architecture-agnostic, we implement.  
- **Datapath** (native FP4 MMA vs software dequant + FP16/FP8 MMA) — architecture-specific, our thesis lives here.

If a 2026 vLLM release lands a **great** Hopper MXFP4 kernel before we finish R2, R0+R1 (protocol + gap decomposition) **still publish**. R2 becomes a comparison, not a unique invention. That is success, not failure.

*End of analysis. Update when a quarterly literature pass moves a row in §1.*
