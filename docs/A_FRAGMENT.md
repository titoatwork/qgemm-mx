# Hopper `wgmma` A-fragment — first opcode, official table, host permute

**Status:** Opcode **picked**. Per-lane table **transcribed** from PTX (not invented). Host permute **implemented**. No `wgmma` issued yet.  
**Authority:** PTX ISA 8.5 (CUDA 12.6, this repo’s toolkit) and PTX ISA 9.3 (current web docs).  
**Code:** `include/qgemm/a_fragment.hpp`, `tests/test_a_fragment.cpp`  
**Date:** 2026-08-13

---

## 1. Opcode we will implement first

| Field | Choice | Why |
|-------|--------|-----|
| Family | `wgmma.mma_async.sync.aligned` | Hopper peak; not `mma.sync` |
| Acc | `.f32` | Contract: FP32 accumulate |
| A/B after upconvert | `.f16` / `.f16` | Software FP4 → FP16 then MMA |
| Shape | **`m64n8k16`** | See below |
| PTX | `wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16` | `wgmma::kFirstOpcode` |
| Warpgroup | **128 threads** (4 warps) | Hardware unit |

**Why n=8, not n=16 / n=64.**  
A is **64×16 for every** `m64nNk16`. N only sizes B and C. Issued problem is `Yᵀ = W @ Xᵀ`, so wgmma-N = logical **M**. Decode has M=1; the ISA minimum N is 8. Larger n wastes C registers (4 f32/thread at n=8 vs 8 at n=16 vs 32 at n=64) and does not change the A-fragment we must pre-shuffle. Prefill can retarget n later; the A permute stays.

We do **not** emit this instruction yet.

---

## 2. Operand mapping (unchanged)

```
logical:  Y = X @ Wᵀ
issued:   Yᵀ = W @ Xᵀ
A = upconverted W [N, K]   registers   ← this fragment
B = Xᵀ            [K, M]   SMEM (TMA)
C = Yᵀ            [N, M]   registers
```

One atom: A is 64 rows of W × 16 of K.

---

## 3. Official A-fragment table (transcribed)

**PTX ISA 8.5 / 9.3** — section *Matrix Fragments for wgmma.mma_async.m64nNk16*  
HTML: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html  
CUDA 12.6 archive: https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html  
Contents listing (ISA 8.5): §9.7.16.5.1.1.1 / neighbouring *m64nNk16* fragment subsection.

The per-warp `(a0…a7)` map is the **same text** as *Matrix Fragments for `mma.m16n8k16` with `.f16` and `.bf16` types*. Four warps cover M=64: warp `w` owns rows `[16w, 16w+16)`.

PTX (structure copied, not paraphrased into a different numbering):

```
groupID           = %laneid >> 2
threadID_in_group = %laneid % 4

row = groupID           for ai where  0 <= i < 2  ||  4 <= i < 6
    = groupID + 8       otherwise

col = (threadID_in_group * 2) + (i & 0x1)         for ai where i <  4
    = (threadID_in_group * 2) + (i & 0x1) + 8     for ai where i >= 4
```

Then `row += 16 * warp_id` with `warp_id = tid / 32`, `tid ∈ [0, 128)`, `laneid = tid % 32`.

| i | row (in warp) | col |
|--:|---------------|----:|
| 0 | groupID | 2·tg + 0 |
| 1 | groupID | 2·tg + 1 |
| 2 | groupID + 8 | 2·tg + 0 |
| 3 | groupID + 8 | 2·tg + 1 |
| 4 | groupID | 2·tg + 8 |
| 5 | groupID | 2·tg + 9 |
| 6 | groupID + 8 | 2·tg + 8 |
| 7 | groupID + 8 | 2·tg + 9 |

`tg = threadID_in_group`. Each 32-bit A register is one `.f16x2`: `(a0,a1)`, `(a2,a3)`, `(a4,a5)`, `(a6,a7)` — **already a consecutive-K pair on one row**. Dual convert applies *inside* those pairs. It does **not** need the even/odd 8-wide word.

Worked example, `tid = 0` (warp 0, lane 0):

| i | (row, col) in the 64×16 A tile |
|--:|--------------------------------|
| 0 | (0, 0) |
| 1 | (0, 1) |
| 2 | (8, 0) |
| 3 | (8, 1) |
| 4 | (0, 8) |
| 5 | (0, 9) |
| 6 | (8, 8) |
| 7 | (8, 9) |

`tid = 32` (warp 1, lane 0) is the same pattern with row += 16.

128 threads × 8 elements = 1024 = 64 × 16. Tested as a bijection (`test_a_fragment`).

### C-fragment (same sitting, n=8, `.f32`)

Same per-warp map as `mma.m16n8k16` C. 4 f32 per thread.

```
row = groupID           for di where i <  2
    = groupID + 8       for di where i >= 2
col = (threadID_in_group * 2) + (i & 0x1)
```

plus `row += 16 * warp_id`. Bijection on 64×8 tested.

---

## 4. Even/odd is not this table

Even/odd word (implemented earlier):

```
one row, 8 consecutive K:  [k0 k2 k4 k6 | k1 k3 k5 k7]
```

tid 0’s A-fragment is **2 rows × 4 K-cols**, not 8 K of one row. Intersection with even/odd word 0 of row 0 is only `(0,0)` and `(0,1)`. Treating even/odd as “the Hopper layout” is the Marlin-port mistake this file exists to prevent.

Pipeline:

```
offline:  sequential pack
       →  even/odd          [optional; fused-ref convert]
       →  A-fragment shuffle [DONE as host permute]
load:     words[n_tile][k_tile][tid]   (32-bit = 8 nibbles in a0..a7 order)
upconvert: nibble i → f16 ai  (pair i,i+1 already same-row consecutive K)
wgmma:    A=regs  B=TMA(Xᵀ)            [NOT DONE]
```

---

## 5. Host permute

`permute_seq_to_a_fragment` / `pack_mxfp4_a_fragment` (also NV, INT4):

```
words[n_tile][k_tile][tid]   n_tile = N/64, k_tile = K/16
word bits [4i : 4i+4) = nibble of W[n_tile*64 + row(tid,i), k_tile*16 + col(tid,i)]
```

Scales stay sequential (indexed by logical `(n,k)`).  
Requires `N % 64 == 0`, `K % 16 == 0`. **Every frozen layer satisfies this**, including `gpt_oss_2880` (2880/64 = 45, 2880/16 = 180). INT4-g128 still rejects K=2880 at *pack* time (group 128), not at the fragment tile.

Inverse is bit-exact. `make test-cpu` runs `test_a_fragment`.

---

## 6. Occupancy: atom tile vs CTA tile

The fragment does **not** force CTA N=64. A CTA may issue 1 atom (tile 64) or 4 (tile 256, the fused-ref default).

| CTA tile | q_o_proj CTAs | occ H100 | still starved? | split-K to ~2×132 |
|---------:|--------------:|---------:|:--------------:|------------------:|
| 256 (4 atoms) | 16 | 0.121 | yes | 17 |
| **64 (1 atom)** | **64** | **0.485** | **yes** | **5** |

One-atom CTAs help a lot and **still** starve 132 SMs without split-K. `kv_gqa` at tile 64 is 16 CTAs / 0.121 — worst layer, unchanged in kind. Full tables: [`OCCUPANCY.md`](OCCUPANCY.md).

MX block 32 = **two** k16 steps. NV block 16 = one k16. 2880 aligns with both.

---

## 7. Checklist

- [x] Pick exact `wgmma` shape → **m64n8k16**
- [x] Transcribe A-fragment table from PTX (cite section + version)
- [x] `permute_to_a_fragment()` + inverse + test vs table
- [x] Recompute occupancy for CTA tile = 64 (and keep 256)
- [ ] TMA descriptor for B = Xᵀ tile
- [ ] Emit `wgmma` on sm_90 (DGX)
- [ ] Do **not** ship a kernel that only does even/odd dual-nibble and claims Hopper peak

---

## 8. What this is not

- Not a kernel. `src/kernels/` stays empty.
- Not a CuTe codegen dump. The table is the ISA, in our types.
- Not a perf claim. Host permute is packing, not throughput.

*End.*
