# Hopper `wgmma` A-fragment — written contract (not CuTe codegen)

**Status:** Schematic + checklist. **Per-lane element table is not invented here.**  
**Authority:** PTX ISA (`wgmma.mma_async` register fragments), Machete/Wilkinson, Colfax WGMMA tutorial, `kernel_contract.hpp`.  
**Date:** 2026-08-13

This document exists so R2 cannot start from folklore. It says what we **know**, what we **will look up in PTX on the day we emit `wgmma`**, and how that maps onto **our** `(M,N,K)` and even/odd pack.

---

## 1. Instruction we will target first

| Field | Choice | Why |
|-------|--------|-----|
| Family | `wgmma.mma_async.sync.aligned` | Hopper peak; not `mma.sync` |
| Acc | `.f32` | Contract: FP32 accumulate |
| A/B | `.f16` / `.f16` after upconvert | Software FP4 → FP16 then MMA |
| Shape (first) | **m64n8k16** or **m64n16k16** | Standard CUTLASS/Hopper tiles; final pick when we implement |
| Warpgroup | **128 threads** (4 warps) | Hardware unit of `wgmma` |

We do **not** emit this instruction yet.

---

## 2. Operand mapping (frozen)

Logical linear layer (`shapes.hpp`):

```
X [M, K]     activations   (row-major)
W [N, K]     weights       (row-major, 4-bit packed)
Y [M, N]     output        (row-major)
Y = X @ Wᵀ
```

`wgmma` rule: **B is SMEM-only; A and C are registers.**  
Upconverted weights live in registers → they must be **A**.

Issued problem (`kernel_contract.hpp::wgmma_view`):

```
A  = upconverted W     [N, K]     registers
B  = Xᵀ                [K, M]     SMEM (TMA)
C  = Yᵀ                [N, M]     registers
Yᵀ = W @ Xᵀ
```

If `M=1` (decode), C is an N-vector; A is still the fat weight matrix. That is why **split-K** and **N-tiles** dominate occupancy, not a fat M-tile.

---

## 3. What an “A-fragment” is

For one `wgmma` of shape **m64 × k16** (A is 64×16 f16):

- 64 rows of A ↔ **64 rows of W** (an N-tile of 64)  
- 16 cols of A ↔ **16 of K** (must be a multiple of MX block 32 after enough steps: two k16 = 32)  
- 128 threads share that 64×16 tile  
- **Elements per thread** = `(64×16) / 128 = 8` f16 values (typically 4× `.f16x2`)

```
          K-tile (16) ---->
        ┌──────────────────┐
  N-tile│   A fragment     │  64 rows × 16 cols  f16
  (64)  │  owned by 128    │
   |    │  threads         │
   v    └──────────────────┘
```

**PTX defines which (row, col) each `%tid` holds.** That table is in:

> PTX ISA → `wgmma.mma_async` → *Register Fragments and Shared Memory Matrix Layouts*  
> (version-locked the day we implement; do not copy from memory)

Until that table is transcribed **from the ISA PDF for the exact `.m64n*k16.f16` variant we pick**, any “lane 7 owns W[n+3,k+12]” claim is **fiction**. We refuse to write fiction into the kernel.

### What we *can* say without the table

| Constraint | Consequence for pack |
|------------|----------------------|
| A is **K-contiguous in the MMA sense**, not in our sequential byte stream | Even/odd words help dual convert; they are **not** automatically A-fragment order |
| 16 of K per instruction | Inner loop steps of 16; MX scale block is 32 → **two** wgmma K-steps per MX block (or pack scales to match) |
| 64 of N per instruction | N-tile 64 (or 128 with two warpgroups) vs our occupancy table’s 256 — occupancy math must be **recomputed** for the real tile |
| No 4-bit `ldmatrix` | Offline permute so a wide load fills the **8 f16s that this lane will hold after upconvert** |

---

## 4. How even/odd sits *under* the fragment (not instead of it)

Even/odd word (implemented, tested):

```
uint32 slots LSB→MSB:  k0 k2 k4 k6 | k1 k3 k5 k7
```

That is a **convert-friendly** 8-wide K bundle.  
A-fragment is a **thread-owned** subset of a 64×16 A tile.

Pipeline we will implement later:

```
offline:  sequential pack
       →  even/odd  (dual convert)     [DONE as optional layout]
       →  A-fragment pre-shuffle       [NOT DONE — needs PTX table]
load:     32/128-bit → registers
upconvert: nibble → f16  (INT4 0x6400 / MX ldexp)   [DONE as decode]
wgmma:    A=regs  B=TMA(Xᵀ)
```

Skipping the third arrow and calling even/odd “the Hopper layout” would be the Marlin-port mistake.

---

## 5. MX / NV scale vs K16

| Format | Block | vs k16 wgmma |
|--------|------:|----------------|
| MXFP4 | 32 | 2 × k16 share one E8M0 per 32 along K |
| NVFP4 | 16 | 1 × k16 = one E4M3 block (aligned — nice) |
| INT4-g128 | 128 | 8 × k16 per group |

**2880 / 16 = 180** — NV and wgmma k16 **align**.  
**2880 / 32 = 90** — MX aligns.  
**2880 / 128 = 22.5** — INT4-g128 **does not**. Unchanged.

---

## 6. Implementation checklist (R2 gate)

- [ ] Pick exact `wgmma` shape (m64n8k16 vs m64n16k16 vs n32…)  
- [ ] Transcribe **A-fragment table** from PTX ISA for that opcode (cite section + version)  
- [ ] Write `permute_to_a_fragment()` + inverse + test vs table  
- [ ] TMA descriptor for B = Xᵀ tile  
- [ ] Recompute CTA/split-K with tile N = 64 (not 256)  
- [ ] Do **not** ship a kernel that only does even/odd dual-nibble and claims Hopper peak  

---

## 7. What we shipped instead (so this file is not vapor)

| Artifact | Role |
|----------|------|
| `kernel_contract.hpp` | Transpose mapping as code |
| `permute.hpp` / even-odd pack | Convert-friendly layout |
| `k_qgemm_ref_mx_even_odd_dual[_splitk]` | Prove pack+dual+split-K math |
| This file | Honest A-fragment contract |

*End. Update when the PTX table is transcribed.*
