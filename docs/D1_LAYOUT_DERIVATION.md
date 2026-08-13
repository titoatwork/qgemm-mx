# D1 — Layout derivation (working draft)

**Status:** Skeleton for study → complete **before** R2 PTX.  
**Rule:** Weight layout is an **output** of the dequant instruction sequence + datapath fragment requirements. Cite Marlin/Machete as prior art; state derive vs adapt per element.

| Work item | State |
|-----------|--------|
| Occupancy / CTA table (tile N=256, no split-K) | **DONE** — [`OCCUPANCY.md`](OCCUPANCY.md), `occupancy.hpp`, `make test-cpu` |
| Dequant instruction sequence (cost model) | **INT4 0x6400 + MX ldexp/expfield proven** (`test_bittrick_splitk`, `test_mx_decode`) |
| Nibble order / 32-bit word diagram | **Sequential + even/odd DONE**. A-fragment host permute **DONE** |
| sm_86 `mma.sync` fragment map | **Open** (dev path; not the flagship) |
| sm_90 TMA + `wgmma` layout | **A-fragment table + permute DONE**. TMA / emit still open |
| Split-K vs scale-group boundaries | **Open** |
| K=2880 sequential-pack correctness | MX fused-ref **done**; R2 device pack **open** |

## 1. Goal

Produce a 32-bit word diagram and global-load story for:

| Arch | Datapath | Kernel family |
|------|----------|----------------|
| sm_86 | `mma.sync` + register B fragments | dev / portability |
| sm_90 | `wgmma` (B from smem) + TMA | **authoritative** |

Formats: MXFP4 (E2M1 + E8M0/32), NVFP4 (E2M1 + E4M3/16), INT4-g128.

## 2. Derivation steps

### Step A — Dequant instruction sequence

| Path | Cost model (decode, small M) | Notes |
|------|------------------------------|--------|
| Naive | shift + mask + `cvt` + scale mul ≈ 5–6 inst/weight | Often **issue-bound**, not pure HBM |
| Marlin-class / TRT interleaved | `lop3` build two FP16 mantissas + bias remove + scale ≈ ~2 inst/weight | Prior art — **cite**, re-derive for our format |
| MXFP4 E8M0 | Scale is **power of two** → fold into exponent (add), not FP mul | Speed win + accuracy cost (v2 thesis) |
| NVFP4 E4M3 | True scale multiply (or approximate) | Slightly costlier dequant than MX |
| INT4-g128 | FP16 group scale mul | Clean roofline arm |

**R0 host pack** (`pack.hpp`) uses sequential nibble order for correctness only.  
**R2 device pack** must follow Step B–D, not host order.

### Step B — Nibble order inside 32-bit words

Target literature pattern for pairwise FP16 construction (verify against chosen ISA sequence):

```
sequential indices:  0 1 2 3 4 5 6 7  (within 8 consecutive 4-bit elems / 32-bit word)
device-friendly:     0 2 4 6 1 3 5 7  (even then odd — enables dual-nibble SIMD paths)
```

**Sequential 32-bit word (`pack.hpp` / fused-ref) — DONE, authoritative for R0/R1:**

```
 byte          0              1              2              3
        ┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐┌──────┴──────┐
 bits   7..4    3..0    7..4    3..0    7..4    3..0    7..4    3..0
 k      1       0       3       2       5       4       7       6
        high    low     high    low     high    low     high    low
```

Along a row of W[N,K], element index `lin = n*K + k`. Byte `lin/2` holds **low nibble = even k**, **high nibble = odd k**.

This is **not** a `wgmma` A-fragment. It is the correctness layout. R2 must **offline-permute** into even/odd (or CuTe-derived) order so one 32/128-bit load feeds two FP16 lanes without extra smem gathers.

**Hopper A-operand view (logical, not yet CuTe):** after `Yᵀ = W Xᵀ`, a thread that owns A-elements corresponding to weight rows `n` and reduction `k` wants consecutive K for MMA K-tiles. Sequential K-packing is *friendly to K-scan fused-ref* and *hostile to wgmma A* unless reshuffled so MMA K-vectors are contiguous in the loaded word.

```
  logical W row n:  k=0 k=1 k=2 k=3 k=4 k=5 k=6 k=7 ...
  sequential pack:  [k0 k1][k2 k3][k4 k5][k6 k7]   ← our R0
  even/odd word:    [k0 k2 k4 k6 | k1 k3 k5 k7]     ← typical dual-convert
  wgmma A fragment: architecture-defined subset of (n,k) per lane
                    (fill with CuTe / PTX doc before R2 codegen)
```

- [x] Sequential 32-bit word diagram (R0 pack)  
- [x] Even/odd 32-bit permute + inverse + dual-nibble extract (`permute.hpp`, `test_permute`, `test_even_odd_gpu`)  
- [x] First-class `pack_mxfp4_even_odd` / `pack_int4_even_odd` + dual-nibble fused-ref (`k_qgemm_ref_mx_even_odd_dual`) — matches sequential (max_abs=0); ~3× faster than scalar sequential on 256² sm_86 (not a roof claim)  
- [x] Written **A-fragment contract** (no invented per-lane table): [`A_FRAGMENT.md`](A_FRAGMENT.md)  
- [x] Opcode **m64n8k16**; official A-fragment table transcribed; `permute_seq_to_a_fragment()` + inverse (`a_fragment.hpp`, `test_a_fragment`)  

### Step C — sm_86 fragment mapping (`mma.sync`)

- Ampere `mma.m16n8k16` (and kin): each lane owns a fixed subset of B-fragment elements.  
- `ldmatrix` does **not** load raw 4-bit; without offline permute → smem staging + shuffles.  
- Marlin’s answer: pack so one 128-bit load / thread is already in fragment order.  
- **Our job:** re-derive for MX/NV block scales (group boundaries ≠ 128 always).  

- [x] **Occupancy / CTA table (paper arithmetic)** — **DONE.** Tile N=256, one CTA per N-tile, no split-K. M=1, N=4096 → **16 CTAs**: 20 SMs (3050) nearly full; **132 SMs (H100) ~12%** → split-K mandatory on sm_90. Every frozen layer is H100-starved without split-K. Table: [`OCCUPANCY.md`](OCCUPANCY.md). Code: `include/qgemm/occupancy.hpp`. Reproduce: `make test-cpu` or `python3 python/occupancy_table.py`.

This table is **not** a fragment map. Steps below remain open.

### Step D — sm_90 TMA + `wgmma`  **(Hopper design law)**

Sourced: Machete (Wilkinson 2024), Colfax WGMMA tutorial, CUTLASS ex. 55. Full writeup: [`RESEARCH_ANALYSIS.md`](RESEARCH_ANALYSIS.md).

- `wgmma` is warpgroup-async. **Operand B is sourced from SMEM; A and C from registers.**  
- We upconvert 4-bit → FP16 **in registers**. Therefore weights **cannot** be `wgmma` B unless we store them back to SMEM (death).  
- **Machete’s move:** compute **`Yᵀ = Wᵀ Xᵀ`** so upconverted weights are operand **A** (registers). CUTLASS/CuTe makes the transpose a layout, not a memcpy.  
- **`ldmatrix` has no 4-bit.** Pre-shuffle offline so one **32- or 128-bit** SMEM load feeds the fragment (Marlin *and* Machete). Sequential `pack.hpp` order is **wrong** for this.  
- TMA: async GMEM→SMEM tiles; warp-specialized producer/consumer.  
- Marlin’s hand `mma` shuffle **does not port**; ~37% Hopper peak lost if we only issue `mma.sync`.  
- Interleaved nibble→f16 prior art: vLLM `vllm_numeric_conversion.cuh` — **cite, do not paste tables blindly**.

- [x] Explicit “transpose GEMM” in code: `include/qgemm/kernel_contract.hpp` (`wgmma_view`)  
- [x] Written 32-bit word diagram for **A-fragment** (not Ampere B-fragment) — [`A_FRAGMENT.md`](A_FRAGMENT.md)  
- [x] Host pre-shuffle = official PTX table (`a_fragment.hpp`). CuTe codegen still optional later.  
- [x] Device A-fragment fused-ref (`k_qgemm_ref_mx_a_fragment`)  
- [x] wgmma SMEM descriptor encode/decode (`wgmma_smem_desc.hpp`). TMA tensormap still open (sm_90).  

Logical `Y = X @ Wᵀ` ↔ issued MMA `Yᵀ = W Xᵀ` (weights = wgmma **A**).

### Step E — Split-K and scale-group boundaries

| Shape | N | K | Notes |
|-------|--:|--:|------|
| q_o_proj | 4096 | 4096 | Canonical |
| gpt_oss_2880 | 2880 | 2880 | **2880 % 128 ≠ 0** — Marlin-class align pain; MX block 32 OK |
| down_proj | 4096 | 14336 | Deep K → split-K heavy |

- Reduction: atomics vs two-pass; document determinism vs correctness tolerances.  
- K-tile must respect group size (32 / 16 / 128) or restructure loop at group edges.  

## 3. Non-128-aligned shapes

- [x] Sequential-pack MXFP4 accepts K=2880; INT4-g128 rejects — host + `make test-gpu` (`test_gptoss_align`)  
- [ ] R2 **device-pack** K=N=2880 path: correctness first, no silent pad that breaks quant blocks  
- [ ] Link to production pain (Marlin 128-align)  

## 4. Sign-off

| Item | Owner | Date |
|------|-------|------|
| Draft complete | | |
| Peer review | | |
| Frozen for R2 coding | | |

## References (to expand)

- Marlin (IST-DASLab) — FP16×INT4  
- Machete / CUTLASS 3.x — Hopper mixed-input  
- OCP MX formats — MXFP4  
- NVFP4 vendor docs  
