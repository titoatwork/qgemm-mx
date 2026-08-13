# D1 — Layout derivation (working draft)

**Status:** Skeleton for study → complete **before** R2 PTX.  
**Rule:** Weight layout is an **output** of the dequant instruction sequence + datapath fragment requirements. Cite Marlin/Machete as prior art; state derive vs adapt per element.

| Work item | State |
|-----------|--------|
| Occupancy / CTA table (tile N=256, no split-K) | **DONE** — [`OCCUPANCY.md`](OCCUPANCY.md), `occupancy.hpp`, `make test-cpu` |
| Dequant instruction sequence (cost model) | Notes only |
| Nibble order / 32-bit word diagram | **Open** (checkboxes below are honest) |
| sm_86 `mma.sync` fragment map | **Open** |
| sm_90 TMA + `wgmma` layout | **Open** |
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

- [ ] Confirm exact order against `lop3` / PTX sequence we implement  
- [ ] Draw 32-bit word diagram once sequence is frozen  
- [ ] Offline permute at pack time so global load delivers per-thread fragments  

### Step C — sm_86 fragment mapping (`mma.sync`)

- Ampere `mma.m16n8k16` (and kin): each lane owns a fixed subset of B-fragment elements.  
- `ldmatrix` does **not** load raw 4-bit; without offline permute → smem staging + shuffles.  
- Marlin’s answer: pack so one 128-bit load / thread is already in fragment order.  
- **Our job:** re-derive for MX/NV block scales (group boundaries ≠ 128 always).  

- [x] **Occupancy / CTA table (paper arithmetic)** — **DONE.** Tile N=256, one CTA per N-tile, no split-K. M=1, N=4096 → **16 CTAs**: 20 SMs (3050) nearly full; **132 SMs (H100) ~12%** → split-K mandatory on sm_90. Every frozen layer is H100-starved without split-K. Table: [`OCCUPANCY.md`](OCCUPANCY.md). Code: `include/qgemm/occupancy.hpp`. Reproduce: `make test-cpu` or `python3 python/occupancy_table.py`.

This table is **not** a fragment map. Steps below remain open.

### Step D — sm_90 TMA + `wgmma`

- `wgmma` reads B from **shared memory** under required swizzles; async pipeline.  
- TMA descriptors constrain 2D tile shapes and swizzle modes.  
- An Ampere/Marlin register-B layout **does not** satisfy this → **Machete-class** redesign, not a port.  
- This is the **highest-value** derivation for the project (authoritative results are sm_90).

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
