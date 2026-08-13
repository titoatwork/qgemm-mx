# D1 — Layout derivation (working draft)

**Status:** Skeleton for study → complete **before** R2 PTX.  
**Rule:** Weight layout is an **output** of the dequant instruction sequence + datapath fragment requirements. Cite Marlin/Machete as prior art; state derive vs adapt per element.

## 1. Goal

Produce a 32-bit word diagram and global-load story for:

| Arch | Datapath | Kernel family |
|------|----------|----------------|
| sm_86 | `mma.sync` + register B fragments | dev / portability |
| sm_90 | `wgmma` (B from smem) + TMA | **authoritative** |

Formats: MXFP4 (E2M1 + E8M0/32), NVFP4 (E2M1 + E4M3/16), INT4-g128.

## 2. Derivation steps (to fill)

### Step A — Dequant instruction sequence

- [ ] Naive cost model (shift/mask/cvt/mul) — issue-bound at small M  
- [ ] `lop3` / mantissa-OR path for two nibbles → f16x2  
- [ ] MXFP4: E8M0 as **exponent add** (speed vs accuracy trade)  
- [ ] NVFP4: E4M3 scale multiply cost  
- [ ] INT4-g128: FP16 group scale  

### Step B — Nibble order inside 32-bit words

- [ ] Required order for pairwise `lop3` (e.g. 0,2,4,6,1,3,5,7 literature pattern)  
- [ ] Diagram: bits → lanes  

### Step C — sm_86 fragment mapping

- [ ] Which B elements each lane needs for chosen `mma` shape  
- [ ] Why offline permute beats smem+shuffle  
- [ ] Relation to Marlin (cite; do not paste tables without attribution)  

### Step D — sm_90 TMA + wgmma

- [ ] TMA descriptor constraints  
- [ ] wgmma B smem swizzle  
- [ ] Why Ampere layout fails → why Machete exists  

### Step E — Split-K and scale-group boundaries

- [ ] CTA count for frozen shapes on 132 SM  
- [ ] Reduction strategy + determinism vs tolerances  
- [ ] Group size 32/16/128 vs K-tile  

## 3. Non-128-aligned shapes

- [ ] K=N=2880 path: correctness first, no silent pad that breaks quant blocks  
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
