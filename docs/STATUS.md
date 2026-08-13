# Project status dashboard

**Last updated:** 2026-08-13  
**Authoritative plan:** [`MASTER_EXECUTION_PLAN.md`](MASTER_EXECUTION_PLAN.md)  
**Thesis:** [`PROJECT-v2.0-fp4-hopper.md`](PROJECT-v2.0-fp4-hopper.md)

---

## Rung status

| Rung | State | Notes |
|------|--------|------|
| **R0 Instrument** | **In progress — local peak strong** | Full frozen shapes on sm_86 (cuBLAS M≤256 + stream all formats). Pack/probes/ref GEMM CPU tests green. **sm_90 + public kernels still open** |
| R1 Quantify | Not started | Needs sm_90 + baselines |
| R2 Close | Not started | D1 layout incomplete; R1 go/no-go |
| R3 Extend | Not started | |

---

## Hardware

| Device | Probe BW | Harness validated | ncu counters | Exclusive runs |
|--------|----------|-------------------|--------------|----------------|
| sm_86 RTX 3050 laptop | **124.2 GB/s** RO | **Yes** | **FAIL** G2 | N/A |
| sm_90 H100 DGX | **UNMEASURED** | No | UNKNOWN G3 | TBD |

---

## Key local artifacts

| Path | Content |
|------|---------|
| `results/raw/stream_ideal_sm86_fullshapes.csv` | Pure weight stream, all layers × formats |
| `results/raw/cublas_fp16_sm86_fullshapes_Mle256.csv` | FP16 cuBLAS, all layers, M≤256 |
| `docs/figures/*fullshapes*` | Latency / %ideal plots |
| `include/qgemm/pack.hpp` + `tests/` | Host quant pack + probes + ref GEMM |

---

## Blockers (need you)

1. **DGX:** `make ARCH=sm_90 run-probe` → PREREGISTRATION sm_90 table  
2. **ncu laptop:** Windows GPU counter ACL (G2)  
3. **PAT workflow scope** if you want CI YAML on GitHub  
4. **Upstream demand** ping (G6)  

---

## Next (automated when possible)

1. Naive dequant+cuBLAS bench (sanity floor)  
2. Public baseline scaffolding (Marlin) when torch available  
3. D1 layout study fill-in  
4. sm_90 R0 when machine available  

---

## Recent commits (selected)

| SHA | Summary |
|-----|---------|
| `30fe16b` | Full shapes sm_86, ref GEMM, baseline scaffold |
| `da4fd7c` | Host pack / probes / CPU tests |
| `45f6c5e` | Stream ideal + analysis |
| `a6bf3a2` | Peak master plan |
