# Project status dashboard

**Last updated:** 2026-08-13  
**Authoritative plan:** [`MASTER_EXECUTION_PLAN.md`](MASTER_EXECUTION_PLAN.md)  
**Thesis:** [`PROJECT-v2.0-fp4-hopper.md`](PROJECT-v2.0-fp4-hopper.md)

---

## Rung status

| Rung | State | Notes |
|------|--------|------|
| **R0 Instrument** | **In progress (local peak largely landed)** | Harness, pack/probes, stream ideal, analysis, CPU tests green on sm_86. **sm_90 probe + public kernel rebench + L2 proof still open** |
| R1 Quantify | Not started | Needs sm_90 + baselines |
| R2 Close | Not started | Blocked on R1 go/no-go + D1 layout complete |
| R3 Extend | Not started | |

---

## Hardware

| Device | Probe BW | Harness validated | ncu counters | Exclusive runs |
|--------|----------|-------------------|--------------|----------------|
| sm_86 RTX 3050 laptop | **124.2 GB/s** RO | **Yes** (FP16 M=1 ~101% ideal) | **FAIL** G2 | N/A (local) |
| sm_90 H100 DGX | **UNMEASURED** | No | UNKNOWN G3 | TBD |

---

## Predictions

See [`../PREREGISTRATION.md`](../PREREGISTRATION.md).  
sm_86 harness outcomes recorded 2026-08-12. sm_90 predictions incomplete until probe.

---

## Blockers (need human / external)

1. **DGX H100** — `make ARCH=sm_90 run-probe` + fill PREREGISTRATION  
2. **ncu on laptop** — Windows GPU Performance Counters ACL (G2 FAIL)  
3. **ncu on DGX** — admin policy  
4. **Upstream demand** ping (G6)  
5. **GitHub PAT `workflow` scope** — `cpu-ci.yml` is ready locally under `.github/workflows/` but **not on remote** until the token includes `workflow`  

---

## Recent milestones

| Date | Milestone |
|------|-----------|
| 2026-08-12 | R0 harness, probes, env capture, preregistration, cuBLAS FP16 sm_86 |
| 2026-08-13 | Peak master plan + governance docs pushed |
| 2026-08-13 | Host pack/shapes/probes + `make test-cpu` OK |
| 2026-08-13 | Stream-ideal bench + analyze/plot scripts |
| 2026-08-13 | G2 ncu gate FAIL logged; CONTRIBUTING |

---

## Next actions (priority)

1. User: fix G2 ncu OR accept profiler-free methodology  
2. User: DGX G1 probe session  
3. Public baseline scaffolding (Marlin/etc.) when torch/GPU ready  
4. Complete D1 layout derivation study before any R2 kernel  
