# Project status dashboard

**Last updated:** 2026-08-13  
**Authoritative plan:** [`MASTER_EXECUTION_PLAN.md`](MASTER_EXECUTION_PLAN.md)  
**Thesis:** [`PROJECT-v2.0-fp4-hopper.md`](PROJECT-v2.0-fp4-hopper.md)

---

## Rung status

| Rung | State | Notes |
|------|--------|------|
| **R0 Instrument** | **In progress** | Harness + FP16 cuBLAS validation on sm_86 done. Pack/probes/stream/CI/analysis pending. sm_90 + public baselines **not started** |
| R1 Quantify | Not started | Needs sm_90 + baselines |
| R2 Close | Not started | Blocked on R1 go/no-go |
| R3 Extend | Not started | |

---

## Hardware

| Device | Probe BW | Harness validated | ncu counters | Exclusive runs |
|--------|----------|-------------------|--------------|----------------|
| sm_86 RTX 3050 laptop | **124.2 GB/s** RO | **Yes** (FP16 M=1 ~101% ideal) | TBD gate | N/A (local) |
| sm_90 H100 DGX | **UNMEASURED** | No | TBD gate | TBD |

---

## Predictions

See [`../PREREGISTRATION.md`](../PREREGISTRATION.md).  
sm_86 harness outcomes recorded 2026-08-12. sm_90 predictions incomplete until probe.

---

## Blockers (need human / external)

1. DGX H100 access for authoritative R0/R1  
2. Nsight counter permissions (laptop Windows ACL + DGX policy)  
3. Upstream demand validation (vLLM / SGLang / torchao)  
4. Git: reconcile local main with origin if histories diverged  

---

## Recent milestones

| Date | Milestone |
|------|-----------|
| 2026-08-12 | R0 harness, probes, env capture, preregistration, cuBLAS FP16 sm_86 |
| 2026-08-13 | Master execution plan (peak standard) + living governance docs |
| 2026-08-13 | CPU CI workflow (`.github/workflows/cpu-ci.yml`) + `CONTRIBUTING.md` |
| 2026-08-13 | G2 ncu gate on sm_86: **FAIL** (`ERR_NVGPUCTRPERM`) — see `docs/GATES.md` |

---

## Next actions (priority)

1. Land host pack + structured probes + CPU tests (`make test-cpu`)  
2. Stream-ideal bench + analysis scripts  
3. User: fix ncu counters on laptop (G2) + DGX probe when available  
4. Upstream demand check (G6) when ready  
