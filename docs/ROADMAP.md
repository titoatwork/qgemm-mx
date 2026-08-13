# Roadmap

Aligned with `MASTER_EXECUTION_PLAN.md`. Dates are targets assuming active part-time (~12–20 h/week) plus intermittent DGX access.

## Now — Phase 0 + R0 local peak

| Week | Deliver |
|------|---------|
| W0 | Master plan, STATUS, GATES, SHAPES, LICENSE, git reconcile |
| W1 | Host pack/unpack, structured probes, CPU tests, CI |
| W1–2 | Stream-ideal bench, analysis/plots, PROTOCOL, sm_86 full re-run |
| W2 | Baseline study scaffolding; D1 layout study notes (no kernel) |

## Next — R0 full (DGX)

| When | Deliver |
|------|---------|
| First DGX day | G1 probe, G2/G3 ncu, fill PREREGISTRATION sm_90 |
| +1–2 sessions | Public kernel rebench under protocol → BASELINE_STUDY.md |
| | Prove L2 hot/cold divergence (G4) |

## Then — R1

| When | Deliver |
|------|---------|
| After R0 full | Gap decomposition, waterfall, Nsight tables, go/no-go for R2 |

## Then — R2 (conditional)

| When | Deliver |
|------|---------|
| After R1 GO | D1 complete, qgemv_mx, qgemm_mx, 2880 shapes, must-pass bars |

## Then — R3 + upstream

| When | Deliver |
|------|---------|
| After R2 or in parallel late R1 | Triton twin, portability, PR, final report |

## Kill criteria

- R1 shows structural impossibility of ≥1.2× MXFP4 vs FP8 at M=1 after full ablation → **publish negative, stop R2**.  
- Upstream explicit “not wanted” + no other consumer → demote D8, keep measurement paper.  
