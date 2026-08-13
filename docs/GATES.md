# Hard gates

Gates are **binary**. Do not pretend R0/R1 on sm_90 is complete while any gate is red.

| ID | Gate | How to check | Status | Notes |
|----|------|--------------|--------|-------|
| G1 | Empirical BW on **sm_90** | `make ARCH=sm_90 run-probe` on H100; record in PREREGISTRATION | **RED** | Denominator for all authoritative claims |
| G2 | Nsight counters **sm_86** | `scripts/check_ncu_gate.sh` | **UNKNOWN** | Windows Control Panel → GPU Performance Counters if WSL |
| G3 | Nsight counters **sm_90** | Same on DGX; may need admin | **UNKNOWN** | Fallback: BW from timing×bytes (weaker) — design in if red |
| G4 | L2 flush/rotation proven where L2 > weights | Hot vs cold diverge on H100 | **RED** | sm_86 cannot prove this (L2 1.5 MB) |
| G5 | Git `main` = origin, pushable | `git status` clean tracking | **YELLOW** | Diverged histories observed 2026-08-13 |
| G6 | Upstream interest | Maintainer/thread says wanted or not | **RED** | Ask before large PR investment |
| G7 | Scope freeze | §4.3 non-goals not reopened | **GREEN** | Enforced by review |

## Fallback if G2/G3 red

- Report `% ideal` from timed kernel and known byte counts  
- Still require graphs + rotation  
- Explicitly mark profiler-free methodology in REPORT  

## User actions required

- [ ] Run probe on DGX; paste/commit results  
- [ ] Fix or confirm ncu on laptop  
- [ ] Confirm ncu on DGX  
- [ ] Ping upstream channels (optional but G6)  
