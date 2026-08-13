# qgemm-mx — Master Execution Plan (Peak Standard)

| Field | Value |
|-------|--------|
| **Repo** | https://github.com/titoatwork/qgemm-mx |
| **Local** | `/home/titoisalive/Documents/projects/qgemm-mx` |
| **Authority** | This document + `docs/PROJECT-v2.0-fp4-hopper.md` + `PREREGISTRATION.md` |
| **Standard** | Peak — no “just enough.” Every rung independently publishable. |
| **Created** | 2026-08-13 |
| **Status** | Active execution plan. Update `docs/STATUS.md` on every milestone. |

---

## 0. One-line mission

> **Measure and recover the missing ~1.9× throughput of block-scaled FP4 (MXFP4/NVFP4) weight-only GEMM on Hopper (no native FP4), with a harness rigorous enough that every number survives review.**

---

## 1. Non-negotiable principles

1. **R0 before kernels.** No fused dequant kernel lands until the harness is proven and predictions are committed.
2. **sm_90 is truth; sm_86 is a lab.** Laptop hides wave quantization and L2 residency. Never publish occupancy conclusions from sm_86 alone.
3. **No straw-man baselines.** Headline metrics: **% of ideal bandwidth**, vs **FP8**, vs **FP16 cuBLAS**, vs **Marlin/Machete** — not “vs naive dequant.”
4. **Preregister then measure.** Edit `PREREGISTRATION.md` outcomes only by append.
5. **Structured correctness before random.** One-hot / primes / ramp before noisy matrices.
6. **Derive layout from the datapath.** Pack format is an *output* of `mma`/`wgmma` + dequant sequence — not an arbitrary convention.
7. **Graphs + L2 discipline mandatory.** `time_graphed` + buffer rotation; report hot vs cold.
8. **Empirical BW denominator only.** Never use spec-sheet peak in `pct_of_ideal`.
9. **Negative results are deliverables.** If gap is structural, stop R2 and publish R0+R1.
10. **Upstream is a goal, not vanity.** Validate demand before large PRs (vLLM / SGLang / torchao).
11. **Cite vs derive.** Marlin/Machete influence documented; no silent reimplementation of tables.
12. **Continuous commits.** One logical change per commit; push when history is clean.

---

## 2. Inventory — what already exists (do not rebuild)

### 2.1 Documentation

| Path | Role | State |
|------|------|--------|
| `docs/PROJECT-v2.0-fp4-hopper.md` | Full technical brief v2.0 | **Canonical thesis** |
| `PREREGISTRATION.md` | Predictions + sm_86 outcomes | **Live; sm_90 TBD** |
| `README.md` | Protocol + status + quick start | Good; keep in sync with STATUS |
| Parent `Documents/projects/REVIEW-*.md` | External review of v1 | Reference only |
| Parent `Documents/projects/POSITIONING-*.md` | Market / standout | Reference; optional copy into `docs/` later |

### 2.2 Code (R0 foundation)

| Path | Role | State |
|------|------|--------|
| `include/qgemm/timing.cuh` | Rotation, L2 flush, graphs, stats, launch overhead | **Solid** |
| `include/qgemm/formats.cuh` | Format enum + byte accounting | **Solid** |
| `src/probe/device_props.cu` | Device constants, ridge, wave-quant notes | **Done sm_86** |
| `src/probe/bandwidth.cu` | Empirical peak BW (denominator) | **Done sm_86: 124.2 GB/s RO** |
| `src/bench/bench_cublas.cu` | FP16 cuBLAS harness validation | **Done sm_86** |
| `scripts/env_capture.sh` | Per-run environment | **Done** |
| `Makefile` | probe + bench_cublas | **Working** |
| `results/cublas_fp16_sm86.csv` | Validation data | **Committed** |
| `results/env_sm86_initial.txt` | Env snapshot | **Committed** |
| `src/kernels/` | — | **Empty (correct for R0)** |
| `tests/` | — | **Empty — must fill** |

### 2.3 Proven on sm_86 (from PREREGISTRATION outcomes)

- Harness + denominator self-consistent: FP16 M=1 ≈ **101% of ideal**.
- L2 inflation **1.00×** (expected; 1.5 MB L2 cannot hold weights).
- Launch share low on laptop (hazard hidden).
- Unexpected: **step drop** in %ideal M=32→64 for cuBLAS — flag for Nsight/sm_90.

### 2.4 Hardware

| Device | Role | Known gaps |
|--------|------|------------|
| RTX 3050 6GB sm_86 (local) | Dev, correctness, harness | Wave quant + L2 hazards invisible; WSL2 launch ~14 µs |
| DGX H100 sm_90 | **Authoritative** | **Not yet run** — BW, ncu, exclusive node |
| Nsight Compute | Counters | **Gate:** confirm `ncu` works (laptop may need Windows GPU counter ACL) |

### 2.5 Git state (as of plan creation)

- Remote: `origin` → `github.com/titoatwork/qgemm-mx.git`
- Local history may be **diverged** from `origin/main` (duplicate commit messages, different SHAs). **Reconcile before force assumptions** — prefer merge/rebase, no force-push unless user approves.

---

## 3. Target repository layout (peak end-state)

```
qgemm-mx/
├── README.md
├── PREREGISTRATION.md
├── LICENSE
├── Makefile / CMakeLists.txt (optional later)
├── CONTRIBUTING.md
├── docs/
│   ├── PROJECT-v2.0-fp4-hopper.md      # thesis (exists)
│   ├── MASTER_EXECUTION_PLAN.md       # this file
│   ├── STATUS.md                      # living dashboard
│   ├── ROADMAP.md                     # rungs + calendar
│   ├── GATES.md                       # hard gates (ncu, DGX, upstream)
│   ├── SHAPES.md                      # frozen shape set + M sweep
│   ├── D1_LAYOUT_DERIVATION.md        # pack layout from datapath (R2 prep)
│   ├── PROTOCOL.md                    # measurement protocol detail
│   ├── BASELINE_STUDY.md              # R0 public-kernel rebench
│   ├── R1_GAP_DECOMPOSITION.md        # R1
│   ├── REPORT.md / paper draft        # D6
│   └── figures/                       # five required figures
├── include/qgemm/
│   ├── timing.cuh                     # exists
│   ├── formats.cuh                    # exists
│   ├── pack.hpp                       # host pack/unpack + reference dequant
│   ├── shapes.hpp                     # frozen shapes
│   ├── correctness.hpp                # structured probes
│   └── kernels/                       # device kernel decls (R2+)
├── src/
│   ├── probe/                         # exists
│   ├── bench/
│   │   ├── bench_cublas.cu            # exists
│   │   ├── bench_stream_ideal.cu      # packed-weight stream → t_ideal
│   │   ├── bench_fp8_cublas.cu        # if TC path available
│   │   └── bench_public_baselines.py  # orchestrate Marlin/etc when deps present
│   ├── kernels/                       # R2: qgemv_mx, qgemm_mx
│   ├── pack/                          # CPU pack tools
│   └── ref/                           # FP32 reference GEMM on dequant weights
├── tests/
│   ├── test_formats.cpp
│   ├── test_pack_roundtrip.cpp
│   ├── test_probes_cpu.cpp
│   └── test_kernel_*.cu               # R2+
├── python/
│   ├── requirements.txt               # exists
│   ├── requirements-baselines.txt     # exists
│   ├── analyze_csv.py
│   ├── plot_figures.py
│   └── shapes.py
├── scripts/
│   ├── env_capture.sh                 # exists
│   ├── run_r0_sm86.sh
│   ├── run_r0_sm90.sh
│   ├── check_ncu_gate.sh
│   └── ci_cpu.sh
├── results/
│   ├── raw/                           # CSVs per run
│   ├── env/
│   └── README.md                      # how to interpret
├── third_party/                       # vendored only with LICENSE
└── .github/workflows/
    └── cpu-ci.yml                     # compile host tests; no GPU required
```

---

## 4. Deliverable ladder (from v2 — unchanged intent)

| Rung | Content | Risk | Exit criterion |
|------|---------|------|----------------|
| **R0 Instrument** | Harness + protocol + rebench public kernels under it | Certain | Third party can run probe+benches; baseline study draft exists |
| **R1 Quantify** | Gap decomposition §3.2 causes + 3-format accuracy/speed | High value | Waterfall + profiler evidence per cause; predictions held/refuted |
| **R2 Close** | sm_90 fused dequant-GEMM (`qgemv`/`qgemm`) | Moderate | Must-pass bars in v2 §8.1 or documented structural miss |
| **R3 Extend** | MoE/grouped, Triton/CuTe twins, sm_86 portability | Swing | Optional; never cut R1 to save R3 |

**Cut order if compressed:** R3 → optional R2 axes. **Never cut R1 for R3.**

---

## 5. Phase plan (detailed)

### Phase 0 — Governance & gates (days 0–2)  ← **NOW**

| ID | Task | Owner type | Depends | Output |
|----|------|------------|---------|--------|
| P0.1 | Master plan + STATUS + ROADMAP + GATES + SHAPES | docs | — | This tree under `docs/` |
| P0.2 | LICENSE (Apache-2.0 or MIT — pick & commit) | docs | — | `LICENSE` |
| P0.3 | Reconcile git local ↔ origin | ops | — | Clean `main`, pushable |
| P0.4 | `scripts/check_ncu_gate.sh` + record result in GATES | ops | — | Pass/fail logged |
| P0.5 | User: DGX access checklist (exclusive, clocks, ncu) | **user** | — | GATES updated |
| P0.6 | User: upstream demand probe (vLLM/SGLang/torchao) | **user** | — | Note in GATES |

**Exit:** Plan committed; git clean; ncu status known; shapes frozen on paper.

---

### Phase 1 — Finish R0 harness peak (days 2–14, local-first)

| ID | Task | Notes |
|----|------|--------|
| R0.1 | `include/qgemm/shapes.hpp` + `docs/SHAPES.md` | Frozen set from v2 §9; M-sweep vector |
| R0.2 | `include/qgemm/pack.hpp` + CPU pack/unpack for INT4-g128, MXFP4, NVFP4 | Reference dequant to FP16/FP32 |
| R0.3 | Structured probes (CPU first) | one-hot, prime scales, ramp |
| R0.4 | `src/bench/bench_stream_ideal.cu` | Pure weight-stream bound for each format’s byte count |
| R0.5 | Analysis: `python/analyze_csv.py`, `plot_figures.py` skeleton | Fig 1–2 partial from existing CSV |
| R0.6 | `tests/` host tests + `make test-cpu` | formats math, pack roundtrip, probes |
| R0.7 | CI: `.github/workflows/cpu-ci.yml` | no GPU |
| R0.8 | `docs/PROTOCOL.md` | Formalize timing.cuh contract |
| R0.9 | Extend Makefile | test-cpu, stream bench, install hooks |
| R0.10 | Re-run full sm_86 matrix → `results/raw/` | env_capture each run |
| R0.11 | Document cuBLAS M=32→64 step anomaly | Issue or PREREGISTRATION append |
| R0.12 | Baseline study scaffolding | scripts to build/run Marlin when torch+GPU ready; may be DGX-only |

**Exit R0 local:**  
- Pack + probes + stream bench + CPU tests green  
- Analysis scripts produce plots from `cublas_fp16_sm86.csv`  
- PROTOCOL + SHAPES frozen  
- STATUS shows R0 local complete; R0 DGX pending  

**Exit R0 full (needs DGX):**  
- `make ARCH=sm_90 run-probe` fills PREREGISTRATION sm_90 table  
- Public kernels rebench under protocol → `docs/BASELINE_STUDY.md`  
- L2 inflation demonstrated (hot vs cold diverge)  

---

### Phase 2 — R1 Gap decomposition (needs sm_90)

| ID | Task |
|----|------|
| R1.1 | Instrument existing FP4/MX paths (vLLM fallback / Marlin FP4) under our harness |
| R1.2 | Ablation waterfall framework (even with placeholder kernels) |
| R1.3 | Attribute: wave quant / launch / layout / dequant issue cost |
| R1.4 | Nsight tables per variant |
| R1.5 | Three-format accuracy proxy plan (perplexity or layer MSE vs FP16) |
| R1.6 | Write `docs/R1_GAP_DECOMPOSITION.md` + figures 3–4 |

**Exit:** Predictions P3–P6 held or refuted with evidence. Go/no-go for R2.

---

### Phase 3 — R2 Kernel (only if R1 go)

| ID | Task |
|----|------|
| R2.0 | `docs/D1_LAYOUT_DERIVATION.md` complete before PTX heroics |
| R2.1 | `qgemv_mx` for M≤8 (split-K first-class) |
| R2.2 | `qgemm_mx` for M≥16 |
| R2.3 | Non-128-aligned K=N=2880 correct + no cliff |
| R2.4 | Dispatch threshold measured |
| R2.5 | Must-pass table v2 §8.1 |
| R2.6 | Ablation waterfall ≥5 steps |

---

### Phase 4 — R3 + upstream

| ID | Task |
|----|------|
| R3.1 | Triton twin + LOC/time/perf accounting |
| R3.2 | Optional CuTe-DSL |
| R3.3 | Grouped/MoE path if demand validated |
| R3.4 | sm_86 portability chapter |
| R3.5 | Upstream PR (start: alignment / MXFP4 pain) |
| R3.6 | Final REPORT + reproduction package |

---

## 6. Frozen technical decisions (already decided — do not reopen)

| Decision | Choice |
|----------|--------|
| Primary formats | MXFP4, NVFP4, INT4-g128 (symmetric) |
| Activations | FP16/BF16 |
| Accumulation | **FP32** |
| Layout convention | Y = X @ Wᵀ; X[M,K], W[N,K], Y[M,N] row-major |
| act_order / desc_act | **Out of scope v1** |
| Asymmetric zeros | **Out of scope v1** |
| Authoritative arch | sm_90 |
| Full engine | **Out of scope** |
| Blackwell native FP4 as target | **Out of scope** (contrast only) |
| Success vs naive dequant | **Sanity only, never headline** |

---

## 7. Success criteria (must-pass) — from v2 §8.1

Summarized; full numbers in `PROJECT-v2.0-fp4-hopper.md`:

- Structured probes **exact**  
- Random vs FP32 dequant ref ≤ √K · eps; error map unstructured  
- Fused vs dequant-then-cuBLAS ≤ 2 ulp (when fused exists)  
- M=1 sm_90: ≥ **60%** ideal BW; MXFP4 vs FP8 ≥ **1.5×**; vs FP16 ≥ **2.5×**  
- Gap to Machete/Marlin within factors in brief  
- K=N=2880 correct, no cliff  
- Crossover M on both arches  
- Gap decomposition with profiler evidence  
- Ablation ≥5 steps  
- Predictions preregistered  
- Third-party repro  

---

## 8. Required figures (experiment design)

1. Latency vs M (log-x), all baselines, both arches  
2. % ideal BW vs M, per format  
3. Gap decomposition waterfall  
4. Roofline with OI points  
5. Speed/accuracy frontier (MX vs NV vs INT4)  
+ Nsight table per variant  

---

## 9. Parallel workstreams (for multi-agent / multi-human)

| Stream | Focus | Can start now? |
|--------|--------|----------------|
| **A · Docs/governance** | STATUS, ROADMAP, GATES, LICENSE, PROTOCOL | **Yes** |
| **B · Host numerics** | pack.hpp, formats tests, probes CPU | **Yes** |
| **C · Harness extension** | stream ideal bench, shapes.hpp, Makefile | **Yes** |
| **D · Analysis** | Python CSV/plot pipeline | **Yes** |
| **E · CI** | cpu-ci.yml, test targets | **Yes** |
| **F · DGX R0** | probe sm_90, ncu, exclusive runs | **Needs user/machine** |
| **G · Public baselines** | Marlin/Machete/vLLM under protocol | **Needs deps + ideally DGX** |
| **H · D1 layout study** | Read Marlin/Machete; draft derivation (no copy) | **Yes (study)** |
| **I · Upstream** | Demand validation, issue notes | **Needs user social** |

**Rule:** Streams A–E and H run in parallel on laptop. F/G/I blocked on external factors — track in GATES.

---

## 10. Commit & GitHub policy

1. **Atomic commits:** one concern (e.g. `pack: MXFP4 host roundtrip tests`).  
2. **Message style:** match repo history (`area: description`).  
3. **Never commit:** `.venv/`, `build/`, secrets, huge checkpoints.  
4. **Always commit:** raw CSVs that support claims, env captures, plot scripts.  
5. **Push:** after each phase slice when `main` is not force-diverged; reconcile first.  
6. **PREREGISTRATION:** never rewrite predictions; append outcomes.  
7. **No force-push to main** without explicit user approval.

---

## 11. Definition of “peak” (quality bar)

A deliverable is not done if:

- [ ] It lacks a stated acceptance test  
- [ ] It uses theoretical BW as denominator  
- [ ] It omits env capture for a published number  
- [ ] It claims sm_86 occupancy findings as general  
- [ ] Correctness is only random tolerance without probes  
- [ ] Layout is “copied from Marlin” without derivation note  
- [ ] Scope creep reopened §4.3 non-goals  

---

## 12. Immediate execution order (this session and next)

### This session (plan + implement offline R0)

1. Commit this master plan + STATUS + ROADMAP + GATES + SHAPES + LICENSE.  
2. Reconcile git with origin.  
3. Parallel implement:  
   - B: pack + tests  
   - C: shapes + stream bench  
   - D: analyze_csv  
   - E: cpu CI  
4. Push commits.  

### Blocked on user (ask when needed)

| Need | Why |
|------|-----|
| DGX H100 session | sm_90 denominator + R0/R1 truth |
| ncu counter permission (Win + DGX) | Profiler evidence |
| Upstream “is this wanted?” ping | D8 viability |
| Force-push approval | Only if history rewrite required |

---

## 13. Risk register (active)

| Risk | Status | Action |
|------|--------|--------|
| ncu ERR_NVGPUCTRPERM | Unknown until check | `check_ncu_gate.sh` |
| sm_90 unmeasured | Open | User DGX |
| Git diverge local/origin | Open | Reconcile |
| Scope creep with DGX access | Controlled | Frozen non-goals |
| Public baseline build complexity | Open | Scaffold now, full on DGX |

---

## 14. Changelog of this plan

| Date | Change |
|------|--------|
| 2026-08-13 | Initial peak plan from live repo + v2.0 + review |

---

*End of master execution plan. Living document — update when rungs complete; do not silently shrink quality bars.*
