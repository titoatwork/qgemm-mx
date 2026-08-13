# Frozen shape set

**Frozen at:** 2026-08-13 (execution plan sign-off)  
**Source:** `PROJECT-v2.0-fp4-hopper.md` §9  
**Do not edit lightly.** New shapes require a dated amendment below.

## Layout convention

- Activations `X`: row-major `[M, K]`  
- Weights `W`: row-major `[N, K]` (as stored for `Y = X @ Wᵀ`)  
- Output `Y`: row-major `[M, N]`  

## Layer archetypes (N × K)

| Name | N | K | Why |
|------|--:|--:|-----|
| `q_o_proj` | 4096 | 4096 | Square canonical |
| `kv_gqa` | 1024 | 4096 | Small N — tails / occupancy |
| `gate_up` | 14336 | 4096 | Wide N — friendly |
| `down_proj` | 4096 | 14336 | Deep K — split-K stress |
| `gpt_oss_2880` | 2880 | 2880 | **Not 128-aligned** — live bug class |
| `fused_qkv` | 6144 | 4096 | End-to-end linear check |

## M sweep

```
M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
```

Full sweep required: **crossover M is a primary result.**

## Formats / groups

| Format | Group / block | Notes |
|--------|---------------|--------|
| MXFP4 | 32 | E2M1 + E8M0 |
| NVFP4 | 16 | E2M1 + E4M3 + tensor scale |
| INT4 | 128 primary; 64 stress | Symmetric scales only |

## Tensor parallel (shape generator only)

`TP ∈ {1, 2, 4, 8}` — per-GPU N shrinks (e.g. 4096/TP). Not building a multi-GPU runtime.

## Out of scope shapes

- `act_order` / permuted-K GPTQ  
- Asymmetric zero-point layouts  
- Training batch megashapes as primary  

## Amendments

| Date | Change | Reason |
|------|--------|--------|
| — | — | — |
