# Wave quantization / CTA occupancy (paper arithmetic)

**Tile N = 256**, one CTA per N-tile, **no split-K**.  
SM counts: laptop **20**, H100 SXM **132** (standard; confirm with probe on each machine).

| layer | N | CTAs | occ sm_86 | occ sm_90 | starved H100? | split-K to ~2×132 SMs |
|-------|--:|-----:|----------:|----------:|:-------------:|----------------------:|
| q_o_proj | 4096 | 16 | 0.800 | 0.121 | yes | 17 |
| kv_gqa | 1024 | 4 | 0.200 | 0.030 | yes | 66 |
| gate_up | 14336 | 56 | 2.800 | 0.424 | yes | 5 |
| down_proj | 4096 | 16 | 0.800 | 0.121 | yes | 17 |
| gpt_oss_2880 | 2880 | 12 | 0.600 | 0.091 | yes | 22 |
| fused_qkv | 6144 | 24 | 1.200 | 0.182 | yes | 11 |

**Finding (by construction):** every frozen decode-like N-partition **starves H100** without split-K. The laptop **cannot** show this except for `kv_gqa`. This is why sm_90 is authoritative.

Reproduce: `make test-cpu` (prints the table) or `python3 python/occupancy_table.py`.
