# Wave quantization / CTA occupancy (paper arithmetic)

SM counts: laptop **20**, H100 SXM **132** (standard; confirm with probe on each machine).  
Two CTA N-tiles: **256** = fused-ref / 4× `m64` atoms; **64** = one first-opcode atom (`m64n8k16`). The fragment does not force CTA size — we choose.

## Tile N = 256 (4 atoms / CTA)

| layer | N | CTAs | occ sm_86 | occ sm_90 | starved H100? | split-K to ~2×132 SMs |
|-------|--:|-----:|----------:|----------:|:-------------:|----------------------:|
| q_o_proj | 4096 | 16 | 0.800 | 0.121 | yes | 17 |
| kv_gqa | 1024 | 4 | 0.200 | 0.030 | yes | 66 |
| gate_up | 14336 | 56 | 2.800 | 0.424 | yes | 5 |
| down_proj | 4096 | 16 | 0.800 | 0.121 | yes | 17 |
| gpt_oss_2880 | 2880 | 12 | 0.600 | 0.091 | yes | 22 |
| fused_qkv | 6144 | 24 | 1.200 | 0.182 | yes | 11 |

## Tile N = 64 (1 atom / CTA)

| layer | N | CTAs | occ sm_86 | occ sm_90 | starved H100? | split-K to ~2×132 SMs |
|-------|--:|-----:|----------:|----------:|:-------------:|----------------------:|
| q_o_proj | 4096 | 64 | 3.200 | 0.485 | yes | 5 |
| kv_gqa | 1024 | 16 | 0.800 | 0.121 | yes | 17 |
| gate_up | 14336 | 224 | 11.200 | 1.697 | no | 2 |
| down_proj | 4096 | 64 | 3.200 | 0.485 | yes | 5 |
| gpt_oss_2880 | 2880 | 45 | 2.250 | 0.341 | yes | 6 |
| fused_qkv | 6144 | 96 | 4.800 | 0.727 | no | 3 |

**Finding (by construction):** every frozen decode-like N-partition **starves H100** at tile 256. At the real atom tile, `q_o_proj` is still under half an H100 (0.485) and `kv_gqa` is still ~12%. Split-K stays mandatory on sm_90. The laptop **cannot** show this except for `kv_gqa` at tile 256. This is why sm_90 is authoritative.

Reproduce: `make test-cpu` (prints both tables) or `python3 python/occupancy_table.py`.
