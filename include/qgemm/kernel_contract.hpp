// kernel_contract.hpp — frozen math ↔ Hopper datapath mapping.
//
// This is the written law from docs/RESEARCH_ANALYSIS.md. R2 kernels that
// violate it are bugs even if they are fast on a laptop.

#pragma once

#include <cstddef>

#include "qgemm/occupancy.hpp"
#include "qgemm/shapes.hpp"

// First Hopper opcode (decode-first). A-fragment table lives in a_fragment.hpp.
//   wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
// A is 64×16 for every n; n=8 is the ISA minimum and matches decode M=1.

namespace qgemm {
namespace contract {

// Logical problem (Transformer linear, matches shapes.hpp / benches):
//   X : row-major [M, K]
//   W : row-major [N, K]
//   Y : row-major [M, N]
//   Y = X @ Wᵀ
//
// Hopper wgmma: operand B is SMEM-only; A,C are registers.
// 4-bit weights are upconverted in registers → they MUST be wgmma A.
// Therefore the issued MMA is the TRANSPOSE problem:
//   Yᵀ = W @ Xᵀ
//   A ← upconverted W  (registers)   shape [N, K] as left operand
//   B ← X               (SMEM via TMA)
//   C ← Yᵀ              (registers) then store as Y
//
// Ampere mma.sync can keep a more Marlin-like B-in-registers story.
// sm_90 authoritative kernels MUST use the transpose mapping above.

enum class Datapath {
  RefScalar,   // qgemv_ref — no MMA
  AmpereMma,   // sm_86 optional
  HopperWgmma  // sm_90 required
};

struct LogicalGemm {
  std::size_t M = 0, N = 0, K = 0;
};

// Dimensions as seen by a wgmma that implements the transpose mapping.
struct WgmmaView {
  std::size_t A_rows = 0;  // N  (weight rows)
  std::size_t A_cols = 0;  // K
  std::size_t B_rows = 0;  // K
  std::size_t B_cols = 0;  // M
  std::size_t C_rows = 0;  // N
  std::size_t C_cols = 0;  // M
};

inline constexpr WgmmaView wgmma_view(LogicalGemm g) {
  return WgmmaView{g.N, g.K, g.K, g.M, g.N, g.M};
}

// Decode CTA plan: N-tiles × split-K. Default tile 256 (4× first-opcode atoms).
// Pass kTileNWgmmaAtom (64) for one-atom CTAs.
struct DecodePlan {
  std::size_t tile_n = 256;
  int n_ctas = 0;
  int split_k = 1;
  int total_ctas = 0;
};

inline DecodePlan plan_decode(std::size_t N, int sm_count,
                              std::size_t tile_n = 256,
                              double oversub = 2.0) {
  DecodePlan p;
  p.tile_n = tile_n;
  p.n_ctas = ctas_n_partition(N, tile_n);
  p.split_k = splitk_to_fill(N, sm_count, tile_n, oversub);
  p.total_ctas = p.n_ctas * p.split_k;
  return p;
}

// Group sizes that the inner K loop must respect (cannot split a scale block).
inline constexpr int kMxBlock = 32;
inline constexpr int kNvBlock = 16;
inline constexpr int kInt4Group = 128;
inline constexpr int kWgmmaAtomK = 16;  // first opcode k16

// MX: one E8M0 covers 32 along K = two k16 atoms. Those two atoms MUST
// share the same scale. NV: one E4M3 per k16 (aligned, one atom).
inline constexpr int kMxAtomsPerScale = kMxBlock / kWgmmaAtomK;  // 2
inline constexpr int kNvAtomsPerScale = kNvBlock / kWgmmaAtomK;  // 1

inline constexpr bool k_multiple_ok(std::size_t K, int group) {
  return group > 0 && (K % static_cast<std::size_t>(group)) == 0;
}

inline constexpr int mx_scale_block(int k) { return k / kMxBlock; }

inline constexpr int mx_scale_block_from_ktile(int k_tile) {
  return k_tile / kMxAtomsPerScale;
}

inline constexpr bool ktiles_share_mx_scale(int kt_a, int kt_b) {
  return mx_scale_block_from_ktile(kt_a) == mx_scale_block_from_ktile(kt_b);
}

// Split-K / K-range must land on scale-group edges (MX: 32).
inline constexpr bool k_range_respects_group(int k0, int k1, int group) {
  return group > 0 && (k0 % group) == 0 && (k1 % group) == 0 && k1 >= k0;
}

}  // namespace contract
}  // namespace qgemm
