// shapes.hpp — frozen GEMM shapes and M-sweep (host, no CUDA).
//
// Source of truth: docs/SHAPES.md (frozen 2026-08-13 from PROJECT-v2.0 §9).
// Do not invent shapes here; amend SHAPES.md with a dated entry first.

#pragma once

#include <cstddef>
#include <cstdint>

namespace qgemm {

// ---------------------------------------------------------------------------
// Layout convention (matches docs/SHAPES.md)
//
//   X  activations  row-major [M, K]
//   W  weights      row-major [N, K]   (stored for Y = X @ Wᵀ)
//   Y  output       row-major [M, N]
// ---------------------------------------------------------------------------

struct LayerShape {
  const char* name;
  std::size_t N;  // output features / rows of W
  std::size_t K;  // reduction / cols of W and X
};

// Layer archetypes. Order matches SHAPES.md.
inline constexpr LayerShape kLayerShapes[] = {
    {"q_o_proj", 4096, 4096},      // square canonical
    {"kv_gqa", 1024, 4096},        // small N — tails / occupancy
    {"gate_up", 14336, 4096},      // wide N — friendly
    {"down_proj", 4096, 14336},    // deep K — split-K stress
    {"gpt_oss_2880", 2880, 2880},  // not 128-aligned — live bug class
    {"fused_qkv", 6144, 4096},     // end-to-end linear check
};

inline constexpr std::size_t kNumLayerShapes =
    sizeof(kLayerShapes) / sizeof(kLayerShapes[0]);

// Full M-sweep. Crossover M is a primary result; do not subsample for headlines.
inline constexpr std::size_t kMSweep[] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
};

inline constexpr std::size_t kNumMSweep = sizeof(kMSweep) / sizeof(kMSweep[0]);

// Tensor-parallel N shrink (shape generator only — not a multi-GPU runtime).
inline constexpr int kTPFactors[] = {1, 2, 4, 8};
inline constexpr std::size_t kNumTPFactors =
    sizeof(kTPFactors) / sizeof(kTPFactors[0]);

// Per-GPU N after TP split. N must be divisible by tp for clean shards.
inline constexpr std::size_t per_gpu_N(std::size_t N, int tp) {
  return N / static_cast<std::size_t>(tp);
}

// Convenience: total problem for one layer × one M.
struct GemmProblem {
  std::size_t M = 0;
  std::size_t N = 0;
  std::size_t K = 0;
  const char* layer_name = nullptr;
};

inline constexpr GemmProblem make_problem(const LayerShape& layer,
                                          std::size_t M) {
  return GemmProblem{M, layer.N, layer.K, layer.name};
}

// Look up a layer by name; returns nullptr if unknown.
inline const LayerShape* find_layer(const char* name) {
  if (name == nullptr) return nullptr;
  for (std::size_t i = 0; i < kNumLayerShapes; ++i) {
    const char* a = kLayerShapes[i].name;
    const char* b = name;
    while (*a && *b && *a == *b) {
      ++a;
      ++b;
    }
    if (*a == '\0' && *b == '\0') return &kLayerShapes[i];
  }
  return nullptr;
}

// Block / group sizes frozen with the formats (see formats.cuh / SHAPES.md).
inline constexpr int kInt4GroupPrimary = 128;
inline constexpr int kInt4GroupStress = 64;
inline constexpr int kMxfp4Block = 32;
inline constexpr int kNvfp4Block = 16;

}  // namespace qgemm
