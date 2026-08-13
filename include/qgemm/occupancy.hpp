// occupancy.hpp — wave-quantization / CTA-count arithmetic (host).
//
// At M=1 a typical N-tile of 256 on N=4096 launches 16 CTAs. That nearly fills
// a 20-SM laptop and starves a 132-SM H100 (~12%). Split-K is the decode
// bottleneck on Hopper; this header makes the arithmetic a testable artifact
// rather than a slide-deck claim.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "qgemm/shapes.hpp"

namespace qgemm {

struct OccupancyEstimate {
  std::size_t N = 0;
  std::size_t tile_n = 256;
  int sm_count = 0;
  int ctas = 0;
  int waves = 0;           // ceil(ctas / sm_count) if 1 CTA/SM
  double occupancy = 0.0;  // ctas / sm_count (can be >1)
  bool starved = false;    // occupancy < 0.5 → machine mostly idle
};

// CTA count if we partition only N (no split-K).
inline int ctas_n_partition(std::size_t N, std::size_t tile_n) {
  if (tile_n == 0) return 0;
  return static_cast<int>((N + tile_n - 1) / tile_n);
}

inline OccupancyEstimate estimate_occupancy(std::size_t N, int sm_count,
                                            std::size_t tile_n = 256) {
  OccupancyEstimate e;
  e.N = N;
  e.tile_n = tile_n;
  e.sm_count = sm_count;
  e.ctas = ctas_n_partition(N, tile_n);
  e.occupancy =
      (sm_count > 0) ? static_cast<double>(e.ctas) / static_cast<double>(sm_count)
                     : 0.0;
  e.waves = (sm_count > 0) ? (e.ctas + sm_count - 1) / sm_count : 0;
  e.starved = e.occupancy < 0.5;
  return e;
}

// Split-K factor so CTA count is at least `target_ctas` (e.g. 2× SMs).
inline int splitk_to_fill(std::size_t N, int sm_count, std::size_t tile_n = 256,
                          double oversub = 2.0) {
  const int base = ctas_n_partition(N, tile_n);
  if (base <= 0 || sm_count <= 0) return 1;
  const int want = static_cast<int>(
      std::ceil(oversub * static_cast<double>(sm_count)));
  const int sk = (want + base - 1) / base;
  return std::max(sk, 1);
}

// Standard SM counts used in docs (not a substitute for cudaGetDeviceProperties).
inline constexpr int kSmCountLaptop3050 = 20;
inline constexpr int kSmCountH100SXM = 132;

}  // namespace qgemm
