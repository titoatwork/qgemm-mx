// a_fragment.hpp — official wgmma A-fragment map + host pre-shuffle.
//
// First opcode (decode-first):
//   wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
//
// A is 64×16 f16 for every m64nNk16. N only changes C/B. We pick n=8 because
// the issued problem is Yᵀ = W @ Xᵀ and decode has M=1 → wgmma-N = 1, padded
// to the ISA minimum of 8. Larger n wastes C registers; it does not change A.
//
// Source (transcribed, not invented):
//   PTX ISA 8.5 (CUDA 12.6, this repo's toolkit) and PTX ISA 9.3 (web)
//   § "Matrix Fragments for wgmma.mma_async.m64nNk16"
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//   Archive: https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html
//
// Per-warp (a0..a7) map is the same text as
//   § "Matrix Fragments for mma.m16n8k16 with .f16 and .bf16 types"
// Warp w of the 128-thread warpgroup owns A rows [16w, 16w+16).
//
// This is NOT the even/odd 8-wide K word. That packs k0..k7 of ONE row.
// A-fragment owns 2 rows × 4 K-cols. See a_fragment_coord().
//
// We do not emit wgmma here. This header is the offline permute + the table
// a future kernel must load against.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"

namespace qgemm {
namespace wgmma {

inline constexpr int kAtomM = 64;
inline constexpr int kAtomN = 8;
inline constexpr int kAtomK = 16;
inline constexpr int kWarpgroupThreads = 128;
inline constexpr int kWarpsPerGroup = 4;
inline constexpr int kElemsAPerThread = 8;  // (64*16)/128
inline constexpr int kElemsCPerThreadN8 = 4;  // (64*8)/128
inline constexpr const char* kFirstOpcode =
    "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16";
inline constexpr const char* kPtxIsaVersion = "8.5 / 9.3";
inline constexpr const char* kPtxIsaSectionA =
    "Matrix Fragments for wgmma.mma_async.m64nNk16";

struct FragCoord {
  int row = 0;
  int col = 0;
};

struct FragOwner {
  int tid = 0;  // 0..127 in the warpgroup
  int i = 0;    // 0..7  (a0..a7), or 0..3 for C
};

// PTX (verbatim structure):
//   groupID           = %laneid >> 2
//   threadID_in_group = %laneid % 4
//   row = groupID           for ai where 0 <= i < 2 || 4 <= i < 6
//       = groupID + 8       otherwise
//   col = (threadID_in_group * 2) + (i & 0x1)        for ai where i <  4
//       = (threadID_in_group * 2) + (i & 0x1) + 8    for ai where i >= 4
// Then row += 16 * warp_id.  // four warps cover m64
inline FragCoord a_fragment_coord(int tid, int i) {
  if (tid < 0 || tid >= kWarpgroupThreads || i < 0 || i >= kElemsAPerThread) {
    throw std::invalid_argument("a_fragment_coord: tid in [0,128), i in [0,8)");
  }
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int groupID = lane >> 2;
  const int tg = lane & 3;
  const int row_in_warp =
      (i < 2 || (i >= 4 && i < 6)) ? groupID : (groupID + 8);
  const int col = (tg * 2) + (i & 1) + (i >= 4 ? 8 : 0);
  return FragCoord{warp * 16 + row_in_warp, col};
}

// Inverse of a_fragment_coord on the 64×16 tile.
inline FragOwner a_fragment_owner(int row, int col) {
  if (row < 0 || row >= kAtomM || col < 0 || col >= kAtomK) {
    throw std::invalid_argument("a_fragment_owner: row in [0,64), col in [0,16)");
  }
  const int warp = row / 16;
  const int row_in_warp = row % 16;
  const int groupID = (row_in_warp < 8) ? row_in_warp : (row_in_warp - 8);
  const int tg = (col < 8) ? (col / 2) : ((col - 8) / 2);
  int i = 0;
  if (col < 8) {
    i = (row_in_warp < 8 ? 0 : 2) + (col & 1);
  } else {
    i = (row_in_warp < 8 ? 4 : 6) + (col & 1);
  }
  return FragOwner{warp * 32 + groupID * 4 + tg, i};
}

// C/D fragment for the first opcode (m64n8, .f32). Same per-warp map as
// mma.m16n8k16 C, stacked on warps. 4 f32 per thread.
//   row = groupID           for di where i <  2
//       = groupID + 8       for di where i >= 2
//   col = (threadID_in_group * 2) + (i & 0x1)
inline FragCoord c_fragment_coord_n8(int tid, int i) {
  if (tid < 0 || tid >= kWarpgroupThreads || i < 0 || i >= kElemsCPerThreadN8) {
    throw std::invalid_argument("c_fragment_coord_n8: tid in [0,128), i in [0,4)");
  }
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int groupID = lane >> 2;
  const int tg = lane & 3;
  const int row_in_warp = (i < 2) ? groupID : (groupID + 8);
  const int col = (tg * 2) + (i & 1);
  return FragCoord{warp * 16 + row_in_warp, col};
}

inline FragOwner c_fragment_owner_n8(int row, int col) {
  if (row < 0 || row >= kAtomM || col < 0 || col >= kAtomN) {
    throw std::invalid_argument("c_fragment_owner_n8: row in [0,64), col in [0,8)");
  }
  const int warp = row / 16;
  const int row_in_warp = row % 16;
  const int groupID = (row_in_warp < 8) ? row_in_warp : (row_in_warp - 8);
  const int tg = col / 2;
  const int i = (row_in_warp < 8 ? 0 : 2) + (col & 1);
  return FragOwner{warp * 32 + groupID * 4 + tg, i};
}

inline void require_a_fragment_shape(std::size_t N, std::size_t K) {
  if (N % static_cast<std::size_t>(kAtomM) != 0) {
    throw std::invalid_argument("a-fragment: N must be a multiple of 64");
  }
  if (K % static_cast<std::size_t>(kAtomK) != 0) {
    throw std::invalid_argument("a-fragment: K must be a multiple of 16");
  }
}

inline std::size_t a_fragment_n_tiles(std::size_t N) {
  return N / static_cast<std::size_t>(kAtomM);
}

inline std::size_t a_fragment_k_tiles(std::size_t K) {
  return K / static_cast<std::size_t>(kAtomK);
}

inline std::size_t a_fragment_word_count(std::size_t N, std::size_t K) {
  require_a_fragment_shape(N, K);
  return a_fragment_n_tiles(N) * a_fragment_k_tiles(K) *
         static_cast<std::size_t>(kWarpgroupThreads);
}

// words layout: [n_tile][k_tile][tid]  — one uint32 = a0..a7 nibbles, i in LSB.
// A thread in a future kernel loads words[nt * k_tiles * 128 + kt * 128 + tid]
// and upconverts nibble i → f16 register ai.
inline std::vector<std::uint32_t> permute_seq_to_a_fragment(
    const std::uint8_t* seq, std::size_t N, std::size_t K) {
  require_a_fragment_shape(N, K);
  const std::size_t n_tiles = a_fragment_n_tiles(N);
  const std::size_t k_tiles = a_fragment_k_tiles(K);
  std::vector<std::uint32_t> words(n_tiles * k_tiles *
                                   static_cast<std::size_t>(kWarpgroupThreads));
  for (std::size_t nt = 0; nt < n_tiles; ++nt) {
    for (std::size_t kt = 0; kt < k_tiles; ++kt) {
      std::uint32_t* tile =
          words.data() + (nt * k_tiles + kt) * static_cast<std::size_t>(kWarpgroupThreads);
      for (int tid = 0; tid < kWarpgroupThreads; ++tid) {
        std::uint32_t w = 0;
        for (int i = 0; i < kElemsAPerThread; ++i) {
          const FragCoord c = a_fragment_coord(tid, i);
          const std::size_t n = nt * static_cast<std::size_t>(kAtomM) +
                                static_cast<std::size_t>(c.row);
          const std::size_t k = kt * static_cast<std::size_t>(kAtomK) +
                                static_cast<std::size_t>(c.col);
          const std::uint8_t nib = unpack_nibble(seq, n * K + k);
          w = u32_insert_nibble(w, i, nib);
        }
        tile[tid] = w;
      }
    }
  }
  return words;
}

inline void permute_a_fragment_to_seq(const std::uint32_t* words, std::uint8_t* seq,
                                      std::size_t N, std::size_t K) {
  require_a_fragment_shape(N, K);
  const std::size_t n_tiles = a_fragment_n_tiles(N);
  const std::size_t k_tiles = a_fragment_k_tiles(K);
  std::fill(seq, seq + (N * K) / 2, static_cast<std::uint8_t>(0));
  for (std::size_t nt = 0; nt < n_tiles; ++nt) {
    for (std::size_t kt = 0; kt < k_tiles; ++kt) {
      const std::uint32_t* tile =
          words + (nt * k_tiles + kt) * static_cast<std::size_t>(kWarpgroupThreads);
      for (int tid = 0; tid < kWarpgroupThreads; ++tid) {
        const std::uint32_t w = tile[tid];
        for (int i = 0; i < kElemsAPerThread; ++i) {
          const FragCoord c = a_fragment_coord(tid, i);
          const std::size_t n = nt * static_cast<std::size_t>(kAtomM) +
                                static_cast<std::size_t>(c.row);
          const std::size_t k = kt * static_cast<std::size_t>(kAtomK) +
                                static_cast<std::size_t>(c.col);
          pack_nibble(seq, n * K + k, nibble_from_u32(w, i));
        }
      }
    }
  }
}

inline std::uint8_t unpack_nibble_a_fragment(const std::uint32_t* words,
                                             std::size_t n, std::size_t k,
                                             std::size_t N, std::size_t K) {
  require_a_fragment_shape(N, K);
  const std::size_t k_tiles = a_fragment_k_tiles(K);
  const std::size_t nt = n / static_cast<std::size_t>(kAtomM);
  const std::size_t kt = k / static_cast<std::size_t>(kAtomK);
  const FragOwner o = a_fragment_owner(static_cast<int>(n % kAtomM),
                                       static_cast<int>(k % kAtomK));
  const std::uint32_t w =
      words[(nt * k_tiles + kt) * static_cast<std::size_t>(kWarpgroupThreads) +
            static_cast<std::size_t>(o.tid)];
  return nibble_from_u32(w, o.i);
}

// First-class A-fragment packed tensors. Scales stay sequential (logical n,k).
struct PackedMxfp4AFrag {
  std::size_t N = 0;
  std::size_t K = 0;
  std::vector<std::uint32_t> words;  // (N/64)*(K/16)*128
  std::vector<std::uint8_t> scales;  // same as PackedMxfp4
};

struct PackedNvfp4AFrag {
  std::size_t N = 0;
  std::size_t K = 0;
  std::vector<std::uint32_t> words;
  std::vector<std::uint8_t> scales;
  float tensor_scale = 1.0f;
};

struct PackedInt4AFrag {
  std::size_t N = 0;
  std::size_t K = 0;
  int group = kInt4Group128;
  std::vector<std::uint32_t> words;
  std::vector<std::uint16_t> scales;
};

inline PackedMxfp4AFrag pack_mxfp4_a_fragment(const float* W, std::size_t N,
                                              std::size_t K) {
  require_a_fragment_shape(N, K);
  PackedMxfp4 seq = pack_mxfp4(W, N, K);
  PackedMxfp4AFrag out;
  out.N = seq.N;
  out.K = seq.K;
  out.scales = std::move(seq.scales);
  out.words = permute_seq_to_a_fragment(seq.data.data(), N, K);
  return out;
}

inline PackedNvfp4AFrag pack_nvfp4_a_fragment(const float* W, std::size_t N,
                                              std::size_t K,
                                              float tensor_scale = 1.0f) {
  require_a_fragment_shape(N, K);
  PackedNvfp4 seq = pack_nvfp4(W, N, K, tensor_scale);
  PackedNvfp4AFrag out;
  out.N = seq.N;
  out.K = seq.K;
  out.tensor_scale = seq.tensor_scale;
  out.scales = std::move(seq.scales);
  out.words = permute_seq_to_a_fragment(seq.data.data(), N, K);
  return out;
}

inline PackedInt4AFrag pack_int4_a_fragment(const float* W, std::size_t N,
                                            std::size_t K,
                                            int group = kInt4Group128) {
  require_a_fragment_shape(N, K);
  PackedInt4 seq = pack_int4(W, N, K, group);
  PackedInt4AFrag out;
  out.N = seq.N;
  out.K = seq.K;
  out.group = seq.group;
  out.scales = std::move(seq.scales);
  out.words = permute_seq_to_a_fragment(seq.data.data(), N, K);
  return out;
}

}  // namespace wgmma
}  // namespace qgemm
