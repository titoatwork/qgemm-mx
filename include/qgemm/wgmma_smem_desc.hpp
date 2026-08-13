// wgmma_smem_desc.hpp — official 64-bit SMEM matrix descriptor (host).
//
// This is what wgmma reads for operand B (always SMEM) and for A in SS mode.
// Our path is RS: A in registers (a_fragment.hpp), B in SMEM via TMA.
// TMA (cuTensorMapEncodeTiled) is a *different* 128-byte object; we do not
// emit it here. TMA's job is to land a tile that this descriptor can name.
//
// Source: PTX ISA 8.5 / 9.3
//   "Asynchronous Warpgroup Level Matrix Shared Memory Layout"
//   "Matrix Descriptor"
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//     #asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
// Packing matches CUTLASS cute GMMA::Descriptor (same ISA figure).
//
// We do not invent a swizzled bank layout. encode() takes LBO/SBO the
// caller computed. One documented example: packed MN-major 16×8 f16 B tile.

#pragma once

#include <cstdint>
#include <stdexcept>

namespace qgemm {
namespace wgmma {

// First-opcode B tile (issued problem: B = Xᵀ, wgmma-N = logical M padded to 8).
inline constexpr int kBTileK = 16;
inline constexpr int kBTileN = 8;
inline constexpr int kBTileBytes = kBTileK * kBTileN * 2;  // f16

enum class SmemSwizzle : unsigned {
  None = 0,  // no swizzle
  B32 = 1,   // 32-byte
  B64 = 2,   // 64-byte
  B128 = 3   // 128-byte
};

struct SmemDescFields {
  std::uint32_t start = 0;       // SMEM byte address, 16-aligned
  std::uint32_t leading = 0;     // LBO bytes, 16-aligned (adjacent core in K)
  std::uint32_t stride = 0;      // SBO bytes, 16-aligned (adjacent core in N)
  unsigned base_offset = 0;      // 0..7
  SmemSwizzle swizzle = SmemSwizzle::None;
};

inline void require_desc_align(std::uint32_t bytes, const char* name) {
  if ((bytes & 0xFu) != 0) {
    throw std::invalid_argument(name);
  }
}

// Pack the 64-bit descriptor. Addresses are encoded as byte_offset >> 4
// in 14-bit fields (SMEM window 2^18 = 256 KiB, Hopper max).
inline std::uint64_t encode_smem_desc(SmemDescFields f) {
  require_desc_align(f.start, "smem desc: start must be 16-byte aligned");
  require_desc_align(f.leading, "smem desc: LBO must be 16-byte aligned");
  require_desc_align(f.stride, "smem desc: SBO must be 16-byte aligned");
  if (f.base_offset > 7u) {
    throw std::invalid_argument("smem desc: base_offset in 0..7");
  }
  const std::uint64_t start14 = (static_cast<std::uint64_t>(f.start) >> 4) & 0x3FFFull;
  const std::uint64_t lbo14 = (static_cast<std::uint64_t>(f.leading) >> 4) & 0x3FFFull;
  const std::uint64_t sbo14 = (static_cast<std::uint64_t>(f.stride) >> 4) & 0x3FFFull;
  const std::uint64_t base = static_cast<std::uint64_t>(f.base_offset) & 7ull;
  const std::uint64_t sw = static_cast<std::uint64_t>(
                               static_cast<unsigned>(f.swizzle)) &
                           3ull;
  return start14 | (lbo14 << 16) | (sbo14 << 32) | (base << 49) | (sw << 62);
}

inline SmemDescFields decode_smem_desc(std::uint64_t desc) {
  SmemDescFields f;
  f.start = static_cast<std::uint32_t>((desc & 0x3FFFull) << 4);
  f.leading = static_cast<std::uint32_t>(((desc >> 16) & 0x3FFFull) << 4);
  f.stride = static_cast<std::uint32_t>(((desc >> 32) & 0x3FFFull) << 4);
  f.base_offset = static_cast<unsigned>((desc >> 49) & 7ull);
  f.swizzle = static_cast<SmemSwizzle>(static_cast<unsigned>((desc >> 62) & 3ull));
  return f;
}

// Packed MN-major B tile: row-major f16 B[k][n], n contiguous, no swizzle.
// Two 8×8-element cores stacked along K (PTX core = 8 strided × 16B contig).
//   core 0 at byte 0  (k = 0..7)
//   core 1 at byte 128 (k = 8..15)  → LBO = 128
//   only one core along N=8          → SBO = 16 (one row of 8 f16)
// This is a legal *dense* layout. Production kernels usually swizzle; that
// is a later choice and does not change encode/decode.
inline SmemDescFields b_tile_mn_major_n8(std::uint32_t smem_start = 0) {
  SmemDescFields f;
  f.start = smem_start;
  f.leading = 128;
  f.stride = 16;
  f.base_offset = 0;
  f.swizzle = SmemSwizzle::None;
  return f;
}

// What a future cuTensorMapEncodeTiled for B = Xᵀ would be asked to copy.
// X is logical [M, K] row-major; issued B is [K, M]. Box is one atom.
struct TmaBBox {
  int rank = 2;
  int box_k = kBTileK;   // 16
  int box_n = kBTileN;   // 8  (logical M padded)
  int elem_bytes = 2;    // f16 after host/device convert of X
  int tile_bytes = kBTileBytes;
};

inline constexpr TmaBBox kTmaBBoxFirstOpcode{};

}  // namespace wgmma
}  // namespace qgemm
