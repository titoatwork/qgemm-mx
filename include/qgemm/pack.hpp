// pack.hpp — host-side pack / unpack / reference dequant for weight formats.
//
// Pure C++17. No CUDA dependency. Safe to compile with g++ for CPU tests.
//
// Scope (R0): sequential host layouts for correctness and byte accounting.
// Kernel-optimal nibble permutations (Marlin-style) are deferred to R2 and
// must be derived from the datapath (docs/D1_LAYOUT_DERIVATION.md), not invented
// here.
//
// ---------------------------------------------------------------------------
// Storage conventions (all formats)
// ---------------------------------------------------------------------------
//
// Weights conceptually row-major W[N, K]. Elements are packed along K.
// Two 4-bit values share one byte:
//
//     byte b holds elements 2i and 2i+1 of a row:
//       low  nibble  = element 2i      (bits [3:0])
//       high nibble  = element 2i+1    (bits [7:4])
//
// Scale buffers are stored separately (not interleaved). Indexing:
//
//     scale_index(n, k, group) = n * (K / group) + (k / group)
//
// K must be divisible by the format's group/block size for the APIs below.
//
// ---------------------------------------------------------------------------
// INT4_G{64,128}  (symmetric, GPTQ/AWQ-style)
// ---------------------------------------------------------------------------
//
// Element: signed 4-bit two's complement in the nibble, range [-8, 7].
// Scale:   one IEEE FP16 value per group, stored as its raw uint16 bits.
//          Dequant:  w_f32 = int4_value * fp16_to_f32(scale)
// Zero points: out of scope (symmetric only).
//
// ---------------------------------------------------------------------------
// MXFP4  (OCP Microscaling, block = 32)
// ---------------------------------------------------------------------------
//
// Element: E2M1 (1 sign, 2 exp, 1 mantissa). Bit layout in the nibble:
//
//     [3] sign | [2:1] exponent | [0] mantissa
//
// Bias = 1. No Inf/NaN encodings (all 16 codes are finite):
//
//     code | bits  | value
//     -----+-------+-------
//      0x0 | 0 00 0|  0.0
//      0x1 | 0 00 1|  0.5     (subnormal)
//      0x2 | 0 01 0|  1.0
//      0x3 | 0 01 1|  1.5
//      0x4 | 0 10 0|  2.0
//      0x5 | 0 10 1|  3.0
//      0x6 | 0 11 0|  4.0
//      0x7 | 0 11 1|  6.0
//      0x8..0xF    | negatives of 0x0..0x7
//
// Block scale: E8M0 — unsigned 8-bit power-of-two exponent (OCP MX):
//
//     if bits == 0xFF → NaN (invalid scale; dequant yields quiet NaN)
//     else            → 2^(bits - 127)
//
// Dequant:  w_f32 = e2m1_decode(elem) * e8m0_decode(scale)
//
// This is a faithful OCP-aligned reference encoding for roundtrip and
// correctness probes. Device pack layout (interleave, bank mapping) is R2.
//
// ---------------------------------------------------------------------------
// NVFP4  (E2M1 + E4M3 per 16 + optional tensor scale)
// ---------------------------------------------------------------------------
//
// Element: same E2M1 as MXFP4.
// Block scale: IEEE-like E4M3 (1 sign, 4 exp, 3 mantissa), bias 7.
//   Finite range includes exp=all-ones except the NaN encodings:
//     NaN when exp==15 and mantissa==7 (codes 0x7F, 0xFF).
//   Subnormals: exp==0, value = (-1)^s * 2^(-6) * (mant/8).
//   Normals:    value = (-1)^s * 2^(exp-7) * (1 + mant/8).
//
// Optional tensor scale: one FP32 multiplier for the whole matrix (stored
// separately). Dequant:
//
//     w_f32 = e2m1_decode(elem) * e4m3_decode(block_scale) * tensor_scale
//
// formats.cuh amortizes only the E4M3 block scale into bytes_per_weight
// (0.5 scale bits/elem); the single FP32 tensor scale is negligible.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qgemm {

// ---------------------------------------------------------------------------
// Group / block sizes
// ---------------------------------------------------------------------------

inline constexpr int kInt4Group128 = 128;
inline constexpr int kInt4Group64 = 64;
inline constexpr int kMxfp4BlockSize = 32;
inline constexpr int kNvfp4BlockSize = 16;

// ---------------------------------------------------------------------------
// Size helpers (must match formats.cuh arithmetic)
// ---------------------------------------------------------------------------

inline std::size_t packed_nibbles_bytes(std::size_t N, std::size_t K) {
  if ((N * K) & 1ull) {
    throw std::invalid_argument("packed_nibbles_bytes: N*K must be even");
  }
  return (N * K) / 2;
}

inline std::size_t int4_scale_count(std::size_t N, std::size_t K, int group) {
  if (group <= 0 || K % static_cast<std::size_t>(group) != 0) {
    throw std::invalid_argument("int4_scale_count: K must be divisible by group");
  }
  return N * (K / static_cast<std::size_t>(group));
}

inline std::size_t mxfp4_scale_count(std::size_t N, std::size_t K) {
  if (K % static_cast<std::size_t>(kMxfp4BlockSize) != 0) {
    throw std::invalid_argument("mxfp4_scale_count: K must be divisible by 32");
  }
  return N * (K / static_cast<std::size_t>(kMxfp4BlockSize));
}

inline std::size_t nvfp4_scale_count(std::size_t N, std::size_t K) {
  if (K % static_cast<std::size_t>(kNvfp4BlockSize) != 0) {
    throw std::invalid_argument("nvfp4_scale_count: K must be divisible by 16");
  }
  return N * (K / static_cast<std::size_t>(kNvfp4BlockSize));
}

// Total host bytes for weight payload + scales (tensor scale excluded).
inline std::size_t int4_total_bytes(std::size_t N, std::size_t K, int group) {
  return packed_nibbles_bytes(N, K) +
         int4_scale_count(N, K, group) * sizeof(std::uint16_t);
}

inline std::size_t mxfp4_total_bytes(std::size_t N, std::size_t K) {
  return packed_nibbles_bytes(N, K) + mxfp4_scale_count(N, K);
}

inline std::size_t nvfp4_total_bytes(std::size_t N, std::size_t K) {
  return packed_nibbles_bytes(N, K) + nvfp4_scale_count(N, K);
}

// ---------------------------------------------------------------------------
// Nibble pack helpers
// ---------------------------------------------------------------------------

inline void pack_nibble(std::uint8_t* data, std::size_t linear_index,
                        std::uint8_t nibble) {
  const std::size_t byte_i = linear_index >> 1;
  const std::uint8_t v = static_cast<std::uint8_t>(nibble & 0xF);
  if ((linear_index & 1ull) == 0) {
    data[byte_i] = static_cast<std::uint8_t>((data[byte_i] & 0xF0) | v);
  } else {
    data[byte_i] =
        static_cast<std::uint8_t>((data[byte_i] & 0x0F) | (v << 4));
  }
}

inline std::uint8_t unpack_nibble(const std::uint8_t* data,
                                  std::size_t linear_index) {
  const std::uint8_t b = data[linear_index >> 1];
  return (linear_index & 1ull) == 0
             ? static_cast<std::uint8_t>(b & 0x0F)
             : static_cast<std::uint8_t>((b >> 4) & 0x0F);
}

// ---------------------------------------------------------------------------
// IEEE FP16 <-> FP32 (host, bit-exact enough for scales)
// ---------------------------------------------------------------------------

inline float fp16_bits_to_float(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1u;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t mant = h & 0x3FFu;

  std::uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      // Subnormal half → normal float
      std::uint32_t m = mant;
      std::uint32_t e = 127 - 15 + 1;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FFu;
      out = (sign << 31) | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7F800000u | (mant << 13);
  } else {
    out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

inline std::uint16_t float_to_fp16_bits(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
  std::uint32_t mant = x & 0x7FFFFFu;

  if (((x >> 23) & 0xFFu) == 0xFFu) {
    // Inf / NaN
    std::uint16_t m = mant ? 0x200u : 0;  // quiet-ish NaN payload
    return static_cast<std::uint16_t>(sign | 0x7C00u | m);
  }
  if (exp <= 0) {
    if (exp < -10) return static_cast<std::uint16_t>(sign);  // underflow → 0
    // Subnormal half
    mant |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    std::uint32_t half_m = mant >> shift;
    // Round to nearest even
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (half_m & 1u))) ++half_m;
    return static_cast<std::uint16_t>(sign | half_m);
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);  // overflow → Inf
  }
  // Round mantissa 23 → 10
  std::uint32_t half_m = mant >> 13;
  if ((mant & 0x1FFFu) > 0x1000u ||
      ((mant & 0x1FFFu) == 0x1000u && (half_m & 1u))) {
    ++half_m;
    if (half_m == 0x400u) {
      half_m = 0;
      ++exp;
      if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                   half_m);
}

// ---------------------------------------------------------------------------
// E2M1 (FP4)  — shared by MXFP4 and NVFP4
// ---------------------------------------------------------------------------

// Exact finite set (positive half). Index is the 3-bit magnitude code.
inline constexpr float kE2M1PosTable[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

inline float e2m1_decode(std::uint8_t code) {
  const std::uint8_t c = static_cast<std::uint8_t>(code & 0xF);
  const float mag = kE2M1PosTable[c & 0x7];
  return (c & 0x8) ? -mag : mag;
}

// Nearest E2M1 code. Both +0 and -0 encode to 0x0 (signed zero is not
// distinguished on the way back in; decode of 0x8 still yields -0.0f).
inline std::uint8_t e2m1_encode(float v) {
  if (!std::isfinite(v)) {
    // Map non-finite to max-magnitude with matching sign.
    return (std::signbit(v) ? 0xFu : 0x7u);
  }
  const float a = std::fabs(v);
  int best = 0;
  float best_err = std::fabs(a - kE2M1PosTable[0]);
  for (int i = 1; i < 8; ++i) {
    const float err = std::fabs(a - kE2M1PosTable[i]);
    if (err < best_err) {
      best_err = err;
      best = i;
    }
  }
  std::uint8_t code = static_cast<std::uint8_t>(best);
  // Preserve sign for nonzero values only; zero collapses to 0x0.
  if (best != 0 && std::signbit(v)) {
    code = static_cast<std::uint8_t>(code | 0x8);
  }
  return code;
}

// ---------------------------------------------------------------------------
// E8M0 scale (OCP MX) — pure power of two
// ---------------------------------------------------------------------------

inline float e8m0_decode(std::uint8_t bits) {
  if (bits == 0xFFu) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  // 2^(bits - 127). ldexp is exact for this integer exponent range.
  return std::ldexp(1.0f, static_cast<int>(bits) - 127);
}

// Nearest E8M0 for a positive scale. Non-positive / non-finite → 0 or NaN bits.
inline std::uint8_t e8m0_encode(float scale) {
  if (!std::isfinite(scale) || scale <= 0.0f) {
    if (!std::isfinite(scale)) return 0xFFu;
    return 0;  // 2^(0-127) = 2^-127, tiny; used as "zero-ish"
  }
  // Ideal exponent e such that 2^e ≈ scale → bits = e + 127
  int e = 0;
  std::frexp(scale, &e);  // scale = m * 2^e with m in [0.5, 1)
  // frexp returns e so that value = m * 2^e; nearest power of two:
  // if m >= sqrt(0.5)≈0.707, round up to 2^e, else 2^(e-1)
  const float m = std::ldexp(scale, -e);
  int pow = e;
  if (m < 0.7071067811865476f) {
    pow = e - 1;
  }
  int bits = pow + 127;
  if (bits < 0) bits = 0;
  if (bits > 254) bits = 254;  // 255 is NaN
  return static_cast<std::uint8_t>(bits);
}

// ---------------------------------------------------------------------------
// E4M3 (FP8) — NVFP4 block scale
// ---------------------------------------------------------------------------

inline float e4m3_decode(std::uint8_t bits) {
  const int sign = (bits >> 7) & 1;
  const int exp = (bits >> 3) & 0xF;
  const int mant = bits & 0x7;

  if (exp == 0xF && mant == 0x7) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float mag;
  if (exp == 0) {
    // Subnormal: 2^(-6) * (mant/8)
    mag = std::ldexp(static_cast<float>(mant), -9);  // mant * 2^-9
  } else {
    // Normal: 2^(exp-7) * (1 + mant/8)
    mag = std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  }
  return sign ? -mag : mag;
}

inline std::uint8_t e4m3_encode(float v) {
  if (!std::isfinite(v)) {
    return (std::signbit(v) ? 0xFFu : 0x7Fu);  // NaN
  }
  const float a = std::fabs(v);
  if (a == 0.0f) return (std::signbit(v) ? 0x80u : 0x00u);

  // Brute-force nearest among 128 finite positive codes (fast, exact for tests).
  int best = 0;
  float best_err = std::numeric_limits<float>::infinity();
  for (int code = 0; code < 128; ++code) {
    if (code == 0x7F) continue;  // NaN
    const float cand = e4m3_decode(static_cast<std::uint8_t>(code));
    const float err = std::fabs(a - cand);
    if (err < best_err) {
      best_err = err;
      best = code;
    }
  }
  std::uint8_t out = static_cast<std::uint8_t>(best);
  if (v < 0.0f) out = static_cast<std::uint8_t>(out | 0x80u);
  return out;
}

// ---------------------------------------------------------------------------
// INT4 nibble <-> signed value
// ---------------------------------------------------------------------------

// Two's complement 4-bit: 0x0..0x7 → 0..7, 0x8..0xF → -8..-1
inline int int4_from_nibble(std::uint8_t n) {
  n = static_cast<std::uint8_t>(n & 0xF);
  return (n & 0x8) ? static_cast<int>(n) - 16 : static_cast<int>(n);
}

inline std::uint8_t int4_to_nibble(int v) {
  if (v < -8) v = -8;
  if (v > 7) v = 7;
  return static_cast<std::uint8_t>(v & 0xF);
}

// ---------------------------------------------------------------------------
// Packed tensor containers
// ---------------------------------------------------------------------------

struct PackedInt4 {
  std::size_t N = 0;
  std::size_t K = 0;
  int group = kInt4Group128;
  std::vector<std::uint8_t> data;     // N*K/2 bytes
  std::vector<std::uint16_t> scales;  // N*(K/group) FP16 bits
};

struct PackedMxfp4 {
  std::size_t N = 0;
  std::size_t K = 0;
  std::vector<std::uint8_t> data;    // N*K/2
  std::vector<std::uint8_t> scales;  // N*(K/32) E8M0
};

struct PackedNvfp4 {
  std::size_t N = 0;
  std::size_t K = 0;
  std::vector<std::uint8_t> data;    // N*K/2
  std::vector<std::uint8_t> scales;  // N*(K/16) E4M3
  float tensor_scale = 1.0f;
};

// ---------------------------------------------------------------------------
// Pack from FP32 weights  (quantize + pack)
// ---------------------------------------------------------------------------

// Symmetric INT4: per-group scale = max(|w|) / 7, then round.
inline PackedInt4 pack_int4(const float* W, std::size_t N, std::size_t K,
                            int group = kInt4Group128) {
  if (group != kInt4Group64 && group != kInt4Group128) {
    throw std::invalid_argument("pack_int4: group must be 64 or 128");
  }
  if (K % static_cast<std::size_t>(group) != 0) {
    throw std::invalid_argument("pack_int4: K must be divisible by group");
  }
  PackedInt4 out;
  out.N = N;
  out.K = K;
  out.group = group;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.resize(int4_scale_count(N, K, group));

  const std::size_t groups_per_row = K / static_cast<std::size_t>(group);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t g = 0; g < groups_per_row; ++g) {
      const std::size_t k0 = g * static_cast<std::size_t>(group);
      float amax = 0.0f;
      for (int t = 0; t < group; ++t) {
        amax = std::max(amax, std::fabs(W[n * K + k0 + static_cast<std::size_t>(t)]));
      }
      // Avoid zero scale; map empty group to scale 1.
      const float scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
      out.scales[n * groups_per_row + g] = float_to_fp16_bits(scale);
      const float inv = 1.0f / scale;
      for (int t = 0; t < group; ++t) {
        const float x = W[n * K + k0 + static_cast<std::size_t>(t)];
        int q = static_cast<int>(std::nearbyint(x * inv));
        if (q < -8) q = -8;
        if (q > 7) q = 7;
        const std::size_t lin = n * K + k0 + static_cast<std::size_t>(t);
        pack_nibble(out.data.data(), lin, int4_to_nibble(q));
      }
    }
  }
  return out;
}

// MXFP4: per-block max sets E8M0 so that amax / scale fits in [-6, 6].
inline PackedMxfp4 pack_mxfp4(const float* W, std::size_t N, std::size_t K) {
  if (K % static_cast<std::size_t>(kMxfp4BlockSize) != 0) {
    throw std::invalid_argument("pack_mxfp4: K must be divisible by 32");
  }
  PackedMxfp4 out;
  out.N = N;
  out.K = K;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.resize(mxfp4_scale_count(N, K));

  const std::size_t blocks = K / static_cast<std::size_t>(kMxfp4BlockSize);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::size_t k0 = b * static_cast<std::size_t>(kMxfp4BlockSize);
      float amax = 0.0f;
      for (int t = 0; t < kMxfp4BlockSize; ++t) {
        amax = std::max(amax, std::fabs(W[n * K + k0 + static_cast<std::size_t>(t)]));
      }
      // Want amax / scale <= 6 → scale >= amax/6. Encode that as E8M0.
      const float target = (amax > 0.0f) ? (amax / 6.0f) : e8m0_decode(127);  // 1.0
      const std::uint8_t sb = e8m0_encode(target);
      out.scales[n * blocks + b] = sb;
      const float scale = e8m0_decode(sb);
      const float inv = 1.0f / scale;
      for (int t = 0; t < kMxfp4BlockSize; ++t) {
        const float x = W[n * K + k0 + static_cast<std::size_t>(t)] * inv;
        const std::size_t lin = n * K + k0 + static_cast<std::size_t>(t);
        pack_nibble(out.data.data(), lin, e2m1_encode(x));
      }
    }
  }
  return out;
}

// NVFP4: E4M3 block scale + optional tensor scale applied before encoding.
// tensor_scale is stored and used in dequant; elements are packed relative to
// (block_scale * tensor_scale).
inline PackedNvfp4 pack_nvfp4(const float* W, std::size_t N, std::size_t K,
                              float tensor_scale = 1.0f) {
  if (K % static_cast<std::size_t>(kNvfp4BlockSize) != 0) {
    throw std::invalid_argument("pack_nvfp4: K must be divisible by 16");
  }
  if (!(tensor_scale > 0.0f) || !std::isfinite(tensor_scale)) {
    throw std::invalid_argument("pack_nvfp4: tensor_scale must be finite and > 0");
  }
  PackedNvfp4 out;
  out.N = N;
  out.K = K;
  out.tensor_scale = tensor_scale;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.resize(nvfp4_scale_count(N, K));

  const float inv_tensor = 1.0f / tensor_scale;
  const std::size_t blocks = K / static_cast<std::size_t>(kNvfp4BlockSize);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t b = 0; b < blocks; ++b) {
      const std::size_t k0 = b * static_cast<std::size_t>(kNvfp4BlockSize);
      float amax = 0.0f;
      for (int t = 0; t < kNvfp4BlockSize; ++t) {
        const float w = W[n * K + k0 + static_cast<std::size_t>(t)] * inv_tensor;
        amax = std::max(amax, std::fabs(w));
      }
      const float target = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
      // Prefer positive E4M3 scales (symmetric weights).
      const std::uint8_t sb = e4m3_encode(target);
      out.scales[n * blocks + b] = sb;
      const float scale = e4m3_decode(sb);
      const float inv = (scale != 0.0f && std::isfinite(scale)) ? (1.0f / scale) : 0.0f;
      for (int t = 0; t < kNvfp4BlockSize; ++t) {
        const float x =
            W[n * K + k0 + static_cast<std::size_t>(t)] * inv_tensor * inv;
        const std::size_t lin = n * K + k0 + static_cast<std::size_t>(t);
        pack_nibble(out.data.data(), lin, e2m1_encode(x));
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Reference dequant → FP32  (row-major N*K)
// ---------------------------------------------------------------------------

inline void dequant_int4(const PackedInt4& p, float* out) {
  const std::size_t groups_per_row = p.K / static_cast<std::size_t>(p.group);
  for (std::size_t n = 0; n < p.N; ++n) {
    for (std::size_t k = 0; k < p.K; ++k) {
      const std::size_t g = k / static_cast<std::size_t>(p.group);
      const float scale =
          fp16_bits_to_float(p.scales[n * groups_per_row + g]);
      const int q = int4_from_nibble(unpack_nibble(p.data.data(), n * p.K + k));
      out[n * p.K + k] = static_cast<float>(q) * scale;
    }
  }
}

inline void dequant_mxfp4(const PackedMxfp4& p, float* out) {
  const std::size_t blocks = p.K / static_cast<std::size_t>(kMxfp4BlockSize);
  for (std::size_t n = 0; n < p.N; ++n) {
    for (std::size_t k = 0; k < p.K; ++k) {
      const std::size_t b = k / static_cast<std::size_t>(kMxfp4BlockSize);
      const float scale = e8m0_decode(p.scales[n * blocks + b]);
      const float elem =
          e2m1_decode(unpack_nibble(p.data.data(), n * p.K + k));
      out[n * p.K + k] = elem * scale;
    }
  }
}

inline void dequant_nvfp4(const PackedNvfp4& p, float* out) {
  const std::size_t blocks = p.K / static_cast<std::size_t>(kNvfp4BlockSize);
  for (std::size_t n = 0; n < p.N; ++n) {
    for (std::size_t k = 0; k < p.K; ++k) {
      const std::size_t b = k / static_cast<std::size_t>(kNvfp4BlockSize);
      const float bscale = e4m3_decode(p.scales[n * blocks + b]);
      const float elem =
          e2m1_decode(unpack_nibble(p.data.data(), n * p.K + k));
      out[n * p.K + k] = elem * bscale * p.tensor_scale;
    }
  }
}

// Generic entry points by format name string (for tests / tools).
inline std::vector<float> dequant_reference_int4(const PackedInt4& p) {
  std::vector<float> out(p.N * p.K);
  dequant_int4(p, out.data());
  return out;
}

inline std::vector<float> dequant_reference_mxfp4(const PackedMxfp4& p) {
  std::vector<float> out(p.N * p.K);
  dequant_mxfp4(p, out.data());
  return out;
}

inline std::vector<float> dequant_reference_nvfp4(const PackedNvfp4& p) {
  std::vector<float> out(p.N * p.K);
  dequant_nvfp4(p, out.data());
  return out;
}

// ---------------------------------------------------------------------------
// Low-level pack of already-quantized codes (exact probes / golden paths)
// ---------------------------------------------------------------------------

// q: signed int in [-8,7], length N*K; scales_f32: length N*(K/group)
inline PackedInt4 pack_int4_codes(const int* q, const float* scales_f32,
                                  std::size_t N, std::size_t K, int group) {
  PackedInt4 out;
  out.N = N;
  out.K = K;
  out.group = group;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.resize(int4_scale_count(N, K, group));
  for (std::size_t i = 0; i < out.scales.size(); ++i) {
    out.scales[i] = float_to_fp16_bits(scales_f32[i]);
  }
  for (std::size_t i = 0; i < N * K; ++i) {
    pack_nibble(out.data.data(), i, int4_to_nibble(q[i]));
  }
  return out;
}

// elem_codes: 4-bit E2M1 codes 0..15, length N*K; scale_bits: E8M0, N*(K/32)
inline PackedMxfp4 pack_mxfp4_codes(const std::uint8_t* elem_codes,
                                    const std::uint8_t* scale_bits,
                                    std::size_t N, std::size_t K) {
  PackedMxfp4 out;
  out.N = N;
  out.K = K;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.assign(scale_bits, scale_bits + mxfp4_scale_count(N, K));
  for (std::size_t i = 0; i < N * K; ++i) {
    pack_nibble(out.data.data(), i, elem_codes[i]);
  }
  return out;
}

inline PackedNvfp4 pack_nvfp4_codes(const std::uint8_t* elem_codes,
                                    const std::uint8_t* scale_bits,
                                    std::size_t N, std::size_t K,
                                    float tensor_scale = 1.0f) {
  PackedNvfp4 out;
  out.N = N;
  out.K = K;
  out.tensor_scale = tensor_scale;
  out.data.assign(packed_nibbles_bytes(N, K), 0);
  out.scales.assign(scale_bits, scale_bits + nvfp4_scale_count(N, K));
  for (std::size_t i = 0; i < N * K; ++i) {
    pack_nibble(out.data.data(), i, elem_codes[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Unpack codes only (no dequant)
// ---------------------------------------------------------------------------

inline void unpack_int4_codes(const PackedInt4& p, int* q_out) {
  for (std::size_t i = 0; i < p.N * p.K; ++i) {
    q_out[i] = int4_from_nibble(unpack_nibble(p.data.data(), i));
  }
}

inline void unpack_e2m1_codes(const std::uint8_t* data, std::size_t N,
                              std::size_t K, std::uint8_t* codes_out) {
  for (std::size_t i = 0; i < N * K; ++i) {
    codes_out[i] = unpack_nibble(data, i);
  }
}

// ---------------------------------------------------------------------------
// Byte-ratio check vs formats.cuh accounting
// ---------------------------------------------------------------------------

// Expected bytes per weight from the amortised formula (no tensor scale).
inline double int4_bytes_per_weight(int group) {
  return (4.0 + 16.0 / static_cast<double>(group)) / 8.0;
}

inline double mxfp4_bytes_per_weight() {
  return (4.0 + 8.0 / 32.0) / 8.0;  // 0.53125
}

inline double nvfp4_bytes_per_weight() {
  return (4.0 + 8.0 / 16.0) / 8.0;  // 0.5625
}

// Measured from actual packed buffer sizes.
inline double measured_bytes_per_weight(std::size_t total_bytes, std::size_t N,
                                        std::size_t K) {
  return static_cast<double>(total_bytes) / static_cast<double>(N * K);
}

}  // namespace qgemm
