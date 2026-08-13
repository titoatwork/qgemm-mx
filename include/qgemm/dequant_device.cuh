// dequant_device.cuh — device decode matching pack.hpp host encodings.
//
// KEEP IN SYNC with pack.hpp. Sequential nibble order (low=even k, high=odd k).
// This is the R0/R1 *correctness* datapath, not the R2 wgmma pack.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace qgemm {
namespace dev {

__device__ __forceinline__ uint8_t unpack_nibble(const uint8_t* data,
                                                 size_t linear_index) {
  const uint8_t b = data[linear_index >> 1];
  return (linear_index & 1) ? static_cast<uint8_t>(b >> 4)
                            : static_cast<uint8_t>(b & 0xF);
}

__device__ __forceinline__ int int4_from_nibble(uint8_t n) {
  n = static_cast<uint8_t>(n & 0xF);
  return (n & 8) ? static_cast<int>(n) - 16 : static_cast<int>(n);
}

// Interleaved-style upconvert (same *math* as 0x6400 mantissa insert).
// Prior art: TRT-LLM / vLLM numeric_conversion — we re-derived the 1024+u
// identity for unsigned nibble u, then map two's-complement.
// MUST match int4_from_nibble exactly (tested).
__device__ __forceinline__ float int4_from_nibble_bittrick(uint8_t n) {
  const unsigned u = static_cast<unsigned>(n & 0xF);
  // FP16 1024.0 = 0x6400; LSBs of mantissa at this exponent are 1.0, so
  // 0x6400+u == 1024+u for u in 0..15. Subtract 1024, then signed adjust.
  const __half h = __ushort_as_half(static_cast<unsigned short>(0x6400u + u));
  float v = __half2float(h) - 1024.0f;
  if (u >= 8u) v -= 16.0f;
  return v;
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t h) {
  __half v = __ushort_as_half(h);
  return __half2float(v);
}

// Same table as pack.hpp kE2M1PosTable.
__device__ __forceinline__ float e2m1_decode(uint8_t code) {
  const float mag_tab[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const uint8_t c = static_cast<uint8_t>(code & 0xF);
  const float mag = mag_tab[c & 7];
  return (c & 8) ? -mag : mag;
}

__device__ __forceinline__ float e8m0_decode(uint8_t bits) {
  if (bits == 0xFFu) return nanf("");
  return ldexpf(1.0f, static_cast<int>(bits) - 127);
}

__device__ __forceinline__ float e4m3_decode(uint8_t bits) {
  const int sign = (bits >> 7) & 1;
  const int exp = (bits >> 3) & 0xF;
  const int mant = bits & 0x7;
  if (exp == 0xF && mant == 0x7) return nanf("");
  float mag;
  if (exp == 0) {
    mag = ldexpf(static_cast<float>(mant), -9);
  } else {
    mag = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
  }
  return sign ? -mag : mag;
}

__device__ __forceinline__ float dequant_int4_elem(const uint8_t* data,
                                                   const uint16_t* scales,
                                                   size_t n, size_t k,
                                                   size_t N, size_t K,
                                                   int group) {
  (void)N;
  const size_t groups_per_row = K / static_cast<size_t>(group);
  const size_t g = k / static_cast<size_t>(group);
  const float scale = fp16_bits_to_float(scales[n * groups_per_row + g]);
  const int q = int4_from_nibble(unpack_nibble(data, n * K + k));
  return static_cast<float>(q) * scale;
}

__device__ __forceinline__ float dequant_mxfp4_elem(const uint8_t* data,
                                                    const uint8_t* scales,
                                                    size_t n, size_t k,
                                                    size_t N, size_t K) {
  (void)N;
  const size_t blocks = K / 32;
  const size_t b = k / 32;
  const float scale = e8m0_decode(scales[n * blocks + b]);
  return e2m1_decode(unpack_nibble(data, n * K + k)) * scale;
}

__device__ __forceinline__ float dequant_nvfp4_elem(const uint8_t* data,
                                                    const uint8_t* scales,
                                                    float tensor_scale,
                                                    size_t n, size_t k,
                                                    size_t N, size_t K) {
  (void)N;
  const size_t blocks = K / 16;
  const size_t b = k / 16;
  const float bscale = e4m3_decode(scales[n * blocks + b]);
  return e2m1_decode(unpack_nibble(data, n * K + k)) * bscale * tensor_scale;
}

}  // namespace dev
}  // namespace qgemm
