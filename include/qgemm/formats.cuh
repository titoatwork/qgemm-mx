// formats.cuh — byte accounting for the weight formats under study.
//
// This file encodes the project's central arithmetic. The thesis is that
// block-scaled 4-bit formats move ~1.9x fewer bytes per weight than FP8 and
// currently deliver ~1.0x the throughput on Hopper. Those ratios are computed
// here rather than quoted, so every reported speedup can be checked against the
// bytes it is supposed to come from.

#pragma once

#include <cstddef>
#include <string>

namespace qgemm {

enum class Format {
  FP16,        // 16-bit, no scale
  BF16,        // 16-bit, no scale
  FP8_E4M3,    // 8-bit, per-tensor or per-channel scale (negligible)
  INT4_G128,   // 4-bit + FP16 scale per 128 elements (GPTQ/AWQ style)
  INT4_G64,    // 4-bit + FP16 scale per 64
  MXFP4,       // E2M1 + E8M0 (8-bit power-of-two) scale per 32   [OCP MX]
  NVFP4,       // E2M1 + E4M3 (8-bit) scale per 16 + FP32 tensor scale
};

inline const char* format_name(Format f) {
  switch (f) {
    case Format::FP16:      return "fp16";
    case Format::BF16:      return "bf16";
    case Format::FP8_E4M3:  return "fp8_e4m3";
    case Format::INT4_G128: return "int4_g128";
    case Format::INT4_G64:  return "int4_g64";
    case Format::MXFP4:     return "mxfp4";
    case Format::NVFP4:     return "nvfp4";
  }
  return "unknown";
}

// Bits per weight element, excluding scales.
inline double element_bits(Format f) {
  switch (f) {
    case Format::FP16:
    case Format::BF16:      return 16.0;
    case Format::FP8_E4M3:  return 8.0;
    case Format::INT4_G128:
    case Format::INT4_G64:
    case Format::MXFP4:
    case Format::NVFP4:     return 4.0;
  }
  return 0.0;
}

// Amortized scale bits per weight element.
inline double scale_bits_per_element(Format f) {
  switch (f) {
    case Format::FP16:
    case Format::BF16:      return 0.0;
    case Format::FP8_E4M3:  return 0.0;   // per-tensor/channel, negligible
    case Format::INT4_G128: return 16.0 / 128.0;  // fp16 per 128 = 0.125
    case Format::INT4_G64:  return 16.0 / 64.0;   // fp16 per 64  = 0.25
    case Format::MXFP4:     return 8.0 / 32.0;    // e8m0 per 32  = 0.25
    case Format::NVFP4:     return 8.0 / 16.0;    // e4m3 per 16  = 0.5
  }
  return 0.0;
}

inline double bytes_per_weight(Format f) {
  return (element_bits(f) + scale_bits_per_element(f)) / 8.0;
}

// Total weight-side bytes a GEMM must stream for an N x K weight matrix.
// At small M this is the entire cost, so it is also the ideal-time numerator.
inline size_t weight_bytes(Format f, size_t N, size_t K) {
  return static_cast<size_t>(bytes_per_weight(f) * static_cast<double>(N) *
                             static_cast<double>(K));
}

// Bytes moved by a full weight-only GEMM: weights + activations in + output.
// Activations and output are negligible at small M but matter past M ~ 256.
inline size_t total_bytes(Format f, size_t M, size_t N, size_t K,
                          size_t act_bytes = 2, size_t out_bytes = 2) {
  return weight_bytes(f, N, K) + M * K * act_bytes + M * N * out_bytes;
}

// The available speedup in the bandwidth-bound regime, from bytes alone.
// MXFP4 vs FP8 -> 1.88x. This is the number the project is trying to realize.
inline double byte_ratio(Format numerator, Format denominator) {
  return bytes_per_weight(numerator) / bytes_per_weight(denominator);
}

inline double flops(size_t M, size_t N, size_t K) {
  return 2.0 * static_cast<double>(M) * static_cast<double>(N) *
         static_cast<double>(K);
}

// Arithmetic intensity in FLOP/byte. Compare against the device ridge point to
// predict where the format stops being bandwidth-bound: for weight-dominated
// traffic this is approximately M / bytes_per_weight, so a lower-precision
// format goes compute-bound at a LOWER M -- which is exactly why the 4-bit
// advantage has a ceiling in batch size.
inline double arithmetic_intensity(Format f, size_t M, size_t N, size_t K) {
  return flops(M, N, K) / static_cast<double>(total_bytes(f, M, N, K));
}

// Predicted batch size at which a format becomes compute-bound on a device with
// the given ridge point. Used to generate the preregistered predictions.
inline double predicted_crossover_M(Format f, double ridge_flop_per_byte) {
  return ridge_flop_per_byte * bytes_per_weight(f) / 2.0;
}

}  // namespace qgemm
