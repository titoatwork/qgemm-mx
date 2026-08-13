// correctness.hpp — structured probes and verification helpers (host).
//
// Random matrices hide nibble / group-index bugs: a single wrong lane still
// looks like FP16 noise after a reduction over K. Structured probes make those
// failures exact and local.
//
// Probe set (PROJECT-v2.0 / PROTOCOL):
//   1. One-hot weights   — only (n0, k0) nonzero → catches index/layout errors
//   2. Prime block scales — distinct primes per group → catches scale mis-index
//   3. Monotone ramp     — W[n,k] = f(n,k) smooth → catches striping / stride
//
// All generators write dense FP32 W[N*K] row-major. Pack with pack.hpp, then
// dequant and compare with the helpers below.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "qgemm/pack.hpp"

namespace qgemm {

// ---------------------------------------------------------------------------
// Small prime table for scale probes (distinct per group index)
// ---------------------------------------------------------------------------

// First 64 primes — enough for K/group up to large test shapes.
inline constexpr int kProbePrimes[] = {
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43,  47,
    53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311,
};
inline constexpr std::size_t kNumProbePrimes =
    sizeof(kProbePrimes) / sizeof(kProbePrimes[0]);

inline int probe_prime(std::size_t index) {
  // Cycle if the test matrix has more groups than the table (still distinct
  // within a short window; for exact-uniqueness keep groups <= 64).
  return kProbePrimes[index % kNumProbePrimes];
}

// ---------------------------------------------------------------------------
// Generators — dense FP32 weights
// ---------------------------------------------------------------------------

// One-hot: W[n0, k0] = value, all else 0.
inline std::vector<float> probe_one_hot(std::size_t N, std::size_t K,
                                        std::size_t n0, std::size_t k0,
                                        float value = 1.0f) {
  if (n0 >= N || k0 >= K) {
    throw std::invalid_argument("probe_one_hot: index out of range");
  }
  std::vector<float> W(N * K, 0.0f);
  W[n0 * K + k0] = value;
  return W;
}

// Prime scales: within each group, all elements equal to the group's prime
// (so after symmetric INT4 quant with scale=prime/7 the dequant recovers
// approximately prime; for exact E2M1 paths use probe_prime_scales_mx*).
//
// For INT4 exactness prefer pack_int4_codes with q=7 and scale=prime/7.
inline std::vector<float> probe_prime_scales(std::size_t N, std::size_t K,
                                             int group) {
  if (group <= 0 || K % static_cast<std::size_t>(group) != 0) {
    throw std::invalid_argument("probe_prime_scales: bad group");
  }
  std::vector<float> W(N * K);
  const std::size_t groups_per_row = K / static_cast<std::size_t>(group);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t g = 0; g < groups_per_row; ++g) {
      const float p =
          static_cast<float>(probe_prime(n * groups_per_row + g));
      for (int t = 0; t < group; ++t) {
        W[n * K + g * static_cast<std::size_t>(group) +
          static_cast<std::size_t>(t)] = p;
      }
    }
  }
  return W;
}

// Monotone ramp: W[n,k] = (n * K + k + 1) * step  (strictly increasing).
inline std::vector<float> probe_monotone_ramp(std::size_t N, std::size_t K,
                                              float step = 1.0f) {
  std::vector<float> W(N * K);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      W[n * K + k] = step * static_cast<float>(n * K + k + 1);
    }
  }
  return W;
}

// ---------------------------------------------------------------------------
// Exact code-level probes (bypass quantizer rounding)
// ---------------------------------------------------------------------------

// INT4: every element code = 1 (nibble), each group scale = prime (as FP16).
// Dequant reference: W[n,k] = 1 * prime(group(n,k)).
struct Int4PrimeProbe {
  PackedInt4 packed;
  std::vector<float> expected;  // N*K
};

inline Int4PrimeProbe make_int4_prime_probe(std::size_t N, std::size_t K,
                                            int group = kInt4Group128) {
  const std::size_t gcount = int4_scale_count(N, K, group);
  std::vector<int> q(N * K, 1);
  std::vector<float> scales(gcount);
  for (std::size_t i = 0; i < gcount; ++i) {
    scales[i] = static_cast<float>(probe_prime(i));
  }
  Int4PrimeProbe probe;
  probe.packed = pack_int4_codes(q.data(), scales.data(), N, K, group);
  probe.expected.resize(N * K);
  const std::size_t groups_per_row = K / static_cast<std::size_t>(group);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      const std::size_t g = k / static_cast<std::size_t>(group);
      const float s = scales[n * groups_per_row + g];
      // Use the same FP16 round-trip the packer applies.
      const float s16 = fp16_bits_to_float(float_to_fp16_bits(s));
      probe.expected[n * K + k] = 1.0f * s16;
    }
  }
  return probe;
}

// INT4 one-hot at codes: q[n0,k0]=qval, else 0; uniform scale=scale_f.
struct Int4OneHotProbe {
  PackedInt4 packed;
  std::vector<float> expected;
};

inline Int4OneHotProbe make_int4_one_hot_probe(std::size_t N, std::size_t K,
                                               std::size_t n0, std::size_t k0,
                                               int qval = 1,
                                               float scale_f = 1.0f,
                                               int group = kInt4Group128) {
  if (n0 >= N || k0 >= K) {
    throw std::invalid_argument("make_int4_one_hot_probe: index OOR");
  }
  std::vector<int> q(N * K, 0);
  q[n0 * K + k0] = qval;
  const std::size_t gcount = int4_scale_count(N, K, group);
  std::vector<float> scales(gcount, scale_f);
  Int4OneHotProbe probe;
  probe.packed = pack_int4_codes(q.data(), scales.data(), N, K, group);
  probe.expected.assign(N * K, 0.0f);
  const float s16 = fp16_bits_to_float(float_to_fp16_bits(scale_f));
  probe.expected[n0 * K + k0] = static_cast<float>(qval) * s16;
  return probe;
}

// MXFP4: element code 0x2 (=1.0), scale E8M0 = 127 + log2(prime) floored to
// exact power-of-two primes... simpler: use E8M0 bits = 127 (scale=1) for
// all but encode distinct *element* patterns per block via E2M1 codes.
//
// Prime-scale MX probe: each block uses E8M0 such that scale = 2^p_i with
// distinct p, and all elements code=0x2 (1.0) → expected = scale.
struct Mxfp4PrimeProbe {
  PackedMxfp4 packed;
  std::vector<float> expected;
};

inline Mxfp4PrimeProbe make_mxfp4_prime_probe(std::size_t N, std::size_t K) {
  const std::size_t scount = mxfp4_scale_count(N, K);
  std::vector<std::uint8_t> codes(N * K, 0x2);  // +1.0
  std::vector<std::uint8_t> scales(scount);
  // Distinct power-of-two scales: 2^(i % 8) via E8M0 bits = 127 + (i%8)
  // Using primes as *exponents offsets* keeps them obviously different.
  for (std::size_t i = 0; i < scount; ++i) {
    const int exp_off = (probe_prime(i) % 13) - 6;  // in [-6, 6] roughly
    int bits = 127 + exp_off;
    if (bits < 0) bits = 0;
    if (bits > 254) bits = 254;
    scales[i] = static_cast<std::uint8_t>(bits);
  }
  Mxfp4PrimeProbe probe;
  probe.packed = pack_mxfp4_codes(codes.data(), scales.data(), N, K);
  probe.expected.resize(N * K);
  const std::size_t blocks = K / static_cast<std::size_t>(kMxfp4BlockSize);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      const std::size_t b = k / static_cast<std::size_t>(kMxfp4BlockSize);
      const float s = e8m0_decode(scales[n * blocks + b]);
      probe.expected[n * K + k] = 1.0f * s;
    }
  }
  return probe;
}

// MXFP4 one-hot: single E2M1 nonzero.
struct Mxfp4OneHotProbe {
  PackedMxfp4 packed;
  std::vector<float> expected;
};

inline Mxfp4OneHotProbe make_mxfp4_one_hot_probe(std::size_t N, std::size_t K,
                                                 std::size_t n0, std::size_t k0,
                                                 std::uint8_t e2m1_code = 0x4,
                                                 std::uint8_t e8m0_bits = 127) {
  if (n0 >= N || k0 >= K) {
    throw std::invalid_argument("make_mxfp4_one_hot_probe: index OOR");
  }
  std::vector<std::uint8_t> codes(N * K, 0);
  codes[n0 * K + k0] = e2m1_code;
  std::vector<std::uint8_t> scales(mxfp4_scale_count(N, K), e8m0_bits);
  Mxfp4OneHotProbe probe;
  probe.packed = pack_mxfp4_codes(codes.data(), scales.data(), N, K);
  probe.expected.assign(N * K, 0.0f);
  probe.expected[n0 * K + k0] =
      e2m1_decode(e2m1_code) * e8m0_decode(e8m0_bits);
  return probe;
}

// NVFP4 prime-scale probe: E2M1=1.0, distinct positive E4M3 scales.
struct Nvfp4PrimeProbe {
  PackedNvfp4 packed;
  std::vector<float> expected;
};

inline Nvfp4PrimeProbe make_nvfp4_prime_probe(std::size_t N, std::size_t K,
                                              float tensor_scale = 1.0f) {
  const std::size_t scount = nvfp4_scale_count(N, K);
  std::vector<std::uint8_t> codes(N * K, 0x2);  // +1.0
  std::vector<std::uint8_t> scales(scount);
  for (std::size_t i = 0; i < scount; ++i) {
    // Encode prime as E4M3 (positive); primes fit easily in E4M3 range.
    scales[i] = e4m3_encode(static_cast<float>(probe_prime(i)));
  }
  Nvfp4PrimeProbe probe;
  probe.packed =
      pack_nvfp4_codes(codes.data(), scales.data(), N, K, tensor_scale);
  probe.expected.resize(N * K);
  const std::size_t blocks = K / static_cast<std::size_t>(kNvfp4BlockSize);
  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      const std::size_t b = k / static_cast<std::size_t>(kNvfp4BlockSize);
      const float s = e4m3_decode(scales[n * blocks + b]);
      probe.expected[n * K + k] = 1.0f * s * tensor_scale;
    }
  }
  return probe;
}

// Monotone ramp of E2M1 codes cycling 0..7 along K (positive), scale=1.
struct Mxfp4RampProbe {
  PackedMxfp4 packed;
  std::vector<float> expected;
};

inline Mxfp4RampProbe make_mxfp4_ramp_probe(std::size_t N, std::size_t K) {
  std::vector<std::uint8_t> codes(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    codes[i] = static_cast<std::uint8_t>(i % 8);  // 0..7 → 0,0.5,1,...,6
  }
  std::vector<std::uint8_t> scales(mxfp4_scale_count(N, K), 127);  // ×1
  Mxfp4RampProbe probe;
  probe.packed = pack_mxfp4_codes(codes.data(), scales.data(), N, K);
  probe.expected.resize(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    probe.expected[i] = e2m1_decode(codes[i]);
  }
  return probe;
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

struct CompareResult {
  bool ok = true;
  std::size_t first_fail = static_cast<std::size_t>(-1);
  std::size_t num_fail = 0;
  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  std::string message;
};

// Exact bitwise equality of floats (for code-level probes with identical path).
inline CompareResult compare_exact(const float* got, const float* ref,
                                   std::size_t n) {
  CompareResult r;
  for (std::size_t i = 0; i < n; ++i) {
    // Treat NaN==NaN as match for invalid-scale tests.
    const bool both_nan = std::isnan(got[i]) && std::isnan(ref[i]);
    std::uint32_t ag, ar;
    std::memcpy(&ag, &got[i], 4);
    std::memcpy(&ar, &ref[i], 4);
    if (!both_nan && ag != ar) {
      r.ok = false;
      ++r.num_fail;
      if (r.first_fail == static_cast<std::size_t>(-1)) r.first_fail = i;
      const float ae = std::fabs(got[i] - ref[i]);
      r.max_abs_err = std::max(r.max_abs_err, ae);
    }
  }
  if (!r.ok) {
    r.message = "exact compare failed at " + std::to_string(r.first_fail) +
                " (" + std::to_string(r.num_fail) + " mismatches)";
  }
  return r;
}

// Absolute + relative tolerance (for quantize-then-dequant paths).
inline CompareResult compare_close(const float* got, const float* ref,
                                   std::size_t n, float atol = 1e-3f,
                                   float rtol = 1e-3f) {
  CompareResult r;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(got[i]) || std::isnan(ref[i])) {
      if (std::isnan(got[i]) && std::isnan(ref[i])) continue;
      r.ok = false;
      ++r.num_fail;
      if (r.first_fail == static_cast<std::size_t>(-1)) r.first_fail = i;
      continue;
    }
    const float ae = std::fabs(got[i] - ref[i]);
    const float re = ae / std::max(1e-12f, std::fabs(ref[i]));
    r.max_abs_err = std::max(r.max_abs_err, ae);
    r.max_rel_err = std::max(r.max_rel_err, re);
    if (ae > atol + rtol * std::fabs(ref[i])) {
      r.ok = false;
      ++r.num_fail;
      if (r.first_fail == static_cast<std::size_t>(-1)) r.first_fail = i;
    }
  }
  if (!r.ok) {
    r.message = "close compare failed at " + std::to_string(r.first_fail) +
                " max_abs=" + std::to_string(r.max_abs_err) +
                " max_rel=" + std::to_string(r.max_rel_err);
  }
  return r;
}

// Max abs error only (useful for logging).
inline float max_abs_error(const float* a, const float* b, std::size_t n) {
  float m = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isnan(a[i]) || std::isnan(b[i])) continue;
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

}  // namespace qgemm
