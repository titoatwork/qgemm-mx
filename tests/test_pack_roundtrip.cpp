// test_pack_roundtrip.cpp — pack → dequant → compare for INT4 / MXFP4 / NVFP4.
//
// Structured probes first (exact), then quantize-path roundtrips (tolerance).

#include "qgemm/correctness.hpp"
#include "qgemm/pack.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_fails = 0;

#define EXPECT_TRUE(cond)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                                \
  do {                                                                        \
    const double _a = (a), _b = (b), _t = (tol);                              \
    if (!(std::fabs(_a - _b) <= _t)) {                                        \
      std::fprintf(stderr,                                                    \
                   "FAIL %s:%d: |%g - %g| > %g\n", __FILE__, __LINE__, _a,    \
                   _b, _t);                                                   \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

void expect_compare(const qgemm::CompareResult& r, const char* label) {
  if (!r.ok) {
    std::fprintf(stderr, "FAIL %s: %s\n", label, r.message.c_str());
    ++g_fails;
  }
}

// ---------------------------------------------------------------------------
// Scalar codec unit tests
// ---------------------------------------------------------------------------

void test_e2m1_table() {
  // Every nonzero code roundtrips through decode→encode.
  // Code 0x8 is signed zero; encode collapses both zeros to 0x0.
  for (int c = 0; c < 16; ++c) {
    if (c == 0x8) continue;
    const float v = qgemm::e2m1_decode(static_cast<std::uint8_t>(c));
    const std::uint8_t back = qgemm::e2m1_encode(v);
    EXPECT_TRUE(back == static_cast<std::uint8_t>(c));
  }
  EXPECT_TRUE(qgemm::e2m1_encode(qgemm::e2m1_decode(0x8)) == 0x0);
  EXPECT_NEAR(qgemm::e2m1_decode(0x0), 0.0, 0);
  EXPECT_NEAR(qgemm::e2m1_decode(0x1), 0.5, 0);
  EXPECT_NEAR(qgemm::e2m1_decode(0x7), 6.0, 0);
  EXPECT_NEAR(qgemm::e2m1_decode(0xF), -6.0, 0);
  EXPECT_NEAR(qgemm::e2m1_decode(0x8), 0.0, 0);  // -0 compares equal to 0
}

void test_e8m0() {
  EXPECT_NEAR(qgemm::e8m0_decode(127), 1.0, 0);
  EXPECT_NEAR(qgemm::e8m0_decode(128), 2.0, 0);
  EXPECT_NEAR(qgemm::e8m0_decode(126), 0.5, 0);
  EXPECT_TRUE(std::isnan(qgemm::e8m0_decode(0xFF)));
  EXPECT_TRUE(qgemm::e8m0_encode(1.0f) == 127);
  EXPECT_TRUE(qgemm::e8m0_encode(2.0f) == 128);
  EXPECT_TRUE(qgemm::e8m0_encode(0.5f) == 126);
  // Nearest: 3 → 2 or 4; 3 is closer to... both equidistant in log? 
  // Our encoder uses mantissa threshold 0.707 on frexp.
  // frexp(3) → m=0.75, e=2 → 0.75 >= 0.707 → 2^2 = 4 → bits 129
  EXPECT_TRUE(qgemm::e8m0_encode(3.0f) == 129);
}

void test_e4m3_roundtrip() {
  // Finite codes (skip NaN 0x7F / 0xFF)
  for (int c = 0; c < 256; ++c) {
    if ((c & 0x7F) == 0x7F) continue;
    const float v = qgemm::e4m3_decode(static_cast<std::uint8_t>(c));
    const std::uint8_t back = qgemm::e4m3_encode(v);
    EXPECT_TRUE(back == static_cast<std::uint8_t>(c));
  }
  EXPECT_NEAR(qgemm::e4m3_decode(0x00), 0.0, 0);
  // 1.0: exp=7, mant=0 → bits = 0b0_0111_000 = 0x38
  EXPECT_NEAR(qgemm::e4m3_decode(0x38), 1.0, 1e-6);
  EXPECT_TRUE(std::isnan(qgemm::e4m3_decode(0x7F)));
}

void test_fp16_roundtrip() {
  const float vals[] = {0.f, 1.f, -1.f, 0.5f, 2.f, 7.f, 0.0001f, 65504.f};
  for (float v : vals) {
    const std::uint16_t h = qgemm::float_to_fp16_bits(v);
    const float back = qgemm::fp16_bits_to_float(h);
    // Half has ~3 decimal digits; allow small relative error.
    if (v == 0.f) {
      EXPECT_NEAR(back, 0.0, 0);
    } else {
      EXPECT_NEAR(back, v, std::fabs(v) * 1e-3 + 1e-6);
    }
  }
}

void test_int4_nibble() {
  for (int v = -8; v <= 7; ++v) {
    EXPECT_TRUE(qgemm::int4_from_nibble(qgemm::int4_to_nibble(v)) == v);
  }
}

// ---------------------------------------------------------------------------
// Nibble packing
// ---------------------------------------------------------------------------

void test_nibble_pack_order() {
  std::uint8_t buf[4] = {0, 0, 0, 0};
  // linear 0 → low nibble, linear 1 → high nibble of byte 0
  qgemm::pack_nibble(buf, 0, 0xA);
  qgemm::pack_nibble(buf, 1, 0xB);
  qgemm::pack_nibble(buf, 2, 0xC);
  qgemm::pack_nibble(buf, 3, 0xD);
  EXPECT_TRUE(buf[0] == 0xBA);
  EXPECT_TRUE(buf[1] == 0xDC);
  EXPECT_TRUE(qgemm::unpack_nibble(buf, 0) == 0xA);
  EXPECT_TRUE(qgemm::unpack_nibble(buf, 1) == 0xB);
  EXPECT_TRUE(qgemm::unpack_nibble(buf, 2) == 0xC);
  EXPECT_TRUE(qgemm::unpack_nibble(buf, 3) == 0xD);
}

// ---------------------------------------------------------------------------
// Structured probes — exact
// ---------------------------------------------------------------------------

void test_int4_prime_probe() {
  const std::size_t N = 4, K = 256;
  auto probe = qgemm::make_int4_prime_probe(N, K, 128);
  auto got = qgemm::dequant_reference_int4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "int4 prime probe");

  // Spot-check: first group scale is prime 2, q=1 → value 2
  EXPECT_NEAR(got[0], 2.0, 1e-3);
}

void test_int4_one_hot_probe() {
  const std::size_t N = 8, K = 128;
  for (std::size_t n0 : {std::size_t{0}, std::size_t{3}, N - 1}) {
    for (std::size_t k0 : {std::size_t{0}, std::size_t{17}, K - 1}) {
      auto probe =
          qgemm::make_int4_one_hot_probe(N, K, n0, k0, /*qval=*/5,
                                         /*scale=*/2.0f, 128);
      auto got = qgemm::dequant_reference_int4(probe.packed);
      expect_compare(
          qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
          "int4 one-hot");
      // All zeros except the hot index
      for (std::size_t i = 0; i < N * K; ++i) {
        if (i == n0 * K + k0) {
          EXPECT_NEAR(got[i], 5.0 * qgemm::fp16_bits_to_float(
                                        qgemm::float_to_fp16_bits(2.0f)),
                      1e-4);
        } else {
          EXPECT_NEAR(got[i], 0.0, 0);
        }
      }
    }
  }
}

void test_mxfp4_prime_probe() {
  const std::size_t N = 3, K = 64;  // 2 blocks per row
  auto probe = qgemm::make_mxfp4_prime_probe(N, K);
  auto got = qgemm::dequant_reference_mxfp4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "mxfp4 prime probe");
}

void test_mxfp4_one_hot_probe() {
  const std::size_t N = 4, K = 32;
  auto probe = qgemm::make_mxfp4_one_hot_probe(N, K, 2, 7, /*code=*/0x5,
                                               /*e8m0=*/128);  // 3.0 * 2.0
  auto got = qgemm::dequant_reference_mxfp4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "mxfp4 one-hot");
  EXPECT_NEAR(got[2 * K + 7], 6.0, 0);  // 3 * 2
  EXPECT_NEAR(got[0], 0.0, 0);
}

void test_mxfp4_ramp_probe() {
  const std::size_t N = 2, K = 32;
  auto probe = qgemm::make_mxfp4_ramp_probe(N, K);
  auto got = qgemm::dequant_reference_mxfp4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "mxfp4 ramp");
  // First 8 along K: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  EXPECT_NEAR(got[0], 0.0, 0);
  EXPECT_NEAR(got[1], 0.5, 0);
  EXPECT_NEAR(got[7], 6.0, 0);
}

void test_nvfp4_prime_probe() {
  const std::size_t N = 2, K = 32;  // 2 blocks of 16
  auto probe = qgemm::make_nvfp4_prime_probe(N, K, /*tensor=*/2.0f);
  auto got = qgemm::dequant_reference_nvfp4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "nvfp4 prime probe");
  // Block 0 scale encodes prime 2 → e4m3 ≈ 2, × tensor 2 → ≈ 4
  EXPECT_NEAR(got[0],
              qgemm::e4m3_decode(qgemm::e4m3_encode(2.0f)) * 2.0f, 1e-5);
}

// ---------------------------------------------------------------------------
// Quantize-path roundtrip (tolerance)
// ---------------------------------------------------------------------------

void test_int4_quantize_roundtrip() {
  const std::size_t N = 4, K = 128;
  // Values well inside symmetric range with known scale.
  std::vector<float> W(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    // Pattern in [-7, 7]
    W[i] = static_cast<float>(static_cast<int>(i % 15) - 7);
  }
  auto packed = qgemm::pack_int4(W.data(), N, K, 128);
  auto got = qgemm::dequant_reference_int4(packed);
  // With per-group scale = max/7 and integer codes, should recover closely.
  expect_compare(
      qgemm::compare_close(got.data(), W.data(), N * K, /*atol=*/0.05f,
                           /*rtol=*/0.05f),
      "int4 quantize roundtrip");
}

void test_mxfp4_quantize_roundtrip() {
  const std::size_t N = 2, K = 32;
  // Exact E2M1 * power-of-two values → perfect recovery after re-quant.
  std::vector<float> W(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    W[i] = qgemm::e2m1_decode(static_cast<std::uint8_t>(i % 16)) * 4.0f;
  }
  auto packed = qgemm::pack_mxfp4(W.data(), N, K);
  auto got = qgemm::dequant_reference_mxfp4(packed);
  expect_compare(
      qgemm::compare_close(got.data(), W.data(), N * K, /*atol=*/1e-5f,
                           /*rtol=*/1e-5f),
      "mxfp4 quantize exact-set roundtrip");
}

void test_nvfp4_quantize_roundtrip() {
  const std::size_t N = 2, K = 16;
  std::vector<float> W(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    W[i] = qgemm::e2m1_decode(static_cast<std::uint8_t>((i * 3) % 16)) * 1.0f;
  }
  auto packed = qgemm::pack_nvfp4(W.data(), N, K, /*tensor=*/1.0f);
  auto got = qgemm::dequant_reference_nvfp4(packed);
  expect_compare(
      qgemm::compare_close(got.data(), W.data(), N * K, /*atol=*/0.05f,
                           /*rtol=*/0.05f),
      "nvfp4 quantize roundtrip");
}

// ---------------------------------------------------------------------------
// Dense generator probes (pack path)
// ---------------------------------------------------------------------------

void test_probe_generators_smoke() {
  auto one = qgemm::probe_one_hot(4, 32, 1, 5, 3.0f);
  EXPECT_NEAR(one[1 * 32 + 5], 3.0, 0);
  EXPECT_NEAR(one[0], 0.0, 0);

  auto primes = qgemm::probe_prime_scales(2, 128, 128);
  EXPECT_NEAR(primes[0], 2.0, 0);  // first prime
  EXPECT_NEAR(primes[127], 2.0, 0);
  EXPECT_NEAR(primes[128], 3.0, 0);  // next group

  auto ramp = qgemm::probe_monotone_ramp(2, 4, 1.0f);
  EXPECT_NEAR(ramp[0], 1.0, 0);
  EXPECT_NEAR(ramp[7], 8.0, 0);
  for (std::size_t i = 1; i < 8; ++i) {
    EXPECT_TRUE(ramp[i] > ramp[i - 1]);
  }
}

// ---------------------------------------------------------------------------
// Awkward shape smoke (2880 divisible by 32, not by 128)
// ---------------------------------------------------------------------------

void test_gpt_oss_dims_mxfp4() {
  // Small stand-in: N=K=96 (div by 32, not by 128) with one-hot exact probe.
  const std::size_t N = 96, K = 96;
  EXPECT_TRUE(K % 128 != 0);
  EXPECT_TRUE(K % 32 == 0);
  auto probe = qgemm::make_mxfp4_one_hot_probe(N, K, 10, 33, 0x6, 127);
  auto got = qgemm::dequant_reference_mxfp4(probe.packed);
  expect_compare(
      qgemm::compare_exact(got.data(), probe.expected.data(), N * K),
      "awkward K mxfp4 one-hot");
  EXPECT_NEAR(got[10 * K + 33], 4.0, 0);
}

// Code unpack consistency
void test_unpack_codes() {
  const std::size_t N = 2, K = 16;
  std::vector<int> q(N * K);
  for (std::size_t i = 0; i < N * K; ++i) {
    q[i] = static_cast<int>(i % 16) - 8;
  }
  // pack_int4 quantizer only allows 64/128; codes API accepts any group | K.
  const int group = 16;
  std::vector<float> scales(qgemm::int4_scale_count(N, K, group), 1.0f);
  auto packed = qgemm::pack_int4_codes(q.data(), scales.data(), N, K, group);
  std::vector<int> back(N * K);
  qgemm::unpack_int4_codes(packed, back.data());
  for (std::size_t i = 0; i < N * K; ++i) {
    EXPECT_TRUE(back[i] == q[i]);
  }
}

}  // namespace

int main() {
  test_e2m1_table();
  test_e8m0();
  test_e4m3_roundtrip();
  test_fp16_roundtrip();
  test_int4_nibble();
  test_nibble_pack_order();

  test_int4_prime_probe();
  test_int4_one_hot_probe();
  test_mxfp4_prime_probe();
  test_mxfp4_one_hot_probe();
  test_mxfp4_ramp_probe();
  test_nvfp4_prime_probe();

  test_int4_quantize_roundtrip();
  test_mxfp4_quantize_roundtrip();
  test_nvfp4_quantize_roundtrip();

  test_probe_generators_smoke();
  test_gpt_oss_dims_mxfp4();
  test_unpack_codes();

  if (g_fails != 0) {
    std::fprintf(stderr, "\ntest_pack_roundtrip: %d failure(s)\n", g_fails);
    return 1;
  }
  std::printf("test_pack_roundtrip: OK\n");
  return 0;
}
