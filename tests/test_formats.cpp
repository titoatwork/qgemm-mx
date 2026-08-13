// test_formats.cpp — host unit tests for formats.cuh byte accounting
// and shapes.hpp frozen tables.
//
// Build: make test-cpu  (g++, no GPU)

#include "qgemm/formats.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/shapes.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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

#define EXPECT_EQ(a, b)                                                       \
  do {                                                                        \
    const auto _a = (a);                                                      \
    const auto _b = (b);                                                      \
    if (!(_a == _b)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s != %s\n", __FILE__, __LINE__,      \
                   #a, #b);                                                   \
      ++g_fails;                                                              \
    }                                                                         \
  } while (0)

void test_bytes_per_weight_table() {
  using qgemm::Format;
  // Canonical ratios from formats.cuh / PROJECT-v2.0 table.
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::FP16), 2.0, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::BF16), 2.0, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::FP8_E4M3), 1.0, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::INT4_G128), 0.515625, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::INT4_G64), 0.53125, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::MXFP4), 0.53125, 1e-12);
  EXPECT_NEAR(qgemm::bytes_per_weight(Format::NVFP4), 0.5625, 1e-12);
}

void test_byte_ratio_vs_fp8() {
  using qgemm::Format;
  // MXFP4 vs FP8 → 1.88× fewer bytes (fp8/mxfp4).
  const double mx_vs_fp8 =
      qgemm::bytes_per_weight(Format::FP8_E4M3) /
      qgemm::bytes_per_weight(Format::MXFP4);
  EXPECT_NEAR(mx_vs_fp8, 1.0 / 0.53125, 1e-12);
  EXPECT_NEAR(mx_vs_fp8, 1.8823529411764706, 1e-9);

  const double nv_vs_fp8 =
      qgemm::bytes_per_weight(Format::FP8_E4M3) /
      qgemm::bytes_per_weight(Format::NVFP4);
  EXPECT_NEAR(nv_vs_fp8, 1.0 / 0.5625, 1e-12);

  const double int4_vs_fp8 =
      qgemm::bytes_per_weight(Format::FP8_E4M3) /
      qgemm::bytes_per_weight(Format::INT4_G128);
  EXPECT_NEAR(int4_vs_fp8, 1.0 / 0.515625, 1e-12);
}

void test_pack_sizes_match_formats() {
  // Measured packed buffer sizes must match amortised bytes_per_weight.
  const std::size_t N = 64;
  const std::size_t K = 256;  // divisible by 128, 64, 32, 16

  const double bpw_i128 = qgemm::measured_bytes_per_weight(
      qgemm::int4_total_bytes(N, K, 128), N, K);
  EXPECT_NEAR(bpw_i128, qgemm::bytes_per_weight(qgemm::Format::INT4_G128),
              1e-12);
  EXPECT_NEAR(bpw_i128, qgemm::int4_bytes_per_weight(128), 1e-12);

  const double bpw_i64 = qgemm::measured_bytes_per_weight(
      qgemm::int4_total_bytes(N, K, 64), N, K);
  EXPECT_NEAR(bpw_i64, qgemm::bytes_per_weight(qgemm::Format::INT4_G64),
              1e-12);

  const double bpw_mx = qgemm::measured_bytes_per_weight(
      qgemm::mxfp4_total_bytes(N, K), N, K);
  EXPECT_NEAR(bpw_mx, qgemm::bytes_per_weight(qgemm::Format::MXFP4), 1e-12);
  EXPECT_NEAR(bpw_mx, qgemm::mxfp4_bytes_per_weight(), 1e-12);

  const double bpw_nv = qgemm::measured_bytes_per_weight(
      qgemm::nvfp4_total_bytes(N, K), N, K);
  EXPECT_NEAR(bpw_nv, qgemm::bytes_per_weight(qgemm::Format::NVFP4), 1e-12);
  EXPECT_NEAR(bpw_nv, qgemm::nvfp4_bytes_per_weight(), 1e-12);
}

void test_weight_bytes_formula() {
  const std::size_t N = 4096, K = 4096;
  const std::size_t wb =
      qgemm::weight_bytes(qgemm::Format::INT4_G128, N, K);
  // 4096*4096 * 0.515625 = 8_650_752
  EXPECT_EQ(wb, static_cast<std::size_t>(4096ull * 4096ull * 0.515625));

  const std::size_t wb_mx = qgemm::weight_bytes(qgemm::Format::MXFP4, N, K);
  EXPECT_EQ(wb_mx, static_cast<std::size_t>(4096ull * 4096ull * 0.53125));
}

void test_shapes_frozen() {
  EXPECT_EQ(qgemm::kNumLayerShapes, static_cast<std::size_t>(6));
  EXPECT_EQ(qgemm::kNumMSweep, static_cast<std::size_t>(12));
  EXPECT_EQ(qgemm::kMSweep[0], static_cast<std::size_t>(1));
  EXPECT_EQ(qgemm::kMSweep[qgemm::kNumMSweep - 1],
            static_cast<std::size_t>(2048));

  EXPECT_EQ(qgemm::kLayerShapes[0].N, static_cast<std::size_t>(4096));
  EXPECT_EQ(qgemm::kLayerShapes[0].K, static_cast<std::size_t>(4096));

  const qgemm::LayerShape* oss = qgemm::find_layer("gpt_oss_2880");
  EXPECT_TRUE(oss != nullptr);
  EXPECT_EQ(oss->N, static_cast<std::size_t>(2880));
  EXPECT_EQ(oss->K, static_cast<std::size_t>(2880));
  // 2880 % 128 != 0 — the live bug class
  EXPECT_TRUE(oss->K % 128 != 0);
  // but divisible by MXFP4 block 32 and NVFP4 block 16
  EXPECT_EQ(oss->K % 32, static_cast<std::size_t>(0));
  EXPECT_EQ(oss->K % 16, static_cast<std::size_t>(0));

  EXPECT_TRUE(qgemm::find_layer("no_such_layer") == nullptr);

  EXPECT_EQ(qgemm::per_gpu_N(4096, 8), static_cast<std::size_t>(512));
  EXPECT_EQ(qgemm::kInt4GroupPrimary, 128);
  EXPECT_EQ(qgemm::kMxfp4Block, 32);
  EXPECT_EQ(qgemm::kNvfp4Block, 16);
}

void test_format_names() {
  EXPECT_TRUE(std::string(qgemm::format_name(qgemm::Format::MXFP4)) ==
              "mxfp4");
  EXPECT_TRUE(std::string(qgemm::format_name(qgemm::Format::NVFP4)) ==
              "nvfp4");
  EXPECT_TRUE(std::string(qgemm::format_name(qgemm::Format::INT4_G128)) ==
              "int4_g128");
}

void test_element_and_scale_bits() {
  EXPECT_NEAR(qgemm::element_bits(qgemm::Format::MXFP4), 4.0, 0);
  EXPECT_NEAR(qgemm::scale_bits_per_element(qgemm::Format::MXFP4), 0.25, 0);
  EXPECT_NEAR(qgemm::scale_bits_per_element(qgemm::Format::NVFP4), 0.5, 0);
  EXPECT_NEAR(qgemm::scale_bits_per_element(qgemm::Format::INT4_G128),
              16.0 / 128.0, 0);
}

void test_predicted_crossover_monotone() {
  // Lower bytes/weight → lower crossover M (goes compute-bound sooner).
  const double ridge = 600.0;  // H100-ish FLOP/byte placeholder
  const double m_fp16 =
      qgemm::predicted_crossover_M(qgemm::Format::FP16, ridge);
  const double m_mx =
      qgemm::predicted_crossover_M(qgemm::Format::MXFP4, ridge);
  EXPECT_TRUE(m_mx < m_fp16);
}

}  // namespace

int main() {
  test_bytes_per_weight_table();
  test_byte_ratio_vs_fp8();
  test_pack_sizes_match_formats();
  test_weight_bytes_formula();
  test_shapes_frozen();
  test_format_names();
  test_element_and_scale_bits();
  test_predicted_crossover_monotone();

  if (g_fails != 0) {
    std::fprintf(stderr, "\ntest_formats: %d failure(s)\n", g_fails);
    return 1;
  }
  std::printf("test_formats: OK\n");
  return 0;
}
