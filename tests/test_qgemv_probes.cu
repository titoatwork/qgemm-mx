// Structured GPU probes for the fused-ref kernel (k_qgemm_ref).
//
// Random / dequant-then-GEMM coverage lives in test_qgemv_gpu.cu. This file
// drives correctness.hpp generators (prime-scale, monotone ramp, one-hot)
// through the device path so scale-index and nibble-lane bugs stay local.

#include "qgemm/correctness.hpp"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/ref_gemm.hpp"
#include "qgemm/timing.cuh"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace qgemm;

static int g_fail = 0;
#define EXPECT(c, msg)                                                        \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", msg);                                \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void expect_compare(const CompareResult& r, const char* tag) {
  std::fprintf(stderr, "%s  max_abs=%.4g  max_rel=%.4g  fails=%zu\n", tag,
               r.max_abs_err, r.max_rel_err, r.num_fail);
  if (!r.ok) {
    std::fprintf(stderr, "FAIL: %s: %s\n", tag, r.message.c_str());
    ++g_fail;
  }
}

static void run_gpu(QFormat fmt, const uint8_t* packed, size_t packed_bytes,
                    const uint16_t* s16, size_t s16_n, const uint8_t* s8,
                    size_t s8_n, float tensor, const float* X, float* Y, int M,
                    int N, int K, int group) {
  uint8_t* d_pk = nullptr;
  uint16_t* d16 = nullptr;
  uint8_t* d8 = nullptr;
  float *dX = nullptr, *dY = nullptr;
  QG_CHECK(cudaMalloc(&d_pk, packed_bytes));
  QG_CHECK(cudaMemcpy(d_pk, packed, packed_bytes, cudaMemcpyHostToDevice));
  if (s16_n) {
    QG_CHECK(cudaMalloc(&d16, s16_n * sizeof(uint16_t)));
    QG_CHECK(cudaMemcpy(d16, s16, s16_n * sizeof(uint16_t),
                        cudaMemcpyHostToDevice));
  }
  if (s8_n) {
    QG_CHECK(cudaMalloc(&d8, s8_n));
    QG_CHECK(cudaMemcpy(d8, s8, s8_n, cudaMemcpyHostToDevice));
  }
  QG_CHECK(cudaMalloc(&dX, static_cast<size_t>(M) * K * sizeof(float)));
  QG_CHECK(cudaMalloc(&dY, static_cast<size_t>(M) * N * sizeof(float)));
  QG_CHECK(cudaMemcpy(dX, X, static_cast<size_t>(M) * K * sizeof(float),
                      cudaMemcpyHostToDevice));
  const int MN = M * N;
  const int block = 128;
  const int grid = (MN + block - 1) / block;
  k_qgemm_ref<<<grid, block>>>(dX, d_pk, d16, d8, tensor, dY, M, N, K, group,
                               fmt);
  QG_CHECK(cudaDeviceSynchronize());
  QG_CHECK(cudaMemcpy(Y, dY, static_cast<size_t>(M) * N * sizeof(float),
                      cudaMemcpyDeviceToHost));
  cudaFree(d_pk);
  cudaFree(d16);
  cudaFree(d8);
  cudaFree(dX);
  cudaFree(dY);
}

// X varies along K so a swapped group scale cannot hide in a uniform reduction.
static void fill_structured_x(std::vector<float>& X, int M, int K) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      X[static_cast<size_t>(m) * K + k] =
          static_cast<float>((k + 3 * m) % 11) + 0.25f;
    }
  }
}

static void test_int4_prime_scale() {
  const int N = 8, K = 256, M = 2, group = 128;  // two groups / row
  auto probe = make_int4_prime_probe(static_cast<size_t>(N),
                                     static_cast<size_t>(K), group);
  std::vector<float> X(static_cast<size_t>(M) * K);
  fill_structured_x(X, M, K);
  auto Yh = gemm_fp32_ref(X, probe.expected, M, N, K);
  std::vector<float> Yg(static_cast<size_t>(M) * N);
  run_gpu(QFormat::Int4, probe.packed.data.data(), probe.packed.data.size(),
          probe.packed.scales.data(), probe.packed.scales.size(), nullptr, 0,
          1.f, X.data(), Yg.data(), M, N, K, group);
  // Expected is independent of host dequant: catches scale-index bugs.
  expect_compare(compare_close(Yg.data(), Yh.data(),
                               static_cast<size_t>(M) * N, 1e-4f, 1e-5f),
                 "int4_prime_scale");
}

static void test_int4_ramp() {
  const int N = 8, K = 128, M = 2;
  auto W = probe_monotone_ramp(static_cast<size_t>(N), static_cast<size_t>(K),
                               0.05f);
  PackedInt4 p = pack_int4(W.data(), N, K, 128);
  auto Wdq = dequant_reference_int4(p);
  std::vector<float> X(static_cast<size_t>(M) * K);
  fill_structured_x(X, M, K);
  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);
  std::vector<float> Yg(static_cast<size_t>(M) * N);
  run_gpu(QFormat::Int4, p.data.data(), p.data.size(), p.scales.data(),
          p.scales.size(), nullptr, 0, 1.f, X.data(), Yg.data(), M, N, K,
          p.group);
  expect_compare(compare_close(Yg.data(), Yh.data(),
                               static_cast<size_t>(M) * N, 1e-4f, 1e-5f),
                 "int4_ramp");
}

static void test_mxfp4_onehot() {
  const int N = 8, K = 64, M = 2;
  struct Case {
    size_t n0, k0;
    uint8_t code, e8;
    const char* tag;
  };
  // Even/odd k (low/high nibble), first/last, and a block boundary.
  const Case cases[] = {
      {0, 0, 0x4, 127, "mx_onehot_00"},
      {3, 17, 0x5, 128, "mx_onehot_odd_k"},
      {1, 32, 0x2, 129, "mx_onehot_block"},
      {static_cast<size_t>(N - 1), static_cast<size_t>(K - 1), 0x6, 127,
       "mx_onehot_last"},
  };
  for (const Case& c : cases) {
    auto probe = make_mxfp4_one_hot_probe(static_cast<size_t>(N),
                                          static_cast<size_t>(K), c.n0, c.k0,
                                          c.code, c.e8);
    std::vector<float> X(static_cast<size_t>(M) * K, 0.f);
    X[0 * K + c.k0] = 1.f;
    X[1 * K + c.k0] = 2.5f;
    auto Yh = gemm_fp32_ref(X, probe.expected, M, N, K);
    std::vector<float> Yg(static_cast<size_t>(M) * N);
    run_gpu(QFormat::Mxfp4, probe.packed.data.data(), probe.packed.data.size(),
            nullptr, 0, probe.packed.scales.data(), probe.packed.scales.size(),
            1.f, X.data(), Yg.data(), M, N, K, 32);
    expect_compare(compare_close(Yg.data(), Yh.data(),
                                 static_cast<size_t>(M) * N, 1e-5f, 1e-6f),
                   c.tag);
    const float w = probe.expected[c.n0 * static_cast<size_t>(K) + c.k0];
    EXPECT(std::fabs(Yg[c.n0] - 1.f * w) < 1e-5f, c.tag);
    EXPECT(std::fabs(Yg[static_cast<size_t>(N) + c.n0] - 2.5f * w) < 1e-5f,
           c.tag);
  }
}

// group=128 requires K % 128 == 0. K=64 would pass only with group 64.
// Use 96 (neither 64 nor 128) and 160 (64-aligned, not 128-aligned).
static void test_int4_pack_rejects_bad_k() {
  const int N = 4;
  for (int K : {96, 160}) {
    bool threw = false;
    try {
      std::vector<float> dummy(static_cast<size_t>(N) * K, 0.1f);
      (void)pack_int4(dummy.data(), N, K, /*group=*/128);
    } catch (const std::exception&) {
      threw = true;
    }
    char buf[80];
    std::snprintf(buf, sizeof(buf),
                  "INT4-g128 must reject K=%d (not multiple of 128)", K);
    EXPECT(threw, buf);
  }
}

int main() {
  test_int4_prime_scale();
  test_int4_ramp();
  test_mxfp4_onehot();
  test_int4_pack_rejects_bad_k();
  if (g_fail) {
    std::fprintf(stderr, "test_qgemv_probes: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_qgemv_probes: OK\n");
  return 0;
}
