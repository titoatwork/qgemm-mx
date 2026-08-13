// GPU fused-ref kernel vs host dequant+GEMM. Structured probes + random.
#include "qgemm/correctness.hpp"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/ref_gemm.hpp"
#include "qgemm/timing.cuh"

#include <cmath>
#include <cstdio>
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

static void compare_vec(const char* tag, const std::vector<float>& a,
                        const std::vector<float>& b, double atol, double rtol) {
  double max_abs = 0.0, max_rel = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(static_cast<double>(a[i] - b[i]));
    const double rel = d / (1e-12 + std::fabs(static_cast<double>(b[i])));
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
    if (rel > max_rel) max_rel = rel;
  }
  std::fprintf(stderr, "%s  n=%zu  max_abs=%.4g  max_rel=%.4g  worst=%zu a=%.6g b=%.6g\n",
               tag, a.size(), max_abs, max_rel, worst,
               a.empty() ? 0.0 : a[worst], b.empty() ? 0.0 : b[worst]);
  EXPECT(max_abs <= atol || max_rel <= rtol, tag);
}

static void run_gpu(QFormat fmt, const uint8_t* packed, size_t packed_bytes,
                    const uint16_t* s16, size_t s16_n, const uint8_t* s8,
                    size_t s8_n, float tensor, const float* X, float* Y, int M,
                    int N, int K, int group) {
  uint8_t *d_pk = nullptr;
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

static void test_int4_onehot() {
  const int N = 16, K = 128, M = 2;
  auto W = probe_one_hot(N, K, 3, 17, 2.0f);
  PackedInt4 p = pack_int4(W.data(), N, K, 128);
  auto Wdq = dequant_reference_int4(p);
  std::vector<float> X(M * K, 0.f);
  X[0 * K + 17] = 1.f;
  X[1 * K + 17] = 3.f;
  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);
  std::vector<float> Yg(M * N);
  run_gpu(QFormat::Int4, p.data.data(), p.data.size(), p.scales.data(),
          p.scales.size(), nullptr, 0, 1.f, X.data(), Yg.data(), M, N, K,
          p.group);
  compare_vec("int4_onehot", Yg, Yh, 1e-4, 1e-4);
}

static void test_mx_random() {
  const int N = 32, K = 64, M = 4;
  std::vector<float> W(N * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.05f * ((i * 17) % 13) - 0.3f;
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);
  auto Wdq = dequant_reference_mxfp4(p);
  std::vector<float> X(M * K);
  for (int i = 0; i < M * K; ++i) X[i] = 0.1f * ((i * 3) % 7);
  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);
  std::vector<float> Yg(M * N);
  run_gpu(QFormat::Mxfp4, p.data.data(), p.data.size(), nullptr, 0,
          p.scales.data(), p.scales.size(), 1.f, X.data(), Yg.data(), M, N, K,
          32);
  // Device and host use the same decode; should be very tight (FP32 acc).
  compare_vec("mxfp4_random", Yg, Yh, 1e-4, 1e-5);
}

static void test_nv_random() {
  const int N = 16, K = 64, M = 1;
  std::vector<float> W(N * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.08f * ((i * 11) % 9) - 0.2f;
  PackedNvfp4 p = pack_nvfp4(W.data(), N, K, 1.25f);
  auto Wdq = dequant_reference_nvfp4(p);
  std::vector<float> X(M * K, 0.5f);
  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);
  std::vector<float> Yg(M * N);
  run_gpu(QFormat::Nvfp4, p.data.data(), p.data.size(), nullptr, 0,
          p.scales.data(), p.scales.size(), p.tensor_scale, X.data(), Yg.data(),
          M, N, K, 16);
  compare_vec("nvfp4_random", Yg, Yh, 1e-4, 1e-5);
}

static void test_gptoss_align() {
  // 2880 is the production pain shape; small M, modest N slice for time.
  const int N = 96, K = 2880, M = 1;  // 2880 % 32 == 0, % 128 == 0? 2880/128=22.5 NO
  // K=2880 is NOT divisible by 128 — INT4-g128 pack should reject.
  bool threw = false;
  try {
    std::vector<float> dummy(static_cast<size_t>(N) * K, 0.1f);
    (void)pack_int4(dummy.data(), N, K, 128);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT(threw, "INT4-g128 must reject K=2880 (not multiple of 128)");

  // MXFP4 must accept 2880.
  std::vector<float> W(static_cast<size_t>(N) * K, 0.15f);
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);
  EXPECT(p.K == 2880, "mx pack K");
  auto Wdq = dequant_reference_mxfp4(p);
  std::vector<float> X(M * K, 1.f);
  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);
  std::vector<float> Yg(M * N);
  run_gpu(QFormat::Mxfp4, p.data.data(), p.data.size(), nullptr, 0,
          p.scales.data(), p.scales.size(), 1.f, X.data(), Yg.data(), M, N, K,
          32);
  compare_vec("mx_k2880", Yg, Yh, 1e-3, 1e-5);
}

int main() {
  test_int4_onehot();
  test_mx_random();
  test_nv_random();
  test_gptoss_align();
  if (g_fail) {
    std::fprintf(stderr, "test_qgemv_gpu: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_qgemv_gpu: OK\n");
  return 0;
}
