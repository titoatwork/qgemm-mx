// Dual-nibble even/odd fused-ref == sequential fused-ref == host GEMM.
#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/ref_gemm.hpp"
#include "qgemm/timing.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace qgemm;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void test_mx_dual() {
  const int N = 24, K = 64, M = 3;
  std::vector<float> W(N * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.06f * ((i * 11) % 13) - 0.28f;
  PackedMxfp4 seq = pack_mxfp4(W.data(), N, K);
  PackedMxfp4Eo eo = pack_mxfp4_even_odd(W.data(), N, K);
  auto Wdq = dequant_reference_mxfp4(seq);
  std::vector<float> X(M * K);
  for (int i = 0; i < M * K; ++i) X[i] = 0.15f * ((i % 6) - 2);

  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);

  uint8_t *dpk = nullptr, *dsc = nullptr;
  unsigned* dwo = nullptr;
  float *dX = nullptr, *dYs = nullptr, *dYd = nullptr;
  QG_CHECK(cudaMalloc(&dpk, seq.data.size()));
  QG_CHECK(cudaMalloc(&dsc, seq.scales.size()));
  QG_CHECK(cudaMalloc(&dwo, eo.words.size() * sizeof(unsigned)));
  QG_CHECK(cudaMalloc(&dX, M * K * sizeof(float)));
  QG_CHECK(cudaMalloc(&dYs, M * N * sizeof(float)));
  QG_CHECK(cudaMalloc(&dYd, M * N * sizeof(float)));
  QG_CHECK(cudaMemcpy(dpk, seq.data.data(), seq.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dsc, seq.scales.data(), seq.scales.size(),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dwo, eo.words.data(), eo.words.size() * sizeof(unsigned),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dX, X.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

  const int MN = M * N;
  k_qgemm_ref<<<(MN + 127) / 128, 128>>>(dX, dpk, nullptr, dsc, 1.f, dYs, M, N,
                                          K, 32, QFormat::Mxfp4);
  k_qgemm_ref_mx_even_odd_dual<<<(MN + 127) / 128, 128>>>(dX, dwo, dsc, dYd, M,
                                                          N, K);
  QG_CHECK(cudaDeviceSynchronize());

  std::vector<float> ys(MN), yd(MN);
  QG_CHECK(cudaMemcpy(ys.data(), dYs, MN * sizeof(float), cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(yd.data(), dYd, MN * sizeof(float), cudaMemcpyDeviceToHost));

  double max_sd = 0, max_dh = 0;
  for (int i = 0; i < MN; ++i) {
    max_sd = std::max(max_sd, std::fabs(static_cast<double>(ys[i] - yd[i])));
    max_dh = std::max(max_dh, std::fabs(static_cast<double>(yd[i] - Yh[i])));
  }
  std::fprintf(stderr, "dual vs seq max_abs=%.4g  dual vs host=%.4g\n", max_sd,
               max_dh);
  EXPECT(max_sd < 1e-5, "dual-nibble even/odd == sequential fused-ref");
  EXPECT(max_dh < 1e-4, "dual-nibble == host dequant+GEMM");

  cudaFree(dpk);
  cudaFree(dsc);
  cudaFree(dwo);
  cudaFree(dX);
  cudaFree(dYs);
  cudaFree(dYd);
}

int main() {
  test_mx_dual();
  if (g_fail) {
    std::fprintf(stderr, "test_dual_nibble: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_dual_nibble: OK\n");
  return 0;
}
