// Dual-nibble even/odd + split-K == sequential fused-ref.
#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/timing.cuh"

#include <algorithm>
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

int main() {
  const int N = 16, K = 128, M = 2, SK = 4;
  std::vector<float> W(N * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.05f * ((i * 9) % 11) - 0.22f;
  PackedMxfp4 seq = pack_mxfp4(W.data(), N, K);
  PackedMxfp4Eo eo = pack_mxfp4_even_odd(W.data(), N, K);
  std::vector<float> X(M * K);
  for (int i = 0; i < M * K; ++i) X[i] = 0.2f * ((i % 5) - 1);

  uint8_t *dpk = nullptr, *dsc = nullptr;
  unsigned* dwo = nullptr;
  float *dX = nullptr, *dY0 = nullptr, *dYsk = nullptr;
  QG_CHECK(cudaMalloc(&dpk, seq.data.size()));
  QG_CHECK(cudaMalloc(&dsc, seq.scales.size()));
  QG_CHECK(cudaMalloc(&dwo, eo.words.size() * sizeof(unsigned)));
  QG_CHECK(cudaMalloc(&dX, M * K * sizeof(float)));
  QG_CHECK(cudaMalloc(&dY0, M * N * sizeof(float)));
  QG_CHECK(cudaMalloc(&dYsk, M * N * sizeof(float)));
  QG_CHECK(cudaMemcpy(dpk, seq.data.data(), seq.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dsc, seq.scales.data(), seq.scales.size(),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dwo, eo.words.data(), eo.words.size() * sizeof(unsigned),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dX, X.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

  const int MN = M * N;
  k_qgemm_ref<<<(MN + 127) / 128, 128>>>(dX, dpk, nullptr, dsc, 1.f, dY0, M, N,
                                          K, 32, QFormat::Mxfp4);
  QG_CHECK(cudaMemset(dYsk, 0, M * N * sizeof(float)));
  dim3 grid((MN + 127) / 128, SK);
  k_qgemm_ref_mx_even_odd_dual_splitk<<<grid, 128>>>(dX, dwo, dsc, dYsk, M, N,
                                                     K, SK);
  QG_CHECK(cudaDeviceSynchronize());

  std::vector<float> y0(MN), ysk(MN);
  QG_CHECK(cudaMemcpy(y0.data(), dY0, MN * sizeof(float), cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(ysk.data(), dYsk, MN * sizeof(float), cudaMemcpyDeviceToHost));
  double maxd = 0;
  for (int i = 0; i < MN; ++i)
    maxd = std::max(maxd, std::fabs(static_cast<double>(y0[i] - ysk[i])));
  std::fprintf(stderr, "dual+splitk vs seq max_abs=%.4g\n", maxd);
  EXPECT(maxd < 1e-5, "dual+split-K matches sequential fused-ref");

  cudaFree(dpk);
  cudaFree(dsc);
  cudaFree(dwo);
  cudaFree(dX);
  cudaFree(dY0);
  cudaFree(dYsk);
  if (g_fail) {
    std::fprintf(stderr, "test_dual_splitk: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_dual_splitk: OK\n");
  return 0;
}
