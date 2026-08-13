// A-fragment packed MX fused-ref == sequential fused-ref == host GEMM.
#include "qgemm/a_fragment.hpp"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/ref_gemm.hpp"
#include "qgemm/timing.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace qgemm;
using namespace qgemm::wgmma;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void run_case(int M, int N, int K, const char* tag) {
  std::vector<float> W(static_cast<size_t>(N) * static_cast<size_t>(K));
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = 0.06f * static_cast<float>((i * 11) % 13) - 0.28f;
  PackedMxfp4 seq = pack_mxfp4(W.data(), static_cast<size_t>(N),
                               static_cast<size_t>(K));
  PackedMxfp4AFrag af =
      pack_mxfp4_a_fragment(W.data(), static_cast<size_t>(N),
                            static_cast<size_t>(K));
  auto Wdq = dequant_reference_mxfp4(seq);
  std::vector<float> X(static_cast<size_t>(M) * static_cast<size_t>(K));
  for (int i = 0; i < M * K; ++i)
    X[static_cast<size_t>(i)] = 0.15f * static_cast<float>((i % 6) - 2);

  auto Yh = gemm_fp32_ref(X, Wdq, M, N, K);

  uint8_t *dpk = nullptr, *dsc = nullptr;
  unsigned* dw = nullptr;
  float *dX = nullptr, *dYs = nullptr, *dYa = nullptr;
  QG_CHECK(cudaMalloc(&dpk, seq.data.size()));
  QG_CHECK(cudaMalloc(&dsc, seq.scales.size()));
  QG_CHECK(cudaMalloc(&dw, af.words.size() * sizeof(unsigned)));
  QG_CHECK(cudaMalloc(&dX, X.size() * sizeof(float)));
  const size_t MN = static_cast<size_t>(M) * static_cast<size_t>(N);
  QG_CHECK(cudaMalloc(&dYs, MN * sizeof(float)));
  QG_CHECK(cudaMalloc(&dYa, MN * sizeof(float)));
  QG_CHECK(cudaMemcpy(dpk, seq.data.data(), seq.data.size(),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dsc, seq.scales.data(), seq.scales.size(),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dw, af.words.data(), af.words.size() * sizeof(unsigned),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dX, X.data(), X.size() * sizeof(float),
                      cudaMemcpyHostToDevice));

  k_qgemm_ref<<<(static_cast<int>(MN) + 127) / 128, 128>>>(
      dX, dpk, nullptr, dsc, 1.f, dYs, M, N, K, 32, QFormat::Mxfp4);
  k_qgemm_ref_mx_a_fragment<<<(static_cast<int>(MN) + 127) / 128, 128>>>(
      dX, dw, dsc, dYa, M, N, K);
  QG_CHECK(cudaDeviceSynchronize());

  std::vector<float> ys(MN), ya(MN);
  QG_CHECK(cudaMemcpy(ys.data(), dYs, MN * sizeof(float), cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(ya.data(), dYa, MN * sizeof(float), cudaMemcpyDeviceToHost));

  double max_sa = 0, max_ah = 0;
  for (size_t i = 0; i < MN; ++i) {
    max_sa = std::max(max_sa, std::fabs(static_cast<double>(ys[i] - ya[i])));
    max_ah = std::max(max_ah, std::fabs(static_cast<double>(ya[i] - Yh[i])));
  }
  std::fprintf(stderr, "%s  afrag vs seq max_abs=%.4g  afrag vs host=%.4g\n",
               tag, max_sa, max_ah);
  EXPECT(max_sa < 1e-5, "A-fragment fused-ref == sequential");
  EXPECT(max_ah < 1e-4, "A-fragment fused-ref == host");

  cudaFree(dpk);
  cudaFree(dsc);
  cudaFree(dw);
  cudaFree(dX);
  cudaFree(dYs);
  cudaFree(dYa);
}

int main() {
  run_case(2, 64, 32, "64x32");
  run_case(1, 64, 2880, "64x2880 M=1");
  if (g_fail) {
    std::fprintf(stderr, "test_a_fragment_gpu: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_a_fragment_gpu: OK\n");
  return 0;
}
