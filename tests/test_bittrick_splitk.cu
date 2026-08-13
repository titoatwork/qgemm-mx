// Bittrick INT4 decode == table decode; split-K fused-ref == single-pass.
#include "qgemm/dequant_device.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/ref_gemm.hpp"
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

__global__ void k_bittrick_table(int* mismatch) {
  const int u = static_cast<int>(threadIdx.x);
  if (u >= 16) return;
  const float a = dev::int4_from_nibble(static_cast<uint8_t>(u));
  const float b = dev::int4_from_nibble_bittrick(static_cast<uint8_t>(u));
  if (a != b) atomicAdd(mismatch, 1);
}

static void test_bittrick() {
  int* d = nullptr;
  QG_CHECK(cudaMalloc(&d, sizeof(int)));
  QG_CHECK(cudaMemset(d, 0, sizeof(int)));
  k_bittrick_table<<<1, 16>>>(d);
  int h = -1;
  QG_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d);
  EXPECT(h == 0, "INT4 bittrick == table for all 16 nibbles");
  std::fprintf(stderr, "bittrick mismatches=%d\n", h);
}

static void test_splitk_mx() {
  const int N = 16, K = 128, M = 2, SK = 4;
  std::vector<float> W(N * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.07f * ((i * 5) % 11) - 0.25f;
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);
  auto Wdq = dequant_reference_mxfp4(p);
  std::vector<float> X(M * K);
  for (int i = 0; i < M * K; ++i) X[i] = 0.2f * ((i % 5) - 2);

  uint8_t *dpk = nullptr, *d8 = nullptr;
  float *dX = nullptr, *dY0 = nullptr, *dYsk = nullptr;
  QG_CHECK(cudaMalloc(&dpk, p.data.size()));
  QG_CHECK(cudaMalloc(&d8, p.scales.size()));
  QG_CHECK(cudaMalloc(&dX, M * K * sizeof(float)));
  QG_CHECK(cudaMalloc(&dY0, M * N * sizeof(float)));
  QG_CHECK(cudaMalloc(&dYsk, M * N * sizeof(float)));
  QG_CHECK(cudaMemcpy(dpk, p.data.data(), p.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(d8, p.scales.data(), p.scales.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dX, X.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

  const int MN = M * N;
  k_qgemm_ref<<<(MN + 127) / 128, 128>>>(dX, dpk, nullptr, d8, 1.f, dY0, M, N,
                                          K, 32, QFormat::Mxfp4);
  QG_CHECK(cudaMemset(dYsk, 0, M * N * sizeof(float)));
  dim3 grid((MN + 127) / 128, SK);
  k_qgemm_ref_splitk<<<grid, 128>>>(dX, dpk, nullptr, d8, 1.f, dYsk, M, N, K,
                                    32, QFormat::Mxfp4, SK);
  QG_CHECK(cudaDeviceSynchronize());

  std::vector<float> y0(MN), ysk(MN);
  QG_CHECK(cudaMemcpy(y0.data(), dY0, MN * sizeof(float), cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(ysk.data(), dYsk, MN * sizeof(float), cudaMemcpyDeviceToHost));
  double maxd = 0;
  for (int i = 0; i < MN; ++i)
    maxd = std::max(maxd, std::fabs(static_cast<double>(y0[i] - ysk[i])));
  std::fprintf(stderr, "splitk vs single max_abs=%.4g\n", maxd);
  EXPECT(maxd < 1e-5, "split-K matches single-pass (FP32, aligned groups)");

  cudaFree(dpk);
  cudaFree(d8);
  cudaFree(dX);
  cudaFree(dY0);
  cudaFree(dYsk);
}

int main() {
  test_bittrick();
  test_splitk_mx();
  if (g_fail) {
    std::fprintf(stderr, "test_bittrick_splitk: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_bittrick_splitk: OK\n");
  return 0;
}
