// bench_qgemv_ref.cu — time the correct-slow fused kernel (not a SOTA claim).
//
// Purpose: fused-path timing vs naive dequant+cuBLAS and vs stream ideal on
// the same shapes, so the waterfall has a real "fused but untuned" step.

#include "qgemm/formats.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/shapes.hpp"
#include "qgemm/timing.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qgemm;

int main(int argc, char** argv) {
  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  const double peak_gbps = (argc > 2) ? std::atof(argv[2]) : 0.0;
  const size_t N = (argc > 3) ? static_cast<size_t>(std::atoll(argv[3])) : 256;
  const size_t K = (argc > 4) ? static_cast<size_t>(std::atoll(argv[4])) : 256;
  const size_t max_M = (argc > 5) ? static_cast<size_t>(std::atoll(argv[5])) : 8;

  QG_CHECK(cudaSetDevice(dev));
  if (peak_gbps <= 0.0) {
    std::printf("usage: %s <dev> <measured_gbps> [N K max_M]\n", argv[0]);
    return 2;
  }

  std::vector<float> W(N * K);
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = 0.05f * static_cast<float>((i * 13) % 11) - 0.2f;
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);

  uint8_t *d_pk = nullptr, *d8 = nullptr;
  QG_CHECK(cudaMalloc(&d_pk, p.data.size()));
  QG_CHECK(cudaMalloc(&d8, p.scales.size()));
  QG_CHECK(cudaMemcpy(d_pk, p.data.data(), p.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(
      cudaMemcpy(d8, p.scales.data(), p.scales.size(), cudaMemcpyHostToDevice));

  std::printf("shape,M,N,K,format,graphed_cold_us,pct_of_ideal_mx_bytes\n");
  std::fprintf(stderr, "=== fused-ref MXFP4 GEMM (correct-slow) N=%zu K=%zu ===\n",
               N, K);

  for (size_t mi = 0; mi < kNumMSweep; ++mi) {
    const size_t M = kMSweep[mi];
    if (M > max_M) break;

    float *dX = nullptr, *dY = nullptr;
    QG_CHECK(cudaMalloc(&dX, M * K * sizeof(float)));
    QG_CHECK(cudaMalloc(&dY, M * N * sizeof(float)));
    QG_CHECK(cudaMemset(dX, 0, M * K * sizeof(float)));

    const int MN = static_cast<int>(M * N);
    const int block = 128;
    const int grid = (MN + block - 1) / block;
    auto step = [&](int, cudaStream_t st) {
      k_qgemm_ref<<<grid, block, 0, st>>>(dX, d_pk, nullptr, d8, 1.f, dY,
                                          static_cast<int>(M),
                                          static_cast<int>(N),
                                          static_cast<int>(K), 32,
                                          QFormat::Mxfp4);
    };
    const TimingResult cold = time_graphed(step, 5, 20, 4);
    const size_t bytes = weight_bytes(Format::MXFP4, N, K);
    const double pct = pct_of_ideal(cold.median_us, bytes, peak_gbps);
    std::fprintf(stderr, "M=%zu  %.2f us  %%ideal(mx bytes)=%.2f\n", M,
                 cold.median_us, pct);
    std::printf("qgemv_ref,%zu,%zu,%zu,mxfp4,%.4f,%.4f\n", M, N, K,
                cold.median_us, pct);
    cudaFree(dX);
    cudaFree(dY);
  }
  cudaFree(d_pk);
  cudaFree(d8);
  return 0;
}
