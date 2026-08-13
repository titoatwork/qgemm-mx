// Sequential fused-ref vs even/odd dual-nibble fused-ref (same math).
// Not SOTA — only asks: does the layout+dual convert pay anything on sm_86?
#include "qgemm/formats.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/timing.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qgemm;

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  const double peak = argc > 2 ? atof(argv[2]) : 0.0;
  const int N = argc > 3 ? atoi(argv[3]) : 256;
  const int K = argc > 4 ? atoi(argv[4]) : 256;
  const int max_M = argc > 5 ? atoi(argv[5]) : 4;
  QG_CHECK(cudaSetDevice(dev));
  if (peak <= 0) {
    std::printf("usage: %s <dev> <gbps> [N K max_M]\n", argv[0]);
    return 2;
  }

  std::vector<float> W((size_t)N * K);
  for (size_t i = 0; i < W.size(); ++i)
    W[i] = 0.04f * float((i * 7) % 11) - 0.2f;
  PackedMxfp4 seq = pack_mxfp4(W.data(), N, K);
  PackedMxfp4Eo eo = pack_mxfp4_even_odd(W.data(), N, K);

  uint8_t *dpk = nullptr, *dsc = nullptr;
  unsigned* dwo = nullptr;
  QG_CHECK(cudaMalloc(&dpk, seq.data.size()));
  QG_CHECK(cudaMalloc(&dsc, seq.scales.size()));
  QG_CHECK(cudaMalloc(&dwo, eo.words.size() * sizeof(unsigned)));
  QG_CHECK(cudaMemcpy(dpk, seq.data.data(), seq.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dsc, seq.scales.data(), seq.scales.size(),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dwo, eo.words.data(), eo.words.size() * sizeof(unsigned),
                      cudaMemcpyHostToDevice));

  const size_t wbytes = weight_bytes(Format::MXFP4, N, K);
  std::printf("path,M,N,K,graphed_us,pct_of_ideal_mx\n");

  const int Ms[] = {1, 2, 4, 8};
  for (int M : Ms) {
    if (M > max_M) break;
    float *dX = nullptr, *dY = nullptr;
    QG_CHECK(cudaMalloc(&dX, (size_t)M * K * sizeof(float)));
    QG_CHECK(cudaMalloc(&dY, (size_t)M * N * sizeof(float)));
    QG_CHECK(cudaMemset(dX, 0, (size_t)M * K * sizeof(float)));
    const int MN = M * N;

    auto seqk = [&](int, cudaStream_t s) {
      k_qgemm_ref<<<(MN + 127) / 128, 128, 0, s>>>(dX, dpk, nullptr, dsc, 1.f,
                                                   dY, M, N, K, 32,
                                                   QFormat::Mxfp4);
    };
    auto dual = [&](int, cudaStream_t s) {
      k_qgemm_ref_mx_even_odd_dual<<<(MN + 127) / 128, 128, 0, s>>>(
          dX, dwo, dsc, dY, M, N, K);
    };
    const auto ts = time_graphed(seqk, 6, 16, 4);
    const auto td = time_graphed(dual, 6, 16, 4);
    std::printf("seq_ref,%d,%d,%d,%.4f,%.4f\n", M, N, K, ts.median_us,
                pct_of_ideal(ts.median_us, wbytes, peak));
    std::printf("dual_eo,%d,%d,%d,%.4f,%.4f\n", M, N, K, td.median_us,
                pct_of_ideal(td.median_us, wbytes, peak));
    std::fprintf(stderr, "M=%d  seq=%.2f us  dual=%.2f us  ratio=%.2f\n", M,
                 ts.median_us, td.median_us, ts.median_us / td.median_us);
    cudaFree(dX);
    cudaFree(dY);
  }
  cudaFree(dpk);
  cudaFree(dsc);
  cudaFree(dwo);
  return 0;
}
