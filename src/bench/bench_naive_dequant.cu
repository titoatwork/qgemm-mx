// bench_naive_dequant.cu — sanity-floor baseline (NOT a headline).
//
// Path: unpack INT4-ish packed buffer to FP16 workspace on device, then
// cuBLAS FP16 GEMM. This is the "obvious wrong way" — it moves ~4× more weight
// bytes than a fused kernel. Report as "×N vs naive" only, never as the result.
//
// Packing here is a stand-in: we allocate packed_bytes = weight_bytes(INT4)
// and dequant by expanding each byte to two FP16 values (not bit-exact GPTQ).
// Purpose is traffic + launch structure, not numerical fidelity.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qgemm/formats.cuh"
#include "qgemm/shapes.hpp"
#include "qgemm/timing.cuh"

using namespace qgemm;

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _st = (expr);                                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[qgemm] cuBLAS error %d\n", static_cast<int>(_st)); \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// Expand packed uint8 (2×INT4 per byte) into FP16 weights [N,K].
// Dummy dequant: low/high nibble -> float in [-8,7] * 0.1f (traffic-shaped).
__global__ void k_naive_dequant(const uint8_t* __restrict__ packed,
                                __half* __restrict__ W, size_t N, size_t K) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t n_elem = N * K;
  if (i >= n_elem) return;
  const size_t byte_i = i / 2;
  const uint8_t b = packed[byte_i];
  const int nib = (i & 1) ? static_cast<int>(b >> 4) : static_cast<int>(b & 0xF);
  int s = (nib & 8) ? (nib - 16) : nib;
  W[i] = __float2half(0.1f * static_cast<float>(s));
}

__global__ void fill_half(__half* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = __float2half(v);
}

int main(int argc, char** argv) {
  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  const double peak_gbps = (argc > 2) ? std::atof(argv[2]) : 0.0;
  const char* layer_filter = (argc > 3) ? argv[3] : "q_o_proj";
  const size_t max_M =
      (argc > 4) ? static_cast<size_t>(std::atoll(argv[4])) : 64;

  QG_CHECK(cudaSetDevice(dev));
  const DeviceInfo d = query_device(dev);
  if (peak_gbps <= 0.0) {
    std::printf("usage: %s <device> <measured_read_gbps> [layer] [max_M]\n",
                argv[0]);
    return 2;
  }

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));

  std::fprintf(stderr, "=== naive dequant + cuBLAS (sanity floor) ===\n");
  std::fprintf(stderr, "device %s  denominator %.1f GB/s  layer %s max_M %zu\n",
               d.name.c_str(), peak_gbps, layer_filter, max_M);

  std::printf(
      "shape,M,N,K,path,graphed_cold_us,pct_of_ideal_fp16_bytes,achieved_gbps\n");

  for (size_t li = 0; li < kNumLayerShapes; ++li) {
    const LayerShape& layer = kLayerShapes[li];
    if (std::string(layer_filter) != "all" &&
        std::string(layer_filter) != layer.name)
      continue;

    const size_t n_elem = layer.N * layer.K;
    const size_t packed_bytes = (n_elem + 1) / 2;
    const size_t w_fp16_bytes = n_elem * sizeof(__half);

    // Rotate on the FP16 workspace size (what dequant writes).
    const int rot = rotation_count(w_fp16_bytes, d.l2_bytes, 2.0);

    uint8_t* packed = nullptr;
    QG_CHECK(cudaMalloc(&packed, packed_bytes));
    QG_CHECK(cudaMemset(packed, 0x12, packed_bytes));

    std::vector<__half*> W(rot, nullptr);
    for (int i = 0; i < rot; ++i) {
      QG_CHECK(cudaMalloc(&W[i], w_fp16_bytes));
    }

    for (size_t mi = 0; mi < kNumMSweep; ++mi) {
      const size_t M = kMSweep[mi];
      if (M > max_M) break;

      __half *X = nullptr, *Y = nullptr;
      QG_CHECK(cudaMalloc(&X, M * layer.K * sizeof(__half)));
      QG_CHECK(cudaMalloc(&Y, M * layer.N * sizeof(__half)));
      fill_half<<<(unsigned)((M * layer.K + 255) / 256), 256>>>(X, M * layer.K,
                                                                0.02f);
      QG_CHECK(cudaDeviceSynchronize());

      const float alpha = 1.f, beta = 0.f;
      const int m = (int)layer.N, n = (int)M, k = (int)layer.K;
      const unsigned dq_grid =
          (unsigned)((n_elem + 255) / 256);

      auto step = [&](int idx, cudaStream_t st) {
        __half* Ww = W[idx % rot];
        k_naive_dequant<<<dq_grid, 256, 0, st>>>(packed, Ww, layer.N, layer.K);
        CUBLAS_CHECK(cublasSetStream(h, st));
        CUBLAS_CHECK(cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, Ww, CUDA_R_16F, k, X,
            CUDA_R_16F, k, &beta, Y, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
      };

      const int inner = std::max(rot, 8);
      const TimingResult cold = time_graphed(step, 10, 30, inner);

      // Report efficiency vs FP16 weight traffic (the fused ideal is lower).
      const size_t moved_fp16 = total_bytes(Format::FP16, M, layer.N, layer.K);
      const double pct = pct_of_ideal(cold.median_us, moved_fp16, peak_gbps);
      const double gbps = achieved_gbps(moved_fp16, cold.median_us);

      std::fprintf(stderr, "%s M=%zu  cold=%.2f us  %%ideal(fp16 bytes)=%.1f\n",
                   layer.name, M, cold.median_us, pct);
      std::printf("%s,%zu,%zu,%zu,naive_dequant_cublas,%.4f,%.2f,%.2f\n",
                  layer.name, M, layer.N, layer.K, cold.median_us, pct, gbps);

      cudaFree(X);
      cudaFree(Y);
    }

    for (auto* p : W) cudaFree(p);
    cudaFree(packed);
  }

  cublasDestroy(h);
  std::fprintf(stderr,
               "\nSanity floor only. Expect much worse %%ideal than fused paths "
               "once they exist — extra dequant write + re-read.\n");
  return 0;
}
