// bench_waterfall.cu — three paths, one shape, one CSV (sm_86 lab).
//
//   stream          packed-weight read (t_ideal traffic)
//   dequant+cublas  real pack → device dequant → FP16 GEMM
//   fused-ref       scalar fused qgemv_ref (correct-slow)
//
// Not a SOTA claim. This is the local waterfall step list so R1 has a
// template when H100 numbers exist.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "qgemm/dequant_device.cuh"
#include "qgemm/formats.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/qgemv_ref.cuh"
#include "qgemm/timing.cuh"

using namespace qgemm;

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _st = (expr);                                              \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[qgemm] cublas %d\n", (int)_st);                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void k_stream(const float4* in, float* sink, size_t n4) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.f;
  if (i < n4) {
    float4 v = in[i];
    acc = v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5f) *sink = acc;
}

__global__ void k_deq_mx(const uint8_t* pk, const uint8_t* sc, __half* W,
                         int N, int K) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_elem = static_cast<size_t>(N) * static_cast<size_t>(K);
  if (i >= n_elem) return;
  const size_t n = i / static_cast<size_t>(K);
  const size_t k = i - n * static_cast<size_t>(K);
  W[i] = __float2half(dev::dequant_mxfp4_elem(pk, sc, n, k, N, K));
}

__global__ void fill_half(__half* p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2half(v);
}

int main(int argc, char** argv) {
  const int dev = argc > 1 ? atoi(argv[1]) : 0;
  const double peak = argc > 2 ? atof(argv[2]) : 0.0;
  const int N = argc > 3 ? atoi(argv[3]) : 256;
  const int K = argc > 4 ? atoi(argv[4]) : 256;
  const int max_M = argc > 5 ? atoi(argv[5]) : 8;
  QG_CHECK(cudaSetDevice(dev));
  if (peak <= 0) {
    std::printf("usage: %s <dev> <measured_gbps> [N K max_M]\n", argv[0]);
    return 2;
  }
  if (K % 32) {
    std::fprintf(stderr, "K must be multiple of 32 for MXFP4\n");
    return 2;
  }

  std::vector<float> Wh(static_cast<size_t>(N) * K);
  for (size_t i = 0; i < Wh.size(); ++i)
    Wh[i] = 0.05f * static_cast<float>((i * 9) % 13) - 0.2f;
  PackedMxfp4 p = pack_mxfp4(Wh.data(), N, K);

  uint8_t *dpk = nullptr, *dsc = nullptr;
  QG_CHECK(cudaMalloc(&dpk, p.data.size()));
  QG_CHECK(cudaMalloc(&dsc, p.scales.size()));
  QG_CHECK(cudaMemcpy(dpk, p.data.data(), p.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dsc, p.scales.data(), p.scales.size(),
                      cudaMemcpyHostToDevice));

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));

  const size_t wbytes = weight_bytes(Format::MXFP4, N, K);
  const size_t alloc = (wbytes + 15) & ~size_t{15};
  float4* stream_buf = nullptr;
  float* sink = nullptr;
  QG_CHECK(cudaMalloc(&stream_buf, alloc));
  QG_CHECK(cudaMalloc(&sink, 4));

  std::printf("path,M,N,K,format,graphed_us,pct_of_ideal_mx\n");
  std::fprintf(stderr, "=== waterfall MXFP4 N=%d K=%d peak=%.1f ===\n", N, K,
               peak);

  const int Ms[] = {1, 2, 4, 8, 16, 32};
  for (int M : Ms) {
    if (M > max_M) break;

    // --- stream ---
    {
      const size_t n4 = alloc / sizeof(float4);
      const unsigned grid =
          (unsigned)std::min(n4, (size_t)256);
      auto st = [&](int, cudaStream_t s) {
        k_stream<<<grid ? grid : 1, 256, 0, s>>>(stream_buf, sink, n4);
      };
      const auto t = time_graphed(st, 8, 24, 8);
      const double pct = pct_of_ideal(t.median_us, wbytes, peak);
      std::printf("stream,%d,%d,%d,mxfp4,%.4f,%.4f\n", M, N, K, t.median_us,
                  pct);
      std::fprintf(stderr, "  stream     M=%2d  %8.2f us  %6.1f%%\n", M,
                   t.median_us, pct);
    }

    // --- dequant + cublas ---
    {
      __half *W = nullptr, *X = nullptr, *Y = nullptr;
      QG_CHECK(cudaMalloc(&W, (size_t)N * K * sizeof(__half)));
      QG_CHECK(cudaMalloc(&X, (size_t)M * K * sizeof(__half)));
      QG_CHECK(cudaMalloc(&Y, (size_t)M * N * sizeof(__half)));
      fill_half<<<((M * K) + 255) / 256, 256>>>(X, (size_t)M * K, 0.02f);
      const unsigned dqg = (unsigned)(((size_t)N * K + 255) / 256);
      const float alpha = 1.f, beta = 0.f;
      auto step = [&](int, cudaStream_t s) {
        k_deq_mx<<<dqg, 256, 0, s>>>(dpk, dsc, W, N, K);
        CUBLAS_CHECK(cublasSetStream(h, s));
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                                  W, CUDA_R_16F, K, X, CUDA_R_16F, K, &beta, Y,
                                  CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
      };
      const auto t = time_graphed(step, 8, 24, 4);
      const double pct = pct_of_ideal(t.median_us, wbytes, peak);
      std::printf("dequant_cublas,%d,%d,%d,mxfp4,%.4f,%.4f\n", M, N, K,
                  t.median_us, pct);
      std::fprintf(stderr, "  deq+cublas M=%2d  %8.2f us  %6.1f%%\n", M,
                   t.median_us, pct);
      cudaFree(W);
      cudaFree(X);
      cudaFree(Y);
    }

    // --- fused-ref ---
    {
      float *X = nullptr, *Y = nullptr;
      QG_CHECK(cudaMalloc(&X, (size_t)M * K * sizeof(float)));
      QG_CHECK(cudaMalloc(&Y, (size_t)M * N * sizeof(float)));
      QG_CHECK(cudaMemset(X, 0, (size_t)M * K * sizeof(float)));
      const int MN = M * N;
      auto step = [&](int, cudaStream_t s) {
        k_qgemm_ref<<<(MN + 127) / 128, 128, 0, s>>>(
            X, dpk, nullptr, dsc, 1.f, Y, M, N, K, 32, QFormat::Mxfp4);
      };
      const auto t = time_graphed(step, 8, 24, 4);
      const double pct = pct_of_ideal(t.median_us, wbytes, peak);
      std::printf("fused_ref,%d,%d,%d,mxfp4,%.4f,%.4f\n", M, N, K, t.median_us,
                  pct);
      std::fprintf(stderr, "  fused-ref  M=%2d  %8.2f us  %6.4f%%\n", M,
                   t.median_us, pct);
      cudaFree(X);
      cudaFree(Y);
    }
  }

  cudaFree(dpk);
  cudaFree(dsc);
  cudaFree(stream_buf);
  cudaFree(sink);
  cublasDestroy(h);
  return 0;
}
