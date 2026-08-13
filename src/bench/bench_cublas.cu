// bench_cublas.cu — FP16 cuBLAS baseline, and the harness's self-validation.
//
// This runs before any custom kernel exists, and it has two jobs beyond
// producing a baseline:
//
//   1. VALIDATE THE HARNESS. cuBLAS FP16 is well understood. If the harness
//      reports something implausible here -- above the memory roofline, or a
//      bandwidth exceeding the empirical peak -- the harness is wrong, not
//      cuBLAS. Catching that now is much cheaper than catching it after a
//      custom kernel is in the picture.
//
//   2. QUANTIFY THE TWO MEASUREMENT HAZARDS on this specific machine, by
//      reporting all four combinations:
//
//                          hot (1 buffer)      cold (rotated buffers)
//        looped launches   both errors         launch overhead only
//        graphed           L2 residency only   <-- the honest number
//
//      The spread between those four is the entire argument for the protocol.
//
// Layout convention throughout the project: activations X are row-major [M,K],
// weights W are row-major [N,K], output Y is row-major [M,N], i.e. Y = X @ W^T,
// which is how a Transformer linear layer is actually stored.

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
      std::fprintf(stderr, "[qgemm] cuBLAS error %d at %s:%d -> %s\n",        \
                   static_cast<int>(_st), __FILE__, __LINE__, #expr);         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

__global__ void fill_half(__half* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = __float2half(v);
}

int main(int argc, char** argv) {
  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  // Measured read-only bandwidth from ./bandwidth. Passed in rather than
  // hardcoded so the denominator is always an explicit, recorded choice.
  const double peak_gbps = (argc > 2) ? std::atof(argv[2]) : 0.0;

  QG_CHECK(cudaSetDevice(dev));
  const DeviceInfo d = query_device(dev);

  if (peak_gbps <= 0.0) {
    std::printf("usage: %s <device> <measured_read_gbps>\n", argv[0]);
    std::printf("  run ./bandwidth first and pass its read-only figure.\n");
    std::printf("  refusing to fall back to the theoretical peak -- that would\n");
    std::printf("  inflate every efficiency number this program prints.\n");
    return 2;
  }

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

  // Optional filter: argv[3] = layer name (e.g. q_o_proj) or "all" (default).
  // Optional argv[4] = max M to run (default full kMSweep).
  const char* layer_filter = (argc > 3) ? argv[3] : "all";
  const size_t max_M =
      (argc > 4) ? static_cast<size_t>(std::atoll(argv[4])) : kMSweep[kNumMSweep - 1];

  std::fprintf(stderr, "=== cuBLAS FP16 baseline + harness validation ===\n");
  std::fprintf(stderr, "device      %s (sm_%d%d, %d SMs, %.2f MB L2)\n", d.name.c_str(),
              d.cc_major, d.cc_minor, d.sm_count, d.l2_bytes / 1048576.0);
  std::fprintf(stderr, "denominator %.1f GB/s (measured read-only)\n", peak_gbps);
  std::fprintf(stderr, "layers      %s  max_M %zu\n", layer_filter, max_M);

  const TimingResult lo = measure_launch_overhead();
  std::fprintf(stderr, "launch cost %.3f us/kernel (median, non-graphed)\n\n",
              lo.median_us);

  std::printf("shape,M,N,K,rotation,graphed_cold_us,graphed_hot_us,"
              "looped_cold_us,pct_of_ideal,achieved_gbps,l2_inflation,"
              "launch_share\n");

  for (size_t li = 0; li < kNumLayerShapes; ++li) {
    const LayerShape& layer = kLayerShapes[li];
    if (std::string(layer_filter) != "all" &&
        std::string(layer_filter) != layer.name) {
      continue;
    }

    const size_t wbytes = weight_bytes(Format::FP16, layer.N, layer.K);
    const int rot = rotation_count(wbytes, d.l2_bytes, 2.0);

    // Weight buffers: `rot` distinct copies so a full rotation exceeds L2.
    std::vector<__half*> W(rot, nullptr);
    for (int i = 0; i < rot; ++i) {
      QG_CHECK(cudaMalloc(&W[i], layer.N * layer.K * sizeof(__half)));
      const size_t n = layer.N * layer.K;
      fill_half<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
          W[i], n, 0.01f * static_cast<float>(i + 1));
    }
    QG_CHECK(cudaDeviceSynchronize());

    std::fprintf(stderr, "\n%s  N=%zu K=%zu  weights %.2f MB  rotation %d buffers "
                 "(%.1f MB total vs %.1f MB L2)\n",
                 layer.name, layer.N, layer.K, wbytes / 1048576.0, rot,
                 rot * wbytes / 1048576.0, d.l2_bytes / 1048576.0);
    std::fprintf(stderr, "%6s %11s %11s %11s %9s %11s %9s %8s\n", "M",
                 "cold(us)", "hot(us)", "looped(us)", "%ideal", "GB/s",
                 "L2infl", "launch%");

    for (size_t mi = 0; mi < kNumMSweep; ++mi) {
      const size_t M = kMSweep[mi];
      if (M > max_M) break;

      __half *X = nullptr, *Y = nullptr;
      QG_CHECK(cudaMalloc(&X, M * layer.K * sizeof(__half)));
      QG_CHECK(cudaMalloc(&Y, M * layer.N * sizeof(__half)));
      {
        const size_t n = M * layer.K;
        fill_half<<<static_cast<unsigned>((n + 255) / 256), 256>>>(X, n, 0.02f);
      }
      QG_CHECK(cudaMemset(Y, 0, M * layer.N * sizeof(__half)));
      QG_CHECK(cudaDeviceSynchronize());

      const float alpha = 1.0f, beta = 0.0f;
      const int m = static_cast<int>(layer.N);
      const int n = static_cast<int>(M);
      const int k = static_cast<int>(layer.K);

      // Y_rm[M,N] = X_rm[M,K] @ W_rm[N,K]^T
      //   == C_cm[N,M] = op_T(W_cm[K,N]) * op_N(X_cm[K,M])
      auto gemm = [&](int idx, cudaStream_t st) {
        CUBLAS_CHECK(cublasSetStream(h, st));
        CUBLAS_CHECK(cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
            W[idx % static_cast<int>(W.size())], CUDA_R_16F, k,
            X, CUDA_R_16F, k, &beta, Y, CUDA_R_16F, m,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      };
      auto gemm_hot = [&](int, cudaStream_t st) { gemm(0, st); };

      // inner >= rot so one graph covers a full rotation cycle.
      const int inner = std::max(rot, 8);
      const TimingResult cold   = time_graphed(gemm,     10, 40, inner);
      const TimingResult hot    = time_graphed(gemm_hot, 10, 40, inner);
      const TimingResult looped = time_looped(gemm,      10, 40, inner);

      const size_t moved = total_bytes(Format::FP16, M, layer.N, layer.K);
      const double pct  = pct_of_ideal(cold.median_us, moved, peak_gbps);
      const double gbps = achieved_gbps(moved, cold.median_us);
      const double l2_infl = cold.median_us / hot.median_us;
      const double launch_share =
          100.0 * (looped.median_us - cold.median_us) / looped.median_us;

      std::fprintf(stderr,
                   "%6zu %11.2f %11.2f %11.2f %8.1f%% %11.1f %8.2fx %7.1f%%\n",
                   M, cold.median_us, hot.median_us, looped.median_us, pct,
                   gbps, l2_infl, launch_share);

      std::printf("%s,%zu,%zu,%zu,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.2f\n",
                  layer.name, M, layer.N, layer.K, rot, cold.median_us,
                  hot.median_us, looped.median_us, pct, gbps, l2_infl,
                  launch_share);

      cudaFree(X);
      cudaFree(Y);
    }

    for (auto* p : W) cudaFree(p);
  }

  std::fprintf(stderr,
      "\nHow to read this:\n"
      "  L2infl  > 1.2x  the hot loop was measuring cache, not HBM. Rotation is\n"
      "                  doing real work; never report the hot number alone.\n"
      "  launch%% > 30%%   dispatch dominates. Any un-graphed benchmark at this M\n"
      "                  is measuring the launcher.\n"
      "  %%ideal > 100%%  IMPOSSIBLE. The harness or the denominator is wrong --\n"
      "                  stop and fix it before trusting anything downstream.\n\n");

  cublasDestroy(h);
  return 0;
}
