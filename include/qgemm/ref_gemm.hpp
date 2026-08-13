// ref_gemm.hpp — host FP32 reference for Y = X @ Wᵀ (row-major).
//
// Used to score dequantized weights and (later) fused kernels.
// Not optimized; correctness only.

#pragma once

#include <cstddef>
#include <vector>

namespace qgemm {

// X: [M,K] row-major, W: [N,K] row-major, Y: [M,N] row-major.
// Y[m,n] = sum_k X[m,k] * W[n,k]
inline void gemm_fp32_ref(const float* X, const float* W, float* Y,
                          std::size_t M, std::size_t N, std::size_t K) {
  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k) {
        acc += static_cast<double>(X[m * K + k]) *
               static_cast<double>(W[n * K + k]);
      }
      Y[m * N + n] = static_cast<float>(acc);
    }
  }
}

inline std::vector<float> gemm_fp32_ref(const std::vector<float>& X,
                                        const std::vector<float>& W,
                                        std::size_t M, std::size_t N,
                                        std::size_t K) {
  std::vector<float> Y(M * N, 0.f);
  gemm_fp32_ref(X.data(), W.data(), Y.data(), M, N, K);
  return Y;
}

}  // namespace qgemm
