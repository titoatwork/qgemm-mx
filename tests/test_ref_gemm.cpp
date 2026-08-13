// Host FP32 reference GEMM + dequant-then-GEMM path for INT4/MXFP4.
#include "qgemm/correctness.hpp"
#include "qgemm/pack.hpp"
#include "qgemm/ref_gemm.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qgemm;

static int g_fail = 0;

#define EXPECT(cond, msg)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL: %s\n", msg);                                \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void test_identity_like() {
  // M=2,N=2,K=2 hand product.
  const float X[] = {1, 2, 3, 4};      // rows
  const float W[] = {5, 6, 7, 8};      // W rows = N
  // Y[0,0] = 1*5+2*6 = 17
  // Y[0,1] = 1*7+2*8 = 23
  // Y[1,0] = 3*5+4*6 = 39
  // Y[1,1] = 3*7+4*8 = 53
  float Y[4];
  gemm_fp32_ref(X, W, Y, 2, 2, 2);
  EXPECT(std::fabs(Y[0] - 17.f) < 1e-5f, "Y00");
  EXPECT(std::fabs(Y[1] - 23.f) < 1e-5f, "Y01");
  EXPECT(std::fabs(Y[2] - 39.f) < 1e-5f, "Y10");
  EXPECT(std::fabs(Y[3] - 53.f) < 1e-5f, "Y11");
}

static void test_int4_dequant_gemm_matches_fp_weights() {
  const std::size_t M = 3, N = 8, K = 128;  // K multiple of group
  std::vector<float> W_fp(N * K);
  // Simple structured weights so packing is stable.
  for (std::size_t n = 0; n < N; ++n)
    for (std::size_t k = 0; k < K; ++k)
      W_fp[n * K + k] = 0.1f * static_cast<float>((n + k) % 7) -
                        0.3f;

  PackedInt4 packed = pack_int4(W_fp.data(), N, K, /*group=*/128);
  std::vector<float> W_dq = dequant_reference_int4(packed);

  std::vector<float> X(M * K, 0.f);
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t k = 0; k < K; ++k)
      X[m * K + k] = ((k + m) % 5) * 0.25f;

  auto Y_ref = gemm_fp32_ref(X, W_dq, M, N, K);
  auto Y_fp = gemm_fp32_ref(X, W_fp, M, N, K);

  // Dequant is lossy; bound relative to scale of products.
  double max_abs = 0.0, max_err = 0.0;
  for (std::size_t i = 0; i < M * N; ++i) {
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(Y_fp[i])));
    max_err = std::max(max_err, std::fabs(static_cast<double>(Y_ref[i] - Y_fp[i])));
  }
  // Loose but catches layout disasters; INT4 quant error accumulates over K.
  const double tol = std::max(1e-2, 0.15 * max_abs + 1e-3 * static_cast<double>(K));
  EXPECT(max_err <= tol, "int4 dequant-GEMM vs FP weights within loose tol");
  std::fprintf(stderr, "int4 dequant-GEMM max_err=%.4g max_abs=%.4g tol=%.4g\n",
               max_err, max_abs, tol);
}

int main() {
  test_identity_like();
  test_int4_dequant_gemm_matches_fp_weights();
  if (g_fail) {
    std::fprintf(stderr, "test_ref_gemm: %d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("test_ref_gemm: OK\n");
  return 0;
}
