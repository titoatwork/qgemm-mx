// Prove MX dequant identities: mul == ldexp == fp16-exponent-add.
#include "qgemm/dequant_device.cuh"
#include "qgemm/timing.cuh"

#include <cmath>
#include <cstdio>

using namespace qgemm;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

__global__ void k_mx_identities(int* fail_mul, int* fail_exp, int* n_nan) {
  const int code = threadIdx.x;          // 0..15
  const int si = blockIdx.x;             // scale index
  if (code >= 16) return;
  // Scales that keep every nonzero E2M1 inside fp16 normals after exp add.
  // Extreme E8M0 is tested only for NaN (0xFF). 127±4 → 2^{±4} * {0.5..6}.
  const uint8_t scales[8] = {123, 125, 126, 127, 128, 129, 131, 0xFF};
  if (si >= 8) return;
  const uint8_t s = scales[si];
  const uint8_t c = static_cast<uint8_t>(code);

  const float mul = dev::e2m1_decode(c) * dev::e8m0_decode(s);
  const float ld = dev::mxfp4_elem_ldexp(c, s);
  const float ex = dev::mxfp4_elem_expfield(c, s);

  if (s == 0xFF) {
    if (!(isnan(mul) && isnan(ld) && isnan(ex))) atomicAdd(n_nan, 1);
    return;
  }
  if (mul != ld && !(mul == 0.f && ld == 0.f)) atomicAdd(fail_mul, 1);
  // expfield may flush underflow to 0; allow that vs tiny mul
  if (ex != ld) {
    if (!(ex == 0.f && fabsf(ld) < 1e-4f) &&
        !(isinf(ex) && isinf(ld) && signbit(ex) == signbit(ld))) {
      atomicAdd(fail_exp, 1);
    }
  }
}

int main() {
  int *a = nullptr, *b = nullptr, *c = nullptr;
  QG_CHECK(cudaMalloc(&a, 4));
  QG_CHECK(cudaMalloc(&b, 4));
  QG_CHECK(cudaMalloc(&c, 4));
  QG_CHECK(cudaMemset(a, 0, 4));
  QG_CHECK(cudaMemset(b, 0, 4));
  QG_CHECK(cudaMemset(c, 0, 4));
  k_mx_identities<<<8, 16>>>(a, b, c);
  int ha = -1, hb = -1, hc = -1;
  QG_CHECK(cudaMemcpy(&ha, a, 4, cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(&hb, b, 4, cudaMemcpyDeviceToHost));
  QG_CHECK(cudaMemcpy(&hc, c, 4, cudaMemcpyDeviceToHost));
  std::fprintf(stderr, "mx identities  mul!=ldexp=%d  expfield!=ldexp=%d  nan_bad=%d\n",
               ha, hb, hc);
  EXPECT(ha == 0, "e2m1*e8m0 == ldexp(e2m1, e8m0-127)");
  EXPECT(hb == 0, "fp16 exp-field add == ldexp (with uf/ov policy)");
  EXPECT(hc == 0, "NaN scale → NaN on all three paths");
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  if (g_fail) {
    std::fprintf(stderr, "test_mx_decode: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_mx_decode: OK\n");
  return 0;
}
