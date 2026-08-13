// qgemv_ref.cuh — correct-and-slow fused weight-only GEMM (device).
//
// Y[m,n] = sum_k X[m,k] * dequant(W)[n,k]
// One thread per (m,n). No Tensor Cores. Exists so we can prove encodings
// on GPU and have a fused path before R2 wgmma work.

#pragma once

#include "qgemm/dequant_device.cuh"

namespace qgemm {

enum class QFormat : int { Int4 = 0, Mxfp4 = 1, Nvfp4 = 2 };

// scales_u16 used for INT4; scales_u8 for MX/NV; tensor_scale for NV only.
__global__ void k_qgemm_ref(const float* __restrict__ X, const uint8_t* packed,
                            const uint16_t* scales_u16, const uint8_t* scales_u8,
                            float tensor_scale, float* __restrict__ Y,
                            int M, int N, int K, int group, QFormat fmt) {
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int MN = M * N;
  if (tid >= MN) return;
  const int m = tid / N;
  const int n = tid - m * N;

  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    float w = 0.f;
    if (fmt == QFormat::Int4) {
      w = dev::dequant_int4_elem(packed, scales_u16, static_cast<size_t>(n),
                                 static_cast<size_t>(k), static_cast<size_t>(N),
                                 static_cast<size_t>(K), group);
    } else if (fmt == QFormat::Mxfp4) {
      w = dev::dequant_mxfp4_elem(packed, scales_u8, static_cast<size_t>(n),
                                  static_cast<size_t>(k), static_cast<size_t>(N),
                                  static_cast<size_t>(K));
    } else {
      w = dev::dequant_nvfp4_elem(packed, scales_u8, tensor_scale,
                                  static_cast<size_t>(n), static_cast<size_t>(k),
                                  static_cast<size_t>(N), static_cast<size_t>(K));
    }
    acc += X[m * K + k] * w;
  }
  Y[m * N + n] = acc;
}

}  // namespace qgemm
