// qgemv_ref.cuh — correct-and-slow fused weight-only GEMM (device).
//
// Y[m,n] = sum_k X[m,k] * dequant(W)[n,k]
// One thread per (m,n). No Tensor Cores, no pre-shuffle, no wgmma.
// This is NOT a Marlin/Machete competitor. Hopper peak requires:
//   wgmma + weights as operand A (Y^T = W^T X^T) + TMA + split-K.
// See docs/RESEARCH_ANALYSIS.md and docs/D1_LAYOUT_DERIVATION.md.

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

// Split-K fused-ref: blockIdx.y is the K-split. Partial sums atomicAdd into Y.
// Caller MUST zero Y. Split boundaries snap to `group` (32 MX / 16 NV / 128 INT4).
__global__ void k_qgemm_ref_splitk(const float* __restrict__ X,
                                   const uint8_t* packed,
                                   const uint16_t* scales_u16,
                                   const uint8_t* scales_u8, float tensor_scale,
                                   float* __restrict__ Y, int M, int N, int K,
                                   int group, QFormat fmt, int split_k) {
  if (split_k < 1) return;
  const int sk = static_cast<int>(blockIdx.y);
  if (sk >= split_k) return;

  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int MN = M * N;
  if (tid >= MN) return;
  const int m = tid / N;
  const int n = tid - m * N;

  const int g = (group > 0) ? group : 1;
  const int k_aligned = (K / g) * g;
  const int chunks = k_aligned / g;
  const int chunk0 = (chunks * sk) / split_k;
  const int chunk1 = (chunks * (sk + 1)) / split_k;
  const int k0 = chunk0 * g;
  const int k1 = chunk1 * g;

  float acc = 0.f;
  for (int k = k0; k < k1; ++k) {
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
  if (acc != 0.f) atomicAdd(&Y[m * N + n], acc);
}

}  // namespace qgemm
