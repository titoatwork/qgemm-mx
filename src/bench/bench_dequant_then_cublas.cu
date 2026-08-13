// bench_dequant_then_cublas.cu — sanity-floor baseline with *real* encodings.
//
// Path: host pack.hpp (INT4 / MXFP4 / optional NVFP4) → upload packed+scales →
// device dequant_device.cuh (same nibble / scale layout) into an FP16 workspace
// → cuBLAS FP16 GEMM. This is the "obvious wrong way": it materializes ~4×
// more weight bytes than a fused kernel. Report as "×N vs naive" only.
//
// Supersedes bench_naive_dequant.cu's dummy nibble expand (traffic-shaped,
// not bit-exact). Timing: CUDA-graph + rotated FP16 workspace, same contract
// as bench_cublas / bench_naive_dequant.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "qgemm/correctness.hpp"
#include "qgemm/dequant_device.cuh"
#include "qgemm/formats.cuh"
#include "qgemm/pack.hpp"
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

enum class DqKind : int { Int4 = 0, Mxfp4 = 1, Nvfp4 = 2 };

__global__ void k_dequant_to_fp16(const uint8_t* __restrict__ packed,
                                  const uint16_t* __restrict__ scales_u16,
                                  const uint8_t* __restrict__ scales_u8,
                                  float tensor_scale, __half* __restrict__ W,
                                  size_t N, size_t K, int group, int kind) {
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t n_elem = N * K;
  if (i >= n_elem) return;
  const size_t n = i / K;
  const size_t k = i - n * K;
  float w = 0.f;
  if (kind == static_cast<int>(DqKind::Int4)) {
    w = dev::dequant_int4_elem(packed, scales_u16, n, k, N, K, group);
  } else if (kind == static_cast<int>(DqKind::Mxfp4)) {
    w = dev::dequant_mxfp4_elem(packed, scales_u8, n, k, N, K);
  } else {
    w = dev::dequant_nvfp4_elem(packed, scales_u8, tensor_scale, n, k, N, K);
  }
  W[i] = __float2half(w);
}

__global__ void fill_half(__half* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n) p[i] = __float2half(v);
}

struct BenchShape {
  const char* name;
  size_t N;
  size_t K;
};

struct PackedHost {
  DqKind kind = DqKind::Mxfp4;
  Format fmt = Format::MXFP4;
  int group = 32;
  float tensor_scale = 1.f;
  std::vector<uint8_t> data;
  std::vector<uint16_t> scales_u16;
  std::vector<uint8_t> scales_u8;
  std::vector<float> ref_f32;
};

static bool parse_nxk(const char* s, size_t* N, size_t* K) {
  if (s == nullptr || N == nullptr || K == nullptr) return false;
  unsigned long long n = 0, k = 0;
  char sep = 0;
  if (std::sscanf(s, "%llu%c%llu", &n, &sep, &k) != 3) return false;
  if (sep != 'x' && sep != 'X') return false;
  if (n == 0 || k == 0) return false;
  *N = static_cast<size_t>(n);
  *K = static_cast<size_t>(k);
  return true;
}

static bool parse_format(const char* s, Format* fmt) {
  if (s == nullptr || fmt == nullptr) return false;
  const std::string t(s);
  if (t == "mxfp4") {
    *fmt = Format::MXFP4;
    return true;
  }
  if (t == "int4" || t == "int4_g128") {
    *fmt = Format::INT4_G128;
    return true;
  }
  if (t == "int4_g64") {
    *fmt = Format::INT4_G64;
    return true;
  }
  if (t == "nvfp4") {
    *fmt = Format::NVFP4;
    return true;
  }
  return false;
}

static bool format_ok_for_K(Format fmt, size_t K) {
  switch (fmt) {
    case Format::INT4_G128:
      return K % static_cast<size_t>(kInt4Group128) == 0;
    case Format::INT4_G64:
      return K % static_cast<size_t>(kInt4Group64) == 0;
    case Format::MXFP4:
      return K % static_cast<size_t>(kMxfp4BlockSize) == 0;
    case Format::NVFP4:
      return K % static_cast<size_t>(kNvfp4BlockSize) == 0;
    default:
      return false;
  }
}

static PackedHost pack_weights(Format fmt, const float* W, size_t N, size_t K) {
  PackedHost out;
  out.fmt = fmt;
  if (fmt == Format::INT4_G128 || fmt == Format::INT4_G64) {
    out.kind = DqKind::Int4;
    out.group = (fmt == Format::INT4_G64) ? kInt4Group64 : kInt4Group128;
    PackedInt4 p = pack_int4(W, N, K, out.group);
    out.ref_f32 = dequant_reference_int4(p);
    out.data = std::move(p.data);
    out.scales_u16 = std::move(p.scales);
    return out;
  }
  if (fmt == Format::NVFP4) {
    out.kind = DqKind::Nvfp4;
    out.group = kNvfp4BlockSize;
    PackedNvfp4 p = pack_nvfp4(W, N, K, 1.0f);
    out.ref_f32 = dequant_reference_nvfp4(p);
    out.tensor_scale = p.tensor_scale;
    out.data = std::move(p.data);
    out.scales_u8 = std::move(p.scales);
    return out;
  }
  out.kind = DqKind::Mxfp4;
  out.group = kMxfp4BlockSize;
  out.fmt = Format::MXFP4;
  PackedMxfp4 p = pack_mxfp4(W, N, K);
  out.ref_f32 = dequant_reference_mxfp4(p);
  out.data = std::move(p.data);
  out.scales_u8 = std::move(p.scales);
  return out;
}

static void fill_host_weights(std::vector<float>* W) {
  for (size_t i = 0; i < W->size(); ++i) {
    (*W)[i] = 0.05f * static_cast<float>((i * 13) % 11) - 0.2f;
  }
}

// Device dequant vs host dequant_reference_* after a host FP16 round-trip
// (device writes __half). Tight enough to catch nibble/scale indexing bugs.
static void check_dequant_matches_host(const PackedHost& packed, const uint8_t* d_pk,
                                       const uint16_t* d16, const uint8_t* d8,
                                       size_t N, size_t K) {
  const size_t n_elem = N * K;
  __half* dW = nullptr;
  QG_CHECK(cudaMalloc(&dW, n_elem * sizeof(__half)));
  const unsigned grid = static_cast<unsigned>((n_elem + 255) / 256);
  k_dequant_to_fp16<<<grid, 256>>>(d_pk, d16, d8, packed.tensor_scale, dW, N, K,
                                   packed.group, static_cast<int>(packed.kind));
  QG_CHECK(cudaDeviceSynchronize());
  QG_CHECK(cudaGetLastError());

  std::vector<uint16_t> bits(n_elem);
  QG_CHECK(cudaMemcpy(bits.data(), dW, n_elem * sizeof(__half),
                      cudaMemcpyDeviceToHost));
  cudaFree(dW);

  std::vector<float> got(n_elem), ref16(n_elem);
  for (size_t i = 0; i < n_elem; ++i) {
    got[i] = fp16_bits_to_float(bits[i]);
    ref16[i] = fp16_bits_to_float(float_to_fp16_bits(packed.ref_f32[i]));
  }
  const CompareResult cr =
      compare_close(got.data(), ref16.data(), n_elem, 2e-3f, 2e-3f);
  std::fprintf(stderr,
               "  dequant check %s  n=%zu  max_abs=%.4g  max_rel=%.4g%s%s\n",
               format_name(packed.fmt), n_elem, cr.max_abs_err, cr.max_rel_err,
               cr.ok ? "" : "  ", cr.ok ? "" : cr.message.c_str());
  if (!cr.ok) {
    std::fprintf(stderr,
                 "[qgemm] device dequant != host dequant_reference (encoding "
                 "mismatch)\n");
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  const double peak_gbps = (argc > 2) ? std::atof(argv[2]) : 0.0;
  const char* shape_filter = (argc > 3) ? argv[3] : "256x256";
  const size_t max_M =
      (argc > 4) ? static_cast<size_t>(std::atoll(argv[4])) : 8;
  const char* fmt_arg = (argc > 5) ? argv[5] : "all";
  const bool do_check = !(argc > 6 && std::string(argv[6]) == "nocheck");

  QG_CHECK(cudaSetDevice(dev));
  const DeviceInfo d = query_device(dev);
  if (peak_gbps <= 0.0) {
    std::printf(
        "usage: %s <device> <measured_read_gbps> [layer|all|NxK] [max_M] "
        "[format] [nocheck]\n",
        argv[0]);
    std::printf(
        "  format: mxfp4 | int4_g128 | int4_g64 | nvfp4 | all\n"
        "  NxK e.g. 256x256 (smoke). Layers from shapes.hpp.\n");
    return 2;
  }

  std::vector<BenchShape> shapes;
  size_t custom_N = 0, custom_K = 0;
  if (parse_nxk(shape_filter, &custom_N, &custom_K)) {
    shapes.push_back({"custom", custom_N, custom_K});
  } else if (std::string(shape_filter) == "all") {
    for (size_t i = 0; i < kNumLayerShapes; ++i) {
      shapes.push_back(
          {kLayerShapes[i].name, kLayerShapes[i].N, kLayerShapes[i].K});
    }
  } else if (const LayerShape* layer = find_layer(shape_filter)) {
    shapes.push_back({layer->name, layer->N, layer->K});
  } else {
    std::fprintf(stderr, "unknown shape '%s' (use layer name, all, or NxK)\n",
                 shape_filter);
    return 2;
  }

  std::vector<Format> formats;
  Format one_fmt;
  if (std::string(fmt_arg) == "all") {
    formats.push_back(Format::INT4_G128);
    formats.push_back(Format::MXFP4);
  } else if (parse_format(fmt_arg, &one_fmt)) {
    formats.push_back(one_fmt);
  } else {
    std::fprintf(stderr, "unknown format '%s'\n", fmt_arg);
    return 2;
  }

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

  std::fprintf(stderr, "=== dequant-then-cuBLAS (real pack.hpp encodings) ===\n");
  std::fprintf(stderr, "device %s  denominator %.1f GB/s  shape %s max_M %zu  fmt %s\n",
               d.name.c_str(), peak_gbps, shape_filter, max_M, fmt_arg);

  std::printf(
      "shape,M,N,K,format,path,rotation,graphed_cold_us,"
      "pct_of_ideal_fp16_bytes,pct_of_ideal_packed_bytes,achieved_gbps\n");

  for (const BenchShape& sh : shapes) {
    if (sh.K == 0 || sh.N == 0) continue;
    const size_t n_elem = sh.N * sh.K;
    std::vector<float> Wh(n_elem);
    fill_host_weights(&Wh);

    for (Format fmt : formats) {
      if (!format_ok_for_K(fmt, sh.K)) {
        std::fprintf(stderr, "%s  skip %s (K=%zu not divisible by group)\n",
                     sh.name, format_name(fmt), sh.K);
        continue;
      }

      PackedHost packed;
      try {
        packed = pack_weights(fmt, Wh.data(), sh.N, sh.K);
      } catch (const std::exception& ex) {
        std::fprintf(stderr, "%s  skip %s (%s)\n", sh.name, format_name(fmt),
                     ex.what());
        continue;
      }

      uint8_t* d_pk = nullptr;
      uint16_t* d16 = nullptr;
      uint8_t* d8 = nullptr;
      QG_CHECK(cudaMalloc(&d_pk, packed.data.size()));
      QG_CHECK(cudaMemcpy(d_pk, packed.data.data(), packed.data.size(),
                          cudaMemcpyHostToDevice));
      if (!packed.scales_u16.empty()) {
        QG_CHECK(cudaMalloc(&d16, packed.scales_u16.size() * sizeof(uint16_t)));
        QG_CHECK(cudaMemcpy(d16, packed.scales_u16.data(),
                            packed.scales_u16.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));
      }
      if (!packed.scales_u8.empty()) {
        QG_CHECK(cudaMalloc(&d8, packed.scales_u8.size()));
        QG_CHECK(cudaMemcpy(d8, packed.scales_u8.data(), packed.scales_u8.size(),
                            cudaMemcpyHostToDevice));
      }

      if (do_check) {
        check_dequant_matches_host(packed, d_pk, d16, d8, sh.N, sh.K);
      }

      const size_t w_fp16_bytes = n_elem * sizeof(__half);
      const int rot = rotation_count(w_fp16_bytes, d.l2_bytes, 2.0);
      std::vector<__half*> W(rot, nullptr);
      for (int i = 0; i < rot; ++i) {
        QG_CHECK(cudaMalloc(&W[i], w_fp16_bytes));
      }

      std::fprintf(stderr,
                   "%s  %s  N=%zu K=%zu  packed %.2f MB  W_fp16 %.2f MB  rot %d\n",
                   sh.name, format_name(fmt), sh.N, sh.K,
                   packed.data.size() / 1048576.0, w_fp16_bytes / 1048576.0,
                   rot);

      const unsigned dq_grid = static_cast<unsigned>((n_elem + 255) / 256);
      const int kind = static_cast<int>(packed.kind);
      const int group = packed.group;
      const float tensor = packed.tensor_scale;

      for (size_t mi = 0; mi < kNumMSweep; ++mi) {
        const size_t M = kMSweep[mi];
        if (M > max_M) break;

        __half *X = nullptr, *Y = nullptr;
        QG_CHECK(cudaMalloc(&X, M * sh.K * sizeof(__half)));
        QG_CHECK(cudaMalloc(&Y, M * sh.N * sizeof(__half)));
        fill_half<<<static_cast<unsigned>((M * sh.K + 255) / 256), 256>>>(
            X, M * sh.K, 0.02f);
        QG_CHECK(cudaDeviceSynchronize());

        const float alpha = 1.f, beta = 0.f;
        const int m = static_cast<int>(sh.N);
        const int n = static_cast<int>(M);
        const int k = static_cast<int>(sh.K);

        auto step = [&](int idx, cudaStream_t st) {
          __half* Ww = W[idx % rot];
          k_dequant_to_fp16<<<dq_grid, 256, 0, st>>>(
              d_pk, d16, d8, tensor, Ww, sh.N, sh.K, group, kind);
          CUBLAS_CHECK(cublasSetStream(h, st));
          CUBLAS_CHECK(cublasGemmEx(
              h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, Ww, CUDA_R_16F, k,
              X, CUDA_R_16F, k, &beta, Y, CUDA_R_16F, m, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        };

        const int inner = std::max(rot, 8);
        const TimingResult cold = time_graphed(step, 10, 30, inner);

        const size_t moved_fp16 = total_bytes(Format::FP16, M, sh.N, sh.K);
        const size_t moved_pk = total_bytes(fmt, M, sh.N, sh.K);
        const double pct_fp16 = pct_of_ideal(cold.median_us, moved_fp16, peak_gbps);
        const double pct_pk = pct_of_ideal(cold.median_us, moved_pk, peak_gbps);
        const double gbps = achieved_gbps(moved_fp16, cold.median_us);

        std::fprintf(stderr,
                     "  M=%zu  cold=%.2f us  %%ideal(fp16)=%.1f  %%ideal(packed)=%.1f\n",
                     M, cold.median_us, pct_fp16, pct_pk);
        std::printf("%s,%zu,%zu,%zu,%s,dequant_then_cublas,%d,%.4f,%.2f,%.2f,%.2f\n",
                    sh.name, M, sh.N, sh.K, format_name(fmt), rot, cold.median_us,
                    pct_fp16, pct_pk, gbps);

        cudaFree(X);
        cudaFree(Y);
      }

      for (auto* p : W) cudaFree(p);
      cudaFree(d_pk);
      cudaFree(d16);
      cudaFree(d8);
    }
  }

  cublasDestroy(h);
  std::fprintf(stderr,
               "\nSanity floor (real encodings). Extra dequant write + FP16 "
               "re-read vs a fused path.\n");
  return 0;
}
