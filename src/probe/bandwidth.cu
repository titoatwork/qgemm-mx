// bandwidth.cu — empirical peak memory bandwidth.
//
// This produces THE denominator for the whole project. Every "percent of ideal"
// figure in the report divides by this number, and using the spec-sheet value
// instead would make every efficiency claim better than it is.
//
// Three access patterns are measured because they bound the achievable rate
// differently, and a fused dequant-GEMM sits closest to the read-only case:
//   read+write (copy-like)   -- the classic STREAM-style figure
//   read-only  (reduction)   -- what a weight-streaming GEMM actually does
//   write-only (fill)        -- reported for completeness
//
// Working set defaults to 256 MiB, far above any current L2, so the numbers are
// genuine HBM rates rather than cache rates.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "qgemm/formats.cuh"
#include "qgemm/timing.cuh"

using namespace qgemm;

__global__ void k_copy(const float4* __restrict__ in, float4* __restrict__ out,
                       size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

__global__ void k_read(const float4* __restrict__ in, float* __restrict__ sink,
                       size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += stride) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Never true at runtime, but the compiler cannot prove it, so the loads stay.
  if (acc == 1234.5678f) sink[0] = acc;
}

__global__ void k_write(float4* __restrict__ out, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  for (; i < n; i += stride) out[i] = v;
}

int main(int argc, char** argv) {
  const size_t mib = (argc > 1) ? static_cast<size_t>(std::atoll(argv[1])) : 256;
  const int dev = (argc > 2) ? std::atoi(argv[2]) : 0;
  QG_CHECK(cudaSetDevice(dev));
  const DeviceInfo d = query_device(dev);

  const size_t bytes = mib << 20;
  const size_t n4 = bytes / sizeof(float4);

  float4 *a = nullptr, *b = nullptr;
  float* sink = nullptr;
  QG_CHECK(cudaMalloc(&a, bytes));
  QG_CHECK(cudaMalloc(&b, bytes));
  QG_CHECK(cudaMalloc(&sink, sizeof(float)));
  QG_CHECK(cudaMemset(a, 1, bytes));
  QG_CHECK(cudaMemset(b, 0, bytes));

  // Grid sized to a few waves so the loop amortizes launch and index setup.
  const int block = 256;
  const unsigned grid =
      static_cast<unsigned>(std::min<size_t>((n4 + block - 1) / block,
                                             static_cast<size_t>(d.sm_count) * 32));

  std::printf("=== empirical bandwidth: %s (sm_%d%d) ===\n", d.name.c_str(),
              d.cc_major, d.cc_minor);
  std::printf("working set %zu MiB per buffer  (L2 is %.2f MB, ratio %.0fx)\n",
              mib, d.l2_bytes / 1048576.0,
              static_cast<double>(bytes) / static_cast<double>(d.l2_bytes));
  std::printf("grid %u x %d\n\n", grid, block);

  struct Row { const char* tag; double bytes_moved; TimingResult t; };
  std::vector<Row> rows;

  {
    auto f = [=](int, cudaStream_t s) {
      k_copy<<<grid, block, 0, s>>>(a, b, n4);
    };
    rows.push_back({"read+write (copy)", 2.0 * bytes,
                    time_graphed(f, 5, 30, 10)});
  }
  {
    auto f = [=](int, cudaStream_t s) {
      k_read<<<grid, block, 0, s>>>(a, sink, n4);
    };
    rows.push_back({"read-only (stream)", 1.0 * bytes,
                    time_graphed(f, 5, 30, 10)});
  }
  {
    auto f = [=](int, cudaStream_t s) {
      k_write<<<grid, block, 0, s>>>(b, n4);
    };
    rows.push_back({"write-only (fill)", 1.0 * bytes,
                    time_graphed(f, 5, 30, 10)});
  }

  const double theo = d.theoretical_peak_gbps();
  std::printf("%-22s %12s %12s %10s\n", "pattern", "median us", "GB/s",
              "% of theo");
  double best_read = 0.0;
  for (const auto& r : rows) {
    const double gbps = achieved_gbps(static_cast<size_t>(r.bytes_moved),
                                      r.t.median_us);
    std::printf("%-22s %12.1f %12.1f %9.1f%%\n", r.tag, r.t.median_us, gbps,
                100.0 * gbps / theo);
    if (std::string(r.tag).find("read-only") != std::string::npos)
      best_read = gbps;
  }

  std::printf("\ntheoretical peak      %12.1f GB/s\n", theo);
  std::printf("\n>>> USE THIS AS THE PROJECT DENOMINATOR: %.1f GB/s (read-only)\n",
              best_read);
  std::printf(">>> record it in PREREGISTRATION.md before any GEMM measurement\n");

  // Ideal times for the frozen shapes, at the measured rate. These are the
  // floors every kernel is measured against.
  std::printf("\n=== t_ideal at %.1f GB/s (weight traffic only) ===\n", best_read);
  std::printf("%-24s %10s %10s %10s %10s\n", "shape", "fp16", "fp8", "mxfp4",
              "int4g128");
  struct Shape { const char* tag; size_t N, K; };
  const Shape shapes[] = {
      {"q/o_proj  4096x4096",  4096, 4096},
      {"k/v_proj  1024x4096",  1024, 4096},
      {"gate/up  14336x4096", 14336, 4096},
      {"down_proj 4096x14336", 4096, 14336},
      {"gpt-oss   2880x2880",  2880, 2880},
  };
  for (const auto& s : shapes) {
    std::printf("%-24s %9.2f %9.2f %9.2f %9.2f   (us)\n", s.tag,
                ideal_us(weight_bytes(Format::FP16, s.N, s.K), best_read),
                ideal_us(weight_bytes(Format::FP8_E4M3, s.N, s.K), best_read),
                ideal_us(weight_bytes(Format::MXFP4, s.N, s.K), best_read),
                ideal_us(weight_bytes(Format::INT4_G128, s.N, s.K), best_read));
  }

  const TimingResult lo = measure_launch_overhead();
  std::printf("\nkernel launch overhead  %.3f us (median, non-graphed)\n",
              lo.median_us);
  std::printf("  -> compare against t_ideal above. Where launch >= t_ideal,\n"
              "     un-graphed measurement reports the launcher, not the kernel.\n\n");

  cudaFree(a); cudaFree(b); cudaFree(sink);
  return 0;
}
