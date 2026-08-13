// bench_stream_ideal.cu — pure weight-stream bound for each packed format.
//
// Streams a buffer of size weight_bytes(format, N, K) with the same rotation +
// CUDA-graph protocol as bench_cublas. There is no GEMM: this measures how
// close pure HBM traffic of that volume gets to the empirical read-only peak.
//
// Purpose:
//   t_ideal for each format is ideal_us(weight_bytes, measured_gbps). The
//   graphed cold stream time is the floor a perfect weight-only kernel can hit
//   before dequant / MMA costs. pct_of_ideal here should sit near 100% if the
//   harness and denominator are honest; if not, fix the harness before any
//   fused-kernel claim.
//
// M appears in the CSV for joinability with other benches. Weight traffic does
// not depend on M; each shape x format is measured once and reported at M=1.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qgemm/formats.cuh"
#include "qgemm/timing.cuh"

using namespace qgemm;

// Coalesced float4 read of `n4` vectors; sink keeps the loads live.
__global__ void k_stream_read(const float4* __restrict__ in,
                              float* __restrict__ sink, size_t n4) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1234.5678f) sink[0] = acc;
}

struct Shape {
  const char* tag;
  size_t N, K;
};

int main(int argc, char** argv) {
  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  // Measured read-only bandwidth from ./bandwidth. Same contract as
  // bench_cublas: never fall back to the theoretical peak.
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

  const Shape shapes[] = {
      {"q/o_proj", 4096, 4096},
      {"gpt-oss", 2880, 2880},
  };
  // Formats whose byte counts drive the R0/R1 claims.
  const Format formats[] = {
      Format::FP16, Format::FP8_E4M3, Format::INT4_G128, Format::MXFP4,
      Format::NVFP4,
  };

  std::fprintf(stderr, "=== packed-weight stream ideal (pure traffic) ===\n");
  std::fprintf(stderr, "device      %s (sm_%d%d, %d SMs, %.2f MB L2)\n",
               d.name.c_str(), d.cc_major, d.cc_minor, d.sm_count,
               d.l2_bytes / 1048576.0);
  std::fprintf(stderr, "denominator %.1f GB/s (measured read-only)\n", peak_gbps);

  const TimingResult lo = measure_launch_overhead();
  std::fprintf(stderr, "launch cost %.3f us/kernel (median, non-graphed)\n\n",
               lo.median_us);

  // shape,M,N,K,format + timing columns + pct_of_ideal
  std::printf("shape,M,N,K,format,rotation,graphed_cold_us,graphed_hot_us,"
              "looped_cold_us,ideal_us,pct_of_ideal,achieved_gbps,"
              "l2_inflation,launch_share\n");

  float* sink = nullptr;
  QG_CHECK(cudaMalloc(&sink, sizeof(float)));
  QG_CHECK(cudaMemset(sink, 0, sizeof(float)));

  const int block = 256;

  for (const auto& s : shapes) {
    for (Format fmt : formats) {
      const size_t wbytes = weight_bytes(fmt, s.N, s.K);
      // Align up to float4 so the stream kernel has a clean vector width;
      // ideal / pct still use the true weight_bytes from formats.cuh.
      const size_t alloc = (wbytes + 15) & ~static_cast<size_t>(15);
      const size_t n4 = alloc / sizeof(float4);
      const int rot = rotation_count(wbytes, d.l2_bytes, 2.0);

      std::vector<float4*> bufs(rot, nullptr);
      for (int i = 0; i < rot; ++i) {
        QG_CHECK(cudaMalloc(&bufs[i], alloc));
        QG_CHECK(cudaMemset(bufs[i], 1 + i, alloc));
      }
      QG_CHECK(cudaDeviceSynchronize());

      const unsigned grid = static_cast<unsigned>(std::min<size_t>(
          (n4 + block - 1) / block, static_cast<size_t>(d.sm_count) * 32));

      const double t_ideal = ideal_us(wbytes, peak_gbps);

      std::fprintf(stderr,
                   "\n%s  %s  N=%zu K=%zu  weights %.2f MB  rotation %d  "
                   "t_ideal %.2f us\n",
                   s.tag, format_name(fmt), s.N, s.K, wbytes / 1048576.0, rot,
                   t_ideal);
      std::fprintf(stderr, "%11s %11s %11s %9s %11s %8s %8s\n", "cold(us)",
                   "hot(us)", "looped(us)", "%ideal", "GB/s", "L2infl",
                   "launch%");

      auto stream = [&](int idx, cudaStream_t st) {
        k_stream_read<<<grid, block, 0, st>>>(
            bufs[idx % static_cast<int>(bufs.size())], sink, n4);
      };
      auto stream_hot = [&](int, cudaStream_t st) { stream(0, st); };

      // inner >= rot so one graph covers a full rotation cycle.
      const int inner = std::max(rot, 8);
      const TimingResult cold = time_graphed(stream, 10, 40, inner);
      const TimingResult hot = time_graphed(stream_hot, 10, 40, inner);
      const TimingResult looped = time_looped(stream, 10, 40, inner);

      const double pct = pct_of_ideal(cold.median_us, wbytes, peak_gbps);
      const double gbps = achieved_gbps(wbytes, cold.median_us);
      const double l2_infl = cold.median_us / hot.median_us;
      const double launch_share =
          100.0 * (looped.median_us - cold.median_us) / looped.median_us;

      // M=1: pure weight stream is M-independent; column kept for CSV joins.
      const int M = 1;

      std::fprintf(stderr,
                   "%11.2f %11.2f %11.2f %8.1f%% %11.1f %8.2fx %7.1f%%\n",
                   cold.median_us, hot.median_us, looped.median_us, pct, gbps,
                   l2_infl, launch_share);

      std::printf("%s,%d,%zu,%zu,%s,%d,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%.2f\n",
                  s.tag, M, s.N, s.K, format_name(fmt), rot, cold.median_us,
                  hot.median_us, looped.median_us, t_ideal, pct, gbps, l2_infl,
                  launch_share);

      for (auto* p : bufs) cudaFree(p);
    }
  }

  std::fprintf(stderr,
      "\nHow to read this:\n"
      "  %%ideal near 100%%  pure traffic saturates the empirical RO peak;\n"
      "                     the denominator and timing protocol agree.\n"
      "  %%ideal << 100%%    stream kernel or rotation is not HBM-bound, or\n"
      "                     the measured_bw_gbps argument is too high.\n"
      "  ideal_us column    theoretical floor at the measured denominator;\n"
      "                     compare fused-kernel cold_us against this, not\n"
      "                     against the stream time alone.\n\n");

  cudaFree(sink);
  return 0;
}
