// device_props.cu — dump the device facts the harness and the report depend on,
// plus the roofline predictions derived from them.
//
// Run this first on every machine. The ridge point it prints is what turns the
// project's predictions from guesses into arithmetic.

#include <cstdio>

#include "qgemm/formats.cuh"
#include "qgemm/timing.cuh"

using namespace qgemm;

int main(int argc, char** argv) {
  int ndev = 0;
  QG_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev == 0) {
    std::printf("no CUDA devices visible\n");
    return 1;
  }

  const int dev = (argc > 1) ? std::atoi(argv[1]) : 0;
  QG_CHECK(cudaSetDevice(dev));
  const DeviceInfo d = query_device(dev);

  std::printf("=== device %d of %d ===\n", dev, ndev);
  std::printf("name                  %s\n", d.name.c_str());
  std::printf("compute capability    sm_%d%d\n", d.cc_major, d.cc_minor);
  std::printf("SMs                   %d\n", d.sm_count);
  std::printf("L2 cache              %.2f MB\n", d.l2_bytes / 1048576.0);
  std::printf("global memory         %.2f GB\n", d.total_mem / 1073741824.0);
  std::printf("memory bus            %d-bit\n", d.bus_width_bits);
  std::printf("SM clock              %.3f GHz\n", d.sm_clock_khz / 1e6);
  std::printf("memory clock          %.0f MHz\n", d.mem_clock_khz / 1e3);
  std::printf("theoretical peak BW   %.1f GB/s  (DO NOT use as denominator)\n",
              d.theoretical_peak_gbps());
  std::printf("est. dense fp16 TC    %.1f TFLOP/s  (estimate, 512 FLOP/clk/SM)\n",
              d.est_fp16_tc_tflops());

  // Ridge point. Two variants, because the achieved bandwidth is what matters
  // and the theoretical figure is what people quote.
  const double ridge_theo =
      d.est_fp16_tc_tflops() * 1e12 / (d.theoretical_peak_gbps() * 1e9);
  std::printf("\nroofline ridge        %.0f FLOP/byte (vs theoretical BW)\n",
              ridge_theo);
  std::printf("  -> re-run with the measured bandwidth from ./bandwidth for the\n"
              "     figure to quote; achieved BW is typically 80-90%% of peak,\n"
              "     which moves the ridge up by the same factor.\n");

  // Occupancy sanity check. This is the calculation that reveals wave
  // quantization: at M=1 with a 256-wide N tile, a 4096-wide projection
  // launches only 16 CTAs.
  // A launch of `ctas` CTAs on `sm` SMs occupies ctas/sm waves. Below one wave
  // the machine is partly idle for the whole kernel and no amount of
  // intra-kernel tuning recovers it -- only splitting the K reduction does.
  std::printf("\n=== wave quantization at M=1, N=4096 ===\n");
  std::printf("  %-8s %6s   %-22s   %-22s\n", "N-tile", "CTAs",
              "this device", "H100 (132 SM) for ref");
  for (int n_tile : {64, 128, 256, 512}) {
    const int ctas = (4096 + n_tile - 1) / n_tile;
    const double waves_local = static_cast<double>(ctas) / d.sm_count;
    const double waves_h100 = static_cast<double>(ctas) / 132.0;
    char loc[64], h100[64];
    std::snprintf(loc, sizeof(loc), "%.2f waves%s", waves_local,
                  waves_local < 1.0 ? "  IDLE SMs" : "");
    std::snprintf(h100, sizeof(h100), "%.2f waves%s", waves_h100,
                  waves_h100 < 1.0 ? "  IDLE SMs" : "");
    std::printf("  %-8d %6d   %-22s   %-22s\n", n_tile, ctas, loc, h100);
  }
  if (d.sm_count < 60) {
    std::printf("\n  NOTE: with only %d SMs this device HIDES wave quantization --\n"
                "  16 CTAs nearly fills it. The same launch leaves an H100 at 12%%.\n"
                "  Draw no split-K conclusions here; validate them on sm_90.\n",
                d.sm_count);
  }

  // Predicted crossovers per format, from the ridge point.
  std::printf("\n=== predicted compute-bound crossover M (ridge %.0f) ===\n",
              ridge_theo);
  const Format fmts[] = {Format::FP16, Format::FP8_E4M3, Format::INT4_G128,
                         Format::MXFP4, Format::NVFP4};
  std::printf("  %-12s %10s %14s %12s\n", "format", "B/weight", "vs fp8",
              "crossover M");
  for (Format f : fmts) {
    std::printf("  %-12s %10.4f %13.2fx %12.0f\n", format_name(f),
                bytes_per_weight(f), byte_ratio(Format::FP8_E4M3, f),
                predicted_crossover_M(f, ridge_theo));
  }

  // L2 residency warning. This is the check that prevents fabricated results.
  std::printf("\n=== L2 residency check (frozen shape set) ===\n");
  struct Shape { const char* tag; size_t N, K; };
  const Shape shapes[] = {
      {"q/o_proj    4096x4096",  4096, 4096},
      {"k/v_proj    1024x4096",  1024, 4096},
      {"gate/up    14336x4096", 14336, 4096},
      {"down_proj   4096x14336", 4096, 14336},
      {"gpt-oss     2880x2880",  2880, 2880},
  };
  // 50 MB is the H100 L2. Printed alongside the local figure because the
  // hazard is invisible on a small-L2 device and catastrophic on a large one.
  const size_t kH100L2 = 50ull << 20;
  std::printf("  %-24s %10s   %-14s   %-24s\n", "shape", "mxfp4", "this device",
              "H100 (50 MB L2)");
  for (const auto& s : shapes) {
    const size_t wb = weight_bytes(Format::MXFP4, s.N, s.K);
    std::printf("  %-24s %7.2f MB   %-14s   %-24s\n", s.tag, wb / 1048576.0,
                wb < d.l2_bytes ? "RESIDENT" : "streams",
                wb < kH100L2 ? "RESIDENT -- rotate/flush" : "streams");
  }
  std::printf("\n  rotation_count needed to defeat L2 by 2x:\n");
  for (const auto& s : shapes) {
    const size_t wb = weight_bytes(Format::MXFP4, s.N, s.K);
    std::printf("    %-24s local %3d buffers,  H100 %3d buffers\n", s.tag,
                rotation_count(wb, d.l2_bytes, 2.0),
                rotation_count(wb, kH100L2, 2.0));
  }
  std::printf("\n  Every frozen shape is L2-RESIDENT on an H100. Benchmarking any\n"
              "  of them in a hot loop there measures L2 bandwidth, not HBM.\n\n");
  return 0;
}
