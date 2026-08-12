// timing.cuh — measurement primitives for low-precision GEMM benchmarking.
//
// The protocol this implements exists because naive kernel timing produces
// numbers that are wrong by large factors for memory-bound kernels:
//
//   1. L2 residency. A 4096x4096 INT4 weight matrix is 8.4 MB. An H100's L2 is
//      50 MB. Timed in a hot loop the weights never leave L2 and you measure L2
//      bandwidth, not HBM. Handled here by BUFFER ROTATION (preferred) or an
//      explicit flush kernel.
//   2. Launch overhead. At M=1 on an H100 an ideal kernel runs in ~3 us while a
//      kernel launch costs 3-8 us. The launcher is larger than the kernel.
//      Handled by capturing many iterations into one CUDA graph.
//   3. Clock drift. Reported separately by scripts/env_capture.sh.
//
// Every timing function returns both the distribution and the launch overhead
// measured under identical conditions, so a caller can always tell how much of
// a reported time is dispatch.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace qgemm {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define QG_CHECK(expr)                                                        \
  do {                                                                        \
    cudaError_t _err = (expr);                                                \
    if (_err != cudaSuccess) {                                                \
      std::fprintf(stderr, "[qgemm] CUDA error %s at %s:%d -> %s\n",          \
                   cudaGetErrorName(_err), __FILE__, __LINE__, #expr);        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Device description
// ---------------------------------------------------------------------------

struct DeviceInfo {
  int device = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;
  int sm_count = 0;
  size_t l2_bytes = 0;
  size_t total_mem = 0;
  int bus_width_bits = 0;
  int mem_clock_khz = 0;
  int sm_clock_khz = 0;

  // Theoretical peak from the memory interface. Always inflated relative to
  // what any kernel achieves; present only so the harness can report the ratio.
  double theoretical_peak_gbps() const {
    return 2.0 * static_cast<double>(mem_clock_khz) * 1e3 *
           (bus_width_bits / 8.0) / 1e9;
  }

  // Dense FP16 tensor-core peak, estimated as 512 FLOP/clk/SM (Ampere and
  // Hopper consumer/datacenter dense rate). An estimate, not a measurement --
  // label it as such anywhere it reaches a figure.
  double est_fp16_tc_tflops() const {
    return sm_count * 512.0 * (static_cast<double>(sm_clock_khz) * 1e3) / 1e12;
  }
};

inline DeviceInfo query_device(int dev = 0) {
  cudaDeviceProp p{};
  QG_CHECK(cudaGetDeviceProperties(&p, dev));
  DeviceInfo d;
  d.device = dev;
  d.name = p.name;
  d.cc_major = p.major;
  d.cc_minor = p.minor;
  d.sm_count = p.multiProcessorCount;
  d.l2_bytes = static_cast<size_t>(p.l2CacheSize);
  d.total_mem = p.totalGlobalMem;
  d.bus_width_bits = p.memoryBusWidth;
  d.mem_clock_khz = p.memoryClockRate;
  d.sm_clock_khz = p.clockRate;
  return d;
}

// ---------------------------------------------------------------------------
// L2 flushing
// ---------------------------------------------------------------------------

__global__ void l2_flush_kernel(int4* __restrict__ buf, size_t n4) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n4) {
    int4 v = buf[i];
    v.x += 1;
    buf[i] = v;
  }
}

// Allocates a scratch buffer several times L2 and streams through it, evicting
// anything the kernel under test left behind.
//
// Prefer BufferRotation where possible: flushing inside a captured graph adds
// its own time to the measurement, and flushing outside the graph forces a
// per-iteration launch, reintroducing the overhead graphs exist to remove.
class L2Flusher {
 public:
  explicit L2Flusher(size_t l2_bytes, double multiplier = 3.0) {
    bytes_ = static_cast<size_t>(l2_bytes * multiplier);
    bytes_ = std::max(bytes_, static_cast<size_t>(8u << 20));  // floor at 8 MiB
    bytes_ = (bytes_ + 15) & ~static_cast<size_t>(15);          // 16B align
    QG_CHECK(cudaMalloc(&buf_, bytes_));
    QG_CHECK(cudaMemset(buf_, 0, bytes_));
    n4_ = bytes_ / sizeof(int4);
  }
  ~L2Flusher() { if (buf_) cudaFree(buf_); }

  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t s = nullptr) const {
    const int block = 256;
    const size_t grid = (n4_ + block - 1) / block;
    l2_flush_kernel<<<static_cast<unsigned>(grid), block, 0, s>>>(
        static_cast<int4*>(buf_), n4_);
  }

  size_t bytes() const { return bytes_; }

 private:
  void* buf_ = nullptr;
  size_t bytes_ = 0;
  size_t n4_ = 0;
};

// ---------------------------------------------------------------------------
// Buffer rotation
// ---------------------------------------------------------------------------

// Computes how many distinct copies of a working set are needed so that
// cycling through them exceeds L2 by `multiplier`, guaranteeing every
// iteration reads from HBM without any flush kernel in the timed region.
//
// This is the preferred way to defeat L2 residency: the timed region stays
// pure kernel, and the memory the kernel touches is genuinely cold.
inline int rotation_count(size_t working_set_bytes, size_t l2_bytes,
                          double multiplier = 2.0) {
  if (working_set_bytes == 0) return 1;
  const double needed = static_cast<double>(l2_bytes) * multiplier;
  int n = static_cast<int>(std::ceil(needed / static_cast<double>(working_set_bytes)));
  return std::max(n, 2);
}

// ---------------------------------------------------------------------------
// Timing results
// ---------------------------------------------------------------------------

struct TimingResult {
  double mean_us = 0.0;
  double median_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  double stddev_us = 0.0;
  double p05_us = 0.0;
  double p95_us = 0.0;
  int samples = 0;
  int inner_reps = 0;
  bool graphed = false;

  void print(const char* label) const {
    std::printf(
        "%-34s median %9.3f us   mean %9.3f  sd %7.3f  min %9.3f  max %9.3f  "
        "[%d x %d%s]\n",
        label, median_us, mean_us, stddev_us, min_us, max_us, samples,
        inner_reps, graphed ? ", graph" : "");
  }
};

inline TimingResult summarize(std::vector<double> us, int inner, bool graphed) {
  TimingResult r;
  if (us.empty()) return r;
  std::sort(us.begin(), us.end());
  r.samples = static_cast<int>(us.size());
  r.inner_reps = inner;
  r.graphed = graphed;
  r.min_us = us.front();
  r.max_us = us.back();
  r.median_us = us[us.size() / 2];
  r.p05_us = us[static_cast<size_t>(us.size() * 0.05)];
  r.p95_us = us[std::min(us.size() - 1, static_cast<size_t>(us.size() * 0.95))];
  double sum = 0.0;
  for (double v : us) sum += v;
  r.mean_us = sum / us.size();
  double acc = 0.0;
  for (double v : us) acc += (v - r.mean_us) * (v - r.mean_us);
  r.stddev_us = std::sqrt(acc / us.size());
  return r;
}

// ---------------------------------------------------------------------------
// Launch overhead
// ---------------------------------------------------------------------------

__global__ void empty_kernel() {}

// Per-launch cost of an empty kernel through the normal (non-graph) path.
// On an H100 this is typically 3-8 us, which at M=1 exceeds the entire kernel.
// Report it next to every latency number so a reader can see how much of the
// measurement is dispatch.
inline TimingResult measure_launch_overhead(int warmup = 100, int samples = 200,
                                            cudaStream_t s = nullptr) {
  cudaEvent_t a, b;
  QG_CHECK(cudaEventCreate(&a));
  QG_CHECK(cudaEventCreate(&b));

  for (int i = 0; i < warmup; ++i) empty_kernel<<<1, 1, 0, s>>>();
  QG_CHECK(cudaStreamSynchronize(s));

  const int inner = 100;  // amortize event overhead over many launches
  std::vector<double> us;
  us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    QG_CHECK(cudaEventRecord(a, s));
    for (int j = 0; j < inner; ++j) empty_kernel<<<1, 1, 0, s>>>();
    QG_CHECK(cudaEventRecord(b, s));
    QG_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    QG_CHECK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1e3 / inner);
  }
  QG_CHECK(cudaEventDestroy(a));
  QG_CHECK(cudaEventDestroy(b));
  return summarize(std::move(us), inner, /*graphed=*/false);
}

// ---------------------------------------------------------------------------
// Core timing
// ---------------------------------------------------------------------------

// Times `f` with each iteration launched normally. Includes per-launch
// overhead, which is the point: the difference against time_graphed() is how
// much dispatch is costing you.
//
// `f` is invoked as f(iteration_index, stream) so callers can rotate buffers.
template <typename F>
TimingResult time_looped(F&& f, int warmup = 20, int samples = 50,
                         int inner = 20, cudaStream_t s = nullptr) {
  cudaEvent_t a, b;
  QG_CHECK(cudaEventCreate(&a));
  QG_CHECK(cudaEventCreate(&b));

  int it = 0;
  for (int i = 0; i < warmup; ++i) f(it++, s);
  QG_CHECK(cudaStreamSynchronize(s));

  std::vector<double> us;
  us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    QG_CHECK(cudaEventRecord(a, s));
    for (int j = 0; j < inner; ++j) f(it++, s);
    QG_CHECK(cudaEventRecord(b, s));
    QG_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    QG_CHECK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1e3 / inner);
  }
  QG_CHECK(cudaEventDestroy(a));
  QG_CHECK(cudaEventDestroy(b));
  QG_CHECK(cudaGetLastError());
  return summarize(std::move(us), inner, /*graphed=*/false);
}

// Times `f` with `inner` iterations captured into a single CUDA graph, so the
// timed region contains one graph launch instead of `inner` kernel launches.
// This is the only honest way to measure a kernel whose runtime is comparable
// to its launch cost.
//
// Note the interaction with L2: iteration index j inside the graph is fixed at
// capture time, so buffer rotation still works (each captured node points at a
// different buffer) but the rotation period is `inner`, not unbounded. Choose
// `inner >= rotation_count(...)` so a full cycle fits inside the graph.
template <typename F>
TimingResult time_graphed(F&& f, int warmup = 20, int samples = 50,
                          int inner = 20, cudaStream_t stream = nullptr) {
  cudaStream_t s = stream;
  bool owns_stream = false;
  if (s == nullptr) {
    QG_CHECK(cudaStreamCreate(&s));
    owns_stream = true;
  }

  int it = 0;
  for (int i = 0; i < warmup; ++i) f(it++, s);
  QG_CHECK(cudaStreamSynchronize(s));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  QG_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int j = 0; j < inner; ++j) f(j, s);
  QG_CHECK(cudaStreamEndCapture(s, &graph));
  QG_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  // Warm the graph itself; the first launch pays instantiation-side costs.
  for (int i = 0; i < 3; ++i) QG_CHECK(cudaGraphLaunch(exec, s));
  QG_CHECK(cudaStreamSynchronize(s));

  cudaEvent_t a, b;
  QG_CHECK(cudaEventCreate(&a));
  QG_CHECK(cudaEventCreate(&b));

  std::vector<double> us;
  us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    QG_CHECK(cudaEventRecord(a, s));
    QG_CHECK(cudaGraphLaunch(exec, s));
    QG_CHECK(cudaEventRecord(b, s));
    QG_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    QG_CHECK(cudaEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1e3 / inner);
  }

  QG_CHECK(cudaEventDestroy(a));
  QG_CHECK(cudaEventDestroy(b));
  QG_CHECK(cudaGraphExecDestroy(exec));
  QG_CHECK(cudaGraphDestroy(graph));
  if (owns_stream) QG_CHECK(cudaStreamDestroy(s));
  QG_CHECK(cudaGetLastError());
  return summarize(std::move(us), inner, /*graphed=*/true);
}

// ---------------------------------------------------------------------------
// Derived metrics
// ---------------------------------------------------------------------------

// Ideal time to stream `bytes` at `achieved_gbps`. The denominator must be an
// EMPIRICAL bandwidth from src/probe/bandwidth.cu, never the spec sheet --
// using the theoretical figure inflates every efficiency claim.
inline double ideal_us(size_t bytes, double achieved_gbps) {
  return static_cast<double>(bytes) / (achieved_gbps * 1e9) * 1e6;
}

// Percent of ideal: the headline metric. Self-normalizing across hardware and
// impossible to inflate, because the numerator and denominator are both
// measured on the same machine under the same protocol.
inline double pct_of_ideal(double measured_us, size_t bytes,
                           double achieved_gbps) {
  return 100.0 * ideal_us(bytes, achieved_gbps) / measured_us;
}

inline double achieved_gbps(size_t bytes, double measured_us) {
  return static_cast<double>(bytes) / (measured_us * 1e-6) / 1e9;
}

}  // namespace qgemm
