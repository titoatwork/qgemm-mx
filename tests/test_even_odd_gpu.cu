// Even/odd pack on device matches sequential nibble decode.
#include "qgemm/dequant_device.cuh"
#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"
#include "qgemm/timing.cuh"

#include <cstdio>
#include <vector>

using namespace qgemm;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

__global__ void k_cmp_layout(const uint8_t* seq, const unsigned* words, int N,
                             int K, int* bad) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N * K) return;
  const int n = i / K;
  const int k = i - n * K;
  const uint8_t a = dev::unpack_nibble(seq, static_cast<size_t>(i));
  const uint8_t b = dev::unpack_nibble_even_odd(
      words, static_cast<size_t>(n), static_cast<size_t>(k), static_cast<size_t>(K));
  if (a != b) atomicAdd(bad, 1);
}

__global__ void k_dual_vs_slots(const unsigned* words, int nwords, int* bad) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= nwords) return;
  const unsigned w = words[i];
  uint8_t ev[4], od[4];
  dev::dual_nibbles_from_word(w, ev, od);
  for (int s = 0; s < 4; ++s) {
    const uint8_t e = static_cast<uint8_t>((w >> (4 * s)) & 0xF);
    const uint8_t o = static_cast<uint8_t>((w >> (16 + 4 * s)) & 0xF);
    if (ev[s] != e || od[s] != o) atomicAdd(bad, 1);
  }
}

int main() {
  const int N = 8, K = 64;
  std::vector<float> W(static_cast<size_t>(N) * K);
  for (int i = 0; i < N * K; ++i) W[i] = 0.12f * ((i * 3) % 9) - 0.3f;
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);
  auto words = matrix_seq_to_even_odd(p.data.data(), N, K);

  uint8_t* dseq = nullptr;
  unsigned* dwo = nullptr;
  int* dbad = nullptr;
  QG_CHECK(cudaMalloc(&dseq, p.data.size()));
  QG_CHECK(cudaMalloc(&dwo, words.size() * sizeof(unsigned)));
  QG_CHECK(cudaMalloc(&dbad, 4));
  QG_CHECK(cudaMemcpy(dseq, p.data.data(), p.data.size(), cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemcpy(dwo, words.data(), words.size() * sizeof(unsigned),
                      cudaMemcpyHostToDevice));
  QG_CHECK(cudaMemset(dbad, 0, 4));
  const int NK = N * K;
  k_cmp_layout<<<(NK + 127) / 128, 128>>>(dseq, dwo, N, K, dbad);
  int h = -1;
  QG_CHECK(cudaMemcpy(&h, dbad, 4, cudaMemcpyDeviceToHost));
  EXPECT(h == 0, "device even-odd unpack == sequential");

  QG_CHECK(cudaMemset(dbad, 0, 4));
  k_dual_vs_slots<<<1, 256>>>(dwo, static_cast<int>(words.size()), dbad);
  QG_CHECK(cudaMemcpy(&h, dbad, 4, cudaMemcpyDeviceToHost));
  EXPECT(h == 0, "dual_nibbles_from_word matches slots");

  std::fprintf(stderr, "even-odd mismatches=%d dual=%d\n", h == 0 ? 0 : h, h);
  cudaFree(dseq);
  cudaFree(dwo);
  cudaFree(dbad);
  if (g_fail) {
    std::fprintf(stderr, "test_even_odd_gpu: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_even_odd_gpu: OK\n");
  return 0;
}
