#include "qgemm/pack.hpp"
#include "qgemm/permute.hpp"

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

int main() {
  EXPECT(even_odd_slot(0) == 0, "k0");
  EXPECT(even_odd_slot(1) == 4, "k1");
  EXPECT(even_odd_slot(2) == 1, "k2");
  EXPECT(even_odd_slot(7) == 7, "k7");

  const std::size_t N = 3, K = 32;
  std::vector<float> W(N * K);
  for (std::size_t i = 0; i < W.size(); ++i)
    W[i] = 0.1f * static_cast<float>((i * 7) % 11) - 0.4f;
  PackedMxfp4 p = pack_mxfp4(W.data(), N, K);

  auto words = matrix_seq_to_even_odd(p.data.data(), N, K);
  EXPECT(words.size() == N * (K / 8), "word count");

  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      const auto a = unpack_nibble(p.data.data(), n * K + k);
      const auto b = unpack_nibble_even_odd(words.data(), n, k, K);
      if (a != b) {
        std::fprintf(stderr, "nibble mismatch n=%zu k=%zu seq=%u eo=%u\n", n, k,
                     a, b);
        ++g_fail;
      }
    }
  }

  std::vector<std::uint8_t> back(p.data.size(), 0);
  const std::size_t row_words = K / 8;
  const std::size_t row_bytes = K / 2;
  for (std::size_t n = 0; n < N; ++n) {
    row_even_odd_to_seq(words.data() + n * row_words, back.data() + n * row_bytes,
                        K);
  }
  EXPECT(back == p.data, "inverse permute bit-exact");

  bool threw = false;
  try {
    std::uint32_t tmp[1];
    row_seq_to_even_odd(p.data.data(), tmp, 12);
  } catch (...) {
    threw = true;
  }
  EXPECT(threw, "K not multiple of 8 rejected");

  if (g_fail) {
    std::fprintf(stderr, "test_permute: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_permute: OK\n");
  return 0;
}
