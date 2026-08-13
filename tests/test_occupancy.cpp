#include "qgemm/occupancy.hpp"
#include "qgemm/shapes.hpp"

#include <cstdio>
#include <cstdlib>

using namespace qgemm;

static int g_fail = 0;
#define EXPECT(c, msg)                                                        \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", msg);                                \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

int main() {
  EXPECT(ctas_n_partition(4096, 256) == 16, "4096/256 = 16 CTAs");
  EXPECT(ctas_n_partition(2880, 256) == 12, "2880/256 = 12 CTAs (ceil)");
  EXPECT(ctas_n_partition(1024, 256) == 4, "GQA small N");

  const auto laptop = estimate_occupancy(4096, kSmCountLaptop3050, 256);
  EXPECT(laptop.ctas == 16, "laptop ctas");
  EXPECT(laptop.occupancy > 0.7 && laptop.occupancy < 0.9, "16/20 = 0.8");
  EXPECT(!laptop.starved, "20 SM almost full — wave quant hidden");

  const auto h100 = estimate_occupancy(4096, kSmCountH100SXM, 256);
  EXPECT(h100.ctas == 16, "h100 ctas");
  EXPECT(h100.occupancy < 0.15, "16/132 ≈ 0.12");
  EXPECT(h100.starved, "H100 starved without split-K");

  const int sk = splitk_to_fill(4096, kSmCountH100SXM, 256, 2.0);
  EXPECT(sk >= 16, "need many K-splits to fill 132 SMs");
  EXPECT(16 * sk >= 2 * kSmCountH100SXM, "oversubscribe ~2x");

  // Print a table for D1 / report.
  std::printf("layer,N,tile,ctas,occ_sm86,occ_sm90,splitk_h100_2x\n");
  for (size_t i = 0; i < kNumLayerShapes; ++i) {
    const auto& L = kLayerShapes[i];
    const auto a = estimate_occupancy(L.N, kSmCountLaptop3050);
    const auto b = estimate_occupancy(L.N, kSmCountH100SXM);
    const int s = splitk_to_fill(L.N, kSmCountH100SXM);
    std::printf("%s,%zu,256,%d,%.3f,%.3f,%d\n", L.name, L.N, a.ctas,
                a.occupancy, b.occupancy, s);
  }

  if (g_fail) {
    std::fprintf(stderr, "test_occupancy: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_occupancy: OK\n");
  return 0;
}
