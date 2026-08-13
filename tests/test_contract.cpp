#include "qgemm/kernel_contract.hpp"
#include "qgemm/pack.hpp"

#include <cstdio>

using namespace qgemm;
using namespace qgemm::contract;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

int main() {
  LogicalGemm g{/*M*/ 1, /*N*/ 4096, /*K*/ 4096};
  const WgmmaView v = wgmma_view(g);
  EXPECT(v.A_rows == 4096 && v.A_cols == 4096, "A is W [N,K]");
  EXPECT(v.B_rows == 4096 && v.B_cols == 1, "B is X^T [K,M]");
  EXPECT(v.C_rows == 4096 && v.C_cols == 1, "C is Y^T [N,M]");

  const DecodePlan d86 = plan_decode(4096, kSmCountLaptop3050);
  const DecodePlan d90 = plan_decode(4096, kSmCountH100SXM);
  EXPECT(d86.n_ctas == 16, "N tiles");
  EXPECT(d90.split_k >= 16, "H100 needs split-K");
  EXPECT(d90.total_ctas >= 2 * kSmCountH100SXM, "oversub ~2x");

  EXPECT(k_multiple_ok(4096, kMxBlock), "4096 % 32");
  EXPECT(k_multiple_ok(2880, kMxBlock), "2880 % 32");
  EXPECT(!k_multiple_ok(2880, kInt4Group), "2880 % 128");
  EXPECT(k_multiple_ok(2880, kNvBlock), "2880 % 16");

  if (g_fail) {
    std::fprintf(stderr, "test_contract: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_contract: OK\n");
  return 0;
}
