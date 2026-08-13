#include "qgemm/a_fragment.hpp"
#include "qgemm/kernel_contract.hpp"
#include "qgemm/wgmma_smem_desc.hpp"

#include <cstdio>

using namespace qgemm;
using namespace qgemm::contract;
using namespace qgemm::wgmma;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void test_scale_contract() {
  EXPECT(kMxAtomsPerScale == 2, "MX = 2 × k16");
  EXPECT(kNvAtomsPerScale == 1, "NV = 1 × k16");
  EXPECT(ktiles_share_mx_scale(0, 1), "k16 0 and 1 share MX scale");
  EXPECT(!ktiles_share_mx_scale(1, 2), "k16 1 and 2 do not");
  EXPECT(mx_scale_block(0) == 0 && mx_scale_block(31) == 0, "k 0..31 block 0");
  EXPECT(mx_scale_block(32) == 1, "k 32 block 1");
  EXPECT(mx_scale_block_from_ktile(0) == mx_scale_block_from_ktile(1),
         "ktile pair");
  EXPECT(k_range_respects_group(0, 64, kMxBlock), "split on MX edge");
  EXPECT(!k_range_respects_group(0, 48, kMxBlock), "48 is not an MX edge");
  EXPECT(k_range_respects_group(0, 16, kNvBlock), "NV k16 is an edge");

  // Live awkward shape: 2880 / 32 = 90 MX blocks = 180 k16 atoms.
  EXPECT(k_multiple_ok(2880, kMxBlock), "2880 MX");
  EXPECT((2880 / kWgmmaAtomK) % kMxAtomsPerScale == 0, "2880 even k16 pairs");
}

static void test_desc_roundtrip() {
  SmemDescFields in;
  in.start = 256;
  in.leading = 128;
  in.stride = 16;
  in.base_offset = 3;
  in.swizzle = SmemSwizzle::B128;
  const std::uint64_t d = encode_smem_desc(in);
  const SmemDescFields out = decode_smem_desc(d);
  EXPECT(out.start == in.start, "start");
  EXPECT(out.leading == in.leading, "LBO");
  EXPECT(out.stride == in.stride, "SBO");
  EXPECT(out.base_offset == in.base_offset, "base");
  EXPECT(out.swizzle == in.swizzle, "swizzle");

  // Official field positions: start in [13:0] as addr>>4.
  EXPECT((d & 0x3FFFull) == (256ull >> 4), "start field");
  EXPECT(((d >> 16) & 0x3FFFull) == (128ull >> 4), "LBO field");
  EXPECT(((d >> 32) & 0x3FFFull) == (16ull >> 4), "SBO field");
  EXPECT(((d >> 49) & 7ull) == 3ull, "base field");
  EXPECT(((d >> 62) & 3ull) == 3ull, "swizzle field");

  bool threw = false;
  try {
    SmemDescFields bad;
    bad.start = 1;
    (void)encode_smem_desc(bad);
  } catch (...) {
    threw = true;
  }
  EXPECT(threw, "misaligned start rejected");
}

static void test_b_tile() {
  EXPECT(kBTileK == kAtomK && kBTileN == kAtomN, "B tile = first opcode");
  EXPECT(kBTileBytes == 256, "16*8*2");
  EXPECT(kTmaBBoxFirstOpcode.box_k == 16 && kTmaBBoxFirstOpcode.box_n == 8,
         "TMA box");
  const SmemDescFields b = b_tile_mn_major_n8(0);
  EXPECT(b.leading == 128 && b.stride == 16, "MN-major cores");
  EXPECT(b.swizzle == SmemSwizzle::None, "no invented swizzle");
  const std::uint64_t d = encode_smem_desc(b);
  EXPECT(decode_smem_desc(d).leading == 128, "example encodes");
}

int main() {
  test_scale_contract();
  test_desc_roundtrip();
  test_b_tile();
  if (g_fail) {
    std::fprintf(stderr, "test_wgmma_desc: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_wgmma_desc: OK\n");
  return 0;
}
