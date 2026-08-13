#include "qgemm/a_fragment.hpp"
#include "qgemm/occupancy.hpp"
#include "qgemm/shapes.hpp"

#include <cstdio>
#include <vector>

using namespace qgemm;
using namespace qgemm::wgmma;

static int g_fail = 0;
#define EXPECT(c, m)                                                          \
  do {                                                                        \
    if (!(c)) {                                                               \
      std::fprintf(stderr, "FAIL: %s\n", m);                                  \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void test_a_bijection() {
  bool seen[kAtomM][kAtomK] = {};
  for (int tid = 0; tid < kWarpgroupThreads; ++tid) {
    for (int i = 0; i < kElemsAPerThread; ++i) {
      const FragCoord c = a_fragment_coord(tid, i);
      EXPECT(c.row >= 0 && c.row < kAtomM, "A row range");
      EXPECT(c.col >= 0 && c.col < kAtomK, "A col range");
      if (seen[c.row][c.col]) {
        std::fprintf(stderr, "A collision tid=%d i=%d (%d,%d)\n", tid, i, c.row,
                     c.col);
        ++g_fail;
      }
      seen[c.row][c.col] = true;
      const FragOwner o = a_fragment_owner(c.row, c.col);
      if (o.tid != tid || o.i != i) {
        std::fprintf(stderr, "A inverse tid=%d i=%d -> owner (%d,%d)\n", tid, i,
                     o.tid, o.i);
        ++g_fail;
      }
    }
  }
  int owned = 0;
  for (int r = 0; r < kAtomM; ++r)
    for (int c = 0; c < kAtomK; ++c)
      if (seen[r][c]) ++owned;
  EXPECT(owned == kAtomM * kAtomK, "A covers 64x16 exactly once");
}

static void test_c_bijection() {
  bool seen[kAtomM][kAtomN] = {};
  for (int tid = 0; tid < kWarpgroupThreads; ++tid) {
    for (int i = 0; i < kElemsCPerThreadN8; ++i) {
      const FragCoord c = c_fragment_coord_n8(tid, i);
      EXPECT(c.row >= 0 && c.row < kAtomM, "C row range");
      EXPECT(c.col >= 0 && c.col < kAtomN, "C col range");
      if (seen[c.row][c.col]) {
        std::fprintf(stderr, "C collision tid=%d i=%d (%d,%d)\n", tid, i, c.row,
                     c.col);
        ++g_fail;
      }
      seen[c.row][c.col] = true;
      const FragOwner o = c_fragment_owner_n8(c.row, c.col);
      if (o.tid != tid || o.i != i) {
        std::fprintf(stderr, "C inverse tid=%d i=%d -> owner (%d,%d)\n", tid, i,
                     o.tid, o.i);
        ++g_fail;
      }
    }
  }
  int owned = 0;
  for (int r = 0; r < kAtomM; ++r)
    for (int c = 0; c < kAtomN; ++c)
      if (seen[r][c]) ++owned;
  EXPECT(owned == kAtomM * kAtomN, "C covers 64x8 exactly once");
}

static void test_ptx_tid0() {
  // Official map, warp 0 lane 0: groupID=0, tg=0
  // i=0,1 row=0  col=0,1
  // i=2,3 row=8  col=0,1
  // i=4,5 row=0  col=8,9
  // i=6,7 row=8  col=8,9
  const int er[8] = {0, 0, 8, 8, 0, 0, 8, 8};
  const int ec[8] = {0, 1, 0, 1, 8, 9, 8, 9};
  for (int i = 0; i < 8; ++i) {
    const FragCoord c = a_fragment_coord(0, i);
    if (c.row != er[i] || c.col != ec[i]) {
      std::fprintf(stderr, "tid0 a%d got (%d,%d) want (%d,%d)\n", i, c.row,
                   c.col, er[i], ec[i]);
      ++g_fail;
    }
  }
  // Warp 1 lane 0 is the same pattern +16 rows.
  const FragCoord w1 = a_fragment_coord(32, 0);
  EXPECT(w1.row == 16 && w1.col == 0, "warp 1 owns rows 16..31");
}

static void test_not_even_odd() {
  // even/odd word 0 of row 0 owns k=0..7 of that row only.
  // tid 0 owns two rows and skips k=2..7. Intersection is just (0,0),(0,1).
  int same_as_row0_k07 = 0;
  for (int i = 0; i < 8; ++i) {
    const FragCoord c = a_fragment_coord(0, i);
    if (c.row == 0 && c.col < 8) ++same_as_row0_k07;
  }
  EXPECT(same_as_row0_k07 == 2, "A-fragment is not the even/odd 8-wide word");
}

static void test_roundtrip_and_2880() {
  const std::size_t N = 64, K = 32;
  std::vector<float> W(N * K);
  for (std::size_t i = 0; i < W.size(); ++i)
    W[i] = 0.1f * static_cast<float>((i * 7) % 11) - 0.4f;
  PackedMxfp4 seq = pack_mxfp4(W.data(), N, K);
  auto words = permute_seq_to_a_fragment(seq.data.data(), N, K);
  EXPECT(words.size() == a_fragment_word_count(N, K), "word count");

  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = 0; k < K; ++k) {
      const auto a = unpack_nibble(seq.data.data(), n * K + k);
      const auto b = unpack_nibble_a_fragment(words.data(), n, k, N, K);
      if (a != b) {
        std::fprintf(stderr, "nibble mismatch n=%zu k=%zu seq=%u af=%u\n", n, k,
                     a, b);
        ++g_fail;
      }
    }
  }

  std::vector<std::uint8_t> back(seq.data.size(), 0);
  permute_a_fragment_to_seq(words.data(), back.data(), N, K);
  EXPECT(back == seq.data, "inverse permute bit-exact");

  PackedMxfp4AFrag packed = pack_mxfp4_a_fragment(W.data(), N, K);
  EXPECT(packed.scales == seq.scales, "scales stay sequential");

  // GPT-OSS 2880 is 64- and 16-aligned — the live awkward shape.
  EXPECT(2880 % kAtomM == 0 && 2880 % kAtomK == 0, "2880 fits A-fragment tiles");
  const std::size_t N2 = 64, K2 = 2880;
  std::vector<float> W2(N2 * K2, 0.25f);
  PackedMxfp4AFrag p2 = pack_mxfp4_a_fragment(W2.data(), N2, K2);
  PackedMxfp4 s2 = pack_mxfp4(W2.data(), N2, K2);
  EXPECT(p2.words.size() == a_fragment_word_count(N2, K2), "2880 word count");
  EXPECT(unpack_nibble_a_fragment(p2.words.data(), 0, 0, N2, K2) ==
             unpack_nibble(s2.data.data(), 0),
         "2880 nibble 0");
  EXPECT(unpack_nibble_a_fragment(p2.words.data(), 8, 9, N2, K2) ==
             unpack_nibble(s2.data.data(), 8 * K2 + 9),
         "2880 tid0 a7 of tile 0");

  bool threw = false;
  try {
    require_a_fragment_shape(32, 32);
  } catch (...) {
    threw = true;
  }
  EXPECT(threw, "N=32 rejected");
}

static void test_frozen_shapes_align() {
  for (std::size_t i = 0; i < kNumLayerShapes; ++i) {
    const auto& L = kLayerShapes[i];
    if (L.N % static_cast<std::size_t>(kAtomM) != 0 ||
        L.K % static_cast<std::size_t>(kAtomK) != 0) {
      std::fprintf(stderr, "frozen shape %s N=%zu K=%zu not 64x16 aligned\n",
                   L.name, L.N, L.K);
      ++g_fail;
    }
  }
}

static void test_occupancy_atom_tile() {
  // CTA tile = one atom (64) vs fused-ref assumption (256).
  const auto h256 = estimate_occupancy(4096, kSmCountH100SXM, 256);
  const auto h64 = estimate_occupancy(4096, kSmCountH100SXM, kTileNWgmmaAtom);
  EXPECT(h256.ctas == 16, "256-tile CTA count unchanged");
  EXPECT(h64.ctas == 64, "64-tile → 64 CTAs");
  EXPECT(h64.occupancy > 0.48 && h64.occupancy < 0.50, "64/132 ≈ 0.485");
  EXPECT(h64.starved, "atom-sized CTAs still starve H100");
  const int sk = splitk_to_fill(4096, kSmCountH100SXM, kTileNWgmmaAtom, 2.0);
  EXPECT(sk == 5, "split-K 5 to reach ~2×132");
  EXPECT(64 * sk >= 2 * kSmCountH100SXM, "oversub");
}

int main() {
  EXPECT(kAtomM == 64 && kAtomN == 8 && kAtomK == 16, "first opcode shape");
  test_a_bijection();
  test_c_bijection();
  test_ptx_tid0();
  test_not_even_odd();
  test_roundtrip_and_2880();
  test_frozen_shapes_align();
  test_occupancy_atom_tile();

  if (g_fail) {
    std::fprintf(stderr, "test_a_fragment: %d fail\n", g_fail);
    return 1;
  }
  std::printf("test_a_fragment: OK  opcode=%s\n", kFirstOpcode);
  return 0;
}
