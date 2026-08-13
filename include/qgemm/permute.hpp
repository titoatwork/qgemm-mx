// permute.hpp — sequential nibble stream ↔ even/odd 32-bit words (host).
//
// Sequential (pack.hpp): along K, byte holds (k even = low, k odd = high).
// Even/odd word (8 nibbles / 32 bits), slots from LSB:
//   [k0 k2 k4 k6 | k1 k3 k5 k7]
// so one 32-bit load yields four evens then four odds for dual convert.
//
// This is NOT a wgmma A-fragment. It is the next packing step after sequential.
// Inverse is exact. K must be a multiple of 8.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "qgemm/pack.hpp"

namespace qgemm {

inline constexpr int kNibblesPerWord = 8;

// Slot in the even/odd word for sequential offset k_off in 0..7.
inline constexpr int even_odd_slot(int k_off) {
  // even k_off -> 0,1,2,3 ; odd -> 4,5,6,7
  return (k_off & 1) ? (4 + (k_off >> 1)) : (k_off >> 1);
}

inline std::uint8_t nibble_from_u32(std::uint32_t w, int slot) {
  return static_cast<std::uint8_t>((w >> (4 * slot)) & 0xFu);
}

inline std::uint32_t u32_insert_nibble(std::uint32_t w, int slot,
                                       std::uint8_t nib) {
  const std::uint32_t mask = 0xFu << (4 * slot);
  return (w & ~mask) | (static_cast<std::uint32_t>(nib & 0xF) << (4 * slot));
}

// One row: sequential packed bytes (K/2) -> even/odd words (K/8 uint32).
inline void row_seq_to_even_odd(const std::uint8_t* seq, std::uint32_t* words,
                                std::size_t K) {
  if (K % 8 != 0) {
    throw std::invalid_argument("even/odd permute: K must be multiple of 8");
  }
  const std::size_t nwords = K / 8;
  for (std::size_t w = 0; w < nwords; ++w) {
    std::uint32_t out = 0;
    for (int off = 0; off < 8; ++off) {
      const std::uint8_t nib = unpack_nibble(seq, w * 8 + static_cast<std::size_t>(off));
      out = u32_insert_nibble(out, even_odd_slot(off), nib);
    }
    words[w] = out;
  }
}

inline void row_even_odd_to_seq(const std::uint32_t* words, std::uint8_t* seq,
                                std::size_t K) {
  if (K % 8 != 0) {
    throw std::invalid_argument("even/odd permute: K must be multiple of 8");
  }
  const std::size_t nwords = K / 8;
  std::memset(seq, 0, K / 2);
  for (std::size_t w = 0; w < nwords; ++w) {
    const std::uint32_t in = words[w];
    for (int off = 0; off < 8; ++off) {
      pack_nibble(seq, w * 8 + static_cast<std::size_t>(off),
                  nibble_from_u32(in, even_odd_slot(off)));
    }
  }
}

inline std::vector<std::uint32_t> matrix_seq_to_even_odd(
    const std::uint8_t* seq, std::size_t N, std::size_t K) {
  std::vector<std::uint32_t> words(N * (K / 8));
  const std::size_t row_bytes = K / 2;
  const std::size_t row_words = K / 8;
  for (std::size_t n = 0; n < N; ++n) {
    row_seq_to_even_odd(seq + n * row_bytes, words.data() + n * row_words, K);
  }
  return words;
}

// Unpack nibble k from even/odd word matrix (host).
inline std::uint8_t unpack_nibble_even_odd(const std::uint32_t* words,
                                           std::size_t n, std::size_t k,
                                           std::size_t K) {
  const std::size_t row_words = K / 8;
  const std::uint32_t w = words[n * row_words + (k / 8)];
  return nibble_from_u32(w, even_odd_slot(static_cast<int>(k % 8)));
}

}  // namespace qgemm
