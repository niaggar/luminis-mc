#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace luminis::math {

std::uint64_t mix_seed(std::uint64_t base, std::uint64_t tid);

/**
 * @brief Per-photon-stream RNG backed by xoshiro256++.
 *
 * xoshiro256++ is a fast, high-quality 256-bit-state generator (Blackman &
 * Vigna).  It replaces `std::mt19937_64` + `std::uniform_real_distribution`,
 * both of which are comparatively slow and, in the distribution's case,
 * implementation-defined.  The state is seeded by streaming SplitMix64 from the
 * 64-bit seed, the canonical initialization recipe for the xoshiro family.
 *
 * Each worker thread owns its own `Rng` seeded via `mix_seed(seed, tid)`, so the
 * streams are independent and the result is deterministic for a fixed seed and
 * thread count (though not bit-identical to the previous mt19937 output).
 */
struct Rng {
  std::uint64_t s[4];

  explicit Rng(std::uint64_t seed = std::random_device{}()) { seed_state(seed); }

  /// Re-seed the 256-bit state from a single 64-bit value via SplitMix64.
  void seed_state(std::uint64_t seed) {
    std::uint64_t z = seed;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      std::uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      s[i] = x ^ (x >> 31);
    }
  }

  static inline std::uint64_t rotl(std::uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  /// xoshiro256++ next 64-bit output.
  inline std::uint64_t next_u64() {
    const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const std::uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }

  double uniform();
  double normal(const double mean, const double stddev);
};

} // namespace luminis::math
