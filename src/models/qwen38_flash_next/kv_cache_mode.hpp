#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_MODE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_MODE_HPP_

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

/// Attention KV cache storage. The lossy modes trade numerical fidelity for
/// per-token bytes; snapshots of different modes are not interchangeable.
enum class KvCacheMode : std::uint8_t {
  kF16,      ///< Both planes as F16 halves (default).
  kQ8_0,     ///< Both planes as Q8_0 blocks (~47% fewer bytes).
  kQ8_0Q4K,  ///< Q8_0 keys, Q4_K values (~59% fewer bytes).
};

/// Canonical `--kv-cache` names, indexed by KvCacheMode.
inline constexpr std::string_view kKvCacheModeNames[] = {"f16", "q8_0",
                                                         "q8_0-q4_k"};
inline constexpr std::string_view kKvCacheModeList = "f16, q8_0, q8_0-q4_k";

/// Parses a canonical mode name into *mode; returns false with *mode
/// untouched for anything else.
constexpr bool ParseKvCacheMode(std::string_view name, KvCacheMode* mode) {
  for (std::size_t i = 0; i < std::size(kKvCacheModeNames); ++i) {
    if (name == kKvCacheModeNames[i]) {
      *mode = static_cast<KvCacheMode>(i);
      return true;
    }
  }
  return false;
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KV_CACHE_MODE_HPP_
