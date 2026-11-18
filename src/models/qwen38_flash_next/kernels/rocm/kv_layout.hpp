// KV cache element formats. The F16 cache stores one __half per element;
// the Q8_0 cache stores GGML-style 32-element blocks with the F32→int8
// codes packed first and their F16 scales trailing, so vector loads of the
// codes stay aligned and the dequant multiply is one scale per block:
// [kv_row int8][kv_row/32 F16 scales] = kv_row + kv_row/16 bytes per row.
// The Q4_K cache stores GGML-style 256-element super-blocks: four-bit codes
// first, then per super-block one F16 scale, one F16 min scale, and eight
// 6-bit sub-block scale/min pairs packed into 12 bytes:
// [kv_row/2 code bytes][kv_row/256 * 16 scale bytes] = kv_row/2 + kv_row/16
// bytes per row. Code nibbles are pair-interleaved (byte 2l low = element
// 2l, high = 2l+1) and a code dequantizes to code * (d * sc) - dmin * m by
// one FMA; sub-block sc/m unpack per GGML's get_scale_min_k4 layout.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

enum class KvCacheDtype : std::uint32_t {
  kF16 = 0,
  kQ8_0 = 1,
  kQ4_K = 2,
};

constexpr std::uint32_t kKvBlock = 32;        // elements per Q8_0 scale block
constexpr std::uint32_t kKvSuperBlock = 256;  // elements per Q4_K super-block

/// Key and value planes may use different formats; K is never quantized
/// below Q8_0 because softmax-amplified logits amplify K error.
struct KvDtypes {
  KvCacheDtype key{KvCacheDtype::kF16};
  KvCacheDtype value{KvCacheDtype::kF16};

  friend constexpr bool operator==(const KvDtypes& a, const KvDtypes& b) {
    return a.key == b.key && a.value == b.value;
  }
  friend constexpr bool operator!=(const KvDtypes& a, const KvDtypes& b) {
    return !(a == b);
  }
};

inline constexpr std::size_t KvRowBytes(std::size_t kv_row_elems,
                                        KvCacheDtype dtype) {
  switch (dtype) {
    case KvCacheDtype::kQ4_K:
      return kv_row_elems / 2 + (kv_row_elems / kKvSuperBlock) * 16;
    case KvCacheDtype::kQ8_0:
      return kv_row_elems + (kv_row_elems / kKvBlock) * 2;
    case KvCacheDtype::kF16:
      break;
  }
  return kv_row_elems * 2;
}

}  // namespace gufo::models::qwen38_flash_next
