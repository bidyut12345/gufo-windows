#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace gufo::minimax_h3::dit_ops {

struct SparseAttentionHints {
  std::uint32_t protected_prefix_rows{2048U};
  std::uint32_t audio_begin_row{0U};
  std::uint32_t audio_end_row{0U};
  std::uint32_t video_begin_row{0U};
  std::uint32_t video_frame_rows{0U};
};

// Dense, noncausal BF16 attention, head dimension 128. Inputs are head-major;
// output is row-major, ready for the attention output projection. For >4096
// rows, packed_values must hold roundUp(rows, 32) * heads * 128 BF16 values
// disjoint from the inputs/output. Short sequences do not use this scratch.
void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint16_t* packed_values, std::uint32_t rows,
                         std::uint32_t heads, hipStream_t stream,
                         const SparseAttentionHints& hints);

void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint16_t* packed_values, std::uint32_t rows,
                         std::uint32_t heads, hipStream_t stream,
                         std::uint32_t protected_prefix_rows = 2048U);

// Long attention with QKV-provided packed values. Output is written directly
// in [heads*128/16][row][16] layout for the native attention output projection.
void LaunchProjectionAttention(const std::uint16_t* query,
                               const std::uint16_t* key,
                               const std::uint16_t* packed_values,
                               std::uint16_t* output, std::uint32_t rows,
                               std::uint32_t heads, hipStream_t stream,
                               const SparseAttentionHints& hints);

void LaunchProjectionAttention(const std::uint16_t* query,
                               const std::uint16_t* key,
                               const std::uint16_t* packed_values,
                               std::uint16_t* output, std::uint32_t rows,
                               std::uint32_t heads, hipStream_t stream,
                               std::uint32_t protected_prefix_rows = 2048U);

}  // namespace gufo::minimax_h3::dit_ops
