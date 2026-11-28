// Short-path softmax ordering adapted from Composable Kernel, ROCm 7.2.3.
// Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
// MIT; see licenses/composable_kernel.txt. No CK headers/runtime are required.
#include <cstddef>

#include "src/models/minimax_h3/attention.hip.hpp"

namespace gufo::minimax_h3::dit_ops {
using BufferResource = __attribute__((ext_vector_type(4))) unsigned;
__device__ BufferResource
LoadBuffer(BufferResource, unsigned, unsigned,
           unsigned) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace {

using Fragment = __attribute__((ext_vector_type(16))) short;
using Accumulator = __attribute__((ext_vector_type(8))) float;

__device__ BufferResource MakeBuffer(const std::uint16_t* pointer,
                                     unsigned bytes) {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  return {unsigned(address), unsigned(address >> 32), bytes, 0x31004000U};
}
__device__ uint4 LoadBufferVector(BufferResource resource, unsigned offset) {
  union {
    BufferResource vector;
    uint4 packed;
  } result;
  result.vector = LoadBuffer(resource, offset, 0, 0);
  return result.packed;
}

__device__ Fragment LoadFragment(const std::uint16_t* source) {
  union {
    Fragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

__device__ std::uint16_t Bf16(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned nan = 0U - unsigned((bits & 0x7fffffffU) > 0x7f800000U);
  const unsigned rounded = bits + 0x7fffU + ((bits >> 16U) & 1U);
  return ((rounded & ~nan) | ((bits | 0x00400000U) & nan)) >> 16U;
}

__global__ void PackValues(const std::uint16_t* input, std::uint16_t* output,
                           unsigned rows) {
  __shared__ __align__(16) std::uint16_t tile[32][136];
  const unsigned tid = threadIdx.x;
  const unsigned padded_rows = (rows + 31U) & ~31U;
  input += std::size_t(blockIdx.y) * rows * 128;
  output += std::size_t(blockIdx.y) * padded_rows * 128;
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned r = tid / 16 + z * 16, d = (tid % 16) * 8;
    uint4 x{};
    if (blockIdx.x * 32 + r < rows)
      x = *reinterpret_cast<const uint4*>(
          input + (std::size_t(blockIdx.x) * 32 + r) * 128 + d);
    *reinterpret_cast<uint4*>(tile[r] + d) = x;
  }
  __syncthreads();
#pragma unroll
  for (unsigned z = 0; z < 2; ++z) {
    const unsigned d = tid / 4 + z * 64, r = (tid % 4) * 8;
    union {
      uint4 packed;
      std::uint16_t half[8];
    } value;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i)
      value.half[i] = tile[r + i][d];
    *reinterpret_cast<uint4*>(output + std::size_t(d) * padded_rows +
                              blockIdx.x * 32 + r) = value.packed;
  }
}

// Values are packed once into dead projection scratch. K and V then reuse
// one LDS allocation in separate phases; vector fragment loads preserve the
// original 32-key softmax/WMMA arithmetic without scalar value gathers.
// Eight waves own sixteen queries each. Transposed QK/PV tiles keep each
// query's softmax reduction within two lanes. Q/K/V and PV inputs use BF16;
// scores, normalization and value accumulation use FP32. No quadratic scratch
// or global probability/LSE buffer. Qualified against the independent teacher.
template<bool PackedOutput>
__global__ __launch_bounds__(256, 1) void WmmaAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t heads) {
  union Staging {
    std::uint16_t key[32][136];
    std::uint16_t value[128][40];
  };
  __shared__ Staging staging;
  auto& staged_key = staging.key;
  auto& staged_value = staging.value;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int first_query = blockIdx.x * 128 + wave * 16;
  const std::size_t head_base = std::size_t(blockIdx.y) * rows * 128;
  query += head_base;
  key += head_base;
  const unsigned padded_rows = (rows + 31U) & ~31U;
  value += std::size_t(blockIdx.y) * padded_rows * 128;
  // Bounded per-head descriptors zero masked key tails without reading the
  // next head. The admitted <=131072 rows fit 32-bit byte offsets.
  const auto key_buffer = MakeBuffer(key, rows * 256);
  const auto value_buffer = MakeBuffer(value, padded_rows * 256);
  Fragment q[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < rows)
      q[d] = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                          d * 16);
  }
  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0;

  for (std::uint32_t base = 0; base < rows; base += 32) {
    // Share each global K/V vector across all eight query waves.
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int kr = tid / 16 + z * 16, d = (tid % 16) * 8;
      const unsigned offset = ((base + kr) * 128 + d) * 2;
      const uint4 k = LoadBufferVector(key_buffer, offset);

      *reinterpret_cast<uint4*>(staged_key[kr] + d) = k;
    }
    __syncthreads();
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const Fragment k =
            LoadFragment(staged_key[half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int vd = tid / 4 + z * 64, vk = (tid % 4) * 8;
      const uint4 v =
          LoadBufferVector(value_buffer, (vd * padded_rows + base + vk) * 2);
      *reinterpret_cast<uint4*>(staged_value[vd] + vk) = v;
    }
    if (base + 32 > rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + half * 16 + 2 * i + lane / 16 >= rows)
            scores[half][i] = -INFINITY;
    }
    float next = maximum;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // log2(e) / sqrt(128), for the base-two online softmax.
        scores[half][i] *= 0.12751743082459868F;
        next = fmaxf(next, scores[half][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));
    __syncthreads();
    const float correction = __builtin_amdgcn_exp2f(maximum - next);
    float subtotal = 0;
    Fragment probability_fragments[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(scores[half][i] - next);
        subtotal += p;
        const unsigned own = Bf16(p);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability_fragments[half][2 * i] = lane < 16 ? own : peer;
        probability_fragments[half][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum = sum * correction + subtotal;
    maximum = next;
#pragma unroll
    for (int d = 0; d < 8; ++d)
      result[d] *= correction;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const Fragment p = probability_fragments[half];
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        const Fragment v =
            LoadFragment(staged_value[d * 16 + lane % 16] + half * 16);
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, result[d]);
        if (d % 2 == 1)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows) {
    const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[PackedOutput
                   ? ((std::size_t(blockIdx.y) * 8 + d) * rows + row) * 16 +
                         2 * i + lane / 16
                   : (std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 +
                         2 * i + lane / 16] = Bf16(result[d][i] * reciprocal);
    }
  }
}

// Short sequences retain 128-key local softmax tiles and normalize after each
// tile. In particular, round local (not running-max-scaled) probabilities to
// BF16 before PV. Contiguous eight-key lane groups preserve the sum tree.
// Explicit FMAs below preserve the rounded recurrence across all 50 blocks.
__global__ __launch_bounds__(128) void ShortAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t heads) {
#pragma clang fp contract(off)
  union Staging {
    std::uint16_t key[128][136];
    struct {
      std::uint16_t probability[64][136];
      std::uint16_t value[128][136];
    } pv;
  };
  __shared__ __align__(16) Staging staging;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int swizzled_lane = ((lane & 1) << 3) | ((lane % 16) >> 1);
  const int first_query = blockIdx.x * 64 + wave * 16;
  const std::size_t head_base = std::size_t(blockIdx.y) * rows * 128;
  query += head_base;
  const auto key_buffer = MakeBuffer(key + head_base, rows * 256);
  const auto value_buffer = MakeBuffer(value + head_base, rows * 256);
  Accumulator result[8]{};
  float running_sum = 0, running_max = -INFINITY;
  for (std::uint32_t base = 0; base < rows; base += 128) {
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int r = tid / 16 + z * 8, d = (tid % 16) * 8;
      *reinterpret_cast<uint4*>(staging.key[r] + d) =
          LoadBufferVector(key_buffer, ((base + r) * 128 + d) * 2);
    }
    __syncthreads();
    Accumulator scores[8]{};
#pragma unroll 1
    for (int d = 0; d < 8; ++d) {
      Fragment q{};
      if (static_cast<std::uint32_t>(first_query + lane % 16) < rows)
        q = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                         d * 16);
#pragma unroll
      for (int half = 0; half < 8; ++half) {
        const auto k =
            LoadFragment(staging.key[half * 16 + swizzled_lane] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q, scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    float local_max = -INFINITY, local_sum = 0;
#pragma unroll
    for (int half = 0; half < 8; ++half)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scores[half][i] = base + half * 16 + i + (lane / 16) * 8 < rows
                              ? scores[half][i] * 0.08838834764831845F
                              : -INFINITY;
        local_max = fmaxf(local_max, scores[half][i]);
      }
    local_max = fmaxf(local_max, __shfl_xor(local_max, 16));
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 8; ++half)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = expf(scores[half][i] - local_max);
        local_sum += p;
        staging.pv.probability[wave * 16 + lane % 16]
                              [half * 16 + i + (lane / 16) * 8] = Bf16(p);
      }
    local_sum += __shfl_xor(local_sum, 16);
#pragma unroll
    for (int z = 0; z < 16; ++z) {
      const int r = tid / 16 + z * 8, d = (tid % 16) * 8;
      *reinterpret_cast<uint4*>(staging.pv.value[r] + d) =
          LoadBufferVector(value_buffer, ((base + r) * 128 + d) * 2);
    }
    __syncthreads();
    const float next_max = fmaxf(running_max, local_max);
    const float previous_scale = running_sum * expf(running_max - next_max);
    const float local_scale = expf(local_max - next_max);
    const float next_sum = fmaf(local_scale, local_sum, previous_scale);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      Accumulator partial{};
#pragma unroll 1
      for (int half = 0; half < 8; ++half) {
        const auto p = LoadFragment(
            staging.pv.probability[wave * 16 + lane % 16] + half * 16);
        Fragment v;
#pragma unroll
        for (int i = 0; i < 16; ++i)
          v[i] = staging.pv.value[half * 16 + i][d * 16 + swizzled_lane];
        partial = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, partial);
        __builtin_amdgcn_sched_barrier(0);
      }
      for (int i = 0; i < 8; ++i)
        result[d][i] =
            fmaf(previous_scale, result[d][i], local_scale * partial[i]) /
            next_sum;
    }
    running_sum = next_sum;
    running_max = next_max;
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows)
#pragma unroll
    for (int d = 0; d < 8; ++d)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[(std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 + i +
               (lane / 16) * 8] = Bf16(result[d][i]);
}

// Coarse block-sparse screening for long sequences (rows >= 8192).
// Mean-pools Q (128-row blocks) and K (32-row blocks), builds a compact list
// of active 32-key tiles per (head, q_block), and runs the exact unmodified
// WmmaAttention inner loop over only the active tiles.
// Always retains the first 2048 tokens (text + Ref2VA conditions + audio) and
// a +/-512-token local window around each query block.
struct SparseAttentionScratch {
  std::uint16_t* q_pool{nullptr};
  std::size_t q_pool_bytes{0};
  std::uint16_t* k_pool{nullptr};
  std::size_t k_pool_bytes{0};
  std::uint16_t* active_tiles{nullptr};
  std::size_t active_tiles_bytes{0};
  std::uint32_t* active_counts{nullptr};
  std::size_t active_counts_bytes{0};

  ~SparseAttentionScratch() {
    if (q_pool != nullptr) (void)hipFree(q_pool);
    if (k_pool != nullptr) (void)hipFree(k_pool);
    if (active_tiles != nullptr) (void)hipFree(active_tiles);
    if (active_counts != nullptr) (void)hipFree(active_counts);
  }

  template<typename T>
  static bool Ensure(T** ptr, std::size_t* cap, std::size_t need) {
    if (*cap >= need) return true;
    if (*ptr != nullptr) {
      (void)hipFree(*ptr);
      *ptr = nullptr;
      *cap = 0;
    }
    if (hipMalloc(reinterpret_cast<void**>(ptr), need) != hipSuccess) {
      *ptr = nullptr;
      *cap = 0;
      return false;
    }
    *cap = need;
    return true;
  }

  bool Prepare(std::uint32_t rows, std::uint32_t heads) {
    const std::size_t q_blocks = (rows + 127U) / 128U;
    const std::size_t k_blocks = (rows + 31U) / 32U;
    return Ensure(&q_pool, &q_pool_bytes,
                  heads * q_blocks * 128U * sizeof(std::uint16_t)) &&
           Ensure(&k_pool, &k_pool_bytes,
                  heads * k_blocks * 128U * sizeof(std::uint16_t)) &&
           Ensure(&active_tiles, &active_tiles_bytes,
                  heads * q_blocks * k_blocks * sizeof(std::uint16_t)) &&
           Ensure(&active_counts, &active_counts_bytes,
                  heads * q_blocks * sizeof(std::uint32_t));
  }
};

SparseAttentionScratch& GetSparseScratch() {
  static SparseAttentionScratch scratch;
  return scratch;
}

__device__ __forceinline__ float Bf16ToF32(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

template<std::uint32_t BlockRows>
__global__ void PoolHeadRowsKernel(const std::uint16_t* input,
                                   std::uint16_t* pooled, std::uint32_t rows,
                                   std::uint32_t num_blocks) {
  const std::uint32_t d = threadIdx.x;  // 0..127
  const std::uint32_t blk = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  if (blk >= num_blocks || d >= 128U) return;
  const std::uint32_t row_start = blk * BlockRows;
  const std::uint32_t row_end =
      row_start + BlockRows < rows ? row_start + BlockRows : rows;
  const std::uint32_t count = row_end - row_start;
  const std::uint16_t* head_in =
      input + static_cast<std::size_t>(head) * rows * 128U;
  float acc = 0.0F;
  for (std::uint32_t r = row_start; r < row_end; ++r) {
    acc += Bf16ToF32(head_in[static_cast<std::size_t>(r) * 128U + d]);
  }
  const float inv = count > 0U ? (1.0F / static_cast<float>(count)) : 0.0F;
  pooled[(static_cast<std::size_t>(head) * num_blocks + blk) * 128U + d] =
      Bf16(acc * inv);
}

inline constexpr std::uint32_t kMaxSparseKBlocks = 4096U;

__global__ void BuildSparseTileListKernel(
    const std::uint16_t* q_pool, const std::uint16_t* k_pool,
    std::uint16_t* active_tiles, std::uint32_t* active_counts,
    std::uint32_t q_blocks, std::uint32_t k_blocks,
    std::uint32_t protected_prefix_tiles, std::uint32_t audio_begin_tile,
    std::uint32_t audio_end_tile, std::uint32_t video_begin_row,
    std::uint32_t video_frame_rows, float log2_threshold) {
  const std::uint32_t qb = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;  // 0..127
  if (qb >= q_blocks) return;

  __shared__ float shared_q[128];
  __shared__ float tile_scores[kMaxSparseKBlocks];
  __shared__ float reduce_max[128];

  const std::uint16_t* q_ptr =
      q_pool + (static_cast<std::size_t>(head) * q_blocks + qb) * 128U;
  // log2(e) / sqrt(128) = 0.12751743082459868F
  shared_q[tid] = Bf16ToF32(q_ptr[tid]) * 0.12751743082459868F;
  __syncthreads();

  const std::uint16_t* k_head =
      k_pool + static_cast<std::size_t>(head) * k_blocks * 128U;
  float local_max = -INFINITY;
  for (std::uint32_t kb = tid; kb < k_blocks; kb += 128U) {
    const std::uint16_t* k_ptr = k_head + static_cast<std::size_t>(kb) * 128U;
    float dot = 0.0F;
#pragma unroll 8
    for (std::uint32_t d = 0; d < 128U; ++d) {
      dot = fmaf(shared_q[d], Bf16ToF32(k_ptr[d]), dot);
    }
    if (kb < kMaxSparseKBlocks) {
      tile_scores[kb] = dot;
    }
    local_max = fmaxf(local_max, dot);
  }
  reduce_max[tid] = local_max;
  __syncthreads();
  for (std::uint32_t s = 64U; s > 0U; s >>= 1U) {
    if (tid < s) {
      reduce_max[tid] = fmaxf(reduce_max[tid], reduce_max[tid + s]);
    }
    __syncthreads();
  }
  const float cutoff = reduce_max[0] - log2_threshold;
  if (tid == 0U) {
    std::uint16_t* out_list =
        active_tiles +
        (static_cast<std::size_t>(head) * q_blocks + qb) * k_blocks;
    std::uint32_t count = 0U;
    const std::int32_t center_kb = static_cast<std::int32_t>(qb * 4U);
    for (std::uint32_t kb = 0U; kb < k_blocks; ++kb) {
      // Keep protected multimodal + image-reference prefix tiles, all audio
      // target tiles, last tile, +/-24 local tiles (+/-768 tokens), and
      // frame-0 / periodic video anchors so background and singing stay coherent.
      const bool in_audio =
          audio_end_tile > audio_begin_tile && kb >= audio_begin_tile &&
          kb < audio_end_tile;
      const bool in_local =
          static_cast<std::int32_t>(kb) >= center_kb - 24 &&
          static_cast<std::int32_t>(kb) <= center_kb + 24;
      bool video_anchor = false;
      if (video_frame_rows > 0U) {
        const std::uint32_t tile_start = kb * 32U;
        const std::uint32_t tile_end = tile_start + 31U;
        if (tile_end >= video_begin_row) {
          const std::uint32_t rel_start =
              tile_start > video_begin_row ? tile_start - video_begin_row : 0U;
          const std::uint32_t rel_end = tile_end - video_begin_row;
          const std::uint32_t f_start = rel_start / video_frame_rows;
          const std::uint32_t f_end = rel_end / video_frame_rows;
          video_anchor = (f_start % 12U == 0U) || (f_end % 12U == 0U);
        }
      }
      const bool protected_tile =
          kb < protected_prefix_tiles || in_audio || kb + 1U >= k_blocks ||
          in_local || video_anchor;
      const bool high_score = kb >= kMaxSparseKBlocks || tile_scores[kb] >= cutoff;
      if (protected_tile || high_score) {
        out_list[count++] = static_cast<std::uint16_t>(kb);
      }
    }
    active_counts[static_cast<std::size_t>(head) * q_blocks + qb] = count;
  }
}

template<bool PackedOutput>
__global__ __launch_bounds__(256, 1) void WmmaAttentionSparse(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output,
    const std::uint16_t* active_tiles, const std::uint32_t* active_counts,
    std::uint32_t rows, std::uint32_t heads, std::uint32_t q_blocks,
    std::uint32_t k_blocks) {
  union Staging {
    std::uint16_t key[32][136];
    std::uint16_t value[128][40];
  };
  __shared__ Staging staging;
  auto& staged_key = staging.key;
  auto& staged_value = staging.value;
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int first_query = blockIdx.x * 128 + wave * 16;
  const std::size_t head_base = std::size_t(blockIdx.y) * rows * 128;
  query += head_base;
  key += head_base;
  const unsigned padded_rows = (rows + 31U) & ~31U;
  value += std::size_t(blockIdx.y) * padded_rows * 128;
  const auto key_buffer = MakeBuffer(key, rows * 256);
  const auto value_buffer = MakeBuffer(value, padded_rows * 256);
  Fragment q[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < rows)
      q[d] = LoadFragment(query + std::size_t(first_query + lane % 16) * 128 +
                          d * 16);
  }
  Accumulator result[8]{};
  float maximum = -INFINITY, sum = 0;

  const std::size_t qb_slot =
      static_cast<std::size_t>(blockIdx.y) * q_blocks + blockIdx.x;
  const std::uint32_t tile_count = active_counts[qb_slot];
  const std::uint16_t* tile_list = active_tiles + qb_slot * k_blocks;

  for (std::uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const std::uint32_t base =
        static_cast<std::uint32_t>(tile_list[tile_idx]) * 32U;
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int kr = tid / 16 + z * 16, d = (tid % 16) * 8;
      const unsigned offset = ((base + kr) * 128 + d) * 2;
      const uint4 k = LoadBufferVector(key_buffer, offset);
      *reinterpret_cast<uint4*>(staged_key[kr] + d) = k;
    }
    __syncthreads();
    Accumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const Fragment k =
            LoadFragment(staged_key[half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
#pragma unroll
    for (int z = 0; z < 2; ++z) {
      const int vd = tid / 4 + z * 64, vk = (tid % 4) * 8;
      const uint4 v =
          LoadBufferVector(value_buffer, (vd * padded_rows + base + vk) * 2);
      *reinterpret_cast<uint4*>(staged_value[vd] + vk) = v;
    }
    if (base + 32 > rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + half * 16 + 2 * i + lane / 16 >= rows)
            scores[half][i] = -INFINITY;
    }
    float next = maximum;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        scores[half][i] *= 0.12751743082459868F;
        next = fmaxf(next, scores[half][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));
    __syncthreads();
    const float correction = __builtin_amdgcn_exp2f(maximum - next);
    float subtotal = 0;
    Fragment probability_fragments[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(scores[half][i] - next);
        subtotal += p;
        const unsigned own = Bf16(p);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability_fragments[half][2 * i] = lane < 16 ? own : peer;
        probability_fragments[half][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum = sum * correction + subtotal;
    maximum = next;
#pragma unroll
    for (int d = 0; d < 8; ++d)
      result[d] *= correction;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const Fragment p = probability_fragments[half];
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        const Fragment v =
            LoadFragment(staged_value[d * 16 + lane % 16] + half * 16);
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, result[d]);
        if (d % 2 == 1)
          __builtin_amdgcn_sched_barrier(0);
      }
    }
    __syncthreads();
  }
  const std::uint32_t row = first_query + lane % 16;
  if (row < rows) {
    const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        output[PackedOutput
                   ? ((std::size_t(blockIdx.y) * 8 + d) * rows + row) * 16 +
                         2 * i + lane / 16
                   : (std::size_t(row) * heads + blockIdx.y) * 128 + d * 16 +
                         2 * i + lane / 16] = Bf16(result[d][i] * reciprocal);
    }
  }
}

template<bool PackedOutput>
void DispatchLongAttention(const std::uint16_t* query, const std::uint16_t* key,
                           const std::uint16_t* packed_values,
                           std::uint16_t* output, std::uint32_t rows,
                           std::uint32_t heads, hipStream_t stream,
                           const SparseAttentionHints& hints) {
  const std::uint32_t q_blocks = (rows + 127U) / 128U;
  const std::uint32_t k_blocks = (rows + 31U) / 32U;
  if (rows >= 8192U && k_blocks <= kMaxSparseKBlocks &&
      GetSparseScratch().Prepare(rows, heads)) {
    const std::uint32_t protected_prefix_tiles =
        std::min(k_blocks, std::max<std::uint32_t>(
                               64U, (hints.protected_prefix_rows + 31U) / 32U));
    const std::uint32_t audio_begin_tile =
        hints.audio_end_row > hints.audio_begin_row
            ? (hints.audio_begin_row / 32U)
            : 0U;
    const std::uint32_t audio_end_tile =
        hints.audio_end_row > hints.audio_begin_row
            ? std::min(k_blocks, (hints.audio_end_row + 31U) / 32U)
            : 0U;
    SparseAttentionScratch& s = GetSparseScratch();
    PoolHeadRowsKernel<128U><<<dim3(q_blocks, heads), 128, 0, stream>>>(
        query, s.q_pool, rows, q_blocks);
    PoolHeadRowsKernel<32U><<<dim3(k_blocks, heads), 128, 0, stream>>>(
        key, s.k_pool, rows, k_blocks);
    // Keep tiles within 7.0 log2 units (~0.8% weight) of peak coarse score.
    BuildSparseTileListKernel<<<dim3(q_blocks, heads), 128, 0, stream>>>(
        s.q_pool, s.k_pool, s.active_tiles, s.active_counts, q_blocks, k_blocks,
        protected_prefix_tiles, audio_begin_tile, audio_end_tile,
        hints.video_begin_row, hints.video_frame_rows, 7.0F);
    WmmaAttentionSparse<PackedOutput><<<dim3(q_blocks, heads), 256, 0, stream>>>(
        query, key, packed_values, output, s.active_tiles, s.active_counts,
        rows, heads, q_blocks, k_blocks);
    return;
  }
  WmmaAttention<PackedOutput><<<dim3(q_blocks, heads), 256, 0, stream>>>(
      query, key, packed_values, output, rows, heads);
}

}  // namespace

void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint16_t* packed_values, std::uint32_t rows,
                         std::uint32_t heads, hipStream_t stream,
                         const SparseAttentionHints& hints) {
  if (rows <= 4096) {
    ShortAttention<<<dim3((rows + 63) / 64, heads), 128, 0, stream>>>(
        query, key, value, output, rows, heads);
    return;
  }
  PackValues<<<dim3((rows + 31) / 32, heads), 256, 0, stream>>>(
      value, packed_values, rows);
  DispatchLongAttention<false>(query, key, packed_values, output, rows, heads,
                               stream, hints);
}

void LaunchWmmaAttention(const std::uint16_t* query, const std::uint16_t* key,
                         const std::uint16_t* value, std::uint16_t* output,
                         std::uint16_t* packed_values, std::uint32_t rows,
                         std::uint32_t heads, hipStream_t stream,
                         std::uint32_t protected_prefix_rows) {
  LaunchWmmaAttention(query, key, value, output, packed_values, rows, heads,
                      stream,
                      SparseAttentionHints{.protected_prefix_rows =
                                               protected_prefix_rows});
}

void LaunchProjectionAttention(const std::uint16_t* query,
                               const std::uint16_t* key,
                               const std::uint16_t* packed_values,
                               std::uint16_t* output, std::uint32_t rows,
                               std::uint32_t heads, hipStream_t stream,
                               const SparseAttentionHints& hints) {
  DispatchLongAttention<true>(query, key, packed_values, output, rows, heads,
                              stream, hints);
}

void LaunchProjectionAttention(const std::uint16_t* query,
                               const std::uint16_t* key,
                               const std::uint16_t* packed_values,
                               std::uint16_t* output, std::uint32_t rows,
                               std::uint32_t heads, hipStream_t stream,
                               std::uint32_t protected_prefix_rows) {
  LaunchProjectionAttention(
      query, key, packed_values, output, rows, heads, stream,
      SparseAttentionHints{.protected_prefix_rows = protected_prefix_rows});
}

}  // namespace gufo::minimax_h3::dit_ops
