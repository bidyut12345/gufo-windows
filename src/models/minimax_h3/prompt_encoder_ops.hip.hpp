#ifndef GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_OPS_HIP_HPP_
#define GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_OPS_HIP_HPP_

// BF16 text-encoder operation boundaries translated from antirez/h3.c
// h3_shaders.metal at 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT).
// The implementation below is native HIP for gfx1151 and does not import
// Metal, MPSGraph, Objective-C, or ccv TensorOps matmul code.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::minimax_h3::ops {

__device__ __forceinline__ float Bf16ToFloat(std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t FloatToBf16(float value) {
  std::uint32_t bits = __float_as_uint(value);
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

static __global__ void EmbeddingKernel(const std::uint16_t* weight,
                                       const std::uint32_t* token_ids,
                                       std::uint16_t* output,
                                       std::uint32_t tokens,
                                       std::uint32_t vocabulary,
                                       std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t token = blockIdx.y;
  if (token >= tokens || column >= width) {
    return;
  }
  const std::uint32_t identifier = token_ids[token];
  output[static_cast<std::size_t>(token) * width + column] =
      identifier < vocabulary
          ? weight[static_cast<std::size_t>(identifier) * width + column]
          : 0;
}

inline void LaunchEmbedding(const std::uint16_t* weight,
                            const std::uint32_t* token_ids,
                            std::uint16_t* output, std::uint32_t tokens,
                            std::uint32_t vocabulary, std::uint32_t width,
                            hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  hipLaunchKernelGGL(EmbeddingKernel,
                     dim3((width + kThreads - 1U) / kThreads, tokens),
                     dim3(kThreads), 0, stream, weight, token_ids, output,
                     tokens, vocabulary, width);
}

static __global__ void RmsNormKernel(const std::uint16_t* input,
                                     const std::uint16_t* weight,
                                     std::uint16_t* output, std::uint32_t rows,
                                     std::uint32_t width, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= rows) {
    return;
  }
  __shared__ float reductions[256];
  const auto* row_input = input + static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    const float value = Bf16ToFloat(row_input[column]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(width) + epsilon);
  for (std::uint32_t column = lane; column < width; column += blockDim.x) {
    output[static_cast<std::size_t>(row) * width + column] = FloatToBf16(
        Bf16ToFloat(row_input[column]) * inverse * Bf16ToFloat(weight[column]));
  }
}

inline void LaunchRmsNorm(const std::uint16_t* input,
                          const std::uint16_t* weight, std::uint16_t* output,
                          std::uint32_t rows, std::uint32_t width,
                          float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(RmsNormKernel, dim3(rows), dim3(256), 0, stream, input,
                     weight, output, rows, width, epsilon);
}

static __global__ void HeadRmsNormKernel(
    std::uint16_t* tensor, const std::uint16_t* weight, std::uint32_t sequence,
    std::uint32_t heads, std::uint32_t head_dimension, float epsilon) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  if (row >= sequence || head >= heads) {
    return;
  }
  __shared__ float reductions[128];
  const std::uint32_t lane = threadIdx.x;
  const std::size_t base =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension;
  float local = 0.0F;
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    const float value = Bf16ToFloat(tensor[base + dimension]);
    local = fmaf(value, value, local);
  }
  reductions[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reductions[lane] += reductions[lane + stride];
    }
    __syncthreads();
  }
  const float inverse =
      rsqrtf(reductions[0] / static_cast<float>(head_dimension) + epsilon);
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    tensor[base + dimension] =
        FloatToBf16(Bf16ToFloat(tensor[base + dimension]) * inverse *
                    Bf16ToFloat(weight[dimension]));
  }
}

inline void LaunchHeadRmsNorm(std::uint16_t* tensor,
                              const std::uint16_t* weight,
                              std::uint32_t sequence, std::uint32_t heads,
                              std::uint32_t head_dimension, float epsilon,
                              hipStream_t stream) {
  hipLaunchKernelGGL(HeadRmsNormKernel, dim3(sequence, heads), dim3(128), 0,
                     stream, tensor, weight, sequence, heads, head_dimension,
                     epsilon);
}

static __global__ void PrepareRopeKernel(float* cosine, float* sine,
                                         std::uint32_t sequence,
                                         std::uint32_t head_dimension,
                                         float theta) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t dimension = threadIdx.x;
  const std::uint32_t half = head_dimension / 2U;
  if (row >= sequence || dimension >= half) {
    return;
  }
  const float inverse_frequency =
      expf(-logf(theta) * (2.0F * static_cast<float>(dimension) /
                           static_cast<float>(head_dimension)));
  const float angle = static_cast<float>(row) * inverse_frequency;
  const std::size_t index = static_cast<std::size_t>(row) * half + dimension;
  cosine[index] = cosf(angle);
  sine[index] = sinf(angle);
}

inline void LaunchPrepareRope(float* cosine, float* sine,
                              std::uint32_t sequence,
                              std::uint32_t head_dimension, float theta,
                              hipStream_t stream) {
  hipLaunchKernelGGL(PrepareRopeKernel, dim3(sequence),
                     dim3(head_dimension / 2U), 0, stream, cosine, sine,
                     sequence, head_dimension, theta);
}

static __global__ void RopeKernel(std::uint16_t* query, std::uint16_t* key,
                                  const float* cosine, const float* sine,
                                  std::uint32_t sequence,
                                  std::uint32_t query_heads,
                                  std::uint32_t key_value_heads,
                                  std::uint32_t head_dimension) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dimension = threadIdx.x;
  const std::uint32_t half = head_dimension / 2U;
  if (row >= sequence || dimension >= half) {
    return;
  }
  const std::size_t rope_index =
      static_cast<std::size_t>(row) * half + dimension;
  const float cosine_value = cosine[rope_index];
  const float sine_value = sine[rope_index];
  if (head < query_heads) {
    const std::size_t base =
        (static_cast<std::size_t>(row) * query_heads + head) * head_dimension;
    const float first = Bf16ToFloat(query[base + dimension]);
    const float second = Bf16ToFloat(query[base + half + dimension]);
    query[base + dimension] =
        FloatToBf16(first * cosine_value - second * sine_value);
    query[base + half + dimension] =
        FloatToBf16(second * cosine_value + first * sine_value);
  }
  if (head < key_value_heads) {
    const std::size_t base =
        (static_cast<std::size_t>(row) * key_value_heads + head) *
        head_dimension;
    const float first = Bf16ToFloat(key[base + dimension]);
    const float second = Bf16ToFloat(key[base + half + dimension]);
    key[base + dimension] =
        FloatToBf16(first * cosine_value - second * sine_value);
    key[base + half + dimension] =
        FloatToBf16(second * cosine_value + first * sine_value);
  }
}

inline void LaunchRope(std::uint16_t* query, std::uint16_t* key,
                       const float* cosine, const float* sine,
                       std::uint32_t sequence, std::uint32_t query_heads,
                       std::uint32_t key_value_heads,
                       std::uint32_t head_dimension, hipStream_t stream) {
  hipLaunchKernelGGL(
      RopeKernel,
      dim3(sequence,
           query_heads > key_value_heads ? query_heads : key_value_heads),
      dim3(head_dimension / 2U), 0, stream, query, key, cosine, sine, sequence,
      query_heads, key_value_heads, head_dimension);
}

inline constexpr std::uint32_t kCausalGqaTile = 64U;

static __global__ void CausalGqaKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t query_heads, std::uint32_t key_value_heads,
    std::uint32_t head_dimension, float scale) {
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t query_head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  if (query_row >= sequence || query_head >= query_heads) {
    return;
  }
  extern __shared__ float shared[];
  float* shared_query = shared;
  float* shared_acc = shared_query + head_dimension;
  float* scores = shared_acc + head_dimension;
  float* reductions = scores + kCausalGqaTile;
  auto* tile_kv = reinterpret_cast<std::uint16_t*>(reductions + 128U);
  const std::uint32_t key_value_head =
      query_head / (query_heads / key_value_heads);
  const std::size_t query_base =
      (static_cast<std::size_t>(query_row) * query_heads + query_head) *
      head_dimension;
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    shared_query[dimension] = Bf16ToFloat(
        FloatToBf16(Bf16ToFloat(query[query_base + dimension]) * scale));
    shared_acc[dimension] = 0.0F;
  }
  __syncthreads();

  const std::uint32_t key_count = query_row + 1U;
  float running_max = -INFINITY;
  float running_sum = 0.0F;
  for (std::uint32_t tile_start = 0; tile_start < key_count;
       tile_start += kCausalGqaTile) {
    const std::uint32_t tile_count =
        key_count - tile_start < kCausalGqaTile ? key_count - tile_start
                                                : kCausalGqaTile;
    const std::size_t tile_elements =
        static_cast<std::size_t>(tile_count) * head_dimension;
    for (std::size_t idx = lane; idx < tile_elements; idx += blockDim.x) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / head_dimension);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % head_dimension);
      const std::size_t key_index =
          (static_cast<std::size_t>(tile_start + k) * key_value_heads +
           key_value_head) *
              head_dimension +
          d;
      tile_kv[idx] = key[key_index];
    }
    __syncthreads();

    float local_max = -INFINITY;
    if (lane < tile_count) {
      const std::uint16_t* key_row =
          tile_kv + static_cast<std::size_t>(lane) * head_dimension;
      float dot = 0.0F;
      for (std::uint32_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        dot = fmaf(shared_query[dimension], Bf16ToFloat(key_row[dimension]),
                   dot);
      }
      scores[lane] = dot;
      local_max = dot;
    }
    reductions[lane] = local_max;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
      if (lane < stride) {
        reductions[lane] = fmaxf(reductions[lane], reductions[lane + stride]);
      }
      __syncthreads();
    }
    const float new_max = fmaxf(running_max, reductions[0]);
    const float rescale =
        running_sum == 0.0F ? 0.0F : expf(running_max - new_max);

    float local_sum = 0.0F;
    if (lane < tile_count) {
      const float probability = expf(scores[lane] - new_max);
      scores[lane] = probability;
      local_sum = probability;
    }
    reductions[lane] = local_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
      if (lane < stride) {
        reductions[lane] += reductions[lane + stride];
      }
      __syncthreads();
    }
    running_sum = running_sum * rescale + reductions[0];
    running_max = new_max;

    for (std::size_t idx = lane; idx < tile_elements; idx += blockDim.x) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / head_dimension);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % head_dimension);
      const std::size_t value_index =
          (static_cast<std::size_t>(tile_start + k) * key_value_heads +
           key_value_head) *
              head_dimension +
          d;
      tile_kv[idx] = value[value_index];
    }
    __syncthreads();

    for (std::uint32_t dimension = lane; dimension < head_dimension;
         dimension += blockDim.x) {
      float sum = shared_acc[dimension] * rescale;
      for (std::uint32_t k = 0; k < tile_count; ++k) {
        sum = fmaf(scores[k],
                   Bf16ToFloat(tile_kv[static_cast<std::size_t>(k) *
                                           head_dimension +
                                       dimension]),
                   sum);
      }
      shared_acc[dimension] = sum;
    }
    __syncthreads();
  }

  const float inverse_sum = 1.0F / running_sum;
  for (std::uint32_t dimension = lane; dimension < head_dimension;
       dimension += blockDim.x) {
    output[query_base + dimension] =
        FloatToBf16(shared_acc[dimension] * inverse_sum);
  }
}

static __global__ void GroupedCausalGqaKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t query_heads, std::uint32_t key_value_heads,
    std::uint32_t head_dimension, float scale) {
  constexpr std::uint32_t kMaximumGroupedHeads = 8;
  constexpr std::uint32_t kLogicalLanes = 128;
  const std::uint32_t query_row = blockIdx.x;
  const std::uint32_t key_value_head = blockIdx.y;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t grouped_heads = query_heads / key_value_heads;
  if (query_row >= sequence || key_value_head >= key_value_heads ||
      grouped_heads == 0 || grouped_heads > kMaximumGroupedHeads) {
    return;
  }
  extern __shared__ float shared[];
  float* scores = shared;
  float* shared_queries =
      scores + static_cast<std::size_t>(grouped_heads) * sequence;
  float* reductions =
      shared_queries + static_cast<std::size_t>(grouped_heads) * head_dimension;
  for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
    const std::uint32_t query_head = key_value_head * grouped_heads + grouped;
    const std::size_t query_base =
        (static_cast<std::size_t>(query_row) * query_heads + query_head) *
        head_dimension;
    for (std::uint32_t dimension = lane; dimension < head_dimension;
         dimension += blockDim.x) {
      shared_queries[static_cast<std::size_t>(grouped) * head_dimension +
                     dimension] =
          Bf16ToFloat(
              FloatToBf16(Bf16ToFloat(query[query_base + dimension]) * scale));
    }
  }
  __syncthreads();

  const std::uint32_t key_count = query_row + 1U;
  for (std::uint32_t logical_lane = lane; logical_lane < kLogicalLanes;
       logical_lane += blockDim.x) {
    for (std::uint32_t key_row = logical_lane; key_row < key_count;
         key_row += kLogicalLanes) {
      const std::size_t key_base =
          (static_cast<std::size_t>(key_row) * key_value_heads +
           key_value_head) *
          head_dimension;
      float dots[kMaximumGroupedHeads] = {};
      for (std::uint32_t dimension = 0; dimension < head_dimension;
           ++dimension) {
        const float key_value = Bf16ToFloat(key[key_base + dimension]);
        for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
          dots[grouped] =
              fmaf(shared_queries[static_cast<std::size_t>(grouped) *
                                      head_dimension +
                                  dimension],
                   key_value, dots[grouped]);
        }
      }
      for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
        scores[static_cast<std::size_t>(grouped) * sequence + key_row] =
            dots[grouped];
      }
    }
  }
  __syncthreads();

  for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
    float* grouped_scores =
        scores + static_cast<std::size_t>(grouped) * sequence;
    for (std::uint32_t logical_lane = lane; logical_lane < kLogicalLanes;
         logical_lane += blockDim.x) {
      float local_maximum = -INFINITY;
      for (std::uint32_t key_row = logical_lane; key_row < key_count;
           key_row += kLogicalLanes) {
        local_maximum = fmaxf(local_maximum, grouped_scores[key_row]);
      }
      reductions[logical_lane] = local_maximum;
    }
    __syncthreads();
    for (std::uint32_t stride = kLogicalLanes / 2U; stride != 0;
         stride >>= 1U) {
      if (lane < stride) {
        reductions[lane] = fmaxf(reductions[lane], reductions[lane + stride]);
      }
      __syncthreads();
    }
    const float maximum = reductions[0];
    for (std::uint32_t logical_lane = lane; logical_lane < kLogicalLanes;
         logical_lane += blockDim.x) {
      float local_sum = 0.0F;
      for (std::uint32_t key_row = logical_lane; key_row < key_count;
           key_row += kLogicalLanes) {
        const float probability = expf(grouped_scores[key_row] - maximum);
        grouped_scores[key_row] = probability;
        local_sum += probability;
      }
      reductions[logical_lane] = local_sum;
    }
    __syncthreads();
    for (std::uint32_t stride = kLogicalLanes / 2U; stride != 0;
         stride >>= 1U) {
      if (lane < stride) {
        reductions[lane] += reductions[lane + stride];
      }
      __syncthreads();
    }
    const float inverse_sum = 1.0F / reductions[0];
    for (std::uint32_t logical_lane = lane; logical_lane < kLogicalLanes;
         logical_lane += blockDim.x) {
      for (std::uint32_t key_row = logical_lane; key_row < key_count;
           key_row += kLogicalLanes) {
        grouped_scores[key_row] *= inverse_sum;
      }
    }
    __syncthreads();
  }

  for (std::uint32_t logical_lane = lane; logical_lane < kLogicalLanes;
       logical_lane += blockDim.x) {
    for (std::uint32_t dimension = logical_lane; dimension < head_dimension;
         dimension += kLogicalLanes) {
      float sums[kMaximumGroupedHeads] = {};
      for (std::uint32_t key_row = 0; key_row < key_count; ++key_row) {
        const std::size_t value_index =
            (static_cast<std::size_t>(key_row) * key_value_heads +
             key_value_head) *
                head_dimension +
            dimension;
        const float value_element = Bf16ToFloat(value[value_index]);
        for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
          sums[grouped] = fmaf(
              scores[static_cast<std::size_t>(grouped) * sequence + key_row],
              value_element, sums[grouped]);
        }
      }
      for (std::uint32_t grouped = 0; grouped < grouped_heads; ++grouped) {
        const std::uint32_t query_head =
            key_value_head * grouped_heads + grouped;
        const std::size_t output_index =
            (static_cast<std::size_t>(query_row) * query_heads + query_head) *
                head_dimension +
            dimension;
        output[output_index] = FloatToBf16(sums[grouped]);
      }
    }
  }
}

inline constexpr std::uint32_t kTiledGqaKeyTile = 32U;
inline constexpr std::uint32_t kCausalGqaRowChunk = 2048U;

// Tiled FlashAttention-style causal GQA kernel: shares K/V global loads across
// all grouped query heads (up to 8) in a 128-thread block, using fixed LDS
// independent of sequence length and supporting row-chunked launches for TDR.
static __global__ void TiledGroupedCausalGqaKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t row_offset, std::uint32_t query_heads,
    std::uint32_t key_value_heads, std::uint32_t head_dimension, float scale) {
  constexpr std::uint32_t kMaxGrouped = 8U;
  constexpr std::uint32_t kThreads = 128U;
  const std::uint32_t query_row = row_offset + blockIdx.x;
  const std::uint32_t kv_head = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t grouped_heads = query_heads / key_value_heads;
  if (query_row >= sequence || kv_head >= key_value_heads ||
      grouped_heads == 0 || grouped_heads > kMaxGrouped ||
      head_dimension > kThreads) {
    return;
  }

  extern __shared__ float shared[];
  float* shared_queries = shared;  // [grouped_heads][head_dimension]
  float* shared_acc =
      shared_queries + static_cast<std::size_t>(grouped_heads) * head_dimension;
  float* scores =
      shared_acc + static_cast<std::size_t>(grouped_heads) * head_dimension;
  float* running_max =
      scores + static_cast<std::size_t>(grouped_heads) * kTiledGqaKeyTile;
  float* running_sum = running_max + kMaxGrouped;
  float* tile_rescale = running_sum + kMaxGrouped;
  auto* tile_kv = reinterpret_cast<std::uint16_t*>(tile_rescale + kMaxGrouped);

  for (std::uint32_t g = 0; g < grouped_heads; ++g) {
    const std::uint32_t q_head = kv_head * grouped_heads + g;
    const std::size_t q_base =
        (static_cast<std::size_t>(query_row) * query_heads + q_head) *
        head_dimension;
    if (tid < head_dimension) {
      shared_queries[static_cast<std::size_t>(g) * head_dimension + tid] =
          Bf16ToFloat(FloatToBf16(Bf16ToFloat(query[q_base + tid]) * scale));
      shared_acc[static_cast<std::size_t>(g) * head_dimension + tid] = 0.0F;
    }
  }
  if (tid < grouped_heads) {
    running_max[tid] = -INFINITY;
    running_sum[tid] = 0.0F;
    tile_rescale[tid] = 0.0F;
  }
  __syncthreads();

  const std::uint32_t key_count = query_row + 1U;
  const std::uint32_t k_idx = tid / 4U;      // 0..31
  const std::uint32_t d_quarter = tid % 4U;  // 0..3
  const std::uint32_t d_chunk = (head_dimension + 3U) / 4U;
  const std::uint32_t d_begin = d_quarter * d_chunk;
  const std::uint32_t d_end =
      d_begin + d_chunk < head_dimension ? d_begin + d_chunk : head_dimension;

  for (std::uint32_t tile_start = 0; tile_start < key_count;
       tile_start += kTiledGqaKeyTile) {
    const std::uint32_t tile_count = key_count - tile_start < kTiledGqaKeyTile
                                         ? key_count - tile_start
                                         : kTiledGqaKeyTile;
    const std::size_t tile_elements =
        static_cast<std::size_t>(tile_count) * head_dimension;
    for (std::size_t idx = tid; idx < tile_elements; idx += kThreads) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / head_dimension);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % head_dimension);
      const std::size_t key_index =
          (static_cast<std::size_t>(tile_start + k) * key_value_heads +
           kv_head) *
              head_dimension +
          d;
      tile_kv[idx] = key[key_index];
    }
    __syncthreads();

    float dots[kMaxGrouped] = {};
    if (k_idx < tile_count) {
      const std::uint16_t* key_row =
          tile_kv + static_cast<std::size_t>(k_idx) * head_dimension;
      for (std::uint32_t d = d_begin; d < d_end; ++d) {
        const float k_val = Bf16ToFloat(key_row[d]);
        for (std::uint32_t g = 0; g < grouped_heads; ++g) {
          dots[g] = fmaf(
              shared_queries[static_cast<std::size_t>(g) * head_dimension + d],
              k_val, dots[g]);
        }
      }
    }
    for (std::uint32_t g = 0; g < grouped_heads; ++g) {
      float sum = dots[g];
      sum += __shfl_xor(sum, 1);
      sum += __shfl_xor(sum, 2);
      if (d_quarter == 0 && k_idx < tile_count) {
        scores[static_cast<std::size_t>(g) * kTiledGqaKeyTile + k_idx] = sum;
      }
    }
    __syncthreads();

    // First `grouped_heads` warps (or warp 0 if grouped_heads <= 8, one head per
    // warp up to 4 warps in two passes) update online softmax in registers.
    const std::uint32_t warp = tid / 32U;
    const std::uint32_t lane = tid % 32U;
    for (std::uint32_t g = warp; g < grouped_heads; g += 4U) {
      float* g_scores = scores + static_cast<std::size_t>(g) * kTiledGqaKeyTile;
      float val = lane < tile_count ? g_scores[lane] : -INFINITY;
      float local_max = val;
      for (std::uint32_t s = 16U; s != 0; s >>= 1U) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, s));
      }
      const float old_max = running_max[g];
      const float new_max = fmaxf(old_max, local_max);
      const float old_sum = running_sum[g];
      const float rescale = old_sum == 0.0F ? 0.0F : expf(old_max - new_max);
      const float prob = lane < tile_count ? expf(val - new_max) : 0.0F;
      if (lane < tile_count) {
        g_scores[lane] = prob;
      }
      float prob_sum = prob;
      for (std::uint32_t s = 16U; s != 0; s >>= 1U) {
        prob_sum += __shfl_xor(prob_sum, s);
      }
      if (lane == 0) {
        running_max[g] = new_max;
        running_sum[g] = old_sum * rescale + prob_sum;
        tile_rescale[g] = rescale;
      }
    }

    for (std::size_t idx = tid; idx < tile_elements; idx += kThreads) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / head_dimension);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % head_dimension);
      const std::size_t value_index =
          (static_cast<std::size_t>(tile_start + k) * key_value_heads +
           kv_head) *
              head_dimension +
          d;
      tile_kv[idx] = value[value_index];
    }
    __syncthreads();

    if (tid < head_dimension) {
      for (std::uint32_t g = 0; g < grouped_heads; ++g) {
        const float* g_scores =
            scores + static_cast<std::size_t>(g) * kTiledGqaKeyTile;
        float acc =
            shared_acc[static_cast<std::size_t>(g) * head_dimension + tid] *
            tile_rescale[g];
        for (std::uint32_t k = 0; k < tile_count; ++k) {
          acc = fmaf(
              g_scores[k],
              Bf16ToFloat(tile_kv[static_cast<std::size_t>(k) * head_dimension +
                                  tid]),
              acc);
        }
        shared_acc[static_cast<std::size_t>(g) * head_dimension + tid] = acc;
      }
    }
    __syncthreads();
  }

  if (tid < head_dimension) {
    for (std::uint32_t g = 0; g < grouped_heads; ++g) {
      const std::uint32_t q_head = kv_head * grouped_heads + g;
      const std::size_t out_index =
          (static_cast<std::size_t>(query_row) * query_heads + q_head) *
              head_dimension +
          tid;
      output[out_index] = FloatToBf16(
          shared_acc[static_cast<std::size_t>(g) * head_dimension + tid] /
          running_sum[g]);
    }
  }
}

inline void LaunchCausalGqa(const std::uint16_t* query,
                            const std::uint16_t* key,
                            const std::uint16_t* value, std::uint16_t* output,
                            std::uint32_t sequence, std::uint32_t query_heads,
                            std::uint32_t key_value_heads,
                            std::uint32_t head_dimension, float scale,
                            hipStream_t stream) {
  const std::uint32_t grouped_heads =
      key_value_heads == 0 ? 0 : query_heads / key_value_heads;
  const std::size_t grouped_shared_bytes =
      (static_cast<std::size_t>(sequence) * grouped_heads +
       static_cast<std::size_t>(head_dimension) * grouped_heads + 128U) *
      sizeof(float);
  constexpr std::size_t kMaximumGroupedSharedBytes = 48U << 10U;
  if (key_value_heads != 0 && query_heads % key_value_heads == 0 &&
      grouped_heads <= 8 &&
      grouped_shared_bytes <= kMaximumGroupedSharedBytes) {
    hipLaunchKernelGGL(GroupedCausalGqaKernel, dim3(sequence, key_value_heads),
                       dim3(64), grouped_shared_bytes, stream, query, key,
                       value, output, sequence, query_heads, key_value_heads,
                       head_dimension, scale);
    return;
  }
  if (key_value_heads != 0 && query_heads % key_value_heads == 0 &&
      grouped_heads <= 8 && head_dimension <= 128U) {
    const std::size_t tiled_shared_bytes =
        (static_cast<std::size_t>(grouped_heads) * head_dimension * 2U +
         static_cast<std::size_t>(grouped_heads) * kTiledGqaKeyTile + 24U) *
            sizeof(float) +
        static_cast<std::size_t>(kTiledGqaKeyTile) * head_dimension *
            sizeof(std::uint16_t);
    for (std::uint32_t row_offset = 0; row_offset < sequence;
         row_offset += kCausalGqaRowChunk) {
      const std::uint32_t chunk_rows = sequence - row_offset < kCausalGqaRowChunk
                                           ? sequence - row_offset
                                           : kCausalGqaRowChunk;
      hipLaunchKernelGGL(TiledGroupedCausalGqaKernel,
                         dim3(chunk_rows, key_value_heads), dim3(128),
                         tiled_shared_bytes, stream, query, key, value, output,
                         sequence, row_offset, query_heads, key_value_heads,
                         head_dimension, scale);
    }
    return;
  }
  const std::size_t shared_bytes =
      (static_cast<std::size_t>(head_dimension) * 2U + kCausalGqaTile + 128U) *
          sizeof(float) +
      static_cast<std::size_t>(kCausalGqaTile) * head_dimension *
          sizeof(std::uint16_t);
  hipLaunchKernelGGL(CausalGqaKernel, dim3(sequence, query_heads), dim3(128),
                     shared_bytes, stream, query, key, value, output, sequence,
                     query_heads, key_value_heads, head_dimension, scale);
}

static __global__ void AddKernel(const std::uint16_t* left,
                                 const std::uint16_t* right,
                                 std::uint16_t* output, std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] =
        FloatToBf16(Bf16ToFloat(left[index]) + Bf16ToFloat(right[index]));
  }
}

inline void LaunchAdd(const std::uint16_t* left, const std::uint16_t* right,
                      std::uint16_t* output, std::size_t elements,
                      hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AddKernel, dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, left, right, output, elements);
}

static __global__ void SwiGluKernel(const std::uint16_t* gate,
                                    const std::uint16_t* up,
                                    std::uint16_t* output,
                                    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float gate_value = Bf16ToFloat(gate[index]);
    output[index] = FloatToBf16(gate_value / (1.0F + expf(-gate_value)) *
                                Bf16ToFloat(up[index]));
  }
}

inline void LaunchSwiGlu(const std::uint16_t* gate, const std::uint16_t* up,
                         std::uint16_t* output, std::size_t elements,
                         hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(SwiGluKernel, dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, gate, up, output, elements);
}

// Interleaved mRoPE for FL2VA keyframe presentation (issue #177). Identical
// to PrepareRopeKernel whenever all three axes agree (every text row), and
// selects axis positions [d < 60 ? d % 3 : 0] per frequency index otherwise
// (transformers' mrope_interleaved with mrope_section [24,20,20] over the
// 64 rope pairs; theta 5e6). positions is int32 [sequence][3] (t, h, w).
static __global__ void PrepareRopeMultiKernel(float* cosine, float* sine,
                                              const std::int32_t* positions,
                                              std::uint32_t sequence,
                                              std::uint32_t head_dimension,
                                              float theta) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t dimension = threadIdx.x;
  const std::uint32_t half = head_dimension / 2U;
  if (row >= sequence || dimension >= half) {
    return;
  }
  // transformers Qwen3VL interleaved mrope: sections [24,20,20] consume
  // the first 60 pairs round-robin (t,h,w); the remaining four pairs stay
  // on the temporal axis. Exact rule of the sibling reference port.
  const std::uint32_t axis = dimension < 60U ? dimension % 3U : 0U;
  const float inverse_frequency =
      expf(-logf(theta) * (2.0F * static_cast<float>(dimension) /
                           static_cast<float>(head_dimension)));
  const std::int32_t selected =
      positions[static_cast<std::size_t>(row) * 3U + axis];
  const float angle = static_cast<float>(selected) * inverse_frequency;
  const std::size_t index = static_cast<std::size_t>(row) * half + dimension;
  cosine[index] = cosf(angle);
  sine[index] = sinf(angle);
}

inline void LaunchPrepareRopeMulti(float* cosine, float* sine,
                                   const std::int32_t* positions,
                                   std::uint32_t sequence,
                                   std::uint32_t head_dimension, float theta,
                                   hipStream_t stream) {
  hipLaunchKernelGGL(PrepareRopeMultiKernel, dim3(sequence),
                     dim3(head_dimension / 2U), 0, stream, cosine, sine,
                     positions, sequence, head_dimension, theta);
}

// Replace selected rows of a bf16 [rows, width] matrix with source rows in
// list order (image_pad placeholders receive vision embeddings).
static __global__ void ReplaceRowsKernel(const std::uint32_t* rows,
                                         const std::uint16_t* source,
                                         std::uint16_t* matrix,
                                         std::uint32_t width,
                                         std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t row = index / width;
  const std::size_t column = index % width;
  matrix[static_cast<std::size_t>(rows[row]) * width + column] = source[index];
}

inline void LaunchReplaceRows(const std::uint32_t* rows,
                              const std::uint16_t* source,
                              std::uint16_t* matrix, std::uint32_t row_count,
                              std::uint32_t width, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  const std::size_t elements = static_cast<std::size_t>(row_count) * width;
  hipLaunchKernelGGL(
      ReplaceRowsKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, rows, source, matrix, width, elements);
}

// DeepStack injection: add one source row (per index entry) into the hidden
// matrix row, single bf16 rounding after an fp32 sum.
static __global__ void AddRowsKernel(const std::uint32_t* rows,
                                     const std::uint16_t* source,
                                     std::uint16_t* matrix, std::uint32_t width,
                                     std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::size_t row = index / width;
  const std::size_t column = index % width;
  std::uint16_t* target =
      matrix + static_cast<std::size_t>(rows[row]) * width + column;
  *target = FloatToBf16(Bf16ToFloat(*target) + Bf16ToFloat(source[index]));
}

inline void LaunchAddRows(const std::uint32_t* rows,
                          const std::uint16_t* source, std::uint16_t* matrix,
                          std::uint32_t row_count, std::uint32_t width,
                          hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  const std::size_t elements = static_cast<std::size_t>(row_count) * width;
  hipLaunchKernelGGL(
      AddRowsKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, rows, source, matrix, width, elements);
}

static __global__ void DequantInt8ToBf16Kernel(const std::int8_t* source,
                                               const float* scale,
                                               std::uint16_t* destination,
                                               std::size_t elements,
                                               std::size_t columns,
                                               std::size_t scale_count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    float s = 1.0F;
    if (scale != nullptr && scale_count != 0U) {
      s = scale_count == 1U ? scale[0] : scale[index / columns];
    }
    destination[index] = FloatToBf16(static_cast<float>(source[index]) * s);
  }
}

inline void LaunchDequantInt8ToBf16(const std::int8_t* source,
                                    const float* scale,
                                    std::uint16_t* destination,
                                    std::size_t rows, std::size_t columns,
                                    std::size_t scale_count,
                                    hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  const std::size_t elements = rows * columns;
  hipLaunchKernelGGL(
      DequantInt8ToBf16Kernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, source, scale, destination, elements, columns,
      scale_count);
}

}  // namespace gufo::minimax_h3::ops

#endif  // GUFO_MODELS_MINIMAX_H3_PROMPT_ENCODER_OPS_HIP_HPP_
