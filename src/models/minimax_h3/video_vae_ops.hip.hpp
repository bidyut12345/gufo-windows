#ifndef GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_HIP_HPP_
#define GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::minimax_h3::video_vae_ops {

static __global__ void AddBiasKernel(float* values, const float* bias,
                                     std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    values[static_cast<std::size_t>(row) * width + column] += bias[column];
  }
}

static __global__ void AddBiasFloat4Kernel(float4* values, const float4* bias,
                                           std::uint32_t rows,
                                           std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < vectors) {
    const std::size_t index = static_cast<std::size_t>(row) * vectors + column;
    const float4 value = values[index];
    const float4 offset = bias[column];
    values[index] = {value.x + offset.x, value.y + offset.y, value.z + offset.z,
                     value.w + offset.w};
  }
}

inline void LaunchAddBias(float* values, const float* bias, std::uint32_t rows,
                          std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (width % 4U == 0U) {
    const std::uint32_t vectors = width / 4U;
    hipLaunchKernelGGL(
        AddBiasFloat4Kernel, dim3((vectors + kThreads - 1U) / kThreads, rows),
        dim3(kThreads), 0, stream, reinterpret_cast<float4*>(values),
        reinterpret_cast<const float4*>(bias), rows, vectors);
    return;
  }
  hipLaunchKernelGGL(AddBiasKernel,
                     dim3((width + kThreads - 1U) / kThreads, rows),
                     dim3(kThreads), 0, stream, values, bias, rows, width);
}

static __global__ void PrepareTileKernel(
    const float* normalized, const float* mean, const float* deviation,
    float* rows, std::uint32_t latent_time, std::uint32_t latent_height,
    std::uint32_t latent_width, std::uint32_t start_time, std::uint32_t start_y,
    std::uint32_t start_x, std::uint32_t tile_time, std::uint32_t tile_height,
    std::uint32_t tile_width, std::uint32_t channels) {
  const std::size_t elements =
      static_cast<std::size_t>(tile_time) * tile_height * tile_width * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t channel = index % channels;
  std::size_t remaining = index / channels;
  const std::uint32_t x = remaining % tile_width;
  remaining /= tile_width;
  const std::uint32_t y = remaining % tile_height;
  const std::uint32_t time = remaining / tile_height;
  const std::size_t source =
      (((static_cast<std::size_t>(channel) * latent_time +
         (start_time + time)) *
            latent_height +
        (start_y + y)) *
           latent_width +
       (start_x + x));
  rows[index] = fmaf(normalized[source], deviation[channel], mean[channel]);
}

inline void LaunchPrepareTile(
    const float* normalized, const float* mean, const float* deviation,
    float* rows, std::uint32_t latent_time, std::uint32_t latent_height,
    std::uint32_t latent_width, std::uint32_t start_time, std::uint32_t start_y,
    std::uint32_t start_x, std::uint32_t tile_time, std::uint32_t tile_height,
    std::uint32_t tile_width, std::uint32_t channels, hipStream_t stream) {
  const std::size_t elements =
      static_cast<std::size_t>(tile_time) * tile_height * tile_width * channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      PrepareTileKernel, dim3((elements + kThreads - 1U) / kThreads),
      dim3(kThreads), 0, stream, normalized, mean, deviation, rows, latent_time,
      latent_height, latent_width, start_time, start_y, start_x, tile_time,
      tile_height, tile_width, channels);
}

static __global__ void RopeKernel(float* cosine, float* sine,
                                  std::uint32_t patches, std::uint32_t sequence,
                                  std::uint32_t tile_time,
                                  std::uint32_t tile_height,
                                  std::uint32_t tile_width) {
  const std::uint32_t frequency = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= sequence || frequency >= 24) {
    return;
  }
  const std::size_t output = static_cast<std::size_t>(row) * 24 + frequency;
  if (row >= patches) {
    cosine[output] = 1.0F;
    sine[output] = 0.0F;
    return;
  }
  std::uint32_t remaining = row;
  const std::uint32_t x = remaining % tile_width;
  remaining /= tile_width;
  const std::uint32_t y = remaining % tile_height;
  const std::uint32_t time = remaining / tile_height;
  const std::uint32_t axis = frequency / 8U;
  const std::uint32_t frequency_in_axis = frequency % 8U;
  float coordinate = 0.0F;
  if (axis == 0) {
    coordinate = 2.0F * ((static_cast<float>(time) + 0.5F) /
                         static_cast<float>(tile_time)) -
                 1.0F;
  } else if (axis == 1) {
    coordinate = 2.0F * ((static_cast<float>(y) + 0.5F) /
                         static_cast<float>(tile_height)) -
                 1.0F;
  } else {
    coordinate = 2.0F * ((static_cast<float>(x) + 0.5F) /
                         static_cast<float>(tile_width)) -
                 1.0F;
  }
  constexpr float kTwoPi = 6.28318530717958647692F;
  const float inverse =
      1.0F / powf(100.0F, static_cast<float>(frequency_in_axis) * 0.125F);
  const float angle = kTwoPi * coordinate * inverse;
  cosine[output] = cosf(angle);
  sine[output] = sinf(angle);
}

inline void LaunchRope(float* cosine, float* sine, std::uint32_t patches,
                       std::uint32_t sequence, std::uint32_t tile_time,
                       std::uint32_t tile_height, std::uint32_t tile_width,
                       hipStream_t stream) {
  hipLaunchKernelGGL(RopeKernel, dim3(1, sequence), dim3(32), 0, stream, cosine,
                     sine, patches, sequence, tile_time, tile_height,
                     tile_width);
}

static __global__ void RmsNormKernel(const float* input, const float* weight,
                                     float* output, std::uint32_t rows,
                                     std::uint32_t width, float epsilon) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock + wave;
  if (row >= rows) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += 32U) {
    local = fmaf(input[base + column], input[base + column], local);
  }
#pragma unroll
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
    local += __shfl_down(local, stride, 32);
  }
  const float inverse =
      rsqrtf(__shfl(local, 0, 32) / static_cast<float>(width) + epsilon);
  for (std::uint32_t column = lane; column < width; column += 32U) {
    output[base + column] = input[base + column] * inverse * weight[column];
  }
}

inline void LaunchRmsNorm(const float* input, const float* weight,
                          float* output, std::uint32_t rows,
                          std::uint32_t width, float epsilon,
                          hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  hipLaunchKernelGGL(RmsNormKernel,
                     dim3((rows + kRowsPerBlock - 1U) / kRowsPerBlock),
                     dim3(kRowsPerBlock * 32U), 0, stream, input, weight,
                     output, rows, width, epsilon);
}

static __global__ void LayerNormKernel(const float* input, const float* weight,
                                       const float* bias, float* output,
                                       std::uint32_t rows, std::uint32_t width,
                                       float epsilon) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock + wave;
  const bool active = row < rows;
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local_sum = 0.0F;
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      local_sum += input[base + column];
    }
  }
  for (std::uint32_t stride = 16U; stride != 0; stride >>= 1U) {
    const float peer = __shfl_down(local_sum, stride, 32);
    if (lane < stride) {
      local_sum += peer;
    }
  }
  const float mean = __shfl(local_sum, 0, 32) / static_cast<float>(width);
  float local_square = 0.0F;
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      const float centered = input[base + column] - mean;
      local_square = fmaf(centered, centered, local_square);
    }
  }
  for (std::uint32_t stride = 16U; stride != 0; stride >>= 1U) {
    const float peer = __shfl_down(local_square, stride, 32);
    if (lane < stride) {
      local_square += peer;
    }
  }
  const float variance =
      __shfl(local_square, 0, 32) / static_cast<float>(width);
  const float inverse = rsqrtf(variance + epsilon);
  if (active) {
    for (std::uint32_t column = lane; column < width; column += 32U) {
      output[base + column] =
          (input[base + column] - mean) * inverse * weight[column] +
          bias[column];
    }
  }
}

inline void LaunchLayerNorm(const float* input, const float* weight,
                            const float* bias, float* output,
                            std::uint32_t rows, std::uint32_t width,
                            float epsilon, hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  hipLaunchKernelGGL(LayerNormKernel,
                     dim3((rows + kRowsPerBlock - 1U) / kRowsPerBlock),
                     dim3(kRowsPerBlock * 32U), 0, stream, input, weight, bias,
                     output, rows, width, epsilon);
}

static __global__ void VideoQkvRopeKernel(
    const float* qkv, const float* bias, const float* cosine, const float* sine,
    float* query, float* key, float* value, std::uint32_t rows,
    std::uint32_t heads, std::uint32_t head_dimension, std::uint32_t rope_half,
    float epsilon) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || head >= heads || lane >= head_dimension) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension * 3U;
  const std::size_t bias_base =
      static_cast<std::size_t>(head) * head_dimension * 3U;
  const float query_value = qkv[base + lane] + bias[bias_base + lane];
  const float key_value = qkv[base + head_dimension + lane] +
                          bias[bias_base + head_dimension + lane];
  __shared__ float query_squares[64];
  __shared__ float key_squares[64];
  query_squares[lane] = query_value * query_value;
  key_squares[lane] = key_value * key_value;
  __syncthreads();
  for (std::uint32_t stride = head_dimension / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      query_squares[lane] += query_squares[lane + stride];
      key_squares[lane] += key_squares[lane + stride];
    }
    __syncthreads();
  }
  const float query_inverse =
      rsqrtf(query_squares[0] / static_cast<float>(head_dimension) + epsilon);
  const float key_inverse =
      rsqrtf(key_squares[0] / static_cast<float>(head_dimension) + epsilon);
  float transformed_query = query_value * query_inverse;
  float transformed_key = key_value * key_inverse;
  if (lane < rope_half) {
    const std::uint32_t pair = lane + rope_half;
    const float pair_query =
        (qkv[base + pair] + bias[bias_base + pair]) * query_inverse;
    const float pair_key = (qkv[base + head_dimension + pair] +
                            bias[bias_base + head_dimension + pair]) *
                           key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + lane];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + lane];
    transformed_query = transformed_query * c - pair_query * s;
    transformed_key = transformed_key * c - pair_key * s;
  } else if (lane < rope_half * 2U) {
    const std::uint32_t pair = lane - rope_half;
    const float pair_query =
        (qkv[base + pair] + bias[bias_base + pair]) * query_inverse;
    const float pair_key = (qkv[base + head_dimension + pair] +
                            bias[bias_base + head_dimension + pair]) *
                           key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + pair];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + pair];
    transformed_query = transformed_query * c + pair_query * s;
    transformed_key = transformed_key * c + pair_key * s;
  }
  const std::size_t output =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension + lane;
  query[output] = transformed_query;
  key[output] = transformed_key;
  value[output] = qkv[base + head_dimension * 2U + lane] +
                  bias[bias_base + head_dimension * 2U + lane];
}

inline void LaunchVideoQkvRope(const float* qkv, const float* bias,
                               const float* cosine, const float* sine,
                               float* query, float* key, float* value,
                               std::uint32_t rows, std::uint32_t heads,
                               std::uint32_t head_dimension,
                               std::uint32_t rope_half, float epsilon,
                               hipStream_t stream) {
  hipLaunchKernelGGL(VideoQkvRopeKernel, dim3(heads, rows),
                     dim3(head_dimension), 0, stream, qkv, bias, cosine, sine,
                     query, key, value, rows, heads, head_dimension, rope_half,
                     epsilon);
}

__device__ __forceinline__ float SoftmaxWaveMaximum(float value) {
#pragma unroll
  for (std::uint32_t offset = 16; offset != 0; offset >>= 1U) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float SoftmaxWaveSum(float value) {
#pragma unroll
  for (std::uint32_t offset = 16; offset != 0; offset >>= 1U) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

// rocBLAS materializes one score matrix per head as
// [head][query_row][key_row]. Eight independent gfx1151 wavefronts normalize
// eight rows per workgroup and retain the probabilities in FP32 for the PV
// GEMM. Each row keeps the same wave-local reduction order.
static __global__ void SoftmaxScoresInPlaceKernel(float* scores,
                                                  std::uint32_t rows,
                                                  std::uint32_t heads) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t query_row = blockIdx.x * kRowsPerBlock + wave;
  const std::uint32_t head = blockIdx.y;
  if (query_row >= rows || head >= heads) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(head) * rows + query_row) * rows;
  float local_maximum = -INFINITY;
  for (std::uint32_t key_row = lane; key_row < rows; key_row += 32U) {
    local_maximum = fmaxf(local_maximum, scores[base + key_row]);
  }
  const float maximum = SoftmaxWaveMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint32_t key_row = lane; key_row < rows; key_row += 32U) {
    const float probability = expf(scores[base + key_row] - maximum);
    scores[base + key_row] = probability;
    local_sum += probability;
  }
  const float inverse_sum = 1.0F / SoftmaxWaveSum(local_sum);
  for (std::uint32_t key_row = lane; key_row < rows; key_row += 32U) {
    scores[base + key_row] *= inverse_sum;
  }
}

inline void LaunchSoftmaxScoresInPlace(float* scores, std::uint32_t rows,
                                       std::uint32_t heads,
                                       hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  hipLaunchKernelGGL(SoftmaxScoresInPlaceKernel,
                     dim3((rows + kRowsPerBlock - 1U) / kRowsPerBlock, heads),
                     dim3(256), 0, stream, scores, rows, heads);
}

static __global__ void SwiGluKernel(const float* fused, const float* bias,
                                    float* output, std::uint32_t rows,
                                    std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t base = static_cast<std::size_t>(row) * width * 2U;
    const float gate = fused[base + column] + bias[column];
    const float up = fused[base + width + column] + bias[width + column];
    output[static_cast<std::size_t>(row) * width + column] =
        (gate / (1.0F + expf(-gate))) * up;
  }
}

static __global__ void SwiGluFloat4Kernel(const float4* fused,
                                          const float4* bias, float4* output,
                                          std::uint32_t rows,
                                          std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= vectors) {
    return;
  }
  const std::size_t input_base = static_cast<std::size_t>(row) * vectors * 2U;
  const float4 gate_value = fused[input_base + column];
  const float4 up_value = fused[input_base + vectors + column];
  const float4 gate_bias = bias[column];
  const float4 up_bias = bias[vectors + column];
  const float gate_x = gate_value.x + gate_bias.x;
  const float gate_y = gate_value.y + gate_bias.y;
  const float gate_z = gate_value.z + gate_bias.z;
  const float gate_w = gate_value.w + gate_bias.w;
  const float up_x = up_value.x + up_bias.x;
  const float up_y = up_value.y + up_bias.y;
  const float up_z = up_value.z + up_bias.z;
  const float up_w = up_value.w + up_bias.w;
  output[static_cast<std::size_t>(row) * vectors + column] = {
      (gate_x / (1.0F + expf(-gate_x))) * up_x,
      (gate_y / (1.0F + expf(-gate_y))) * up_y,
      (gate_z / (1.0F + expf(-gate_z))) * up_z,
      (gate_w / (1.0F + expf(-gate_w))) * up_w};
}

inline void LaunchSwiGlu(const float* fused, const float* bias, float* output,
                         std::uint32_t rows, std::uint32_t width,
                         hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (width % 4U == 0U) {
    const std::uint32_t vectors = width / 4U;
    hipLaunchKernelGGL(
        SwiGluFloat4Kernel, dim3((vectors + kThreads - 1U) / kThreads, rows),
        dim3(kThreads), 0, stream, reinterpret_cast<const float4*>(fused),
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<float4*>(output), rows, vectors);
    return;
  }
  hipLaunchKernelGGL(
      SwiGluKernel, dim3((width + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, fused, bias, output, rows, width);
}

__device__ __forceinline__ float Bf16ToFloatScalar(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

__device__ __forceinline__ float4 UnpackBf16x4(uint2 packed) {
  return {
      __uint_as_float((packed.x & 0xFFFFU) << 16U),
      __uint_as_float((packed.x >> 16U) << 16U),
      __uint_as_float((packed.y & 0xFFFFU) << 16U),
      __uint_as_float((packed.y >> 16U) << 16U),
  };
}

static __global__ void ScaleAddKernel(float* residual, const float* branch,
                                      const float* bias, const float* scale,
                                      std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    const float biased = branch[index] + bias[column];
    residual[index] = fmaf(biased, scale[column], residual[index]);
  }
}

static __global__ void ScaleAddFromBf16Kernel(
    float* residual, const std::uint16_t* branch, const float* bias,
    const float* scale, std::uint32_t rows, std::uint32_t width) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row < rows && column < width) {
    const std::size_t index = static_cast<std::size_t>(row) * width + column;
    const float biased = Bf16ToFloatScalar(branch[index]) + bias[column];
    residual[index] = fmaf(biased, scale[column], residual[index]);
  }
}

static __global__ void ScaleAddFloat4Kernel(
    float4* residual, const float4* branch, const float4* bias,
    const float4* scale, std::uint32_t rows, std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= vectors) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * vectors + column;
  const float4 current = residual[index];
  const float4 value = branch[index];
  const float4 offset = bias[column];
  const float4 multiplier = scale[column];
  residual[index] = {
      fmaf(value.x + offset.x, multiplier.x, current.x),
      fmaf(value.y + offset.y, multiplier.y, current.y),
      fmaf(value.z + offset.z, multiplier.z, current.z),
      fmaf(value.w + offset.w, multiplier.w, current.w),
  };
}

static __global__ void ScaleAddFloat4FromBf16Kernel(
    float4* residual, const uint2* branch_bf16, const float4* bias,
    const float4* scale, std::uint32_t rows, std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= vectors) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * vectors + column;
  const float4 current = residual[index];
  const float4 value = UnpackBf16x4(branch_bf16[index]);
  const float4 offset = bias[column];
  const float4 multiplier = scale[column];
  residual[index] = {
      fmaf(value.x + offset.x, multiplier.x, current.x),
      fmaf(value.y + offset.y, multiplier.y, current.y),
      fmaf(value.z + offset.z, multiplier.z, current.z),
      fmaf(value.w + offset.w, multiplier.w, current.w),
  };
}

inline void LaunchScaleAdd(float* residual, const float* branch,
                           const float* bias, const float* scale,
                           std::uint32_t rows, std::uint32_t width,
                           hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (width % 4U == 0U) {
    const std::uint32_t vectors = width / 4U;
    hipLaunchKernelGGL(
        ScaleAddFloat4Kernel, dim3((vectors + kThreads - 1U) / kThreads, rows),
        dim3(kThreads), 0, stream, reinterpret_cast<float4*>(residual),
        reinterpret_cast<const float4*>(branch),
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<const float4*>(scale), rows, vectors);
    return;
  }
  hipLaunchKernelGGL(
      ScaleAddKernel, dim3((width + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, residual, branch, bias, scale, rows, width);
}

inline void LaunchScaleAddFromBf16(float* residual, const std::uint16_t* branch,
                                   const float* bias, const float* scale,
                                   std::uint32_t rows, std::uint32_t width,
                                   hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (width % 4U == 0U) {
    const std::uint32_t vectors = width / 4U;
    hipLaunchKernelGGL(
        ScaleAddFloat4FromBf16Kernel,
        dim3((vectors + kThreads - 1U) / kThreads, rows), dim3(kThreads), 0,
        stream, reinterpret_cast<float4*>(residual),
        reinterpret_cast<const uint2*>(branch),
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<const float4*>(scale), rows, vectors);
    return;
  }
  hipLaunchKernelGGL(
      ScaleAddFromBf16Kernel, dim3((width + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, residual, branch, bias, scale, rows, width);
}

static __global__ void UnpackRgbKernel(const float* projected,
                                       const std::int32_t* frames, float* rgb,
                                       std::uint32_t frame_count,
                                       std::uint32_t tile_height,
                                       std::uint32_t tile_width,
                                       std::uint32_t output_frames) {
  const std::size_t elements = static_cast<std::size_t>(frame_count) *
                               tile_height * 16U * tile_width * 16U * 3U;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t channel = index % 3U;
  std::size_t remaining = index / 3U;
  const std::uint32_t x = remaining % (tile_width * 16U);
  remaining /= tile_width * 16U;
  const std::uint32_t y = remaining % (tile_height * 16U);
  const std::uint32_t selected = remaining / (tile_height * 16U);
  const std::uint32_t frame = static_cast<std::uint32_t>(frames[selected]);
  if (frame >= output_frames) {
    return;
  }
  std::uint32_t decoded_time = frame + 3U;
  if (output_frames == 22U && frame >= 17U) {
    decoded_time += 3U;
  }
  const std::uint32_t patch_time = decoded_time / 4U;
  const std::uint32_t within_time = decoded_time % 4U;
  const std::uint32_t patch_y = y / 16U;
  const std::uint32_t within_y = y % 16U;
  const std::uint32_t patch_x = x / 16U;
  const std::uint32_t within_x = x % 16U;
  const std::size_t patch =
      (static_cast<std::size_t>(patch_time) * tile_height + patch_y) *
          tile_width +
      patch_x;
  const std::size_t component =
      (((static_cast<std::size_t>(channel) * 4U + within_time) * 16U +
        within_y) *
           16U +
       within_x);
  constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float kDeviation[3] = {0.229F, 0.224F, 0.225F};
  const float value = fmaf(projected[patch * 3072U + component],
                           kDeviation[channel], kMean[channel]);
  rgb[index] = fminf(1.0F, fmaxf(0.0F, value));
}

inline void LaunchUnpackRgb(const float* projected, const std::int32_t* frames,
                            float* rgb, std::uint32_t frame_count,
                            std::uint32_t tile_height, std::uint32_t tile_width,
                            std::uint32_t output_frames, hipStream_t stream) {
  const std::size_t elements = static_cast<std::size_t>(frame_count) *
                               tile_height * 16U * tile_width * 16U * 3U;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(UnpackRgbKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, projected, frames, rgb,
                     frame_count, tile_height, tile_width, output_frames);
}

static __global__ void FloatToBf16Kernel(const float* source,
                                         std::uint16_t* destination,
                                         std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::uint32_t bits = __float_as_uint(source[index]);
    if ((bits & 0x7F800000U) == 0x7F800000U) {
      destination[index] = static_cast<std::uint16_t>((bits | 0x0040U) >> 16U);
    } else {
      const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
      destination[index] = static_cast<std::uint16_t>((bits + rounding) >> 16U);
    }
  }
}

inline void LaunchFloatToBf16(const float* source, std::uint16_t* destination,
                              std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      FloatToBf16Kernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, source, destination, elements);
}

static __global__ void Bf16ToFloatKernel(const std::uint16_t* source,
                                         float* destination,
                                         std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] =
        __uint_as_float(static_cast<std::uint32_t>(source[index]) << 16U);
  }
}

inline void LaunchBf16ToFloat(const std::uint16_t* source, float* destination,
                              std::size_t elements, hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(
      Bf16ToFloatKernel,
      dim3(static_cast<std::uint32_t>((elements + kThreads - 1U) / kThreads)),
      dim3(kThreads), 0, stream, source, destination, elements);
}

static __global__ void SoftmaxBf16InPlaceKernel(std::uint16_t* scores,
                                                std::uint32_t rows,
                                                std::uint32_t heads) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  if (row >= rows || head >= heads) {
    return;
  }
  std::uint16_t* row_scores =
      scores +
      (static_cast<std::size_t>(head) * rows + row) *
          static_cast<std::size_t>(rows);
  float local_maximum = -INFINITY;
  for (std::uint32_t column = threadIdx.x; column < rows;
       column += blockDim.x) {
    const float value = __uint_as_float(
        static_cast<std::uint32_t>(row_scores[column]) << 16U);
    local_maximum = fmaxf(local_maximum, value);
  }
  const float maximum = SoftmaxWaveMaximum(local_maximum);
  float local_sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < rows;
       column += blockDim.x) {
    const float value = __uint_as_float(
        static_cast<std::uint32_t>(row_scores[column]) << 16U);
    local_sum += expf(value - maximum);
  }
  const float inverse_sum = 1.0F / SoftmaxWaveSum(local_sum);
  for (std::uint32_t column = threadIdx.x; column < rows;
       column += blockDim.x) {
    const float value = __uint_as_float(
        static_cast<std::uint32_t>(row_scores[column]) << 16U);
    const float prob = expf(value - maximum) * inverse_sum;
    const std::uint32_t bits = __float_as_uint(prob);
    if ((bits & 0x7F800000U) == 0x7F800000U) {
      row_scores[column] = static_cast<std::uint16_t>((bits | 0x0040U) >> 16U);
    } else {
      const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
      row_scores[column] = static_cast<std::uint16_t>((bits + rounding) >> 16U);
    }
  }
}

inline void LaunchSoftmaxBf16InPlace(std::uint16_t* scores, std::uint32_t rows,
                                     std::uint32_t heads, hipStream_t stream) {
  hipLaunchKernelGGL(SoftmaxBf16InPlaceKernel, dim3(rows, heads), dim3(256), 0,
                     stream, scores, rows, heads);
}

__device__ __forceinline__ std::uint16_t FloatToBf16Scalar(float value) {
  const std::uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    return static_cast<std::uint16_t>((bits | 0x0040U) >> 16U);
  }
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

static __global__ void RmsNormBf16Kernel(const float* input,
                                         const float* weight,
                                         std::uint16_t* output_bf16,
                                         std::uint32_t rows,
                                         std::uint32_t width, float epsilon) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t wave = threadIdx.x >> 5U;
  const std::uint32_t row = blockIdx.x * kRowsPerBlock + wave;
  if (row >= rows) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(row) * width;
  float local = 0.0F;
  for (std::uint32_t column = lane; column < width; column += 32U) {
    const float v = input[base + column];
    local = fmaf(v, v, local);
  }
#pragma unroll
  for (std::uint32_t stride = 16U; stride != 0U; stride >>= 1U) {
    local += __shfl_down(local, stride, 32);
  }
  const float inverse =
      rsqrtf(__shfl(local, 0, 32) / static_cast<float>(width) + epsilon);
  for (std::uint32_t column = lane; column < width; column += 32U) {
    output_bf16[base + column] =
        FloatToBf16Scalar(input[base + column] * inverse * weight[column]);
  }
}

inline void LaunchRmsNormBf16(const float* input, const float* weight,
                              std::uint16_t* output_bf16, std::uint32_t rows,
                              std::uint32_t width, float epsilon,
                              hipStream_t stream) {
  constexpr std::uint32_t kRowsPerBlock = 8;
  hipLaunchKernelGGL(RmsNormBf16Kernel,
                     dim3((rows + kRowsPerBlock - 1U) / kRowsPerBlock),
                     dim3(kRowsPerBlock * 32U), 0, stream, input, weight,
                     output_bf16, rows, width, epsilon);
}

__device__ __forceinline__ float ReadScalarAsF32(const float* ptr,
                                               std::size_t idx) {
  return ptr[idx];
}

__device__ __forceinline__ float ReadScalarAsF32(const std::uint16_t* ptr,
                                               std::size_t idx) {
  return Bf16ToFloatScalar(ptr[idx]);
}

template<typename QkvT>
static __global__ void VideoQkvRopeBf16HeadMajorKernel(
    const QkvT* qkv, const float* bias, const float* cosine, const float* sine,
    std::uint16_t* query, std::uint16_t* key, std::uint16_t* value,
    std::uint32_t rows, std::uint32_t heads, std::uint32_t head_dimension,
    std::uint32_t rope_half, float epsilon) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || head >= heads || lane >= head_dimension) {
    return;
  }
  const std::size_t base =
      (static_cast<std::size_t>(row) * heads + head) * head_dimension * 3U;
  const std::size_t bias_base =
      static_cast<std::size_t>(head) * head_dimension * 3U;
  const float query_value =
      ReadScalarAsF32(qkv, base + lane) + bias[bias_base + lane];
  const float key_value =
      ReadScalarAsF32(qkv, base + head_dimension + lane) +
      bias[bias_base + head_dimension + lane];
  __shared__ float query_squares[64];
  __shared__ float key_squares[64];
  query_squares[lane] = query_value * query_value;
  key_squares[lane] = key_value * key_value;
  __syncthreads();
  for (std::uint32_t stride = head_dimension / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      query_squares[lane] += query_squares[lane + stride];
      key_squares[lane] += key_squares[lane + stride];
    }
    __syncthreads();
  }
  const float query_inverse =
      rsqrtf(query_squares[0] / static_cast<float>(head_dimension) + epsilon);
  const float key_inverse =
      rsqrtf(key_squares[0] / static_cast<float>(head_dimension) + epsilon);
  float transformed_query = query_value * query_inverse;
  float transformed_key = key_value * key_inverse;
  if (lane < rope_half) {
    const std::uint32_t pair = lane + rope_half;
    const float pair_query =
        (ReadScalarAsF32(qkv, base + pair) + bias[bias_base + pair]) *
        query_inverse;
    const float pair_key =
        (ReadScalarAsF32(qkv, base + head_dimension + pair) +
         bias[bias_base + head_dimension + pair]) *
        key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + lane];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + lane];
    transformed_query = transformed_query * c - pair_query * s;
    transformed_key = transformed_key * c - pair_key * s;
  } else if (lane < rope_half * 2U) {
    const std::uint32_t pair = lane - rope_half;
    const float pair_query =
        (ReadScalarAsF32(qkv, base + pair) + bias[bias_base + pair]) *
        query_inverse;
    const float pair_key =
        (ReadScalarAsF32(qkv, base + head_dimension + pair) +
         bias[bias_base + head_dimension + pair]) *
        key_inverse;
    const float c = cosine[static_cast<std::size_t>(row) * rope_half + pair];
    const float s = sine[static_cast<std::size_t>(row) * rope_half + pair];
    transformed_query = transformed_query * c + pair_query * s;
    transformed_key = transformed_key * c + pair_key * s;
  }
  // Head-major layout [head][row][head_dim] for coalesced FlashAttention loads.
  const std::size_t output =
      (static_cast<std::size_t>(head) * rows + row) * head_dimension + lane;
  query[output] = FloatToBf16Scalar(transformed_query);
  key[output] = FloatToBf16Scalar(transformed_key);
  value[output] =
      FloatToBf16Scalar(ReadScalarAsF32(qkv, base + head_dimension * 2U + lane) +
                        bias[bias_base + head_dimension * 2U + lane]);
}

inline void LaunchVideoQkvRopeBf16HeadMajor(
    const float* qkv, const float* bias, const float* cosine, const float* sine,
    std::uint16_t* query, std::uint16_t* key, std::uint16_t* value,
    std::uint32_t rows, std::uint32_t heads, std::uint32_t head_dimension,
    std::uint32_t rope_half, float epsilon, hipStream_t stream) {
  hipLaunchKernelGGL(VideoQkvRopeBf16HeadMajorKernel<float>, dim3(heads, rows),
                     dim3(head_dimension), 0, stream, qkv, bias, cosine, sine,
                     query, key, value, rows, heads, head_dimension, rope_half,
                     epsilon);
}

inline void LaunchVideoQkvRopeBf16HeadMajorFromBf16(
    const std::uint16_t* qkv, const float* bias, const float* cosine,
    const float* sine, std::uint16_t* query, std::uint16_t* key,
    std::uint16_t* value, std::uint32_t rows, std::uint32_t heads,
    std::uint32_t head_dimension, std::uint32_t rope_half, float epsilon,
    hipStream_t stream) {
  hipLaunchKernelGGL(VideoQkvRopeBf16HeadMajorKernel<std::uint16_t>,
                     dim3(heads, rows), dim3(head_dimension), 0, stream, qkv,
                     bias, cosine, sine, query, key, value, rows, heads,
                     head_dimension, rope_half, epsilon);
}

static __global__ void SwiGluBf16Float4Kernel(const float4* fused,
                                              const float4* bias,
                                              uint2* output_bf16,
                                              std::uint32_t rows,
                                              std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= vectors) {
    return;
  }
  const std::size_t input_base = static_cast<std::size_t>(row) * vectors * 2U;
  const float4 gate_value = fused[input_base + column];
  const float4 up_value = fused[input_base + vectors + column];
  const float4 gate_bias = bias[column];
  const float4 up_bias = bias[vectors + column];
  const float gx = gate_value.x + gate_bias.x;
  const float gy = gate_value.y + gate_bias.y;
  const float gz = gate_value.z + gate_bias.z;
  const float gw = gate_value.w + gate_bias.w;
  const std::uint32_t b0 =
      FloatToBf16Scalar((gx / (1.0F + expf(-gx))) * (up_value.x + up_bias.x));
  const std::uint32_t b1 =
      FloatToBf16Scalar((gy / (1.0F + expf(-gy))) * (up_value.y + up_bias.y));
  const std::uint32_t b2 =
      FloatToBf16Scalar((gz / (1.0F + expf(-gz))) * (up_value.z + up_bias.z));
  const std::uint32_t b3 =
      FloatToBf16Scalar((gw / (1.0F + expf(-gw))) * (up_value.w + up_bias.w));
  output_bf16[static_cast<std::size_t>(row) * vectors + column] = {
      b0 | (b1 << 16U), b2 | (b3 << 16U)};
}

static __global__ void SwiGluBf16Float4FromBf16Kernel(const uint2* fused_bf16,
                                                      const float4* bias,
                                                      uint2* output_bf16,
                                                      std::uint32_t rows,
                                                      std::uint32_t vectors) {
  const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows || column >= vectors) {
    return;
  }
  const std::size_t input_base = static_cast<std::size_t>(row) * vectors * 2U;
  const float4 gate_value = UnpackBf16x4(fused_bf16[input_base + column]);
  const float4 up_value =
      UnpackBf16x4(fused_bf16[input_base + vectors + column]);
  const float4 gate_bias = bias[column];
  const float4 up_bias = bias[vectors + column];
  const float gx = gate_value.x + gate_bias.x;
  const float gy = gate_value.y + gate_bias.y;
  const float gz = gate_value.z + gate_bias.z;
  const float gw = gate_value.w + gate_bias.w;
  const std::uint32_t b0 =
      FloatToBf16Scalar((gx / (1.0F + expf(-gx))) * (up_value.x + up_bias.x));
  const std::uint32_t b1 =
      FloatToBf16Scalar((gy / (1.0F + expf(-gy))) * (up_value.y + up_bias.y));
  const std::uint32_t b2 =
      FloatToBf16Scalar((gz / (1.0F + expf(-gz))) * (up_value.z + up_bias.z));
  const std::uint32_t b3 =
      FloatToBf16Scalar((gw / (1.0F + expf(-gw))) * (up_value.w + up_bias.w));
  output_bf16[static_cast<std::size_t>(row) * vectors + column] = {
      b0 | (b1 << 16U), b2 | (b3 << 16U)};
}

inline void LaunchSwiGluBf16(const float* fused, const float* bias,
                             std::uint16_t* output_bf16, std::uint32_t rows,
                             std::uint32_t width, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t vectors = width / 4U;
  hipLaunchKernelGGL(
      SwiGluBf16Float4Kernel, dim3((vectors + kThreads - 1U) / kThreads, rows),
      dim3(kThreads), 0, stream, reinterpret_cast<const float4*>(fused),
      reinterpret_cast<const float4*>(bias),
      reinterpret_cast<uint2*>(output_bf16), rows, vectors);
}

inline void LaunchSwiGluBf16FromBf16(const std::uint16_t* fused,
                                     const float* bias,
                                     std::uint16_t* output_bf16,
                                     std::uint32_t rows, std::uint32_t width,
                                     hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t vectors = width / 4U;
  hipLaunchKernelGGL(
      SwiGluBf16Float4FromBf16Kernel,
      dim3((vectors + kThreads - 1U) / kThreads, rows), dim3(kThreads), 0,
      stream, reinterpret_cast<const uint2*>(fused),
      reinterpret_cast<const float4*>(bias),
      reinterpret_cast<uint2*>(output_bf16), rows, vectors);
}

// Multi-query tiled BF16 WMMA FlashAttention for VisualVAE decoder
// (head_dim = 64, wave32 gfx1151). Q/K/V are head-major [heads, seq, 64] BF16;
// output is row-major [seq, heads * 64] BF16. Scores, softmax, and value
// accumulation stay in FP32 with no quadratic score buffer.
using VideoWmmaFragment = __attribute__((ext_vector_type(16))) short;
using VideoWmmaAccumulator = __attribute__((ext_vector_type(8))) float;

__device__ __forceinline__ VideoWmmaFragment LoadVideoFragment(
    const std::uint16_t* source) {
  union {
    VideoWmmaFragment fragment;
    uint4 words[2];
  } value;
  value.words[0] = *reinterpret_cast<const uint4*>(source);
  value.words[1] = *reinterpret_cast<const uint4*>(source + 8);
  return value.fragment;
}

static __global__ __launch_bounds__(256) void VideoWmmaFlashAttentionKernel(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t rows,
    std::uint32_t heads) {
  __shared__ __align__(16) std::uint16_t staged_key[32][72];
  __shared__ __align__(16) std::uint16_t staged_value[64][40];
  const int tid = threadIdx.x;
  const int lane = tid % 32;
  const int wave = tid / 32;
  const int first_query = blockIdx.x * 128 + wave * 16;
  const std::size_t head_base = static_cast<std::size_t>(blockIdx.y) * rows * 64;
  query += head_base;
  key += head_base;
  value += head_base;

  VideoWmmaFragment q[4];
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    q[d] = {};
    if (static_cast<std::uint32_t>(first_query + lane % 16) < rows) {
      q[d] = LoadVideoFragment(
          query + static_cast<std::size_t>(first_query + lane % 16) * 64 +
          d * 16);
    }
  }

  VideoWmmaAccumulator result[4]{};
  float maximum = -INFINITY;
  float sum = 0.0F;

  const int kv_row = tid / 8;         // 0..31
  const int kv_col = (tid % 8) * 8;   // 0, 8, ..., 56

  for (std::uint32_t base = 0; base < rows; base += 32U) {
    const bool valid_kv = (base + static_cast<std::uint32_t>(kv_row)) < rows;
    const uint4 k_vec =
        valid_kv
            ? *reinterpret_cast<const uint4*>(
                  key + static_cast<std::size_t>(base + kv_row) * 64 + kv_col)
            : uint4{0, 0, 0, 0};
    *reinterpret_cast<uint4*>(staged_key[kv_row] + kv_col) = k_vec;

    // Load V tile concurrently and transpose into [64][40] for vector fragment loads.
    union {
      uint4 packed;
      std::uint16_t h[8];
    } v_vec;
    v_vec.packed =
        valid_kv
            ? *reinterpret_cast<const uint4*>(
                  value + static_cast<std::size_t>(base + kv_row) * 64 + kv_col)
            : uint4{0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      staged_value[kv_col + i][kv_row] = v_vec.h[i];
    }
    __syncthreads();

    VideoWmmaAccumulator scores[2]{};
#pragma unroll
    for (int d = 0; d < 4; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const VideoWmmaFragment k =
            LoadVideoFragment(staged_key[half * 16 + lane % 16] + d * 16);
        scores[half] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(k, q[d], scores[half]);
      }
    }

    if (base + 32U > rows) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (base + static_cast<std::uint32_t>(half * 16 + 2 * i + lane / 16) >=
              rows) {
            scores[half][i] = -INFINITY;
          }
        }
      }
    }

    float next = maximum;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        // log2(e) / sqrt(64) = 1.4426950408889634 / 8
        scores[half][i] *= 0.18033688011112042F;
        next = fmaxf(next, scores[half][i]);
      }
    }
    next = fmaxf(next, __uint_as_float(__builtin_amdgcn_permlanex16(
                           __float_as_uint(next), __float_as_uint(next),
                           0x76543210U, 0xfedcba98U, false, false)));

    if (next > maximum) {
      const float correction = __builtin_amdgcn_exp2f(maximum - next);
      sum *= correction;
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        result[d] *= correction;
      }
      maximum = next;
    }

    float subtotal = 0.0F;
    VideoWmmaFragment probability_fragments[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = __builtin_amdgcn_exp2f(scores[half][i] - maximum);
        subtotal += p;
        const unsigned own = FloatToBf16Scalar(p);
        const unsigned peer = __builtin_amdgcn_permlanex16(
            own, own, 0x76543210U, 0xfedcba98U, false, false);
        probability_fragments[half][2 * i] = lane < 16 ? own : peer;
        probability_fragments[half][2 * i + 1] = lane < 16 ? peer : own;
      }
    }
    subtotal += __uint_as_float(__builtin_amdgcn_permlanex16(
        __float_as_uint(subtotal), __float_as_uint(subtotal), 0x76543210U,
        0xfedcba98U, false, false));
    sum += subtotal;

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const VideoWmmaFragment p = probability_fragments[half];
#pragma unroll
      for (int d = 0; d < 4; ++d) {
        const VideoWmmaFragment v =
            LoadVideoFragment(staged_value[d * 16 + lane % 16] + half * 16);
        result[d] =
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v, p, result[d]);
      }
    }
    __syncthreads();
  }

  const std::uint32_t row = static_cast<std::uint32_t>(first_query + lane % 16);
  if (row < rows) {
    const float reciprocal = __builtin_amdgcn_rcpf(sum);
#pragma unroll
    for (int d = 0; d < 4; ++d) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        output[(static_cast<std::size_t>(row) * heads + blockIdx.y) * 64 +
               d * 16 + 2 * i + lane / 16] =
            FloatToBf16Scalar(result[d][i] * reciprocal);
      }
    }
  }
}

inline void LaunchVideoWmmaFlashAttention(
    const std::uint16_t* query, const std::uint16_t* key,
    const std::uint16_t* value, std::uint16_t* output, std::uint32_t sequence,
    std::uint32_t heads, hipStream_t stream) {
  hipLaunchKernelGGL(VideoWmmaFlashAttentionKernel,
                     dim3((sequence + 127U) / 128U, heads), dim3(256), 0,
                     stream, query, key, value, output, sequence, heads);
}

// Tiled FlashAttention for VisualVAE decoder: [sequence, heads, head_dim=64].
// Eliminates materializing the [heads, sequence, sequence] score matrix.
static __global__ void VideoFlashAttentionKernel(
    const float* query, const float* key, const float* value, float* output,
    std::uint32_t sequence, std::uint32_t heads, float scale) {
  constexpr std::uint32_t kHeadDim = 64U;
  constexpr std::uint32_t kTileK = 32U;
  const std::uint32_t q_row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;  // 0..63
  if (q_row >= sequence || head >= heads) {
    return;
  }

  __shared__ float shared_q[kHeadDim];
  __shared__ float shared_acc[kHeadDim];
  __shared__ float shared_kv[kTileK * kHeadDim];
  __shared__ float scores[kTileK];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float tile_rescale;

  const std::size_t row_stride = static_cast<std::size_t>(heads) * kHeadDim;
  const std::size_t q_base =
      static_cast<std::size_t>(q_row) * row_stride + head * kHeadDim;
  shared_q[tid] = query[q_base + tid] * scale;
  shared_acc[tid] = 0.0F;
  if (tid == 0) {
    running_max = -INFINITY;
    running_sum = 0.0F;
    tile_rescale = 0.0F;
  }
  __syncthreads();

  const std::uint32_t k_idx = tid / 2U;    // 0..31
  const std::uint32_t d_half = tid % 2U;   // 0..1
  const std::uint32_t d_begin = d_half * 32U;
  const std::uint32_t d_end = d_begin + 32U;

  for (std::uint32_t tile_start = 0; tile_start < sequence;
       tile_start += kTileK) {
    const std::uint32_t tile_count =
        sequence - tile_start < kTileK ? sequence - tile_start : kTileK;
    const std::size_t tile_elements =
        static_cast<std::size_t>(tile_count) * kHeadDim;
    for (std::size_t idx = tid; idx < tile_elements; idx += kHeadDim) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / kHeadDim);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % kHeadDim);
      shared_kv[idx] =
          key[static_cast<std::size_t>(tile_start + k) * row_stride +
              head * kHeadDim + d];
    }
    __syncthreads();

    float dot = 0.0F;
    if (k_idx < tile_count) {
      const float* k_row =
          shared_kv + static_cast<std::size_t>(k_idx) * kHeadDim;
      for (std::uint32_t d = d_begin; d < d_end; ++d) {
        dot = fmaf(shared_q[d], k_row[d], dot);
      }
    }
    dot += __shfl_xor(dot, 1);
    if (d_half == 0 && k_idx < tile_count) {
      scores[k_idx] = dot;
    }
    __syncthreads();

    if (tid < 32U) {
      const float val = tid < tile_count ? scores[tid] : -INFINITY;
      float local_max = val;
      for (std::uint32_t s = 16U; s != 0; s >>= 1U) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, s));
      }
      const float old_max = running_max;
      const float new_max = fmaxf(old_max, local_max);
      const float old_sum = running_sum;
      const float rescale = old_sum == 0.0F ? 0.0F : expf(old_max - new_max);
      const float prob = tid < tile_count ? expf(val - new_max) : 0.0F;
      if (tid < tile_count) {
        scores[tid] = prob;
      }
      float prob_sum = prob;
      for (std::uint32_t s = 16U; s != 0; s >>= 1U) {
        prob_sum += __shfl_xor(prob_sum, s);
      }
      if (tid == 0) {
        running_max = new_max;
        running_sum = old_sum * rescale + prob_sum;
        tile_rescale = rescale;
      }
    }

    for (std::size_t idx = tid; idx < tile_elements; idx += kHeadDim) {
      const std::uint32_t k = static_cast<std::uint32_t>(idx / kHeadDim);
      const std::uint32_t d = static_cast<std::uint32_t>(idx % kHeadDim);
      shared_kv[idx] =
          value[static_cast<std::size_t>(tile_start + k) * row_stride +
                head * kHeadDim + d];
    }
    __syncthreads();

    float acc = shared_acc[tid] * tile_rescale;
    for (std::uint32_t k = 0; k < tile_count; ++k) {
      acc = fmaf(scores[k],
                 shared_kv[static_cast<std::size_t>(k) * kHeadDim + tid], acc);
    }
    shared_acc[tid] = acc;
    __syncthreads();
  }

  output[q_base + tid] = shared_acc[tid] / running_sum;
}

inline void LaunchVideoFlashAttention(const float* query, const float* key,
                                      const float* value, float* output,
                                      std::uint32_t sequence,
                                      std::uint32_t heads, float scale,
                                      hipStream_t stream) {
  hipLaunchKernelGGL(VideoFlashAttentionKernel, dim3(sequence, heads), dim3(64),
                     0, stream, query, key, value, output, sequence, heads,
                     scale);
}

}  // namespace gufo::minimax_h3::video_vae_ops

#endif  // GUFO_MODELS_MINIMAX_H3_VIDEO_VAE_OPS_HIP_HPP_
