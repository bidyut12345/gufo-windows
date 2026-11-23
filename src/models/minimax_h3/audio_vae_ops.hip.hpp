#ifndef GUFO_MODELS_MINIMAX_H3_AUDIO_VAE_OPS_HIP_HPP_
#define GUFO_MODELS_MINIMAX_H3_AUDIO_VAE_OPS_HIP_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/minimax_h3/audio_vae.hpp"

namespace gufo::minimax_h3::audio_vae_ops {

static __global__ void PrepareLatentKernel(const float* normalized,
                                           const float* mean,
                                           const float* deviation,
                                           float* output,
                                           std::uint32_t length) {
  const std::size_t count =
      static_cast<std::size_t>(kH3AudioVaeStereoChannels) * length *
      kH3AudioVaeLatentChannels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t channel =
      static_cast<std::uint32_t>(index % kH3AudioVaeLatentChannels);
  const std::size_t row = index / kH3AudioVaeLatentChannels;
  const std::uint32_t time = static_cast<std::uint32_t>(row % length);
  const std::uint32_t stereo = static_cast<std::uint32_t>(row / length);
  const std::size_t source =
      (static_cast<std::size_t>(channel) * kH3AudioVaeStereoChannels + stereo) *
          length +
      time;
  output[index] = fmaf(normalized[source], deviation[channel], mean[channel]);
}

inline void LaunchPrepareLatent(const float* normalized, const float* mean,
                                const float* deviation, float* output,
                                std::uint32_t length, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(kH3AudioVaeStereoChannels) * length *
      kH3AudioVaeLatentChannels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(PrepareLatentKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, normalized, mean, deviation, output, length);
}

static __global__ void WeightNormKernel(const float* vector,
                                        const float* magnitude, float* output,
                                        std::uint32_t outer,
                                        std::uint32_t inner) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (row >= outer) {
    return;
  }
  __shared__ float reduction[256];
  const std::size_t base = static_cast<std::size_t>(row) * inner;
  float local = 0.0F;
  for (std::uint32_t index = lane; index < inner; index += blockDim.x) {
    const float value = vector[base + index];
    local = fmaf(value, value, local);
  }
  reduction[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    __syncthreads();
  }
  const float scale = magnitude[row] * rsqrtf(reduction[0]);
  for (std::uint32_t index = lane; index < inner; index += blockDim.x) {
    output[base + index] = vector[base + index] * scale;
  }
}

inline void LaunchWeightNorm(const float* vector, const float* magnitude,
                             float* output, std::uint32_t outer,
                             std::uint32_t inner, hipStream_t stream) {
  hipLaunchKernelGGL(WeightNormKernel, dim3(outer), dim3(256), 0, stream,
                     vector, magnitude, output, outer, inner);
}

static __global__ void WeightNormTransposeKernel(const float* vector,
                                                 const float* magnitude,
                                                 float* output,
                                                 std::uint32_t input_channels,
                                                 std::uint32_t output_channels,
                                                 std::uint32_t kernel) {
  const std::uint32_t input_channel = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (input_channel >= input_channels) {
    return;
  }
  __shared__ float reduction[256];
  const std::uint32_t inner = output_channels * kernel;
  const std::size_t base = static_cast<std::size_t>(input_channel) * inner;
  float local = 0.0F;
  for (std::uint32_t index = lane; index < inner; index += blockDim.x) {
    const float value = vector[base + index];
    local = fmaf(value, value, local);
  }
  reduction[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2U; stride != 0; stride >>= 1U) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    __syncthreads();
  }
  const float scale = magnitude[input_channel] * rsqrtf(reduction[0]);
  for (std::uint32_t index = lane; index < inner; index += blockDim.x) {
    const std::uint32_t output_channel = index / kernel;
    const std::uint32_t tap = index % kernel;
    const std::size_t target =
        (static_cast<std::size_t>(output_channel) * input_channels +
         input_channel) *
            kernel +
        tap;
    output[target] = vector[base + index] * scale;
  }
}

inline void LaunchWeightNormTranspose(const float* vector,
                                      const float* magnitude, float* output,
                                      std::uint32_t input_channels,
                                      std::uint32_t output_channels,
                                      std::uint32_t kernel,
                                      hipStream_t stream) {
  hipLaunchKernelGGL(WeightNormTransposeKernel, dim3(input_channels), dim3(256),
                     0, stream, vector, magnitude, output, input_channels,
                     output_channels, kernel);
}

static __global__ void Conv1dIm2ColKernel(
    const float* input, float* columns, std::uint32_t batch,
    std::uint32_t input_length, std::uint32_t input_channels,
    std::uint32_t output_length, std::uint32_t kernel, std::uint32_t padding,
    std::uint32_t dilation) {
  const std::size_t column_width =
      static_cast<std::size_t>(input_channels) * kernel;
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * column_width;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t tap = static_cast<std::uint32_t>(index % kernel);
  std::size_t remaining = index / kernel;
  const std::uint32_t input_channel =
      static_cast<std::uint32_t>(remaining % input_channels);
  remaining /= input_channels;
  const std::uint32_t output_time =
      static_cast<std::uint32_t>(remaining % output_length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / output_length);
  const std::int64_t input_time = static_cast<std::int64_t>(output_time) -
                                  padding +
                                  static_cast<std::int64_t>(tap) * dilation;
  if (input_time < 0 || input_time >= static_cast<std::int64_t>(input_length)) {
    columns[index] = 0.0F;
    return;
  }
  const std::size_t source =
      (static_cast<std::size_t>(batch_index) * input_length +
       static_cast<std::uint32_t>(input_time)) *
          input_channels +
      input_channel;
  columns[index] = input[source];
}

inline void LaunchConv1dIm2Col(const float* input, float* columns,
                               std::uint32_t batch, std::uint32_t input_length,
                               std::uint32_t input_channels,
                               std::uint32_t output_length,
                               std::uint32_t kernel, std::uint32_t padding,
                               std::uint32_t dilation, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * input_channels * kernel;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(Conv1dIm2ColKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, columns, batch, input_length,
                     input_channels, output_length, kernel, padding, dilation);
}

static __global__ void ConvTranspose1dIm2ColKernel(
    const float* input, float* columns, std::uint32_t batch,
    std::uint32_t input_length, std::uint32_t input_channels,
    std::uint32_t output_length, std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t padding) {
  const std::size_t column_width =
      static_cast<std::size_t>(input_channels) * kernel;
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * column_width;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t tap = static_cast<std::uint32_t>(index % kernel);
  std::size_t remaining = index / kernel;
  const std::uint32_t input_channel =
      static_cast<std::uint32_t>(remaining % input_channels);
  remaining /= input_channels;
  const std::uint32_t output_time =
      static_cast<std::uint32_t>(remaining % output_length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / output_length);
  const std::int64_t numerator =
      static_cast<std::int64_t>(output_time) + padding - tap;
  if (numerator < 0 || numerator % stride != 0) {
    columns[index] = 0.0F;
    return;
  }
  const std::int64_t input_time = numerator / stride;
  if (input_time >= static_cast<std::int64_t>(input_length)) {
    columns[index] = 0.0F;
    return;
  }
  const std::size_t source =
      (static_cast<std::size_t>(batch_index) * input_length +
       static_cast<std::uint32_t>(input_time)) *
          input_channels +
      input_channel;
  columns[index] = input[source];
}

inline void LaunchConvTranspose1dIm2Col(
    const float* input, float* columns, std::uint32_t batch,
    std::uint32_t input_length, std::uint32_t input_channels,
    std::uint32_t output_length, std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t padding, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * input_channels * kernel;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(ConvTranspose1dIm2ColKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, columns, batch, input_length,
                     input_channels, output_length, kernel, stride, padding);
}

static __global__ void AddConvBiasKernel(float* output, const float* bias,
                                         std::size_t rows,
                                         std::uint32_t columns) {
  const std::size_t count = rows * columns;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] += bias[index % columns];
  }
}

inline void LaunchAddConvBias(float* output, const float* bias,
                              std::size_t rows, std::uint32_t columns,
                              hipStream_t stream) {
  const std::size_t count = rows * columns;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AddConvBiasKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, output, bias, rows, columns);
}

static __global__ void Conv1dKernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t batch, std::uint32_t input_length,
    std::uint32_t input_channels, std::uint32_t output_length,
    std::uint32_t output_channels, std::uint32_t kernel, std::uint32_t padding,
    std::uint32_t dilation) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * output_channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t output_channel =
      static_cast<std::uint32_t>(index % output_channels);
  std::size_t remaining = index / output_channels;
  const std::uint32_t output_time =
      static_cast<std::uint32_t>(remaining % output_length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / output_length);
  float value = bias == nullptr ? 0.0F : bias[output_channel];
  const std::size_t weight_base =
      static_cast<std::size_t>(output_channel) * input_channels * kernel;
  for (std::uint32_t input_channel = 0; input_channel < input_channels;
       ++input_channel) {
    const std::size_t channel_weight =
        weight_base + static_cast<std::size_t>(input_channel) * kernel;
    for (std::uint32_t tap = 0; tap < kernel; ++tap) {
      const std::int64_t input_time = static_cast<std::int64_t>(output_time) -
                                      padding +
                                      static_cast<std::int64_t>(tap) * dilation;
      if (input_time < 0 ||
          input_time >= static_cast<std::int64_t>(input_length)) {
        continue;
      }
      const std::size_t source =
          (static_cast<std::size_t>(batch_index) * input_length +
           static_cast<std::uint32_t>(input_time)) *
              input_channels +
          input_channel;
      value = fmaf(input[source], weight[channel_weight + tap], value);
    }
  }
  output[index] = value;
}

inline void LaunchConv1d(const float* input, const float* weight,
                         const float* bias, float* output, std::uint32_t batch,
                         std::uint32_t input_length,
                         std::uint32_t input_channels,
                         std::uint32_t output_length,
                         std::uint32_t output_channels, std::uint32_t kernel,
                         std::uint32_t padding, std::uint32_t dilation,
                         hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * output_channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(Conv1dKernel, dim3((count + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, weight, bias, output,
                     batch, input_length, input_channels, output_length,
                     output_channels, kernel, padding, dilation);
}

static __global__ void ConvTranspose1dKernel(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t batch, std::uint32_t input_length,
    std::uint32_t input_channels, std::uint32_t output_length,
    std::uint32_t output_channels, std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t padding) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * output_channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t output_channel =
      static_cast<std::uint32_t>(index % output_channels);
  std::size_t remaining = index / output_channels;
  const std::uint32_t output_time =
      static_cast<std::uint32_t>(remaining % output_length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / output_length);
  float value = bias == nullptr ? 0.0F : bias[output_channel];
  for (std::uint32_t input_channel = 0; input_channel < input_channels;
       ++input_channel) {
    const std::size_t weight_base =
        (static_cast<std::size_t>(input_channel) * output_channels +
         output_channel) *
        kernel;
    for (std::uint32_t tap = 0; tap < kernel; ++tap) {
      const std::int64_t numerator =
          static_cast<std::int64_t>(output_time) + padding - tap;
      if (numerator < 0 || numerator % stride != 0) {
        continue;
      }
      const std::int64_t input_time = numerator / stride;
      if (input_time >= static_cast<std::int64_t>(input_length)) {
        continue;
      }
      const std::size_t source =
          (static_cast<std::size_t>(batch_index) * input_length +
           static_cast<std::uint32_t>(input_time)) *
              input_channels +
          input_channel;
      value = fmaf(input[source], weight[weight_base + tap], value);
    }
  }
  output[index] = value;
}

inline void LaunchConvTranspose1d(
    const float* input, const float* weight, const float* bias, float* output,
    std::uint32_t batch, std::uint32_t input_length,
    std::uint32_t input_channels, std::uint32_t output_length,
    std::uint32_t output_channels, std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t padding, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * output_length * output_channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(ConvTranspose1dKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, weight, bias, output, batch,
                     input_length, input_channels, output_length,
                     output_channels, kernel, stride, padding);
}

static __global__ void AliasFreeSnakeKernel(
    const float* input, const float* alpha_log, const float* beta_log,
    const float* upsample_filter, const float* downsample_filter, float* output,
    std::uint32_t batch, std::uint32_t length, std::uint32_t channels) {
  const std::size_t count = static_cast<std::size_t>(batch) * length * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  std::size_t remaining = index / channels;
  const std::uint32_t time = static_cast<std::uint32_t>(remaining % length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / length);
  const float alpha = expf(alpha_log[channel]);
  const float beta = expf(beta_log[channel]);
  float result = 0.0F;
  for (int down_tap = 0; down_tap < 12; ++down_tap) {
    int up_time = static_cast<int>(time * 2U) + down_tap - 5;
    up_time = max(0, min(up_time, static_cast<int>(length * 2U) - 1));
    const int raw_time = up_time + 15;
    float upsampled = 0.0F;
    for (int up_tap = 0; up_tap < 12; ++up_tap) {
      const int numerator = raw_time - up_tap;
      if (numerator < 0 || (numerator & 1) != 0) {
        continue;
      }
      int source_time = numerator / 2 - 5;
      source_time = max(0, min(source_time, static_cast<int>(length) - 1));
      const std::size_t source =
          (static_cast<std::size_t>(batch_index) * length +
           static_cast<std::uint32_t>(source_time)) *
              channels +
          channel;
      upsampled =
          fmaf(input[source], 2.0F * upsample_filter[up_tap], upsampled);
    }
    const float sine = sinf(alpha * upsampled);
    const float activated = upsampled + sine * sine / (beta + 1.0e-9F);
    result = fmaf(activated, downsample_filter[down_tap], result);
  }
  output[index] = result;
}

inline void LaunchAliasFreeSnake(const float* input, const float* alpha_log,
                                 const float* beta_log,
                                 const float* upsample_filter,
                                 const float* downsample_filter, float* output,
                                 std::uint32_t batch, std::uint32_t length,
                                 std::uint32_t channels, hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(batch) * length * channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AliasFreeSnakeKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, alpha_log, beta_log, upsample_filter,
                     downsample_filter, output, batch, length, channels);
}

static __global__ void AliasFreeSnakeUpsampleKernel(
    const float* input, const float* alpha_log, const float* beta_log,
    const float* upsample_filter, float* activated, std::uint32_t batch,
    std::uint32_t length, std::uint32_t channels) {
  const std::uint32_t upsampled_length = length * 2U;
  const std::size_t count =
      static_cast<std::size_t>(batch) * upsampled_length * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  std::size_t remaining = index / channels;
  const std::uint32_t upsampled_time =
      static_cast<std::uint32_t>(remaining % upsampled_length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / upsampled_length);
  const int raw_time = static_cast<int>(upsampled_time) + 15;
  float upsampled = 0.0F;
  for (int tap = 0; tap < 12; ++tap) {
    const int numerator = raw_time - tap;
    if (numerator < 0 || (numerator & 1) != 0) {
      continue;
    }
    int source_time = numerator / 2 - 5;
    source_time = max(0, min(source_time, static_cast<int>(length) - 1));
    const std::size_t source = (static_cast<std::size_t>(batch_index) * length +
                                static_cast<std::uint32_t>(source_time)) *
                                   channels +
                               channel;
    upsampled = fmaf(input[source], 2.0F * upsample_filter[tap], upsampled);
  }
  const float alpha = expf(alpha_log[channel]);
  const float beta = expf(beta_log[channel]);
  const float sine = sinf(alpha * upsampled);
  activated[index] = upsampled + sine * sine / (beta + 1.0e-9F);
}

inline void LaunchAliasFreeSnakeUpsample(
    const float* input, const float* alpha_log, const float* beta_log,
    const float* upsample_filter, float* activated, std::uint32_t batch,
    std::uint32_t length, std::uint32_t channels, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(batch) * length * 2U * channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AliasFreeSnakeUpsampleKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, alpha_log, beta_log, upsample_filter,
                     activated, batch, length, channels);
}

static __global__ void AliasFreeSnakeDownsampleKernel(
    const float* activated, const float* downsample_filter, float* output,
    std::uint32_t batch, std::uint32_t length, std::uint32_t channels) {
  const std::size_t count = static_cast<std::size_t>(batch) * length * channels;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const std::uint32_t channel = static_cast<std::uint32_t>(index % channels);
  std::size_t remaining = index / channels;
  const std::uint32_t time = static_cast<std::uint32_t>(remaining % length);
  const std::uint32_t batch_index =
      static_cast<std::uint32_t>(remaining / length);
  const std::uint32_t upsampled_length = length * 2U;
  float result = 0.0F;
  for (int tap = 0; tap < 12; ++tap) {
    int upsampled_time = static_cast<int>(time * 2U) + tap - 5;
    upsampled_time =
        max(0, min(upsampled_time, static_cast<int>(upsampled_length) - 1));
    const std::size_t source =
        (static_cast<std::size_t>(batch_index) * upsampled_length +
         static_cast<std::uint32_t>(upsampled_time)) *
            channels +
        channel;
    result = fmaf(activated[source], downsample_filter[tap], result);
  }
  output[index] = result;
}

inline void LaunchAliasFreeSnakeDownsample(const float* activated,
                                           const float* downsample_filter,
                                           float* output, std::uint32_t batch,
                                           std::uint32_t length,
                                           std::uint32_t channels,
                                           hipStream_t stream) {
  const std::size_t count = static_cast<std::size_t>(batch) * length * channels;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AliasFreeSnakeDownsampleKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, activated, downsample_filter, output, batch,
                     length, channels);
}

static __global__ void CopyKernel(const float* input, float* output,
                                  std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = input[index];
  }
}

inline void LaunchCopy(const float* input, float* output, std::size_t elements,
                       hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(CopyKernel, dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, input, output, elements);
}

static __global__ void AddScaledKernel(const float* left, const float* right,
                                       float* output, std::size_t elements,
                                       float left_scale, float right_scale) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = left[index] * left_scale + right[index] * right_scale;
  }
}

inline void LaunchAddScaled(const float* left, const float* right,
                            float* output, std::size_t elements,
                            float left_scale, float right_scale,
                            hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(AddScaledKernel,
                     dim3((elements + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, left, right, output, elements,
                     left_scale, right_scale);
}

static __global__ void ClipAndReorderKernel(const float* input,
                                            float* channel_major,
                                            std::uint32_t samples) {
  const std::size_t count =
      static_cast<std::size_t>(kH3AudioVaeStereoChannels) * samples;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  channel_major[index] = fminf(1.0F, fmaxf(-1.0F, input[index]));
}

inline void LaunchClipAndReorder(const float* input, float* channel_major,
                                 std::uint32_t samples, hipStream_t stream) {
  const std::size_t count =
      static_cast<std::size_t>(kH3AudioVaeStereoChannels) * samples;
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(ClipAndReorderKernel,
                     dim3((count + kThreads - 1U) / kThreads), dim3(kThreads),
                     0, stream, input, channel_major, samples);
}

static __global__ void SingleOutputConvGemmKernel(const float* weight,
                                                  const float* columns,
                                                  float* output,
                                                  std::size_t batch_rows,
                                                  std::size_t reduction) {
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= batch_rows) {
    return;
  }
  const float* col_row = columns + row * reduction;
  float sum = 0.0F;
  for (std::size_t k = 0; k < reduction; ++k) {
    sum = fmaf(weight[k], col_row[k], sum);
  }
  output[row] = sum;
}

inline void LaunchSingleOutputConvGemm(const float* weight,
                                       const float* columns, float* output,
                                       std::size_t batch_rows,
                                       std::size_t reduction,
                                       hipStream_t stream) {
  constexpr std::size_t kThreads = 256;
  hipLaunchKernelGGL(SingleOutputConvGemmKernel,
                     dim3((batch_rows + kThreads - 1U) / kThreads),
                     dim3(kThreads), 0, stream, weight, columns, output,
                     batch_rows, reduction);
}

}  // namespace gufo::minimax_h3::audio_vae_ops

#endif  // GUFO_MODELS_MINIMAX_H3_AUDIO_VAE_OPS_HIP_HPP_
