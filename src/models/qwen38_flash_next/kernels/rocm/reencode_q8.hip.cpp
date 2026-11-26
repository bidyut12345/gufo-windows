// Load-time requantization of dense IQ4_NL / IQ4_XS / Q5_K / Q6_K weights to
// Q8_0 on device. Built without fast math or contraction (extension.cmake) so
// dequantization and rounding match the CPU path.
#include <hip/hip_fp16.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/experimental/ggml_dequant.hpp>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

#pragma clang fp contract(off)

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float HalfBitsToFloat(std::uint16_t bits) noexcept {
  return __half2float(std::bit_cast<__half>(bits));
}

__device__ __forceinline__ void WriteQ8_0Block(
    const float (&x)[32], quant::block_q8_0& out) noexcept {
  float amax = 0.0F;
  for (int j = 0; j < 32; ++j) {
    amax = fmaxf(amax, fabsf(x[j]));
  }
  const float d = amax / 127.0F;
  const float id = d != 0.0F ? 1.0F / d : 0.0F;
  out.d = std::bit_cast<std::uint16_t>(__float2half_rn(d));
  for (int j = 0; j < 32; ++j) {
    const long q = llroundf(x[j] * id);
    const long clamped = q < -127L ? -127L : (q > 127L ? 127L : q);
    out.qs[j] = static_cast<std::int8_t>(clamped);
  }
}

__device__ __forceinline__ void GetQ4ScaleMin(std::uint32_t index,
                                              const std::uint8_t* packed,
                                              std::uint8_t& scale,
                                              std::uint8_t& minimum) noexcept {
  if (index < 4U) {
    scale = packed[index] & 0x3FU;
    minimum = packed[index + 4U] & 0x3FU;
    return;
  }
  scale = static_cast<std::uint8_t>((packed[index + 4U] & 0x0FU) |
                                    ((packed[index - 4U] >> 6U) << 4U));
  minimum = static_cast<std::uint8_t>((packed[index + 4U] >> 4U) |
                                      ((packed[index] >> 6U) << 4U));
}

__global__ void ReencodeIQ4NLToQ8_0Kernel(
    const quant::block_iq4_nl* __restrict__ src,
    quant::block_q8_0* __restrict__ dst, std::size_t blocks) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= blocks) {
    return;
  }
  const quant::block_iq4_nl b = src[idx];
  const float d = HalfBitsToFloat(b.d);
  float x[32];
  for (int j = 0; j < 16; ++j) {
    const std::uint8_t packed = b.qs[j];
    x[j] = d * static_cast<float>(quant::kValuesIq4Nl[packed & 0x0FU]);
    x[j + 16] = d * static_cast<float>(quant::kValuesIq4Nl[packed >> 4U]);
  }
  WriteQ8_0Block(x, dst[idx]);
}

__global__ void ReencodeIQ4XSToQ8_0Kernel(
    const quant::block_iq4_xs* __restrict__ src,
    quant::block_q8_0* __restrict__ dst, std::size_t superblocks) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t sb = idx / 8U;
  if (sb >= superblocks) {
    return;
  }
  const std::uint32_t ib = static_cast<std::uint32_t>(idx % 8U);
  const quant::block_iq4_xs& b = src[sb];
  const int ls =
      static_cast<int>((b.scales_l[ib / 2U] >> (4U * (ib % 2U))) & 0x0FU) |
      static_cast<int>(((b.scales_h >> (2U * ib)) & 0x03U) << 4U);
  const float dl = HalfBitsToFloat(b.d) * static_cast<float>(ls - 32);
  float x[32];
  const std::uint32_t qs_base = ib * 16U;
  for (std::uint32_t j = 0; j < 16U; ++j) {
    const std::uint8_t packed = b.qs[qs_base + j];
    x[j] = dl * static_cast<float>(quant::kValuesIq4Nl[packed & 0x0FU]);
    x[j + 16U] = dl * static_cast<float>(quant::kValuesIq4Nl[packed >> 4U]);
  }
  WriteQ8_0Block(x, dst[idx]);
}

__global__ void ReencodeQ5KToQ8_0Kernel(
    const quant::block_q5_K* __restrict__ src,
    quant::block_q8_0* __restrict__ dst, std::size_t superblocks) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t sb = idx / 8U;
  if (sb >= superblocks) {
    return;
  }
  const std::uint32_t ib = static_cast<std::uint32_t>(idx % 8U);
  const quant::block_q5_K& b = src[sb];
  std::uint8_t sc = 0;
  std::uint8_t m = 0;
  GetQ4ScaleMin(ib, b.scales, sc, m);
  const float d_sc = HalfBitsToFloat(b.d) * static_cast<float>(sc);
  const float dm_m = HalfBitsToFloat(b.dmin) * static_cast<float>(m);
  const std::uint32_t gg = ib / 2U;
  const bool lohalf = (ib % 2U) == 0U;
  const std::uint32_t qs_base = gg * 32U;
  float x[32];
  for (std::uint32_t lane = 0; lane < 32U; ++lane) {
    const std::uint8_t qb = b.qs[qs_base + lane];
    const std::uint8_t quant4 = lohalf ? (qb & 0x0FU) : (qb >> 4U);
    const std::uint8_t quant = static_cast<std::uint8_t>(
        quant4 + (((b.qh[lane] >> ib) & 1U) ? 16U : 0U));
    x[lane] = d_sc * static_cast<float>(quant) - dm_m;
  }
  WriteQ8_0Block(x, dst[idx]);
}

__global__ void ReencodeQ6KToQ8_0Kernel(
    const quant::block_q6_K* __restrict__ src,
    quant::block_q8_0* __restrict__ dst, std::size_t superblocks) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t sb = idx / 8U;
  if (sb >= superblocks) {
    return;
  }
  const std::uint32_t ib = static_cast<std::uint32_t>(idx % 8U);
  const quant::block_q6_K& b = src[sb];
  const std::uint32_t half = ib / 4U;
  const std::uint32_t segment = ib % 4U;
  const std::uint32_t ql_base = half * 64U + ((segment & 1U) ? 32U : 0U);
  const std::uint32_t qh_base = half * 32U;
  const std::uint32_t qh_shift = 2U * segment;
  const bool high_nibble = segment >= 2U;
  const float d_f = HalfBitsToFloat(b.d);
  const float sc0 =
      static_cast<float>(b.scales[half * 8U + segment * 2U + 0U]);
  const float sc1 =
      static_cast<float>(b.scales[half * 8U + segment * 2U + 1U]);
  float x[32];
  for (std::uint32_t lane = 0; lane < 32U; ++lane) {
    const std::uint8_t ql = b.ql[ql_base + lane];
    const std::uint8_t low = high_nibble ? (ql >> 4U) : (ql & 0x0FU);
    const std::uint8_t high = (b.qh[qh_base + lane] >> qh_shift) & 0x03U;
    const auto quant =
        static_cast<std::int8_t>(static_cast<int>((high << 4U) | low) - 32);
    const float sc = lane < 16U ? sc0 : sc1;
    x[lane] = d_f * sc * static_cast<float>(quant);
  }
  WriteQ8_0Block(x, dst[idx]);
}

}  // namespace

bool ReencodeToQ8_0(gufo::core::GgmlType src_type, const void* src, void* dst,
                    std::size_t elements, hipStream_t stream) {
  if (elements == 0) {
    return true;
  }
  const std::size_t q8_blocks = elements / 32U;
  const dim3 grid(static_cast<unsigned>((q8_blocks + kThreads - 1U) / kThreads));
  const dim3 block(kThreads);
  switch (src_type) {
    case gufo::core::GgmlType::kIQ4_NL:
      hipLaunchKernelGGL(
          ReencodeIQ4NLToQ8_0Kernel, grid, block, 0, stream,
          static_cast<const quant::block_iq4_nl*>(src),
          static_cast<quant::block_q8_0*>(dst), q8_blocks);
      return true;
    case gufo::core::GgmlType::kIQ4_XS:
      hipLaunchKernelGGL(
          ReencodeIQ4XSToQ8_0Kernel, grid, block, 0, stream,
          static_cast<const quant::block_iq4_xs*>(src),
          static_cast<quant::block_q8_0*>(dst), elements / 256U);
      return true;
    case gufo::core::GgmlType::kQ5_K:
      hipLaunchKernelGGL(
          ReencodeQ5KToQ8_0Kernel, grid, block, 0, stream,
          static_cast<const quant::block_q5_K*>(src),
          static_cast<quant::block_q8_0*>(dst), elements / 256U);
      return true;
    case gufo::core::GgmlType::kQ6_K:
      hipLaunchKernelGGL(
          ReencodeQ6KToQ8_0Kernel, grid, block, 0, stream,
          static_cast<const quant::block_q6_K*>(src),
          static_cast<quant::block_q8_0*>(dst), elements / 256U);
      return true;
    default:
      return false;
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm
