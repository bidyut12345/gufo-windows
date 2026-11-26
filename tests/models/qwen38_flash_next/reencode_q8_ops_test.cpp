#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

struct DeviceDelete {
  void operator()(void* pointer) const { (void)hipFree(pointer); }
};

auto AllocateBytes(std::size_t bytes) {
  void* pointer = nullptr;
  CheckHip(hipMalloc(&pointer, bytes), "hipMalloc");
  return std::unique_ptr<void, DeviceDelete>(pointer);
}

std::uint16_t FloatToHalfBits(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t abs_bits = bits & 0x7fffffffU;
  if (abs_bits > 0x7f800000U) {
    return static_cast<std::uint16_t>(sign | 0x7e00U |
                                      ((abs_bits >> 13) & 0x01ffU));
  }
  if (abs_bits >= 0x47800000U) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (abs_bits < 0x38800000U) {
    if (abs_bits < 0x33000000U) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t shift = 126U - (abs_bits >> 23);
    const std::uint32_t mant = (abs_bits & 0x007fffffU) | 0x00800000U;
    const std::uint32_t rounded =
        mant + (1U << (shift - 1U)) - 1U + ((mant >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded =
      abs_bits + 0x00000fffU + ((abs_bits >> 13) & 1U);
  if (rounded >= 0x47800000U) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  const std::uint32_t half_exp = ((rounded >> 23) - 112U) << 10;
  const std::uint32_t half_mant = (rounded >> 13) & 0x03ffU;
  return static_cast<std::uint16_t>(sign | half_exp | half_mant);
}

void QuantizeRowQ8_0Cpu(const float* src, std::uint8_t* dst,
                        std::size_t cols) noexcept {
  for (std::size_t offset = 0; offset < cols; offset += 32) {
    float amax = 0.0F;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::abs(src[offset + j]));
    }
    const float d = amax / 127.0F;
    const float id = d != 0.0F ? 1.0F / d : 0.0F;
    const std::uint16_t d_half = FloatToHalfBits(d);
    std::memcpy(dst, &d_half, sizeof(d_half));
    auto* qs = reinterpret_cast<std::int8_t*>(dst + 2);
    for (std::size_t j = 0; j < 32; ++j) {
      const long v = std::clamp(std::lround(src[offset + j] * id), -127L, 127L);
      qs[j] = static_cast<std::int8_t>(v);
    }
    dst += 34;
  }
}

void FillValidScales(gufo::core::GgmlType type, std::vector<std::uint8_t>& buf,
                     std::size_t blocks, std::mt19937& rng) {
  static constexpr std::uint16_t kScales[] = {0x0000, 0x3400, 0x3800, 0x3C00,
                                              0xB400, 0xBC00, 0x4000};
  std::uniform_int_distribution<int> sdist(0, 6);
  const std::size_t bsize = buf.size() / blocks;
  for (std::size_t b = 0; b < blocks; ++b) {
    const std::uint16_t d = kScales[sdist(rng)];
    const std::uint16_t dm = kScales[sdist(rng)];
    if (type == gufo::core::GgmlType::kQ6_K) {
      std::memcpy(buf.data() + b * bsize + 208, &d, 2);
    } else if (type == gufo::core::GgmlType::kQ5_K) {
      std::memcpy(buf.data() + b * bsize, &d, 2);
      std::memcpy(buf.data() + b * bsize + 2, &dm, 2);
    } else {
      std::memcpy(buf.data() + b * bsize, &d, 2);
    }
  }
}

void CheckFormat(gufo::core::GgmlType type, std::size_t elements) {
  std::mt19937 rng(42 + static_cast<unsigned>(type));
  std::uniform_int_distribution<unsigned> bdist(0, 255);
  const std::size_t src_bytes = gufo::quant::QuantizedRowBytes(type, elements);
  const std::size_t dst_bytes = (elements / 32) * 34;
  const std::size_t qk = gufo::quant::QuantizedBlockElements(type);
  std::vector<std::uint8_t> src(src_bytes);
  for (auto& byte : src) {
    byte = static_cast<std::uint8_t>(bdist(rng));
  }
  FillValidScales(type, src, elements / qk, rng);

  std::vector<float> f32(elements);
  if (!gufo::quant::DequantizeRow(type, src.data(), f32.data(), elements)) {
    throw std::runtime_error("DequantizeRow failed");
  }
  std::vector<std::uint8_t> cpu_q8(dst_bytes);
  QuantizeRowQ8_0Cpu(f32.data(), cpu_q8.data(), elements);

  auto d_src = AllocateBytes(src_bytes);
  auto d_dst = AllocateBytes(dst_bytes);
  CheckHip(hipMemcpy(d_src.get(), src.data(), src_bytes, hipMemcpyHostToDevice),
           "upload src");
  if (!q::ReencodeToQ8_0(type, d_src.get(), d_dst.get(), elements, nullptr)) {
    throw std::runtime_error("ReencodeToQ8_0 returned false");
  }
  CheckHip(hipDeviceSynchronize(), "sync");
  std::vector<std::uint8_t> gpu_q8(dst_bytes);
  CheckHip(hipMemcpy(gpu_q8.data(), d_dst.get(), dst_bytes,
                     hipMemcpyDeviceToHost),
           "download dst");

  const std::size_t q8_blocks = elements / 32;
  for (std::size_t b = 0; b < q8_blocks; ++b) {
    const auto* cpu_blk =
        reinterpret_cast<const gufo::quant::block_q8_0*>(cpu_q8.data()) + b;
    const auto* gpu_blk =
        reinterpret_cast<const gufo::quant::block_q8_0*>(gpu_q8.data()) + b;
    if (cpu_blk->d != gpu_blk->d) {
      throw std::runtime_error("scale mismatch at block " + std::to_string(b));
    }
    for (int j = 0; j < 32; ++j) {
      if (cpu_blk->qs[j] != gpu_blk->qs[j]) {
        throw std::runtime_error("quant mismatch at block " +
                                 std::to_string(b) + " elem " +
                                 std::to_string(j));
      }
    }
  }
}

}  // namespace

int main() {
  try {
    for (gufo::core::GgmlType type : {gufo::core::GgmlType::kIQ4_NL,
                                      gufo::core::GgmlType::kIQ4_XS,
                                      gufo::core::GgmlType::kQ5_K,
                                      gufo::core::GgmlType::kQ6_K}) {
      CheckFormat(type, 4096 * 8);
    }
    std::cout << "PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL: " << ex.what() << "\n";
    return 1;
  }
}

