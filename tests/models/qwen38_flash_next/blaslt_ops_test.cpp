#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Check(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}

float AsFloat(std::uint16_t value, hipDataType type) {
  return type == HIP_R_16BF
             ? std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16)
             : __half2float(std::bit_cast<__half>(value));
}

struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

}  // namespace

int main() {
  try {
    using gufo::models::qwen38_flash_next::rocm::BlasLt;
    std::string error;
    auto warmed = BlasLt::Create(nullptr, &error);
    Require(warmed != nullptr, error);
    constexpr int max_n = 4096;
    for (const auto [type, m, k] :
         {std::tuple{HIP_R_16F, 96, 2560}, std::tuple{HIP_R_16F, 513, 2560},
          std::tuple{HIP_R_16F, 33, 96}, std::tuple{HIP_R_16BF, 64, 2560},
          std::tuple{HIP_R_16BF, 512, 2560}}) {
      // Prepare before inputs exist, then compare against lazy plan creation
      // in a fresh instance, including ragged shapes after the prepared one.
      Require(warmed->Prepare(type, m, max_n, k, &error), error);
      std::vector<std::uint16_t> weights(static_cast<std::size_t>(m) * k);
      std::vector<std::uint16_t> input(static_cast<std::size_t>(max_n) * k);
      unsigned rng = 123;
      for (auto* values : {&weights, &input}) {
        for (auto& v : *values) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          const float value =
              static_cast<float>(int(rng % 8193) - 4096) / 4096.0F;
          // Independent round-to-nearest-even for these finite BF16 inputs.
          const auto bits = std::bit_cast<std::uint32_t>(value);
          v = type == HIP_R_16BF
                  ? static_cast<std::uint16_t>(
                        (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16)
                  : std::bit_cast<std::uint16_t>(__float2half_rn(value));
        }
      }
      Buffer a(weights.size() * sizeof(__half));
      Buffer b(input.size() * sizeof(__half));
      Buffer c(static_cast<std::size_t>(m) * max_n * sizeof(float));
      Check(hipMemcpy(a.data, weights.data(), weights.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(b.data, input.data(), input.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
      if (type == HIP_R_16F && k == 2560) {
        // Exercise both an unaligned cache boundary and a wide prefill split.
        for (const auto [total, boundary] :
             {std::pair{136, 94}, std::pair{2048, 1025}}) {
          std::vector<float> bulk(std::size_t(m) * total), split(bulk.size());
          Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                               HIP_R_16F, m, total, k, &error),
                  error);
          Check(hipMemcpy(bulk.data(), c.data, bulk.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          for (const auto [off, n] : {std::pair{0, boundary},
                                      std::pair{boundary, total - boundary}}) {
            Require(
                warmed->Gemm(
                    a.data, static_cast<__half*>(b.data) + std::size_t(off) * k,
                    static_cast<float*>(c.data) + std::size_t(off) * m,
                    HIP_R_16F, m, n, k, &error),
                error);
          }
          Check(hipMemcpy(split.data(), c.data, split.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          Require(bulk == split, "projection changed across prefill chunks");
        }
      }
      // Vary shape order and use ragged sizes between the calibrated widths.
      for (int n : {4096, 2048, 9, 1024, 512, 129, 128, 513, 256}) {
        auto fresh = BlasLt::Create(nullptr, &error);
        Require(fresh != nullptr, error);
        std::vector<float> reference(static_cast<std::size_t>(m) * n);
        std::vector<float> actual(reference.size());
        for (int pass = 0; pass < 3; ++pass) {
          auto& blas = pass == 1 ? fresh : warmed;
          Require(blas->Gemm(a.data, b.data, static_cast<float*>(c.data), type,
                             m, n, k, &error),
                  error);
          Check(hipDeviceSynchronize());
          Check(hipMemcpy(actual.data(), c.data, actual.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          if (pass == 0) {
            reference = actual;
          } else {
            Require(std::memcmp(reference.data(), actual.data(),
                                actual.size() * sizeof(float)) == 0,
                    "projection changed across instances or plan reuse");
          }
        }
        for (int sample = 0; sample < 32; ++sample) {
          const int row = (sample * 79) % n;
          const int col = (sample * 127) % m;
          double expected = 0.0;
          for (int j = 0; j < k; ++j) {
            expected += static_cast<double>(
                            AsFloat(weights[std::size_t(col) * k + j], type)) *
                        AsFloat(input[std::size_t(row) * k + j], type);
          }
          const float result = actual[std::size_t(row) * m + col];
          Require(std::isfinite(result) && std::abs(result - expected) < 0.002,
                  "projection differs from the F64 reference");
        }
      }
    }
    // The BF16 indexer projections run the own kernel at every token count
    // with 512-token slabs aligned to absolute positions: a suffix chunk's
    // rows must take the same K chains a whole-prompt prefill used, at every
    // boundary alignment.
    {
      constexpr int m = 128, k = 2560;
      const auto bf16_bits = [](std::size_t count, std::uint32_t seed) {
        std::vector<std::uint16_t> v(count);
        for (auto& e : v) {
          seed ^= seed << 13;
          seed ^= seed >> 17;
          seed ^= seed << 5;
          const float f =
              static_cast<float>(static_cast<int>(seed % 8193) - 4096) /
              4096.0F;
          std::uint32_t bits = 0;
          std::memcpy(&bits, &f, 4);
          e = static_cast<std::uint16_t>(
              (bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
        }
        return v;
      };
      const auto weights = bf16_bits(static_cast<std::size_t>(m) * k, 456);
      const auto input = bf16_bits(static_cast<std::size_t>(max_n) * k, 789);
      Buffer a(weights.size() * sizeof(std::uint16_t));
      Buffer b(input.size() * sizeof(std::uint16_t));
      Buffer c(static_cast<std::size_t>(m) * max_n * sizeof(float));
      Buffer position(sizeof(std::uint32_t));
      Check(hipMemcpy(a.data, weights.data(),
                      weights.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(b.data, input.data(),
                      input.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
      for (const auto [total, boundary] :
           {std::pair{136, 94}, std::pair{1025, 600}, std::pair{2048, 1025}}) {
        std::vector<float> bulk(std::size_t(m) * total), split(bulk.size());
        const std::uint32_t zero = 0;
        Check(hipMemcpy(position.data, &zero, sizeof(zero),
                        hipMemcpyHostToDevice));
        warmed->SetTokenOffset(
            static_cast<const std::uint32_t*>(position.data));
        Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                             HIP_R_16BF, m, total, k, &error),
                error);
        Check(hipMemcpy(bulk.data(), c.data, bulk.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        for (const auto [off, n] :
             {std::pair{0, boundary}, std::pair{boundary, total - boundary}}) {
          const std::uint32_t start = static_cast<std::uint32_t>(off);
          Check(hipMemcpy(position.data, &start, sizeof(start),
                          hipMemcpyHostToDevice));
          Require(
              warmed->Gemm(
                  a.data,
                  static_cast<std::uint16_t*>(b.data) + std::size_t(off) * k,
                  static_cast<float*>(c.data) + std::size_t(off) * m,
                  HIP_R_16BF, m, n, k, &error),
              error);
        }
        Check(hipMemcpy(split.data(), c.data, split.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
        Require(bulk == split, "BF16 projection changed across prefill chunks");
      }
      // The own kernel serves every token count: the shape's plan must not
      // depend on n, or a short suffix would compute different arithmetic.
      for (int n : {7, 23, 623, 1024, 3023, 4096, 4223}) {
        Require(warmed->UsesOwnKernel(HIP_R_16BF, m, n, k, &error), error);
        Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                             HIP_R_16BF, m, n, k, &error),
                error);
        Check(hipDeviceSynchronize());
      }
      const std::uint32_t zero = 0;
      Check(
          hipMemcpy(position.data, &zero, sizeof(zero), hipMemcpyHostToDevice));
      warmed->SetTokenOffset(nullptr);
    }
    std::cout << "PASS: F16/BF16 prepared/lazy projection replay, F16 chunk "
                 "boundaries, ragged shapes and F64\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
