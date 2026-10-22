#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host, std::size_t extra_bytes = 0) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + extra_bytes + 4096),
           "hipMalloc");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

/// Packed expert weights plus their dequantized values [e][m][k].
struct Experts {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;
};

/// Random Q4_K blocks with the real 6-bit scale/min packing.
Experts MakeQ4K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 144);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 144;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qs = blk + 16;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code = (qs[base] >> shift) & 0xF;
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_K blocks: the Q4_K header, 32 high-bit bytes, 128 nibble bytes.
Experts MakeQ5K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 176);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 176;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half dmin = __float2half(Uniform(&seed, 0.01F) + 0.015F);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &dmin, 2);
    std::uint8_t sc[8];
    std::uint8_t mn[8];
    for (int i = 0; i < 8; ++i) {
      sc[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      mn[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
    }
    std::uint8_t* scales = blk + 4;
    for (int i = 0; i < 4; ++i) {
      scales[i] = static_cast<std::uint8_t>(sc[i] | ((sc[i + 4] >> 4) << 6));
      scales[i + 4] =
          static_cast<std::uint8_t>(mn[i] | ((mn[i + 4] >> 4) << 6));
      scales[i + 8] = static_cast<std::uint8_t>((sc[i + 4] & 0xF) |
                                                ((mn[i + 4] & 0xF) << 4));
    }
    std::uint8_t* qh = blk + 16;
    for (int i = 0; i < 32; ++i) {
      qh[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    std::uint8_t* qs = blk + 48;
    for (int i = 0; i < 128; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(dmin);
    for (int j = 0; j < 256; ++j) {
      const int sb32 = j / 32;
      const int base = (sb32 / 2) * 32 + (j % 32);
      const int shift = 4 * (sb32 & 1);
      const int code =
          ((qs[base] >> shift) & 0xF) | (((qh[j % 32] >> sb32) & 1) << 4);
      w.values[b * 256 + j] =
          df * static_cast<float>(sc[sb32]) * static_cast<float>(code) -
          mf * static_cast<float>(mn[sb32]);
    }
  }
  return w;
}

/// Random Q5_1 blocks: d, m, 32 high bits, 16 packed low nibbles.
Experts MakeQ5_1(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 24);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 24;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    const __half mn = __float2half(Uniform(&seed, 0.5F));
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &mn, 2);
    const std::uint32_t qh = NextRandom(&seed);
    std::memcpy(blk + 4, &qh, 4);
    std::uint8_t* qs = blk + 8;
    for (int i = 0; i < 16; ++i) {
      qs[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    const float mf = __half2float(mn);
    for (int j = 0; j < 32; ++j) {
      const int low = j < 16 ? (qs[j] & 0xF) : (qs[j - 16] >> 4);
      const int code = low | (static_cast<int>((qh >> j) & 1U) << 4);
      w.values[b * 32 + j] = df * static_cast<float>(code) + mf;
    }
  }
  return w;
}

/// Q5_1 blocks quantized from Gaussian weights as llama.cpp's
/// quantize_row_q5_1 does (m = block minimum, d = (max - min) / 31), so the
/// dequantized values centre on zero like a model's weights.
Experts MakeQuantizedQ5_1(std::size_t e, std::size_t m, std::size_t k,
                          std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 24);
  w.values.resize(e * m * k);
  constexpr float kTwoPi = 6.28318530717958647692F;
  for (std::size_t b = 0; b < blocks; ++b) {
    float x[32];
    for (float& v : x) {
      const float u1 =
          (static_cast<float>(NextRandom(&seed) >> 8) + 1.0F) / 16777217.0F;
      const float u2 = static_cast<float>(NextRandom(&seed) >> 8) / 16777216.0F;
      v = 0.02F * std::sqrt(-2.0F * std::log(u1)) * std::cos(kTwoPi * u2);
    }
    const float lo = *std::min_element(x, x + 32);
    const float hi = *std::max_element(x, x + 32);
    const float scale = (hi - lo) / 31.0F;
    const float inverse = scale != 0.0F ? 1.0F / scale : 0.0F;
    std::uint8_t* blk = w.packed.data() + b * 24;
    const __half d = __float2half(scale);
    const __half mn = __float2half(lo);
    std::memcpy(blk, &d, 2);
    std::memcpy(blk + 2, &mn, 2);
    std::uint32_t qh = 0;
    std::uint8_t* qs = blk + 8;
    int codes[32];
    for (int j = 0; j < 32; ++j) {
      codes[j] = std::min(31, static_cast<int>((x[j] - lo) * inverse + 0.5F));
      qh |= static_cast<std::uint32_t>(codes[j] >> 4) << j;
    }
    for (int j = 0; j < 16; ++j) {
      qs[j] = static_cast<std::uint8_t>((codes[j] & 0xF) |
                                        ((codes[j + 16] & 0xF) << 4));
    }
    std::memcpy(blk + 4, &qh, 4);
    const float df = __half2float(d);
    const float mf = __half2float(mn);
    for (int j = 0; j < 32; ++j) {
      w.values[b * 32 + j] = df * static_cast<float>(codes[j]) + mf;
    }
  }
  return w;
}

/// Random Q8_0 blocks: d and 32 signed codes.
Experts MakeQ8_0(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 34);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 34;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    const float df = __half2float(d);
    for (int j = 0; j < 32; ++j) {
      const auto q = static_cast<std::int8_t>(NextRandom(&seed) & 0xFF);
      blk[2 + j] = static_cast<std::uint8_t>(q);
      w.values[b * 32 + j] = df * static_cast<float>(q);
    }
  }
  return w;
}

struct Result {
  double vs_mmq;
  double vs_ref;
  double mmq_vs_ref;
  double scale;
  double mmq_sq_error = 0.0;
  double ref_sq = 0.0;
};


/// Random Q6_K blocks: 128 low-nibble bytes, 64 high-bit bytes, 16 int8

/// scales and one scale, matching ggml's dequantize_row_q6_K layout.

Experts MakeQ6K(std::size_t e, std::size_t m, std::size_t k,
                std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 210);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 210;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk + 208, &d, 2);
    std::uint8_t* ql = blk;
    std::uint8_t* qh = blk + 128;
    std::int8_t* sc = reinterpret_cast<std::int8_t*>(blk + 192);
    for (int i = 0; i < 128; ++i) {
      ql[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int i = 0; i < 64; ++i) {
      qh[i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int i = 0; i < 16; ++i) {
      sc[i] = static_cast<std::int8_t>(NextRandom(&seed) & 0xFF);
    }
    const float df = __half2float(d);
    for (int n = 0; n < 256; n += 128) {
      const std::uint8_t* qln = ql + n / 2;
      const std::uint8_t* qhn = qh + n / 4;
      const std::int8_t* scn = sc + n / 16;
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int q1 =
            static_cast<std::int8_t>((qln[l] & 0xF) | ((qhn[l] & 0x03) << 4)) -
            32;
        const int q2 = static_cast<std::int8_t>((qln[l + 32] & 0xF) |
                                                ((qhn[l] & 0x0c) << 2)) -
                       32;
        const int q3 =
            static_cast<std::int8_t>((qln[l] >> 4) | ((qhn[l] & 0x30) << 0)) -
            32;
        const int q4 = static_cast<std::int8_t>((qln[l + 32] >> 4) |
                                                ((qhn[l] & 0xc0) >> 2)) -
                       32;
        w.values[b * 256 + n + l] =
            df * static_cast<float>(scn[is]) * static_cast<float>(q1);
        w.values[b * 256 + n + l + 32] =
            df * static_cast<float>(scn[is + 2]) * static_cast<float>(q2);
        w.values[b * 256 + n + l + 64] =
            df * static_cast<float>(scn[is + 4]) * static_cast<float>(q3);
        w.values[b * 256 + n + l + 96] =
            df * static_cast<float>(scn[is + 6]) * static_cast<float>(q4);
      }
    }
  }
  return w;
}

// IQ3_S grid codes and the IQ4_NL codebook (vendored from llama.cpp,
// ggml-common.h) for the host-side reference dequantization.
constexpr std::uint32_t kIq3sGrid[512] = {
    0x01010101U, 0x01010103U, 0x01010105U, 0x0101010bU, 0x0101010fU,
    0x01010301U, 0x01010303U, 0x01010305U, 0x01010309U, 0x0101030dU,
    0x01010501U, 0x01010503U, 0x0101050bU, 0x01010707U, 0x01010901U,
    0x01010905U, 0x0101090bU, 0x0101090fU, 0x01010b03U, 0x01010b07U,
    0x01010d01U, 0x01010d05U, 0x01010f03U, 0x01010f09U, 0x01010f0fU,
    0x01030101U, 0x01030103U, 0x01030105U, 0x01030109U, 0x01030301U,
    0x01030303U, 0x0103030bU, 0x01030501U, 0x01030507U, 0x0103050fU,
    0x01030703U, 0x0103070bU, 0x01030909U, 0x01030d03U, 0x01030d0bU,
    0x01030f05U, 0x01050101U, 0x01050103U, 0x0105010bU, 0x0105010fU,
    0x01050301U, 0x01050307U, 0x0105030dU, 0x01050503U, 0x0105050bU,
    0x01050701U, 0x01050709U, 0x01050905U, 0x0105090bU, 0x0105090fU,
    0x01050b03U, 0x01050b07U, 0x01050f01U, 0x01050f07U, 0x01070107U,
    0x01070303U, 0x0107030bU, 0x01070501U, 0x01070505U, 0x01070703U,
    0x01070707U, 0x0107070dU, 0x01070909U, 0x01070b01U, 0x01070b05U,
    0x01070d0fU, 0x01070f03U, 0x01070f0bU, 0x01090101U, 0x01090307U,
    0x0109030fU, 0x01090503U, 0x01090509U, 0x01090705U, 0x01090901U,
    0x01090907U, 0x01090b03U, 0x01090f01U, 0x010b0105U, 0x010b0109U,
    0x010b0501U, 0x010b0505U, 0x010b050dU, 0x010b0707U, 0x010b0903U,
    0x010b090bU, 0x010b090fU, 0x010b0d0dU, 0x010b0f07U, 0x010d010dU,
    0x010d0303U, 0x010d0307U, 0x010d0703U, 0x010d0b05U, 0x010d0f03U,
    0x010f0101U, 0x010f0105U, 0x010f0109U, 0x010f0501U, 0x010f0505U,
    0x010f050dU, 0x010f0707U, 0x010f0b01U, 0x010f0b09U, 0x03010101U,
    0x03010103U, 0x03010105U, 0x03010109U, 0x03010301U, 0x03010303U,
    0x03010307U, 0x0301030bU, 0x0301030fU, 0x03010501U, 0x03010505U,
    0x03010703U, 0x03010709U, 0x0301070dU, 0x03010b09U, 0x03010b0dU,
    0x03010d03U, 0x03010f05U, 0x03030101U, 0x03030103U, 0x03030107U,
    0x0303010dU, 0x03030301U, 0x03030309U, 0x03030503U, 0x03030701U,
    0x03030707U, 0x03030903U, 0x03030b01U, 0x03030b05U, 0x03030f01U,
    0x03030f0dU, 0x03050101U, 0x03050305U, 0x0305030bU, 0x0305030fU,
    0x03050501U, 0x03050509U, 0x03050705U, 0x03050901U, 0x03050907U,
    0x03050b0bU, 0x03050d01U, 0x03050f05U, 0x03070103U, 0x03070109U,
    0x0307010fU, 0x03070301U, 0x03070307U, 0x03070503U, 0x0307050fU,
    0x03070701U, 0x03070709U, 0x03070903U, 0x03070d05U, 0x03070f01U,
    0x03090107U, 0x0309010bU, 0x03090305U, 0x03090309U, 0x03090703U,
    0x03090707U, 0x03090905U, 0x0309090dU, 0x03090b01U, 0x03090b09U,
    0x030b0103U, 0x030b0301U, 0x030b0307U, 0x030b0503U, 0x030b0701U,
    0x030b0705U, 0x030b0b03U, 0x030d0501U, 0x030d0509U, 0x030d050fU,
    0x030d0909U, 0x030d090dU, 0x030f0103U, 0x030f0107U, 0x030f0301U,
    0x030f0305U, 0x030f0503U, 0x030f070bU, 0x030f0903U, 0x030f0d05U,
    0x030f0f01U, 0x05010101U, 0x05010103U, 0x05010107U, 0x0501010bU,
    0x0501010fU, 0x05010301U, 0x05010305U, 0x05010309U, 0x0501030dU,
    0x05010503U, 0x05010507U, 0x0501050fU, 0x05010701U, 0x05010705U,
    0x05010903U, 0x05010907U, 0x0501090bU, 0x05010b01U, 0x05010b05U,
    0x05010d0fU, 0x05010f01U, 0x05010f07U, 0x05010f0bU, 0x05030101U,
    0x05030105U, 0x05030301U, 0x05030307U, 0x0503030fU, 0x05030505U,
    0x0503050bU, 0x05030703U, 0x05030709U, 0x05030905U, 0x05030b03U,
    0x05050103U, 0x05050109U, 0x0505010fU, 0x05050503U, 0x05050507U,
    0x05050701U, 0x0505070fU, 0x05050903U, 0x05050b07U, 0x05050b0fU,
    0x05050f03U, 0x05050f09U, 0x05070101U, 0x05070105U, 0x0507010bU,
    0x05070303U, 0x05070505U, 0x05070509U, 0x05070703U, 0x05070707U,
    0x05070905U, 0x05070b01U, 0x05070d0dU, 0x05090103U, 0x0509010fU,
    0x05090501U, 0x05090507U, 0x05090705U, 0x0509070bU, 0x05090903U,
    0x05090f05U, 0x05090f0bU, 0x050b0109U, 0x050b0303U, 0x050b0505U,
    0x050b070fU, 0x050b0901U, 0x050b0b07U, 0x050b0f01U, 0x050d0101U,
    0x050d0105U, 0x050d010fU, 0x050d0503U, 0x050d0b0bU, 0x050d0d03U,
    0x050f010bU, 0x050f0303U, 0x050f050dU, 0x050f0701U, 0x050f0907U,
    0x050f0b01U, 0x07010105U, 0x07010303U, 0x07010307U, 0x0701030bU,
    0x0701030fU, 0x07010505U, 0x07010703U, 0x07010707U, 0x0701070bU,
    0x07010905U, 0x07010909U, 0x0701090fU, 0x07010b03U, 0x07010d07U,
    0x07010f03U, 0x07030103U, 0x07030107U, 0x0703010bU, 0x07030309U,
    0x07030503U, 0x07030507U, 0x07030901U, 0x07030d01U, 0x07030f05U,
    0x07030f0dU, 0x07050101U, 0x07050305U, 0x07050501U, 0x07050705U,
    0x07050709U, 0x07050b01U, 0x07070103U, 0x07070301U, 0x07070309U,
    0x07070503U, 0x07070507U, 0x0707050fU, 0x07070701U, 0x07070903U,
    0x07070907U, 0x0707090fU, 0x07070b0bU, 0x07070f07U, 0x07090107U,
    0x07090303U, 0x0709030dU, 0x07090505U, 0x07090703U, 0x07090b05U,
    0x07090d01U, 0x07090d09U, 0x070b0103U, 0x070b0301U, 0x070b0305U,
    0x070b050bU, 0x070b0705U, 0x070b0909U, 0x070b0b0dU, 0x070b0f07U,
    0x070d030dU, 0x070d0903U, 0x070f0103U, 0x070f0107U, 0x070f0501U,
    0x070f0505U, 0x070f070bU, 0x09010101U, 0x09010109U, 0x09010305U,
    0x09010501U, 0x09010509U, 0x0901050fU, 0x09010705U, 0x09010903U,
    0x09010b01U, 0x09010f01U, 0x09030105U, 0x0903010fU, 0x09030303U,
    0x09030307U, 0x09030505U, 0x09030701U, 0x0903070bU, 0x09030907U,
    0x09030b03U, 0x09030b0bU, 0x09050103U, 0x09050107U, 0x09050301U,
    0x0905030bU, 0x09050503U, 0x09050707U, 0x09050901U, 0x09050b0fU,
    0x09050d05U, 0x09050f01U, 0x09070109U, 0x09070303U, 0x09070307U,
    0x09070501U, 0x09070505U, 0x09070703U, 0x0907070bU, 0x09090101U,
    0x09090105U, 0x09090509U, 0x0909070fU, 0x09090901U, 0x09090f03U,
    0x090b010bU, 0x090b010fU, 0x090b0503U, 0x090b0d05U, 0x090d0307U,
    0x090d0709U, 0x090d0d01U, 0x090f0301U, 0x090f030bU, 0x090f0701U,
    0x090f0907U, 0x090f0b03U, 0x0b010105U, 0x0b010301U, 0x0b010309U,
    0x0b010505U, 0x0b010901U, 0x0b010909U, 0x0b01090fU, 0x0b010b05U,
    0x0b010d0dU, 0x0b010f09U, 0x0b030103U, 0x0b030107U, 0x0b03010bU,
    0x0b030305U, 0x0b030503U, 0x0b030705U, 0x0b030f05U, 0x0b050101U,
    0x0b050303U, 0x0b050507U, 0x0b050701U, 0x0b05070dU, 0x0b050b07U,
    0x0b070105U, 0x0b07010fU, 0x0b070301U, 0x0b07050fU, 0x0b070909U,
    0x0b070b03U, 0x0b070d0bU, 0x0b070f07U, 0x0b090103U, 0x0b090109U,
    0x0b090501U, 0x0b090705U, 0x0b09090dU, 0x0b0b0305U, 0x0b0b050dU,
    0x0b0b0b03U, 0x0b0b0b07U, 0x0b0d0905U, 0x0b0f0105U, 0x0b0f0109U,
    0x0b0f0505U, 0x0d010303U, 0x0d010307U, 0x0d01030bU, 0x0d010703U,
    0x0d010707U, 0x0d010d01U, 0x0d030101U, 0x0d030501U, 0x0d03050fU,
    0x0d030d09U, 0x0d050305U, 0x0d050709U, 0x0d050905U, 0x0d050b0bU,
    0x0d050d05U, 0x0d050f01U, 0x0d070101U, 0x0d070309U, 0x0d070503U,
    0x0d070901U, 0x0d09050bU, 0x0d090907U, 0x0d090d05U, 0x0d0b0101U,
    0x0d0b0107U, 0x0d0b0709U, 0x0d0b0d01U, 0x0d0d010bU, 0x0d0d0901U,
    0x0d0f0303U, 0x0d0f0307U, 0x0f010101U, 0x0f010109U, 0x0f01010fU,
    0x0f010501U, 0x0f010505U, 0x0f01070dU, 0x0f010901U, 0x0f010b09U,
    0x0f010d05U, 0x0f030105U, 0x0f030303U, 0x0f030509U, 0x0f030907U,
    0x0f03090bU, 0x0f050103U, 0x0f050109U, 0x0f050301U, 0x0f05030dU,
    0x0f050503U, 0x0f050701U, 0x0f050b03U, 0x0f070105U, 0x0f070705U,
    0x0f07070bU, 0x0f070b07U, 0x0f090103U, 0x0f09010bU, 0x0f090307U,
    0x0f090501U, 0x0f090b01U, 0x0f0b0505U, 0x0f0b0905U, 0x0f0d0105U,
    0x0f0d0703U, 0x0f0f0101U,
};

constexpr float kValuesIq4Nl[16] = {-127.F, -104.F, -83.F, -65.F, -49.F, -35.F,
                                    -22.F,  -10.F,  1.F,   13.F,  25.F,  38.F,
                                    53.F,   69.F,   89.F,  113.F};

/// Random IQ4_NL blocks: d and 16 codebook-index bytes.
Experts MakeIQ4NL(std::size_t e, std::size_t m, std::size_t k,
                  std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 32);
  w.packed.resize(blocks * 18);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 18;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    const float df = __half2float(d);
    for (int i = 0; i < 16; ++i) {
      blk[2 + i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int j = 0; j < 32; ++j) {
      const int code = j < 16 ? (blk[2 + j] & 0xF) : (blk[2 + j - 16] >> 4);
      w.values[b * 32 + j] = df * kValuesIq4Nl[code];
    }
  }
  return w;
}

/// Random IQ3_S blocks: d, 64 grid-index bytes, 8 high-bit bytes, 32 sign

/// bytes and 4 scale bytes over the 256-value superblock.
Experts MakeIQ3S(std::size_t e, std::size_t m, std::size_t k,
                 std::uint32_t seed) {
  Experts w;
  const std::size_t blocks = e * m * (k / 256);
  w.packed.resize(blocks * 110);
  w.values.resize(e * m * k);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* blk = w.packed.data() + b * 110;
    const __half d = __float2half(Uniform(&seed, 0.02F) + 0.03F);
    std::memcpy(blk, &d, 2);
    for (int i = 0; i < 64; ++i) {
      blk[2 + i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int i = 0; i < 8; ++i) {
      blk[66 + i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int i = 0; i < 32; ++i) {
      blk[74 + i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFF);
    }
    for (int i = 0; i < 4; ++i) {
      blk[106 + i] = static_cast<std::uint8_t>(NextRandom(&seed) & 0x0F);
    }
    const float df = __half2float(d);
    for (int c = 0; c < 8; ++c) {
      const float ls = static_cast<float>(
          1 + 2 * ((blk[106 + c / 2] >> (4 * (c & 1))) & 0xF));
      for (int l = 0; l < 8; ++l) {
        const std::uint32_t idx =
            blk[2 + 8 * c + l] |
            ((static_cast<std::uint32_t>(blk[66 + c] >> l) & 1U) << 8);
        const std::uint32_t grid = kIq3sGrid[idx];
        const std::uint8_t sb = blk[74 + 4 * c + l / 2];
        const std::uint32_t bits = (l & 1) != 0 ? sb >> 4 : sb & 0xF;
        for (int j = 0; j < 4; ++j) {
          const float g = static_cast<float>((grid >> (8 * j)) & 0xFFU);
          const float v = ((bits >> j) & 1U) != 0U ? -g : g;
          w.values[b * 256 + 32 * c + 4 * l + j] = df * ls * v;
        }
      }
    }
  }
  return w;
}

const char* TypeName(q::WeightType type) {
  return type == q::WeightType::kQ4_K   ? "Q4_K"
         : type == q::WeightType::kQ5_K ? "Q5_K"
         : type == q::WeightType::kQ5_1 ? "Q5_1"
         : type == q::WeightType::kIQ3_S ? "IQ3_S"
         : type == q::WeightType::kIQ4_NL ? "IQ4_NL"
                                          : "Q8_0";

}

/// Runs the MMQ tier and the F16 route over the same routing and reports
/// the worst absolute errors of the F16 route against the tier and against
/// an F64 reference over the dequantized weights (the tier's own F64 gap is
/// its activation quantization).
Result Run(q::WeightType type, std::size_t n_tokens, std::size_t used,
           std::size_t experts, std::size_t m, std::size_t k,
           std::uint32_t seed, std::uint32_t tile_rows = 48,
           float input_scale = 2.0F) {
  const Experts w =
      type == q::WeightType::kQ4_K     ? MakeQ4K(experts, m, k, seed)
      : type == q::WeightType::kQ5_K   ? MakeQ5K(experts, m, k, seed)
      : type == q::WeightType::kQ5_1   ? MakeQ5_1(experts, m, k, seed)
      : type == q::WeightType::kIQ3_S  ? MakeIQ3S(experts, m, k, seed)
      : type == q::WeightType::kIQ4_NL ? MakeIQ4NL(experts, m, k, seed)
                                       : MakeQ8_0(experts, m, k, seed);
  const std::size_t slots = n_tokens * used;
  // Skewed top-k without replacement: the low experts see most rows.
  std::vector<std::int32_t> ids(slots);
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  for (std::size_t t = 0; t < n_tokens; ++t) {
    std::vector<std::int32_t> chosen;
    while (chosen.size() < used) {
      const std::uint32_t r = NextRandom(&state);
      const auto e =
          static_cast<std::int32_t>((r % 3 == 0 ? r / 3 % 4 : r / 3) % experts);
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
        chosen.push_back(e);
      }
    }
    std::copy(chosen.begin(), chosen.end(), ids.begin() + t * used);
  }
  std::vector<float> x(n_tokens * k);
  for (float& v : x) {
    v = Uniform(&state, input_scale);
  }
  std::vector<std::uint32_t> counts(experts, 0);
  for (std::int32_t e : ids) {
    ++counts[e];
  }
  std::uint32_t max_bucket = 0;
  for (std::uint32_t c : counts) {
    max_bucket = std::max(max_bucket, (c + 15u) / 16u * 16u);
  }

  std::uint8_t* d_w = Upload(w.packed);
  float* d_x = Upload(x);
  std::int32_t* d_ids = Upload(ids);
  std::uint32_t* d_counts = Upload(counts);
  const std::size_t compact = q::RoutedCompactRows(slots, experts);
  std::vector<std::int32_t> zeros(compact + experts + 1, 0);
  std::int32_t* d_bounds = Upload(std::vector<std::int32_t>(experts + 1, 0));
  std::int32_t* d_cursors = Upload(std::vector<std::int32_t>(experts, 0));
  std::int32_t* d_rows_token = Upload(zeros);
  std::int32_t* d_rows_slot = Upload(zeros);
  std::vector<float> zero_out(slots * m, 0.0F);
  float* d_mmq = Upload(zero_out);

  const auto mmq_raw = type == q::WeightType::kQ4_K     ? qfn_mmq_q4_K_moe_raw
                       : type == q::WeightType::kQ5_K   ? qfn_mmq_q5_K_moe_raw
                       : type == q::WeightType::kQ5_1   ? qfn_mmq_q5_1_moe_raw
                       : type == q::WeightType::kIQ3_S  ? qfn_mmq_iq3_s_moe_raw
                       : type == q::WeightType::kIQ4_NL ? qfn_mmq_iq4_nl_moe_raw
                                                        : qfn_mmq_q8_0_moe_raw;
  const int rc =
      mmq_raw(d_w, d_x, d_ids, d_mmq, static_cast<int>(m), static_cast<int>(k),
              static_cast<int>(n_tokens), static_cast<int>(experts),
              static_cast<int>(used), nullptr);
  if (rc != 0) {
    throw std::runtime_error("MMQ routed GEMM failed");
  }
  // The F16 route over the same assignments: F16 activation rows and the
  // host-built tile map. `used == 1` (the down projection's view) gathers
  // slot rows; the gate/up view gathers tokens.
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rows_token,
                   d_rows_slot, static_cast<std::uint32_t>(n_tokens),
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(experts), nullptr);
  const auto bounds = Download(d_bounds, experts + 1);
  const auto token_rows = Download(d_rows_token, compact);
  const auto slot_rows = Download(d_rows_slot, compact);
  std::vector<bool> seen(slots, false);
  for (std::size_t e = 0; e < experts; ++e) {
    for (auto row = bounds[e]; row < bounds[e + 1]; ++row) {
      const auto slot = slot_rows[row];
      if (static_cast<std::uint32_t>(row - bounds[e]) >= counts[e]) {
        if (slot != -1 || token_rows[row] != -1)
          throw std::runtime_error(
              "routed padding retained a stale assignment");
      } else {
        if (slot < 0 || static_cast<std::size_t>(slot) >= slots || seen[slot] ||
            ids[slot] != static_cast<std::int32_t>(e) ||
            token_rows[row] != slot / static_cast<std::int32_t>(used))
          throw std::runtime_error("routed compaction lost an assignment");
        seen[slot] = true;
      }
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::runtime_error("routed compaction omitted a slot");
  std::vector<__half> x_half(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    x_half[i] = __float2half(x[i]);
  }
  std::vector<std::int32_t> tiles;
  for (std::size_t e = 0; e < experts; ++e) {
    const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
    for (std::uint32_t j = 0; j < (padded + tile_rows - 1) / tile_rows; ++j) {
      tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
    }
  }
  __half* d_x_half = nullptr;
  CheckHip(hipMalloc(&d_x_half, x_half.size() * sizeof(__half)),
           "exact F16 input allocation");
  CheckHip(hipMemcpy(d_x_half, x_half.data(), x_half.size() * sizeof(__half),
                     hipMemcpyHostToDevice),
           "F16 input upload");
  std::int32_t* d_tiles = Upload(tiles);
  float* d_f16 = Upload(zero_out);
  if (!q::RoutedF16Gemm(d_w, type, d_x_half, d_tiles,
                        static_cast<std::uint32_t>(tiles.size()), tile_rows,
                        d_bounds, d_rows_token, d_rows_slot, nullptr, d_f16,
                        nullptr, m, k, nullptr)) {
    throw std::runtime_error("routed F16 GEMM rejected the shape");
  }
  // The F16-output route (the up and down projections' rows) must match
  // the F32 output to F16 rounding.
  constexpr std::size_t kHalfGuard = 16;
  const __half half_guard = __float2half(123.0F);
  __half* d_f16_half =
      Upload(std::vector<__half>(slots * m + kHalfGuard, half_guard));
  if (!q::RoutedF16Gemm(d_w, type, d_x_half, d_tiles,
                        static_cast<std::uint32_t>(tiles.size()), tile_rows,
                        d_bounds, d_rows_token, d_rows_slot, nullptr, nullptr,
                        d_f16_half, m, k, nullptr)) {
    throw std::runtime_error("routed F16 GEMM (F16 out) rejected the shape");
  }
  CheckHip(hipDeviceSynchronize(), "routed GEMMs");
  const auto mmq = Download(d_mmq, slots * m);
  const auto f16 = Download(d_f16, slots * m);
  const auto f16_half = Download(d_f16_half, slots * m);
  if (tile_rows == 64) {
    std::vector<std::int32_t> narrow_tiles;
    for (std::size_t e = 0; e < experts; ++e) {
      const std::uint32_t padded = (counts[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 47u) / 48u; ++j) {
        narrow_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
      }
    }
    auto* d_narrow_tiles = Upload(narrow_tiles);
    if (!q::RoutedF16Gemm(d_w, type, d_x_half, d_narrow_tiles,
                          static_cast<std::uint32_t>(narrow_tiles.size()), 48,
                          d_bounds, d_rows_token, d_rows_slot, nullptr, d_f16,
                          nullptr, m, k, nullptr)) {
      throw std::runtime_error("routed reference tile rejected");
    }
    CheckHip(hipDeviceSynchronize(), "routed tile comparison");
    const auto narrow = Download(d_f16, slots * m);
    if (std::memcmp(narrow.data(), f16.data(), f16.size() * sizeof(float)) !=
        0) {
      throw std::runtime_error("routed tile widths change output bits");
    }
    CheckHip(hipFree(d_narrow_tiles), "free reference tiles");
  }
  for (std::size_t i = 0; i < slots * m; ++i) {
    const __half expected = __float2half(f16[i]);
    if (std::memcmp(&expected, &f16_half[i], sizeof(__half)) != 0) {
      throw std::runtime_error("routed F16 output changed rounding");
    }
  }
  const auto half_tail = Download(d_f16_half + slots * m, kHalfGuard);
  for (__half value : half_tail) {
    if (std::memcmp(&value, &half_guard, sizeof(__half)) != 0)
      throw std::runtime_error("routed F16 output overwrote its guard");
  }

  // Prefill pairs gate/up for buckets the separate 16- and 48-row tiles take.
  if ((tile_rows == 16 || tile_rows == 48) &&
      (type == q::WeightType::kQ4_K || type == q::WeightType::kQ5_K)) {
    // Different gate/up weights, with the up scale small enough that these
    // deliberately large synthetic weights do not overflow the F16 output.
    auto up = w.packed;
    const std::size_t block_bytes = type == q::WeightType::kQ4_K ? 144 : 176;
    for (std::size_t b = 0; b < up.size(); b += block_bytes) {
      for (std::size_t offset : {0U, 2U}) {
        __half scale;
        std::memcpy(&scale, up.data() + b + offset, sizeof(scale));
        scale = __float2half(__half2float(scale) / 1024.0F);
        std::memcpy(up.data() + b + offset, &scale, sizeof(scale));
      }
    }
    auto* d_up = Upload(up);
    __half* d_pair = nullptr;
    CheckHip(hipMalloc(&d_pair, (slots * m + kHalfGuard) * sizeof(__half)),
             "paired output allocation");
    const std::vector<__half> pair_initial(slots * m + kHalfGuard, half_guard);
    CheckHip(
        hipMemcpy(d_pair, pair_initial.data(),
                  pair_initial.size() * sizeof(__half), hipMemcpyHostToDevice),
        "paired output upload");
    if (!q::RoutedF16Gemm(d_up, type, d_x_half, d_tiles,
                          static_cast<std::uint32_t>(tiles.size()), tile_rows,
                          d_bounds, d_rows_token, d_rows_slot, d_f16, nullptr,
                          d_f16_half, m, k, nullptr)) {
      throw std::runtime_error("separate routed SwiGLU launch failed");
    }
    const auto separate = Download(d_f16_half, slots * m);
    {
      constexpr std::uint32_t pair_rows = 128;
      std::vector<std::int32_t> gate_tiles;
      for (std::size_t e = 0; e < experts; ++e) {
        for (std::uint32_t j = 0; j < (counts[e] + pair_rows - 1) / pair_rows;
             ++j) {
          gate_tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
        }
      }
      auto* d_gate_tiles = Upload(gate_tiles);
      auto launch_pair = [&] {
        if (!q::RoutedGatedF16Gemm(
                d_w, d_up, type, d_x_half, d_gate_tiles,
                static_cast<std::uint32_t>(gate_tiles.size()), d_bounds,
                d_rows_token, d_rows_slot, d_pair, m, k, nullptr)) {
          throw std::runtime_error("paired routed SwiGLU launch failed");
        }
      };
      // The paired launch must write the whole output itself; stale results
      // can hide an incomplete launch grid.
      CheckHip(hipMemcpy(d_pair, pair_initial.data(),
                         pair_initial.size() * sizeof(__half),
                         hipMemcpyHostToDevice),
               "paired output reset");
      launch_pair();
      const auto paired = Download(d_pair, slots * m);
      for (std::size_t i = 0; i < paired.size(); ++i) {
        const float a = __half2float(separate[i]);
        const float b = __half2float(paired[i]);
        // The independent epilogue permits either sign of zero; every
        // numerical value must match, including subnormal F16 values.
        if (!std::isfinite(a) || !std::isfinite(b) || a != b) {
          throw std::runtime_error(
              "paired routed SwiGLU differs at slot " + std::to_string(i / m) +
              " row " + std::to_string(i % m) + ": " + std::to_string(a) +
              " vs " + std::to_string(b));
        }
      }
      const auto guard = Download(d_pair + slots * m, kHalfGuard);
      if (std::memcmp(guard.data(), pair_initial.data() + slots * m,
                      kHalfGuard * sizeof(__half)) != 0)
        throw std::runtime_error("paired output overwrote its guard");
      launch_pair();
      const auto replay = Download(d_pair, slots * m);
      if (std::memcmp(paired.data(), replay.data(),
                      paired.size() * sizeof(__half)) != 0) {
        throw std::runtime_error("paired routed SwiGLU replay changed bits");
      }
      CheckHip(hipFree(d_gate_tiles), "paired tile map free");
    }
    CheckHip(hipFree(d_up), "paired up free");
    CheckHip(hipFree(d_pair), "paired output free");
  }

  Result r{0.0, 0.0, 0.0, 0.0};
  for (std::size_t s = 0; s < slots; ++s) {
    const std::size_t t = s / used;
    const std::int32_t e = ids[s];
    for (std::size_t row = 0; row < m; ++row) {
      double ref = 0.0;
      const float* wr = w.values.data() + (e * m + row) * k;
      const float* xr = x.data() + t * k;
      for (std::size_t i = 0; i < k; ++i) {
        ref += static_cast<double>(wr[i]) * xr[i];
      }
      const float f = f16[s * m + row];
      if (!std::isfinite(f)) {
        throw std::runtime_error("routed F16 output is not finite");
      }
      const float v = mmq[s * m + row];
      r.vs_mmq = std::max(r.vs_mmq, std::abs(static_cast<double>(v - f)));
      r.vs_ref = std::max(r.vs_ref, std::abs(ref - f));
      r.mmq_vs_ref = std::max(r.mmq_vs_ref, std::abs(ref - v));
      r.scale = std::max(r.scale, std::abs(ref));
      r.mmq_sq_error += (ref - v) * (ref - v);
      r.ref_sq += ref * ref;
    }
  }

  std::cout << TypeName(type) << " routed n=" << n_tokens << " used=" << used
            << " E=" << experts << " m=" << m << " k=" << k
            << ": worst |F16 - MMQ| " << r.vs_mmq << ", worst |F16 - F64| "
            << r.vs_ref << ", worst |MMQ - F64| " << r.mmq_vs_ref
            << ", MMQ NMSE " << r.mmq_sq_error / r.ref_sq << " (scale "
            << r.scale << ", widest bucket " << max_bucket << ")\n";

  for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x),
                  static_cast<void*>(d_ids), static_cast<void*>(d_counts),
                  static_cast<void*>(d_bounds), static_cast<void*>(d_cursors),
                  static_cast<void*>(d_rows_token),
                  static_cast<void*>(d_rows_slot), static_cast<void*>(d_mmq),
                  static_cast<void*>(d_x_half), static_cast<void*>(d_tiles),
                  static_cast<void*>(d_f16), static_cast<void*>(d_f16_half)}) {
    (void)hipFree(p);
  }
  return r;
}

bool Ok(const Result& r) {
  // The tier quantizes the activations to 8 bits per 32-wide block, the F16
  // route rounds them (and the dequantized weights) to F16, so the two
  // agree to the tier's quantization; the F16 route sits well inside it.
  return r.vs_mmq < 1e-2 * r.scale && r.vs_ref < 2e-3 * r.scale &&
         r.mmq_vs_ref < 1e-2 * r.scale;
}

/// The activations as quantize_q8_1 hands them to the dot products: each
/// 32-wide block becomes its F16 scale times the rounded codes.
std::vector<float> Q8_1Activations(const std::vector<float>& x) {
  std::vector<float> out(x.size());
  for (std::size_t b = 0; b < x.size(); b += 32) {
    float amax = 0.0F;
    for (std::size_t i = b; i < b + 32; ++i)
      amax = std::max(amax, std::abs(x[i]));
    const float d = amax / 127.0F;
    const float dh = __half2float(__float2half(d));
    for (std::size_t i = b; i < b + 32; ++i)
      out[i] = amax == 0.0F ? 0.0F : dh * std::round(x[i] / d);
  }
  return out;
}

/// Reports the vector tier (decode and verify widths) against F64 dots over
/// the dequantized weights, for the gate/up (640 x 2560) and down
/// (2560 x 640) shapes: with the exact activations (the tier's error is its
/// activation quantization, so this reports rather than gates) and with the
/// q8_1 activations it consumes (ggml's q8_1 contract; only F16 scale and sum
/// rounding remain). "Q5_1 quantized" blocks centre like a model's weights.
void ReportVectorAccuracy() {
  constexpr int experts = 8;
  struct Case {
    q::WeightType type;
    bool quantized;
  };
  for (const bool down : {false, true}) {
    const int rows = down ? 2560 : 640;
    const int cols = down ? 640 : 2560;
    const int used = down ? 1 : 2;
    const std::vector<Case> cases =
        down ? std::vector<Case>{{q::WeightType::kQ5_1, false},
                                 {q::WeightType::kQ5_1, true},
                                 {q::WeightType::kQ8_0, false}}
             : std::vector<Case>{{q::WeightType::kQ4_K, false},
                                 {q::WeightType::kQ5_K, false},
                                 {q::WeightType::kQ5_1, false},
                                 {q::WeightType::kQ5_1, true},
                                 {q::WeightType::kQ8_0, false}};
    for (const auto [type, quantized] : cases) {
      for (const int n : {1, 4, 8}) {
        const std::uint32_t seed =
            0x7EC70000U + 256U * static_cast<int>(type) + (down ? 64U : 0U) + n;
        const Experts w =
            quantized ? MakeQuantizedQ5_1(experts, rows, cols, seed)
            : type == q::WeightType::kQ4_K ? MakeQ4K(experts, rows, cols, seed)
            : type == q::WeightType::kQ5_K ? MakeQ5K(experts, rows, cols, seed)
            : type == q::WeightType::kQ5_1
                ? MakeQ5_1(experts, rows, cols, seed)
                : MakeQ8_0(experts, rows, cols, seed);
        std::uint32_t state = seed ^ 0xA5A5A5A5U;
        std::vector<float> x(static_cast<std::size_t>(n) * cols);
        for (float& v : x)
          v = Uniform(&state, 2.0F);
        std::vector<std::int32_t> ids(static_cast<std::size_t>(n) * used);
        for (std::size_t i = 0; i < ids.size(); ++i)
          ids[i] = static_cast<std::int32_t>((3 * i + 1) % experts);
        auto* dw = Upload(w.packed);
        auto* dx = Upload(x);
        auto* di = Upload(ids);
        auto* dout = Upload(std::vector<float>(ids.size() * rows, 0.0F));
        if (qfn_mmq_moe_vec(static_cast<int>(type), dw, dx, di, dout, rows,
                            cols, n, experts, used, nullptr))
          throw std::runtime_error("vector accuracy launch failed");
        const auto out = Download(dout, ids.size() * rows);
        const auto xq = Q8_1Activations(x);
        double worst = 0.0, error = 0.0, norm = 0.0;
        double worst_q = 0.0, error_q = 0.0;
        for (std::size_t s = 0; s < ids.size(); ++s) {
          const float* xr = x.data() + (s / used) * cols;
          const float* xqr = xq.data() + (s / used) * cols;
          for (int row = 0; row < rows; ++row) {
            const float* wr =
                w.values.data() +
                (static_cast<std::size_t>(ids[s]) * rows + row) * cols;
            double ref = 0.0, ref_q = 0.0;
            for (int i = 0; i < cols; ++i) {
              ref += static_cast<double>(wr[i]) * xr[i];
              ref_q += static_cast<double>(wr[i]) * xqr[i];
            }
            const double e = ref - out[s * rows + row];
            const double eq = ref_q - out[s * rows + row];
            worst = std::max(worst, std::abs(e));
            worst_q = std::max(worst_q, std::abs(eq));
            error += e * e;
            error_q += eq * eq;
            norm += ref * ref;
          }
        }
        std::cout << TypeName(type) << (quantized ? " quantized" : "")
                  << " vector n=" << n << " m=" << rows << " k=" << cols
                  << ": worst |vector - F64| " << worst << ", NMSE "
                  << error / norm << "; against the q8_1 activations: worst "
                  << worst_q << ", NMSE " << error_q / norm << '\n';
        for (void* p : {static_cast<void*>(dw), static_cast<void*>(dx),
                        static_cast<void*>(di), static_cast<void*>(dout)})
          (void)hipFree(p);
      }
    }
  }
}

void CheckVectorGrouping(bool down = false) {
  constexpr int experts = 8;
  const int cols = down ? 640 : 2560;
  const int tokens = down ? 640 : 64;
  const int used = down ? 1 : 3;
  constexpr std::size_t guard = 16;
  constexpr float poison = -1234567.0F;
  const std::vector<q::WeightType> formats =
      down ? std::vector{q::WeightType::kQ5_1, q::WeightType::kQ8_0,
                         q::WeightType::kQ6_K}
           : std::vector{q::WeightType::kQ4_K, q::WeightType::kQ5_K,
                         q::WeightType::kQ5_1, q::WeightType::kQ8_0,
                         q::WeightType::kQ6_K};
  std::vector<float> x(tokens * cols);
  std::uint32_t seed = 37;
  // Mix ordinary and small activations: saturated SwiGLU alone misses
  // rounding changes that appear in real predictor inputs.
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = Uniform(&seed, !down && (i / cols) % 2 ? 1.0F / 128.0F : 2.0F);
  std::vector<std::int32_t> ids(tokens * used);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      ids[t * used + j] =
          j == 1 || (down && t % 11 == 1) ? -1 : (t + j) % experts;
  if (!down) {
    // Exercise both repeated experts across tokens and duplicate slots
    // within one token; each output must retain its original slot.
    ids[2] = ids[0];
    ids[6 * used + 2] = ids[6 * used];
  }
  auto* dx = Upload(x);
  auto* di = Upload(ids);
  for (int rows : {down ? 2560 : 64, down ? 2561 : 65}) {
    std::vector<Experts> weights, paired_weights;
    if (down) {
      // Q6_K has no down view: its 256-element blocks do not divide the
      // 640-element expert FFN, and the loader rejects Q6_K expert tensors.
      weights = {MakeQ5_1(experts, rows, cols, 17),
                 MakeQ8_0(experts, rows, cols, 23)};
      paired_weights = {MakeQ5_1(experts, rows, cols, 79),
                        MakeQ8_0(experts, rows, cols, 83)};
    } else {
      weights = {
          MakeQ4K(experts, rows, cols, 11), MakeQ5K(experts, rows, cols, 13),
          MakeQ5_1(experts, rows, cols, 17), MakeQ8_0(experts, rows, cols, 23),
          MakeQ6K(experts, rows, cols, 29)};
      paired_weights = {
          MakeQ4K(experts, rows, cols, 71), MakeQ5K(experts, rows, cols, 73),
          MakeQ5_1(experts, rows, cols, 79), MakeQ8_0(experts, rows, cols, 83),
          MakeQ6K(experts, rows, cols, 89)};
    }
    const std::size_t count = tokens * used * rows;
    const std::vector<float> dirty(count + 2 * guard, poison);
    auto* scalar = Upload(dirty);
    auto* batch = Upload(dirty);
    auto* paired_scalar = Upload(dirty);
    auto* paired_batch = Upload(dirty);
    const auto check_output = [&](const float* device, int n) {
      const auto output = Download(device, dirty.size());
      for (std::size_t i = 0; i < output.size(); ++i) {
        if (i < guard || i >= guard + n * used * rows) {
          if (output[i] != poison)
            throw std::runtime_error("routed vector output guard changed");
          continue;
        }
        const std::size_t slot = (i - guard) / rows;
        const bool zero =
            ids[slot] < 0 || (ids[slot] == 0 && (i - guard) % rows == 0);
        if (!std::isfinite(output[i]) || output[i] == poison ||
            (zero && output[i] != 0.0F))
          throw std::runtime_error("routed vector did not write a valid row");
      }
      return output;
    };
    for (std::size_t f = 0; f < weights.size(); ++f) {
      // Every format starts with an F16 scale — except Q6_K, whose scale
      // sits at the end of the block behind ql/qh/scales. A nonfinite
      // projection must still produce zero, just as an inactive expert does.
      const __half infinite_scale = __float2half(INFINITY);
      const std::size_t scale_offset =
          formats[f] == q::WeightType::kQ6_K ? 208 : 0;
      std::memcpy(weights[f].packed.data() + scale_offset, &infinite_scale,
                  sizeof(infinite_scale));
      std::memcpy(paired_weights[f].packed.data() + scale_offset,
                  &infinite_scale, sizeof(infinite_scale));
      auto* wb = Upload(paired_weights[f].packed);
      auto* w = Upload(weights[f].packed);
      CheckHip(hipMemcpy(scalar, dirty.data(), dirty.size() * sizeof(float),
                         hipMemcpyHostToDevice),
               "poison scalar output");
      for (int t = 0; t < tokens; ++t)
        if (qfn_mmq_moe_vec(static_cast<int>(formats[f]), w, dx + t * cols,
                            di + t * used, scalar + guard + t * used * rows,
                            rows, cols, 1, experts, used, nullptr))
          throw std::runtime_error("scalar routed vector launch failed");
      const auto reference = check_output(scalar, tokens);
      CheckHip(hipMemcpy(paired_scalar, dirty.data(),
                         dirty.size() * sizeof(float), hipMemcpyHostToDevice),
               "poison paired reference");
      if (qfn_mmq_moe_vec(static_cast<int>(formats[f]), wb, dx, di,
                          paired_scalar + guard, rows, cols, tokens, experts,
                          used, nullptr))
        throw std::runtime_error("paired reference projection failed");
      const auto reference_b = check_output(paired_scalar, tokens);
      const auto widths =
          down ? std::vector{2,  3,  4,  7,  8,   10,  20, 21,
                             30, 40, 79, 80, 160, 319, 640}
               : std::vector{2, 3, 4, 7, 8, 9, 16, 31, 32, 33, 64};
      for (int n : widths) {
        CheckHip(hipMemcpy(batch, dirty.data(), dirty.size() * sizeof(float),
                           hipMemcpyHostToDevice),
                 "poison batched output");
        if (qfn_mmq_moe_vec(static_cast<int>(formats[f]), w, dx, di,
                            batch + guard, rows, cols, n, experts, used,
                            nullptr))
          throw std::runtime_error("batched routed vector launch failed");
        const auto output = check_output(batch, n);
        if (std::memcmp(reference.data() + guard, output.data() + guard,
                        n * used * rows * sizeof(float)) != 0) {
          for (int t = 0; t < n; ++t) {
            for (int j = 0; j < used; ++j) {
              for (int rr = 0; rr < rows; ++rr) {
                const std::size_t o =
                    (static_cast<std::size_t>(t) * used + j) * rows + rr;
                if (reference[guard + o] != output[guard + o]) {
                  throw std::runtime_error(
                      "routed vector grouping changed format " +
                      std::to_string(f) + " width " + std::to_string(n) +
                      " at token " + std::to_string(t) + " slot " +
                      std::to_string(j) + " row " + std::to_string(rr));
                }
              }
            }
          }
          throw std::runtime_error("routed vector grouping changed format " +
                                   std::to_string(f) + " width " +
                                   std::to_string(n));
        }
      }
      for (int n : {1, 4, 7}) {
        CheckHip(hipMemcpy(batch, dirty.data(), dirty.size() * sizeof(float),
                           hipMemcpyHostToDevice),
                 "poison paired gate");
        CheckHip(hipMemcpy(paired_batch, dirty.data(),
                           dirty.size() * sizeof(float), hipMemcpyHostToDevice),
                 "poison paired up");
        if (qfn_mmq_moe_vec(static_cast<int>(formats[f]), w, dx, di,
                            batch + guard, rows, cols, n, experts, used,
                            nullptr, wb, paired_batch + guard))
          throw std::runtime_error("paired vector projection failed");
        const auto gate = check_output(batch, n);
        const auto up = check_output(paired_batch, n);
        const std::size_t bytes = n * used * rows * sizeof(float);
        if (std::memcmp(reference.data() + guard, gate.data() + guard, bytes) ||
            std::memcmp(reference_b.data() + guard, up.data() + guard, bytes))
          throw std::runtime_error(
              "paired vector differs from separate projections");
      }
      if (formats[f] == q::WeightType::kQ4_K ||
          formats[f] == q::WeightType::kQ5_K ||
          formats[f] == q::WeightType::kQ8_0) {
        // Compare the fused vector path with the original GPU SwiGLU too:
        // inactive experts, nonfinite scales and a ragged last row must
        // retain their exact values and output guards.
        q::Swiglu(scalar + guard, paired_scalar + guard, count, nullptr);
        const auto gated_reference = Download(scalar, dirty.size());
        const auto gated_widths =
            !down && formats[f] == q::WeightType::kQ4_K
                ? std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32, 33, 64}
                : std::vector{1, 2, 3, 4, 5, 6, 7, 8};
        for (int n : gated_widths) {
          CheckHip(hipMemcpy(batch, dirty.data(), dirty.size() * sizeof(float),
                             hipMemcpyHostToDevice),
                   "poison gated output");
          if (qfn_mmq_moe_gated_vec(static_cast<int>(formats[f]), w, wb, dx, di,
                                    batch + guard, rows, cols, n, experts, used,
                                    nullptr))
            throw std::runtime_error("gated vector launch failed");
          const auto gated = check_output(batch, n);
          if (std::memcmp(gated_reference.data() + guard, gated.data() + guard,
                          n * used * rows * sizeof(float)) != 0) {
            for (int i = 0; i < n * used * rows; ++i) {
              if (std::memcmp(&gated_reference[guard + i], &gated[guard + i],
                              sizeof(float)) != 0) {
                std::cerr << "first changed slot/row=" << i
                          << " expected=" << std::hexfloat
                          << gated_reference[guard + i]
                          << " actual=" << gated[guard + i] << std::defaultfloat
                          << '\n';
                break;
              }
            }
            throw std::runtime_error(
                "gated vector changed SwiGLU output for format " +
                std::to_string(f) + " width " + std::to_string(n) +
                (down ? " down rows" : " gate rows"));
          }
        }
      }
      CheckHip(hipFree(wb), "paired weight free");
      CheckHip(hipFree(w), "grouping weight free");
    }
    CheckHip(hipFree(scalar), "scalar output free");
    CheckHip(hipFree(batch), "batch output free");
    CheckHip(hipFree(paired_scalar), "paired reference free");
    CheckHip(hipFree(paired_batch), "paired output free");
  }
  CheckHip(hipFree(dx), "grouping input free");
  CheckHip(hipFree(di), "grouping IDs free");
}

void CheckPairedMmq() {
  constexpr int experts = 8, rows = 64, cols = 512, tokens = 65, used = 3;
  const auto a = MakeQ4K(experts, rows, cols, 11);
  const auto b = MakeQ4K(experts, rows, cols, 23);
  std::vector<float> x(tokens * cols);
  std::uint32_t seed = 37;
  for (auto& v : x)
    v = Uniform(&seed, 2.0F);
  std::vector<std::int32_t> ids(tokens * used);
  for (int t = 0; t < tokens; ++t)
    for (int j = 0; j < used; ++j)
      ids[t * used + j] = (t + j) % experts;
  auto* wa = Upload(a.packed);
  auto* wb = Upload(b.packed);
  auto* dx = Upload(x);
  auto* di = Upload(ids);
  const std::vector<float> zeros(tokens * used * rows);
  auto* ra = Upload(zeros);
  auto* rb = Upload(zeros);
  auto* pa = Upload(zeros);
  auto* pb = Upload(zeros);
  if (qfn_mmq_q4_K_moe_raw(wa, dx, di, ra, rows, cols, tokens, experts, used,
                           nullptr) ||
      qfn_mmq_q4_K_moe_raw(wb, dx, di, rb, rows, cols, tokens, experts, used,
                           nullptr) ||
      qfn_mmq_q4_K_moe_pair_unique(wa, wb, dx, di, pa, pb, rows, cols, tokens,
                                   experts, used, nullptr)) {
    throw std::runtime_error("paired MMQ launch failed");
  }
  const bool equal = Download(ra, zeros.size()) == Download(pa, zeros.size()) &&
                     Download(rb, zeros.size()) == Download(pb, zeros.size());
  for (void* ptr :
       {static_cast<void*>(wa), static_cast<void*>(wb), static_cast<void*>(dx),
        static_cast<void*>(di), static_cast<void*>(ra), static_cast<void*>(rb),
        static_cast<void*>(pa), static_cast<void*>(pb)})
    CheckHip(hipFree(ptr), "paired MMQ free");
  if (!equal)
    throw std::runtime_error("paired MMQ differs from separate projections");
}

}  // namespace

int main() {
  try {
    ReportVectorAccuracy();
    CheckVectorGrouping();
    CheckVectorGrouping(true);
    CheckPairedMmq();
    bool ok = true;
    // Gate/up view: 64 experts, top-10, 640 x 2560 Q4_K.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 10, 64, 640, 2560, 0x1234ABCDU)) &&
         ok;
    // Down view: single-expert slot rows, 2560 x 640 Q5_1.
    ok = Ok(Run(q::WeightType::kQ5_1, 3000, 1, 64, 2560, 640, 0x0BADF00DU,
                64)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 97, 1, 8, 129, 640, 0x51516464U, 64)) &&
         ok;
    // Q5_K gate/up view (one layer).
    ok = Ok(Run(q::WeightType::kQ5_K, 300, 10, 64, 640, 2560, 0x5A5A0001U)) &&
         ok;
    // Q8_0 down view (five layers keep Q8_0 down projections).
    ok = Ok(Run(q::WeightType::kQ8_0, 3000, 1, 64, 2560, 640, 0x0C0FFEE0U,
                64)) &&
         ok;
    // IQ3_S gate/up view and IQ4_NL gate/up and down views.
    ok = Ok(Run(q::WeightType::kIQ3_S, 300, 10, 64, 640, 2560, 0x00C0FFEEU)) &&
         ok;
    ok = Ok(Run(q::WeightType::kIQ4_NL, 3000, 1, 64, 2560, 640, 0x00BAD002U,
                48)) &&
         ok;
    ok = Ok(Run(q::WeightType::kIQ4_NL, 97, 1, 8, 129, 640, 0x00BAD003U, 64)) &&
         ok;
    // Ragged rows against the 128-row tile and a tiny batch.
    ok = Ok(Run(q::WeightType::kQ4_K, 40, 4, 8, 201, 512, 0xDEADBEEFU)) && ok;
    // Paired waves must retain unsaturated SiLU arithmetic and protect the
    // last odd output row in short and partial token groups.
    for (auto type : {q::WeightType::kQ4_K, q::WeightType::kQ5_K}) {
      for (std::size_t rows : {1U, 8U, 9U, 32U, 33U, 65U, 129U}) {
        ok = Ok(Run(type, rows, 1, 1, 65, 256, 0x5A1711U, 48, 0.002F)) && ok;
      }
    }
    // The 16-row tile (small buckets) on every type.
    ok = Ok(Run(q::WeightType::kQ4_K, 300, 10, 64, 640, 2560, 0x16161616U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_K, 300, 10, 64, 640, 2560, 0x16160002U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ5_1, 3000, 1, 64, 2560, 640, 0x16160003U,
                16)) &&
         ok;
    ok = Ok(Run(q::WeightType::kQ8_0, 3000, 1, 64, 2560, 640, 0x16160004U,
                16)) &&
         ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
