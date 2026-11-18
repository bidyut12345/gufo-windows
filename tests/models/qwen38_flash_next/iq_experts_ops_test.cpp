#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace {

using gufo::core::GgmlType;

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

float Uniform(std::uint32_t* state, float scale) noexcept {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

template<typename T>
T* Upload(const std::vector<T>& host, std::size_t extra_bytes = 4096) {
  T* device = nullptr;
  const std::size_t bytes = host.size() * sizeof(T);
  CheckHip(hipMalloc(&device, bytes + extra_bytes), "hipMalloc");
  if (bytes > 0) {
    CheckHip(hipMemcpy(device, host.data(), bytes, hipMemcpyHostToDevice),
             "hipMemcpy H2D");
  }
  if (extra_bytes > 0) {
    CheckHip(hipMemset(reinterpret_cast<std::uint8_t*>(device) + bytes, 0,
                       extra_bytes),
             "hipMemset tail");
  }
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  if (count > 0) {
    CheckHip(hipMemcpy(host.data(), device, count * sizeof(T),
                       hipMemcpyDeviceToHost),
             "hipMemcpy D2H");
  }
  return host;
}

std::uint16_t FloatToHalfBits(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t abs_bits = bits & 0x7FFFFFFFU;
  if (abs_bits >= 0x47800000U) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (abs_bits < 0x38800000U) {
    if (abs_bits < 0x33000000U) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t shift = 126U - (abs_bits >> 23);
    const std::uint32_t mant = (abs_bits & 0x007FFFFFU) | 0x00800000U;
    const std::uint32_t rounded =
        mant + (1U << (shift - 1U)) - 1U + ((mant >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded =
      abs_bits + 0x00000FFFU + ((abs_bits >> 13) & 1U);
  if (rounded >= 0x47800000U) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  return static_cast<std::uint16_t>(sign | (((rounded >> 23) - 112U) << 10) |
                                    ((rounded >> 13) & 0x03FFU));
}

struct PackedExperts {
  std::vector<std::uint8_t> packed;
  std::vector<float> values;  // [e][m][k]
};

PackedExperts MakeExperts(GgmlType type, std::size_t e, std::size_t m,
                          std::size_t k, std::uint32_t seed) {
  const std::size_t block_elems = gufo::quant::QuantizedBlockElements(type);
  const std::size_t block_bytes =
      gufo::quant::QuantizedRowBytes(type, block_elems);
  if (block_elems == 0 || block_bytes == 0 || k % block_elems != 0) {
    throw std::runtime_error("invalid test shape for format");
  }
  const std::size_t blocks_per_row = k / block_elems;
  const std::size_t row_bytes = blocks_per_row * block_bytes;
  PackedExperts w;
  w.packed.resize(e * m * row_bytes);
  for (auto& byte : w.packed) {
    byte = static_cast<std::uint8_t>(NextRandom(&seed) & 0xFFU);
  }
  for (std::size_t b = 0; b < e * m * blocks_per_row; ++b) {
    std::uint8_t* blk = w.packed.data() + b * block_bytes;
    const float s = (Uniform(&seed, 0.04F) >= 0.0F ? 1.0F : -1.0F) *
                    (0.03F + std::abs(Uniform(&seed, 0.05F)));
    const std::uint16_t h = FloatToHalfBits(s);
    if (type == GgmlType::kQ6_K) {
      std::memcpy(blk + 208, &h, 2);
    } else {
      std::memcpy(blk, &h, 2);
    }
  }
  w.values.resize(e * m * k);
  for (std::size_t row = 0; row < e * m; ++row) {
    if (!gufo::quant::DequantizeRow(type, w.packed.data() + row * row_bytes,
                                    w.values.data() + row * k, k)) {
      throw std::runtime_error("DequantizeRow failed in test fixture");
    }
  }
  return w;
}

std::vector<float> MakeInputs(std::size_t n, std::size_t k,
                              std::uint32_t seed) {
  std::vector<float> x(n * k);
  for (float& v : x) {
    v = Uniform(&seed, 0.8F);
  }
  return x;
}

std::vector<std::int32_t> MakeIds(std::size_t tokens, std::size_t used,
                                  std::size_t experts, std::uint32_t seed) {
  std::vector<std::int32_t> ids(tokens * used);
  for (std::size_t t = 0; t < tokens; ++t) {
    std::size_t start = NextRandom(&seed) % experts;
    for (std::size_t u = 0; u < used; ++u) {
      ids[t * used + u] =
          static_cast<std::int32_t>((start + u * 2 + 1) % experts);
    }
  }
  return ids;
}

std::vector<float> CpuMoe(const std::vector<float>& w,
                          const std::vector<float>& x,
                          const std::vector<std::int32_t>& ids,
                          std::size_t tokens, std::size_t used, std::size_t m,
                          std::size_t k) {
  std::vector<float> out(tokens * used * m, 0.0F);
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t u = 0; u < used; ++u) {
      const auto e = static_cast<std::size_t>(ids[t * used + u]);
      for (std::size_t r = 0; r < m; ++r) {
        double acc = 0.0;
        const float* wr = w.data() + (e * m + r) * k;
        const float* xr = x.data() + t * k;
        for (std::size_t c = 0; c < k; ++c) {
          acc += static_cast<double>(wr[c]) * static_cast<double>(xr[c]);
        }
        out[(t * used + u) * m + r] = static_cast<float>(acc);
      }
    }
  }
  return out;
}

std::vector<float> CpuMoeDown(const std::vector<float>& w,
                              const std::vector<float>& x,
                              const std::vector<std::int32_t>& ids,
                              std::size_t slots, std::size_t m, std::size_t k) {
  std::vector<float> out(slots * m, 0.0F);
  for (std::size_t s = 0; s < slots; ++s) {
    const auto e = static_cast<std::size_t>(ids[s]);
    for (std::size_t r = 0; r < m; ++r) {
      double acc = 0.0;
      const float* wr = w.data() + (e * m + r) * k;
      const float* xr = x.data() + s * k;
      for (std::size_t c = 0; c < k; ++c) {
        acc += static_cast<double>(wr[c]) * static_cast<double>(xr[c]);
      }
      out[s * m + r] = static_cast<float>(acc);
    }
  }
  return out;
}

void CheckSimilar(const std::vector<float>& ref,
                  const std::vector<float>& actual, const std::string& label) {
  if (ref.size() != actual.size() || ref.empty()) {
    throw std::runtime_error(label + ": size mismatch");
  }
  double dot = 0.0;
  double norm_r = 0.0;
  double norm_a = 0.0;
  double max_diff = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double r = ref[i];
    const double a = actual[i];
    if (!std::isfinite(a)) {
      throw std::runtime_error(label + ": non-finite output at " +
                               std::to_string(i) + " val=" + std::to_string(a));
    }
    dot += r * a;
    norm_r += r * r;
    norm_a += a * a;
    max_diff = std::max(max_diff, std::abs(r - a));
  }
  const double denom = std::sqrt(norm_r) * std::sqrt(norm_a);
  const double cos_sim = denom > 0.0 ? dot / denom : 1.0;
  const double rms_ref = std::sqrt(norm_r / static_cast<double>(ref.size()));
  const double rel_max = rms_ref > 0.0 ? max_diff / rms_ref : max_diff;
  // Q8_1 activation quantization on random inputs allows ~1-2% relative jitter,
  // while wrong layouts or unpacking fail cosine by orders of magnitude.
  if (cos_sim < 0.997 || rel_max > 0.25) {
    throw std::runtime_error(label +
                             ": mismatch cos=" + std::to_string(cos_sim) +
                             " rel_max=" + std::to_string(rel_max));
  }
}

int CallMoeRaw(GgmlType type, const void* w, const float* x,
               const std::int32_t* ids, float* out, int m, int k, int t, int e,
               int u) {
  switch (type) {
    case GgmlType::kQ2_0:
      return qfn_mmq_q2_0_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kQ5_0:
      return qfn_mmq_q5_0_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ2_S:
      return qfn_mmq_iq2_s_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ3_XXS:
      return qfn_mmq_iq3_xxs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ3_S:
      return qfn_mmq_iq3_s_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ4_NL:
      return qfn_mmq_iq4_nl_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    case GgmlType::kIQ4_XS:
      return qfn_mmq_iq4_xs_moe_raw(w, x, ids, out, m, k, t, e, u, nullptr);
    default:
      return -1;
  }
}

int CallMoePairUnique(GgmlType type, const void* wa, const void* wb,
                      const float* x, const std::int32_t* ids, float* out_a,
                      float* out_b, int m, int k, int t, int e, int u) {
  switch (type) {
    case GgmlType::kIQ2_S:
      return qfn_mmq_iq2_s_moe_pair_unique(wa, wb, x, ids, out_a, out_b, m, k,
                                           t, e, u, nullptr);
    case GgmlType::kIQ3_XXS:
      return qfn_mmq_iq3_xxs_moe_pair_unique(wa, wb, x, ids, out_a, out_b, m, k,
                                             t, e, u, nullptr);
    case GgmlType::kIQ3_S:
      return qfn_mmq_iq3_s_moe_pair_unique(wa, wb, x, ids, out_a, out_b, m, k,
                                           t, e, u, nullptr);
    case GgmlType::kIQ4_XS:
      return qfn_mmq_iq4_xs_moe_pair_unique(wa, wb, x, ids, out_a, out_b, m, k,
                                            t, e, u, nullptr);
    default:
      return -1;
  }
}

void TestRoutedFormat(GgmlType type) {
  constexpr std::size_t kE = 4;
  constexpr std::size_t kU = 2;
  constexpr std::size_t kM = 128;
  constexpr std::size_t kK = 512;
  const std::string name(gufo::core::ToString(type));
  const auto w = MakeExperts(type, kE, kM, kK,
                             0x10203040U ^ static_cast<std::uint32_t>(type));
  auto* d_w = Upload(w.packed);

  // MMVQ decode (T=1) and multi-token vector (T=5).
  for (std::size_t tokens : {std::size_t{1}, std::size_t{5}}) {
    const auto x =
        MakeInputs(tokens, kK, 0x551100U + static_cast<std::uint32_t>(tokens));
    const auto ids =
        MakeIds(tokens, kU, kE, 0x770011U + static_cast<std::uint32_t>(tokens));
    auto* d_x = Upload(x);
    auto* d_ids = Upload(ids, 256);
    float* d_out = nullptr;
    CheckHip(hipMalloc(&d_out, tokens * kU * kM * sizeof(float)),
             "hipMalloc out");
    const int rc = qfn_mmq_moe_vec(
        static_cast<int>(type), d_w, d_x, d_ids, d_out, static_cast<int>(kM),
        static_cast<int>(kK), static_cast<int>(tokens), static_cast<int>(kE),
        static_cast<int>(kU), nullptr);
    if (rc != 0) {
      throw std::runtime_error(name + " qfn_mmq_moe_vec failed");
    }
    CheckHip(hipDeviceSynchronize(), "sync vec");
    CheckSimilar(CpuMoe(w.values, x, ids, tokens, kU, kM, kK),
                 Download(d_out, tokens * kU * kM),
                 name + " vec T=" + std::to_string(tokens));
    (void)hipFree(d_x);
    (void)hipFree(d_ids);
    (void)hipFree(d_out);
  }

  // Down projection pattern: slots = T * U rows with U=1 expert per slot
  // (covers 4-row down tile too).
  {
    constexpr std::size_t kSlots = 8;
    const auto x = MakeInputs(kSlots, kK, 0x998877U);
    const auto ids = MakeIds(kSlots, 1, kE, 0x443322U);
    auto* d_x = Upload(x);
    auto* d_ids = Upload(ids, 256);
    float* d_out = nullptr;
    CheckHip(hipMalloc(&d_out, kSlots * kM * sizeof(float)), "hipMalloc down");
    const int rc = qfn_mmq_moe_vec(
        static_cast<int>(type), d_w, d_x, d_ids, d_out, static_cast<int>(kM),
        static_cast<int>(kK), static_cast<int>(kSlots), static_cast<int>(kE), 1,
        nullptr);
    if (rc != 0) {
      throw std::runtime_error(name + " qfn_mmq_moe_vec down failed");
    }
    CheckHip(hipDeviceSynchronize(), "sync down vec");
    CheckSimilar(CpuMoeDown(w.values, x, ids, kSlots, kM, kK),
                 Download(d_out, kSlots * kM), name + " down vec");
    (void)hipFree(d_x);
    (void)hipFree(d_ids);
    (void)hipFree(d_out);
  }

  // MMQ tiled prefill across both tile column regimes.
  for (int tile_cols : {16, 128}) {
    qfn_mmq_set_routed_tile_cols(tile_cols);
    for (std::size_t tokens : {std::size_t{5}, std::size_t{17}}) {
      const auto x = MakeInputs(tokens, kK,
                                0xABCD01U + static_cast<std::uint32_t>(tokens));
      const auto ids = MakeIds(tokens, kU, kE,
                               0xDEAD02U + static_cast<std::uint32_t>(tokens));
      auto* d_x = Upload(x);
      auto* d_ids = Upload(ids, 256);
      float* d_out = nullptr;
      CheckHip(hipMalloc(&d_out, tokens * kU * kM * sizeof(float)),
               "hipMalloc raw");
      if (CallMoeRaw(type, d_w, d_x, d_ids, d_out, static_cast<int>(kM),
                     static_cast<int>(kK), static_cast<int>(tokens),
                     static_cast<int>(kE), static_cast<int>(kU)) != 0) {
        throw std::runtime_error(name + " moe_raw failed");
      }
      CheckHip(hipDeviceSynchronize(), "sync raw");
      CheckSimilar(CpuMoe(w.values, x, ids, tokens, kU, kM, kK),
                   Download(d_out, tokens * kU * kM),
                   name + " raw T=" + std::to_string(tokens) +
                       " tile=" + std::to_string(tile_cols));
      (void)hipFree(d_x);
      (void)hipFree(d_ids);
      (void)hipFree(d_out);
    }
  }
  qfn_mmq_set_routed_tile_cols(0);
  (void)hipFree(d_w);
}

void TestGatedIqFormat(GgmlType type) {
  constexpr std::size_t kE = 4;
  constexpr std::size_t kU = 2;
  constexpr std::size_t kM = 128;
  constexpr std::size_t kK = 512;
  const std::string name(gufo::core::ToString(type));
  const auto gate = MakeExperts(type, kE, kM, kK,
                                0x11112222U ^ static_cast<std::uint32_t>(type));
  const auto up = MakeExperts(type, kE, kM, kK,
                              0x33334444U ^ static_cast<std::uint32_t>(type));
  auto* d_gate = Upload(gate.packed);
  auto* d_up = Upload(up.packed);

  // Decode gated SwiGLU vector path (T=1, T=4, T=8).
  for (std::size_t tokens : {std::size_t{1}, std::size_t{4}, std::size_t{8}}) {
    const auto x =
        MakeInputs(tokens, kK, 0xCAFE01U + static_cast<std::uint32_t>(tokens));
    const auto ids =
        MakeIds(tokens, kU, kE, 0xCAFE02U + static_cast<std::uint32_t>(tokens));
    auto* d_x = Upload(x);
    auto* d_ids = Upload(ids, 256);
    float* d_out = nullptr;
    CheckHip(hipMalloc(&d_out, tokens * kU * kM * sizeof(float)),
             "hipMalloc gated");
    if (qfn_mmq_moe_gated_vec(static_cast<int>(type), d_gate, d_up, d_x, d_ids,
                              d_out, static_cast<int>(kM), static_cast<int>(kK),
                              static_cast<int>(tokens), static_cast<int>(kE),
                              static_cast<int>(kU), nullptr) != 0) {
      throw std::runtime_error(name + " moe_gated_vec failed");
    }
    CheckHip(hipDeviceSynchronize(), "sync gated");
    const auto g_ref = CpuMoe(gate.values, x, ids, tokens, kU, kM, kK);
    const auto u_ref = CpuMoe(up.values, x, ids, tokens, kU, kM, kK);
    std::vector<float> ref(g_ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const float g = g_ref[i];
      ref[i] = (g / (1.0F + std::exp(-g))) * u_ref[i];
    }
    CheckSimilar(ref, Download(d_out, tokens * kU * kM),
                 name + " gated_vec T=" + std::to_string(tokens));
    (void)hipFree(d_x);
    (void)hipFree(d_ids);
    (void)hipFree(d_out);
  }

  // Prefill unique pair path (T=5, T=17).
  for (std::size_t tokens : {std::size_t{5}, std::size_t{17}}) {
    const auto x =
        MakeInputs(tokens, kK, 0xBEEF01U + static_cast<std::uint32_t>(tokens));
    const auto ids =
        MakeIds(tokens, kU, kE, 0xBEEF02U + static_cast<std::uint32_t>(tokens));
    auto* d_x = Upload(x);
    auto* d_ids = Upload(ids, 256);
    float* d_out_a = nullptr;
    float* d_out_b = nullptr;
    CheckHip(hipMalloc(&d_out_a, tokens * kU * kM * sizeof(float)),
             "hipMalloc a");
    CheckHip(hipMalloc(&d_out_b, tokens * kU * kM * sizeof(float)),
             "hipMalloc b");
    if (CallMoePairUnique(type, d_gate, d_up, d_x, d_ids, d_out_a, d_out_b,
                          static_cast<int>(kM), static_cast<int>(kK),
                          static_cast<int>(tokens), static_cast<int>(kE),
                          static_cast<int>(kU)) != 0) {
      throw std::runtime_error(name + " moe_pair_unique failed");
    }
    CheckHip(hipDeviceSynchronize(), "sync pair_unique");
    CheckSimilar(CpuMoe(gate.values, x, ids, tokens, kU, kM, kK),
                 Download(d_out_a, tokens * kU * kM),
                 name + " pair_unique gate T=" + std::to_string(tokens));
    CheckSimilar(CpuMoe(up.values, x, ids, tokens, kU, kM, kK),
                 Download(d_out_b, tokens * kU * kM),
                 name + " pair_unique up T=" + std::to_string(tokens));
    (void)hipFree(d_x);
    (void)hipFree(d_ids);
    (void)hipFree(d_out_a);
    (void)hipFree(d_out_b);
  }
  (void)hipFree(d_gate);
  (void)hipFree(d_up);
}

void TestDenseReencodeToQ8(GgmlType type) {
  constexpr std::size_t kM = 64;
  constexpr std::size_t kN = 4;
  constexpr std::size_t kK = 512;
  const auto w = MakeExperts(type, 1, kM, kK,
                             0x76543210U ^ static_cast<std::uint32_t>(type));
  std::vector<std::uint8_t> q8(kM * (kK / 32) * 34);
  for (std::size_t r = 0; r < kM; ++r) {
    const float* src = w.values.data() + r * kK;
    std::uint8_t* dst = q8.data() + r * (kK / 32) * 34;
    for (std::size_t c = 0; c < kK; c += 32, dst += 34) {
      float amax = 0.0F;
      for (std::size_t j = 0; j < 32; ++j) {
        amax = std::max(amax, std::abs(src[c + j]));
      }
      const float d = amax / 127.0F;
      const float id = d != 0.0F ? 1.0F / d : 0.0F;
      const std::uint16_t d_half = FloatToHalfBits(d);
      std::memcpy(dst, &d_half, 2);
      auto* qs = reinterpret_cast<std::int8_t*>(dst + 2);
      for (std::size_t j = 0; j < 32; ++j) {
        qs[j] = static_cast<std::int8_t>(
            std::clamp(std::lround(src[c + j] * id), -127L, 127L));
      }
    }
  }
  const auto x = MakeInputs(kN, kK, 0x13572468U);
  auto* d_w = Upload(q8);
  auto* d_x = Upload(x);
  float* d_out = nullptr;
  CheckHip(hipMalloc(&d_out, kN * kM * sizeof(float)), "hipMalloc dense");
  if (qfn_mmq_q8_0_dense(d_w, d_x, d_out, static_cast<int>(kM),
                         static_cast<int>(kN), static_cast<int>(kK),
                         nullptr) != 0) {
    throw std::runtime_error("qfn_mmq_q8_0_dense failed after re-encode");
  }
  CheckHip(hipDeviceSynchronize(), "sync dense");
  std::vector<float> ref(kN * kM, 0.0F);
  for (std::size_t n = 0; n < kN; ++n) {
    for (std::size_t r = 0; r < kM; ++r) {
      double acc = 0.0;
      for (std::size_t c = 0; c < kK; ++c) {
        acc += static_cast<double>(w.values[r * kK + c]) *
               static_cast<double>(x[n * kK + c]);
      }
      ref[n * kM + r] = static_cast<float>(acc);
    }
  }
  CheckSimilar(ref, Download(d_out, kN * kM),
               std::string(gufo::core::ToString(type)) + "->Q8_0 dense");
  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_out);
}

void TestWmmaGatedIqFormat(GgmlType type) {
  namespace q = gufo::models::qwen38_flash_next::rocm;
  constexpr std::size_t kE = 4;
  constexpr std::size_t kU = 2;
  constexpr std::size_t kM = 128;
  constexpr std::size_t kK = 512;
  constexpr std::size_t kT = 40;
  const std::string name(gufo::core::ToString(type));
  const auto wg = MakeExperts(type, kE, kM, kK,
                              0x44332211U ^ static_cast<std::uint32_t>(type));
  const auto wu = MakeExperts(type, kE, kM, kK,
                              0x88776655U ^ static_cast<std::uint32_t>(type));
  auto x = MakeInputs(kT, kK, 0x19283746U);
  for (float& v : x) {
    v *= 0.01F;
  }
  const auto ids = MakeIds(kT, kU, kE, 0x55667788U);
  auto* d_wg = Upload(wg.packed);
  auto* d_wu = Upload(wu.packed);
  auto* d_x = Upload(x);
  auto* d_ids = Upload(ids, 256);

  __half* d_xh = nullptr;
  __half* d_out_h = nullptr;
  std::uint32_t* d_counts = nullptr;
  std::int32_t* d_bounds = nullptr;
  std::int32_t* d_cursors = nullptr;
  std::int32_t* d_rtok = nullptr;
  std::int32_t* d_rslot = nullptr;
  std::int32_t* d_tiles = nullptr;
  const std::size_t compact_rows = q::RoutedCompactRows(kT * kU, kE);
  CheckHip(hipMalloc(&d_xh, kT * kK * sizeof(__half)), "xh");
  CheckHip(hipMalloc(&d_out_h, kT * kU * kM * sizeof(__half)), "out_h");
  CheckHip(hipMalloc(&d_counts, kE * sizeof(std::uint32_t)), "counts");
  CheckHip(hipMemset(d_counts, 0, kE * sizeof(std::uint32_t)), "zero counts");
  CheckHip(hipMalloc(&d_bounds, (kE + 1) * sizeof(std::int32_t)), "bounds");
  CheckHip(hipMalloc(&d_cursors, kE * sizeof(std::int32_t)), "cursors");
  CheckHip(hipMalloc(&d_rtok, compact_rows * sizeof(std::int32_t)), "rtok");
  CheckHip(hipMalloc(&d_rslot, compact_rows * sizeof(std::int32_t)), "rslot");
  CheckHip(hipMalloc(&d_tiles, (compact_rows / 16 + kE) * sizeof(std::int32_t)),
           "tiles");

  q::NarrowActivations(d_x, d_xh, false, kT * kK, nullptr);
  q::ExpertCounts(d_ids, d_counts, kT, kE, kU, nullptr);
  std::vector<std::uint32_t> counts_h(kE);
  CheckHip(hipMemcpy(counts_h.data(), d_counts, kE * sizeof(std::uint32_t),
                     hipMemcpyDeviceToHost),
           "counts d2h");
  std::vector<std::int32_t> tiles_h;
  for (std::uint32_t e = 0; e < kE; ++e) {
    for (std::uint32_t j = 0; j < (counts_h[e] + 127U) / 128U; ++j) {
      tiles_h.push_back(static_cast<std::int32_t>(e | (j << 16U)));
    }
  }
  CheckHip(
      hipMemcpy(d_tiles, tiles_h.data(), tiles_h.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "tiles h2d");
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rtok, d_rslot, kT,
                   kU, kE, nullptr);
  const q::WeightType wt = type == GgmlType::kIQ2_S ? q::WeightType::kIQ2_S
                           : type == GgmlType::kIQ3_XXS
                               ? q::WeightType::kIQ3_XXS
                               : q::WeightType::kIQ4_XS;
  if (!q::RoutedGatedF16Gemm(d_wg, d_wu, wt, d_xh, d_tiles,
                             static_cast<std::uint32_t>(tiles_h.size()),
                             d_bounds, d_rtok, d_rslot, d_out_h, kM, kK,
                             nullptr)) {
    throw std::runtime_error(name + " RoutedGatedF16Gemm failed");
  }
  CheckHip(hipDeviceSynchronize(), "sync wmma gated");
  std::vector<__half> out_h(kT * kU * kM);
  CheckHip(hipMemcpy(out_h.data(), d_out_h, out_h.size() * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "out d2h");
  std::vector<float> actual(out_h.size());
  for (std::size_t i = 0; i < out_h.size(); ++i) {
    actual[i] = __half2float(out_h[i]);
  }
  const auto ref_g = CpuMoe(wg.values, x, ids, kT, kU, kM, kK);
  const auto ref_u = CpuMoe(wu.values, x, ids, kT, kU, kM, kK);
  std::vector<float> ref(ref_g.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double g = ref_g[i];
    ref[i] = static_cast<float>((g / (1.0 + std::exp(-g))) * ref_u[i]);
  }
  CheckSimilar(ref, actual, name + " wmma gated");

  (void)hipFree(d_wg);
  (void)hipFree(d_wu);
  (void)hipFree(d_x);
  (void)hipFree(d_ids);
  (void)hipFree(d_xh);
  (void)hipFree(d_out_h);
  (void)hipFree(d_counts);
  (void)hipFree(d_bounds);
  (void)hipFree(d_cursors);
  (void)hipFree(d_rtok);
  (void)hipFree(d_rslot);
  (void)hipFree(d_tiles);
}

void TestWmmaDownFormat(GgmlType type) {
  namespace q = gufo::models::qwen38_flash_next::rocm;
  constexpr std::size_t kE = 4;
  constexpr std::size_t kU = 2;
  constexpr std::size_t kM = 256;
  constexpr std::size_t kK = 256;
  constexpr std::size_t kT = 40;
  constexpr std::size_t kSlots = kT * kU;
  const std::string name(gufo::core::ToString(type));
  const auto w = MakeExperts(type, kE, kM, kK,
                             0x31415926U ^ static_cast<std::uint32_t>(type));
  auto x = MakeInputs(kSlots, kK, 0x27182818U);
  for (float& v : x) {
    v *= 0.01F;
  }
  const auto ids = MakeIds(kT, kU, kE, 0x11223344U);
  auto* d_w = Upload(w.packed);
  auto* d_x = Upload(x);
  auto* d_ids = Upload(ids, 256);

  __half* d_xh = nullptr;
  __half* d_out_h = nullptr;
  std::uint32_t* d_counts = nullptr;
  std::int32_t* d_bounds = nullptr;
  std::int32_t* d_cursors = nullptr;
  std::int32_t* d_rtok = nullptr;
  std::int32_t* d_rslot = nullptr;
  std::int32_t* d_tiles = nullptr;
  const std::size_t compact_rows = q::RoutedCompactRows(kSlots, kE);
  CheckHip(hipMalloc(&d_xh, kSlots * kK * sizeof(__half)), "xh down");
  CheckHip(hipMalloc(&d_out_h, kSlots * kM * sizeof(__half)), "out_h down");
  CheckHip(hipMemset(d_out_h, 0, kSlots * kM * sizeof(__half)), "zero out_h");
  CheckHip(hipMalloc(&d_counts, kE * sizeof(std::uint32_t)), "counts down");
  CheckHip(hipMemset(d_counts, 0, kE * sizeof(std::uint32_t)), "zero counts");
  CheckHip(hipMalloc(&d_bounds, (kE + 1) * sizeof(std::int32_t)),
           "bounds down");
  CheckHip(hipMalloc(&d_cursors, kE * sizeof(std::int32_t)), "cursors down");
  CheckHip(hipMalloc(&d_rtok, compact_rows * sizeof(std::int32_t)),
           "rtok down");
  CheckHip(hipMalloc(&d_rslot, compact_rows * sizeof(std::int32_t)),
           "rslot down");
  CheckHip(hipMalloc(&d_tiles, (compact_rows / 16 + kE) * sizeof(std::int32_t)),
           "tiles down");

  q::NarrowActivations(d_x, d_xh, false, kSlots * kK, nullptr);
  q::ExpertCounts(d_ids, d_counts, kT, kE, kU, nullptr);
  std::vector<std::uint32_t> counts_h(kE);
  CheckHip(hipMemcpy(counts_h.data(), d_counts, kE * sizeof(std::uint32_t),
                     hipMemcpyDeviceToHost),
           "counts d2h");
  std::vector<std::int32_t> tiles_h;
  for (std::uint32_t e = 0; e < kE; ++e) {
    for (std::uint32_t j = 0; j < (counts_h[e] + 127U) / 128U; ++j) {
      tiles_h.push_back(static_cast<std::int32_t>(e | (j << 16U)));
    }
  }
  CheckHip(
      hipMemcpy(d_tiles, tiles_h.data(), tiles_h.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "tiles h2d");
  q::RoutedCompact(d_ids, d_counts, d_bounds, d_cursors, d_rtok, d_rslot, kT,
                   kU, kE, nullptr);
  const q::WeightType wt = type == GgmlType::kQ5_0     ? q::WeightType::kQ5_0
                           : type == GgmlType::kIQ4_NL ? q::WeightType::kIQ4_NL
                                                       : q::WeightType::kQ2_0;
  if (!q::RoutedF16Gemm(d_w, wt, d_xh, d_tiles,
                        static_cast<std::uint32_t>(tiles_h.size()), 128,
                        d_bounds, d_rslot, d_rslot, nullptr, nullptr, d_out_h,
                        kM, kK, nullptr)) {
    throw std::runtime_error(name + " RoutedF16Gemm down failed");
  }
  CheckHip(hipDeviceSynchronize(), "sync wmma down");
  std::vector<__half> out_h(kSlots * kM);
  CheckHip(hipMemcpy(out_h.data(), d_out_h, out_h.size() * sizeof(__half),
                     hipMemcpyDeviceToHost),
           "out d2h");
  std::vector<float> actual(out_h.size());
  for (std::size_t i = 0; i < out_h.size(); ++i) {
    actual[i] = __half2float(out_h[i]);
  }
  CheckSimilar(CpuMoeDown(w.values, x, ids, kSlots, kM, kK), actual,
               name + " wmma down");

  (void)hipFree(d_w);
  (void)hipFree(d_x);
  (void)hipFree(d_ids);
  (void)hipFree(d_xh);
  (void)hipFree(d_out_h);
  (void)hipFree(d_counts);
  (void)hipFree(d_bounds);
  (void)hipFree(d_cursors);
  (void)hipFree(d_rtok);
  (void)hipFree(d_rslot);
  (void)hipFree(d_tiles);
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0) {
      throw std::runtime_error("qfn_mmq_init failed");
    }
    for (const auto type : {GgmlType::kQ2_0, GgmlType::kQ5_0, GgmlType::kIQ2_S,
                            GgmlType::kIQ3_XXS, GgmlType::kIQ3_S,
                            GgmlType::kIQ4_NL, GgmlType::kIQ4_XS}) {
      TestRoutedFormat(type);
    }
    for (const auto type : {GgmlType::kIQ2_S, GgmlType::kIQ3_XXS,
                            GgmlType::kIQ3_S, GgmlType::kIQ4_XS}) {
      TestGatedIqFormat(type);
    }
    for (const auto type :
         {GgmlType::kIQ2_S, GgmlType::kIQ3_XXS, GgmlType::kIQ4_XS}) {
      TestWmmaGatedIqFormat(type);
    }
    for (const auto type :
         {GgmlType::kQ5_0, GgmlType::kIQ4_NL, GgmlType::kQ2_0}) {
      TestWmmaDownFormat(type);
    }
    TestDenseReencodeToQ8(GgmlType::kQ6_K);
    TestDenseReencodeToQ8(GgmlType::kIQ4_XS);
    std::cout << "iq_experts_ops_test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
