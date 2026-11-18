// Decode-GEMV microbenchmark for the qwen35moe dense shapes: the Q8_0
// dequantized route (quantize + dense_vec_preq) against native Q4_K through
// the routed vector kernel with a single expert. Reports effective GB/s so
// kernel efficiency is comparable to the measured device bandwidth.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <vector>

#include "qfn_mmq.h"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

constexpr std::uint32_t kHidden = 2048;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::uint32_t NextRandom(std::uint32_t* s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}

/// Pseudo-Q4_K blocks: 256 elements -> 144 bytes (scales + nibbles).
std::vector<std::uint8_t> MakeQ4K(std::size_t rows, std::uint32_t seed) {
  const std::size_t blocks = kHidden / 256;
  std::vector<std::uint8_t> w(rows * blocks * 144);
  std::uint32_t state = seed;
  for (auto& b : w) {
    b = static_cast<std::uint8_t>(NextRandom(&state));
  }
  return w;
}

/// Pseudo-Q8_0 blocks: 32 elements -> 34 bytes.
std::vector<std::uint8_t> MakeQ80(std::size_t rows, std::uint32_t seed) {
  const std::size_t blocks = kHidden / 32;
  std::vector<std::uint8_t> w(rows * blocks * 34);
  std::uint32_t state = seed;
  for (auto& b : w) {
    b = static_cast<std::uint8_t>(NextRandom(&state));
  }
  // Keep scales finite halves in a sane range so outputs stay finite.
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      w[(r * blocks + b) * 34] = 0x38;
      w[(r * blocks + b) * 34 + 1] = 0x00;
    }
  }
  return w;
}

/// Pseudo-Q6_K blocks: 256 elements -> 210 bytes (low nibbles, high bits,
/// int8 scales, one F16 scale).
std::vector<std::uint8_t> MakeQ6K(std::size_t rows, std::uint32_t seed) {
  const std::size_t blocks = kHidden / 256;
  std::vector<std::uint8_t> w(rows * blocks * 210);
  std::uint32_t state = seed;
  for (auto& b : w) {
    b = static_cast<std::uint8_t>(NextRandom(&state));
  }
  // Keep the block scale a finite half in a sane range so outputs stay
  // finite; the int8 scales and 6-bit codes need no clamping.
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      w[(r * blocks + b) * 210 + 208] = 0x38;
      w[(r * blocks + b) * 210 + 209] = 0x00;
    }
  }
  return w;
}

double Seconds(std::uint32_t iters, const std::function<void()>& body) {
  body();
  Check(hipDeviceSynchronize(), "warmup");
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t i = 0; i < iters; ++i) {
    body();
  }
  Check(hipDeviceSynchronize(), "run");
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
                 .count() /
         iters;
}

void Run(const char* name, std::size_t rows, std::uint32_t iters,
         double q8_bytes, double q4_bytes, double q6_bytes) {
  // Q8_0 route: quantize + dense_vec_preq.
  auto q8 = MakeQ80(rows, 0x1111);
  void* d_q8 = nullptr;
  void* d_xq = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  std::size_t xq_bytes = qfn_mmq_q8_1_bytes(1, kHidden);
  Check(hipMalloc(&d_q8, q8.size()), "q8 malloc");
  Check(hipMalloc(&d_xq, xq_bytes), "xq malloc");
  Check(hipMalloc(&d_x, kHidden * sizeof(float)), "x malloc");
  Check(hipMalloc(&d_out, rows * sizeof(float)), "out malloc");
  Check(hipMemcpy(d_q8, q8.data(), q8.size(), hipMemcpyHostToDevice),
        "q8 upload");
  std::vector<float> x(kHidden);
  for (auto& v : x) v = 0.01F;
  Check(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
  const double t_q8 =
      Seconds(iters, [&] {
        qfn_mmq_quantize_q8_1(d_x, d_xq, 1, kHidden, nullptr);
        qfn_mmq_q8_0_dense_vec_preq(d_q8, nullptr, d_xq, d_out,
                                    static_cast<int>(rows), 1, kHidden,
                                    nullptr);
      });

  // Native Q4_K route: moe_vec with one expert.
  auto q4 = MakeQ4K(rows, 0x2222);
  void* d_q4 = nullptr;
  void* d_q6 = nullptr;
  std::int32_t* d_ids = nullptr;
  Check(hipMalloc(&d_q4, q4.size()), "q4 malloc");
  Check(hipMalloc(&d_ids, 64 * sizeof(std::int32_t)), "ids malloc");
  Check(hipMemcpy(d_q4, q4.data(), q4.size(), hipMemcpyHostToDevice),
        "q4 upload");
  std::vector<std::int32_t> ids(64, 0);
  Check(hipMemcpy(d_ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice),
        "ids");
  const double t_q4 =
      Seconds(iters, [&] {
        qfn_mmq_moe_vec(12 /* GGML type q4_K */, d_q4, d_x, d_ids, d_out,
                        static_cast<int>(rows), kHidden, 1, 1, 1, nullptr);
      });
  // Also the Q8_0 weights through moe_vec (kernel choice comparison at
  // equal bytes, relevant for the LM head).
  const double t_q8m =
      Seconds(iters, [&] {
        qfn_mmq_moe_vec(8 /* q8_0 */, d_q8, d_x, d_ids, d_out,
                        static_cast<int>(rows), kHidden, 1, 1, 1, nullptr);
      });
  // Q6_K moe_vec: the LM head and routed down format of the Q4_K_M artifact.
  auto q6 = MakeQ6K(rows, 0x3333);
  Check(hipMalloc(&d_q6, q6.size()), "q6 malloc");
  Check(hipMemcpy(d_q6, q6.data(), q6.size(), hipMemcpyHostToDevice),
        "q6 upload");
  const double t_q6 =
      Seconds(iters, [&] {
        qfn_mmq_moe_vec(14 /* q6_K */, d_q6, d_x, d_ids, d_out,
                        static_cast<int>(rows), kHidden, 1, 1, 1, nullptr);
      });

  std::printf(
      "%-26s rows=%6zu  q8_0: %7.3f ms (%5.0f GB/s)  q4_K moe_vec: %7.3f ms "
      "(%5.0f GB/s)  q8_0 moe_vec: %7.3f ms (%5.0f GB/s)  q6_K moe_vec: "
      "%7.3f ms (%5.0f GB/s)\n",
      name, rows, t_q8 * 1e3, q8_bytes / t_q8 / 1e9, t_q4 * 1e3,
      q4_bytes / t_q4 / 1e9, t_q8m * 1e3, q8_bytes / t_q8m / 1e9, t_q6 * 1e3,
      q6_bytes / t_q6 / 1e9);
  Check(hipFree(d_q8), "free");
  Check(hipFree(d_xq), "free");
  Check(hipFree(d_q4), "free");
  Check(hipFree(d_q6), "free");
  Check(hipFree(d_ids), "free");
  Check(hipFree(d_x), "free");
  Check(hipFree(d_out), "free");
}

/// Validates the E=1 native GEMV paths against a CPU F32 reference over
/// real artifact weights (files extracted beside the bench). `q6` selects
/// the Q6_K layout (210-byte blocks) instead of Q4_K (144-byte blocks).
void Validate(const char* name, const char* path, std::size_t rows,
              bool gated_pair = false, const char* up_path = nullptr,
              bool q6 = false) {
  std::FILE* fh = std::fopen(path, "rb");
  if (fh == nullptr) {
    std::printf("  %-22s SKIPPED (no %s)\n", name, path);
    return;
  }
  const std::size_t row_bytes = kHidden / 256 * (q6 ? 210 : 144);
  std::vector<std::uint8_t> w(rows * row_bytes);
  std::fread(w.data(), 1, w.size(), fh);
  std::fclose(fh);
  std::vector<std::uint8_t> up_w;
  if (gated_pair) {
    fh = std::fopen(up_path, "rb");
    up_w.resize(rows * row_bytes);
    std::fread(up_w.data(), 1, up_w.size(), fh);
    std::fclose(fh);
  }
  std::uint32_t seed = 7;
  std::vector<float> x(kHidden);
  for (auto& v : x) {
    v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 65536.0F;
  }
  // CPU reference over the first 64 rows.
  const std::size_t check = std::min<std::size_t>(rows, 64);
  std::vector<float> ref(check);
  std::vector<float> deq(kHidden);
  for (std::size_t r = 0; r < check; ++r) {
    if (q6) {
      gufo::quant::DequantizeQ6_K(w.data() + r * row_bytes, deq.data(),
                                  kHidden);
    } else {
      gufo::quant::DequantizeQ4_K(w.data() + r * row_bytes, deq.data(),
                                  kHidden);
    }
    double acc = 0;
    for (std::size_t i = 0; i < kHidden; ++i) {
      acc += static_cast<double>(deq[i]) * x[i];
    }
    ref[r] = static_cast<float>(acc);
    if (gated_pair) {
      gufo::quant::DequantizeQ4_K(up_w.data() + r * row_bytes, deq.data(),
                                  kHidden);
      double up = 0;
      for (std::size_t i = 0; i < kHidden; ++i) {
        up += static_cast<double>(deq[i]) * x[i];
      }
      const double g = ref[r];
      ref[r] = static_cast<float>(g / (1.0 + std::exp(-g)) * up);
    }
  }
  void* d_w = nullptr;
  void* d_up = nullptr;
  std::int32_t* d_ids = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  Check(hipMalloc(&d_w, w.size()), "val malloc");
  Check(hipMalloc(&d_x, x.size() * 4), "val x");
  Check(hipMalloc(&d_out, rows * 4), "val out");
  Check(hipMalloc(&d_ids, 64 * 4), "val ids");
  Check(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice), "val up");
  Check(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "val x");
  if (gated_pair) {
    Check(hipMalloc(&d_up, up_w.size()), "val up malloc");
    Check(hipMemcpy(d_up, up_w.data(), up_w.size(), hipMemcpyHostToDevice),
          "val up");
  }
  Check(hipMemset(d_ids, 0, 64 * 4), "val ids");
  const int type_id = q6 ? 14 : 12;
  if (gated_pair) {
    qfn_mmq_moe_gated_vec(type_id, d_w, d_up, d_x, d_ids, d_out,
                          static_cast<int>(rows), kHidden, 1, 1, 1, nullptr);
  } else {
    qfn_mmq_moe_vec(type_id, d_w, d_x, d_ids, d_out, static_cast<int>(rows),
                    kHidden, 1, 1, 1, nullptr);
  }
  std::vector<float> got(rows);
  Check(hipMemcpy(got.data(), d_out, rows * 4, hipMemcpyDeviceToHost),
        "val down");
  double worst = 0;
  double peak = 0;
  for (std::size_t r = 0; r < check; ++r) {
    peak = std::max(peak, std::abs(static_cast<double>(ref[r])));
  }
  std::size_t bad_row = 0;
  for (std::size_t r = 0; r < check; ++r) {
    const double e = std::abs(static_cast<double>(ref[r] - got[r]));
    if (std::abs(static_cast<double>(ref[r])) > 0.05 * peak &&
        e / peak > worst) {
      worst = e / peak;
      bad_row = r;
    }
  }
  std::printf("  %-22s worst error vs row peak: %g\n", name, worst);
  if (worst > 2e-2) {
    std::printf("  FAILED: ref[0]=%f got[0]=%f\n", ref[0], got[0]);
  }
  Check(hipFree(d_w), "vf");
  Check(hipFree(d_x), "vf");
  Check(hipFree(d_out), "vf");
  Check(hipFree(d_ids), "vf");
  if (d_up != nullptr) {
    Check(hipFree(d_up), "vf");
  }
}

/// Validates one batched (multi-token) moe_vec call against the CPU F32
/// reference: distinct activations per token row, expert 0 everywhere, so
/// the y-dim token batching and per-token strides are both exercised.
void ValidateBatched(const char* name, const char* path, std::size_t rows,
                     std::uint32_t tokens, bool q6) {
  std::FILE* fh = std::fopen(path, "rb");
  if (fh == nullptr) {
    std::printf("  %-22s SKIPPED (no %s)\n", name, path);
    return;
  }
  const std::size_t row_bytes = kHidden / 256 * (q6 ? 210 : 144);
  std::vector<std::uint8_t> w(rows * row_bytes);
  std::fread(w.data(), 1, w.size(), fh);
  std::fclose(fh);
  std::uint32_t seed = 11;
  std::vector<float> x(tokens * kHidden);
  for (auto& v : x) {
    v = (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 65536.0F;
  }
  const std::size_t check = std::min<std::size_t>(rows, 64);
  std::vector<float> ref(tokens * check);
  std::vector<float> deq(kHidden);
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::size_t r = 0; r < check; ++r) {
      if (q6) {
        gufo::quant::DequantizeQ6_K(w.data() + r * row_bytes, deq.data(),
                                    kHidden);
      } else {
        gufo::quant::DequantizeQ4_K(w.data() + r * row_bytes, deq.data(),
                                    kHidden);
      }
      double acc = 0;
      for (std::size_t i = 0; i < kHidden; ++i) {
        acc += static_cast<double>(deq[i]) * x[t * kHidden + i];
      }
      ref[t * check + r] = static_cast<float>(acc);
    }
  }
  void* d_w = nullptr;
  std::int32_t* d_ids = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  Check(hipMalloc(&d_w, w.size()), "vb malloc");
  Check(hipMalloc(&d_x, x.size() * 4), "vb x");
  Check(hipMalloc(&d_out, tokens * rows * 4), "vb out");
  Check(hipMalloc(&d_ids, tokens * 4), "vb ids");
  Check(hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice), "vb up");
  Check(hipMemcpy(d_x, x.data(), x.size() * 4, hipMemcpyHostToDevice), "vb x");
  Check(hipMemset(d_ids, 0, tokens * 4), "vb ids");
  qfn_mmq_moe_vec(q6 ? 14 : 12, d_w, d_x, d_ids, d_out, static_cast<int>(rows),
                  kHidden, static_cast<int>(tokens), 1, 1, nullptr);
  std::vector<float> got(tokens * rows);
  Check(hipMemcpy(got.data(), d_out, tokens * rows * 4, hipMemcpyDeviceToHost),
        "vb down");
  double peak = 0;
  for (float v : ref) {
    peak = std::max(peak, std::abs(static_cast<double>(v)));
  }
  double worst = 0;
  for (std::uint32_t t = 0; t < tokens; ++t) {
    for (std::size_t r = 0; r < check; ++r) {
      if (std::abs(static_cast<double>(ref[t * check + r])) <= 0.05 * peak) {
        continue;
      }
      worst = std::max(worst,
                       std::abs(static_cast<double>(ref[t * check + r]) -
                                static_cast<double>(got[t * rows + r])) /
                           peak);
    }
  }
  std::printf("  %-22s tokens=%u worst error vs row peak: %g\n", name, tokens,
              worst);
  if (worst > 2e-2) {
    std::printf("  FAILED: ref[0]=%f got[0]=%f\n", ref[0], got[0]);
  }
  Check(hipFree(d_w), "vbf");
  Check(hipFree(d_x), "vbf");
  Check(hipFree(d_out), "vbf");
  Check(hipFree(d_ids), "vbf");
}

}  // namespace

int main() {
  if (qfn_mmq_init(0) != 0) {
    std::fprintf(stderr, "qfn_mmq init failed\n");
    return 1;
  }
  Validate("attn_q moe_vec E=1", "val_attn_q.bin", 8192);
  Validate("shexp gated_vec E=1", "val_shgate.bin", 512, true, "val_shup.bin");
  Validate("head q6_K moe_vec E=1", "val_head_q6.bin", 8192, false, nullptr,
           true);
  ValidateBatched("head q6_K moe_vec w5", "val_head_q6.bin", 8192, 5, true);
  ValidateBatched("attn_q moe_vec w5", "val_attn_q.bin", 8192, 5, false);
  const std::uint32_t iters = 300;
  const double kb = kHidden;
  Run("ssm qkv|gate", 12288, iters, 12288 * kb * 34 / 32,
      12288 * kb * 144 / 256, 12288 * kb * 210 / 256);
  Run("ssm_out", 2048, iters, 2048 * kb * 34 / 32, 2048 * kb * 144 / 256,
      2048 * kb * 210 / 256);
  Run("attn q|g|k|v", 9216, iters, 9216 * kb * 34 / 32,
      9216 * kb * 144 / 256, 9216 * kb * 210 / 256);
  Run("shexp pair", 512, iters, 512 * kb * 34 / 32, 512 * kb * 144 / 256,
      512 * kb * 210 / 256);
  Run("LM head", 248320, 100, 248320 * kb * 34 / 32,
      248320 * kb * 144 / 256, 248320 * kb * 210 / 256);
  return 0;
}
