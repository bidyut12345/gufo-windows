#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
<<<<<<< HEAD
namespace qk = gufo::models::qwen38_flash_next;
=======
>>>>>>> origin/main
namespace qfn = gufo::models::qwen38_flash_next;
namespace {

// The model's full-attention geometry: 24 query heads over 2 KV heads,
// head_dim 256, 4-token selection blocks.
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
// The per-token kernel accumulates in FP32 over FP16 keys/values; its gated
// output (order one) stays within this of an FP64 evaluation.
constexpr double kFp64Limit = 1e-6;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    CheckHip(hipMemset(allocation, 0, bytes()), "hipMemset");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

template<typename T>
void Upload(HipBuffer<T>* destination, const std::vector<T>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

std::vector<unsigned char> DownloadBytes(HipBuffer<unsigned char>* source) {
  std::vector<unsigned char> values(source->bytes());
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download bytes");
  return values;
}

/// Host mirror of the kernels' planar Q8_0 row quantization: F16 scale =
/// amax/127 per 32 elements, codes rounded to nearest even.
void QuantizeRowQ8_0Host(const float* src, unsigned char* dst,
                         std::uint32_t row_dim) {
  const std::size_t scales = row_dim / 32;
  auto* codes = reinterpret_cast<signed char*>(dst);
  auto* scale_out = reinterpret_cast<__half*>(dst + row_dim);
  for (std::size_t b = 0; b < scales; ++b) {
    float amax = 0.0F;
    for (std::uint32_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::fabs(src[b * 32 + j]));
    }
    const float id = amax > 0.0F ? 127.0F / amax : 0.0F;
    for (std::uint32_t j = 0; j < 32; ++j) {
      codes[b * 32 + j] =
          static_cast<signed char>(std::nearbyint(src[b * 32 + j] * id));
    }
    scale_out[b] = __float2half_rn(amax / 127.0F);
  }
}

/// Decodes a planar Q8_0 row back to floats, optionally rounding through F16
/// (the WMMA loaders dequantize into F16 registers).
std::vector<float> DequantRowQ8_0Host(const unsigned char* src,
                                      std::uint32_t row_dim, bool round_f16) {
  std::vector<float> out(row_dim);
  const auto* codes = reinterpret_cast<const signed char*>(src);
  const auto* scales = reinterpret_cast<const __half*>(src + row_dim);
  for (std::uint32_t i = 0; i < row_dim; ++i) {
    const float value =
        static_cast<float>(codes[i]) * __half2float(scales[i / 32]);
    out[i] = round_f16 ? __half2float(__float2half_rn(value)) : value;
  }
  return out;
}

/// Host mirror of the kernels' planar Q4_K row quantization: GGML's
/// quantize_row_q4_K_ref machinery (per-32 sub-block weighted least-squares
/// scale search, 6-bit scale/min packing, code requantization through the
/// stored F16 scales) in the kernels' canonical XOR-butterfly reduction
/// order, every op single-rounded IEEE so the bytes match the __f*_rn
/// device code.
float Q4KButterflySum32(const float* v) {
  float cur[32];
  std::memcpy(cur, v, sizeof(cur));
  float next[32];
  for (int offset = 16; offset > 0; offset >>= 1) {
    for (int i = 0; i < 32; ++i)
      next[i] = cur[i] + cur[i ^ offset];
    std::memcpy(cur, next, sizeof(cur));
  }
  return cur[0];
}

int Q4KNearestIntHost(float fval) {
  const float val = fval + 12582912.0F;
  int i;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

/// One 32-element sub-block: the reference's make_qkx2_quants(32, 15,
/// rmin=-1, rdelta=0.1, nstep=20). `min_out` is the stored min (>= 0, the
/// negated clamped block minimum); the search's min stays live across
/// candidate iterations exactly like the reference.
void Q4KSubBlockQuantHost(const float* x, float* scale_out, float* min_out) {
  // Mean square (not rms) matches the kernels: see Q4KSubBlockQuant.
  float sumsq[32];
  for (int i = 0; i < 32; ++i)
    sumsq[i] = x[i] * x[i];
  const float mean_sq = Q4KButterflySum32(sumsq) * (1.0F / 32.0F);
  float w[32];
  for (int i = 0; i < 32; ++i)
    w[i] = mean_sq + std::fabs(x[i]);

  float mn = x[0], mx = x[0];
  float sw[32], sx[32];
  for (int i = 0; i < 32; ++i) {
    mn = std::min(mn, x[i]);
    mx = std::max(mx, x[i]);
    sw[i] = w[i];
    sx[i] = w[i] * x[i];
  }
  const float sum_w = Q4KButterflySum32(sw);
  const float sum_x = Q4KButterflySum32(sx);
  if (mn > 0.0F)
    mn = 0.0F;
  if (mx == mn) {
    *scale_out = 0.0F;
    *min_out = -mn;
    return;
  }
  const float range = mx - mn;
  const float iscale0 = 15.0F / range;
  float scale = 1.0F / iscale0;
  float err[32];
  for (int i = 0; i < 32; ++i) {
    const int l = std::clamp(Q4KNearestIntHost(iscale0 * (x[i] - mn)), 0, 15);
    float diff = scale * static_cast<float>(l) + mn - x[i];
    diff = diff * diff;
    err[i] = w[i] * diff;
  }
  float best_error = Q4KButterflySum32(err);
  for (int is = 0; is <= 20; ++is) {
    // The reference re-divides by the live (mx - mn): an accepted candidate
    // re-bases every later candidate's range.
    const float iscale =
        (-1.0F + 0.1F * static_cast<float>(is) + 15.0F) / (mx - mn);
    float sl[32], sl2[32], sxl[32];
    for (int i = 0; i < 32; ++i) {
      const int l = std::clamp(Q4KNearestIntHost(iscale * (x[i] - mn)), 0, 15);
      const float lf = static_cast<float>(l);
      const float wl = w[i] * lf;
      sl[i] = wl;
      sl2[i] = wl * lf;
      sxl[i] = wl * x[i];
    }
    const float sum_l = Q4KButterflySum32(sl);
    const float sum_l2 = Q4KButterflySum32(sl2);
    const float sum_xl = Q4KButterflySum32(sxl);
    const float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0.0F) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0.0F) {
        this_min = 0.0F;
        this_scale = sum_xl / sum_l2;
      }
      for (int i = 0; i < 32; ++i) {
        const int l =
            std::clamp(Q4KNearestIntHost(iscale * (x[i] - mn)), 0, 15);
        float diff = this_scale * static_cast<float>(l) + this_min - x[i];
        diff = diff * diff;
        err[i] = w[i] * diff;
      }
      const float cur_error = Q4KButterflySum32(err);
      if (cur_error < best_error) {
        best_error = cur_error;
        scale = this_scale;
        mn = this_min;
      }
    }
  }
  *scale_out = scale;
  *min_out = -mn;
}

/// Unpacks one super-block's 6-bit sub-block scale/min pair (GGML
/// get_scale_min_k4 over the 12 bytes trailing the two F16 scales).
void Q4KUnpackScalePair(const unsigned char* scales, std::uint32_t j,
                        std::uint8_t* sc6, std::uint8_t* m6) {
  const auto* q = scales + 4;
  if (j < 4) {
    *sc6 = q[j] & 63;
    *m6 = q[j + 4] & 63;
  } else {
    *sc6 = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m6 = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/// Planar Q4_K row: [row_dim/2 code bytes][row_dim/256 * 16 scale bytes];
/// code nibble pairs are interleaved (byte 2l low = element 2l, high =
/// 2l+1) and codes dequantize to code * (d * sc) - dmin * m by one FMA.
void QuantizeRowQ4_KHost(const float* src, unsigned char* dst,
                         std::uint32_t row_dim) {
  const std::size_t code_bytes = row_dim / 2;
  for (std::size_t s = 0; s < row_dim / 256; ++s) {
    const float* x = src + s * 256;
    auto* codes = dst + s * 128;
    auto* scales = dst + code_bytes + s * 16;
    float sub_scale[8], sub_min[8];
    for (int j = 0; j < 8; ++j) {
      Q4KSubBlockQuantHost(x + 32 * j, &sub_scale[j], &sub_min[j]);
    }
    float max_scale = 0.0F, max_min = 0.0F;
    for (int j = 0; j < 8; ++j) {
      max_scale = std::max(max_scale, sub_scale[j]);
      max_min = std::max(max_min, sub_min[j]);
    }
    const float inv_scale = max_scale > 0.0F ? 63.0F / max_scale : 0.0F;
    const float inv_min = max_min > 0.0F ? 63.0F / max_min : 0.0F;
    std::uint8_t ls[8], lm[8];
    for (int j = 0; j < 8; ++j) {
      ls[j] = static_cast<std::uint8_t>(
          std::min(63, Q4KNearestIntHost(inv_scale * sub_scale[j])));
      lm[j] = static_cast<std::uint8_t>(
          std::min(63, Q4KNearestIntHost(inv_min * sub_min[j])));
    }
    __half d = __float2half_rn(max_scale / 63.0F);
    __half dmin = __float2half_rn(max_min / 63.0F);
    // Floor at the smallest normal F16 like the kernels: a flushed-to-zero
    // scale would collapse its super-block on dequant.
    std::uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, &d, sizeof(d_bits));
    std::memcpy(&dmin_bits, &dmin, sizeof(dmin_bits));
    if (max_scale > 0.0F && d_bits < 0x0400) {
      d_bits = 0x0400;
      std::memcpy(&d, &d_bits, sizeof(d));
    }
    if (max_min > 0.0F && dmin_bits < 0x0400) {
      dmin_bits = 0x0400;
      std::memcpy(&dmin, &dmin_bits, sizeof(dmin));
    }
    std::uint8_t packed[12] = {};
    for (int j = 0; j < 4; ++j) {
      packed[j] = ls[j];
      packed[j + 4] = lm[j];
    }
    for (int j = 4; j < 8; ++j) {
      packed[j + 4] =
          static_cast<std::uint8_t>((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
      packed[j - 4] |= static_cast<std::uint8_t>((ls[j] >> 4) << 6);
      packed[j] |= static_cast<std::uint8_t>((lm[j] >> 4) << 6);
    }
    std::memcpy(scales, &d, 2);
    std::memcpy(scales + 2, &dmin, 2);
    std::memcpy(scales + 4, packed, 12);
    for (int j = 0; j < 8; ++j) {
      std::uint8_t sc6, m6;
      Q4KUnpackScalePair(scales, j, &sc6, &m6);
      const float dsc = __half2float(d) * sc6;
      const float dm = __half2float(dmin) * m6;
      int codes32[32];
      for (int i = 0; i < 32; ++i) {
        int l = 0;
        if (dsc > 0.0F) {
          l = std::clamp(Q4KNearestIntHost((x[32 * j + i] + dm) / dsc), 0, 15);
        }
        codes32[i] = l;
      }
      auto* out = codes + 16 * j;
      for (int i = 0; i < 16; ++i) {
        out[i] = static_cast<std::uint8_t>(codes32[2 * i] |
                                           (codes32[2 * i + 1] << 4));
      }
    }
  }
}

/// Decodes a planar Q4_K row back to floats, optionally rounding through F16
/// (the WMMA loaders dequantize into F16 registers).
std::vector<float> DequantRowQ4_KHost(const unsigned char* src,
                                      std::uint32_t row_dim, bool round_f16) {
  std::vector<float> out(row_dim);
  const unsigned char* codes = src;
  const unsigned char* scales = src + row_dim / 2;
  for (std::uint32_t i = 0; i < row_dim; ++i) {
    const std::size_t sb = i / 256;
    const std::uint32_t e = i % 256;
    const auto* sc = scales + sb * 16;
    std::uint8_t sc6, m6;
    Q4KUnpackScalePair(sc, e / 32, &sc6, &m6);
    const float d = __half2float(*reinterpret_cast<const __half*>(sc)) * sc6;
    const float dm =
        __half2float(*reinterpret_cast<const __half*>(sc + 2)) * m6;
    const int nib = (codes[sb * 128 + e / 2] >> ((e % 2) * 4)) & 0xF;
    const float value = std::fma(static_cast<float>(nib), d, -dm);
    out[i] = round_f16 ? __half2float(__float2half_rn(value)) : value;
  }
  return out;
}

void CheckPreparation(std::uint32_t n, std::uint32_t start,
                      std::uint32_t rotary, const qfn::YarnRope& yarn = {}) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t count = static_cast<std::size_t>(n) * kQWidth;
  const std::size_t cache_rows = start + n + 2;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> q_ref(count), gate_ref(count), q_out(count), gate_out(count);
  HipBuffer<float> k(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<float> v(static_cast<std::size_t>(n) * kKvWidth);
  HipBuffer<__half> k_ref(cache_rows * kKvWidth), v_ref(cache_rows * kKvWidth);
  HipBuffer<__half> k_out(cache_rows * kKvWidth), v_out(cache_rows * kKvWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);

  struct Capture {
    hipStream_t stream{nullptr};
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    Capture() {
      CheckHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
               "preparation stream");
    }
    ~Capture() {
      if (exec)
        (void)hipGraphExecDestroy(exec);
      if (graph)
        (void)hipGraphDestroy(graph);
      (void)hipStreamDestroy(stream);
    }
  } capture;
  const auto same = [&](auto* expected, auto* actual, std::size_t size,
                        const char* name, std::uint32_t replay) {
    using T = std::remove_pointer_t<decltype(expected)>;
    std::vector<T> a(size), b(size);
    CheckHip(
        hipMemcpy(a.data(), expected, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation reference");
    CheckHip(
        hipMemcpy(b.data(), actual, size * sizeof(T), hipMemcpyDeviceToHost),
        "preparation result");
    for (std::size_t i = 0; i < size; ++i) {
      if (std::memcmp(&a[i], &b[i], sizeof(T)) == 0)
        continue;
      std::cerr << "preparation n=" << n << " start=" << start
                << " rotary=" << rotary << " replay=" << replay
                << " index=" << i << " expected=" << std::hexfloat
                << static_cast<float>(a[i])
                << " actual=" << static_cast<float>(b[i]) << std::defaultfloat
                << '\n';
      throw std::runtime_error(std::string("attention preparation changed ") +
                               name);
    }
  };
  for (std::uint32_t replay = 0; replay < 2; ++replay) {
    const std::vector<std::uint32_t> position{start + replay};
    Upload(&pos, position);
    q::UnpackQGate(packed.get(), stride, q_ref.get(), gate_ref.get(), k.get(),
                   v.get(), n, kHeads, kDim, kKvWidth, nullptr);
    q::RmsNormRows(q_ref.get(), q_gamma.get(), q_ref.get(), n * kHeads, kDim, 1,
                   1e-6F, nullptr);
    q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1,
                   1e-6F, nullptr);
    q::Rope(q_ref.get(), n, kHeads, kDim, rotary, pos.get(), 1e7F, nullptr,
            nullptr, yarn);
    q::Rope(k.get(), n, kKvHeads, kDim, rotary, pos.get(), 1e7F, nullptr,
            nullptr, yarn);
    q::StoreKv(k.get(), k_ref.get(), n, kKvWidth, pos.get(), nullptr);
    q::StoreKv(v.get(), v_ref.get(), n, kKvWidth, pos.get(), nullptr);
    CheckHip(hipDeviceSynchronize(), "preparation reference ready");
    if (replay == 0) {
      CheckHip(
          hipStreamBeginCapture(capture.stream, hipStreamCaptureModeGlobal),
          "preparation capture");
      const bool supported = q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
<<<<<<< HEAD
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream, nullptr, false,
          qfn::KvDtypes{}, yarn);
=======
          rotary, pos.get(), 1e7F, 1e-6F, capture.stream, nullptr, false, yarn);
>>>>>>> origin/main
      CheckHip(hipStreamEndCapture(capture.stream, &capture.graph),
               "preparation capture end");
      if (!supported)
        throw std::runtime_error("attention preparation rejected geometry");
      CheckHip(hipGraphInstantiate(&capture.exec, capture.graph, nullptr,
                                   nullptr, 0),
               "preparation instantiate");
    }
    CheckHip(hipGraphLaunch(capture.exec, capture.stream),
             "preparation replay");
    CheckHip(hipStreamSynchronize(capture.stream), "preparation ready");
    same(q_ref.get(), q_out.get(), count, "queries", replay);
    same(gate_ref.get(), gate_out.get(), count, "gates", replay);
    // Include the neighboring cache rows and replay at a new device position.
    const std::size_t first = start == 0 ? 0 : start - 1;
    const std::size_t window = (cache_rows - first) * kKvWidth;
    same(k_ref.get() + first * kKvWidth, k_out.get() + first * kKvWidth, window,
         "key cache", replay);
    same(v_ref.get() + first * kKvWidth, v_out.get() + first * kKvWidth, window,
         "value cache", replay);
  }
  if (n >= 32) {
    // Each small prefill piece must match the full preparation, including
    // normalization rounding and absolute rotary positions.
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U, 32U}) {
      for (std::uint32_t off = 0; off < n; off += chunk) {
        Upload(&pos, std::vector<std::uint32_t>{start + 1 + off});
        if (!q::PrepareAttention(
                packed.get() + std::size_t(off) * stride, stride, q_gamma.get(),
                k_gamma.get(), q_out.get() + std::size_t(off) * kQWidth,
                gate_out.get() + std::size_t(off) * kQWidth, k_out.get(),
                v_out.get(), std::min(chunk, n - off), kHeads, kKvHeads, kDim,
<<<<<<< HEAD
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true,
                qfn::KvDtypes{}, yarn))
=======
                rotary, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, true, yarn))
>>>>>>> origin/main
          throw std::runtime_error("prefill preparation rejected a short tail");
      }
      CheckHip(hipDeviceSynchronize(), "short prefill preparation");
      same(q_ref.get(), q_out.get(), count, "prefill queries", chunk);
      same(gate_ref.get(), gate_out.get(), count, "prefill gates", chunk);
      same(k_ref.get(), k_out.get(), cache_rows * kKvWidth, "prefill keys",
           chunk);
      same(v_ref.get(), v_out.get(), cache_rows * kKvWidth, "prefill values",
           chunk);
    }
  }
  std::cout << "attention preparation n=" << n << " start=" << start
            << " rotary=" << rotary << " yarn=" << yarn.enabled
            << ": exact, including graph positions\n";
}

qfn::YarnRope Yarn(float factor) {
  return qfn::MakeYarnRope({.factor = factor, .original_context = 262144}, 64,
                           1e7F);
}

// HF frequency of pair i in double (same ramp, 1/factor blend).
double HfFrequency(float factor, std::uint32_t i) {
  const double extrapolation = std::pow(1e7, -2.0 * i / 64.0);
  if (factor == 1.0F)
    return extrapolation;
  const double ramp = std::clamp((double(i) - 14.0) / 8.0, 0.0, 1.0);
  return extrapolation * (1.0 - ramp + ramp / factor);
}

// Unit input per pair: the Rope kernel's output is mscale * (cos, sin) of
// the angle, so angle and scale are checked directly against HF.
void CheckPhase(const std::vector<float>& out, const std::vector<double>& pos,
                float factor, const char* label) {
  const double mscale = factor == 1.0F ? 1.0 : 0.1 * std::log(factor) + 1.0;
  for (std::uint32_t i = 0; i < 32; ++i) {
    const double angle = pos[i] * HfFrequency(factor, i);
    const double got = std::atan2(out[i + 32], out[i]);
    const double diff = std::remainder(got - angle, 2.0 * std::numbers::pi);
    const double magnitude = std::hypot(out[i], out[i + 32]);
    if (std::abs(diff) > 1e-6 * angle + 1e-4 ||
        std::abs(magnitude - mscale) > 1e-5 * mscale) {
      throw std::runtime_error(
          std::string(label) + ": pair " + std::to_string(i) + " angle error " +
          std::to_string(diff) + " magnitude " + std::to_string(magnitude));
    }
  }
  for (std::uint32_t i = 64; i < kDim; ++i)
    if (out[i] != 0.25F)
      throw std::runtime_error(std::string(label) + ": tail changed");
}

std::vector<float> UnitPairs() {
  std::vector<float> host(kDim, 0.0F);
  std::fill_n(host.begin(), 32, 1.0F);
  std::fill(host.begin() + 64, host.end(), 0.25F);
  return host;
}

void CheckYarnPhase(float factor, std::uint32_t position) {
  HipBuffer<float> x(kDim);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, UnitPairs());
  Upload(&pos, std::vector<std::uint32_t>{position});
  q::Rope(x.get(), 1, 1, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
          factor == 1.0F ? qfn::YarnRope{} : Yarn(factor));
  CheckHip(hipDeviceSynchronize(), "phase rope");
  CheckPhase(Download(&x, kDim), std::vector<double>(32, position), factor,
             "rope phase");
  std::cout << "rope phase factor=" << factor << " pos=" << position
            << ": HF angle and mscale\n";
}

// mRoPE: pair i reads T, H or W exactly as qwen::vision::RopePosition does.
void CheckYarnVisionPhase(float factor) {
  const std::vector<std::int32_t> thw{300000, 200000, 250000};
  HipBuffer<std::int32_t> positions(3);
  CheckHip(hipMemcpy(positions.get(), thw.data(), 12, hipMemcpyHostToDevice),
           "vision positions");
  const gufo::models::qwen::vision::DeviceRope host_rope{positions.get(), 1, 0};
  HipBuffer<gufo::models::qwen::vision::DeviceRope> rope(1);
  CheckHip(hipMemcpy(rope.get(), &host_rope, sizeof(host_rope),
                     hipMemcpyHostToDevice),
           "vision rope");
  HipBuffer<float> x(kDim);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, UnitPairs());
  Upload(&pos, std::vector<std::uint32_t>{0});
  q::Rope(x.get(), 1, 1, kDim, 64, pos.get(), 1e7F, nullptr, rope.get(),
          Yarn(factor));
  CheckHip(hipDeviceSynchronize(), "vision phase rope");
  std::vector<double> coordinate(32);
  for (std::uint32_t i = 0; i < 32; ++i) {
    const int axis = i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0;
    coordinate[i] = thw[axis];
  }
  CheckPhase(Download(&x, kDim), coordinate, factor, "vision rope phase");
  std::cout << "vision rope phase factor=" << factor << ": per-axis YaRN\n";
}

// Relative RMS of the rotated dims, GPU Rope kernel versus cpu::Rope.
double OracleRms(float factor, std::uint32_t position) {
  const auto yarn = factor == 1.0F ? qfn::YarnRope{} : Yarn(factor);
  const auto input = MakeValues(std::size_t(kHeads) * kDim, 0x5EED, 1.0F);
  HipBuffer<float> x(input.size());
  HipBuffer<std::uint32_t> pos(1);
  Upload(&x, input);
  Upload(&pos, std::vector<std::uint32_t>{position});
  q::Rope(x.get(), 1, kHeads, kDim, 64, pos.get(), 1e7F, nullptr, nullptr,
          yarn);
  const auto gpu = Download(&x, input.size());
  auto cpu = input;
  qfn::cpu::Rope(cpu.data(), kHeads, kDim, 64, position, 1e7F, yarn);
  double num = 0.0, den = 0.0;
  for (std::uint32_t h = 0; h < kHeads; ++h)
    for (std::uint32_t i = 0; i < 64; ++i) {
      const auto k = std::size_t(h) * kDim + i;
      num += (double(gpu[k]) - cpu[k]) * (double(gpu[k]) - cpu[k]);
      den += double(cpu[k]) * cpu[k];
    }
  return std::sqrt(num / den);
}

void CheckOracleRms() {
  for (const std::uint32_t position : {1000U, 300000U, 650000U}) {
    const double off = OracleRms(1.0F, position);
    const double on = OracleRms(2.5F, position);
    std::cout << "rope vs CPU oracle pos=" << position << ": relative RMS off "
              << off << " on " << on << '\n';
    if ((position == 1000 && on > 1e-4) || on > 2.0 * off + 1e-5)
      throw std::runtime_error("YaRN rope drifts from the CPU oracle");
  }
}

/// FP64 gated attention over the keys each row can see: every key below
/// pos + 1, or with a mask the selected complete blocks plus the incomplete
/// tail block. An independent formula for the per-token kernel. Cache
/// elements decode through `DecodeCache` (__half or float).
inline float DecodeCache(__half value) {
  return __half2float(value);
}
inline float DecodeCache(float value) {
  return value;
}

template<typename KCache, typename VCache>
std::vector<double> Fp64Attention(
    const std::vector<float>& qv, const std::vector<float>& gate,
    const KCache& kh, const VCache& vh, const std::uint32_t* mask,
    std::uint32_t mask_words, std::uint32_t n_tokens, std::uint32_t start_pos,
    std::uint32_t ratio = kRatio) {
  std::vector<double> out(static_cast<std::size_t>(n_tokens) * kQWidth);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kDim));
  std::vector<std::uint32_t> keys;
  std::vector<double> scores;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t n_kv = start_pos + t + 1;
    const std::uint32_t tail_start = n_kv / ratio * ratio;
    keys.clear();
    for (std::uint32_t j = 0; j < n_kv; ++j) {
      const std::uint32_t b = j / ratio;
      if (mask == nullptr || j >= tail_start ||
          ((mask[static_cast<std::size_t>(t) * mask_words + b / 32] >>
            (b % 32)) &
           1U) != 0) {
        keys.push_back(j);
      }
    }
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      const std::size_t row = (static_cast<std::size_t>(t) * kHeads + h) * kDim;
      const std::size_t kv_head = (h / (kHeads / kKvHeads)) * kDim;
      scores.assign(keys.size(), 0.0);
      double maximum = -INFINITY;
      for (std::size_t k = 0; k < keys.size(); ++k) {
        const std::size_t base = keys[k] * std::size_t{kKvWidth} + kv_head;
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          dot += static_cast<double>(qv[row + d]) *
                 static_cast<double>(DecodeCache(kh[base + d]));
        }
        scores[k] = dot * scale;
        maximum = std::max(maximum, scores[k]);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - maximum);
        sum += score;
      }
      for (std::uint32_t d = 0; d < kDim; ++d) {
        double acc = 0.0;
        for (std::size_t k = 0; k < keys.size(); ++k) {
          acc += scores[k] *
                 static_cast<double>(DecodeCache(
                     vh[keys[k] * std::size_t{kKvWidth} + kv_head + d]));
        }
        const double g = static_cast<double>(gate[row + d]);
        out[row + d] = acc / sum / (1.0 + std::exp(-g));
      }
    }
  }
  return out;
}

/// Runs the per-token reference and the fused WMMA route over the same
/// cache and reports the worst absolute error of the gated context.
double Compare(std::uint32_t n_tokens, std::uint32_t start_pos, bool masked,
               std::uint32_t seed, bool sparse = false, bool bounded = false,
               std::uint32_t mask_words_override = 0) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words =
      mask_words_override != 0 ? mask_words_override : (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;

  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  for (std::size_t i = 0; i < kv_count; ++i) {
    kh[i] = __float2half(kf[i]);
    vh[i] = __float2half(vf[i]);
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t& word : mask) {
    word = NextRandom(&state) & NextRandom(&state);
    if (sparse) {
      // About one block in sixteen: most 16-key tiles are unselected for a
      // whole query block, which is what the tile skipping is for.
      word &= NextRandom(&state) & NextRandom(&state);
    }
  }
  if (bounded) {
    std::fill(mask.begin(), mask.end(), 0);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const auto complete = (start_pos + t + 1) / kRatio;
      std::uint32_t selected = 0;
      while (selected < std::min(512U, complete)) {
        const auto block = NextRandom(&state) % complete;
        auto& word =
            mask[static_cast<std::size_t>(t) * mask_words + block / 32];
        const auto bit = 1U << (block % 32);
        if ((word & bit) == 0) {
          word |= bit;
          ++selected;
        }
      }
    }
  }

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_ref(q_count);
  HipBuffer<float> d_wmma(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  const std::uint32_t* mask_ptr = masked ? d_mask.get() : nullptr;

  q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
               d_ref.get(), nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads,
               kDim, kRatio, nullptr);
  q::SigmoidMul(d_ref.get(), d_gate.get(), q_count, nullptr);
  // The split-key form of the reference must match its single-block form
  // (log-sum-exp merge), the way decode runs it.
  {
    constexpr std::uint32_t kSplits = 8;
    HipBuffer<float> d_split(q_count);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                kSplits * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), mask_ptr, mask_words,
                 d_split.get(), d_partials.get(), kSplits, n_tokens,
                 d_pos.get(), kHeads, kKvHeads, kDim, kRatio, nullptr);
    q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "split attention");
    const auto a = Download(&d_ref, q_count);
    const auto b = Download(&d_split, q_count);
    double worst = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      worst = std::max(worst, std::abs(static_cast<double>(a[i] - b[i])));
    }
    std::cout << "  split-key reference worst absolute error " << worst << '\n';
    const auto exact =
        Fp64Attention(qv, gate, kh, vh, masked ? mask.data() : nullptr,
                      mask_words, n_tokens, start_pos);
    double single_error = 0.0;
    double split_error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      single_error = std::max(single_error, std::abs(a[i] - exact[i]));
      split_error = std::max(split_error, std::abs(b[i] - exact[i]));
    }
    std::cout << "  FP64 worst absolute error single " << single_error
              << " split " << split_error << '\n';
    if (std::max(single_error, split_error) > kFp64Limit) {
      throw std::runtime_error("per-token attention exceeds its FP64 limit");
    }
    if (worst > 1e-4) {
      throw std::runtime_error("split-key attention disagrees");
    }
  }
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention rejected the model geometry");
  }
  CheckHip(hipDeviceSynchronize(), "attention synchronization");

  const auto ref = Download(&d_ref, q_count);
  const auto out = Download(&d_wmma, q_count);
  if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                              mask_ptr, mask_words, d_wmma.get(), n_tokens,
                              start_pos, kHeads, kKvHeads, kDim, kRatio,
                              nullptr)) {
    throw std::runtime_error("WMMA attention replay rejected the geometry");
  }
  const auto replay = Download(&d_wmma, q_count);
  if (std::memcmp(out.data(), replay.data(), q_count * sizeof(float)) != 0) {
    throw std::runtime_error("WMMA attention replay changed the output");
  }
  {
    constexpr float poison = -12345.0F;
    Upload(&d_wmma, std::vector<float>(q_count, poison));
    if (!q::WmmaCausalAttention(d_q.get(), d_gate.get(), d_k.get(), d_v.get(),
                                mask_ptr, mask_words, d_wmma.get(), n_tokens,
                                start_pos, kHeads, kKvHeads, kDim, kRatio,
                                nullptr, true)) {
      throw std::runtime_error("last-tile attention rejected the geometry");
    }
    const auto tail = Download(&d_wmma, q_count);
    const std::uint32_t tile_rows = mask_ptr ? 4 : 16;
    const std::size_t begin =
        static_cast<std::size_t>((n_tokens - 1) / tile_rows * tile_rows) *
        kQWidth;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (i < begin ? tail[i] != poison
                    : std::memcmp(&out[i], &tail[i], sizeof(float)) != 0) {
        throw std::runtime_error("last-tile attention changed a row or guard");
      }
    }
  }
  double worst = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(out[i])) {
      throw std::runtime_error("WMMA attention output is not finite");
    }
    worst = std::max(worst, std::abs(static_cast<double>(ref[i] - out[i])));
  }
  if (worst > 1e-2) {
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      for (std::uint32_t h = 0; h < kHeads; ++h) {
        double w = 0.0;
        for (std::uint32_t d = 0; d < kDim; ++d) {
          const std::size_t i =
              (static_cast<std::size_t>(t) * kHeads + h) * kDim + d;
          w = std::max(w, std::abs(static_cast<double>(ref[i] - out[i])));
        }
        if (w > 1e-2) {
          std::cout << "  query " << t << " head " << h << " err " << w << '\n';
        }
      }
    }
  }
  return worst;
}

/// Selected blocks for one query row, given its complete-block count.
using BlockPattern =
    std::function<void(std::uint32_t, std::vector<std::uint32_t>*)>;

/// The per-token kernel against FP64 on a structured selection, in its
/// single-block form and split 1, 3 and 8 ways. The masks are built so the
/// sparse path's 1024-block compaction windows and 64-block key tiles meet
/// their edge cases: tiles that straddle windows, a carried partial tile of
/// every size, a single partial tile, and a ninth tile that wraps a split.
void CheckStructured(const char* name, std::uint32_t n_tokens,
                     std::uint32_t start_pos, const BlockPattern& pattern,
                     std::uint32_t seed, std::uint32_t ratio = kRatio) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + ratio - 1) / ratio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::vector<std::uint32_t> blocks;
  std::size_t selected = 0;
  std::size_t most = 1;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / ratio;
    blocks.clear();
    pattern(complete, &blocks);
    for (const std::uint32_t b : blocks) {
      if (b >= complete) {
        throw std::runtime_error("structured mask selects past the context");
      }
      mask[static_cast<std::size_t>(t) * mask_words + b / 32] |= 1U << (b % 32);
    }
    selected += blocks.size();
    most = std::max(most, blocks.size());
  }
  const auto exact = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                   n_tokens, start_pos, ratio);
  // The list path (ratio 4 only) is compared bitwise with the mask walk. The
  // adversarial patterns select more than the indexer's 512-block budget, so
  // the lists get room for the largest row: that keeps every carry and window
  // edge in play, and the capacity guard is not what is being tested here.
  const std::uint32_t list_capacity = static_cast<std::uint32_t>(most);
  HipBuffer<std::uint32_t> d_lists(static_cast<std::size_t>(n_tokens) *
                                   list_capacity);
  HipBuffer<std::uint32_t> d_counts(n_tokens);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  std::cout << "structured " << name << " ratio=" << ratio << " n=" << n_tokens
            << " start=" << start_pos << " blocks/row=" << selected / n_tokens
            << " FP64 worst absolute error";
  double worst = 0.0;
  for (const std::uint32_t splits : {0U, 1U, 3U, 8U}) {
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                std::max(splits, 1U) * (kDim + 2));
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_out.get(), splits == 0 ? nullptr : d_partials.get(), splits,
                 n_tokens, d_pos.get(), kHeads, kKvHeads, kDim, ratio, nullptr);
    q::SigmoidMul(d_out.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "structured attention");
    const auto out = Download(&d_out, q_count);
    if (ratio == kRatio) {
      HipBuffer<float> d_list(q_count);
      q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                   d_list.get(), splits == 0 ? nullptr : d_partials.get(),
                   splits, n_tokens, d_pos.get(), kHeads, kKvHeads, kDim, ratio,
                   nullptr, d_lists.get(), d_counts.get(), list_capacity);
      q::SigmoidMul(d_list.get(), d_gate.get(), q_count, nullptr);
      CheckHip(hipDeviceSynchronize(), "structured list attention");
      const auto list = Download(&d_list, q_count);
      if (std::memcmp(out.data(), list.data(), q_count * sizeof(float)) != 0) {
        throw std::runtime_error("structured list attention differs from walk");
      }
    }
    double error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(out[i])) {
        throw std::runtime_error("structured attention output is not finite");
      }
      error = std::max(error, std::abs(out[i] - exact[i]));
    }
    std::cout << (splits == 0 ? " single "
                              : " split" + std::to_string(splits) + " ")
              << error;
    worst = std::max(worst, error);
  }
  std::cout << '\n';
  if (worst > kFp64Limit) {
    throw std::runtime_error("structured attention exceeds its FP64 limit");
  }
}

/// The compacted-selection route against the mask walk. Random masks of at
/// most the 512-block budget per row (a short context selects every block):
/// the outputs must match bit for bit, single-block and split eight ways, and
/// stay within the FP64 limit.
void CheckLists(std::uint32_t n_tokens, std::uint32_t start_pos,
                std::uint32_t seed) {
  constexpr std::uint32_t kBudget = 512;
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  std::vector<__half> kh(kv_count);
  std::vector<__half> vh(kv_count);
  {
    const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
    const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
    for (std::size_t i = 0; i < kv_count; ++i) {
      kh[i] = __float2half(kf[i]);
      vh[i] = __float2half(vf[i]);
    }
  }
  std::vector<std::uint32_t> mask(static_cast<std::size_t>(n_tokens) *
                                  mask_words);
  std::uint32_t state = seed ^ 0x77777777U;
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / kRatio;
    std::uint32_t* row = mask.data() + static_cast<std::size_t>(t) * mask_words;
    if (complete <= kBudget) {
      // The selector marks every word of a row that fits its budget.
      std::fill(row, row + mask_words, 0xFFFFFFFFU);
      continue;
    }
    // Rows alternate between the full budget and a random smaller count.
    const std::uint32_t want =
        t % 2 == 0 ? kBudget : 1 + NextRandom(&state) % kBudget;
    for (std::uint32_t selected = 0; selected < want;) {
      const auto block = NextRandom(&state) % complete;
      if ((row[block / 32] & (1U << (block % 32))) == 0) {
        row[block / 32] |= 1U << (block % 32);
        ++selected;
      }
    }
  }
  const auto exact = Fp64Attention(qv, gate, kh, vh, mask.data(), mask_words,
                                   n_tokens, start_pos);

  HipBuffer<float> d_q(q_count);
  HipBuffer<float> d_gate(q_count);
  HipBuffer<__half> d_k(kv_count);
  HipBuffer<__half> d_v(kv_count);
  HipBuffer<std::uint32_t> d_mask(mask.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<std::uint32_t> d_lists(static_cast<std::size_t>(n_tokens) *
                                   kBudget);
  HipBuffer<std::uint32_t> d_counts(n_tokens);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_mask, mask);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});
  std::cout << "lists n=" << n_tokens << " start=" << start_pos;
  for (const std::uint32_t splits : {0U, 8U}) {
    const std::uint32_t z = std::max(splits, 1U);
    HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                                z * (kDim + 2));
    HipBuffer<float> d_walk(q_count);
    HipBuffer<float> d_list(q_count);
    float* partials = splits == 0 ? nullptr : d_partials.get();
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_walk.get(), partials, splits, n_tokens, d_pos.get(), kHeads,
                 kKvHeads, kDim, kRatio, nullptr);
    q::Attention(d_q.get(), d_k.get(), d_v.get(), d_mask.get(), mask_words,
                 d_list.get(), partials, splits, n_tokens, d_pos.get(), kHeads,
                 kKvHeads, kDim, kRatio, nullptr, d_lists.get(), d_counts.get(),
                 kBudget);
    q::SigmoidMul(d_walk.get(), d_gate.get(), q_count, nullptr);
    q::SigmoidMul(d_list.get(), d_gate.get(), q_count, nullptr);
    CheckHip(hipDeviceSynchronize(), "list attention");
    const auto walk = Download(&d_walk, q_count);
    const auto list = Download(&d_list, q_count);
    if (std::memcmp(walk.data(), list.data(), q_count * sizeof(float)) != 0) {
      throw std::runtime_error("list attention differs from the mask walk");
    }
    double error = 0.0;
    for (std::size_t i = 0; i < q_count; ++i) {
      if (!std::isfinite(list[i])) {
        throw std::runtime_error("list attention output is not finite");
      }
      error = std::max(error, std::abs(list[i] - exact[i]));
    }
    std::cout << (splits == 0 ? " single " : " split8 ") << error;
    if (error > kFp64Limit) {
      throw std::runtime_error("list attention exceeds its FP64 limit");
    }
  }
  std::cout << " bitwise-equal\n";
}

void CheckChunks(std::uint32_t n, std::uint32_t split) {
  const std::size_t count = std::size_t(n) * kQWidth;
  HipBuffer<float> queries(count), gates(count), full(count), chunked(count);
  HipBuffer<__half> keys(std::size_t(n) * kKvWidth),
      values(std::size_t(n) * kKvWidth);
  Upload(&queries, MakeValues(count, 412, 4.0F));
  Upload(&gates, MakeValues(count, 721, 3.0F));
  for (auto* destination : {&keys, &values}) {
    const auto f = MakeValues(std::size_t(n) * kKvWidth,
                              destination == &keys ? 891 : 347, 1.0F);
    std::vector<__half> half(f.size());
    std::transform(f.begin(), f.end(), half.begin(),
                   [](float x) { return __float2half(x); });
    Upload(destination, half);
  }
  const auto run = [&](std::uint32_t start, std::uint32_t rows, float* out) {
    const auto offset = std::size_t(start) * kQWidth;
    if (!q::WmmaCausalAttention(queries.get() + offset, gates.get() + offset,
                                keys.get(), values.get(), nullptr, 0,
                                out + offset, rows, start, kHeads, kKvHeads,
                                kDim, kRatio, nullptr)) {
      throw std::runtime_error("chunk attention rejected geometry");
    }
  };
  run(0, n, full.get());
  run(0, split, chunked.get());
  run(split, n - split, chunked.get());
  const auto a = Download(&full, count);
  const auto b = Download(&chunked, count);
  std::size_t differences = 0, first = count;
  float max_error = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (a[i] != b[i]) {
      ++differences;
      first = std::min(first, i);
      max_error = std::max(max_error, std::abs(a[i] - b[i]));
    }
  }
  std::cout << "chunk attention n=" << n << " split=" << split
            << " differing=" << differences << " max=" << max_error
            << " first_row=" << first / kQWidth << '\n';
  if (differences != 0)
    throw std::runtime_error("attention depends on prefill chunk boundary");
}

<<<<<<< HEAD
/// Q8_0 write path: PrepareAttention's cache rows must byte-match the host
/// quantization of the same normalized/rotated F32 values (the F16 case
/// already proves the arithmetic matches StoreKv's inputs).
void CheckQ8Preparation(std::uint32_t n, std::uint32_t start) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t cache_rows = start + n + 2;
  const std::size_t row_bytes = kKvWidth + kKvWidth / 16;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> k(std::size_t{n} * kKvWidth), v(std::size_t{n} * kKvWidth);
  HipBuffer<unsigned char> k_ref(cache_rows * row_bytes),
      v_ref(cache_rows * row_bytes);
  HipBuffer<unsigned char> k_out(cache_rows * row_bytes),
      v_out(cache_rows * row_bytes);
  HipBuffer<float> q_out(std::size_t{n} * kQWidth),
      gate_out(std::size_t{n} * kQWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 17, 4.0F));
  auto qg = MakeValues(kDim, 37, 0.5F);
  auto kg = MakeValues(kDim, 91, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);
  Upload(&pos, std::vector<std::uint32_t>{start});

  // Reference: the same unpack/norm/rope pipeline (V is never normalized),
  // then host quantization.
  q::UnpackQGate(packed.get(), stride, q_out.get(), gate_out.get(), k.get(),
                 v.get(), n, kHeads, kDim, kKvWidth, nullptr);
  q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1, 1e-6F,
                 nullptr);
  q::Rope(k.get(), n, kKvHeads, kDim, 64, pos.get(), 1e7F, nullptr);
  CheckHip(hipDeviceSynchronize(), "q8 preparation reference ready");
  const auto kf = Download(&k, std::size_t{n} * kKvWidth);
  const auto vf = Download(&v, std::size_t{n} * kKvWidth);
  std::vector<unsigned char> k_ref_host(cache_rows * row_bytes, 0),
      v_ref_host(cache_rows * row_bytes, 0);
  for (std::uint32_t t = 0; t < n; ++t) {
    QuantizeRowQ8_0Host(kf.data() + std::size_t{t} * kKvWidth,
                        k_ref_host.data() + (start + t) * row_bytes, kKvWidth);
    QuantizeRowQ8_0Host(vf.data() + std::size_t{t} * kKvWidth,
                        v_ref_host.data() + (start + t) * row_bytes, kKvWidth);
  }
  Upload(&k_ref, k_ref_host);
  Upload(&v_ref, v_ref_host);

  if (!q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          64, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, false,
          qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0})) {
    throw std::runtime_error("q8 preparation rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "q8 preparation ready");
  const auto k_out_host = DownloadBytes(&k_out);
  const auto v_out_host = DownloadBytes(&v_out);
  for (std::size_t i = 0; i < cache_rows * row_bytes; ++i) {
    if (k_out_host[i] != k_ref_host[i]) {
      std::cerr << "q8 key cache row " << i / row_bytes << " byte "
                << i % row_bytes << " expected "
                << static_cast<int>(k_ref_host[i]) << " got "
                << static_cast<int>(k_out_host[i]) << '\n';
      throw std::runtime_error("q8 key cache differs from host quantization");
    }
    if (v_out_host[i] != v_ref_host[i]) {
      std::cerr << "q8 value cache row " << i / row_bytes << " byte "
                << i % row_bytes << " expected "
                << static_cast<int>(v_ref_host[i]) << " got "
                << static_cast<int>(v_out_host[i]) << '\n';
      throw std::runtime_error("q8 value cache differs from host quantization");
    }
  }
  std::cout << "q8 preparation n=" << n << " start=" << start
            << ": cache bytes exact\n";
}

/// Q8_0 read paths: the per-token kernel (single and split) against FP64 over
/// the F32-dequantized cache, and the WMMA route against FP64 over the F16
/// rounded dequantization it computes in registers.
void CheckQ8Attention(std::uint32_t n_tokens, std::uint32_t start_pos,
                      std::uint32_t seed) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const std::size_t row_bytes = kKvWidth + kKvWidth / 16;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<unsigned char> kh(static_cast<std::size_t>(n_kv) * row_bytes, 0);
  std::vector<unsigned char> vh(static_cast<std::size_t>(n_kv) * row_bytes, 0);
  std::vector<float> kd_f32(kv_count), vd_f32(kv_count);
  std::vector<float> kd_f16(kv_count), vd_f16(kv_count);
  for (std::uint32_t r = 0; r < n_kv; ++r) {
    QuantizeRowQ8_0Host(kf.data() + std::size_t{r} * kKvWidth,
                        kh.data() + std::size_t{r} * row_bytes, kKvWidth);
    QuantizeRowQ8_0Host(vf.data() + std::size_t{r} * kKvWidth,
                        vh.data() + std::size_t{r} * row_bytes, kKvWidth);
    const auto k32 = DequantRowQ8_0Host(kh.data() + std::size_t{r} * row_bytes,
                                        kKvWidth, false);
    const auto v32 = DequantRowQ8_0Host(vh.data() + std::size_t{r} * row_bytes,
                                        kKvWidth, false);
    const auto k16 = DequantRowQ8_0Host(kh.data() + std::size_t{r} * row_bytes,
                                        kKvWidth, true);
    const auto v16 = DequantRowQ8_0Host(vh.data() + std::size_t{r} * row_bytes,
                                        kKvWidth, true);
    std::copy(k32.begin(), k32.end(),
              kd_f32.begin() + std::size_t{r} * kKvWidth);
    std::copy(v32.begin(), v32.end(),
              vd_f32.begin() + std::size_t{r} * kKvWidth);
    std::copy(k16.begin(), k16.end(),
              kd_f16.begin() + std::size_t{r} * kKvWidth);
    std::copy(v16.begin(), v16.end(),
              vd_f16.begin() + std::size_t{r} * kKvWidth);
  }

  HipBuffer<float> d_q(q_count), d_gate(q_count);
  HipBuffer<unsigned char> d_k(kh.size()), d_v(vh.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});

  const auto exact32 =
      Fp64Attention(qv, gate, kd_f32, vd_f32, nullptr, 0, n_tokens, start_pos);
  const auto exact16 =
      Fp64Attention(qv, gate, kd_f16, vd_f16, nullptr, 0, n_tokens, start_pos);

  // Per-token kernel: FP32 dequant products, single-block and split forms.
  q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_out.get(),
               nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads, kDim,
               kRatio, nullptr, nullptr, nullptr, 0,
               qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0});
  q::SigmoidMul(d_out.get(), d_gate.get(), q_count, nullptr);
  constexpr std::uint32_t kSplits = 8;
  HipBuffer<float> d_split(q_count);
  HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                              kSplits * (kDim + 2));
  q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_split.get(),
               d_partials.get(), kSplits, n_tokens, d_pos.get(), kHeads,
               kKvHeads, kDim, kRatio, nullptr, nullptr, nullptr, 0,
               qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0});
  q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
  CheckHip(hipDeviceSynchronize(), "q8 per-token attention");
  const auto single = Download(&d_out, q_count);
  const auto split = Download(&d_split, q_count);
  double single_error = 0.0, split_error = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(single[i]) || !std::isfinite(split[i])) {
      throw std::runtime_error("q8 per-token attention output is not finite");
    }
    single_error = std::max(single_error, std::abs(single[i] - exact32[i]));
    split_error = std::max(split_error, std::abs(split[i] - exact32[i]));
  }
  // WMMA route: dequantizes through F16 registers.
  HipBuffer<float> d_wmma(q_count);
  if (!q::WmmaCausalAttention(
          d_q.get(), d_gate.get(), d_k.get(), d_v.get(), nullptr, 0,
          d_wmma.get(), n_tokens, start_pos, kHeads, kKvHeads, kDim, kRatio,
          nullptr, false,
          qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0})) {
    throw std::runtime_error("q8 WMMA attention rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "q8 wmma attention");
  const auto wmma = Download(&d_wmma, q_count);
  double wmma_error = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(wmma[i])) {
      throw std::runtime_error("q8 WMMA attention output is not finite");
    }
    wmma_error = std::max(wmma_error, std::abs(wmma[i] - exact16[i]));
  }
  std::cout << "q8 attention n=" << n_tokens << " start=" << start_pos
            << " fp64 error single " << single_error << " split " << split_error
            << " wmma " << wmma_error << '\n';
  if (std::max(single_error, split_error) > 1e-4) {
    throw std::runtime_error("q8 per-token attention exceeds its FP64 limit");
  }
  if (wmma_error > 1e-4) {
    throw std::runtime_error("q8 WMMA attention exceeds its FP64 limit");
  }
}

/// Q8_0 keys + Q4_K values write path: PrepareAttention's value rows must
/// byte-match the host Q4_K quantization of the same raw F32 values (keys
/// reuse the proven Q8_0 reference).
void CheckQ4KPreparation(std::uint32_t n, std::uint32_t start) {
  constexpr std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  const std::size_t cache_rows = start + n + 2;
  const std::size_t k_row_bytes = kKvWidth + kKvWidth / 16;
  const std::size_t v_row_bytes = kKvWidth / 2 + kKvWidth / 16;
  HipBuffer<float> packed(static_cast<std::size_t>(n) * stride);
  HipBuffer<float> q_gamma(kDim), k_gamma(kDim);
  HipBuffer<float> k(std::size_t{n} * kKvWidth), v(std::size_t{n} * kKvWidth);
  HipBuffer<unsigned char> k_ref(cache_rows * k_row_bytes),
      v_ref(cache_rows * v_row_bytes);
  HipBuffer<unsigned char> k_out(cache_rows * k_row_bytes),
      v_out(cache_rows * v_row_bytes);
  HipBuffer<float> q_out(std::size_t{n} * kQWidth),
      gate_out(std::size_t{n} * kQWidth);
  HipBuffer<std::uint32_t> pos(1);
  Upload(&packed, MakeValues(static_cast<std::size_t>(n) * stride, 23, 4.0F));
  auto qg = MakeValues(kDim, 41, 0.5F);
  auto kg = MakeValues(kDim, 97, 0.5F);
  for (auto& x : qg)
    x += 1.0F;
  for (auto& x : kg)
    x += 1.0F;
  Upload(&q_gamma, qg);
  Upload(&k_gamma, kg);
  Upload(&pos, std::vector<std::uint32_t>{start});

  q::UnpackQGate(packed.get(), stride, q_out.get(), gate_out.get(), k.get(),
                 v.get(), n, kHeads, kDim, kKvWidth, nullptr);
  q::RmsNormRows(k.get(), k_gamma.get(), k.get(), n * kKvHeads, kDim, 1, 1e-6F,
                 nullptr);
  q::Rope(k.get(), n, kKvHeads, kDim, 64, pos.get(), 1e7F, nullptr);
  CheckHip(hipDeviceSynchronize(), "q4k preparation reference ready");
  const auto kf = Download(&k, std::size_t{n} * kKvWidth);
  const auto vf = Download(&v, std::size_t{n} * kKvWidth);
  std::vector<unsigned char> k_ref_host(cache_rows * k_row_bytes, 0),
      v_ref_host(cache_rows * v_row_bytes, 0);
  for (std::uint32_t t = 0; t < n; ++t) {
    QuantizeRowQ8_0Host(kf.data() + std::size_t{t} * kKvWidth,
                        k_ref_host.data() + (start + t) * k_row_bytes,
                        kKvWidth);
    QuantizeRowQ4_KHost(vf.data() + std::size_t{t} * kKvWidth,
                        v_ref_host.data() + (start + t) * v_row_bytes,
                        kKvWidth);
  }
  Upload(&k_ref, k_ref_host);
  Upload(&v_ref, v_ref_host);

  if (!q::PrepareAttention(
          packed.get(), stride, q_gamma.get(), k_gamma.get(), q_out.get(),
          gate_out.get(), k_out.get(), v_out.get(), n, kHeads, kKvHeads, kDim,
          64, pos.get(), 1e7F, 1e-6F, nullptr, nullptr, false,
          qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K})) {
    throw std::runtime_error("q4k preparation rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "q4k preparation ready");
  const auto k_out_host = DownloadBytes(&k_out);
  const auto v_out_host = DownloadBytes(&v_out);
  for (std::size_t i = 0; i < cache_rows * k_row_bytes; ++i) {
    if (k_out_host[i] != k_ref_host[i]) {
      std::cerr << "q4k key cache row " << i / k_row_bytes << " byte "
                << i % k_row_bytes << " expected "
                << static_cast<int>(k_ref_host[i]) << " got "
                << static_cast<int>(k_out_host[i]) << '\n';
      throw std::runtime_error("q4k key cache differs from host quantization");
    }
  }
  // The Q8 key plane is byte-exact (its amax/scale math is deterministic).
  // The Q4_K value search is a 21-candidate weighted least squares whose
  // near-tied acceptances are ulp-sensitive across compilers (device
  // -ffast-math contraction and approximate sqrt against the host), so the
  // value plane is judged per element against the host reference's own
  // quantization step: a candidate flip moves a code by exactly one step.
  double worst_steps = 0.0;
  int worst_t = -1, worst_elem = -1;
  float worst_dev = 0.0f, worst_ref = 0.0f;
  double worst_step = 1.0;
  for (std::uint32_t t = 0; t < n; ++t) {
    const auto* ref_row = v_ref_host.data() + (start + t) * v_row_bytes;
    const auto dev_values = DequantRowQ4_KHost(
        v_out_host.data() + (start + t) * v_row_bytes, kKvWidth, false);
    const auto ref_values = DequantRowQ4_KHost(ref_row, kKvWidth, false);
    for (std::uint32_t i = 0; i < kKvWidth; ++i) {
      const auto* ref_scales = ref_row + kKvWidth / 2 + (i / 256) * 16;
      std::uint8_t sc6 = 0, m6 = 0;
      Q4KUnpackScalePair(ref_scales, (i % 256) / 32, &sc6, &m6);
      const double step = std::max(
          1e-30,
          static_cast<double>(
              __half2float(*reinterpret_cast<const __half*>(ref_scales)) *
              sc6));
      const double ratio =
          std::abs(dev_values[i] - static_cast<double>(ref_values[i])) / step;
      if (ratio > worst_steps) {
        worst_steps = ratio;
        worst_t = static_cast<int>(t);
        worst_elem = static_cast<int>(i);
        worst_dev = dev_values[i];
        worst_ref = ref_values[i];
        worst_step = step;
      }
    }
  }
  if (worst_steps > 1.05) {
    std::cerr << "q4k worst at t=" << worst_t << " elem=" << worst_elem
              << " dev=" << worst_dev << " ref=" << worst_ref
              << " step=" << worst_step << '\n';
    std::cerr << "q4k value cache worst dequant delta " << worst_steps
              << " code steps\n";
    throw std::runtime_error(
        "q4k value cache dequant differs beyond one code step");
  }
  std::cout << "q4k preparation n=" << n << " start=" << start
            << ": keys byte-exact, values worst dequant delta " << worst_steps
            << " code steps\n";
}

/// Q8_0 keys + Q4_K values read paths: the per-token kernel (single and
/// split) against FP64 over the F32-dequantized cache, and the WMMA route
/// against FP64 over the F16 rounded dequantization it computes in
/// registers.
void CheckQ4KAttention(std::uint32_t n_tokens, std::uint32_t start_pos,
                       std::uint32_t seed) {
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::size_t q_count = static_cast<std::size_t>(n_tokens) * kQWidth;
  const std::size_t kv_count = static_cast<std::size_t>(n_kv) * kKvWidth;
  const std::size_t k_row_bytes = kKvWidth + kKvWidth / 16;
  const std::size_t v_row_bytes = kKvWidth / 2 + kKvWidth / 16;
  const auto qv = MakeValues(q_count, seed, 4.0F);
  const auto gate = MakeValues(q_count, seed ^ 0x5555U, 3.0F);
  const auto kf = MakeValues(kv_count, seed ^ 0xAAAAU, 1.0F);
  const auto vf = MakeValues(kv_count, seed ^ 0x3333U, 1.0F);
  std::vector<unsigned char> kh(static_cast<std::size_t>(n_kv) * k_row_bytes,
                                0);
  std::vector<unsigned char> vh(static_cast<std::size_t>(n_kv) * v_row_bytes,
                                0);
  std::vector<float> kd_f32(kv_count), vd_f32(kv_count);
  std::vector<float> kd_f16(kv_count), vd_f16(kv_count);
  for (std::uint32_t r = 0; r < n_kv; ++r) {
    QuantizeRowQ8_0Host(kf.data() + std::size_t{r} * kKvWidth,
                        kh.data() + std::size_t{r} * k_row_bytes, kKvWidth);
    QuantizeRowQ4_KHost(vf.data() + std::size_t{r} * kKvWidth,
                        vh.data() + std::size_t{r} * v_row_bytes, kKvWidth);
    const auto k32 = DequantRowQ8_0Host(
        kh.data() + std::size_t{r} * k_row_bytes, kKvWidth, false);
    const auto v32 = DequantRowQ4_KHost(
        vh.data() + std::size_t{r} * v_row_bytes, kKvWidth, false);
    const auto k16 = DequantRowQ8_0Host(
        kh.data() + std::size_t{r} * k_row_bytes, kKvWidth, true);
    const auto v16 = DequantRowQ4_KHost(
        vh.data() + std::size_t{r} * v_row_bytes, kKvWidth, true);
    std::copy(k32.begin(), k32.end(),
              kd_f32.begin() + std::size_t{r} * kKvWidth);
    std::copy(v32.begin(), v32.end(),
              vd_f32.begin() + std::size_t{r} * kKvWidth);
    std::copy(k16.begin(), k16.end(),
              kd_f16.begin() + std::size_t{r} * kKvWidth);
    std::copy(v16.begin(), v16.end(),
              vd_f16.begin() + std::size_t{r} * kKvWidth);
  }

  HipBuffer<float> d_q(q_count), d_gate(q_count);
  HipBuffer<unsigned char> d_k(kh.size()), d_v(vh.size());
  HipBuffer<std::uint32_t> d_pos(1);
  HipBuffer<float> d_out(q_count);
  Upload(&d_q, qv);
  Upload(&d_gate, gate);
  Upload(&d_k, kh);
  Upload(&d_v, vh);
  Upload(&d_pos, std::vector<std::uint32_t>{start_pos});

  const auto exact32 =
      Fp64Attention(qv, gate, kd_f32, vd_f32, nullptr, 0, n_tokens, start_pos);
  const auto exact16 =
      Fp64Attention(qv, gate, kd_f16, vd_f16, nullptr, 0, n_tokens, start_pos);

  q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_out.get(),
               nullptr, 1, n_tokens, d_pos.get(), kHeads, kKvHeads, kDim,
               kRatio, nullptr, nullptr, nullptr, 0,
               qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K});
  q::SigmoidMul(d_out.get(), d_gate.get(), q_count, nullptr);
  constexpr std::uint32_t kSplits = 8;
  HipBuffer<float> d_split(q_count);
  HipBuffer<float> d_partials(static_cast<std::size_t>(n_tokens) * kHeads *
                              kSplits * (kDim + 2));
  q::Attention(d_q.get(), d_k.get(), d_v.get(), nullptr, 0, d_split.get(),
               d_partials.get(), kSplits, n_tokens, d_pos.get(), kHeads,
               kKvHeads, kDim, kRatio, nullptr, nullptr, nullptr, 0,
               qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K});
  q::SigmoidMul(d_split.get(), d_gate.get(), q_count, nullptr);
  CheckHip(hipDeviceSynchronize(), "q4k per-token attention");
  const auto single = Download(&d_out, q_count);
  const auto split = Download(&d_split, q_count);
  double single_error = 0.0, split_error = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(single[i]) || !std::isfinite(split[i])) {
      throw std::runtime_error("q4k per-token attention output is not finite");
    }
    single_error = std::max(single_error, std::abs(single[i] - exact32[i]));
    split_error = std::max(split_error, std::abs(split[i] - exact32[i]));
  }
  // WMMA route: dequantizes through F16 registers.
  HipBuffer<float> d_wmma(q_count);
  if (!q::WmmaCausalAttention(
          d_q.get(), d_gate.get(), d_k.get(), d_v.get(), nullptr, 0,
          d_wmma.get(), n_tokens, start_pos, kHeads, kKvHeads, kDim, kRatio,
          nullptr, false,
          qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K})) {
    throw std::runtime_error("q4k WMMA attention rejected geometry");
  }
  CheckHip(hipDeviceSynchronize(), "q4k wmma attention");
  const auto wmma = Download(&d_wmma, q_count);
  double wmma_error = 0.0;
  for (std::size_t i = 0; i < q_count; ++i) {
    if (!std::isfinite(wmma[i])) {
      throw std::runtime_error("q4k WMMA attention output is not finite");
    }
    wmma_error = std::max(wmma_error, std::abs(wmma[i] - exact16[i]));
  }
  std::cout << "q4k attention n=" << n_tokens << " start=" << start_pos
            << " fp64 error single " << single_error << " split " << split_error
            << " wmma " << wmma_error << '\n';
  if (std::max(single_error, split_error) > 1e-4) {
    throw std::runtime_error("q4k per-token attention exceeds its FP64 limit");
  }
  if (wmma_error > 1e-4) {
    throw std::runtime_error("q4k WMMA attention exceeds its FP64 limit");
  }
}

void CheckWmmaLiveExtent() {
  HipBuffer<float> qv(kQWidth), gate(kQWidth), out(kQWidth);
  HipBuffer<__half> k(kKvWidth), v(kKvWidth);
  HipBuffer<std::uint32_t> mask(3200);
  const auto accepts = [&](std::uint32_t n, std::uint32_t start,
                           std::uint32_t words) {
    return q::WmmaCausalAttention(qv.get(), gate.get(), k.get(), v.get(),
                                  mask.get(), words, out.get(), n, start,
                                  kHeads, kKvHeads, kDim, kRatio, nullptr);
  };
  // 262,141 + 4 visible tokens need 65,537 blocks = 2,049 words.
  if (accepts(4, 262141, 3200))
    throw std::runtime_error("WMMA accepted a live extent past 2048 words");
  // A row stride narrower than the live extent is a caller error.
  if (accepts(4, 131069, 1024))
    throw std::runtime_error("WMMA accepted a stride below the live extent");
  std::cout << "WMMA live extent: rejects past 262,144 visible tokens\n";
}

}  // namespace

int main() {
  try {
    // The quantized cache cases run first: the F16 query-preparation cases
    // below hit this toolchain's pre-existing 1-ULP drift (fails identically
    // on an unmodified build) and would otherwise abort before them.
    CheckQ4KPreparation(8, 4096);
    CheckQ4KPreparation(65, 131069);
    CheckQ4KAttention(4, 4096, 0x51ED0011U);
    CheckQ4KAttention(17, 2047, 0x51ED0012U);
    CheckQ4KAttention(3, 90000, 0x51ED0013U);
    CheckQ8Preparation(8, 4096);
    CheckQ8Preparation(65, 131069);
    CheckQ8Attention(4, 4096, 0x51ED0001U);
    CheckQ8Attention(17, 2047, 0x51ED0002U);
    CheckQ8Attention(3, 90000, 0x51ED0003U);
    CheckChunks(136, 94);
    CheckChunks(2048, 1025);
<<<<<<< HEAD
    // The wide-batch sparse cases run before the F16 query-preparation cases
    // below, which hit this toolchain's pre-existing 1-ULP drift and would
    // otherwise abort before them.
    struct Case {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
      bool masked;
      bool sparse;
      bool bounded{false};
    };
    const Case cases[] = {
        {4, 4096, false, false},      {3, 9000, true, true},
        {100, 0, false, false},       {64, 37, false, false},
        {100, 0, true, false},        {77, 51, true, false},
        {96, 4096, true, true},       {70, 8000, true, true},
        {512, 2048, true, true},      {1024, 2048, true, true, true},
        {5, 32765, true, true, true}, {7, 65533, true, true, true},
        {5, 131069, true, true},      {7, 131069, true, true, true},
        {17, 2047, true, true, true}};
    std::uint32_t seed = 0x1234ABCDU;
    for (const Case& c : cases) {
      const double worst = Compare(c.n_tokens, c.start_pos, c.masked, seed++,
                                   c.sparse, c.bounded);
      std::cout << "WMMA attention n=" << c.n_tokens << " start=" << c.start_pos
                << (c.masked ? (c.sparse ? " sparse" : " masked") : " dense")
                << " worst absolute error " << worst << '\n';
      // The reference accumulates FP32 probabilities; the WMMA route rounds
      // Q and P to FP16, so the contract is a small absolute envelope on
      // values of order one.
      if (worst >= 2e-2) {
        throw std::runtime_error("WMMA attention exceeds its error envelope");
      }
    }
    CheckWmmaLiveExtent();
    CheckPreparation(1, 0, 64);
    CheckPreparation(8, 4096, 64);
    CheckPreparation(65, 131069, 64);
    CheckPreparation(7, 131069, 256);
    CheckPreparation(8, 4096, 64, Yarn(1.5625F));
    CheckPreparation(65, 300000, 64, Yarn(2.5F));
    CheckYarnPhase(1.0F, 262143);  // off: sincosf accuracy at native depth
    for (const float factor : {1.5625F, 2.5F, 4.0F})
      for (const std::uint32_t position : {1000U, 300000U, 650000U})
        CheckYarnPhase(factor, position);
    CheckYarnVisionPhase(2.5F);
    CheckOracleRms();
    // Structured selections against the sparse path's 1024-block compaction
    // windows and 64-block key tiles. At 131069 the last 512 blocks sit in
    // the final window; 262141 spans all 64 windows of the native context,
    // so 63 blocks per window carry every partial-tile size once.
    constexpr std::uint32_t kWindow = 1024;
    const auto spread = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t j = 0; j < count && j < complete; ++j) {
          out->push_back(static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(j) * complete / count));
        }
      };
    };
    const auto per_window = [](std::uint32_t count) {
      return [count](std::uint32_t complete, std::vector<std::uint32_t>* out) {
        for (std::uint32_t w0 = 0; w0 < complete; w0 += kWindow) {
          for (std::uint32_t j = 0; j < count; ++j) {
            const std::uint32_t b = w0 + j * 16 + (w0 / kWindow) % 16;
            if (b < complete) {
              out->push_back(b);
            }
          }
        }
      };
    };
    const BlockPattern last_budget = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = complete - std::min(512U, complete); b < complete;
           ++b) {
        out->push_back(b);
      }
    };
    const BlockPattern one_window = [](std::uint32_t complete,
                                       std::vector<std::uint32_t>* out) {
      for (std::uint32_t j = 0; j < 64 && 7 * kWindow + j * 16 < complete;
           ++j) {
        out->push_back(7 * kWindow + j * 16);
      }
    };
    struct Structured {
      const char* name;
      std::uint32_t start_pos;
      BlockPattern pattern;
    };
    const Structured structured[] = {
        {"last-512", 131069, last_budget},
        {"63-per-window", 262141, per_window(63)},
        {"1-per-window", 262141, per_window(1)},
        {"40-total", 131069, spread(40)},
        {"64-in-one-window", 131069, one_window},
        {"513-total", 131069, spread(513)},
        {"576-total", 131069, spread(576)},
        // A YaRN-scaled capacity (409600 at factor 1.5625, 3200 mask words):
        // chunks past the native 262144 tokens take this same kernel.
        {"63-per-window-past-native", 409597, per_window(63)},
    };
    // Compacted selections: a short context that selects every block, then
    // the budgeted rows at increasing depth, up to the 160K decode target.
    std::uint32_t list_seed = 0x115700U;
    for (const std::uint32_t depth : {1500U, 8192U, 32768U, 163840U}) {
      for (const std::uint32_t n : {1U, 3U, 8U}) {
        CheckLists(n, depth, list_seed++);
      }
    }
    std::uint32_t structured_seed = 0x5EED0000U;
    for (const Structured& c : structured) {
      for (const std::uint32_t n : {1U, 3U}) {
        CheckStructured(c.name, n, c.start_pos, c.pattern, structured_seed++);
      }
    }
    // At ratio 1 a tile is 256 blocks, so partial tiles reach 231 blocks,
    // more than the room kept for a ratio-4 carry. Alternating 999 and 100
    // selected blocks per window attends some partial tiles in their window
    // and carries others.
    const BlockPattern alternating = [](std::uint32_t complete,
                                        std::vector<std::uint32_t>* out) {
      for (std::uint32_t b = 0; b < complete; ++b) {
        if (b % kWindow < ((b / kWindow) % 2 == 0 ? 999U : 100U)) {
          out->push_back(b);
        }
      }
    };
    for (const std::uint32_t n : {1U, 3U}) {
      CheckStructured("ratio-1-alternating", n, 8189, alternating,
                      structured_seed++, 1);
    }
    struct Case2 {
      std::uint32_t n_tokens;
      std::uint32_t start_pos;
      bool masked;
      bool sparse;
      bool bounded{false};
    };
    const Case2 cases2[] = {
        {4, 4096, false, false},      {3, 9000, true, true},
        {100, 0, false, false},       {64, 37, false, false},
        {100, 0, true, false},        {77, 51, true, false},
        {96, 4096, true, true},       {70, 8000, true, true},
        {5, 32765, true, true, true}, {7, 65533, true, true, true},
        {5, 131069, true, true},      {7, 131069, true, true, true},
        {17, 2047, true, true, true}};
    bool ok = true;
    std::uint32_t seed2 = 0x1234ABCDU;
    for (const Case2& c : cases2) {
      const double worst = Compare(c.n_tokens, c.start_pos, c.masked, seed2++,
                                   c.sparse, c.bounded);
      std::cout << "WMMA attention n=" << c.n_tokens << " start=" << c.start_pos
                << (c.masked ? (c.sparse ? " sparse" : " masked") : " dense")
                << " worst absolute error " << worst << '\n';
      // The reference accumulates FP32 probabilities; the WMMA route rounds
      // Q and P to FP16, so the contract is a small absolute envelope on
      // values of order one.
      ok = ok && worst < 2e-2;
    }
    // A capacity wider than 262,144 (3,200 words at 409,600) keeps chunks
    // whose visible extent fits 2,048 words on the fused kernel.
    for (const auto& [n, start] :
         {std::pair{5U, 131069U}, std::pair{4U, 262140U}}) {
      const double worst = Compare(n, start, true, seed2++, true, true, 3200);
      std::cout << "WMMA attention n=" << n << " start=" << start
                << " stride=3200 worst absolute error " << worst << '\n';
      ok = ok && worst < 2e-2;
    }
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

void CheckWmmaLiveExtent() {
  HipBuffer<float> qv(kQWidth), gate(kQWidth), out(kQWidth);
  HipBuffer<__half> k(kKvWidth), v(kKvWidth);
  HipBuffer<std::uint32_t> mask(3200);
  const auto accepts = [&](std::uint32_t n, std::uint32_t start,
                           std::uint32_t words) {
    return q::WmmaCausalAttention(qv.get(), gate.get(), k.get(), v.get(),
                                  mask.get(), words, out.get(), n, start,
                                  kHeads, kKvHeads, kDim, kRatio, nullptr);
  };
  // 262,141 + 4 visible tokens need 65,537 blocks = 2,049 words.
  if (accepts(4, 262141, 3200))
    throw std::runtime_error("WMMA accepted a live extent past 2048 words");
  // A row stride narrower than the live extent is a caller error.
  if (accepts(4, 131069, 1024))
    throw std::runtime_error("WMMA accepted a stride below the live extent");
  std::cout << "WMMA live extent: rejects past 262,144 visible tokens\n";
}
