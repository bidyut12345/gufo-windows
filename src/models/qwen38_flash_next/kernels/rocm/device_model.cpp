#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/hip/managed_alloc.hpp"
#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// Round-to-nearest-even float to half, matching the kernels' conversion.
std::uint16_t Fp32ToF16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000;
  const std::uint32_t mantissa = bits & 0x7FFFFF;
  const int exponent = static_cast<int>((bits >> 23) & 0xFF);
  if (exponent == 0xFF) {
    return static_cast<std::uint16_t>(sign | (mantissa != 0 ? 0x7E00 : 0x7C00));
  }
  const int unbiased = exponent - 127 + 15;
  if (unbiased >= 0x1F) {
    return static_cast<std::uint16_t>(sign | 0x7C00);
  }
  if (unbiased <= 0) {
    if (unbiased < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t whole = mantissa | 0x800000;
    const int shift = 14 - unbiased;
    const std::uint32_t half = whole >> shift;
    const std::uint32_t remainder = whole & ((1u << shift) - 1);
    const std::uint32_t middle = 1u << (shift - 1);
    return static_cast<std::uint16_t>(
        sign + ((remainder > middle || (remainder == middle && (half & 1) != 0))
                    ? 1u
                    : 0u));
  }
  std::uint32_t half =
      sign | (static_cast<std::uint32_t>(unbiased) << 10) | (mantissa >> 13);
  const std::uint32_t remainder = mantissa & 0x1FFF;
  if (remainder > 0x1000 || (remainder == 0x1000 && (half & 1) != 0)) {
    ++half;
  }
  return static_cast<std::uint16_t>(half);
}

float Fp16ToF32(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
  const std::uint32_t exponent = (value >> 10) & 0x1F;
  const std::uint32_t mantissa = value & 0x3FF;
  if (exponent == 0) {
    const auto bits =
        sign | std::bit_cast<std::uint32_t>(mantissa * 5.960464477539063e-8f);
    return std::bit_cast<float>(bits);
  }
  if (exponent == 0x1F) {
    return std::bit_cast<float>(sign | 0x7F800000u |
                                (mantissa != 0 ? 0x400000u : 0u));
  }
  return std::bit_cast<float>(sign | ((exponent + 112u) << 23) |
                              (mantissa << 13));
}

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

#pragma pack(push, 1)
/// Q4_0 block: fp16 scale + 16 packed nibbles (QK=32). Quantized values are
/// zero-based codes; the kernels subtract the 8 offset in the dot product.
struct BlockQ4_0 {
  std::uint16_t d;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 must be 18 bytes");
#pragma pack(pop)

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
  /// Expands Q8_0 weights to F32 instead of narrowing F32 to F16.
  bool expand_q8{false};
  std::size_t source_bytes{0};
};

/// The trunk and the optional MTP block as one traversal's output.
struct Uploaded {
  DeviceTensor token_embd;
  DeviceTensor output;
  DeviceMixer hc_head;
  std::vector<DeviceLayer> layers;
  DeviceLayer mtp;
  DeviceTensor output_draft;
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  const std::vector<core::GgufMappedRegion>& shards;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  std::uint64_t requested{0};

  [[nodiscard]] bool Measuring() const noexcept { return weights == nullptr; }

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  [[nodiscard]] hipError_t Allocate(void** pointer, std::size_t size) {
    requested += size;
    if (Measuring()) {
      return hipSuccess;
    }
    return weights->Allocate(pointer, size);
  }

  /// Host-dequantizes a Q6_K tensor to Q8_0 rows and uploads the result.
  /// The LM head is the only expected caller: no production kernel reads
  /// Q6_K in place, and a Q8_0 copy routes the head through the fast dense
  /// tiers (DenseF16Gemm wide batches, the Q8 vector path at decode) where
  /// an F16 copy would fall to the workspace-free hipBLASLt GEMM, which has
  /// no competent Tensile kernel for the head shape on gfx1151.
  DeviceTensor CopyDequantQ8_0(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.type != core::GgmlType::kQ6_K) {
      Fail("dequant upload needs Q6_K: " + std::string(t.name));
      return d;
    }
    if (t.cols % 256 != 0) {
      Fail("Q6_K dequant upload needs 256-aligned columns: " +
           std::string(t.name));
      return d;
    }
    const std::size_t dst_bytes =
        t.rows * (t.cols / 32 * sizeof(quant::block_q8_0));
    void* ptr = nullptr;
    if (Allocate(&ptr, dst_bytes + kTailMargin) != hipSuccess) {
      Fail("Q8_0 upload allocation failed for " + std::string(t.name));
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += dst_bytes + kTailMargin;
    const auto& region = shards[t.shard];
    const auto* src =
        static_cast<const std::uint8_t*>(region.data) + t.file_offset;
    const std::size_t row_src_bytes = t.cols / 256 * 210;
    const std::size_t row_dst_bytes = t.cols / 32 * sizeof(quant::block_q8_0);
    constexpr std::size_t kChunkRows = 4096;
    const std::size_t chunk_dst_bytes = kChunkRows * row_dst_bytes;
    std::vector<std::uint8_t> staged(chunk_dst_bytes);
    const std::uint32_t workers =
        std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));

    for (std::uint32_t chunk_start = 0; chunk_start < t.rows;
         chunk_start += kChunkRows) {
      const std::uint32_t chunk_rows = static_cast<std::uint32_t>(
          std::min<std::size_t>(kChunkRows, t.rows - chunk_start));
      const std::size_t current_chunk_bytes = chunk_rows * row_dst_bytes;
      std::vector<std::thread> threads;
      for (std::uint32_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
          std::vector<float> chunk(t.cols);
          const std::uint32_t begin = static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(chunk_rows) * w / workers);
          const std::uint32_t end = static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(chunk_rows) * (w + 1) / workers);
          for (std::uint32_t r = begin; r < end; ++r) {
            const std::uint32_t row = chunk_start + r;
            quant::DequantizeQ6_K(src + row * row_src_bytes, chunk.data(),
                                  t.cols);
            auto* out = reinterpret_cast<quant::block_q8_0*>(
                staged.data() + static_cast<std::size_t>(r) * row_dst_bytes);
            for (std::size_t b = 0; b < t.cols / 32; ++b) {
              const float* v = chunk.data() + b * 32;
              float amax = 0.0f;
              for (std::size_t j = 0; j < 32; ++j) {
                amax = std::max(amax, std::fabs(v[j]));
              }
              const float dblock = amax / 127.0f;
              const float id = dblock != 0.0f ? 1.0f / dblock : 0.0f;
              out[b].d = Fp32ToF16(dblock);
              for (std::size_t j = 0; j < 32; ++j) {
                out[b].qs[j] = static_cast<std::int8_t>(std::roundf(v[j] * id));
              }
            }
          }
        });
      }
      for (auto& thread : threads) {
        thread.join();
      }
      if (hipMemcpy(static_cast<std::uint8_t*>(ptr) +
                        static_cast<std::size_t>(chunk_start) * row_dst_bytes,
                    staged.data(), current_chunk_bytes,
                    hipMemcpyHostToDevice) != hipSuccess) {
        Fail("Q8_0 upload failed for " + std::string(t.name));
        return d;
      }
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + dst_bytes, 0,
                         kTailMargin, nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    return d;
  }

  /// Host-requantizes a Q8_0 tensor to Q4_0 rows and uploads the result.
  /// The draft head is the only expected caller: its logits only rank
  /// proposals the target verifies exactly, so the coarser format costs
  /// nothing but the per-pass weight bytes, which halve.
  DeviceTensor RequantQ4_0(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.type != core::GgmlType::kQ8_0) {
      Fail("Q4_0 requant needs Q8_0: " + std::string(t.name));
      return d;
    }
    if (t.cols % 32 != 0) {
      Fail("Q4_0 requant needs 32-aligned columns: " + std::string(t.name));
      return d;
    }
    const auto& region = shards[t.shard];
    const auto* src =
        static_cast<const std::uint8_t*>(region.data) + t.file_offset;
    const std::size_t row_src_bytes = t.cols / 32 * sizeof(quant::block_q8_0);
    const std::size_t row_dst_bytes = t.cols / 32 * sizeof(BlockQ4_0);
    const std::size_t dst_bytes = t.rows * row_dst_bytes;
    void* ptr = nullptr;
    if (Allocate(&ptr, dst_bytes + kTailMargin) != hipSuccess) {
      Fail("Q4_0 upload allocation failed for " + std::string(t.name));
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += dst_bytes + kTailMargin;
    constexpr std::size_t kChunkRows = 4096;
    const std::size_t chunk_dst_bytes = kChunkRows * row_dst_bytes;
    std::vector<std::uint8_t> staged(chunk_dst_bytes);
    const std::uint32_t workers =
        std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));

    for (std::uint32_t chunk_start = 0; chunk_start < t.rows;
         chunk_start += kChunkRows) {
      const std::uint32_t chunk_rows = static_cast<std::uint32_t>(
          std::min<std::size_t>(kChunkRows, t.rows - chunk_start));
      const std::size_t current_chunk_bytes = chunk_rows * row_dst_bytes;
      std::vector<std::thread> threads;
      for (std::uint32_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
          const std::uint32_t begin = static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(chunk_rows) * w / workers);
          const std::uint32_t end = static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(chunk_rows) * (w + 1) / workers);
          for (std::uint32_t r = begin; r < end; ++r) {
            const std::uint32_t row = chunk_start + r;
            const auto* in = reinterpret_cast<const quant::block_q8_0*>(
                src + static_cast<std::size_t>(row) * row_src_bytes);
            auto* out = reinterpret_cast<BlockQ4_0*>(
                staged.data() + static_cast<std::size_t>(r) * row_dst_bytes);
            for (std::size_t b = 0; b < t.cols / 32; ++b) {
              const float dblock = Fp16ToF32(in[b].d);
              float v[32];
              for (std::size_t j = 0; j < 32; ++j) {
                v[j] = dblock * static_cast<float>(in[b].qs[j]);
              }
              float amax = 0.0f;
              for (std::size_t j = 0; j < 32; ++j) {
                amax = std::max(amax, std::fabs(v[j]));
              }
              const float scale = amax / 7.0f;
              const float id = scale != 0.0f ? 1.0f / scale : 0.0f;
              out[b].d = Fp32ToF16(scale);
              for (std::size_t j = 0; j < 16; ++j) {
                const int lo = std::clamp(
                    static_cast<int>(std::lroundf(v[j] * id)) + 8, 0, 15);
                const int hi = std::clamp(
                    static_cast<int>(std::lroundf(v[j + 16] * id)) + 8, 0, 15);
                out[b].qs[j] = static_cast<std::uint8_t>(lo | (hi << 4));
              }
            }
          }
        });
      }
      for (auto& thread : threads) {
        thread.join();
      }
      if (hipMemcpy(static_cast<std::uint8_t*>(ptr) +
                        static_cast<std::size_t>(chunk_start) * row_dst_bytes,
                    staged.data(), current_chunk_bytes,
                    hipMemcpyHostToDevice) != hipSuccess) {
        Fail("Q4_0 upload failed for " + std::string(t.name));
        return d;
      }
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + dst_bytes, 0,
                         kTailMargin, nullptr);
    d.data = ptr;
    d.type = core::GgmlType::kQ4_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    return d;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Uploads a narrow projection the quantized tier does not cover (the
  /// hyper-connection inject) as F32. A Q8_0 tensor is uploaded as is and
  /// expanded once the disk pipeline drains, so it runs the F32 kernels with
  /// its dequantized values instead of quantizing their activations.
  DeviceTensor CopyF32(const TensorRef& t) {
    if (t.empty() || !ok || t.type != core::GgmlType::kQ8_0) {
      return Copy(t);
    }
    DeviceTensor d = Copy(t);
    const std::size_t count = t.cols * t.rows * t.experts;
    void* f32 = nullptr;
    if (!ok ||
        Allocate(&f32, count * sizeof(float) + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name));
      return {};
    }
    if (Measuring()) {
      return {};
    }
    allocations.push_back(f32);
    bytes += count * sizeof(float) + kTailMargin;
    conversions.push_back({d.data, f32, count, true, t.SizeBytes()});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(f32) + count * 4, 0,
                         kTailMargin, nullptr);
    d.data = f32;
    d.type = core::GgmlType::kF32;
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || (!Measuring() && !stager.Finish(error))) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (Allocate(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      if (Measuring()) {
        continue;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    if (Measuring()) {
      return;
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV, unless `expand_q8` asks for the F32 stack's F16
  /// route over its dequantized values.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts,
                     bool expand_q8 = false) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (Allocate(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (!Measuring()) {
      allocations.push_back(ptr);
      bytes += size + kTailMargin;
      std::size_t offset = 0;
      for (const TensorRef* t : parts) {
        if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(ptr) + offset, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += t->SizeBytes();
      }
    }
    if (type == core::GgmlType::kQ8_0 && !expand_q8) {
      if (Measuring()) {
        return d;
      }
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    void* f32 = ptr;
    std::size_t f32_bytes = size;
    if (type == core::GgmlType::kQ8_0) {
      const std::size_t elements = rows * (*parts.begin())->cols;
      f32_bytes = elements * sizeof(float);
      if (Allocate(&f32, f32_bytes + kTailMargin) != hipSuccess) {
        Fail("stacked tensor allocation failed");
        return d;
      }
      if (!Measuring()) {
        allocations.push_back(f32);
        bytes += f32_bytes + kTailMargin;
        conversions.push_back({ptr, f32, elements, true, size});
      }
    }
    const std::size_t count = f32_bytes / sizeof(float);
    void* half = nullptr;
    if (Allocate(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("stacked tensor allocation failed");
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({f32, half, count, false, f32_bytes});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), Copy(m.down), Copy(m.up), CopyF32(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta}, true);
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = Copy(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  Uploaded Model(const ModelWeights& w, const MtpWeights* mtp,
                 std::uint32_t trunk_shards) {
    Uploaded u;
    u.token_embd = Copy(w.token_embd);
    u.output = w.output.data == w.token_embd.data
                   ? u.token_embd
                   : (w.output.type == core::GgmlType::kQ6_K
                          ? CopyDequantQ8_0(w.output)
                          : Copy(w.output));
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
    }
    if (mtp != nullptr) {
      // The sidecar has its own shard index; reuse the target's readers and
      // staging pool.
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
      // The draft head only ranks proposals the target verifies exactly, so
      // a Q4_0 copy halves its weight bytes at no quality cost.
      u.output_draft = w.output.type == core::GgmlType::kQ8_0
                           ? RequantQ4_0(w.output)
                           : DeviceTensor{};
    }
    return u;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

bool DeviceModel::SetDraftVocab(std::uint32_t prefix_rows,
                                std::vector<std::int32_t> specials,
                                std::string* error_msg) {
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr)
      *error_msg = message;
    return false;
  };
  if (output_.empty() || output_.rows == 0 || output_bytes_ % output_.rows != 0)
    return fail("draft vocabulary needs a row-addressable output head");
  if (prefix_rows == 0 || prefix_rows >= output_.rows)
    return fail("draft vocabulary prefix must leave rows out");
  for (const auto id : specials) {
    if (id < static_cast<std::int32_t>(prefix_rows) ||
        static_cast<std::uint32_t>(id) >= output_.rows)
      return fail("draft special tokens must lie above the prefix");
  }
  // The prefix is the head's own leading rows: no copy.
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  draft_specials_ = std::move(specials);
  return true;
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The routed expert kernels cover Q4_K, Q5_K, Q5_1, Q8_0, IQ3_S and
  // IQ4_XS. The LM head may be Q6_K: it is host-dequantized to Q8_0 instead,
  // which keeps it on the fast dense-GEMM tiers.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (t.experts != 1)
      *error_msg =
          "unsupported HIP expert tensor format Q6_K: " + std::string(t.name);
    return t.experts == 1;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  // Measure first: the placement is decided once, from the bytes this walk
  // will request of the device heap.
  std::uint64_t measured = 0;
  {
    std::vector<Conversion> conversions;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager, nullptr,       conversions, allocations,
                     bytes,   max_half_cols, max_q8_cols, shards,
                     error_msg};
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  std::vector<Conversion> conversions;
  Uploader up{*stager,   &weights,          conversions,     m->allocations_,
              m->bytes_, m->max_half_cols_, m->max_q8_cols_, shards,
              error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
  m->output_draft_ = uploaded.output_draft;
  for (const auto& c : conversions) {
    if (c.expand_q8) {
      ExpandQ8_0(c.source, static_cast<float*>(c.destination), c.count,
                 nullptr);
    } else {
      NarrowActivations(static_cast<const float*>(c.source), c.destination,
                        false, c.count, nullptr);
    }
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.source_bytes + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
