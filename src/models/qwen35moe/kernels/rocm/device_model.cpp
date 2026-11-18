#include "src/models/qwen35moe/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <thread>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen35moe/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen35moe::rocm {
namespace {

using core::GgmlType;

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

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// Encoded bytes of one `cols`-wide row in a dequant-capable format.
std::size_t RowBytesOf(GgmlType type, std::uint64_t cols) {
  switch (type) {
    case GgmlType::kQ4_K:
      return static_cast<std::size_t>(cols / 256) * 144;
    case GgmlType::kQ5_K:
      return static_cast<std::size_t>(cols / 256) * 176;
    case GgmlType::kQ6_K:
      return static_cast<std::size_t>(cols / 256) * 210;
    case GgmlType::kQ8_0:
      return static_cast<std::size_t>(cols / 32) * 34;
    case GgmlType::kF32:
      return static_cast<std::size_t>(cols) * 4;
    default:
      return 0;
  }
}

/// Formats qfn_mmq_moe_vec decodes natively; a raw copy of these halves the
/// decode-time bytes next to the Q8_0 wide-batch view.
bool VecSupported(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ5_1:
    case GgmlType::kQ6_K:
    case GgmlType::kQ8_0:
    case GgmlType::kIQ3_S:
    case GgmlType::kIQ4_XS:
    case GgmlType::kIQ4_NL:
      return true;
    default:
      return false;
  }
}

/// Dequantizes one GGUF row to F32 through the scalar decode paths.
void DequantizeRow(GgmlType type, const void* src, float* dst,
                   std::size_t cols) {
  switch (type) {
    case GgmlType::kQ4_K:
      quant::DequantizeQ4_K(src, dst, cols);
      return;
    case GgmlType::kQ5_K:
      quant::DequantizeQ5_K(src, dst, cols);
      return;
    case GgmlType::kQ6_K:
      quant::DequantizeQ6_K(src, dst, cols);
      return;
    case GgmlType::kQ8_0:
      quant::DequantizeQ8_0(src, dst, cols);
      return;
    default:
      std::memcpy(dst, src, cols * sizeof(float));
      return;
  }
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  const std::vector<core::GgufMappedRegion>& shards;
  std::string* error;
  bool ok{true};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  void* Allocate(std::size_t size) {
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed (" + std::to_string(size) + " bytes)");
      return nullptr;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    return ptr;
  }

  /// Raw byte copy of a mapped tensor.
  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    void* ptr = Allocate(t.SizeBytes());
    if (ptr == nullptr) {
      return d;
    }
    if (!stager.Copy(t.shard, t.file_offset, t.SizeBytes(), ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + t.SizeBytes(), 0,
                         kTailMargin, nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (VecSupported(t.type)) {
      d.native_data = ptr;
      d.native_type = t.type;
    }
    if (t.type == GgmlType::kBF16 || t.type == GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Concatenates the raw bytes of same-format parts into one decode view
  /// next to their dequantized stack. Returns an empty native view when the
  /// parts mix formats or the format has no vector kernel.
  DeviceTensor NativeStack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    const GgmlType type = (*parts.begin())->type;
    std::size_t size = 0;
    std::size_t rows = 0;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || !VecSupported(type)) {
        return DeviceTensor{};
      }
      size += t->SizeBytes();
      rows += t->rows;
    }
    void* ptr = Allocate(size);
    if (ptr == nullptr) {
      return DeviceTensor{};
    }
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("native stack upload failed for " + std::string(t->name));
        return DeviceTensor{};
      }
      offset += t->SizeBytes();
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                         kTailMargin, nullptr);
    d.native_data = ptr;
    d.native_type = type;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    return d;
  }

  /// Uploads the parts as one stacked Q8_0 matrix. Q8_0 parts stream to the
  /// device unchanged; Q4_K/Q5_K/Q6_K parts are host-dequantized to Q8_0
  /// rows. Every part shares `cols`; parts concatenate along rows, and a
  /// routed part contributes experts after its rows.
  DeviceTensor CopyDequantQ8_0(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    const std::uint64_t cols = (*parts.begin())->cols;
    if (cols % 256 != 0) {
      Fail("dequant upload needs 256-aligned columns");
      return d;
    }
    struct Source {
      const std::uint8_t* bytes;
      GgmlType type;
      std::size_t first;
      std::size_t rows;
    };
    std::vector<Source> sources;
    std::size_t rows = 0;
    std::size_t output_rows = 0;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->cols != cols) {
        Fail("dequant upload needs tensors of one width");
        return d;
      }
      sources.push_back(
          {static_cast<const std::uint8_t*>(shards[t->shard].data) +
               t->file_offset,
           t->type, rows,
           static_cast<std::size_t>(t->rows) *
               static_cast<std::size_t>(t->experts)});
      rows += static_cast<std::size_t>(t->rows) *
              static_cast<std::size_t>(t->experts);
      output_rows += t->rows;
    }
    const std::size_t row_dst_bytes =
        static_cast<std::size_t>(cols) / 32 * sizeof(quant::block_q8_0);
    const std::size_t dst_bytes = rows * row_dst_bytes;
    void* ptr = Allocate(dst_bytes);
    if (ptr == nullptr) {
      return d;
    }
    // Q8_0 parts are already in the destination layout: their bytes stream
    // straight to the device, and the dequant+requant round trip (exact but
    // expensive at load) only runs for the other formats.
    bool any_host = false;
    {
      auto part = parts.begin();
      for (const auto& source : sources) {
        const TensorRef* t = *part++;
        if (source.type != GgmlType::kQ8_0) {
          any_host = true;
          continue;
        }
        if (!stager.Copy(t->shard, t->file_offset, t->SizeBytes(),
                         static_cast<std::uint8_t*>(ptr) +
                             source.first * row_dst_bytes,
                         error)) {
          Fail("Q8_0 upload failed for " + std::string(t->name));
          return d;
        }
      }
    }
    if (any_host) {
      std::vector<std::uint8_t> staged(dst_bytes);
      const std::uint32_t workers =
          std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
      std::vector<std::thread> threads;
      for (std::uint32_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
          std::vector<float> chunk(cols);
          const std::size_t begin = rows * w / workers;
          const std::size_t end = rows * (w + 1) / workers;
          for (const auto& source : sources) {
            if (source.type == GgmlType::kQ8_0) {
              continue;
            }
            const std::size_t last = source.first + source.rows;
            const std::size_t lo = std::max(begin, source.first);
            const std::size_t hi = std::min(end, last);
            const std::size_t step = RowBytesOf(source.type, cols);
            for (std::size_t index = lo; index < hi; ++index) {
              DequantizeRow(source.type,
                            source.bytes + (index - source.first) * step,
                            chunk.data(), cols);
              auto* out = reinterpret_cast<quant::block_q8_0*>(
                  staged.data() + index * row_dst_bytes);
              for (std::size_t b = 0; b < cols / 32; ++b) {
                const float* v = chunk.data() + b * 32;
                float amax = 0.0f;
                for (std::size_t j = 0; j < 32; ++j) {
                  amax = std::max(amax, std::fabs(v[j]));
                }
                const float dblock = amax / 127.0f;
                const float id = dblock != 0.0f ? 1.0f / dblock : 0.0f;
                out[b].d = Fp32ToF16(dblock);
                for (std::size_t j = 0; j < 32; ++j) {
                  out[b].qs[j] =
                      static_cast<std::int8_t>(std::roundf(v[j] * id));
                }
              }
            }
          }
        });
      }
      for (auto& thread : threads) {
        thread.join();
      }
      // Copy only the dequantized row ranges: the Q8_0 parts' device bytes
      // are already in place and `staged` holds zeros for them.
      for (const auto& source : sources) {
        if (source.type == GgmlType::kQ8_0) {
          continue;
        }
        if (hipMemcpy(static_cast<std::uint8_t*>(ptr) +
                          source.first * row_dst_bytes,
                      staged.data() + source.first * row_dst_bytes,
                      source.rows * row_dst_bytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
          Fail("Q8_0 upload failed");
          return d;
        }
      }
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + dst_bytes, 0,
                         kTailMargin, nullptr);
    d.data = ptr;
    d.type = GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(output_rows);
    d.experts = static_cast<std::uint32_t>((*parts.begin())->experts);
    max_q8_cols = std::max<std::size_t>(max_q8_cols, cols);
    // Dequantized dense matrices are F16-GEMM-eligible (DenseF16Route), so
    // their activation width bounds the F16 staging buffer too.
    if ((*parts.begin())->experts == 1) {
      max_half_cols = std::max<std::size_t>(max_half_cols, cols);
    }
    return d;
  }

  DeviceTensor CopyDequantOne(const TensorRef& t) {
    // A Q8_0 source is the decode format already: the raw copy is the
    // dequantized tensor, and it carries its own decode view.
    if (t.type == GgmlType::kQ8_0) {
      return Copy(t);
    }
    DeviceTensor d = CopyDequantQ8_0({&t});
    if (d.data == nullptr) {
      return d;
    }
    // The decode view: raw bytes beside the dequantized wide-batch copy.
    const DeviceTensor native = NativeStack({&t});
    d.native_data = native.native_data;
    d.native_type = native.native_type;
    return d;
  }

  /// Uploads F32 matrices stacked along rows and queues the device-side
  /// narrowing to F16 the wide GEMM tier reads.
  DeviceTensor StackF32(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    std::size_t rows = 0;
    std::size_t size = 0;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != GgmlType::kF32 ||
          t->cols != (*parts.begin())->cols) {
        Fail("stacked upload needs F32 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = Allocate(size);
    if (ptr == nullptr) {
      return d;
    }
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  /// Host-dequantizes the stacked alpha/beta projections (any supported
  /// format) into F32 rows and uploads them, queuing the same F16
  /// narrowing the F32 router stacks use.
  DeviceTensor AlphaBeta(const TensorRef& alpha, const TensorRef& beta) {
    if (alpha.type == GgmlType::kF32 && beta.type == GgmlType::kF32) {
      return StackF32({&alpha, &beta});
    }
    DeviceTensor d;
    if (alpha.empty() || beta.empty() || !ok) {
      return d;
    }
    const std::uint64_t cols = alpha.cols;
    if (beta.type != alpha.type || beta.cols != cols) {
      Fail("alpha/beta uploads need one supported format");
      return d;
    }
    const std::size_t row_src_bytes = RowBytesOf(alpha.type, cols);
    if (row_src_bytes == 0) {
      Fail("unsupported alpha/beta format");
      return d;
    }
    const std::size_t rows = alpha.rows + beta.rows;
    std::vector<float> staged(rows * cols);
    const auto fill = [&](const TensorRef& t, std::size_t row_base) {
      const auto* src = static_cast<const std::uint8_t*>(shards[t.shard].data) +
                        t.file_offset;
      std::vector<float> chunk(cols);
      for (std::uint32_t r = 0; r < t.rows; ++r) {
        DequantizeRow(t.type, src + r * row_src_bytes, chunk.data(), cols);
        std::copy_n(chunk.begin(), cols,
                    staged.begin() +
                        static_cast<std::ptrdiff_t>((row_base + r) * cols));
      }
    };
    fill(alpha, 0);
    fill(beta, alpha.rows);
    const std::size_t src_bytes = staged.size() * sizeof(float);
    void* ptr = Allocate(src_bytes);
    if (ptr == nullptr) {
      return d;
    }
    if (hipMemcpy(ptr, staged.data(), src_bytes, hipMemcpyHostToDevice) !=
        hipSuccess) {
      Fail("alpha/beta upload failed");
      return d;
    }
    const std::size_t count = staged.size();
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("alpha/beta staging allocation failed");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>(cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, cols);
    return d;
  }

  /// Concatenates already-uploaded Q8_0 tensors (post-dequant) along rows.
  DeviceTensor StackQ8(std::initializer_list<const DeviceTensor*> parts) {
    DeviceTensor d;
    std::size_t rows = 0;
    std::size_t size = 0;
    for (const DeviceTensor* t : parts) {
      if (t->empty() || t->type != GgmlType::kQ8_0 ||
          t->cols != (*parts.begin())->cols) {
        Fail("Q8_0 stacks need Q8_0 tensors of one width");
        return d;
      }
      rows += t->rows;
      size += std::size_t{t->rows} * t->cols / 32 * 34;
    }
    void* ptr = Allocate(size);
    if (ptr == nullptr) {
      return d;
    }
    std::size_t offset = 0;
    for (const DeviceTensor* t : parts) {
      const std::size_t part = std::size_t{t->rows} * t->cols / 32 * 34;
      if (hipMemcpyAsync(static_cast<std::uint8_t*>(ptr) + offset, t->data,
                         part, hipMemcpyDeviceToDevice,
                         nullptr) != hipSuccess) {
        Fail("stacked copy failed");
        return d;
      }
      offset += part;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = GgmlType::kQ8_0;
    d.cols = (*parts.begin())->cols;
    d.rows = static_cast<std::uint32_t>(rows);
    max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
    return d;
  }

  /// GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  /// quantization-block boundary without changing any weight.
  void SplitMtpProjection(const DeviceTensor& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    // Each half covers half of every raw row; a whole-row decode view of
    // the combined tensor must never leak into the split outputs.
    if (t.native_data != nullptr) {
      Fail("MTP projection split requires the dequantized tensor");
      return;
    }
    const std::size_t total_row_bytes = std::size_t{t.cols} / 32 * 34;
    const std::size_t row_bytes = total_row_bytes / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = t;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(t.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, t.data);
    (void)hipFree(t.data);
    bytes -= total_row_bytes * t.rows + kTailMargin;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.attn_norm = Copy(l.attn_norm);
    d.post_attention_norm = Copy(l.post_attention_norm);
    if (l.linear) {
      // The stacked qkv|gate projection: dequantized parts land in one GEMV,
      // with a raw byte-identical stack as the decode view when the formats
      // allow it. Mixed formats keep separate parts, each with its own view.
      d.ssm_in = CopyDequantQ8_0({&l.ssm_qkv, &l.ssm_gate});
      if (l.ssm_qkv.type == GgmlType::kQ8_0 &&
          l.ssm_gate.type == GgmlType::kQ8_0) {
        // The dequantized stack already is the raw Q8_0 layout; reuse it as
        // the decode view instead of uploading the bytes a second time.
        d.ssm_in.native_data = d.ssm_in.data;
        d.ssm_in.native_type = GgmlType::kQ8_0;
      } else {
        const DeviceTensor native = NativeStack({&l.ssm_qkv, &l.ssm_gate});
        d.ssm_in.native_data = native.native_data;
        d.ssm_in.native_type = native.native_type;
      }
      d.ssm_conv1d = Copy(l.ssm_conv1d);
      d.ssm_alpha_beta = AlphaBeta(l.ssm_alpha, l.ssm_beta);
      d.ssm_dt = Copy(l.ssm_dt);
      d.ssm_a = Copy(l.ssm_a);
      d.ssm_norm = Copy(l.ssm_norm);
      d.ssm_out = CopyDequantOne(l.ssm_out);
    } else {
      d.attn_qkv = CopyDequantQ8_0({&l.attn_q, &l.attn_k, &l.attn_v});
      if (l.attn_q.type == GgmlType::kQ8_0 &&
          l.attn_k.type == GgmlType::kQ8_0 &&
          l.attn_v.type == GgmlType::kQ8_0) {
        d.attn_qkv.native_data = d.attn_qkv.data;
        d.attn_qkv.native_type = GgmlType::kQ8_0;
      } else {
        const DeviceTensor native =
            NativeStack({&l.attn_q, &l.attn_k, &l.attn_v});
        d.attn_qkv.native_data = native.native_data;
        d.attn_qkv.native_type = native.native_type;
      }
      d.attn_out = CopyDequantOne(l.attn_out);
      d.attn_q_norm = Copy(l.attn_q_norm);
      d.attn_k_norm = Copy(l.attn_k_norm);
    }
    d.router = StackF32({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = l.ffn_down_exps.type == GgmlType::kQ6_K
                          ? CopyDequantOne(l.ffn_down_exps)
                          : Copy(l.ffn_down_exps);
    d.shexp_gate = CopyDequantOne(l.shexp_gate);
    d.shexp_up = CopyDequantOne(l.shexp_up);
    d.shexp_down = CopyDequantOne(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_head_norm = Copy(l.nextn_head_norm);
    if (!l.nextn_eh_proj.empty()) {
      // The raw eh_proj rows pack [fc_embedding | fc_hidden] across 4096
      // columns; a raw decode view of the full tensor is meaningless after
      // the column split, so this projection keeps only the Q8_0 halves.
      const auto combined = CopyDequantQ8_0({&l.nextn_eh_proj});
      // The split frees the combined buffer, so the streamed upload must be
      // complete first. This is the final layer upload; the drain costs no
      // overlap with later work.
      if (!stager.Finish(error)) {
        Fail("eh_proj upload failed" +
             (error != nullptr && !error->empty() ? ": " + *error
                                                  : std::string()));
        return d;
      }
      SplitMtpProjection(combined, d.nextn_fc_embedding, d.nextn_fc_hidden);
    }
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(const ModelWeights& w,
                                                 const core::GgufReader& reader,
                                                 const MtpWeights* mtp,
                                                 std::string* error_msg) {
  // Routed gate/up keep their native format (the MMQ and F16 WMMA tiers
  // decode in place); routed Q6_K downs are host-dequantized to Q8_0. Every
  // dense projection is dequantized to Q8_0 unless the artifact already is
  // Q8_0, in which case its bytes stream through unchanged — this keeps the
  // whole dense tier (decode GEMVs, W8A8 and F16 WMMA prefill routes) on its
  // fast paths.
  const auto supported = [&](const TensorRef& t) {
    if (t.experts == 1) {
      switch (t.type) {
        case GgmlType::kQ4_K:
        case GgmlType::kQ5_K:
        case GgmlType::kQ6_K:
        case GgmlType::kQ8_0:
        case GgmlType::kBF16:
        case GgmlType::kF16:
        case GgmlType::kF32:
          return true;
        default:
          *error_msg = "unsupported HIP tensor format: " + std::string(t.name);
          return false;
      }
    }
    switch (t.type) {
      case GgmlType::kQ4_K:
      case GgmlType::kQ5_K:
      case GgmlType::kQ8_0:
      case GgmlType::kIQ3_S:
      case GgmlType::kIQ4_XS:
      case GgmlType::kQ6_K:
        return true;
      default:
        *error_msg =
            "unsupported HIP expert tensor format: " + std::string(t.name);
        return false;
    }
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
  const std::vector<core::GgufMappedRegion> shards(regions.begin(),
                                                   regions.end());
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, shards,          error_msg};
  if (!up.ok) {
    return nullptr;
  }
  m->token_embd_ = w.token_embd.type == GgmlType::kQ8_0
                       ? up.Copy(w.token_embd)
                       : up.CopyDequantOne(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data
          ? m->token_embd_
          : (w.output.type == GgmlType::kQ8_0 ? up.Copy(w.output)
                                              : up.CopyDequantOne(w.output));
  m->output_norm_ = up.Copy(w.output_norm);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
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
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen35moe::rocm
