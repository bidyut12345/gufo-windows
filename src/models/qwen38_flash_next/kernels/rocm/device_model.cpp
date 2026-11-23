#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/platform/fs.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include <gufo/ext/platform/pages.hpp>
#include <gufo/ext/platform/prefault.hpp>
#include <initializer_list>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

[[nodiscard]] constexpr bool NeedsQ8Reencode(const TensorRef& t) noexcept {
  return (t.type == core::GgmlType::kQ6_K ||
          t.type == core::GgmlType::kIQ4_XS) &&
         t.experts == 1 && (t.cols % 32) == 0;
}

[[nodiscard]] inline std::size_t DeviceBytesOf(const TensorRef& t) noexcept {
  if (NeedsQ8Reencode(t)) {
    return t.rows * (t.cols / 32) * 34;
  }
  return t.SizeBytes();
}

[[nodiscard]] inline std::uint16_t FloatToHalfBits(float value) noexcept {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t abs_bits = bits & 0x7fffffffu;
  if (abs_bits > 0x7f800000u) {
    return static_cast<std::uint16_t>(sign | 0x7e00u |
                                      ((abs_bits >> 13) & 0x01ffu));
  }
  if (abs_bits >= 0x47800000u) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (abs_bits < 0x38800000u) {
    if (abs_bits < 0x33000000u) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t shift = 126u - (abs_bits >> 23);
    const std::uint32_t mant = (abs_bits & 0x007fffffu) | 0x00800000u;
    const std::uint32_t rounded =
        mant + (1u << (shift - 1u)) - 1u + ((mant >> shift) & 1u);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded =
      abs_bits + 0x00000fffu + ((abs_bits >> 13) & 1u);
  if (rounded >= 0x47800000u) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  const std::uint32_t half_exp = ((rounded >> 23) - 112u) << 10;
  const std::uint32_t half_mant = (rounded >> 13) & 0x03ffu;
  return static_cast<std::uint16_t>(sign | half_exp | half_mant);
}

inline void QuantizeRowQ8_0(const float* src, std::uint8_t* dst,
                            std::size_t cols) noexcept {
  for (std::size_t offset = 0; offset < cols; offset += 32) {
    float amax = 0.0f;
    for (std::size_t j = 0; j < 32; ++j) {
      amax = std::max(amax, std::abs(src[offset + j]));
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
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

struct ReencodeScratch {
  std::vector<platform::PlatformFile> direct_shards;
  std::vector<float> row_f32;
  std::vector<std::uint8_t> chunk_src;
  std::vector<std::uint8_t> chunk_q8;

  explicit ReencodeScratch(std::size_t shard_count)
      : direct_shards(shard_count, platform::kInvalidFile) {}

  ~ReencodeScratch() {
    for (platform::PlatformFile f : direct_shards) {
      platform::CloseFile(f);
    }
  }
};

[[nodiscard]] bool ReadTensorSlice(platform::PlatformFile fd,
                                   platform::PlatformFile direct_fd,
                                   std::uint64_t offset, std::size_t bytes,
                                   std::vector<std::uint8_t>& buffer,
                                   const std::uint8_t*& out_ptr) {
  constexpr std::size_t kReadAlign = 4096;
  const std::uint64_t aligned_begin = offset & ~(std::uint64_t{kReadAlign - 1});
  const auto skip = static_cast<std::size_t>(offset - aligned_begin);
  const std::size_t aligned_len =
      (skip + bytes + kReadAlign - 1) & ~(kReadAlign - 1);
  if (buffer.size() < aligned_len + kReadAlign) {
    buffer.resize(aligned_len + kReadAlign);
  }
  const auto addr = reinterpret_cast<std::uintptr_t>(buffer.data());
  const std::size_t align_pad = (kReadAlign - (addr % kReadAlign)) % kReadAlign;
  auto* aligned_buf = buffer.data() + align_pad;

  auto read_loop = [&](platform::PlatformFile handle, std::uint64_t pos,
                       std::size_t req_len, std::size_t min_bytes) -> bool {
    std::size_t got = 0;
    while (got < min_bytes) {
      const std::size_t n = platform::ReadFileAt(handle, aligned_buf + got,
                                                 req_len - got, pos + got);
      if (n == 0) {
        return false;
      }
      got += n;
    }
    return true;
  };

  if (platform::FileValid(direct_fd) &&
      read_loop(direct_fd, aligned_begin, aligned_len, skip + bytes)) {
    out_ptr = aligned_buf + skip;
    return true;
  }
  if (read_loop(fd, offset, bytes, bytes)) {
    out_ptr = aligned_buf;
    return true;
  }
  return false;
}

[[nodiscard]] bool UploadReencodedQ8_0(
    const TensorRef& t, void* device_dst,
    std::span<const core::GgufMappedRegion> shards, std::uint32_t shard_idx,
    ReencodeScratch& scratch, std::string* error) {
  const bool has_fd = shard_idx < shards.size() &&
                      platform::FileValid(shards[shard_idx].file_descriptor);
  if (!has_fd && t.data == nullptr) {
    if (error != nullptr) {
      *error = "missing mapped host pointer for " + std::string(t.name);
    }
    return false;
  }
  const std::size_t cols = t.cols;
  const std::size_t src_row_bytes = t.SizeBytes() / t.rows;
  const std::size_t dst_row_bytes = (cols / 32) * 34;
  // Cap host scratch around 16 MiB regardless of vocabulary row count.
  const std::size_t chunk_rows =
      std::max<std::size_t>(1, (std::size_t{16} << 20) / dst_row_bytes);
  if (scratch.row_f32.size() < cols) {
    scratch.row_f32.resize(cols);
  }
  const std::size_t max_q8_bytes =
      std::min<std::size_t>(chunk_rows, t.rows) * dst_row_bytes;
  if (scratch.chunk_q8.size() < max_q8_bytes) {
    scratch.chunk_q8.resize(max_q8_bytes);
  }
  platform::PlatformFile fd = platform::kInvalidFile;
  platform::PlatformFile direct_fd = platform::kInvalidFile;
  if (has_fd) {
    fd = shards[shard_idx].file_descriptor;
    if (!platform::FileValid(scratch.direct_shards[shard_idx])) {
      scratch.direct_shards[shard_idx] = platform::ReopenUnbuffered(fd);
    }
    direct_fd = scratch.direct_shards[shard_idx];
  }
  const auto* src_base = static_cast<const std::uint8_t*>(t.data);
  auto* dst_base = static_cast<std::uint8_t*>(device_dst);
  for (std::size_t row = 0; row < t.rows; row += chunk_rows) {
    const std::size_t count = std::min<std::size_t>(chunk_rows, t.rows - row);
    const std::uint8_t* chunk_ptr = nullptr;
    if (has_fd) {
      const std::uint64_t slice_offset =
          t.file_offset + static_cast<std::uint64_t>(row) * src_row_bytes;
      const std::size_t slice_bytes = count * src_row_bytes;
      if (!ReadTensorSlice(fd, direct_fd, slice_offset, slice_bytes,
                           scratch.chunk_src, chunk_ptr)) {
        if (error != nullptr) {
          *error = "unbuffered host read failed for " + std::string(t.name);
        }
        return false;
      }
    } else {
      chunk_ptr = src_base + row * src_row_bytes;
    }
    for (std::size_t r = 0; r < count; ++r) {
      if (!gufo::quant::DequantizeRow(t.type, chunk_ptr + r * src_row_bytes,
                                      scratch.row_f32.data(), cols)) {
        if (error != nullptr) {
          *error = "unsupported host dequantization for " + std::string(t.name);
        }
        return false;
      }
      QuantizeRowQ8_0(scratch.row_f32.data(),
                      scratch.chunk_q8.data() + r * dst_row_bytes, cols);
    }
    const hipError_t rc =
        hipMemcpy(dst_base + row * dst_row_bytes, scratch.chunk_q8.data(),
                  count * dst_row_bytes, hipMemcpyHostToDevice);
    if (rc != hipSuccess) {
      if (error != nullptr) {
        *error = "HIP upload failed for " + std::string(t.name) + ": " +
                 hipGetErrorString(rc);
      }
      return false;
    }
  }
  return true;
}

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
};

/// Walks the model's tensors in upload order. With `weights` null the walk
/// only measures: it sums the bytes every allocation would request and does
/// no device work, so the weight placement can be decided from the tensors.
struct Uploader {
  hip::WeightUpload& stager;
  std::span<const core::GgufMappedRegion> shards;
  ReencodeScratch& reencode_scratch;
  const hip::WeightAllocator* weights;
  std::vector<Conversion>& conversions;
  std::vector<void*>& tail_zeros;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
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

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const bool reencode_q8 = NeedsQ8Reencode(t);
    const std::size_t size = DeviceBytesOf(t);
    void* ptr = nullptr;
    if (const hipError_t alloc_err = Allocate(&ptr, size + kTailMargin);
        alloc_err != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes): " + hipGetErrorString(alloc_err));
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (reencode_q8) {
      // Drain queued async staging copies before synchronous pageable-host
      // hipMemcpy in UploadReencodedQ8_0 to avoid concurrent stream/paging faults.
      if (!stager.Finish(error) ||
          !UploadReencodedQ8_0(t, ptr, shards, shard_base + t.shard,
                               reencode_scratch, error)) {
        Fail(error != nullptr && !error->empty()
                 ? *error
                 : ("upload failed for " + std::string(t.name)));
        return d;
      }
    } else if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr,
                            error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    tail_zeros.push_back(static_cast<std::uint8_t*>(ptr) + size);
    d.data = ptr;
    d.type = reencode_q8 ? core::GgmlType::kQ8_0 : t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (d.type == core::GgmlType::kBF16 || d.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (d.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Uploads a narrow projection the quantized tier does not cover (the
  /// hyper-connection inject) as F32. A Q8_0 tensor is uploaded as is and
  /// expanded once the disk pipeline drains, so it runs the F32 kernels with
  /// its dequantized values instead of quantizing their activations. A BF16
  /// tensor is widened exactly on the host (see HcInjectUploadFor).
  DeviceTensor CopyF32(const TensorRef& t) {
    if (t.empty() || !ok) {
      return {};
    }
    switch (HcInjectUploadFor(t.type)) {
      case HcInjectUpload::kCopy:
        return Copy(t);
      case HcInjectUpload::kWidenBF16:
        return UploadWidened(t);
      case HcInjectUpload::kExpandQ8_0:
        break;
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
    tail_zeros.push_back(static_cast<std::uint8_t*>(f32) + count * 4);
    d.data = f32;
    d.type = core::GgmlType::kF32;
    return d;
  }

  DeviceTensor UploadWidened(const TensorRef& t) {
    DeviceTensor d;
    const std::size_t count = t.cols * t.rows * t.experts;
    void* f32 = nullptr;
    if (Allocate(&f32, count * sizeof(float) + kTailMargin) != hipSuccess) {
      Fail("device allocation failed for " + std::string(t.name));
      return d;
    }
    if (Measuring()) {
      return d;
    }
    allocations.push_back(f32);
    bytes += count * sizeof(float) + kTailMargin;
    std::vector<float> host(count);
    WidenBF16(static_cast<const std::uint16_t*>(t.data), host.data(), count);
    if (hipMemcpy(f32, host.data(), count * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(f32) + count * sizeof(float), 0,
                  kTailMargin) != hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = f32;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
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
    const std::size_t total_bytes = DeviceBytesOf(t);
    const std::size_t row_bytes = total_bytes / t.rows / 2;
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
    std::erase(tail_zeros,
               static_cast<std::uint8_t*>(combined.data) + total_bytes);
    (void)hipFree(combined.data);
    bytes -= total_bytes + kTailMargin;
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
    const auto is_q8_like = [](const TensorRef* t) noexcept {
      return t->type == core::GgmlType::kQ8_0 || NeedsQ8Reencode(*t);
    };
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType first_type = (*parts.begin())->type;
    const bool all_f32 = first_type == core::GgmlType::kF32;
    const core::GgmlType type =
        all_f32 ? core::GgmlType::kF32 : core::GgmlType::kQ8_0;
    for (const TensorRef* t : parts) {
      const bool valid_type =
          all_f32 ? (t->type == core::GgmlType::kF32) : is_q8_like(t);
      if (t->empty() || !valid_type || t->cols != (*parts.begin())->cols) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += DeviceBytesOf(*t);
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
        auto* dst = static_cast<std::uint8_t*>(ptr) + offset;
        const std::size_t part_bytes = DeviceBytesOf(*t);
        if (NeedsQ8Reencode(*t)) {
          if (!stager.Finish(error) ||
              !UploadReencodedQ8_0(*t, dst, shards, shard_base + t->shard,
                                   reencode_scratch, error)) {
            Fail(error != nullptr && !error->empty()
                     ? *error
                     : ("upload failed for " + std::string(t->name)));
            return d;
          }
        } else if (!stager.Copy(shard_base + t->shard, t->file_offset,
                                part_bytes, dst, error)) {
          Fail("upload failed for " + std::string(t->name));
          return d;
        }
        offset += part_bytes;
      }
    }
    if (type == core::GgmlType::kQ8_0 && !expand_q8) {
      if (Measuring()) {
        return d;
      }
      tail_zeros.push_back(static_cast<std::uint8_t*>(ptr) + size);
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
    tail_zeros.push_back(static_cast<std::uint8_t*>(half) + count * 2);
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
        if (t->empty() ||
            (t->type != core::GgmlType::kQ8_0 && !NeedsQ8Reencode(*t)) ||
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
    u.output =
        w.output.data == w.token_embd.data ? u.token_embd : Copy(w.output);
    u.hc_head = Mixer(w.hc_head);
    u.layers.reserve(w.layers.size());
    for (const auto& l : w.layers) {
      u.layers.push_back(Layer(l));
      if (!ok) {
        return u;
      }
      if (!Measuring() && bytes >= (std::size_t{30} << 30)) {
        platform::TrimWorkingSet();
      }
    }
    if (mtp != nullptr) {
      shard_base = trunk_shards;
      u.mtp = Layer(mtp->block);
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
  draft_output_ = output_;
  draft_output_.rows = prefix_rows;
  if (output_.type == core::GgmlType::kQ8_0 && output_.cols % 256 == 0) {
    // Drafts read a Q4_K copy of the Q8_0 prefix: the head is the larger
    // part of each draft step's weight traffic, and its proposals are
    // verified, so this changes acceptance only. Each row of the copy is
    // the same at any batch width.
    const std::size_t bytes =
        std::size_t{prefix_rows} * (output_.cols / 256) * 144;
    void* q4k = nullptr;
    if (hipMalloc(&q4k, bytes + kTailMargin) != hipSuccess)
      return fail("device allocation failed for the Q4_K draft head");
    allocations_.push_back(q4k);
    bytes_ += bytes + kTailMargin;
    QuantizeQ8_0ToQ4K(output_.data, q4k, prefix_rows, output_.cols, nullptr);
    if (hipMemsetAsync(static_cast<std::uint8_t*>(q4k) + bytes, 0, kTailMargin,
                       nullptr) != hipSuccess ||
        hipStreamSynchronize(nullptr) != hipSuccess)
      return fail("Q4_K draft head quantization failed");
    draft_output_.data = q4k;
    draft_output_.type = core::GgmlType::kQ4_K;
  }
  draft_specials_ = std::move(specials);
  return true;
}

void WidenBF16(const std::uint16_t* src, float* dst,
               std::size_t count) noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
    std::memcpy(dst + i, &bits, sizeof(float));
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // One-expert Q6_K/IQ4_XS matrices (dense, embedding, output) are re-encoded
  // to Q8_0 on upload; routed experts reject Q6_K because only the tuned MMQ
  // and IQ expert kernels are implemented for multi-expert tensors.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K || t.experts <= 1)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
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
    ReencodeScratch measure_scratch(0);
    std::vector<Conversion> conversions;
    std::vector<void*> tail_zeros;
    std::vector<void*> allocations;
    std::size_t bytes = 0;
    std::size_t max_half_cols = 1;
    std::size_t max_q8_cols = 32;
    Uploader measure{*stager,     shards,      measure_scratch, nullptr,
                     conversions, tail_zeros,  allocations,     bytes,
                     max_half_cols, max_q8_cols, error_msg};
    (void)measure.Model(w, mtp, shard_count);
    if (!measure.ok) {
      return nullptr;
    }
    measured = measure.requested;
  }
  const hip::WeightAllocator weights(measured);
  ReencodeScratch reencode_scratch(shards.size());
  std::vector<Conversion> conversions;
  std::vector<void*> tail_zeros;
  Uploader up{
      *stager,      shards,           reencode_scratch, &weights,
      conversions,  tail_zeros,       m->allocations_,  m->bytes_,
      m->max_half_cols_, m->max_q8_cols_, error_msg};
  Uploaded uploaded = up.Model(w, mtp, shard_count);
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  static constexpr std::uint8_t kZeroTail[kTailMargin]{};
  for (void* tail : tail_zeros) {
    const hipError_t tail_rc =
        hipMemcpy(tail, kZeroTail, kTailMargin, hipMemcpyHostToDevice);
    if (tail_rc != hipSuccess) {
      if (error_msg != nullptr) {
        *error_msg =
            "tail zero copy failed: " + std::string(hipGetErrorString(tail_rc));
      }
      return nullptr;
    }
  }
  assert(up.requested == measured);
  m->token_embd_ = uploaded.token_embd;
  m->output_ = uploaded.output;
  m->output_bytes_ = m->output_.type == core::GgmlType::kQ8_0
                         ? static_cast<std::size_t>(m->output_.rows) *
                               (m->output_.cols / 32) * 34
                         : w.output.SizeBytes();
  m->hc_head_ = uploaded.hc_head;
  m->layers_ = std::move(uploaded.layers);
  m->mtp_ = uploaded.mtp;
  m->has_mtp_ = mtp != nullptr;
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
  for (const auto& region : shards) {
    platform::ReleaseMappedPages(region.data, region.size);
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
