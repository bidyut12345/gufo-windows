#if defined(ENGINE_ENABLE_HIP)

// Qwen3-VL layer sequencing and tensor boundaries adapted from antirez/h3.c
// h3_text_encoder.c at 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (MIT).
// Dense projections and residency/prefetch are independent ROCm/HIP code.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <gufo/ext/mapped_prefetch.hpp>
#include <gufo/ext/platform/mapped_file.hpp>
#include "src/models/minimax_h3/prompt_encoder.hpp"
#include "src/models/minimax_h3/prompt_encoder_ops.hip.hpp"

namespace gufo::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kRopeTheta = 5'000'000.0F;
constexpr std::size_t kMaximumPromptTokens = 32768;

std::atomic<std::uint64_t> g_registered_host_bytes{0};
std::atomic<bool> g_prompt_encoder_active{false};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::string HipError(std::string_view operation, hipError_t status) {
  return std::string(operation) + ": " + hipGetErrorString(status);
}

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool IsCancelled(const CancellationToken* cancellation) {
  return cancellation != nullptr && cancellation->IsCancelled();
}

class PromptPhaseLease {
public:
  static std::optional<PromptPhaseLease> Acquire(std::string* error) {
    bool expected = false;
    if (!g_prompt_encoder_active.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      SetError(error, "MiniMax H3 prompt encoder already has an active phase");
      return std::nullopt;
    }
    return PromptPhaseLease();
  }

  ~PromptPhaseLease() {
    if (active_) {
      g_prompt_encoder_active.store(false, std::memory_order_release);
    }
  }
  PromptPhaseLease(const PromptPhaseLease&) = delete;
  PromptPhaseLease& operator=(const PromptPhaseLease&) = delete;
  PromptPhaseLease(PromptPhaseLease&& other) noexcept
      : active_(std::exchange(other.active_, false)) {}
  PromptPhaseLease& operator=(PromptPhaseLease&&) = delete;

private:
  PromptPhaseLease() = default;
  bool active_{true};
};

bool ValidateTarget(int* device, std::string* error) {
  hipError_t status = hipGetDevice(device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDevice", status));
    return false;
  }
  hipDeviceProp_t properties{};
  status = hipGetDeviceProperties(&properties, *device);
  if (status != hipSuccess) {
    SetError(error, HipError("hipGetDeviceProperties", status));
    return false;
  }
  if (!std::string_view(properties.gcnArchName).starts_with("gfx1151") ||
      properties.integrated == 0) {
    SetError(error,
             "MiniMax H3 prompt encoder supports only integrated gfx1151 "
             "Strix Halo; detected " +
                 std::string(properties.gcnArchName));
    return false;
  }
  return true;
}

class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        bytes_(std::exchange(other.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }

  bool Allocate(std::size_t bytes, std::string* error) {
    Reset();
    if (bytes == 0) {
      return true;
    }
    const hipError_t status = hipMalloc(&data_, bytes);
    if (status != hipSuccess) {
      SetError(error, HipError("hipMalloc", status));
      return false;
    }
    bytes_ = bytes;
    return true;
  }

  void Reset() noexcept {
    if (data_ != nullptr) {
      (void)hipFree(data_);
      data_ = nullptr;
    }
    bytes_ = 0;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  template<typename T>
  [[nodiscard]] T* As() noexcept {
    return static_cast<T*>(data_);
  }
  template<typename T>
  [[nodiscard]] const T* As() const noexcept {
    return static_cast<const T*>(data_);
  }

private:
  void* data_{nullptr};
  std::size_t bytes_{0};
};

struct LoadedTensor {
  const TensorDescriptor* descriptor{nullptr};
  const std::uint16_t* data{nullptr};
};

struct HostMapping;

struct LoadedBatch {
  std::vector<LoadedTensor> tensors;
  std::vector<HostMapping> mappings;
  std::uint64_t bytes{0};
  std::uint64_t peak_registered_bytes{0};
  double load_ms{0.0};

  [[nodiscard]] const LoadedTensor* FindTensor(std::string_view name) const {
    auto match = [&](std::string_view target) -> const LoadedTensor* {
      for (const auto& tensor : tensors) {
        if (tensor.descriptor != nullptr && tensor.descriptor->name == target) {
          return &tensor;
        }
      }
      return nullptr;
    };
    if (const LoadedTensor* found = match(name)) {
      return found;
    }
    if (name.starts_with("model.language_model.")) {
      return match("model." + std::string(name.substr(21)));
    }
    return nullptr;
  }

  [[nodiscard]] const std::uint16_t* Find(std::string_view name) const {
    const LoadedTensor* tensor = FindTensor(name);
    return tensor != nullptr ? tensor->data : nullptr;
  }
};

struct FileInterval {
  std::filesystem::path shard;
  std::uint64_t begin{0};
  std::uint64_t end{0};
};

struct PinnedAllocation {
  void* host{nullptr};
  std::size_t capacity{0};
};

class PinnedPool {
public:
  ~PinnedPool() { ReleaseAll(); }

  bool Acquire(std::size_t min_bytes, PinnedAllocation* out,
               std::string* error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = free_.begin(); it != free_.end(); ++it) {
        if (it->capacity >= min_bytes) {
          *out = *it;
          free_.erase(it);
          return true;
        }
      }
      if (!free_.empty()) {
        PinnedAllocation old = free_.back();
        free_.pop_back();
        (void)hipHostFree(old.host);
        g_registered_host_bytes.fetch_sub(old.capacity,
                                          std::memory_order_acq_rel);
      }
    }
    void* ptr = nullptr;
    const hipError_t status =
        hipHostMalloc(&ptr, min_bytes, hipHostMallocMapped);
    if (status != hipSuccess) {
      *error = HipError("hipHostMalloc(H3 text weight)", status);
      return false;
    }
    g_registered_host_bytes.fetch_add(min_bytes, std::memory_order_acq_rel);
    *out = {.host = ptr, .capacity = min_bytes};
    return true;
  }

  void Release(PinnedAllocation alloc) noexcept {
    if (alloc.host == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(alloc);
  }

  void ReleaseAll() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const PinnedAllocation& alloc : free_) {
      if (alloc.host != nullptr) {
        (void)hipHostFree(alloc.host);
        g_registered_host_bytes.fetch_sub(alloc.capacity,
                                          std::memory_order_acq_rel);
      }
    }
    free_.clear();
  }

private:
  std::mutex mutex_;
  std::vector<PinnedAllocation> free_;
};

struct HostMapping {
  std::filesystem::path shard;
  platform::RawMappedFile mapping_file;
  void* host{nullptr};
  std::size_t bytes{0};
  std::size_t capacity{0};
  std::uint64_t file_offset{0};
  void* device{nullptr};
  DeviceBuffer device_storage;
  bool registered{false};
  // `host` is pinned memory filled from the file, not a mapping view.
  bool pinned{false};
  PinnedPool* pool{nullptr};

  HostMapping() = default;
  HostMapping(const HostMapping&) = delete;
  HostMapping& operator=(const HostMapping&) = delete;
  HostMapping(HostMapping&& other) noexcept
      : shard(std::move(other.shard)),
        mapping_file(std::exchange(other.mapping_file, {})),
        host(std::exchange(other.host, nullptr)),
        bytes(std::exchange(other.bytes, 0)),
        capacity(std::exchange(other.capacity, 0)),
        file_offset(other.file_offset),
        device(std::exchange(other.device, nullptr)),
        device_storage(std::move(other.device_storage)),
        registered(std::exchange(other.registered, false)),
        pinned(std::exchange(other.pinned, false)),
        pool(std::exchange(other.pool, nullptr)) {}
  HostMapping& operator=(HostMapping&& other) noexcept {
    if (this != &other) {
      Release();
      shard = std::move(other.shard);
      mapping_file = std::exchange(other.mapping_file, {});
      host = std::exchange(other.host, nullptr);
      bytes = std::exchange(other.bytes, 0);
      capacity = std::exchange(other.capacity, 0);
      file_offset = other.file_offset;
      device = std::exchange(other.device, nullptr);
      device_storage = std::move(other.device_storage);
      registered = std::exchange(other.registered, false);
      pinned = std::exchange(other.pinned, false);
      pool = std::exchange(other.pool, nullptr);
    }
    return *this;
  }
  ~HostMapping() { Release(); }

  void ReleaseHost() noexcept {
    if (pinned && host != nullptr) {
      if (pool != nullptr) {
        pool->Release({.host = host, .capacity = capacity});
      } else {
        (void)hipHostFree(host);
        g_registered_host_bytes.fetch_sub(capacity, std::memory_order_acq_rel);
      }
      pinned = false;
    }
    if (registered && host != nullptr) {
      (void)hipHostUnregister(host);
      g_registered_host_bytes.fetch_sub(bytes, std::memory_order_acq_rel);
      registered = false;
    }
    mapping_file.Close();
    host = nullptr;
    bytes = 0;
    capacity = 0;
    pool = nullptr;
  }

  void Release() noexcept {
    ReleaseHost();
    device_storage.Reset();
    device = nullptr;
  }
};

std::vector<FileInterval> BuildIntervals(
    std::span<const TensorDescriptor* const> descriptors) {
  const auto page_size = static_cast<std::uint64_t>(platform::SystemPageSize());
  std::vector<FileInterval> intervals;
  intervals.reserve(descriptors.size());
  for (const TensorDescriptor* descriptor : descriptors) {
    if (descriptor->payload_bytes >
        std::numeric_limits<std::uint64_t>::max() - descriptor->file_offset) {
      throw std::overflow_error("H3 tensor mapping interval overflows");
    }
    intervals.push_back({
        .shard = descriptor->shard,
        .begin = (descriptor->file_offset / page_size) * page_size,
        .end = descriptor->file_offset + descriptor->payload_bytes,
    });
  }
  std::ranges::sort(intervals, [](const auto& left, const auto& right) {
    return std::tie(left.shard, left.begin, left.end) <
           std::tie(right.shard, right.begin, right.end);
  });
  std::vector<FileInterval> merged;
  for (const auto& interval : intervals) {
    if (!merged.empty() && merged.back().shard == interval.shard &&
        interval.begin <= merged.back().end) {
      merged.back().end = std::max(merged.back().end, interval.end);
    } else {
      merged.push_back(interval);
    }
  }
  return merged;
}

bool HasShape(const TensorDescriptor& descriptor,
              std::initializer_list<std::uint64_t> expected,
              bool allow_int8 = false) {
  const bool dtype_ok =
      descriptor.dtype == DType::kBFloat16 ||
      (allow_int8 && descriptor.dtype == DType::kInt8);
  return dtype_ok &&
         descriptor.shape ==
             std::vector<std::uint64_t>(expected.begin(), expected.end());
}

const TensorDescriptor* FindPromptTensor(const ModelInventory& inventory,
                                         std::string_view name) {
  if (const TensorDescriptor* descriptor = inventory.FindTensor(name)) {
    return descriptor;
  }
  if (name.starts_with("model.language_model.")) {
    return inventory.FindTensor("model." + std::string(name.substr(21)));
  }
  return nullptr;
}

const TensorDescriptor* RequireTensor(
    const ModelInventory& inventory, std::string_view name,
    std::initializer_list<std::uint64_t> shape, std::string* error,
    bool allow_int8 = false) {
  const TensorDescriptor* descriptor = FindPromptTensor(inventory, name);
  if (descriptor == nullptr) {
    SetError(error, "required H3 text tensor is absent: " + std::string(name));
    return nullptr;
  }
  if (descriptor->phase != Phase::kPromptEncoder ||
      !HasShape(*descriptor, shape, allow_int8)) {
    SetError(error, "H3 text tensor contract mismatch: " + std::string(name));
    return nullptr;
  }
  return descriptor;
}

struct BatchLoadResult {
  std::optional<LoadedBatch> batch;
  std::string error;
};

// Windows HIP rejects hipHostRegister on file mapping views, so there the
// interval is read into mapped pinned memory that the GPU reads the same way.
bool ReadIntervalPinned(const std::filesystem::path& path, std::uint64_t offset,
                        std::size_t length, PinnedPool* pool,
                        HostMapping* mapping, std::string* error) {
  if (pool != nullptr) {
    PinnedAllocation alloc;
    if (!pool->Acquire(length, &alloc, error)) {
      return false;
    }
    mapping->host = alloc.host;
    mapping->capacity = alloc.capacity;
    mapping->pool = pool;
  } else {
    hipError_t status =
        hipHostMalloc(&mapping->host, length, hipHostMallocMapped);
    if (status != hipSuccess) {
      mapping->host = nullptr;
      *error = HipError("hipHostMalloc(H3 text weight)", status);
      return false;
    }
    mapping->capacity = length;
    g_registered_host_bytes.fetch_add(length, std::memory_order_acq_rel);
  }
  mapping->pinned = true;
  mapping->bytes = length;
  std::string open_error;
  const platform::PlatformFile file = platform::OpenReadFile(path, &open_error);
  if (!platform::FileValid(file)) {
    *error = "cannot open H3 text shard " + path.string() + ": " + open_error;
    return false;
  }
  std::size_t read_bytes = 0;
  while (read_bytes < length) {
    const std::size_t result = platform::ReadFileAt(
        file, static_cast<std::byte*>(mapping->host) + read_bytes,
        length - read_bytes, offset + read_bytes);
    if (result == 0) {
      break;
    }
    read_bytes += result;
  }
  platform::CloseFile(file);
  if (read_bytes != length) {
    *error = "short read loading H3 text shard " + path.string();
    return false;
  }
  return true;
}

BatchLoadResult LoadBatch(const ModelInventory& inventory,
                          std::vector<const TensorDescriptor*> descriptors,
                          int device, PinnedPool* pool,
                          const CancellationToken* cancellation) {
  const auto begin = Clock::now();
  BatchLoadResult result;
  if (IsCancelled(cancellation)) {
    result.error = "MiniMax H3 prompt encoder cancelled during weight load";
    return result;
  }
  hipError_t status = hipSetDevice(device);
  if (status != hipSuccess) {
    result.error = HipError("hipSetDevice", status);
    return result;
  }

  try {
    LoadedBatch batch;
    std::vector<HostMapping> mappings;
    for (const auto& interval : BuildIntervals(descriptors)) {
      if (IsCancelled(cancellation)) {
        result.error =
            "MiniMax H3 prompt encoder cancelled during file mapping";
        return result;
      }
      const auto path = inventory.model_root() / interval.shard;
      const std::uint64_t length64 = interval.end - interval.begin;
      if (length64 > std::numeric_limits<std::size_t>::max()) {
        result.error = "H3 text mapping is too large";
        return result;
      }
      const auto length = static_cast<std::size_t>(length64);
      HostMapping mapping;
      mapping.shard = interval.shard;
      mapping.file_offset = interval.begin;
#if defined(_WIN32)
      if (!ReadIntervalPinned(path, interval.begin, length, pool, &mapping,
                              &result.error)) {
        return result;
      }
      // Stage into device memory on Windows so hipblasLtMatmul operates on
      // hipMalloc pointers instead of host-mapped memory.
      std::string device_error;
      if (mapping.device_storage.Allocate(length, &device_error) &&
          hipMemcpy(mapping.device_storage.data(), mapping.host, length,
                    hipMemcpyHostToDevice) == hipSuccess) {
        mapping.device = mapping.device_storage.data();
        const std::size_t staged_bytes = length;
        mapping.ReleaseHost();
        mapping.bytes = staged_bytes;
      } else {
        mapping.device_storage.Reset();
        status = hipHostGetDevicePointer(&mapping.device, mapping.host, 0);
        if (status != hipSuccess) {
          result.error =
              HipError("hipHostGetDevicePointer(H3 text weight)", status);
          return result;
        }
      }
#else
      (void)pool;
      std::string map_error;
      if (!mapping.mapping_file.OpenRange(path, interval.begin, length,
                                          platform::MapHint::kNone,
                                          &map_error)) {
        result.error =
            "cannot map H3 text shard " + path.string() + ": " + map_error;
        return result;
      }
      mapping.host = mapping.mapping_file.data;
      mapping.bytes = length;
      core::PrefaultMappedRange(mapping.host, length);
      status = hipHostRegister(mapping.host, length,
                               hipHostRegisterMapped | hipHostRegisterReadOnly);
      if (status != hipSuccess) {
        result.error = HipError("hipHostRegister(mapped|readonly)", status);
        return result;
      }
      mapping.registered = true;
      g_registered_host_bytes.fetch_add(length, std::memory_order_acq_rel);
      status = hipHostGetDevicePointer(&mapping.device, mapping.host, 0);
      if (status != hipSuccess) {
        result.error =
            HipError("hipHostGetDevicePointer(H3 text weight)", status);
        return result;
      }
#endif
      const std::uint64_t registered =
          g_registered_host_bytes.load(std::memory_order_acquire);
      batch.peak_registered_bytes =
          std::max(batch.peak_registered_bytes, registered);
      mappings.push_back(std::move(mapping));
    }

    batch.tensors.reserve(descriptors.size());
    for (const TensorDescriptor* descriptor : descriptors) {
      if (IsCancelled(cancellation)) {
        result.error = "MiniMax H3 prompt encoder cancelled during weight map";
        return result;
      }
      if (descriptor->payload_bytes > std::numeric_limits<std::size_t>::max()) {
        result.error = "H3 text tensor is too large";
        return result;
      }
      const auto bytes = static_cast<std::size_t>(descriptor->payload_bytes);
      const HostMapping* mapping = nullptr;
      for (const auto& candidate : mappings) {
        if (candidate.shard == descriptor->shard &&
            descriptor->file_offset >= candidate.file_offset &&
            descriptor->file_offset + descriptor->payload_bytes <=
                candidate.file_offset + candidate.bytes) {
          mapping = &candidate;
          break;
        }
      }
      if (mapping == nullptr) {
        result.error = "H3 text tensor mapping was not found";
        return result;
      }
      const auto delta = static_cast<std::size_t>(descriptor->file_offset -
                                                  mapping->file_offset);
      LoadedTensor tensor{
          .descriptor = descriptor,
          .data = reinterpret_cast<const std::uint16_t*>(
              static_cast<const std::byte*>(mapping->device) + delta),
      };
      batch.bytes += bytes;
      batch.tensors.push_back(std::move(tensor));
    }
    batch.mappings = std::move(mappings);
    batch.load_ms = Milliseconds(begin, Clock::now());
    result.batch = std::move(batch);
    return result;
  } catch (const std::exception& exception) {
    result.error = "MiniMax H3 prompt weight load failed: " +
                   std::string(exception.what());
    return result;
  }
}

std::vector<const TensorDescriptor*> LayerDescriptors(
    const ModelInventory& inventory, std::size_t layer, std::string* error) {
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  struct Expected {
    std::string_view suffix;
    std::initializer_list<std::uint64_t> shape;
  };
  static const std::array<Expected, 11> kExpected = {{
      {"input_layernorm.weight", {kPromptEncoderHiddenSize}},
      {"self_attn.q_proj.weight",
       {kPromptEncoderQueryHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.k_proj.weight",
       {kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.v_proj.weight",
       {kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
        kPromptEncoderHiddenSize}},
      {"self_attn.q_norm.weight", {kPromptEncoderHeadDimension}},
      {"self_attn.k_norm.weight", {kPromptEncoderHeadDimension}},
      {"self_attn.o_proj.weight",
       {kPromptEncoderHiddenSize,
        kPromptEncoderQueryHeads * kPromptEncoderHeadDimension}},
      {"post_attention_layernorm.weight", {kPromptEncoderHiddenSize}},
      {"mlp.gate_proj.weight",
       {kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize}},
      {"mlp.up_proj.weight",
       {kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize}},
      {"mlp.down_proj.weight",
       {kPromptEncoderHiddenSize, kPromptEncoderIntermediateSize}},
  }};
  std::vector<const TensorDescriptor*> result;
  result.reserve(kExpected.size() * 2U);
  for (const auto& expected : kExpected) {
    const bool is_projection = expected.shape.size() == 2U;
    const TensorDescriptor* descriptor =
        RequireTensor(inventory, prefix + std::string(expected.suffix),
                      expected.shape, error, is_projection);
    if (descriptor == nullptr) {
      return {};
    }
    result.push_back(descriptor);
    if (descriptor->dtype == DType::kInt8) {
      const std::string scale_name = descriptor->name + "_scale";
      const TensorDescriptor* scale_desc = inventory.FindTensor(scale_name);
      if (scale_desc == nullptr && descriptor->name.ends_with(".weight")) {
        scale_desc = inventory.FindTensor(
            descriptor->name.substr(0, descriptor->name.size() - 7U) +
            "._weight_scale");
      }
      if (scale_desc == nullptr || scale_desc->phase != Phase::kPromptEncoder ||
          scale_desc->dtype != DType::kFloat32 ||
          (scale_desc->payload_bytes != 4U &&
           scale_desc->payload_bytes != *expected.shape.begin() * 4U)) {
        SetError(error, "H3 text INT8 scale tensor invalid: " + scale_name);
        return {};
      }
      result.push_back(scale_desc);
    }
  }
  return result;
}

class Bf16Gemm {
public:
  Bf16Gemm() {
    if (rocblas_create_handle(&handle_) != rocblas_status_success) {
      handle_ = nullptr;
      return;
    }
    (void)rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed);
  }
  ~Bf16Gemm() {
    if (handle_ != nullptr) {
      (void)rocblas_destroy_handle(handle_);
      handle_ = nullptr;
    }
  }
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  [[nodiscard]] std::size_t workspace_bytes() const noexcept { return 0; }

  bool Run(const std::uint16_t* weight, const std::uint16_t* input,
           std::uint16_t* output, std::size_t batch, std::size_t rows,
           std::size_t columns, hipStream_t stream, std::string* error) {
    if (handle_ == nullptr || batch == 0 || rows == 0 || columns == 0 ||
        rows > static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max()) ||
        columns >
            static_cast<std::size_t>(std::numeric_limits<rocblas_int>::max())) {
      SetError(error, "invalid rocBLAS BF16 matmul dimensions");
      return false;
    }
    if (rocblas_set_stream(handle_, stream) != rocblas_status_success) {
      SetError(error, "rocBLAS set_stream failed for H3 prompt encoder");
      return false;
    }
    constexpr std::size_t kGemmBatchChunk = 4096;
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    for (std::size_t offset = 0; offset < batch; offset += kGemmBatchChunk) {
      const std::size_t chunk = std::min(kGemmBatchChunk, batch - offset);
      const rocblas_status status = rocblas_gemm_ex(
          handle_, rocblas_operation_transpose, rocblas_operation_none,
          static_cast<rocblas_int>(rows), static_cast<rocblas_int>(chunk),
          static_cast<rocblas_int>(columns), &alpha, weight,
          rocblas_datatype_bf16_r, static_cast<rocblas_int>(columns),
          input + offset * columns, rocblas_datatype_bf16_r,
          static_cast<rocblas_int>(columns), &beta, output + offset * rows,
          rocblas_datatype_bf16_r, static_cast<rocblas_int>(rows),
          output + offset * rows, rocblas_datatype_bf16_r,
          static_cast<rocblas_int>(rows), rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0);
      if (status != rocblas_status_success) {
        SetError(error, "rocBLAS BF16 matmul dispatch failed: " +
                            std::to_string(static_cast<int>(status)));
        return false;
      }
    }
    return true;
  }

private:
  rocblas_handle handle_{nullptr};
};

struct Activations {
  DeviceBuffer ids;
  DeviceBuffer hidden;
  DeviceBuffer norm;
  DeviceBuffer query;
  DeviceBuffer key;
  DeviceBuffer rope_cosine;
  DeviceBuffer rope_sine;
  DeviceBuffer value;
  DeviceBuffer attention;
  DeviceBuffer projection;
  DeviceBuffer gate;
  DeviceBuffer up;
  DeviceBuffer mlp;

  [[nodiscard]] std::uint64_t Bytes() const noexcept {
    return ids.bytes() + hidden.bytes() + norm.bytes() + query.bytes() +
           key.bytes() + rope_cosine.bytes() + rope_sine.bytes() +
           value.bytes() + attention.bytes() + projection.bytes() +
           gate.bytes() + up.bytes() + mlp.bytes();
  }
};

bool AllocateActivations(std::size_t tokens, Activations* activations,
                         std::string* error) {
  const auto allocate_bf16 = [&](DeviceBuffer* buffer, std::size_t elements) {
    if (elements >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
      SetError(error, "H3 prompt activation size overflows");
      return false;
    }
    return buffer->Allocate(elements * sizeof(std::uint16_t), error);
  };
  const auto allocate_f32 = [&](DeviceBuffer* buffer, std::size_t elements) {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      SetError(error, "H3 prompt activation size overflows");
      return false;
    }
    return buffer->Allocate(elements * sizeof(float), error);
  };
  return activations->ids.Allocate(tokens * sizeof(std::uint32_t), error) &&
         allocate_bf16(&activations->hidden,
                       tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->norm, tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->query, tokens * kPromptEncoderQueryHeads *
                                                kPromptEncoderHeadDimension) &&
         allocate_bf16(&activations->key, tokens * kPromptEncoderKeyValueHeads *
                                              kPromptEncoderHeadDimension) &&
         allocate_f32(&activations->rope_cosine,
                      tokens * (kPromptEncoderHeadDimension / 2U)) &&
         allocate_f32(&activations->rope_sine,
                      tokens * (kPromptEncoderHeadDimension / 2U)) &&
         allocate_bf16(&activations->value, tokens *
                                                kPromptEncoderKeyValueHeads *
                                                kPromptEncoderHeadDimension) &&
         allocate_bf16(
             &activations->attention,
             tokens * kPromptEncoderQueryHeads * kPromptEncoderHeadDimension) &&
         allocate_bf16(&activations->projection,
                       tokens * kPromptEncoderHiddenSize) &&
         allocate_bf16(&activations->gate,
                       tokens * kPromptEncoderIntermediateSize) &&
         allocate_bf16(&activations->up,
                       tokens * kPromptEncoderIntermediateSize) &&
         allocate_bf16(&activations->mlp, tokens * kPromptEncoderHiddenSize);
}

struct LayerEvents {
  hipEvent_t start{nullptr};
  hipEvent_t stop{nullptr};

  void Destroy() noexcept {
    if (stop != nullptr) {
      (void)hipEventDestroy(stop);
      stop = nullptr;
    }
    if (start != nullptr) {
      (void)hipEventDestroy(start);
      start = nullptr;
    }
  }
};

bool CollectLayerTiming(LayerEvents* events,
                        PromptEncoderLayerTelemetry* telemetry,
                        std::string* error) {
  const hipError_t wait_status = hipEventSynchronize(events->stop);
  if (wait_status != hipSuccess) {
    events->Destroy();
    SetError(error, HipError("H3 prompt encoder layer wait", wait_status));
    return false;
  }
  float elapsed_ms = 0.0F;
  if (hipEventElapsedTime(&elapsed_ms, events->start, events->stop) !=
      hipSuccess) {
    events->Destroy();
    SetError(error, "cannot read H3 prompt encoder GPU timing");
    return false;
  }
  telemetry->gpu_ms = elapsed_ms;
  events->Destroy();
  return true;
}

bool RunLayer(const LoadedBatch& weights, std::size_t layer, std::size_t tokens,
              Bf16Gemm* gemm, DeviceBuffer* dequant_scratch,
              Activations* activations, hipStream_t stream,
              PromptEncoderLayerTelemetry* telemetry, LayerEvents* events,
              std::string* error) {
  const std::string prefix =
      "model.language_model.layers." + std::to_string(layer) + ".";
  const auto tensor = [&](std::string_view suffix) {
    return weights.Find(prefix + std::string(suffix));
  };
  const auto proj_bf16 = [&](std::string_view suffix, std::size_t rows,
                             std::size_t cols) -> const std::uint16_t* {
    const LoadedTensor* loaded =
        weights.FindTensor(prefix + std::string(suffix));
    if (loaded == nullptr || loaded->descriptor == nullptr) {
      return nullptr;
    }
    if (loaded->descriptor->dtype == DType::kBFloat16) {
      return loaded->data;
    }
    if (loaded->descriptor->dtype != DType::kInt8) {
      return nullptr;
    }
    const std::string& wname = loaded->descriptor->name;
    const LoadedTensor* scale = weights.FindTensor(wname + "_scale");
    if (scale == nullptr && wname.ends_with(".weight")) {
      scale = weights.FindTensor(wname.substr(0, wname.size() - 7U) +
                                 "._weight_scale");
    }
    if (scale == nullptr || scale->descriptor == nullptr) {
      return nullptr;
    }
    const std::size_t max_bytes = kPromptEncoderIntermediateSize *
                                  kPromptEncoderHiddenSize *
                                  sizeof(std::uint16_t);
    if (dequant_scratch->bytes() < max_bytes &&
        !dequant_scratch->Allocate(max_bytes, error)) {
      return nullptr;
    }
    const std::size_t scale_count =
        static_cast<std::size_t>(scale->descriptor->payload_bytes / 4U);
    ops::LaunchDequantInt8ToBf16(
        reinterpret_cast<const std::int8_t*>(loaded->data),
        reinterpret_cast<const float*>(scale->data),
        dequant_scratch->As<std::uint16_t>(), rows, cols, scale_count, stream);
    return dequant_scratch->As<std::uint16_t>();
  };
  const std::uint16_t* input_norm = tensor("input_layernorm.weight");
  const std::uint16_t* query_norm = tensor("self_attn.q_norm.weight");
  const std::uint16_t* key_norm = tensor("self_attn.k_norm.weight");
  const std::uint16_t* post_norm = tensor("post_attention_layernorm.weight");
  if (input_norm == nullptr || query_norm == nullptr || key_norm == nullptr ||
      post_norm == nullptr ||
      weights.FindTensor(prefix + "self_attn.q_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "self_attn.k_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "self_attn.v_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "self_attn.o_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "mlp.gate_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "mlp.up_proj.weight") == nullptr ||
      weights.FindTensor(prefix + "mlp.down_proj.weight") == nullptr) {
    SetError(error, "loaded H3 text layer is incomplete");
    return false;
  }
  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  if (hipEventCreate(&start_event) != hipSuccess ||
      hipEventCreate(&stop_event) != hipSuccess) {
    if (start_event != nullptr) {
      (void)hipEventDestroy(start_event);
    }
    SetError(error, "cannot create H3 prompt encoder timing events");
    return false;
  }
  const auto submit_begin = Clock::now();
  (void)hipEventRecord(start_event, stream);
  const auto token_count = static_cast<std::uint32_t>(tokens);
  ops::LaunchRmsNorm(activations->hidden.As<std::uint16_t>(), input_norm,
                     activations->norm.As<std::uint16_t>(), token_count,
                     kPromptEncoderHiddenSize, kRmsEpsilon, stream);
  const std::uint16_t* query_weight = proj_bf16(
      "self_attn.q_proj.weight",
      kPromptEncoderQueryHeads * kPromptEncoderHeadDimension,
      kPromptEncoderHiddenSize);
  if (query_weight == nullptr ||
      !gemm->Run(query_weight, activations->norm.As<std::uint16_t>(),
                 activations->query.As<std::uint16_t>(), tokens,
                 kPromptEncoderQueryHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  const std::uint16_t* key_weight = proj_bf16(
      "self_attn.k_proj.weight",
      kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
      kPromptEncoderHiddenSize);
  if (key_weight == nullptr ||
      !gemm->Run(key_weight, activations->norm.As<std::uint16_t>(),
                 activations->key.As<std::uint16_t>(), tokens,
                 kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  const std::uint16_t* value_weight = proj_bf16(
      "self_attn.v_proj.weight",
      kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
      kPromptEncoderHiddenSize);
  if (value_weight == nullptr ||
      !gemm->Run(value_weight, activations->norm.As<std::uint16_t>(),
                 activations->value.As<std::uint16_t>(), tokens,
                 kPromptEncoderKeyValueHeads * kPromptEncoderHeadDimension,
                 kPromptEncoderHiddenSize, stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchHeadRmsNorm(activations->query.As<std::uint16_t>(), query_norm,
                         token_count, kPromptEncoderQueryHeads,
                         kPromptEncoderHeadDimension, kRmsEpsilon, stream);
  ops::LaunchHeadRmsNorm(activations->key.As<std::uint16_t>(), key_norm,
                         token_count, kPromptEncoderKeyValueHeads,
                         kPromptEncoderHeadDimension, kRmsEpsilon, stream);
  ops::LaunchRope(activations->query.As<std::uint16_t>(),
                  activations->key.As<std::uint16_t>(),
                  activations->rope_cosine.As<float>(),
                  activations->rope_sine.As<float>(), token_count,
                  kPromptEncoderQueryHeads, kPromptEncoderKeyValueHeads,
                  kPromptEncoderHeadDimension, stream);
  ops::LaunchCausalGqa(
      activations->query.As<std::uint16_t>(),
      activations->key.As<std::uint16_t>(),
      activations->value.As<std::uint16_t>(),
      activations->attention.As<std::uint16_t>(), token_count,
      kPromptEncoderQueryHeads, kPromptEncoderKeyValueHeads,
      kPromptEncoderHeadDimension,
      1.0F / std::sqrt(static_cast<float>(kPromptEncoderHeadDimension)),
      stream);
  const std::uint16_t* output_weight = proj_bf16(
      "self_attn.o_proj.weight", kPromptEncoderHiddenSize,
      kPromptEncoderQueryHeads * kPromptEncoderHeadDimension);
  if (output_weight == nullptr ||
      !gemm->Run(output_weight, activations->attention.As<std::uint16_t>(),
                 activations->projection.As<std::uint16_t>(), tokens,
                 kPromptEncoderHiddenSize,
                 kPromptEncoderQueryHeads * kPromptEncoderHeadDimension, stream,
                 error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchAdd(activations->hidden.As<std::uint16_t>(),
                 activations->projection.As<std::uint16_t>(),
                 activations->hidden.As<std::uint16_t>(),
                 tokens * kPromptEncoderHiddenSize, stream);
  ops::LaunchRmsNorm(activations->hidden.As<std::uint16_t>(), post_norm,
                     activations->norm.As<std::uint16_t>(), token_count,
                     kPromptEncoderHiddenSize, kRmsEpsilon, stream);
  const std::uint16_t* gate_weight = proj_bf16(
      "mlp.gate_proj.weight", kPromptEncoderIntermediateSize,
      kPromptEncoderHiddenSize);
  if (gate_weight == nullptr ||
      !gemm->Run(gate_weight, activations->norm.As<std::uint16_t>(),
                 activations->gate.As<std::uint16_t>(), tokens,
                 kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize,
                 stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  const std::uint16_t* up_weight = proj_bf16(
      "mlp.up_proj.weight", kPromptEncoderIntermediateSize,
      kPromptEncoderHiddenSize);
  if (up_weight == nullptr ||
      !gemm->Run(up_weight, activations->norm.As<std::uint16_t>(),
                 activations->up.As<std::uint16_t>(), tokens,
                 kPromptEncoderIntermediateSize, kPromptEncoderHiddenSize,
                 stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchSwiGlu(activations->gate.As<std::uint16_t>(),
                    activations->up.As<std::uint16_t>(),
                    activations->gate.As<std::uint16_t>(),
                    tokens * kPromptEncoderIntermediateSize, stream);
  const std::uint16_t* down_weight = proj_bf16(
      "mlp.down_proj.weight", kPromptEncoderHiddenSize,
      kPromptEncoderIntermediateSize);
  if (down_weight == nullptr ||
      !gemm->Run(down_weight, activations->gate.As<std::uint16_t>(),
                 activations->mlp.As<std::uint16_t>(), tokens,
                 kPromptEncoderHiddenSize, kPromptEncoderIntermediateSize,
                 stream, error)) {
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  ops::LaunchAdd(activations->hidden.As<std::uint16_t>(),
                 activations->mlp.As<std::uint16_t>(),
                 activations->hidden.As<std::uint16_t>(),
                 tokens * kPromptEncoderHiddenSize, stream);
  const hipError_t launch_status = hipGetLastError();
  if (launch_status != hipSuccess) {
    SetError(error, HipError("H3 prompt encoder kernel launch", launch_status));
    (void)hipEventDestroy(stop_event);
    (void)hipEventDestroy(start_event);
    return false;
  }
  (void)hipEventRecord(stop_event, stream);
  telemetry->submit_ms = Milliseconds(submit_begin, Clock::now());
  telemetry->dispatches = 17;
  events->start = start_event;
  events->stop = stop_event;
  return true;
}

}  // namespace

std::uint64_t PromptEncoderLiveRegisteredHostBytes() noexcept {
  return g_registered_host_bytes.load(std::memory_order_acquire);
}

namespace {

// Device-side bundle of FL2VA vision data for one multimodal prompt
// (issue #177). All pointers are device memory owned by the caller of
// EncodePromptCore for the duration of the call.
struct PromptVisionData {
  std::uint32_t row_count{0};
  const std::uint32_t* rows_device{nullptr};        // [row_count] image rows
  const std::uint16_t* embeddings_device{nullptr};  // [row_count][width]
  const std::uint16_t* deepstack_device{nullptr};   // 3 x [row_count][width]
  const std::int32_t* positions_device{nullptr};    // [sequence][3] (t,h,w)
};

bool EncodePromptCore(const ModelInventory& inventory,
                      std::span<const std::uint32_t> token_ids,
                      const PromptEncoderOptions& options,
                      const PromptVisionData* vision,
                      const CancellationToken* cancellation,
                      PromptEmbedding* output,
                      PromptEncoderTelemetry* telemetry, std::string* error);

}  // namespace

bool EncodePromptLayer50(const ModelInventory& inventory,
                         std::span<const std::uint32_t> token_ids,
                         const PromptEncoderOptions& options,
                         const CancellationToken* cancellation,
                         PromptEmbedding* output,
                         PromptEncoderTelemetry* telemetry,
                         std::string* error) {
  if (options.vision_inputs || options.deepstack_inputs) {
    SetError(error,
             "MiniMax H3 initial prompt encoder is text-only; vision and "
             "deepstack inputs are rejected");
    return false;
  }
  return EncodePromptCore(inventory, token_ids, options, nullptr, cancellation,
                          output, telemetry, error);
}

namespace {

bool EncodePromptCore(const ModelInventory& inventory,
                      std::span<const std::uint32_t> token_ids,
                      const PromptEncoderOptions& options,
                      const PromptVisionData* vision,
                      const CancellationToken* cancellation,
                      PromptEmbedding* output,
                      PromptEncoderTelemetry* telemetry, std::string* error) {
  if (output != nullptr) {
    *output = {};
  }
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (token_ids.size() > kMaximumPromptTokens) {
    SetError(error, "MiniMax H3 prompt exceeds " +
                        std::to_string(kMaximumPromptTokens) + " tokens");
    return false;
  }
  if (output == nullptr || telemetry == nullptr || token_ids.empty() ||
      options.layer_count == 0 || options.layer_count > kPromptEncoderLayers) {
    SetError(error, "invalid MiniMax H3 prompt encoder arguments");
    return false;
  }
  for (const std::uint32_t token : token_ids) {
    if (token >= kPromptEncoderVocabularySize) {
      SetError(error, "MiniMax H3 prompt token is outside the vocabulary");
      return false;
    }
  }
  if (IsCancelled(cancellation)) {
    SetError(error, "MiniMax H3 prompt encoder cancelled");
    return false;
  }
  auto phase_lease = PromptPhaseLease::Acquire(error);
  if (!phase_lease.has_value()) {
    return false;
  }

  int device = 0;
  if (!ValidateTarget(&device, error)) {
    return false;
  }
  hipStream_t compute_stream = nullptr;
  hipError_t status =
      hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking);
  if (status != hipSuccess) {
    SetError(error, HipError("hipStreamCreateWithFlags", status));
    return false;
  }
  const auto destroy_stream = [&]() {
    if (compute_stream != nullptr) {
      (void)hipStreamDestroy(compute_stream);
      compute_stream = nullptr;
    }
  };

  Activations activations;
  if (!AllocateActivations(token_ids.size(), &activations, error)) {
    destroy_stream();
    return false;
  }
  telemetry->activation_bytes = activations.Bytes();
  status = hipMemcpyAsync(activations.ids.data(), token_ids.data(),
                          token_ids.size_bytes(), hipMemcpyHostToDevice,
                          compute_stream);
  if (status != hipSuccess) {
    SetError(error, HipError("hipMemcpyAsync(prompt token IDs)", status));
    destroy_stream();
    return false;
  }
  if (vision == nullptr) {
    ops::LaunchPrepareRope(
        activations.rope_cosine.As<float>(), activations.rope_sine.As<float>(),
        static_cast<std::uint32_t>(token_ids.size()),
        kPromptEncoderHeadDimension, kRopeTheta, compute_stream);
  } else {
    ops::LaunchPrepareRopeMulti(
        activations.rope_cosine.As<float>(), activations.rope_sine.As<float>(),
        vision->positions_device, static_cast<std::uint32_t>(token_ids.size()),
        kPromptEncoderHeadDimension, kRopeTheta, compute_stream);
  }

  const TensorDescriptor* embedding_descriptor = RequireTensor(
      inventory, "model.language_model.embed_tokens.weight",
      {kPromptEncoderVocabularySize, kPromptEncoderHiddenSize}, error);
  if (embedding_descriptor == nullptr) {
    destroy_stream();
    return false;
  }
  PinnedPool pinned_pool;
  BatchLoadResult embedding_result =
      LoadBatch(inventory, {embedding_descriptor}, device, &pinned_pool,
                cancellation);
  if (!embedding_result.batch.has_value()) {
    SetError(error, std::move(embedding_result.error));
    destroy_stream();
    return false;
  }
  LoadedBatch embedding = std::move(*embedding_result.batch);
  telemetry->embedding_weight_bytes = embedding.bytes;
  telemetry->peak_registered_host_bytes = std::max(
      telemetry->peak_registered_host_bytes, embedding.peak_registered_bytes);
  telemetry->peak_device_bytes = telemetry->activation_bytes + embedding.bytes;
  const std::uint16_t* embedding_weight =
      embedding.Find("model.language_model.embed_tokens.weight");
  if (embedding_weight == nullptr) {
    SetError(error, "loaded H3 embedding tensor is absent");
    destroy_stream();
    return false;
  }
  ops::LaunchEmbedding(embedding_weight, activations.ids.As<std::uint32_t>(),
                       activations.hidden.As<std::uint16_t>(),
                       static_cast<std::uint32_t>(token_ids.size()),
                       kPromptEncoderVocabularySize, kPromptEncoderHiddenSize,
                       compute_stream);
  status = hipStreamSynchronize(compute_stream);
  if (status != hipSuccess) {
    SetError(error, HipError("H3 embedding lookup", status));
    destroy_stream();
    return false;
  }
  embedding = {};
  if (vision != nullptr && vision->row_count != 0U) {
    ops::LaunchReplaceRows(vision->rows_device, vision->embeddings_device,
                           activations.hidden.As<std::uint16_t>(),
                           vision->row_count, kPromptEncoderHiddenSize,
                           compute_stream);
    if (hipGetLastError() != hipSuccess) {
      SetError(error, "MiniMax H3 prompt encoder vision row splice failed");
      destroy_stream();
      return false;
    }
    status = hipStreamSynchronize(compute_stream);
    if (status != hipSuccess) {
      SetError(error, HipError("H3 vision row splice", status));
      destroy_stream();
      return false;
    }
  }

  Bf16Gemm gemm;
  DeviceBuffer dequant_scratch;
  telemetry->peak_device_bytes = std::max<std::uint64_t>(
      telemetry->peak_device_bytes,
      telemetry->activation_bytes + gemm.workspace_bytes());

  std::string descriptor_error;
  auto current_descriptors = LayerDescriptors(inventory, 0, &descriptor_error);
  if (current_descriptors.empty()) {
    SetError(error, std::move(descriptor_error));
    destroy_stream();
    return false;
  }
  auto load_start = Clock::now();
  BatchLoadResult current_result =
      LoadBatch(inventory, std::move(current_descriptors), device, &pinned_pool,
                cancellation);
  double current_wait_ms = Milliseconds(load_start, Clock::now());
  if (!current_result.batch.has_value()) {
    SetError(error, std::move(current_result.error));
    destroy_stream();
    return false;
  }
  std::optional<LoadedBatch> current = std::move(*current_result.batch);

  for (std::size_t layer = 0; layer < options.layer_count; ++layer) {
    if (IsCancelled(cancellation)) {
      SetError(error, "MiniMax H3 prompt encoder cancelled between layers");
      destroy_stream();
      return false;
    }
    std::future<BatchLoadResult> next_future;
    bool future_valid = false;
    if (layer + 1 < options.layer_count && options.prefetch_layers) {
      auto next_descriptors =
          LayerDescriptors(inventory, layer + 1, &descriptor_error);
      if (next_descriptors.empty()) {
        SetError(error, std::move(descriptor_error));
        destroy_stream();
        return false;
      }
      next_future =
          std::async(std::launch::async,
                     [&inventory, descriptors = std::move(next_descriptors),
                      device, &pinned_pool, cancellation]() mutable {
                       return LoadBatch(inventory, std::move(descriptors),
                                        device, &pinned_pool, cancellation);
                     });
      future_valid = true;
    }

    PromptEncoderLayerTelemetry layer_telemetry;
    layer_telemetry.layer = layer;
    layer_telemetry.weight_bytes = current->bytes;
    layer_telemetry.load_ms = current->load_ms;
    layer_telemetry.prefetch_wait_ms = current_wait_ms;
    telemetry->peak_registered_host_bytes = std::max(
        telemetry->peak_registered_host_bytes, current->peak_registered_bytes);
    telemetry->peak_device_bytes = std::max<std::uint64_t>(
        telemetry->peak_device_bytes,
        telemetry->activation_bytes + gemm.workspace_bytes() + current->bytes);
    LayerEvents layer_events;
    if (!RunLayer(*current, layer, token_ids.size(), &gemm, &dequant_scratch,
                  &activations, compute_stream, &layer_telemetry, &layer_events,
                  error)) {
      layer_events.Destroy();
      if (future_valid) {
        (void)next_future.get();
      }
      destroy_stream();
      return false;
    }
    if (vision != nullptr && layer < 3U && vision->row_count != 0U) {
      // Qwen3-VL deepstack: tower taps (blocks 8/16/24) add into the hidden
      // state at language layers 0, 1, 2 on the image rows.
      ops::LaunchAddRows(
          vision->rows_device,
          vision->deepstack_device + static_cast<std::size_t>(layer) *
                                         vision->row_count *
                                         kPromptEncoderHiddenSize,
          activations.hidden.As<std::uint16_t>(), vision->row_count,
          kPromptEncoderHiddenSize, compute_stream);
      if (hipGetLastError() != hipSuccess) {
        layer_events.Destroy();
        if (future_valid) {
          (void)next_future.get();
        }
        SetError(error, "MiniMax H3 prompt encoder deepstack add failed");
        destroy_stream();
        return false;
      }
    }
    telemetry->total_dispatches += layer_telemetry.dispatches;
    telemetry->layers.push_back(layer_telemetry);

    if (layer + 1 >= options.layer_count) {
      if (!CollectLayerTiming(&layer_events, &telemetry->layers.back(),
                              error)) {
        destroy_stream();
        return false;
      }
      break;
    }
    const auto wait_begin = Clock::now();
    BatchLoadResult next_result;
    if (future_valid) {
      next_result = next_future.get();
      if (!CollectLayerTiming(&layer_events, &telemetry->layers.back(),
                              error)) {
        destroy_stream();
        return false;
      }
    } else {
      if (!CollectLayerTiming(&layer_events, &telemetry->layers.back(),
                              error)) {
        destroy_stream();
        return false;
      }
      current.reset();
      auto next_descriptors =
          LayerDescriptors(inventory, layer + 1, &descriptor_error);
      if (next_descriptors.empty()) {
        SetError(error, std::move(descriptor_error));
        destroy_stream();
        return false;
      }
      next_result = LoadBatch(inventory, std::move(next_descriptors), device,
                              &pinned_pool, cancellation);
    }
    current_wait_ms = Milliseconds(wait_begin, Clock::now());
    if (!next_result.batch.has_value()) {
      SetError(error, std::move(next_result.error));
      destroy_stream();
      return false;
    }
    const std::uint64_t old_bytes = current.has_value() ? current->bytes : 0;
    const std::uint64_t next_bytes = next_result.batch->bytes;
    telemetry->peak_device_bytes = std::max<std::uint64_t>(
        telemetry->peak_device_bytes, telemetry->activation_bytes +
                                          gemm.workspace_bytes() + old_bytes +
                                          next_bytes);
    current = std::move(*next_result.batch);
  }

  output->tokens = token_ids.size();
  output->width = kPromptEncoderHiddenSize;
  output->values.resize(token_ids.size() * kPromptEncoderHiddenSize);
  status = hipMemcpy(output->values.data(), activations.hidden.data(),
                     output->values.size() * sizeof(std::uint16_t),
                     hipMemcpyDeviceToHost);
  if (status != hipSuccess) {
    *output = {};
    SetError(error, HipError("hipMemcpy(H3 prompt output)", status));
    destroy_stream();
    return false;
  }
  current.reset();
  pinned_pool.ReleaseAll();
  destroy_stream();
  telemetry->live_registered_host_bytes_after =
      PromptEncoderLiveRegisteredHostBytes();
  if (telemetry->live_registered_host_bytes_after != 0) {
    *output = {};
    SetError(error, "MiniMax H3 prompt encoder leaked registered host pages");
    return false;
  }
  return true;
}

}  // namespace

bool EncodePromptWithVisions(const ModelInventory& inventory,
                             const Tokenizer& tokenizer,
                             std::string_view prompt,
                             std::span<const MultimodalBlock> blocks,
                             const std::vector<KeyframeVision>& visions,
                             const CancellationToken* cancellation,
                             PromptEmbedding* output,
                             PromptEncoderTelemetry* telemetry,
                             std::string* error) {
  if (output != nullptr) {
    *output = {};
  }
  if (telemetry != nullptr) {
    *telemetry = {};
  }
  if (output == nullptr || telemetry == nullptr || blocks.empty() ||
      blocks.size() != visions.size()) {
    SetError(error, "MiniMax H3 multimodal prompt requires matching blocks");
    return false;
  }
  KeyframePresentation presentation;
  if (!BuildMultimodalPresentation(tokenizer, prompt, blocks, &presentation,
                                   error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "MiniMax H3 multimodal presentation failed");
    }
    return false;
  }

  // Tower encodes, concatenated in presentation order; deepstack is laid out
  // [tap][row][width] to match the core's per-layer offset.
  std::size_t total_rows = presentation.image_rows.size();
  std::vector<std::uint16_t> all_embeddings;
  all_embeddings.reserve(total_rows * kPromptEncoderHiddenSize);
  std::vector<std::uint16_t> all_deep;
  all_deep.reserve(3U * total_rows * kPromptEncoderHiddenSize);
  std::vector<std::vector<std::uint16_t>> tap_buffers(3);
  std::vector<std::vector<std::uint16_t>> embedded_batch;
  std::vector<std::vector<std::uint16_t>> deep_batch;
  if (!EncodeVisionBatch(inventory, visions, cancellation, &embedded_batch,
                         &deep_batch, error)) {
    if (error != nullptr && error->empty()) {
      SetError(error, "MiniMax H3 vision tower failed");
    }
    return false;
  }
  for (std::size_t i = 0; i < visions.size(); ++i) {
    const KeyframeVision& vision = visions[i];
    const std::vector<std::uint16_t>& embedded = embedded_batch[i];
    const std::vector<std::uint16_t>& deep = deep_batch[i];
    const std::size_t rows =
        static_cast<std::size_t>(vision.grid_h / 2) * (vision.grid_w / 2);
    if (embedded.size() != rows * kPromptEncoderHiddenSize ||
        deep.size() != 3U * rows * kPromptEncoderHiddenSize) {
      SetError(error, "MiniMax H3 vision tower output shape differs");
      return false;
    }
    all_embeddings.insert(all_embeddings.end(), embedded.begin(),
                          embedded.end());
    const std::size_t tap_stride = rows * kPromptEncoderHiddenSize;
    for (std::size_t tap = 0; tap < 3U; ++tap) {
      tap_buffers[tap].insert(tap_buffers[tap].end(),
                              deep.begin() +
                                  static_cast<std::ptrdiff_t>(tap *
                                                              tap_stride),
                              deep.begin() + static_cast<std::ptrdiff_t>(
                                                 (tap + 1U) * tap_stride));
    }
  }
  for (const std::vector<std::uint16_t>& tap_buffer : tap_buffers) {
    all_deep.insert(all_deep.end(), tap_buffer.begin(), tap_buffer.end());
  }
  std::vector<std::uint8_t>& tags = presentation.tags;
  std::vector<std::uint32_t>& image_rows = presentation.image_rows;

  // Device staging (synchronous copies on the default stream; freed below).
  auto device_alloc = [](void** slot, std::size_t bytes,
                         std::string* failure) -> bool {
    if (hipMalloc(slot, bytes) != hipSuccess) {
      SetError(failure, "MiniMax H3 multimodal device staging failed");
      return false;
    }
    return true;
  };
  void* rows_device = nullptr;
  void* embeddings_device = nullptr;
  void* deep_device = nullptr;
  void* positions_device = nullptr;
  const auto free_all = [&]() {
    if (rows_device != nullptr) {
      (void)hipFree(rows_device);
    }
    if (embeddings_device != nullptr) {
      (void)hipFree(embeddings_device);
    }
    if (deep_device != nullptr) {
      (void)hipFree(deep_device);
    }
    if (positions_device != nullptr) {
      (void)hipFree(positions_device);
    }
  };
  bool staged =
      device_alloc(&rows_device, image_rows.size() * sizeof(std::uint32_t),
                   error) &&
      device_alloc(&embeddings_device,
                   all_embeddings.size() * sizeof(std::uint16_t), error) &&
      device_alloc(&deep_device, all_deep.size() * sizeof(std::uint16_t),
                   error) &&
      device_alloc(&positions_device,
                   presentation.rope_positions.size() * sizeof(std::int32_t),
                   error);
  if (staged) {
    staged =
        hipMemcpy(rows_device, image_rows.data(),
                  image_rows.size() * sizeof(std::uint32_t),
                  hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(embeddings_device, all_embeddings.data(),
                  all_embeddings.size() * sizeof(std::uint16_t),
                  hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(deep_device, all_deep.data(),
                  all_deep.size() * sizeof(std::uint16_t),
                  hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(positions_device, presentation.rope_positions.data(),
                  presentation.rope_positions.size() * sizeof(std::int32_t),
                  hipMemcpyHostToDevice) == hipSuccess;
    if (!staged) {
      SetError(error, "MiniMax H3 multimodal staging copy failed");
    }
  }
  if (!staged) {
    free_all();
    return false;
  }
  const PromptVisionData data{static_cast<std::uint32_t>(image_rows.size()),
                              static_cast<const std::uint32_t*>(rows_device),
                              static_cast<const std::uint16_t*>(
                                  embeddings_device),
                              static_cast<const std::uint16_t*>(deep_device),
                              static_cast<const std::int32_t*>(
                                  positions_device)};
  const PromptEncoderOptions options{
      .layer_count = kPromptEncoderLayers,
      .prefetch_layers = true,
      .vision_inputs = true,
      .deepstack_inputs = true,
  };
  const bool ok = EncodePromptCore(inventory, presentation.ids, options,
                                   &data, cancellation,
                                   output, telemetry, error);
  free_all();
  if (!ok && error != nullptr && error->empty()) {
    SetError(error, "MiniMax H3 prompt encoder core failed");
  }
  if (ok) {
    output->tags = std::move(tags);
  }
  return ok;
}

bool EncodePromptMultimodal(const ModelInventory& inventory,
                            const Tokenizer& tokenizer,
                            std::string_view prompt,
                            const std::vector<KeyframeVision>& keyframes,
                            const CancellationToken* cancellation,
                            PromptEmbedding* output,
                            PromptEncoderTelemetry* telemetry,
                            std::string* error) {
  std::vector<MultimodalBlock> blocks;
  blocks.reserve(keyframes.size());
  for (std::size_t i = 0; i < keyframes.size(); ++i) {
    blocks.push_back({.prefix_label = "<Picture " + std::to_string(i + 1) + ">: ",
                      .is_video = false,
                      .grid_h = keyframes[i].grid_h,
                      .grid_w = keyframes[i].grid_w});
  }
  return EncodePromptWithVisions(inventory, tokenizer, prompt, blocks,
                                 keyframes, cancellation, output, telemetry,
                                 error);
}

}  // namespace gufo::minimax_h3

#endif
