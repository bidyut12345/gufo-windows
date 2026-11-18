// Event-timed A/B of the wide attention reader over F16, Q8_0 and
// Q8_0-keys/Q4_K-values caches.
// Timing-only: outputs are downloaded and checksummed so the compiler and
// runtime cannot elide work, but no numerics are judged here (the operator
// test owns correctness). Usage: kvq8_reader_bench [start_pos] [n_tokens]
// [rounds]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace qk = gufo::models::qwen38_flash_next;

namespace {

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

void CheckHip(hipError_t error, const char* what) {
  if (error != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
    std::exit(1);
  }
}

template<typename T>
T* DeviceBytes(std::size_t bytes) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, bytes), "hipMalloc");
  CheckHip(hipMemset(p, 0, bytes), "hipMemset");
  return static_cast<T*>(p);
}

double TimeKernel(const char* label, int rounds, const std::function<void()>& run) {
  hipEvent_t begin, end;
  CheckHip(hipEventCreate(&begin), "event");
  CheckHip(hipEventCreate(&end), "event");
  run();  // warm (allocator, instruction cache)
  CheckHip(hipDeviceSynchronize(), "warm sync");
  CheckHip(hipEventRecord(begin), "record");
  for (int i = 0; i < rounds; ++i) {
    run();
  }
  CheckHip(hipEventRecord(end), "record");
  CheckHip(hipEventSynchronize(end), "sync");
  float ms = 0.0F;
  CheckHip(hipEventElapsedTime(&ms, begin, end), "elapsed");
  hipEventDestroy(begin);
  hipEventDestroy(end);
  std::printf("%-28s %8.3f ms/round (%d rounds)\n", label,
              static_cast<double>(ms) / rounds, rounds);
  return static_cast<double>(ms) / rounds;
}

}  // namespace

int main(int argc, char** argv) {
  const std::uint32_t start_pos =
      argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 16384;
  const std::uint32_t n_tokens =
      argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 2048;
  const int rounds = argc > 3 ? std::atoi(argv[3]) : 20;
  const std::uint32_t n_kv = start_pos + n_tokens;
  const std::size_t kv_elems = std::size_t{n_kv} * kKvWidth;
  const std::size_t q_count = std::size_t{n_tokens} * kQWidth;
  const std::size_t f16_bytes = kv_elems * 2;
  const std::size_t q8_bytes =
      std::size_t{n_kv} * (kKvWidth + kKvWidth / 16);

  std::uint32_t seed = 12345;
  std::vector<float> qf(q_count), gf(q_count);
  for (float& v : qf) {
    v = 4.0F * (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0F;
  }
  for (float& v : gf) {
    v = 3.0F * (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0F;
  }
  std::vector<__half> kh(f16_bytes / 2), vh(f16_bytes / 2);
  for (std::size_t i = 0; i < kh.size(); ++i) {
    kh[i] = __float2half(
        (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0F);
    vh[i] = __float2half(
        (static_cast<int>(NextRandom(&seed) & 0xFFFF) - 32768) / 32768.0F);
  }

  auto* d_q = DeviceBytes<float>(q_count * 4);
  auto* d_gate = DeviceBytes<float>(q_count * 4);
  auto* d_k16 = DeviceBytes<__half>(f16_bytes);
  auto* d_v16 = DeviceBytes<__half>(f16_bytes);
  auto* d_k8 = DeviceBytes<unsigned char>(q8_bytes);
  auto* d_v8 = DeviceBytes<unsigned char>(q8_bytes);
  auto* d_out = DeviceBytes<float>(q_count * 4);
  auto* d_pos = DeviceBytes<std::uint32_t>(4);
  CheckHip(hipMemcpy(d_q, qf.data(), q_count * 4, hipMemcpyHostToDevice),
           "upload q");
  CheckHip(hipMemcpy(d_gate, gf.data(), q_count * 4, hipMemcpyHostToDevice),
           "upload gate");
  CheckHip(hipMemcpy(d_k16, kh.data(), f16_bytes, hipMemcpyHostToDevice),
           "upload k16");
  CheckHip(hipMemcpy(d_v16, vh.data(), f16_bytes, hipMemcpyHostToDevice),
           "upload v16");
  CheckHip(
      hipMemcpy(d_pos, &start_pos, 4, hipMemcpyHostToDevice), "upload pos");

  // Pack the F16 rows into the planar Q8_0 layout on the host so the Q8
  // reader sees a realistic (nonzero) scale region.
  {
    const std::size_t row_bytes = kKvWidth + kKvWidth / 16;
    std::vector<unsigned char> packed(q8_bytes);
    for (std::uint32_t r = 0; r < n_kv; ++r) {
      const auto* src16 =
          reinterpret_cast<const unsigned char*>(kh.data()) + r * kKvWidth * 2;
      unsigned char* dst = packed.data() + r * row_bytes;
      std::memcpy(dst, src16, kKvWidth);  // codes region = raw bytes (timing)
      std::memcpy(dst + kKvWidth, src16, kKvWidth / 16);  // scales pattern
    }
    CheckHip(hipMemcpy(d_k8, packed.data(), q8_bytes, hipMemcpyHostToDevice),
             "upload k8 packed");
    for (std::uint32_t r = 0; r < n_kv; ++r) {
      const auto* src16 =
          reinterpret_cast<const unsigned char*>(vh.data()) + r * kKvWidth * 2;
      unsigned char* dst = packed.data() + r * row_bytes;
      std::memcpy(dst, src16, kKvWidth);
      std::memcpy(dst + kKvWidth, src16, kKvWidth / 16);
    }
    CheckHip(hipMemcpy(d_v8, packed.data(), q8_bytes, hipMemcpyHostToDevice),
             "upload v8 packed");
  }
  CheckHip(hipDeviceSynchronize(), "uploads ready");

  std::printf("start_pos=%u n_tokens=%u (rows=%u) rounds=%d\n", start_pos,
              n_tokens, n_kv, rounds);
  const auto f16 = TimeKernel("F16 reader", rounds, [&] {
    if (!q::WmmaCausalAttention(d_q, d_gate, d_k16, d_v16, nullptr, 0, d_out,
                                n_tokens, start_pos, kHeads, kKvHeads, kDim,
                                kRatio, nullptr)) {
      std::exit(2);
    }
  });
  std::uint64_t sum16 = 0;
  CheckHip(hipMemcpy(&sum16, d_out, 8, hipMemcpyDeviceToHost), "download");
  const auto q8 = TimeKernel("Q8 reader", rounds, [&] {
    if (!q::WmmaCausalAttention(
            d_q, d_gate, d_k8, d_v8, nullptr, 0, d_out, n_tokens, start_pos,
            kHeads, kKvHeads, kDim, kRatio, nullptr, false,
            qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0})) {
      std::exit(2);
    }
  });
  std::uint64_t sum8 = 0;
  CheckHip(hipMemcpy(&sum8, d_out, 8, hipMemcpyDeviceToHost), "download");
  std::printf("ratio q8/f16 = %.3f (checksums %llx / %llx)\n", q8 / f16,
              static_cast<unsigned long long>(sum16),
              static_cast<unsigned long long>(sum8));

  // Planar Q4_K value rows (synthetic nonzero bytes; timing-only).
  const std::size_t q4k_bytes =
      std::size_t{n_kv} * (kKvWidth / 2 + kKvWidth / 16);
  auto* d_v4k = DeviceBytes<unsigned char>(q4k_bytes);
  {
    std::vector<unsigned char> packed(q4k_bytes);
    for (std::size_t i = 0; i < packed.size(); ++i) {
      packed[i] = static_cast<unsigned char>(NextRandom(&seed) & 0xFF);
    }
    CheckHip(hipMemcpy(d_v4k, packed.data(), q4k_bytes, hipMemcpyHostToDevice),
             "upload v4k packed");
  }
  const auto q4k = TimeKernel("Q8K+Q4KV reader", rounds, [&] {
    if (!q::WmmaCausalAttention(
            d_q, d_gate, d_k8, d_v4k, nullptr, 0, d_out, n_tokens, start_pos,
            kHeads, kKvHeads, kDim, kRatio, nullptr, false,
            qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K})) {
      std::exit(2);
    }
  });
  std::uint64_t sum4k = 0;
  CheckHip(hipMemcpy(&sum4k, d_out, 8, hipMemcpyDeviceToHost), "download");
  std::printf("ratio q8k+q4kv/f16 = %.3f (checksum %llx)\n", q4k / f16,
              static_cast<unsigned long long>(sum4k));

  // Sparse selection, the shape serving actually runs: ~512 selected blocks
  // per query plus the visible tail.
  const std::uint32_t max_blocks = (n_kv + kRatio - 1) / kRatio;
  const std::uint32_t mask_words = (max_blocks + 31) / 32;
  std::vector<std::uint32_t> mask(std::size_t{n_tokens} * mask_words, 0);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const std::uint32_t complete = (start_pos + t + 1) / kRatio;
    std::uint32_t selected = 0;
    while (selected < std::min(512U, complete)) {
      const std::uint32_t block = NextRandom(&seed) % complete;
      auto& word = mask[std::size_t{t} * mask_words + block / 32];
      const std::uint32_t bit = 1U << (block % 32);
      if ((word & bit) == 0) {
        word |= bit;
        ++selected;
      }
    }
  }
  auto* d_mask = DeviceBytes<std::uint32_t>(mask.size() * 4);
  CheckHip(
      hipMemcpy(d_mask, mask.data(), mask.size() * 4, hipMemcpyHostToDevice),
      "upload mask");
  const auto f16s = TimeKernel("F16 reader (sparse)", rounds, [&] {
    if (!q::WmmaCausalAttention(d_q, d_gate, d_k16, d_v16, d_mask, mask_words,
                                d_out, n_tokens, start_pos, kHeads, kKvHeads,
                                kDim, kRatio, nullptr)) {
      std::exit(2);
    }
  });
  const auto q8s = TimeKernel("Q8 reader (sparse)", rounds, [&] {
    if (!q::WmmaCausalAttention(
            d_q, d_gate, d_k8, d_v8, d_mask, mask_words, d_out, n_tokens,
            start_pos, kHeads, kKvHeads, kDim, kRatio, nullptr, false,
            qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ8_0})) {
      std::exit(2);
    }
  });
  std::printf("sparse ratio q8/f16 = %.3f\n", q8s / f16s);
  const auto q4ks = TimeKernel("Q8K+Q4KV reader (sparse)", rounds, [&] {
    if (!q::WmmaCausalAttention(
            d_q, d_gate, d_k8, d_v4k, d_mask, mask_words, d_out, n_tokens,
            start_pos, kHeads, kKvHeads, kDim, kRatio, nullptr, false,
            qk::KvDtypes{qk::KvCacheDtype::kQ8_0, qk::KvCacheDtype::kQ4_K})) {
      std::exit(2);
    }
  });
  std::printf("sparse ratio q8k+q4kv/f16 = %.3f\n", q4ks / f16s);
  return 0;
}
